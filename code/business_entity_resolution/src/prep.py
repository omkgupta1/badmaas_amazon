"""Stage 1 — load the raw TSVs, mine lexicons, parse every record, cache parquet.

Outputs in ``work/<split>/``:
  raw_s1.parquet / raw_q.parquet        original strings (reports, cross-encoder)
  records_s1/part-*.parquet            parsed S1 records   (idx = row in test/train_source1)
  records_q/part-*.parquet             parsed S2+S3 records (idx: S2 rows first, then S3)
  gt_pairs.parquet, q_true.npy, s_ntrue.npy, s1_fold.npy, s1_half.npy   (train only)
and the lexicons in ``work/lexicon/``.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from collections import deque

import numpy as np
import polars as pl

from .config import lexicon_dir, report_dir, split_dir
from .io_utils import read_ground_truth, read_source
from .lexicon_mining import (merge_address_lexicons, mine_address_lexicon, mine_indic_lexicon,
                             pseudo_pairs, s1_component_counts, sample_indices, state_vocab)
from .parse_address import ADDR_FIELDS, parse_address
from .text_norm import NAME_FIELDS, normalize_name
from .utils import (LOG, chunks, is_done, load_json, mark_done, part_path, reset_dir, save_json,
                    stage, write_df)

LITE_FIELDS = ["name_sorted", "hn", "comps"]
RECORD_FIELDS = NAME_FIELDS + [f for f in ADDR_FIELDS if f != "comps"]
INT_FIELDS = {
    "legal_mask": pl.Int64, "is_domain": pl.Int8, "has_alias": pl.Int8, "name_script": pl.Int8,
    "n_name_tok": pl.Int16, "hn_int": pl.Int64, "hn_hi": pl.Int64, "has_po": pl.Int8,
    "addr_empty": pl.Int8, "n_comps": pl.Int16, "addr_script": pl.Int8,
}
_NAME_I = {f: i for i, f in enumerate(NAME_FIELDS)}
_ADDR_I = {f: i for i, f in enumerate(ADDR_FIELDS)}
_INDIC_CLASS = "[ऀ-ൿ]"

_G: dict = {}


def _init_worker(indic_lex: dict, addr_lex: dict, lite: bool) -> None:
    _G["indic"], _G["addr"], _G["lite"] = indic_lex, addr_lex, lite


def _parse_chunk(task: tuple[list[str], list[str], list[str]]) -> pl.DataFrame:
    names, addrs, countries = task
    indic, addr_lex, lite = _G["indic"], _G["addr"], _G["lite"]
    if lite:
        cols: dict[str, list] = {f: [] for f in LITE_FIELDS}
        for nm, ad in zip(names, addrs):
            n = normalize_name(nm, indic)
            a = parse_address(ad, indic)
            cols["name_sorted"].append(n[_NAME_I["name_sorted"]])
            cols["hn"].append(a[_ADDR_I["hn"]])
            cols["comps"].append(a[_ADDR_I["comps"]])
        return pl.DataFrame(cols, schema={f: pl.Utf8 for f in LITE_FIELDS})
    cols = {f: [] for f in RECORD_FIELDS}
    for nm, ad, country in zip(names, addrs, countries):
        lx = addr_lex.get(country) or {}
        n = normalize_name(nm, indic)
        a = parse_address(ad, indic, lx.get("state"), lx.get("abbr"))
        for f, v in zip(NAME_FIELDS, n):
            cols[f].append(v)
        for f, v in zip(ADDR_FIELDS, a):
            if f != "comps":
                cols[f].append(v)
    return pl.DataFrame(cols, schema={f: INT_FIELDS.get(f, pl.Utf8) for f in RECORD_FIELDS})


def _parse_frame(df: pl.DataFrame, cfg: dict, indic_lex: dict, addr_lex: dict, lite: bool,
                 out_dir=None, meta_cols=None) -> pl.DataFrame | None:
    """Parse ``df`` in parallel. Lite results are returned; full results are written as parts."""
    size = cfg["prep"]["chunk_size"]
    n_workers = max(1, int(cfg["runtime"]["n_workers"]))
    bounds = list(chunks(df.height, size))
    names, addrs, countries = df["business_name"], df["business_address"], df["country"]
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    results: list[pl.DataFrame] = []
    if out_dir is not None:
        reset_dir(out_dir)

    def consume(i: int, res: pl.DataFrame) -> None:
        if out_dir is None:
            results.append(res)
        else:
            a, b = bounds[i]
            meta = df.slice(a, b - a).select(meta_cols)
            write_df(pl.concat([meta, res], how="horizontal"), part_path(out_dir, i))

    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(indic_lex, addr_lex, lite)) as pool:
        inflight: deque = deque()
        for i, (a, b) in enumerate(bounds):
            task = (names.slice(a, b - a).to_list(), addrs.slice(a, b - a).to_list(),
                    countries.slice(a, b - a).to_list())
            inflight.append((i, pool.apply_async(_parse_chunk, (task,))))
            while len(inflight) >= 3 * n_workers:
                j, r = inflight.popleft()
                consume(j, r.get())
            if i % 20 == 0:
                LOG.info("  parse %s: submitted %d/%d chunks", "lite" if lite else "full", i + 1,
                         len(bounds))
        while inflight:
            j, r = inflight.popleft()
            consume(j, r.get())
    if out_dir is None:
        return pl.concat(results, how="vertical")
    return None


def _load_raw(cfg: dict, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    data_dir = cfg["paths"]["data_dir"]
    s1 = read_source(data_dir, split, 1).with_columns(pl.lit(1, pl.Int8).alias("src"))
    s2 = read_source(data_dir, split, 2).with_columns(pl.lit(2, pl.Int8).alias("src"))
    s3 = read_source(data_dir, split, 3).with_columns(pl.lit(3, pl.Int8).alias("src"))
    q = pl.concat([s2, s3], how="vertical")
    s1 = s1.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    q = q.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    return s1, q


def _ground_truth(cfg: dict, s1: pl.DataFrame, q: pl.DataFrame, out) -> pl.DataFrame:
    gt = read_ground_truth(cfg["paths"]["data_dir"])
    gtp = (gt.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s")),
                   on="s1_id", how="inner")
             .join(q.select(pl.col("entity_id").alias("q_id"), pl.col("idx").alias("q")),
                   on="q_id", how="inner")
             .select(["s", "q"]))
    if gtp.height != gt.height:
        LOG.warning("ground truth: %d of %d pairs reference unknown ids", gt.height - gtp.height,
                    gt.height)
    write_df(gtp, out / "gt_pairs.parquet")
    q_true = np.full(q.height, -1, dtype=np.int32)
    q_true[gtp["q"].to_numpy()] = gtp["s"].to_numpy()
    np.save(out / "q_true.npy", q_true)
    np.save(out / "s_ntrue.npy", np.bincount(gtp["s"].to_numpy(), minlength=s1.height)
            .astype(np.int32))
    rng = np.random.default_rng(cfg["runtime"]["seed"])
    np.save(out / "s1_fold.npy", rng.integers(0, cfg["model"]["n_folds"], s1.height)
            .astype(np.int8))
    np.save(out / "s1_half.npy", rng.integers(0, 2, s1.height).astype(np.int8))
    return gtp


def _mine_indic(cfg: dict, s1: pl.DataFrame, q: pl.DataFrame, gtp: pl.DataFrame) -> dict:
    qi = q.filter(pl.col("business_name").str.contains(_INDIC_CLASS)
                  | pl.col("business_address").str.contains(_INDIC_CLASS))
    pairs = (gtp.join(qi.select(pl.col("idx").alias("q"), pl.col("business_name").alias("bn"),
                                pl.col("business_address").alias("ba")), on="q", how="inner")
                .join(s1.select(pl.col("idx").alias("s"), pl.col("business_name").alias("an"),
                                pl.col("business_address").alias("aa")), on="s", how="inner"))
    LOG.info("indic lexicon: %d aligned pairs with native script", pairs.height)
    p = cfg["prep"]["indic_lexicon"]
    return mine_indic_lexicon(
        pairs["bn"].to_list() + pairs["ba"].to_list(),
        pairs["an"].to_list() + pairs["aa"].to_list(),
        min_count=p["min_count"], min_share=p["min_share"], min_sim=p["min_sim"])


def run_prep(cfg: dict, split: str) -> None:
    out = split_dir(cfg, split)
    if is_done(out / "records_q") and is_done(out / "records_s1"):
        LOG.info("prep[%s]: cached", split)
        return
    rep = report_dir(cfg)
    lex_dir = lexicon_dir(cfg)
    with stage(f"prep[{split}] load", rep):
        s1, q = _load_raw(cfg, split)
        write_df(s1.select(["idx", "entity_id", "business_name", "business_address", "country"]),
                 out / "raw_s1.parquet")
        write_df(q.select(["idx", "entity_id", "src", "business_name", "business_address",
                           "country"]), out / "raw_q.parquet")
        gtp = _ground_truth(cfg, s1, q, out) if split == "train" else None

    with stage(f"prep[{split}] indic lexicon", rep):
        if split == "train":
            indic_lex = _mine_indic(cfg, s1, q, gtp)
            save_json(indic_lex, lex_dir / "indic.json")
        else:
            path = lex_dir / "indic.json"
            if not path.exists():
                raise FileNotFoundError("run `prep --split train` first (needs the Indic lexicon)")
            indic_lex = load_json(path)

    with stage(f"prep[{split}] pass 1 (lite parse)", rep):
        s1_lite = _parse_frame(s1, cfg, indic_lex, {}, lite=True)
        s1_lite = pl.concat([s1.select(["idx", "country"]), s1_lite], how="horizontal")
        q_lite = _parse_frame(q, cfg, indic_lex, {}, lite=True)
        q_lite = pl.concat([q.select(["idx", "country"]), q_lite], how="horizontal")

    with stage(f"prep[{split}] address lexicon", rep):
        p = cfg["prep"]["address_lexicon"]
        states = state_vocab(s1_lite, p["state_min_s1_count"])
        comp_counts = s1_component_counts(s1_lite)
        aligned = pseudo_pairs(s1_lite, q_lite, p["max_pseudo_pairs"], cfg["runtime"]["seed"])
        if gtp is not None:
            take = sample_indices(gtp.height, p["max_pseudo_pairs"], cfg["runtime"]["seed"])
            aligned = pl.concat([aligned, gtp.select(["s", "q"]).select(pl.all().gather(take))],
                                how="vertical")
        s_idx = aligned["s"].to_numpy()
        q_idx = aligned["q"].to_numpy()
        mined = mine_address_lexicon(
            s1_lite["comps"].gather(s_idx).to_list(), q_lite["comps"].gather(q_idx).to_list(),
            s1_lite["country"].gather(s_idx).to_list(), states, comp_counts, cfg)
        addr_lex = {c: {"state": {**states.get(c, {}), **mined.get(c, {}).get("state", {})},
                        "abbr": mined.get(c, {}).get("abbr", {})}
                    for c in set(states) | set(mined)}
        if split != "train":
            train_path = lex_dir / "address_train.json"
            if train_path.exists():
                addr_lex = merge_address_lexicons(load_json(train_path), addr_lex)
        save_json(addr_lex, lex_dir / f"address_{split}.json")
        del s1_lite, q_lite, aligned

    meta = ["idx", "entity_id", "src", "country"]
    with stage(f"prep[{split}] pass 2 (full parse)", rep):
        _parse_frame(s1, cfg, indic_lex, addr_lex, lite=False, out_dir=out / "records_s1",
                     meta_cols=meta)
        mark_done(out / "records_s1", {"rows": s1.height})
        _parse_frame(q, cfg, indic_lex, addr_lex, lite=False, out_dir=out / "records_q",
                     meta_cols=meta)
        mark_done(out / "records_q", {"rows": q.height})

    _prep_report(cfg, split)


def _prep_report(cfg: dict, split: str) -> None:
    from .records import load_records

    out = split_dir(cfg, split)
    stats = {}
    for side in ("s1", "q"):
        r = load_records(cfg, split, side, ["country", "src", "hn", "state", "city", "street",
                                            "addr_empty", "is_domain", "has_alias",
                                            "name_script"])
        stats[side] = (
            r.group_by(["country", "src"]).agg(
                pl.len().alias("n"),
                (pl.col("hn") != "").mean().alias("has_hn"),
                (pl.col("state") != "").mean().alias("has_state"),
                (pl.col("city") != "").mean().alias("has_city"),
                (pl.col("street") != "").mean().alias("has_street"),
                pl.col("addr_empty").cast(pl.Float64).mean().alias("addr_empty"),
                pl.col("is_domain").cast(pl.Float64).mean().alias("is_domain"),
                pl.col("has_alias").cast(pl.Float64).mean().alias("has_alias"),
                (pl.col("name_script") > 0).mean().alias("indic_name"),
            ).sort(["country", "src"]).to_dicts())
    save_json(stats, report_dir(cfg) / f"prep_{split}.json")
    save_json(stats, out / "prep_stats.json")
    LOG.info("prep[%s] parse coverage: %s", split, stats)
