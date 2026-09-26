"""Post-hoc decisions from cached stage predictions — no retraining.

``probe`` rebuilds a stage's tuned decision (calibration, many-to-one assignment, singleton model,
rule) for train (OOF) and test, drops the kept pairs matching the decoy-twin signature, reports what
that costs on train OOF, and writes both TSVs into a separate folder (never ``output/``).

Decoy-twin signature: the S2/S3 record's house number differs from the S1's by a small arithmetic
shift (not a substring / range relation, i.e. not truncation noise), while BOTH the S1's number and
the record's number are backed by other records of the S1's group. Test decoy businesses come as
several records sharing their shifted number; train decoys were single records, so the consensus
features accept them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import models_dir, split_dir
from .decide import apply_calibration, apply_rule, assign, load_scored_pairs, singleton_table
from .evaluate import breakdown
from .io_utils import write_grouped_tsv
from .records import load_raw, load_records, load_truth
from .utils import LOG, list_parts, load_json, save_json

RULE_FEATS = {"feat": ["hn_eq", "hn_substr", "hn_range", "hn_logdiff"],
              "ctx": ["g_hn_a", "g_hn_b", "g_hn_b_x"]}


def kept_pairs(cfg: dict, split: str, tag: str) -> pl.DataFrame:
    """(s, q, p1[, label]) pairs kept by the stage's tuned rule — OOF on train."""
    md = models_dir(cfg)
    floor = float(cfg["decision"]["floor"])
    df = load_scored_pairs(cfg, split, tag)
    cal = load_json(md / f"calibration_{tag}.json")
    df = df.with_columns(pl.Series("p", apply_calibration(cal, df["p_raw"].to_numpy())))
    asg = assign(df)
    X = singleton_table(cfg, split, df, asg, floor)
    del df
    n_folds = int(cfg["model"]["n_folds"])
    models = [lgb.Booster(model_file=str(md / f"singleton_{tag}_fold{k}.txt"))
              for k in range(n_folds)]
    if split == "train":
        fold = load_truth(cfg)["s1_fold"]
        p_single = np.empty(len(fold), dtype=np.float32)
        for k, m in enumerate(models):
            p_single[fold == k] = m.predict(X[fold == k])
    else:
        p_single = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
    rule = load_json(md / f"decision_{tag}.json")
    keep = apply_rule(rule, asg["s"].to_numpy().astype(np.int64), asg["p1"].to_numpy(),
                      asg["margin"].to_numpy(), p_single, floor)
    return asg.filter(pl.Series(keep))


def attach_features(cfg: dict, split: str, kept: pl.DataFrame) -> pl.DataFrame:
    out = split_dir(cfg, split)
    parts = {d: list_parts(out / d) for d in ["cand", *RULE_FEATS]}
    keys = kept.select(["q", "s"])
    frames = []
    for i, cp in enumerate(parts["cand"]):
        cols = [pl.read_parquet(cp, columns=["q", "s"])]
        cols += [pl.read_parquet(parts[d][i], columns=names) for d, names in RULE_FEATS.items()]
        frames.append(pl.concat(cols, how="horizontal").join(keys, on=["q", "s"], how="semi"))
    return kept.join(pl.concat(frames, how="vertical"), on=["q", "s"], how="left")


def twin_mask(df: pl.DataFrame, max_shift: float, cross_source_only: bool = False) -> np.ndarray:
    dh = np.expm1(df["hn_logdiff"].to_numpy().astype(np.float64))  # |delta| (NaN when unknown)
    support = pl.col("g_hn_b_x") if cross_source_only else pl.col("g_hn_b")
    m = df.select(((pl.col("hn_eq") == 0) & (pl.col("hn_substr") != 1) & (pl.col("hn_range") != 1)
                   & (pl.col("g_hn_a") > 0) & (support > 0)).fill_null(False)).to_series().to_numpy()
    with np.errstate(invalid="ignore"):
        return m & (dh <= max_shift + 0.5)


def run_probe(cfg: dict, tag: str, out_dir: Path) -> dict:
    pc = cfg.get("probe", {})
    max_shift = float(pc.get("max_shift", 99))
    cross = bool(pc.get("cross_source_only", False))
    tr = load_truth(cfg)
    n_true = tr["s_ntrue"]
    info: dict = {"tag": tag, "max_shift": max_shift, "cross_source_only": cross}

    # ---- cost / benefit on train OOF
    k = attach_features(cfg, "train", kept_pairs(cfg, "train", tag))
    s1c = load_records(cfg, "train", "s1", ["country"])["country"].to_numpy()
    groups = {f"country={c}": s1c == c for c in sorted(set(s1c.tolist()))}
    groups["all"] = np.ones(len(n_true), dtype=bool)
    s = k["s"].to_numpy().astype(np.int64)
    lab = k["label"].to_numpy()
    tw = twin_mask(k, max_shift, cross)
    info["train_before"] = breakdown(s, lab, n_true, groups)
    info["train_after"] = breakdown(s[~tw], lab[~tw], n_true, groups)
    info["train_dropped"] = {"pairs": int(tw.sum()), "precision": float(lab[tw].mean()) if tw.any()
                             else None}
    LOG.info("probe[train]: before %s | after %s | dropped %s", info["train_before"],
             info["train_after"], info["train_dropped"])

    # ---- test: drop the signature, write both files
    k = attach_features(cfg, "test", kept_pairs(cfg, "test", tag))
    tw = twin_mask(k, max_shift, cross)
    s1c = load_records(cfg, "test", "s1", ["country"])["country"].to_numpy()
    n_s1 = {c: int((s1c == c).sum()) for c in sorted(set(s1c.tolist()))}
    ks = s1c[k["s"].to_numpy()]
    info["test_kept_per_s1"] = {c: round(float((ks == c).sum()) / n, 4) for c, n in n_s1.items()}
    info["test_dropped_per_s1"] = {c: round(float(((ks == c) & tw).sum()) / n, 4)
                                   for c, n in n_s1.items()}
    LOG.info("probe[test]: kept/S1 %s | dropped/S1 %s", info["test_kept_per_s1"],
             info["test_dropped_per_s1"])
    k = k.filter(pl.Series(~tw))

    out_dir.mkdir(parents=True, exist_ok=True)
    s1_ids = load_raw(cfg, "test", "s1", ["entity_id"])["entity_id"].to_list()
    q_ids = np.array(load_raw(cfg, "test", "q", ["entity_id"])["entity_id"].to_list(), dtype=object)
    info["s1_with_matches"] = write_grouped_tsv(
        out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, k["s"].to_numpy(),
        k["q"].to_numpy(), q_ids, order_key=k["p1"].to_numpy())
    cand = pl.concat([pl.read_parquet(p, columns=["s", "q", "s0"])
                      for p in list_parts(split_dir(cfg, "test") / "cand")], how="vertical")
    write_grouped_tsv(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids,
                      cand["s"].to_numpy(), cand["q"].to_numpy(), q_ids,
                      order_key=cand["s0"].to_numpy())
    info["matched_pairs"] = k.height
    validator = Path(cfg["paths"]["validator"])
    res = subprocess.run([sys.executable, str(validator),
                          "--matching", str(out_dir / "matching_results.tsv"),
                          "--candidate", str(out_dir / "candidate_pairs.tsv"),
                          "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test"),
                          "--check-ids"], capture_output=True, text=True)
    info["validator_exit"] = res.returncode
    LOG.info("validator (exit %d):\n%s%s", res.returncode, res.stdout[-1500:], res.stderr[-500:])
    save_json(info, out_dir / "probe_info.json")
    return info


def blank_country(cfg: dict, src_dir: Path, country: str, out_dir: Path) -> dict:
    """Leaderboard probe: a copy of ``src_dir/matching_results.tsv`` with every S1 of ``country``
    emptied. Then  F(country) ~= (LB_src - LB_blank) / share + singleton_share(country),
    with share / singleton-share estimates saved in blank_info.json."""
    s1 = load_raw(cfg, "test", "s1", ["entity_id", "country"]).rename(
        {"entity_id": "source1_entity_id"})
    m = (pl.read_csv(src_dir / "matching_results.tsv", separator="\t", quote_char=None,
                     infer_schema_length=0).fill_null("").with_row_index("_row"))
    m = m.join(s1, on="source1_entity_id", how="left").sort("_row")
    is_c = (m["country"] == country).fill_null(False)
    info = {"country": country, "source": str(src_dir),
            "share_of_s1": float(is_c.mean()),
            "singleton_share_estimate": float((m.filter(is_c)["matched_entity_ids"] == "").mean()),
            "rows_blanked": int(is_c.sum())}
    out = m.with_columns(pl.when(is_c).then(pl.lit("")).otherwise(pl.col("matched_entity_ids"))
                         .alias("matched_entity_ids")).select(["source1_entity_id",
                                                               "matched_entity_ids"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    res = subprocess.run([sys.executable, str(Path(cfg["paths"]["validator"])),
                          "--matching", str(out_dir / "matching_results.tsv"),
                          "--candidate", str(src_dir / "candidate_pairs.tsv"),
                          "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test"),
                          "--check-ids"], capture_output=True, text=True)
    info["validator_exit"] = res.returncode
    info["read_back"] = (f"F({country}) = (LB_source - LB_blank) / {info['share_of_s1']:.4f} "
                         f"+ {info['singleton_share_estimate']:.4f}")
    save_json(info, out_dir / "blank_info.json")
    LOG.info("blank probe: %s\nvalidator (exit %d): %s", info, res.returncode, res.stdout[-400:])
    return info
