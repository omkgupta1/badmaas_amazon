"""Stage 3b — context features: competition among candidates and label-free cluster consensus.

Per query q (within its candidate list):
  rank and margin (value minus the best competing value) of several scores, candidate count,
  number of candidates with an exact house number / exact core name.
Per S1 record s (over every query that lists it):
  candidate count, rank of q among them, and the consensus of G_s = the queries whose top-1
  stage-0 candidate is s ("the records that look like s"):
    g_n         |G_s| without q
    g_hn_a      members agreeing with s's house number
    g_hn_b      members agreeing with q's house number (support for q's number)
    g_hn_b_x    same, members from the OTHER source only (cross-source support)
    g_hn_a_src  number of sources (S2/S3) whose members agree with s's house number
    g_name_max / g_addr_max  best name / address similarity between q and a member
A decoy (shifted house number, extra word) typically has no support for its own number while
the S1 number is supported across sources; a typo'd true record shares everything else.
Output: work/<split>/ctx/part-*.parquet, row-aligned with cand parts.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .config import report_dir, split_dir
from .records import load_records
from .utils import LOG, is_done, list_parts, mark_done, part_path, reset_dir, stage, write_df

RANK_COLS = ["s0", "cos_na", "n_tset", "n_core_ratio", "ad_tset", "n_idf_jacc", "hn_eq"]
_HN_SCALE = np.int64(10**12)


def _lookup_counts(keys_all: np.ndarray, keys_query: np.ndarray) -> np.ndarray:
    """Count of each ``keys_query`` value inside ``keys_all``."""
    if len(keys_all) == 0:
        return np.zeros(len(keys_query), dtype=np.float32)
    uniq, cnt = np.unique(keys_all, return_counts=True)
    pos = np.searchsorted(uniq, keys_query)
    pos_c = np.minimum(pos, len(uniq) - 1)
    hit = uniq[pos_c] == keys_query
    return np.where(hit, cnt[pos_c], 0).astype(np.float32)


def s_level_features(s: np.ndarray, q: np.ndarray, s0: np.ndarray, top1: np.ndarray,
                     q_hn: np.ndarray, q_src: np.ndarray, a_hn: np.ndarray,
                     n_s1: int) -> dict[str, np.ndarray]:
    s64 = s.astype(np.int64)
    hq = q_hn[q]            # house number of the query (-1 missing)
    ha = a_hn[s]            # house number of the S1 record
    srcq = q_src[q]
    f: dict[str, np.ndarray] = {}
    f["s_ncand"] = np.bincount(s, minlength=n_s1).astype(np.float32)[s]
    order = np.lexsort((-s0, s64))
    rank = np.empty(len(s), dtype=np.float32)
    s_sorted = s64[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(s_sorted)]))
    rank[order] = (np.arange(len(s_sorted)) - group_start + 1).astype(np.float32)
    f["s_rank_s0"] = rank

    t = top1.astype(bool)
    f["q_top1_is_s"] = t.astype(np.float32)
    f["g_n"] = np.bincount(s[t], minlength=n_s1).astype(np.float32)[s] - t
    has_q = hq >= 0
    has_a = ha >= 0
    tk = t & has_q
    keys_T = s64[tk] * _HN_SCALE + hq[tk]
    key_b = s64 * _HN_SCALE + np.where(has_q, hq, 0)
    key_a = s64 * _HN_SCALE + np.where(has_a, ha, 0)
    cnt_b = _lookup_counts(keys_T, key_b)
    cnt_a = _lookup_counts(keys_T, key_a)
    self_in = (t & has_q).astype(np.float32)
    f["g_hn_b"] = np.where(has_q, cnt_b - self_in, np.nan).astype(np.float32)
    self_a = (t & has_q & has_a & (hq == ha)).astype(np.float32)
    f["g_hn_a"] = np.where(has_a, cnt_a - self_a, np.nan).astype(np.float32)
    by_src = {}
    for src in (2, 3):
        m = tk & (srcq == src)
        keys_src = s64[m] * _HN_SCALE + hq[m]
        by_src[src] = (_lookup_counts(keys_src, key_b), _lookup_counts(keys_src, key_a))
    own_b = np.where(srcq == 2, by_src[2][0], by_src[3][0])
    f["g_hn_b_x"] = np.where(has_q, cnt_b - own_b, np.nan).astype(np.float32)
    a2 = by_src[2][1] - (self_a * (srcq == 2))
    a3 = by_src[3][1] - (self_a * (srcq == 3))
    f["g_hn_a_src"] = np.where(has_a, (a2 > 0).astype(np.float32) + (a3 > 0), np.nan
                               ).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["g_hn_a_share"] = np.where(f["g_n"] > 0, f["g_hn_a"] / f["g_n"], np.nan
                                     ).astype(np.float32)
        f["g_hn_b_share"] = np.where(f["g_n"] > 0, f["g_hn_b"] / f["g_n"], np.nan
                                     ).astype(np.float32)
    # house-number groups of G_s: decoy twins form a second, smaller group sharing a shifted
    # number, so "is q's number the best-supported one" separates them from the true group
    grp_n = np.zeros(n_s1, dtype=np.float32)
    grp_max = np.zeros(n_s1, dtype=np.float32)
    if len(keys_T):
        uk, uc = np.unique(keys_T, return_counts=True)
        us = uk // _HN_SCALE
        grp_n = np.bincount(us, minlength=n_s1).astype(np.float32)
        np.maximum.at(grp_max, us, uc.astype(np.float32))
    f["g_n_hn"] = grp_n[s]
    f["g_hn_b_is_max"] = np.where(has_q, (cnt_b > 0) & (cnt_b >= grp_max[s]), np.nan
                                  ).astype(np.float32)
    f["g_hn_a_is_max"] = np.where(has_a, (cnt_a > 0) & (cnt_a >= grp_max[s]), np.nan
                                  ).astype(np.float32)
    f["g_hn_ba_diff"] = (f["g_hn_b"] - f["g_hn_a"]).astype(np.float32)
    return f


def q_level_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rank / margin of several scores among the candidates of each query (one part).

    Built in steps so that no window expression is nested inside another one.
    """
    work = df.select(
        [pl.col("q")]
        + [pl.col(c).cast(pl.Float32).fill_nan(None).fill_null(-1.0).alias(f"_{c}")
           for c in RANK_COLS]
        + [pl.col("hn_eq").cast(pl.Float32).fill_nan(None).alias("_hn_raw"),
           pl.col("n_core_ratio").cast(pl.Float32).fill_nan(None).alias("_nc_raw")])
    work = work.with_columns(
        [pl.col(f"_{c}").max().over("q").alias(f"_mx_{c}") for c in RANK_COLS])
    work = work.with_columns(
        [(pl.col(f"_{c}") == pl.col(f"_mx_{c}")).cast(pl.Int32).sum().over("q")
         .alias(f"_nmx_{c}") for c in RANK_COLS]
        + [pl.when(pl.col(f"_{c}") < pl.col(f"_mx_{c}")).then(pl.col(f"_{c}"))
           .otherwise(None).max().over("q").alias(f"_mx2_{c}") for c in RANK_COLS])
    exprs = [pl.len().over("q").cast(pl.Float32).alias("q_ncand"),
             (pl.col("_hn_raw") == 1).fill_null(False).cast(pl.Float32).sum().over("q")
             .alias("q_n_hn_eq"),
             (pl.col("_nc_raw") >= 99.5).fill_null(False).cast(pl.Float32).sum().over("q")
             .alias("q_n_name_exact")]
    for c in RANK_COLS:
        v, mx = pl.col(f"_{c}"), pl.col(f"_mx_{c}")
        other = pl.when((v == mx) & (pl.col(f"_nmx_{c}") == 1)).then(pl.col(f"_mx2_{c}")) \
            .otherwise(mx)
        exprs.append(v.rank("min", descending=True).over("q").cast(pl.Float32)
                     .alias(f"q_rank_{c}"))
        exprs.append((v - other).cast(pl.Float32).alias(f"q_margin_{c}"))
    return work.select(exprs)


def _group_similarity(q: np.ndarray, s: np.ndarray, sel: np.ndarray, members: np.ndarray,
                      offsets: np.ndarray, q_name: pl.Series, q_addr: pl.Series,
                      nt: int, cap: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Max name / address similarity of q to the other members of G_s (at most ``cap`` of
    them per pair, which bounds the cost of the rare huge groups)."""
    n = len(q)
    out_n = np.full(n, np.nan, dtype=np.float32)
    out_a = np.full(n, np.nan, dtype=np.float32)
    rows = np.flatnonzero(sel)
    if len(rows) == 0:
        return out_n, out_a
    cnt = np.minimum(offsets[s[rows] + 1] - offsets[s[rows]], cap)
    rep = np.repeat(rows, cnt)
    base = np.repeat(offsets[s[rows]], cnt)
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    mem = members[base + within]
    keep = mem != q[rep]
    rep, mem = rep[keep], mem[keep]
    if len(rep) == 0:
        return out_n, out_a
    qa = q[rep]
    for series, out, scorer in ((q_name, out_n, fuzz.ratio), (q_addr, out_a, fuzz.token_set_ratio)):
        a = series.gather(qa)
        b = series.gather(mem)
        v = process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=nt,
                           dtype=np.float32)
        v[((a.str.len_chars() == 0) | (b.str.len_chars() == 0)).to_numpy()] = -1.0
        tmp = np.full(n, -1.0, dtype=np.float32)
        np.maximum.at(tmp, rep, v)
        got = tmp >= 0
        out[got] = tmp[got]
    return out_n, out_a


def run_context(cfg: dict, split: str) -> None:
    out = split_dir(cfg, split)
    ctx_dir = out / "ctx"
    if is_done(ctx_dir):
        LOG.info("context[%s]: cached", split)
        return
    rep = report_dir(cfg)
    nt = int(cfg["runtime"]["n_threads"])
    top_rank = int(cfg["features"]["consensus_top_rank"])
    cand_parts = list_parts(out / "cand")
    feat_parts = list_parts(out / "feat")
    assert len(cand_parts) == len(feat_parts), "cand / feat parts are not aligned"

    qrec = load_records(cfg, split, "q", ["hn_int", "src", "hn", "name_core", "addr_norm"])
    q_hn = np.where(qrec["hn"].str.len_chars().to_numpy() > 0,
                    qrec["hn_int"].to_numpy(), -1).astype(np.int64)
    q_src = qrec["src"].to_numpy().astype(np.int8)
    q_name, q_addr = qrec["name_core"], qrec["addr_norm"]
    del qrec
    s1rec = load_records(cfg, split, "s1", ["hn_int", "hn", "country"])
    a_hn = np.where(s1rec["hn"].str.len_chars().to_numpy() > 0,
                    s1rec["hn_int"].to_numpy(), -1).astype(np.int64)
    s_country = s1rec["country"].to_numpy()
    n_s1 = s1rec.height
    del s1rec

    # blocking writes the parts of one country contiguously: group parts by country
    part_country = []
    for p in cand_parts:
        s_first = pl.read_parquet(p, columns=["s"], n_rows=1)["s"][0]
        part_country.append(s_country[s_first])
    groups: list[tuple[str, list[int]]] = []
    for i, c in enumerate(part_country):
        if groups and groups[-1][0] == c:
            groups[-1][1].append(i)
        else:
            groups.append((c, [i]))

    reset_dir(ctx_dir)
    for country, idxs in groups:
        with stage(f"context[{split}] {country}", rep):
            cols = ["q", "s", "s0", "s0_rank"]
            cand = pl.concat([pl.read_parquet(cand_parts[i], columns=cols) for i in idxs],
                             how="vertical")
            sizes = [pl.scan_parquet(cand_parts[i]).select(pl.len()).collect().item()
                     for i in idxs]
            s = cand["s"].to_numpy().astype(np.int64)
            q = cand["q"].to_numpy().astype(np.int64)
            top1 = cand["s0_rank"].to_numpy() == 1
            sf = s_level_features(s, q, cand["s0"].to_numpy().astype(np.float32), top1, q_hn,
                                  q_src, a_hn, n_s1)
            # member index of G_s (queries whose top-1 stage-0 candidate is s)
            ms, mq = s[top1], q[top1]
            order = np.argsort(ms, kind="stable")
            members = mq[order]
            offsets = np.searchsorted(ms[order], np.arange(n_s1 + 1)).astype(np.int64)
            del cand, ms, mq, order
            start = 0
            for k, i in enumerate(idxs):
                n = sizes[k]
                c = pl.read_parquet(cand_parts[i], columns=["q", "s0", "s0_rank", "cos_na"])
                f = pl.read_parquet(feat_parts[i], columns=["n_tset", "n_core_ratio", "ad_tset",
                                                            "n_idf_jacc", "hn_eq"])
                df = pl.concat([c, f], how="horizontal")
                ctx = q_level_features(df)
                sl = slice(start, start + n)
                gs_n, gs_a = _group_similarity(q[sl], s[sl],
                                               df["s0_rank"].to_numpy() <= top_rank,
                                               members, offsets, q_name, q_addr, nt,
                                               int(cfg["features"].get("group_sim_cap", 200)))
                extra = {key: val[sl] for key, val in sf.items()}
                extra["g_name_max"] = gs_n
                extra["g_addr_max"] = gs_a
                ctx = pl.concat([ctx, pl.DataFrame(extra)], how="horizontal")
                write_df(ctx, part_path(ctx_dir, i))
                start += n
            LOG.info("  context %s: %d parts, %d pairs", country, len(idxs), len(s))
            del s, q, sf, members, offsets
    mark_done(ctx_dir)
