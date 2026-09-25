"""Stage 5a — collective features from stage-A probabilities + record-level "orphan" model.

For a pair (s, q) with stage-A probability pa:
  q_pa_rank / q_pa_margin / q_pa_max_other / q_pa_sum     competition inside q's candidates
  s_pa_sum_other / s_pa_max_other / s_n50_other           how much mass s already attracts
  s_pa_sum_other_src / s_n50_other_src                     ... from q's own source (S2 or S3)
  psup_a / psup_b (+ shares)                               probability-weighted support among
                                                           s's other candidates for s's house
                                                           number vs q's house number
  conf_name / conf_addr / conf_hn_eq                       similarity of q to s's most
                                                           confident other record
  p_match                                                  P(q matches any S1) — OOF model on
                                                           the top candidate of each query
Written to work/<split>/coll/part-*.parquet, row-aligned with cand parts.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .config import models_dir, report_dir, split_dir
from .context_features import _HN_SCALE
from .records import load_records, load_truth
from .train_gbdt import lgb_params, s1_valid_mask
from .utils import (LOG, is_done, list_parts, load_json, mark_done, part_path, reset_dir,
                    save_json, stage, write_df)

ORPHAN_FEATS = {
    "feat": ["n_tset", "n_core_ratio", "ad_tset", "hn_eq", "hn_logdiff", "hn_single_sub",
             "hn_substr", "b_addr_empty", "b_is_domain", "b_has_alias", "b_name_script", "b_src",
             "n_extra_b_cnt", "n_extra_b_maxidf", "a_name_family", "b_name_amb", "st_ratio",
             "city_eq", "state_eq", "n_idf_jacc", "ad_idf_jacc"],
    "ctx": ["g_n", "g_hn_a", "g_hn_b", "g_hn_b_x", "g_hn_a_src", "g_name_max", "g_addr_max",
            "q_ncand", "q_n_hn_eq"],
    "coll": ["q_pa_max_other", "q_pa_margin", "q_pa_sum", "s_pa_sum_other", "s_pa_max_other",
             "psup_a", "psup_b", "conf_name", "conf_addr", "conf_hn_eq", "pa"],
}


def _lookup_sums(keys_all: np.ndarray, w_all: np.ndarray, keys_q: np.ndarray) -> np.ndarray:
    if len(keys_all) == 0:
        return np.zeros(len(keys_q), dtype=np.float64)
    uniq, inv = np.unique(keys_all, return_inverse=True)
    sums = np.bincount(inv, weights=w_all, minlength=len(uniq))
    pos = np.minimum(np.searchsorted(uniq, keys_q), len(uniq) - 1)
    return np.where(uniq[pos] == keys_q, sums[pos], 0.0)


def collective_features(s: np.ndarray, q: np.ndarray, pa: np.ndarray, q_hn: np.ndarray,
                        a_hn: np.ndarray, q_name: pl.Series, q_addr: pl.Series, q_src: np.ndarray,
                        n_s1: int, nt: int) -> dict[str, np.ndarray]:
    s64 = s.astype(np.int64)
    pa64 = pa.astype(np.float64)
    f: dict[str, np.ndarray] = {"pa": pa.astype(np.float32)}
    # ---- competition inside the query
    dq = pl.DataFrame({"q": q, "p": pa}).with_columns(
        pl.col("p").max().over("q").alias("mx"),
        pl.col("p").sum().over("q").alias("sum"))
    dq = dq.with_columns(
        (pl.col("p") == pl.col("mx")).cast(pl.Int32).sum().over("q").alias("nmx"),
        pl.when(pl.col("p") < pl.col("mx")).then(pl.col("p")).otherwise(None)
        .max().over("q").alias("mx2"))
    other = dq.select(pl.when((pl.col("p") == pl.col("mx")) & (pl.col("nmx") == 1))
                      .then(pl.col("mx2")).otherwise(pl.col("mx"))).to_series().to_numpy()
    other = np.asarray(other, dtype=np.float64)
    f["q_pa_max_other"] = np.where(np.isnan(other), np.nan, other).astype(np.float32)
    f["q_pa_margin"] = (pa64 - np.nan_to_num(other, nan=0.0)).astype(np.float32)
    f["q_pa_sum"] = dq["sum"].to_numpy().astype(np.float32)
    f["q_pa_rank"] = (pl.DataFrame({"q": q, "p": pa})
                      .select(pl.col("p").rank("ordinal", descending=True).over("q"))
                      .to_series().to_numpy().astype(np.float32))
    # ---- mass attracted by s
    s_sum = np.bincount(s, weights=pa64, minlength=n_s1)
    f["s_pa_sum_other"] = (s_sum[s] - pa64).astype(np.float32)
    s_n50 = np.bincount(s, weights=(pa64 >= 0.5).astype(np.float64), minlength=n_s1)
    f["s_n50_other"] = (s_n50[s] - (pa64 >= 0.5)).astype(np.float32)
    # same-source mass: an S1 has only 1-5 S2 / 1-6 S3 matches, mostly 1-3
    key = s64 * 4 + q_src[q].astype(np.int64)
    ks_sum = np.bincount(key, weights=pa64, minlength=n_s1 * 4)
    f["s_pa_sum_other_src"] = (ks_sum[key] - pa64).astype(np.float32)
    ks_n50 = np.bincount(key, weights=(pa64 >= 0.5).astype(np.float64), minlength=n_s1 * 4)
    f["s_n50_other_src"] = (ks_n50[key] - (pa64 >= 0.5)).astype(np.float32)
    order = np.lexsort((-pa64, s64))
    ss = s64[order]
    first = np.r_[True, ss[1:] != ss[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    top1_q = np.full(n_s1, -1, dtype=np.int64)
    top1_v = np.full(n_s1, np.nan)
    top2_q = np.full(n_s1, -1, dtype=np.int64)
    top2_v = np.full(n_s1, np.nan)
    top1_q[ss[first]] = q[order][first]
    top1_v[ss[first]] = pa64[order][first]
    top2_q[ss[second]] = q[order][second]
    top2_v[ss[second]] = pa64[order][second]
    is_top = top1_q[s] == q
    f["s_pa_max_other"] = np.where(is_top, top2_v[s], top1_v[s]).astype(np.float32)
    conf_q = np.where(is_top, top2_q[s], top1_q[s])
    # ---- probability-weighted house-number support
    hq, ha = q_hn[q], a_hn[s]
    m = hq >= 0
    keys_all = s64[m] * _HN_SCALE + hq[m]
    w_all = pa64[m]
    sup_b = _lookup_sums(keys_all, w_all, s64 * _HN_SCALE + np.where(m, hq, 0))
    sup_a = _lookup_sums(keys_all, w_all, s64 * _HN_SCALE + np.where(ha >= 0, ha, 0))
    self_b = np.where(m, pa64, 0.0)
    self_a = np.where(m & (ha >= 0) & (hq == ha), pa64, 0.0)
    psup_b = np.where(m, sup_b - self_b, np.nan)
    psup_a = np.where(ha >= 0, sup_a - self_a, np.nan)
    denom = s_sum[s] - pa64
    with np.errstate(divide="ignore", invalid="ignore"):
        f["psup_a"] = psup_a.astype(np.float32)
        f["psup_b"] = psup_b.astype(np.float32)
        f["psup_a_share"] = np.where(denom > 1e-6, psup_a / denom, np.nan).astype(np.float32)
        f["psup_b_share"] = np.where(denom > 1e-6, psup_b / denom, np.nan).astype(np.float32)
    # ---- similarity to s's most confident other record
    has = conf_q >= 0
    conf_name = np.full(len(q), np.nan, dtype=np.float32)
    conf_addr = np.full(len(q), np.nan, dtype=np.float32)
    conf_hn = np.full(len(q), np.nan, dtype=np.float32)
    if has.any():
        qa, qb = q[has], conf_q[has]
        for series, out, scorer in ((q_name, conf_name, fuzz.ratio),
                                    (q_addr, conf_addr, fuzz.token_set_ratio)):
            a, b = series.gather(qa), series.gather(qb)
            v = process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=nt,
                               dtype=np.float32)
            v[((a.str.len_chars() == 0) | (b.str.len_chars() == 0)).to_numpy()] = np.nan
            out[has] = v
        ha_, hb_ = q_hn[qa], q_hn[qb]
        conf_hn[has] = np.where((ha_ >= 0) & (hb_ >= 0), (ha_ == hb_).astype(np.float32), np.nan)
    f["conf_name"], f["conf_addr"], f["conf_hn_eq"] = conf_name, conf_addr, conf_hn
    return f


def _country_groups(cand_parts, s_country) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    for i, p in enumerate(cand_parts):
        c = s_country[pl.read_parquet(p, columns=["s"], n_rows=1)["s"][0]]
        if groups and groups[-1][0] == c:
            groups[-1][1].append(i)
        else:
            groups.append((c, [i]))
    return groups


def run_collective(cfg: dict, split: str) -> None:
    out = split_dir(cfg, split)
    coll_dir = out / "coll"
    if is_done(coll_dir):
        LOG.info("collective[%s]: cached", split)
        return
    rep = report_dir(cfg)
    nt = int(cfg["runtime"]["n_threads"])
    cand_parts = list_parts(out / "cand")
    pred_parts = list_parts(out / "pred_A")
    assert len(cand_parts) == len(pred_parts), "run stage A predictions first"
    qrec = load_records(cfg, split, "q", ["hn_int", "hn", "name_core", "addr_norm", "src"])
    q_hn = np.where(qrec["hn"].str.len_chars().to_numpy() > 0, qrec["hn_int"].to_numpy(), -1
                    ).astype(np.int64)
    q_name, q_addr = qrec["name_core"], qrec["addr_norm"]
    q_src = qrec["src"].to_numpy().astype(np.int64)
    n_q = qrec.height
    del qrec
    s1rec = load_records(cfg, split, "s1", ["hn_int", "hn", "country"])
    a_hn = np.where(s1rec["hn"].str.len_chars().to_numpy() > 0, s1rec["hn_int"].to_numpy(), -1
                    ).astype(np.int64)
    s_country = s1rec["country"].to_numpy()
    n_s1 = s1rec.height
    del s1rec

    reset_dir(coll_dir)
    for country, idxs in _country_groups(cand_parts, s_country):
        with stage(f"collective[{split}] {country}", rep):
            cand = pl.concat([pl.read_parquet(cand_parts[i], columns=["q", "s"]) for i in idxs],
                             how="vertical")
            pa = pl.concat([pl.read_parquet(pred_parts[i], columns=["p_A"]) for i in idxs],
                           how="vertical")["p_A"].to_numpy().astype(np.float32)
            sizes = [pl.scan_parquet(cand_parts[i]).select(pl.len()).collect().item()
                     for i in idxs]
            f = collective_features(cand["s"].to_numpy().astype(np.int64),
                                    cand["q"].to_numpy().astype(np.int64), pa, q_hn, a_hn,
                                    q_name, q_addr, q_src, n_s1, nt)
            start = 0
            for k, i in enumerate(idxs):
                sl = slice(start, start + sizes[k])
                write_df(pl.DataFrame({key: val[sl] for key, val in f.items()}),
                         part_path(coll_dir, i))
                start += sizes[k]
            del cand, pa, f

    with stage(f"orphan model [{split}]", rep):
        p_match = orphan_model(cfg, split, n_q)
    for i, p in enumerate(list_parts(coll_dir)):
        q = pl.read_parquet(cand_parts[i], columns=["q"])["q"].to_numpy()
        df = pl.read_parquet(p).with_columns(pl.Series("p_match", p_match[q].astype(np.float32)))
        write_df(df, p)
    mark_done(coll_dir)


# ---------------------------------------------------------------------------------------------
# orphan model: P(query matches any S1)
# ---------------------------------------------------------------------------------------------

def _top_rows(cfg: dict, split: str) -> tuple[pl.DataFrame, list[str]]:
    out = split_dir(cfg, split)
    cols: list[str] = []
    frames = []
    for i, cp in enumerate(list_parts(out / "cand")):
        coll = pl.read_parquet(part_path(out / "coll", i))
        top = coll["q_pa_rank"].to_numpy() == 1
        rows = np.flatnonzero(top)
        parts = [pl.read_parquet(cp, columns=["q", "s"]).select(pl.all().gather(rows))]
        for d, names in ORPHAN_FEATS.items():
            src = coll if d == "coll" else pl.read_parquet(part_path(out / d, i), columns=names)
            parts.append(src.select(names).select(pl.all().gather(rows)))
        frames.append(pl.concat(parts, how="horizontal"))
    cols = [c for names in ORPHAN_FEATS.values() for c in names]
    return pl.concat(frames, how="vertical"), cols


def orphan_model(cfg: dict, split: str, n_q: int) -> np.ndarray:
    top, cols = _top_rows(cfg, split)
    X = np.column_stack([top[c].cast(pl.Float32).to_numpy() for c in cols]).astype(np.float32)
    q = top["q"].to_numpy()
    s = top["s"].to_numpy()
    params = lgb_params(cfg, "aux_lgb")
    mdir = models_dir(cfg)
    n_folds = int(cfg["model"]["n_folds"])
    p = np.full(n_q, np.nan, dtype=np.float32)
    if split == "train":
        tr = load_truth(cfg)
        y = (tr["q_true"][q] >= 0).astype(np.float32)
        fold = tr["s1_fold"][s]
        es = s1_valid_mask(cfg, len(tr["s1_fold"]))[s]
        pred = np.empty(len(q), dtype=np.float32)
        for k in range(n_folds):
            trn = (fold != k) & ~es
            val = (fold != k) & es
            dtr = lgb.Dataset(X[trn], y[trn], feature_name=cols, params=params)
            dva = lgb.Dataset(X[val], y[val], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=int(cfg["model"]["aux_rounds"]),
                          valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            b.save_model(str(mdir / f"orphan_fold{k}.txt"), num_iteration=b.best_iteration)
            sel = fold == k
            pred[sel] = b.predict(X[sel], num_threads=params["num_threads"])
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, pred))
        save_json({"features": cols, "oof_auc": auc}, mdir / "orphan_meta.json")
        save_json({"oof_auc": auc, "positive_rate": float(y.mean())},
                  report_dir(cfg) / "orphan_model.json")
        LOG.info("orphan model OOF AUC %.4f (match rate %.3f)", auc, y.mean())
    else:
        meta = load_json(mdir / "orphan_meta.json")
        assert meta["features"] == cols
        models = [lgb.Booster(model_file=str(mdir / f"orphan_fold{k}.txt"))
                  for k in range(n_folds)]
        pred = np.mean([m.predict(X, num_threads=params["num_threads"]) for m in models], axis=0)
    p[q] = pred
    return p
