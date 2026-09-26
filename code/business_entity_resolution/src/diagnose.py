"""Read-only diagnostics.

``shift``: adversarial validation on the top stage-0 candidate of each query. For train (all) vs
test France, train US vs test US and train India vs test India, a LightGBM learns to tell the two
slices apart; its AUC says how different they are and its top features say where (with NaN rates
and means of both sides). Report: work/reports/shift.json (+ log).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from .config import report_dir, split_dir
from .records import load_records
from .utils import LOG, list_parts, load_json, save_json

_DROP = {"q", "s", "label", "pair_id"}


def _top1(cfg: dict, split: str, per_part: int, seed: int) -> pl.DataFrame:
    out = split_dir(cfg, split)
    parts = {d: list_parts(out / d) for d in ("cand", "feat", "ctx")}
    qc = load_records(cfg, split, "q", ["country"])["country"].to_numpy()
    n_real = int(load_json(out / "records_q" / "_DONE")["rows"])  # skips synthetic twins
    rng = np.random.default_rng(seed)
    frames = []
    for i, cp in enumerate(parts["cand"]):
        c = pl.read_parquet(cp)
        q = c["q"].to_numpy()
        rows = np.flatnonzero((c["s0_rank"].to_numpy() == 1) & (q < n_real))
        if len(rows) > per_part:
            rows = np.sort(rng.choice(rows, size=per_part, replace=False))
        take = [c.select([x for x in c.columns if x not in _DROP]).select(pl.all().gather(rows))]
        for d in ("feat", "ctx"):
            f = pl.read_parquet(parts[d][i])
            take.append(f.select([x for x in f.columns if x not in _DROP])
                        .select(pl.all().gather(rows)))
        df = pl.concat(take, how="horizontal")
        frames.append(df.with_columns(pl.Series("_country", qc[q[rows]])))
    return pl.concat(frames, how="vertical_relaxed")


def _compare(a: pl.DataFrame, b: pl.DataFrame, name: str, cfg: dict, cap: int = 300_000) -> dict:
    rng = np.random.default_rng(0)
    n = min(a.height, b.height, cap)
    a = a.sample(n, seed=1)
    b = b.sample(n, seed=2)
    cols = [c for c in a.columns if c != "_country" and c in b.columns]
    X = np.vstack([a.select(cols).cast(pl.Float32).to_numpy(),
                   b.select(cols).cast(pl.Float32).to_numpy()])
    y = np.r_[np.zeros(n), np.ones(n)].astype(np.float32)
    tr = rng.random(len(y)) < 0.7
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "verbose": -1,
              "num_threads": int(cfg["runtime"]["n_threads"]), "seed": 0}
    booster = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=cols), 200)
    auc = float(roc_auc_score(y[~tr], booster.predict(X[~tr])))
    gain = booster.feature_importance("gain")
    top = [cols[i] for i in np.argsort(-gain)[:20]]
    rows = []
    for c in top:
        va, vb = a[c].cast(pl.Float64).to_numpy(), b[c].cast(pl.Float64).to_numpy()
        rows.append({"feature": c, "gain_share": float(gain[cols.index(c)] / gain.sum()),
                     "nan_a": float(np.isnan(va).mean()), "nan_b": float(np.isnan(vb).mean()),
                     "mean_a": float(np.nanmean(va)) if (~np.isnan(va)).any() else None,
                     "mean_b": float(np.nanmean(vb)) if (~np.isnan(vb)).any() else None})
    LOG.info("shift %s: AUC %.3f (0.5 = no shift)", name, auc)
    for r in rows[:15]:
        LOG.info("   %-22s gain %5.1f%% | NaN %.3f -> %.3f | mean %s -> %s", r["feature"],
                 100 * r["gain_share"], r["nan_a"], r["nan_b"],
                 None if r["mean_a"] is None else round(r["mean_a"], 3),
                 None if r["mean_b"] is None else round(r["mean_b"], 3))
    return {"auc": auc, "n_per_side": n, "top": rows}


def run_shift(cfg: dict) -> dict:
    tr = _top1(cfg, "train", 40_000, 0)
    te = _top1(cfg, "test", 40_000, 1)
    ne = pl.col("b_addr_empty") == 0
    rep = {"state_eq_nan_nonempty_addr": {
        "train": float(tr.filter(ne)["state_eq"].is_nan().mean()),
        **{f"test {c}": float(te.filter(ne & (pl.col("_country") == c))["state_eq"].is_nan().mean())
           for c in sorted(set(te["_country"].to_list()))}}}
    LOG.info("state_eq NaN share among non-empty addresses: %s", rep["state_eq_nan_nonempty_addr"])
    rep["train_vs_test_France"] = _compare(tr, te.filter(pl.col("_country") == "France"),
                                           "train vs test France", cfg)
    for c in ("US", "India"):
        rep[f"train_vs_test_{c}"] = _compare(tr.filter(pl.col("_country") == c),
                                             te.filter(pl.col("_country") == c),
                                             f"train {c} vs test {c}", cfg)
    save_json(rep, report_dir(cfg) / "shift.json")
    return rep


def agreement_stats(d: pl.DataFrame, n_real: int) -> pl.DataFrame:
    """Per synthetic / real: summed |stage-0 score - pair feature| and hn agreement counts."""
    both_nan = pl.col("hn_eq0").is_nan() & pl.col("hn_eq").is_nan()
    agree = ((pl.col("hn_eq0") == pl.col("hn_eq")) | both_nan).fill_null(False)
    return d.with_columns((pl.col("q") >= n_real).alias("synthetic")).group_by("synthetic").agg(
        (pl.col("name_ratio0") - pl.col("n_ratio")).abs().fill_nan(0).sum().alias("d_name"),
        (pl.col("addr_tset0") - pl.col("ad_tset")).abs().fill_nan(0).sum().alias("d_addr"),
        agree.cast(pl.Int64).sum().alias("hn_agree"),
        pl.len().alias("n"))


def run_aug_check(cfg: dict) -> dict:
    """Giveaway check for synthetic train records: on real rows the stage-0 fuzzy scores equal
    the pair features computed from the same strings (name_ratio0 = n_ratio, addr_tset0 = ad_tset,
    hn_eq0 = hn_eq); synthetic rows must agree just as well, or a model can spot them."""
    out = split_dir(cfg, "train")
    n_real = int(load_json(out / "records_q" / "_DONE")["rows"])
    stats = []
    for c, f in zip(list_parts(out / "cand"), list_parts(out / "feat")):
        d = pl.concat([pl.read_parquet(c, columns=["q", "name_ratio0", "addr_tset0", "hn_eq0"]),
                       pl.read_parquet(f, columns=["n_ratio", "ad_tset", "hn_eq"])],
                      how="horizontal")
        stats.append(agreement_stats(d, n_real))
    t = pl.concat(stats).group_by("synthetic").sum().sort("synthetic")
    rep = {("synthetic" if r["synthetic"] else "real"): {
        "rows": int(r["n"]), "mean_abs_name_ratio0_minus_n_ratio": r["d_name"] / r["n"],
        "mean_abs_addr_tset0_minus_ad_tset": r["d_addr"] / r["n"],
        "hn_eq0_agrees_with_hn_eq": r["hn_agree"] / r["n"]} for r in t.iter_rows(named=True)}
    LOG.info("aug-check: %s", rep)
    save_json(rep, report_dir(cfg) / "aug_check.json")
    return rep


def run_gates(cfg: dict, tag: str = "B") -> dict:
    """Offline gates for a finished run: accepted matches per S1 split by house-number relation,
    train (real queries) vs test per country, plus the share of synthetic train decoys accepted.
    Test should look like train (e.g. US 'hn differs' near its train value)."""
    from .postprocess import attach_features, kept_pairs

    n_real = int(load_json(split_dir(cfg, "train") / "records_q" / "_DONE")["rows"])
    rep: dict = {}
    for split in ("train", "test"):
        k = attach_features(cfg, split, kept_pairs(cfg, split, tag))
        sc = load_records(cfg, split, "s1", ["country"])["country"].to_numpy()
        ks = sc[k["s"].to_numpy()]
        syn = k["q"].to_numpy() >= n_real if split == "train" else np.zeros(k.height, bool)
        hn = k["hn_eq"].to_numpy()
        for c in sorted(set(sc.tolist())):
            n = float((sc == c).sum())
            m = (ks == c) & ~syn
            row = {"kept_per_s1": m.sum() / n, "hn_equal": (m & (hn == 1)).sum() / n,
                   "hn_differs": (m & (hn == 0)).sum() / n,
                   "hn_missing": (m & np.isnan(hn)).sum() / n}
            if split == "train":
                row["synthetic_accepted_per_s1"] = ((ks == c) & syn).sum() / n
            rep[f"{split} {c}"] = {key: round(float(v), 4) for key, v in row.items()}
            LOG.info("gates %-14s %s", f"{split} {c}", rep[f"{split} {c}"])
    save_json(rep, report_dir(cfg) / f"gates_{tag}.json")
    return rep
