"""Stage 6 — calibration, many-to-one assignment, S1 singleton model, macro-F0.5 decision rule.

1. calibrate stage-B probabilities with isotonic regression fitted on train OOF
2. every S2/S3 record keeps only its best S1 (each record matches at most one S1)
3. an S1-level model estimates P(S1 has no match at all) from its candidates (OOF on train)
4. rules tuned on OOF macro F0.5 over ALL train S1:
     T : keep assigned pairs with p >= tau and margin >= delta
     E : per S1 choose k maximising EF(k) = 1.25*sum_{i<=k} p_i / (0.25*(sum p + mhat) + k)
         against EF(0) = P(singleton)   (E0 uses prod(1 - p_i) instead of the model)
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .config import models_dir, report_dir, split_dir
from .evaluate import breakdown, macro_f05_pairs, pair_metrics
from .records import load_records, load_truth
from .train_gbdt import lgb_params, s1_valid_mask
from .utils import LOG, list_parts, load_json, save_json, stage

SINGLETON_FEATS = ["n_assigned", "n_assigned_50", "p_max_assigned", "p_2nd_assigned",
                   "p_sum_assigned", "margin_max", "p_max_any", "s_ncand", "a_name_family",
                   "a_addr_share", "a_hn_missing"]


def load_scored_pairs(cfg: dict, split: str, tag: str) -> pl.DataFrame:
    """Candidates with their stage predictions (parts are row-aligned, so no join is needed)."""
    out = split_dir(cfg, split)
    cols = ["pair_id", "q", "s"] + (["label"] if split == "train" else [])
    cand_parts = list_parts(out / "cand")
    pred_parts = list_parts(out / f"pred_{tag}")
    assert len(cand_parts) == len(pred_parts), f"pred_{tag} is not aligned with cand"
    frames = []
    for cp, pp in zip(cand_parts, pred_parts):
        c = pl.read_parquet(cp, columns=cols)
        p = pl.read_parquet(pp)
        if not (c["pair_id"] == p["pair_id"]).all():
            raise ValueError(f"{pp} is not row-aligned with {cp}")
        frames.append(c.with_columns(p[f"p_{tag}"].alias("p_raw")))
    return pl.concat(frames, how="vertical")


# ---------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------

def fit_calibration(p: np.ndarray, y: np.ndarray, seed: int, max_rows: int = 8_000_000) -> dict:
    if len(p) > max_rows:
        idx = np.random.default_rng(seed).choice(len(p), size=max_rows, replace=False)
        p, y = p[idx], y[idx]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p.astype(np.float64), y.astype(np.float64))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibration(cal: dict, p: np.ndarray) -> np.ndarray:
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"])).astype(np.float32)


def prior_shift(p: np.ndarray, prior_train: float, iters: int = 50) -> tuple[np.ndarray, float]:
    """Saerens-style EM re-estimation of the positive-pair prior on unlabeled data."""
    prior = prior_train
    q = p
    for _ in range(iters):
        a = (prior / prior_train) * p
        b = ((1 - prior) / (1 - prior_train)) * (1 - p)
        q = a / np.maximum(a + b, 1e-12)
        new = float(q.mean())
        if abs(new - prior) < 1e-6:
            break
        prior = new
    return q.astype(np.float32), prior


# ---------------------------------------------------------------------------------------------
# assignment + singleton model
# ---------------------------------------------------------------------------------------------

def assign(df: pl.DataFrame) -> pl.DataFrame:
    """Best S1 per query with its probability and margin over the runner-up."""
    aggs = [pl.col("s").first().alias("s"), pl.col("p").first().alias("p1"),
            pl.col("p").slice(1, 1).first().alias("p2")]
    if "label" in df.columns:
        aggs.append(pl.col("label").first().alias("label"))
    g = (df.sort(["q", "p"], descending=[False, True])
           .group_by("q", maintain_order=True).agg(aggs))
    return g.with_columns((pl.col("p1") - pl.col("p2").fill_null(0.0)).alias("margin"))


def singleton_table(cfg: dict, split: str, df: pl.DataFrame, asg: pl.DataFrame,
                    floor: float) -> np.ndarray:
    n_s1 = load_records(cfg, split, "s1", ["hn"]).height
    a = asg.filter(pl.col("p1") >= floor)
    per_s = (a.sort(["s", "p1"], descending=[False, True]).group_by("s").agg(
        pl.len().alias("n_assigned"),
        (pl.col("p1") >= 0.5).sum().alias("n_assigned_50"),
        pl.col("p1").first().alias("p_max_assigned"),
        pl.col("p1").slice(1, 1).first().alias("p_2nd_assigned"),
        pl.col("p1").sum().alias("p_sum_assigned"),
        pl.col("margin").max().alias("margin_max")))
    any_s = df.group_by("s").agg(pl.col("p").max().alias("p_max_any"),
                                 pl.len().alias("s_ncand"))
    X = np.full((n_s1, len(SINGLETON_FEATS)), np.nan, dtype=np.float32)
    X[:, 0] = 0
    X[:, 1] = 0
    X[:, 7] = 0
    col = {c: i for i, c in enumerate(SINGLETON_FEATS)}
    s_idx = per_s["s"].to_numpy()
    for c in ("n_assigned", "n_assigned_50", "p_max_assigned", "p_2nd_assigned",
              "p_sum_assigned", "margin_max"):
        X[s_idx, col[c]] = per_s[c].cast(pl.Float32).to_numpy()
    s_idx = any_s["s"].to_numpy()
    X[s_idx, col["p_max_any"]] = any_s["p_max_any"].cast(pl.Float32).to_numpy()
    X[s_idx, col["s_ncand"]] = any_s["s_ncand"].cast(pl.Float32).to_numpy()
    from .features import ambiguity_arrays

    s1 = load_records(cfg, split, "s1", ["hn"])
    amb = ambiguity_arrays(cfg, split, s1)
    X[:, col["a_name_family"]] = np.log1p(amb["a_name_family"])
    X[:, col["a_addr_share"]] = np.log1p(amb["a_addr_share"])
    X[:, col["a_hn_missing"]] = (s1["hn"].str.len_chars() == 0).to_numpy().astype(np.float32)
    return X


def singleton_model(cfg: dict, split: str, X: np.ndarray, tag: str) -> np.ndarray:
    params = lgb_params(cfg, "aux_lgb")
    n_folds = int(cfg["model"]["n_folds"])
    mdir = models_dir(cfg)
    if split == "train":
        tr = load_truth(cfg)
        y = (tr["s_ntrue"] == 0).astype(np.float32)
        fold = tr["s1_fold"]
        es = s1_valid_mask(cfg, len(fold))
        pred = np.empty(len(y), dtype=np.float32)
        for k in range(n_folds):
            trn, val = (fold != k) & ~es, (fold != k) & es
            dtr = lgb.Dataset(X[trn], y[trn], feature_name=SINGLETON_FEATS, params=params)
            dva = lgb.Dataset(X[val], y[val], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=int(cfg["model"]["aux_rounds"]),
                          valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            b.save_model(str(mdir / f"singleton_{tag}_fold{k}.txt"), num_iteration=b.best_iteration)
            pred[fold == k] = b.predict(X[fold == k], num_threads=params["num_threads"])
        LOG.info("singleton model OOF: %s (singleton rate %.4f)", pair_metrics(y, pred), y.mean())
        return pred
    models = [lgb.Booster(model_file=str(mdir / f"singleton_{tag}_fold{k}.txt"))
              for k in range(n_folds)]
    return np.mean([m.predict(X, num_threads=params["num_threads"]) for m in models], axis=0
                   ).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------------------------

class EFGroups:
    """Sorted per-S1 structure of assigned items, reused for every ``mhat`` / EF(0) choice."""

    def __init__(self, s: np.ndarray, p: np.ndarray):
        self.n = len(s)
        if self.n == 0:
            return
        self.order = np.lexsort((-p, s))
        s_o, p_o = s[self.order], p[self.order].astype(np.float64)
        self.starts = np.r_[0, np.flatnonzero(np.diff(s_o)) + 1]
        self.lens = np.diff(np.r_[self.starts, len(s_o)])
        self.k = np.arange(len(s_o)) - np.repeat(self.starts, self.lens) + 1
        cs = np.cumsum(p_o)
        self.within = cs - np.repeat(cs[self.starts] - p_o[self.starts], self.lens)
        self.total = np.repeat(self.within[self.starts + self.lens - 1], self.lens)
        self.group_s = s_o[self.starts]

    def select(self, ef0_by_s: np.ndarray, mhat: float) -> np.ndarray:
        keep = np.zeros(self.n, dtype=bool)
        if self.n == 0:
            return keep
        ef = 1.25 * self.within / (0.25 * (self.total + mhat) + self.k)
        best = np.maximum.reduceat(ef, self.starts)
        is_best = ef >= np.repeat(best, self.lens) - 1e-12
        kstar = np.minimum.reduceat(np.where(is_best, self.k, np.iinfo(np.int64).max),
                                    self.starts)
        take = best > ef0_by_s[self.group_s]
        keep[self.order] = (self.k <= np.repeat(kstar, self.lens)) & np.repeat(take, self.lens)
        return keep


def ef_select(s: np.ndarray, p: np.ndarray, ef0_by_s: np.ndarray, mhat: float) -> np.ndarray:
    """Keep mask for the expected-F0.5 rule on assigned items (s, p)."""
    return EFGroups(s, p).select(ef0_by_s, mhat)


def _prod_ef0(s: np.ndarray, p: np.ndarray, n_s1: int) -> np.ndarray:
    logs = np.bincount(s, weights=np.log(np.clip(1.0 - p.astype(np.float64), 1e-9, 1.0)),
                       minlength=n_s1)
    return np.exp(logs)


def apply_rule(rule: dict, s: np.ndarray, p: np.ndarray, margin: np.ndarray,
               p_single: np.ndarray, floor: float) -> np.ndarray:
    base = (p >= max(floor, rule.get("item_floor", 0.0))) & (margin >= rule["delta"])
    if rule["kind"] == "T":
        return base & (p >= rule["tau"])
    keep = np.zeros(len(s), dtype=bool)
    idx = np.flatnonzero(base)
    ef0 = p_single if rule["kind"] == "E" else _prod_ef0(s[idx], p[idx], len(p_single))
    keep[idx] = ef_select(s[idx], p[idx], ef0, rule["mhat"])
    return keep


def tune(cfg: dict, asg: pl.DataFrame, p_single: np.ndarray, n_true: np.ndarray) -> tuple[dict, list]:
    d = cfg["decision"]
    s = asg["s"].to_numpy().astype(np.int64)
    p = asg["p1"].to_numpy().astype(np.float32)
    margin = asg["margin"].to_numpy().astype(np.float32)
    lab = asg["label"].to_numpy().astype(np.float64)
    floor = float(d["floor"])
    n_s1 = len(n_true)
    results = []
    for delta in d["margins"]:
        base = (p >= floor) & (margin >= delta)
        for tau in d["taus"]:
            keep = base & (p >= tau)
            results.append({"kind": "T", "delta": delta, "tau": tau,
                            "score": macro_f05_pairs(s[keep], lab[keep], n_true)})
        for phi in d["item_floors"]:
            idx = np.flatnonzero(base & (p >= phi))
            groups = EFGroups(s[idx], p[idx])
            ef0_prod = _prod_ef0(s[idx], p[idx], n_s1)
            for mhat in d["mhat"]:
                for kind, ef0 in (("E", p_single), ("E0", ef0_prod)):
                    sel = idx[groups.select(ef0, mhat)]
                    results.append({"kind": kind, "delta": delta, "item_floor": phi,
                                    "mhat": mhat,
                                    "score": macro_f05_pairs(s[sel], lab[sel], n_true)})
    results.sort(key=lambda x: -x["score"])
    LOG.info("decision tuning: best %s", results[0])
    return results[0], results


# ---------------------------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------------------------

def run_decide_train(cfg: dict, tag: str = "B") -> dict:
    rep = report_dir(cfg)
    mdir = models_dir(cfg)
    tr = load_truth(cfg)
    with stage("decide[train] calibrate + assign", rep):
        df = load_scored_pairs(cfg, "train", tag)
        y = df["label"].to_numpy()
        cal = fit_calibration(df["p_raw"].to_numpy(), y, int(cfg["runtime"]["seed"]))
        save_json(cal, mdir / f"calibration_{tag}.json")
        df = df.with_columns(pl.Series("p", apply_calibration(cal, df["p_raw"].to_numpy())))
        pm = pair_metrics(y, df["p"].to_numpy())
        asg = assign(df)
    with stage("decide[train] singleton model", rep):
        X = singleton_table(cfg, "train", df, asg, float(cfg["decision"]["floor"]))
        p_single = singleton_model(cfg, "train", X, tag)
    with stage("decide[train] tune rule", rep):
        best, results = tune(cfg, asg, p_single, tr["s_ntrue"])
    s = asg["s"].to_numpy().astype(np.int64)
    keep = apply_rule(best, s, asg["p1"].to_numpy(), asg["margin"].to_numpy(), p_single,
                      float(cfg["decision"]["floor"]))
    s1 = load_records(cfg, "train", "s1", ["country"])
    countries = s1["country"].to_numpy()
    n_true = tr["s_ntrue"]
    groups = {f"country={c}": countries == c for c in sorted(set(countries.tolist()))}
    groups.update({"singletons": n_true == 0, "has_matches": n_true > 0,
                   "n_true=1": n_true == 1, "n_true=2-3": (n_true >= 2) & (n_true <= 3),
                   "n_true>=4": n_true >= 4})
    lab = asg["label"].to_numpy()
    summary = {
        "tag": tag, "rule": best, "oof_macro_f05": best["score"],
        "breakdown": breakdown(s[keep], lab[keep], n_true, groups),
        "pair_metrics_calibrated": pm,
        "kept_pairs": int(keep.sum()), "kept_precision": float(lab[keep].mean()),
        "kept_recall_of_all_true": float(lab[keep].sum() / max(int((tr["q_true"] >= 0).sum()), 1)),
        "top_rules": results[:15],
        "pair_prior_train": float(y.mean()),
        "assigned_p_hist": np.histogram(asg["p1"].to_numpy(), bins=10, range=(0, 1))[0].tolist(),
    }
    save_json(best, mdir / f"decision_{tag}.json")
    save_json(summary, rep / f"decision_train_{tag}.json")
    LOG.info("OOF macro F0.5 = %.5f | %s", best["score"], summary["breakdown"])
    _error_samples(cfg, asg, keep, int(cfg["report"]["n_examples"]))
    return summary


def decide_test(cfg: dict, tag: str = "B") -> pl.DataFrame:
    """Kept (s, q, p) pairs for the test split, using the rule tuned on train."""
    mdir = models_dir(cfg)
    df = load_scored_pairs(cfg, "test", tag)
    cal = load_json(mdir / f"calibration_{tag}.json")
    df = df.with_columns(pl.Series("p", apply_calibration(cal, df["p_raw"].to_numpy())))
    if cfg["decision"].get("prior_shift", False):
        prior_train = load_json(report_dir(cfg) / f"decision_train_{tag}.json")["pair_prior_train"]
        p_adj, prior = prior_shift(df["p"].to_numpy(), prior_train)
        LOG.info("prior shift: train %.4f -> test %.4f", prior_train, prior)
        df = df.with_columns(pl.Series("p", p_adj))
    asg = assign(df)
    X = singleton_table(cfg, "test", df, asg, float(cfg["decision"]["floor"]))
    p_single = singleton_model(cfg, "test", X, tag)
    rule = load_json(mdir / f"decision_{tag}.json")
    keep = apply_rule(rule, asg["s"].to_numpy().astype(np.int64), asg["p1"].to_numpy(),
                      asg["margin"].to_numpy(), p_single, float(cfg["decision"]["floor"]))
    kept = asg.filter(pl.Series(keep)).select(["s", "q", pl.col("p1").alias("p")])
    _test_stats(cfg, df, asg, kept, tag)
    return kept


def _test_stats(cfg: dict, df: pl.DataFrame, asg: pl.DataFrame, kept: pl.DataFrame,
                tag: str) -> None:
    s1 = load_records(cfg, "test", "s1", ["country"])
    q = load_records(cfg, "test", "q", ["country"])
    sc = s1["country"].to_numpy()
    qc = q["country"].to_numpy()
    n_kept = np.bincount(kept["s"].to_numpy(), minlength=s1.height)
    q_top = np.full(q.height, np.nan)
    q_top[asg["q"].to_numpy()] = asg["p1"].to_numpy()
    stats = {}
    for c in sorted(set(sc.tolist())):
        m = sc == c
        mq = qc == c
        stats[c] = {"n_s1": int(m.sum()), "share_s1_with_match": float((n_kept[m] > 0).mean()),
                    "mean_matches_per_s1": float(n_kept[m].mean()),
                    "n_queries": int(mq.sum()),
                    "share_queries_assigned": float(np.isin(np.flatnonzero(mq),
                                                            kept["q"].to_numpy()).mean()),
                    "mean_top_p": float(np.nanmean(q_top[mq])),
                    "share_top_p_ge_0.5": float(np.nanmean(q_top[mq] >= 0.5))}
    stats["pair_mean_p"] = float(df["p"].mean())
    save_json(stats, report_dir(cfg) / f"decision_test_{tag}.json")
    LOG.info("test prediction stats: %s", stats)


def _error_samples(cfg: dict, asg: pl.DataFrame, keep: np.ndarray, n: int) -> None:
    """Markdown dump of false positives / false negatives on train OOF."""
    from .records import load_raw

    tr = load_truth(cfg)
    rs = load_raw(cfg, "train", "s1", ["entity_id", "business_name", "business_address"])
    rq = load_raw(cfg, "train", "q", ["entity_id", "business_name", "business_address"])
    rng = np.random.default_rng(0)
    lab = asg["label"].to_numpy()
    s = asg["s"].to_numpy()
    q = asg["q"].to_numpy()
    p = asg["p1"].to_numpy()
    lines = ["# OOF error samples", ""]

    def row(si, qi, pp, tag):
        a = rs.row(int(si))
        b = rq.row(int(qi))
        return f"- **{tag}** p={pp:.3f}\n  - S1 `{a[0]}` {a[1]!r} | {a[2]!r}\n  - Q  `{b[0]}` {b[1]!r} | {b[2]!r}"

    fp = np.flatnonzero(keep & (lab == 0))
    lines.append(f"## False positives ({len(fp)} total)")
    for i in rng.choice(fp, size=min(n, len(fp)), replace=False) if len(fp) else []:
        lines.append(row(s[i], q[i], p[i], "FP"))
    fn = np.flatnonzero(~keep & (lab == 1))
    lines.append(f"\n## Rejected true pairs ({len(fn)} total)")
    for i in rng.choice(fn, size=min(n, len(fn)), replace=False) if len(fn) else []:
        lines.append(row(s[i], q[i], p[i], "FN"))
    wrong = np.flatnonzero((lab == 0) & (tr["q_true"][q] >= 0))
    lines.append(f"\n## Queries whose best candidate is the wrong S1 ({len(wrong)} total)")
    for i in rng.choice(wrong, size=min(n, len(wrong)), replace=False) if len(wrong) else []:
        lines.append(row(s[i], q[i], p[i], "WRONG-S1"))
        t = tr["q_true"][q[i]]
        a = rs.row(int(t))
        lines.append(f"  - true S1 `{a[0]}` {a[1]!r} | {a[2]!r}")
    (report_dir(cfg) / "errors_train.md").write_text("\n".join(lines), encoding="utf-8")
