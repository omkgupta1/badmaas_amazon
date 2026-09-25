"""Stage-0 ranker: a light LightGBM over cheap union features that keeps the top-M S1
candidates per query.

Cross-fitted so that no train score is in-sample: the union parts are split into two halves by
part parity, one booster is trained on (a query sample of) each half, and part ``i`` is always
scored by the booster of the *other* half (``1 - i % 2``). Test parts use the same rule, so train
and test ``s0`` come from the same distribution — ``s0`` feeds the matching models and the
context features, so an in-sample train score would be a leak. M is chosen on out-of-sample
scores as the smallest value that loses at most ``max_recall_loss`` of the union's true pairs.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import UNION_FEATURES
from .config import models_dir, report_dir
from .utils import LOG, list_parts, load_json, part_path, save_json, write_df

MODEL_FILE = "prerank_{}.txt"
META_FILE = "prerank.json"


class Ranker:
    def __init__(self, boosters: list[lgb.Booster], m: int, n_threads: int):
        self.boosters = boosters
        self.m = int(m)
        self.n_threads = n_threads

    def booster_for(self, part: int) -> lgb.Booster:
        return self.boosters[(1 - part % 2) % len(self.boosters)]

    def score(self, df: pl.DataFrame, part: int) -> np.ndarray:
        X = df.select([pl.col(c).cast(pl.Float32) for c in UNION_FEATURES]).to_numpy()
        return self.booster_for(part).predict(X, num_threads=self.n_threads).astype(np.float32)


def cut_top_m(ranker: Ranker, df: pl.DataFrame, part: int) -> pl.DataFrame:
    df = df.with_columns(pl.Series("s0", ranker.score(df, part)))
    df = df.with_columns(pl.col("s0").rank("ordinal", descending=True).over("q")
                         .cast(pl.Int16).alias("s0_rank"))
    return df.filter(pl.col("s0_rank") <= ranker.m).sort(["q", "s0_rank"])


def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.select([pl.col(c).cast(pl.Float32) for c in UNION_FEATURES]).to_numpy()
    return X, df["label"].to_numpy().astype(np.float32)


def _n_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _sample(parts: list[Path], halves: list[int], p: dict, rng: np.random.Generator):
    """One pass over the union parts: per half, a training sample (negatives thinned so the
    half stays under ``max_train_rows``) and an evaluation sample of complete query groups."""
    frac = float(p["train_query_frac"])
    eval_frac = float(p.get("eval_query_frac", 0.05))
    cols = ["q", "label"] + UNION_FEATURES
    rows_half = [0, 0]
    for path, h in zip(parts, halves):
        rows_half[h] += _n_rows(path)
    neg_rate = [min(1.0, float(p["max_train_rows"]) / max(frac * r, 1.0)) for r in rows_half]
    train: list[list[pl.DataFrame]] = [[], []]
    evals: list[list[pl.DataFrame]] = [[], []]
    for path, h in zip(parts, halves):
        df = pl.read_parquet(path, columns=cols)
        uq = df["q"].unique().to_numpy()
        r = rng.random(len(uq))
        tr = _with_queries(df, uq[r < frac])
        if neg_rate[h] < 1.0:
            keep = (tr["label"].to_numpy() == 1) | (rng.random(tr.height) < neg_rate[h])
            tr = tr.filter(pl.Series(keep))
        train[h].append(tr)
        evals[h].append(_with_queries(df, uq[(r >= frac) & (r < frac + eval_frac)]))
    return ([pl.concat(x, how="vertical") for x in train],
            [pl.concat(x, how="vertical") for x in evals])


def _with_queries(df: pl.DataFrame, qs: np.ndarray, anti: bool = False) -> pl.DataFrame:
    """Rows of ``df`` whose query is (``anti``: is not) in ``qs`` — a join, not ``is_in``."""
    keys = pl.DataFrame({"q": qs}).cast({"q": df.schema["q"]})
    return df.join(keys, on="q", how="anti" if anti else "semi")


def _fit(tr: pl.DataFrame, p: dict, nt: int, seed: int) -> lgb.Booster:
    uq = tr["q"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    vq = uq[rng.random(len(uq)) < 0.15]
    va = _with_queries(tr, vq)
    tr = _with_queries(tr, vq, anti=True)
    Xt, yt = _xy(tr)
    Xv, yv = _xy(va)
    LOG.info("stage-0 ranker: train %d rows (%d pos), early-stopping valid %d rows", len(yt),
             int(yt.sum()), len(yv))
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63,
              "min_data_in_leaf": 100, "feature_fraction": 0.9, "bagging_fraction": 0.8,
              "bagging_freq": 1, "num_threads": nt, "verbose": -1, "seed": seed}
    dtrain = lgb.Dataset(Xt, yt, feature_name=UNION_FEATURES, free_raw_data=True)
    dvalid = lgb.Dataset(Xv, yv, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=int(p["num_boost_round"]),
                     valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)])


def train_prerank(cfg: dict, union_dir: Path) -> Ranker:
    p = cfg["blocking"]["prerank"]
    seed = int(cfg["runtime"]["seed"])
    nt = int(cfg["runtime"]["n_threads"])
    parts = list_parts(union_dir)
    rng = np.random.default_rng(seed)
    cross = len(parts) >= 2
    # half of part i = i % 2 (the parity rule of Ranker.booster_for); a single part -> one model
    halves = [i % 2 for i in range(len(parts))] if cross else [0] * len(parts)
    train, evals = _sample(parts, halves, p, rng)

    boosters, ranks_oos, pos_oos = [], [], []
    for h in ((0, 1) if cross else (0,)):
        booster = _fit(train[h], p, nt, seed + h)
        boosters.append(booster)
        ev = evals[1 - h] if cross else evals[0]  # out-of-sample when cross-fitted
        if ev.height == 0:
            continue
        s = booster.predict(_xy(ev)[0], num_threads=nt)
        ranks = (ev.select("q").with_columns(pl.Series("s0", s))
                 .select(pl.col("s0").rank("ordinal", descending=True).over("q"))
                 .to_series().to_numpy())
        ranks_oos.append(ranks)
        pos_oos.append(ev["label"].to_numpy() == 1)
    del train

    ranks = np.concatenate(ranks_oos) if ranks_oos else np.zeros(0)
    pos = np.concatenate(pos_oos) if pos_oos else np.zeros(0, dtype=bool)
    n_pos = max(int(pos.sum()), 1)
    max_m = int(p["max_m"])
    curve = {m: float(((ranks <= m) & pos).sum() / n_pos) for m in range(1, max_m + 1)}
    ok = [m for m in range(int(p["min_m"]), max_m + 1)
          if 1.0 - curve[m] <= float(p["max_recall_loss"])]
    ranker = Ranker(boosters, min(ok) if ok else max_m, nt)
    LOG.info("stage-0 ranker (%s): out-of-sample recall-within-union curve %s -> M = %d",
             "cross-fitted" if cross else "single model",
             {k: round(v, 5) for k, v in curve.items()}, ranker.m)

    mdir = models_dir(cfg)
    for h, b in enumerate(boosters):
        b.save_model(str(mdir / MODEL_FILE.format(h)))
    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    meta = {"m": ranker.m, "n_models": len(boosters), "cross_fitted": cross,
            "recall_within_union": curve, "n_eval_pos": int(pos.sum()),
            "best_iteration": [int(b.best_iteration) for b in boosters],
            "importance": dict(zip(UNION_FEATURES, gain.round(1).tolist()))}
    save_json(meta, mdir / META_FILE)
    save_json(meta, report_dir(cfg) / "prerank.json")
    return ranker


def load_prerank(cfg: dict) -> Ranker | None:
    mdir = models_dir(cfg)
    if not (mdir / META_FILE).exists():
        return None
    meta = load_json(mdir / META_FILE)
    n = int(meta.get("n_models", 1))
    files = [mdir / MODEL_FILE.format(h) for h in range(n)]
    if not all(f.exists() for f in files):
        return None
    return Ranker([lgb.Booster(model_file=str(f)) for f in files], meta["m"],
                  int(cfg["runtime"]["n_threads"]))


def apply_prerank(cfg: dict, ranker: Ranker, union_dir: Path, cand_dir: Path) -> int:
    offset, n_written = 0, 0
    for i, p in enumerate(list_parts(union_dir)):
        df = cut_top_m(ranker, pl.read_parquet(p), part=i)
        if df.height == 0:
            continue
        df = df.with_columns(pl.int_range(offset, offset + df.height, dtype=pl.Int64)
                             .alias("pair_id"))
        offset += df.height
        write_df(df, part_path(cand_dir, n_written))
        n_written += 1
    LOG.info("stage-0 cut: %d candidate pairs kept (M=%d)", offset, ranker.m)
    return offset
