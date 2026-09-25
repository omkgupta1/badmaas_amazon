"""Stages 4/5 — LightGBM pair models (stage A: pairwise + context; stage B: + collective).

Feature tables are row-aligned parquet part directories (cand / feat / ctx / coll / nn).
Training rows are sampled per part with fixed seeds (all positives kept at ``pos_frac``,
hard negatives = stage-0 rank <= ``hard_rank``, easy negatives) and inverse-probability
weighted, so the model stays calibrated to the real candidate distribution. Folds are
GroupKFold by S1 record; a fixed subset of training-fold S1 records drives early stopping.
Train predictions are out-of-fold; test predictions average the fold models.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import models_dir, report_dir, split_dir
from .records import load_truth
from .utils import (LOG, list_parts, load_json, mark_done, part_path, reset_dir, save_json, stage,
                    write_df)

META_COLS = {"pair_id", "q", "s", "label"}


def stage_dirs(cfg: dict, split: str, stage_name: str) -> list[Path]:
    out = split_dir(cfg, split)
    dirs = [out / "cand", out / "feat", out / "ctx"]
    if stage_name == "B":
        dirs.append(out / "coll")
        if cfg["neural"]["enabled"] and list_parts(out / "nn"):
            dirs.append(out / "nn")
    return dirs


def column_map(dirs: list[Path], exclude: set[str] | None = None) -> dict[str, int]:
    """feature column -> index of the directory it is read from (first occurrence wins)."""
    exclude = META_COLS | (exclude or set())
    cmap: dict[str, int] = {}
    for di, d in enumerate(dirs):
        parts = list_parts(d)
        if not parts:
            raise FileNotFoundError(f"{d} has no parts")
        for c in pl.read_parquet_schema(parts[0]):
            if c not in exclude and c not in cmap:
                cmap[c] = di
    return cmap


def read_features(dirs: list[Path], cmap: dict[str, int], cols: list[str], i: int,
                  rows: np.ndarray | None = None) -> np.ndarray:
    """float32 matrix of ``cols`` for part ``i`` (optionally only ``rows``)."""
    frames = []
    for di, d in enumerate(dirs):
        want = [c for c in cols if cmap[c] == di]
        if want:
            frames.append(pl.read_parquet(part_path(d, i), columns=want))
    df = pl.concat(frames, how="horizontal").select(cols)
    if rows is not None:
        df = df.select(pl.all().gather(rows))
    X = np.empty((df.height, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = df[c].cast(pl.Float32).to_numpy()
    return X


def part_meta(cfg: dict, split: str) -> list[pl.DataFrame]:
    cols = ["pair_id", "q", "s", "s0_rank"] + (["label"] if split == "train" else [])
    return [pl.read_parquet(p, columns=cols) for p in list_parts(split_dir(cfg, split) / "cand")]


def s1_valid_mask(cfg: dict, n_s1: int) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["runtime"]["seed"]) + 7)
    return rng.random(n_s1) < float(cfg["model"]["valid_s1_frac"])


def sample_plan(cfg: dict, metas: list[pl.DataFrame]) -> list[dict[str, np.ndarray]]:
    """Per part: selected rows, inverse-probability weights, fold and early-stop flags."""
    sm = cfg["model"]["sample"]
    tr = load_truth(cfg)
    fold, valid_s = tr["s1_fold"], s1_valid_mask(cfg, len(tr["s1_fold"]))
    probs = []
    expected = 0.0
    for m in metas:
        y = m["label"].to_numpy()
        hard = m["s0_rank"].to_numpy() <= int(sm["hard_rank"])
        p = np.where(y == 1, sm["pos_frac"], np.where(hard, sm["hard_neg_frac"],
                                                      sm["easy_neg_frac"])).astype(np.float64)
        probs.append(p)
        expected += p.sum()
    scale = min(1.0, float(sm["max_rows"]) / max(expected, 1.0))
    plan = []
    for i, (m, p) in enumerate(zip(metas, probs)):
        p = np.clip(p * scale, 1e-6, 1.0)
        u = np.random.default_rng(int(cfg["runtime"]["seed"]) * 1000 + i).random(len(p))
        sel = u < p
        s = m["s"].to_numpy()
        plan.append({"rows": np.flatnonzero(sel), "w": (1.0 / p[sel]).astype(np.float32),
                     "y": m["label"].to_numpy()[sel].astype(np.float32),
                     "fold": fold[s[sel]], "es": valid_s[s[sel]]})
    LOG.info("training sample: %d rows (scale %.3f)", sum(len(x["rows"]) for x in plan), scale)
    return plan


def _matrix(dirs, cmap, cols, plan, keep_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = [int(keep_fn(pp).sum()) for pp in plan]
    X = np.empty((sum(sizes), len(cols)), dtype=np.float32)
    y = np.empty(sum(sizes), dtype=np.float32)
    w = np.empty(sum(sizes), dtype=np.float32)
    at = 0
    for i, pp in enumerate(plan):
        k = keep_fn(pp)
        n = int(k.sum())
        if n == 0:
            continue
        X[at:at + n] = read_features(dirs, cmap, cols, i, pp["rows"][k])
        y[at:at + n] = pp["y"][k]
        w[at:at + n] = pp["w"][k]
        at += n
    return X, y, w


def lgb_params(cfg: dict, key: str = "lgb") -> dict:
    p = dict(cfg["model"][key])
    p["num_threads"] = int(cfg["runtime"]["n_threads"])
    p["seed"] = int(cfg["runtime"]["seed"])
    return p


def train_stage(cfg: dict, stage_name: str, exclude: set[str] | None = None,
                tag: str | None = None) -> dict:
    """Train the K fold models of a stage on the train split; write OOF predictions."""
    tag = tag or stage_name
    rep = report_dir(cfg)
    dirs = stage_dirs(cfg, "train", stage_name)
    cmap = column_map(dirs, exclude)
    cols = list(cmap)
    metas = part_meta(cfg, "train")
    plan = sample_plan(cfg, metas)
    n_folds = int(cfg["model"]["n_folds"])
    params = lgb_params(cfg)
    mdir = models_dir(cfg)
    importance = np.zeros(len(cols))
    best_iters = []
    for k in range(n_folds):
        with stage(f"train[{tag}] fold {k}", rep):
            Xt, yt, wt = _matrix(dirs, cmap, cols, plan, lambda pp: (pp["fold"] != k) & ~pp["es"])
            dtrain = lgb.Dataset(Xt, yt, weight=wt, feature_name=cols, free_raw_data=True,
                                 params=params)
            dtrain.construct()
            del Xt, yt, wt
            Xv, yv, wv = _matrix(dirs, cmap, cols, plan, lambda pp: (pp["fold"] != k) & pp["es"])
            dvalid = lgb.Dataset(Xv, yv, weight=wv, reference=dtrain, free_raw_data=True)
            booster = lgb.train(
                params, dtrain, num_boost_round=int(cfg["model"]["num_boost_round"]),
                valid_sets=[dvalid], valid_names=["es"],
                callbacks=[lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]),
                                              verbose=False),
                           lgb.log_evaluation(200)])
            del Xv, yv, wv, dtrain, dvalid
            booster.save_model(str(mdir / f"{tag}_fold{k}.txt"),
                               num_iteration=booster.best_iteration)
            importance += booster.feature_importance("gain", iteration=booster.best_iteration)
            best_iters.append(int(booster.best_iteration))
            LOG.info("  fold %d: best iteration %d", k, booster.best_iteration)
    imp = sorted(zip(cols, (importance / n_folds).round(1).tolist()), key=lambda x: -x[1])
    meta = {"features": cols, "best_iterations": best_iters, "importance_gain": imp}
    save_json(meta, mdir / f"{tag}_meta.json")
    save_json(meta, rep / f"model_{tag}.json")
    with stage(f"predict[{tag}] train OOF", rep):
        predict_stage(cfg, "train", stage_name, tag)
    return meta


def load_models(cfg: dict, tag: str) -> tuple[list[lgb.Booster], list[str]]:
    mdir = models_dir(cfg)
    meta = load_json(mdir / f"{tag}_meta.json")
    models = [lgb.Booster(model_file=str(mdir / f"{tag}_fold{k}.txt"))
              for k in range(int(cfg["model"]["n_folds"]))]
    return models, meta["features"]


def predict_stage(cfg: dict, split: str, stage_name: str, tag: str | None = None) -> None:
    """Train: out-of-fold predictions. Test: average of the fold models."""
    tag = tag or stage_name
    models, cols = load_models(cfg, tag)
    dirs = stage_dirs(cfg, split, stage_name)
    cmap = column_map(dirs)
    missing = [c for c in cols if c not in cmap]
    if missing:
        raise ValueError(f"features missing for {split}: {missing[:10]}")
    out_dir = reset_dir(split_dir(cfg, split) / f"pred_{tag}")
    nt = int(cfg["runtime"]["n_threads"])
    fold = load_truth(cfg)["s1_fold"] if split == "train" else None
    for i, p in enumerate(list_parts(split_dir(cfg, split) / "cand")):
        meta = pl.read_parquet(p, columns=["pair_id", "s"])
        X = read_features(dirs, cmap, cols, i)
        if fold is not None:
            f = fold[meta["s"].to_numpy()]
            pred = np.empty(len(X), dtype=np.float32)
            for k, m in enumerate(models):
                sel = f == k
                if sel.any():
                    pred[sel] = m.predict(X[sel], num_threads=nt)
        else:
            pred = np.mean([m.predict(X, num_threads=nt) for m in models], axis=0)
        write_df(pl.DataFrame({"pair_id": meta["pair_id"], f"p_{tag}": pred.astype(np.float32)}),
                 part_path(out_dir, i))
    mark_done(out_dir)


def read_predictions(cfg: dict, split: str, tag: str) -> pl.DataFrame:
    d = split_dir(cfg, split) / f"pred_{tag}"
    return pl.concat([pl.read_parquet(p) for p in list_parts(d)], how="vertical")
