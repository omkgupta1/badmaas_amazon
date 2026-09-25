"""Leave-one-country-out check — the proxy for the unseen test country (France).

For every ordered pair (src, dst) of training countries: train one stage-A model on src pairs
only, score all dst pairs, assign each query to its best S1 and report dst macro F0.5 for a
grid of thresholds (plus the threshold that is best on src itself, i.e. what we would ship).
Use it to compare feature / normalisation choices by how well they transfer across countries.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import report_dir, split_dir
from .decide import assign
from .evaluate import f05_per_s1
from .records import load_records, load_truth
from .train_gbdt import (_matrix, column_map, lgb_params, part_meta, read_features,
                         sample_plan, stage_dirs)
from .utils import LOG, list_parts, save_json, stage

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _scores(asg: pl.DataFrame, n_true: np.ndarray, mask_s1: np.ndarray) -> dict[float, float]:
    s = asg["s"].to_numpy()
    p = asg["p1"].to_numpy()
    lab = asg["label"].to_numpy().astype(np.float64)
    out = {}
    for tau in TAUS:
        keep = p >= tau
        tp = np.bincount(s[keep], weights=lab[keep], minlength=len(n_true))
        npred = np.bincount(s[keep], minlength=len(n_true))
        out[tau] = float(f05_per_s1(tp, npred, n_true)[mask_s1].mean())
    return out


def _predict_country(booster, dirs, cmap, cols, metas, s_country, country, nt) -> pl.DataFrame:
    frames = []
    for i, m in enumerate(metas):
        rows = np.flatnonzero(s_country[m["s"].to_numpy()] == country)
        if len(rows) == 0:
            continue
        X = read_features(dirs, cmap, cols, i, rows)
        frames.append(m.select(pl.all().gather(rows)).with_columns(
            pl.Series("p", booster.predict(X, num_threads=nt).astype(np.float32))))
    return pl.concat(frames, how="vertical")


def run_loco(cfg: dict) -> dict:
    rep = report_dir(cfg)
    nt = int(cfg["runtime"]["n_threads"])
    dirs = stage_dirs(cfg, "train", "A")
    cmap = column_map(dirs)
    cols = list(cmap)
    metas = part_meta(cfg, "train")
    plan = sample_plan(cfg, metas)
    s_country = load_records(cfg, "train", "s1", ["country"])["country"].to_numpy()
    for pp, m in zip(plan, metas):
        pp["country"] = s_country[m["s"].to_numpy()[pp["rows"]]]
    countries = sorted(set(s_country.tolist()))
    n_true = load_truth(cfg)["s_ntrue"]
    params = lgb_params(cfg)
    results: dict = {}
    for src in countries:
        with stage(f"loco train on {src}", rep):
            Xt, yt, wt = _matrix(dirs, cmap, cols, plan, lambda pp: (pp["country"] == src) & ~pp["es"])
            Xv, yv, wv = _matrix(dirs, cmap, cols, plan, lambda pp: (pp["country"] == src) & pp["es"])
            dtr = lgb.Dataset(Xt, yt, weight=wt, feature_name=cols, params=params)
            dva = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
            booster = lgb.train(params, dtr, num_boost_round=int(cfg["model"]["num_boost_round"]),
                                valid_sets=[dva],
                                callbacks=[lgb.early_stopping(
                                    int(cfg["model"]["early_stopping_rounds"]), verbose=False)])
            del Xt, yt, wt, Xv, yv, wv
        for dst in countries:
            with stage(f"loco {src} -> {dst}", rep):
                asg = assign(_predict_country(booster, dirs, cmap, cols, metas, s_country, dst, nt))
                sc = _scores(asg, n_true, s_country == dst)
                results[f"{src}->{dst}"] = sc
                LOG.info("LOCO %s -> %s: %s", src, dst, sc)
    for src in countries:
        own = results[f"{src}->{src}"]
        tau = max(own, key=own.get)
        for dst in countries:
            if dst != src:
                results[f"{src}->{dst}"]["shipped_tau"] = tau
                results[f"{src}->{dst}"]["shipped_f05"] = results[f"{src}->{dst}"][tau]
    save_json({str(k): v for k, v in results.items()}, rep / "loco.json")
    return results
