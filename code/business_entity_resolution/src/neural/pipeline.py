"""Optional cross-encoder: 2-fold cross-fitting over S1 halves, scores the uncertain band.

export  -> work/neural/{train_half0,train_half1,score_train,score_test}.parquet
train   -> work/neural/ce_half{h}/   (model trained on S1 half h scores train pairs of half 1-h)
infer   -> work/<split>/nn/part-*.parquet  (ce_score, NaN outside the band), row-aligned with cand
The model sees raw "name | address" strings (native scripts kept) so it can learn
transliteration; default model paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params).
Requires: pip install -r requirements-neural.txt
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from ..config import split_dir, work_dir
from ..records import load_raw, load_truth
from ..utils import LOG, list_parts, mark_done, part_path, reset_dir, write_df


def _neural_dir(cfg: dict) -> Path:
    d = work_dir(cfg) / "neural"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _texts(cfg: dict, split: str, side: str) -> pl.Series:
    raw = load_raw(cfg, split, side, ["business_name", "business_address"])
    return raw.select((pl.col("business_name").str.strip_chars() + " | "
                       + pl.col("business_address").str.strip_chars()).alias("t"))["t"]


def _pairs(cfg: dict, split: str) -> pl.DataFrame:
    out = split_dir(cfg, split)
    frames = []
    for i, cp in enumerate(list_parts(out / "cand")):
        cols = ["pair_id", "q", "s"] + (["label"] if split == "train" else [])
        c = pl.read_parquet(cp, columns=cols)
        a = pl.read_parquet(part_path(out / "pred_A", i), columns=["p_A"])
        r = pl.read_parquet(part_path(out / "coll", i), columns=["q_pa_rank"])
        frames.append(pl.concat([c, a, r], how="horizontal"))
    return pl.concat(frames, how="vertical")


def export_pairs(cfg: dict) -> None:
    n = cfg["neural"]
    nd = _neural_dir(cfg)
    seed = int(cfg["runtime"]["seed"])
    for split in ("train", "test"):
        df = _pairs(cfg, split)
        ta, tb = _texts(cfg, split, "s1"), _texts(cfg, split, "q")
        band = ((pl.col("q_pa_rank") <= int(n["top_per_query"]))
                & (pl.col("p_A") >= float(n["band_low"])) & (pl.col("p_A") <= float(n["band_high"])))
        score = df.filter(band)
        if split == "train":
            half = load_truth(cfg)["s1_half"]
            score = score.with_columns(pl.Series("half", half[score["s"].to_numpy()]))
            train_pool = df.filter((pl.col("label") == 1) | (pl.col("q_pa_rank") <= 3))
            train_pool = train_pool.with_columns(pl.Series("half",
                                                           half[train_pool["s"].to_numpy()]))
            for h in (0, 1):
                t = train_pool.filter(pl.col("half") == h)
                cap = int(n["train_pairs_per_half"])
                if t.height > cap:
                    t = t.sample(n=cap, seed=seed + h)
                t = t.with_columns(ta.gather(t["s"].to_numpy()).alias("text_a"),
                                   tb.gather(t["q"].to_numpy()).alias("text_b"))
                write_df(t.select(["text_a", "text_b", "label"]), nd / f"train_half{h}.parquet")
                LOG.info("neural export: train_half%d %d pairs (%.3f positive)", h, t.height,
                         t["label"].mean())
        score = score.with_columns(ta.gather(score["s"].to_numpy()).alias("text_a"),
                                   tb.gather(score["q"].to_numpy()).alias("text_b"))
        keep = ["pair_id", "text_a", "text_b"] + (["half"] if split == "train" else [])
        write_df(score.select(keep), nd / f"score_{split}.parquet")
        LOG.info("neural export: score_%s %d pairs", split, score.height)


def _device(cfg: dict):
    import torch

    want = cfg["neural"].get("device", "auto")
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_half(cfg: dict, h: int) -> Path:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    n = cfg["neural"]
    nd = _neural_dir(cfg)
    dev = _device(cfg)
    torch.manual_seed(int(cfg["runtime"]["seed"]) + h)
    data = pl.read_parquet(nd / f"train_half{h}.parquet").sample(fraction=1.0, shuffle=True,
                                                                 seed=int(cfg["runtime"]["seed"]))
    tok = AutoTokenizer.from_pretrained(n["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(n["model_name"], num_labels=1)
    model.to(dev)
    bs = int(n["batch_size"])
    steps = math.ceil(data.height / bs) * int(n["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(n["lr"]), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * float(n["warmup_frac"])), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_amp = dev.type == "cuda"
    a, b = data["text_a"].to_list(), data["text_b"].to_list()
    y = data["label"].to_numpy().astype(np.float32)
    model.train()
    step = 0
    for epoch in range(int(n["epochs"])):
        for i in range(0, data.height, bs):
            enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=int(n["max_len"]),
                      padding=True, return_tensors="pt").to(dev)
            target = torch.tensor(y[i:i + bs], device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                LOG.info("ce half %d epoch %d step %d/%d loss %.4f", h, epoch, step, steps,
                         loss.item())
    path = nd / f"ce_half{h}"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _score(cfg: dict, path: Path, a: list[str], b: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    n = cfg["neural"]
    dev = _device(cfg)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    bs = int(n["infer_batch_size"])
    out = np.empty(len(a), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(a), bs):
            enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=int(n["max_len"]),
                      padding=True, return_tensors="pt").to(dev)
            out[i:i + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy()
            if (i // bs) % 2000 == 0:
                LOG.info("ce scoring %d/%d", i, len(a))
    return out


def infer_all(cfg: dict) -> None:
    nd = _neural_dir(cfg)
    for split in ("train", "test"):
        sc = pl.read_parquet(nd / f"score_{split}.parquet")
        a, b = sc["text_a"].to_list(), sc["text_b"].to_list()
        if split == "train":
            half = sc["half"].to_numpy()
            ce = np.full(sc.height, np.nan, dtype=np.float32)
            for h in (0, 1):
                sel = np.flatnonzero(half == 1 - h)  # model h never saw S1 half 1-h
                ce[sel] = _score(cfg, nd / f"ce_half{h}", [a[i] for i in sel],
                                 [b[i] for i in sel])
        else:
            ce = np.mean([_score(cfg, nd / f"ce_half{h}", a, b) for h in (0, 1)], axis=0)
        lookup = pl.DataFrame({"pair_id": sc["pair_id"], "ce_score": ce.astype(np.float32)})
        nn_dir = reset_dir(split_dir(cfg, split) / "nn")
        for i, cp in enumerate(list_parts(split_dir(cfg, split) / "cand")):
            pid = pl.read_parquet(cp, columns=["pair_id"])
            part = pid.join(lookup, on="pair_id", how="left").sort("pair_id")
            write_df(part, part_path(nn_dir, i))
        mark_done(nn_dir)
        LOG.info("ce scores written for %s", split)


def run_neural(cfg: dict) -> None:
    nd = _neural_dir(cfg)
    if not (nd / "score_test.parquet").exists():
        export_pairs(cfg)
    for h in (0, 1):
        if not (nd / f"ce_half{h}").exists():
            train_half(cfg, h)
    if not list_parts(split_dir(cfg, "test") / "nn"):
        infer_all(cfg)
