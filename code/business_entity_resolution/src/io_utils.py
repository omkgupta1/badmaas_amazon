"""Reading the challenge TSVs and writing the two submission TSVs.

All inputs are read with quoting disabled: a few hundred records contain a literal ``"``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .utils import LOG

COLS = ["entity_id", "business_name", "business_address", "country"]


def source_path(data_dir: str | Path, split: str, src: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{src}.tsv"


def count_data_lines(path: Path) -> int:
    """Number of data lines (excluding the header), robust to a missing trailing newline."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 24), b""):
            n += buf.count(b"\n")
            last = buf[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def read_tsv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        infer_schema_length=0,  # every column as String
        encoding="utf8",
    )
    return df.with_columns([pl.col(c).fill_null("") for c in df.columns])


def read_source(data_dir: str | Path, split: str, src: int) -> pl.DataFrame:
    path = source_path(data_dir, split, src)
    df = read_tsv(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.select(COLS).with_columns(pl.col("entity_id").str.strip_chars())
    expected = count_data_lines(path)
    if df.height != expected:
        LOG.warning("%s: parsed %d rows but the file has %d data lines", path.name, df.height, expected)
    else:
        LOG.info("%s: %d rows", path.name, df.height)
    return df


def read_ground_truth(data_dir: str | Path) -> pl.DataFrame:
    """Ground-truth pairs as a (s1_id, q_id) table, one row per matched S2/S3 record."""
    path = Path(data_dir) / "train" / "train_ground_truth.tsv"
    gt = read_tsv(path).rename({"source1_entity_id": "s1_id", "matched_entity_ids": "ids"})
    pairs = (
        gt.with_columns(pl.col("ids").str.split(","))
        .explode("ids")
        .with_columns(pl.col("ids").str.strip_chars())
        .filter(pl.col("ids").str.len_chars() > 0)
        .rename({"ids": "q_id"})
        .select(["s1_id", "q_id"])
    )
    LOG.info("ground truth: %d S1 rows, %d matched pairs", gt.height, pairs.height)
    return pairs


def write_grouped_tsv(
    path: Path,
    value_header: str,
    s1_ids: list[str],
    s_idx: np.ndarray,
    q_idx: np.ndarray,
    q_ids: np.ndarray,
    order_key: np.ndarray | None = None,
) -> int:
    """Write one row per S1 (in ``s1_ids`` order) with the comma-joined ids of its pairs.

    ``s_idx``/``q_idx`` are integer pair arrays, ``q_ids`` maps q index -> entity id string
    (numpy object array). Pairs are listed by descending ``order_key`` when given.
    Returns the number of non-empty rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_s1 = len(s1_ids)
    s_idx = np.asarray(s_idx, dtype=np.int64)
    q_idx = np.asarray(q_idx, dtype=np.int64)
    if order_key is None:
        order = np.lexsort((q_idx, s_idx))
    else:
        order = np.lexsort((-np.asarray(order_key, dtype=np.float64), s_idx))
    s_sorted = s_idx[order]
    q_sorted = q_idx[order]
    bounds = np.searchsorted(s_sorted, np.arange(n_s1 + 1))
    non_empty = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{value_header}\n")
        for i in range(n_s1):
            a, b = bounds[i], bounds[i + 1]
            if a == b:
                f.write(f"{s1_ids[i]}\t\n")
            else:
                ids = q_ids[q_sorted[a:b]]
                # a q appears at most once per S1 by construction; dedupe defensively
                seen = dict.fromkeys(ids.tolist())
                f.write(s1_ids[i] + "\t" + ",".join(seen) + "\n")
                non_empty += 1
    return non_empty
