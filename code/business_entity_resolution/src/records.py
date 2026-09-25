"""Loading cached parsed records and ground-truth arrays of a split."""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import split_dir
from .utils import list_parts


def load_records(cfg: dict, split: str, side: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Parsed records of ``side`` ('s1' or 'q'), ordered by ``idx`` (= row position)."""
    d = split_dir(cfg, split) / f"records_{side}"
    parts = list_parts(d)
    if not parts:
        raise FileNotFoundError(f"{d} is empty — run `prep --split {split}` first")
    cols = None if columns is None else list(dict.fromkeys(["idx"] + columns))
    df = pl.concat([pl.read_parquet(p, columns=cols) for p in parts], how="vertical")
    return df


def load_raw(cfg: dict, split: str, side: str, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(split_dir(cfg, split) / f"raw_{side}.parquet", columns=columns)


def load_truth(cfg: dict) -> dict[str, np.ndarray]:
    d = split_dir(cfg, "train")
    return {
        "q_true": np.load(d / "q_true.npy"),
        "s_ntrue": np.load(d / "s_ntrue.npy"),
        "s1_fold": np.load(d / "s1_fold.npy"),
        "s1_half": np.load(d / "s1_half.npy"),
    }
