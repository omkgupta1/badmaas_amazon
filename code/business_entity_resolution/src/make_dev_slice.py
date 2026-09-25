"""Build a ~5% development dataset that keeps the hard structure of the full data.

S1 records are selected by city (stable hash of the city component), so name families, shared
addresses and decoys of the selected cities stay together. S2/S3 records are kept when they
match a selected S1 (train) or, for unmatched records, when one of their address components is
a selected city. The result mirrors dataset/ (train/*.tsv, test/*.tsv) under work/dev_data/.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import polars as pl

from .io_utils import read_ground_truth, read_source
from .utils import LOG

KEEP_MOD = 20  # 1 / 20 of the cities


def _comps(col: str) -> pl.Expr:
    return (pl.col(col).str.to_lowercase().str.split(",")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(~pl.element().str.contains(r"\d")
                                           & (pl.element().str.len_chars() >= 3))))


def _city_key() -> pl.Expr:
    c = _comps("business_address")
    return pl.when(c.list.len() >= 2).then(c.list.get(-2, null_on_oob=True)) \
        .otherwise(c.list.last()).fill_null("")


def _selected(key: str) -> bool:
    return bool(key) and zlib.crc32(key.encode("utf-8")) % KEEP_MOD == 0


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never")
    LOG.info("wrote %s (%d rows)", path, df.height)


def _slice_split(src_dir: str, dst: Path, split: str) -> None:
    s1 = read_source(src_dir, split, 1)
    s1 = s1.with_columns(_city_key().alias("_city"))
    s1 = s1.with_columns(pl.col("_city").map_elements(_selected, return_dtype=pl.Boolean)
                         .alias("_keep"))
    cities = set(s1.filter(pl.col("_keep"))["_city"].to_list())
    s1_keep = s1.filter(pl.col("_keep")).drop(["_city", "_keep"])
    matched_keep: set[str] = set()
    matched_all: set[str] = set()
    if split == "train":
        gt = read_ground_truth(src_dir)
        matched_all = set(gt["q_id"].to_list())
        keep_ids = set(s1_keep["entity_id"].to_list())
        gt_keep = gt.filter(pl.col("s1_id").is_in(list(keep_ids)))
        matched_keep = set(gt_keep["q_id"].to_list())
        gt_rows = (s1_keep.select(pl.col("entity_id").alias("s1_id"))
                   .join(gt_keep.group_by("s1_id").agg(pl.col("q_id").str.join(",")),
                         on="s1_id", how="left")
                   .with_columns(pl.col("q_id").fill_null(""))
                   .rename({"s1_id": "source1_entity_id", "q_id": "matched_entity_ids"}))
        _write(gt_rows, dst / split / f"{split}_ground_truth.tsv")
    _write(s1_keep, dst / split / f"{split}_source1.tsv")
    city_list = list(cities)
    for src in (2, 3):
        q = read_source(src_dir, split, src)
        in_city = _comps("business_address").list.eval(pl.element().is_in(city_list)).list.any()
        empty = pl.col("business_address").str.strip_chars() == ""
        sample_empty = pl.col("entity_id").map_elements(
            lambda x: zlib.crc32(x.encode("utf-8")) % KEEP_MOD == 0, return_dtype=pl.Boolean)
        if split == "train":
            ids = pl.col("entity_id")
            keep = ids.is_in(list(matched_keep)) | (~ids.is_in(list(matched_all))
                                                     & (in_city | (empty & sample_empty)))
        else:
            keep = in_city | (empty & sample_empty)
        _write(q.filter(keep.fill_null(False)), dst / split / f"{split}_source{src}.tsv")


def make_dev_slice(cfg: dict) -> Path:
    src_dir = cfg["paths"]["data_dir"]
    dst = Path(cfg["paths"]["work_dir"]) / "dev_data"
    for split in ("train", "test"):
        _slice_split(src_dir, dst, split)
    LOG.info("dev slice ready in %s — run with --config configs/dev.yaml", dst)
    return dst
