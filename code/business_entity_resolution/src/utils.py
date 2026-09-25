"""Logging, stage timing / memory accounting, JSON and parquet-part helpers."""
from __future__ import annotations

import json
import logging
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

LOG = logging.getLogger("ber")


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in LOG.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        LOG.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


def rss_gb() -> float:
    """Current resident memory of this process in GB (NaN if psutil is missing)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # pragma: no cover
        return float("nan")


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2


@contextmanager
def stage(name: str, report_dir: Path | None = None):
    """Log wall time and memory of a pipeline stage; append it to ``timings.jsonl``."""
    t0 = time.time()
    LOG.info("▶ %s", name)
    ok = False
    try:
        yield
        ok = True
    finally:
        minutes = (time.time() - t0) / 60
        LOG.info("%s %s: %.1f min | rss %.1f GB | peak %.1f GB",
                 "✔" if ok else "✘", name, minutes, rss_gb(), peak_rss_gb())
        if report_dir is not None:
            rec = {"stage": name, "ok": ok, "minutes": round(minutes, 2),
                   "peak_rss_gb": round(peak_rss_gb(), 2),
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(Path(report_dir) / "timings.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------------------------
# parquet "part" directories: a table stored as ordered files part-00000.parquet, part-00001...
# ---------------------------------------------------------------------------------------------

def part_path(directory: Path, i: int) -> Path:
    return directory / f"part-{i:05d}.parquet"


def list_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob("part-*.parquet"))


def reset_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for p in directory.glob("part-*.parquet"):
        p.unlink()
    return directory


def read_parts(directory: Path, columns: list[str] | None = None) -> pl.DataFrame:
    parts = list_parts(directory)
    if not parts:
        raise FileNotFoundError(f"no parquet parts in {directory}")
    return pl.concat([pl.read_parquet(p, columns=columns) for p in parts], how="vertical")


def write_df(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3)


def done_marker(directory: Path) -> Path:
    return directory / "_DONE"


def is_done(directory: Path) -> bool:
    return done_marker(directory).exists()


def mark_done(directory: Path, info: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(done_marker(directory), "w", encoding="utf-8") as f:
        json.dump(info or {}, f, default=_json_default)


def chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(n, start + size)
