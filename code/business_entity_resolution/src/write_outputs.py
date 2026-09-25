"""Stage 7 — write output/matching_results.tsv and output/candidate_pairs.tsv, then validate."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from .config import report_dir, split_dir
from .decide import decide_test
from .io_utils import write_grouped_tsv
from .records import load_raw
from .utils import LOG, list_parts, save_json, stage


def run_write(cfg: dict, tag: str = "B") -> dict:
    rep = report_dir(cfg)
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with stage("write[test] decide", rep):
        kept = decide_test(cfg, tag)
    with stage("write[test] tsv files", rep):
        s1_ids = load_raw(cfg, "test", "s1", ["entity_id"])["entity_id"].to_list()
        q_ids = np.array(load_raw(cfg, "test", "q", ["entity_id"])["entity_id"].to_list(),
                         dtype=object)
        n_match = write_grouped_tsv(out_dir / "matching_results.tsv", "matched_entity_ids",
                                    s1_ids, kept["s"].to_numpy(), kept["q"].to_numpy(), q_ids,
                                    order_key=kept["p"].to_numpy())
        cand = pl.concat([pl.read_parquet(p, columns=["s", "q", "s0"])
                          for p in list_parts(split_dir(cfg, "test") / "cand")], how="vertical")
        n_cand = write_grouped_tsv(out_dir / "candidate_pairs.tsv", "candidate_entity_ids",
                                   s1_ids, cand["s"].to_numpy(), cand["q"].to_numpy(), q_ids,
                                   order_key=cand["s0"].to_numpy())
    info = {"s1_rows": len(s1_ids), "s1_with_matches": n_match, "matched_pairs": kept.height,
            "s1_with_candidates": n_cand, "candidate_pairs": cand.height}
    LOG.info("outputs: %s", info)
    info["validator"] = validate(cfg)
    save_json(info, rep / "outputs.json")
    return info


def validate(cfg: dict) -> dict:
    validator = Path(cfg["paths"]["validator"])
    out_dir = Path(cfg["paths"]["out_dir"])
    if not validator.exists():
        LOG.warning("validator not found at %s — skipping", validator)
        return {"ran": False}
    cmd = [sys.executable, str(validator),
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"),
           "--test-dir", str(Path(cfg["paths"]["data_dir"]) / "test"), "--check-ids"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    LOG.info("validator (exit %d):\n%s%s", res.returncode, res.stdout, res.stderr)
    return {"ran": True, "exit_code": res.returncode, "stdout": res.stdout[-4000:]}
