"""Collect the JSON reports of every stage into work/reports/summary.md (paste-able)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import report_dir
from .utils import LOG


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(obj, max_len: int = 3000) -> str:
    s = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    return s if len(s) <= max_len else s[:max_len] + "\n... (truncated)"


def build_summary(cfg: dict) -> Path:
    rep = report_dir(cfg)
    lines = ["# Business entity resolution — run summary", ""]
    t = rep / "timings.jsonl"
    if t.exists():
        lines += ["## Timings", "", "| stage | ok | minutes | peak RSS GB |", "|---|---|---|---|"]
        for line in t.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            lines.append(f"| {r['stage']} | {r['ok']} | {r['minutes']} | {r['peak_rss_gb']} |")
        lines.append("")
    sections = [
        ("Prep coverage (train)", "prep_train.json"),
        ("Prep coverage (test)", "prep_test.json"),
        ("Stage-0 ranker", "prerank.json"),
        ("Blocking (train)", "blocking_train.json"),
        ("Blocking (test)", "blocking_test.json"),
        ("Orphan model", "orphan_model.json"),
        ("Decision on train OOF (stage A)", "decision_train_A.json"),
        ("Decision on train OOF (stage B)", "decision_train_B.json"),
        ("Test prediction stats", "decision_test_B.json"),
        ("Outputs", "outputs.json"),
        ("Leave-one-country-out", "loco.json"),
    ]
    for title, name in sections:
        obj = _load(rep / name)
        if obj is not None:
            lines += [f"## {title}", "", "```json", _fmt(obj), "```", ""]
    for tag in ("A", "B"):
        obj = _load(rep / f"model_{tag}.json")
        if obj is not None:
            top = obj.get("importance_gain", [])[:30]
            lines += [f"## Model {tag}: best iterations {obj.get('best_iterations')}", "",
                      "| feature | gain |", "|---|---|"]
            lines += [f"| {f} | {g} |" for f, g in top]
            lines.append("")
    path = rep / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("summary written to %s", path)
    return path
