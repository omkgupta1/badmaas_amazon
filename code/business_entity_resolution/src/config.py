"""Configuration: YAML defaults + optional override file + CLI ``key=value`` overrides.

Relative paths in the ``paths`` section are resolved against the package root
(``code/business_entity_resolution/``) so the pipeline can be launched from anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PKG_ROOT / "configs" / "default.yaml"


def _deep_update(base: dict, upd: dict) -> dict:
    for key, val in (upd or {}).items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    """Load ``configs/default.yaml``, merge ``path`` on top, then apply ``key=value`` overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = PKG_ROOT / p
        if p.resolve() != DEFAULT_CONFIG.resolve():
            with open(p, encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
    for key, val in list(cfg["paths"].items()):
        p = Path(os.path.expanduser(str(val)))
        if not p.is_absolute():
            p = (PKG_ROOT / p).resolve()
        cfg["paths"][key] = str(p)
    return cfg


def work_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["work_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_dir(cfg: dict, split: str) -> Path:
    """Cache directory of one split (``train`` or ``test``)."""
    d = work_dir(cfg) / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(cfg: dict) -> Path:
    d = work_dir(cfg) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(cfg: dict) -> Path:
    d = work_dir(cfg) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lexicon_dir(cfg: dict) -> Path:
    d = work_dir(cfg) / "lexicon"
    d.mkdir(parents=True, exist_ok=True)
    return d
