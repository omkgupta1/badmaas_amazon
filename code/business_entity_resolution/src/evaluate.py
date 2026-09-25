"""Metrics that reproduce the leaderboard scorer exactly.

Per S1 entity: F0.5 = 1.25*P*R / (0.25*P + R) = 1.25*TP / (0.25*|truth| + |pred|);
an entity with no true matches scores 1.0 for an empty prediction and 0.0 otherwise; an entity
with true matches and an empty prediction scores 0.0. The macro average runs over ALL S1.
"""
from __future__ import annotations

import numpy as np


def f05_per_s1(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    tp = tp.astype(np.float64)
    n_pred = n_pred.astype(np.float64)
    n_true = n_true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((n_true > 0) & (n_pred > 0), 1.25 * tp / (0.25 * n_true + n_pred), 0.0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, f)


def macro_f05_pairs(s_kept: np.ndarray, label_kept: np.ndarray, n_true: np.ndarray) -> float:
    """Macro F0.5 from kept (s, label) pairs; ``n_true`` has one entry per S1 (all of them)."""
    n = len(n_true)
    tp = np.bincount(s_kept, weights=label_kept.astype(np.float64), minlength=n)
    n_pred = np.bincount(s_kept, minlength=n)
    return float(f05_per_s1(tp, n_pred, n_true).mean())


def macro_f05_sets(pred: dict[str, set], truth: dict[str, set], s1_ids: list[str]) -> float:
    """Reference implementation on id sets (used by tests)."""
    scores = []
    for sid in s1_ids:
        p, t = pred.get(sid, set()), truth.get(sid, set())
        if not t and not p:
            scores.append(1.0)
        elif not t or not p:
            scores.append(0.0)
        else:
            tp = len(p & t)
            prec, rec = tp / len(p), tp / len(t)
            scores.append(0.0 if tp == 0 else 1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(scores)) if scores else 0.0


def breakdown(s_kept: np.ndarray, label_kept: np.ndarray, n_true: np.ndarray,
              groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Macro F0.5 restricted to boolean S1 masks (e.g. per country, singletons)."""
    n = len(n_true)
    tp = np.bincount(s_kept, weights=label_kept.astype(np.float64), minlength=n)
    n_pred = np.bincount(s_kept, minlength=n)
    f = f05_per_s1(tp, n_pred, n_true)
    return {name: float(f[mask].mean()) for name, mask in groups.items() if mask.any()}


def pair_metrics(y: np.ndarray, p: np.ndarray, max_rows: int = 5_000_000,
                 seed: int = 0) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
        y, p = y[idx], p[idx]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if y.min() == y.max():
        return {"n": int(len(y))}
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, p)), "n": int(len(y))}
