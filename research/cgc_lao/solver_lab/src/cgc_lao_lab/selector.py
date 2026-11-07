"""Acquisition policies. Pure functions of public logits, the learner's own volatile state and counts.

``choose_probe_early`` is a line-by-line numpy port of ``cgc.lao_q4e5.choose_probe_early`` (the
checked-in reference): sigmoid prediction, first-index argmax, ``-1e9`` masking of seen experts, probe of
the best unseen expert at steps 1 and 2, and deadline coverage. ``horizon`` defaults to the reference's 8.
"""
from __future__ import annotations

import numpy as np

from .env import K, T_PRIMARY, sigmoid

STATE_CAP = 1.5


def _check(t: int, horizon: int, pred: np.ndarray, counts: np.ndarray) -> None:
    if not (isinstance(t, int) and not isinstance(t, bool) and 0 <= t < horizon):
        raise ValueError("SELECT_STEP_OUT_OF_RANGE")
    if pred.ndim != 2 or pred.shape[1] != K or counts.shape != pred.shape:
        raise ValueError("SELECT_SHAPE_B3")
    if not (np.isfinite(pred).all() and np.isfinite(counts).all()):
        raise ValueError("SELECT_NONFINITE")
    if not ((counts >= 0).all() and (counts == np.floor(counts)).all() and (counts.sum(-1) == t).all()):
        raise ValueError("SELECT_COUNTS_DO_NOT_MATCH_STEP")


def probe_early_from_pred(t: int, pred: np.ndarray, counts: np.ndarray, horizon: int = T_PRIMARY) -> np.ndarray:
    """PROBE_EARLY given predicted success probabilities ``[E, 3]``."""
    _check(t, horizon, pred, counts)
    best = pred.argmax(-1)
    unseen = counts == 0
    need = unseen.sum(-1)
    unseen_best = np.where(unseen, pred, pred.dtype.type(-1e9)).argmax(-1)
    deadline_force = need >= (horizon - t)
    probe = (t in (1, 2)) & (need > 0)
    return np.where(deadline_force | probe, unseen_best, best)


def choose_probe_early(t: int, base_logits: np.ndarray, state: np.ndarray, counts: np.ndarray,
                       horizon: int = T_PRIMARY) -> np.ndarray:
    """Reference-semantics selector: ``pred = sigmoid(base + state)`` then PROBE_EARLY."""
    if state.shape != base_logits.shape or state.dtype != base_logits.dtype:
        raise ValueError("SELECT_STATE_SHAPE_OR_DTYPE")
    if not (np.abs(state) <= STATE_CAP + 1e-6).all():
        raise ValueError("SELECT_STATE_CAP")
    return probe_early_from_pred(t, sigmoid(base_logits + state), counts, horizon)


def greedy_with_deadline(t: int, score: np.ndarray, counts: np.ndarray, horizon: int) -> np.ndarray:
    """Argmax of ``score`` except that the best-scoring unseen expert is forced when coverage is due."""
    _check(t, horizon, score, counts)
    unseen = counts == 0
    need = unseen.sum(-1)
    unseen_best = np.where(unseen, score, -1e300).argmax(-1)
    return np.where(need >= (horizon - t), unseen_best, score.argmax(-1))
