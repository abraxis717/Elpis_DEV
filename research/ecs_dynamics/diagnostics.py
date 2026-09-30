"""Finite-time phase diagnostics. RESEARCH_ONLY.

Conservative by construction:

* :func:`finite_time_growth` is a finite-time, finite-horizon tangent growth
  rate. It is NOT a rigorous Lyapunov exponent: no limit is taken and no
  convergence is claimed.
* :func:`classify_tail` reports FIXED_POINT, PERIODIC (exact repetition
  within tolerance, across several cycles), UNBOUNDED, or
  UNRESOLVED_NONPERIODIC. The last label means only "not resolved as fixed
  or periodic within this horizon and tolerance"; it includes slow transients,
  quasi-periodic motion and long periods. Nothing here labels a trajectory
  "chaotic".
* :func:`two_time_correlation` follows the definition
  ``C(t, s) = (1/N) sum_i phi(x_i(t)) phi(x_i(s))`` of Vaidya
  (arXiv:2609.19288, Eq. 2.5). Here the state is already the rate, so ``phi``
  is the identity.
"""
from __future__ import annotations

import numpy as np

from .systems import RecurrentSystem, perturb

CLASSES = ("FIXED_POINT", "PERIODIC", "UNRESOLVED_NONPERIODIC", "UNBOUNDED")
RECOVERY = ("DECAY", "PERSIST", "GROW")


def classify_tail(tail: np.ndarray, *, fixed_tol: float, recurrence_tol: float, max_period: int,
                  min_cycles: int = 3, bound: float = 1e8) -> dict:
    """Classify the last ``len(tail)`` states of a trajectory."""
    tail = np.asarray(tail, dtype=np.float64)
    length = tail.shape[0]
    if not np.all(np.isfinite(tail)) or np.max(np.abs(tail)) > bound:
        return {"class": "UNBOUNDED", "period": None, "tail_length": length}
    steps = np.max(np.abs(np.diff(tail, axis=0)), axis=1) if length > 1 else np.zeros(1)
    max_step = float(np.max(steps))
    if max_step <= fixed_tol:
        return {"class": "FIXED_POINT", "period": 1, "tail_length": length, "max_step": max_step}
    for p in range(2, max_period + 1):
        if length < (min_cycles + 1) * p:
            break
        residual = float(np.max(np.abs(tail[p:] - tail[:-p])))
        if residual <= recurrence_tol:
            return {"class": "PERIODIC", "period": p, "tail_length": length, "max_step": max_step,
                    "recurrence_residual": residual, "cycles": length // p}
    return {"class": "UNRESOLVED_NONPERIODIC", "period": None, "tail_length": length, "max_step": max_step}


def finite_time_growth(system: RecurrentSystem, x0: np.ndarray, rng: np.random.Generator, horizon: int,
                       magnitude: float = 1e-8) -> float:
    """Mean log growth per step of a tangent vector over ``horizon`` steps (finite-time).

    Uses the analytic Jacobian when the system has one, otherwise a
    renormalized two-trajectory estimate at separation ``magnitude``.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    v = rng.standard_normal(x.size)
    v /= np.linalg.norm(v)
    total = 0.0
    try:
        system.jacobian(x)
        use_jacobian = True
    except NotImplementedError:
        use_jacobian = False
    for _ in range(horizon):
        if use_jacobian:
            v = system.jacobian(x) @ v
            x = system.step(x)
        else:
            y = system.step(x + magnitude * v)
            x = system.step(x)
            v = (y - x) / magnitude
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return float("-inf") if n == 0.0 else float("inf")
        total += np.log(n)
        v /= n
    return total / horizon


def recovery(system: RecurrentSystem, x: np.ndarray, rng: np.random.Generator, *, magnitude: float, horizon: int,
             decay_ratio: float = 1e-2, grow_ratio: float = 1e2) -> dict:
    """Perturb, propagate both states for ``horizon`` steps, classify the separation ratio."""
    y = perturb(x, rng, magnitude)
    a, b = np.array(x, copy=True), y
    for _ in range(horizon):
        a, b = system.step(a), system.step(b)
    ratio = float(np.linalg.norm(a - b) / magnitude)
    if not np.isfinite(ratio) or ratio >= grow_ratio:
        label = "GROW"
    elif ratio <= decay_ratio:
        label = "DECAY"
    else:
        label = "PERSIST"
    return {"class": label, "ratio": ratio if np.isfinite(ratio) else None}


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of a scalar series at lags 0..max_lag (mean removed)."""
    s = np.asarray(series, dtype=np.float64)
    s = s - s.mean()
    var = float(np.dot(s, s))
    if var == 0.0:
        return np.concatenate([[1.0], np.zeros(max_lag)])
    return np.array([np.dot(s[: s.size - k], s[k:]) / var for k in range(max_lag + 1)])


def two_time_correlation(trajectory: np.ndarray, times, lags) -> np.ndarray:
    """``C(t, t + tau) = (1/d) x_t . x_{t+tau}`` for each ``t`` in ``times`` and ``tau`` in ``lags``."""
    traj = np.asarray(trajectory, dtype=np.float64)
    d = traj.shape[1]
    out = np.full((len(times), len(lags)), np.nan)
    for i, t in enumerate(times):
        for j, tau in enumerate(lags):
            if t + tau < traj.shape[0]:
                out[i, j] = float(traj[t] @ traj[t + tau]) / d
    return out
