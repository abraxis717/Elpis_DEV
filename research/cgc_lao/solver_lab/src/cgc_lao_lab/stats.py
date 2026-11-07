"""Paired router-seed statistics: Student-t intervals without SciPy.

The t CDF uses the regularised incomplete beta function (modified Lentz continued fraction); the
quantile is found by bisection. ``paired`` summarises per-seed differences A - B (negative favours A).
"""
from __future__ import annotations

import math

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("BETACF_NO_CONVERGENCE")


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: int) -> float:
    if not (0.0 < p < 1.0) or df < 1:
        raise ValueError("T_QUANTILE_RANGE")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci(values, level: float = 0.95) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        raise ValueError("CI_NEEDS_TWO_SEEDS")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    half = t_quantile(0.5 + level / 2.0, n - 1) * se
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "se": se, "n": n}


def paired(a, b, level: float = 0.95) -> dict:
    """Per-seed paired difference a - b with a two-sided Student-t interval; wins = seeds with a < b."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("PAIRED_SHAPE")
    out = mean_ci(a - b, level)
    out["wins"] = int((a < b).sum())
    return out
