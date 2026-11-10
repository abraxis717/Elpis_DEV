"""Exact sufficient conditions for the R119 guards, computed with rational arithmetic.

G_GATES (kernel guard 2). For one slot let m = max |coordinate| over A, B, sal, col. The
kernel (R118 arithmetic) computes, per gate q,

    raw_j = dot4(.,.)  (9 features, plus raw_9 = 0),   f_j = raw_j / (1 + |raw_j|),
    z = (((0 + W_q0 f_0) + W_q1 f_1) + ... + W_q9 f_9) + bias_q.

With u = 2^-53 and eta = 2^-1075 (the absolute rounding error of a result in the
subnormal range; sums of subnormal-range values are exact):
|fl(x y)| <= |x y| (1+u) + eta, |fl(a+b)| <= (|a|+|b|)(1+u), so
|raw_j| <= 4 m^2 (1+5u) + 5 eta, |f_j| <= 4 m^2 (1+6u) + 7 eta, and the W.f sum obeys
|z_sum| <= 4 S_q m^2 (1+19u) + (7 S_q + 11) eta <= 4 S_q m^2 (1 + 2^-47) + 2^-1065
for S_q = sum_j |W_qj| <= 100. If |bias_q| in [2^E, 2^(E+1)) (normal, nonzero), every
value within 2^(E-54) of bias_q rounds to bias_q (half the smallest adjacent gap), so
|z_sum| <= 2^(E-53) / 4 guarantees fl(z_sum + bias_q) == bias_q bit for bit.

``gate_theta`` returns the largest double theta with theta^2 <= the tightest such bound
over all eight gates; the kernel takes the guard only if m < theta (strict).
"""
from __future__ import annotations

import math
from fractions import Fraction

TINY = 2.2250738585072014e-308
U47 = Fraction(1, 2 ** 47)
ETA_SUM = Fraction(1, 2 ** 1065)
S_LIMIT = 100


def gate_bound_m2(W, bias):
    """Tightest admissible m^2 over the eight gates (Fraction), or None if the guard is unusable."""
    bound = None
    for q in range(8):
        b = float(bias[q])
        if not math.isfinite(b) or b == 0.0 or abs(b) < TINY:
            return None
        _, e = math.frexp(abs(b))          # |b| = mant * 2^e, mant in [0.5, 1)  =>  E = e - 1
        gap = Fraction(2) ** (e - 1 - 53)  # smallest gap adjacent to b, conservatively
        S = sum((Fraction(abs(float(w))) for w in W[q]), Fraction(0))
        if S > S_LIMIT:
            return None
        if S == 0:
            continue                       # z_sum is +-0 exactly: no constraint from this gate
        lim = (gap / 4 - ETA_SUM) / (4 * S * (1 + U47))
        if lim <= 0:
            return None
        bound = lim if bound is None else min(bound, lim)
    return bound


def gate_theta(W, bias) -> float:
    bound = gate_bound_m2(W, bias)
    if bound is None:
        return 0.0                         # guard never taken
    t = math.sqrt(float(bound))
    while Fraction(t) ** 2 > bound:
        t = math.nextafter(t, 0.0)
    return min(t, 1.0)
