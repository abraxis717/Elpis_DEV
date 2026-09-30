"""Static-feedback reading of Vaidya, arXiv:2609.19288. RESEARCH_ONLY.

What the paper states (cited by section and equation):

* A continuous-time tanh network ``tau x' = -x + sum_j J_ij phi(x_j) + W_fb y``
  with a learned readout ``y = w_out . phi(x)`` (Eqs. 2.1, 2.2, 2.4).
* With learning slow compared with the network (Sec. 4), the feedback
  enters the fast dynamics as a quasi-static field of variance ``yhat^2``
  (Eq. 4.2, with sigma_fb = 1).
* The fast fluctuations disappear at the marginal point
  ``1 = g^2 int Dz phi'(sqrt(u_c) z)^2`` (Eq. 4.5), where
  ``u_c = g^2 int Dz phi(sqrt(u_c) z)^2 + yhat_c^2`` (Eq. 4.6). The paper reports
  ``yhat_c = 0.2`` at ``g = 1.3`` (Sec. 4, Fig. 1(f); Sec. 6).

This module reproduces that number from Eqs. (4.5)-(4.6) by Gauss-Hermite
quadrature (:func:`critical_feedback`).

Two remarks on the text, recorded here rather than silently corrected:

* Section 2 gives the coupling variance as ``1/sqrt(N)`` and writes ``g^2 J``
  in Eq. (2.1), while Eqs. (3.2) and (A.2) use ``P(J) ~ exp(-N J^2 / 2 g^2)``,
  i.e. variance ``g^2/N``. The mean-field equations follow (3.2)/(A.2), and
  so does this module.
* Section 5.3 says the kernel vanishes "beyond u = u_c ..., which is 0.2 for
  g=1.3". Solving (4.5)-(4.6) gives ``u_c ~= 0.50`` and ``yhat_c ~= 0.196``, so
  the 0.2 there is ``yhat_c``.

The laboratory's own derivation, which is heuristic mean-field and not from
the paper: for the discrete-time probe
``x_{t+1} = tanh(g J x_t + b)``, with ``J`` i.i.d. ``N(0, 1/d)`` (``gamma = 1`` in
:func:`systems.nonreciprocal_coupling`) and ``b_i ~ N(0, yhat^2)``, the
pre-activation variance obeys ``Delta = g^2 E[tanh^2(sqrt(Delta) z)] + yhat^2``.
The squared tangent-norm growth factor of the fixed-point branch is
``chi = g^2 E[tanh'^2(sqrt(Delta) z)]``, and ``chi = 1`` is the same condition as
(4.5)-(4.6). The discrete probe's predicted stability change in ``yhat``
therefore sits at the paper's ``yhat_c``. Only this static stability condition
is transferred. The learning rule, the critical time ``t_cr``, the output
trajectory and the two-time DMFT solution are NOT tested here.
"""
from __future__ import annotations

import numpy as np

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(200)
_WEIGHTS = _WEIGHTS / np.sqrt(2.0 * np.pi)


def gaussian_expectation(f) -> float:
    """``int Dz f(z)`` for the standard normal measure (200-point Gauss-Hermite)."""
    return float(np.sum(_WEIGHTS * f(_NODES)))


def chi(u: float, g: float) -> float:
    """``g^2 int Dz phi'(sqrt(u) z)^2`` with ``phi = tanh``."""
    return g * g * gaussian_expectation(lambda z: (1.0 - np.tanh(np.sqrt(u) * z) ** 2) ** 2)


def critical_feedback(g: float) -> dict:
    """Solve Eq. (4.5) for ``u_c`` by bisection, then Eq. (4.6) for ``yhat_c``."""
    if g <= 1.0:
        return {"u_c": 0.0, "yhat_c": 0.0, "note": "g <= 1: marginal at u = 0"}
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi(mid, g) > 1.0:
            lo = mid
        else:
            hi = mid
    u_c = 0.5 * (lo + hi)
    y2 = u_c - g * g * gaussian_expectation(lambda z: np.tanh(np.sqrt(u_c) * z) ** 2)
    return {"u_c": u_c, "yhat_c": float(np.sqrt(max(y2, 0.0))), "chi_at_u_c": chi(u_c, g)}


def stationary_variance(g: float, yhat: float, iterations: int = 5000) -> float:
    """Fixed point of ``Delta -> g^2 E[tanh^2(sqrt(Delta) z)] + yhat^2`` (discrete-time mean field)."""
    delta = 1.0
    for _ in range(iterations):
        new = g * g * gaussian_expectation(lambda z: np.tanh(np.sqrt(delta) * z) ** 2) + yhat * yhat
        if abs(new - delta) < 1e-15:
            break
        delta = new
    return delta


def meanfield_growth(g: float, yhat: float) -> dict:
    """Mean-field squared growth factor ``chi`` on the fixed-point branch and its per-step log-norm rate."""
    delta = stationary_variance(g, yhat)
    c = chi(delta, g)
    return {"delta": delta, "chi": c, "log_norm_rate": 0.5 * float(np.log(c))}
