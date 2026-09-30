"""Exact cubic coarse-graining control. RESEARCH_ONLY. An algebraic control, not the production architecture.

Family::

    phi(z)  = 0.5 z + 0.5 z^2 + 0.5 z^3
    f_W(x)  = sum_i phi(x . w_i),       W = [w_1 .. w_N]  (d x N)

Raw (uncentred, unnormalized) moments of the columns::

    m1 = sum_i w_i          (d)
    m2 = sum_i w_i w_i^T    (d x d)
    m3 = sum_i w_i (x) w_i (x) w_i  (d x d x d)

give the identity ``f_W(x) = 0.5 (x.m1 + x^T m2 x + m3[x, x, x])``.
:func:`forward_microscopic` evaluates the per-unit sum; :func:`forward_contracted`
evaluates the moment contraction from the moments alone. They are independent code
paths and are compared numerically across deterministic worlds.

What the identity establishes, and nothing more: ``S3 = (m1, m2, m3)`` is
sufficient for the INSTANTANEOUS_OUTPUT target of this family. It does not
establish sufficiency for a one-step transition, a trajectory, an attractor, an
intervention response, or anything about production ECS. :func:`transition`
is a deterministic rule update ``F(W) = W + eta W (.) W (.) W`` under which
``S3(F(W))`` needs fourth- and sixth-order moments of ``W``, so ``S3`` is not
transition-sufficient; :func:`pte_collision` builds an exact witness pair.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

PHI_COEFFICIENTS = (0.5, 0.5, 0.5)


def phi(z):
    return 0.5 * z + 0.5 * z * z + 0.5 * z * z * z


def dphi(z):
    return 0.5 + z + 1.5 * z * z


def forward_microscopic(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-unit evaluation ``sum_i phi(x . w_i)`` for each row of ``X``."""
    X = np.atleast_2d(X)
    return phi(X @ W).sum(axis=1)


def raw_moments(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m1 = W.sum(axis=1)
    m2 = W @ W.T
    m3 = np.einsum("ai,bi,ci->abc", W, W, W)
    return m1, m2, m3


def forward_contracted(moments, X: np.ndarray) -> np.ndarray:
    """``0.5 (x.m1 + x^T m2 x + m3[x,x,x])`` from the moments only (no access to ``W``)."""
    m1, m2, m3 = moments
    X = np.atleast_2d(X)
    return 0.5 * (X @ m1 + np.einsum("na,ab,nb->n", X, m2, X) + np.einsum("abc,na,nb,nc->n", m3, X, X, X))


# ---------------------------------------------------------------------------
# Coarse maps S1 / S2 / S3 on the flattened microscopic rule vec(W)
# ---------------------------------------------------------------------------


def _unique_pairs(d):
    return list(combinations_with_replacement(range(d), 2))


def _unique_triples(d):
    return list(combinations_with_replacement(range(d), 3))


def moment_coordinates(W: np.ndarray, order: int) -> np.ndarray:
    """``S_order`` as a vector of non-redundant moment entries (m2, m3 are symmetric)."""
    d = W.shape[0]
    m1, m2, m3 = raw_moments(W)
    parts = [m1]
    if order >= 2:
        parts.append(np.array([m2[a, b] for a, b in _unique_pairs(d)]))
    if order >= 3:
        parts.append(np.array([m3[a, b, c] for a, b, c in _unique_triples(d)]))
    return np.concatenate(parts)


def moment_jacobian(W: np.ndarray, order: int) -> np.ndarray:
    """Analytic ``d S_order / d vec(W)`` (C order: index ``e * N + i``)."""
    d, n = W.shape
    rows = []
    for a in range(d):
        g = np.zeros((d, n))
        g[a, :] = 1.0
        rows.append(g.ravel())
    if order >= 2:
        for a, b in _unique_pairs(d):
            g = np.zeros((d, n))
            g[a, :] += W[b, :]
            g[b, :] += W[a, :]
            rows.append(g.ravel())
    if order >= 3:
        for a, b, c in _unique_triples(d):
            g = np.zeros((d, n))
            g[a, :] += W[b, :] * W[c, :]
            g[b, :] += W[a, :] * W[c, :]
            g[c, :] += W[a, :] * W[b, :]
            rows.append(g.ravel())
    return np.vstack(rows)


def coarse_dimension(d: int, order: int) -> int:
    return d + (len(_unique_pairs(d)) if order >= 2 else 0) + (len(_unique_triples(d)) if order >= 3 else 0)


# ---------------------------------------------------------------------------
# Targets as functions of the rule W (with analytic Jacobians)
# ---------------------------------------------------------------------------


def output_jacobian(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``d f_W(x_n) / d vec(W)`` for each probe ``x_n``: rows ``x_e phi'(x . w_i)``."""
    X = np.atleast_2d(X)
    D = dphi(X @ W)  # (n_probe, N)
    return np.einsum("ne,ni->nei", X, D).reshape(X.shape[0], -1)


def transition(W: np.ndarray, eta: float) -> np.ndarray:
    """Synthetic deterministic rule update ``F(W) = W + eta W^3`` (elementwise). Not a learning rule of Elpis."""
    return W + eta * W * W * W


def transition_jacobian_diag(W: np.ndarray, eta: float) -> np.ndarray:
    """``dF/dvec(W)`` is diagonal; this returns the diagonal."""
    return (1.0 + 3.0 * eta * W * W).ravel()


def transition_coarse_target(W: np.ndarray, eta: float, order: int = 3) -> np.ndarray:
    """ONE_STEP_TRANSITION target: the coarse state after one rule update."""
    return moment_coordinates(transition(W, eta), order)


def transition_coarse_jacobian(W: np.ndarray, eta: float, order: int = 3) -> np.ndarray:
    return moment_jacobian(transition(W, eta), order) * transition_jacobian_diag(W, eta)[None, :]


def trajectory_output_target(W: np.ndarray, eta: float, X: np.ndarray, steps: int) -> np.ndarray:
    """K_STEP_TRAJECTORY target: outputs at the probes after 1..steps rule updates."""
    out, cur = [], W
    for _ in range(steps):
        cur = transition(cur, eta)
        out.append(forward_microscopic(cur, X))
    return np.concatenate(out)


def trajectory_output_jacobian(W: np.ndarray, eta: float, X: np.ndarray, steps: int) -> np.ndarray:
    rows, cur, chain = [], W, np.ones(W.size)
    for _ in range(steps):
        chain = chain * transition_jacobian_diag(cur, eta)
        cur = transition(cur, eta)
        rows.append(output_jacobian(cur, X) * chain[None, :])
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Exact witness pairs (Prouhet-Tarry-Escott multisets)
# ---------------------------------------------------------------------------

# Equal power sums of degree 1..3, different at degree 4:
PTE3 = ((0, 4, 7, 11), (1, 2, 9, 10))
# Equal power sums of degree 1..2, different at degree 3:
PTE2 = ((1, 5, 6), (2, 3, 7))


def pte_collision(base: np.ndarray, degree: int, scale: float = 1.0 / 16.0, axis: int = 0):
    """Two rules ``[base | scale * s * e_axis]`` whose raw moments agree through ``degree``.

    ``scale`` is dyadic and the PTE entries are small integers, so the appended
    columns' power sums agree exactly in binary64; the shared ``base`` enters
    both rules identically.
    """
    sets = {3: PTE3, 2: PTE2}[degree]
    d = base.shape[0]
    out = []
    for s in sets:
        tail = np.zeros((d, len(s)))
        tail[axis, :] = scale * np.asarray(s, dtype=np.float64)
        out.append(np.concatenate([base, tail], axis=1))
    return out[0], out[1]
