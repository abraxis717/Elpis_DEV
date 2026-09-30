"""Coarse-preserving interventions, fibre response capacity and prediction loss. RESEARCH_ONLY.

An intervention changes the microscopic object ``z`` (a state or a rule) while
keeping a coarse map ``C`` fixed. Where ``C`` is linear, a step in the null space
of its Jacobian preserves ``C`` exactly (up to rounding). Otherwise it
preserves ``C`` only to first order; :func:`retract` applies Gauss-Newton steps
back onto the level set. :func:`coarse_preserving_interventions` measures the
preservation residual and REJECTS any intervention above the tolerance.

The fibre quantities follow Wen & Lu (arXiv:2609.29834):

* the fibre of ``C`` through ``z`` is ``{z' : C(z') = C(z)}``; at a regular
  point, its tangent space is ``ker DC`` (main text, paragraph after the
  abstract);
* response capacity ``Gamma = rank(D Phi restricted to ker DC)`` (their Eq. 2):
  the number of independent first-order target changes that the coarse map
  cannot see;
* prediction loss: their Eq. (3) uses an L-infinity kernel budget. This
  laboratory uses a EUCLIDEAN budget ``||dz||_2 <= eps`` instead, for which
  the largest first-order target change is ``eps * sigma_max(D Phi N)``, with
  ``N`` an orthonormal basis of ``ker DC``. For a scalar target this equals
  ``eps * min_a ||grad Phi - DC^T a||_2``, the L2 analogue of their
  ``inf_a int q0 |Psi - a.C|``. A predictor that sees only ``C`` returns one
  value on the fibre, so ``eps * sigma_max`` is also, to leading order, the
  smallest worst-case error of any such predictor within the budget. That
  adaptation is this laboratory's, not the paper's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NullSpace:
    basis: np.ndarray        # (n, k) orthonormal columns
    rank: int
    nominal_dimension: int   # number of coarse coordinates (rows of the Jacobian)
    microscopic_dimension: int
    singular_values: np.ndarray

    @property
    def null_dimension(self) -> int:
        return self.basis.shape[1]

    def summary(self) -> dict:
        return {"nominal_dimension": self.nominal_dimension, "microscopic_dimension": self.microscopic_dimension,
                "rank": self.rank, "null_dimension": self.null_dimension}


def null_space(jacobian: np.ndarray, rtol: float = 1e-10) -> NullSpace:
    """SVD null space; rank counts singular values above ``rtol * s_max``."""
    jacobian = np.atleast_2d(jacobian)
    rows, cols = jacobian.shape
    _, s, vt = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0
    return NullSpace(vt[rank:].T.copy(), rank, rows, cols, s)


def response_capacity(target_jacobian: np.ndarray, basis: np.ndarray, rtol: float = 1e-8,
                      scale: float | None = None) -> tuple[int, np.ndarray]:
    """``rank(D Phi N)`` and its singular values.

    The rank threshold is relative to ``scale``, the norm of the unrestricted
    target Jacobian by default, so a response that is zero only up to
    rounding is counted as zero.
    """
    restricted = np.atleast_2d(target_jacobian) @ basis
    if restricted.size == 0:
        return 0, np.zeros(0)
    s = np.linalg.svd(restricted, compute_uv=False)
    ref = np.linalg.norm(target_jacobian, 2) if scale is None else scale
    if ref == 0:
        return 0, s
    return int(np.sum(s > rtol * ref)), s


def prediction_loss(target_jacobian: np.ndarray, basis: np.ndarray) -> float:
    """Largest first-order target change per unit Euclidean budget inside the fibre: ``sigma_max(D Phi N)``."""
    restricted = np.atleast_2d(target_jacobian) @ basis
    if restricted.size == 0:
        return 0.0
    return float(np.linalg.norm(restricted, 2))


def retract(z: np.ndarray, coarse, jacobian, value: np.ndarray, steps: int) -> np.ndarray:
    """Gauss-Newton steps toward ``coarse(z) == value`` (minimum-norm corrections)."""
    for _ in range(steps):
        r = coarse(z) - value
        if not np.all(np.isfinite(r)):
            break
        dz, *_ = np.linalg.lstsq(np.atleast_2d(jacobian(z)), r, rcond=None)
        z = z - dz
    return z


def preservation_residual(z0, z1, coarse) -> float:
    c0 = coarse(z0)
    return float(np.linalg.norm(coarse(z1) - c0) / max(1.0, np.linalg.norm(c0)))


def coarse_preserving_interventions(z: np.ndarray, coarse, jacobian, rng: np.random.Generator, *,
                                    magnitude: float, count: int, tolerance: float, retraction_steps: int = 0,
                                    rtol: float = 1e-10):
    """Random null-space interventions at ``z``; returns ``(accepted states, report)``.

    Each candidate is ``z + magnitude * N c / ||N c||`` with Gaussian ``c``,
    optionally retracted onto the level set. Candidates with preservation
    residual above ``tolerance`` are rejected and counted, never used.
    """
    ns = null_space(jacobian(z), rtol)
    value = coarse(z)
    accepted, residuals, rejected = [], [], 0
    if ns.null_dimension == 0:
        return [], dict(ns.summary(), attempted=count, accepted=0, rejected=count, residual_max=None,
                        residual_median=None, tolerance=tolerance)
    for _ in range(count):
        c = rng.standard_normal(ns.null_dimension)
        step = ns.basis @ c
        cand = z + magnitude * step / np.linalg.norm(step)
        if retraction_steps:
            cand = retract(cand, coarse, jacobian, value, retraction_steps)
        res = preservation_residual(z, cand, coarse)
        if np.isfinite(res) and res <= tolerance:
            accepted.append(cand)
            residuals.append(res)
        else:
            rejected += 1
    report = dict(ns.summary(), attempted=count, accepted=len(accepted), rejected=rejected, tolerance=tolerance,
                  magnitude=magnitude, retraction_steps=retraction_steps,
                  residual_max=float(max(residuals)) if residuals else None,
                  residual_median=float(np.median(residuals)) if residuals else None)
    return accepted, report


def divergence_statistics(reference: np.ndarray, others, threshold: float) -> dict:
    """Distances ``||T(z') - T(z)||`` (absolute and relative to ``max(1, ||T(z)||)``)."""
    ref = np.asarray(reference, dtype=np.float64)
    scale = max(1.0, float(np.linalg.norm(ref)))
    d = np.array([np.linalg.norm(np.asarray(o) - ref) for o in others]) if len(others) else np.zeros(0)
    rel = d / scale
    return {"count": int(d.size), "threshold": threshold,
            "relative_median": float(np.median(rel)) if d.size else None,
            "relative_max": float(np.max(rel)) if d.size else None,
            "relative_min": float(np.min(rel)) if d.size else None,
            "fraction_above_threshold": float(np.mean(rel > threshold)) if d.size else None}
