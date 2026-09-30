"""Coarse observables and prediction targets. RESEARCH_ONLY.

A coarse observable is an explicit map ``C(z) -> vector``. Every sufficiency
statement must name a target from :data:`TARGETS`; there is no API that says
"C is sufficient" without one.

Coarse maps may declare ``linear=True``. For a linear map, a perturbation in
the null space of its (constant) Jacobian preserves the map exactly up to
floating point; for a nonlinear map it preserves it only to first order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

TARGETS = (
    "INSTANTANEOUS_OUTPUT",
    "ONE_STEP_TRANSITION",
    "K_STEP_TRAJECTORY",
    "EVENT",
    "ATTRACTOR_CLASS",
    "LONG_RUN_STATISTIC",
    "INTERVENTION_RESPONSE",
)


@dataclass(frozen=True)
class CoarseMap:
    """A named coarse observable with an optional analytic Jacobian."""
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray] | None = None
    linear: bool = False

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.fn(np.asarray(z, dtype=np.float64)), dtype=np.float64)).ravel()


def _mean_jac(z):
    return np.full((1, z.size), 1.0 / z.size)


def _second_jac(z):
    return (2.0 / z.size) * z.reshape(1, -1)


def _third_jac(z):
    return (3.0 / z.size) * (z ** 2).reshape(1, -1)


def _norm_jac(z):
    n = np.linalg.norm(z)
    return (z / n).reshape(1, -1) if n > 0 else np.zeros((1, z.size))


MEAN = CoarseMap("mean", lambda z: np.array([z.mean()]), _mean_jac, linear=True)
SECOND_MOMENT = CoarseMap("second_moment", lambda z: np.array([np.mean(z ** 2)]), _second_jac)
THIRD_MOMENT = CoarseMap("third_moment", lambda z: np.array([np.mean(z ** 3)]), _third_jac)
NORM = CoarseMap("norm", lambda z: np.array([np.linalg.norm(z)]), _norm_jac)


def projection(indices) -> CoarseMap:
    """Selected coordinates (linear; exact null-space preservation)."""
    idx = tuple(int(i) for i in indices)

    def jac(z):
        out = np.zeros((len(idx), z.size))
        out[np.arange(len(idx)), idx] = 1.0
        return out
    return CoarseMap(f"projection{list(idx)}", lambda z: z[list(idx)], jac, linear=True)


def combine(*maps: CoarseMap) -> CoarseMap:
    """Concatenation of coarse maps; linear only if every part is linear."""
    def fn(z):
        return np.concatenate([m(z) for m in maps])

    jac = None
    if all(m.jacobian is not None for m in maps):
        def jac(z):
            return np.vstack([m.jacobian(z) for m in maps])
    return CoarseMap("+".join(m.name for m in maps), fn, jac, linear=all(m.linear for m in maps))


def spectral_summary(shape, k: int) -> CoarseMap:
    """Top-k singular values of a matrix-shaped microscopic vector (no analytic Jacobian)."""
    rows, cols = shape

    def fn(z):
        return np.linalg.svd(z.reshape(rows, cols), compute_uv=False)[:k]
    return CoarseMap(f"spectral_top{k}", fn)


MEAN_AND_SECOND = combine(MEAN, SECOND_MOMENT)
MOMENTS_123 = combine(MEAN, SECOND_MOMENT, THIRD_MOMENT)

REGISTRY = {m.name: m for m in (MEAN, SECOND_MOMENT, THIRD_MOMENT, NORM, MEAN_AND_SECOND, MOMENTS_123)}


def coarse_map(name: str) -> CoarseMap:
    if name in REGISTRY:
        return REGISTRY[name]
    if name.startswith("projection"):
        import json
        return projection(json.loads(name[len("projection"):]))
    raise KeyError(f"unknown coarse observable {name!r}")
