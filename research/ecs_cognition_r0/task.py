"""The synthetic Cognitive R0 task. RESEARCH_ONLY. TRAINING=SYNTHETIC SEMANTICS=NONE.

Per world, from the frozen specification only:

* ``w0``: the ECS_G initialization (``dim x width``, i.i.d. normal);
* teacher A and teacher B: independent cubic ridge functions of the same
  family (``teacher_width`` columns). They exist only to *generate* targets
  for experience and evaluation; no query ever consults them;
* experience A / B (``train_rows``) and held-out evaluation A / B
  (``test_rows``): inputs ``N(+offset e0, scale^2 I)`` for A and
  ``N(-offset e0, scale^2 I)`` for B, with targets from the teacher;
* a target permutation (shuffled-target control) and an independent fresh
  initialization (fresh-state control);
* an exact Prouhet-Tarry-Escott pair of microstates with identical raw
  moments through degree 3 (identical S3), for microstate authority.

The numbers mean nothing outside this test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import world_rng

# Equal power sums of degree 1..3, different at degree 4 (same construction
# family as the frozen v1 laboratory; rebuilt here, not imported).
PTE3 = ((0, 4, 7, 11), (1, 2, 9, 10))


def cubic(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``sum_i phi(x . w_i)`` with ``phi(z) = 0.5 z + 0.5 z^2 + 0.5 z^3`` (target generation and cross-checks)."""
    Z = np.atleast_2d(X) @ W
    return (0.5 * Z + 0.5 * Z * Z + 0.5 * Z * Z * Z).sum(axis=1)


def _inputs(spec: dict, world: str, stream: str, rows: int, sign: float) -> np.ndarray:
    rng = world_rng(spec["seed"], spec["name"], world, stream)
    centre = np.zeros(spec["dim"])
    centre[0] = sign * spec["region_offset"]
    return rng.normal(0.0, spec["input_scale"], size=(rows, spec["dim"])) + centre


def query_inputs(spec: dict, world: str, which: str) -> np.ndarray:
    """Evaluation inputs only (no targets): what a clean replay process may regenerate."""
    sign = {"A": 1.0, "B": -1.0}[which]
    return _inputs(spec, world, f"inputs-{which}-test", spec["test_rows"], sign)


@dataclass(frozen=True)
class World:
    w0: np.ndarray
    teacher_a: np.ndarray
    teacher_b: np.ndarray
    xa_train: np.ndarray
    ya_train: np.ndarray
    xa_test: np.ndarray
    ya_test: np.ndarray
    xb_train: np.ndarray
    yb_train: np.ndarray
    xb_test: np.ndarray
    yb_test: np.ndarray
    permutation: np.ndarray
    w_fresh: np.ndarray


def build_world(spec: dict, world: str) -> World:
    d, n = spec["dim"], spec["width"]

    def rng(stream):
        return world_rng(spec["seed"], spec["name"], world, stream)

    w0 = rng("init").normal(0.0, spec["init_scale"], size=(d, n))
    ta = rng("teacher-A").normal(0.0, spec["teacher_scale"], size=(d, spec["teacher_width"]))
    tb = rng("teacher-B").normal(0.0, spec["teacher_scale"], size=(d, spec["teacher_width"]))
    xa = _inputs(spec, world, "inputs-A-train", spec["train_rows"], 1.0)
    xb = _inputs(spec, world, "inputs-B-train", spec["train_rows"], -1.0)
    xa_t, xb_t = query_inputs(spec, world, "A"), query_inputs(spec, world, "B")
    permutation = rng("shuffle").permutation(spec["train_rows"])
    w_fresh = rng("fresh").normal(0.0, spec["init_scale"], size=(d, n))
    return World(w0, ta, tb, xa, cubic(ta, xa), xa_t, cubic(ta, xa_t), xb, cubic(tb, xb), xb_t, cubic(tb, xb_t),
                 permutation, w_fresh)


def pte_pair(spec: dict, w0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[w0[:, :base] | tail]`` for both PTE3 sets on axis 0: identical S3, different microstates."""
    base = w0[:, :spec["pte_base_width"]]
    out = []
    for values in PTE3:
        tail = np.zeros((spec["dim"], len(values)))
        tail[0, :] = spec["pte_tail_scale"] * np.asarray(values, dtype=np.float64)
        out.append(np.concatenate([base, tail], axis=1))
    return out[0], out[1]


def nmse(prediction, target: np.ndarray) -> float:
    """Mean squared error normalized by the variance of the evaluation targets."""
    p = np.asarray(prediction, dtype=np.float64)
    return float(np.mean((p - target) ** 2) / np.var(target))
