"""The Retention R3 task family, inherited unchanged from closed R2: aliased hidden ridges. RESEARCH_ONLY.
TRAINING=SYNTHETIC SEMANTICS=NONE.

Per world, from the write-once specification only (spec task, regime, seed):

* ``w0``: the ECS_G initialization (``dim x width``, i.i.d. normal);
* a Haar orthonormal basis ``q1..q6`` (QR of a Gaussian matrix, columns signed by the diagonal of R);
* the witness teacher ``T* = [r_A, g q4, g q5, g q6]`` with ``r_A`` a random unit vector in span(q1, q2, q3);
* planes: A observes span(q1, q2, q3); B, C, D replace one axis by ``cos(theta) q_k + sin(theta) q_{k+3}``;
  inputs ``x = M_t z`` with ``z ~ N(0, input_scale^2 I_3)``; targets ``g*(x) = sum_j phi(x . t*_j)``;
* the mismatched-statistics inputs (the never-observed plane span(q4, q5, q6)) and the PTE microstate pair
  placed along B's aliased axis (identical S3; a B step reads exactly the differing coordinate).

Inputs are drawn as standard normals per stream and then mapped, so every hidden gain shares the same draws.
The numbers mean nothing outside this test.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .protocol import world_rng

TASKS = ("A", "B", "C", "D")
PTE3 = ((0, 4, 7, 11), (1, 2, 9, 10))   # equal power sums of degree 1..3
PTE_BASE_WIDTH, PTE_TAIL_SCALE = 32, 0.125


def cubic(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``sum_i phi(x . w_i)`` with ``phi(z) = 0.5 z + 0.5 z^2 + 0.5 z^3`` (target generation only)."""
    Z = np.atleast_2d(X) @ W
    return (0.5 * Z + 0.5 * Z * Z + 0.5 * Z * Z * Z).sum(axis=1)


@dataclass(frozen=True)
class Indices:
    """Packed index sets in the order of ``ecsg_math.h``: mu, M (a <= b), T3 (a <= b <= c)."""
    dim: int
    pairs: np.ndarray
    triples: np.ndarray
    m2: np.ndarray
    m3: np.ndarray

    @property
    def size(self) -> int:
        return self.dim + len(self.pairs) + len(self.triples)


def indices(dim: int) -> Indices:
    pairs = np.array(list(combinations_with_replacement(range(dim), 2)), dtype=np.intp)
    triples = np.array(list(combinations_with_replacement(range(dim), 3)), dtype=np.intp)
    m2 = np.where(pairs[:, 0] == pairs[:, 1], 1.0, 2.0)
    distinct = np.array([len(set(t)) for t in triples.tolist()])
    m3 = np.select([distinct == 1, distinct == 2], [1.0, 3.0], 6.0)
    return Indices(dim, pairs, triples, m2, m3)


def features(X: np.ndarray, idx: Indices) -> np.ndarray:
    """Packed cubic features ``phi(x)``: ``f_W(x) = 1/2 phi(x) . S3(W)`` exactly."""
    X = np.atleast_2d(X)
    p, t = idx.pairs, idx.triples
    return np.concatenate([X, idx.m2 * X[:, p[:, 0]] * X[:, p[:, 1]],
                           idx.m3 * X[:, t[:, 0]] * X[:, t[:, 1]] * X[:, t[:, 2]]], axis=1)


@dataclass(frozen=True)
class Experience:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class World:
    world: str
    gain: float
    w0: np.ndarray
    basis: np.ndarray       # Q, columns q1..q6
    planes: dict            # task name -> 6 x 3 orthonormal frame M_t
    tasks: dict             # task name -> Experience
    witness: np.ndarray
    mismatch: dict          # task name -> mismatched consolidation inputs
    pte_axis: np.ndarray    # B's aliased axis b


def _rng(spec, world, stream):
    return world_rng(spec["seed"], spec["name"], world, stream)


def haar_basis(spec: dict, world: str) -> np.ndarray:
    d = spec["regime"]["dim"]
    Q, R = np.linalg.qr(_rng(spec, world, "basis").standard_normal(size=(d, d)))
    return np.ascontiguousarray(Q * np.sign(np.diag(R)))


def planes_of(Q: np.ndarray, theta: float) -> dict:
    c, s = np.cos(theta), np.sin(theta)
    planes = {"A": Q[:, :3].copy()}
    for k, t in enumerate("BCD"):
        M = Q[:, :3].copy()
        M[:, k] = c * Q[:, k] + s * Q[:, 3 + k]
        planes[t] = M
    return {t: np.ascontiguousarray(M) for t, M in planes.items()}


def _plane_inputs(spec, world, stream, frame, rows, scale):
    z = _rng(spec, world, stream).standard_normal(size=(rows, frame.shape[1])) * scale
    return np.ascontiguousarray(z @ frame.T)


def build_world(spec: dict, world: str, gain: float) -> World:
    reg, task = spec["regime"], spec["task"]
    d, n = reg["dim"], reg["width"]
    scale, theta = task["input_scale"], np.deg2rad(task["aliasing_angle_degrees"])
    w0 = np.ascontiguousarray(_rng(spec, world, "init").normal(0.0, reg["init_scale"], size=(d, n)))
    Q = haar_basis(spec, world)
    z = Q[:, :3] @ _rng(spec, world, "teacher-A").standard_normal(size=3)
    teacher = np.stack([z / np.linalg.norm(z), gain * Q[:, 3], gain * Q[:, 4], gain * Q[:, 5]], axis=1)
    planes = planes_of(Q, theta)
    tasks, mismatch = {}, {}
    for t in TASKS:
        x_train = _plane_inputs(spec, world, f"inputs-{t}-train", planes[t], reg["train_rows"], scale)
        x_test = _plane_inputs(spec, world, f"inputs-{t}-test", planes[t], reg["test_rows"], scale)
        tasks[t] = Experience(x_train, cubic(teacher, x_train), x_test, cubic(teacher, x_test))
        mismatch[t] = _plane_inputs(spec, world, f"mismatch-{t}", Q[:, 3:6], reg["train_rows"], scale)
    witness = np.ascontiguousarray(np.concatenate([teacher, np.zeros((d, n - teacher.shape[1]))], axis=1))
    return World(world, float(gain), w0, Q, planes, tasks, witness, mismatch, planes["B"][:, 0].copy())


def uninformed_rng(spec: dict, world: str, task: str) -> np.random.Generator:
    return _rng(spec, world, f"uninformed-{task}")


def pte_pair(w0: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[w0[:, :32] | tail]``, tail entities ``0.125 v axis`` for both PTE3 sets: identical S3."""
    base = w0[:, :PTE_BASE_WIDTH]
    out = []
    for values in PTE3:
        tail = np.outer(axis, PTE_TAIL_SCALE * np.asarray(values, dtype=np.float64))
        out.append(np.ascontiguousarray(np.concatenate([base, tail], axis=1)))
    return out[0], out[1]


def nmse(prediction, target: np.ndarray) -> float:
    """Mean squared error over held-out rows divided by the population variance of the held-out targets."""
    p = np.asarray(prediction, dtype=np.float64)
    return float(np.mean((p - target) ** 2) / np.var(target))
