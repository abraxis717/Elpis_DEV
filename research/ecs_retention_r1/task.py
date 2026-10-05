"""The synthetic Retention R1 task families. RESEARCH_ONLY. TRAINING=SYNTHETIC SEMANTICS=NONE.

Per world, from the frozen specification only (spec arms, regime, seed):

* ``w0``: the ECS_G initialization (``dim x width``, i.i.d. normal), shared by both arms;
* arm S: one world teacher ``T*``; experience ``t`` has inputs on its active coordinates only and targets
  ``g*(x)``; the witness ``W* = [T* | 0]`` answers every experience exactly. Inputs are standard normals per
  stream, then scaled, so every input scale shares the same draws;
* arm R: the Retention R0 v1 family rebuilt here (not imported): independent ridge teachers per experience,
  inputs ``N(sign * offset * e_axis, input_scale^2 I)``;
* the mismatched-statistics inputs (complement coordinates) and the PTE microstate pair (identical S3).

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
    pairs: np.ndarray       # (P, 2)
    triples: np.ndarray     # (T, 3)
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
    arm: str
    scale: float                 # arm S input scale; arm R input scale
    w0: np.ndarray
    tasks: dict                  # task name -> Experience
    witness: np.ndarray | None   # arm S only
    mismatch: dict | None        # arm S only: task name -> mismatched consolidation inputs


def _rng(spec, world, stream):
    return world_rng(spec["seed"], spec["name"], world, stream)


def initial_w(spec: dict, world: str) -> np.ndarray:
    reg = spec["regime"]
    return np.ascontiguousarray(_rng(spec, world, "init").normal(0.0, reg["init_scale"],
                                                                 size=(reg["dim"], reg["width"])))


def _subspace_inputs(spec, world, stream, coords, rows, scale):
    d = spec["regime"]["dim"]
    z = _rng(spec, world, stream).standard_normal(size=(rows, len(coords)))
    x = np.zeros((rows, d))
    x[:, coords] = z * scale
    return np.ascontiguousarray(x)


def build_world_s(spec: dict, world: str, scale: float) -> World:
    reg, arm = spec["regime"], spec["arms"]["S"]
    d, n = reg["dim"], reg["width"]
    w0 = initial_w(spec, world)
    teacher = _rng(spec, world, "world-teacher").normal(0.0, reg["teacher_scale"], size=(d, reg["teacher_width"]))
    tasks, mismatch = {}, {}
    for t in TASKS:
        coords = arm["active_coordinates"][t]
        x_train = _subspace_inputs(spec, world, f"inputs-S-{t}-train", coords, reg["train_rows"], scale)
        x_test = _subspace_inputs(spec, world, f"inputs-S-{t}-test", coords, reg["test_rows"], scale)
        tasks[t] = Experience(x_train, cubic(teacher, x_train), x_test, cubic(teacher, x_test))
        mismatch[t] = _subspace_inputs(spec, world, f"mismatch-{t}", arm["complement_coordinates"][t],
                                       reg["train_rows"], scale)
    witness = np.ascontiguousarray(np.concatenate([teacher, np.zeros((d, n - teacher.shape[1]))], axis=1))
    return World(world, "S", float(scale), w0, tasks, witness, mismatch)


def build_world_r(spec: dict, world: str) -> World:
    reg, arm = spec["regime"], spec["arms"]["R"]
    d = reg["dim"]
    w0 = initial_w(spec, world)
    tasks = {}
    for t in TASKS:
        task = arm["tasks"][t]
        centre = np.zeros(d)
        centre[task["axis"]] = task["sign"] * arm["offset"]
        teacher = _rng(spec, world, f"teacher-{t}").normal(0.0, reg["teacher_scale"], size=(d, reg["teacher_width"]))

        def inputs(kind, rows):
            z = _rng(spec, world, f"inputs-R-{t}-{kind}").standard_normal(size=(rows, d))
            return np.ascontiguousarray(z * arm["input_scale"] + centre)
        x_train, x_test = inputs("train", reg["train_rows"]), inputs("test", reg["test_rows"])
        tasks[t] = Experience(x_train, cubic(teacher, x_train), x_test, cubic(teacher, x_test))
    return World(world, "R", float(arm["input_scale"]), w0, tasks, None, None)


def uninformed_rng(spec: dict, world: str, task: str) -> np.random.Generator:
    return _rng(spec, world, f"uninformed-{task}")


def pte_pair(w0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[w0[:, :32] | tail]`` for both PTE3 sets on axis 0: identical S3, different microstates."""
    base = w0[:, :PTE_BASE_WIDTH]
    out = []
    for values in PTE3:
        tail = np.zeros((w0.shape[0], len(values)))
        tail[0, :] = PTE_TAIL_SCALE * np.asarray(values, dtype=np.float64)
        out.append(np.ascontiguousarray(np.concatenate([base, tail], axis=1)))
    return out[0], out[1]


def nmse(prediction, target: np.ndarray) -> float:
    """Mean squared error over held-out rows divided by the population variance of the held-out targets."""
    p = np.asarray(prediction, dtype=np.float64)
    return float(np.mean((p - target) ** 2) / np.var(target))
