"""The synthetic Retention R0 task family. RESEARCH_ONLY. TRAINING=SYNTHETIC SEMANTICS=NONE.

The Cognitive R0 v1 family, rebuilt here (not imported) and extended to four
experiences. Per world, from the frozen specification only:

* ``w0``: the ECS_G initialization (``dim x width``, i.i.d. normal);
* for each task t in A, B, C, D: an independent cubic ridge teacher
  (``teacher_width`` columns) that exists only to generate targets, and
  experience / held-out inputs ``N(sign * offset * e_axis, input_scale^2 I)``.
  Inputs are drawn as standard normals per stream and then scaled and
  shifted, so every offset shares the same underlying draws;
* the exact Prouhet-Tarry-Escott microstate pair of Cognitive R0 (identical
  S3, different microstates) for the microstate check.

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
    m2: np.ndarray          # multiplicities of the quadratic features
    m3: np.ndarray          # multiplicities of the cubic features

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
    offset: float
    w0: np.ndarray
    tasks: dict          # task name -> Experience


def build_world(spec: dict, world: str, offset: float) -> World:
    reg, name, seed = spec["regime"], spec["name"], spec["seed"]
    d, n = reg["dim"], reg["width"]

    def rng(stream):
        return world_rng(seed, name, world, stream)

    w0 = rng("init").normal(0.0, reg["init_scale"], size=(d, n))
    tasks = {}
    for t in TASKS:
        spec_t = spec["tasks"][t]
        centre = np.zeros(d)
        centre[spec_t["axis"]] = spec_t["sign"] * offset
        teacher = rng(f"teacher-{t}").normal(0.0, reg["teacher_scale"], size=(d, reg["teacher_width"]))

        def inputs(kind, rows):
            return rng(f"inputs-{t}-{kind}").standard_normal(size=(rows, d)) * reg["input_scale"] + centre
        x_train, x_test = inputs("train", reg["train_rows"]), inputs("test", reg["test_rows"])
        tasks[t] = Experience(np.ascontiguousarray(x_train), cubic(teacher, x_train),
                              np.ascontiguousarray(x_test), cubic(teacher, x_test))
    return World(world, float(offset), np.ascontiguousarray(w0), tasks)


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
