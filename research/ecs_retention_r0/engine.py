"""Retention R0 learning engines. RESEARCH_ONLY.

* Canonical paths (M0, M1): the canonical core and executor, unchanged.
* Candidate engine (C1, C2): every step is the canonical native G1 step (an
  executor created from the current W performs one learn step) followed by
  subtracting ``rate * grad P(W)`` evaluated at the pre-step W. With the
  consolidation term removed it is bitwise the canonical core.
* Consolidation objects receive the consolidated W and experienced inputs
  only; no target ever reaches them. Queries never see them: every response is
  the canonical native forward map of W.

S3 coordinates follow ``ecsg_math.h``: ``f_W(x) = 1/2 phi(x) . S3(W)``.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.native import ECSGError, Executor

from .task import Indices, features

SNAPSHOT_HEADER = 40


def w_digest(W: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(W, dtype="<f8").tobytes()).hexdigest()


def _flat(W: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(W, dtype=np.float64).reshape(-1))


def w_of(snapshot: bytes, dim: int, width: int) -> np.ndarray:
    return np.frombuffer(snapshot, dtype="<f8", offset=SNAPSHOT_HEADER).reshape(dim, width).astype(np.float64)


# --- canonical paths --------------------------------------------------------------------------------------------


def canonical_learn(api, W: np.ndarray, X: np.ndarray, y: np.ndarray, rate: float, steps: int) -> np.ndarray:
    """K canonical G1 steps through CognitiveCore (one native call); returns the committed W."""
    dim, width = W.shape
    with CognitiveCore.create(api, dim, width, _flat(W), learning_rate=rate,
                              max_rows=max(len(y), 1)) as core:
        core.learn(memoryview(np.ascontiguousarray(X)), memoryview(np.ascontiguousarray(y)), steps=steps,
                   receipt=False)
        return w_of(core.snapshot(), dim, width)


def responses(api, W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The canonical native forward map of W on X."""
    dim, width = W.shape
    with Executor.create(api, dim, width, _flat(W)) as e:
        return np.asarray(e.forward(memoryview(np.ascontiguousarray(X))), dtype=np.float64)


def native_s3(api, W: np.ndarray) -> np.ndarray:
    dim, width = W.shape
    with Executor.create(api, dim, width, _flat(W)) as e:
        return np.asarray(e.s3(), dtype=np.float64)


# --- S3 coordinates ---------------------------------------------------------------------------------------------


def s3(W: np.ndarray, idx: Indices) -> np.ndarray:
    p, t = idx.pairs, idx.triples
    return np.concatenate([W.sum(axis=1), (W[p[:, 0]] * W[p[:, 1]]).sum(axis=1),
                           (W[t[:, 0]] * W[t[:, 1]] * W[t[:, 2]]).sum(axis=1)])


def s3_vjp(W: np.ndarray, u: np.ndarray, idx: Indices) -> np.ndarray:
    """``d(u . S3(W)) / dW`` (shape of W): ``u_1 + 2 U_2 w_i + 3 U_3(w_i, w_i, .)`` per entity."""
    d = idx.dim
    P = len(idx.pairs)
    u1, u2, u3 = u[:d], u[d:d + P], u[d + P:]
    p, t = idx.pairs, idx.triples
    G = np.repeat(u1[:, None], W.shape[1], axis=1)
    U2 = np.zeros((d, d))
    np.add.at(U2, (p[:, 0], p[:, 1]), u2)
    np.add.at(U2, (p[:, 1], p[:, 0]), u2)
    G += U2 @ W
    np.add.at(G, t[:, 0], u3[:, None] * W[t[:, 1]] * W[t[:, 2]])
    np.add.at(G, t[:, 1], u3[:, None] * W[t[:, 0]] * W[t[:, 2]])
    np.add.at(G, t[:, 2], u3[:, None] * W[t[:, 0]] * W[t[:, 1]])
    return G


def s3_jacobian(W: np.ndarray, idx: Indices) -> np.ndarray:
    """``dS3/dW`` as an ``F x (d N)`` matrix; column ``e * N + i`` is the derivative w.r.t. ``W[e, i]``."""
    d, n = W.shape
    P = len(idx.pairs)
    p, t = idx.pairs, idx.triples
    J = np.zeros((idx.size, d, n))
    J[np.arange(d), np.arange(d), :] = 1.0
    rows = d + np.arange(P)
    np.add.at(J, (rows, p[:, 0]), W[p[:, 1]])
    np.add.at(J, (rows, p[:, 1]), W[p[:, 0]])
    rows = d + P + np.arange(len(t))
    np.add.at(J, (rows, t[:, 0]), W[t[:, 1]] * W[t[:, 2]])
    np.add.at(J, (rows, t[:, 1]), W[t[:, 0]] * W[t[:, 2]])
    np.add.at(J, (rows, t[:, 2]), W[t[:, 0]] * W[t[:, 1]])
    return J.reshape(idx.size, d * n)


def input_statistics(inputs: np.ndarray, idx: Indices) -> np.ndarray:
    """``Sigma = (1/n) sum_r phi(x_r) phi(x_r)^T``: statistics of experienced inputs (no targets)."""
    Phi = features(inputs, idx)
    return Phi.T @ Phi / len(inputs)


# --- candidates ---------------------------------------------------------------------------------------------------


class C1:
    """Coarse S3-space functional consolidation (CANDIDATES.md C1). State: H (F x F symmetric), b (F)."""

    family = "C1"

    def __init__(self, idx: Indices, lam: float, *, uninformed: bool = False):
        self.idx, self.lam, self.uninformed = idx, float(lam), uninformed
        self.H = np.zeros((idx.size, idx.size))
        self.b = np.zeros(idx.size)

    def consolidate(self, W: np.ndarray, inputs: np.ndarray) -> None:
        sigma = input_statistics(inputs, self.idx)
        if self.uninformed:
            sigma = np.eye(self.idx.size) * (np.trace(sigma) / self.idx.size)
        self.H += sigma
        self.b += sigma @ s3(W, self.idx)

    def grad(self, W: np.ndarray) -> np.ndarray:
        u = 0.5 * self.lam * (self.H @ s3(W, self.idx) - self.b)
        return s3_vjp(W, u, self.idx)

    def penalty(self, W: np.ndarray, anchors) -> float:
        """``P(W)`` for one consolidation (tests): ``(lambda/4) (S3 - s)^T Sigma (S3 - s)``."""
        delta = s3(W, self.idx) - anchors
        return 0.25 * self.lam * float(delta @ self.H @ delta)

    def state_bytes(self) -> int:
        F = self.idx.size
        return 8 * (F * (F + 1) // 2 + F)


class C2:
    """Local per-weight consolidation (CANDIDATES.md C2). State: Omega, A (d x N each)."""

    family = "C2"

    def __init__(self, idx: Indices, lam: float, shape: tuple, *, uninformed: bool = False):
        self.idx, self.lam, self.uninformed = idx, float(lam), uninformed
        self.Omega = np.zeros(shape)
        self.A = np.zeros(shape)

    def importance(self, W: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        sigma = input_statistics(inputs, self.idx)
        J = s3_jacobian(W, self.idx)
        omega = 0.25 * np.einsum("fk,fk->k", J, sigma @ J).reshape(W.shape)
        return np.full_like(omega, omega.mean()) if self.uninformed else omega

    def consolidate(self, W: np.ndarray, inputs: np.ndarray) -> None:
        omega = self.importance(W, inputs)
        self.Omega += omega
        self.A += omega * W

    def grad(self, W: np.ndarray) -> np.ndarray:
        return 2.0 * self.lam * (self.Omega * W - self.A)

    def state_bytes(self) -> int:
        return 8 * 2 * self.Omega.size


def make_candidate(family: str, lam: float, idx: Indices, shape: tuple, *, uninformed: bool = False):
    if family == "C1":
        return C1(idx, lam, uninformed=uninformed)
    if family == "C2":
        return C2(idx, lam, shape, uninformed=uninformed)
    raise ValueError(family)


def engine_learn(api, W_start: np.ndarray, X: np.ndarray, y: np.ndarray, rate: float, steps: int,
                 consolidation=None) -> tuple[np.ndarray, int, float]:
    """``W <- G1(W) - rate * grad P(W)`` for ``steps`` steps; G1 is the canonical native step.

    Returns (W, refused_step, seconds). A refusal leaves W at ``W_start`` (canonical semantics).
    """
    dim, width = W_start.shape
    xm, ym = memoryview(np.ascontiguousarray(X)), memoryview(np.ascontiguousarray(y))
    rows = len(y)
    W = np.ascontiguousarray(W_start, dtype=np.float64).copy()
    t0 = time.perf_counter()
    for k in range(steps):
        with np.errstate(over="ignore", invalid="ignore"):   # divergence is detected below and refused
            g = consolidation.grad(W) if consolidation is not None else None
        try:
            with Executor.create(api, dim, width, _flat(W), max_rows=rows) as e:
                e.learn(xm, ym, rate, 1)
                stepped = np.asarray(e.w(), dtype=np.float64).reshape(dim, width)
        except ECSGError:
            return W_start.copy(), k + 1, time.perf_counter() - t0
        if g is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                stepped = stepped - rate * g
            if not np.all(np.isfinite(stepped)):
                return W_start.copy(), k + 1, time.perf_counter() - t0
        W = stepped
    return W, 0, time.perf_counter() - t0


# --- representability analysis (EXPERIMENT_ONLY) --------------------------------------------------------------


def ceiling(experiences, tests, idx: Indices) -> list[np.ndarray]:
    """Unconstrained least-squares ``c`` for ``1/2 phi(x) . c`` on the joint experience; predictions per test set."""
    Phi = np.vstack([0.5 * features(x, idx) for x, _ in experiences])
    y = np.concatenate([t for _, t in experiences])
    c = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return [0.5 * features(x, idx) @ c for x in tests]


def flops_per_step(dim: int, width: int, rows: int, idx: Indices) -> dict:
    """Analytic floating-point operations per step (approximate, for capacity accounting)."""
    g1 = 4 * rows * width * dim + 12 * rows * width
    P, T, F = len(idx.pairs), len(idx.triples), idx.size
    c1 = width * (dim + 2 * P + 3 * T) + 2 * F * F + width * (dim + 2 * dim * dim + 9 * T)
    c2 = 4 * dim * width
    return {"G1": g1, "C1_extra": c1, "C2_extra": c2}
