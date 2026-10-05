"""Retention R3 learning engine: the K1 law of the closed R1/R2 lineage, and the C1R reference. RESEARCH_ONLY.

* Canonical paths (M0, M1): the canonical core and executor, unchanged.
* Mechanism engine (K1, C1R): every step is the canonical native G1 step of the current W (an executor created
  from W performs one learn step), followed by the mechanism's correction evaluated at the pre-step W. With the
  mechanism removed the engine is bitwise the canonical core.
* Consolidation receives the current W and the current experience's inputs only; no target ever reaches it, and
  it is closed form (no iteration).
* The authoritative K1 state is (W, epoch, H, a). ``query`` reads W only. ``reset`` empties H and a and keeps W
  and the epoch. ``import_w_only`` turns a canonical ELPISG01 W-only snapshot into an UNCONSOLIDATED state, never a
  retained one.

S3 coordinates follow ``ecsg_math.h``: ``f_W(x) = 1/2 phi(x) . S3(W)``. H is kept exactly symmetric, so its packed
serialization is lossless. The mathematics of K1 and C1R is the R2 laboratory's, unchanged.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time

import numpy as np

from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.native import ECSGError, Executor

from .task import Indices, features

SNAPSHOT_HEADER = 40
SNAPSHOT_MAGIC = b"ELPISG01"
STATE_FORMAT = "elpis.research.ecs-retention-r3.state.v1"
PROVENANCE = ("COMPLETE", "RESET", "UNCONSOLIDATED_IMPORT")


def w_digest(W: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(W, dtype="<f8").tobytes()).hexdigest()


def _flat(W: np.ndarray) -> memoryview:
    return memoryview(np.ascontiguousarray(W, dtype=np.float64).reshape(-1))


def w_of(snapshot: bytes, dim: int, width: int) -> np.ndarray:
    return np.frombuffer(snapshot, dtype="<f8", offset=SNAPSHOT_HEADER).reshape(dim, width).astype(np.float64)


# --- canonical paths --------------------------------------------------------------------------------------------


def canonical_learn(api, W: np.ndarray, X: np.ndarray, y: np.ndarray, rate: float, steps: int):
    """K canonical G1 steps through CognitiveCore (one native call). Returns (W, refused_step); refusal keeps W."""
    dim, width = W.shape
    with CognitiveCore.create(api, dim, width, _flat(W), learning_rate=rate, max_rows=max(len(y), 1)) as core:
        try:
            core.learn(memoryview(np.ascontiguousarray(X)), memoryview(np.ascontiguousarray(y)), steps=steps,
                       receipt=False)
        except ECSGError as exc:
            return np.array(W, dtype=np.float64, copy=True), max(int(exc.step), 1)
        return w_of(core.snapshot(), dim, width), 0


def responses(api, W: np.ndarray, X: np.ndarray) -> np.ndarray:
    dim, width = W.shape
    with Executor.create(api, dim, width, _flat(W)) as e:
        return np.asarray(e.forward(memoryview(np.ascontiguousarray(X))), dtype=np.float64)


def native_s3(api, W: np.ndarray) -> np.ndarray:
    dim, width = W.shape
    with Executor.create(api, dim, width, _flat(W)) as e:
        return np.asarray(e.s3(), dtype=np.float64)


def query(api, state, X: np.ndarray) -> np.ndarray:
    """The response of a complete state ``(mechanism, W, epoch)``: the canonical forward map of W, nothing else."""
    _, W, _ = state
    return responses(api, W, X)


# --- S3 calculus ------------------------------------------------------------------------------------------------


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


def jacobian3(W: np.ndarray, idx: Indices) -> np.ndarray:
    """``dS3/dW`` as ``(F, d, N)``; flattened to ``F x (d N)`` its column ``e * N + i`` is ``d/dW[e, i]``."""
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
    return J


def jacobian(W: np.ndarray, idx: Indices) -> np.ndarray:
    return jacobian3(W, idx).reshape(idx.size, W.size)


def input_statistics(inputs: np.ndarray, idx: Indices) -> np.ndarray:
    """``Sigma = (1/n) sum_r phi(x_r) phi(x_r)^T`` from experienced inputs only, made exactly symmetric."""
    Phi = features(inputs, idx)
    S = Phi.T @ Phi / len(inputs)
    return 0.5 * (S + S.T)


def pack_upper(H: np.ndarray) -> np.ndarray:
    return H[np.triu_indices(H.shape[0])]


def unpack_upper(packed: np.ndarray, F: int) -> np.ndarray:
    H = np.zeros((F, F))
    H[np.triu_indices(F)] = packed
    return H + np.triu(H, 1).T


# --- analytic operation counts (capacity accounting and the native budget; spec native_budget) --------------------


def g1_ops(dim: int, width: int, rows: int) -> int:
    return 4 * rows * width * dim + 12 * rows * width


def k1_ops(dim: int, width: int, idx: Indices) -> int:
    """Extra operations of one K1 correction: S3 (N(d + 2P + 3T)), the F x F mat-vec, the S3 VJP."""
    P, T, F = len(idx.pairs), len(idx.triples), idx.size
    return width * (dim + 2 * P + 3 * T) + 2 * F * F + width * (dim + 2 * dim * dim + 9 * T)


def k1_consolidation_ops(dim: int, width: int, rows: int, idx: Indices) -> int:
    """One K1 consolidation: features (R F), packed Sigma (R F (F + 1)), H update (F (F + 1) / 2), S3."""
    P, T, F = len(idx.pairs), len(idx.triples), idx.size
    return rows * F + rows * F * (F + 1) + F * (F + 1) // 2 + width * (dim + 2 * P + 3 * T)


# --- mechanisms ----------------------------------------------------------------------------------------------------


class Mechanism:
    """Common interface. ``correct`` maps (pre-step W, canonical stepped W) to the next W; ``consolidate`` returns
    W unchanged (both R3 mechanisms consolidate in closed form) and an information dict."""

    kind = "NONE"

    def __init__(self, idx: Indices, removed: bool = False):
        self.idx, self.removed = idx, removed

    def correct(self, W, stepped, rate):
        return stepped

    def consolidate(self, W, inputs):
        return W, {}

    def persistent_bytes(self) -> int:
        return 0

    def hot_ops(self, dim, width) -> int:
        return 0

    def consolidation_ops(self, dim, width, rows) -> int:
        return 0

    def state_vectors(self) -> list:
        return []

    def constants(self) -> dict:
        return {}

    def clone(self):
        raise NotImplementedError


class C1R(Mechanism):
    """Retention R0's C1 unchanged (lambda 4): H, b dense exactly as R0 computed them (REFERENCE)."""

    kind = "C1R"

    def __init__(self, idx, lam=4.0, removed=False):
        super().__init__(idx, removed)
        self.lam = float(lam)
        self.H = np.zeros((idx.size, idx.size))
        self.b = np.zeros(idx.size)

    def clone(self):
        m = C1R(self.idx, self.lam, self.removed)
        m.H, m.b = self.H.copy(), self.b.copy()
        return m

    def consolidate(self, W, inputs):
        if self.removed:
            return W, {}
        Phi = features(inputs, self.idx)
        sigma = Phi.T @ Phi / len(inputs)
        self.H += sigma
        self.b += sigma @ s3(W, self.idx)
        return W, {}

    def grad(self, W):
        u = 0.5 * self.lam * (self.H @ s3(W, self.idx) - self.b)
        return s3_vjp(W, u, self.idx)

    def correct(self, W, stepped, rate):
        if self.removed:
            return stepped
        return stepped - rate * self.grad(W)

    def persistent_bytes(self):
        F = self.idx.size
        return 8 * (F * (F + 1) // 2 + F)

    def hot_ops(self, dim, width):
        return 0 if self.removed else k1_ops(dim, width, self.idx)

    def state_vectors(self):
        return [self.H.reshape(-1), self.b]

    def constants(self):
        return {"lambda": self.lam}


class K1(Mechanism):
    """Exact functional consolidation (S3 Laplace, re-anchored), lambda = 1: persistent H (packed) and a."""

    kind = "K1"

    def __init__(self, idx, removed=False, provenance="COMPLETE"):
        super().__init__(idx, removed)
        if provenance not in PROVENANCE:
            raise ValueError(f"unknown K1 provenance {provenance!r}")
        self.provenance = provenance
        self.H = np.zeros((idx.size, idx.size))
        self.a = np.zeros(idx.size)

    def clone(self):
        m = K1(self.idx, self.removed, self.provenance)
        m.H, m.a = self.H.copy(), self.a.copy()
        return m

    def consolidate(self, W, inputs):
        if self.removed:
            return W, {}
        self.H = self.H + input_statistics(inputs, self.idx)
        self.a = s3(W, self.idx)
        return W, {}

    def grad(self, W):
        u = 0.5 * (self.H @ (s3(W, self.idx) - self.a))
        return s3_vjp(W, u, self.idx)

    def objective(self, W):
        delta = s3(W, self.idx) - self.a
        return 0.25 * float(delta @ self.H @ delta)

    def correct(self, W, stepped, rate):
        if self.removed:
            return stepped
        return stepped - rate * self.grad(W)

    def persistent_bytes(self):
        F = self.idx.size
        return 8 * (F * (F + 1) // 2 + F)

    def hot_ops(self, dim, width):
        return 0 if self.removed else k1_ops(dim, width, self.idx)

    def consolidation_ops(self, dim, width, rows):
        return 0 if self.removed else k1_consolidation_ops(dim, width, rows, self.idx)

    def state_vectors(self):
        return [pack_upper(self.H), self.a]

    def constants(self):
        return {"lambda": 1.0}


def make(kind: str, idx: Indices, *, removed=False, lam=None) -> Mechanism:
    if kind == "C1R":
        return C1R(idx, 4.0 if lam is None else lam, removed=removed)
    if kind == "K1":
        return K1(idx, removed=removed)
    raise ValueError(f"R3 runs K1 and C1R only, not {kind!r}")


def reset(state):
    """The registered reset: H <- 0, a <- 0; W and the epoch kept bitwise (copies)."""
    mech, W, epoch = state
    if not isinstance(mech, K1):
        raise TypeError("reset applies to a K1 state")
    return K1(mech.idx, provenance="RESET"), np.array(W, dtype=np.float64, copy=True), int(epoch)


# --- the declared complete state --------------------------------------------------------------------------------


def serialize(mech: Mechanism, W: np.ndarray, epoch: int) -> bytes:
    """Deterministic bytes of the complete declared state: header, W, persistent consolidation state."""
    header = {"format": STATE_FORMAT, "kind": mech.kind, "dim": int(W.shape[0]), "width": int(W.shape[1]),
              "features": mech.idx.size, "epoch": int(epoch), "removed": bool(mech.removed),
              "provenance": getattr(mech, "provenance", "COMPLETE"), "constants": mech.constants()}
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    body = [np.ascontiguousarray(W, dtype="<f8").tobytes()]
    body += [np.ascontiguousarray(v, dtype="<f8").tobytes() for v in mech.state_vectors()]
    return struct.pack("<I", len(head)) + head + b"".join(body)


def deserialize(blob: bytes, idx: Indices):
    """Inverse of ``serialize`` for K1. Returns (mechanism, W, epoch). Refuses anything else or a short body."""
    (n,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4:4 + n])
    if header.get("format") != STATE_FORMAT or header.get("kind") != "K1" or header.get("features") != idx.size:
        raise ValueError("not a Retention R3 K1 state")
    d, w, F = header["dim"], header["width"], idx.size
    packed = F * (F + 1) // 2
    data = np.frombuffer(blob, dtype="<f8", offset=4 + n).astype(np.float64)
    if data.size != d * w + packed + F:
        raise ValueError("truncated or oversized Retention R3 K1 state")
    mech = K1(idx, removed=header["removed"], provenance=header["provenance"])
    mech.H = unpack_upper(data[d * w:d * w + packed], F)
    mech.a = data[d * w + packed:].copy()
    return mech, data[:d * w].reshape(d, w).copy(), header["epoch"]


def import_w_only(snapshot: bytes, idx: Indices, dim: int, width: int):
    """A canonical ELPISG01 W-only snapshot imported as K1: UNCONSOLIDATED (H = 0, a = 0), never a retained state."""
    # ELPISG01 R0 layout: magic[8], version u32, reserved u32, dim u64, width u64, epoch u64, W (little-endian).
    shape = (int.from_bytes(snapshot[16:24], "little"), int.from_bytes(snapshot[24:32], "little"))
    if snapshot[:8] != SNAPSHOT_MAGIC or shape != (dim, width) or len(snapshot) != SNAPSHOT_HEADER + 8 * dim * width:
        raise ValueError("not a canonical W-only snapshot of this shape")
    epoch = int.from_bytes(snapshot[32:40], "little")
    return K1(idx, provenance="UNCONSOLIDATED_IMPORT"), w_of(snapshot, dim, width), epoch


def state_digest(mech: Mechanism, W: np.ndarray, epoch: int) -> str:
    return hashlib.sha256(serialize(mech, W, epoch)).hexdigest()


# --- the mechanism engine -----------------------------------------------------------------------------------------


def engine_learn(api, W_start: np.ndarray, X: np.ndarray, y: np.ndarray, rate: float, steps: int,
                 mech: Mechanism):
    """``steps`` canonical native G1 steps of W on (X, y), each followed by the mechanism's correction.

    Returns (W, refused_step, seconds). A refused native step or any non-finite value refuses the whole
    experience: W is returned as ``W_start`` (canonical refusal semantics).
    """
    dim, width = W_start.shape
    xm, ym = memoryview(np.ascontiguousarray(X)), memoryview(np.ascontiguousarray(y))
    rows = len(y)
    W = np.ascontiguousarray(W_start, dtype=np.float64).copy()
    t0 = time.perf_counter()
    for k in range(steps):
        try:
            with Executor.create(api, dim, width, _flat(W), max_rows=rows) as e:
                e.learn(xm, ym, rate, 1)
                stepped = np.asarray(e.w(), dtype=np.float64).reshape(dim, width)
        except ECSGError:
            return W_start.copy(), k + 1, time.perf_counter() - t0
        with np.errstate(over="ignore", invalid="ignore"):
            nxt = mech.correct(W, stepped, rate)
        if not np.all(np.isfinite(nxt)):
            return W_start.copy(), k + 1, time.perf_counter() - t0
        W = np.ascontiguousarray(nxt)
    return W, 0, time.perf_counter() - t0


# --- representability ceilings (EXPERIMENT_ONLY) --------------------------------------------------------------


def ceiling(experiences, tests, idx: Indices) -> list[np.ndarray]:
    """Minimum-norm least-squares ``c`` for ``1/2 phi(x) . c`` on the joint training rows; predictions per test set."""
    Phi = np.vstack([0.5 * features(x, idx) for x, _ in experiences])
    y = np.concatenate([t for _, t in experiences])
    c = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return [0.5 * features(x, idx) @ c for x in tests]
