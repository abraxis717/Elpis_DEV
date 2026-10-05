"""Retention R2 learning engines and mechanisms (CANDIDATES.md sections 6-8). RESEARCH_ONLY.

* Canonical paths (M0, M1): the canonical core and executor, unchanged.
* Candidate engine (C1R, K1, K2, K3): every step is the canonical native G1 step of the current W (an executor
  created from W performs one learn step), followed by the mechanism's correction evaluated at the pre-step W.
  A mechanism in ``removed`` mode applies no correction and no consolidation: the engine is then bitwise the
  canonical core.
* Consolidation receives the current W and the current experience's inputs only (and, for the K2 uninformed
  ablation, its own random stream); no target ever reaches it. Queries never see consolidation state: every
  response is the canonical native forward map of W.

S3 coordinates follow ``ecsg_math.h``: ``f_W(x) = 1/2 phi(x) . S3(W)``. Persistent consolidation state of K1, K2
and K3 is kept exactly symmetric, so its packed serialization is lossless.
"""
from __future__ import annotations

import hashlib
from itertools import permutations
import json
import struct
import time

import numpy as np

from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.native import ECSGError, Executor

from .task import Indices, features

SNAPSHOT_HEADER = 40
STATE_FORMAT = "elpis.research.ecs-retention-r2.state.v1"

# Registered constants (spec mechanisms K2, K3).
PROTECTED_CUTOFF = 1e-9
GRAM_CUTOFF = 1e-12
FIBRE_MAX_ITERATIONS = 200
FIBRE_INITIAL_STEP = 0.01
FIBRE_MAX_CONSECUTIVE_REJECTIONS = 20
RETRACTION_MAX_ITERATIONS = 10
RETRACTION_TOLERANCE = 1e-13


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


def hessian_vjp(W: np.ndarray, G: np.ndarray, idx: Indices) -> np.ndarray:
    """``d <G, J(W)> / dW`` for ``G`` shaped like ``jacobian3``: the S3 Hessian contracted with ``G``."""
    d = idx.dim
    P = len(idx.pairs)
    p, t = idx.pairs, idx.triples
    out = np.zeros_like(W)
    rows = d + np.arange(P)
    for a, b in ((0, 1), (1, 0)):
        np.add.at(out, p[:, a], G[rows, p[:, b], :])
    rows = d + P + np.arange(len(t))
    for a, b, c in permutations(range(3)):
        np.add.at(out, t[:, a], G[rows, t[:, b], :] * W[t[:, c]])
    return out


def pinv_sym(M: np.ndarray, cutoff: float) -> np.ndarray:
    """Pseudo-inverse of a symmetric positive semidefinite matrix with a relative eigenvalue cutoff."""
    w, V = np.linalg.eigh(M)
    top = float(w[-1]) if w.size else 0.0
    if not np.isfinite(top) or top <= 0.0:
        return np.zeros_like(M)
    keep = w > cutoff * top
    Vk = V[:, keep]
    return (Vk / w[keep]) @ Vk.T


def input_statistics(inputs: np.ndarray, idx: Indices) -> np.ndarray:
    """``Sigma = (1/n) sum_r phi(x_r) phi(x_r)^T`` from experienced inputs only, made exactly symmetric."""
    Phi = features(inputs, idx)
    S = Phi.T @ Phi / len(inputs)
    return 0.5 * (S + S.T)


def protected_basis(H: np.ndarray) -> np.ndarray:
    """Eigenvectors of H with eigenvalue above PROTECTED_CUTOFF x the largest (K2, K3): ``F x k``."""
    w, V = np.linalg.eigh(H)
    top = float(w[-1]) if w.size else 0.0
    if not np.isfinite(top) or top <= 0.0:
        return np.zeros((H.shape[0], 0))
    return np.ascontiguousarray(V[:, w > PROTECTED_CUTOFF * top])


def pack_upper(H: np.ndarray) -> np.ndarray:
    return H[np.triu_indices(H.shape[0])]


def unpack_upper(packed: np.ndarray, F: int) -> np.ndarray:
    H = np.zeros((F, F))
    H[np.triu_indices(F)] = packed
    return H + np.triu(H, 1).T


# --- analytic operation counts (capacity accounting; spec L) ----------------------------------------------------


def g1_ops(dim: int, width: int, rows: int) -> int:
    return 4 * rows * width * dim + 12 * rows * width


def k1_ops(dim: int, width: int, idx: Indices) -> int:
    P, T, F = len(idx.pairs), len(idx.triples), idx.size
    return width * (dim + 2 * P + 3 * T) + 2 * F * F + width * (dim + 2 * dim * dim + 9 * T)


def k2_ops(dim: int, width: int, idx: Indices, k: int) -> int:
    if k == 0:
        return 0
    P, T, F = len(idx.pairs), len(idx.triples), idx.size
    dn = dim * width
    jac = width * (2 * P + 6 * T)
    return jac + 2 * k * F * dn + 2 * k * k * dn + 9 * k ** 3 + 2 * k * k + 4 * k * dn


def _jk_ops(idx: Indices, dn: int) -> int:
    F = idx.size
    return 2 * F * F * dn + 9 * F ** 3


# --- mechanisms ----------------------------------------------------------------------------------------------------


class Mechanism:
    """Common interface. ``correct`` maps (pre-step W, canonical stepped W) to the next W; ``consolidate`` returns
    the W after consolidation (changed only by K3's function-preserving reconditioning)."""

    kind = "NONE"

    def __init__(self, idx: Indices, removed: bool = False):
        self.idx, self.removed = idx, removed

    def correct(self, W, stepped, rate):
        return stepped

    def consolidate(self, W, inputs, rng=None):
        return W, {}

    def persistent_bytes(self) -> int:
        return 0

    def hot_ops(self, dim, width) -> int:
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

    def __init__(self, idx, lam=4.0, removed=False, uninformed=False):
        super().__init__(idx, removed)
        self.lam, self.uninformed = float(lam), uninformed
        self.H = np.zeros((idx.size, idx.size))
        self.b = np.zeros(idx.size)

    def clone(self):
        m = C1R(self.idx, self.lam, self.removed, self.uninformed)
        m.H, m.b = self.H.copy(), self.b.copy()
        return m

    def consolidate(self, W, inputs, rng=None):
        if self.removed:
            return W, {}
        Phi = features(inputs, self.idx)
        sigma = Phi.T @ Phi / len(inputs)
        if self.uninformed:
            sigma = np.eye(self.idx.size) * (np.trace(sigma) / self.idx.size)
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
    """Exact functional consolidation (S3 Laplace, re-anchored), lambda = 1: H, a."""

    kind = "K1"

    def __init__(self, idx, removed=False, uninformed=False):
        super().__init__(idx, removed)
        self.uninformed = uninformed
        self.H = np.zeros((idx.size, idx.size))
        self.a = np.zeros(idx.size)

    def _copy_into(self, m):
        m.H, m.a = self.H.copy(), self.a.copy()
        return m

    def clone(self):
        return self._copy_into(K1(self.idx, self.removed, self.uninformed))

    def _accumulate(self, inputs):
        sigma = input_statistics(inputs, self.idx)
        if self.uninformed:
            sigma = np.eye(self.idx.size) * (np.trace(sigma) / self.idx.size)
        self.H = self.H + sigma

    def consolidate(self, W, inputs, rng=None):
        if self.removed:
            return W, {}
        self._accumulate(inputs)
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

    def state_vectors(self):
        return [pack_upper(self.H), self.a]

    def constants(self):
        return {"lambda": 1.0}


class K2(Mechanism):
    """Protected functional subspace: H; U derived; hard first-order constraint through the current J."""

    kind = "K2"

    def __init__(self, idx, removed=False, uninformed=False):
        super().__init__(idx, removed)
        self.uninformed = uninformed
        self.H = np.zeros((idx.size, idx.size))
        self.U = np.zeros((idx.size, 0))

    def clone(self):
        m = K2(self.idx, self.removed, self.uninformed)
        m.H, m.U = self.H.copy(), self.U.copy()
        return m

    def derive(self, rng=None):
        U = protected_basis(self.H)
        if self.uninformed and U.shape[1]:
            Z = rng.standard_normal(size=(self.idx.size, U.shape[1]))
            U = np.linalg.qr(Z)[0]
        self.U = np.ascontiguousarray(U)

    def consolidate(self, W, inputs, rng=None):
        if self.removed:
            return W, {}
        self.H = self.H + input_statistics(inputs, self.idx)
        self.derive(rng)
        return W, {"protected_dimension": int(self.U.shape[1])}

    def projection(self, W, delta):
        """``C^T (C C^T)^+ C delta`` with ``C = U^T J(W)`` (delta flattened like W).

        Computed from the singular value decomposition of C itself: with ``C = A diag(s) V^T`` and the registered
        Gram cutoff (keep ``s_i^2 > GRAM_CUTOFF * s_max^2``, the relative eigenvalue cutoff of ``C C^T``), the
        operator is exactly ``V_k V_k^T``. The same mathematics as forming ``(C C^T)^+``, without squaring the
        condition number (RET2B implementation decision, recorded before DEV).
        """
        C = self.U.T @ jacobian(W, self.idx)
        _, sv, Vt = np.linalg.svd(C, full_matrices=False)
        if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] <= 0.0:
            return np.zeros(delta.size)
        Vk = Vt[sv * sv > GRAM_CUTOFF * sv[0] * sv[0]]
        return Vk.T @ (Vk @ delta.reshape(-1))

    def correct(self, W, stepped, rate):
        if self.removed or self.U.shape[1] == 0:
            return stepped
        return stepped - self.projection(W, stepped - W).reshape(W.shape)

    def persistent_bytes(self):
        F = self.idx.size
        return 8 * (F * (F + 1) // 2)

    def hot_ops(self, dim, width):
        return 0 if self.removed else k2_ops(dim, width, self.idx, int(self.U.shape[1]))

    def state_vectors(self):
        return [pack_upper(self.H)]

    def constants(self):
        return {"protected_eigenvalue_cutoff": PROTECTED_CUTOFF, "gram_pseudoinverse_cutoff": GRAM_CUTOFF}


def omega_terms(W: np.ndarray, P: np.ndarray, idx: Indices):
    """``Omega(W) = |P K (I - P)|_F^2 / |K|_F^2`` and its gradient; also returns J (flat) and K."""
    J3 = jacobian3(W, idx)
    J = J3.reshape(idx.size, W.size)
    K = J @ J.T
    Q = np.eye(idx.size) - P
    A = P @ K @ Q
    f, g = float(np.sum(A * A)), float(np.sum(K * K))
    omega = f / g
    B = (A + A.T - 2.0 * omega * K) / g
    GJ = (2.0 * (B @ J)).reshape(J3.shape)
    return omega, hessian_vjp(W, GJ, idx), J, K


def retract(W: np.ndarray, target: np.ndarray, idx: Indices):
    """Minimum-norm Gauss-Newton retraction onto ``S3 = target``. Returns (success, W, newton_iterations)."""
    tol = RETRACTION_TOLERANCE * max(1.0, float(np.max(np.abs(target))))
    for it in range(RETRACTION_MAX_ITERATIONS + 1):
        r = target - s3(W, idx)
        if not np.all(np.isfinite(r)):
            return False, W, it
        if float(np.max(np.abs(r))) <= tol:
            return True, W, it
        if it == RETRACTION_MAX_ITERATIONS:
            return False, W, it
        J = jacobian(W, idx)
        W = W + (J.T @ (pinv_sym(J @ J.T, GRAM_CUTOFF) @ r)).reshape(W.shape)
    return False, W, RETRACTION_MAX_ITERATIONS


def recondition(W: np.ndarray, U: np.ndarray, idx: Indices):
    """K3's fixed procedure over the fibre ``{W' : S3(W') = S3(W)}`` (spec mechanisms.K3.procedure)."""
    F, k = idx.size, U.shape[1]
    info = {"protected_dimension": int(k), "iterations": 0, "accepted": 0, "newton_iterations": 0,
            "omega_before": 0.0, "omega_after": 0.0, "displacement": 0.0, "cold_ops": 0}
    if k == 0 or k == F:
        return W, info
    P = U @ U.T
    target = s3(W, idx)
    start = W
    omega, grad, J, K = omega_terms(W, P, idx)
    info["omega_before"] = omega
    dn = W.size
    per_eval = _jk_ops(idx, dn) + 4 * F ** 3 + 2 * F * F * dn
    ops = per_eval
    alpha, rejections = FIBRE_INITIAL_STEP, 0
    while info["iterations"] < FIBRE_MAX_ITERATIONS and rejections < FIBRE_MAX_CONSECUTIVE_REJECTIONS:
        info["iterations"] += 1
        g = grad.reshape(-1)
        gf = g - J.T @ (pinv_sym(K, GRAM_CUTOFF) @ (J @ g))
        ops += 9 * F ** 3 + 4 * F * dn
        norm = float(np.linalg.norm(gf))
        if not np.isfinite(norm) or norm == 0.0:
            break
        trial = W - (alpha * float(np.linalg.norm(W)) / norm) * gf.reshape(W.shape)
        ok, trial, newton = retract(trial, target, idx)
        info["newton_iterations"] += newton
        ops += newton * _jk_ops(idx, dn)
        accepted = False
        if ok:
            t_omega, t_grad, t_J, t_K = omega_terms(trial, P, idx)
            ops += per_eval
            if np.isfinite(t_omega) and t_omega < omega:
                W, omega, grad, J, K = trial, t_omega, t_grad, t_J, t_K
                accepted = True
        if accepted:
            info["accepted"] += 1
            rejections = 0
        else:
            alpha *= 0.5
            rejections += 1
    info["omega_after"] = omega
    info["displacement"] = float(np.linalg.norm(W - start) / np.linalg.norm(start))
    info["cold_ops"] = int(ops)
    return W, info


class K3(K1):
    """K1 plus the function-preserving fibre reconditioning after every consolidation."""

    kind = "K3"

    def clone(self):
        return self._copy_into(K3(self.idx, self.removed, self.uninformed))

    def consolidate(self, W, inputs, rng=None):
        if self.removed:
            return W, {}
        self._accumulate(inputs)
        W2, info = recondition(W, protected_basis(self.H), self.idx)
        self.a = s3(W2, self.idx)
        return W2, info

    def constants(self):
        return {"lambda": 1.0, "protected_eigenvalue_cutoff": PROTECTED_CUTOFF, "procedure": {
            "max_iterations": FIBRE_MAX_ITERATIONS, "initial_relative_step": FIBRE_INITIAL_STEP,
            "stop_after_consecutive_rejections": FIBRE_MAX_CONSECUTIVE_REJECTIONS,
            "retraction_iterations": RETRACTION_MAX_ITERATIONS, "retraction_tolerance": RETRACTION_TOLERANCE,
            "gram_pseudoinverse_cutoff": GRAM_CUTOFF}}


def make(kind: str, idx: Indices, *, removed=False, uninformed=False, lam=None) -> Mechanism:
    if kind == "C1R":
        return C1R(idx, 4.0 if lam is None else lam, removed=removed, uninformed=uninformed)
    return {"K1": K1, "K2": K2, "K3": K3}[kind](idx, removed=removed, uninformed=uninformed)


# --- the declared extended state -------------------------------------------------------------------------------


def serialize(mech: Mechanism, W: np.ndarray, epoch: int) -> bytes:
    """Deterministic bytes of the complete declared state: header, W, persistent consolidation state."""
    header = {"format": STATE_FORMAT, "kind": mech.kind, "dim": int(W.shape[0]), "width": int(W.shape[1]),
              "features": mech.idx.size, "epoch": int(epoch), "removed": bool(mech.removed),
              "uninformed": bool(getattr(mech, "uninformed", False)), "constants": mech.constants()}
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    body = [np.ascontiguousarray(W, dtype="<f8").tobytes()]
    body += [np.ascontiguousarray(v, dtype="<f8").tobytes() for v in mech.state_vectors()]
    return struct.pack("<I", len(head)) + head + b"".join(body)


def deserialize(blob: bytes, idx: Indices, rng=None):
    """Inverse of ``serialize`` for K1, K2 and K3. Returns (mechanism, W, epoch); derived state is recomputed."""
    (n,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4:4 + n])
    if header["format"] != STATE_FORMAT or header["features"] != idx.size:
        raise ValueError("not a Retention R2 state")
    d, w, F = header["dim"], header["width"], idx.size
    data = np.frombuffer(blob, dtype="<f8", offset=4 + n).astype(np.float64)
    W = data[:d * w].reshape(d, w).copy()
    rest = data[d * w:]
    mech = make(header["kind"], idx, removed=header["removed"], uninformed=header["uninformed"])
    packed = F * (F + 1) // 2
    mech.H = unpack_upper(rest[:packed], F)
    if header["kind"] in ("K1", "K3"):
        mech.a = rest[packed:packed + F].copy()
    if header["kind"] == "K2" and not mech.removed:
        mech.derive(rng)
    return mech, W, header["epoch"]


def state_digest(mech: Mechanism, W: np.ndarray, epoch: int) -> str:
    return hashlib.sha256(serialize(mech, W, epoch)).hexdigest()


# --- the candidate engine ----------------------------------------------------------------------------------------


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
