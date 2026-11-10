"""R122 measurement harness: full-state trajectories over the R118-E composed system.

Everything numerical is the already-qualified machinery, consumed read-only:

* composed registry, distractor structure and windows: ``r118_scalable_awr.characterize_e``
  (``structure``, ``composed_registry``), exactly as R118-E / R120 / R121 used them;
* HCA steps: ``r119_numerical_hotpath.engine_r119.R119Engine`` in ``EXACT_GUARDED`` mode
  (the real 88-parameter frozen checkpoints, shared immutable weights, real gauge
  transport, MEAN reconciliation); controller choice per step from the frozen R121
  ``policy.controller_at``;
* teacher: the frozen R30 ``NonlinearTeacher`` composed through
  ``reference_adapter.oracle_candidates`` / ``reconcile_mean`` (R118-E's teacher path).

What is new is only *what is recorded*: the complete relation state at every step
instead of the paired readout at two horizons, plus offline teacher-forced one-step and
teacher-prefix ("restart") diagnostics.

Teacher data flow (R122 contract). Teacher states are produced first and are only ever
(a) compared with HCA outputs, or (b) loaded as the *input* of an offline diagnostic call
whose output never feeds an autonomous trajectory: autonomous rollouts start from the
shared initial state ``X0`` and call nothing but engine steps. ``measure.py`` re-runs
autonomous trajectories after all diagnostics and requires bit identity.

All state is volatile (numpy arrays and the AWRState pool); nothing here writes files.
"""
from __future__ import annotations

import math

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import characterize_e as e118
from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr import gauge as G118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr import reference_adapter as RA
from research.yang_mills_hca.r118_scalable_awr.awr_state import AWRState, StateCapacity
from research.yang_mills_hca.r118_scalable_awr.registry import blocks_to_matrix
from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine
from research.yang_mills_hca.r121_temporal_topology.policy import controller_at

COPIES = e118.COPIES            # 8: historical pair_state(batch=4) rows i and i + 4 form sample i
PAIRS = COPIES // 2
HMAX = 20
RELATION_BASE = 9_100_000_000_000  # characterize_e.composed_registry relation-ID base
VIEWS = ("READOUT", "PAIRED_STATE", "WEAK_STATE", "STRONG_STATE")
POLICIES = ("NATIVE", "R114", "SIGNED_ONLY", "DELAYED8")
TRAJECTORIES = ("NATIVE", "R114", "DELAYED8")   # distinct autonomous trajectories per checkpoint
SIGNED = "STATE8_SIGNED8"


def is_signed(checkpoint: str) -> bool:
    return checkpoint.endswith("/" + SIGNED)


def trajectory_of(checkpoint: str, policy: str) -> str:
    """Which distinct autonomous trajectory realizes ``policy`` for ``checkpoint`` (exact identity).

    SIGNED_ONLY and DELAYED8 never change the controller for unsigned checkpoints, and
    SIGNED_ONLY equals R114 at every step for signed ones; this is the frozen R120 / R121
    policy algebra, asserted against ``controller_at`` for all twenty steps.
    """
    for name in TRAJECTORIES:
        if all(controller_at(checkpoint, policy, s) == controller_at(checkpoint, name, s) for s in range(1, HMAX + 1)):
            return name
    raise RuntimeError("POLICY_TRAJECTORY:" + checkpoint + ":" + policy)


class System:
    """One composed system (fixture graph, distractor structure, overlap k) with a reusable AWR state.

    The AWRState and its engines are reused across fixtures; a rollout depends only on
    the pool contents it is given (``measure.py`` checks this against fresh states).
    """

    def __init__(self, registry, windows, fid, k: int, budgets=None):
        self.reg, self.fid, self.k = registry, fid, k
        wins = windows(k)
        n = len(wins)
        self.windows_n = n
        self.st = AWRState(registry, StateCapacity(max_instances=n, max_materialized_relations=4 * n,
                                                   max_bindings_per_relation=64))
        mask = np.array([[r is not None for r in w] for w in wins])
        rel = np.array([[0 if r is None else r for r in w] for w in wins], dtype=np.uint64)
        self.h, _ = self.st.create_instances(np.arange(n, dtype=np.uint64) + 1, rel, mask)
        sr = self.st.slot_row[self.h]
        self.rows = np.unique(sr[sr >= 0])
        frow = np.array([[int(self.st.rel_row[int(registry.relation_index([fid(c, d)])[0])]) for d in range(4)]
                         for c in range(COPIES)])
        self.fidx = np.searchsorted(self.rows, frow)       # fixture relation rows inside the compact list
        if not np.array_equal(self.rows[self.fidx], frow):
            raise RuntimeError("FIXTURE_ROW_INDEX")
        ids = registry.relation_ids[self.st.row_rel[self.rows]].astype(np.int64) - RELATION_BASE
        self.row_copy = (ids // 1000).astype(np.int64)     # copy index of every compact row
        self.row_local = (ids % 1000).astype(np.int64)     # 0..3 fixture relation, >= 100 distractor relation
        if self.row_copy.min() < 0 or self.row_copy.max() >= COPIES:
            raise RuntimeError("COPY_LAYOUT")
        self.budgets = budgets if budgets is not None else E118.Budgets(max_passes=HMAX)
        self._engines = {}
        self.engine_steps = 0

    # -- state I/O -------------------------------------------------------------------------
    def x0(self, state_a, state_b, scale: float = 1.0):
        """Initial compact state from historical ``pair_state`` arrays ``[8, 4, 4]`` (A, B)."""
        X = np.zeros((len(self.rows), 8))
        for c in range(COPIES):
            for d in range(4):
                X[self.fidx[c, d], :4] = np.asarray(state_a[c][d], dtype=np.float64) * scale
                X[self.fidx[c, d], 4:] = np.asarray(state_b[c][d], dtype=np.float64) * scale
        return X

    def load(self, X):
        self.st.pool[self.rows] = X

    def dump(self):
        return self.st.pool[self.rows].copy()

    # -- HCA ---------------------------------------------------------------------------------
    def engine(self, checkpoint: str, controller: str, mode: str = "EXACT_GUARDED"):
        key = (checkpoint, controller, mode)
        if key not in self._engines:
            self._engines[key] = R119Engine(self.st, checkpoint, controller=controller, mode=mode,
                                            budgets=self.budgets)
        return self._engines[key]

    def _advance(self, checkpoint, controller, mode="EXACT_GUARDED"):
        res = self.engine(checkpoint, controller, mode).step(self.h)
        if not res.exact or res.disposition != "EXECUTED":
            raise RuntimeError("ENGINE_REFUSAL:" + str(res.disposition))
        self.engine_steps += 1

    def step(self, X, checkpoint: str, controller: str, mode: str = "EXACT_GUARDED"):
        """One HCA pass from compact state ``X`` (offline diagnostic call)."""
        self.load(X)
        self._advance(checkpoint, controller, mode)
        return self.dump()

    def rollout(self, X, checkpoint: str, policy: str, start: int = 0, stop: int = HMAX, mode: str = "EXACT_GUARDED"):
        """Autonomous rollout from state ``X`` at time ``start``; returns states ``start..stop``.

        The controller at absolute step ``s`` is ``controller_at(checkpoint, policy, s)``.
        """
        out = np.empty((stop - start + 1,) + X.shape)
        out[0] = X
        self.load(X)
        for s in range(start + 1, stop + 1):
            self._advance(checkpoint, controller_at(checkpoint, policy, s), mode)
            out[s - start] = self.dump()
        return out

    # -- teacher (evaluation only) ---------------------------------------------------------
    def teacher(self, oracle, X0, steps: int = HMAX):
        """Composed R30 teacher trajectory ``[steps + 1, rows, 8]`` (R118-E teacher path)."""
        law = oracle.r30.NonlinearTeacher(oracle.torch)
        cur = {int(r): X0[i].copy() for i, r in enumerate(self.rows)}
        out = np.empty((steps + 1,) + X0.shape)
        out[0] = X0
        for s in range(1, steps + 1):
            cands = RA.oracle_candidates(self.reg, self.st.slot_rel[self.h], self.st.slot_row[self.h], cur,
                                         P.CHECKPOINTS[0], "NATIVE", law=law)
            if cands is None:
                raise RuntimeError("TEACHER_NONFINITE")
            cur = RA.reconcile_mean(cands)
            out[s] = np.array([cur[int(r)] for r in self.rows])
        return out

    def teacher_step(self, oracle, X):
        return self.teacher(oracle, X, steps=1)[1]

    # -- views ------------------------------------------------------------------------------
    def view(self, X, name: str):
        """Registered views of compact states ``[..., rows, 8]`` -> ``[..., 4 samples, dim]``.

        READOUT       A of the readout relation, copy i minus copy i + 4   (4-d, the historical metric)
        PAIRED_STATE  whole fixture window, copy i minus copy i + 4         (32-d)
        WEAK_STATE    whole fixture window of the weak-source copies 0..3   (32-d)
        STRONG_STATE  whole fixture window of the strong-source copies 4..7 (32-d)
        """
        if name == "READOUT":
            a3 = X[..., self.fidx[:, 3], :4]
            return a3[..., :PAIRS, :] - a3[..., PAIRS:, :]
        w = X[..., self.fidx, :]
        w = w.reshape(w.shape[:-2] + (32,))
        if name == "PAIRED_STATE":
            return w[..., :PAIRS, :] - w[..., PAIRS:, :]
        if name == "WEAK_STATE":
            return w[..., :PAIRS, :]
        if name == "STRONG_STATE":
            return w[..., PAIRS:, :]
        raise ValueError("VIEW:" + name)

    def view_parts(self, X, name: str):
        """The two halves (a, b) of a paired view (for conditioning), or None for unpaired views."""
        if name == "READOUT":
            a3 = X[..., self.fidx[:, 3], :4]
            return a3[..., :PAIRS, :], a3[..., PAIRS:, :]
        if name == "PAIRED_STATE":
            w = X[..., self.fidx, :]
            w = w.reshape(w.shape[:-2] + (32,))
            return w[..., :PAIRS, :], w[..., PAIRS:, :]
        return None


def build_systems(oracle, graph, structure_seed: int, ks=(0, 2, 6)):
    """Systems for one fixture graph: R118-E structure spec from ``structure_seed`` (already offset by graph index)."""
    spec = e118.structure(np.random.default_rng(structure_seed))
    reg, windows, fid = e118.composed_registry(oracle, graph, spec)
    return {k: System(reg, windows, fid, k) for k in ks}, (reg, windows, fid)


def gauged_system(base: System, parts, k: int, rng):
    """A jointly gauge-transformed copy of a system (frames and, via ``gauge_state``, states).

    The relation gauge is copy-symmetric (the same SO(2)xSO(2) element for corresponding
    relations of every copy), which is the symmetry of the paired views. Returns
    ``(system, g_rows)`` where ``g_rows`` are the gauge blocks of the compact rows.
    """
    reg, windows, fid = parts
    local = (reg.relation_ids.astype(np.int64) - RELATION_BASE) % 1000
    table = G118.random_so2x2(rng, 1000)
    g_rel = table[local]
    reg_g = G118.gauge_registry(reg, g_rel=g_rel)
    sys_g = System(reg_g, windows, fid, k)
    if not np.array_equal(sys_g.rows, base.rows):
        raise RuntimeError("GAUGED_ROW_LAYOUT")
    return sys_g, g_rel[sys_g.st.row_rel[sys_g.rows]]


def gauge_state(X, g_rows):
    """Apply relation gauges to compact states ``[..., rows, 8]``."""
    out = np.empty_like(X)
    out[..., :4] = G118.bapply(g_rows, X[..., :4])
    out[..., 4:] = G118.bapply(g_rows, X[..., 4:])
    return out


# ---- independent teacher reference (no torch) -----------------------------------------------
def _squash(x):
    return x / (1.0 + abs(x))


def _sigm(x):
    return 1.0 / (1.0 + math.exp(-x))


def teacher_coefficients(a, b, s, c):
    """The R30 NonlinearTeacher coefficient law (frozen ``r30_source.py``), re-implemented with ``math``."""
    def dot(x, y):
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
    q = [_squash(v) for v in (dot(a, a), dot(b, b), dot(s, s), dot(c, c), dot(a, s), dot(a, c), dot(b, s),
                              dot(b, c), dot(s, c))]
    return (_sigm(.65 + .55 * math.sin(2.3 * q[4]) - .25 * q[0] * q[8]),
            math.tanh(.25 + .70 * q[4] * q[4] - .32 * math.sin(2.1 * q[8])),
            math.tanh(.10 + .40 * math.sin(2.3 * q[5])),
            math.tanh(.03 + .32 * q[6] * q[7]),
            _sigm(.80 - .52 * math.sin(2.0 * q[7]) + .40 * q[1] * q[8]),
            math.tanh(-.10 - .38 * q[6] * q[6] + .27 * q[0]),
            math.tanh(.18 + .58 * math.sin(2.0 * q[8]) - .24 * q[5] * q[6]),
            math.tanh(.02 + .40 * q[4] * q[7]))


def reference_teacher_step(system: System, X):
    """One composed teacher pass in plain Python floats: an independent check of the torch path."""
    reg, st = system.reg, system.st
    pos = {int(r): i for i, r in enumerate(system.rows)}
    cands = {}
    for hnd in system.h.tolist():
        slot_rel, slot_row = st.slot_rel[hnd], st.slot_row[hnd]
        A = [[0.0] * 4 for _ in range(4)]
        B = [[0.0] * 4 for _ in range(4)]
        for d in range(4):
            if slot_row[d] >= 0:
                v = X[pos[int(slot_row[d])]]
                A[d], B[d] = [float(x) for x in v[:4]], [float(x) for x in v[4:]]
        links = reg.window_links(slot_rel)
        for d in range(4):
            if slot_row[d] < 0:
                continue
            w, wc = [], []
            for (dd, s, _p, inc_d, inc_s) in links:
                if dd != d:
                    continue
                L = blocks_to_matrix(reg.frames[inc_d]) @ blocks_to_matrix(reg.frames[inc_s]).T
                w.append([sum(L[i][j] * A[s][j] for j in range(4)) for i in range(4)])
                wc.append([sum(L[i][j] * B[s][j] for j in range(4)) for i in range(4)])
            sal, col = [0.0] * 4, [0.0] * 4
            if w:
                x = [8.0 * math.sqrt(sum(v * v for v in vec)) for vec in w]
                mx = max(x)
                ex = [math.exp(v - mx) for v in x]
                tot = sum(ex)
                for wt, vec in zip(ex, w):
                    for i in range(4):
                        sal[i] += wt / tot * vec[i]
                for i in range(4):
                    col[i] = sum(vec[i] for vec in wc) / len(wc)
            kk = teacher_coefficients(A[d], B[d], sal, col)
            ua = [kk[0] * A[d][i] + kk[1] * sal[i] + kk[2] * col[i] + kk[3] * B[d][i] for i in range(4)]
            ub = [kk[4] * B[d][i] + kk[5] * sal[i] + kk[6] * col[i] + kk[7] * A[d][i] for i in range(4)]
            for v in (ua, ub):
                scale = min(4.0 / (math.sqrt(sum(x * x for x in v)) + 1e-30), 1.0)
                for i in range(4):
                    v[i] *= scale
            cands.setdefault(int(slot_row[d]), []).append(ua + ub)
    out = np.empty_like(X)
    for row, vecs in cands.items():
        for i in range(8):
            vals = sorted(v[i] for v in vecs)
            acc = vals[0]
            for x in vals[1:]:
                acc += x
            out[pos[row], i] = acc / len(vecs)
    return out
