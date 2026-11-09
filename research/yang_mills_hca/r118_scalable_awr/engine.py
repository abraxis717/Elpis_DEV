"""R118 hot path: active selection -> sparse gather -> gauge transport -> shared frozen
HCA -> controller -> sparse reconciliation -> updated volatile AWR.

Semantics (fixed, documented in ARCHITECTURE.md):

* **Synchronous (Jacobi) passes.** Every active instance in a pass reads the same
  pre-pass relation states; candidates are reconciled; then all touched rows
  change together. Relations not bound by an active instance are untouched.
* **Reconciliation.** ``MEAN``: a relation reached by several active
  (instance, slot) bindings receives the per-coordinate mean of its candidates,
  summed in ascending value order (independent of active-set order; covariant
  because every candidate is expressed in that relation's own frame).
  ``EXCLUSIVE``: overlapping active windows are refused.
* **Budgets.** Checked before work; exhaustion returns a ``REFUSED_*``
  disposition and leaves the AWR unchanged. Nothing is silently dropped:
  ``exact`` is True only for ``EXECUTED``.
* **Transactional.** Pool rows are written only after every pass succeeded and
  every candidate was finite.
* **Weights.** Shared and immutable: one checkpoint per engine, loaded from the
  authenticated R77 record and pinned by hash (``weights_sha256``); no torch is
  needed on the native path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from . import provenance as P
from .awr_state import AWRState, StateError

RECONCILIATION = ("MEAN", "EXCLUSIVE")
BACKENDS = ("native", "oracle")


@dataclass(frozen=True)
class Budgets:
    max_active_instances: int = 1024
    max_active_relations: int = 4096
    max_active_memberships: int = 1 << 16
    max_links_per_pass: int = 1 << 17
    max_passes: int = 8
    max_work_units: int = 1 << 22      # passes * (links + bound slots)
    max_expansion_incidences: int = 1 << 16
    max_scratch_bytes: int = 64 << 20


@dataclass
class StepResult:
    disposition: str
    exact: bool
    active_instances: int = 0
    active_relations: int = 0
    memberships: int = 0
    links: int = 0
    passes: int = 0
    work_units: int = 0
    detail: str = ""


@dataclass
class Selection:
    handles: np.ndarray
    generations: np.ndarray
    complete: bool
    disposition: str
    expansion_incidences: int = 0
    omitted_instances: int = 0
    relations_considered: int = 0
    notes: dict = field(default_factory=dict)


def load_weights(checkpoint: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Shared frozen weights from the authenticated R77 record (stdlib + numpy)."""
    rec = P.checkpoint_records()[checkpoint]
    W = np.ascontiguousarray(np.array(rec["weights"], dtype=np.float64))
    b = np.ascontiguousarray(np.array(rec["bias"], dtype=np.float64))
    sha = P.checkpoint_hash(W.tolist(), b.tolist())
    if sha != P.CHECKPOINT_SHA256[checkpoint]:
        raise P.ProvenanceError("ENGINE_WEIGHT_PIN:" + checkpoint)
    W.setflags(write=False)
    b.setflags(write=False)
    return W, b, sha


class Engine:
    def __init__(self, state: AWRState, checkpoint: str, controller: str = "R114",
                 budgets: Budgets = Budgets(), reconciliation: str = "MEAN", backend: str = "native"):
        if reconciliation not in RECONCILIATION:
            raise ValueError("RECONCILIATION")
        if backend not in BACKENDS:
            raise ValueError("BACKEND")
        if controller not in C.CONTROLLERS:
            raise ValueError("CONTROLLER")
        self.state = state
        self.registry = state.registry
        self.checkpoint = checkpoint
        self.controller = controller
        self.gains = C.controller_gains(controller, checkpoint)
        self.budgets = budgets
        self.reconciliation = reconciliation
        self.backend = backend
        self.W, self.b, self.weights_sha256 = load_weights(checkpoint)
        A = budgets.max_active_instances
        R = min(budgets.max_active_relations, A * C.SLOTS)
        self._rmax = R
        self._buf = {
            "handles": np.zeros(A, dtype=np.int32),
            "slot_rel": np.zeros(A * C.SLOTS, dtype=np.int32),
            "slot_cmp": np.zeros(A * C.SLOTS, dtype=np.int32),
            "rows": np.zeros(max(R, 1), dtype=np.int32),
            "buf_a": np.zeros(max(R, 1) * C.SLOT_COORDS, dtype=np.float64),
            "buf_b": np.zeros(max(R, 1) * C.SLOT_COORDS, dtype=np.float64),
            "cand": np.zeros(A * C.SLOTS * C.SLOT_COORDS, dtype=np.float64),
            "count": np.zeros(max(R, 1), dtype=np.int32),
            "off": np.zeros(max(R, 1) + 1, dtype=np.int32),
            "list": np.zeros(A * C.SLOTS, dtype=np.int32),
            "nr": np.zeros(1, dtype=np.int32),
            "links": np.zeros(1, dtype=np.int64),
        }
        self.scratch_bytes = sum(a.nbytes for a in self._buf.values())
        if self.scratch_bytes > budgets.max_scratch_bytes:
            raise StateError("SCRATCH_MEMORY_BUDGET", f"{self.scratch_bytes}>{budgets.max_scratch_bytes}")
        self._kernel = None
        if backend == "native":
            from . import native

            self._kernel = native.build()
            self._ptr = {k: v.ctypes.data for k, v in self._buf.items()}
            reg = self.registry
            self._reg_ptr = (reg.rel_offsets.ctypes.data, reg.members.ctypes.data, reg.frames.ctypes.data)
            self._w_ptr = (self.W.ctypes.data, self.b.ctypes.data)
            st = state
            self._st_ptr = (st.inst_live.ctypes.data, st.slot_rel.ctypes.data, st.slot_row.ctypes.data,
                            st.mark.ctypes.data, st.pool.ctypes.data)
        self.steps_executed = 0
        self.steps_refused = 0

    # -- selection ---------------------------------------------------------------
    def select_participants(self, participant_ids, allow_partial: bool = False) -> Selection:
        """Instances whose windows bind any relation containing the given participants.

        Bounded expansion: participant -> incidences (reverse CSR) -> relations ->
        materialized rows -> bindings. Over budget: refusal, or (only if explicitly
        allowed) a PARTIAL selection that reports exactly how much was omitted.
        """
        reg, st, bud = self.registry, self.state, self.budgets
        p = np.unique(reg.participant_index(participant_ids))
        lo, hi = reg.part_offsets[p], reg.part_offsets[p + 1]
        total = int((hi - lo).sum())
        empty = np.zeros(0, dtype=np.int32)
        if total > bud.max_expansion_incidences:
            return Selection(empty, empty.astype(np.uint32), False, "REFUSED_EXPANSION_BUDGET", total)
        if total:
            idx = np.repeat(lo - np.cumsum(np.concatenate(([0], (hi - lo)[:-1]))), hi - lo) + np.arange(total)
            rels = np.unique(reg.inc_relation[reg.part_incidences[idx]])
        else:
            rels = empty
        rows = st.rel_row[rels]
        rows = rows[rows >= 0]
        if rows.size:
            refs = st.row_ref[rows]
            binds = st.row_bind[rows]
            mask = np.arange(binds.shape[1])[None, :] < refs[:, None]
            handles = np.unique(binds[mask] // C.SLOTS).astype(np.int32)
        else:
            handles = empty
        omitted = max(0, handles.size - bud.max_active_instances)
        if omitted:
            if not allow_partial:
                return Selection(empty, empty.astype(np.uint32), False, "REFUSED_INSTANCE_BUDGET", total,
                                 int(handles.size), int(rels.size))
            # Deterministic truncation by stable instance ID, reported as PARTIAL.
            order = np.argsort(st.inst_id[handles], kind="stable")[:bud.max_active_instances]
            handles = np.sort(handles[order])
        return Selection(handles, st.inst_gen[handles].copy(), omitted == 0,
                         "COMPLETE" if omitted == 0 else "PARTIAL", total, int(omitted), int(rels.size))

    # -- step -----------------------------------------------------------------------
    def _refuse(self, code: str, res: StepResult, detail: str = "") -> StepResult:
        self.steps_refused += 1
        res.disposition = "REFUSED_" + code
        res.exact = False
        res.detail = detail
        return res

    def step(self, handles, generations=None, passes: int = 1) -> StepResult:
        st, bud = self.state, self.budgets
        res = StepResult("PENDING", False, passes=int(passes))
        if not 1 <= int(passes) <= bud.max_passes:
            return self._refuse("PASS_BUDGET", res, f"{passes}>{bud.max_passes}")
        h = np.asarray(handles, dtype=np.int64).reshape(-1)
        if generations is not None:
            g = np.asarray(generations, dtype=np.uint32).reshape(-1)
            if g.shape != h.shape:
                return self._refuse("ARGUMENT", res, "generation shape")
            if h.size and (h.min() < 0 or h.max() >= st.capacity.max_instances):
                return self._refuse("DEAD_HANDLE", res)
            if not np.array_equal(st.inst_gen[h], g):
                return self._refuse("STALE_HANDLE", res)
        h = np.unique(h).astype(np.int32)
        n = int(h.size)
        res.active_instances = n
        if n > bud.max_active_instances:
            return self._refuse("INSTANCE_BUDGET", res, f"{n}>{bud.max_active_instances}")
        if n == 0:
            res.disposition, res.exact = "EXECUTED", True
            return res
        if h.min() < 0 or h.max() >= st.capacity.max_instances or not np.all(st.inst_live[h] == 1):
            return self._refuse("DEAD_HANDLE", res)
        memberships = int(st.inst_memberships[h].sum())
        links = int(st.inst_links[h].sum())
        bound = int(st.inst_bound[h].sum())
        res.memberships, res.links = memberships, links
        res.work_units = int(passes) * (links + bound)
        if memberships > bud.max_active_memberships:
            return self._refuse("MEMBERSHIP_BUDGET", res, f"{memberships}>{bud.max_active_memberships}")
        if links > bud.max_links_per_pass:
            return self._refuse("LINK_BUDGET", res, f"{links}>{bud.max_links_per_pass}")
        if res.work_units > bud.max_work_units:
            return self._refuse("WORK_BUDGET", res, f"{res.work_units}>{bud.max_work_units}")
        if self.backend == "native":
            return self._step_native(h, n, bound, int(passes), res)
        return self._step_oracle(h, n, bound, int(passes), res)

    def _step_native(self, h, n, bound, passes, res) -> StepResult:
        k, buf, ptr = self._kernel.lib, self._buf, self._ptr
        st = self.state
        buf["handles"][:n] = h
        live, slot_rel_t, slot_row_t, mark, pool = self._st_ptr
        rc = k.awr_compact(n, ptr["handles"], live, st.capacity.max_instances, slot_rel_t, slot_row_t, mark,
                           ptr["slot_rel"], ptr["slot_cmp"], ptr["rows"], ptr["nr"], self._rmax)
        nr = int(buf["nr"][0])
        if rc != 0:
            k.awr_release_marks(nr, ptr["rows"], mark)
            from .native import STATUS

            code = STATUS[rc]
            return self._refuse("RELATION_BUDGET" if code == "ROW_CAPACITY" else code, res)
        res.active_relations = nr
        if nr > self.budgets.max_active_relations:
            k.awr_release_marks(nr, ptr["rows"], mark)
            return self._refuse("RELATION_BUDGET", res, f"{nr}>{self.budgets.max_active_relations}")
        if self.reconciliation == "EXCLUSIVE" and nr < bound:
            k.awr_release_marks(nr, ptr["rows"], mark)
            return self._refuse("OVERLAP_CONFLICT", res, f"{bound - nr} shared bindings")
        rel_off, members, frames = self._reg_ptr
        W, b = self._w_ptr
        alpha, beta = self.gains if self.gains is not None else (0.0, 0.0)
        rc = k.awr_execute(n, ptr["slot_rel"], ptr["slot_cmp"], nr, ptr["rows"], pool, ptr["buf_a"], ptr["buf_b"],
                           ptr["cand"], rel_off, members, frames, W, b, int(self.gains is not None), alpha, beta,
                           passes, 1 if self.reconciliation == "EXCLUSIVE" else 0, ptr["count"], ptr["off"],
                           ptr["list"], ptr["links"])
        k.awr_release_marks(nr, ptr["rows"], mark)
        if rc != 0:
            from .native import STATUS

            return self._refuse(STATUS[rc], res)
        processed = int(buf["links"][0])
        if processed != res.links * passes:
            raise RuntimeError(f"LINK_ACCOUNTING {processed} != {res.links * passes}")
        st.mutation_epoch += 1
        self.steps_executed += 1
        res.disposition, res.exact = "EXECUTED", True
        return res

    def _step_oracle(self, h, n, bound, passes, res) -> StepResult:
        from . import reference_adapter as RA

        st = self.state
        rows = sorted({int(r) for r in st.slot_row[h].reshape(-1) if r >= 0})
        res.active_relations = len(rows)
        if len(rows) > self.budgets.max_active_relations:
            return self._refuse("RELATION_BUDGET", res)
        if self.reconciliation == "EXCLUSIVE" and len(rows) < bound:
            return self._refuse("OVERLAP_CONFLICT", res)
        cur = {r: st.pool[r].copy() for r in rows}
        for _ in range(passes):
            cands = RA.oracle_candidates(self.registry, st.slot_rel[h], st.slot_row[h], cur, self.checkpoint,
                                         self.controller)
            if cands is None:
                return self._refuse("NONFINITE", res)
            cur = RA.reconcile_mean(cands)
        for r, v in cur.items():
            st.pool[r] = v
        st.mutation_epoch += 1
        self.steps_executed += 1
        res.disposition, res.exact = "EXECUTED", True
        return res
