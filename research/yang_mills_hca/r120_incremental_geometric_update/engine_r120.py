"""R120 incremental engine: bounded batched geometric updates fused with selective propagation.

    state  = IncrementalState(registry, capacity)          # R118 AWRState + three side tables
    engine = IncrementalEngine(state, checkpoint)          # shared immutable HCA88, frozen CGC
    report = engine.apply(rel_ids, seq, part, obs, weight, aux)

One ``apply`` is one native call: validation, CGC coefficients, in-place update of the
*existing* relation rows, discovery of the affected windows through the existing binding
lists, the frozen R119 EXACT_GUARDED passes, reconciliation and a transactional commit.
Nothing is created per event: no relation, no instance, no operator, no CGC parameter set.
All buffers are allocated once, here.

Separation of concerns. Relation *identity* is the registry's 64-bit ID (never changed
here). *Evidence authority* is the caller's: an event is assumed admitted upstream; this
engine neither admits nor infers evidence. The *learned geometric state* is the volatile
pool row, which is the only thing an update mutates (plus the row's event sequence).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import contract as C118
from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr.awr_state import AWRState, StateCapacity, StateError
from research.yang_mills_hca.r119_numerical_hotpath import engine_r119 as E119
from research.yang_mills_hca.r119_numerical_hotpath import guards

from . import cgc_bridge as CGC
from . import native_r120 as N


@dataclass(frozen=True)
class Limits:
    max_events: int = 1024               # events per batch
    max_active_instances: int = 1024     # affected windows per call
    max_active_relations: int = 4096     # compact rows per call
    max_links: int = 1 << 17             # links x passes per call
    max_passes: int = 8
    max_radius: int = 2
    window_cache_bytes: int = 128 << 20  # volatile; 0 disables the cache
    threads: int = 1                     # > 1: exact parallel Jacobi pass (opt-in)
    parallel_min_instances: int = 96     # below this the pass stays serial


class IncrementalState(AWRState):
    """R118 ``AWRState`` (unmodified) plus R120 side tables, all preallocated.

    row_gen      materialization generation of a pool row; bumped when the row is freed, so a
                 reference (row, generation) captured earlier is refused after re-materialization
    row_seq      last applied event sequence of the relation in that row (0 = none)
    row_pending  1 if an update of that row was committed with its propagation deferred
    """

    def __init__(self, registry, capacity: StateCapacity = StateCapacity()):
        super().__init__(registry, capacity)
        n_r = capacity.max_materialized_relations
        self.row_gen = np.ones(n_r, dtype=np.uint32)
        self.row_seq = np.zeros(n_r, dtype=np.uint64)
        self.row_pending = np.zeros(n_r, dtype=np.uint8)
        self.pending_list = np.zeros(n_r, dtype=np.int32)
        self.inst_mark = np.zeros(capacity.max_instances, dtype=np.uint8)
        self._engines = []

    def create_instances(self, instance_ids, relation_ids, bound_mask):
        handles, gens = super().create_instances(instance_ids, relation_ids, bound_mask)
        for eng in self._engines:
            eng._uncache(handles)
        return handles, gens

    def release_instance(self, handle: int, generation: int) -> None:
        rows = self.slot_row[int(handle)].copy()
        rows = rows[rows >= 0]
        super().release_instance(handle, generation)
        freed = rows[self.row_rel[rows] < 0]
        if freed.size:
            self.row_gen[freed] += 1
            self.row_seq[freed] = 0
        for eng in self._engines:
            eng._uncache(np.array([int(handle)], dtype=np.int32))
            eng._drop_pending(freed)

    def relation_ref(self, relation_id: int):
        """(row, generation) of a materialized relation, for generation-checked events."""
        r = int(self.registry.relation_index([relation_id])[0])
        row = int(self.rel_row[r])
        if row < 0:
            raise StateError("RELATION_NOT_MATERIALIZED", str(relation_id))
        return row, int(self.row_gen[row])


class Report:
    """Snapshot of one native call's report array (see native_r120.REPORT_FIELDS)."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __getattr__(self, name):
        try:
            return int(self.v[N.REPORT_FIELDS.index(name)])
        except ValueError:
            raise AttributeError(name) from None

    @property
    def disposition(self) -> str:
        return N.STATUS[int(self.v[0])]

    @property
    def committed(self) -> bool:
        return int(self.v[0]) in (0, 1)

    @property
    def deferred(self) -> bool:
        return int(self.v[0]) == 1

    def as_dict(self) -> dict:
        d = {k: int(x) for k, x in zip(N.REPORT_FIELDS, self.v)}
        d["disposition"] = self.disposition
        return d


class IncrementalEngine:
    def __init__(self, state: IncrementalState, checkpoint: str, controller: str = "NATIVE",
                 limits: Limits = Limits(), cgc_weights=None, eta: float = 1.0, gamma: float = 0.25,
                 rho_b: float = CGC.RHO_B, kappa: float = CGC.KAPPA, reconciliation: str = "MEAN",
                 guard_mask: int = E119.ALL_GUARDS):
        if not isinstance(state, IncrementalState):
            raise TypeError("IncrementalState required")
        if reconciliation not in E118.RECONCILIATION or controller not in C118.CONTROLLERS:
            raise ValueError("CONFIGURATION")
        if not (0.0 <= eta <= 1.0 and 0.0 <= gamma <= 1.0 and 0.0 <= rho_b <= 1.0 and 0.0 < kappa <= 0.25):
            raise ValueError("UPDATE_PARAMETER_RANGE")
        self.state, self.registry, self.limits = state, state.registry, limits
        self.checkpoint, self.controller, self.reconciliation = checkpoint, controller, reconciliation
        self.W, self.b, self.weights_sha256 = E118.load_weights(checkpoint)      # authenticated, read-only
        self.cgc_w = CGC.check_weights(CGC.synthetic_weights() if cgc_weights is None else cgc_weights)
        self.cgc_sha256 = CGC.weights_sha256(self.cgc_w)
        self.gains = C118.controller_gains(controller, checkpoint)
        self.eta, self.gamma, self.rho_b, self.kappa = float(eta), float(gamma), float(rho_b), float(kappa)
        self._lib, self.kernel_info = N.kernel()
        mask = int(guard_mask) & E119.ALL_GUARDS
        if not self._lib.awr119_libm_selfcheck():
            mask &= ~E119.G_SOFTMAX
        self.guard_mask = mask
        self.theta = guards.gate_theta(self.W.tolist(), self.b.tolist())
        self.guard_counts = np.zeros(8, dtype=np.int64)
        st, reg, lim = state, state.registry, limits
        A, R, Ev = lim.max_active_instances, lim.max_active_relations, lim.max_events
        self._buf = {
            "handles": np.zeros(A, np.int32), "s_slot_rel": np.zeros(4, np.int32),
            "s_slot_cmp": np.zeros(A * 4, np.int32), "rows": np.zeros(R, np.int32), "count": np.zeros(R, np.int32),
            "off": np.zeros(R + 1, np.int32), "list": np.zeros(A * 4, np.int32), "last_event": np.zeros(R, np.int32),
            "buf_a": np.zeros(R * 8), "buf_b": np.zeros(R * 8), "cand": np.zeros(A * 32),
            "seq_new": np.zeros(R, np.uint64),
        }
        self.ev = {
            "rel_id": np.zeros(Ev, np.uint64), "seq": np.zeros(Ev, np.uint64), "exp_gen": np.zeros(Ev, np.uint32),
            "part": np.full(Ev, -1, np.int32), "obs": np.zeros((Ev, 4)), "weight": np.zeros(Ev), "aux": np.zeros(Ev),
        }
        self._ev_ptr = tuple(self.ev[k].ctypes.data for k in ("rel_id", "seq", "exp_gen", "part", "obs", "weight", "aux"))
        self.wc_off = np.full(st.capacity.max_instances, -1, dtype=np.int64)
        self.wc_data = np.zeros(max(int(lim.window_cache_bytes), 8), dtype=np.uint8)
        self.wc_used = 0
        self._rep = np.zeros(len(N.REPORT_FIELDS), dtype=np.int64)
        self._step_handles = np.zeros(A, np.int32)
        self._step_ptr = self._step_handles.ctypes.data
        ctx = N.Ctx()
        for name, arr in (("relation_ids", reg.relation_ids), ("rel_off", reg.rel_offsets), ("members", reg.members),
                          ("frames", reg.frames), ("pool", st.pool), ("rel_row", st.rel_row),
                          ("inst_live", st.inst_live), ("slot_rel", st.slot_rel), ("slot_row", st.slot_row),
                          ("row_head", st.row_head), ("bind_next", st.bind_next), ("mark", st.mark),
                          ("inst_links", st.inst_links), ("row_seq", st.row_seq), ("row_gen", st.row_gen),
                          ("row_pending", st.row_pending), ("pending_list", st.pending_list),
                          ("inst_mark", st.inst_mark), ("wc_off", self.wc_off), ("wc_data", self.wc_data),
                          ("W", self.W), ("bias", self.b), ("cgc_w", self.cgc_w), ("gc", self.guard_counts)):
            setattr(ctx, name, arr.ctypes.data)
        for name, arr in self._buf.items():
            setattr(ctx, name, arr.ctypes.data)
        if lim.window_cache_bytes <= 0:
            ctx.wc_off = None
        ctx.n_rel, ctx.max_links, ctx.pending_count = reg.num_relations, lim.max_links, 0
        ctx.alpha, ctx.beta = self.gains if self.gains is not None else (0.0, 0.0)
        ctx.theta, ctx.eta, ctx.gamma_v, ctx.rho_b, ctx.kappa = self.theta, self.eta, self.gamma, self.rho_b, self.kappa
        ctx.inst_capacity, ctx.row_capacity = st.capacity.max_instances, st.capacity.max_materialized_relations
        ctx.max_events, ctx.max_instances, ctx.max_rows = Ev, A, R
        ctx.correct, ctx.mode = int(self.gains is not None), 1 if reconciliation == "EXCLUSIVE" else 0
        ctx.pending_capacity, ctx.gmask = st.capacity.max_materialized_relations, mask
        ctx.threads = max(1, int(lim.threads)) if self._lib.awr120_openmp() > 0 else 1
        ctx.parallel_min = int(lim.parallel_min_instances)
        self.threads = int(ctx.threads)
        self._ctx = ctx
        self._ctx_ref = ctypes.byref(ctx)
        self._rep_ptr = self._rep.ctypes.data
        self.scratch_bytes = (sum(a.nbytes for a in self._buf.values()) + sum(a.nbytes for a in self.ev.values())
                              + self._rep.nbytes + self._step_handles.nbytes)
        self.calls = 0
        if st._engines:
            raise StateError("ONE_INCREMENTAL_ENGINE_PER_STATE")   # the pending list is owned by one engine
        st._engines.append(self)

    # -- window cache (volatile, bounded, optional) -------------------------------------------
    def build_cache(self, handles=None) -> dict:
        """Cache the window blocks of live instances (all by default). Bounded by ``window_cache_bytes``;
        instances that do not fit stay on the registry path, which gives identical results."""
        if self.limits.window_cache_bytes <= 0:
            return {"cached": 0, "bytes": 0, "uncached": int(self.state.inst_live.sum())}
        if handles is None:
            self.wc_off[:] = -1
            self.wc_used = 0
            handles = np.nonzero(self.state.inst_live)[0]
        h = np.ascontiguousarray(handles, dtype=np.int32)
        built = ctypes.c_int32(0)
        self.wc_used = int(self._lib.awr120_cache_build(self._ctx_ref, h.size, h.ctypes.data, self.wc_off.ctypes.data,
                                                        self.wc_data.ctypes.data, self.wc_data.size, self.wc_used,
                                                        ctypes.byref(built)))
        return {"cached": int((self.wc_off >= 0).sum()), "bytes": self.wc_used,
                "uncached": int(self.state.inst_live.sum() - (self.wc_off >= 0).sum())}

    def _uncache(self, handles) -> None:
        self.wc_off[np.asarray(handles, dtype=np.int64)] = -1

    def _drop_pending(self, rows) -> None:
        st = self.state
        n = int(self._ctx.pending_count)
        if not n or not len(rows):
            return
        keep = st.pending_list[:n][~np.isin(st.pending_list[:n], rows)]
        st.row_pending[rows] = 0
        st.pending_list[:keep.size] = keep
        self._ctx.pending_count = int(keep.size)

    @property
    def pending_rows(self) -> int:
        return int(self._ctx.pending_count)

    # -- hot path -------------------------------------------------------------------------------
    def apply(self, rel_ids, seq, part, obs, weight, aux, exp_gen=None, propagate: bool = True,
              defer_on_budget: bool = False, radius: int = 1, passes: int = 1, profile: bool = False) -> Report:
        """Copy an event batch into the preallocated buffers and run the fused native call."""
        n = len(rel_ids)
        if n > self.limits.max_events:
            self._rep[:] = 0
            self._rep[0], self._rep[1] = 17, -1
            return Report(self._rep.copy())
        ev = self.ev
        ev["rel_id"][:n] = rel_ids
        ev["seq"][:n] = seq
        ev["part"][:n] = part
        ev["obs"][:n] = obs
        ev["weight"][:n] = weight
        ev["aux"][:n] = aux
        if exp_gen is None:
            ev["exp_gen"][:n] = 0
        else:
            ev["exp_gen"][:n] = exp_gen
        return self.commit(n, propagate, defer_on_budget, radius, passes, profile)

    def commit(self, n: int, propagate: bool = True, defer_on_budget: bool = False, radius: int = 1,
               passes: int = 1, profile: bool = False) -> Report:
        """Run the first ``n`` events already written into ``self.ev`` (zero-copy entry)."""
        if not (1 <= radius <= self.limits.max_radius and 1 <= passes <= self.limits.max_passes):
            self._rep[:] = 0
            self._rep[0], self._rep[1] = 10, -1
            return Report(self._rep.copy())
        flags = (N.FLAG_PROPAGATE if propagate else 0) | (N.FLAG_DEFER_ON_BUDGET if defer_on_budget else 0) | (
            N.FLAG_PROFILE if profile else 0)
        p = self._ev_ptr
        self._lib.awr120_update_propagate(self._ctx_ref, n, p[0], p[1], p[2], p[3], p[4], p[5], p[6], flags, radius,
                                          passes, self._rep_ptr)
        self.calls += 1
        if self._rep[0] in (0, 1):
            self.state.mutation_epoch += 1
        return Report(self._rep.copy())

    def propagate_pending(self, limit: int = 1 << 30, radius: int = 1, passes: int = 1) -> Report:
        self._lib.awr120_propagate_pending(self._ctx_ref, min(int(limit), 1 << 30), radius, passes, self._rep_ptr)
        if self._rep[0] == 0:
            self.state.mutation_epoch += 1
        return Report(self._rep.copy())

    def step(self, handles, passes: int = 1) -> Report:
        """Plain synchronous step over instance handles (R119 step semantics, through the window cache)."""
        h = np.asarray(handles, dtype=np.int32).reshape(-1)
        if h.size > self._step_handles.size:
            self._rep[:] = 0
            self._rep[0], self._rep[1] = 18, -1
            return Report(self._rep.copy())
        self._step_handles[:h.size] = h
        self._lib.awr120_step(self._ctx_ref, h.size, self._step_ptr, passes, self._rep_ptr)
        if self._rep[0] == 0:
            self.state.mutation_epoch += 1
        return Report(self._rep.copy())

    # -- cold helpers ----------------------------------------------------------------------------
    def replace_cgc_weights(self, weights) -> str:
        """Cold path: install another frozen CGC parameter set for subsequent calls (for example a
        trained one, or the next candidate of an offline search). Never part of an update: no event
        creates, changes or selects CGC parameters."""
        self.cgc_w = CGC.check_weights(weights)
        self.cgc_sha256 = CGC.weights_sha256(self.cgc_w)
        self._ctx.cgc_w = self.cgc_w.ctypes.data
        return self.cgc_sha256

    def participant_index(self, participant_ids):
        return self.registry.participant_index(participant_ids)

    def update_probe(self, state8, frame8, obs4, weight: float, aux: float):
        """The native update law on one detached state (for reference-parity tests)."""
        s = np.array(state8, dtype=np.float64)
        rc = ctypes.c_int32(0)
        fr = None if frame8 is None else np.ascontiguousarray(frame8, dtype=np.float64)
        ob = np.ascontiguousarray(obs4, dtype=np.float64)
        self._lib.awr120_update_probe(self._ctx_ref, s.ctypes.data, None if fr is None else fr.ctypes.data,
                                      ob.ctypes.data, float(weight), float(aux), ctypes.byref(rc))
        return s, int(rc.value)
