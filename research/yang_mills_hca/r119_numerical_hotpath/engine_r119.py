"""R119 engine: numerical modes of the R118 hot path.

=================  ==============================================  ======================
mode               kernel                                          numerical class
=================  ==============================================  ======================
EXACT_R118         frozen R118 kernel (csrc/awr_kernel.c), FTZ off  exact reference
EXACT_GUARDED      R119 kernel, guards G_SOFTMAX|G_GATES|G_CLAMP|   exact by proof; R119
                   G_RADIAL, FTZ off                                verifies bit identity
FTZ_DAZ            frozen R118 kernel with flush-to-zero /          opt-in; qualified only
                   denormals-are-zero scoped to the call            against a tolerance
=================  ==============================================  ======================

Everything except the kernel call is the frozen R118 ``Engine`` (budgets, refusals,
compaction, transactional commit, reconciliation semantics, shared immutable weights).
"""
from __future__ import annotations

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import engine as E118

from . import guards
from . import native_r119 as N

MODES = ("EXACT_R118", "EXACT_GUARDED", "FTZ_DAZ")
G_SOFTMAX, G_GATES, G_CLAMP, G_RADIAL = 1, 2, 4, 8
ALL_GUARDS = G_SOFTMAX | G_GATES | G_CLAMP | G_RADIAL
GUARD_NAMES = ("softmax", "gates", "clamp", "radial")


class R119Engine(E118.Engine):
    def __init__(self, state, checkpoint, controller="R114", budgets=E118.Budgets(), reconciliation="MEAN",
                 mode="EXACT_GUARDED", guard_mask=ALL_GUARDS):
        if mode not in MODES:
            raise ValueError("MODE:" + str(mode))
        super().__init__(state, checkpoint, controller=controller, budgets=budgets, reconciliation=reconciliation,
                         backend="native", flush_subnormals=(mode == "FTZ_DAZ"))
        self.mode = mode
        self.guard_counts = np.zeros(8, dtype=np.int64)  # hits[4], evaluations[4]
        self.libm_selfcheck = None
        self.theta = 0.0
        self.guard_mask = 0
        if mode == "EXACT_GUARDED":
            self._k119, self.kernel_info = N.kernel()
            self.libm_selfcheck = bool(self._k119.awr119_libm_selfcheck())
            mask = int(guard_mask) & ALL_GUARDS
            if not self.libm_selfcheck:
                mask &= ~G_SOFTMAX  # precondition of the softmax guard not established: never take it
            self.guard_mask = mask
            self.theta = guards.gate_theta(self.W.tolist(), self.b.tolist())
        else:
            from research.yang_mills_hca.r118_scalable_awr import native

            self.kernel_info = native.build().info

    def guard_report(self) -> dict:
        g = self.guard_counts
        return {name: {"hits": int(g[i]), "evaluations": int(g[4 + i])} for i, name in enumerate(GUARD_NAMES)}

    def _step_native(self, h, n, bound, passes, res):
        if self.mode != "EXACT_GUARDED":
            return super()._step_native(h, n, bound, passes, res)
        k, buf, ptr = self._k119, self._buf, self._ptr
        st = self.state
        buf["handles"][:n] = h
        live, slot_rel_t, slot_row_t, mark, pool = self._st_ptr
        rc = k.awr119_compact(n, ptr["handles"], live, st.capacity.max_instances, slot_rel_t, slot_row_t, mark,
                              ptr["slot_rel"], ptr["slot_cmp"], ptr["rows"], ptr["nr"], self._rmax)
        nr = int(buf["nr"][0])
        if rc != 0:
            k.awr119_release_marks(nr, ptr["rows"], mark)
            code = N.STATUS[rc]
            return self._refuse("RELATION_BUDGET" if code == "ROW_CAPACITY" else code, res)
        res.active_relations = nr
        if nr > self.budgets.max_active_relations:
            k.awr119_release_marks(nr, ptr["rows"], mark)
            return self._refuse("RELATION_BUDGET", res, f"{nr}>{self.budgets.max_active_relations}")
        if self.reconciliation == "EXCLUSIVE" and nr < bound:
            k.awr119_release_marks(nr, ptr["rows"], mark)
            return self._refuse("OVERLAP_CONFLICT", res, f"{bound - nr} shared bindings")
        rel_off, members, frames = self._reg_ptr
        W, b = self._w_ptr
        alpha, beta = self.gains if self.gains is not None else (0.0, 0.0)
        rc = k.awr119_execute(n, ptr["slot_rel"], ptr["slot_cmp"], nr, ptr["rows"], pool, ptr["buf_a"],
                              ptr["buf_b"], ptr["cand"], rel_off, members, frames, W, b, int(self.gains is not None),
                              alpha, beta, passes, 1 if self.reconciliation == "EXCLUSIVE" else 0, ptr["count"],
                              ptr["off"], ptr["list"], ptr["links"], 0, self.guard_mask, self.theta,
                              self.guard_counts.ctypes.data)
        k.awr119_release_marks(nr, ptr["rows"], mark)
        if rc != 0:
            return self._refuse(N.STATUS[rc], res)
        processed = int(buf["links"][0])
        if processed != res.links * passes:
            raise RuntimeError(f"LINK_ACCOUNTING {processed} != {res.links * passes}")
        st.mutation_epoch += 1
        self.steps_executed += 1
        res.disposition, res.exact = "EXECUTED", True
        return res
