"""R118-B engine over the sparse substrate, against the intentionally simple reference.

The native engine must equal ``simple_reference`` (pure Python) on every graph
family; budgets refuse before work and leave the AWR unchanged; isolation,
reconciliation, order independence and replay hold.
"""
from __future__ import annotations

import unittest

import numpy as np

from .. import engine as E
from ..awr_state import AWRState, StateCapacity
from ..fixtures import synthetic_registry
from ..registry import RegistryLimits, build_registry, rotation_blocks
from ._util import compare_with_simple, engine, random_windows, seed_states

TOL = 1e-12  # engine vs simple reference (observed: bitwise; the tolerance only absorbs signed zeros)


def family(name, seed):
    if name == "connected_sparse":
        return synthetic_registry(300, 1200, seed, arity=(2, 5))
    if name == "heterogeneous_zipf":
        return synthetic_registry(300, 1200, seed, arity=(1, 8), zipf=1.3)
    if name == "disconnected":
        return synthetic_registry(400, 1200, seed, arity=(2, 4), components=8)
    if name == "hub_adversarial":
        return synthetic_registry(120, 600, seed, arity=(3, 16), hub_fraction=0.9, limits=RegistryLimits(max_arity=16))
    raise KeyError(name)


def relation_components(reg):
    """Connected components of relations (linked when they share a participant), union-find."""
    parent = list(range(reg.num_relations))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in range(reg.num_participants):
        rels = reg.inc_relation[reg.incidences_of_participant(p)].tolist()
        for r in rels[1:]:
            a, b = find(rels[0]), find(r)
            if a != b:
                parent[b] = a
    return np.array([find(r) for r in range(reg.num_relations)])


def populate(reg, n, seed, cap=None):
    st = AWRState(reg, StateCapacity(max_instances=cap or n, max_materialized_relations=4 * (cap or n),
                                     max_bindings_per_relation=256))
    rel, mask = random_windows(reg, n, seed)
    h, g = st.create_instances(np.arange(n, dtype=np.uint64) + 1000, rel, mask)
    seed_states(st, seed + 1)
    return st, h, g


class EngineVsSimpleReference(unittest.TestCase):
    def test_graph_families_controllers_and_passes(self):
        worst = 0.0
        for fam in ("connected_sparse", "heterogeneous_zipf", "disconnected", "hub_adversarial"):
            reg = family(fam, 31)
            for controller in ("NATIVE", "R104", "R114"):
                st, h, _ = populate(reg, 96, seed=7)
                eng = engine(st, controller=controller)
                rng = np.random.default_rng(3)
                for passes in (1, 3):
                    active = rng.choice(h, size=48, replace=False)
                    res, diff, eq, n = compare_with_simple(eng, active, passes)
                    self.assertEqual(res.disposition, "EXECUTED", fam)
                    self.assertTrue(res.exact)
                    self.assertLessEqual(diff, TOL, f"{fam}/{controller}/{passes}")
                    worst = max(worst, diff)
        self.assertLessEqual(worst, TOL)

    def test_overlap_reconciliation_shared_relations(self):
        """Relations reached by several active windows receive the order-independent mean."""
        reg = synthetic_registry(60, 80, 5, arity=(3, 6))
        st = AWRState(reg, StateCapacity(max_instances=64, max_materialized_relations=80, max_bindings_per_relation=64))
        rng = np.random.default_rng(9)
        rel = reg.relation_ids[rng.integers(0, 12, size=(48, 4))]  # 48 windows over only 12 relations
        mask = np.ones((48, 4), dtype=bool)
        keep = [i for i in range(48) if len(set(rel[i].tolist())) == 4]
        h, _ = st.create_instances(np.arange(len(keep), dtype=np.uint64), rel[keep], mask[keep])
        self.assertGreater(int(st.row_ref.max()), 4)
        seed_states(st, 2)
        res, diff, _, _ = compare_with_simple(engine(st), h, passes=4)
        self.assertEqual(res.disposition, "EXECUTED")
        self.assertLessEqual(diff, TOL)
        # EXCLUSIVE refuses the same overlapping set without touching the state.
        before = st.pool.copy()
        res = engine(st, reconciliation="EXCLUSIVE").step(h)
        self.assertEqual(res.disposition, "REFUSED_OVERLAP_CONFLICT")
        np.testing.assert_array_equal(before, st.pool)


class EngineSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = synthetic_registry(400, 1600, 77, arity=(2, 6), zipf=0.8)

    def test_order_duplicates_and_replay_are_bitwise(self):
        st, h, _ = populate(self.reg, 128, seed=12)
        init = st.pool.copy()
        outs = []
        for active in (h[:96], h[:96][::-1], np.concatenate([h[:96], h[:40]]), np.random.default_rng(1).permutation(h[:96])):
            st.pool[:] = init
            res = engine(st).step(active, passes=3)
            self.assertEqual(res.disposition, "EXECUTED")
            self.assertEqual(res.active_instances, 96)
            outs.append(st.pool.copy())
        for o in outs[1:]:
            np.testing.assert_array_equal(outs[0], o)

    def test_inactive_and_unbound_untouched(self):
        st, h, _ = populate(self.reg, 128, seed=13)
        before = st.pool.copy()
        active = h[:30]
        engine(st).step(active, passes=2)
        touched = np.zeros(st.pool.shape[0], dtype=bool)
        touched[st.slot_row[active][st.slot_row[active] >= 0]] = True
        np.testing.assert_array_equal(before[~touched], st.pool[~touched])
        self.assertTrue(np.any(before[touched] != st.pool[touched]))
        self.assertTrue(np.all(st.mark == -1))  # compaction scratch fully released

    def test_disconnected_components_are_isolated(self):
        reg = synthetic_registry(400, 1200, 8, arity=(2, 4), components=4)
        comp = relation_components(reg)
        self.assertGreaterEqual(len(set(comp.tolist())), 4)
        st, h, _ = populate(reg, 160, seed=4)
        inst_comp = {}
        for i in h.tolist():
            cs = {int(comp[r]) for r in st.slot_rel[i] if r >= 0}
            self.assertEqual(len(cs), 1)  # extracted windows never straddle components
            inst_comp[i] = cs.pop()
        target = max(set(inst_comp.values()), key=lambda c: sum(v == c for v in inst_comp.values()))
        init = st.pool.copy()
        eng = engine(st)
        eng.step(h, passes=3)
        base = st.pool.copy()
        victims = [i for i in h.tolist() if inst_comp[i] == target]
        others = [i for i in h.tolist() if inst_comp[i] != target]
        rows0 = np.unique(st.slot_row[victims][st.slot_row[victims] >= 0])
        rows1 = np.unique(st.slot_row[others][st.slot_row[others] >= 0])
        self.assertEqual(np.intersect1d(rows0, rows1).size, 0)
        st.pool[:] = init
        st.pool[rows0] += 0.25  # perturb one component only
        eng.step(h, passes=3)
        np.testing.assert_array_equal(base[rows1], st.pool[rows1])     # bitwise: no leakage
        self.assertTrue(np.any(base[rows0] != st.pool[rows0]))

    def test_budgets_refuse_before_work_and_leave_state_unchanged(self):
        st, h, _ = populate(self.reg, 200, seed=14)
        links = int(st.inst_links[h[:50]].sum())
        mem = int(st.inst_memberships[h[:50]].sum())
        cases = {
            "REFUSED_INSTANCE_BUDGET": E.Budgets(max_active_instances=49),
            "REFUSED_RELATION_BUDGET": E.Budgets(max_active_relations=10),
            "REFUSED_MEMBERSHIP_BUDGET": E.Budgets(max_active_memberships=mem - 1),
            "REFUSED_LINK_BUDGET": E.Budgets(max_links_per_pass=links - 1),
            "REFUSED_WORK_BUDGET": E.Budgets(max_work_units=10),
            "REFUSED_PASS_BUDGET": E.Budgets(max_passes=2),
        }
        for code, bud in cases.items():
            before = st.pool.copy()
            epoch = st.mutation_epoch
            res = E.Engine(st, "54031/STATE8", controller="R114", budgets=bud).step(h[:50], passes=3)
            self.assertEqual(res.disposition, code)
            self.assertFalse(res.exact)
            np.testing.assert_array_equal(before, st.pool)
            self.assertEqual(epoch, st.mutation_epoch)
            self.assertTrue(np.all(st.mark == -1))
        with self.assertRaises(Exception):
            E.Engine(st, "54031/STATE8", budgets=E.Budgets(max_scratch_bytes=1000))

    def test_dead_and_stale_handles_refused(self):
        st, h, g = populate(self.reg, 20, seed=15)
        eng = engine(st)
        st.release_instance(int(h[3]), int(g[3]))
        self.assertEqual(eng.step(h).disposition, "REFUSED_DEAD_HANDLE")
        nh, ng = st.create_instance(10**15, [int(self.reg.relation_ids[0]), None, None, None])
        self.assertEqual(nh, int(h[3]))
        self.assertEqual(eng.step(h, g).disposition, "REFUSED_STALE_HANDLE")
        self.assertEqual(eng.step(h[:3], g[:3]).disposition, "EXECUTED")

    def test_nonfinite_is_refused_transactionally(self):
        st, h, _ = populate(self.reg, 20, seed=16)
        row = int(st.slot_row[h[5]][st.slot_row[h[5]] >= 0][0])
        st.pool[row, 2] = np.nan  # corruption bypassing the API's finiteness check
        before = st.pool.copy()
        res = engine(st).step(h, passes=2)
        self.assertEqual(res.disposition, "REFUSED_NONFINITE")
        np.testing.assert_array_equal(np.isnan(before), np.isnan(st.pool))
        np.testing.assert_array_equal(np.nan_to_num(before), np.nan_to_num(st.pool))

    def test_participant_selection_bounded_and_partial_is_labelled(self):
        st, h, _ = populate(self.reg, 300, seed=17, cap=300)
        eng = engine(st, budgets=E.Budgets(max_active_instances=40, max_passes=20, max_expansion_incidences=10**6))
        hub = int(np.argmax(self.reg.participant_degree()))
        pid = [int(self.reg.participant_ids[hub])]
        sel = eng.select_participants(pid)
        exp_rel = np.unique(self.reg.inc_relation[self.reg.incidences_of_participant(hub)])
        expected = np.unique(np.concatenate([st.instances_binding(int(r)) for r in exp_rel] + [np.zeros(0, int)]))
        if expected.size > 40:
            self.assertEqual(sel.disposition, "REFUSED_INSTANCE_BUDGET")
            part = eng.select_participants(pid, allow_partial=True)
            self.assertEqual(part.disposition, "PARTIAL")
            self.assertFalse(part.complete)
            self.assertEqual(part.omitted_instances, expected.size - 40)
            self.assertTrue(set(part.handles.tolist()) <= set(expected.tolist()))
        else:
            self.assertEqual(sel.disposition, "COMPLETE")
            np.testing.assert_array_equal(sel.handles, expected)
        small = engine(st, budgets=E.Budgets(max_expansion_incidences=1))
        self.assertEqual(small.select_participants(pid).disposition, "REFUSED_EXPANSION_BUDGET")

    def test_repeated_passes_bounded_without_resource_growth(self):
        """Per-slot invariant bound M* = ((1+b)sqrt32 + |a-b|sqrt128)/(1-a), any convex reconciliation."""
        for checkpoint, gains in (("54007/STATE8_SIGNED8", (0.70, 0.45)), ("54031/STATE8", (0.50, 0.50))):
            st, h, _ = populate(self.reg, 128, seed=18)
            self.assertGreater(int(st.row_ref.max()), 1)  # overlapping windows present
            st.pool *= 30.0
            eng = engine(st, checkpoint=checkpoint, budgets=E.Budgets(max_passes=8))
            free_rows, free_handles = len(st._free_rows), len(st._free_handles)
            a, b = gains
            bound = ((1 + b) * np.sqrt(32.0) + abs(a - b) * np.sqrt(128.0)) / (1 - a)
            m_prev = float(np.linalg.norm(st.pool, axis=1).max())
            for _ in range(60):
                self.assertEqual(eng.step(h, passes=8).disposition, "EXECUTED")
                m = float(np.linalg.norm(st.pool, axis=1).max())
                # one-pass recursion M' <= (bound * (1 - a)) + a M, applied 8 times
                limit = m_prev
                for _ in range(8):
                    limit = bound * (1 - a) + a * limit
                self.assertLessEqual(m, limit + 1e-9)
                m_prev = m
            self.assertTrue(np.all(np.isfinite(st.pool)))
            self.assertLessEqual(m_prev, bound + 1e-9)
            self.assertEqual((free_rows, free_handles), (len(st._free_rows), len(st._free_handles)))
            self.assertTrue(np.all(st.mark == -1))

    def test_maximum_in_degree_window(self):
        """Four arity-64 relations with identical members: in-degree 3 x 64 = 192 per slot,
        the native kernel's hard limit (AWR_MAX_IN); still equal to the simple reference."""
        pids = list(range(1, 65))
        frames = rotation_blocks(np.linspace(-3, 3, 256), np.linspace(2, -1, 256))
        reg = build_registry(pids, [1, 2, 3, 4], [0, 64, 128, 192, 256], pids * 4, frames,
                             RegistryLimits(max_arity=64))
        st = AWRState(reg, StateCapacity(max_instances=2, max_materialized_relations=4))
        h, _ = st.create_instance(7, [1, 2, 3, 4])
        self.assertEqual(int(st.inst_links[h]), 4 * 192)
        seed_states(st, 3)
        res, diff, eq, n = compare_with_simple(engine(st), [h], passes=4)
        self.assertEqual((res.disposition, res.links), ("EXECUTED", 768))
        self.assertLessEqual(diff, TOL)


class FrameEdgeCases(unittest.TestCase):
    def test_multilinks_identical_members_and_arity_one(self):
        # Two relations with identical member sets (4 parallel links per ordered pair), arity-1 relations.
        pids = [1, 2, 3, 4, 5]
        rids = [10, 11, 12, 13]
        offsets = [0, 4, 8, 9, 10]
        mids = [1, 2, 3, 4, 1, 2, 3, 4, 5, 1]
        frames = rotation_blocks(np.linspace(-3, 3, 10), np.linspace(1, -2, 10))
        reg = build_registry(pids, rids, offsets, mids, frames)
        st = AWRState(reg, StateCapacity(max_instances=4, max_materialized_relations=8))
        h1, _ = st.create_instance(1, [10, 11, 12, 13])
        self.assertEqual(int(st.inst_links[h1]), 8 + 2 + 2)  # 4 shared x 2 directions, + {10,13},{11,13} via 1
        seed_states(st, 5)
        res, diff, _, _ = compare_with_simple(engine(st), [h1], passes=5)
        self.assertEqual(res.disposition, "EXECUTED")
        self.assertLessEqual(diff, TOL)


if __name__ == "__main__":
    unittest.main()
