"""R118-B volatile state: addressing, capacities, materialization, transactional refusals."""
from __future__ import annotations

import unittest

import numpy as np

from ..awr_state import AWRState, StateCapacity, StateError
from ..fixtures import bulk_synthetic_registry, synthetic_registry
from ._util import random_windows


def snapshot(st):
    names = ("inst_id", "inst_live", "inst_gen", "slot_rel", "slot_row", "inst_links", "inst_memberships",
             "inst_bound", "pool", "row_rel", "row_ref", "row_head", "bind_next", "bind_prev", "rel_row", "mark")
    return {n: getattr(st, n).copy() for n in names} | {"handles": dict(st._handle_of),
                                                         "free_h": list(st._free_handles),
                                                         "free_r": list(st._free_rows)}


def assert_same(tc, a, b):
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            tc.assertEqual(a[k], b[k], k)


class StateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = synthetic_registry(500, 4000, seed=21, arity=(2, 6), zipf=0.9)

    def ids(self, idx):
        return [int(self.reg.relation_ids[i]) for i in idx]

    def test_link_count_identity_matches_explicit_enumeration(self):
        st = AWRState(self.reg, StateCapacity(max_instances=400, max_materialized_relations=1600))
        rel, mask = random_windows(self.reg, 300, seed=4)
        h, _ = st.create_instances(np.arange(300, dtype=np.uint64) + 9, rel, mask)
        for i in h.tolist():
            self.assertEqual(int(st.inst_links[i]), len(self.reg.window_links(st.slot_rel[i])))
            self.assertEqual(int(st.inst_memberships[i]),
                             int(self.reg.arity(st.slot_rel[i][st.slot_rel[i] >= 0]).sum()))

    def test_refusals_are_transactional(self):
        st = AWRState(self.reg, StateCapacity(max_instances=3, max_materialized_relations=6,
                                              max_bindings_per_relation=2))
        st.create_instance(1, self.ids([0, 1]) + [None, None])
        cases = [
            ("DUPLICATE_INSTANCE_ID", 1, self.ids([2]) + [None] * 3),
            ("EMPTY_WINDOW", 2, [None] * 4),
            ("DUPLICATE_RELATION_IN_WINDOW", 2, self.ids([5, 5]) + [None, None]),
            ("UNKNOWN_RELATION", 2, [12345, None, None, None]),
        ]
        for code, iid, rels in cases:
            before = snapshot(st)
            with self.assertRaises(StateError) as cm:
                st.create_instance(iid, rels)
            self.assertEqual(cm.exception.code, code)
            assert_same(self, before, snapshot(st))
        st.create_instance(2, self.ids([0, 7]) + [None, None])          # relation 0 now bound twice (limit)
        before = snapshot(st)
        with self.assertRaises(StateError) as cm:
            st.create_instance(3, self.ids([0]) + [None] * 3)
        self.assertEqual(cm.exception.code, "BINDING_CAPACITY")
        assert_same(self, before, snapshot(st))
        with self.assertRaises(StateError) as cm:                        # 3 rows used, 4 new needed
            st.create_instance(3, self.ids([10, 11, 12, 13]))
        self.assertEqual(cm.exception.code, "STATE_ROW_CAPACITY")
        assert_same(self, before, snapshot(st))
        st.create_instance(3, self.ids([10]) + [None] * 3)
        with self.assertRaises(StateError) as cm:
            st.create_instance(4, self.ids([20]) + [None] * 3)
        self.assertEqual(cm.exception.code, "INSTANCE_CAPACITY")

    def test_materialization_refcount_and_release(self):
        st = AWRState(self.reg, StateCapacity(max_instances=8, max_materialized_relations=16))
        r = self.ids([3, 4, 5])
        h1, g1 = st.create_instance(100, [r[0], r[1], None, None])
        h2, g2 = st.create_instance(200, [None, r[1], r[2], None])
        row = st.rel_row[self.reg.relation_index([r[1]])[0]]
        self.assertEqual(int(st.row_ref[row]), 2)
        st.set_relation_state(r[1], np.arange(8.0))
        np.testing.assert_array_equal(st.instance_state32(h1, g1)[[4, 5, 6, 7, 20, 21, 22, 23]], np.arange(8.0))
        np.testing.assert_array_equal(st.instance_state32(h2, g2)[[4, 5, 6, 7, 20, 21, 22, 23]], np.arange(8.0))
        self.assertEqual(sorted(st.instances_binding(int(self.reg.relation_index([r[1]])[0])).tolist()),
                         sorted([h1, h2]))
        st.release_instance(h1, g1)
        self.assertEqual(int(st.row_ref[row]), 1)
        np.testing.assert_array_equal(st.relation_state(r[1]), np.arange(8.0))  # still bound by 200
        self.assertIsNone(st.relation_state(r[0]))                             # discarded (volatile)
        with self.assertRaises(StateError) as cm:
            st.instance_state32(h1, g1)
        self.assertEqual(cm.exception.code, "DEAD_HANDLE")
        h3, g3 = st.create_instance(300, [r[0], None, None, None])            # handle reuse
        self.assertEqual(h3, h1)
        self.assertNotEqual(g3, g1)
        with self.assertRaises(StateError) as cm:
            st.release_instance(h1, g1)
        self.assertEqual(cm.exception.code, "STALE_HANDLE")
        st.release_instance(h2, g2)
        self.assertIsNone(st.relation_state(r[1]))
        self.assertEqual(int(st.row_ref.sum()), 1)

    def test_binding_index_matches_full_scan_under_churn(self):
        st = AWRState(self.reg, StateCapacity(max_instances=120, max_materialized_relations=480,
                                              max_bindings_per_relation=64))
        rng = np.random.default_rng(99)
        live = {}
        next_id = 1
        for step in range(400):
            if live and (rng.random() < 0.45 or len(live) == 120):
                iid = int(rng.choice(sorted(live)))
                h, g = live.pop(iid)
                st.release_instance(h, g)
            else:
                rels = rng.choice(60, size=int(rng.integers(1, 5)), replace=False)  # heavy overlap on 60 relations
                slots = [None] * 4
                for d, r in zip(rng.permutation(4), rels):
                    slots[d] = int(self.reg.relation_ids[r])
                try:
                    live[next_id] = st.create_instance(next_id, slots)
                except StateError as exc:
                    self.assertEqual(exc.code, "BINDING_CAPACITY")
                next_id += 1
            if step % 20 == 0 or step == 399:
                expect = {}
                for h in np.nonzero(st.inst_live)[0].tolist():
                    for d in range(4):
                        if st.slot_row[h, d] >= 0:
                            expect.setdefault(int(st.slot_row[h, d]), set()).add(h * 4 + d)
                for row in range(480):
                    codes = set(st.binding_codes(np.array([row])).tolist())
                    self.assertEqual(codes, expect.get(row, set()), row)
                    self.assertEqual(int(st.row_ref[row]), len(codes))
                    if not codes:
                        self.assertEqual(int(st.row_rel[row]), -1)

    def test_capacities(self):
        """Distinct logical instance capacities 1 .. 50,000: fill, refuse N+1, release, refill."""
        reg = bulk_synthetic_registry(60_000, 220_000, seed=8, arity=(2, 5), zipf=0.6)
        for cap in (1, 10, 100, 1_000, 10_000, 50_000):
            st = AWRState(reg, StateCapacity(max_instances=cap, max_materialized_relations=4 * cap))
            rel, mask = random_windows(reg, cap, seed=cap) if cap <= 1000 else _disjoint(reg, cap)
            h, g = st.create_instances(np.arange(cap, dtype=np.uint64) * 7919 + 1, rel, mask)
            self.assertEqual(st.live_count, cap)
            self.assertEqual(len(set(h.tolist())), cap)
            with self.assertRaises(StateError) as cm:
                st.create_instance(10**12, [int(reg.relation_ids[0]), None, None, None])
            self.assertEqual(cm.exception.code, "INSTANCE_CAPACITY")
            st.release_instance(int(h[-1]), int(g[-1]))
            st.create_instance(10**12, [int(reg.relation_ids[0]), None, None, None])
            self.assertEqual(st.live_count, cap)
            self.assertLessEqual(int(st.row_ref.sum()), 4 * cap)
            self.assertEqual(int(st.row_ref.sum()), int(st.inst_bound.sum()))


def _disjoint(reg, n):
    """n windows of 1..4 relations with no relation shared between windows."""
    rng = np.random.default_rng(n)
    perm = rng.permutation(reg.num_relations)[:4 * n].reshape(n, 4)
    mask = rng.random((n, 4)) < 0.8
    mask[:, 0] = True
    return reg.relation_ids[perm], mask


if __name__ == "__main__":
    unittest.main()
