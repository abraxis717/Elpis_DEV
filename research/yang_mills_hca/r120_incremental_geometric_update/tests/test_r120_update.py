"""R120 incremental update: unit tests (numpy + gcc; no torch).

    python -B -m unittest research.yang_mills_hca.r120_incremental_geometric_update.tests.test_r120_update
"""
from __future__ import annotations

import unittest

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr import gauge as G118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine

from .. import cgc_bridge as CGC
from .. import engine_r120 as E
from .. import fixtures_r120 as F
from .. import native_r120 as N
from .. import reference as REF


def bits(a, b):
    return np.array_equal(np.ascontiguousarray(a, dtype=np.float64).view(np.uint64),
                          np.ascontiguousarray(b, dtype=np.float64).view(np.uint64))


def pool_vs_reference(st, ref_state):
    reg = st.registry
    return sum(not bits(st.pool[st.rel_row[int(reg.relation_index([rid])[0])]], vec) for rid, vec in ref_state.items())


class Law(unittest.TestCase):
    def setUp(self):
        self.reg, self.st, *_ = F.small_state()
        self.eng = E.IncrementalEngine(self.st, F.CHECKPOINT)

    def test_kernel_identity(self):
        lib, info = N.kernel()
        self.assertEqual(info["r119_kernel_sha256"], N.R119_KERNEL_SHA256)
        self.assertEqual(P.sha256_file(P.YM_ROOT / "r119_numerical_hotpath/csrc/awr_kernel_r119.c"), N.R119_KERNEL_SHA256)
        self.assertIn("-ffp-contract=off", info["cflags"])
        self.assertNotIn("-ffast-math", " ".join(info["cflags"]))

    def test_cgc_native_equals_reference(self):
        rng = np.random.default_rng(1)
        out = np.zeros(3, dtype=np.float32)
        for _ in range(300):
            x = rng.uniform(-1, 1, 5).astype(np.float32)
            self.eng._lib.awr120_cgc_probe(self.eng.cgc_w.ctypes.data, x.ctypes.data, out.ctypes.data)
            ref = np.array(CGC.forward_reference(self.eng.cgc_w, x), dtype=np.float32)
            self.assertTrue(np.array_equal(out.view(np.uint32), ref.view(np.uint32)))

    def test_cgc_shape_and_parameter_count(self):
        self.assertEqual(self.eng.cgc_w.shape, (3, 5, 8))
        self.assertEqual(self.eng.cgc_w.size, 120)
        self.assertFalse(self.eng.cgc_w.flags.writeable)
        self.assertEqual(self.eng.W.size + self.eng.b.size, 88)
        with self.assertRaises(ValueError):
            CGC.check_weights(np.zeros((3, 5, 7), dtype=np.float32))

    def test_update_law_native_equals_reference(self):
        rng = np.random.default_rng(2)
        frames = G118.random_so2x2(rng, 16)
        for scale in (1.0, 3.9, 1e-160, 1e-310, 0.0):
            for _ in range(60):
                s, o = rng.normal(size=8) * scale, rng.normal(size=4)
                wt, aux = float(rng.uniform(-1, 1)), float(rng.uniform(0, 1))
                fr = frames[rng.integers(0, 16)]
                got, rc = self.eng.update_probe(s, fr, o, wt, aux)
                ref = REF.update_state(s, fr, o, wt, aux, self.eng.cgc_w, 1.0, 0.25, CGC.RHO_B, CGC.KAPPA)
                self.assertEqual(rc, 0)
                self.assertTrue(bits(got, ref))

    def test_zero_and_bounds(self):
        z, _ = self.eng.update_probe(np.zeros(8), None, np.zeros(4), 1.0, 1.0)
        self.assertFalse(z.any())
        rng = np.random.default_rng(3)
        fr = G118.random_so2x2(rng, 1)[0]
        big, rc = self.eng.update_probe(rng.normal(size=8) * 3.9, fr, rng.normal(size=4) * 1e6, 1.0, 0.5)
        self.assertEqual(rc, 0)
        self.assertLessEqual(np.linalg.norm(big[:4]), 4.0 + 1e-9)
        self.assertLessEqual(np.linalg.norm(big[4:]), 4.0 + 1e-9)

    def test_single_update_gauge_covariance(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            g, fr = G118.random_so2x2(rng, 2)
            s, o = rng.normal(size=8), rng.normal(size=4)
            out, _ = self.eng.update_probe(s, fr, o, 0.7, 0.2)
            sg = np.concatenate([G118.bapply(g, s[:4]), G118.bapply(g, s[4:])])
            outg, _ = self.eng.update_probe(sg, G118.bmul(g, fr), o, 0.7, 0.2)
            exp = np.concatenate([G118.bapply(g, out[:4]), G118.bapply(g, out[4:])])
            self.assertLess(np.abs(outg - exp).max(), 1e-12)

    def test_parameter_ranges(self):
        self.st._engines.clear()
        with self.assertRaises(ValueError):
            E.IncrementalEngine(self.st, F.CHECKPOINT, kappa=0.5)
        with self.assertRaises(ValueError):
            E.IncrementalEngine(self.st, F.CHECKPOINT, eta=2.0)


class Batches(unittest.TestCase):
    def test_reference_parity_bitwise(self):
        for controller, cache, threads in (("NATIVE", True, 1), ("R114", False, 1), ("R114", True, 4), ("R104", True, 2)):
            reg, st, *_ = F.small_state(seed=5 + threads)
            eng = E.IncrementalEngine(st, F.CHECKPOINT, controller=controller,
                                      limits=E.Limits(threads=threads, parallel_min_instances=2))
            if cache:
                eng.build_cache()
            rng = np.random.default_rng(17)
            for _ in range(8):
                batch = F.random_batch(st, rng, int(rng.integers(1, 12)))
                radius, passes = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                model, params = F.model_of(st, eng)
                disp, ref_state, _, info = REF.apply_batch(model, F.events_of(st, *batch), params, radius=radius, passes=passes)
                rep = eng.apply(*batch, radius=radius, passes=passes)
                self.assertEqual(rep.disposition, disp)
                self.assertEqual(rep.instances, len(info["active"]))
                self.assertEqual(pool_vs_reference(st, ref_state), 0)

    def test_step_equals_r119(self):
        reg, st, handles, *_ = F.small_state(seed=9)
        eng = E.IncrementalEngine(st, F.CHECKPOINT, controller="R114", limits=E.Limits(threads=4, parallel_min_instances=2))
        eng.build_cache()
        r119 = R119Engine(st, F.CHECKPOINT, controller="R114", mode="EXACT_GUARDED", budgets=E118.Budgets(max_passes=8))
        rng = np.random.default_rng(9)
        for _ in range(6):
            a = np.sort(rng.choice(handles, size=40, replace=False))
            base = st.pool.copy()
            self.assertTrue(r119.step(a, passes=2).exact)
            ref = st.pool.copy()
            st.pool[:] = base
            self.assertTrue(eng.step(a, passes=2).committed)
            self.assertTrue(bits(st.pool, ref))

    def test_identity_is_reused(self):
        reg, st, *_ = F.small_state(seed=11)
        eng = E.IncrementalEngine(st, F.CHECKPOINT)
        snap = F.snapshot(st, eng)
        ptr = st.pool.ctypes.data
        rng = np.random.default_rng(11)
        for _ in range(200):
            self.assertTrue(eng.apply(*F.random_batch(st, rng, 6)).committed)
        for k in ("row_rel", "rel_row", "slot_rel", "slot_row", "inst_gen", "row_gen"):
            self.assertTrue(np.array_equal(getattr(st, k), snap[k]), k)
        self.assertEqual(st.live_count, snap["live"])
        self.assertEqual(st.pool.ctypes.data, ptr)

    def test_duplicates_conflicts_order(self):
        reg, st, *_ = F.small_state(seed=12)
        eng = E.IncrementalEngine(st, F.CHECKPOINT)
        rng = np.random.default_rng(12)
        batch = F.random_batch(st, rng, 7, repeat=False)
        self.assertEqual(eng.apply(*batch).applied, 7)
        after = st.pool.copy()
        rep = eng.apply(*batch)
        self.assertEqual((rep.applied, rep.skipped_duplicate, rep.disposition), (0, 7, "OK"))
        self.assertTrue(bits(st.pool, after))
        ids, seq, parts, obs, wt, aux = [np.concatenate([x[:1], x[:1]]) for x in F.random_batch(st, rng, 1)]
        obs[1] += 1.0
        snap = F.snapshot(st, eng)
        self.assertEqual(eng.apply(ids, seq, parts, obs, wt, aux).disposition, "REFUSED_CONFLICT")
        self.assertTrue(F.unchanged(st, snap, eng))

    def test_refusals_are_atomic(self):
        reg, st, *_ = F.small_state(seed=13)
        rng = np.random.default_rng(13)

        def attempt(expect, mutate, limits=E.Limits(), n=6, **kw):
            st._engines.clear()
            eng = E.IncrementalEngine(st, F.CHECKPOINT, limits=limits)
            batch = list(F.random_batch(st, rng, n, repeat=False))
            gen = np.zeros(n, dtype=np.uint32)
            mutate(batch, gen)
            snap = F.snapshot(st, eng)
            rep = eng.apply(*batch, exp_gen=gen, **kw)
            self.assertEqual(rep.disposition, expect)
            self.assertTrue(F.unchanged(st, snap, eng), expect)

        attempt("REFUSED_INVALID_VALUE", lambda b, g: b[3].__setitem__((5, 1), np.nan))
        attempt("REFUSED_INVALID_VALUE", lambda b, g: b[4].__setitem__(5, -1.01))
        attempt("REFUSED_UNKNOWN_RELATION", lambda b, g: b[0].__setitem__(5, np.uint64(3)))
        attempt("REFUSED_STALE_RELATION", lambda b, g: g.__setitem__(5, 9))
        attempt("REFUSED_NOT_INCIDENT", lambda b, g: b[2].__setitem__(5, -1))
        attempt("REFUSED_INSTANCE_BUDGET", lambda b, g: None, limits=E.Limits(max_active_instances=2), n=12)
        attempt("REFUSED_LINK_BUDGET", lambda b, g: None, limits=E.Limits(max_links=2), n=3)
        attempt("REFUSED_EVENT_BUDGET", lambda b, g: None, limits=E.Limits(max_events=4), n=5)
        attempt("REFUSED_ARGUMENT", lambda b, g: None, passes=99)

    def test_stale_reference_after_rematerialization(self):
        reg, st, *_ = F.small_state(seed=14)
        eng = E.IncrementalEngine(st, F.CHECKPOINT)
        row = int(np.nonzero(st.row_ref == 1)[0][0])
        r = int(st.row_rel[row])
        rid = int(reg.relation_ids[r])
        _, gen0 = st.relation_ref(rid)
        h = int(st.instances_binding(r)[0])
        slots = [None if q < 0 else int(reg.relation_ids[q]) for q in st.slot_rel[h]]
        st.release_instance(h, int(st.inst_gen[h]))
        st.create_instance(424242, slots)
        ev = (np.array([rid], np.uint64), np.array([1], np.uint64), np.array([int(reg.members_of(r)[0])], np.int32),
              np.ones((1, 4)), np.array([0.5]), np.array([0.5]))
        self.assertEqual(eng.apply(*ev, exp_gen=np.array([gen0], np.uint32)).disposition, "REFUSED_STALE_RELATION")
        _, gen1 = st.relation_ref(rid)
        self.assertNotEqual(gen0, gen1)
        self.assertEqual(eng.apply(*ev, exp_gen=np.array([gen1], np.uint32)).disposition, "OK")

    def test_deferred_equals_immediate(self):
        reg, st, *_ = F.small_state(seed=15)
        eng = E.IncrementalEngine(st, F.CHECKPOINT)
        rng = np.random.default_rng(15)
        batch = F.random_batch(st, rng, 6, repeat=False)
        base, seq0 = st.pool.copy(), st.row_seq.copy()
        eng.apply(*batch)
        immediate = st.pool.copy()
        st.pool[:], st.row_seq[:] = base, seq0
        rep = eng.apply(*batch, propagate=False)
        self.assertTrue(rep.deferred)
        self.assertEqual(eng.pending_rows, 6)
        self.assertEqual(eng.propagate_pending().disposition, "OK")
        self.assertEqual(eng.pending_rows, 0)
        self.assertTrue(bits(st.pool, immediate))
        st._engines.clear()
        small = E.IncrementalEngine(st, F.CHECKPOINT, limits=E.Limits(max_active_instances=1))
        st.pool[:], st.row_seq[:] = base, seq0
        rep = small.apply(*batch, defer_on_budget=True)
        self.assertEqual((rep.disposition, small.pending_rows), ("COMMITTED_DEFERRED", 6))

    def test_end_to_end_gauge_covariance(self):
        reg, st, handles, gens, wins = F.small_state(seed=16)
        rng = np.random.default_rng(16)
        g_rel = G118.random_so2x2(rng, reg.num_relations)
        h_part = G118.random_so2x2(rng, reg.num_participants)
        batches = []
        for k in range(3):
            b = F.random_batch(st, rng, 8)
            b[1][:] = 10 * (k + 1) + np.arange(8, dtype=np.uint64)
            batches.append(b)

        def run(reg_x, st_x, rotate_obs=False):
            eng = E.IncrementalEngine(st_x, F.CHECKPOINT, controller="R114")
            for b in batches:
                obs = G118.bapply(h_part[b[2]], b[3]) if rotate_obs else b[3]
                self.assertTrue(eng.apply(b[0], b[1], b[2], obs, b[4], b[5], radius=2, passes=2).committed)
            return {int(reg_x.relation_ids[st_x.row_rel[r]]): st_x.pool[r].copy() for r in np.nonzero(st_x.row_ref > 0)[0]}

        base = run(reg, F.clone_state(reg, st, wins))
        reg_g = G118.gauge_registry(reg, g_rel=g_rel, h_part=h_part)
        st_g = F.clone_state(reg_g, st, wins)
        st_g.pool[:] = G118.gauge_rows(st_g, g_rel)
        out = run(reg_g, st_g, rotate_obs=True)
        worst = 0.0
        for rid, vec in base.items():
            g = g_rel[int(reg.relation_index([rid])[0])]
            exp = np.concatenate([G118.bapply(g, vec[:4]), G118.bapply(g, vec[4:])])
            worst = max(worst, float(np.abs(out[rid] - exp).max()))
        self.assertLess(worst, 1e-12)


if __name__ == "__main__":
    unittest.main()
