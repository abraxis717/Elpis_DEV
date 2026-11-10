"""R119: the guarded kernel is bit-identical to the frozen R118 kernel; guard bounds are sound."""
from __future__ import annotations

import math
import unittest
from fractions import Fraction

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.awr_state import AWRState, StateCapacity
from research.yang_mills_hca.r118_scalable_awr.engine import Budgets
from research.yang_mills_hca.r118_scalable_awr.fixtures import synthetic_registry
from research.yang_mills_hca.r118_scalable_awr.registry import RegistryLimits, build_registry, rotation_blocks
from research.yang_mills_hca.r118_scalable_awr.tests._util import compare_with_simple, random_windows, seed_states

from .. import differential as D
from .. import guards
from .. import native_r119 as N
from ..engine_r119 import ALL_GUARDS, R119Engine


def families():
    return {
        "connected": synthetic_registry(300, 1200, 7, arity=(2, 5)),
        "zipf": synthetic_registry(200, 700, 8, arity=(1, 8), zipf=1.3),
        "hub16": synthetic_registry(80, 300, 9, arity=(8, 16), hub_fraction=0.9, limits=RegistryLimits(max_arity=16)),
    }


class GuardBounds(unittest.TestCase):
    def test_theta_is_sound_and_tight(self):
        for key, rec in P.checkpoint_records().items():
            W, b = rec["weights"], rec["bias"]
            theta = guards.gate_theta(W, b)
            self.assertGreater(theta, 0.0, key)
            bound = guards.gate_bound_m2(W, b)
            self.assertLessEqual(Fraction(theta) ** 2, bound)
            self.assertGreater(Fraction(math.nextafter(theta, math.inf)) ** 2, bound)
            for q in range(8):  # re-derive the per-gate inequality independently
                _, e = math.frexp(abs(b[q]))
                gap = Fraction(2) ** (e - 1 - 53)
                S = sum((Fraction(abs(w)) for w in W[q]), Fraction(0))
                lhs = 4 * S * Fraction(theta) ** 2 * (1 + Fraction(1, 2 ** 47)) + Fraction(1, 2 ** 1065)
                self.assertLessEqual(lhs, gap / 4, (key, q))

    def test_theta_refuses_unusable_biases(self):
        W = [[0.1] * 10 for _ in range(8)]
        self.assertEqual(guards.gate_theta(W, [0.5] * 7 + [0.0]), 0.0)
        self.assertEqual(guards.gate_theta(W, [0.5] * 7 + [1e-310]), 0.0)
        self.assertEqual(guards.gate_theta([[50.0] * 10] * 8, [0.5] * 8), 0.0)  # S > 100

    def test_libm_precondition(self):
        lib, _ = N.kernel()
        self.assertEqual(lib.awr119_libm_selfcheck(), 1)
        for t in (0.0, -0.0, -2.0 ** -56, -2.0 ** -57, -2.0 ** -1074, -5e-324, -3e-300, -1.3e-17):
            self.assertEqual(math.exp(t), 1.0, t)


class KernelDifferential(unittest.TestCase):
    def test_bit_identity_across_regimes_and_thresholds(self):
        rng = np.random.default_rng(1190)
        hits = np.zeros(8, dtype=np.int64)
        for fname, reg in families().items():
            for ck in ("54007/STATE8_SIGNED8", "54031/STATE8", "54007/STATE8_TF1"):
                for ctrl in ("NATIVE", "R104", "R114"):
                    dif = D.Differential(reg, ck, ctrl)
                    for mode in ("uniform_scale", "mixed_rows", "mixed_coords", "zeros_and_signs", "subnormal",
                                 "threshold"):
                        for _ in range(3):
                            slot_rel, slot_cmp, nr = D.windows_for(reg, 32, rng)
                            cin = (D.threshold_states(rng, nr, dif.theta) if mode == "threshold"
                                   else D.random_states(rng, nr, mode))
                            rc, mb, mu = dif.compare(slot_rel, slot_cmp, cin)
                            self.assertEqual((rc, mb, mu), (0, 0, 0), (fname, ck, ctrl, mode))
                    hits += dif.gc
        for i in range(4):  # every guard was both taken and declined
            self.assertGreater(hits[i], 0, i)
            self.assertGreater(hits[4 + i] - hits[i], 0, i)

    def test_maximum_in_degree_window(self):
        pids = list(range(1, 65))
        frames = rotation_blocks(np.linspace(-3, 3, 256), np.linspace(2, -1, 256))
        reg = build_registry(pids, [1, 2, 3, 4], [0, 64, 128, 192, 256], pids * 4, frames, RegistryLimits(max_arity=64))
        rng = np.random.default_rng(5)
        dif = D.Differential(reg, "54007/STATE8_SIGNED8", "R114")
        for mode in ("uniform_scale", "mixed_coords", "zeros_and_signs", "subnormal"):
            for _ in range(10):
                cin = D.random_states(rng, 4, mode)
                rc, mb, mu = dif.compare(np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2, 3]]), cin)
                self.assertEqual((rc, mb, mu), (0, 0, 0), mode)


class EngineModes(unittest.TestCase):
    def test_guarded_engine_equals_simple_reference_and_r118(self):
        for fname, reg in families().items():
            for scale in (1.0, 1e-105, 1e-152, 1e-158, 1e-300, 1e-310):
                states = []
                for mode in ("EXACT_R118", "EXACT_GUARDED"):
                    st = AWRState(reg, StateCapacity(max_instances=96, max_materialized_relations=384,
                                                     max_bindings_per_relation=256))
                    rel, mask = random_windows(reg, 96, 11)
                    h, _ = st.create_instances(np.arange(96, dtype=np.uint64) + 1, rel, mask)
                    seed_states(st, 12)
                    st.pool *= scale
                    eng = R119Engine(st, "54007/STATE8_SIGNED8", budgets=Budgets(max_passes=20), mode=mode)
                    res, diff, eq, n = compare_with_simple(eng, h[:64], passes=3)
                    self.assertEqual(res.disposition, "EXECUTED")
                    self.assertEqual(eq, n, (fname, scale, mode))  # value-equal to the simple reference
                    states.append(st.pool.copy())
                np.testing.assert_array_equal(states[0].view(np.uint64), states[1].view(np.uint64))

    def test_refusals_identical_across_modes(self):
        reg = families()["connected"]
        outs = []
        for mode in ("EXACT_R118", "EXACT_GUARDED", "FTZ_DAZ"):
            st = AWRState(reg, StateCapacity(max_instances=60, max_materialized_relations=240))
            rel, mask = random_windows(reg, 60, 3)
            h, g = st.create_instances(np.arange(60, dtype=np.uint64), rel, mask)
            disp = []
            for bud in (Budgets(max_active_instances=10), Budgets(max_passes=1), Budgets(max_active_relations=5)):
                disp.append(R119Engine(st, "54031/STATE8", budgets=bud, mode=mode).step(h, passes=2).disposition)
            st.release_instance(int(h[0]), int(g[0]))
            disp.append(R119Engine(st, "54031/STATE8", mode=mode).step(h).disposition)
            outs.append(disp)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])
        self.assertEqual(outs[0], ["REFUSED_INSTANCE_BUDGET", "REFUSED_PASS_BUDGET", "REFUSED_RELATION_BUDGET",
                                   "REFUSED_DEAD_HANDLE"])

    def test_guard_mask_subsets_are_bit_identical(self):
        reg = families()["zipf"]
        pools = {}
        for mask in (0, 1, 2, 4, 8, ALL_GUARDS):
            st = AWRState(reg, StateCapacity(max_instances=80, max_materialized_relations=320,
                                             max_bindings_per_relation=256))
            rel, mask_ = random_windows(reg, 80, 4)
            h, _ = st.create_instances(np.arange(80, dtype=np.uint64), rel, mask_)
            seed_states(st, 6)
            st.pool *= 1e-150
            R119Engine(st, "54007/STATE8", controller="R104", budgets=Budgets(max_passes=8),
                       guard_mask=mask).step(h, passes=8)
            pools[mask] = st.pool.view(np.uint64).copy()
        for mask, pool in pools.items():
            np.testing.assert_array_equal(pool, pools[0], err_msg=str(mask))


if __name__ == "__main__":
    unittest.main()
