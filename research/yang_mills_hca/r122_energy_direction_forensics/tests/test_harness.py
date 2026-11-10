"""Harness checks on consumed R121 material only (needs torch for the frozen teacher)."""
import json
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r122_energy_direction_forensics import controls as C
from research.yang_mills_hca.r122_energy_direction_forensics import harness as H
from research.yang_mills_hca.r122_energy_direction_forensics import linear as L
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M


@unittest.skipUnless(HAVE_TORCH, "torch is required for the frozen teacher")
class TestHarnessOnConsumedFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
        cls.o = get_oracle()
        cfg = C.regression_config()
        cls.systems, cls.parts = H.build_systems(cls.o, cfg["graphs"][C.REGRESSION_GRAPH], cfg["structure_seed"], ks=(0, 2))
        st0 = cls.o.r64.pair_state(cfg["evaluation_seeds"][0], "weak", 160, batch=4)
        cls.A0, cls.B0 = st0[0].numpy(), st0[1].numpy()

    def test_known_inversion_and_native_alignment(self):
        S = self.systems[0]
        X0 = S.x0(self.A0, self.B0)
        T = S.teacher(self.o, X0)
        ck = "54007/STATE8"
        r114 = M.decompose(S.view(S.rollout(X0, ck, "R114")[20], "READOUT"), S.view(T[20], "READOUT"))
        nat = M.decompose(S.view(S.rollout(X0, ck, "NATIVE")[20], "READOUT"), S.view(T[20], "READOUT"))
        self.assertTrue(np.all(r114["dir_class"] == M.D_REVERSED) and float(r114["edir"].min()) > 3.99)
        self.assertTrue(np.all(nat["dir_class"] == M.D_ALIGNED) and float(nat["c"].min()) > 0.999)

    def test_telescoping_identities_and_teacher_isolation(self):
        S = self.systems[2]
        X0 = S.x0(self.A0, self.B0)
        T = S.teacher(self.o, X0)
        ck = "54031/STATE8_SIGNED8"
        X = S.rollout(X0, ck, "DELAYED8")
        F20 = S.step(T[19], ck, "R114")
        e, L_, Pp = X[20] - T[20], F20 - T[20], X[20] - F20
        self.assertLess(float(np.abs(e - (Pp + L_)).max()), 1e-15)
        ys = [X[20]] + [S.rollout(T[j], ck, "DELAYED8", start=j)[-1] for j in (4, 8, 12, 16)] + [T[20]]
        total = sum(ys[b] - ys[b + 1] for b in range(5))
        self.assertLess(float(np.abs(total - e).max()), 1e-15)
        again = S.rollout(X0, ck, "DELAYED8")              # after teacher-forced calls: bit identical
        self.assertTrue(np.array_equal(again.view(np.uint64), X.view(np.uint64)))
        self.assertTrue(np.array_equal(X[8].view(np.uint64), S.rollout(X0, ck, "NATIVE")[8].view(np.uint64)))

    def test_independent_teacher_reference(self):
        S = self.systems[2]
        X0 = S.x0(self.A0, self.B0)
        T = S.teacher(self.o, X0, steps=3)
        for h in (1, 2, 3):
            self.assertLess(float(np.abs(H.reference_teacher_step(S, T[h - 1]) - T[h]).max()), 1e-13)

    def test_decayed_regime_operator_is_the_eight_gate_operator(self):
        S = self.systems[0]
        rec, ops = L.analyze_system(S, self.o, checkpoints=("54007/STATE8", "54007/STATE8_SIGNED8"))
        self.assertEqual(rec["max_probe_homogeneity_abs"], 0.0)
        self.assertLess(rec["max_analytic_vs_probe_rel"], 1e-12)
        self.assertLess(rec["checkpoints"]["54007/STATE8"]["r104_controller_vs_1p5M_minus_0p5I_max_abs"], 1e-12)

    def test_gauge_controls(self):
        S = self.systems[0]
        X0 = S.x0(self.A0, self.B0)
        T = S.teacher(self.o, X0)
        tol = {"gauge_covariance_abs": 1e-9, "gauge_metric_invariance": 1e-9, "gauge_negative_control_min": 1e-6}
        res = C.gauge_controls(self.o, S, self.parts, X0, T, ("54007/STATE8",), np.random.default_rng(1), tol)
        self.assertTrue(res["pass"], res)

    def test_weights_unchanged(self):
        self.assertEqual(self.o.parameter_pins(), P.CHECKPOINT_SHA256)


if __name__ == "__main__":
    unittest.main()
