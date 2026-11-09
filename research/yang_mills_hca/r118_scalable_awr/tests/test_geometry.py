"""R118-A/C geometry with the authentic frozen operator (needs torch; skipped explicitly without)."""
from __future__ import annotations

import importlib.util
import unittest

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "torch (the frozen operator's historical dependency) is not installed")
class FrozenOperatorInSubstrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .. import frozen_oracle, qualify_a, qualify_c

        cls.o = frozen_oracle.get_oracle()
        cls.A, cls.Cq = qualify_a, qualify_c

    def test_r118a_covariance_negative_controls_and_isolation(self):
        cov = self.A.covariance_suite(self.o)
        for k, v in cov.items():
            if k == "cases":
                continue
            if k.startswith("negative"):
                self.assertGreaterEqual(v, self.A.NEGATIVE_MIN, k)
            else:
                self.assertLessEqual(v, self.A.PASS_TOL, k)
        iso = self.A.isolation_suite(self.o)
        self.assertEqual(iso["cross_instance_leak_h20"], 0.0)
        self.assertEqual(iso["isolated_zero_slot_max_abs_h20"], 0.0)
        self.assertLessEqual(self.A.controller_parity(self.o)["vectorized_vs_reference_max_abs"], self.A.PASS_TOL)

    def test_r118c_historical_replay_through_substrate(self):
        hr = self.Cq.historical_replay(self.o)
        self.assertEqual(hr["oracle_bitwise_cases"], hr["cases"])
        self.assertEqual(hr["oracle_max_abs"], 0.0)
        self.assertLessEqual(hr["native_h1_max_abs"], self.Cq.ONE_PASS_TOL)
        self.assertLessEqual(hr["native_h20_max_abs"], self.Cq.MULTI_PASS_TOL)
        self.assertLessEqual(hr["native_controlled_vs_oracle_h20_max_abs"], self.Cq.MULTI_PASS_TOL)

    def test_r118c_sparse_parity_covariance_hidden_channels(self):
        for name, r in self.Cq.sparse_parity(self.o).items():
            self.assertLessEqual(r["one_pass_max_abs"], self.Cq.ONE_PASS_TOL, name)
            self.assertLessEqual(r["multi_pass_max_abs"], self.Cq.MULTI_PASS_TOL, name)
            self.assertLessEqual(r["native_vs_simple_max_abs"], self.Cq.ONE_PASS_TOL, name)
        for tag, c in self.Cq.covariance(self.o).items():
            if not isinstance(c, dict):
                continue
            self.assertLessEqual(c["relation_gauge_covariance"], self.Cq.COV_TOL, tag)
            self.assertLessEqual(c["participant_gauge_invariance"], self.Cq.COV_TOL, tag)
            for neg in ("negative_state_only", "negative_frames_only", "negative_inverse_frames"):
                self.assertGreaterEqual(c[neg], self.Cq.NEG_MIN, tag + neg)
        hc = self.Cq.hidden_channels_and_passes(self.o)
        self.assertTrue(hc["active_rows_bitwise_unchanged"])
        self.assertTrue(hc["five_passes_one_call_equals_five_calls_bitwise"])

    def test_weights_never_mutated(self):
        from .. import provenance as P

        self.assertEqual(self.o.parameter_pins(), P.CHECKPOINT_SHA256)


if __name__ == "__main__":
    unittest.main()
