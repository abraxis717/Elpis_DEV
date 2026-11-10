"""Registered decision logic and protocol consistency (no model execution)."""
import json
import unittest

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r122_energy_direction_forensics import analyze as A
from research.yang_mills_hca.r122_energy_direction_forensics import harness as H
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M
from research.yang_mills_hca.r122_energy_direction_forensics import qualify as Q

PROTO = json.loads(Q.PROTOCOL.read_bytes())
THR = PROTO["thresholds"]


class TestPolicies(unittest.TestCase):
    def test_policy_trajectory_identities(self):
        for ck in P.CHECKPOINTS:
            signed = H.is_signed(ck)
            self.assertEqual(H.trajectory_of(ck, "NATIVE"), "NATIVE")
            self.assertEqual(H.trajectory_of(ck, "R114"), "R114")
            self.assertEqual(H.trajectory_of(ck, "SIGNED_ONLY"), "R114" if signed else "NATIVE")
            self.assertEqual(H.trajectory_of(ck, "DELAYED8"), "DELAYED8" if signed else "NATIVE")


class TestDecisions(unittest.TestCase):
    def item(self, lam, aden=1.0, frac=1.0):
        return {"eligible_fraction": frac, "a_den_median": aden, "lambda_local_median": lam}

    def test_stratum_label_order(self):
        self.assertEqual(A._label(self.item(0.01, frac=0.5), THR), "INVALID")
        self.assertEqual(A._label(self.item(0.01, aden=10.0), THR), "DENOMINATOR_COLLAPSE")
        self.assertEqual(A._label(self.item(0.5), THR), "LOCAL_TRANSITION")
        self.assertEqual(A._label(self.item(0.1), THR), "ACCUMULATED_DRIFT")
        self.assertEqual(A._label(self.item(0.3), THR), "MIXED_LOCAL_ACCUMULATED")
        self.assertEqual(A._label(self.item(None), THR), "INVALID")

    def test_dominance_needs_overall_share_and_every_graph(self):
        acc, mix = "ACCUMULATED_DRIFT", "MIXED_LOCAL_ACCUMULATED"
        d = A.dominance({f"g{i}": [acc] * 6 for i in range(6)}, A.MECHANISMS, THR)
        self.assertEqual(d["dominant"], acc)
        # 30 of 36 overall (83 %), but one graph class contradicts: not dominant
        labels = {f"g{i}": [acc] * 6 for i in range(5)}
        labels["g5"] = [mix] * 6
        self.assertIsNone(A.dominance(labels, A.MECHANISMS, THR)["dominant"])
        # every graph has a majority but the overall share is 67 %: not dominant
        self.assertIsNone(A.dominance({f"g{i}": [acc] * 4 + [mix] * 2 for i in range(6)}, A.MECHANISMS, THR)["dominant"])

    def test_disposition(self):
        def dec(dom):
            return {"ALL": {"dominant": dom}}
        self.assertEqual(A.disposition(dec("ACCUMULATED_DRIFT"), 0.0, THR), "MECHANISM_LOCALIZED_BOUNDED")
        self.assertEqual(A.disposition(dec(None), 0.0, THR), "MECHANISM_MIXED_BOUNDED")
        self.assertEqual(A.disposition(dec("MIXED_LOCAL_ACCUMULATED"), 0.0, THR), "MECHANISM_MIXED_BOUNDED")
        self.assertEqual(A.disposition(dec("ACCUMULATED_DRIFT"), 0.2, THR), "MECHANISM_INDETERMINATE")


class TestProtocol(unittest.TestCase):
    def test_status_and_source_pins(self):
        self.assertEqual(PROTO["status"], Q.REGISTERED_STATUS)
        self.assertEqual(PROTO["source_sha256"], Q.source_sha256())
        self.assertEqual(PROTO["r93"], "ENERGY_NORMALIZED_DENOMINATOR_HOLD")

    def test_classifier_constants(self):
        c = PROTO["classifier"]
        self.assertEqual((c["r_floor"], c["cos_aligned"], c["cos_reversed"]), (M.R_FLOOR, M.COS_ALIGNED, M.COS_REVERSED))

    def test_seeds_are_disjoint_from_history(self):
        audit = Q.seed_audit(PROTO)
        self.assertTrue(audit["pass"], audit)
        self.assertGreater(audit["consumed_seed_count"], 300)

    def test_scales_and_horizons(self):
        self.assertEqual(PROTO["scales"], {"2^-20": 2.0 ** -20, "2^-2": 0.25, "1": 1.0, "2^2": 4.0})
        self.assertEqual(PROTO["horizons"], [1, 4, 8, 12, 16, 20])
        self.assertIn(THR["primary_scale"], PROTO["scales"])


if __name__ == "__main__":
    unittest.main()
