"""Pure derivation checks; no trained model or runtime benchmark."""
from __future__ import annotations
import unittest
from research.yang_mills_hca.r120_checkpoint_policy.qualify import derive


class TestDerivation(unittest.TestCase):
    def test_policy_uses_whole_native_or_r114_cell(self):
        by_graph = {}
        for graph in ("g0", "g1", "g2", "g3"):
            for k in (0, 2, 6):
                for ck in ("54007/STATE8", "54007/STATE8_SIGNED8", "54007/STATE8_TF1",
                           "54031/STATE8", "54031/STATE8_SIGNED8", "54031/STATE8_TF1"):
                    for h in (8, 20):
                        for ctrl, nse in (("NATIVE", 1.0), ("R114", 0.8)):
                            by_graph[f"{graph}/k{k}/{ctrl}/{ck}/H{h}"] = {
                                "nse_vs_composed_teacher": nse,
                                "matched_nse_vs_composed_teacher": 0.1,
                                "energy_ratio_vs_composed_teacher": 1.0,
                            }
        o = derive({"by_graph": by_graph, "phase": "fixture", "k_values": [0, 2, 6],
                    "horizons": [8, 20], "seed_leak": False, "weights_unchanged": True})
        self.assertEqual(o["summary"]["k0/SIGNED_ONLY/H20"]["nse"], (4 * 1.0 + 2 * 0.8) / 6)
        self.assertEqual(o["gate"], "FRESH_SEED_NONQUAL")

if __name__ == "__main__": unittest.main()
