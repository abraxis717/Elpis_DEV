"""Incidence signatures, the historical inventory and the registered graphs (stdlib only)."""
import itertools
import json
import random
import unittest

from research.yang_mills_hca.r122_energy_direction_forensics import qualify as Q
from research.yang_mills_hca.r122_energy_direction_forensics import topology as T


class TestTopology(unittest.TestCase):
    def test_invariance_under_relabeling_and_relation_permutation(self):
        rng = random.Random(3)
        for label, g in list(T.historical_graphs().items())[:20]:
            ids = sorted({p for row in g for p in row})
            perm = dict(zip(ids, rng.sample(range(100, 200), len(ids))))
            order = rng.sample(range(4), 4)
            h = [sorted(perm[p] for p in g[i]) for i in order]
            self.assertEqual(T.signature(g), T.signature(h), label)
            self.assertEqual(T.canonical_form(g) == T.canonical_form(h), True)

    def test_two_complete_invariants_agree(self):
        hg = list(T.historical_graphs().values())
        for a, b in itertools.combinations(hg, 2):
            self.assertEqual(T.signature(a) == T.signature(b), T.canonical_form(a) == T.canonical_form(b))

    def test_r117_training_graphs_share_a_class_but_not_a_rooted_class(self):
        r61 = json.loads((T.SNAPSHOT / "FROZEN_PROTOCOL_R61.json").read_bytes())["r55_structural"]["graphs"]
        self.assertEqual(T.signature(r61["D1_P1"]), T.signature(r61["D2_P1"]))
        self.assertNotEqual(T.rooted_signature(r61["D1_P1"]), T.rooted_signature(r61["D2_P1"]))

    def test_r121_classes_against_the_complete_inventory(self):
        r121 = json.loads(T.R121_PROTOCOL.read_bytes())["graphs"]
        others = {}
        for label, g in T.historical_graphs().items():
            if not label.startswith("R121:"):
                others.setdefault(T.signature(g), set()).add(label.rsplit("/", 1)[-1])
        hit = {n: sorted(others.get(T.signature(g), [])) for n, g in r121.items()}
        self.assertEqual(hit["R121_HOLDOUT_1"], ["r52_fan"])
        self.assertEqual(hit["R121_HOLDOUT_2"], [])
        self.assertEqual(hit["R121_HOLDOUT_3"], ["R72_G2"])
        self.assertEqual(hit["R121_HOLDOUT_4"], ["R70_STAR"])

    def test_registered_graphs_are_novel_and_regenerable(self):
        proto = json.loads(Q.PROTOCOL.read_bytes())
        gen = proto["graph_generation"]
        again = T.generate(gen["seed"], gen["degree_schedule"], set(T.historical_classes()), attempts=gen["attempts_per_graph"])
        self.assertEqual(again, list(proto["graphs"].values()))
        audit = T.audit(proto["graphs"])
        self.assertTrue(audit["all_novel"])
        self.assertEqual(len({tuple(v["signature"]) for v in audit["graphs"].values()}), len(proto["graphs"]))
        for g in proto["graphs"].values():
            self.assertTrue(T.connected(g) and T.participants(g) == 10 and T.hop_distance(g) >= 1)

    def test_family_inventory(self):
        classes = set()

        def rec(start, left, counts, chosen):
            if left == 0:
                if counts == [4, 4, 4, 4]:
                    classes.add(tuple(chosen))
                return
            for m in range(start, 16):
                c = [counts[b] + (m >> b & 1) for b in range(4)]
                if max(c) <= 4:
                    rec(m, left - 1, c, chosen + [m])

        rec(1, 10, [0, 0, 0, 0], [])
        valid = {}
        for ms in classes:
            g = [[p for p, m in enumerate(ms) if m >> r & 1] for r in range(4)]
            if T.connected(g) and len({tuple(r) for r in g}) == 4:
                valid.setdefault(T.signature(g), g)
        hist = set(T.historical_classes())
        unused = [g for s, g in valid.items() if s not in hist]
        self.assertEqual(len(valid), 50)
        self.assertEqual(sorted(T.max_degree(g) for g in unused).count(4), 0)
        self.assertEqual(len(unused), 17)


if __name__ == "__main__":
    unittest.main()
