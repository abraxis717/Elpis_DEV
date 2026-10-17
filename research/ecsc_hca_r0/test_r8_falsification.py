"""R8: assertions cover reproducibility, no oracle leakage, refusal and epistemic gates."""
import json
import unittest
from unittest.mock import patch

from .r8_falsification import (METHODS, SCHEMA, admitted_task, index_only, profile, run, summarize)
from .r2_trajectories import trajectory
from .task import predict


class EpistemicFalsificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first = profile(4, False)

    def test_exact_index_is_zero_hop(self):
        event = next(iter(trajectory(arity=4)))
        result = index_only(event.verified.graph, event.task)
        self.assertEqual(result.coordinate, event.verified.graph.view.coordinate)
        self.assertEqual(result.query, event.task)
        self.assertTrue(all(p.hit.grant.axiom.key in event.task.seeds
                            for p in result.predictions))

    def test_admitted_comparator_rejects_candidate_predicates(self):
        event = next(iter(trajectory(arity=4)))
        restricted = admitted_task(event.task)
        self.assertNotEqual(restricted, event.task)
        self.assertTrue(all(p.statuses == ("ADMITTED",) for p in restricted.predicates))
        result = predict(event.verified.graph, restricted, "B0")
        self.assertTrue(all(r.relation.status == "ADMITTED"
                            for p in result.predictions for r in p.relations))

    def test_existing_oracle_never_appears_in_task_seeds(self):
        for event in trajectory(arity=4):
            self.assertFalse(event.withheld.intersection(event.task.seeds))

    def test_all_methods_report_one_disposition_per_event(self):
        self.assertEqual(len(self.first["events"]), 20)
        for row in self.first["events"]:
            self.assertEqual(tuple(row["methods"]), METHODS)
            for item in row["methods"].values():
                self.assertNotEqual(item["score"] is None, item["refusal"] is None)

    def test_b0_and_b0i_exact_prediction_and_refusal_equality(self):
        for row in self.first["events"]:
            a, b = row["methods"]["B0"], row["methods"]["B0I"]
            self.assertEqual(a["refusal"], b["refusal"])
            if a["score"]:
                for field in ("predicted_keys", "tp", "fp", "fn", "binding_count", "invalid"):
                    self.assertEqual(a["score"][field], b["score"][field])

    def test_candidate_evidence_does_not_appear_in_admitted_only(self):
        for row in self.first["events"]:
            data = row["methods"]["ADMITTED_ONLY"]["score"]
            if data is not None:
                self.assertEqual(data["candidate_evidence_links"], 0)

    def test_replay_determinism_non_time_data(self):
        self.assertEqual(self.first, profile(4, False))
        self.assertEqual(SCHEMA, "elpis.ecsc.r8.dev-epistemic-falsification.v1")

    def test_summary_is_exactly_derived_from_rows(self):
        aggregate = summarize([self.first])
        self.assertEqual(sum(v["answered"] + sum(v["refusals"].values())
                             for v in aggregate.values()), 20 * len(METHODS))
        self.assertEqual(aggregate["B0"]["tp"], aggregate["B0I"]["tp"])


if __name__ == "__main__":
    unittest.main()
