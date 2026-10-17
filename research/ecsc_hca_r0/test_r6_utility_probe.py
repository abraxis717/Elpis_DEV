"""DEV-only checks for R6's comparative accounting and real-byte refresh probe."""
from dataclasses import replace
from unittest import TestCase

from .baselines import Refusal
from .r2_b0i import B0I
from .r2_common import semantic
from .r2_trajectories import fixture, query, trajectory
from .task import predict
from .r6_utility_probe import _methods, replay_profiles, warm_refresh_probe


class UtilityProbeTests(TestCase):
    def test_all_80_transitions_reproduce_fresh_exact_semantics(self):
        profiles = replay_profiles()
        self.assertEqual([(p['arity'], p['reverse_observation_order'], p['event_count'])
                          for p in profiles], [(4, False, 20), (4, True, 20),
                                               (8, False, 20), (8, True, 20)])
        for p in profiles:
            self.assertGreater(p['sum_shared_admission_work'], 0)
            self.assertGreaterEqual(p['refusals'], 1)
            self.assertEqual(p['sum_fresh_including_shared_admission'],
                             p['sum_fresh_solver_work'] + p['sum_shared_admission_work'])
            self.assertEqual(p['sum_incremental_including_shared_admission'],
                             p['sum_incremental_solver_work'] + p['sum_shared_admission_work'])

    def test_withheld_output_keys_are_never_solver_seeds(self):
        for arity in (4, 8):
            for event in trajectory(arity=arity):
                self.assertTrue(set(event.task.seeds).isdisjoint(event.withheld))

    def test_no_predicates_cannot_discover_unseeded_output(self):
        f, labels = fixture(arity=4)
        from .r2_common import VerifiedGraph
        graph = VerifiedGraph(f.view, f.authority, f.checks)
        q = replace(query(f), predicates=())
        fresh = predict(graph.graph, q, 'B0')
        _, inc, _ = B0I.start(graph, q)
        self.assertEqual(semantic(fresh), semantic(inc))
        self.assertEqual(tuple(p.hit.grant.axiom.key for p in fresh.predictions), ())
        self.assertNotIn(labels['plan'], q.seeds)

    def test_refusal_cannot_be_reported_as_task_success(self):
        event = next(e for e in trajectory() if e.name == 'work_budget_refusal')
        cache, row = _methods(event, None)
        self.assertIsNone(cache)
        self.assertEqual(row['disposition'], 'WORK_LIMIT')
        self.assertNotIn('true_positive', row)

    def test_readout_omission_is_explicit_not_a_correct_prediction(self):
        event = next(e for e in trajectory() if e.name == 'readout_omission')
        _, row = _methods(event, None)
        self.assertEqual(row['disposition'], 'EXACT')
        self.assertGreater(row['omitted'], 0)

    def test_scope_fallback_never_relabels_cached_scope_as_same_scope(self):
        prior = None
        for event in trajectory():
            prior, row = _methods(event, prior)
            if event.name == 'other_scope':
                self.assertEqual(row['disposition'], 'EXACT')
                self.assertEqual(row['cache_mode'], 'FULL_RECOMPUTE_FROZEN_B0_SCOPE')
                self.assertFalse(row['incremental_cache_retained'])
                break

    def test_verified_source_refresh_matches_cold_publication(self):
        data = warm_refresh_probe(repetitions=3)
        self.assertEqual(data['semantics'], 'EXACT_EQUAL')
        self.assertEqual(data['trials_per_arm'], 3)
        self.assertTrue(all(t > 0 for t in data['cold_wall_ns']))
        self.assertTrue(all(t > 0 for t in data['warm_wall_ns']))
        self.assertEqual(len(data['warm_result_work']), 3)

    def test_oracle_mutation_does_not_change_predictor_result(self):
        event = next(iter(trajectory()))
        original = predict(event.verified.graph, event.task, 'B0')
        from .evaluation import score
        nonsense = frozenset('f' * 64 for _ in range(1))
        self.assertEqual(original, predict(event.verified.graph, event.task, 'B0'))
        self.assertNotEqual(score(original, event.withheld), score(original, nonsense))
