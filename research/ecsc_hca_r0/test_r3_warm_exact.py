"""R3 DEV WARM exact-memory adapter regression (not scientific QUAL)."""
from dataclasses import replace
import unittest

from elpis.runtime.context import ContextLimits, Observation, successor

from .baselines import Refusal
from .fixtures import h
from .r2_trajectories import fixture, query
from .r3_warm_exact import ExactWarmContext
from .task import predict
from .r2_common import semantic


class WarmExactContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.keys = fixture(arity=4)
        self.task = query(self.fixture)
        self.state, self.result, self.meta = ExactWarmContext.open(
            self.fixture.view, self.fixture.authority, self.fixture.checks, self.task)

    def refusal(self, code, fn):
        with self.assertRaises(Refusal) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)

    def exact(self, state, task, result):
        self.assertEqual(semantic(result), semantic(predict(state.verified.graph, task, "B0")))
        self.assertEqual(result.coordinate, state.verified.graph.view.coordinate)

    def successor_view(self, old, *observations):
        return successor(old, tuple(observations), ContextLimits(),
                         through=old.through + 1, dropped=old.dropped)

    def test_explicit_cold_open_matches_frozen_b0(self):
        self.assertEqual(self.meta["lifetime"], "EXPLICIT_COLD_OPEN")
        self.assertGreater(self.meta["graph_build_work"], 0)
        self.exact(self.state, self.task, self.result)

    def test_same_published_view_uses_exact_cache(self):
        nxt, result, meta = self.state.advance(self.fixture.view, self.fixture.authority,
                                               self.fixture.checks, self.task)
        self.assertIs(nxt.cache, self.state.cache)
        self.assertEqual(meta["lifetime"], "SAME_PUBLISHED_VIEW")
        self.assertEqual(meta["mode"], "EXACT_CACHE_REUSE")
        self.assertEqual(meta["graph_build_work"], 0)
        self.exact(nxt, self.task, result)

    def test_successor_preserves_exactness_and_old_snapshot(self):
        v = self.successor_view(self.fixture.view)
        old = repr(self.state)
        nxt, result, meta = self.state.advance(v, self.fixture.authority,
                                               self.fixture.checks, self.task)
        self.assertEqual(meta["lifetime"], "VERIFIED_IMMEDIATE_SUCCESSOR")
        self.assertGreater(meta["graph_build_work"], 0)
        self.exact(nxt, self.task, result)
        self.assertEqual(repr(self.state), old)

    def test_real_source_refresh_requires_successor(self):
        ref = next(a.references[0] for a in self.fixture.view.axioms if a.references)
        new_checks = tuple(replace(c, observed_digest=h("different-source"))
                           if c.reference == ref else c for c in self.fixture.checks)
        self.refusal("SOURCE_REFRESH_REQUIRES_SUCCESSOR", lambda: self.state.advance(
            self.fixture.view, self.fixture.authority, new_checks, self.task))
        v = self.successor_view(self.fixture.view)
        nxt, result, _ = self.state.advance(v, self.fixture.authority, new_checks, self.task)
        self.exact(nxt, self.task, result)
        self.assertEqual(self.state.checks, self.fixture.checks)

    def test_skipped_and_alien_views_refuse(self):
        intermediate = self.successor_view(self.fixture.view)
        skipped = self.successor_view(intermediate)
        self.refusal("PREDECESSOR", lambda: self.state.advance(
            skipped, self.fixture.authority, self.fixture.checks, self.task))
        alien = replace(intermediate, predecessor="a" * 64)
        self.refusal("PREDECESSOR", lambda: self.state.advance(
            alien, self.fixture.authority, self.fixture.checks, self.task))

    def test_scope_authority_and_budget_refuse_without_mutation(self):
        old = repr(self.state)
        self.refusal("SCOPE_TRANSITION", lambda: self.state.advance(
            self.fixture.view, self.fixture.authority, self.fixture.checks,
            query(self.fixture, "project-debian")))
        self.refusal("AUTHORITY_CONTINUITY", lambda: self.state.advance(
            self.fixture.view, replace(self.fixture.authority, epoch="another-policy"),
            self.fixture.checks, self.task))
        self.refusal("WORK_LIMIT", lambda: self.state.advance(
            self.fixture.view, self.fixture.authority, self.fixture.checks,
            replace(self.task, work_budget=1)))
        self.assertEqual(repr(self.state), old)


if __name__ == "__main__":
    unittest.main()
