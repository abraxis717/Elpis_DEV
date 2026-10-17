"""DEV-only R2 adversarial transition and exact-comparator invariants."""
from dataclasses import FrozenInstanceError, replace
import builtins
import io
import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from elpis.runtime.context import ContextLimits, Observation, successor
from .baselines import Limits, Refusal
from .b2 import B2, predict_b2
from .evaluation import score
from .fixtures import h
from .task import predict
from .test_b2 import ring
from .r2_common import VerifiedGraph, semantic
from .r2_b0i import B0I
from .r2_b2s import B2S, Snapshot
from .r2_trajectories import externally_seeded_ring, fixture, query, trajectory


class R2Mechanics(unittest.TestCase):
    def setUp(self):
        self.f, self.labels = fixture()
        self.g = VerifiedGraph(self.f.view, self.f.authority, self.f.checks)
        self.q = query(self.f)

    def refused(self, code, fn):
        with self.assertRaises(Refusal) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def next(self, *, observations=(), checks=None, authority=None, limits=None, previous=True):
        view = successor(self.g.graph.view, observations, ContextLimits(),
                         through=self.g.graph.view.through + 1, dropped=self.g.graph.view.dropped)
        return VerifiedGraph(view, self.f.authority if authority is None else authority,
                             self.f.checks if checks is None else checks,
                             self.g.graph.limits if limits is None else limits,
                             previous=self.g if previous else None)

    def rebase(self, s, graph=None, task=None):
        return s.warm_rebase(s.snapshot, s.verified, s.task,
                             self.g if graph is None else graph, self.q if task is None else task)

    def test_full_trajectories_exactness_provenance_constraints_and_caps(self):
        for arity in (4, 8):
            inc = warm = None
            for event in trajectory(arity=arity):
                graph, task = event.verified, event.task
                before = repr((inc, warm, graph))
                if event.name == "work_budget_refusal":
                    for fn in (lambda: predict(graph.graph, task, "B0"),
                               lambda: predict(graph.graph, task, "B1"),
                               lambda: predict_b2(graph.graph, task),
                               lambda: inc.warm(graph, task), lambda: self.rebase(warm, graph, task)):
                        self.assertLessEqual(self.refused("WORK_LIMIT", fn).used, task.work_budget)
                    self.assertEqual(before, repr((inc, warm, graph)))
                    continue
                exact = predict(graph.graph, task, "B0")
                inc, result, meta = B0I.start(graph, task) if inc is None else inc.warm(graph, task)
                self.assertEqual(semantic(result), semantic(exact), event.name)
                self.assertEqual(sum(meta["phase_work"].values()), result.work)
                if event.name == "other_scope":
                    self.assertIsNone(inc)
                    self.assertEqual(meta["mode"], "FULL_RECOMPUTE_FROZEN_B0_SCOPE")
                    self.refused("SCOPE_TRANSITION", lambda: self.rebase(warm, graph, task))
                    # Explicit fresh scope experiment, never a replacement for refusal.
                    _, fresh, _ = B2S.start(graph, task)
                    self.assertEqual(semantic(fresh), semantic(exact))
                    continue
                warm, result, meta = B2S.start(graph, task) if warm is None else self.rebase(warm, graph, task)
                self.assertEqual(semantic(result), semantic(exact), event.name)
                self.assertEqual(sum(meta["phase_work"].values()), result.work)
                self.assertLessEqual(result.work, task.work_budget)
                self.assertLessEqual(warm.reservation, graph.graph.limits.state_bytes)
                self.assertEqual((score(result, event.withheld).false_positive,
                                  score(result, event.withheld).false_negative),
                                 (0, 5) if event.name == "readout_omission" else (0, 0))
                for p in result.predictions:
                    self.assertEqual(p.claim, "SALIENCE_ONLY")
                    self.assertEqual(p.hit, graph.graph.axioms[p.hit.grant.axiom.key])
                    for r in p.relations:
                        self.assertEqual(r, graph.graph.relations[r.relation.key])
                self.assertEqual(result.constraints, exact.constraints)

    def test_usable_retention_through_source_change_without_output_firing(self):
        s, _, _ = B2S.start(self.g, self.q)
        key = self.labels["plan"]
        ref = self.g.graph.axioms[key].grant.axiom.references[0]
        checks = tuple(replace(c, observed_digest=h("stale")) if c.reference == ref else c for c in self.f.checks)
        g = self.next(checks=checks)
        # Choose a deterministic schedule whose only fired cell is not the retained
        # output. This is a mechanical intervention, not selecting a science threshold.
        stable = self.labels["stable"]
        from hashlib import sha256
        from .r2_b2s import SCHEDULE_DOMAIN
        seed = next(seed for seed in range(100) if min(s.snapshot.scores, key=lambda k:
                    (sha256(SCHEDULE_DOMAIN + seed.to_bytes(8, "big") + s.snapshot.step.to_bytes(8, "big")
                            + bytes.fromhex(k)).digest(), k)) != stable)
        q = replace(self.q, steps=1, updates_per_tick=1, schedule_seed=seed)
        new, result, meta = self.rebase(s, g, q)
        self.assertEqual(meta["retained_derived_positive"], 1)
        self.assertEqual(new.snapshot.scores[stable], s.snapshot.scores[stable])
        for label in ("plan", "tail", "cycle-a", "cycle-b"):
            self.assertEqual(new.snapshot.scores[self.labels[label]], 0)
        self.assertEqual({p.hit.grant.axiom.key for p in result.predictions}, {stable})
        self.assertEqual(dict(result.operations)["cell_fire"], 1)

    def test_ring_seed_removal_clears_whole_cycle_exactly(self):
        f = externally_seeded_ring()
        g = VerifiedGraph(f.view, f.authority, f.checks)
        q = query(f)
        self.assertFalse(set(q.seeds) & {k for _, _, k in f.keys})
        inc, _, _ = B0I.start(g, q)
        s, _, _ = B2S.start(g, q)
        self.assertTrue(all(inc.scores.values()))
        q = replace(q, seeds=(), steps=1, updates_per_tick=1)
        inc, r, _ = inc.warm(g, q)
        s, result, meta = self.rebase(s, g, q)
        self.assertEqual(semantic(r), semantic(predict(g.graph, q, "B0")))
        self.assertEqual(result.predictions, ())
        self.assertTrue(all(v == 0 for v in s.snapshot.scores.values()))
        self.assertEqual(meta["mode"], "TOTAL_REINITIALIZATION")
        self.assertEqual(meta["retained"], 0)

    def test_explicit_cyclic_damage_recovery_and_frozen_negative_control(self):
        f = ring()
        g = VerifiedGraph(f.view, f.authority, f.checks)
        q = query(f)
        s, _, _ = B2S.start(g, q)
        saved = repr(s)
        damaged, operations = s.damage(s.snapshot, tuple((k, 1000) for k in s.snapshot.scores))
        self.assertGreater(sum(n for _, n in operations), 0)
        self.refused("TAINTED_STATE", lambda: damaged.observe(damaged.snapshot))
        fixed, result, meta = self.rebase(damaged, g, replace(q, steps=1, updates_per_tick=1))
        self.assertEqual(result.predictions, ())
        self.assertEqual(meta["retained"], 0)
        self.assertFalse(fixed.snapshot.tainted)
        self.assertEqual(repr(s), saved)
        original, initial = B2.start(g.graph, q)
        state = original.damage(initial, tuple((k, 1000) for k in original.cells))
        for _ in range(q.steps):
            state = original.warm(state)
        self.assertEqual(len(original.observe(state).predictions), len(original.cells))
        self.assertEqual(predict(g.graph, q, "B0").predictions, ())
        self.assertEqual(original.observe(original.reset()).predictions, ())

    def test_unissued_snapshot_and_raw_activation_rejected_before_copy(self):
        s, _, _ = B2S.start(self.g, self.q)
        forged = replace(s.snapshot, scores={k: 1000 for k in s.snapshot.scores})
        for state in (forged, replace(s.snapshot), b"\x03\xe8", Snapshot({}, 0)):
            self.refused("UNISSUED_OR_UNBOUND_STATE", lambda: s.warm_rebase(state, self.g, self.q, self.g, self.q))
        with self.assertRaises(TypeError):
            replace(s, snapshot=forged)
        with self.assertRaises(TypeError):
            s.snapshot.scores["bad"] = 1000
        with self.assertRaises(FrozenInstanceError):
            s.snapshot.step = 99
        other = VerifiedGraph(self.f.view, self.f.authority, self.f.checks)
        self.refused("UNISSUED_OR_UNBOUND_STATE", lambda: s.warm_rebase(s.snapshot, other, self.q, self.g, self.q))

    def test_policy_epoch_scope_weight_limits_and_unrelated_refused(self):
        s, _, _ = B2S.start(self.g, self.q)
        changes = (replace(self.f.authority, epoch="other"),
                   replace(self.f.authority, schemas=(replace(self.f.authority.schemas[0], weight=749),)
                           + self.f.authority.schemas[1:]))
        for a in changes:
            self.refused("AUTHORITY_CONTINUITY", lambda: self.next(authority=a))
            independent = self.next(authority=a, previous=False)
            self.refused("AUTHORITY_CONTINUITY", lambda: self.rebase(s, independent))
        g = self.next(limits=replace(self.g.graph.limits, degree=9), previous=False)
        self.refused("AUTHORITY_CONTINUITY", lambda: self.rebase(s, g))
        self.refused("SCOPE_TRANSITION", lambda: self.rebase(s, task=query(self.f, "project-debian")))
        unrelated = VerifiedGraph(self.f.view, self.f.authority, self.f.checks)
        self.refused("PREDECESSOR", lambda: self.rebase(s, unrelated))
        refreshed_same_view = VerifiedGraph(self.f.view, self.f.authority, ())
        self.refused("PREDECESSOR", lambda: self.rebase(s, refreshed_same_view))
        nxt = self.next()
        v = successor(nxt.graph.view, (), ContextLimits(), through=nxt.graph.view.through + 1, dropped=0)
        skipped = VerifiedGraph(v, self.f.authority, self.f.checks, previous=nxt)
        self.refused("PREDECESSOR", lambda: self.rebase(s, skipped))

    def test_malformed_views_counters_and_binding_removal_fail_closed(self):
        self.refused("VIEW_INVALID", lambda: VerifiedGraph(replace(self.f.view, digest=h("tampered")),
                                                           self.f.authority, self.f.checks))
        s, _, _ = B2S.start(self.g, self.q)
        binding = self.labels["policy"]
        self.refused("BINDING_RETENTION", lambda: self.next(observations=(Observation(retract=binding, required=True),)))
        independent = self.next(observations=(Observation(retract=binding, required=True),), previous=False)
        self.refused("BINDING_RETENTION", lambda: self.rebase(s, independent))
        v = successor(self.f.view, (), ContextLimits(), through=0, dropped=0)
        self.refused("PREDECESSOR", lambda: VerifiedGraph(v, self.f.authority, self.f.checks, previous=self.g))
        independent = VerifiedGraph(v, self.f.authority, self.f.checks)
        self.refused("PREDECESSOR", lambda: self.rebase(s, independent))

    def test_invalid_seed_and_binding_visibility_are_not_silent_fallbacks(self):
        inc, _, _ = B0I.start(self.g, self.q)
        s, _, _ = B2S.start(self.g, self.q)
        ref = self.f.view.axiom(self.labels["policy"]).references[0]
        for digest in (None, h("stale")):
            g = self.next(checks=tuple(replace(c, observed_digest=digest) if c.reference == ref else c
                                      for c in self.f.checks))
            self.refused("SOURCE_INVALID", lambda: inc.warm(g, self.q))
            self.refused("SOURCE_INVALID", lambda: self.rebase(s, g))
            q = replace(self.q, seeds=tuple(k for k in self.q.seeds if k != self.labels["policy"]))
            _, r, _ = inc.warm(g, q)
            _, warm, _ = self.rebase(s, g, q)
            self.assertEqual(semantic(r), semantic(predict(g.graph, q, "B0")))
            self.assertEqual(warm.predictions, ())
            self.assertEqual(len(warm.constraints), 1)
            self.assertFalse(warm.constraints[0].source_valid)

    def test_predicate_changes_invalidate_candidate_downstream_and_cycles(self):
        inc, _, _ = B0I.start(self.g, self.q)
        s, _, _ = B2S.start(self.g, self.q)
        for predicates in ((), self.q.predicates[1:], self.q.predicates):
            q = replace(self.q, predicates=predicates)
            inc, r, _ = inc.warm(self.g, q)
            s, warm, _ = self.rebase(s, self.g, q)
            exact = predict(self.g.graph, q, "B0")
            self.assertEqual(semantic(r), semantic(exact))
            self.assertEqual(semantic(warm), semantic(exact))

    def test_all_seed_subsets_match_full_score_closure(self):
        # Cached non-output cells are also checked, so readout cannot hide stale state.
        q = replace(self.q, output_kinds=tuple(sorted({g.kind for g in self.f.authority.axioms})), readout_budget=128)
        inc = None
        for mask in (7, 0, 1, 2, 3, 4, 5, 6, 7, 2, 0, 7):
            task = replace(q, seeds=tuple(k for i, k in enumerate(q.seeds) if mask & (1 << i)))
            inc, result, _ = B0I.start(self.g, task) if inc is None else inc.warm(self.g, task)
            exact = predict(self.g.graph, task, "B0")
            self.assertEqual(semantic(result), semantic(exact))
            expected = {p.hit.grant.axiom.key: p.salience for p in exact.predictions}
            self.assertEqual({k: v for k, v in inc.scores.items() if v}, expected)

    def test_withheld_truth_changes_never_change_any_solver_prediction(self):
        results = [predict(self.g.graph, self.q, x) for x in ("B0", "B1")]
        results += [predict_b2(self.g.graph, self.q), B0I.start(self.g, self.q)[1], B2S.start(self.g, self.q)[1]]
        truth = frozenset(p.hit.grant.axiom.key for p in results[0].predictions)
        for result in results:
            saved = repr(result)
            self.assertNotEqual(score(result, truth), score(result, frozenset((h("different withheld truth"),))))
            self.assertEqual(repr(result), saved)
        self.assertEqual(semantic(B0I.start(self.g, self.q)[1]), semantic(results[0]))

    def test_readout_caps_and_atomic_work_refusals(self):
        inc, _, _ = B0I.start(self.g, self.q)
        s, _, _ = B2S.start(self.g, self.q)
        before = repr((inc, s))
        for budget in (1, 10, 100):
            q = replace(self.q, work_budget=budget)
            for fn in (lambda: inc.warm(self.g, q), lambda: self.rebase(s, task=q)):
                self.assertLessEqual(self.refused("WORK_LIMIT", fn).used, budget)
        for solver in (B0I, B2S):
            self.refused("BINDING_READOUT_LIMIT", lambda: solver.start(self.g, replace(self.q, readout_budget=0)))
            _, result, _ = solver.start(self.g, replace(self.q, readout_budget=1))
            self.assertEqual(result.predictions, ())
            self.assertEqual(result.omitted, 5)
            self.assertEqual(len(result.constraints), 1)
        self.assertEqual(repr((inc, s)), before)

    def test_equal_task_cache_reuse_and_old_snapshot_immutability(self):
        inc, cold, _ = B0I.start(self.g, self.q)
        old = repr(inc)
        _, warm, meta = inc.warm(self.g, replace(self.q))
        self.assertEqual(meta["mode"], "EXACT_CACHE_REUSE")
        self.assertLess(warm.work, predict(self.g.graph, self.q, "B0").work)
        self.assertEqual(semantic(warm), semantic(cold))
        self.assertEqual(repr(inc), old)
        with self.assertRaises(TypeError):
            replace(inc, scores={k: 1000 for k in inc.scores})

    def test_partial_schedule_bound_and_retained_scores_never_exceed_exact(self):
        s = None
        for e in trajectory():
            if e.name in ("work_budget_refusal", "other_scope"):
                continue
            task = replace(e.task, steps=1, updates_per_tick=1, readout_budget=128,
                           output_kinds=("environment", "dependency", "policy", "plan"))
            s, result, _ = B2S.start(e.verified, task) if s is None else self.rebase(s, e.verified, task)
            exact = predict(e.verified.graph, task, "B0")
            expected = {p.hit.grant.axiom.key: p.salience for p in exact.predictions}
            self.assertTrue(all(v <= expected.get(k, 0) for k, v in s.snapshot.scores.items()))
            self.assertEqual(dict(result.operations)["cell_fire"], 1)

    def test_order_permutations_and_order_sensitive_coordinates(self):
        f, _ = fixture(reverse=True)
        self.assertEqual(f.view, self.f.view)
        self.assertEqual(f.authority, self.f.authority)
        g = VerifiedGraph(f.view, f.authority, f.checks)
        self.assertEqual(g.graph.evidence_digest, self.g.graph.evidence_digest)
        for solver in (B0I, B2S):
            self.assertEqual(solver.start(g, self.q)[1], solver.start(self.g, self.q)[1])
            altered = replace(self.q, seeds=tuple(reversed(self.q.seeds)), predicates=tuple(reversed(self.q.predicates)))
            self.assertEqual(solver.start(g, altered)[1].predictions, solver.start(g, self.q)[1].predictions)
        # Frozen sidecars intentionally encode tuple order; do not retrofit a hash registry.
        checks_reordered = VerifiedGraph(f.view, f.authority, tuple(reversed(f.checks)))
        self.assertNotEqual(checks_reordered.graph.evidence_digest, g.graph.evidence_digest)
        self.assertEqual(B0I.start(checks_reordered, self.q)[1].predictions, B0I.start(g, self.q)[1].predictions)

    def test_hash_seed_replay_subprocesses(self):
        script = '''from research.ecsc_hca_r0.r2_trajectories import trajectory
from research.ecsc_hca_r0.r2_b0i import B0I
from research.ecsc_hca_r0.r2_b2s import B2S
from research.ecsc_hca_r0.baselines import Refusal
from hashlib import sha256
digest=sha256(); inc=warm=None
for e in trajectory():
 for method in ('B0I','B2S'):
  try:
   if method=='B0I': inc,r,m=B0I.start(e.verified,e.task) if inc is None else inc.warm(e.verified,e.task)
   else: warm,r,m=B2S.start(e.verified,e.task) if warm is None else warm.warm_rebase(warm.snapshot,warm.verified,warm.task,e.verified,e.task)
   digest.update(repr((e.name,method,r,m)).encode())
  except Refusal as exc: digest.update(repr((e.name,method,exc.code,exc.used)).encode())
print(digest.hexdigest())
'''
        outputs = []
        for seed in ("1", "271828"):
            p = subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=True,
                               env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:."))
            print("R2 hash seed", seed, p.stdout.strip(), flush=True)
            outputs.append(p.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_state_capacity_and_no_core_io_workers_or_old_mutation(self):
        g = VerifiedGraph(self.f.view, self.f.authority, self.f.checks, Limits(state_bytes=1))
        for solver in (B0I, B2S):
            self.refused("STATE_LIMIT", lambda: solver.start(g, self.q))
        saved = repr(self.g)
        with patch.object(builtins, "open", side_effect=AssertionError("I/O")), \
             patch.object(io, "open", side_effect=AssertionError("I/O")), \
             patch.object(os, "open", side_effect=AssertionError("I/O")), \
             patch.object(threading, "Thread", side_effect=AssertionError("worker")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
            s, _, _ = B2S.start(self.g, self.q)
            s, _, _ = self.rebase(s)
            before = repr(s)
            s.observe(s.snapshot)
            self.assertEqual(repr(s), before)
            inc, _, _ = B0I.start(self.g, self.q)
            inc.warm(self.g, self.q)
        self.assertEqual(repr(self.g), saved)


if __name__ == "__main__":
    unittest.main()
