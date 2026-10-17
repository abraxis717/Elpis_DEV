"""B2 DEV mechanical checks against frozen baseline sources; no QUAL."""
from dataclasses import FrozenInstanceError, replace
import builtins
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from hashlib import sha256

from elpis.runtime.context import ContextLimits, ContextRelation, EMPTY_VIEW, Observation, successor
from .b2 import B2, Packed, footprint, predict_b2
from .baselines import Authority, AxiomGrant, Baselines, Limits, Meter, Refusal, Schema
from .fixtures import Fixture, coding, h
from .task import predict
from .test_task import chain, keys, query


def ring(n=8):
    """Tiny source-valid candidate cycle; deliberately lacks an external seed."""
    f = coding()
    original = f.authority.axioms[0]
    scope = original.scope
    axioms = tuple(replace(original.axiom, key=h(f"ring-{i}")) for i in range(n))
    relations = tuple(ContextRelation(91, (axioms[i].key, axioms[(i+1) % n].key), h("ring-proposal"))
                      for i in range(n))
    schema = Schema(91, ("input", "output"), ("plan", "plan"))
    authority = Authority(tuple(AxiomGrant(a, scope, "plan") for a in axioms),
                          (f.authority.sources[0],), relations, (schema,))
    view = successor(EMPTY_VIEW, tuple(Observation(axiom=a) for a in axioms)
                     + tuple(Observation(relation=r) for r in relations), ContextLimits(), through=1, dropped=0)
    return Fixture(view, authority, (f.checks[0],), tuple((scope, "plan", a.key) for a in axioms))


class B2Mechanics(unittest.TestCase):
    def setUp(self):
        self.f = coding()
        self.b = Baselines.build(self.f.view, self.f.authority, self.f.checks)
        self.q = query(self.f)
        self.target = self.f.key(self.q.scope, "plan")

    def refused(self, code, call):
        with self.assertRaises(Refusal) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_baseline_freeze_is_unchanged(self):
        root = Path(__file__).parent
        manifest = json.loads((root / "BASELINE_R1_FREEZE.json").read_text())
        repo = root.parent.parent
        for name, expected in manifest["revisions"].items():
            self.assertEqual(sha256((repo / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(sha256((repo / "src/elpis/runtime/context.py").read_bytes()).hexdigest(),
                         manifest["canonical_context_sha256"])

    def test_same_query_evidence_scores_and_labels(self):
        b2 = predict_b2(self.b, self.q)
        for method in ("B0", "B1"):
            other = predict(self.b, self.q, method)
            self.assertEqual(b2.predictions, other.predictions)
            self.assertEqual(b2.constraints, other.constraints)
            self.assertEqual(b2.query, other.query)
            self.assertEqual(b2.evidence_digest, other.evidence_digest)
            self.assertEqual(b2.policy_digest, other.policy_digest)
        p, = b2.predictions
        self.assertEqual(p.hit, self.b.axioms[self.target])
        self.assertEqual(p.hit.grant.axiom.status, "PROVISIONAL")
        self.assertEqual({r.relation.status for r in p.relations}, {"CANDIDATE", "ADMITTED"})
        self.assertTrue(all(self.b.relations[r.relation.key] == r for r in p.relations))
        self.assertEqual(p.claim, "SALIENCE_ONLY")

    def test_every_missing_nary_input_and_crossed_scope(self):
        for arity in (4, 5, 8):
            f = coding(arity=arity)
            b = Baselines.build(f.view, f.authority, f.checks)
            for scope in ("project-arch", "project-debian"):
                q = query(f, scope)
                self.assertEqual(keys(predict_b2(b, q)), {f.key(scope, "plan"): 750})
                for i in range(len(q.seeds)):
                    partial = replace(q, seeds=q.seeds[:i] + q.seeds[i+1:])
                    result = predict_b2(b, partial)
                    self.assertEqual(result.predictions, ())
                    self.assertNotIn(f.key(scope, "plan"), repr(result))
                foreign = f.key("project-debian" if scope == "project-arch" else "project-arch", "environment")
                self.refused("SCOPE_MISMATCH", lambda: predict_b2(b, replace(q, seeds=(foreign,))))
        self.assertEqual(predict_b2(self.b, replace(self.q, predicates=())).predictions, ())

    def test_source_invalid_targets_and_bindings_cannot_activate(self):
        for kind in ("plan", "policy"):
            key = self.f.key(self.q.scope, kind)
            ref = self.f.view.axiom(key).references[0]
            for observation, state_label in ((None, "UNKNOWN"), (h("stale"), "STALE")):
                checks = tuple(replace(c, observed_digest=observation) if c.reference == ref else c for c in self.f.checks)
                b = Baselines.build(self.f.view, self.f.authority, checks)
                q = replace(self.q, seeds=tuple(k for k in self.q.seeds if k != key))
                session, state = B2.start(b, q)
                damaged = session.damage(state, ((key, 1000),))
                result = session.observe(session.warm(damaged))
                self.assertEqual(result.predictions, ())
                self.assertEqual(b.axioms[key].source_state, state_label)
                self.assertEqual(len(result.constraints), 1)
                if kind == "policy":
                    self.assertEqual(result.constraints[0].source_state, state_label)
                    self.refused("SOURCE_INVALID", lambda: predict_b2(b, self.q))

    def test_two_step_candidate_dependency_does_not_gain_authority(self):
        f, output = chain()
        b = Baselines.build(f.view, f.authority, f.checks)
        result = predict_b2(b, query(f))
        self.assertEqual(keys(result), {self.target: 750, output: 600})
        downstream = next(p for p in result.predictions if p.hit.grant.axiom.key == output)
        self.assertEqual(downstream.relations[0].relation.status, "ADMITTED")
        self.assertEqual(downstream.hit.grant.axiom.status, "PROVISIONAL")
        self.assertEqual(downstream.claim, "SALIENCE_ONLY")

    def test_frozen_snapshots_perturbation_repair_and_reset(self):
        f, _ = chain()
        b = Baselines.build(f.view, f.authority, f.checks)
        q = query(f)
        session, initial = B2.start(b, q)
        saved = repr((b, initial))
        state = initial
        for _ in range(q.steps):
            state = session.warm(state)
        expected = predict(b, q, "B0")
        self.assertEqual(session.observe(state).predictions, expected.predictions)
        healthy = state
        damaged = session.damage(state, tuple((key, 0) for key in session.cells))
        self.assertNotEqual(damaged.values, state.values)
        # Bindings survive even when their activation is explicitly zeroed.
        self.assertEqual(session.observe(damaged).constraints, expected.constraints)
        for _ in range(q.steps):
            damaged = session.warm(damaged)
        self.assertEqual(session.observe(damaged).predictions, expected.predictions)
        self.assertEqual(state, healthy)
        self.assertEqual(repr((b, initial)), saved)
        reset = session.reset()
        self.assertEqual(reset.values, initial.values)
        self.assertEqual(reset.step, 0)
        self.assertEqual(predict(b, q, "B1").predictions, expected.predictions)
        with self.assertRaises(FrozenInstanceError):
            healthy.values = b""
        with self.assertRaises(TypeError):
            session.factors["new"] = None

    def test_cycle_damage_is_not_a_recovery_claim(self):
        f = ring()
        b = Baselines.build(f.view, f.authority, f.checks)
        q = query(f)
        session, initial = B2.start(b, q)
        self.assertEqual(session.observe(initial).predictions, ())
        state = session.damage(initial, tuple((key, 1000) for key in session.cells))
        for _ in range(q.steps):
            state = session.warm(state)
        # A unit-weight unsupported cycle retains damage: explicit negative control.
        self.assertEqual(len(session.observe(state).predictions), len(session.cells))
        self.assertEqual(predict(b, q, "B0").predictions, ())
        self.assertEqual(predict(b, q, "B1").predictions, ())
        self.assertEqual(session.observe(session.reset()).predictions, ())

    def test_activation_authority_blocks_cross_scope_and_stale_transplants(self):
        session, state = B2.start(self.b, self.q)
        other, _ = B2.start(self.b, query(self.f, "project-debian"))
        self.refused("INVALID_STATE", lambda: other.warm(state))
        b = Baselines.build(self.f.view, self.f.authority, self.f.checks[:-1])
        other, _ = B2.start(b, self.q)
        self.refused("INVALID_STATE", lambda: other.observe(state))
        self.assertEqual(session.observe(state).predictions, ())

    def test_tick_work_readout_caps_and_atomicity(self):
        session, state = B2.start(self.b, self.q)
        saved = repr((session, state))
        warm = session.warm(state)
        self.assertEqual(session.warm(state, budget=warm.work), warm)
        exc = self.refused("WORK_LIMIT", lambda: session.warm(state, budget=warm.work-1))
        self.assertLessEqual(exc.used, warm.work-1)
        observed = session.observe(warm)
        self.assertEqual(session.observe(warm, budget=observed.work), observed)
        self.refused("WORK_LIMIT", lambda: session.observe(warm, budget=observed.work-1))
        self.refused("WORK_LIMIT", lambda: B2.start(self.b, self.q, budget=1))
        self.refused("WORK_LIMIT", lambda: predict_b2(self.b, replace(self.q, work_budget=1)))
        self.refused("WORK_LIMIT", lambda: session.damage(state, ((self.target, 1),), budget=1))
        self.refused("BINDING_READOUT_LIMIT", lambda: predict_b2(self.b, replace(self.q, readout_budget=0)))
        limited = predict_b2(self.b, replace(self.q, readout_budget=1))
        self.assertEqual(limited.predictions, ())
        self.assertEqual(limited.omitted, 1)
        self.assertEqual(repr((session, state)), saved)
        self.refused("INVALID_DAMAGE", lambda: session.damage(state, (("unknown", 1),)))
        self.refused("INVALID_DAMAGE", lambda: session.damage(state, ((self.target, 1001),)))
        self.refused("INVALID_STATE", lambda: session.warm(replace(state, values=b"\xff\xff" * len(session.cells))))
        self.refused("STEP_LIMIT", lambda: session.warm(replace(state, step=2**64-1)))

    def test_degree_state_and_update_boundedness(self):
        f = ring(32)
        b = Baselines.build(f.view, f.authority, f.checks)
        q = query(f, updates_per_tick=2)
        session, state = B2.start(b, q)
        self.assertEqual(len(state.values), 64)
        self.assertLessEqual(footprint((session, state), (b, q)), session.auxiliary_reservation)
        self.assertLessEqual(session.auxiliary_reservation, b.limits.state_bytes)
        warm = session.warm(state)
        self.assertEqual(dict(warm.operations)["cell_fire"], 2)
        self.assertLessEqual(dict(warm.operations)["incidence_visit"], 2 * b.limits.degree)
        self.assertLessEqual(dict(warm.operations).get("input_read", 0), 2 * b.limits.degree * (b.limits.arity-1))
        small = Baselines.build(self.f.view, self.f.authority, self.f.checks, Limits(state_bytes=1))
        self.refused("STATE_LIMIT", lambda: B2.start(small, self.q))
        small = Baselines.build(self.f.view, self.f.authority, self.f.checks, Limits(updates_per_tick=1))
        self.refused("UPDATE_LIMIT", lambda: B2.start(small, self.q))

    def test_schedule_and_asynchronous_order_replay(self):
        q = replace(self.q, updates_per_tick=1)
        session, state = B2.start(self.b, q)
        order = session._schedule(0, Meter(100_000))
        warm = session.warm(state)
        self.assertEqual(dict(warm.operations)["cell_fire"], 1)
        before, after = Packed(state.values, session.offsets), Packed(warm.values, session.offsets)
        self.assertTrue(all(before[k] == after[k] for k in session.cells if k != order[0]))
        other, _ = B2.start(self.b, replace(q, schedule_seed=q.schedule_seed+1))
        self.assertNotEqual(order, other._schedule(0, Meter(100_000)))
        self.assertEqual(session.warm(state), warm)
        script = '''from research.ecsc_hca_r0.fixtures import coding
from research.ecsc_hca_r0.baselines import Baselines
from research.ecsc_hca_r0.b2 import B2, predict_b2
from research.ecsc_hca_r0.test_task import query
f=coding(); b=Baselines.build(f.view,f.authority,f.checks); q=query(f,updates_per_tick=1)
s,a=B2.start(b,q)
for _ in range(q.steps): a=s.warm(a)
print(a.values.hex(), repr(predict_b2(b,q)))
'''
        results = []
        for seed in ("1", "271828"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:.")
            results.append(subprocess.run([sys.executable, "-c", script], env=env, check=True,
                                          capture_output=True, text=True).stdout)
        self.assertEqual(results[0], results[1])

    def test_core_has_no_file_or_worker_operations(self):
        before = repr(self.b)
        with patch.object(builtins, "open", side_effect=AssertionError("I/O")), \
             patch.object(io, "open", side_effect=AssertionError("I/O")), \
             patch.object(os, "open", side_effect=AssertionError("I/O")), \
             patch.object(threading, "Thread", side_effect=AssertionError("worker")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
            session, state = B2.start(self.b, self.q)
            state = session.warm(session.damage(state, ((self.target, 1000),)))
            session.observe(state)
            session.reset()
            predict_b2(self.b, self.q)
        self.assertEqual(repr(self.b), before)
        self.b.view.verify()


if __name__ == "__main__":
    unittest.main()
