"""Adversarial R1A invariants, before B2 is implemented or observed."""
from dataclasses import fields, replace
import builtins
import io
import os
import unittest
from unittest.mock import patch

from elpis.runtime.context import ADMITTED, BINDING, ContextLimits, ContextRelation, Observation, successor
from .baselines import AxiomGrant, Baselines, Limits, Refusal, Schema, bounded_digest
from .evaluation import score
from .fixtures import coding, h
from .task import Predicate, Task, predict


def query(f, scope="project-arch", **changes):
    # Construction uses only input kinds and public schema; no output key lookup.
    seeds = tuple(k for s, kind, k in f.keys if s == scope and kind != "plan")
    predicates = tuple(Predicate(s.relation_type, s.roles, s.kinds) for s in f.authority.schemas)
    return replace(Task(scope, seeds, predicates, ("plan",)), **changes)


def chain():
    f = coding()
    scope = "project-arch"
    original = f.view.axiom(f.key(scope, "plan"))
    output = replace(original, key=h("chain-output"))
    relation = ContextRelation(73, (original.key, f.key(scope, "policy"), output.key), h("chain"), ADMITTED)
    schema = Schema(73, ("plan-input", "policy-input", "plan-output"), ("plan", "policy", "plan"), 800)
    authority = replace(f.authority, axioms=f.authority.axioms + (AxiomGrant(output, scope, "plan"),),
                        relations=f.authority.relations + (relation,), schemas=f.authority.schemas + (schema,))
    view = successor(f.view, (Observation(axiom=output), Observation(relation=relation)),
                     ContextLimits(), through=99, dropped=0)
    return replace(f, authority=authority, view=view), output.key


def keys(result):
    return {p.hit.grant.axiom.key: p.salience for p in result.predictions}


class TaskMechanics(unittest.TestCase):
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

    def test_envelope_has_no_answer_oracle_and_inspector_is_not_prediction(self):
        self.assertEqual({f.name for f in fields(Task)},
                         {"scope", "seeds", "predicates", "output_kinds", "readout_budget",
                          "work_budget", "steps", "schedule_seed", "updates_per_tick"})
        self.assertNotIn(self.target, repr(self.q))
        inspected = self.b.b1(self.q.scope, ())
        self.assertIn(self.target, {h.grant.axiom.key for h in inspected.axioms})
        self.assertIn(self.target, dict(inspected.scores))
        for method in ("B0", "B1"):
            result = predict(self.b, replace(self.q, seeds=()), method)
            self.assertNotIn(self.target, repr(result))
            self.assertEqual(result.predictions, ())
            self.assertEqual(len(result.constraints), 1)
            self.assertEqual(result.constraints[0].grant.axiom.status, BINDING)

    def test_all_omitted_nary_premises_and_contradictory_scopes(self):
        for arity in (4, 5, 8):
            f = coding(arity=arity)
            b = Baselines.build(f.view, f.authority, f.checks)
            for scope in ("project-arch", "project-debian"):
                q = query(f, scope)
                for method in ("B0", "B1"):
                    self.assertEqual(keys(predict(b, q, method)), {f.key(scope, "plan"): 750})
                    for i in range(len(q.seeds)):
                        partial = replace(q, seeds=q.seeds[:i] + q.seeds[i+1:])
                        self.assertEqual(keys(predict(b, partial, method)), {})
                    other = f.key("project-debian" if scope == "project-arch" else "project-arch", "environment")
                    self.refused("SCOPE_MISMATCH", lambda: predict(b, replace(q, seeds=(other,)), method))

    def test_predicate_filters_type_role_kind_status(self):
        for method in ("B0", "B1"):
            self.assertEqual(keys(predict(self.b, replace(self.q, predicates=()), method)), {})
            p = self.q.predicates[0]
            for bad in (replace(p, relation_type=99), replace(p, roles=tuple(reversed(p.roles))),
                        replace(p, kinds=tuple(reversed(p.kinds)))):
                self.assertEqual(keys(predict(self.b, replace(self.q, predicates=(bad,)), method)), {})
            p = replace(self.q.predicates[1], statuses=(ADMITTED,))
            result = predict(self.b, replace(self.q, predicates=(p,)), method)
            self.assertEqual(keys(result), {self.target: 500})
            self.assertTrue(all(r.relation.status == ADMITTED for r in result.predictions[0].relations))

    def test_external_scoring_does_not_mutate_or_admit(self):
        for method in ("B0", "B1"):
            result = predict(self.b, self.q, method)
            before = repr(result)
            correct = score(result, frozenset((self.target,)))
            wrong = score(result, frozenset((h("withheld-distractor"),)))
            self.assertEqual((correct.true_positive, correct.false_positive, correct.false_negative), (1, 0, 0))
            self.assertEqual((wrong.true_positive, wrong.false_positive, wrong.false_negative), (0, 1, 1))
            self.assertEqual(correct.axiom_statuses, (("PROVISIONAL", 1),))
            self.assertEqual(correct.relation_statuses, (("ADMITTED", 1), ("CANDIDATE", 1)))
            self.assertEqual(result.predictions[0].claim, "SALIENCE_ONLY")
            self.assertEqual(repr(result), before)
            self.assertEqual(predict(self.b, self.q, method), result)

    def test_invalid_output_and_binding_source_states(self):
        for kind in ("plan", "policy"):
            ref = self.f.view.axiom(self.f.key(self.q.scope, kind)).references[0]
            for observed, state in ((None, "UNKNOWN"), (h("changed"), "STALE")):
                checks = tuple(replace(c, observed_digest=observed) if c.reference == ref else c for c in self.f.checks)
                b = Baselines.build(self.f.view, self.f.authority, checks)
                self.assertEqual(b.axioms[self.f.key(self.q.scope, kind)].source_state, state)
                for method in ("B0", "B1"):
                    q = self.q if kind == "plan" else replace(self.q, seeds=self.q.seeds[:-1])
                    result = predict(b, q, method)
                    self.assertEqual(result.predictions, ())
                    self.assertEqual(len(result.constraints), 1)
                    if kind == "policy":
                        self.assertEqual(result.constraints[0].source_state, state)
                        self.refused("SOURCE_INVALID", lambda: predict(b, self.q, method))
        # The scorer also never credits invalid data supplied as a forged result.
        result = predict(self.b, self.q, "B0")
        hit = replace(result.predictions[0].hit, source_valid=False, source_state="UNKNOWN")
        bad = replace(result, predictions=(replace(result.predictions[0], hit=hit),))
        self.assertEqual(score(bad, frozenset((self.target,))).invalid_predictions, 1)
        self.assertEqual(score(bad, frozenset((self.target,))).true_positive, 0)

    def test_empty_reference_binding_is_authority_only(self):
        old = self.f.view.axiom(self.f.key(self.q.scope, "policy"))
        binding = replace(old, references=())
        authority = replace(self.f.authority, axioms=tuple(replace(g, axiom=binding)
                            if g.axiom.key == binding.key else g for g in self.f.authority.axioms))
        view = successor(self.f.view, (Observation(axiom=binding, required=True),), ContextLimits(), through=99, dropped=0)
        b = Baselines.build(view, authority, self.f.checks)
        self.assertEqual(predict(b, self.q, "B0").constraints[0].source_state, "AUTHORITY_ONLY")
        self.assertEqual(b.axioms[binding.key].grant.axiom.references, ())

    def test_chain_candidate_to_admitted_remains_salience(self):
        f, output = chain()
        b = Baselines.build(f.view, f.authority, f.checks)
        q = query(f)
        self.assertNotIn(output, keys(predict(b, replace(q, steps=1), "B1")))
        expected = {self.target: 750, output: 600}
        self.assertEqual(keys(predict(b, replace(q, steps=2), "B1")), expected)
        self.assertEqual(keys(predict(b, replace(q, steps=1), "B0")), expected)
        result = predict(b, q, "B0")
        self.assertTrue(all(p.hit.grant.axiom.status == "PROVISIONAL" and p.claim == "SALIENCE_ONLY"
                            for p in result.predictions))
        self.assertEqual(keys(predict(b, q, "B1")), expected)

    def test_readout_work_budget_atomicity_and_replay(self):
        for method in ("B0", "B1"):
            full = predict(self.b, self.q, method)
            self.refused("BINDING_READOUT_LIMIT", lambda: predict(self.b, replace(self.q, readout_budget=0), method))
            limited = predict(self.b, replace(self.q, readout_budget=1), method)
            self.assertEqual(limited.predictions, ())
            self.assertEqual(limited.omitted, 1)
            self.assertEqual(len(limited.constraints), 1)
            exact = predict(self.b, replace(self.q, work_budget=full.work), method)
            self.assertEqual(exact.operations, full.operations)
            self.assertEqual(exact.predictions, full.predictions)
            exc = self.refused("WORK_LIMIT", lambda: predict(self.b, replace(self.q, work_budget=full.work-1), method))
            self.assertLessEqual(exc.used, full.work-1)
            self.assertEqual(full, predict(self.b, self.q, method))
            self.assertEqual(full.predictions, predict(self.b, replace(self.q, seeds=tuple(reversed(self.q.seeds))), method).predictions)

    def test_policy_successors_refuse_all_changes_and_allow_observation_refresh(self):
        view = successor(self.f.view, (), ContextLimits(), through=99, dropped=0)
        def build(authority=self.f.authority, limits=self.b.limits, checks=self.f.checks):
            return Baselines.build(view, authority, checks, limits, previous=self.b)
        same = build()
        self.assertEqual(same.policy_digest, self.b.policy_digest)
        changed = build(checks=())
        self.assertEqual(changed.policy_digest, same.policy_digest)
        self.assertNotEqual(changed.evidence_digest, same.evidence_digest)
        schema = self.f.authority.schemas[0]
        variants = [replace(self.f.authority, schemas=(replace(schema, weight=749),) + self.f.authority.schemas[1:]),
                    replace(self.f.authority, schemas=(replace(schema, roles=tuple(reversed(schema.roles))),) + self.f.authority.schemas[1:]),
                    replace(self.f.authority, sources=(replace(self.f.authority.sources[0], content_digest=h("new")),) + self.f.authority.sources[1:]),
                    replace(self.f.authority, epoch="TEST_ONLY_R2")]
        for authority in variants:
            self.refused("AUTHORITY_CONTINUITY", lambda: build(authority=authority))
        for limits in (replace(self.b.limits, work=99_999), replace(self.b.limits, degree=9),
                       replace(self.b.limits, sidecar_bytes=200_000)):
            self.refused("AUTHORITY_CONTINUITY", lambda: build(limits=limits))
        # Scope/kind grants changed consistently with schemas still cannot migrate.
        authority = replace(self.f.authority,
                            axioms=tuple(replace(g, kind="other") if g.kind == "environment" else g for g in self.f.authority.axioms),
                            schemas=tuple(replace(s, kinds=("other",) + s.kinds[1:]) for s in self.f.authority.schemas))
        self.refused("AUTHORITY_CONTINUITY", lambda: build(authority=authority))
        authority = replace(self.f.authority, axioms=tuple(replace(g, scope="renamed") if g.scope == self.q.scope else g for g in self.f.authority.axioms),
                            sources=tuple(replace(s, scope="renamed") if s.scope == self.q.scope else s for s in self.f.authority.sources))
        checks = tuple(replace(c, scope="renamed") if c.scope == self.q.scope else c for c in self.f.checks)
        # A renamed binding hits the stricter retention gate first.
        self.refused("BINDING_RETENTION", lambda: build(authority=authority, checks=checks))
        self.assertTrue(all(hit.source_valid for hit in self.b.axioms.values()))

    def test_sidecar_admission_precedes_hash_and_never_assembles_json(self):
        with patch("research.ecsc_hca_r0.baselines.sha256", side_effect=AssertionError("hash before admission")):
            self.refused("SIDECAR_LIMIT", lambda: Baselines.build(self.f.view, self.f.authority, self.f.checks,
                                                                  replace(self.b.limits, sidecar_bytes=100)))
        # Failure to reserve hashing work cannot invoke the hash implementation.
        from .baselines import Meter
        with patch("research.ecsc_hca_r0.baselines.sha256", side_effect=AssertionError("hash before budget")):
            self.refused("WORK_LIMIT", lambda: bounded_digest(("abc",), self.b.limits, Meter(8), b"test"))
        b = Baselines.build(self.f.view, self.f.authority, self.f.checks)
        self.assertEqual(b.evidence_digest, self.b.evidence_digest)
        ops = dict(b.build_operations)
        self.assertEqual(ops["sidecar_preflight_byte"], ops["sidecar_hash_byte"])
        work = sum(n for _, n in b.build_operations)
        self.refused("WORK_LIMIT", lambda: Baselines.build(self.f.view, self.f.authority, self.f.checks, budget=work-1))

    def test_no_io_or_mutation(self):
        before = repr(self.b)
        with patch.object(builtins, "open", side_effect=AssertionError("I/O")), \
             patch.object(io, "open", side_effect=AssertionError("I/O")), \
             patch.object(os, "open", side_effect=AssertionError("I/O")):
            for method in ("B0", "B1"):
                predict(self.b, self.q, method)
        self.assertEqual(repr(self.b), before)
        self.b.view.verify()


if __name__ == "__main__":
    unittest.main()
