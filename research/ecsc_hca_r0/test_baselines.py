"""Local DEV mechanics only; no frozen QUAL or native builds."""
from dataclasses import FrozenInstanceError, replace
import builtins
import unittest
from unittest.mock import patch

from elpis.runtime.context import (
    ADMITTED, BINDING, CANDIDATE, ContextLimits, ContextRelation, Observation, successor,
)
from .baselines import AxiomGrant, Baselines, Limits, Refusal, Schema
from .fixtures import coding, delayed, h


class Mechanics(unittest.TestCase):
    def setUp(self):
        self.f = coding()
        self.b = Baselines.build(self.f.view, self.f.authority, self.f.checks)
        self.scope = "project-arch"
        self.members = next(r.members for r in self.f.view.relations
                            if self.b.relations[r.key].scope == self.scope)

    def refusal(self, code, fn):
        before = repr(self.b.view)
        with self.assertRaises(Refusal) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(repr(self.b.view), before)
        self.b.view.verify()
        return caught.exception

    def build(self, f, **kw):
        return Baselines.build(f.view, f.authority, f.checks, **kw)

    def test_exact_lookup_provenance_status_and_binding(self):
        keys = tuple(k for k, hit in self.b.relations.items() if hit.scope == self.scope)
        result = self.b.b0(self.scope, self.members, keys)
        self.assertEqual({x.relation.status for x in result.relations}, {CANDIDATE, ADMITTED})
        self.assertEqual({x.grant.axiom for x in result.axioms},
                         {g.axiom for g in self.f.authority.axioms if g.scope == self.scope})
        self.assertTrue(all(x.source_valid for x in result.axioms + result.relations))
        self.assertEqual(result.coordinate, self.f.view.coordinate)
        binding = self.b.b0(self.scope).axioms
        self.assertEqual(len(binding), 1)
        self.assertEqual(binding[0].grant.axiom.status, BINDING)
        self.assertEqual(binding[0].grant.axiom.references, self.f.view.axiom(self.members[2]).references)

    def test_nary_all_premises_required_for_each_arity(self):
        for arity in (4, 5, 8):
            f = coding(arity=arity)
            b = self.build(f)
            members = next(r.members for r in f.view.relations if b.relations[r.key].scope == self.scope)
            full = b.b1(self.scope, members[:-1], rounds=2)
            self.assertEqual(dict(full.scores)[members[-1]], 750)
            self.assertEqual(sorted(dict(full.factors).values()), [500, 750])
            self.assertEqual({r.relation.status for r in full.relations}, {CANDIDATE, ADMITTED})
            for omitted in range(arity - 1):
                seeds = tuple(m for i, m in enumerate(members[:-1]) if i != omitted)
                partial = b.b1(self.scope, seeds, rounds=3)
                self.assertEqual(dict(partial.scores)[members[-1]], 0)
                self.assertTrue(all(v == 0 for _, v in partial.factors))
            self.assertEqual(sum(len(v) for v in b.incidence.values()), 4 * arity)

    def test_contradictory_scopes_do_not_mix(self):
        foreign = self.f.key("project-debian", "environment")
        self.refusal("SCOPE_MISMATCH", lambda: self.b.b0(self.scope, (foreign,)))
        self.refusal("SCOPE_MISMATCH", lambda: self.b.b1(self.scope, (foreign,)))
        result = self.b.b1(self.scope, self.members[:-1])
        self.assertNotIn(foreign, dict(result.scores))
        self.assertTrue(all(x.grant.scope == self.scope for x in result.axioms))
        for scope in (self.scope, "project-debian"):
            env = self.f.key(scope, "environment")
            hits = self.b.b0(scope, (env,)).axioms
            self.assertIn(self.f.view.axiom(env), {x.grant.axiom for x in hits})
            self.assertTrue(all(x.grant.scope == scope for x in hits))

    def test_role_order_and_cross_scope_edges_refused_even_if_allowlisted(self):
        relation = self.f.view.relations[0]
        for replacement, code in ((tuple(reversed(relation.members)), "ROLE_MISMATCH"),
                                  ((self.f.key("project-debian" if self.b.relations[relation.key].scope == self.scope
                                               else self.scope, "environment"),) + relation.members[1:], "SCOPE_MISMATCH")):
            altered = replace(relation, members=replacement)
            view = successor(self.f.view, (Observation(relation=altered),), ContextLimits(), through=99, dropped=0)
            authority = replace(self.f.authority, relations=self.f.authority.relations + (altered,))
            self.refusal(code, lambda: Baselines.build(view, authority, self.f.checks))

    def test_candidate_cannot_promote_itself(self):
        candidate = next(r for r in self.f.view.relations if r.status == CANDIDATE)
        altered = replace(candidate, status=ADMITTED)
        view = successor(self.f.view, (Observation(relation=altered),), ContextLimits(), through=99, dropped=0)
        self.refusal("RELATION_AUTHORITY", lambda: Baselines.build(view, self.f.authority, self.f.checks))
        result = self.b.b1(self.scope, self.members, rounds=8)
        self.assertEqual({r.relation for r in result.relations},
                         {r for r in self.f.view.relations if self.b.relations[r.key].scope == self.scope})

    def test_tampered_provenance_and_axiom_status_refused(self):
        relation = self.f.view.relations[0]
        view = successor(self.f.view, (Observation(relation=replace(relation, provenance_digest=h("forged"))),),
                         ContextLimits(), through=99, dropped=0)
        self.refusal("RELATION_AUTHORITY", lambda: Baselines.build(view, self.f.authority, self.f.checks))
        axiom = self.f.view.axiom(self.members[0])
        for altered in (replace(axiom, status=BINDING),
                        replace(axiom, references=(replace(axiom.references[0], provenance_digest=h("forged")),))):
            view = successor(self.f.view, (Observation(axiom=altered, required=True),), ContextLimits(), through=99, dropped=0)
            self.refusal("AXIOM_AUTHORITY", lambda: Baselines.build(view, self.f.authority, self.f.checks))

    def test_missing_altered_and_stale_manifest_sources(self):
        ref = self.f.view.axiom(self.members[0]).references[0]
        for observed in (None, h("altered")):
            checks = tuple(replace(c, observed_digest=observed) if c.reference == ref else c for c in self.f.checks)
            b = Baselines.build(self.f.view, self.f.authority, checks)
            result = b.b0(self.scope, (self.members[0],))
            self.assertFalse(next(x for x in result.axioms if x.grant.axiom.key == self.members[0]).source_valid)
            self.refusal("SOURCE_INVALID", lambda: b.b1(self.scope, self.members[:-1]))
            self.assertTrue(all(v == 0 for _, v in b.b1(self.scope, self.members[1:-1]).factors))
        missing = tuple(c for c in self.f.checks if c.reference != ref)
        self.assertFalse(Baselines.build(self.f.view, self.f.authority, missing).axioms[self.members[0]].source_valid)
        checks = tuple(replace(c, reference=replace(ref, corpus_digest=h("v2"))) if c.reference == ref else c
                       for c in self.f.checks)
        self.refusal("SOURCE_AUTHORITY", lambda: Baselines.build(self.f.view, self.f.authority, checks))
        sources = tuple(s for s in self.f.authority.sources if s.reference != ref)
        self.refusal("SOURCE_AUTHORITY", lambda: Baselines.build(self.f.view, replace(self.f.authority, sources=sources), missing))

    def test_invalid_binding_retained_and_blocks_factor(self):
        binding = self.members[2]
        ref = self.f.view.axiom(binding).references[0]
        f = replace(self.f, checks=tuple(c for c in self.f.checks if c.reference != ref))
        b = self.build(f)
        hit, = b.b0(self.scope).axioms
        self.assertEqual(hit.grant.axiom.status, BINDING)
        self.assertFalse(hit.source_valid)
        self.assertTrue(all(v == 0 for _, v in b.b1(self.scope, self.members[:2]).factors))

    def test_source_refresh_has_separate_coordinate_and_keeps_old_snapshot(self):
        invalid = self.build(replace(self.f, checks=()))
        self.assertEqual(invalid.view.coordinate, self.b.view.coordinate)
        self.assertNotEqual(invalid.evidence_digest, self.b.evidence_digest)
        self.assertTrue(all(not x.source_valid for x in invalid.axioms.values()))
        self.assertTrue(all(x.source_valid for x in self.b.axioms.values()))
        self.assertEqual(self.build(self.f).evidence_digest, self.b.evidence_digest)

    def test_two_round_chain_and_weighted_role_messages(self):
        # A candidate factor's output can affect salience downstream; neither
        # this nor a downstream ADMITTED label constitutes a proof of the output.
        original = self.f.view.axiom(self.members[-1])
        output = replace(original, key=h("chain-output"))
        relation = ContextRelation(73, (self.members[-1], self.members[2], output.key), h("chain-proposal"))
        schema = Schema(73, ("plan-input", "policy-input", "plan-output"), ("plan", "policy", "plan"), 800)
        authority = replace(self.f.authority,
                            axioms=self.f.authority.axioms + (AxiomGrant(output, self.scope, "plan"),),
                            relations=self.f.authority.relations + (relation,),
                            schemas=self.f.authority.schemas + (schema,))
        view = successor(self.f.view, (Observation(axiom=output), Observation(relation=relation)),
                         ContextLimits(), through=99, dropped=0)
        b = Baselines.build(view, authority, self.f.checks)
        first = b.b1(self.scope, self.members[:-1])
        second = b.b1(self.scope, self.members[:-1], rounds=2)
        self.assertEqual(dict(first.scores)[output.key], 0)
        self.assertEqual(dict(second.scores)[output.key], 600)
        self.assertEqual(b.relations[relation.key].relation.status, CANDIDATE)
        self.assertIn((relation.key, 2), b.incidence[output.key])

    def test_binding_retention_and_predecessor_atomicity(self):
        binding = self.members[2]
        view = successor(self.f.view, (Observation(retract=binding, required=True),), ContextLimits(), through=99, dropped=0)
        self.refusal("BINDING_RETENTION", lambda: Baselines.build(view, self.f.authority, self.f.checks, previous=self.b))
        self.refusal("PREDECESSOR", lambda: self.build(self.f, previous=self.b))
        old = self.f.view.axiom(binding)
        downgraded = replace(old, status="PROVISIONAL")
        view = successor(self.f.view, (Observation(axiom=downgraded, required=True),),
                         ContextLimits(), through=99, dropped=0)
        authority = replace(self.f.authority, axioms=tuple(replace(g, axiom=downgraded)
                            if g.axiom.key == binding else g for g in self.f.authority.axioms))
        self.refusal("BINDING_RETENTION", lambda: Baselines.build(view, authority, self.f.checks, previous=self.b))
        view = successor(self.f.view, (), ContextLimits(), through=99, dropped=0)
        next_b = Baselines.build(view, self.f.authority, self.f.checks, previous=self.b)
        self.assertEqual(next_b.view.predecessor, self.b.view.digest)
        self.assertEqual(next_b.axioms, self.b.axioms)
        with self.assertRaises(TypeError):
            next_b.axioms[binding] = None
        with self.assertRaises(FrozenInstanceError):
            next_b.view = self.f.view

    def test_delayed_references_bounded_at_all_dev_horizons(self):
        for turns in (8, 32, 128, 512):
            previous = self.b
            for f in delayed(turns=turns, drop_every=3):
                if f.authority != previous.authority:
                    # R1 forbids silent authority evolution, including new source grants.
                    # R0 delay fixtures are independent authority snapshots, not migrations.
                    self.refusal("AUTHORITY_CONTINUITY", lambda: self.build(f, previous=previous))
                    previous = self.build(f)
                else:
                    previous = self.build(f, previous=previous)
                self.assertLessEqual(len(previous.axioms), 9)
                self.assertLessEqual(len(f.authority.sources), 9)
            result = previous.b0(self.scope, self.members)
            self.assertEqual({x.grant.axiom for x in result.axioms},
                             {self.f.view.axiom(k) for k in self.members})
            self.assertEqual(previous.view.dropped, turns // 3)
            self.assertEqual(dict(previous.b1(self.scope, self.members[:-1]).scores)[self.members[-1]], 750)

    def test_capacity_degree_arity_reference_and_work_limits(self):
        for limits, code in ((Limits(axioms=7), "CAPACITY"), (Limits(relations=3), "CAPACITY"),
                             (Limits(arity=3), "ARITY_LIMIT"), (Limits(degree=1), "DEGREE_LIMIT"),
                             (Limits(work=1), "WORK_LIMIT")):
            self.refusal(code, lambda: self.build(self.f, limits=limits))
        self.refusal("ROUND_LIMIT", lambda: self.b.b1(self.scope, (), rounds=9))
        self.refusal("CAPACITY", lambda: self.b.b0(self.scope, self.members * 9))
        for method in (lambda **kw: self.b.b0(self.scope, self.members, **kw),
                       lambda **kw: self.b.b1(self.scope, self.members[:-1], rounds=3, **kw)):
            result = method()
            self.assertEqual(method(budget=result.work), result)
            exc = self.refusal("WORK_LIMIT", lambda: method(budget=result.work - 1))
            self.assertLessEqual(exc.used, result.work - 1)
            self.assertEqual(method(), result)
        work = sum(n for _, n in self.b.build_operations)
        self.assertEqual(self.build(self.f, budget=work).build_operations, self.b.build_operations)
        self.refusal("WORK_LIMIT", lambda: self.build(self.f, budget=work - 1))

    def test_reference_capacity_and_output_source_invalidation(self):
        axiom = self.f.view.axiom(self.members[0])
        extra = self.f.view.axiom(self.members[1]).references[0]
        changed = replace(axiom, references=tuple(sorted(axiom.references + (extra,))))
        view = successor(self.f.view, (Observation(axiom=changed),), ContextLimits(), through=99, dropped=0)
        self.refusal("REFERENCES_LIMIT", lambda: Baselines.build(view, self.f.authority, self.f.checks,
                                                                 Limits(references=1)))
        output_ref = self.f.view.axiom(self.members[-1]).references[0]
        b = self.build(replace(self.f, checks=tuple(c for c in self.f.checks if c.reference != output_ref)))
        result = b.b1(self.scope, self.members[:-1])
        self.assertEqual(dict(result.scores)[self.members[-1]], 0)
        self.assertTrue(all(not x.source_valid for x in result.relations))

    def test_deterministic_seeds_replay_and_work_formula(self):
        self.assertEqual(coding(), coding())
        self.assertNotEqual(coding(1).view.digest, coding(2).view.digest)
        first = self.b.b1(self.scope, self.members[:-1], rounds=3)
        self.assertEqual(first, self.build(coding()).b1(self.scope, tuple(reversed(self.members[:-1])), rounds=3))
        ops = dict(first.operations)
        self.assertEqual(ops["factor_update"], 6)
        self.assertEqual(ops["incidence_visit"], 24)
        self.assertEqual(ops["cell_update"], 6)

    def test_forged_view_duplicate_authority_and_invalid_query(self):
        self.refusal("VIEW_INVALID", lambda: self.build(replace(self.f, view=replace(self.f.view, digest=h("tampered")))))
        duplicate = replace(self.f.authority, axioms=self.f.authority.axioms + self.f.authority.axioms[:1])
        self.refusal("DUPLICATE", lambda: self.build(replace(self.f, authority=duplicate)))
        self.refusal("NOT_FOUND", lambda: self.b.b0(self.scope, (h("absent"),)))
        self.refusal("INVALID_QUERY", lambda: self.b.b0(self.scope, relation_keys=(([], ()),)))
        self.refusal("INVALID_BUDGET", lambda: self.b.b0(self.scope, budget=True))
        self.refusal("INVALID_SCHEMA", lambda: Schema(1, ("same", "same"), ("a", "b")))

    def test_operations_never_open_files(self):
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected I/O")):
            b = self.build(self.f)
            b.b0(self.scope, self.members)
            b.b1(self.scope, self.members[:-1])


if __name__ == "__main__":
    unittest.main()
