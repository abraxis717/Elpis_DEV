"""Deterministic DEV harness only. Withheld synthetic answers never enter solvers."""
from dataclasses import dataclass, replace

from elpis.runtime.context import (ContextAxiom, ContextReference, ContextRelation,
                                   ContextLimits, Observation, successor)
from .baselines import AxiomGrant, Schema, SourceCheck, SourceGrant
from .fixtures import coding, h
from .r2_common import VerifiedGraph
from .task import Predicate, Task


def fixture(*, arity=4, reverse=False):
    base = coding(arity=arity)
    scope = "project-arch"
    grants, sources, checks = list(base.authority.axioms), list(base.authority.sources), list(base.checks)
    relations, observations = list(base.authority.relations), []
    labels = dict((kind, key) for s, kind, key in base.keys if s == scope)
    for name in ("stable", "tail", "cycle-a", "cycle-b", "distractor"):
        ref = ContextReference(h("r2-ref:" + name), h("r2-corpus"), h("r2-source-proposal:" + name))
        axiom = ContextAxiom(h("r2-cell:" + name), (ref,))
        grants.append(AxiomGrant(axiom, scope, "plan"))
        sources.append(SourceGrant(ref, scope, h("r2-bytes:" + name)))
        checks.append(SourceCheck(ref, scope, h("r2-bytes:" + name)))
        observations.append(Observation(axiom=axiom))
        labels[name] = axiom.key
    original = next(r for r in relations if r.members[-1] == labels["plan"] and r.relation_type == 71)
    new_relations = [ContextRelation(71, original.members[:-1] + (labels["stable"],), h("r2-stable"))]
    for a, b in (("plan", "tail"), ("plan", "cycle-a"), ("cycle-a", "cycle-b"), ("cycle-b", "cycle-a")):
        new_relations.append(ContextRelation(73, (labels[a], labels[b]), h("r2-edge:" + a + b)))
    relations.extend(new_relations)
    # Only independent assertions are permuted. Prerequisite order is meaningful.
    if reverse:
        observations.reverse()
        new_relations.reverse()
    observations += [Observation(relation=r) for r in new_relations]
    view = successor(base.view, tuple(observations), ContextLimits(),
                     through=base.view.through + len(observations), dropped=0)
    authority = replace(base.authority, axioms=tuple(grants), sources=tuple(sources),
                        relations=tuple(relations), schemas=base.authority.schemas +
                        (Schema(73, ("input", "output"), ("plan", "plan")),))
    return replace(base, view=view, authority=authority, checks=tuple(checks)), labels


def query(f, scope="project-arch", **changes):
    # Public input roles/kinds only. No lookup of a withheld output identity.
    seeds = tuple(g.axiom.key for g in f.authority.axioms if g.scope == scope and g.kind != "plan")
    predicates = tuple(Predicate(s.relation_type, s.roles, s.kinds) for s in f.authority.schemas)
    return replace(Task(scope, seeds, predicates, ("plan",), readout_budget=64), **changes)


def externally_seeded_ring():
    """A distinct public input kind supports a cycle; no output key is a seed."""
    from .test_b2 import ring
    f = ring()
    original = f.authority.axioms[0]
    premise = replace(original.axiom, key=h("r2-external-ring-premise"))
    edge = ContextRelation(92, (premise.key, original.axiom.key), h("r2-external-ring-edge"))
    authority = replace(f.authority,
                        axioms=f.authority.axioms + (AxiomGrant(premise, original.scope, "environment"),),
                        relations=f.authority.relations + (edge,),
                        schemas=f.authority.schemas + (Schema(92, ("premise", "output"), ("environment", "plan")),))
    view = successor(f.view, (Observation(axiom=premise), Observation(relation=edge)),
                     ContextLimits(), through=f.view.through + 1, dropped=0)
    return replace(f, view=view, authority=authority)


@dataclass(frozen=True)
class Event:
    name: str
    verified: VerifiedGraph
    task: Task
    withheld: frozenset  # Harness/scorer only; excluded from every solver call.


def trajectory(*, arity=4, reverse=False):
    f, labels = fixture(arity=arity, reverse=reverse)
    graph = VerifiedGraph(f.view, f.authority, f.checks)
    task = query(f)
    full = frozenset(labels[n] for n in ("plan", "stable", "tail", "cycle-a", "cycle-b"))
    stable = frozenset((labels["stable"],))
    yield Event("cold", graph, task, full)
    yield Event("identical_task", graph, task, full)
    yield Event("seed_removed", graph, replace(task, seeds=task.seeds[1:]), frozenset())
    yield Event("seed_restored", graph, task, full)

    def advance(observations=(), checks=f.checks, drops=0):
        nonlocal graph
        view = successor(graph.graph.view, observations, ContextLimits(),
                         through=graph.graph.view.through + 1,
                         dropped=graph.graph.view.dropped + drops)
        graph = VerifiedGraph(view, f.authority, checks, graph.graph.limits, previous=graph)

    advance()
    yield Event("same_policy_successor", graph, task, full)
    ref = f.view.axiom(labels["plan"]).references[0]
    for name, value, truth in (("source_stale", h("r2-stale"), stable),
                               ("source_valid_again", "VALID", full),
                               ("source_unknown", None, stable),
                               ("source_reverified", "VALID", full)):
        checks = f.checks if value == "VALID" else tuple(
            replace(c, observed_digest=value) if c.reference == ref else c for c in f.checks)
        advance(checks=checks)
        yield Event(name, graph, task, truth)
    advance((Observation(retract=labels["distractor"]),))
    yield Event("disconnected_distractor_removed", graph, task, full)
    advance((Observation(axiom=f.view.axiom(labels["distractor"])),))
    yield Event("disconnected_distractor_restored", graph, task, full)
    advance((Observation(retract=labels["plan"]),))
    yield Event("optional_axiom_and_input_edges_removed", graph, task, stable)
    observations = (Observation(axiom=f.view.axiom(labels["plan"])),) + tuple(
        Observation(relation=r) for r in f.view.relations if labels["plan"] in r.members)
    advance(observations)
    yield Event("optional_axiom_and_edges_restored", graph, task, full)
    # Supported way to remove incident relations while retaining the same axiom.
    advance((Observation(retract=labels["plan"]), Observation(axiom=f.view.axiom(labels["plan"]))))
    yield Event("incident_relations_removed_member_retained", graph, task, stable)
    advance(tuple(Observation(relation=r) for r in f.view.relations if labels["plan"] in r.members))
    yield Event("incident_relations_restored", graph, task, full)
    advance(drops=1)
    yield Event("optional_queue_drop_counter_binding_retained", graph, task, full)
    yield Event("partial_asynchronous_schedule", graph, replace(task, steps=1, updates_per_tick=1), full)
    yield Event("readout_omission", graph, replace(task, readout_budget=1), full)
    yield Event("work_budget_refusal", graph, replace(task, work_budget=1), full)
    yield Event("other_scope", graph, query(f, "project-debian"),
                frozenset((f.key("project-debian", "plan"),)))
