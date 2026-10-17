"""R2 DEV admission and dependency invalidation; no state import, I/O or authority."""
from dataclasses import dataclass
from heapq import heappop, heappush
from types import MappingProxyType

from elpis.runtime.context import BINDING
from .baselines import Baselines, Limits, Refusal, require
from .b2 import footprint
from .task import Task, prepare


@dataclass(frozen=True, init=False)
class VerifiedGraph:
    """In-process admission receipt. Construction always calls frozen build.

    No serialization/import API. Hostile Python reflection is not a trust boundary.
    The receipt retains only the current graph, never its predecessor/history.
    """
    graph: Baselines

    def __init__(self, view, authority, checks, limits=Limits(), *, previous=None):
        require(previous is None or type(previous) is VerifiedGraph, "INVALID_GRAPH")
        graph = Baselines.build(view, authority, checks, limits,
                                previous=None if previous is None else previous.graph)
        object.__setattr__(self, "graph", graph)


def meter_for(verified, task):
    require(type(verified) is VerifiedGraph and type(task) is Task, "INVALID_TASK")
    return verified.graph._meter(task.work_budget)


def transition(old, new, old_task, new_task, meter, *, same_scope=True):
    require(type(old) is VerifiedGraph and type(new) is VerifiedGraph
            and type(old_task) is Task and type(new_task) is Task, "INVALID_TRANSITION")
    a, b = old.graph, new.graph
    meter.tick("transition_check")
    if same_scope:
        require(old_task.scope == new_task.scope, "SCOPE_TRANSITION")
    meter.tick("policy_compare", 1 + len(a.authority.axioms) + len(a.authority.sources)
               + len(a.authority.relations) + len(a.authority.schemas)
               + sum(len(r.members) for r in a.authority.relations))
    require(a.authority == b.authority and a.limits == b.limits, "AUTHORITY_CONTINUITY")
    if old is not new:
        require(b.view.predecessor == a.view.digest and b.view.version == a.view.version + 1
                and b.view.through >= a.view.through and b.view.dropped >= a.view.dropped,
                "PREDECESSOR")
    for key, hit in a.axioms.items():
        meter.tick("binding_check")
        if hit.grant.axiom.status == BINDING:
            require(key in b.axioms and b.axioms[key].grant == hit.grant, "BINDING_RETENTION")


def reserve(graph, task, meter):
    # Conservative auxiliary reservation includes two dependency maps and scratch.
    # Graph/Task are borrowed; Python footprint backstop below is not an RSS claim.
    # Include absent grants: old/new union scratch can exceed either active view.
    # Count all authorized factors conservatively, including other scopes.
    meter.tick("memory_preflight", len(graph.authority.axioms) + len(graph.authority.relations))
    n = sum(g.scope == task.scope for g in graph.authority.axioms)
    edges = graph.authority.relations
    size = (8192 + 1536 * n + 512 * len(edges)
            + 128 * sum(len(r.members) for r in edges) + 256 * len(task.predicates))
    require(size <= graph.limits.state_bytes, "STATE_LIMIT")
    return size


def dependencies(scores, factors, meter):
    meter.tick("dependency_cell", len(scores))
    out = {k: [] for k in scores}
    for rk, hit in factors.items():
        for key in hit.relation.members[:-1]:
            meter.tick("dependency_input")
            out[key].append(rk)
    return MappingProxyType({k: tuple(sorted(v)) for k, v in out.items()})


def prepared(verified, task, meter):
    graph = verified.graph
    reservation = reserve(graph, task, meter)
    scores, factors = prepare(graph, task, meter)
    return scores, MappingProxyType(factors), dependencies(scores, factors, meter), reservation


def affected(old_graph, new_graph, old_task, new_task, old_scores, new_scores,
             old_factors, new_factors, old_deps, new_deps, meter, tainted=()):
    """Directed union closure. Every path from a changed input is invalidated."""
    roots = set(tainted)
    meter.tick("taint_read", len(tainted))
    meter.tick("seed_compare", len(old_task.seeds) + len(new_task.seeds))
    roots.update(set(old_task.seeds) ^ set(new_task.seeds))
    for key in sorted(set(old_scores) | set(new_scores)):
        meter.tick("cell_compare")
        if old_graph.axioms.get(key) != new_graph.axioms.get(key):
            roots.add(key)
    for key in sorted(set(old_factors) | set(new_factors)):
        meter.tick("factor_compare")
        if old_factors.get(key) != new_factors.get(key):
            roots.add(key[1][-1])
    queue = sorted(roots)
    meter.tick("invalidate_queue", len(queue))
    while queue:
        key = heappop(queue)
        meter.tick("invalidate_pop")
        for factors, deps in ((old_factors, old_deps), (new_factors, new_deps)):
            for rk in deps.get(key, ()):
                meter.tick("invalidate_edge")
                target = factors[rk].relation.members[-1]
                if target not in roots:
                    roots.add(target)
                    heappush(queue, target)
                    meter.tick("invalidate_push")
    return frozenset(roots)


def carry(old_scores, initial, region, meter):
    retained = retained_positive = reset = 0
    for key in initial:
        if key in old_scores and key not in region:
            meter.tick("activation_copy")
            initial[key] = old_scores[key]
            retained += 1
            retained_positive += int(initial[key] > 0)
        else:
            meter.tick("activation_reset")
            reset += 1
    return retained, retained_positive, reset


def seal_scores(scores, meter):
    meter.tick("activation_seal", len(scores))
    return MappingProxyType(dict(sorted(scores.items())))


def memory_check(value, graph, task, reservation, meter):
    meter.tick("memory_backstop", len(graph.axioms) + len(graph.relations) + 1)
    require(footprint(value, (graph, task)) <= reservation, "STATE_LIMIT")


def semantic(result):
    """Equality channel includes every result field except method-specific counters."""
    from dataclasses import replace
    return replace(result, operations=())
