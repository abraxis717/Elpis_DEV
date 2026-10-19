"""Offline differential: the native HCA research substrate (native/ECS_C/hca, docs/ECS_C_HCA.md) equals the DEV
reference (research/ecsc_hca_r0) on every prediction, salience, evidence set, constraint, omitted count and refusal,
and on every B2 activation of a trajectory (ticks, damage, reset), for the reference's own fixtures and for seeded
random graphs.

SCIENTIFIC lane. The reference is imported, never modified (tests/research/ecsc_hca_r0/test_record.py keeps it
byte-identical). Work accounting differs by design (the native meter counts its own logical units), so every
comparison runs under budgets neither side reaches. Passing grants nothing: it shows the port is faithful, not that
any of the three comparators is useful (DEV/ecsc_hca/README.md: no useful repair advantage is demonstrated).
"""
from __future__ import annotations

import ctypes as C
import random
from dataclasses import replace

import pytest

from elpis.runtime.context import (ADMITTED, BINDING, CANDIDATE, EMPTY_VIEW, ContextAxiom, ContextLimits,
                                   ContextReference, ContextRelation, Observation, successor)
from research.ecsc_hca_r0.b2 import B2, predict_b2
from research.ecsc_hca_r0.baselines import (AxiomGrant, Authority, Baselines, Limits, Refusal, Schema, SourceCheck,
                                            SourceGrant)
from research.ecsc_hca_r0.fixtures import coding, delayed, h
from research.ecsc_hca_r0.task import Predicate, Task, predict
from tests.conftest import require_native_library

from ._native import CODES, Activation, Damage, NativeGraph, Result, bind, reference

WORK = 10_000_000
BIG = Limits(axioms=64, relations=64, arity=8, degree=8, rounds=8, work=WORK, state_bytes=1_048_576,
             updates_per_tick=256)


@pytest.fixture(scope="module")
def lib():
    lib = bind(require_native_library("elpis_ecs_c_hca"))
    assert lib.elpis_hca_abi_version() == 1
    return lib


def query(f, scope="project-arch", **changes):
    """The reference's own query (research/ecsc_hca_r0/test_task.py): every input kind seeded, every schema."""
    seeds = tuple(k for s, kind, k in f.keys if s == scope and kind != "plan")
    predicates = tuple(Predicate(s.relation_type, s.roles, s.kinds) for s in f.authority.schemas)
    return replace(Task(scope, seeds, predicates, ("plan",), work_budget=WORK), **changes)


def run_reference(graph, task, method):
    try:
        result = predict_b2(graph, task) if method == "B2" else predict(graph, task, method)
    except Refusal as exc:
        return exc.code, None
    return "OK", reference(result)


def agree(lib, graph, tasks, methods=("B0", "B1", "B2")):
    native = NativeGraph(lib, graph)
    assert native.code == "OK"
    try:
        compared = 0
        for task in tasks:
            for method in methods:
                code, result = run_reference(graph, task, method)
                native_code, native_result, _ = native.predict(task, method)
                assert (native_code, native_result) == (code, result), (method, task)
                compared += 1
        return compared
    finally:
        native.close()


# -- the reference's fixtures -------------------------------------------------------------------------------------

@pytest.mark.parametrize("arity", [4, 5, 6, 7, 8])
def test_the_coding_fixture_every_omitted_premise_and_both_scopes(lib, arity):
    f = coding(arity=arity)
    graph = Baselines.build(f.view, f.authority, f.checks, BIG)
    tasks = []
    for scope in ("project-arch", "project-debian"):
        q = query(f, scope)
        tasks.append(q)
        tasks += [replace(q, seeds=q.seeds[:i] + q.seeds[i + 1:]) for i in range(len(q.seeds))]   # n-ary premises
        other = f.key("project-debian" if scope == "project-arch" else "project-arch", "environment")
        tasks.append(replace(q, seeds=q.seeds + (other,)))                                       # SCOPE_MISMATCH
        tasks.append(replace(q, seeds=q.seeds + (h("absent"),)))                                 # NOT_FOUND
        tasks.append(replace(q, seeds=()))                                                       # constraints only
        tasks.append(replace(q, readout_budget=0))                                               # BINDING_READOUT_LIMIT
        tasks.append(replace(q, readout_budget=1))
        tasks.append(replace(q, readout_budget=2))
        tasks.append(replace(q, output_kinds=("plan", "policy", "environment")))
        tasks.append(replace(q, predicates=q.predicates[:1]))
        tasks.append(replace(q, predicates=tuple(replace(p, statuses=(ADMITTED,)) for p in q.predicates)))
        tasks.append(replace(q, predicates=tuple(replace(p, statuses=(CANDIDATE,)) for p in q.predicates)))
        tasks.append(replace(q, predicates=tuple(replace(p, roles=p.roles[::-1]) for p in q.predicates)))
        tasks.append(replace(q, steps=1, updates_per_tick=1))
        tasks.append(replace(q, steps=8, updates_per_tick=3, schedule_seed=7))
    assert agree(lib, graph, tasks) == 3 * len(tasks)


def test_a_chain_deeper_than_the_rounds(lib):
    """The reference's chain fixture (research/ecsc_hca_r0/test_task.py `chain`): the plan feeds a second, admitted
    relation. B1 is synchronous, so one round leaves the chained output at zero; B0 closes it exactly."""
    f = coding()
    scope = "project-arch"
    original = f.view.axiom(f.key(scope, "plan"))
    output = replace(original, key=h("chain-output"))
    relation = ContextRelation(73, (original.key, f.key(scope, "policy"), output.key), h("chain"), ADMITTED)
    schema = Schema(73, ("plan-input", "policy-input", "plan-output"), ("plan", "policy", "plan"), 800)
    authority = replace(f.authority, axioms=f.authority.axioms + (AxiomGrant(output, scope, "plan"),),
                        relations=f.authority.relations + (relation,), schemas=f.authority.schemas + (schema,))
    view = successor(f.view, (Observation(axiom=output), Observation(relation=relation)), ContextLimits(),
                     through=99, dropped=0)
    graph = Baselines.build(view, authority, f.checks, BIG)
    q = query(replace(f, authority=authority, view=view), scope)
    tasks = [replace(q, steps=n, updates_per_tick=u) for n in (1, 2, 3) for u in (1, 2, 32)]
    agree(lib, graph, tasks)
    salience = {k: s for k, s, _ in run_reference(graph, replace(q, steps=1), "B1")[1][0]}
    assert h("chain-output") not in salience                       # one round: the chain is not closed
    salience = {k: s for k, s, _ in run_reference(graph, replace(q, steps=1), "B0")[1][0]}
    assert salience[h("chain-output")] == 600                      # B0: 800 * min(750, 1000) / 1000


def test_source_invalidation(lib):
    f = coding()
    stale = tuple(replace(c, observed_digest=h("altered")) if i == 0 else c for i, c in enumerate(f.checks))
    missing = tuple(replace(c, observed_digest=None) if i == 1 else c for i, c in enumerate(f.checks))
    for checks in (stale, missing, f.checks[2:]):
        graph = Baselines.build(f.view, f.authority, checks, BIG)
        tasks = [query(f, s) for s in ("project-arch", "project-debian")]
        tasks += [replace(t, seeds=t.seeds[1:]) for t in tasks]
        tasks += [replace(t, output_kinds=("plan", "environment", "dependency")) for t in tasks]
        agree(lib, graph, tasks)


def test_delayed_distractor_streams(lib):
    for turns, drop in ((8, 0), (32, 3), (128, 5)):
        for f in delayed(turns=turns, drop_every=drop):
            graph = Baselines.build(f.view, f.authority, f.checks, BIG)
            agree(lib, graph, [query(f, "project-arch"), query(f, "noise", seeds=(), output_kinds=("distractor",))])


# -- seeded random graphs ---------------------------------------------------------------------------------------

def random_world(rng: random.Random):
    """A random bounded authorized graph: up to 3 scopes, 5 kinds, 6 schemas (arity 2..6), mixed source states,
    binding constraints with and without references, candidate and admitted relations, cycles allowed."""
    kinds = [f"k{i}" for i in range(rng.randint(2, 4))]
    scopes = [f"s{i}" for i in range(rng.randint(1, 3))]
    axioms, sources, checks, cells = [], [], [], []
    for scope in scopes:
        for n in range(rng.randint(2, 10)):
            kind = rng.choice(kinds)
            tag = f"{rng.random()}:{scope}:{n}"
            refs, binding = [], rng.random() < 0.15
            for r in range(0 if binding and rng.random() < 0.5 else rng.randint(1, 2)):
                ref = ContextReference(h(f"chunk:{tag}:{r}"), h(f"corpus:{scope}"), h(f"who:{tag}:{r}"))
                content = h(f"content:{tag}:{r}")
                refs.append(ref)
                sources.append(SourceGrant(ref, scope, content))
                roll = rng.random()
                if roll < 0.8:
                    checks.append(SourceCheck(ref, scope, content))
                elif roll < 0.9:
                    checks.append(SourceCheck(ref, scope, h("stale")))
                elif roll < 0.95:
                    checks.append(SourceCheck(ref, scope, None))
            axiom = ContextAxiom(h(f"axiom:{tag}"), tuple(sorted(refs)), BINDING if binding else "PROVISIONAL")
            axioms.append(AxiomGrant(axiom, scope, kind))
            cells.append((scope, kind, axiom.key))
    schemas = []
    for t in range(1, rng.randint(2, 6) + 1):
        arity = rng.randint(2, 6)
        roles = tuple(f"r{t}.{i}" for i in range(arity))
        schemas.append(Schema(t, roles, tuple(rng.choice(kinds) for _ in range(arity)), rng.randint(1, 1000)))
    relations, degree, seen = [], {}, set()
    for _ in range(rng.randint(0, 60)):
        schema = rng.choice(schemas)
        scope = rng.choice(scopes)
        members = []
        for kind in schema.kinds:
            pool = [k for s, kk, k in cells if s == scope and kk == kind and k not in members
                    and degree.get(k, 0) < BIG.degree]
            if not pool:
                break
            members.append(rng.choice(pool))
        if len(members) != len(schema.kinds) or (schema.relation_type, tuple(members)) in seen:
            continue
        seen.add((schema.relation_type, tuple(members)))
        for k in members:
            degree[k] = degree.get(k, 0) + 1
        relations.append(ContextRelation(schema.relation_type, tuple(members), h(f"prov:{rng.random()}"),
                                         rng.choice((CANDIDATE, ADMITTED))))
    observations = [Observation(axiom=g.axiom, required=g.axiom.status == BINDING) for g in axioms]
    observations += [Observation(relation=r) for r in relations]
    view = successor(EMPTY_VIEW, tuple(observations), ContextLimits(), through=len(observations), dropped=0)
    authority = Authority(tuple(axioms), tuple(sources), tuple(relations), tuple(schemas))
    return view, authority, tuple(checks), cells, schemas, scopes, kinds


def random_task(rng, cells, schemas, scopes, kinds, graph):
    scope = rng.choice(scopes + ["elsewhere"] if rng.random() < 0.05 else scopes)
    scoped = [k for s, _, k in cells if s == scope]
    valid = [k for k in scoped if graph.axioms[k].source_valid]
    factors = [hit for hit in graph.relations.values() if hit.scope == scope]
    output = tuple(rng.sample(kinds, rng.randint(1, len(kinds))))
    if factors and rng.random() < 0.7:
        # Seed the complete premises of a few factors (sometimes one fewer), so messages propagate and chain.
        chosen = rng.sample(factors, rng.randint(1, min(4, len(factors))))
        seeds = list(dict.fromkeys(m for hit in chosen for m in hit.relation.members[:-1]
                                   if graph.axioms[m].source_valid))
        if seeds and rng.random() < 0.3:
            seeds.pop(rng.randrange(len(seeds)))
        if rng.random() < 0.6:
            output = tuple(dict.fromkeys(graph.axioms[hit.relation.members[-1]].grant.kind for hit in chosen))
    else:
        seeds = rng.sample(valid, rng.randint(0, len(valid))) if valid else []
    roll = rng.random()
    if roll < 0.04 and scoped:
        seeds.append(rng.choice([k for k in scoped if k not in seeds] or scoped))   # possibly SOURCE_INVALID
    elif roll < 0.07:
        seeds.append(h(f"absent:{rng.random()}"))                                     # NOT_FOUND
    elif roll < 0.10:
        others = [k for s, _, k in cells if s != scope]
        if others:
            seeds.append(rng.choice(others))                                         # SCOPE_MISMATCH
    seeds = tuple(dict.fromkeys(seeds))
    predicates = []
    for s in rng.sample(schemas, rng.randint(0, len(schemas))):
        statuses = rng.choice(((CANDIDATE,), (ADMITTED,), (CANDIDATE, ADMITTED), (ADMITTED, CANDIDATE)))
        roles = s.roles if rng.random() < 0.9 else tuple(f"x{i}" for i in range(len(s.roles)))
        predicates.append(Predicate(s.relation_type, roles, s.kinds, statuses))
    return Task(scope, seeds, tuple(dict.fromkeys(predicates)), output,
                readout_budget=rng.choice((0, 1, 2, 3, 5, 8, 16, 64, 1024)), work_budget=WORK,
                steps=rng.randint(1, 8), schedule_seed=rng.randrange(2**64), updates_per_tick=rng.randint(1, 32))


RANDOM_GRAPHS, TASKS_PER_GRAPH = 60, 16


def random_corpus(seed):
    rng = random.Random(1729 + seed)
    view, authority, checks, cells, schemas, scopes, kinds = random_world(rng)
    graph = Baselines.build(view, authority, checks, BIG)
    return graph, [random_task(rng, cells, schemas, scopes, kinds, graph) for _ in range(TASKS_PER_GRAPH)]


@pytest.mark.parametrize("seed", range(RANDOM_GRAPHS))
def test_random_graphs(lib, seed):
    graph, tasks = random_corpus(seed)
    assert agree(lib, graph, tasks) == 3 * TASKS_PER_GRAPH


def test_the_random_corpus_is_not_vacuous():
    """The comparisons above mean something only if the corpus propagates: messages reach unseeded cells with
    sub-maximal salience and direct evidence, and every refusal both sides share occurs."""
    codes, propagated, evidence, partial = set(), 0, 0, 0
    for seed in range(RANDOM_GRAPHS):
        graph, tasks = random_corpus(seed)
        for task in tasks:
            for method in ("B0", "B1", "B2"):
                code, result = run_reference(graph, task, method)
                codes.add(code)
                for key, salience, relations in (result[0] if result else ()):
                    propagated += key not in task.seeds
                    evidence += len(relations)
                    partial += salience < 1000
    assert {"OK", "NOT_FOUND", "SCOPE_MISMATCH", "SOURCE_INVALID", "BINDING_READOUT_LIMIT"} <= codes, codes
    # Measured on this corpus (deterministic): 74 propagated, 269 evidence, 74 partial.
    assert propagated >= 60 and evidence >= 200 and partial >= 60, (propagated, evidence, partial)


# -- B2 trajectories ----------------------------------------------------------------------------------------------

def values(native_activation, n):
    return bytes(native_activation.values[:2 * n])


def b2_agree(lib, graph, task, rng, ticks=6):
    native = NativeGraph(lib, graph)
    assert native.code == "OK"
    t = native.task(task)
    session, state, used = C.c_void_p(), Activation(), C.c_uint64()
    try:
        try:
            ref_session, ref_state = B2.start(graph, task)
        except Refusal as exc:
            rc = lib.elpis_hca_b2_start(native.handle, C.byref(t), 0, C.byref(session), C.byref(state), C.byref(used))
            assert CODES[rc] == exc.code
            return 0
        rc = lib.elpis_hca_b2_start(native.handle, C.byref(t), 0, C.byref(session), C.byref(state), C.byref(used))
        assert rc == 0
        n = len(ref_session.cells)
        assert (state.cells, state.step, values(state, n)) == (n, 0, ref_state.values)
        steps = 0

        def check():
            assert (state.step, values(state, n)) == (ref_state.step, ref_state.values)
            r = Result()
            rc = lib.elpis_hca_b2_observe(session, C.byref(state), 0, C.byref(r), C.byref(used))
            try:
                expected = ("OK", reference(ref_session.observe(ref_state)))
            except Refusal as exc:
                expected = (exc.code, None)
            assert (CODES[rc], native.result(r) if rc == 0 else None) == expected

        for _ in range(ticks):
            nxt = Activation()
            assert lib.elpis_hca_b2_warm(session, C.byref(state), 0, C.byref(nxt), C.byref(used)) == 0
            state, ref_state = nxt, ref_session.warm(ref_state)
            check()
            steps += 1
        # Activation damage: explicit, local, and repaired (or not) by the same ticks on both sides.
        chosen = rng.sample(range(n), rng.randint(0, n))
        changes = tuple((ref_session.cells[i], rng.choice((0, 1000, rng.randint(0, 1000)))) for i in chosen)
        damage = (Damage * max(1, len(changes)))(*((native.index[k], v) for k, v in changes))
        nxt = Activation()
        assert lib.elpis_hca_b2_damage(session, C.byref(state), damage, len(changes), 0, C.byref(nxt),
                                       C.byref(used)) == 0
        state, ref_state = nxt, ref_session.damage(ref_state, changes)
        check()
        for _ in range(3):
            nxt = Activation()
            assert lib.elpis_hca_b2_warm(session, C.byref(state), 0, C.byref(nxt), C.byref(used)) == 0
            state, ref_state = nxt, ref_session.warm(ref_state)
            check()
        nxt = Activation()
        assert lib.elpis_hca_b2_reset(session, 0, C.byref(nxt), C.byref(used)) == 0
        state, ref_state = nxt, ref_session.reset()
        check()
        return steps
    finally:
        if session:
            lib.elpis_hca_b2_destroy(session)
        native.close()


def test_b2_trajectories_on_the_fixture(lib):
    f = coding()
    graph = Baselines.build(f.view, f.authority, f.checks, BIG)
    rng = random.Random(7)
    for scope in ("project-arch", "project-debian"):
        for updates in (1, 2, 4, 32):
            assert b2_agree(lib, graph, query(f, scope, updates_per_tick=updates, schedule_seed=rng.randrange(2**64)),
                            rng) == 6


@pytest.mark.parametrize("seed", range(40))
def test_b2_trajectories_on_random_graphs(lib, seed):
    rng = random.Random(4242 + seed)
    view, authority, checks, cells, schemas, scopes, kinds = random_world(rng)
    graph = Baselines.build(view, authority, checks, BIG)
    for _ in range(4):
        b2_agree(lib, graph, random_task(rng, cells, schemas, scopes, kinds, graph), rng)
