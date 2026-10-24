"""Adapter to the REAL reference substrate in Elpis_DEV_HCA.

This imports the actual predecessor objects (research.ecsc_hca_r0.baselines /
task / b2) rather than an imitation, so the differential proves that the
H1D0 port of B0/B1/B2 behaves like the reference on real objects.

Declared encoding (the only way H1D0 provenance fits the reference contract):
  * every role value becomes an axiom cell with a kind = its layer.
  * every factor gets a dedicated provenance cell of kind 'P'.
  * a factor is a 5-ary ContextRelation
        (from,L0) & (scope,S) & (kind,K) & (prov,P) => (to,L1)
    whose provenance cell's source check is VALID / STALE / None according to
    the current status of the factor's source.
  * task seeds are the query's from/scope/kind cells plus the provenance cells
    of every currently admitted factor.
"""
import sys
from pathlib import Path
from hashlib import sha256

# Publication derivative: locate the actual predecessor from this checkout.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if not (_REPO_ROOT / "research/ecsc_hca_r0/baselines.py").is_file():
    raise RuntimeError("H1D0: predecessor reference absent from checkout")
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from elpis.runtime.context import (BINDING, ADMITTED, CANDIDATE, EMPTY_VIEW,
                                   ContextAxiom, ContextLimits, ContextReference,
                                   ContextRelation, Observation, successor)
from research.ecsc_hca_r0.baselines import (Authority, AxiomGrant, Baselines, Limits,
                                           SourceCheck, SourceGrant, Schema)
from research.ecsc_hca_r0.task import Predicate, Task, predict
from research.ecsc_hca_r0.b2 import B2, predict_b2

LAYER_IN = {"R0": "L0", "R1": "L1", "R2": "L2"}
LAYER_OUT = {"R0": "L1", "R1": "L2", "R2": "L3"}
REL_TYPE = {"R0": 101, "R1": 102, "R2": 103}
SCOPE = "h1d0"
CONTENT = sha256(b"elpis.hca.h1d0.reference.source\0").hexdigest()
STALE = sha256(b"elpis.hca.h1d0.reference.stale\0").hexdigest()


def cell_key(token, layer):
    return sha256(("h1d0-cell:" + token + ":" + layer).encode()).hexdigest()


def prefix_state(case, n):
    status, present, src = {}, {}, {}
    for e in case["events"][:n]:
        t = e["type"]
        if t == "SOURCE":
            status[e["source"]] = (e["status"], e["epoch"])
        elif t == "EMIT":
            present[e["factor"]] = True
        elif t == "DROP":
            present[e["factor"]] = False
        else:
            src[e["factor"]] = e["source"]
    return status, present, src


def build_graph(case, n):
    status, present, src = prefix_state(case, n)
    cells = {}
    relations = []
    prov_live = {}
    for f in case["factors"]:
        if f["rel"] not in REL_TYPE:
            continue
        s = src.get(f["id"], f["source"])
        st = status.get(s)
        live = present.get(f["id"], False) and st == ("VALID", f["version"])
        members = [
            cell_key(f["roles"]["from"], LAYER_IN[f["rel"]]),
            cell_key(f["roles"]["scope"], "S"),
            cell_key(f["roles"]["kind"], "K"),
            cell_key(f["id"], "P"),
            cell_key(f["roles"]["to"], LAYER_OUT[f["rel"]]),
        ]
        kinds = (LAYER_IN[f["rel"]], "S", "K", "P", LAYER_OUT[f["rel"]])
        relations.append((f, members, kinds, live))
        for key, kind in zip(members, kinds):
            cells.setdefault(key, kind)
        prov_live[members[3]] = live
    for q in case["queries"]:
        cells.setdefault(cell_key(q["from"], "L0"), "L0")
        cells.setdefault(cell_key(q["to"], "L3"), "L3")
        cells.setdefault(cell_key(q["scope"], "S"), "S")
        cells.setdefault(cell_key(q["kind"], "K"), "K")

    axioms, sources, checks, observations = [], [], [], []
    for key, kind in cells.items():
        ref = ContextReference(sha256(("ref:" + key).encode()).hexdigest(),
                               sha256(b"h1d0-corpus").hexdigest(),
                               sha256(("prov:" + key).encode()).hexdigest())
        axiom = ContextAxiom(key, (ref,), "PROVISIONAL")
        axioms.append(AxiomGrant(axiom, SCOPE, kind))
        sources.append(SourceGrant(ref, SCOPE, CONTENT))
        if kind == "P":
            observed = CONTENT if prov_live[key] else STALE
            checks.append(SourceCheck(ref, SCOPE, observed))
        else:
            checks.append(SourceCheck(ref, SCOPE, CONTENT))
        observations.append(Observation(axiom=axiom, required=False))

    schemas = tuple(Schema(REL_TYPE[r], ("from", "scope", "kind", "prov", "to"),
                           (LAYER_IN[r], "S", "K", "P", LAYER_OUT[r]), 1000)
                    for r in REL_TYPE)
    rel_objs = []
    for f, members, kinds, live in relations:
        rel = ContextRelation(REL_TYPE[f["rel"]], tuple(members),
                              sha256(("rel:" + f["id"]).encode()).hexdigest(), ADMITTED)
        rel_objs.append(rel)
        observations.append(Observation(relation=rel, required=False))

    limits = Limits(axioms=256, relations=512, arity=8, degree=32, references=2,
                   rounds=8, work=10_000_000, sidecar_bytes=4_194_304,
                   state_bytes=1_048_576, updates_per_tick=256)
    view = successor(EMPTY_VIEW, tuple(observations),
                     ContextLimits(capacity=len(observations) + 8, max_axioms=256,
                                   max_relations=512, max_arity=8, max_references=2),
                     through=len(observations), dropped=0)
    return Baselines.build(view, Authority(tuple(axioms), tuple(sources), tuple(rel_objs), schemas),
                           tuple(checks), limits=limits), cells


def task_for(case, query, n, cells, rounds=4, updates=256):
    status, present, src = prefix_state(case, n)
    seeds = [cell_key(query["from"], "L0"), cell_key(query["scope"], "S"), cell_key(query["kind"], "K")]
    for f in case["factors"]:
        if f["rel"] not in REL_TYPE:
            continue
        s = src.get(f["id"], f["source"])
        if present.get(f["id"], False) and status.get(s) == ("VALID", f["version"]):
            seeds.append(cell_key(f["id"], "P"))
    predicates = tuple(Predicate(REL_TYPE[r], ("from", "scope", "kind", "prov", "to"),
                                 (LAYER_IN[r], "S", "K", "P", LAYER_OUT[r])) for r in REL_TYPE)
    return Task(SCOPE, tuple(dict.fromkeys(seeds)), predicates, ("L3",),
                readout_budget=1024, work_budget=1_000_000, steps=rounds,
                schedule_seed=1729, updates_per_tick=updates)


def reference_predict(case, n, query, method, rounds=4):
    graph, cells = build_graph(case, n)
    task = task_for(case, query, n, cells, rounds=rounds)
    if method == "B2":
        result = predict_b2(graph, task)
    else:
        result = predict(graph, task, method)
    target = cell_key(query["to"], "L3")
    return int(any(p.hit.grant.axiom.key == target for p in result.predictions)), result.work


def reference_b2_damage(case, n, query, changes, rounds=4):
    graph, cells = build_graph(case, n)
    task = task_for(case, query, n, cells, rounds=rounds)
    session, state = B2.start(graph, task)
    for _ in range(rounds):
        state = session.warm(state)
    state = session.damage(state, changes)
    state = session.warm(state)
    result = session.observe(state)
    target = cell_key(query["to"], "L3")
    return int(any(p.hit.grant.axiom.key == target for p in result.predictions))
