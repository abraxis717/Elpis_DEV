"""Layer C, natively: the projection of one query as a real native HCA graph (native/ECS_C/hca, ABI v1), read out
by B0 (exact closure). Research-only ctypes consumer (tests/boundary/test_ecs_c_hca.py: no canonical use). One-way:
saliences in, SALIENCE_ONLY scores out; nothing is written to the learner or to ECS_C.

Graph of a query: per relation r a seed cell (VALID) and a relation cell, joined by a unary factor of a schema
whose weight is r's salience (relations with salience 0 get no factor); per path a factor (its relation cells ->
path cell) of weight 1000; per alternative a unary factor (path cell -> alternative cell) of weight 1000 per path.
"""
from __future__ import annotations

import ctypes as C
import hashlib

from .ecsc import find_library

MAX_ARITY, MAX_READOUT = 16, 1024
SEED, REL, PATH, ALT = 1, 2, 3, 4
ROLE_SEED, ROLE_REL, ROLE_IN, ROLE_PATH, ROLE_ALT = 1, 2, 10, 20, 30


class Limits(C.Structure):
    _fields_ = [("cells", C.c_uint32), ("factors", C.c_uint32), ("arity", C.c_uint32), ("degree", C.c_uint32),
                ("rounds", C.c_uint32), ("updates_per_tick", C.c_uint32), ("work", C.c_uint64),
                ("state_bytes", C.c_uint64)]


class Cell(C.Structure):
    _fields_ = [("key", C.c_uint8 * 32), ("scope", C.c_uint32), ("kind", C.c_uint32), ("binding", C.c_uint8),
                ("source_state", C.c_uint8), ("reserved", C.c_uint8 * 6)]


class Schema(C.Structure):
    _fields_ = [("relation_type", C.c_uint32), ("arity", C.c_uint32), ("weight", C.c_uint32),
                ("reserved", C.c_uint32), ("roles", C.c_uint32 * MAX_ARITY), ("kinds", C.c_uint32 * MAX_ARITY)]


class Factor(C.Structure):
    _fields_ = [("relation_type", C.c_uint32), ("status", C.c_uint8), ("arity", C.c_uint8),
                ("reserved", C.c_uint8 * 2), ("members", C.c_uint32 * MAX_ARITY)]


class Predicate(C.Structure):
    _fields_ = [("relation_type", C.c_uint32), ("arity", C.c_uint32), ("status_mask", C.c_uint32),
                ("reserved", C.c_uint32), ("roles", C.c_uint32 * MAX_ARITY), ("kinds", C.c_uint32 * MAX_ARITY)]


class Task(C.Structure):
    _fields_ = [("scope", C.c_uint32), ("seed_count", C.c_uint32), ("predicate_count", C.c_uint32),
                ("output_kind_count", C.c_uint32), ("readout_budget", C.c_uint32), ("steps", C.c_uint32),
                ("updates_per_tick", C.c_uint32), ("reserved", C.c_uint32), ("work_budget", C.c_uint64),
                ("schedule_seed", C.c_uint64), ("seeds", C.c_void_p), ("predicates", C.POINTER(Predicate)),
                ("output_kinds", C.POINTER(C.c_uint32))]


class Prediction(C.Structure):
    _fields_ = [("cell", C.c_uint32), ("salience", C.c_uint32), ("evidence_first", C.c_uint32),
                ("evidence_count", C.c_uint32)]


class Result(C.Structure):
    _fields_ = [("prediction_count", C.c_uint32), ("evidence_count", C.c_uint32), ("constraint_count", C.c_uint32),
                ("omitted", C.c_uint32), ("work", C.c_uint64), ("predictions", Prediction * MAX_READOUT),
                ("evidence", C.c_uint32 * MAX_READOUT), ("constraints", C.c_uint32 * MAX_READOUT)]


_LIB = []


def library():
    if not _LIB:
        for kind, size in ((Cell, 48), (Schema, 144), (Factor, 72), (Predicate, 144)):
            if C.sizeof(kind) != size:
                raise RuntimeError(f"HCA layout {kind.__name__}")
        lib = C.CDLL(str(find_library("elpis_ecs_c_hca")))
        vp, u64p = C.c_void_p, C.POINTER(C.c_uint64)
        lib.elpis_hca_limits_default.argtypes = [C.POINTER(Limits)]
        lib.elpis_hca_limits_default.restype = None
        lib.elpis_hca_graph_create.argtypes = [C.POINTER(Limits), C.POINTER(Cell), C.c_size_t, C.POINTER(Schema),
                                               C.c_size_t, C.POINTER(Factor), C.c_size_t, C.c_uint64,
                                               C.POINTER(vp), u64p]
        lib.elpis_hca_graph_destroy.argtypes = [vp]
        lib.elpis_hca_graph_destroy.restype = None
        lib.elpis_hca_predict.argtypes = [vp, C.POINTER(Task), C.c_uint32, C.POINTER(Result), C.POINTER(C.c_uint16),
                                          u64p]
        _LIB.append(lib)
    return _LIB[0]


def _key(*labels) -> bytes:
    return hashlib.sha256("/".join(str(x) for x in ("rlb1-hca",) + labels).encode("utf-8")).digest()


def query_scores(query, saliences, budget: int = 1_000_000) -> list:
    """Native B0 salience of every alternative of `query` (a world.Query) given relation saliences (u16 list)."""
    lib = library()
    rels = sorted({r for alt in query.alternatives for p in alt for r in p})
    cells = []                                                  # (key, kind, label)
    for r in rels:
        cells += [(_key("seed", r), SEED, ("seed", r)), (_key("rel", r), REL, ("rel", r))]
    for a, alt in enumerate(query.alternatives):
        cells.append((_key("alt", a), ALT, ("alt", a)))
        for i, _ in enumerate(alt):
            cells.append((_key("path", a, i), PATH, ("path", a, i)))
    cells.sort(key=lambda c: c[0])
    index = {c[2]: i for i, c in enumerate(cells)}
    schemas, factors = {}, []
    for r in rels:
        if saliences[r] >= 1:
            t = 100 + r
            schemas[t] = (t, (ROLE_SEED, ROLE_REL), (SEED, REL), int(saliences[r]))
            factors.append((t, (index[("seed", r)], index[("rel", r)])))
    for a, alt in enumerate(query.alternatives):
        for i, path in enumerate(alt):
            t = 200 + len(path)
            schemas[t] = (t, tuple(ROLE_IN + k for k in range(len(path))) + (ROLE_PATH,),
                          (REL,) * len(path) + (PATH,), 1000)
            factors.append((t, tuple(index[("rel", r)] for r in path) + (index[("path", a, i)],)))
            schemas[300] = (300, (ROLE_PATH, ROLE_ALT), (PATH, ALT), 1000)
            factors.append((300, (index[("path", a, i)], index[("alt", a)])))
    factors.sort(key=lambda f: (f[0], tuple(cells[m][0] for m in f[1])))
    c_cells = (Cell * len(cells))()
    for i, (key, kind, _) in enumerate(cells):
        c_cells[i].key[:] = key
        c_cells[i].scope, c_cells[i].kind, c_cells[i].source_state = 1, kind, 1
    ordered = sorted(schemas.values())
    c_schemas = (Schema * len(ordered))()
    for i, (t, roles, kinds, w) in enumerate(ordered):
        c_schemas[i].relation_type, c_schemas[i].arity, c_schemas[i].weight = t, len(roles), w
        for k, (ro, ki) in enumerate(zip(roles, kinds)):
            c_schemas[i].roles[k], c_schemas[i].kinds[k] = ro, ki
    c_factors = (Factor * max(1, len(factors)))()
    for i, (t, members) in enumerate(factors):
        c_factors[i].relation_type, c_factors[i].status, c_factors[i].arity = t, 2, len(members)
        for k, m in enumerate(members):
            c_factors[i].members[k] = m
    limits = Limits(64, 64, 8, 16, 8, 256, budget, 65536)
    graph, used = C.c_void_p(), C.c_uint64()
    code = lib.elpis_hca_graph_create(C.byref(limits), c_cells, len(cells), c_schemas, len(ordered), c_factors,
                                      len(factors), budget, C.byref(graph), C.byref(used))
    if code != 0:
        raise RuntimeError(f"HCA_GRAPH_{code}")
    try:
        seed_keys = [cells[index[("seed", r)]][0] for r in rels]
        c_seeds = (C.c_uint8 * 32 * len(seed_keys))()
        for i, k in enumerate(seed_keys):
            c_seeds[i][:] = k
        preds = (Predicate * len(ordered))()
        for i, (t, roles, kinds, _) in enumerate(ordered):
            preds[i].relation_type, preds[i].arity, preds[i].status_mask = t, len(roles), 2
            for k, (ro, ki) in enumerate(zip(roles, kinds)):
                preds[i].roles[k], preds[i].kinds[k] = ro, ki
        outs = (C.c_uint32 * 1)(ALT)
        task = Task(1, len(seed_keys), len(ordered), 1, MAX_READOUT, 4, 1, 0, budget, 0,
                    C.cast(c_seeds, C.c_void_p), preds, outs)
        res, scores = Result(), (C.c_uint16 * len(cells))()
        code = lib.elpis_hca_predict(graph, C.byref(task), 0, C.byref(res), scores, C.byref(used))
        if code != 0:
            raise RuntimeError(f"HCA_PREDICT_{code}")
        return [int(scores[index[("alt", a)]]) for a in range(len(query.alternatives))]
    finally:
        lib.elpis_hca_graph_destroy(graph)
