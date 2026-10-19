"""The offline differential's bridge: the DEV reference (research/ecsc_hca_r0) translated into the native research
substrate's C ABI (native/ECS_C/hca, elpis/ecs_c_hca.h), and native results translated back.

Test harness only (SCIENTIFIC lane): the reference stays byte-identical and nothing in src/ imports either side.
Labels (scopes, kinds, roles) are interned to positive integers in first-seen order; keys stay the reference's
digests, so canonical orders agree by construction.
"""
from __future__ import annotations

import ctypes as C

from elpis.runtime.context import ADMITTED, BINDING, CANDIDATE

MAX_ARITY, MAX_CELLS, MAX_READOUT = 16, 256, 1024
SOURCE = {"VALID": 1, "AUTHORITY_ONLY": 2, "UNKNOWN": 3, "STALE": 4}
STATUS = {CANDIDATE: 1, ADMITTED: 2}
MASK = {CANDIDATE: 1, ADMITTED: 2}
METHOD = {"B0": 0, "B1": 1, "B2": 2}
# Native codes by the reference's refusal names (baselines.py / task.py / b2.py), where both refuse the same input.
CODES = {0: "OK", 1: "INVALID", 2: "CAPACITY", 3: "WORK_LIMIT", 4: "NOT_FOUND", 5: "SCOPE_MISMATCH",
         6: "SOURCE_INVALID", 7: "ROLE_MISMATCH", 8: "ROLE_SCHEMA", 9: "DEGREE_LIMIT", 10: "ARITY_LIMIT",
         11: "DUPLICATE", 12: "ROUND_LIMIT", 13: "UPDATE_LIMIT", 14: "BINDING_READOUT_LIMIT", 15: "STATE_LIMIT",
         16: "INVALID_STATE", 17: "STEP_LIMIT", 18: "INVALID_DAMAGE"}


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


class Activation(C.Structure):
    _fields_ = [("step", C.c_uint64), ("owner", C.c_uint8 * 32), ("cells", C.c_uint32), ("reserved", C.c_uint32),
                ("values", C.c_uint8 * (2 * MAX_CELLS))]


class Damage(C.Structure):
    _fields_ = [("cell", C.c_uint32), ("value", C.c_uint32)]


def bind(path) -> C.CDLL:
    lib = C.CDLL(str(path))
    sizes = {"cell": (Cell, 48), "schema": (Schema, 144), "factor": (Factor, 72), "predicate": (Predicate, 144),
             "activation": (Activation, 560)}
    for name, (kind, size) in sizes.items():
        assert C.sizeof(kind) == size, name
    u64p, vp = C.POINTER(C.c_uint64), C.c_void_p
    lib.elpis_hca_abi_version.restype = C.c_uint32
    lib.elpis_hca_graph_create.argtypes = [C.POINTER(Limits), C.POINTER(Cell), C.c_size_t, C.POINTER(Schema),
                                           C.c_size_t, C.POINTER(Factor), C.c_size_t, C.c_uint64,
                                           C.POINTER(vp), u64p]
    lib.elpis_hca_graph_destroy.argtypes = [vp]
    lib.elpis_hca_graph_destroy.restype = None
    lib.elpis_hca_graph_evidence.argtypes = [vp, C.c_uint8 * 32]
    lib.elpis_hca_predict.argtypes = [vp, C.POINTER(Task), C.c_uint32, C.POINTER(Result), C.POINTER(C.c_uint16), u64p]
    lib.elpis_hca_b2_start.argtypes = [vp, C.POINTER(Task), C.c_uint64, C.POINTER(vp), C.POINTER(Activation), u64p]
    lib.elpis_hca_b2_warm.argtypes = [vp, C.POINTER(Activation), C.c_uint64, C.POINTER(Activation), u64p]
    lib.elpis_hca_b2_damage.argtypes = [vp, C.POINTER(Activation), C.POINTER(Damage), C.c_size_t, C.c_uint64,
                                        C.POINTER(Activation), u64p]
    lib.elpis_hca_b2_reset.argtypes = [vp, C.c_uint64, C.POINTER(Activation), u64p]
    lib.elpis_hca_b2_observe.argtypes = [vp, C.POINTER(Activation), C.c_uint64, C.POINTER(Result), u64p]
    lib.elpis_hca_b2_destroy.argtypes = [vp]
    lib.elpis_hca_b2_destroy.restype = None
    return lib


class Labels:
    """Positive integer ids for labels, in first-seen order (scopes, kinds and roles share one table)."""

    def __init__(self):
        self.ids = {}

    def __call__(self, label: str) -> int:
        return self.ids.setdefault(label, len(self.ids) + 1)


class NativeGraph:
    """The reference graph (a built ``Baselines``) as a native graph, with the key <-> index maps."""

    def __init__(self, lib, graph, *, budget=None, limits=None):
        self.lib, self.labels = lib, Labels()
        self.keys = list(graph.axioms)                      # the view's order: sorted keys
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.relation_keys = list(graph.relations)          # the view's order: sorted (type, members)
        self.relation_index = {k: i for i, k in enumerate(self.relation_keys)}
        cells = (Cell * max(1, len(self.keys)))()
        for i, (key, hit) in enumerate(graph.axioms.items()):
            cells[i].key[:] = bytes.fromhex(key)
            cells[i].scope = self.labels(hit.grant.scope)
            cells[i].kind = self.labels(hit.grant.kind)
            cells[i].binding = int(hit.grant.axiom.status == BINDING)
            cells[i].source_state = SOURCE[hit.source_state]
        schemas_sorted = sorted(graph.schemas.values(), key=lambda s: s.relation_type)
        schemas = (Schema * max(1, len(schemas_sorted)))()
        for i, s in enumerate(schemas_sorted):
            schemas[i].relation_type, schemas[i].arity, schemas[i].weight = s.relation_type, len(s.roles), s.weight
            for p, (role, kind) in enumerate(zip(s.roles, s.kinds)):
                schemas[i].roles[p], schemas[i].kinds[p] = self.labels(role), self.labels(kind)
        factors = (Factor * max(1, len(self.relation_keys)))()
        for i, hit in enumerate(graph.relations.values()):
            r = hit.relation
            factors[i].relation_type, factors[i].status, factors[i].arity = r.relation_type, STATUS[r.status], len(r.members)
            for p, m in enumerate(r.members):
                factors[i].members[p] = self.index[m]
        lim = limits or graph.limits
        self.limits = Limits(lim.axioms, lim.relations, lim.arity, lim.degree, lim.rounds, lim.updates_per_tick,
                             lim.work, lim.state_bytes)
        handle, used = C.c_void_p(), C.c_uint64()
        rc = lib.elpis_hca_graph_create(C.byref(self.limits), cells, len(self.keys), schemas, len(schemas_sorted),
                                        factors, len(self.relation_keys), budget or lim.work, C.byref(handle),
                                        C.byref(used))
        self.code, self.handle, self.build_work = CODES[rc], handle, used.value

    def close(self):
        if self.handle:
            self.lib.elpis_hca_graph_destroy(self.handle)
            self.handle = C.c_void_p()

    def task(self, task):
        """A reference ``Task`` as the native task (its arrays kept alive on the returned object)."""
        t = Task()
        t.scope = self.labels(task.scope)
        t.seed_count, t.predicate_count, t.output_kind_count = len(task.seeds), len(task.predicates), len(task.output_kinds)
        t.readout_budget, t.steps, t.updates_per_tick = task.readout_budget, task.steps, task.updates_per_tick
        t.work_budget, t.schedule_seed = task.work_budget, task.schedule_seed
        seeds = C.create_string_buffer(b"".join(bytes.fromhex(k) for k in task.seeds), max(1, 32 * len(task.seeds)))
        predicates = (Predicate * max(1, len(task.predicates)))()
        for i, p in enumerate(task.predicates):
            predicates[i].relation_type, predicates[i].arity = p.relation_type, len(p.roles)
            predicates[i].status_mask = sum(MASK[s] for s in set(p.statuses))
            for j, (role, kind) in enumerate(zip(p.roles, p.kinds)):
                predicates[i].roles[j], predicates[i].kinds[j] = self.labels(role), self.labels(kind)
        kinds = (C.c_uint32 * len(task.output_kinds))(*(self.labels(k) for k in task.output_kinds))
        t.seeds = C.cast(seeds, C.c_void_p)
        t.predicates = C.cast(predicates, C.POINTER(Predicate))
        t.output_kinds = C.cast(kinds, C.POINTER(C.c_uint32))
        t._keep = (seeds, predicates, kinds)
        return t

    def result(self, r: Result):
        """A native result as the reference's comparable form: predictions (key, salience, evidence relation
        keys), constraint keys and the omitted count."""
        predictions = []
        for p in r.predictions[:r.prediction_count]:
            evidence = tuple(self.relation_keys[f] for f in r.evidence[p.evidence_first:p.evidence_first + p.evidence_count])
            predictions.append((self.keys[p.cell], p.salience, evidence))
        return (tuple(predictions), tuple(self.keys[c] for c in r.constraints[:r.constraint_count]), r.omitted)

    def predict(self, task, method):
        t, r, used = self.task(task), Result(), C.c_uint64()
        scores = (C.c_uint16 * max(1, len(self.keys)))()
        rc = self.lib.elpis_hca_predict(self.handle, C.byref(t), METHOD[method], C.byref(r), scores, C.byref(used))
        if rc:
            return CODES[rc], None, None
        return "OK", self.result(r), {k: scores[i] for i, k in enumerate(self.keys) if scores[i] != 0xFFFF}


def reference(result):
    """A reference ``PredictionResult`` in the same comparable form."""
    predictions = tuple((p.hit.grant.axiom.key, p.salience, tuple(r.relation.key for r in p.relations))
                        for p in result.predictions)
    return predictions, tuple(c.grant.axiom.key for c in result.constraints), result.omitted
