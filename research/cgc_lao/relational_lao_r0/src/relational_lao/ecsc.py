"""Layer A: ctypes access to the ECS_C relational view (native/ECS_C/relational, ABI v1) and the memory's relation
key. Research-only consumer: canonical code never uses the view (tests/boundary/test_ecs_c_relational.py), and this
module only ADMITS observations, SYNCS authority tables and READS live evidence. It never writes anything that the
learner inferred.

The synthetic authority (``AuthorityWorld``) builds real ``elpis_ecs_c_entity`` rows exactly as the view's own tests
do (native/ECS_C/relational/tests/relational_test_util.h): an OPERATOR and a RUNTIME founder, then SOURCE rows; every
change bumps the generation; a scope changes only with a version; reclamation removes a row and counts it.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
from pathlib import Path

from . import REPO

OK, INVALID, CAPACITY, STALE_VIEW, UNKNOWN_SOURCE, NOT_SOURCE, LIFECYCLE_CODE, SOURCE_STALE, SCOPE_CONFLICT, \
    BINDING_MISMATCH, CLOCK, LINEAGE, REGRESSION, CORRUPT, NOT_FOUND, NOMEM = range(16)
LIVE, SUSPENDED = 1, 2
OPERATOR, RUNTIME, SOURCE = 1, 2, 3
FOUNDED, ACTIVE, DORMANT, TERMINATED = 1, 2, 3, 4
ASSERT, NEGATE, UNCERTAIN = 1, 2, 3
REL_ABI, MEM_ABI = 1, 1
RELATION_TYPE = 7            # the binary typed relation of RLB1 (roles FROM=1, TO=2)
ROLE_FROM, ROLE_TO = 1, 2
U8x32 = C.c_uint8 * 32


class Digest(C.Structure):
    _fields_ = [("b", U8x32)]


class Entity(C.Structure):
    _fields_ = [("id", U8x32), ("kind", C.c_uint8), ("lifecycle", C.c_uint8), ("reserved", C.c_uint8 * 2),
                ("per_turn_limit", C.c_uint32), ("founding_index", C.c_uint64), ("founded_generation", C.c_uint64),
                ("last_generation", C.c_uint64), ("sender_watermark", C.c_uint64), ("source_version", C.c_uint64),
                ("label", U8x32), ("scope", U8x32), ("source_authority", C.c_uint32), ("reserved2", C.c_uint32)]


class RelConfig(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("entity_capacity", C.c_uint32), ("relation_capacity", C.c_uint32),
                ("contribution_capacity", C.c_uint32), ("byte_budget", C.c_uint64)]


class Authority(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("entity_count", C.c_uint32), ("generation", C.c_uint64),
                ("next_founding_index", C.c_uint64), ("reclaimed_count", C.c_uint64), ("lineage", U8x32),
                ("state_root", U8x32), ("entities", C.POINTER(Entity))]


class Coordinate(C.Structure):
    _fields_ = [("generation", C.c_uint64), ("state_root", U8x32)]


class Work(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("rows", "index_probes", "contributions", "relations", "recomputes", "sorted")]


class SyncResult(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("sources_changed", "contributions_flipped", "contributions_retired",
                                          "relations_recomputed", "relations_released", "noop")] + [("work", Work)]


class Support(C.Structure):
    _fields_ = [("source", U8x32), ("scope", U8x32), ("version", C.c_uint64), ("source_authority", C.c_uint32),
                ("state", C.c_uint32), ("assert_count", C.c_uint32), ("negate_count", C.c_uint32),
                ("uncertain_count", C.c_uint32), ("reserved", C.c_uint32), ("first_tick", C.c_uint64),
                ("last_tick", C.c_uint64), ("first_provenance", Digest), ("last_provenance", Digest)]


class Participant(C.Structure):
    _fields_ = [("role", C.c_uint32), ("ordinal", C.c_uint32), ("entity", Digest)]


class Observation(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("relation_type", C.c_uint32), ("polarity", C.c_uint32),
                ("participant_count", C.c_uint32), ("tick", C.c_uint64), ("source_authority", C.c_uint32),
                ("reserved", C.c_uint32), ("payload", Digest), ("provenance", Digest), ("structure_ref", Digest),
                ("participants", Participant * 8)]


class Slot(C.Structure):
    _fields_ = [(n, Digest) for n in ("key", "payload", "first_provenance", "last_provenance", "structure_ref")] + \
               [(n, C.c_uint32) for n in ("relation_type", "participant_count", "assert_count", "negate_count",
                                          "uncertain_count", "last_polarity", "last_authority", "learn_count")] + \
               [(n, C.c_uint64) for n in ("first_tick", "last_tick", "last_assert_tick", "last_negate_tick")] + \
               [("participants", Participant * 8)]


SIZES = {Entity: 152, Coordinate: 40, Support: 176, Observation: 448, Slot: 544}


class NativeUnavailable(RuntimeError):
    pass


def find_library(stem: str) -> Path:
    """The single built lib<stem>.so under ELPIS_NATIVE_BUILD (default: the repository's build/)."""
    root = Path(os.environ.get("ELPIS_NATIVE_BUILD", str(REPO / "build"))).resolve()
    found = sorted(root.rglob(f"lib{stem}.so")) if root.is_dir() else []
    if len(found) != 1:
        raise NativeUnavailable(f"expected exactly one lib{stem}.so under {root}, found {len(found)}")
    return found[0]


_LIBS: dict = {}


def libraries():
    """(relational view, memory) CDLLs with their prototypes; layouts checked against the C headers."""
    if not _LIBS:
        for kind, size in SIZES.items():
            if C.sizeof(kind) != size:
                raise NativeUnavailable(f"layout of {kind.__name__}: {C.sizeof(kind)} != {size}")
        mem = C.CDLL(str(find_library("elpis_ecs_memory")))
        rel = C.CDLL(str(find_library("elpis_ecs_c_relational")))
        vp, u64p, szp = C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)
        mem.elpis_ecs_mem_relation_key.argtypes = [C.POINTER(Observation), C.POINTER(Digest)]
        mem.elpis_ecs_mem_relation_key.restype = C.c_int
        sig = {
            "elpis_ecs_c_rel_config_default": ([C.POINTER(RelConfig)], None),
            "elpis_ecs_c_rel_required_bytes": ([C.POINTER(RelConfig), u64p], C.c_int),
            "elpis_ecs_c_rel_create": ([C.POINTER(RelConfig), C.POINTER(vp)], C.c_int),
            "elpis_ecs_c_rel_destroy": ([vp], None),
            "elpis_ecs_c_rel_sync": ([vp, C.POINTER(Authority), C.POINTER(SyncResult)], C.c_int),
            "elpis_ecs_c_rel_check": ([vp, U8x32, C.c_uint64, U8x32], C.c_int),
            "elpis_ecs_c_rel_admit": ([vp, C.POINTER(Coordinate), U8x32, C.c_uint64, U8x32, C.POINTER(Observation),
                                       C.POINTER(Digest), C.POINTER(Work)], C.c_int),
            "elpis_ecs_c_rel_relation_read": ([vp, C.POINTER(Coordinate), C.c_void_p, C.POINTER(Digest),
                                               C.POINTER(Slot)], C.c_int),
            "elpis_ecs_c_rel_support_of": ([vp, C.POINTER(Digest), C.POINTER(Support), C.c_size_t, szp], C.c_int),
            "elpis_ecs_c_rel_dependents": ([vp, U8x32, C.POINTER(Digest), C.c_size_t, szp], C.c_int),
            "elpis_ecs_c_rel_identity": ([vp, U8x32], C.c_int),
            "elpis_ecs_c_rel_verify": ([vp, u64p], C.c_int),
            "elpis_ecs_c_rel_reset": ([vp], C.c_int),
        }
        for name, (args, res) in sig.items():
            fn = getattr(rel, name)
            fn.argtypes, fn.restype = args, res
        _LIBS["rel"], _LIBS["mem"] = rel, mem
    return _LIBS["rel"], _LIBS["mem"]


def digest(*parts) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, (bytes, bytearray)) else str(p).encode("utf-8"))
        h.update(b"\0")
    return h.digest()


def _u8(data: bytes) -> U8x32:
    return U8x32(*data)


class AuthorityWorld:
    """A synthetic ECS_C authority table (real elpis_ecs_c_entity rows) obeying the successor law."""

    def __init__(self, lineage: bytes):
        self.lineage, self.generation, self.next_founding, self.reclaimed = lineage, 1, 0, 0
        self.rows: list[Entity] = []
        self.found(OPERATOR, ACTIVE)
        self.found(RUNTIME, ACTIVE)

    def found(self, kind, lifecycle, authority=0, version=0, scope=b"\0" * 32) -> int:
        e = Entity()
        e.founding_index = self.next_founding
        self.next_founding += 1
        e.id = _u8(digest(self.lineage, "entity", e.founding_index))
        e.label = _u8(digest(self.lineage, "label", e.founding_index))
        e.kind, e.lifecycle = kind, lifecycle
        e.founded_generation = e.last_generation = self.generation
        if kind == SOURCE:
            e.per_turn_limit, e.source_version, e.scope, e.source_authority = 16, version, _u8(scope), authority
        self.rows.append(e)
        self.generation += 1
        return len(self.rows) - 1

    def find(self, ident: bytes) -> int | None:
        for i, e in enumerate(self.rows):
            if bytes(e.id) == ident:
                return i
        return None

    def set_version(self, i: int, version: int, scope: bytes | None = None) -> None:
        self.rows[i].source_version = version
        if scope is not None:
            self.rows[i].scope = _u8(scope)
        self.generation += 1

    def set_lifecycle(self, i: int, lifecycle: int) -> None:
        self.rows[i].lifecycle = lifecycle
        self.generation += 1

    def reclaim(self, i: int) -> None:
        del self.rows[i]
        self.reclaimed += 1
        self.generation += 1

    def root(self) -> bytes:
        return digest(self.lineage, "root", self.generation)

    def coordinate(self) -> Coordinate:
        c = Coordinate()
        c.generation, c.state_root = self.generation, _u8(self.root())
        return c

    def table(self):
        rows = (Entity * max(1, len(self.rows)))(*self.rows)
        t = Authority()
        t.abi_version, t.entity_count, t.generation = REL_ABI, len(self.rows), self.generation
        t.next_founding_index, t.reclaimed_count = self.next_founding, self.reclaimed
        t.lineage, t.state_root = _u8(self.lineage), _u8(self.root())
        t.entities = C.cast(rows, C.POINTER(Entity))
        return t, rows

    def binding(self, i: int):
        e = self.rows[i]
        return bytes(e.id), int(e.source_version), bytes(e.scope)


def observation(polarity: int, tick: int, source_authority: int, entity_from: bytes, entity_to: bytes,
                provenance: bytes, structure_ref: bytes = b"\0" * 32) -> Observation:
    o = Observation()
    o.abi_version, o.relation_type, o.polarity, o.participant_count = MEM_ABI, RELATION_TYPE, polarity, 2
    o.tick, o.source_authority = tick, source_authority
    o.provenance.b, o.structure_ref.b = _u8(provenance), _u8(structure_ref)
    o.participants[0].role, o.participants[0].entity.b = ROLE_FROM, _u8(entity_from)
    o.participants[1].role, o.participants[1].entity.b = ROLE_TO, _u8(entity_to)
    return o


def relation_key(entity_from: bytes, entity_to: bytes) -> bytes:
    """The memory's canonical relation key of the binary typed relation entity_from -> entity_to."""
    _, mem = libraries()
    out = Digest()
    code = mem.elpis_ecs_mem_relation_key(C.byref(observation(ASSERT, 0, 0, entity_from, entity_to, b"\0" * 32)),
                                          C.byref(out))
    if code != 0:
        raise ValueError(f"RELATION_KEY_{code}")
    return bytes(out.b)


class RelationalView:
    """One ECS_C relational view (one allocation; destroyed on close)."""

    def __init__(self, entities: int = 64, relations: int = 64, contributions: int = 512):
        self.rel, _ = libraries()
        cfg = RelConfig()
        self.rel.elpis_ecs_c_rel_config_default(C.byref(cfg))
        cfg.entity_capacity, cfg.relation_capacity, cfg.contribution_capacity = entities, relations, contributions
        need = C.c_uint64()
        if self.rel.elpis_ecs_c_rel_required_bytes(C.byref(cfg), C.byref(need)) != OK:
            raise ValueError("REL_CONFIG")
        cfg.byte_budget = need.value
        self.h = C.c_void_p()
        if self.rel.elpis_ecs_c_rel_create(C.byref(cfg), C.byref(self.h)) != OK:
            raise MemoryError("REL_CREATE")
        self.arena_bytes = int(need.value)

    def close(self) -> None:
        if self.h:
            self.rel.elpis_ecs_c_rel_destroy(self.h)
            self.h = C.c_void_p()

    def sync(self, world: AuthorityWorld):
        table, keep = world.table()
        res = SyncResult()
        code = self.rel.elpis_ecs_c_rel_sync(self.h, C.byref(table), C.byref(res))
        del keep
        return code, res

    def check(self, source: bytes, version: int, scope: bytes) -> int:
        return self.rel.elpis_ecs_c_rel_check(self.h, _u8(source), version, _u8(scope))

    def admit(self, world: AuthorityWorld, row: int, obs: Observation):
        source, version, scope = world.binding(row)
        key = Digest()
        code = self.rel.elpis_ecs_c_rel_admit(self.h, C.byref(world.coordinate()), _u8(source), version, _u8(scope),
                                              C.byref(obs), C.byref(key), None)
        return code, bytes(key.b)

    def support_of(self, key: bytes, capacity: int = 16):
        out = (Support * capacity)()
        n = C.c_size_t()
        k = Digest()
        k.b = _u8(key)
        code = self.rel.elpis_ecs_c_rel_support_of(self.h, C.byref(k), out, capacity, C.byref(n))
        return code, [out[i] for i in range(n.value)] if code == OK else []

    def dependents(self, source: bytes, capacity: int = 64):
        out = (Digest * capacity)()
        n = C.c_size_t()
        code = self.rel.elpis_ecs_c_rel_dependents(self.h, _u8(source), out, capacity, C.byref(n))
        return code, [bytes(out[i].b) for i in range(n.value)] if code == OK else []

    def relation_read(self, world: AuthorityWorld, key: bytes):
        k, slot = Digest(), Slot()
        k.b = _u8(key)
        code = self.rel.elpis_ecs_c_rel_relation_read(self.h, C.byref(world.coordinate()), None, C.byref(k),
                                                      C.byref(slot))
        return code, slot

    def identity(self) -> bytes:
        out = U8x32()
        if self.rel.elpis_ecs_c_rel_identity(self.h, out) != OK:
            raise RuntimeError("REL_IDENTITY")
        return bytes(out)

    def verify(self) -> int:
        n = C.c_uint64()
        if self.rel.elpis_ecs_c_rel_verify(self.h, C.byref(n)) != OK:
            raise RuntimeError("REL_VERIFY")
        return int(n.value)

    def reset(self) -> int:
        return self.rel.elpis_ecs_c_rel_reset(self.h)
