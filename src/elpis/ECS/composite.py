"""Python control plane for the ECS composite retained state (``ecs_composite.h``, docs/ECS_COMPOSITE_STATE.md).

The one ECS retained state is the qualified K1 state ``(W, epoch, H, a)`` together with the ECS memory
(``ecs_memory.h``, docs/ECS_MEMORY.md): one native object owns both, mutates both only inside one composite
transaction and identifies both with one ELPISEC1 digest (never a K1 identity). This module binds it as
``elpis.ECS.k1`` binds K1: every method admits its arguments, makes one native call and packages the result;
nothing here computes ECS mathematics, a memory law, an encoding or an identity.

ADMITTED OPERATIONS ONLY. The memory configuration built here never carries a capability (its ``capabilities``
word is always 0): OBSERVE (exact typed evidence storage) and LOOKUP (an exact read-only scan) are reachable;
memory LEARN and QUERY FAST (unqualified recurrent learning) are not, and the slow-bank channel is not provided.
A composite whose envelope carries learned content this configuration does not admit is refused on restore
(``UNQUALIFIED``). An observation is evidence, never truth, a learning instruction or an authority.

Ownership: a native composite is freed only by an explicit :meth:`CompositeState.close` (no finalizer), so its
owner decides its lifetime, exactly as for a standalone K1 state; a managed runtime keeps the bound state's owner
alive until it has ended the turn. While a runtime holds its lease, every unmanaged mutation here is refused
``LEASED`` before anything is touched; reads stay open. Like ``elpis.ECS.k1`` this module imports only the
standard library and the ECS bindings.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
from struct import unpack_from as _unpack_from
from types import SimpleNamespace

from .k1 import K1Error, _Transition, _bytes_view
from .native import DEFAULT_MAX_ROWS, _P, _U8P, _VP, _admit_rows, _admit_vector

__all__ = ("CompositeConfig", "CompositeError", "CompositeLibrary", "CompositeState", "CompositeTransaction",
           "MemoryAnswer", "MemoryObservation", "MemorySlot", "POLARITY")

_U32, _U64 = C.c_uint32, C.c_uint64
_CODES = {-1: "INVALID", -2: "NONFINITE", -3: "STALE", -4: "BUSY", -5: "CAPACITY", -6: "NOMEM", -7: "CORRUPT",
          -8: "LEASED", -9: "UNQUALIFIED", -10: "REFUSED"}
MEMORY_CODES = {-1: "INVALID", -2: "CAPACITY", -3: "NUMERIC", -4: "CLOCK", -5: "BUSY", -6: "LEASED", -7: "CORRUPT",
                -8: "NOMEM", -9: "NOT_FOUND", -10: "NO_NET_EVIDENCE", -11: "UNQUALIFIED"}
POLARITY = {"ASSERT": 1, "NEGATE": 2, "UNCERTAIN": 3}
_POLARITY_NAMES = {v: k for k, v in POLARITY.items()}
_POLICY = {"REFUSE": 0, "EVICT_LRU": 1}
MAX_PARTICIPANTS = 8
_LOOKUP = 1
_ABI_V1 = 1


class CompositeError(K1Error):
    """A native composite call refused (the committed state is unchanged). ``memory_status`` names the memory's
    own refusal behind ``REFUSED`` (or ``None``)."""

    def __init__(self, code, message, memory_status=None):
        super().__init__(code, message)
        self.memory_status = memory_status


def _refused(rc, what, memory=0):
    rc, memory = int(rc), int(memory)
    status = MEMORY_CODES.get(memory, f"STATUS_{memory}") if memory else None
    detail = f"{what} (memory {status})" if status else what
    return CompositeError(_CODES.get(rc, "INVALID"), detail, status)


# -- the native records (elpis/ecs_memory.h, elpis/ecs_composite.h); sizes asserted against the C ABI -------------

_Digest = C.c_uint8 * 32


class _MemConfig(C.Structure):
    _fields_ = [(n, _U32) for n in ("abi_version", "dim", "banks", "slot_capacity", "codebook_capacity",
                                    "slot_policy", "codebook_policy", "law", "learn_rate_q16", "retain_q16",
                                    "slow_rate_q16", "capabilities")] + [("byte_budget", _U64)]


class _Config(C.Structure):
    _fields_ = [("abi_version", _U32), ("reserved", _U32), ("dim", _U64), ("width", _U64), ("max_rows", _U64),
                ("memory", _MemConfig)]


class _Participant(C.Structure):
    _fields_ = [("role", _U32), ("ordinal", _U32), ("entity", _Digest)]


class _Observation(C.Structure):
    _fields_ = [("abi_version", _U32), ("relation_type", _U32), ("polarity", _U32), ("participant_count", _U32),
                ("tick", _U64), ("source_authority", _U32), ("reserved", _U32), ("payload", _Digest),
                ("provenance", _Digest), ("structure_ref", _Digest),
                ("participants", _Participant * MAX_PARTICIPANTS)]


class _Slot(C.Structure):
    _fields_ = [("key", _Digest), ("payload", _Digest), ("first_provenance", _Digest), ("last_provenance", _Digest),
                ("structure_ref", _Digest)] + [(n, _U32) for n in (
                    "relation_type", "participant_count", "assert_count", "negate_count", "uncertain_count",
                    "last_polarity", "last_authority", "learn_count")] + [(n, _U64) for n in (
                        "first_tick", "last_tick", "last_assert_tick", "last_negate_tick")] + [
                    ("participants", _Participant * MAX_PARTICIPANTS)]


class _Request(C.Structure):
    _fields_ = [(n, _U32) for n in ("abi_version", "channel", "relation_type", "target_role", "target_ordinal",
                                    "cue_count", "candidate_source", "candidate_count")] + [
        ("candidates", C.c_void_p), ("cue", _Participant * MAX_PARTICIPANTS)]


class _Work(C.Structure):
    _fields_ = [(n, _U64) for n in ("index_probes", "slot_scans", "sha_blocks", "mac_ops", "candidates_scored")]


class _Answer(C.Structure):
    _fields_ = [(n, _U32) for n in ("found", "matches", "conflicts", "contradicted", "uncertain", "reserved")] + [
        ("answer", _Digest), ("score", C.c_int64), ("runner_up", C.c_int64), ("signal", C.c_int64),
        ("answer_tick", _U64), ("answer_key", _Digest), ("work", _Work)]


class _Commit(C.Structure):
    _fields_ = [("k1", _Transition)] + [(n, _Digest) for n in (
        "before", "after", "k1_before", "k1_after", "memory_before", "memory_after")] + [
        ("memory_changed", _U32), ("reserved", _U32)]


class _Stats(C.Structure):
    _fields_ = [(n, _U64) for n in ("envelope_bytes", "k1_envelope_bytes", "memory_capacity_bytes",
                                    "memory_state_bytes", "allocated_bytes", "k1_workspace_bytes",
                                    "memory_arena_bytes", "staging_bytes", "k1_epoch", "memory_clock")] + [
        (n, _U32) for n in ("memory_slots_used", "memory_codebook_used", "capabilities", "txn_open", "lease_held",
                            "reserved")] + [
        (n, _U64) for n in ("commits", "aborts", "refusals", "memory_observes", "memory_learns")]


for _record, _size in ((_MemConfig, 56), (_Config, 88), (_Participant, 40), (_Observation, 448), (_Slot, 544),
                       (_Request, 360), (_Answer, 160), (_Commit, 248), (_Stats, 144)):
    assert C.sizeof(_record) == _size, (_record.__name__, C.sizeof(_record))

_DP = C.POINTER(_Digest)
_IP = C.POINTER(C.c_int)
_ABI = {
    "code_name": ([C.c_int], C.c_char_p),
    "envelope_bytes": ([C.POINTER(_Config)], C.c_size_t),
    "create": ([C.POINTER(_Config), _P, C.POINTER(_VP)], C.c_int),
    "restore": ([C.POINTER(_Config), _U8P, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "import_k1": ([C.POINTER(_Config), _U8P, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "destroy": ([_VP], None),
    "state_digest": ([_VP, _U8P], C.c_int),
    "shape": ([_VP, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)], C.c_int),
    "query_identity": ([_VP, C.c_size_t, _P, C.c_size_t, _P, _U8P], C.c_int),
    "memory_query": ([_VP, C.POINTER(_Request), C.POINTER(_Answer), _U8P, _IP], C.c_int),
    "memory_slot_read": ([_VP, _DP, C.POINTER(_Slot), _IP], C.c_int),
    "snapshot_write": ([_VP, _U8P, C.c_size_t], C.c_int),
    "inspect": ([_VP, C.POINTER(_Stats)], C.c_int),
    "lease_of": ([_VP], _U64),
    "leased_txn_begin": ([_VP, _U64, _U8P, C.POINTER(_U64)], C.c_int),
    "leased_txn_observe": ([_VP, _U64, _U64, C.POINTER(_Observation), _DP, _IP], C.c_int),
    "txn_snapshot_write": ([_VP, _U64, _U8P, C.c_size_t], C.c_int),
    "leased_txn_commit": ([_VP, _U64, _U64, C.POINTER(_Commit)], C.c_int),
    "txn_abort": ([_VP, _U64], C.c_int),
    "envelope_identities": ([_U8P, C.c_size_t, _U8P, _U8P, _U8P], C.c_int),
}


def _bind(lib):
    out = {}
    for name, (args, result) in _ABI.items():
        try:
            fn = getattr(lib, "elpis_ecs_composite_" + name)
        except AttributeError as exc:
            raise CompositeError("UNSUPPORTED", "composite symbol elpis_ecs_composite_" + name) from exc
        fn.argtypes, fn.restype = args, result
        out[name] = fn
    try:
        default = lib.elpis_ecs_mem_config_default   # the memory's own defaults (resolved through the composite)
    except AttributeError as exc:
        raise CompositeError("UNSUPPORTED", "memory symbol elpis_ecs_mem_config_default") from exc
    default.argtypes, default.restype = [C.POINTER(_MemConfig)], None
    out["memory_default"] = default
    return SimpleNamespace(**out)


def _digest(value, what):
    if type(value) is not bytes or len(value) != 32:
        raise CompositeError("INVALID", what + ": 32 bytes")
    return _Digest.from_buffer_copy(value)


def _u32(value, what):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise CompositeError("INVALID", what + ": a 32-bit unsigned integer")
    return value


# -- values ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeConfig:
    """A composite's configuration: the K1 shape and row reservation, and the memory's shape. ``max_rows`` and
    ``byte_budget`` are not state; everything else is part of the ELPISEC1 envelope and must match on restore.
    The memory's law words keep the native defaults; no capability is ever set."""

    dim: int
    width: int
    max_rows: int = DEFAULT_MAX_ROWS
    memory_dim: int = 64
    banks: int = 2
    slot_capacity: int = 16
    codebook_capacity: int = 32
    slot_policy: str = "REFUSE"
    codebook_policy: str = "EVICT_LRU"
    byte_budget: int | None = None   # None: the native default

    def __post_init__(self):
        for name in ("dim", "width", "max_rows", "memory_dim", "banks", "slot_capacity", "codebook_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise CompositeError("INVALID", f"{name}: a positive integer")
        if self.slot_policy not in _POLICY or self.codebook_policy not in _POLICY:
            raise CompositeError("INVALID", "slot and codebook policy: REFUSE or EVICT_LRU")
        if self.byte_budget is not None and (type(self.byte_budget) is not int or self.byte_budget <= 0):
            raise CompositeError("INVALID", "byte_budget: a positive integer or None")

    def _native(self, f):
        memory = _MemConfig()
        f.memory_default(C.byref(memory))
        memory.abi_version, memory.dim, memory.banks = _ABI_V1, self.memory_dim, self.banks
        memory.slot_capacity, memory.codebook_capacity = self.slot_capacity, self.codebook_capacity
        memory.slot_policy, memory.codebook_policy = _POLICY[self.slot_policy], _POLICY[self.codebook_policy]
        memory.capabilities = 0   # admitted operations only: never an unqualified capability
        if self.byte_budget is not None:
            memory.byte_budget = self.byte_budget
        return _Config(_ABI_V1, 0, self.dim, self.width, self.max_rows, memory)


@dataclass(frozen=True)
class MemoryObservation:
    """One typed observation for OBSERVE (``elpis_ecs_mem_observation``): a relation of ``relation_type`` over
    ``participants`` (``(role, ordinal, entity)`` triples; entity: 32 bytes), with an explicit ``polarity``
    (``ASSERT``, ``NEGATE`` or ``UNCERTAIN``, supplied by the caller), the caller's logical ``tick`` (never older
    than the memory clock), the ``source_authority`` and ``provenance`` it is recorded with (recorded, never
    interpreted), an optional ``payload`` (part of the relation identity) and ``structure_ref`` (an opaque external
    identity, recorded only). The native memory validates it completely; this is only its typed transport."""

    relation_type: int
    polarity: str
    participants: tuple
    tick: int
    source_authority: int
    provenance: bytes
    payload: bytes = bytes(32)
    structure_ref: bytes = bytes(32)

    def __post_init__(self):
        if self.polarity not in POLARITY:
            raise CompositeError("INVALID", "polarity: ASSERT, NEGATE or UNCERTAIN")
        if type(self.participants) is not tuple or not 1 <= len(self.participants) <= MAX_PARTICIPANTS:
            raise CompositeError("INVALID", f"participants: a tuple of 1..{MAX_PARTICIPANTS}")
        for p in self.participants:
            if type(p) is not tuple or len(p) != 3:
                raise CompositeError("INVALID", "a participant is a (role, ordinal, entity) triple")
            _u32(p[0], "role"), _u32(p[1], "ordinal"), _digest(p[2], "entity")
        _u32(self.relation_type, "relation_type"), _u32(self.source_authority, "source_authority")
        if type(self.tick) is not int or not 0 <= self.tick < 1 << 64:
            raise CompositeError("INVALID", "tick: a 64-bit unsigned integer")
        for name in ("provenance", "payload", "structure_ref"):
            _digest(getattr(self, name), name)

    def _native(self):
        o = _Observation()
        o.abi_version, o.relation_type, o.polarity = _ABI_V1, self.relation_type, POLARITY[self.polarity]
        o.participant_count, o.tick, o.source_authority = len(self.participants), self.tick, self.source_authority
        o.payload, o.provenance, o.structure_ref = (_digest(self.payload, "payload"),
                                                    _digest(self.provenance, "provenance"),
                                                    _digest(self.structure_ref, "structure_ref"))
        for i, (role, ordinal, entity) in enumerate(self.participants):
            o.participants[i] = _Participant(role, ordinal, _digest(entity, "entity"))
        return o

    @classmethod
    def _from_native(cls, o):
        """The exact typed value of a native observation (an adapter's output); refuses a nonzero reserved field and
        an out-of-range polarity or count rather than dropping them."""
        if o.abi_version != _ABI_V1 or o.reserved or o.polarity not in _POLARITY_NAMES or \
                not 1 <= o.participant_count <= MAX_PARTICIPANTS:
            raise CompositeError("INVALID", "not an admissible observation record")
        if any(bytes(C.string_at(C.addressof(o.participants[i]), C.sizeof(_Participant)))
               != bytes(C.sizeof(_Participant)) for i in range(o.participant_count, MAX_PARTICIPANTS)):
            raise CompositeError("INVALID", "participants past the count must be zero")
        return cls(int(o.relation_type), _POLARITY_NAMES[o.polarity],
                   tuple((int(p.role), int(p.ordinal), bytes(p.entity)) for p in o.participants[:o.participant_count]),
                   int(o.tick), int(o.source_authority), bytes(o.provenance), bytes(o.payload), bytes(o.structure_ref))


@dataclass(frozen=True)
class MemorySlot:
    """A resident evidence slot, read back exactly (participants canonical: sorted by role, ordinal, entity)."""

    key: bytes
    relation_type: int
    participants: tuple
    payload: bytes
    first_provenance: bytes
    last_provenance: bytes
    structure_ref: bytes
    assert_count: int
    negate_count: int
    uncertain_count: int
    last_polarity: str
    last_authority: int
    learn_count: int
    first_tick: int
    last_tick: int
    record: object = field(default=None, compare=False, repr=False)   # the native record (an adapter inverts it)


@dataclass(frozen=True)
class MemoryAnswer:
    """An exact LOOKUP answer, bound to the composite identity of the state that computed it. ``found`` is False
    when nothing, or more than one filler with net evidence, matches: a conflict is reported, never resolved."""

    found: bool
    answer: bytes | None
    matches: int
    conflicts: int
    contradicted: int
    uncertain: int
    score: int
    answer_key: bytes | None
    state_digest: bytes


# -- the library and the state ---------------------------------------------------------------------------------------

class CompositeLibrary:
    """Typed view of one loaded ``libelpis_ecs_composite`` (composite ABI v1)."""

    __slots__ = ("_c", "_lib")

    def __init__(self, lib):
        self._c = _bind(lib)
        self._lib = lib  # the loaded library itself: its entry points for native callers (RuntimeCore)

    def envelope_bytes(self, config):
        """The fixed ELPISEC1 size of ``config`` (refused ``CAPACITY`` beyond the declared bounds)."""
        if type(config) is not CompositeConfig:
            raise CompositeError("INVALID", "a CompositeConfig is required")
        n = int(self._c.envelope_bytes(C.byref(config._native(self._c))))
        if n == 0:
            raise CompositeError("CAPACITY", "configuration beyond the composite's declared bounds")
        return n

    def envelope_identities(self, envelope):
        """``(composite, k1, memory)`` identities an ELPISEC1 envelope records (structure and trailer verified, not
        its components). Pure."""
        c, k, m = _Digest(), _Digest(), _Digest()
        rc = self._c.envelope_identities(_bytes_view(envelope, "envelope"), len(envelope), c, k, m)
        if rc != 0:
            raise _refused(rc, "composite envelope")
        return bytes(c), bytes(k), bytes(m)


class CompositeState:
    """One native ECS composite retained state (K1 + memory under one ELPISEC1 identity). One writer XOR many readers
    (an overlapping call is ``BUSY``)."""

    __slots__ = ("_c", "_lib", "_handle", "_config", "_dim", "_width")

    def __init__(self, library, handle, config):
        self._c, self._lib, self._handle, self._config = library._c, library._lib, handle, config
        dim, width = C.c_size_t(), C.c_size_t()
        rc = self._c.shape(handle, C.byref(dim), C.byref(width))
        if rc != 0:
            self._c.destroy(handle)
            raise _refused(rc, "composite shape")
        self._dim, self._width = int(dim.value), int(width.value)

    @staticmethod
    def _admit(library, config):
        if type(library) is not CompositeLibrary:
            raise CompositeError("INVALID", "a loaded CompositeLibrary is required")
        if type(config) is not CompositeConfig:
            raise CompositeError("INVALID", "a CompositeConfig is required")
        return config._native(library._c)

    @classmethod
    def create(cls, library, config, initial_w):
        """A fresh composite: K1 from W (H = 0, a = 0) and an empty memory."""
        native = cls._admit(library, config)
        w = _admit_vector(initial_w, config.dim * config.width, "initial W")
        handle = _VP()
        rc = library._c.create(C.byref(native), w, C.byref(handle))
        if rc != 0:
            raise _refused(rc, "composite create")
        return cls(library, handle, config)

    @classmethod
    def restore(cls, library, config, envelope):
        """Exactly the composite an ELPISEC1 envelope holds, validated completely before anything is returned (the
        configuration must match; unadmitted learned memory content is ``UNQUALIFIED``; a bare ELPISGK1 envelope is
        ``CORRUPT``: it becomes a composite only through :meth:`import_k1`)."""
        native = cls._admit(library, config)
        handle = _VP()
        rc = library._c.restore(C.byref(native), _bytes_view(envelope, "envelope"), len(envelope), C.byref(handle))
        if rc != 0:
            raise _refused(rc, "composite restore")
        return cls(library, handle, config)

    @classmethod
    def import_k1(cls, library, config, k1_envelope):
        """LEGACY, explicit: a K1-only ELPISGK1 retained state as a composite with an empty memory and a new
        (composite) identity; the K1 component's bytes are preserved exactly."""
        native = cls._admit(library, config)
        handle = _VP()
        rc = library._c.import_k1(C.byref(native), _bytes_view(k1_envelope, "K1 envelope"), len(k1_envelope),
                                  C.byref(handle))
        if rc != 0:
            raise _refused(rc, "composite K1 import")
        return cls(library, handle, config)

    def _live(self):
        if not self._handle:
            raise CompositeError("INVALID", "composite closed")
        return self._handle

    @property
    def dim(self):
        return self._dim

    @property
    def width(self):
        return self._width

    @property
    def config(self):
        return self._config

    @property
    def managed_lease(self):
        """The lease of the managed owner (RuntimeCore) holding this composite, 0 when unmanaged."""
        return int(self._c.lease_of(self._live()))

    def state_digest(self):
        """The composite retained-state identity (the ELPISEC1 trailer of the committed state)."""
        out = _Digest()
        rc = self._c.state_digest(self._live(), out)
        if rc != 0:
            raise _refused(rc, "composite state digest")
        return bytes(out)

    def query_identity(self, x_rows):
        """K1 QUERY ``f_W(X)`` and the composite identity of the same committed state (one read-guarded call)."""
        x, rows = _admit_rows(x_rows, self._dim, "X")
        out, digest = (C.c_double * rows)(), _Digest()
        rc = self._c.query_identity(self._live(), self._dim, x, rows, out, digest)
        if rc != 0:
            raise _refused(rc, "composite query")
        return _unpack_from(f"{rows}d", out), bytes(digest)

    def lookup(self, relation_type, target_role, cues, *, target_ordinal=0):
        """LOOKUP (read-only exact scan of resident evidence): the filler of ``(target_role, target_ordinal)`` in
        relations of ``relation_type`` matching ``cues`` (``(role, ordinal, entity)`` triples), bound to the
        composite identity of the state that answered. Never learns, never changes recency."""
        if type(cues) is not tuple or len(cues) > MAX_PARTICIPANTS - 1:
            raise CompositeError("INVALID", f"cues: a tuple of at most {MAX_PARTICIPANTS - 1}")
        q = _Request()
        q.abi_version, q.channel = _ABI_V1, _LOOKUP
        q.relation_type, q.target_role = _u32(relation_type, "relation_type"), _u32(target_role, "target_role")
        q.target_ordinal, q.cue_count = _u32(target_ordinal, "target_ordinal"), len(cues)
        for i, cue in enumerate(cues):
            if type(cue) is not tuple or len(cue) != 3:
                raise CompositeError("INVALID", "a cue is a (role, ordinal, entity) triple")
            q.cue[i] = _Participant(_u32(cue[0], "role"), _u32(cue[1], "ordinal"), _digest(cue[2], "entity"))
        answer, digest, memory = _Answer(), _Digest(), C.c_int(0)
        rc = self._c.memory_query(self._live(), C.byref(q), C.byref(answer), digest, C.byref(memory))
        if rc != 0:
            raise _refused(rc, "composite lookup", memory.value)
        return MemoryAnswer(bool(answer.found), bytes(answer.answer) if answer.found else None, int(answer.matches),
                            int(answer.conflicts), int(answer.contradicted), int(answer.uncertain), int(answer.score),
                            bytes(answer.answer_key) if answer.found else None, bytes(digest))

    def slot(self, key):
        """The resident evidence slot of relation ``key`` exactly as recorded, or None when no slot holds it."""
        out, memory = _Slot(), C.c_int(0)
        rc = self._c.memory_slot_read(self._live(), C.byref(_digest(key, "relation key")), C.byref(out),
                                      C.byref(memory))
        if rc != 0:
            if memory.value == -9:   # NOT_FOUND
                return None
            raise _refused(rc, "composite slot read", memory.value)
        return MemorySlot(bytes(out.key), int(out.relation_type),
                          tuple((int(p.role), int(p.ordinal), bytes(p.entity))
                                for p in out.participants[:out.participant_count]),
                          bytes(out.payload), bytes(out.first_provenance), bytes(out.last_provenance),
                          bytes(out.structure_ref), int(out.assert_count), int(out.negate_count),
                          int(out.uncertain_count), _POLARITY_NAMES.get(out.last_polarity, str(out.last_polarity)),
                          int(out.last_authority), int(out.learn_count), int(out.first_tick), int(out.last_tick),
                          out)

    def snapshot(self):
        """The committed ELPISEC1 envelope (fixed size for the configuration; its trailer is the identity)."""
        size = int(self.inspect()["envelope_bytes"])
        out = (C.c_uint8 * size)()
        rc = self._c.snapshot_write(self._live(), out, size)
        if rc != 0:
            raise _refused(rc, "composite snapshot")
        return bytes(out)

    def inspect(self):
        """Resource figures and telemetry (read-only): envelope and arena sizes, allocated bytes, clocks, slot use,
        the admitted capabilities (always 0 here) and non-authoritative counters."""
        s = _Stats()
        rc = self._c.inspect(self._live(), C.byref(s))
        if rc != 0:
            raise _refused(rc, "composite inspect")
        return {name: int(getattr(s, name)) for name, _ in _Stats._fields_ if name != "reserved"}

    def transaction(self):
        """An UNMANAGED composite transaction (lease 0): refused ``LEASED`` while a runtime holds the composite."""
        source, token = _Digest(), _U64()
        rc = self._c.leased_txn_begin(self._live(), 0, source, C.byref(token))
        if rc != 0:
            raise _refused(rc, "composite transaction begin")
        return CompositeTransaction(self, token.value, bytes(source))

    def close(self):
        """Free the native composite (aborting any open unmanaged transaction). Never while a runtime is bound to
        it: its owner keeps it alive until the runtime has released it or closed."""
        if self._handle:
            self._c.destroy(self._handle)
            self._handle = _VP()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CompositeTransaction:
    """An open unmanaged composite transaction: OBSERVE on the memory candidate, then one commit (both components
    and the new identity together) or an abort (nothing installed). A refusal leaves the candidate unchanged."""

    __slots__ = ("_state", "_token", "source_digest", "_open")

    def __init__(self, state, token, source_digest):
        self._state, self._token, self.source_digest, self._open = state, token, source_digest, True

    @property
    def open(self):
        return self._open

    def observe(self, observation):
        """OBSERVE one :class:`MemoryObservation` on the candidate; returns its relation key."""
        if type(observation) is not MemoryObservation:
            raise CompositeError("INVALID", "a MemoryObservation is required")
        key, memory = _Digest(), C.c_int(0)
        native = observation._native()
        rc = self._state._c.leased_txn_observe(self._state._live(), 0, self._token, C.byref(native), C.byref(key),
                                               C.byref(memory))
        if rc != 0:
            raise _refused(rc, "composite observe", memory.value)
        return bytes(key)

    def commit(self):
        """Commit both candidates as one transition; returns the composite, K1 and memory identities before and
        after, and whether the memory changed."""
        out = _Commit()
        rc = self._state._c.leased_txn_commit(self._state._live(), 0, self._token, C.byref(out))
        if rc in (0, -3, -2):   # committed, or discarded by K1's own contract (STALE, NONFINITE)
            self._open = False
        if rc != 0:
            raise _refused(rc, "composite commit")
        return {name: bytes(getattr(out, name)) for name in ("before", "after", "k1_before", "k1_after",
                                                             "memory_before", "memory_after")} | {
            "memory_changed": bool(out.memory_changed), "generation_after": int(out.k1.generation_after)}

    def abort(self):
        if self._open:
            self._open = False
            rc = self._state._c.txn_abort(self._state._live(), self._token)
            if rc != 0:
                raise _refused(rc, "composite transaction abort")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()
