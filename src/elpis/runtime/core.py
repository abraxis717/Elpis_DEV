"""Thin Python adapter over RuntimeCore (``native/runtime``, ``include/elpis/runtime.h``, ABI v8).

RuntimeCore (Rust) owns the runtime's mutable systems authority: lifecycle, fail-stop, the binding of the K1 lineage
to one native state and its live coordinate, the managed turn's native K1 transaction (a VOLATILE commit), and the
evolution attempt's reservation and finalization. ABI v8 is volatile only (docs/ELPIS_MISSION.md, Amendment 3): its
authority is one volatile ECS_C session (native/ECS_C/session) per open, created from a :class:`SessionConfig`
(``.authority``) at a fresh genesis and gone at close. Nothing is published, checkpointed, recovered or resumed, and no
directory is involved anywhere. This module only marshals values across the C ABI and raises the stable code
RuntimeCore returns. It holds no authority state, decides no transition and calls no K1 function: it hands RuntimeCore
the K1 library's own entry points and the state's native handle.

``turn_commit`` returns a :class:`VolatileCommit`: the native transition and the live generation it established. The
v7 durability lane, checkpoint store, restart recovery, legacy import and evolution reconciliation are retired with
their codes; a v7 library is refused (version check).

The substrate descriptor names the native state, its library's entry points, its declared dimension (verified
natively) and the identity of the Python object that owns the handle (``owner``). The caller keeps that owner
alive while it is bound, so its identity cannot be reused by another state (lifetime only; no decision).

QUERY is read-only: ``query`` answers from the lineage's live state with one native K1 crossing that also reports its
generation (no digest), and opens no transaction. LEARN is the managed turn.

Turn lifecycle (ABI v2). Once ``turn_begin`` has opened a native K1 transaction, RuntimeCore ends it with exactly
one native commit or abort before it forgets the turn: ``turn_commit``, ``turn_abort``, ``close`` and destruction
(``__del__`` -> ``elpis_runtime_destroy``) all end it natively; ``open`` on an open runtime is refused and keeps it.
To do so RuntimeCore retains the state's handle and abort entry until the turn ends, so the native state must stay
live for that interval. Native K1 handles are freed only by an explicit ``close`` (no finalizer frees them), the
FMS adapter refuses to close a resident state with an open transaction, and the facade keeps the bound owner
alive; a standalone ``K1State`` must not be closed while a managed turn on it is open.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from as _unpack_from

from elpis.ECS.composite import CompositeState, MemoryObservation, MEMORY_CODES as _MEMORY_CODES
from elpis.ECS.k1 import K1Error, K1FMSState, K1State

from elpis.substrate.contracts import ContractError
from elpis.substrate.native_admission import AdmittedLibrary, admission_of, admit_library

from .authority import (
    _Evolution,
    _Snapshot,
    _evolution_arg,
    _from_c,
    _hex_digest,
    ContinuitySnapshot,
    EvolutionAuthority,
    SessionConfig,
    label,
)
from .errors import CompositionError
from .fuel import CEILING as _FUEL_CEILING

__all__ = ("RuntimeCore", "RuntimeLibrary", "TurnBegun", "VolatileCommit")

_ABI_VERSION = 8
_U64 = C.c_uint64
_K1_NAMES = {-1: "INVALID", -2: "NONFINITE", -3: "STALE", -4: "BUSY", -5: "CAPACITY", -6: "NOMEM", -7: "CORRUPT",
             -8: "LEASED"}


class _Experience(C.Structure):
    _fields_ = [("rows", _U64), ("steps", _U64)]


class _Schedule(C.Structure):
    _fields_ = [(n, _U64) for n in ("epoch_before", "epoch_after", "experiences_applied", "failed_experience",
                                    "failed_step")]


class _Transition(C.Structure):
    _fields_ = [(n, _U64) for n in ("epoch_before", "epoch_after", "generation_before", "generation_after", "steps",
                                    "failed_step")]


class _Begin(C.Structure):
    _fields_ = [("schedule", _Schedule), ("k1_status", C.c_int32), ("reserved", C.c_uint32)]


class _Commit(C.Structure):
    """``elpis_runtime_volatile_commit``."""
    _fields_ = [("transition", _Transition), ("live_generation", _U64), ("committed", C.c_uint32),
                ("k1_status", C.c_int32)]


class _Query(C.Structure):
    _fields_ = [("live_generation", _U64), ("k1_status", C.c_int32), ("reserved", C.c_uint32)]


class _Counters(C.Structure):
    """``elpis_runtime_counters`` (ABI v8, 96 bytes)."""
    _fields_ = [(n, _U64) for n in ("k1_state_digests", "k1_reserves", "k1_txn_begins", "k1_run_schedules",
                                    "k1_commits", "k1_aborts", "session_transitions", "k1_queries", "k1_shapes",
                                    "k1_lease_claims", "k1_identities", "k1_observes")]


class _Config(C.Structure):
    """``elpis_runtime_config`` (ABI v8, 104 bytes): what every session of a runtime is created with."""
    _fields_ = [("abi_version", C.c_uint32), ("capacity", C.c_uint32), ("nonce", C.c_uint8 * 32),
                ("operator_label", C.c_uint8 * 32), ("runtime_label", C.c_uint8 * 32)]


class _Budget(C.Structure):
    _fields_ = [(n, _U64) for n in ("max_experiences", "max_rows", "max_experience_rows", "max_steps",
                                    "max_work_units", "max_query_rows", "max_observations")]


# The K1 library's own entry points RuntimeCore calls, in the ABI v8 table order. Every mutating entry is a leased
# one (ecsg_k1.h, "Managed ownership"); the live-coordinate entries report the generation (ecsg_k1.h, "Live
# coordinate"). v8 removed leased_capture and envelope_seal (they existed for durable publication).
_API_ENTRIES = ("state_digest", "shape", "lease_claim", "lease_release", "leased_reserve", "leased_identity",
                "leased_query", "leased_txn_begin_at", "leased_txn_run_schedule", "leased_txn_commit", "txn_abort")
# The K1 FMS adapter's table appends its residency hold (docs/RUNTIME_CORE.md, "HOT residency").
_FMS_ENTRIES = _API_ENTRIES + ("resident_hold", "resident_release")
# The ECS composite's table (elpis_runtime_composite_api): the K1 table's entries over composite identities, in the
# same order (its leased_txn_commit is the composite's leased_txn_commit_at), then the memory OBSERVE of the open turn.
_COMPOSITE_ENTRIES = tuple("leased_txn_commit_at" if n == "leased_txn_commit" else n for n in _API_ENTRIES) + (
    "leased_txn_observe",)


class _AuthorityStatus(C.Structure):
    """``elpis_runtime_authority`` (ABI v8, 240 bytes): the open session; while closed only ``backend``, ``sessions``
    and the code of the last refused open."""
    _fields_ = [("backend", C.c_uint32), ("ecs_c_code", C.c_int32), ("entity_count", C.c_uint32),
                ("capacity", C.c_uint32), ("generation", _U64), ("instance", _U64), ("receipt_count", _U64),
                ("evidence_count", _U64), ("volatile_bytes", _U64), ("sessions", _U64),
                ("next_founding_index", _U64), ("reclaimed_count", _U64), ("session", C.c_uint8 * 32),
                ("commit", C.c_uint8 * 32), ("state_root", C.c_uint8 * 32), ("receipt_head", C.c_uint8 * 32),
                ("evidence_head", C.c_uint8 * 32)]


_AUTHORITY_DIGESTS = frozenset({"session", "commit", "state_root", "receipt_head", "evidence_head"})
BACKEND_ECS_C_SESSION = 3


class _AuthorityEntity(C.Structure):
    """``elpis_runtime_authority_entity`` (152 bytes, laid out as ``elpis_ecs_c_entity``)."""
    _fields_ = [("id", C.c_uint8 * 32), ("kind", C.c_uint8), ("lifecycle", C.c_uint8), ("reserved", C.c_uint8 * 2),
                ("per_turn_limit", C.c_uint32), ("founding_index", _U64), ("founded_generation", _U64),
                ("last_generation", _U64), ("sender_watermark", _U64), ("source_version", _U64),
                ("label", C.c_uint8 * 32), ("scope", C.c_uint8 * 32), ("source_authority", C.c_uint32),
                ("reserved2", C.c_uint32)]


assert C.sizeof(_AuthorityStatus) == 240 and C.sizeof(_AuthorityEntity) == 152 and C.sizeof(_Config) == 104


@dataclass(frozen=True)
class VolatileCommit:
    """A managed commit: VOLATILE. The native transition and the live generation it established. Nothing was
    written anywhere (ABI v8 has no persistent path)."""

    epoch_before: int
    epoch_after: int
    generation_before: int
    generation_after: int
    steps: int
    live_generation: int


class _K1Api(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in _API_ENTRIES]


class _K1FmsApi(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in _FMS_ENTRIES]


class _CompositeApi(C.Structure):
    _fields_ = [("base", _K1Api), ("leased_txn_observe", C.c_void_p)]


class _ObserveResult(C.Structure):
    """``elpis_runtime_observe_result``."""
    _fields_ = [("relation_key", C.c_uint8 * 32), ("k1_status", C.c_int32), ("memory_status", C.c_int32)]


class _Substrate(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("reserved", C.c_uint32), ("handle", C.c_void_p), ("id", _U64),
                ("owner", _U64), ("dim", _U64), ("api", C.c_void_p)]


assert C.sizeof(_Begin) == 48 and C.sizeof(_Commit) == 64 and C.sizeof(_Counters) == 96
assert C.sizeof(_Budget) == 56 and C.sizeof(_Query) == 16
assert C.sizeof(_CompositeApi) == 12 * C.sizeof(C.c_void_p) and C.sizeof(_ObserveResult) == 40


def _api(lib, prefix):
    """The library's own entry points (addresses of the loaded symbols), as RuntimeCore's function table."""
    if prefix == "elpis_ecs_composite_":
        address = [C.cast(getattr(lib, prefix + n), C.c_void_p).value for n in _COMPOSITE_ENTRIES]
        return _CompositeApi(_K1Api(*address[:-1]), address[-1])
    struct, entries = (_K1FmsApi, _FMS_ENTRIES) if prefix == "elpis_ecsg_k1_fms_" else (_K1Api, _API_ENTRIES)
    return struct(*(C.cast(getattr(lib, prefix + n), C.c_void_p).value for n in entries))


_TABLES: dict[int, tuple[object, _K1Api]] = {}   # id(loaded library) -> (library, table); libraries are never unloaded


def _table(lib, prefix):
    entry = _TABLES.get(id(lib))
    if entry is None or entry[0] is not lib:
        entry = _TABLES[id(lib)] = (lib, _api(lib, prefix))
    return entry[1]


K1_LIBRARY_IDS = {K1State: "elpis_ecsg_k1", K1FMSState: "elpis_ecsg_k1_fms", CompositeState: "elpis_ecs_composite"}


def _require_admitted(lib, kind):
    """The managed runtime drives only K1 code it can prove: the state's library must have been admitted (sealed,
    pinned) under the K1 library identifier of its kind."""
    admitted = admission_of(lib)
    if admitted is None or admitted.library_id != K1_LIBRARY_IDS[kind]:
        raise CompositionError("ECS_NATIVE_UNADMITTED",
                               "the K1 state's native library was not admitted from sealed, pinned bytes")


def describe(substrate):
    """``(descriptor, owner)`` for a native K1 state or ECS composite; raises ``ECS_STATE`` for anything else or a
    closed state, and ``ECS_NATIVE_UNADMITTED`` for a state whose native library was not admitted."""
    try:
        if type(substrate) is K1State:
            _require_admitted(substrate._lib, K1State)
            table = _table(substrate._lib, "elpis_ecsg_k1_")
            d = _Substrate(1, 0, substrate._live(), 0, id(substrate), substrate.dim, C.addressof(table))
            return d, substrate
        if type(substrate) is K1FMSState:
            runtime = substrate._r
            _require_admitted(runtime._lib, K1FMSState)
            table = _table(runtime._lib, "elpis_ecsg_k1_fms_")
            d = _Substrate(2, 0, runtime._live(), substrate.id, id(runtime), substrate.dim, C.addressof(table))
            return d, runtime
        if type(substrate) is CompositeState:
            _require_admitted(substrate._lib, CompositeState)
            table = _table(substrate._lib, "elpis_ecs_composite_")
            d = _Substrate(3, 0, substrate._live(), 0, id(substrate), substrate.dim, C.addressof(table))
            return d, substrate
    except K1Error as exc:
        raise CompositionError("ECS_STATE", str(exc)) from exc
    raise CompositionError("ECS_STATE", "a native K1 state (K1State or K1FMSState) or an ECS CompositeState is "
                                        "required")


class TurnBegun(tuple):
    """What RuntimeCore's turn begin returns: the readout of the candidate and its epochs."""

    __slots__ = ()

    def __new__(cls, s3, epoch_before, epoch_after):
        return tuple.__new__(cls, (s3, epoch_before, epoch_after))

    s3 = property(lambda self: self[0])
    epoch_before = property(lambda self: self[1])
    epoch_after = property(lambda self: self[2])


RUNTIME_LIBRARY_IDS = ("elpis_runtime",)   # ABI v7's testing build (fault injection) retired with the durability lane


class RuntimeLibrary:
    """The RuntimeCore library, admitted: deployment-pinned identity, opened beneath a trusted root, verified and
    loaded from sealed bytes (``elpis.substrate.native_admission``). It is never loaded from a pathname. It exports
    RuntimeCore's C ABI only (ABI v8: the session crate is embedded without its C ABI)."""

    @classmethod
    def admit(cls, root, path, authority, library_id: str = "elpis_runtime") -> "RuntimeLibrary":
        """Admit ``path`` beneath ``root`` as ``library_id`` of the deployment ``authority``, then bind it."""
        if library_id not in RUNTIME_LIBRARY_IDS:
            raise CompositionError("RUNTIME_UNPINNED", f"not a RuntimeCore library identifier: {library_id!r}")
        try:
            return cls(admit_library(root, path, authority, library_id))
        except ContractError as exc:
            raise CompositionError("RUNTIME_UNPINNED", str(exc)) from exc

    def __init__(self, admitted: AdmittedLibrary):
        if type(admitted) is not AdmittedLibrary or admission_of(admitted.lib) is not admitted \
                or admitted.library_id not in RUNTIME_LIBRARY_IDS:
            raise CompositionError("RUNTIME_UNPINNED", "RuntimeCore runs only from an admitted, pinned library")
        lib = admitted.lib
        lib.elpis_runtime_abi_version.restype = C.c_uint32
        if lib.elpis_runtime_abi_version() != _ABI_VERSION:
            raise CompositionError("RUNTIME_INVALID", "runtime ABI version mismatch")
        lib.elpis_runtime_code_name.restype = C.c_char_p
        lib.elpis_runtime_code_name.argtypes = [C.c_int]
        lib.elpis_runtime_features.restype = C.c_size_t
        lib.elpis_runtime_features.argtypes = [C.c_size_t]
        S, E, P, D = C.POINTER(_Snapshot), C.POINTER(_Evolution), C.c_void_p, C.POINTER(C.c_double)
        sig = {
            "create": [C.POINTER(_Config), C.POINTER(C.c_void_p)],
            "authority_status": [P, C.POINTER(_AuthorityStatus)],
            "authority_transition": [P, P, C.POINTER(_U64)],
            "authority_entity_at": [P, C.c_uint32, C.POINTER(_AuthorityEntity)],
            "authority_next_entity_id": [P, C.c_uint8, P, P],
            "open": [P, S],
            "fault": [P],
            "snapshot": [P, S],
            "read_counters": [P, C.POINTER(_Counters), C.c_int],
            "anchor": [P, C.POINTER(_Substrate), S],
            "query": [P, C.POINTER(_Substrate), D, C.c_size_t, D, C.c_size_t, C.POINTER(_Budget), C.POINTER(_Query)],
            "turn_begin": [P, C.POINTER(_Substrate), D, C.c_size_t, D, C.c_size_t, C.POINTER(_Experience),
                           C.c_size_t, C.c_double, C.POINTER(_Budget), D, C.c_size_t, C.POINTER(_Begin)],
            "fuel_ceiling": [C.POINTER(_Budget)],
            "work_units": [C.c_size_t, C.c_size_t, C.POINTER(_Experience), C.c_size_t, C.POINTER(_U64)],
            "query_work_units": [C.c_size_t, C.c_size_t, C.c_size_t, C.POINTER(_U64)],
            "turn_commit": [P, C.POINTER(_Substrate), C.POINTER(_Commit)],
            "live_identity": [P, C.POINTER(_Substrate), P, C.POINTER(_U64)],
            "turn_abort": [P, C.POINTER(_Substrate)],
            "release": [P, C.POINTER(_Substrate)],
            "evolution_authority": [P, S],
            "evolution_reserve": [P, E, P, S],
            "evolution_finalize": [P, P, S],
            "evolution_abandon": [P],
            "turn_observe": [P, C.POINTER(_Substrate), P, C.POINTER(_ObserveResult)],
            "turn_begin_observe": [P, C.POINTER(_Substrate), C.POINTER(_Budget), C.POINTER(_Query)],
        }
        for name, args in sig.items():
            fn = getattr(lib, "elpis_runtime_" + name)
            fn.argtypes, fn.restype = args, C.c_int
        for name, args in (("close", [P]), ("destroy", [C.POINTER(C.c_void_p)])):
            fn = getattr(lib, "elpis_runtime_" + name)
            fn.argtypes, fn.restype = args, None
        ceiling = _Budget()
        if lib.elpis_runtime_fuel_ceiling(C.byref(ceiling)) != 0 or \
                tuple(getattr(ceiling, n) for n, _ in _Budget._fields_) != _FUEL_CEILING.native_fields():
            raise CompositionError("RUNTIME_INVALID", "the compiled fuel ceiling is not the canonical one")
        self.admission = admitted
        self._lib = lib
        self._features: dict[int, int] = {}

    def features(self, dim: int) -> int:
        """The S3 readout length for an input dimension (a pure native function, asked once per dimension)."""
        n = self._features.get(dim)
        if n is None:
            n = self._features[dim] = int(self._lib.elpis_runtime_features(dim))
        return n

    def work_units(self, dim: int, width: int, schedule) -> int:
        """Native ECS work units of a ``(rows, steps)`` uint64 schedule buffer (the fuel authority's formula)."""
        n = len(schedule) // 2
        out = _U64()
        self.check(self._lib.elpis_runtime_work_units(dim, width, (_Experience * n).from_buffer_copy(schedule), n,
                                                      C.byref(out)))
        return int(out.value)

    def query_work_units(self, dim: int, width: int, rows: int) -> int:
        out = _U64()
        self.check(self._lib.elpis_runtime_query_work_units(dim, width, rows, C.byref(out)))
        return int(out.value)

    def code(self, rc: int) -> str:
        name = self._lib.elpis_runtime_code_name(rc)
        return name.decode() if name else f"RUNTIME_UNKNOWN_{rc}"

    def check(self, rc: int, detail: str = "", k1_status: int = 0) -> None:
        if rc == 0:
            return
        if k1_status:
            detail = f"{detail}; K1 {_K1_NAMES.get(k1_status, k1_status)}" if detail else f"K1 {_K1_NAMES.get(k1_status, k1_status)}"
        raise CompositionError(self.code(rc), detail, k1_status=k1_status)


class RuntimeCore:
    """One RuntimeCore handle whose sessions are created from ``session`` (a :class:`SessionConfig`): each
    :meth:`open` is a new volatile ECS_C session at a fresh genesis, :meth:`close` ends it. No directory, file or
    path is involved. Every method is one call into RuntimeCore."""

    def __init__(self, library: RuntimeLibrary, session: SessionConfig):
        if type(library) is not RuntimeLibrary:
            raise TypeError("RuntimeCore takes a RuntimeLibrary")
        if type(session) is not SessionConfig:
            raise TypeError("RuntimeCore takes a SessionConfig")
        self.library = library
        self.session = session
        digest = C.c_uint8 * 32
        config = _Config(_ABI_VERSION, session.capacity, digest.from_buffer_copy(session.nonce_seed or bytes(32)),
                         digest.from_buffer_copy(label(session.operator_label)),
                         digest.from_buffer_copy(label(session.runtime_label)))
        handle = C.c_void_p()
        library.check(library._lib.elpis_runtime_create(C.byref(config), C.byref(handle)))
        self._handle = handle
        self._f = library._lib

    def __del__(self):
        # Destruction aborts an open managed turn natively before the handle is freed (RuntimeCore's Drop).
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self.library._lib.elpis_runtime_destroy(C.byref(handle))

    def _snapshot(self, name: str, *args, detail: str = "") -> ContinuitySnapshot:
        out = _Snapshot()
        self.library.check(getattr(self._f, "elpis_runtime_" + name)(self._handle, *args, C.byref(out)), detail)
        return _from_c(out)   # RuntimeCore is format-aware: a lineage is K1 or ECS composite

    # -- lifecycle --------------------------------------------------------------------------------------------
    def open(self) -> ContinuitySnapshot:
        """Start a new session: a fresh genesis (the founding operator and runtime), nothing anchored."""
        return self._snapshot("open")

    def close(self) -> None:
        """End the session; an open managed turn is aborted natively first. Everything the session held is gone."""
        self._f.elpis_runtime_close(self._handle)

    def fault(self) -> str | None:
        rc = self._f.elpis_runtime_fault(self._handle)
        return None if rc == 0 else self.library.code(rc)

    def require_live(self) -> None:
        """Raise the fail-stop disposition (or RUNTIME_CLOSED), if any."""
        rc = self._f.elpis_runtime_fault(self._handle)
        self.library.check(rc, "runtime is fail-stopped: close it and open a new session")

    def snapshot(self) -> ContinuitySnapshot:
        return self._snapshot("snapshot")

    def counters(self, reset: bool = False) -> dict[str, int]:
        out = _Counters()
        self.library.check(self._f.elpis_runtime_read_counters(self._handle, C.byref(out), int(reset)))
        return {name: getattr(out, name) for name, _ in _Counters._fields_}

    # -- K1 lineage and the managed turn ----------------------------------------------------------------------------
    def anchor(self, descriptor) -> ContinuitySnapshot:
        return self._snapshot("anchor", C.byref(descriptor), detail="no K1 mutation happened")

    def query(self, descriptor, stimulus, budget) -> tuple[tuple, int]:
        """QUERY (read-only): ``(f_W(x) per row, live generation)`` from the lineage's live state.

        One call into RuntimeCore, one native K1 crossing (after a one-time read of the state's shape); nothing is
        begun, committed, published or hashed. RuntimeCore admits the query rows and work units under ``budget``
        first.
        """
        rows = stimulus.rows
        out, result = (C.c_double * rows)(), _Query()
        x = (C.c_double * len(stimulus.x)).from_buffer(stimulus.x)
        rc = self._f.elpis_runtime_query(self._handle, C.byref(descriptor), x, len(stimulus.x), out, rows,
                                         C.byref(_Budget(*budget.native_fields())), C.byref(result))
        self.library.check(rc, "", result.k1_status)
        return _unpack_from(f"{rows}d", out), int(result.live_generation)

    def turn_begin(self, descriptor, stimulus, learning_rate: float, budget) -> TurnBegun:
        features = self.library.features(descriptor.dim)
        s3, out = (C.c_double * max(features, 1))(), _Begin()
        x = (C.c_double * len(stimulus.x)).from_buffer(stimulus.x)
        y = (C.c_double * len(stimulus.y)).from_buffer(stimulus.y)
        schedule = (_Experience * stimulus.experiences).from_buffer(stimulus.schedule)
        rc = self._f.elpis_runtime_turn_begin(self._handle, C.byref(descriptor), x, len(stimulus.x), y,
                                              len(stimulus.y), schedule, stimulus.experiences, learning_rate,
                                              C.byref(_Budget(*budget.native_fields())), s3, features, C.byref(out))
        self.library.check(rc, "", out.k1_status)
        return TurnBegun(_unpack_from(f"{features}d", s3), int(out.schedule.epoch_before),
                         int(out.schedule.epoch_after))

    def turn_observe(self, descriptor, observation) -> bytes:
        """OBSERVE one :class:`~elpis.ECS.composite.MemoryObservation` on the open turn's memory candidate (an ECS
        composite substrate only); returns its relation key. Authoritative only with the turn's commit. A refusal
        (``ECS_REFUSED`` with the composite and memory statuses) leaves the turn open and its candidate unchanged;
        RuntimeCore never reads the observation."""
        if type(observation) is not MemoryObservation:
            raise CompositionError("RUNTIME_INVALID", "a MemoryObservation is required")
        native, out = observation._native(), _ObserveResult()
        rc = self._f.elpis_runtime_turn_observe(self._handle, C.byref(descriptor), C.byref(native), C.byref(out))
        if rc != 0 and out.memory_status:
            memory = _MEMORY_CODES.get(out.memory_status, out.memory_status)
            self.library.check(rc, f"memory {memory}", out.k1_status)
        self.library.check(rc, "", out.k1_status)
        return bytes(out.relation_key)

    def turn_begin_observe(self, descriptor, budget) -> int:
        """Begin an OBSERVE-ONLY managed turn on an ECS composite substrate (ABI v5): the lineage is verified and bound
        as by :meth:`turn_begin` and the composite transaction is opened at the live generation, but no K1 schedule
        runs (the K1 retained state stays byte-identical). Observations enter through :meth:`turn_observe` (at most
        ``budget.max_observations``, refused ``COGNITION_FUEL_EXCEEDED`` beyond); end it with :meth:`turn_commit`
        (refused ``RUNTIME_INVALID``, the turn kept open, while nothing was observed) or :meth:`turn_abort`. Returns
        the live generation the turn was opened on."""
        out = _Query()   # layout-identical to elpis_runtime_observe_begin_result
        rc = self._f.elpis_runtime_turn_begin_observe(self._handle, C.byref(descriptor),
                                                      C.byref(_Budget(*budget.native_fields())), C.byref(out))
        self.library.check(rc, "", out.k1_status)
        return int(out.live_generation)

    def turn_commit(self, descriptor) -> VolatileCommit:
        """The volatile native commit of the open turn: nothing is checkpointed or published."""
        out = _Commit()
        rc = self._f.elpis_runtime_turn_commit(self._handle, C.byref(descriptor), C.byref(out))
        if rc != 0 and out.committed:
            self.library.check(rc, "K1 turn committed; it does not continue the live coordinate")
        self.library.check(rc, "", out.k1_status)
        t = out.transition
        return VolatileCommit(int(t.epoch_before), int(t.epoch_after), int(t.generation_before),
                              int(t.generation_after), int(t.steps), int(out.live_generation))

    def live_identity(self, descriptor) -> tuple[bytes, int]:
        """The bound state's retained-state identity and live generation (cold path: one hashing native call)."""
        digest, generation = (C.c_uint8 * 32)(), _U64()
        self.library.check(self._f.elpis_runtime_live_identity(self._handle, C.byref(descriptor), digest,
                                                               C.byref(generation)))
        return bytes(digest), int(generation.value)

    def turn_abort(self, descriptor) -> None:
        self.library.check(self._f.elpis_runtime_turn_abort(self._handle, C.byref(descriptor)))

    def release(self, descriptor) -> None:
        """Release the bound state (its K1 lease) and unbind it: its owner may use it unmanaged again."""
        self.library.check(self._f.elpis_runtime_release(self._handle, C.byref(descriptor)))

    # -- evolution ----------------------------------------------------------------------------------------------------
    def evolution_authority(self) -> ContinuitySnapshot:
        return self._snapshot("evolution_authority", detail="explicit reconciliation required")

    def evolution_reserve(self, observed: EvolutionAuthority, assertion_digest: str) -> ContinuitySnapshot:
        return self._snapshot("evolution_reserve", _ref(_evolution_arg(observed)), _hex_digest(assertion_digest),
                              detail="evolution reservation failed; no attempt executed")

    def evolution_finalize(self, receipt_digest: str) -> ContinuitySnapshot:
        return self._snapshot("evolution_finalize", _hex_digest(receipt_digest),
                              detail="evolution attempt executed; finalization not certain")

    def evolution_abandon(self) -> None:
        self.library.check(self._f.elpis_runtime_evolution_abandon(self._handle))

    # -- the session (operator events while it is open) ----------------------------------------------------------------
    def authority_status(self) -> dict:
        """``elpis_runtime_authority_status``: the session's backend (always the volatile ECS_C session), last refusal,
        and while open its generation, counts, footprint and chain heads. Readable closed."""
        out = _AuthorityStatus()
        self.library.check(self._f.elpis_runtime_authority_status(self._handle, C.byref(out)))
        return {name: (bytes(getattr(out, name)).hex() if name in _AUTHORITY_DIGESTS else getattr(out, name))
                for name, _ in _AuthorityStatus._fields_ if not name.startswith("reserved")}

    def authority_entity(self, index: int) -> dict:
        out = _AuthorityEntity()
        self.library.check(self._f.elpis_runtime_authority_entity_at(self._handle, index, C.byref(out)))
        return {"id": bytes(out.id), "kind": out.kind, "lifecycle": out.lifecycle, "per_turn": out.per_turn_limit,
                "founding_index": out.founding_index, "founded_generation": out.founded_generation,
                "last_generation": out.last_generation, "sender_watermark": out.sender_watermark,
                "source_version": out.source_version, "source_authority": out.source_authority,
                "label": bytes(out.label), "scope": bytes(out.scope)}

    def authority_next_entity_id(self, kind: int, label: bytes) -> bytes:
        out = (C.c_uint8 * 32)()
        self.library.check(self._f.elpis_runtime_authority_next_entity_id(self._handle, kind, _digest_arg(label), out))
        return bytes(out)

    def authority_transition(self, event: bytes) -> int:
        """One operator event (512 bytes) on the open session; returns its new generation."""
        if type(event) is not bytes or len(event) != 512:
            raise CompositionError("RUNTIME_INVALID", "an operator event is 512 bytes")
        generation = _U64()
        self.library.check(self._f.elpis_runtime_authority_transition(
            self._handle, (C.c_uint8 * 512).from_buffer_copy(event), C.byref(generation)))
        return int(generation.value)


def _ref(value):
    return C.byref(value) if value is not None else None


def _digest_arg(value):
    if type(value) is not bytes or len(value) != 32:
        raise CompositionError("RUNTIME_INVALID", "a 32-byte retained-state identity is required")
    return (C.c_uint8 * 32).from_buffer_copy(value)
