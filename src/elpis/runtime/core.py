"""Thin Python adapter over RuntimeCore (``native/runtime``, ``include/elpis/runtime.h``, ABI v4).

RuntimeCore (Rust) owns the runtime's mutable systems authority: lifecycle, fail-stop, the binding of the K1 lineage
to one native state and its live coordinate, the managed turn's native K1 transaction (a VOLATILE commit), the
durability lane (explicitly requested durable publication, on its own worker thread: recoverable = checkpoint slot,
read back verified, then continuity record; identity-only = continuity record alone), and the evolution attempt's reservation and finalization. This module only marshals values across
the C ABI and raises the stable code RuntimeCore returns. It holds no authority state, decides no transition and
calls no K1 function: it hands RuntimeCore the K1 library's own entry points and the state's native handle.

Live and durable authority (ABI v4, docs/RUNTIME_CORE.md). ``turn_commit`` returns a :class:`VolatileCommit`: the
native transition and the live generation it established; nothing was checkpointed or published. Durability is an
explicit request of one of two kinds, never confused: ``recoverable_request`` (the complete state; a checkpoint store
must be attached; ``ACKNOWLEDGED`` only with a read-back verified slot) or ``identity_request`` (only the identity; no
store attached; ``IDENTITY_PUBLISHED``, never ``ACKNOWLEDGED``). Either returns a PENDING :class:`DurableTicket` at
once; ``durable_wait`` is the caller's choice to wait; ``durable_status`` reads the lane without the HOT lock. A v3
library is refused (version check).

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

from elpis.continuity.adapter import (
    _Counters as _IOCounters,
    _Evolution,
    _Snapshot,
    _evolution_arg,
    _from_c,
    _hex_digest,
    ContinuityLibrary,
    ContinuityProcessDeath,
    ContinuitySnapshot,
    EvolutionAuthority,
)
from elpis.ECS.composite import CompositeState, MemoryObservation, MEMORY_CODES as _MEMORY_CODES
from elpis.ECS.k1 import K1Error, K1FMSState, K1State

from elpis.substrate.contracts import ContractError
from elpis.substrate.native_admission import AdmittedLibrary, admission_of, admit_library

from .errors import CompositionError
from .fuel import CEILING as _FUEL_CEILING

__all__ = ("DurabilityStatus", "DurableTicket", "K1Recovery", "RuntimeCore", "RuntimeLibrary", "TurnBegun",
           "VolatileCommit")

_ABI_VERSION = 4
_TESTING_PROCESS_DEATH = 255
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


class _Ticket(C.Structure):
    """``elpis_runtime_durable_ticket``."""
    _fields_ = [("sequence", _U64), ("generation", _U64), ("status", C.c_uint32), ("recoverable", C.c_uint32),
                ("code", C.c_int32), ("kind", C.c_uint32), ("identity", C.c_uint8 * 32)]


class _Durability(C.Structure):
    """``elpis_runtime_durability``."""
    _fields_ = [("last_requested", _Ticket), ("last_acknowledged", _Ticket), ("running", _Ticket),
                ("live_generation", _U64), ("pending", C.c_uint32), ("fault", C.c_int32), ("parked", C.c_uint32),
                ("durable_known", C.c_uint32), ("checkpoint_capacity", _U64), ("durable", _Snapshot)]


class _CheckpointCounters(C.Structure):
    _fields_ = [(n, _U64) for n in ("preads", "pread_bytes", "pwrites", "pwrite_bytes", "data_syncs")]


class _Counters(C.Structure):
    _fields_ = [(n, _U64) for n in ("k1_state_digests", "k1_reserves", "k1_txn_begins", "k1_run_schedules",
                                    "k1_commits", "k1_aborts", "publications", "k1_queries", "k1_shapes",
                                    "k1_lease_claims", "k1_identities", "k1_captures", "durable_requests",
                                    "durable_superseded", "durable_acknowledged", "durable_failed")]


class _Budget(C.Structure):
    _fields_ = [(n, _U64) for n in ("max_experiences", "max_rows", "max_experience_rows", "max_steps",
                                    "max_work_units", "max_query_rows")]


# The K1 library's own entry points RuntimeCore calls, in the ABI v4 table order. Every mutating entry is a leased
# one (ecsg_k1.h, "Managed ownership"); the live-coordinate entries report the generation (ecsg_k1.h, "Live
# coordinate"); envelope_seal is pure (the durability lane calls it on its own thread).
_API_ENTRIES = ("state_digest", "shape", "lease_claim", "lease_release", "leased_reserve", "leased_identity",
                "leased_query", "leased_txn_begin_at", "leased_txn_run_schedule", "leased_txn_commit", "txn_abort",
                "leased_capture", "envelope_seal")
# The K1 FMS adapter's table appends its residency hold (docs/RUNTIME_CORE.md, "HOT residency").
_FMS_ENTRIES = _API_ENTRIES + ("resident_hold", "resident_release")
# The ECS composite's table (elpis_runtime_composite_api): the K1 table's entries over composite identities, in the
# same order (its leased_txn_commit is the composite's leased_txn_commit_at), then the memory OBSERVE of the open turn.
_COMPOSITE_ENTRIES = tuple("leased_txn_commit_at" if n == "leased_txn_commit" else n for n in _API_ENTRIES) + (
    "leased_txn_observe",)


class _Recovery(C.Structure):
    _fields_ = [("disposition", C.c_uint32), ("authorized_present", C.c_uint32), ("authorized", C.c_uint8 * 32),
                ("candidate", C.c_uint8 * 32), ("envelope_bytes", _U64)]


_DISPOSITIONS = {1: "NOTHING_TO_RECOVER", 2: "RESUMABLE", 3: "CANDIDATE_UNRESOLVED", 4: "CHECKPOINT_MISSING"}
_DURABLE = {0: "NONE", 1: "PENDING", 2: "ACKNOWLEDGED", 3: "FAILED", 4: "UNCERTAIN", 5: "SUPERSEDED", 6: "ABANDONED",
            7: "UNKNOWN", 8: "IDENTITY_PUBLISHED"}
_KIND = {1: "RECOVERABLE", 2: "IDENTITY"}   # elpis_runtime_durable_ticket.kind
_DURABLE_PENDING = 81   # RUNTIME_DURABLE_PENDING: a wait that timed out


@dataclass(frozen=True)
class VolatileCommit:
    """A managed commit (ABI v4): VOLATILE. The native transition and the live generation it established. Nothing
    was checkpointed or published: the durable authority is unchanged until an explicit durable publication."""

    epoch_before: int
    epoch_after: int
    generation_before: int
    generation_after: int
    steps: int
    live_generation: int


@dataclass(frozen=True)
class DurableTicket:
    """One durable publication request (``elpis_runtime_durable_ticket``).

    ``kind`` is what was asked: ``RECOVERABLE`` (the complete state) or ``IDENTITY`` (only the identity).
    ``status``: ``PENDING`` (captured; waiting for, or in, the durability worker), ``ACKNOWLEDGED`` (RECOVERABLE only:
    a checkpoint slot holding exactly the captured envelope was synced and read back verified, and continuity names
    ``identity``; ``recoverable`` is always True: restart resumes exactly this state), ``IDENTITY_PUBLISHED``
    (IDENTITY only: continuity names ``identity``; no store holds the bytes, the caller keeps them; ``recoverable`` is
    always False; never an acknowledgment), ``FAILED`` (nothing durable changed; ``code`` says why), ``UNCERTAIN``
    (outcome unknown: continuity's record was being written or synced when the publication failed or the process
    died; the lane is faulted until reopen; restart resolves it to exactly one complete record, possibly this one; it
    never means "not durable"), ``SUPERSEDED`` (a newer request replaced it before the worker took it), ``ABANDONED``
    (not published: closed, or the process died, before continuity's record was begun) or ``UNKNOWN`` (resolved too
    long ago to be recorded). ``generation`` is the live generation captured.
    """

    sequence: int
    generation: int
    status: str
    recoverable: bool
    code: str | None
    identity: bytes | None
    kind: str | None = None

    @property
    def acknowledged(self) -> bool:
        """Recoverable state acknowledged: restart resumes exactly ``identity``. Never true for identity-only."""
        return self.status == "ACKNOWLEDGED"

    @property
    def identity_published(self) -> bool:
        """Only the identity is durable (the caller keeps the bytes). Never a recoverable acknowledgment."""
        return self.status == "IDENTITY_PUBLISHED"


@dataclass(frozen=True)
class DurabilityStatus:
    """The durability lane as a status read sees it (lane lock only; never waits behind a HOT operation)."""

    last_requested: DurableTicket
    last_acknowledged: DurableTicket
    running: DurableTicket | None
    live_generation: int
    pending: bool
    fault: str | None
    parked: int
    checkpoint_capacity: int
    durable: ContinuitySnapshot | None


@dataclass(frozen=True)
class K1Recovery:
    """What restart may do with the K1 lineage (K1 Recovery R0, docs/K1_RECOVERY_R0.md). Continuity decides; the
    checkpoint slots are evidence and never authorize themselves.

    ``disposition`` is one of ``NOTHING_TO_RECOVER`` (continuity unanchored), ``RESUMABLE`` (``envelope`` is the
    complete authorized ``(W, epoch, H, a)`` envelope: restore it into a K1 state), ``CANDIDATE_UNRESOLVED`` (a
    newer unauthorized ``candidate`` needs an explicit operator discard or adopt) or ``CHECKPOINT_MISSING`` (no
    verified slot holds continuity's identity ``authorized``).
    """

    disposition: str
    authorized: bytes | None
    authorized_present: bool
    candidate: bytes | None
    envelope: bytes | None


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


assert C.sizeof(_Begin) == 48 and C.sizeof(_Commit) == 64 and C.sizeof(_Counters) == 128
assert C.sizeof(_Budget) == 48 and C.sizeof(_Ticket) == 64
assert C.sizeof(_Query) == 16 and C.sizeof(_Recovery) == 80
assert C.sizeof(_CompositeApi) == 14 * C.sizeof(C.c_void_p) and C.sizeof(_ObserveResult) == 40
assert C.sizeof(_Durability) == 3 * 64 + 32 + C.sizeof(_Snapshot)


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


RUNTIME_LIBRARY_IDS = ("elpis_runtime", "elpis_runtime_testing")


class RuntimeLibrary:
    """The RuntimeCore library, admitted: deployment-pinned identity, opened beneath a trusted root, verified and
    loaded from sealed bytes (``elpis.substrate.native_admission``). It is never loaded from a pathname.

    It also exports the continuity C ABI of the store it embeds: ``continuity`` is the record codec over it.
    """

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
            "create": [C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p)],
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
            "recoverable_request": [P, C.POINTER(_Substrate), C.POINTER(_Ticket)],
            "identity_request": [P, C.POINTER(_Substrate), C.POINTER(_Ticket)],
            "durable_wait": [P, _U64, _U64, C.POINTER(_Ticket)],
            "durable_status": [P, C.POINTER(_Durability)],
            "live_identity": [P, C.POINTER(_Substrate), P, C.POINTER(_U64)],
            "turn_abort": [P, C.POINTER(_Substrate)],
            "release": [P, C.POINTER(_Substrate)],
            "evolution_authority": [P, S],
            "evolution_reserve": [P, E, P, S],
            "evolution_finalize": [P, P, S],
            "evolution_abandon": [P],
            "evolution_reconcile": [P, E, P, S],
            "checkpoint_provision": [C.c_char_p, C.c_size_t, C.c_size_t],
            "checkpoint_attach": [P, C.c_char_p, C.c_size_t],
            "checkpoint_recover": [P, C.POINTER(_Recovery), P, C.c_size_t],
            "checkpoint_discard": [P, P],
            "checkpoint_adopt": [P, P, S],
            "turn_observe": [P, C.POINTER(_Substrate), P, C.POINTER(_ObserveResult)],
            "import_composite": [P, C.POINTER(_Substrate), S],
        }
        for name, args in sig.items():
            fn = getattr(lib, "elpis_runtime_" + name)
            fn.argtypes, fn.restype = args, C.c_int
        for name, args in (("close", [P]), ("destroy", [C.POINTER(C.c_void_p)])):
            fn = getattr(lib, "elpis_runtime_" + name)
            fn.argtypes, fn.restype = args, None
        self.testing = hasattr(lib, "elpis_runtime_testing_fault")
        if self.testing:
            lib.elpis_runtime_testing_fault.argtypes = [P, C.c_uint64, C.c_uint32, C.c_uint64]
            lib.elpis_runtime_testing_io_counters.argtypes = [P, C.POINTER(_IOCounters), C.c_int]
            lib.elpis_runtime_testing_durable_hold.argtypes = [P, C.c_int]
            lib.elpis_runtime_testing_durable_fault.argtypes = [P, C.c_uint32, C.c_uint32, C.c_uint64]
            lib.elpis_runtime_testing_checkpoint_io_counters.argtypes = [P, C.POINTER(_CheckpointCounters), C.c_int]
        ceiling = _Budget()
        if lib.elpis_runtime_fuel_ceiling(C.byref(ceiling)) != 0 or \
                tuple(getattr(ceiling, n) for n, _ in _Budget._fields_) != _FUEL_CEILING.native_fields():
            raise CompositionError("RUNTIME_INVALID", "the compiled fuel ceiling is not the canonical one")
        self.admission = admitted
        self.continuity = ContinuityLibrary(lib)   # the continuity ABI of the same admitted bytes
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
        if rc == _TESTING_PROCESS_DEATH:
            raise ContinuityProcessDeath("simulated process death")
        if k1_status:
            detail = f"{detail}; K1 {_K1_NAMES.get(k1_status, k1_status)}" if detail else f"K1 {_K1_NAMES.get(k1_status, k1_status)}"
        raise CompositionError(self.code(rc), detail, k1_status=k1_status)


class RuntimeCore:
    """One RuntimeCore handle over one continuity directory. Every method is one call into RuntimeCore."""

    def __init__(self, library: RuntimeLibrary, directory: str | Path):
        if type(library) is not RuntimeLibrary:
            raise TypeError("RuntimeCore takes a RuntimeLibrary")
        self.library = library
        self.directory = Path(directory)
        raw = str(self.directory).encode()
        handle = C.c_void_p()
        library.check(library._lib.elpis_runtime_create(raw, len(raw), C.byref(handle)))
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
        return _from_c(out, composite=True)   # RuntimeCore is format-aware: a lineage is K1 or ECS composite

    # -- lifecycle --------------------------------------------------------------------------------------------
    def open(self) -> ContinuitySnapshot:
        return self._snapshot("open")

    def close(self) -> None:
        """Close; an open managed turn is aborted natively first (nothing installed, nothing published)."""
        self._f.elpis_runtime_close(self._handle)

    def fault(self) -> str | None:
        rc = self._f.elpis_runtime_fault(self._handle)
        return None if rc == 0 else self.library.code(rc)

    def require_live(self) -> None:
        """Raise the fail-stop disposition (or RUNTIME_CLOSED), if any."""
        rc = self._f.elpis_runtime_fault(self._handle)
        self.library.check(rc, "runtime is fail-stopped pending restart/reconciliation")

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

    def import_composite(self, descriptor) -> ContinuitySnapshot:
        """LEGACY, explicit: move the K1 lineage to the ECS composite ``descriptor`` names (made by
        ``CompositeState.import_k1`` from exactly the authorized K1 state, empty memory); durable before continuity
        publishes the import. Needs an attached checkpoint store sized for the composite envelope."""
        return self._snapshot("import_composite", C.byref(descriptor))

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

    # -- durable publication (the explicit WARM lane) -----------------------------------------------------------------
    def _ticket(self, t: _Ticket) -> DurableTicket:
        identity = bytes(t.identity)
        code = self.library.code(t.code) if t.code else None
        ticket = DurableTicket(int(t.sequence), int(t.generation), _DURABLE.get(t.status, f"STATUS_{t.status}"),
                               bool(t.recoverable), code, identity if any(identity) else None, _KIND.get(t.kind))
        # The ABI's law, re-checked at the boundary: an acknowledgment is recoverable and only ever a recoverable
        # request's; identity-only publication is never recoverable. Anything else is a contract violation.
        if ticket.recoverable != ticket.acknowledged or (ticket.acknowledged and ticket.kind != "RECOVERABLE") or \
                (ticket.identity_published and ticket.kind != "IDENTITY"):
            raise CompositionError("RUNTIME_INVALID", f"durable ticket violates the acknowledgment law: {ticket}")
        return ticket

    def recoverable_request(self, descriptor) -> DurableTicket:
        """Capture the live state (unsealed, under the lease) and hand it to the durability lane for RECOVERABLE
        publication (checkpoint slot, read back verified, then continuity): PENDING at once. Refused with
        ``RUNTIME_RECOVERY_UNAVAILABLE`` (nothing bound or captured) unless a checkpoint store is attached."""
        out = _Ticket()
        self.library.check(self._f.elpis_runtime_recoverable_request(self._handle, C.byref(descriptor), C.byref(out)))
        return self._ticket(out)

    def identity_request(self, descriptor) -> DurableTicket:
        """Capture the live state and hand it to the durability lane for IDENTITY-ONLY publication (continuity alone;
        the caller keeps the bytes): PENDING at once. Refused with ``RUNTIME_CHECKPOINT_ATTACHED`` (nothing bound or
        captured) while a checkpoint store is attached."""
        out = _Ticket()
        self.library.check(self._f.elpis_runtime_identity_request(self._handle, C.byref(descriptor), C.byref(out)))
        return self._ticket(out)

    def durable_wait(self, sequence: int, timeout: float | None = None) -> DurableTicket:
        """Wait (the caller's explicit choice) until request ``sequence`` resolves, at most ``timeout`` seconds (None:
        without limit). Returns the ticket as it stands (``PENDING`` after a timeout)."""
        ms = (1 << 64) - 1 if timeout is None else max(0, int(timeout * 1000))
        out = _Ticket()
        rc = self._f.elpis_runtime_durable_wait(self._handle, sequence, ms, C.byref(out))
        if rc != _DURABLE_PENDING:
            self.library.check(rc)
        return self._ticket(out)

    def durable_status(self) -> DurabilityStatus:
        out = _Durability()
        self.library.check(self._f.elpis_runtime_durable_status(self._handle, C.byref(out)))
        return DurabilityStatus(self._ticket(out.last_requested), self._ticket(out.last_acknowledged),
                                self._ticket(out.running) if out.running.sequence else None,
                                int(out.live_generation), bool(out.pending),
                                self.library.code(out.fault) if out.fault else None, int(out.parked),
                                int(out.checkpoint_capacity), _from_c(out.durable, composite=True) if out.durable_known else None)

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

    def evolution_reconcile(self, expected: EvolutionAuthority, receipt_digest: str) -> ContinuitySnapshot:
        return self._snapshot("evolution_reconcile", _ref(_evolution_arg(expected)), _hex_digest(receipt_digest))

    # -- K1 Recovery R0 ------------------------------------------------------------------------------------------------
    def checkpoint_attach(self, directory: Path) -> None:
        raw = str(directory).encode()
        self.library.check(self._f.elpis_runtime_checkpoint_attach(self._handle, raw, len(raw)))

    def checkpoint_recover(self) -> K1Recovery:
        """Read-only: the disposition, and the authorized envelope when resumable (one sizing call, one copy)."""
        out = _Recovery()
        self.library.check(self._f.elpis_runtime_checkpoint_recover(self._handle, C.byref(out), None, 0))
        envelope = None
        if out.envelope_bytes:
            size = int(out.envelope_bytes)
            buffer, again = (C.c_uint8 * size)(), _Recovery()
            self.library.check(self._f.elpis_runtime_checkpoint_recover(self._handle, C.byref(again), buffer, size))
            if bytes(again.authorized) != bytes(out.authorized) or again.envelope_bytes != size:
                raise CompositionError("CHECKPOINT_INVALID", "the checkpoint changed while it was read")
            envelope = bytes(buffer)
        present = bool(out.authorized_present)
        authorized, candidate = bytes(out.authorized), bytes(out.candidate)
        return K1Recovery(_DISPOSITIONS[out.disposition], authorized if any(authorized) else None, present,
                          candidate if any(candidate) else None, envelope)

    def checkpoint_discard(self, candidate: bytes) -> None:
        self.library.check(self._f.elpis_runtime_checkpoint_discard(self._handle, _digest_arg(candidate)))

    def checkpoint_adopt(self, candidate: bytes) -> ContinuitySnapshot:
        return self._snapshot("checkpoint_adopt", _digest_arg(candidate),
                              detail="the candidate was not adopted")

    # -- testing library only -------------------------------------------------------------------------------------------
    def testing_fault(self, publication: int, action: int, arg: int = 0) -> None:
        self._require_testing()
        self.library.check(self._f.elpis_runtime_testing_fault(self._handle, publication, action, arg))

    def testing_io_counters(self, reset: bool = False) -> dict[str, int]:
        self._require_testing()
        out = _IOCounters()
        self.library.check(self._f.elpis_runtime_testing_io_counters(self._handle, C.byref(out), int(reset)))
        return {name: getattr(out, name) for name, _ in _IOCounters._fields_}

    def testing_durable_hold(self, hold: bool) -> None:
        """Hold (or release) the durability worker before it takes a new job (the HOT lane must not care)."""
        self._require_testing()
        self.library.check(self._f.elpis_runtime_testing_durable_hold(self._handle, int(bool(hold))))

    def testing_durable_fault(self, action: int, step: int = 0, arg: int = 0) -> None:
        """Plan one fault of the next durable publication (runtime.c ABI: 1 die / 2 park at ``step``; 3..9 I/O
        failures; 10 a checkpoint slot that does not read back exactly); docs/RUNTIME_CORE.md, "Crash law"."""
        self._require_testing()
        self.library.check(self._f.elpis_runtime_testing_durable_fault(self._handle, action, step, arg))

    def testing_checkpoint_io_counters(self, reset: bool = False) -> dict[str, int]:
        self._require_testing()
        out = _CheckpointCounters()
        self.library.check(self._f.elpis_runtime_testing_checkpoint_io_counters(self._handle, C.byref(out),
                                                                                int(reset)))
        return {name: getattr(out, name) for name, _ in _CheckpointCounters._fields_}

    def _require_testing(self) -> None:
        if not self.library.testing:
            raise CompositionError("RUNTIME_INVALID", "fault injection exists only in the testing library")


def _ref(value):
    return C.byref(value) if value is not None else None


def _digest_arg(value):
    if type(value) is not bytes or len(value) != 32:
        raise CompositionError("RUNTIME_INVALID", "a 32-byte retained-state identity is required")
    return (C.c_uint8 * 32).from_buffer_copy(value)
