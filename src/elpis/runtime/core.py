"""Thin Python adapter over RuntimeCore (``native/runtime``, ``include/elpis/runtime.h``).

RuntimeCore (Rust) owns the runtime's mutable systems authority: lifecycle, fail-stop, the continuity store, the
binding of the K1 lineage to one native state, the managed turn's native K1 transaction and its continuity
publication, and the evolution attempt's reservation and finalization. This module only marshals values across
the C ABI and raises the stable code RuntimeCore returns. It holds no authority state, decides no transition and
calls no K1 function: it hands RuntimeCore the K1 library's own entry points and the state's native handle.

The substrate descriptor names the native state, its library's entry points, its declared dimension (verified
natively) and the identity of the Python object that owns the handle (``owner``). The caller keeps that owner
alive while it is bound, so its identity cannot be reused by another state (lifetime only; no decision).

QUERY (ABI v3) is read-only: ``query`` answers from the lineage's authoritative state with one native K1 crossing that
also identifies it, and opens no transaction. LEARN is the managed turn.

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
from elpis.ECS.k1 import CommitIdentity, K1Error, K1FMSState, K1State
from elpis.ECS.native import Commit

from elpis.substrate.contracts import ContractError
from elpis.substrate.native_admission import AdmittedLibrary, admission_of, admit_library

from .errors import CompositionError
from .fuel import CEILING as _FUEL_CEILING

__all__ = ("K1Recovery", "RuntimeCore", "RuntimeLibrary", "TurnBegun")

_ABI_VERSION = 3
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


class _Identity(C.Structure):
    _fields_ = [("transition", _Transition), ("state_before_digest", C.c_uint8 * 32),
                ("state_after_digest", C.c_uint8 * 32)]


class _Begin(C.Structure):
    _fields_ = [("schedule", _Schedule), ("k1_status", C.c_int32), ("reserved", C.c_uint32)]


class _Commit(C.Structure):
    _fields_ = [("identity", _Identity), ("committed", C.c_uint32), ("k1_status", C.c_int32)]


class _Query(C.Structure):
    _fields_ = [("state_digest", C.c_uint8 * 32), ("k1_status", C.c_int32), ("reserved", C.c_uint32)]


class _Counters(C.Structure):
    _fields_ = [(n, _U64) for n in ("k1_state_digests", "k1_reserves", "k1_txn_begins", "k1_run_schedules",
                                    "k1_commits", "k1_aborts", "publications", "k1_queries", "k1_shapes",
                                    "k1_lease_claims")]


class _Budget(C.Structure):
    _fields_ = [(n, _U64) for n in ("max_experiences", "max_rows", "max_experience_rows", "max_steps",
                                    "max_work_units", "max_query_rows")]


# The K1 library's own entry points RuntimeCore calls, in the ABI v3 table order. Every mutating entry is a leased
# one (ecsg_k1.h, "Managed ownership").
_API_ENTRIES = ("state_digest", "shape", "query_identity", "lease_claim", "lease_release", "leased_reserve",
                "leased_txn_begin", "leased_txn_run_schedule", "leased_txn_commit_identity", "txn_abort",
                "snapshot_write", "txn_snapshot_write")


class _Recovery(C.Structure):
    _fields_ = [("disposition", C.c_uint32), ("authorized_present", C.c_uint32), ("authorized", C.c_uint8 * 32),
                ("candidate", C.c_uint8 * 32), ("envelope_bytes", _U64)]


_DISPOSITIONS = {1: "NOTHING_TO_RECOVER", 2: "RESUMABLE", 3: "CANDIDATE_UNRESOLVED", 4: "CHECKPOINT_MISSING"}


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


class _Substrate(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("reserved", C.c_uint32), ("handle", C.c_void_p), ("id", _U64),
                ("owner", _U64), ("dim", _U64), ("api", C.c_void_p)]


assert C.sizeof(_Begin) == 48 and C.sizeof(_Commit) == 120 and C.sizeof(_Counters) == 80
assert C.sizeof(_Budget) == 48
assert C.sizeof(_Query) == 40 and C.sizeof(_Recovery) == 80


def _api(lib, prefix):
    """The library's own entry points (addresses of the loaded symbols), as RuntimeCore's function table."""
    return _K1Api(*(C.cast(getattr(lib, prefix + n), C.c_void_p).value for n in _API_ENTRIES))


_TABLES: dict[int, tuple[object, _K1Api]] = {}   # id(loaded library) -> (library, table); libraries are never unloaded


def _table(lib, prefix):
    entry = _TABLES.get(id(lib))
    if entry is None or entry[0] is not lib:
        entry = _TABLES[id(lib)] = (lib, _api(lib, prefix))
    return entry[1]


K1_LIBRARY_IDS = {K1State: "elpis_ecsg_k1", K1FMSState: "elpis_ecsg_k1_fms"}


def _require_admitted(lib, kind):
    """The managed runtime drives only K1 code it can prove: the state's library must have been admitted (sealed,
    pinned) under the K1 library identifier of its kind."""
    admitted = admission_of(lib)
    if admitted is None or admitted.library_id != K1_LIBRARY_IDS[kind]:
        raise CompositionError("ECS_NATIVE_UNADMITTED",
                               "the K1 state's native library was not admitted from sealed, pinned bytes")


def describe(substrate):
    """``(descriptor, owner)`` for a native K1 state; raises ``ECS_STATE`` for anything else or a closed state, and
    ``ECS_NATIVE_UNADMITTED`` for a state whose native library was not admitted."""
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
    except K1Error as exc:
        raise CompositionError("ECS_STATE", str(exc)) from exc
    raise CompositionError("ECS_STATE", "a native K1 state (K1State or K1FMSState) is required")


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
            "turn_commit": [P, C.POINTER(_Substrate), C.POINTER(_Commit), S],
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
            lib.elpis_runtime_testing_checkpoint_crash.argtypes = [P, C.c_uint32]
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
        return _from_c(out)

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

    def query(self, descriptor, stimulus, budget) -> tuple[tuple, bytes]:
        """QUERY (read-only): ``(f_W(x) per row, retained-state identity)`` from the lineage's authoritative state.

        One call into RuntimeCore, one native K1 crossing (after a one-time read of the state's shape); nothing is
        begun, committed or published. RuntimeCore admits the query rows and work units under ``budget`` first.
        """
        rows = stimulus.rows
        out, result = (C.c_double * rows)(), _Query()
        x = (C.c_double * len(stimulus.x)).from_buffer(stimulus.x)
        rc = self._f.elpis_runtime_query(self._handle, C.byref(descriptor), x, len(stimulus.x), out, rows,
                                         C.byref(_Budget(*budget.native_fields())), C.byref(result))
        self.library.check(rc, "", result.k1_status)
        return _unpack_from(f"{rows}d", out), bytes(result.state_digest)

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

    def turn_commit(self, descriptor) -> tuple[CommitIdentity, ContinuitySnapshot]:
        out, snap = _Commit(), _Snapshot()
        rc = self._f.elpis_runtime_turn_commit(self._handle, C.byref(descriptor), C.byref(out), C.byref(snap))
        if rc != 0 and out.committed:
            self.library.check(rc, "K1 turn committed; continuity publication failed")
        self.library.check(rc, "", out.k1_status)
        t = out.identity.transition
        commit = Commit(t.epoch_before, t.epoch_after, t.generation_before, t.generation_after, t.steps)
        identity = CommitIdentity(commit, bytes(out.identity.state_before_digest),
                                  bytes(out.identity.state_after_digest))
        return identity, _from_c(snap)

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

    def testing_checkpoint_crash(self, point: int) -> None:
        """Simulated process death at a checkpointed-LEARN boundary: 0 none, 1 after the candidate checkpoint,
        2 after the native commit (before the continuity publication)."""
        self._require_testing()
        self.library.check(self._f.elpis_runtime_testing_checkpoint_crash(self._handle, point))

    def _require_testing(self) -> None:
        if not self.library.testing:
            raise CompositionError("RUNTIME_INVALID", "fault injection exists only in the testing library")


def _ref(value):
    return C.byref(value) if value is not None else None


def _digest_arg(value):
    if type(value) is not bytes or len(value) != 32:
        raise CompositionError("RUNTIME_INVALID", "a 32-byte retained-state identity is required")
    return (C.c_uint8 * 32).from_buffer_copy(value)
