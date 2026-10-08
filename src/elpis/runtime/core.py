"""Thin Python adapter over RuntimeCore (``native/runtime``, ``include/elpis/runtime.h``).

RuntimeCore (Rust) owns the runtime's mutable systems authority: lifecycle, fail-stop, the continuity store, the
binding of the K1 lineage to one native state, the managed turn's native K1 transaction and its continuity
publication, and the evolution attempt's reservation and finalization. This module only marshals values across
the C ABI and raises the stable code RuntimeCore returns. It holds no authority state, decides no transition and
calls no K1 function: it hands RuntimeCore the K1 library's own entry points and the state's native handle.

The substrate descriptor names the native state, its library's entry points, its declared dimension (verified
natively) and the identity of the Python object that owns the handle (``owner``). The caller keeps that owner
alive while it is bound, so its identity cannot be reused by another state (lifetime only; no decision).

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

from .errors import CompositionError

__all__ = ("RuntimeCore", "RuntimeLibrary", "TurnBegun")

_ABI_VERSION = 2
_TESTING_PROCESS_DEATH = 255
_U64 = C.c_uint64
_K1_NAMES = {-1: "INVALID", -2: "NONFINITE", -3: "STALE", -4: "BUSY", -5: "CAPACITY", -6: "NOMEM", -7: "CORRUPT"}


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


class _Counters(C.Structure):
    _fields_ = [(n, _U64) for n in ("k1_state_digests", "k1_reserves", "k1_txn_begins", "k1_run_schedules",
                                    "k1_commits", "k1_aborts", "publications")]


_API_ENTRIES = ("state_digest", "reserve", "txn_begin", "txn_run_schedule", "txn_commit_identity", "txn_abort")


class _K1Api(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in _API_ENTRIES]


class _Substrate(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("reserved", C.c_uint32), ("handle", C.c_void_p), ("id", _U64),
                ("owner", _U64), ("dim", _U64), ("api", C.c_void_p)]


assert C.sizeof(_Begin) == 48 and C.sizeof(_Commit) == 120 and C.sizeof(_Counters) == 56


def _api(lib, prefix):
    """The library's own entry points (addresses of the loaded symbols), as RuntimeCore's function table."""
    return _K1Api(*(C.cast(getattr(lib, prefix + n), C.c_void_p).value for n in _API_ENTRIES))


_TABLES: dict[int, tuple[object, _K1Api]] = {}   # id(loaded library) -> (library, table); libraries are never unloaded


def _table(lib, prefix):
    entry = _TABLES.get(id(lib))
    if entry is None or entry[0] is not lib:
        entry = _TABLES[id(lib)] = (lib, _api(lib, prefix))
    return entry[1]


def describe(substrate):
    """``(descriptor, owner)`` for a native K1 state; raises ``ECS_STATE`` for anything else or a closed state."""
    try:
        if type(substrate) is K1State:
            table = _table(substrate._lib, "elpis_ecsg_k1_")
            d = _Substrate(1, 0, substrate._live(), 0, id(substrate), substrate.dim, C.addressof(table))
            return d, substrate
        if type(substrate) is K1FMSState:
            runtime = substrate._r
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


class RuntimeLibrary:
    """The loaded RuntimeCore library (explicit path, like every Elpis native library).

    It also exports the continuity C ABI of the store it embeds: ``continuity`` is the record codec over it.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise CompositionError("RUNTIME_PATH", f"runtime library not found: {path}")
        lib = C.CDLL(str(path))
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
            "turn_begin": [P, C.POINTER(_Substrate), D, C.c_size_t, D, C.c_size_t, C.POINTER(_Experience),
                           C.c_size_t, C.c_double, D, C.c_size_t, C.POINTER(_Begin)],
            "turn_commit": [P, C.POINTER(_Substrate), C.POINTER(_Commit), S],
            "turn_abort": [P, C.POINTER(_Substrate)],
            "evolution_authority": [P, S],
            "evolution_reserve": [P, E, P, S],
            "evolution_finalize": [P, P, S],
            "evolution_abandon": [P],
            "evolution_reconcile": [P, E, P, S],
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
        self.path = path
        self.continuity = ContinuityLibrary(path)
        self._lib = lib
        self._features: dict[int, int] = {}

    def features(self, dim: int) -> int:
        """The S3 readout length for an input dimension (a pure native function, asked once per dimension)."""
        n = self._features.get(dim)
        if n is None:
            n = self._features[dim] = int(self._lib.elpis_runtime_features(dim))
        return n

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

    def turn_begin(self, descriptor, stimulus, learning_rate: float) -> TurnBegun:
        features = self.library.features(descriptor.dim)
        s3, out = (C.c_double * max(features, 1))(), _Begin()
        x = (C.c_double * len(stimulus.x)).from_buffer(stimulus.x)
        y = (C.c_double * len(stimulus.y)).from_buffer(stimulus.y)
        schedule = (_Experience * stimulus.experiences).from_buffer(stimulus.schedule)
        rc = self._f.elpis_runtime_turn_begin(self._handle, C.byref(descriptor), x, len(stimulus.x), y,
                                              len(stimulus.y), schedule, stimulus.experiences, learning_rate, s3,
                                              features, C.byref(out))
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

    # -- testing library only -------------------------------------------------------------------------------------------
    def testing_fault(self, publication: int, action: int, arg: int = 0) -> None:
        self._require_testing()
        self.library.check(self._f.elpis_runtime_testing_fault(self._handle, publication, action, arg))

    def testing_io_counters(self, reset: bool = False) -> dict[str, int]:
        self._require_testing()
        out = _IOCounters()
        self.library.check(self._f.elpis_runtime_testing_io_counters(self._handle, C.byref(out), int(reset)))
        return {name: getattr(out, name) for name, _ in _IOCounters._fields_}

    def _require_testing(self) -> None:
        if not self.library.testing:
            raise CompositionError("RUNTIME_INVALID", "fault injection exists only in the testing library")


def _ref(value):
    return C.byref(value) if value is not None else None
