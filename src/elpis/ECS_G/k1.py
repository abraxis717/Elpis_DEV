"""Python control plane for the native K1 consolidation runtime (``ecsg_k1.h``, ``ecsg_k1_fms.h``).

K1 is Retention R3's qualified mechanism (``OUTCOME_A``, docs/research/ECS_RETENTION_R3_RESULTS.md); the runtime and
its contract are documented in docs/ECS_K1_RUNTIME.md. The complete cognitive state is ``(W, epoch, H, a)``; it
lives in native memory (a standalone :class:`K1State`) or as one generic FMS object (:class:`K1FMSRuntime`).

PYTHON MAY CONTROL THE ECS; IT MAY NOT EXECUTE THE HOT PATH. Every method here admits its arguments, makes one
native call and packages the result: a query is one call, a ``K``-step learn is one call (the ``K`` loop, the G1
step, S3, the Jacobian contraction and the ``H`` mat-vec are native), a consolidation is one call, and a
transaction commits ``W``, epoch, ``H`` and ``a`` together natively. Nothing here computes cognitive mathematics.
Like ``elpis.ECS_G.native`` this module imports only the standard library and that binding.
"""
from __future__ import annotations

import ctypes as C
from struct import unpack_from as _unpack_from
from types import SimpleNamespace

from .native import (DEFAULT_MAX_ROWS, Commit, ECSGError, _P, _U8P, _VP, _admit_out, _admit_rows, _admit_vector,
                     _rate, _steps, _view)

__all__ = ("CommitIdentity", "K1Error", "K1FMSRuntime", "K1FMSState", "K1Library", "K1State", "K1Transaction",
           "MAX_EXPERIENCES", "PROVENANCE")

_U64 = C.c_uint64
_CODES = {-1: "INVALID", -2: "NONFINITE", -3: "STALE", -4: "BUSY", -5: "CAPACITY", -6: "NOMEM", -7: "CORRUPT",
          -101: "FMS_INVALID", -102: "NOMEM", -103: "MISSING", -104: "BUSY", -105: "UNSUPPORTED", -106: "IO",
          -107: "CAPACITY", -108: "STATE", -109: "DIGEST", -110: "DEVICE", -111: "TIMEOUT"}
PROVENANCE = {0: "COMPLETE", 1: "RESET", 2: "UNCONSOLIDATED_IMPORT"}
_STALE, _NONFINITE = -3, -2


def _discards(rc, mutating):
    """The refusal contract of ecsg_k1.h: STALE, or NONFINITE from a candidate-mutating call, discards the
    transaction; INVALID, CAPACITY and BUSY leave it open and unchanged (the call may be retried)."""
    return rc == _STALE or (mutating and rc == _NONFINITE)


class K1Error(ECSGError):
    """A native K1 call refused (the complete state is unchanged)."""


def _refused(rc, what, step=0):
    rc = int(rc)
    return K1Error(_CODES.get(rc, "INVALID"), what if not step else f"{what} (step {step} refused)", step=step)


class _Transition(C.Structure):
    _fields_ = [("epoch_before", _U64), ("epoch_after", _U64), ("generation_before", _U64),
                ("generation_after", _U64), ("steps", _U64), ("failed_step", _U64)]


class CommitIdentity(tuple):
    """Immutable committed-K1 transition plus exact retained-state identities."""

    __slots__ = ()

    def __new__(cls, commit, state_before_digest, state_after_digest):
        return tuple.__new__(
            cls,
            (
                commit,
                state_before_digest,
                state_after_digest,
            ),
        )

    @property
    def commit(self):
        return self[0]

    @property
    def state_before_digest(self):
        return self[1]

    @property
    def state_after_digest(self):
        return self[2]


class _CommitIdentity(C.Structure):
    _fields_ = [("transition", _Transition),
                ("state_before_digest", C.c_uint8 * 32),
                ("state_after_digest", C.c_uint8 * 32)]


class _Counters(C.Structure):
    _fields_ = [("workspace_bytes", C.c_size_t), ("max_rows", C.c_size_t), ("heap_allocations", _U64),
                ("forward_calls", _U64), ("learn_calls", _U64), ("steps_executed", _U64),
                ("corrected_steps", _U64), ("consolidations", _U64), ("commits", _U64), ("refusals", _U64),
                ("txn_begins", _U64), ("txn_aborts", _U64), ("stale_refusals", _U64), ("busy_refusals", _U64)]


class _Info(C.Structure):
    _fields_ = [("logical_identity", C.c_uint8 * 32), ("object_handle", _U64), ("epoch", _U64),
                ("generation", _U64), ("dim", C.c_size_t), ("width", C.c_size_t), ("max_rows", C.c_size_t),
                ("image_bytes", C.c_size_t), ("envelope_bytes", C.c_size_t), ("workspace_bytes", C.c_size_t),
                ("acquisitions", _U64), ("lease_failures", _U64), ("commits", _U64), ("aborts", _U64),
                ("materialization_ns", _U64), ("query_ns", _U64), ("learn_ns", _U64), ("consolidate_ns", _U64),
                ("commit_ns", _U64), ("lease_count", C.c_uint32), ("provenance", C.c_uint32), ("tier", C.c_uint8),
                ("residency_state", C.c_uint8), ("cold_replica", C.c_uint8), ("transaction_open", C.c_uint8)]


_TP = C.POINTER(_Transition)
_CIP = C.POINTER(_CommitIdentity)


class _Experience(C.Structure):
    _fields_ = [("rows", _U64), ("steps", _U64)]


class _ScheduleResult(C.Structure):
    _fields_ = [("epoch_before", _U64), ("epoch_after", _U64), ("experiences_applied", _U64),
                ("failed_experience", _U64), ("failed_step", _U64)]


_EP = C.POINTER(_Experience)
_SRP = C.POINTER(_ScheduleResult)
MAX_EXPERIENCES = 64


def _admit_schedule(x, y, schedule, dim):
    """The native-ready experience schedule: X (total rows x dim binary64), y (total rows binary64) and the
    descriptors (a uint64 buffer of (rows, steps) pairs, at most MAX_EXPERIENCES). Contiguous buffers cross as they
    are; nothing is walked in Python. Spans, counts and finiteness are validated natively before any mutation."""
    xs, values = _view(x, "X")
    ys, rows = _view(y, "y")
    if xs is None or ys is None:
        raise K1Error("INVALID", "X and y: C-contiguous binary64 buffers")
    try:
        view = memoryview(schedule)
    except TypeError as exc:
        raise K1Error("INVALID", "schedule: a uint64 buffer of (rows, steps) pairs") from exc
    if view.format not in ("Q", "<Q", "=Q") or view.itemsize != 8 or view.ndim != 1 or not view.c_contiguous:
        raise K1Error("INVALID", "schedule: a uint64 buffer of (rows, steps) pairs")
    pairs = len(view) // 2
    if len(view) != 2 * pairs or not 1 <= pairs <= MAX_EXPERIENCES or values != rows * dim:
        raise K1Error("INVALID", "schedule: 1..64 (rows, steps) pairs over rows x dim inputs")
    descriptors = (_Experience * pairs).from_buffer_copy(view) if view.readonly else \
        (_Experience * pairs).from_buffer(view)
    return xs, ys, rows, descriptors, pairs


def _prepared(result, s3, features):
    return SimpleNamespace(s3=_unpack_from(f"{features}d", s3), epoch_before=int(result.epoch_before),
                           epoch_after=int(result.epoch_after), experiences=int(result.experiences_applied))


def _commit(t):
    return Commit(t.epoch_before, t.epoch_after, t.generation_before, t.generation_after, t.steps)


def _commit_identity(identity):
    return CommitIdentity(
        _commit(identity.transition),
        bytes(identity.state_before_digest),
        bytes(identity.state_after_digest),
    )


def _record(record):
    return {name: (bytes(getattr(record, name)) if isinstance(getattr(record, name), C.Array)
                   else int(getattr(record, name))) for name, _ in record._fields_}


def _bind(lib, prefix, table):
    out = {}
    for name, (args, result) in table.items():
        try:
            fn = getattr(lib, prefix + name)
        except AttributeError as exc:
            raise K1Error("UNSUPPORTED", "K1 symbol " + prefix + name) from exc
        fn.argtypes, fn.restype = args, result
        out[name] = fn
    return SimpleNamespace(**out)


_K1_ABI = {
    "abi_version": ([], C.c_uint32),
    "features": ([C.c_size_t], C.c_size_t),
    "envelope_bytes": ([C.c_size_t, C.c_size_t], C.c_size_t),
    "workspace_bytes": ([C.c_size_t, C.c_size_t, C.c_size_t], C.c_size_t),
    "create": ([C.c_size_t, C.c_size_t, C.c_size_t, _P, C.POINTER(_VP)], C.c_int),
    "restore": ([_U8P, C.c_size_t, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "import_w_only": ([_U8P, C.c_size_t, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "destroy": ([C.POINTER(_VP)], C.c_int),
    "reserve": ([_VP, C.c_size_t], C.c_int),
    "dim": ([_VP], C.c_size_t),
    "width": ([_VP], C.c_size_t),
    "max_rows": ([_VP], C.c_size_t),
    "epoch": ([_VP], _U64),
    "generation": ([_VP], _U64),
    "provenance_of": ([_VP], C.c_uint32),
    "forward": ([_VP, _P, C.c_size_t, _P], C.c_int),
    "learn": ([_VP, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
    "consolidate": ([_VP, _P, C.c_size_t, _TP], C.c_int),
    "reset": ([_VP, _TP], C.c_int),
    "copy_w": ([_VP, _P, C.c_size_t], C.c_int),
    "copy_h_packed": ([_VP, _P, C.c_size_t], C.c_int),
    "copy_a": ([_VP, _P, C.c_size_t], C.c_int),
    "snapshot_size": ([_VP], C.c_size_t),
    "snapshot_write": ([_VP, _U8P, C.c_size_t], C.c_int),
    "state_digest": ([_VP, _U8P], C.c_int),
    "stats": ([_VP, C.POINTER(_Counters)], C.c_int),
    "txn_begin": ([_VP, C.POINTER(_U64)], C.c_int),
    "txn_learn": ([_VP, _U64, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
    "txn_consolidate": ([_VP, _U64, _P, C.c_size_t], C.c_int),
    "txn_forward": ([_VP, _U64, _P, C.c_size_t, _P], C.c_int),
    "txn_epoch": ([_VP, _U64, C.POINTER(_U64)], C.c_int),
    "txn_commit": ([_VP, _U64, _TP], C.c_int),
    "txn_commit_identity": ([_VP, _U64, _CIP], C.c_int),
    "txn_abort": ([_VP, _U64], C.c_int),
    "txn_run_schedule": ([_VP, _U64, _P, _P, C.c_size_t, _EP, C.c_size_t, C.c_double, _P, C.c_size_t, _SRP], C.c_int),
}


class K1Library:
    """Typed view of one loaded ``libelpis_ecsg_k1`` (K1 ABI v1)."""

    __slots__ = ("_k",)

    def __init__(self, lib):
        self._k = _bind(lib, "elpis_ecsg_k1_", _K1_ABI)
        if self._k.abi_version() != 1:
            raise K1Error("UNSUPPORTED", "K1 ABI version")

    def features(self, dim):
        return int(self._k.features(dim))

    def envelope_bytes(self, dim, width):
        return int(self._k.envelope_bytes(dim, width))

    def workspace_bytes(self, dim, width, max_rows=DEFAULT_MAX_ROWS):
        return int(self._k.workspace_bytes(dim, width, max_rows))


def _bytes_view(data, what):
    if type(data) is not bytes or not data:
        raise K1Error("INVALID", what + ": bytes required")
    return (C.c_uint8 * len(data)).from_buffer_copy(data)


class K1State:
    """One standalone native K1 state ``(W, epoch, H, a)``. SINGLE_WRITER (an overlapping call is BUSY)."""

    __slots__ = ("_k", "_handle", "_dim", "_width", "_features")

    def __init__(self, library, handle):
        self._k, self._handle = library._k, handle
        self._dim, self._width = int(self._k.dim(handle)), int(self._k.width(handle))
        self._features = int(self._k.features(self._dim))

    @classmethod
    def create(cls, library, dim, width, initial_w, *, max_rows=DEFAULT_MAX_ROWS):
        """A complete state from W with an empty consolidation (H = 0, a = 0)."""
        if type(library) is not K1Library:
            raise K1Error("INVALID", "a loaded K1Library is required")
        w = _admit_vector(initial_w, dim * width, "initial W")
        handle = _VP()
        rc = library._k.create(dim, width, max_rows, w, C.byref(handle))
        if rc != 0:
            raise _refused(rc, "K1 create")
        return cls(library, handle)

    @classmethod
    def restore(cls, library, envelope, *, max_rows=DEFAULT_MAX_ROWS):
        """A complete state from a retained-state envelope (validated, checksum verified)."""
        handle = _VP()
        rc = library._k.restore(_bytes_view(envelope, "envelope"), len(envelope), max_rows, C.byref(handle))
        if rc != 0:
            raise _refused(rc, "K1 restore")
        return cls(library, handle)

    @classmethod
    def import_w_only(cls, library, snapshot, *, max_rows=DEFAULT_MAX_ROWS):
        """A canonical ELPISG01 W-only snapshot, admitted only as an UNCONSOLIDATED state (H = 0, a = 0)."""
        handle = _VP()
        rc = library._k.import_w_only(_bytes_view(snapshot, "snapshot"), len(snapshot), max_rows, C.byref(handle))
        if rc != 0:
            raise _refused(rc, "K1 W-only import")
        return cls(library, handle)

    def _live(self):
        if not self._handle:
            raise K1Error("INVALID", "K1 state closed")
        return self._handle

    @property
    def dim(self):
        return self._dim

    @property
    def width(self):
        return self._width

    @property
    def max_rows(self):
        return int(self._k.max_rows(self._live()))

    @property
    def epoch(self):
        return int(self._k.epoch(self._live()))

    @property
    def generation(self):
        return int(self._k.generation(self._live()))

    @property
    def provenance(self):
        return PROVENANCE[int(self._k.provenance_of(self._live()))]

    def reserve(self, max_rows):
        rc = self._k.reserve(self._live(), max_rows)
        if rc != 0:
            raise _refused(rc, "K1 reserve")

    def query(self, x_rows):
        """f_W(X): one native call; reads W only."""
        x, rows = _admit_rows(x_rows, self._dim, "X")
        out = (C.c_double * rows)()
        rc = self._k.forward(self._live(), x, rows, out)
        if rc != 0:
            raise _refused(rc, "K1 query")
        return _unpack_from(f"{rows}d", out)

    def query_into(self, x, out):
        x, rows = _admit_rows(x, self._dim, "X")
        rc = self._k.forward(self._live(), x, rows, _admit_out(out, rows, "out"))
        if rc != 0:
            raise _refused(rc, "K1 query")
        return rows

    def learn(self, x_rows, y, learning_rate, steps=1):
        """``steps`` K1 steps (G1 + the K1 correction) in one native call, committed as one transition."""
        x, rows = _admit_rows(x_rows, self._dim, "X")
        t = _Transition()
        rc = self._k.learn(self._live(), x, _admit_vector(y, rows, "y"), rows, _rate(learning_rate), _steps(steps),
                           C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 learn", int(t.failed_step))
        return _commit(t)

    def consolidate(self, x_rows):
        """H <- H + Sigma(X), a <- S3(W): inputs only, one native call."""
        x, rows = _admit_rows(x_rows, self._dim, "X")
        t = _Transition()
        rc = self._k.consolidate(self._live(), x, rows, C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 consolidate")
        return _commit(t)

    def reset(self):
        """H <- 0, a <- 0; W and the epoch kept (provenance RESET)."""
        t = _Transition()
        rc = self._k.reset(self._live(), C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 reset")
        return _commit(t)

    def transaction(self):
        token = _U64()
        rc = self._k.txn_begin(self._live(), C.byref(token))
        if rc != 0:
            raise _refused(rc, "K1 transaction begin")
        return K1Transaction(self, token.value)

    def _copy(self, fn, count, what):
        out = (C.c_double * count)()
        rc = fn(self._live(), out, count)
        if rc != 0:
            raise _refused(rc, what)
        return tuple(out)

    def w(self):
        return self._copy(self._k.copy_w, self._dim * self._width, "K1 copy W")

    def h_packed(self):
        f = int(self._k.features(self._dim))
        return self._copy(self._k.copy_h_packed, f * (f + 1) // 2, "K1 copy H")

    def a(self):
        return self._copy(self._k.copy_a, int(self._k.features(self._dim)), "K1 copy a")

    def snapshot(self):
        """The portable retained-state envelope (ELPISGK1 v1, little-endian, SHA-256 trailer)."""
        size = int(self._k.snapshot_size(self._live()))
        out = (C.c_uint8 * size)()
        rc = self._k.snapshot_write(self._live(), out, size)
        if rc != 0:
            raise _refused(rc, "K1 snapshot")
        return bytes(out)

    def state_digest(self):
        out = (C.c_uint8 * 32)()
        rc = self._k.state_digest(self._live(), out)
        if rc != 0:
            raise _refused(rc, "K1 state digest")
        return bytes(out)

    def stats(self):
        s = _Counters()
        rc = self._k.stats(self._live(), C.byref(s))
        if rc != 0:
            raise _refused(rc, "K1 stats")
        return _record(s)

    def close(self):
        if self._handle:
            rc = self._k.destroy(C.byref(self._handle))
            if rc != 0:
                raise _refused(rc, "K1 destroy")
            self._handle = _VP()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class K1Transaction:
    """A native candidate of the complete state; ``commit`` installs W, epoch, H and a together, or nothing."""

    __slots__ = ("_state", "_token", "_open")

    def __init__(self, state, token):
        self._state, self._token, self._open = state, token, True

    def _live(self):
        if not self._open:
            raise K1Error("INVALID", "K1 transaction closed")
        return self._state._live()

    def _settle(self, rc, what, step=0, mutating=False):
        if rc != 0:
            if _discards(rc, mutating):
                self._open = False
            raise _refused(rc, what, step)

    def learn(self, x_rows, y, learning_rate, steps=1):
        k = self._state._k
        x, rows = _admit_rows(x_rows, self._state._dim, "X")
        t = _Transition()
        rc = k.txn_learn(self._live(), self._token, x, _admit_vector(y, rows, "y"), rows, _rate(learning_rate),
                         _steps(steps), C.byref(t))
        self._settle(rc, "K1 transaction learn", int(t.failed_step), mutating=True)
        return _commit(t)

    def consolidate(self, x_rows):
        x, rows = _admit_rows(x_rows, self._state._dim, "X")
        self._settle(self._state._k.txn_consolidate(self._live(), self._token, x, rows), "K1 transaction consolidate",
                     mutating=True)

    def query(self, x_rows):
        x, rows = _admit_rows(x_rows, self._state._dim, "X")
        out = (C.c_double * rows)()
        self._settle(self._state._k.txn_forward(self._live(), self._token, x, rows, out), "K1 transaction query")
        return _unpack_from(f"{rows}d", out)

    def run_schedule(self, x, y, schedule, learning_rate):
        """The ordered experience schedule on the candidate, one native call: per experience, its K1 steps then the
        consolidation of its rows; returns S3 of the final candidate W (the readout) and the candidate epochs."""
        state = self._state
        xs, ys, rows, descriptors, pairs = _admit_schedule(x, y, schedule, state._dim)
        s3, result = (C.c_double * state._features)(), _ScheduleResult()
        rc = state._k.txn_run_schedule(self._live(), self._token, xs, ys, rows, descriptors, pairs,
                                       _rate(learning_rate), s3, state._features, C.byref(result))
        self._settle(rc, "K1 transaction schedule", int(result.failed_step), mutating=True)
        return _prepared(result, s3, state._features)

    def epoch(self):
        value = _U64()
        self._settle(self._state._k.txn_epoch(self._live(), self._token, C.byref(value)), "K1 transaction epoch")
        return int(value.value)

    def commit(self):
        t = _Transition()
        rc = self._state._k.txn_commit(self._live(), self._token, C.byref(t))
        if rc == 0 or _discards(rc, False):
            self._open = False
        if rc != 0:
            raise _refused(rc, "K1 transaction commit")
        return _commit(t)


    def commit_identity(self):
        """Commit once and return transition metadata plus exact retained-state identities."""
        identity = _CommitIdentity()
        rc = self._state._k.txn_commit_identity(self._live(), self._token, C.byref(identity))
        if rc == 0 or _discards(rc, False):
            self._open = False
        if rc != 0:
            raise _refused(rc, "K1 transaction commit identity")
        return _commit_identity(identity)

    def abort(self):
        if self._open:
            self._open = False
            rc = self._state._k.txn_abort(self._state._live(), self._token)
            if rc != 0:
                raise _refused(rc, "K1 transaction abort")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()


_FMS_ABI = {
    "abi_version": ([], C.c_uint32),
    "create": ([_VP, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "destroy": ([C.POINTER(_VP)], C.c_int),
    "register": ([_VP, _U8P, C.c_size_t, C.c_size_t, C.c_size_t, _P, C.POINTER(_U64)], C.c_int),
    "restore": ([_VP, _U8P, _U8P, C.c_size_t, C.c_size_t, C.POINTER(_U64)], C.c_int),
    "import_w_only": ([_VP, _U8P, _U8P, C.c_size_t, C.c_size_t, C.POINTER(_U64)], C.c_int),
    "close": ([_VP, C.POINTER(_U64)], C.c_int),
    "inspect": ([_VP, _U64, C.POINTER(_Info)], C.c_int),
    "k1_stats": ([_VP, _U64, C.POINTER(_Counters)], C.c_int),
    "pump": ([_VP], C.c_int),
    "reserve": ([_VP, _U64, C.c_size_t], C.c_int),
    "forward": ([_VP, _U64, _P, C.c_size_t, _P], C.c_int),
    "learn": ([_VP, _U64, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
    "consolidate": ([_VP, _U64, _P, C.c_size_t, _TP], C.c_int),
    "reset": ([_VP, _U64, _TP], C.c_int),
    "copy_w": ([_VP, _U64, _P, C.c_size_t], C.c_int),
    "snapshot_write": ([_VP, _U64, _U8P, C.c_size_t], C.c_int),
    "state_digest": ([_VP, _U64, _U8P], C.c_int),
    "txn_begin": ([_VP, _U64, C.POINTER(_U64)], C.c_int),
    "txn_learn": ([_VP, _U64, _U64, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
    "txn_consolidate": ([_VP, _U64, _U64, _P, C.c_size_t], C.c_int),
    "txn_forward": ([_VP, _U64, _U64, _P, C.c_size_t, _P], C.c_int),
    "txn_commit": ([_VP, _U64, _U64, _TP], C.c_int),
    "txn_commit_identity": ([_VP, _U64, _U64, _CIP], C.c_int),
    "txn_abort": ([_VP, _U64, _U64], C.c_int),
    "txn_run_schedule": ([_VP, _U64, _U64, _P, _P, C.c_size_t, _EP, C.c_size_t, C.c_double, _P, C.c_size_t, _SRP],
                         C.c_int),
}


def _key(key):
    if type(key) is not bytes or len(key) != 32:
        raise K1Error("INVALID", "logical key: 32 bytes")
    return (C.c_uint8 * 32).from_buffer_copy(key)


class K1FMSRuntime:
    """K1 states materialized by generic FMS: pin -> native K1 over the resident bytes -> unpin.

    ``context`` is an owned, empty FMS context handle (``context.handle``) created by the caller; the runtime takes
    ownership of it. ``library`` is the loaded ``libelpis_ecsg_k1_fms``. States are numeric logical ids.
    """

    def __init__(self, k1_library, context, library, *, max_states):
        if type(k1_library) is not K1Library or not getattr(context, "handle", None) or not context.handle.value:
            raise K1Error("INVALID", "loaded K1 library and owned FMS context required")
        if type(max_states) is not int or not 1 <= max_states < 1 << 32:
            raise K1Error("INVALID", "state capacity")
        self._k1 = k1_library
        self._f = _bind(library, "elpis_ecsg_k1_fms_", _FMS_ABI)
        if self._f.abi_version() != 1:
            raise K1Error("UNSUPPORTED", "K1 FMS ABI")
        self._handle = _VP()
        rc = self._f.create(context.handle, max_states, C.byref(self._handle))
        if rc != 0:
            raise _refused(rc, "K1 FMS runtime create")
        context.handle.value = None
        self._shape = {}

    def _live(self):
        if not self._handle:
            raise K1Error("INVALID", "K1 FMS runtime closed")
        return self._handle

    def register(self, key, dim, width, initial_w, *, max_rows=DEFAULT_MAX_ROWS):
        out = _U64()
        rc = self._f.register(self._live(), _key(key), dim, width, max_rows,
                              _admit_vector(initial_w, dim * width, "initial W"), C.byref(out))
        if rc != 0:
            raise _refused(rc, "K1 FMS register")
        self._shape[out.value] = (dim, width, self._k1.features(dim), max_rows)
        return int(out.value)

    def _install(self, fn, key, data, what, max_rows):
        out = _U64()
        rc = fn(self._live(), _key(key), _bytes_view(data, what), len(data), max_rows, C.byref(out))
        if rc != 0:
            raise _refused(rc, "K1 FMS " + what)
        info = self.inspect(out.value)
        self._shape[out.value] = (info["dim"], info["width"], self._k1.features(info["dim"]), info["max_rows"])
        return int(out.value)

    def restore(self, key, envelope, *, max_rows=DEFAULT_MAX_ROWS):
        return self._install(self._f.restore, key, envelope, "restore", max_rows)

    def import_w_only(self, key, snapshot, *, max_rows=DEFAULT_MAX_ROWS):
        return self._install(self._f.import_w_only, key, snapshot, "W-only import", max_rows)

    def close_state(self, state_id):
        sid = _U64(state_id)
        rc = self._f.close(self._live(), C.byref(sid))
        if rc != 0:
            raise _refused(rc, "K1 FMS close")
        self._shape.pop(state_id, None)

    def _dim(self, state_id):
        try:
            return self._shape[state_id][0]
        except KeyError as exc:
            raise K1Error("INVALID", "unknown K1 state") from exc

    def inspect(self, state_id):
        info = _Info()
        rc = self._f.inspect(self._live(), state_id, C.byref(info))
        if rc != 0:
            raise _refused(rc, "K1 FMS inspect")
        out = _record(info)
        out["provenance"] = PROVENANCE[out["provenance"]]
        return out

    def k1_stats(self, state_id):
        s = _Counters()
        rc = self._f.k1_stats(self._live(), state_id, C.byref(s))
        if rc != 0:
            raise _refused(rc, "K1 FMS stats")
        return _record(s)

    def pump(self):
        rc = self._f.pump(self._live())
        if rc != 0:
            raise _refused(rc, "K1 FMS pump")

    def query(self, state_id, x_rows):
        x, rows = _admit_rows(x_rows, self._dim(state_id), "X")
        out = (C.c_double * rows)()
        rc = self._f.forward(self._live(), state_id, x, rows, out)
        if rc != 0:
            raise _refused(rc, "K1 FMS query")
        return _unpack_from(f"{rows}d", out)

    def query_into(self, state_id, x, out):
        x, rows = _admit_rows(x, self._dim(state_id), "X")
        rc = self._f.forward(self._live(), state_id, x, rows, _admit_out(out, rows, "out"))
        if rc != 0:
            raise _refused(rc, "K1 FMS query")
        return rows

    def learn(self, state_id, x_rows, y, learning_rate, steps=1):
        x, rows = _admit_rows(x_rows, self._dim(state_id), "X")
        t = _Transition()
        rc = self._f.learn(self._live(), state_id, x, _admit_vector(y, rows, "y"), rows, _rate(learning_rate),
                           _steps(steps), C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 FMS learn", int(t.failed_step))
        return _commit(t)

    def consolidate(self, state_id, x_rows):
        x, rows = _admit_rows(x_rows, self._dim(state_id), "X")
        t = _Transition()
        rc = self._f.consolidate(self._live(), state_id, x, rows, C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 FMS consolidate")
        return _commit(t)

    def reset(self, state_id):
        t = _Transition()
        rc = self._f.reset(self._live(), state_id, C.byref(t))
        if rc != 0:
            raise _refused(rc, "K1 FMS reset")
        return _commit(t)

    def snapshot(self, state_id):
        size = int(self.inspect(state_id)["envelope_bytes"])
        out = (C.c_uint8 * size)()
        rc = self._f.snapshot_write(self._live(), state_id, out, size)
        if rc != 0:
            raise _refused(rc, "K1 FMS snapshot")
        return bytes(out)

    def state_digest(self, state_id):
        out = (C.c_uint8 * 32)()
        rc = self._f.state_digest(self._live(), state_id, out)
        if rc != 0:
            raise _refused(rc, "K1 FMS state digest")
        return bytes(out)

    def reserve(self, state_id, max_rows):
        """Grow one state's admitted row capacity (cold path; refused while a transaction is open)."""
        rc = self._f.reserve(self._live(), state_id, max_rows)
        if rc != 0:
            raise _refused(rc, "K1 FMS reserve")
        dim, width, features, current = self._shape[state_id]
        self._shape[state_id] = (dim, width, features, max(current, max_rows))

    def state(self, state_id):
        """A typed handle on one resident state (for the canonical turn and other control code)."""
        return K1FMSState(self, state_id)

    def transaction(self, state_id):
        token = _U64()
        rc = self._f.txn_begin(self._live(), state_id, C.byref(token))
        if rc != 0:
            raise _refused(rc, "K1 FMS transaction begin")
        return _FMSTransaction(self, state_id, token.value)

    def close(self):
        if self._handle:
            rc = self._f.destroy(C.byref(self._handle))
            if rc != 0:
                raise _refused(rc, "K1 FMS runtime destroy")
            self._handle = _VP()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _FMSTransaction:
    """A K1 candidate under a held WRITE pin; ``commit`` installs W, epoch, H and a together."""

    __slots__ = ("_r", "_id", "_token", "_open")

    def __init__(self, runtime, state_id, token):
        self._r, self._id, self._token, self._open = runtime, state_id, token, True

    def _settle(self, rc, what, step=0, mutating=False):
        if rc != 0:
            if _discards(rc, mutating):
                self._open = False
            raise _refused(rc, what, step)

    def learn(self, x_rows, y, learning_rate, steps=1):
        x, rows = _admit_rows(x_rows, self._r._dim(self._id), "X")
        t = _Transition()
        rc = self._r._f.txn_learn(self._r._live(), self._id, self._token, x, _admit_vector(y, rows, "y"), rows,
                                  _rate(learning_rate), _steps(steps), C.byref(t))
        self._settle(rc, "K1 FMS transaction learn", int(t.failed_step), mutating=True)
        return _commit(t)

    def consolidate(self, x_rows):
        x, rows = _admit_rows(x_rows, self._r._dim(self._id), "X")
        self._settle(self._r._f.txn_consolidate(self._r._live(), self._id, self._token, x, rows),
                     "K1 FMS transaction consolidate", mutating=True)

    def query(self, x_rows):
        x, rows = _admit_rows(x_rows, self._r._dim(self._id), "X")
        out = (C.c_double * rows)()
        self._settle(self._r._f.txn_forward(self._r._live(), self._id, self._token, x, rows, out),
                     "K1 FMS transaction query")
        return _unpack_from(f"{rows}d", out)

    def run_schedule(self, x, y, schedule, learning_rate):
        """The experience schedule on the resident candidate, one native call (see K1Transaction.run_schedule)."""
        dim, _, features, _ = self._r._shape[self._id]
        xs, ys, rows, descriptors, pairs = _admit_schedule(x, y, schedule, dim)
        s3, result = (C.c_double * features)(), _ScheduleResult()
        rc = self._r._f.txn_run_schedule(self._r._live(), self._id, self._token, xs, ys, rows, descriptors, pairs,
                                         _rate(learning_rate), s3, features, C.byref(result))
        self._settle(rc, "K1 FMS transaction schedule", int(result.failed_step), mutating=True)
        return _prepared(result, s3, features)

    def commit(self):
        t = _Transition()
        rc = self._r._f.txn_commit(self._r._live(), self._id, self._token, C.byref(t))
        if rc == 0 or _discards(rc, False):
            self._open = False
        if rc != 0:
            raise _refused(rc, "K1 FMS transaction commit")
        return _commit(t)


    def commit_identity(self):
        """Commit resident K1 once under its WRITE pin and return exact retained-state identities."""
        identity = _CommitIdentity()
        rc = self._r._f.txn_commit_identity(
            self._r._live(), self._id, self._token, C.byref(identity)
        )
        if rc == 0 or _discards(rc, False):
            self._open = False
        if rc != 0:
            raise _refused(rc, "K1 FMS transaction commit identity")
        return _commit_identity(identity)

    def abort(self):
        if self._open:
            self._open = False
            rc = self._r._f.txn_abort(self._r._live(), self._id, self._token)
            if rc != 0:
                raise _refused(rc, "K1 FMS transaction abort")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()


class K1FMSState:
    """A typed handle on one FMS-resident K1 state: its runtime and logical id. It holds no cognitive state and no
    mathematics; every operation is the runtime's (pin -> native K1 over the resident bytes -> unpin)."""

    __slots__ = ("_r", "_id")

    def __init__(self, runtime, state_id):
        if type(runtime) is not K1FMSRuntime or state_id not in runtime._shape:
            raise K1Error("INVALID", "a K1FMSRuntime and one of its registered state ids are required")
        self._r, self._id = runtime, state_id

    @property
    def id(self):
        return self._id

    @property
    def dim(self):
        return self._r._shape[self._id][0]

    @property
    def width(self):
        return self._r._shape[self._id][1]

    @property
    def max_rows(self):
        return self._r._shape[self._id][3]

    @property
    def epoch(self):
        return self._r.inspect(self._id)["epoch"]

    @property
    def provenance(self):
        return self._r.inspect(self._id)["provenance"]

    def reserve(self, max_rows):
        self._r.reserve(self._id, max_rows)

    def transaction(self):
        return self._r.transaction(self._id)

    def query(self, x_rows):
        return self._r.query(self._id, x_rows)

    def snapshot(self):
        return self._r.snapshot(self._id)

    def state_digest(self):
        return self._r.state_digest(self._id)
