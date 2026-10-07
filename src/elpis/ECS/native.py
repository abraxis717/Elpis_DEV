"""Python binding to the native ECS kernel (``native/ECS``).

The binding owns no loading policy: the caller passes an already loaded
``libelpis_ecsg_math`` handle (for example one opened through the substrate's
sealed loader by the runtime). ECS therefore keeps its declared dependency
set empty: this module imports only the standard library, never numpy,
inference, runtime or continuity.

Two native surfaces, one mathematics (it lives in C; nothing here
re-implements it):

* :class:`Executor` is the runtime form of an ECS state
  (``ecsg_executor.h``, docs/ECS_RUNTIME_R1.md). Python controls it and never
  executes its hot path: a query is one native call, a ``K``-step learn is one
  native call (the ``K`` loop is native), and a candidate commits natively by
  pointer exchange, refused as ``STALE`` when its source state was replaced
  (a generation check, no hashing). Contiguous binary64 buffers
  (``array('d')``, ``memoryview``) cross without copying or per-element
  work; lists and tuples are a convenience converted once, in C.
  SINGLE_WRITER: calls on one executor must be serialized; an overlapping
  call is refused with ``BUSY``.
* :class:`WorldState` is the scalar reference state (``ecsg_state.h``): owned
  ``W[dim, width]``, epoch, forward map, ``S3``, one explicit atomic gradient
  step, snapshot/restore. It is the correctness authority the executor is
  bitwise-equal to, not a runtime path.

Snapshots of both are the same bytes and restore into either.
"""
from __future__ import annotations

import ctypes as C
from itertools import chain, repeat
import math
from struct import error as _StructError, pack as _pack, unpack_from as _unpack_from
from typing import NamedTuple

__all__ = ("Commit", "ECSGError", "ECSGLibrary", "Executor", "Transaction", "WorldState")

_OK, _INVALID, _NONFINITE = 0, -1, -2
_CODES = {-1: "INVALID", -2: "NONFINITE", -3: "STALE", -4: "BUSY", -5: "CAPACITY", -6: "NOMEM"}
_MAX_FORWARD_ROWS = 1 << 16
DEFAULT_MAX_ROWS = 256
_NUMBERS = frozenset((float, int))
_ROWS = frozenset((tuple, list))
_P = C.POINTER(C.c_double)
_U8P = C.POINTER(C.c_uint8)
_VP = C.c_void_p


class ECSGError(ValueError):
    """A native ECS call refused its input (state is unchanged)."""

    def __init__(self, code: str, detail: str = "", *, step: int = 0):
        self.code, self.step = code, step
        super().__init__(f"{code}: {detail}" if detail else code)


def _check(rc, what):
    if rc != _OK:
        raise ECSGError("NONFINITE" if rc == _NONFINITE else "INVALID", what)


def _refused(rc, what, step=0):
    return ECSGError(_CODES.get(rc, "INVALID"), what if not step else f"{what} (step {step} refused)", step=step)


def _doubles(values, count, what):
    values = tuple(values)
    if len(values) != count or not all(type(v) in (float, int) and math.isfinite(v) for v in values):
        raise ECSGError("INVALID", what)
    return (C.c_double * count)(*values)


class _Transition(C.Structure):
    _fields_ = [("epoch_before", C.c_uint64), ("epoch_after", C.c_uint64),
                ("generation_before", C.c_uint64), ("generation_after", C.c_uint64),
                ("steps", C.c_uint64), ("failed_step", C.c_uint64)]


class _Drive(C.Structure):
    _fields_ = [("rows", C.c_size_t), ("steps", C.c_uint64)]


class _Stats(C.Structure):
    _fields_ = [("workspace_bytes", C.c_size_t), ("max_rows", C.c_size_t), ("heap_allocations", C.c_uint64),
                ("forward_calls", C.c_uint64), ("learn_calls", C.c_uint64), ("steps_executed", C.c_uint64),
                ("commits", C.c_uint64), ("refusals", C.c_uint64), ("txn_begins", C.c_uint64),
                ("txn_aborts", C.c_uint64), ("stale_refusals", C.c_uint64), ("busy_refusals", C.c_uint64)]


_STATE_ABI = {
    "elpis_ecsg_math_abi_version": ([], C.c_uint32),
    "elpis_ecsg_state_abi_version": ([], C.c_uint32),
    "elpis_ecsg_s3_size": ([C.c_size_t], C.c_size_t),
    "elpis_ecsg_symmetric2_size": ([C.c_size_t], C.c_size_t),
    "elpis_ecsg_symmetric3_size": ([C.c_size_t], C.c_size_t),
    "elpis_ecsg_state_create": ([C.c_size_t, C.c_size_t, _P, C.POINTER(_VP)], C.c_int),
    "elpis_ecsg_state_destroy": ([C.POINTER(_VP)], C.c_int),
    "elpis_ecsg_state_dim": ([_VP], C.c_size_t),
    "elpis_ecsg_state_width": ([_VP], C.c_size_t),
    "elpis_ecsg_state_epoch": ([_VP], C.c_uint64),
    "elpis_ecsg_state_copy_w": ([_VP, _P, C.c_size_t], C.c_int),
    "elpis_ecsg_state_project_s3_f64": ([_VP, _P, _P, _P], C.c_int),
    "elpis_ecsg_state_forward_f64": ([_VP, _P, C.c_size_t, _P], C.c_int),
    "elpis_ecsg_state_gd_step_scratch_f64": ([C.c_size_t, C.c_size_t, C.c_size_t], C.c_size_t),
    "elpis_ecsg_state_gd_step_f64": ([_VP, _P, _P, C.c_size_t, C.c_double, _P, C.c_size_t], C.c_int),
    "elpis_ecsg_state_snapshot_size": ([_VP], C.c_size_t),
    "elpis_ecsg_state_snapshot_write": ([_VP, _U8P, C.c_size_t], C.c_int),
    "elpis_ecsg_state_snapshot_restore": ([_U8P, C.c_size_t, C.POINTER(_VP)], C.c_int),
}

# Executor ABI v1: attribute name -> (symbol suffix, argtypes, restype).
_TP = C.POINTER(_Transition)
_EXECUTOR_ABI = {
    "abi_version": ("abi_version", [], C.c_uint32),
    "workspace_bytes": ("workspace_bytes", [C.c_size_t, C.c_size_t, C.c_size_t], C.c_size_t),
    "create": ("create", [C.c_size_t, C.c_size_t, C.c_size_t, _P, C.POINTER(_VP)], C.c_int),
    "restore": ("restore", [_U8P, C.c_size_t, C.c_size_t, C.POINTER(_VP)], C.c_int),
    "destroy": ("destroy", [C.POINTER(_VP)], C.c_int),
    "reserve": ("reserve", [_VP, C.c_size_t], C.c_int),
    "dim": ("dim", [_VP], C.c_size_t),
    "width": ("width", [_VP], C.c_size_t),
    "max_rows": ("max_rows", [_VP], C.c_size_t),
    "epoch": ("epoch", [_VP], C.c_uint64),
    "generation": ("generation", [_VP], C.c_uint64),
    "forward": ("forward", [_VP, _P, C.c_size_t, _P], C.c_int),
    "learn": ("learn", [_VP, _P, _P, C.c_size_t, C.c_double, C.c_uint64, _TP], C.c_int),
    "learn_schedule": ("learn_schedule", [_VP, _P, _P, C.POINTER(_Drive), C.c_size_t, C.c_double, _TP], C.c_int),
    "copy_w": ("copy_w", [_VP, _P, C.c_size_t], C.c_int),
    "project_s3": ("project_s3", [_VP, _P, _P, _P], C.c_int),
    "snapshot_size": ("snapshot_size", [_VP], C.c_size_t),
    "snapshot_write": ("snapshot_write", [_VP, _U8P, C.c_size_t], C.c_int),
    "stats": ("stats", [_VP, C.POINTER(_Stats)], C.c_int),
    "txn_begin": ("txn_begin", [_VP, C.POINTER(C.c_uint64)], C.c_int),
    "txn_learn": ("txn_learn", [_VP, C.c_uint64, _P, _P, C.c_size_t, C.c_double, C.c_uint64, _TP], C.c_int),
    "txn_learn_schedule": ("txn_learn_schedule",
                           [_VP, C.c_uint64, _P, _P, C.POINTER(_Drive), C.c_size_t, C.c_double, _TP], C.c_int),
    "txn_forward": ("txn_forward", [_VP, C.c_uint64, _P, C.c_size_t, _P], C.c_int),
    "txn_project_s3": ("txn_project_s3", [_VP, C.c_uint64, _P, _P, _P], C.c_int),
    "txn_epoch": ("txn_epoch", [_VP, C.c_uint64, C.POINTER(C.c_uint64)], C.c_int),
    "txn_commit": ("txn_commit", [_VP, C.c_uint64, _TP], C.c_int),
    "txn_abort": ("txn_abort", [_VP, C.c_uint64], C.c_int),
}


class _ExecutorABI:
    """The executor entry points, resolved and typed once."""

    __slots__ = tuple(_EXECUTOR_ABI)


class ECSGLibrary:
    """Typed view of one loaded ECS library (math v1, state v1, executor v1)."""

    __slots__ = ("_lib", "_x")

    def __init__(self, lib):
        def bind(name, args, result):
            try:
                fn = getattr(lib, name)
            except AttributeError as exc:
                raise ECSGError("UNSUPPORTED", "ECS symbol " + name) from exc
            fn.argtypes, fn.restype = args, result
            return fn

        for name, (args, result) in _STATE_ABI.items():
            bind(name, args, result)
        x = _ExecutorABI()
        for attr, (suffix, args, result) in _EXECUTOR_ABI.items():
            setattr(x, attr, bind("elpis_ecsg_executor_" + suffix, args, result))
        if (lib.elpis_ecsg_math_abi_version() != 1 or lib.elpis_ecsg_state_abi_version() != 1
                or x.abi_version() != 1):
            raise ECSGError("UNSUPPORTED", "ECS ABI version")
        self._lib, self._x = lib, x

    def s3_size(self, dim):
        return int(self._lib.elpis_ecsg_s3_size(dim))

    def workspace_bytes(self, dim, width, max_rows=DEFAULT_MAX_ROWS):
        """Native workspace an executor of this shape owns (0 if the shape is invalid)."""
        return int(self._x.workspace_bytes(dim, width, max_rows))


# --- admission: validate the top-level shape, hand native code a contiguous binary64 view --------------------


def _view(values, what, dim=None):
    """Zero-copy ctypes view of a C-contiguous binary64 buffer, and its element count.

    Vectors are one-dimensional; rows may also be a ``rows x dim`` buffer.
    """
    try:
        view = memoryview(values)
    except TypeError:
        return None, 0
    if view.format != "d" or not view.c_contiguous or not (
            view.ndim == 1 or (dim is not None and view.ndim == 2 and view.shape[1] == dim)):
        raise ECSGError("INVALID", what + ": a C-contiguous binary64 buffer is required")
    count = view.nbytes // 8
    if view.readonly:
        return (C.c_double * count).from_buffer_copy(view), count
    return (C.c_double * count).from_buffer(view), count


def _convert(values, what):
    """A convenience list of numbers (float and int only, as before), packed once in C."""
    if not set(map(type, values)) <= _NUMBERS:
        raise ECSGError("INVALID", what + ": numbers")
    try:
        packed = _pack(f"{len(values)}d", *values)
    except (_StructError, OverflowError) as exc:
        raise ECSGError("INVALID", what) from exc
    return (C.c_double * len(values)).from_buffer_copy(packed)


def _admit_rows(x_rows, dim, what):
    """X as a native row-major binary64 view and its row count.

    A buffer (``array('d')``, a ``memoryview`` of doubles, flat or ``rows x dim``)
    crosses as is; a sequence of rows of numbers is converted once, in C.
    """
    if type(x_rows) not in _ROWS:
        x, count = _view(x_rows, what, dim)
        if x is not None:
            rows = count // dim if dim else 0
            if rows < 1 or rows * dim != count:
                raise ECSGError("INVALID", what + ": rows x dim values")
            return x, rows
        try:
            x_rows = tuple(x_rows)
        except TypeError as exc:
            raise ECSGError("INVALID", what) from exc
    if not x_rows or not set(map(type, x_rows)) <= _ROWS or set(map(len, x_rows)) != {dim}:
        raise ECSGError("INVALID", what + ": rows of dim numbers")
    return _convert(list(chain.from_iterable(x_rows)), what), len(x_rows)


def _admit_vector(values, count, what):
    """A length-``count`` binary64 vector as a native view (buffer as is, or a sequence converted in C)."""
    if type(values) not in _ROWS:
        v, n = _view(values, what)
        if v is not None:
            if n != count:
                raise ECSGError("INVALID", what + ": length")
            return v
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ECSGError("INVALID", what) from exc
    if len(values) != count:
        raise ECSGError("INVALID", what + ": length")
    return _convert(values, what)


def _admit_out(out, rows, what):
    """A caller-owned writable binary64 buffer with room for ``rows`` values."""
    try:
        view = memoryview(out)
    except TypeError as exc:
        raise ECSGError("INVALID", what + ": a writable binary64 buffer is required") from exc
    if view.format != "d" or view.readonly or not view.c_contiguous or view.nbytes // 8 < rows:
        raise ECSGError("INVALID", what + ": a writable binary64 buffer with room for every row")
    return (C.c_double * (view.nbytes // 8)).from_buffer(view)


def _admit_drives(drive_rows, steps, total, what):
    """An ordered schedule: drive j covers drive_rows[j] consecutive rows, each for ``steps`` steps."""
    try:
        counts = tuple(drive_rows)
    except TypeError as exc:
        raise ECSGError("INVALID", what) from exc
    if not counts or not set(map(type, counts)) <= {int} or min(counts) < 1 or sum(counts) != total:
        raise ECSGError("INVALID", what + ": positive row counts covering every row")
    return (_Drive * len(counts))(*zip(counts, repeat(steps))), len(counts)


def _steps(steps):
    if type(steps) is not int or not 1 <= steps < 1 << 64:
        raise ECSGError("INVALID", "steps must be a positive int")
    return steps


def _rate(learning_rate):
    if type(learning_rate) is not float:
        raise ECSGError("INVALID", "learning rate must be a float")
    return learning_rate


class Commit(NamedTuple):
    """What a native transition did: epochs, commit generations and steps."""
    epoch_before: int
    epoch_after: int
    generation_before: int
    generation_after: int
    steps: int


def _commit(t):
    return Commit(t.epoch_before, t.epoch_after, t.generation_before, t.generation_after, t.steps)


class Executor:
    """One native ECS executor (ecsg_executor.h): the runtime form of an ECS state.

    Owns, natively, the authoritative ``W``, staging and candidate buffers,
    scratch and the admitted-experience capacity (``max_rows``). Identity is
    the snapshot bytes; nothing is keyed to this object's address.
    """

    __slots__ = ("_api", "_x", "_handle", "dim", "width", "_snapshot_size", "_t")

    def __init__(self, api, handle):
        self._api, self._x, self._handle, self._t = api, api._x, handle, _Transition()
        self.dim, self.width = int(self._x.dim(handle)), int(self._x.width(handle))
        self._snapshot_size = int(self._x.snapshot_size(handle))

    @classmethod
    def create(cls, api, dim, width, initial_w, *, max_rows=DEFAULT_MAX_ROWS):
        if type(api) is not ECSGLibrary:
            raise ECSGError("INVALID", "ECSGLibrary required")
        if type(dim) is not int or type(width) is not int or dim < 1 or width < 1 or type(max_rows) is not int:
            raise ECSGError("INVALID", "executor shape")
        w = _admit_vector(initial_w, dim * width, "initial W")
        handle = _VP()
        rc = api._x.create(dim, width, max_rows, w, C.byref(handle))
        if rc != _OK:
            raise _refused(rc, "executor create")
        return cls(api, handle)

    @classmethod
    def restore(cls, api, snapshot, *, max_rows=DEFAULT_MAX_ROWS):
        if type(api) is not ECSGLibrary or type(snapshot) is not bytes or not snapshot or type(max_rows) is not int:
            raise ECSGError("INVALID", "snapshot bytes")
        data = (C.c_uint8 * len(snapshot)).from_buffer_copy(snapshot)
        handle = _VP()
        rc = api._x.restore(data, len(snapshot), max_rows, C.byref(handle))
        if rc != _OK:
            raise _refused(rc, "snapshot restore")
        return cls(api, handle)

    def _live(self):
        if self._handle is None or not self._handle.value:
            raise ECSGError("CLOSED", "ECS executor closed")
        return self._handle

    @property
    def epoch(self):
        return int(self._x.epoch(self._live()))

    @property
    def generation(self):
        """Commit counter of this executor (process-local; not persisted, not identity)."""
        return int(self._x.generation(self._live()))

    @property
    def max_rows(self):
        return int(self._x.max_rows(self._live()))

    def reserve(self, max_rows):
        """Cold path: grow the admitted-experience capacity (one native allocation)."""
        if type(max_rows) is not int or max_rows < 1:
            raise ECSGError("INVALID", "max_rows")
        rc = self._x.reserve(self._live(), max_rows)
        if rc != _OK:
            raise _refused(rc, "reserve")

    def forward(self, x_rows):
        """QUERY: ``f_W(x)`` for each row, from the current authoritative W, as a tuple. Read-only."""
        x, rows = _admit_rows(x_rows, self.dim, "forward rows")
        if rows > _MAX_FORWARD_ROWS:
            raise ECSGError("INVALID", "forward rows")
        out = (C.c_double * rows)()
        rc = self._x.forward(self._live(), x, rows, out)
        if rc != _OK:
            raise _refused(rc, "forward")
        return _unpack_from(f"{rows}d", out)

    def forward_into(self, x, out):
        """QUERY into a caller-owned buffer: no allocation, no conversion. Returns the row count."""
        xv, rows = _admit_rows(x, self.dim, "forward rows")
        rc = self._x.forward(self._live(), xv, rows, _admit_out(out, rows, "forward out"))
        if rc != _OK:
            raise _refused(rc, "forward")
        return rows

    def learn(self, x_rows, y, learning_rate, steps=1):
        """LEARN: ``steps`` G1 steps on ``(X, y)`` in one native call, committed as one transition.

        A refusal leaves W, epoch and generation unchanged.
        """
        x, rows = _admit_rows(x_rows, self.dim, "learn X")
        t = self._t
        rc = self._x.learn(self._live(), x, _admit_vector(y, rows, "learn y"), rows, _rate(learning_rate),
                           _steps(steps), C.byref(t))
        if rc != _OK:
            raise _refused(rc, "learn", t.failed_step)
        return _commit(t)

    def transaction(self):
        """Open the native candidate transaction (one at a time); use it as a context manager."""
        token = C.c_uint64()
        rc = self._x.txn_begin(self._live(), C.byref(token))
        if rc != _OK:
            raise _refused(rc, "transaction begin")
        return Transaction(self, token.value)

    def w(self):
        count = self.dim * self.width
        out = (C.c_double * count)()
        rc = self._x.copy_w(self._live(), out, count)
        if rc != _OK:
            raise _refused(rc, "copy W")
        return tuple(out)

    def s3(self):
        """Packed raw-sum S3 = (mu, M upper-packed, T3 upper-packed), length s3_size(dim). Diagnostic."""
        mu, m, t3 = _s3_buffers(self._api, self.dim)
        rc = self._x.project_s3(self._live(), mu, m, t3)
        if rc != _OK:
            raise _refused(rc, "project S3")
        return tuple(mu) + tuple(m) + tuple(t3)

    def snapshot(self):
        """Portable bytes of W and its epoch, identical to the reference state's snapshot."""
        out = (C.c_uint8 * self._snapshot_size)()
        rc = self._x.snapshot_write(self._live(), out, self._snapshot_size)
        if rc != _OK:
            raise _refused(rc, "snapshot write")
        return bytes(out)

    def stats(self):
        """Native counters and owned workspace (see elpis_ecsg_exec_stats)."""
        s = _Stats()
        rc = self._x.stats(self._live(), C.byref(s))
        if rc != _OK:
            raise _refused(rc, "stats")
        return {name: int(getattr(s, name)) for name, _ in _Stats._fields_}

    def close(self):
        if self._handle is not None and self._handle.value:
            rc = self._x.destroy(C.byref(self._handle))
            if rc != _OK:
                raise _refused(rc, "executor destroy")
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Transaction:
    """The native candidate transaction of one executor.

    The candidate starts as the authoritative W; ``learn`` advances it without
    touching the authoritative W; ``forward``/``s3``/``epoch`` read it;
    ``commit`` installs it natively, or raises ``STALE`` if another commit
    replaced its source state meanwhile. A refused learn or a stale check ends
    the transaction; leaving the ``with`` block uncommitted aborts it.
    """

    __slots__ = ("_executor", "_token")

    def __init__(self, executor, token):
        self._executor, self._token = executor, token

    def _live(self):
        if self._token is None:
            raise ECSGError("CLOSED", "transaction finished")
        return self._executor._live()

    def _fail(self, rc, what, step=0, learning=False):
        if rc == -3 or (learning and rc == -2):  # STALE, or a refused learn: native code discarded it
            self._token = None
        return _refused(rc, what, step)

    @property
    def open(self):
        return self._token is not None

    def learn(self, x_rows, y, learning_rate, steps=1):
        """``steps`` G1 steps on the candidate in one native call. Returns candidate epochs."""
        e = self._executor
        x, rows = _admit_rows(x_rows, e.dim, "learn X")
        t = e._t
        rc = e._x.txn_learn(self._live(), self._token, x, _admit_vector(y, rows, "learn y"), rows,
                            _rate(learning_rate), _steps(steps), C.byref(t))
        if rc != _OK:
            raise self._fail(rc, "transaction learn", t.failed_step, True)
        return _commit(t)

    def learn_schedule(self, x_rows, y, drive_rows, learning_rate, steps=1):
        """Ordered drives over consecutive row blocks of ``(X, y)``, ``steps`` each, in one native call."""
        e = self._executor
        x, rows = _admit_rows(x_rows, e.dim, "learn X")
        drives, count = _admit_drives(drive_rows, _steps(steps), rows, "drives")
        t = e._t
        rc = e._x.txn_learn_schedule(self._live(), self._token, x, _admit_vector(y, rows, "learn y"), drives, count,
                                     _rate(learning_rate), C.byref(t))
        if rc != _OK:
            raise self._fail(rc, "transaction learn", t.failed_step, True)
        return _commit(t)

    def forward(self, x_rows):
        e = self._executor
        x, rows = _admit_rows(x_rows, e.dim, "forward rows")
        out = (C.c_double * rows)()
        rc = e._x.txn_forward(self._live(), self._token, x, rows, out)
        if rc != _OK:
            raise self._fail(rc, "transaction forward")
        return tuple(out)

    def s3(self):
        e = self._executor
        mu, m, t3 = _s3_buffers(e._api, e.dim)
        rc = e._x.txn_project_s3(self._live(), self._token, mu, m, t3)
        if rc != _OK:
            raise self._fail(rc, "transaction S3")
        return tuple(mu) + tuple(m) + tuple(t3)

    @property
    def epoch(self):
        value = C.c_uint64()
        rc = self._executor._x.txn_epoch(self._live(), self._token, C.byref(value))
        if rc != _OK:
            raise self._fail(rc, "transaction epoch")
        return int(value.value)

    def commit(self):
        """Install the candidate (one native pointer exchange) or raise STALE; the transaction ends either way."""
        e = self._executor
        t = e._t
        rc = e._x.txn_commit(self._live(), self._token, C.byref(t))
        if rc == _OK:
            self._token = None
            return _commit(t)
        raise self._fail(rc, "transaction commit")

    def abort(self):
        if self._token is not None and self._executor._handle is not None:
            token, self._token = self._token, None
            rc = self._executor._x.txn_abort(self._executor._live(), token)
            if rc != _OK:
                self._token = token
                raise _refused(rc, "transaction abort")
        self._token = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()


def _s3_buffers(api, d):
    lib = api._lib
    return ((C.c_double * d)(), (C.c_double * int(lib.elpis_ecsg_symmetric2_size(d)))(),
            (C.c_double * int(lib.elpis_ecsg_symmetric3_size(d)))())


class WorldState:
    """The scalar reference ECS state. W is authoritative; S3 is its projection.

    Correctness authority for the executor, one explicit step per call. Not
    the runtime path: runtime callers use :class:`Executor`.
    """

    __slots__ = ("_api", "_handle", "dim", "width")

    def __init__(self, api, handle):
        self._api, self._handle = api, handle
        lib = api._lib
        self.dim, self.width = int(lib.elpis_ecsg_state_dim(handle)), int(lib.elpis_ecsg_state_width(handle))

    @classmethod
    def create(cls, api, dim, width, initial_w):
        if type(api) is not ECSGLibrary:
            raise ECSGError("INVALID", "ECSGLibrary required")
        w = _doubles(initial_w, dim * width, "initial W")
        handle = _VP()
        _check(api._lib.elpis_ecsg_state_create(dim, width, w, C.byref(handle)), "state create")
        return cls(api, handle)

    @classmethod
    def restore(cls, api, snapshot):
        if type(api) is not ECSGLibrary or type(snapshot) is not bytes or not snapshot:
            raise ECSGError("INVALID", "snapshot bytes")
        data = (C.c_uint8 * len(snapshot)).from_buffer_copy(snapshot)
        handle = _VP()
        _check(api._lib.elpis_ecsg_state_snapshot_restore(data, len(snapshot), C.byref(handle)), "snapshot restore")
        return cls(api, handle)

    def _live(self):
        if self._handle is None or not self._handle.value:
            raise ECSGError("CLOSED", "ECS state closed")
        return self._handle

    @property
    def epoch(self):
        return int(self._api._lib.elpis_ecsg_state_epoch(self._live()))

    def w(self):
        count = self.dim * self.width
        out = (C.c_double * count)()
        _check(self._api._lib.elpis_ecsg_state_copy_w(self._live(), out, count), "copy W")
        return tuple(out)

    def forward(self, x_rows):
        """``f_W(x)`` for each query row, computed natively from the current W. Read-only."""
        rows = tuple(x_rows)
        if not 1 <= len(rows) <= _MAX_FORWARD_ROWS:
            raise ECSGError("INVALID", "forward rows")
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != self.dim:
                raise ECSGError("INVALID", "forward row width")
        x = _doubles((v for row in rows for v in row), len(rows) * self.dim, "forward rows")
        out = (C.c_double * len(rows))()
        _check(self._api._lib.elpis_ecsg_state_forward_f64(self._live(), x, len(rows), out), "forward")
        return tuple(out)

    def s3(self):
        """Packed raw-sum S3 = (mu, M upper-packed, T3 upper-packed), length s3_size(dim)."""
        mu, m, t3 = _s3_buffers(self._api, self.dim)
        _check(self._api._lib.elpis_ecsg_state_project_s3_f64(self._live(), mu, m, t3), "project S3")
        return tuple(mu) + tuple(m) + tuple(t3)

    def step(self, x_rows, y, learning_rate):
        """One atomic cubic full-batch gradient step. Rejection leaves W and epoch unchanged."""
        rows = len(y)
        if rows < 1 or type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ECSGError("INVALID", "step rows/learning rate")
        x = _doubles((v for row in x_rows for v in row), rows * self.dim, "drive X")
        target = _doubles(y, rows, "drive y")
        lib = self._api._lib
        count = int(lib.elpis_ecsg_state_gd_step_scratch_f64(self.dim, self.width, rows))
        scratch = (C.c_double * max(count, 1))()
        _check(lib.elpis_ecsg_state_gd_step_f64(self._live(), x, target, rows, learning_rate, scratch, count),
               "gradient step")

    def fork(self):
        """An independent copy of this state, through the native snapshot/restore surface."""
        return WorldState.restore(self._api, self.snapshot())

    def snapshot(self):
        lib, handle = self._api._lib, self._live()
        size = int(lib.elpis_ecsg_state_snapshot_size(handle))
        out = (C.c_uint8 * size)()
        _check(lib.elpis_ecsg_state_snapshot_write(handle, out, size), "snapshot write")
        return bytes(out)

    def close(self):
        if self._handle is not None and self._handle.value:
            _check(self._api._lib.elpis_ecsg_state_destroy(C.byref(self._handle)), "state destroy")
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
