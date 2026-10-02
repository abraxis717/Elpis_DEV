"""Thin binding to the existing generic execution port (``elpis/execution.h``).

It binds only what an adapter needs to drive a provider-owned
``ELPIS_EXEC_BACKEND_ONLY`` stream: owned buffers, a runtime created with an
explicitly attached backend, BACKEND_ONLY submission, the single ordered
``take``, ``notify``, metrics, shutdown and destroy. It never registers a CPU
compute callback, so no pool thread can ever call into Python. It adds nothing
to the port contract and knows no vendor concept.

The runtime library is opened only through ``RootCapability`` + ``load_native``
against an independently pinned identity. One sealed copy per process serves
every runtime: a provider must use that copy's entry points (``entry_point``),
never link a private runtime of its own.
"""
from __future__ import annotations

import ctypes as C

from .authority import PinnedAuthority
from .boundary import RootCapability, load_native
from .contracts import Code, ContractError, require

OK, INVALID, CLOSED, WOULD_BLOCK, CANCELLED = 0, 1, 2, 3, 4
BACKEND_UNAVAILABLE, BACKEND_REJECTED, INTERNAL, DEFER = 5, 6, 7, 8
STATUS_NAMES = {OK: "OK", INVALID: "INVALID", CLOSED: "CLOSED", WOULD_BLOCK: "WOULD_BLOCK",
                CANCELLED: "CANCELLED", BACKEND_UNAVAILABLE: "BACKEND_UNAVAILABLE",
                BACKEND_REJECTED: "BACKEND_REJECTED", INTERNAL: "INTERNAL", DEFER: "DEFER"}
BACKEND_ONLY = 2
BUFFER_LIMIT = 64 << 20


class Backend(C.Structure):
    """``elpis_exec_backend``: opaque to Python; filled by a provider's attach."""
    _fields_ = [("context", C.c_void_p), ("init", C.c_void_p), ("submit", C.c_void_p),
                ("poll", C.c_void_p), ("abort", C.c_void_p), ("shutdown", C.c_void_p)]


class _Config(C.Structure):
    _fields_ = [("workers", C.c_uint), ("capacity", C.c_uint), ("max_input_bytes", C.c_size_t),
                ("max_output_bytes", C.c_size_t), ("backend_poll_limit", C.c_uint),
                ("backend", C.POINTER(Backend))]


class _Task(C.Structure):
    _fields_ = [("operation", C.c_uint32), ("stage", C.c_uint32), ("affinity", C.c_uint32),
                ("flags", C.c_uint32), ("tag", C.c_uint64), ("compute", C.c_void_p)]


class _Result(C.Structure):
    _fields_ = [("sequence", C.c_uint64), ("tag", C.c_uint64), ("operation", C.c_uint32),
                ("stage", C.c_uint32), ("status", C.c_int), ("queue_ns", C.c_uint64),
                ("compute_ns", C.c_uint64), ("output", C.c_void_p), ("worker", C.c_uint32),
                ("completed_ns", C.c_uint64), ("retire_ns", C.c_uint64)]


class _Metrics(C.Structure):
    _fields_ = [("workers", C.c_uint), ("capacity", C.c_uint), ("outstanding", C.c_uint),
                ("high_water", C.c_uint), ("submitted", C.c_uint64), ("completed", C.c_uint64),
                ("retired", C.c_uint64), ("cancelled", C.c_uint64), ("queue_full", C.c_uint64),
                ("queue_ns", C.c_uint64), ("compute_ns", C.c_uint64), ("backend_accepted", C.c_uint64),
                ("backend_fallback", C.c_uint64), ("queued", C.c_uint), ("running", C.c_uint),
                ("parked", C.c_uint), ("steals", C.c_uint64), ("retire_wait_ns", C.c_uint64),
                ("backend_polls", C.c_uint64), ("backend_wait_ns", C.c_uint64), ("deferred", C.c_uint64)]


class ExecutionLibrary:
    """One sealed, pinned ``libelpis_execution`` with the subset of the ABI bound."""

    __slots__ = ("_lib", "identity", "library_id", "counts")

    def __init__(self, root, library, *, authority, library_id):
        require(type(authority) is PinnedAuthority and authority.provenance in ("deployment", "synthetic-test"),
                Code.IDENTITY, "execution runtime authority")
        require(type(library_id) is str and library_id in authority.libraries, Code.IDENTITY,
                "execution runtime library identifier")
        identity = authority.libraries[library_id]
        with RootCapability(root) as boundary:
            lib = load_native(boundary, library, identity)
        vp, sz, u64 = C.c_void_p, C.c_size_t, C.c_uint64
        sig = {
            "elpis_exec_buffer_alloc": ([sz], vp),
            "elpis_exec_buffer_mutable_data": ([vp], vp),
            "elpis_exec_buffer_data": ([vp], vp),
            "elpis_exec_buffer_size": ([vp], sz),
            "elpis_exec_buffer_release": ([vp], None),
            "elpis_exec_create": ([C.POINTER(_Config), C.POINTER(vp)], C.c_int),
            "elpis_exec_submit": ([vp, C.POINTER(_Task), C.POINTER(vp), C.POINTER(u64)], C.c_int),
            "elpis_exec_take": ([vp, C.c_uint, C.POINTER(_Result)], C.c_int),
            "elpis_exec_get_metrics": ([vp, C.POINTER(_Metrics)], None),
            "elpis_exec_notify": ([vp], None),
            "elpis_exec_shutdown": ([vp, C.c_int], C.c_int),
            "elpis_exec_destroy": ([vp], None),
        }
        for name, (args, result) in sig.items():
            try:
                fn = getattr(lib, name)
            except AttributeError as exc:
                raise ContractError(Code.UNSUPPORTED, "execution symbol: " + name) from exc
            fn.argtypes, fn.restype = args, result
        self._lib, self.identity, self.library_id = lib, identity, library_id
        # Buffer accounting for leak qualification: every request buffer is either consumed
        # by an accepted submission or released; every taken output is released.
        self.counts = dict(allocated=0, released=0, consumed=0, outputs=0, outputs_released=0)

    def entry_point(self, name):
        """Address of one runtime entry point, for a provider's host table."""
        require(name in ("elpis_exec_buffer_alloc", "elpis_exec_buffer_mutable_data", "elpis_exec_buffer_data",
                         "elpis_exec_buffer_size", "elpis_exec_buffer_release", "elpis_exec_notify"),
                Code.UNSUPPORTED, "execution entry point")
        return C.cast(getattr(self._lib, name), C.c_void_p).value

    def buffer(self, size):
        require(type(size) is int and 0 < size <= BUFFER_LIMIT, Code.LIMIT, "execution buffer size")
        ptr = self._lib.elpis_exec_buffer_alloc(size)
        require(bool(ptr), Code.LIMIT, "execution buffer allocation")
        self.counts["allocated"] += 1
        return Buffer(self, ptr, size)


class Buffer:
    """An owned ``elpis_exec_buffer``: writable until submitted or released."""

    __slots__ = ("_owner", "_ptr", "size", "state")

    def __init__(self, owner, ptr, size):
        self._owner, self._ptr, self.size, self.state = owner, ptr, size, "OWNED"

    def view(self):
        require(self.state == "OWNED", Code.CLOSED, "execution buffer no longer writable")
        data = self._owner._lib.elpis_exec_buffer_mutable_data(self._ptr)
        require(bool(data), Code.CLOSED, "sealed execution buffer")
        return memoryview((C.c_ubyte * self.size).from_address(data)).cast("B")

    def release(self):
        if self.state == "OWNED":
            self._owner._lib.elpis_exec_buffer_release(self._ptr)
            self._owner.counts["released"] += 1
        self.state = "RELEASED"


class Result:
    __slots__ = ("sequence", "tag", "status", "output")

    def __init__(self, sequence, tag, status, output):
        self.sequence, self.tag, self.status, self.output = sequence, tag, status, output


class Runtime:
    """One execution context with an attached backend. Single lifecycle owner."""

    __slots__ = ("_owner", "_handle", "_config", "_backend", "closed")

    def __init__(self, owner, backend, *, workers, capacity, max_input_bytes, max_output_bytes, poll_limit):
        require(type(owner) is ExecutionLibrary and type(backend) is Backend, Code.IDENTITY, "execution runtime")
        self._owner, self._backend = owner, backend
        self._config = _Config(workers, capacity, max_input_bytes, max_output_bytes, poll_limit, C.pointer(backend))
        handle = C.c_void_p()
        rc = owner._lib.elpis_exec_create(C.byref(self._config), C.byref(handle))
        require(rc == OK and bool(handle.value), Code.UNSUPPORTED,
                "execution runtime create: " + STATUS_NAMES.get(rc, str(rc)))
        self._handle, self.closed = handle, False

    @property
    def handle(self):
        return self._handle.value

    def submit_backend_only(self, operation, stage, tag, buffer):
        """BACKEND_ONLY submission. OK consumes the buffer; any other status keeps it owned."""
        require(not self.closed and type(buffer) is Buffer and buffer.state == "OWNED", Code.CLOSED,
                "BACKEND_ONLY submission")
        task = _Task(operation, stage, 0, BACKEND_ONLY, tag, None)
        ptr = C.c_void_p(buffer._ptr)
        sequence = C.c_uint64(0)
        rc = self._owner._lib.elpis_exec_submit(self._handle, C.byref(task), C.byref(ptr), C.byref(sequence))
        if rc == OK:
            buffer.state = "SUBMITTED"
            self._owner.counts["consumed"] += 1
        return rc, sequence.value

    def take(self, wait_ms):
        """Ordered retirement. Returns (status, Result|None); output bytes are copied and released."""
        result = _Result()
        rc = self._owner._lib.elpis_exec_take(self._handle, wait_ms, C.byref(result))
        if rc != OK:
            return rc, None
        output = None
        if result.output:
            lib = self._owner._lib
            self._owner.counts["outputs"] += 1
            try:
                output = C.string_at(lib.elpis_exec_buffer_data(result.output),
                                     lib.elpis_exec_buffer_size(result.output))
            finally:
                lib.elpis_exec_buffer_release(result.output)
                self._owner.counts["outputs_released"] += 1
        return rc, Result(result.sequence, result.tag, result.status, output)

    def metrics(self):
        m = _Metrics()
        self._owner._lib.elpis_exec_get_metrics(self._handle, C.byref(m))
        return {name: getattr(m, name) for name, _ in _Metrics._fields_}

    def notify(self):
        self._owner._lib.elpis_exec_notify(self._handle)

    def shutdown(self, cancel=True):
        if not self.closed:
            self._owner._lib.elpis_exec_shutdown(self._handle, 1 if cancel else 0)

    def destroy(self):
        """Shut down (runs backend.shutdown once) and free the context. Idempotent."""
        if not self.closed:
            self._owner._lib.elpis_exec_destroy(self._handle)
            self.closed = True
