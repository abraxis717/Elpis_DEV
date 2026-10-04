"""Mutable ECS_G logical states backed by generic FMS residency.

This module is an additive control plane over the existing qualified Executor.
The native adapter owns materialization and state publication.  Python performs
only coarse lifecycle calls.  FMSRuntime returns *actual* ``Executor`` objects
whose ABI table is redirected to the native residency adapter, so existing
CognitiveCore/runtime exact-type checks and numerical methods remain unchanged.
"""
from __future__ import annotations

import ctypes as C
from types import SimpleNamespace

from .native import (
    DEFAULT_MAX_ROWS,
    ECSGError,
    ECSGLibrary,
    Executor,
    _Drive,
    _P,
    _Stats,
    _TP,
    _Transition,
    _U8P,
    _VP,
    _admit_vector,
    _refused,
)

__all__ = ("FMSRuntime",)

_U64 = C.c_uint64
_FMS_CODES = {
    -101: "INVALID",
    -102: "NOMEM",
    -103: "MISSING",
    -104: "BUSY",
    -105: "UNSUPPORTED",
    -106: "IO",
    -107: "CAPACITY",
    -108: "STATE",
    -109: "DIGEST",
    -110: "DEVICE",
    -111: "TIMEOUT",
}


class _FMSInfo(C.Structure):
    _fields_ = [
        ("logical_identity", C.c_uint8 * 32),
        ("object_handle", _U64),
        ("epoch", _U64),
        ("generation", _U64),
        ("dim", C.c_size_t),
        ("width", C.c_size_t),
        ("max_rows", C.c_size_t),
        ("snapshot_bytes", C.c_size_t),
        ("logical_bytes", _U64),
        ("resident_authoritative_bytes", _U64),
        ("active_workspace_bytes", _U64),
        ("acquisitions", _U64),
        ("lease_failures", _U64),
        ("commits", _U64),
        ("aborts", _U64),
        ("materialization_ns", _U64),
        ("query_ns", _U64),
        ("learn_ns", _U64),
        ("commit_ns", _U64),
        ("lease_count", C.c_uint32),
        ("tier", C.c_uint8),
        ("residency_state", C.c_uint8),
        ("cold_replica", C.c_uint8),
        ("transaction_open", C.c_uint8),
    ]


class _FMSStats(C.Structure):
    _fields_ = [
        ("tier_bytes", _U64 * 3),
        ("domain_bytes", _U64 * 3),
        ("objects", _U64),
        ("pinned_bytes", _U64),
        ("inflight_ops", _U64),
        ("promotions", _U64),
        ("demotions", _U64),
        ("bytes_promoted", _U64),
        ("bytes_demoted", _U64),
        ("zero_copy_placements", _U64),
        ("cold_replica_reuse", _U64),
        ("cold_writes", _U64),
        ("cold_reads", _U64),
        ("digest_failures", _U64),
        ("device_failures", _U64),
        ("forced_cpu_fallbacks", _U64),
        ("forced_placements", _U64),
        ("move_failures", _U64),
        ("fence_timeouts", _U64),
        ("move_p50_ns", _U64),
        ("move_p95_ns", _U64),
    ]


class _FMSMetrics(C.Structure):
    _fields_ = [
        ("residency", _FMSStats),
        ("logical_bytes", _U64),
        ("resident_authoritative_bytes", _U64),
        ("active_workspace_bytes", _U64),
        ("resident_high_water", _U64),
        ("leases", _U64),
        ("states", _U64),
        ("demotion_ns", _U64),
    ]


def _fields(record):
    def value(v):
        if isinstance(v, C.Array):
            return tuple(v)
        if isinstance(v, C.Structure):
            return _fields(v)
        return int(v)

    return {name: value(getattr(record, name)) for name, _ in record._fields_}


def _sid(value):
    if hasattr(value, "value"):
        value = value.value
    return int(value or 0)


def _fms_result(rc, what):
    """Translate FMS-domain errors; preserve executor-domain status for Executor proxy calls."""
    rc = int(rc)
    if rc <= -101:
        raise ECSGError(_FMS_CODES.get(rc, "INVALID"), what)
    return rc


def _control_result(rc, what):
    """Direct runtime calls raise both FMS-domain and executor-domain failures."""
    rc = _fms_result(rc, what)
    if rc != 0:
        raise _refused(rc, what)
    return rc


class FMSRuntime:
    """Own one generic FMS context and a bounded set of logical ECS states."""

    def __init__(self, api, context, library, *, max_states):
        if type(api) is not ECSGLibrary or not getattr(context, "handle", None) or not context.handle.value:
            raise ECSGError("INVALID", "loaded ECS library and owned FMS context required")
        if type(max_states) is not int or not 1 <= max_states < 1 << 32:
            raise ECSGError("INVALID", "state capacity")

        if library is None:
            raise ECSGError("INVALID", "mutable FMS adapter library required")
        self._api = api
        self._library = library
        self._handle = _VP()
        lib = library

        def bind(name, args, result=C.c_int):
            try:
                fn = getattr(lib, "elpis_ecsg_fms_" + name)
            except AttributeError as exc:
                raise ECSGError("UNSUPPORTED", "mutable FMS symbol " + name) from exc
            fn.argtypes, fn.restype = args, result
            return fn

        self._f = SimpleNamespace(
            abi_version=bind("abi_version", [], C.c_uint32),
            create=bind("create", [_VP, C.c_size_t, C.POINTER(_VP)]),
            destroy=bind("destroy", [C.POINTER(_VP)]),
            register=bind("register", [_VP, _U8P, C.c_size_t, C.c_size_t, C.c_size_t,
                                       _P, C.POINTER(_U64)]),
            restore=bind("restore", [_VP, _U8P, _U8P, C.c_size_t, C.c_size_t,
                                     C.POINTER(_U64)]),
            close=bind("close", [_VP, C.POINTER(_U64)]),
            inspect=bind("inspect", [_VP, _U64, C.POINTER(_FMSInfo)]),
            exec_stats=bind("exec_stats", [_VP, _U64, C.POINTER(_Stats)]),
            stats=bind("stats", [_VP, C.POINTER(_FMSMetrics)]),
            pump=bind("pump", [_VP]),
        )
        if self._f.abi_version() != 1:
            raise ECSGError("UNSUPPORTED", "mutable FMS ABI")
        rc = self._f.create(context.handle, max_states, C.byref(self._handle))
        _control_result(rc, "mutable FMS runtime create")
        context.handle.value = None

        native = {}
        signatures = {
            "reserve": ([_VP, _U64, C.c_size_t], C.c_int),
            "forward": ([_VP, _U64, _P, C.c_size_t, _P], C.c_int),
            "learn": ([_VP, _U64, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
            "learn_schedule": ([_VP, _U64, _P, _P, C.POINTER(_Drive), C.c_size_t, C.c_double, _TP], C.c_int),
            "copy_w": ([_VP, _U64, _P, C.c_size_t], C.c_int),
            "project_s3": ([_VP, _U64, _P, _P, _P], C.c_int),
            "snapshot_write": ([_VP, _U64, _U8P, C.c_size_t], C.c_int),
            "txn_begin": ([_VP, _U64, C.POINTER(_U64)], C.c_int),
            "txn_learn": ([_VP, _U64, _U64, _P, _P, C.c_size_t, C.c_double, _U64, _TP], C.c_int),
            "txn_learn_schedule": ([_VP, _U64, _U64, _P, _P, C.POINTER(_Drive), C.c_size_t,
                                    C.c_double, _TP], C.c_int),
            "txn_forward": ([_VP, _U64, _U64, _P, C.c_size_t, _P], C.c_int),
            "txn_project_s3": ([_VP, _U64, _U64, _P, _P, _P], C.c_int),
            "txn_epoch": ([_VP, _U64, _U64, C.POINTER(_U64)], C.c_int),
            "txn_commit": ([_VP, _U64, _U64, _TP], C.c_int),
            "txn_abort": ([_VP, _U64, _U64], C.c_int),
        }
        for name, (args, result) in signatures.items():
            native[name] = bind(name, args, result)
        self._native = native
        self._x = self._make_executor_proxy()

    def _live(self):
        if not self._handle.value:
            raise ECSGError("CLOSED", "mutable FMS runtime closed")
        return self._handle

    @staticmethod
    def _key(key):
        if type(key) is not bytes or len(key) != 32:
            raise ECSGError("INVALID", "explicit 32-byte logical namespace key required")
        return (C.c_uint8 * 32).from_buffer_copy(key)

    def _inspect_id(self, sid):
        info = _FMSInfo()
        rc = self._f.inspect(self._live(), sid, C.byref(info))
        _control_result(rc, "mutable FMS state inspection")
        return info

    def _make_executor_proxy(self):
        proxy = SimpleNamespace()
        proxy.abi_version = lambda: 1
        proxy.workspace_bytes = self._api._x.workspace_bytes

        def scalar(name):
            return lambda handle: int(getattr(self._inspect_id(_sid(handle)), name))

        proxy.dim = scalar("dim")
        proxy.width = scalar("width")
        proxy.max_rows = scalar("max_rows")
        proxy.epoch = scalar("epoch")
        proxy.generation = scalar("generation")
        proxy.snapshot_size = scalar("snapshot_bytes")

        def numeric(name):
            fn = self._native[name]
            def call(handle, *args):
                return _fms_result(fn(self._live(), _sid(handle), *args), "mutable FMS " + name)
            return call

        for name in ("reserve", "forward", "learn", "learn_schedule", "copy_w", "project_s3",
                     "snapshot_write", "txn_begin", "txn_learn", "txn_learn_schedule", "txn_forward",
                     "txn_project_s3", "txn_epoch", "txn_commit", "txn_abort"):
            setattr(proxy, name, numeric(name))

        def stats(handle, out):
            return _fms_result(self._f.exec_stats(self._live(), _sid(handle), out), "mutable FMS executor stats")
        proxy.stats = stats

        def destroy(handle_ptr):
            p = C.cast(handle_ptr, C.POINTER(_U64))
            return _fms_result(self._f.close(self._live(), p), "mutable FMS state close")
        proxy.destroy = destroy
        return proxy

    def _executor(self, sid):
        info = self._inspect_id(sid)
        state = Executor.__new__(Executor)
        state._api = self._api
        state._x = self._x
        state._handle = _U64(sid)
        state._t = _Transition()
        state.dim = int(info.dim)
        state.width = int(info.width)
        state._snapshot_size = int(info.snapshot_bytes)
        return state

    def create(self, key, dim, width, initial_w, *, max_rows=DEFAULT_MAX_ROWS):
        if (type(dim) is not int or type(width) is not int or min(dim, width) < 1
                or type(max_rows) is not int or max_rows < 1):
            raise ECSGError("INVALID", "state geometry")
        w = _admit_vector(initial_w, dim * width, "initial W")
        sid = _U64()
        rc = self._f.register(self._live(), self._key(key), dim, width, max_rows, w, C.byref(sid))
        _control_result(rc, "mutable FMS state registration")
        return self._executor(sid.value)

    def restore(self, key, snapshot, *, max_rows=DEFAULT_MAX_ROWS):
        if type(snapshot) is not bytes or not snapshot or type(max_rows) is not int or max_rows < 1:
            raise ECSGError("INVALID", "snapshot and capacity")
        data = (C.c_uint8 * len(snapshot)).from_buffer_copy(snapshot)
        sid = _U64()
        rc = self._f.restore(self._live(), self._key(key), data, len(snapshot), max_rows, C.byref(sid))
        _control_result(rc, "mutable FMS state restore")
        return self._executor(sid.value)

    def inspect(self, state):
        if type(state) is not Executor or state._x is not self._x:
            raise ECSGError("INVALID", "state is not owned by this mutable FMS runtime")
        if state._handle is None or not state._handle.value:
            raise ECSGError("CLOSED", "mutable FMS state closed")
        result = _fields(self._inspect_id(_sid(state._handle)))
        result["logical_identity"] = bytes(result["logical_identity"]).hex()
        return result

    def pump(self):
        _control_result(self._f.pump(self._live()), "mutable FMS pump")

    def stats(self):
        metrics = _FMSMetrics()
        _control_result(self._f.stats(self._live(), C.byref(metrics)), "mutable FMS metrics")
        result = _fields(metrics)
        result["residency"] = _fields(metrics.residency)
        return result

    def close(self):
        if self._handle.value:
            _control_result(self._f.destroy(C.byref(self._handle)), "mutable FMS runtime destroy")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
