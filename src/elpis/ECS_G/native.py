"""Python binding to the native ECS_G kernel (``native/ECS_G``).

The binding owns no loading policy: the caller passes an already loaded
``libelpis_ecsg_math`` handle (for example one opened through the substrate's
sealed loader by the runtime). ECS_G therefore keeps its declared dependency
set empty: this module imports only the standard library, never numpy,
inference, runtime or ECS_C.

It exposes exactly the qualified native surface: owned microscopic state
``W[dim, width]`` (binary64, row-major), its epoch, the raw-sum ``S3``
projection, the explicit atomic cubic gradient step, and portable
snapshot/restore. The mathematics live in C; nothing here re-implements them.
"""
from __future__ import annotations

import ctypes as C
import math

__all__ = ("ECSGError", "ECSGLibrary", "WorldState")

_OK, _INVALID, _NONFINITE = 0, -1, -2
_P = C.POINTER(C.c_double)
_VP = C.c_void_p


class ECSGError(ValueError):
    """A native ECS_G call refused its input (state is unchanged)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _check(rc, what):
    if rc != _OK:
        raise ECSGError("NONFINITE" if rc == _NONFINITE else "INVALID", what)


def _doubles(values, count, what):
    values = tuple(values)
    if len(values) != count or not all(type(v) in (float, int) and math.isfinite(v) for v in values):
        raise ECSGError("INVALID", what)
    return (C.c_double * count)(*values)


class ECSGLibrary:
    """Typed view of one loaded ECS_G library (math ABI v1, state ABI v1)."""

    __slots__ = ("_lib",)

    def __init__(self, lib):
        sig = {
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
            "elpis_ecsg_state_gd_step_scratch_f64": ([C.c_size_t, C.c_size_t, C.c_size_t], C.c_size_t),
            "elpis_ecsg_state_gd_step_f64": ([_VP, _P, _P, C.c_size_t, C.c_double, _P, C.c_size_t], C.c_int),
            "elpis_ecsg_state_snapshot_size": ([_VP], C.c_size_t),
            "elpis_ecsg_state_snapshot_write": ([_VP, C.POINTER(C.c_uint8), C.c_size_t], C.c_int),
            "elpis_ecsg_state_snapshot_restore": ([C.POINTER(C.c_uint8), C.c_size_t, C.POINTER(_VP)], C.c_int),
        }
        for name, (args, result) in sig.items():
            try:
                fn = getattr(lib, name)
            except AttributeError as exc:
                raise ECSGError("UNSUPPORTED", "ECS_G symbol " + name) from exc
            fn.argtypes, fn.restype = args, result
        if lib.elpis_ecsg_math_abi_version() != 1 or lib.elpis_ecsg_state_abi_version() != 1:
            raise ECSGError("UNSUPPORTED", "ECS_G ABI version")
        self._lib = lib

    def s3_size(self, dim):
        return int(self._lib.elpis_ecsg_s3_size(dim))


class WorldState:
    """One owned native ECS_G state. W is authoritative; S3 is its projection."""

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
            raise ECSGError("CLOSED", "ECS_G state closed")
        return self._handle

    @property
    def epoch(self):
        return int(self._api._lib.elpis_ecsg_state_epoch(self._live()))

    def w(self):
        count = self.dim * self.width
        out = (C.c_double * count)()
        _check(self._api._lib.elpis_ecsg_state_copy_w(self._live(), out, count), "copy W")
        return tuple(out)

    def s3(self):
        """Packed raw-sum S3 = (mu, M upper-packed, T3 upper-packed), length s3_size(dim)."""
        lib, d = self._api._lib, self.dim
        mu = (C.c_double * d)()
        m = (C.c_double * int(lib.elpis_ecsg_symmetric2_size(d)))()
        t3 = (C.c_double * int(lib.elpis_ecsg_symmetric3_size(d)))()
        _check(lib.elpis_ecsg_state_project_s3_f64(self._live(), mu, m, t3), "project S3")
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
