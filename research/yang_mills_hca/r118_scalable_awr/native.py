"""Runtime build and ctypes binding of the C11 kernel (csrc/awr_kernel.c).

The shared object is compiled into a volatile scratch directory, loaded, and the
scratch directory is removed immediately (the mapping stays valid). No binary is
written under the repository. If no C compiler is available the native backend
is reported unavailable (``NativeUnavailable``); nothing silently falls back.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import threading

from . import provenance as P

SOURCE = P.COMPONENT / "csrc" / "awr_kernel.c"
CFLAGS = ("-std=c11", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror")
ABI_VERSION = 4

STATUS = {
    0: "OK",
    1: "NONFINITE",
    2: "IN_DEGREE_LIMIT",
    3: "ARGUMENT",
    4: "OVERLAP_CONFLICT",
    5: "BINDING_LIMIT",
    6: "DEAD_HANDLE",
    7: "DUPLICATE_HANDLE",
    8: "ROW_CAPACITY",
}


class NativeUnavailable(RuntimeError):
    pass


class NativeKernel:
    def __init__(self, lib: ctypes.CDLL, info: dict):
        self.lib = lib
        self.info = info
        vp, i32, i64p, f64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_double
        lib.awr_abi_version.restype = ctypes.c_int
        lib.awr_fsum_probe.argtypes = [vp, i32]
        lib.awr_fsum_probe.restype = f64
        lib.awr_compact.argtypes = [i32, vp, vp, i32, vp, vp, vp, vp, vp, vp, vp, i32]
        lib.awr_compact.restype = ctypes.c_int
        lib.awr_release_marks.argtypes = [i32, vp, vp]
        lib.awr_release_marks.restype = None
        lib.awr_execute.argtypes = [i32, vp, vp, i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, f64, f64,
                                    i32, i32, vp, vp, vp, i64p, i32]
        lib.awr_mxcsr_probe.restype = ctypes.c_uint
        lib.awr_execute.restype = ctypes.c_int
        lib.awr_candidates.argtypes = [i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, f64, f64, i64p]
        lib.awr_candidates.restype = ctypes.c_int
        if lib.awr_abi_version() != ABI_VERSION:
            raise NativeUnavailable("ABI_VERSION")


_LOCK = threading.Lock()
_KERNEL: NativeKernel | None = None


def _compiler() -> str:
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    raise NativeUnavailable("NO_C_COMPILER")


def build() -> NativeKernel:
    global _KERNEL
    with _LOCK:
        if _KERNEL is not None:
            return _KERNEL
        cc = _compiler()
        source = SOURCE.read_bytes()
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        with P.volatile_scratch("ym_r118_native_") as d:
            src = d / "awr_kernel.c"
            src.write_bytes(source)
            so = d / "awr_kernel.so"
            proc = subprocess.run([cc, *CFLAGS, str(src), "-o", str(so), "-lm"], capture_output=True, text=True)
            if proc.returncode != 0:
                raise NativeUnavailable("COMPILE_FAILED:" + proc.stderr[-2000:])
            so_sha = hashlib.sha256(so.read_bytes()).hexdigest()
            lib = ctypes.CDLL(str(so))
        info = {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "binary_sha256": so_sha,
            "compiler": cc,
            "compiler_version": version,
            "cflags": list(CFLAGS),
            "abi_version": ABI_VERSION,
        }
        _KERNEL = NativeKernel(lib, info)
        return _KERNEL
