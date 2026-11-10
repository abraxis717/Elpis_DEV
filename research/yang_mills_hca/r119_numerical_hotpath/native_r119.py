"""Runtime build of R119 C sources into volatile scratch (no binary is ever written to the repo).

Same discipline as R118 ``native.py``: compile into ``/dev/shm``, load, delete the
directory immediately, record compiler identity, flags, source and binary SHA-256.
"""
from __future__ import annotations

import ctypes
import hashlib
import pathlib
import shutil
import subprocess
import threading

from research.yang_mills_hca.r118_scalable_awr import provenance as P

HERE = pathlib.Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / "csrc" / "awr_kernel_r119.c"
MICROBENCH_SOURCE = HERE / "csrc" / "fp_microbench.c"
KERNEL_CFLAGS = ("-std=c11", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror")
MICROBENCH_CFLAGS = ("-std=c11", "-O2", "-ffp-contract=off", "-fno-tree-vectorize", "-fPIC", "-shared", "-Wall",
                     "-Wextra", "-Werror")
KERNEL_ABI = 119


class BuildError(RuntimeError):
    pass


_LOCK = threading.Lock()
_CACHE: dict[str, tuple] = {}


def compiler() -> str:
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    raise BuildError("NO_C_COMPILER")


def build(source: pathlib.Path, cflags: tuple, key: str):
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        cc = compiler()
        data = source.read_bytes()
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        with P.volatile_scratch("ym_r119_native_") as d:
            src = d / source.name
            src.write_bytes(data)
            so = d / (source.stem + ".so")
            proc = subprocess.run([cc, *cflags, str(src), "-o", str(so), "-lm"], capture_output=True, text=True)
            if proc.returncode != 0:
                raise BuildError("COMPILE_FAILED:" + proc.stderr[-2000:])
            lib = ctypes.CDLL(str(so))
            info = {"source": source.name, "source_sha256": hashlib.sha256(data).hexdigest(),
                    "binary_sha256": hashlib.sha256(so.read_bytes()).hexdigest(), "compiler": cc,
                    "compiler_version": version, "cflags": list(cflags)}
        _CACHE[key] = (lib, info)
        return lib, info


def microbench():
    lib, info = build(MICROBENCH_SOURCE, MICROBENCH_CFLAGS, "microbench")
    lib.fpmb_run.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                             ctypes.c_int32, ctypes.c_int32]
    lib.fpmb_run.restype = ctypes.c_double
    return lib, info


STATUS = {0: "OK", 1: "NONFINITE", 2: "IN_DEGREE_LIMIT", 3: "ARGUMENT", 4: "OVERLAP_CONFLICT", 5: "BINDING_LIMIT",
          6: "DEAD_HANDLE", 7: "DUPLICATE_HANDLE", 8: "ROW_CAPACITY"}


def kernel():
    """The R119 guarded kernel (exported symbols prefixed awr119_)."""
    lib, info = build(KERNEL_SOURCE, KERNEL_CFLAGS, "kernel")
    if getattr(lib, "_r119_bound", False):
        return lib, info
    vp, i32, u32, f64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_double
    lib.awr119_abi_version.restype = ctypes.c_int
    lib.awr119_libm_selfcheck.restype = ctypes.c_int
    lib.awr119_compact.argtypes = [i32, vp, vp, i32, vp, vp, vp, vp, vp, vp, vp, i32]
    lib.awr119_compact.restype = ctypes.c_int
    lib.awr119_release_marks.argtypes = [i32, vp, vp]
    lib.awr119_release_marks.restype = None
    lib.awr119_execute.argtypes = [i32, vp, vp, i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, f64, f64,
                                   i32, i32, vp, vp, vp, vp, i32, u32, f64, vp]
    lib.awr119_execute.restype = ctypes.c_int
    lib.awr119_candidates.argtypes = [i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, i32, f64, f64, vp, u32, f64, vp]
    lib.awr119_candidates.restype = ctypes.c_int
    lib.awr119_fsum_probe.argtypes = [vp, i32]
    lib.awr119_fsum_probe.restype = f64
    if lib.awr119_abi_version() != KERNEL_ABI:
        raise BuildError("ABI")
    lib._r119_bound = True
    return lib, info
