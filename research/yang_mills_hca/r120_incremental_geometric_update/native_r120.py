"""Runtime build of the R120 kernel (csrc/awr_kernel_r120.c) into volatile scratch.

The R120 translation unit ``#include``s the frozen R119 kernel source, so both files are
copied into one scratch directory, compiled with the R119 flags, loaded and removed. No
binary is written under the repository. The R119 source hash is checked before the build.
"""
from __future__ import annotations

import ctypes
import hashlib
import pathlib
import subprocess
import threading

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r119_numerical_hotpath import native_r119 as N119

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "csrc" / "awr_kernel_r120.c"
R119_KERNEL_SHA256 = "125e138b11d0189bc46093b257ec0885ea7e49e875c67fc11b4f91ca21d291e2"
CFLAGS = N119.KERNEL_CFLAGS + ("-D_POSIX_C_SOURCE=200809L", "-fopenmp")   # clock_gettime; opt-in parallel Jacobi
ABI = 120

STATUS = {0: "OK", 1: "COMMITTED_DEFERRED", 10: "REFUSED_ARGUMENT", 11: "REFUSED_INVALID_VALUE",
          12: "REFUSED_UNKNOWN_RELATION", 13: "REFUSED_NOT_MATERIALIZED", 14: "REFUSED_STALE_RELATION",
          15: "REFUSED_NOT_INCIDENT", 16: "REFUSED_CONFLICT", 17: "REFUSED_EVENT_BUDGET",
          18: "REFUSED_INSTANCE_BUDGET", 19: "REFUSED_ROW_BUDGET", 20: "REFUSED_LINK_BUDGET",
          21: "REFUSED_NONFINITE", 22: "REFUSED_KERNEL", 23: "REFUSED_PENDING_CAPACITY"}
REPORT_FIELDS = ("status", "fail_index", "applied", "skipped_duplicate", "touched_rows", "instances", "rows",
                 "links", "pending_rows", "kernel_rc", "ns_validate_update", "ns_select", "ns_hca", "ns_reconcile",
                 "ns_commit", "ns_total", "cached_instances", "ns_cgc")
FLAG_PROPAGATE, FLAG_DEFER_ON_BUDGET, FLAG_PROFILE = 1, 2, 4

_P, _I64, _F64, _I32, _U32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int32, ctypes.c_uint32
POINTER_FIELDS = ("relation_ids", "rel_off", "members", "frames", "pool", "rel_row", "inst_live", "slot_rel",
                  "slot_row", "row_head", "bind_next", "mark", "inst_links", "row_seq", "row_gen", "row_pending",
                  "pending_list", "inst_mark", "wc_off", "wc_data", "W", "bias", "cgc_w", "handles", "s_slot_rel",
                  "s_slot_cmp", "rows", "count", "off", "list", "last_event", "buf_a", "buf_b", "cand", "seq_new",
                  "gc")


class Ctx(ctypes.Structure):
    """Mirror of ``awr120_ctx`` (field order and types are the C declaration)."""
    _fields_ = ([(n, _P) for n in POINTER_FIELDS]
                + [("n_rel", _I64), ("max_links", _I64), ("pending_count", _I64)]
                + [(n, _F64) for n in ("alpha", "beta", "theta", "eta", "gamma_v", "rho_b", "kappa")]
                + [(n, _I32) for n in ("inst_capacity", "row_capacity", "max_events", "max_instances", "max_rows",
                                       "correct", "mode", "pending_capacity")]
                + [("gmask", _U32), ("threads", _I32), ("parallel_min", _I32), ("pad_", _I32)])


_LOCK = threading.Lock()
_CACHE = None


def kernel():
    """(lib, info): the loaded R120 kernel with bound signatures."""
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        r119 = N119.KERNEL_SOURCE.read_bytes()
        if hashlib.sha256(r119).hexdigest() != R119_KERNEL_SHA256:
            raise N119.BuildError("R119_KERNEL_SOURCE_DRIFT")
        src = SOURCE.read_bytes()
        cc = N119.compiler()
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        with P.volatile_scratch("ym_r120_native_") as d:
            (d / "awr_kernel_r119.c").write_bytes(r119)
            (d / "awr_kernel_r120.c").write_bytes(src)
            so = d / "awr_kernel_r120.so"
            proc = subprocess.run([cc, *CFLAGS, str(d / "awr_kernel_r120.c"), "-o", str(so), "-lm"],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise N119.BuildError("COMPILE_FAILED:" + proc.stderr[-2000:])
            lib = ctypes.CDLL(str(so))
            binary = hashlib.sha256(so.read_bytes()).hexdigest()
        cp = ctypes.POINTER(Ctx)
        lib.awr120_abi_version.restype = ctypes.c_int
        lib.awr120_openmp.restype = ctypes.c_int
        lib.awr120_ctx_bytes.restype = ctypes.c_int
        lib.awr119_libm_selfcheck.restype = ctypes.c_int
        lib.awr120_update_propagate.argtypes = [cp, _I32, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _P]
        lib.awr120_update_propagate.restype = ctypes.c_int
        lib.awr120_propagate_pending.argtypes = [cp, _I32, _I32, _I32, _P]
        lib.awr120_propagate_pending.restype = ctypes.c_int
        lib.awr120_step.argtypes = [cp, _I32, _P, _I32, _P]
        lib.awr120_step.restype = ctypes.c_int
        lib.awr120_cache_build.argtypes = [cp, _I32, _P, _P, _P, _I64, _I64, _P]
        lib.awr120_cache_build.restype = _I64
        lib.awr120_cgc_probe.argtypes = [_P, _P, _P]
        lib.awr120_cgc_probe.restype = None
        lib.awr120_update_probe.argtypes = [cp, _P, _P, _P, _F64, _F64, _P]
        lib.awr120_update_probe.restype = None
        if lib.awr120_abi_version() != ABI or lib.awr120_ctx_bytes() != ctypes.sizeof(Ctx):
            raise N119.BuildError("ABI_OR_LAYOUT")
        info = {"source_sha256": hashlib.sha256(src).hexdigest(), "r119_kernel_sha256": R119_KERNEL_SHA256,
                "binary_sha256": binary, "compiler_version": version, "cflags": list(CFLAGS)}
        _CACHE = (lib, info)
        return _CACHE
