"""Effective numerical environment of this process (spec numerical_binding). RESEARCH_ONLY.

Records what is actually in effect, not only environment strings: the OpenBLAS kernel family NumPy's bundled
library selected (``OPENBLAS_CORETYPE`` forces it) and the thread count it really uses, the NumPy build
configuration, the CPU model and the flags that decide kernel dispatch.
"""
from __future__ import annotations

import ctypes
import os
import platform

import numpy as np

_CORE = ("openblas_get_corename64_", "scipy_openblas_get_corename64_", "openblas_get_corename")
_THREADS = ("openblas_get_num_threads64_", "scipy_openblas_get_num_threads64_", "openblas_get_num_threads")
_CONFIG = ("openblas_get_config64_", "scipy_openblas_get_config64_", "openblas_get_config")
CPU_FLAGS = ("sse4_2", "avx", "avx2", "fma", "avx512f")


def _openblas():
    try:
        with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
            path = next(line.split()[-1] for line in maps if "openblas" in line.lower())
        return ctypes.CDLL(path)
    except (OSError, StopIteration):
        return None


def _call(names, restype):
    lib = _openblas()
    for name in names if lib is not None else ():
        if hasattr(lib, name):
            fn = getattr(lib, name)
            fn.restype = restype
            return fn()
    return None


def blas_core() -> str:
    value = _call(_CORE, ctypes.c_char_p)
    return value.decode("ascii", "replace").strip() if value else "unknown"


def blas_threads() -> int:
    value = _call(_THREADS, ctypes.c_int)
    return -1 if value is None else int(value)


def blas_runtime_config() -> str:
    value = _call(_CONFIG, ctypes.c_char_p)
    return value.decode("ascii", "replace").strip() if value else "unknown"


def cpu() -> dict:
    info = {"model_name": "unknown", "family": "unknown", "model": "unknown", "flags": []}
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name" and info["model_name"] == "unknown":
                    info["model_name"] = value
                elif key == "cpu family" and info["family"] == "unknown":
                    info["family"] = value
                elif key == "model" and info["model"] == "unknown":
                    info["model"] = value
                elif key == "flags" and not info["flags"]:
                    present = set(value.split())
                    info["flags"] = [f for f in CPU_FLAGS if f in present]
    except OSError:
        pass
    return info


def numpy_build() -> dict:
    deps = np.show_config(mode="dicts")["Build Dependencies"]
    keep = ("name", "version", "openblas configuration")
    return {lib: {k: deps[lib].get(k) for k in keep} for lib in ("blas", "lapack") if lib in deps}


def profile() -> dict:
    """Everything the evidence binds about the numerical environment (spec numerical_binding)."""
    return {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "numpy": np.__version__,
        "numpy_build": numpy_build(),
        "blas": {"runtime_config": blas_runtime_config(), "core": blas_core(), "threads": blas_threads()},
        "thread_environment": {v: os.environ.get(v, "unset")
                               for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                         "OPENBLAS_CORETYPE")},
        "cpu": cpu(),
        "machine": platform.machine(),
        "os": {"system": platform.system(), "release": platform.release()},
        "libc": list(platform.libc_ver()),
        "float": "IEEE-754 binary64",
    }
