"""Effective numerical runtime of this process, for replay preconditions and diagnostics (tests only).

NumPy's bundled OpenBLAS picks its kernel family per CPU at load time (``OPENBLAS_CORETYPE`` overrides it).
The v1 Cognitive R0 and Retention R0 records bind Python, NumPy and machine, not this kernel; which kernels
reproduce their bytes was established after the fact (docs/research/COGNITION_R0_RESULTS.md and
ECS_RETENTION_R0_RESULTS.md, "Reproduction contract").
"""
from __future__ import annotations

import ctypes

_NAMES = ("openblas_get_corename64_", "scipy_openblas_get_corename64_", "openblas_get_corename")
_THREADS = ("openblas_get_num_threads64_", "scipy_openblas_get_num_threads64_", "openblas_get_num_threads")


def _library():
    try:
        with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
            path = next(line.split()[-1] for line in maps if "openblas" in line.lower())
        return ctypes.CDLL(path)
    except (OSError, StopIteration):
        return None


def openblas_core() -> str:
    """The OpenBLAS kernel family in use (``"unknown"`` if NumPy is not on OpenBLAS or not loaded yet)."""
    lib = _library()
    for name in _NAMES if lib is not None else ():
        if hasattr(lib, name):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_char_p
            return fn().decode("ascii", "replace").strip()
    return "unknown"


def openblas_threads() -> int:
    """Threads OpenBLAS actually uses (``-1`` if unknown); the environment variables may say otherwise."""
    lib = _library()
    for name in _THREADS if lib is not None else ():
        if hasattr(lib, name):
            return int(getattr(lib, name)())
    return -1
