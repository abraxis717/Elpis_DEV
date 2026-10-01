"""R4 native Engram gated-write differential against the qualified NumPy tower."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from elpis.inference.drivers.dsv41.engram import gated_write


P = C.POINTER(C.c_float)


def ptr(a):
    return a.ctypes.data_as(P)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64
    lib.elpis_dsv41_engram_gated_write_f32.argtypes = [
        P, P, P, P, P, C.c_float, P, P,
        C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_engram_gated_write_f32.restype = C.c_int
    return lib


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_engram_math_diff.py /path/to/libelpis_dsv41_native.so"
        )

    lib = bind(sys.argv[1])
    assert lib.elpis_dsv41_native_capabilities() & (1 << 13)

    rng = np.random.default_rng(41404)
    worst = 0.0

    for copies, dim, row_shape, eps in (
        (2, 8, (3, 4), 1e-6),
        (4, 8, (5, 8), 1e-6),
        (4, 16, (7, 8), 1e-20),
        (4, 32, (9, 16), 1e-6),
    ):
        c = SimpleNamespace(hc_mult=copies, dimension=dim, norm_eps=eps)
        row_values = int(np.prod(row_shape))
        kv_dim = (copies + 1) * dim

        for _ in range(64):
            stream = rng.normal(size=(copies, dim)).astype("<f4")
            rows = rng.normal(size=row_shape).astype("<f4")
            rows_flat = np.ascontiguousarray(rows.reshape(-1))
            weights = {
                "wkv": rng.normal(size=(kv_dim, row_values)).astype("<f4"),
                "q_weight": rng.normal(size=(copies, dim)).astype("<f4"),
                "k_weight": rng.normal(size=(copies, dim)).astype("<f4"),
            }

            scratch = np.empty(kv_dim, dtype="<f4")
            out = np.empty((copies, dim), dtype="<f4")

            rc = lib.elpis_dsv41_engram_gated_write_f32(
                ptr(stream),
                ptr(rows_flat),
                ptr(weights["wkv"]),
                ptr(weights["q_weight"]),
                ptr(weights["k_weight"]),
                eps,
                ptr(scratch),
                ptr(out),
                copies,
                dim,
                row_values,
            )
            assert rc == 0

            expected = gated_write(stream, rows, weights, c)
            error = float(np.max(np.abs(out - expected)))
            worst = max(worst, error)
            np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    print("PASS_NATIVE_ENGRAM_MATH_DIFFERENTIAL")
    print(f"engram_gated_write_max_abs={worst:.9g}")


if __name__ == "__main__":
    main()
