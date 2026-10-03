"""Standalone R0 native-vs-NumPy differential.

This is deliberately executable without pytest or the donor/Torch oracle.
It compares the new C ABI directly against the already-qualified NumPy F32
reference. Donor parity for the NumPy side is frozen by the existing tower
qualification.
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys

import numpy as np

from research.dsv41_tower.numerics import F32, hc_post, hc_pre, linear, rms


def _ptr(array):
    return array.ctypes.data_as(C.POINTER(C.c_float))


def load(path: Path):
    lib = C.CDLL(str(path))
    p = C.POINTER(C.c_float)
    lib.elpis_dsv41_native_abi_version.argtypes = []
    lib.elpis_dsv41_native_abi_version.restype = C.c_uint32
    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64
    lib.elpis_dsv41_linear_f32.argtypes = [p, p, p, C.c_size_t, C.c_size_t]
    lib.elpis_dsv41_linear_f32.restype = C.c_int
    lib.elpis_dsv41_rms_f32.argtypes = [p, p, C.c_float, p, C.c_size_t]
    lib.elpis_dsv41_rms_f32.restype = C.c_int
    lib.elpis_dsv41_hc_pre_f32.argtypes = [p, p, p, C.c_size_t, C.c_size_t]
    lib.elpis_dsv41_hc_pre_f32.restype = C.c_int
    lib.elpis_dsv41_hc_post_f32.argtypes = [p, p, p, p, p, C.c_size_t, C.c_size_t]
    lib.elpis_dsv41_hc_post_f32.restype = C.c_int
    return lib


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: native_primitives_diff.py /path/to/libelpis_dsv41_native.so")
    lib = load(Path(sys.argv[1]).resolve())
    assert lib.elpis_dsv41_native_abi_version() == 1
    assert lib.elpis_dsv41_native_capabilities() & 0xF == 0xF

    rng = np.random.default_rng(41041)
    worst = {"linear": 0.0, "rms": 0.0, "hc_pre": 0.0, "hc_post": 0.0}

    for dim, out_dim, copies in ((4, 3, 2), (16, 11, 4), (32, 29, 4), (64, 67, 4)):
        for _ in range(32):
            x = rng.normal(size=dim).astype("<f4")
            w = rng.normal(size=(out_dim, dim)).astype("<f4")
            actual = np.empty(out_dim, dtype="<f4")
            assert lib.elpis_dsv41_linear_f32(_ptr(x), _ptr(w), _ptr(actual), dim, out_dim) == 0
            expected = linear(x, w)
            worst["linear"] = max(worst["linear"], float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)

            rw = rng.normal(size=dim).astype("<f4")
            actual_rms = np.empty(dim, dtype="<f4")
            eps = np.float32(1e-6)
            assert lib.elpis_dsv41_rms_f32(_ptr(x), _ptr(rw), float(eps), _ptr(actual_rms), dim) == 0
            expected_rms = rms(x, rw, eps)
            worst["rms"] = max(worst["rms"], float(np.max(np.abs(actual_rms - expected_rms))))
            np.testing.assert_allclose(actual_rms, expected_rms, rtol=3e-6, atol=2e-6)

            stream = rng.normal(size=(copies, dim)).astype("<f4")
            pre = rng.random(size=copies).astype("<f4")
            actual_pre = np.empty(dim, dtype="<f4")
            assert lib.elpis_dsv41_hc_pre_f32(
                _ptr(stream), _ptr(pre), _ptr(actual_pre), copies, dim
            ) == 0
            expected_pre = hc_pre(stream, pre)
            worst["hc_pre"] = max(worst["hc_pre"], float(np.max(np.abs(actual_pre - expected_pre))))
            np.testing.assert_allclose(actual_pre, expected_pre, rtol=3e-6, atol=2e-6)

            residual = rng.normal(size=(copies, dim)).astype("<f4")
            post = rng.random(size=copies).astype("<f4")
            comb = rng.random(size=(copies, copies)).astype("<f4")
            y = rng.normal(size=dim).astype("<f4")
            actual_post = np.empty((copies, dim), dtype="<f4")
            assert lib.elpis_dsv41_hc_post_f32(
                _ptr(y), _ptr(residual), _ptr(post), _ptr(comb),
                _ptr(actual_post), copies, dim
            ) == 0
            expected_post = hc_post(y, residual, post, comb)
            worst["hc_post"] = max(worst["hc_post"], float(np.max(np.abs(actual_post - expected_post))))
            np.testing.assert_allclose(actual_post, expected_post, rtol=3e-6, atol=2e-6)

    print("PASS_NATIVE_NUMPY_DIFFERENTIAL")
    for name, value in worst.items():
        print(f"{name}_max_abs={value:.9g}")


if __name__ == "__main__":
    main()
