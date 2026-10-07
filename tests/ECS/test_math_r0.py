"""Differential qualification for the native ECS cubic/S3 kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
import os

import numpy as np
import pytest

from ._math_oracle import forward, forward_s3, project_s3


REPO = Path(__file__).resolve().parents[2]


def _library_path() -> Path:
    root = Path(
        os.environ.get("ELPIS_NATIVE_BUILD", str(REPO / "build"))
    ).resolve()

    matches = sorted(root.rglob("libelpis_ecsg_math.so")) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]

    if os.environ.get("ELPIS_REQUIRE_NATIVE") == "1":
        raise AssertionError(
            f"expected exactly one libelpis_ecsg_math.so under {root}, got {matches}"
        )

    pytest.skip(f"native ECS math library unavailable under {root}")


@pytest.fixture(scope="module")
def lib():
    dll = ctypes.CDLL(str(_library_path()))
    p = ctypes.POINTER(ctypes.c_double)

    dll.elpis_ecsg_math_abi_version.restype = ctypes.c_uint32

    dll.elpis_ecsg_symmetric2_size.argtypes = [ctypes.c_size_t]
    dll.elpis_ecsg_symmetric2_size.restype = ctypes.c_size_t
    dll.elpis_ecsg_symmetric3_size.argtypes = [ctypes.c_size_t]
    dll.elpis_ecsg_symmetric3_size.restype = ctypes.c_size_t
    dll.elpis_ecsg_s3_size.argtypes = [ctypes.c_size_t]
    dll.elpis_ecsg_s3_size.restype = ctypes.c_size_t

    dll.elpis_ecsg_project_s3_f64.argtypes = [
        p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        p,
        p,
        p,
    ]
    dll.elpis_ecsg_project_s3_f64.restype = ctypes.c_int

    dll.elpis_ecsg_forward_f64.argtypes = [
        p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        p,
        ctypes.c_size_t,
        p,
    ]
    dll.elpis_ecsg_forward_f64.restype = ctypes.c_int

    dll.elpis_ecsg_forward_s3_f64.argtypes = [
        p,
        p,
        p,
        ctypes.c_size_t,
        p,
        ctypes.c_size_t,
        p,
    ]
    dll.elpis_ecsg_forward_s3_f64.restype = ctypes.c_int

    return dll


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def test_s3_dimension_is_83_for_frozen_d6(lib):
    assert lib.elpis_ecsg_math_abi_version() == 1
    assert lib.elpis_ecsg_symmetric2_size(6) == 21
    assert lib.elpis_ecsg_symmetric3_size(6) == 56
    assert lib.elpis_ecsg_s3_size(6) == 83


@pytest.mark.parametrize("width", [36, 48, 72])
def test_native_projection_and_forward_match_frozen_math(lib, width):
    dim = 6
    rng = np.random.Generator(np.random.PCG64(717 + width))

    w = np.ascontiguousarray(
        rng.normal(0.0, 0.18, size=(dim, width)),
        dtype=np.float64,
    )

    x = rng.normal(size=(64, dim)).astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x *= np.sqrt(float(dim))
    x = np.ascontiguousarray(x)

    mu_ref, m_ref, t3_ref = project_s3(w)
    y_ref = forward(w, x)
    y_s3_ref = forward_s3(mu_ref, m_ref, t3_ref, x)

    mu = np.empty(dim, dtype=np.float64)
    m = np.empty(21, dtype=np.float64)
    t3 = np.empty(56, dtype=np.float64)
    y = np.empty(x.shape[0], dtype=np.float64)
    y_s3 = np.empty(x.shape[0], dtype=np.float64)

    assert lib.elpis_ecsg_project_s3_f64(
        _ptr(w), dim, width, _ptr(mu), _ptr(m), _ptr(t3)
    ) == 0

    assert lib.elpis_ecsg_forward_f64(
        _ptr(w), dim, width, _ptr(x), x.shape[0], _ptr(y)
    ) == 0

    assert lib.elpis_ecsg_forward_s3_f64(
        _ptr(mu), _ptr(m), _ptr(t3), dim, _ptr(x), x.shape[0], _ptr(y_s3)
    ) == 0

    np.testing.assert_allclose(mu, mu_ref, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(m, m_ref, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(t3, t3_ref, rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(y, y_ref, rtol=5e-13, atol=5e-13)

    # Exact algebra, floating implementation tolerance only.
    np.testing.assert_allclose(y_s3_ref, y_ref, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(y_s3, y_ref, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("width", [36, 48, 72])
def test_entity_permutation_preserves_collective_state(lib, width):
    dim = 6
    rng = np.random.Generator(np.random.PCG64(1717 + width))

    w = np.ascontiguousarray(
        rng.normal(0.0, 0.18, size=(dim, width)),
        dtype=np.float64,
    )
    permuted = np.ascontiguousarray(w[:, rng.permutation(width)])

    left = [
        np.empty(dim, dtype=np.float64),
        np.empty(21, dtype=np.float64),
        np.empty(56, dtype=np.float64),
    ]
    right = [np.empty_like(a) for a in left]

    assert lib.elpis_ecsg_project_s3_f64(
        _ptr(w), dim, width, *(_ptr(a) for a in left)
    ) == 0
    assert lib.elpis_ecsg_project_s3_f64(
        _ptr(permuted), dim, width, *(_ptr(a) for a in right)
    ) == 0

    for a, b in zip(left, right):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)


def test_nonfinite_input_fails_closed(lib):
    w = np.zeros((6, 36), dtype=np.float64)
    w[2, 7] = np.nan

    mu = np.empty(6, dtype=np.float64)
    m = np.empty(21, dtype=np.float64)
    t3 = np.empty(56, dtype=np.float64)

    assert lib.elpis_ecsg_project_s3_f64(
        _ptr(w), 6, 36, _ptr(mu), _ptr(m), _ptr(t3)
    ) == -2
