"""Differential qualification for ECS_G G1 stateful microscopic recurrence."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from ._math_oracle import forward, gd_step, project_s3
from .test_math_r0 import _library_path, _ptr


@pytest.fixture(scope="module")
def state_lib():
    dll = ctypes.CDLL(str(_library_path()))
    p = ctypes.POINTER(ctypes.c_double)
    vp = ctypes.c_void_p
    vpp = ctypes.POINTER(vp)

    dll.elpis_ecsg_state_abi_version.restype = ctypes.c_uint32

    dll.elpis_ecsg_state_create.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        p,
        vpp,
    ]
    dll.elpis_ecsg_state_create.restype = ctypes.c_int

    dll.elpis_ecsg_state_destroy.argtypes = [vpp]
    dll.elpis_ecsg_state_destroy.restype = ctypes.c_int

    dll.elpis_ecsg_state_dim.argtypes = [vp]
    dll.elpis_ecsg_state_dim.restype = ctypes.c_size_t
    dll.elpis_ecsg_state_width.argtypes = [vp]
    dll.elpis_ecsg_state_width.restype = ctypes.c_size_t
    dll.elpis_ecsg_state_epoch.argtypes = [vp]
    dll.elpis_ecsg_state_epoch.restype = ctypes.c_uint64

    dll.elpis_ecsg_state_copy_w.argtypes = [
        vp,
        p,
        ctypes.c_size_t,
    ]
    dll.elpis_ecsg_state_copy_w.restype = ctypes.c_int

    dll.elpis_ecsg_state_project_s3_f64.argtypes = [vp, p, p, p]
    dll.elpis_ecsg_state_project_s3_f64.restype = ctypes.c_int

    dll.elpis_ecsg_state_gd_step_scratch_f64.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    dll.elpis_ecsg_state_gd_step_scratch_f64.restype = ctypes.c_size_t

    dll.elpis_ecsg_state_gd_step_f64.argtypes = [
        vp,
        p,
        p,
        ctypes.c_size_t,
        ctypes.c_double,
        p,
        ctypes.c_size_t,
    ]
    dll.elpis_ecsg_state_gd_step_f64.restype = ctypes.c_int

    return dll


@pytest.mark.parametrize("width", [36, 48, 72])
def test_multistep_matches_frozen_branch36_transition(state_lib, width):
    dim = 6
    base = 32
    rng = np.random.Generator(np.random.PCG64(36000 + width))

    teacher = rng.normal(0.0, 0.18, size=(dim, width)).astype(np.float64)
    student = (
        teacher
        + rng.normal(0.0, 0.12, size=(dim, width))
    ).astype(np.float64)

    x_half = rng.normal(size=(base, dim)).astype(np.float64)
    x_half /= np.linalg.norm(x_half, axis=1, keepdims=True)
    x_half *= np.sqrt(float(dim))
    x = np.ascontiguousarray(
        np.concatenate([x_half, -x_half], axis=0),
        dtype=np.float64,
    )
    y = np.ascontiguousarray(forward(teacher, x), dtype=np.float64)

    reference = np.ascontiguousarray(student.copy())
    state = ctypes.c_void_p()

    assert state_lib.elpis_ecsg_state_abi_version() == 1
    assert state_lib.elpis_ecsg_state_create(
        dim, width, _ptr(student), ctypes.byref(state)
    ) == 0
    assert state.value is not None
    assert state_lib.elpis_ecsg_state_dim(state) == dim
    assert state_lib.elpis_ecsg_state_width(state) == width
    assert state_lib.elpis_ecsg_state_epoch(state) == 0

    scratch_count = state_lib.elpis_ecsg_state_gd_step_scratch_f64(
        dim, width, x.shape[0]
    )
    assert scratch_count > 0
    scratch = np.empty(scratch_count, dtype=np.float64)

    for epoch in range(1, 9):
        reference = np.ascontiguousarray(
            gd_step(reference, x, y, 0.002),
            dtype=np.float64,
        )

        assert state_lib.elpis_ecsg_state_gd_step_f64(
            state,
            _ptr(x),
            _ptr(y),
            x.shape[0],
            0.002,
            _ptr(scratch),
            scratch_count,
        ) == 0

        assert state_lib.elpis_ecsg_state_epoch(state) == epoch

    actual = np.empty((dim, width), dtype=np.float64)
    assert state_lib.elpis_ecsg_state_copy_w(
        state, _ptr(actual), actual.size
    ) == 0

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=3e-12,
        atol=3e-12,
    )

    mu_ref, m_ref, t3_ref = project_s3(reference)
    mu = np.empty(dim, dtype=np.float64)
    m = np.empty(dim * (dim + 1) // 2, dtype=np.float64)
    t3 = np.empty(dim * (dim + 1) * (dim + 2) // 6, dtype=np.float64)

    assert state_lib.elpis_ecsg_state_project_s3_f64(
        state, _ptr(mu), _ptr(m), _ptr(t3)
    ) == 0

    np.testing.assert_allclose(mu, mu_ref, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(m, m_ref, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(t3, t3_ref, rtol=8e-12, atol=8e-12)

    assert state_lib.elpis_ecsg_state_destroy(ctypes.byref(state)) == 0
    assert state.value is None


def test_rejected_step_is_atomic(state_lib):
    dim = 6
    width = 36

    w = np.zeros((dim, width), dtype=np.float64)
    x = np.zeros((2, dim), dtype=np.float64)
    y = np.asarray([0.0, np.nan], dtype=np.float64)

    state = ctypes.c_void_p()
    assert state_lib.elpis_ecsg_state_create(
        dim, width, _ptr(w), ctypes.byref(state)
    ) == 0

    scratch_count = state_lib.elpis_ecsg_state_gd_step_scratch_f64(
        dim, width, len(y)
    )
    scratch = np.empty(scratch_count, dtype=np.float64)

    before = np.empty_like(w)
    assert state_lib.elpis_ecsg_state_copy_w(
        state, _ptr(before), before.size
    ) == 0

    assert state_lib.elpis_ecsg_state_gd_step_f64(
        state,
        _ptr(x),
        _ptr(y),
        len(y),
        0.002,
        _ptr(scratch),
        scratch_count,
    ) == -2

    after = np.empty_like(w)
    assert state_lib.elpis_ecsg_state_copy_w(
        state, _ptr(after), after.size
    ) == 0

    assert state_lib.elpis_ecsg_state_epoch(state) == 0
    assert np.array_equal(before, after)

    assert state_lib.elpis_ecsg_state_destroy(ctypes.byref(state)) == 0
