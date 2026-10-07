"""Portable deterministic snapshot qualification for ECS state."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from .test_math_r0 import _library_path, _ptr


@pytest.fixture(scope="module")
def lib():
    dll = ctypes.CDLL(str(_library_path()))
    p = ctypes.POINTER(ctypes.c_double)
    b = ctypes.POINTER(ctypes.c_uint8)
    vp = ctypes.c_void_p
    vpp = ctypes.POINTER(vp)

    dll.elpis_ecsg_state_create.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, p, vpp
    ]
    dll.elpis_ecsg_state_create.restype = ctypes.c_int

    dll.elpis_ecsg_state_destroy.argtypes = [vpp]
    dll.elpis_ecsg_state_destroy.restype = ctypes.c_int

    dll.elpis_ecsg_state_epoch.argtypes = [vp]
    dll.elpis_ecsg_state_epoch.restype = ctypes.c_uint64

    dll.elpis_ecsg_state_copy_w.argtypes = [vp, p, ctypes.c_size_t]
    dll.elpis_ecsg_state_copy_w.restype = ctypes.c_int

    dll.elpis_ecsg_state_snapshot_size.argtypes = [vp]
    dll.elpis_ecsg_state_snapshot_size.restype = ctypes.c_size_t

    dll.elpis_ecsg_state_snapshot_write.argtypes = [
        vp, b, ctypes.c_size_t
    ]
    dll.elpis_ecsg_state_snapshot_write.restype = ctypes.c_int

    dll.elpis_ecsg_state_snapshot_restore.argtypes = [
        b, ctypes.c_size_t, vpp
    ]
    dll.elpis_ecsg_state_snapshot_restore.restype = ctypes.c_int

    return dll


def _bptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


@pytest.mark.parametrize("width", [36, 48, 72])
def test_snapshot_roundtrip_is_byte_exact_and_repeatable(lib, width):
    rng = np.random.Generator(np.random.PCG64(22000 + width))
    w = np.ascontiguousarray(
        rng.normal(0.0, 0.18, size=(6, width)),
        dtype=np.float64,
    )

    state = ctypes.c_void_p()
    assert lib.elpis_ecsg_state_create(
        6, width, _ptr(w), ctypes.byref(state)
    ) == 0

    size = lib.elpis_ecsg_state_snapshot_size(state)
    assert size == 40 + w.size * 8

    left = np.empty(size, dtype=np.uint8)
    right = np.empty(size, dtype=np.uint8)

    assert lib.elpis_ecsg_state_snapshot_write(
        state, _bptr(left), size
    ) == 0
    assert lib.elpis_ecsg_state_snapshot_write(
        state, _bptr(right), size
    ) == 0

    assert np.array_equal(left, right)

    restored = ctypes.c_void_p()
    assert lib.elpis_ecsg_state_snapshot_restore(
        _bptr(left), size, ctypes.byref(restored)
    ) == 0

    actual = np.empty_like(w)
    assert lib.elpis_ecsg_state_copy_w(
        restored, _ptr(actual), actual.size
    ) == 0

    assert lib.elpis_ecsg_state_epoch(restored) == 0
    assert np.array_equal(actual.view(np.uint64), w.view(np.uint64))

    again = np.empty(size, dtype=np.uint8)
    assert lib.elpis_ecsg_state_snapshot_write(
        restored, _bptr(again), size
    ) == 0
    assert np.array_equal(again, left)

    assert lib.elpis_ecsg_state_destroy(ctypes.byref(state)) == 0
    assert lib.elpis_ecsg_state_destroy(ctypes.byref(restored)) == 0


def test_truncated_snapshot_fails_closed(lib):
    w = np.zeros((6, 36), dtype=np.float64)
    state = ctypes.c_void_p()

    assert lib.elpis_ecsg_state_create(
        6, 36, _ptr(w), ctypes.byref(state)
    ) == 0

    size = lib.elpis_ecsg_state_snapshot_size(state)
    blob = np.empty(size, dtype=np.uint8)

    assert lib.elpis_ecsg_state_snapshot_write(
        state, _bptr(blob), size
    ) == 0

    restored = ctypes.c_void_p()
    assert lib.elpis_ecsg_state_snapshot_restore(
        _bptr(blob), size - 1, ctypes.byref(restored)
    ) == -1
    assert restored.value is None

    assert lib.elpis_ecsg_state_destroy(ctypes.byref(state)) == 0
