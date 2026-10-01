from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from elpis.inference.drivers.dsv41.attention import AttentionState
from elpis.inference.drivers.dsv41.numerics import softmax


P = C.POINTER(C.c_float)
U32P = C.POINTER(C.c_uint32)


class State(C.Structure):
    pass


StateP = C.POINTER(State)


def ptr(a):
    return a.ctypes.data_as(P)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))

    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64

    lib.elpis_dsv41_attention_state_create.argtypes = [
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_int, C.POINTER(StateP),
    ]
    lib.elpis_dsv41_attention_state_create.restype = C.c_int

    lib.elpis_dsv41_attention_state_destroy.argtypes = [C.POINTER(StateP)]
    lib.elpis_dsv41_attention_state_destroy.restype = C.c_int

    lib.elpis_dsv41_attention_state_bytes.argtypes = [StateP]
    lib.elpis_dsv41_attention_state_bytes.restype = C.c_size_t

    lib.elpis_dsv41_attention_state_count.argtypes = [StateP]
    lib.elpis_dsv41_attention_state_count.restype = C.c_size_t

    lib.elpis_dsv41_attention_local_store_f32.argtypes = [StateP, C.c_size_t, P]
    lib.elpis_dsv41_attention_local_store_f32.restype = C.c_int

    lib.elpis_dsv41_attention_local_copy_f32.argtypes = [
        StateP, C.c_size_t, P, C.c_size_t, C.POINTER(C.c_size_t),
    ]
    lib.elpis_dsv41_attention_local_copy_f32.restype = C.c_int

    lib.elpis_dsv41_attention_compress_push_f32.argtypes = [
        StateP, C.c_size_t, P, P, P, C.POINTER(C.c_int),
    ]
    lib.elpis_dsv41_attention_compress_push_f32.restype = C.c_int

    lib.elpis_dsv41_attention_publish_group_f32.argtypes = [
        StateP, C.c_size_t, P, P,
    ]
    lib.elpis_dsv41_attention_publish_group_f32.restype = C.c_int

    lib.elpis_dsv41_attention_index_copy_f32.argtypes = [
        StateP, P, C.c_size_t, C.POINTER(C.c_size_t),
    ]
    lib.elpis_dsv41_attention_index_copy_f32.restype = C.c_int

    lib.elpis_dsv41_attention_gather_compressed_f32.argtypes = [
        StateP, U32P, C.c_size_t, P,
    ]
    lib.elpis_dsv41_attention_gather_compressed_f32.restype = C.c_int

    return lib


def make_config(local_window, head_dim, index_dim, max_tokens, ratio, owner):
    return SimpleNamespace(
        kv_sources=(0,) if owner else (),
        compress_ratios=(ratio,),
        max_tokens=max_tokens,
        local_window=local_window,
        head_dim=head_dim,
        index_dim=index_dim,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: native_attention_state_diff.py /path/to/libelpis_dsv41_native.so")

    lib = bind(sys.argv[1])
    assert lib.elpis_dsv41_native_capabilities() & (1 << 12)

    rng = np.random.default_rng(41303)
    worst_compress = 0.0

    cases = (
        (5, 8, 8, 24, 0, False),
        (5, 8, 8, 24, 2, False),
        (7, 8, 8, 24, 1, True),
        (7, 8, 8, 24, 2, True),
        (7, 8, 8, 24, 3, True),
    )

    for local_window, head_dim, index_dim, max_tokens, ratio, owner in cases:
        py = AttentionState.create(
            make_config(local_window, head_dim, index_dim, max_tokens, ratio, owner),
            0,
        )

        native = StateP()
        assert lib.elpis_dsv41_attention_state_create(
            local_window, head_dim, index_dim, max_tokens, ratio,
            int(owner), C.byref(native),
        ) == 0

        try:
            assert lib.elpis_dsv41_attention_state_bytes(native) == py.nbytes
            assert lib.elpis_dsv41_attention_state_count(native) == py.count == 0

            for position in range(min(max_tokens, local_window * 2 + 3)):
                local = rng.normal(size=head_dim).astype("<f4")
                py.window[position % local_window] = local

                assert lib.elpis_dsv41_attention_local_store_f32(
                    native, position, ptr(local)
                ) == 0

                rows_expected = min(position + 1, local_window)
                start = position + 1 - rows_expected
                slots = np.arange(start, position + 1) % local_window
                expected = py.window[slots]

                out = np.empty((local_window, head_dim), dtype="<f4")
                rows = C.c_size_t()

                assert lib.elpis_dsv41_attention_local_copy_f32(
                    native, position, ptr(out), local_window, C.byref(rows)
                ) == 0

                assert rows.value == rows_expected
                np.testing.assert_array_equal(out[:rows.value], expected)

            if not owner:
                continue

            pending = np.zeros((ratio, head_dim), dtype="<f4") if ratio > 1 else None
            scores = np.zeros((ratio, head_dim), dtype="<f4") if ratio > 1 else None

            expected_compressed = []
            expected_indexes = []

            for position in range(max_tokens):
                kv = rng.normal(size=head_dim).astype("<f4")
                gate = rng.normal(size=head_dim).astype("<f4")
                latent = np.empty(head_dim, dtype="<f4")
                ready = C.c_int(-1)

                assert lib.elpis_dsv41_attention_compress_push_f32(
                    native, position, ptr(kv), ptr(gate),
                    ptr(latent), C.byref(ready)
                ) == 0

                if ratio == 1:
                    expected_ready = True
                    expected_latent = kv
                else:
                    slot = position % ratio
                    pending[slot] = kv
                    scores[slot] = gate
                    expected_ready = ((position + 1) % ratio) == 0
                    if expected_ready:
                        expected_latent = np.sum(
                            pending * softmax(scores, axis=0),
                            axis=0,
                            dtype=np.float32,
                        )

                assert bool(ready.value) == expected_ready

                if not expected_ready:
                    continue

                worst_compress = max(
                    worst_compress,
                    float(np.max(np.abs(latent - expected_latent))),
                )

                np.testing.assert_allclose(
                    latent, expected_latent, rtol=5e-6, atol=3e-6
                )

                compressed = rng.normal(size=head_dim).astype("<f4")
                index_key = rng.normal(size=index_dim).astype("<f4")

                assert lib.elpis_dsv41_attention_publish_group_f32(
                    native, position, ptr(compressed), ptr(index_key)
                ) == 0

                expected_compressed.append(compressed.copy())
                expected_indexes.append(index_key.copy())

                assert lib.elpis_dsv41_attention_state_count(native) == len(expected_indexes)

                index_out = np.empty((max_tokens // ratio, index_dim), dtype="<f4")
                rows = C.c_size_t()

                assert lib.elpis_dsv41_attention_index_copy_f32(
                    native, ptr(index_out), len(index_out), C.byref(rows)
                ) == 0

                assert rows.value == len(expected_indexes)

                np.testing.assert_array_equal(
                    index_out[:rows.value],
                    np.asarray(expected_indexes, dtype="<f4"),
                )

                selected = np.arange(
                    len(expected_compressed), dtype=np.uint32
                )[::-1].copy()

                gathered = np.empty(
                    (len(selected), head_dim), dtype="<f4"
                )

                assert lib.elpis_dsv41_attention_gather_compressed_f32(
                    native,
                    selected.ctypes.data_as(U32P),
                    len(selected),
                    ptr(gathered),
                ) == 0

                np.testing.assert_array_equal(
                    gathered,
                    np.asarray(expected_compressed, dtype="<f4")[selected],
                )

        finally:
            assert lib.elpis_dsv41_attention_state_destroy(
                C.byref(native)
            ) == 0
            assert not bool(native)
            assert lib.elpis_dsv41_attention_state_destroy(
                C.byref(native)
            ) == 0

    print("PASS_NATIVE_ATTENTION_STATE_DIFFERENTIAL")
    print(f"compress_max_abs={worst_compress:.9g}")


if __name__ == "__main__":
    main()
