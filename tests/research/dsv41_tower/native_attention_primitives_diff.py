"""R1 native-vs-qualified-NumPy differential for stateless attention primitives."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys

import numpy as np

from research.dsv41_tower.numerics import quant_dequant, rotate
from research.dsv41_tower.attention import (
    candidate_mask,
    select_positions,
    sparse_attention,
)


P = C.POINTER(C.c_float)
U8P = C.POINTER(C.c_uint8)
U32P = C.POINTER(C.c_uint32)


def ptr(a):
    return a.ctypes.data_as(P)


def load(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.elpis_dsv41_native_abi_version.argtypes = []
    lib.elpis_dsv41_native_abi_version.restype = C.c_uint32
    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64

    lib.elpis_dsv41_quant_dequant_f32.argtypes = [
        P, P, C.c_size_t, C.c_size_t, C.c_uint32
    ]
    lib.elpis_dsv41_quant_dequant_f32.restype = C.c_int

    lib.elpis_dsv41_rope_f32.argtypes = [
        P, P, P, C.c_size_t, C.c_size_t, C.c_size_t, C.c_int
    ]
    lib.elpis_dsv41_rope_f32.restype = C.c_int

    lib.elpis_dsv41_candidate_mask_f32.argtypes = [
        P, U8P, C.c_size_t, C.c_size_t, C.c_size_t
    ]
    lib.elpis_dsv41_candidate_mask_f32.restype = C.c_int

    lib.elpis_dsv41_select_positions_f32.argtypes = [
        P, U32P, C.c_size_t, C.c_size_t, C.POINTER(C.c_size_t)
    ]
    lib.elpis_dsv41_select_positions_f32.restype = C.c_int

    lib.elpis_dsv41_sparse_attention_f32.argtypes = [
        P, P, P, P, P, C.c_size_t, C.c_size_t, C.c_size_t
    ]
    lib.elpis_dsv41_sparse_attention_f32.restype = C.c_int
    return lib


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_attention_primitives_diff.py "
            "/path/to/libelpis_dsv41_native.so"
        )

    lib = load(sys.argv[1])
    assert lib.elpis_dsv41_native_abi_version() == 1
    assert lib.elpis_dsv41_native_capabilities() & 0x1FF == 0x1FF

    rng = np.random.default_rng(41117)

    # Discrete donor cache formats must match exactly.
    for mode_name, mode_id, width in (
        ("local", 0, 64),
        ("compressed", 1, 64),
        ("index", 2, 64),
    ):
        for _ in range(48):
            x = rng.normal(0, 3, (7, width)).astype("<f4")
            x[0] = 0
            out = np.empty_like(x)
            assert (
                lib.elpis_dsv41_quant_dequant_f32(
                    ptr(x), ptr(out), x.shape[0], width, mode_id
                )
                == 0
            )
            expected = quant_dequant(x, mode_name)
            np.testing.assert_array_equal(out, expected)

    worst_rope = 0.0
    for vectors, dim, pairs in ((1, 8, 2), (4, 16, 4), (7, 32, 8)):
        for _ in range(32):
            x = rng.normal(size=(vectors, dim)).astype("<f4")
            angles = rng.normal(size=pairs).astype("<f4")
            freq = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype("<f4")
            for inverse in (0, 1):
                out = np.empty_like(x)
                assert (
                    lib.elpis_dsv41_rope_f32(
                        ptr(x),
                        ptr(freq),
                        ptr(out),
                        vectors,
                        dim,
                        pairs,
                        inverse,
                    )
                    == 0
                )
                expected = rotate(x, freq, inverse=bool(inverse))
                worst_rope = max(
                    worst_rope, float(np.max(np.abs(out - expected)))
                )
                np.testing.assert_allclose(
                    out, expected, rtol=2e-6, atol=2e-6
                )

    for width, top_blocks, block_size in (
        (1, 1, 4),
        (17, 2, 4),
        (64, 3, 8),
        (65, 4, 7),
    ):
        for _ in range(32):
            logits = rng.normal(size=width).astype("<f4")
            if width > 5:
                logits[1] = logits[4]
            mask = np.empty(width, dtype=np.uint8)
            assert (
                lib.elpis_dsv41_candidate_mask_f32(
                    ptr(logits),
                    mask.ctypes.data_as(U8P),
                    width,
                    top_blocks,
                    block_size,
                )
                == 0
            )
            np.testing.assert_array_equal(
                mask.astype(bool),
                candidate_mask(logits, top_blocks, block_size),
            )

    for width, topk in ((1, 1), (9, 3), (33, 8), (64, 17)):
        for _ in range(32):
            scores = rng.normal(size=width).astype("<f4")
            if width > 4:
                scores[1] = scores[3]
            pos = np.empty(min(width, topk), dtype=np.uint32)
            count = C.c_size_t()
            assert (
                lib.elpis_dsv41_select_positions_f32(
                    ptr(scores),
                    pos.ctypes.data_as(U32P),
                    width,
                    topk,
                    C.byref(count),
                )
                == 0
            )
            assert count.value == len(pos)
            np.testing.assert_array_equal(
                pos.astype(np.int64),
                select_positions(scores, topk),
            )

    worst_sparse = 0.0
    for heads, n, dim in (
        (1, 1, 4),
        (2, 7, 8),
        (4, 19, 16),
        (5, 31, 32),
    ):
        for _ in range(24):
            q = rng.normal(size=(heads, dim)).astype("<f4")
            kv = rng.normal(size=(n, dim)).astype("<f4")
            sink = rng.normal(size=heads).astype("<f4")
            out = np.empty_like(q)
            scratch = np.empty(n, dtype="<f4")
            assert (
                lib.elpis_dsv41_sparse_attention_f32(
                    ptr(q),
                    ptr(kv),
                    ptr(sink),
                    ptr(out),
                    ptr(scratch),
                    heads,
                    n,
                    dim,
                )
                == 0
            )
            expected = sparse_attention(q, kv, sink)
            worst_sparse = max(
                worst_sparse, float(np.max(np.abs(out - expected)))
            )
            np.testing.assert_allclose(
                out, expected, rtol=5e-6, atol=3e-6
            )

    print("PASS_NATIVE_ATTENTION_PRIMITIVES_DIFFERENTIAL")
    print(f"rope_max_abs={worst_rope:.9g}")
    print(f"sparse_attention_max_abs={worst_sparse:.9g}")


if __name__ == "__main__":
    main()
