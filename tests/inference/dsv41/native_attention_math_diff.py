"""R5 native attention-specific reduction differential."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import sys

import numpy as np


P = C.POINTER(C.c_float)
RTOL = 1e-4
ATOL = 1e-5


def ptr(a):
    return a.ctypes.data_as(P)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64

    lib.elpis_dsv41_index_scores_f32.argtypes = [
        P, P, P, P, C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_index_scores_f32.restype = C.c_int

    lib.elpis_dsv41_grouped_output_f32.argtypes = [
        P, P, P, P, P,
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_grouped_output_f32.restype = C.c_int
    return lib


def numpy_index_scores(iq, keys, iw):
    pair = np.einsum("hd,nd->hn", iq, keys, dtype=np.float32, optimize=False)
    return np.sum(np.maximum(pair, np.float32(0)) * iw[:, None],
                  axis=0, dtype=np.float32)


def numpy_grouped_output(attn_out, wo_a, wo_b, o_groups, o_rank):
    grouped = attn_out.reshape(o_groups, -1)
    projection = wo_a.reshape(o_groups, o_rank, -1)
    tmp = np.einsum("gi,gri->gr", grouped, projection,
                    dtype=np.float32, optimize=False)
    return np.einsum("i,oi->o", tmp.reshape(-1), wo_b,
                     dtype=np.float32, optimize=False)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_attention_math_diff.py /path/to/libelpis_dsv41_native.so"
        )

    lib = bind(sys.argv[1])
    caps = lib.elpis_dsv41_native_capabilities()
    assert caps & (1 << 14)
    assert caps & (1 << 15)

    rng = np.random.default_rng(41505)
    worst_index = 0.0
    worst_output = 0.0

    for heads, dim, count in (
        (1, 8, 1),
        (2, 8, 7),
        (4, 16, 17),
        (8, 32, 31),
    ):
        for _ in range(64):
            iq = rng.normal(size=(heads, dim)).astype("<f4")
            keys = rng.normal(size=(count, dim)).astype("<f4")
            iw = rng.normal(size=heads).astype("<f4")
            out = np.empty(count, dtype="<f4")

            rc = lib.elpis_dsv41_index_scores_f32(
                ptr(iq), ptr(keys), ptr(iw), ptr(out),
                heads, dim, count,
            )
            assert rc == 0

            expected = numpy_index_scores(iq, keys, iw)
            worst_index = max(
                worst_index, float(np.max(np.abs(out - expected), initial=0.0))
            )
            np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)

    for heads, head_dim, groups, rank, dimension in (
        (2, 8, 1, 3, 8),
        (4, 8, 2, 4, 16),
        (8, 16, 4, 3, 32),
        (8, 32, 4, 8, 64),
    ):
        width = heads * head_dim // groups
        for _ in range(64):
            attn = rng.normal(size=(heads, head_dim)).astype("<f4")
            wo_a = rng.normal(size=(groups * rank, width)).astype("<f4")
            wo_b = rng.normal(size=(dimension, groups * rank)).astype("<f4")
            scratch = np.empty(groups * rank, dtype="<f4")
            out = np.empty(dimension, dtype="<f4")

            rc = lib.elpis_dsv41_grouped_output_f32(
                ptr(attn), ptr(wo_a), ptr(wo_b), ptr(scratch), ptr(out),
                heads, head_dim, groups, rank, dimension,
            )
            assert rc == 0

            expected = numpy_grouped_output(attn, wo_a, wo_b, groups, rank)
            worst_output = max(
                worst_output, float(np.max(np.abs(out - expected), initial=0.0))
            )
            np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)

    print("PASS_NATIVE_ATTENTION_MATH_DIFFERENTIAL")
    print(f"index_scores_max_abs={worst_index:.9g}")
    print(f"grouped_output_max_abs={worst_output:.9g}")


if __name__ == "__main__":
    main()
