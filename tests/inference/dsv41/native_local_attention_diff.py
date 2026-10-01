"""R6 native pure-local Attention.apply orchestration differential."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from elpis.inference.drivers.dsv41.attention import Attention, AttentionState


P = C.POINTER(C.c_float)
StateP = C.c_void_p
RTOL = 1e-4
ATOL = 1e-5


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

    lib.elpis_dsv41_attention_local_copy_f32.argtypes = [
        StateP, C.c_size_t, P, C.c_size_t, C.POINTER(C.c_size_t),
    ]
    lib.elpis_dsv41_attention_local_copy_f32.restype = C.c_int

    lib.elpis_dsv41_local_attention_scratch_floats.argtypes = [
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_local_attention_scratch_floats.restype = C.c_size_t

    lib.elpis_dsv41_local_attention_apply_f32.argtypes = [
        StateP, C.c_size_t,
        P, P, P, P, P, P, P, P, P, P,
        C.c_float, P, C.c_size_t, P,
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_local_attention_apply_f32.restype = C.c_int
    return lib


def make_case(seed, dimension, q_rank, heads, head_dim,
              rope_pairs, groups, rank, local_window, max_tokens):
    rng = np.random.default_rng(seed)
    c = SimpleNamespace(
        kv_sources=(),
        index_sources=(),
        compress_ratios=(0,),
        max_tokens=max_tokens,
        local_window=local_window,
        head_dim=head_dim,
        index_dim=head_dim,
        heads=heads,
        o_groups=groups,
        o_rank=rank,
        norm_eps=1e-6,
    )

    width = heads * head_dim // groups
    w = {
        "wq_a": rng.normal(size=(q_rank, dimension)).astype("<f4"),
        "q_norm": rng.normal(size=q_rank).astype("<f4"),
        "wq_b": rng.normal(size=(heads * head_dim, q_rank)).astype("<f4"),
        "wkv": rng.normal(size=(head_dim, dimension)).astype("<f4"),
        "kv_norm": rng.normal(size=head_dim).astype("<f4"),
        "attn_sink": rng.normal(size=heads).astype("<f4"),
        "wo_a": rng.normal(size=(groups * rank, width)).astype("<f4"),
        "wo_b": rng.normal(size=(dimension, groups * rank)).astype("<f4"),
    }

    angles = rng.normal(size=(max_tokens, rope_pairs)).astype("<f4")
    freqs = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype("<f4")
    xs = rng.normal(size=(max_tokens, dimension)).astype("<f4")
    return c, w, freqs, xs


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_local_attention_diff.py /path/to/libelpis_dsv41_native.so"
        )

    lib = bind(sys.argv[1])
    assert lib.elpis_dsv41_native_capabilities() & (1 << 16)

    worst_out = 0.0
    worst_cache = 0.0

    cases = (
        (41601, 16, 8, 1, 32, 4, 1, 3, 3, 11),
        (41602, 24, 12, 2, 32, 8, 2, 4, 5, 14),
        (41603, 32, 16, 4, 32, 8, 2, 5, 7, 18),
    )

    for args in cases:
        _, dimension, q_rank, heads, head_dim, rope_pairs, groups, rank, local_window, max_tokens = args
        c, w, freqs, xs = make_case(*args)
        py_state = AttentionState.create(c, 0)
        attention = Attention(c, 0, w, freqs)

        native = StateP()
        assert lib.elpis_dsv41_attention_state_create(
            local_window, head_dim, head_dim, max_tokens, 0, 0,
            C.byref(native),
        ) == 0
        assert native.value

        scratch_n = lib.elpis_dsv41_local_attention_scratch_floats(
            dimension, q_rank, heads, head_dim, groups, rank, local_window,
        )
        assert scratch_n > 0
        scratch = np.empty(scratch_n, dtype="<f4")

        try:
            for position in range(max_tokens):
                x = xs[position]
                py_out, selected = attention.apply(
                    x, py_state, SimpleNamespace(owner=None, selected=None, candidates=None),
                    position,
                )
                assert selected == ()

                native_out = np.empty(dimension, dtype="<f4")
                rc = lib.elpis_dsv41_local_attention_apply_f32(
                    native, position,
                    ptr(x),
                    ptr(w["wq_a"]),
                    ptr(w["q_norm"]),
                    ptr(w["wq_b"]),
                    ptr(w["wkv"]),
                    ptr(w["kv_norm"]),
                    ptr(w["attn_sink"]),
                    ptr(w["wo_a"]),
                    ptr(w["wo_b"]),
                    ptr(freqs[position]),
                    c.norm_eps,
                    ptr(scratch),
                    scratch_n,
                    ptr(native_out),
                    dimension,
                    q_rank,
                    heads,
                    head_dim,
                    rope_pairs,
                    groups,
                    rank,
                    local_window,
                )
                assert rc == 0

                err = float(np.max(np.abs(native_out - py_out), initial=0.0))
                worst_out = max(worst_out, err)
                np.testing.assert_allclose(
                    native_out, py_out, rtol=RTOL, atol=ATOL,
                    err_msg=f"position={position} config={args}",
                )

                rows_expected = min(position + 1, local_window)
                start = position + 1 - rows_expected
                slots = np.arange(start, position + 1) % local_window
                expected_cache = py_state.window[slots]

                cache = np.empty((local_window, head_dim), dtype="<f4")
                rows = C.c_size_t()
                assert lib.elpis_dsv41_attention_local_copy_f32(
                    native, position, ptr(cache), local_window, C.byref(rows)
                ) == 0
                assert rows.value == rows_expected

                cache_err = float(np.max(
                    np.abs(cache[:rows.value] - expected_cache), initial=0.0
                ))
                worst_cache = max(worst_cache, cache_err)
                np.testing.assert_array_equal(
                    cache[:rows.value], expected_cache
                )
        finally:
            assert lib.elpis_dsv41_attention_state_destroy(C.byref(native)) == 0
            assert not native.value

    print("PASS_NATIVE_LOCAL_ATTENTION_ORCHESTRATION_DIFFERENTIAL")
    print(f"local_attention_output_max_abs={worst_out:.9g}")
    print(f"local_attention_cache_max_abs={worst_cache:.9g}")


if __name__ == "__main__":
    main()
