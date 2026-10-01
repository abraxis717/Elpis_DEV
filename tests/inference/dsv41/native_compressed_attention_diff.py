"""R7 native compressed/global Attention.apply orchestration differential."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from elpis.inference.drivers.dsv41.attention import (
    Attention,
    AttentionState,
    SharedAttention,
)


P = C.POINTER(C.c_float)
U32P = C.POINTER(C.c_uint32)
U8P = C.POINTER(C.c_uint8)
StateP = C.c_void_p
RTOL = 1e-4
ATOL = 1e-5


def ptr(a):
    return a.ctypes.data_as(P)


def u32ptr(a):
    return a.ctypes.data_as(U32P)


def u8ptr(a):
    return a.ctypes.data_as(U8P)


def fptr_or_none(a):
    return None if a is None else ptr(a)


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
    lib.elpis_dsv41_attention_state_count.argtypes = [StateP]
    lib.elpis_dsv41_attention_state_count.restype = C.c_size_t

    lib.elpis_dsv41_attention_local_copy_f32.argtypes = [
        StateP, C.c_size_t, P, C.c_size_t, C.POINTER(C.c_size_t),
    ]
    lib.elpis_dsv41_attention_local_copy_f32.restype = C.c_int
    lib.elpis_dsv41_attention_index_copy_f32.argtypes = [
        StateP, P, C.c_size_t, C.POINTER(C.c_size_t),
    ]
    lib.elpis_dsv41_attention_index_copy_f32.restype = C.c_int
    lib.elpis_dsv41_attention_gather_compressed_f32.argtypes = [
        StateP, U32P, C.c_size_t, P,
    ]
    lib.elpis_dsv41_attention_gather_compressed_f32.restype = C.c_int

    lib.elpis_dsv41_compressed_attention_scratch_floats.argtypes = [
        C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_compressed_attention_scratch_floats.restype = C.c_size_t

    lib.elpis_dsv41_compressed_attention_apply_f32.argtypes = [
        StateP, StateP, C.c_size_t,
        P, P, P, P, P, P, P, P, P,
        P, P, P, P, P, P, P,
        P, P, C.c_float,
        U32P, C.c_size_t, C.POINTER(C.c_size_t),
        U8P, C.c_size_t, C.POINTER(C.c_size_t),
        P, C.c_size_t, P,
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_int, C.c_int, C.c_int,
        C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_compressed_attention_apply_f32.restype = C.c_int
    return lib


def config_for(*, ratios, kv_sources, index_sources, candidate_source,
               candidate_blocks=0, candidate_block_size=0,
               dimension=24, q_rank=12, heads=2, head_dim=32,
               index_heads=2, index_dim=32, groups=2, rank=4,
               local_window=5, max_tokens=12, index_topk=3):
    return SimpleNamespace(
        compress_ratios=tuple(ratios),
        kv_sources=tuple(kv_sources),
        index_sources=tuple(index_sources),
        candidate_source=candidate_source,
        candidate_blocks=candidate_blocks,
        candidate_block_size=candidate_block_size,
        dimension=dimension,
        q_rank=q_rank,
        heads=heads,
        head_dim=head_dim,
        index_heads=index_heads,
        index_dim=index_dim,
        o_groups=groups,
        o_rank=rank,
        local_window=local_window,
        max_tokens=max_tokens,
        index_topk=index_topk,
        norm_eps=1e-6,
    )


def layer_weights(rng, c, layer):
    width = c.heads * c.head_dim // c.o_groups
    w = {
        "wq_a": rng.normal(size=(c.q_rank, c.dimension)).astype("<f4"),
        "q_norm": rng.normal(size=c.q_rank).astype("<f4"),
        "wq_b": rng.normal(size=(c.heads * c.head_dim, c.q_rank)).astype("<f4"),
        "wkv": rng.normal(size=(c.head_dim, c.dimension)).astype("<f4"),
        "kv_norm": rng.normal(size=c.head_dim).astype("<f4"),
        "attn_sink": rng.normal(size=c.heads).astype("<f4"),
        "wo_a": rng.normal(
            size=(c.o_groups * c.o_rank, width)
        ).astype("<f4"),
        "wo_b": rng.normal(
            size=(c.dimension, c.o_groups * c.o_rank)
        ).astype("<f4"),
    }
    if layer in c.kv_sources:
        w.update({
            "compressor.wkv": rng.normal(
                size=(c.head_dim, c.dimension)
            ).astype("<f4"),
            "compressor.norm": rng.normal(size=c.head_dim).astype("<f4"),
            "indexer.wk": rng.normal(
                size=(c.index_dim, c.head_dim)
            ).astype("<f4"),
            "indexer.k_norm": rng.normal(size=c.index_dim).astype("<f4"),
        })
        if c.compress_ratios[layer] > 1:
            w["compressor.wgate"] = rng.normal(
                size=(c.head_dim, c.dimension)
            ).astype("<f4")
    if layer in c.index_sources:
        w.update({
            "indexer.wq_b": rng.normal(
                size=(c.index_heads * c.index_dim, c.q_rank)
            ).astype("<f4"),
            "indexer.weights_proj": rng.normal(
                size=(c.index_heads, c.dimension)
            ).astype("<f4"),
        })
    return w


def run_scenario(lib, seed, c):
    rng = np.random.default_rng(seed)
    layers = len(c.compress_ratios)
    rope_pairs = 4
    angles = rng.normal(
        size=(c.max_tokens, rope_pairs)
    ).astype("<f4")
    freqs = np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    ).astype("<f4")

    weights = [layer_weights(rng, c, i) for i in range(layers)]
    attentions = [Attention(c, i, weights[i], freqs) for i in range(layers)]
    py_states = [AttentionState.create(c, i) for i in range(layers)]
    py_shared = SharedAttention()

    native_states = []
    for i in range(layers):
        st = StateP()
        ratio = c.compress_ratios[i]
        owner = int(i in c.kv_sources)
        assert lib.elpis_dsv41_attention_state_create(
            c.local_window, c.head_dim, c.index_dim, c.max_tokens,
            ratio, owner, C.byref(st),
        ) == 0
        native_states.append(st)

    max_groups = c.max_tokens // min(r for r in c.compress_ratios if r > 0)
    selected = np.zeros(c.index_topk, dtype=np.uint32)
    candidates = np.zeros(max_groups, dtype=np.uint8)
    selected_count = C.c_size_t(0)
    candidate_count = C.c_size_t(0)

    scratch_n = lib.elpis_dsv41_compressed_attention_scratch_floats(
        c.q_rank, c.heads, c.head_dim,
        c.index_heads, c.index_dim,
        c.o_groups, c.o_rank,
        c.local_window, max_groups, c.index_topk,
    )
    assert scratch_n > 0
    scratch = np.empty(scratch_n, dtype="<f4")

    xs = rng.normal(
        size=(c.max_tokens, layers, c.dimension)
    ).astype("<f4")

    worst_out = 0.0
    worst_local = 0.0
    worst_index = 0.0
    worst_compressed = 0.0
    owner_native = StateP()

    try:
        for position in range(c.max_tokens):
            for layer in range(layers):
                x = xs[position, layer]
                w = weights[layer]
                py_out, py_selected = attentions[layer].apply(
                    x, py_states[layer], py_shared, position
                )

                ratio = c.compress_ratios[layer]
                is_owner = layer in c.kv_sources
                is_index = layer in c.index_sources
                if is_owner:
                    owner_native = native_states[layer]

                assert owner_native.value

                if is_index and layer == c.candidate_source:
                    candidate_mode = 1
                elif is_index and 0 <= c.candidate_source < layer:
                    candidate_mode = 2
                else:
                    candidate_mode = 0

                group_freq = None
                if is_owner and (position + 1) % ratio == 0:
                    group_freq = ptr(freqs[position + 1 - ratio])

                layer_max_groups = c.max_tokens // ratio
                if layer_max_groups > max_groups:
                    raise AssertionError("scenario max_groups bound")

                out = np.empty(c.dimension, dtype="<f4")
                rc = lib.elpis_dsv41_compressed_attention_apply_f32(
                    native_states[layer], owner_native, position,
                    ptr(x),
                    ptr(w["wq_a"]),
                    ptr(w["q_norm"]),
                    ptr(w["wq_b"]),
                    ptr(w["wkv"]),
                    ptr(w["kv_norm"]),
                    ptr(w["attn_sink"]),
                    ptr(w["wo_a"]),
                    ptr(w["wo_b"]),
                    fptr_or_none(w.get("compressor.wkv")),
                    fptr_or_none(w.get("compressor.wgate")),
                    fptr_or_none(w.get("compressor.norm")),
                    fptr_or_none(w.get("indexer.wk")),
                    fptr_or_none(w.get("indexer.k_norm")),
                    fptr_or_none(w.get("indexer.wq_b")),
                    fptr_or_none(w.get("indexer.weights_proj")),
                    ptr(freqs[position]),
                    group_freq,
                    c.norm_eps,
                    u32ptr(selected), c.index_topk, C.byref(selected_count),
                    u8ptr(candidates), max_groups, C.byref(candidate_count),
                    ptr(scratch), scratch_n, ptr(out),
                    c.dimension, c.q_rank, c.heads, c.head_dim, rope_pairs,
                    c.index_heads, c.index_dim,
                    c.o_groups, c.o_rank, c.local_window,
                    ratio, max_groups, c.index_topk,
                    int(is_owner), int(is_index), candidate_mode,
                    c.candidate_blocks, c.candidate_block_size,
                )
                assert rc == 0, (position, layer, rc)

                worst_out = max(
                    worst_out,
                    float(np.max(np.abs(out - py_out), initial=0.0)),
                )
                np.testing.assert_allclose(
                    out, py_out, rtol=RTOL, atol=ATOL,
                    err_msg=f"position={position} layer={layer}",
                )

                got_selected = tuple(
                    int(v) for v in selected[:selected_count.value]
                )
                assert got_selected == py_selected, (
                    position, layer, got_selected, py_selected
                )

                rows_expected = min(position + 1, c.local_window)
                start = position + 1 - rows_expected
                slots = np.arange(start, position + 1) % c.local_window
                expected_local = py_states[layer].window[slots]
                local_copy = np.empty(
                    (c.local_window, c.head_dim), dtype="<f4"
                )
                local_rows = C.c_size_t()
                assert lib.elpis_dsv41_attention_local_copy_f32(
                    native_states[layer], position,
                    ptr(local_copy), c.local_window, C.byref(local_rows)
                ) == 0
                assert local_rows.value == rows_expected
                worst_local = max(
                    worst_local,
                    float(np.max(
                        np.abs(local_copy[:rows_expected] - expected_local),
                        initial=0.0,
                    )),
                )
                np.testing.assert_array_equal(
                    local_copy[:rows_expected], expected_local
                )

                if is_owner:
                    py_owner = py_states[layer]
                    count = py_owner.count
                    assert lib.elpis_dsv41_attention_state_count(
                        native_states[layer]
                    ) == count

                    if count:
                        index_copy = np.empty(
                            (max_groups, c.index_dim), dtype="<f4"
                        )
                        index_rows = C.c_size_t()
                        assert lib.elpis_dsv41_attention_index_copy_f32(
                            native_states[layer], ptr(index_copy),
                            max_groups, C.byref(index_rows)
                        ) == 0
                        assert index_rows.value == count
                        worst_index = max(
                            worst_index,
                            float(np.max(
                                np.abs(
                                    index_copy[:count] -
                                    py_owner.index_keys[:count]
                                ),
                                initial=0.0,
                            )),
                        )
                        np.testing.assert_allclose(
                            index_copy[:count],
                            py_owner.index_keys[:count],
                            rtol=RTOL, atol=ATOL,
                        )

                        positions = np.arange(count, dtype=np.uint32)
                        compressed = np.empty(
                            (count, c.head_dim), dtype="<f4"
                        )
                        assert lib.elpis_dsv41_attention_gather_compressed_f32(
                            native_states[layer], u32ptr(positions),
                            count, ptr(compressed)
                        ) == 0
                        worst_compressed = max(
                            worst_compressed,
                            float(np.max(
                                np.abs(
                                    compressed -
                                    py_owner.compressed[:count]
                                ),
                                initial=0.0,
                            )),
                        )
                        np.testing.assert_allclose(
                            compressed,
                            py_owner.compressed[:count],
                            rtol=RTOL, atol=ATOL,
                        )

                if (
                    is_index and layer == c.candidate_source and
                    (position + 1) // ratio > 0
                ):
                    assert py_shared.candidates is not None
                    assert candidate_count.value == len(py_shared.candidates)
                    np.testing.assert_array_equal(
                        candidates[:candidate_count.value],
                        py_shared.candidates.astype(np.uint8),
                    )
    finally:
        for st in native_states:
            assert lib.elpis_dsv41_attention_state_destroy(
                C.byref(st)
            ) == 0

    return worst_out, worst_local, worst_index, worst_compressed


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_compressed_attention_diff.py "
            "/path/to/libelpis_dsv41_native.so"
        )
    lib = bind(sys.argv[1])
    assert lib.elpis_dsv41_native_capabilities() & (1 << 17)

    scenarios = [
        (
            41701,
            config_for(
                ratios=(2, 2, 2, 2),
                kv_sources=(0,),
                index_sources=(0, 2),
                candidate_source=-1,
                max_tokens=12,
            ),
        ),
        (
            41702,
            config_for(
                ratios=(1, 1, 1),
                kv_sources=(0,),
                index_sources=(0, 1),
                candidate_source=0,
                candidate_blocks=2,
                candidate_block_size=2,
                max_tokens=10,
                dimension=16,
                q_rank=8,
                heads=2,
                groups=1,
                rank=3,
                local_window=4,
                index_topk=3,
            ),
        ),
    ]

    worst = np.zeros(4, dtype=np.float64)
    for seed, c in scenarios:
        values = run_scenario(lib, seed, c)
        worst = np.maximum(worst, np.array(values, dtype=np.float64))

    print("PASS_NATIVE_COMPRESSED_ATTENTION_ORCHESTRATION_DIFFERENTIAL")
    print(f"compressed_attention_output_max_abs={worst[0]:.9g}")
    print(f"compressed_attention_local_cache_max_abs={worst[1]:.9g}")
    print(f"compressed_attention_index_cache_max_abs={worst[2]:.9g}")
    print(f"compressed_attention_compressed_cache_max_abs={worst[3]:.9g}")


if __name__ == "__main__":
    main()
