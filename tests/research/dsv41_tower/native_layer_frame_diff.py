"""R9 native split Layer.apply frame differential."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from research.dsv41_tower.engram import gated_write
from research.dsv41_tower.moe import route
from research.dsv41_tower.numerics import (
    hc_mixes,
    hc_post,
    hc_pre,
    rms,
)


P = C.POINTER(C.c_float)
U32P = C.POINTER(C.c_uint32)
RTOL = 1e-4
ATOL = 1e-5


def ptr(a):
    return a.ctypes.data_as(P)


def u32ptr(a):
    return a.ctypes.data_as(U32P)


def fptr_or_none(a):
    return None if a is None else ptr(a)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64

    lib.elpis_dsv41_layer_frame_scratch_floats.argtypes = [
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t, C.c_int,
    ]
    lib.elpis_dsv41_layer_frame_scratch_floats.restype = C.c_size_t

    lib.elpis_dsv41_layer_begin_f32.argtypes = [
        P, P,
        P, P, P, P,
        C.c_int, C.c_size_t,
        P, P, P, P,
        C.c_float, C.c_float, C.c_size_t,
        P, C.c_size_t,
        P, P, P, P, P,
        C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_layer_begin_f32.restype = C.c_int

    lib.elpis_dsv41_layer_after_attention_route_f32.argtypes = [
        P, P, P, P, P,
        P, P, P, P,
        P, P,
        C.c_float, C.c_float, C.c_size_t,
        C.c_uint32, C.c_float, C.c_int, C.c_float,
        P, C.c_size_t,
        P, P, P, P, P,
        U32P, P, U32P,
        C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_layer_after_attention_route_f32.restype = C.c_int

    lib.elpis_dsv41_layer_finish_f32.argtypes = [
        P, P, P, P, P, C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_layer_finish_f32.restype = C.c_int
    return lib


SCORE_IDS = {
    "softmax": 0,
    "sigmoid": 1,
    "sqrtsoftplus": 2,
}


def make_config(copies, dim, active, score):
    return SimpleNamespace(
        hc_mult=copies,
        dimension=dim,
        hc_sinkhorn_iters=7,
        norm_eps=1e-6,
        hc_eps=1e-6,
        active_experts=active,
        score_func=score,
        gate_temp=0.85,
        norm_topk_prob=True,
        route_scale=1.25,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_layer_frame_diff.py "
            "/path/to/libelpis_dsv41_native.so"
        )

    lib = bind(sys.argv[1])
    caps = lib.elpis_dsv41_native_capabilities()
    assert caps & (1 << 20)
    assert caps & (1 << 21)
    assert caps & (1 << 22)

    rng = np.random.default_rng(41909)
    worst = {
        "begin_stream": 0.0,
        "attention_x": 0.0,
        "after_stream": 0.0,
        "moe_x": 0.0,
        "finish_stream": 0.0,
    }

    cases = (
        (2, 8, 5, 2, "softmax", False, 0),
        (4, 8, 7, 3, "sigmoid", True, 96),
        (4, 16, 8, 4, "sqrtsoftplus", False, 0),
        (4, 24, 9, 4, "sqrtsoftplus", True, 128),
    )

    for copies, dim, expert_count, active, score_mode, engram, row_values in cases:
        c = make_config(copies, dim, active, score_mode)
        mix_dim = (2 + copies) * copies

        for _ in range(36):
            stream = rng.normal(size=(copies, dim)).astype("<f4")
            pre_mix = rng.normal(size=copies).astype("<f4")

            hc_attn_fn = rng.normal(
                0, .12, size=(mix_dim, copies * dim)
            ).astype("<f4")
            hc_attn_scale = np.array([.08, .09, .07], dtype="<f4")
            hc_attn_base = rng.normal(0, .12, size=mix_dim).astype("<f4")
            attn_norm = (
                1 + rng.normal(0, .04, size=dim)
            ).astype("<f4")

            hc_ffn_fn = rng.normal(
                0, .12, size=(mix_dim, copies * dim)
            ).astype("<f4")
            hc_ffn_scale = np.array([.08, .09, .07], dtype="<f4")
            hc_ffn_base = rng.normal(0, .12, size=mix_dim).astype("<f4")
            ffn_norm = (
                1 + rng.normal(0, .04, size=dim)
            ).astype("<f4")

            router = rng.normal(
                0, .12, size=(expert_count, dim)
            ).astype("<f4")
            bias = rng.normal(0, .12, size=expert_count).astype("<f4")

            if engram:
                rows = rng.normal(0, .12, size=row_values).astype("<f4")
                e_wkv = rng.normal(
                    0, .12, size=((copies + 1) * dim, row_values)
                ).astype("<f4")
                e_q = (
                    1 + rng.normal(0, .04, size=(copies, dim))
                ).astype("<f4")
                e_k = (
                    1 + rng.normal(0, .04, size=(copies, dim))
                ).astype("<f4")
                expected_stream_work = gated_write(
                    stream,
                    rows,
                    {"wkv": e_wkv, "q_weight": e_q, "k_weight": e_k},
                    c,
                )
            else:
                rows = e_wkv = e_q = e_k = None
                expected_stream_work = stream.copy()

            e_ap, e_apost, e_ac = hc_mixes(
                expected_stream_work,
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                c,
            )
            e_attention_x = rms(
                hc_pre(expected_stream_work, pre_mix),
                attn_norm,
                c.norm_eps,
            )

            scratch_n = lib.elpis_dsv41_layer_frame_scratch_floats(
                copies, dim, row_values, expert_count, int(engram)
            )
            assert scratch_n > 0
            scratch = np.empty(scratch_n, dtype="<f4")

            stream_work = np.empty((copies, dim), dtype="<f4")
            ap = np.empty(copies, dtype="<f4")
            apost = np.empty(copies, dtype="<f4")
            ac = np.empty((copies, copies), dtype="<f4")
            attention_x = np.empty(dim, dtype="<f4")

            assert lib.elpis_dsv41_layer_begin_f32(
                ptr(stream),
                ptr(pre_mix),
                fptr_or_none(rows),
                fptr_or_none(e_wkv),
                fptr_or_none(e_q),
                fptr_or_none(e_k),
                int(engram),
                row_values,
                ptr(hc_attn_fn),
                ptr(hc_attn_scale),
                ptr(hc_attn_base),
                ptr(attn_norm),
                c.norm_eps,
                c.hc_eps,
                c.hc_sinkhorn_iters,
                ptr(scratch),
                scratch_n,
                ptr(stream_work),
                ptr(ap),
                ptr(apost),
                ptr(ac),
                ptr(attention_x),
                copies,
                dim,
            ) == 0

            worst["begin_stream"] = max(
                worst["begin_stream"],
                float(np.max(
                    np.abs(stream_work - expected_stream_work),
                    initial=0.0,
                )),
            )
            worst["attention_x"] = max(
                worst["attention_x"],
                float(np.max(
                    np.abs(attention_x - e_attention_x),
                    initial=0.0,
                )),
            )

            np.testing.assert_allclose(
                stream_work, expected_stream_work,
                rtol=RTOL, atol=ATOL,
            )
            np.testing.assert_allclose(ap, e_ap, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(apost, e_apost, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(ac, e_ac, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(
                attention_x, e_attention_x,
                rtol=RTOL, atol=ATOL,
            )

            # R9 frames an already-qualified attention call. Exercise a
            # production-scale attention result without reimplementing R6/R7.
            attention_out = rng.normal(0, .12, size=dim).astype("<f4")

            e_stream_after = hc_post(
                attention_out, expected_stream_work, e_apost, e_ac
            )
            e_fp, e_fpost, e_fc = hc_mixes(
                e_stream_after,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
                c,
            )
            e_moe_x = rms(
                hc_pre(e_stream_after, e_ap),
                ffn_norm,
                c.norm_eps,
            )
            e_chosen, e_values = route(e_moe_x, router, bias, c)
            e_order = np.argsort(
                e_chosen, kind="stable"
            ).astype(np.uint32)

            stream_after = np.empty((copies, dim), dtype="<f4")
            fp = np.empty(copies, dtype="<f4")
            fpost = np.empty(copies, dtype="<f4")
            fc = np.empty((copies, copies), dtype="<f4")
            moe_x = np.empty(dim, dtype="<f4")
            chosen = np.empty(active, dtype=np.uint32)
            values = np.empty(active, dtype="<f4")
            order = np.empty(active, dtype=np.uint32)

            assert lib.elpis_dsv41_layer_after_attention_route_f32(
                ptr(stream_work),
                ptr(attention_out),
                ptr(ap),
                ptr(apost),
                ptr(ac),
                ptr(hc_ffn_fn),
                ptr(hc_ffn_scale),
                ptr(hc_ffn_base),
                ptr(ffn_norm),
                ptr(router),
                ptr(bias),
                c.norm_eps,
                c.hc_eps,
                c.hc_sinkhorn_iters,
                SCORE_IDS[score_mode],
                c.gate_temp,
                int(c.norm_topk_prob),
                c.route_scale,
                ptr(scratch),
                scratch_n,
                ptr(stream_after),
                ptr(fp),
                ptr(fpost),
                ptr(fc),
                ptr(moe_x),
                u32ptr(chosen),
                ptr(values),
                u32ptr(order),
                copies,
                dim,
                expert_count,
                active,
            ) == 0

            worst["after_stream"] = max(
                worst["after_stream"],
                float(np.max(
                    np.abs(stream_after - e_stream_after),
                    initial=0.0,
                )),
            )
            worst["moe_x"] = max(
                worst["moe_x"],
                float(np.max(
                    np.abs(moe_x - e_moe_x),
                    initial=0.0,
                )),
            )

            np.testing.assert_allclose(
                stream_after, e_stream_after,
                rtol=RTOL, atol=ATOL,
            )
            np.testing.assert_allclose(fp, e_fp, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(fpost, e_fpost, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(fc, e_fc, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(moe_x, e_moe_x, rtol=RTOL, atol=ATOL)
            np.testing.assert_array_equal(
                chosen, e_chosen.astype(np.uint32)
            )
            np.testing.assert_allclose(
                values, e_values, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_array_equal(order, e_order)

            # R9 finish frames the already-qualified one-expert-at-a-time R8
            # accumulation result.
            moe_out = rng.normal(0, .12, size=dim).astype("<f4")
            e_stream_out = hc_post(
                moe_out, e_stream_after, e_fpost, e_fc
            )
            stream_out = np.empty((copies, dim), dtype="<f4")

            assert lib.elpis_dsv41_layer_finish_f32(
                ptr(moe_out),
                ptr(stream_after),
                ptr(fpost),
                ptr(fc),
                ptr(stream_out),
                copies,
                dim,
            ) == 0

            worst["finish_stream"] = max(
                worst["finish_stream"],
                float(np.max(
                    np.abs(stream_out - e_stream_out),
                    initial=0.0,
                )),
            )
            np.testing.assert_allclose(
                stream_out, e_stream_out,
                rtol=RTOL, atol=ATOL,
            )

    print("PASS_NATIVE_LAYER_FRAME_DIFFERENTIAL")
    for name, value in worst.items():
        print(f"layer_frame_{name}_max_abs={value:.9g}")


if __name__ == "__main__":
    main()
