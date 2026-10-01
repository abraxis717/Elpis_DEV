"""R8 native MoE dispatch/one-expert materialization differential."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from elpis.inference.drivers.dsv41.moe import route, expert


P = C.POINTER(C.c_float)
U32P = C.POINTER(C.c_uint32)
RTOL = 1e-4
ATOL = 1e-5


def ptr(a):
    return a.ctypes.data_as(P)


def u32ptr(a):
    return a.ctypes.data_as(U32P)


def bind(path):
    lib = C.CDLL(str(Path(path).resolve()))

    lib.elpis_dsv41_native_capabilities.argtypes = []
    lib.elpis_dsv41_native_capabilities.restype = C.c_uint64

    lib.elpis_dsv41_route_f32.argtypes = [
        P, P, P, U32P, P, P,
        C.c_size_t, C.c_size_t, C.c_size_t,
        C.c_uint32, C.c_float, C.c_int, C.c_float,
    ]
    lib.elpis_dsv41_route_f32.restype = C.c_int

    lib.elpis_dsv41_route_order_u32.argtypes = [
        U32P, U32P, C.c_size_t,
    ]
    lib.elpis_dsv41_route_order_u32.restype = C.c_int

    lib.elpis_dsv41_expert_accumulate_f32.argtypes = [
        P, P, P, P,
        C.c_float, C.c_int, C.c_float,
        P, P, P, P,
        C.c_size_t, C.c_size_t,
    ]
    lib.elpis_dsv41_expert_accumulate_f32.restype = C.c_int
    return lib


SCORE_IDS = {
    "softmax": 0,
    "sigmoid": 1,
    "sqrtsoftplus": 2,
}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: native_moe_dispatch_diff.py "
            "/path/to/libelpis_dsv41_native.so"
        )

    lib = bind(sys.argv[1])
    caps = lib.elpis_dsv41_native_capabilities()
    assert caps & (1 << 18)
    assert caps & (1 << 19)

    rng = np.random.default_rng(41808)
    worst = 0.0

    cases = (
        (8, 4, 2, 12, "softmax", True, 0.0),
        (16, 6, 3, 24, "sigmoid", True, 0.4),
        (24, 8, 4, 32, "sqrtsoftplus", False, 0.0),
        (32, 12, 4, 48, "sqrtsoftplus", True, 1.25),
    )

    for dim, expert_count, active, expert_dim, score, normalize, limit in cases:
        c = SimpleNamespace(
            gate_temp=1.0,
            score_func=score,
            active_experts=active,
            norm_topk_prob=normalize,
            route_scale=0.73,
            swiglu_limit=limit,
        )

        for _ in range(48):
            x = rng.normal(size=dim).astype("<f4")
            # Match PRODUCTION_SHAPED_FIXTURE parameter amplitude: ordinary
            # learned tensors (including router/bias/experts) use sigma=0.12.
            router = rng.normal(0, .12, size=(expert_count, dim)).astype("<f4")
            bias = rng.normal(0, .12, size=expert_count).astype("<f4")

            # Qualification keeps all synthetic expert tensors resident. The
            # native API itself consumes them one at a time, exactly matching
            # the production TensorStore.expert() staging lifetime.
            experts = []
            for _e in range(expert_count + 1):
                experts.append((
                    rng.normal(0, .12, size=(expert_dim, dim)).astype("<f4"),
                    rng.normal(0, .12, size=(expert_dim, dim)).astype("<f4"),
                    rng.normal(0, .12, size=(dim, expert_dim)).astype("<f4"),
                ))

            expected_ids, expected_weights = route(x, router, bias, c)
            expected = np.zeros(dim, dtype="<f4")
            for index in np.argsort(expected_ids):
                expected += expert(
                    x,
                    experts[int(expected_ids[index])],
                    expected_weights[index],
                    c.swiglu_limit,
                )
            expected += expert(
                x,
                experts[-1],
                None,
                c.swiglu_limit,
            )

            chosen = np.empty(active, dtype=np.uint32)
            values = np.empty(active, dtype="<f4")
            score_scratch = np.empty(expert_count, dtype="<f4")

            rc = lib.elpis_dsv41_route_f32(
                ptr(x), ptr(router), ptr(bias),
                u32ptr(chosen), ptr(values), ptr(score_scratch),
                dim, expert_count, active,
                SCORE_IDS[score], c.gate_temp,
                int(normalize), c.route_scale,
            )
            assert rc == 0

            np.testing.assert_array_equal(
                chosen, expected_ids.astype(np.uint32)
            )
            np.testing.assert_allclose(
                values, expected_weights, rtol=2e-6, atol=2e-6
            )

            order = np.empty(active, dtype=np.uint32)
            assert lib.elpis_dsv41_route_order_u32(
                u32ptr(chosen), u32ptr(order), active
            ) == 0
            np.testing.assert_array_equal(
                order,
                np.argsort(chosen, kind="stable").astype(np.uint32),
            )

            gate = np.empty(expert_dim, dtype="<f4")
            up = np.empty(expert_dim, dtype="<f4")
            one = np.empty(dim, dtype="<f4")
            accum = np.zeros(dim, dtype="<f4")

            for route_index in order:
                ri = int(route_index)
                eid = int(chosen[ri])
                w1, w3, w2 = experts[eid]
                assert lib.elpis_dsv41_expert_accumulate_f32(
                    ptr(x), ptr(w1), ptr(w3), ptr(w2),
                    float(values[ri]), 1, float(limit),
                    ptr(gate), ptr(up), ptr(one), ptr(accum),
                    dim, expert_dim,
                ) == 0

            w1, w3, w2 = experts[-1]
            assert lib.elpis_dsv41_expert_accumulate_f32(
                ptr(x), ptr(w1), ptr(w3), ptr(w2),
                0.0, 0, float(limit),
                ptr(gate), ptr(up), ptr(one), ptr(accum),
                dim, expert_dim,
            ) == 0

            error = float(np.max(np.abs(accum - expected), initial=0.0))
            worst = max(worst, error)
            np.testing.assert_allclose(
                accum, expected, rtol=RTOL, atol=ATOL
            )

    print("PASS_NATIVE_MOE_DISPATCH_DIFFERENTIAL")
    print(f"moe_dispatch_output_max_abs={worst:.9g}")


if __name__ == "__main__":
    main()
