
#include "elpis/dsv41_native.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

elpis_dsv41_native_status
elpis_dsv41_route_order_u32(const uint32_t *chosen,
                            uint32_t *order,
                            size_t active) {
    size_t i, j;

    if (chosen == NULL || order == NULL || active == 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    /* Route is contractually unique. Reject duplicates rather than silently
     * changing donor accumulation semantics. */
    for (i = 0u; i < active; ++i) {
        for (j = i + 1u; j < active; ++j) {
            if (chosen[i] == chosen[j]) {
                return ELPIS_DSV41_NATIVE_INVALID;
            }
        }
    }

    /* Stable insertion sort over indices. active is top-k, not expert_count. */
    for (i = 0u; i < active; ++i) {
        size_t at = i;
        uint32_t idx = (uint32_t)i;

        while (at > 0u) {
            uint32_t prev = order[at - 1u];
            if (chosen[prev] <= chosen[idx]) {
                break;
            }
            order[at] = prev;
            --at;
        }
        order[at] = idx;
    }

    return ELPIS_DSV41_NATIVE_OK;
}

elpis_dsv41_native_status
elpis_dsv41_expert_accumulate_f32(const float *x,
                                  const float *w1,
                                  const float *w3,
                                  const float *w2,
                                  float route_weight,
                                  int weighted,
                                  float swiglu_limit,
                                  float *gate_scratch,
                                  float *up_scratch,
                                  float *expert_out_scratch,
                                  float *accumulator,
                                  size_t dim,
                                  size_t expert_dim) {
    elpis_dsv41_native_status rc;
    size_t d;

    if (accumulator == NULL) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    rc = elpis_dsv41_expert_f32(
        x, w1, w3, w2,
        route_weight, weighted, swiglu_limit,
        gate_scratch, up_scratch, expert_out_scratch,
        dim, expert_dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) {
        return rc;
    }

    for (d = 0u; d < dim; ++d) {
        float value = accumulator[d] + expert_out_scratch[d];
        if (!isfinite(value)) {
            return ELPIS_DSV41_NATIVE_NONFINITE;
        }
        accumulator[d] = value;
    }

    return ELPIS_DSV41_NATIVE_OK;
}
