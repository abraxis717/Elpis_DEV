
#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

int main(void) {
    uint32_t chosen[3] = {3u, 1u, 2u};
    uint32_t order[3] = {99u, 99u, 99u};

    float x[2] = {1.0f, -1.0f};
    float w1[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float w3[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float w2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float gate[2] = {0.0f, 0.0f};
    float up[2] = {0.0f, 0.0f};
    float expert_out[2] = {0.0f, 0.0f};
    float accum[2] = {0.0f, 0.0f};

    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_ROUTE_ORDER_U32) != 0u);
    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_EXPERT_ACCUMULATE_F32) != 0u);

    assert(elpis_dsv41_route_order_u32(chosen, order, 3u)
           == ELPIS_DSV41_NATIVE_OK);
    assert(order[0] == 1u);
    assert(order[1] == 2u);
    assert(order[2] == 0u);

    assert(elpis_dsv41_expert_accumulate_f32(
        x, w1, w3, w2,
        0.5f, 1, 0.0f,
        gate, up, expert_out, accum,
        2u, 2u) == ELPIS_DSV41_NATIVE_OK);

    assert(isfinite(accum[0]));
    assert(isfinite(accum[1]));

    /* Duplicate route IDs are invalid rather than ambiguously ordered. */
    chosen[2] = 1u;
    assert(elpis_dsv41_route_order_u32(chosen, order, 3u)
           == ELPIS_DSV41_NATIVE_INVALID);

    return 0;
}
