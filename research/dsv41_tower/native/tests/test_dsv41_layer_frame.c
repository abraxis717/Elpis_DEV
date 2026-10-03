
#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

int main(void) {
    const size_t copies = 2u;
    const size_t dim = 2u;
    const size_t expert_count = 2u;
    const size_t active = 1u;
    const size_t mix_dim = (2u + copies) * copies;

    float stream[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float pre_mix[2] = {1.0f, 0.0f};
    float hc_fn[32] = {0};
    float hc_scale[3] = {0};
    float hc_base[8] = {0};
    float norm[2] = {1.0f, 1.0f};
    float router[4] = {0};
    float bias[2] = {0};

    float stream_work[4] = {0};
    float ap[2] = {0};
    float apost[2] = {0};
    float ac[4] = {0};
    float attention_x[2] = {0};

    float attention_out[2] = {0};
    float stream_after[4] = {0};
    float fp[2] = {0};
    float fpost[2] = {0};
    float fc[4] = {0};
    float moe_x[2] = {0};
    uint32_t chosen[1] = {99u};
    float values[1] = {0};
    uint32_t order[1] = {99u};

    float moe_out[2] = {0};
    float stream_out[4] = {0};

    size_t scratch_floats;
    float *scratch;
    size_t i;

    (void)mix_dim;

    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_LAYER_BEGIN_F32) != 0u);
    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_LAYER_AFTER_ATTN_ROUTE_F32) != 0u);
    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_LAYER_FINISH_F32) != 0u);

    scratch_floats = elpis_dsv41_layer_frame_scratch_floats(
        copies, dim, 0u, expert_count, 0);
    assert(scratch_floats > 0u);
    scratch = (float *)calloc(scratch_floats, sizeof(float));
    assert(scratch != NULL);

    assert(elpis_dsv41_layer_begin_f32(
        stream, pre_mix,
        NULL, NULL, NULL, NULL, 0, 0u,
        hc_fn, hc_scale, hc_base, norm,
        1e-6f, 1e-6f, 2u,
        scratch, scratch_floats,
        stream_work, ap, apost, ac, attention_x,
        copies, dim) == ELPIS_DSV41_NATIVE_OK);

    for (i = 0u; i < 4u; ++i) {
        assert(isfinite(stream_work[i]));
    }
    for (i = 0u; i < 2u; ++i) {
        assert(isfinite(attention_x[i]));
    }

    assert(elpis_dsv41_layer_after_attention_route_f32(
        stream_work, attention_out, ap, apost, ac,
        hc_fn, hc_scale, hc_base, norm,
        router, bias,
        1e-6f, 1e-6f, 2u,
        ELPIS_DSV41_SCORE_SOFTMAX,
        1.0f, 1, 1.0f,
        scratch, scratch_floats,
        stream_after, fp, fpost, fc, moe_x,
        chosen, values, order,
        copies, dim, expert_count, active)
        == ELPIS_DSV41_NATIVE_OK);

    assert(chosen[0] == 0u);
    assert(order[0] == 0u);

    assert(elpis_dsv41_layer_finish_f32(
        moe_out, stream_after, fpost, fc,
        stream_out, copies, dim)
        == ELPIS_DSV41_NATIVE_OK);

    for (i = 0u; i < 4u; ++i) {
        assert(isfinite(stream_out[i]));
    }

    free(scratch);
    return 0;
}
