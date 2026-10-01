
#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int main(void) {
    elpis_dsv41_attention_state *state = NULL;
    const size_t dimension = 2u;
    const size_t q_rank = 2u;
    const size_t heads = 1u;
    const size_t head_dim = 32u;
    const size_t rope_pairs = 1u;
    const size_t groups = 1u;
    const size_t rank = 1u;
    const size_t local_window = 2u;
    size_t scratch_floats;
    float x[2] = {1.0f, -1.0f};
    float wq_a[4] = {0};
    float q_norm[2] = {1.0f, 1.0f};
    float wq_b[64] = {0};
    float wkv[64] = {0};
    float kv_norm[32];
    float sink[1] = {0.0f};
    float wo_a[32] = {0};
    float wo_b[2] = {0};
    float freq[2] = {1.0f, 0.0f};
    float out[2] = {123.0f, 456.0f};
    float *scratch;
    size_t i;

    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_LOCAL_ATTN_APPLY_F32) != 0u);

    for (i = 0u; i < 32u; ++i) {
        kv_norm[i] = 1.0f;
    }

    assert(elpis_dsv41_attention_state_create(
        local_window, head_dim, head_dim, 4u, 0u, 0, &state)
        == ELPIS_DSV41_NATIVE_OK);

    scratch_floats = elpis_dsv41_local_attention_scratch_floats(
        dimension, q_rank, heads, head_dim, groups, rank, local_window);
    assert(scratch_floats > 0u);
    scratch = (float *)calloc(scratch_floats, sizeof(float));
    assert(scratch != NULL);

    assert(elpis_dsv41_local_attention_apply_f32(
        state, 0u, x,
        wq_a, q_norm, wq_b, wkv, kv_norm, sink, wo_a, wo_b, freq,
        1e-6f, scratch, scratch_floats, out,
        dimension, q_rank, heads, head_dim, rope_pairs,
        groups, rank, local_window) == ELPIS_DSV41_NATIVE_OK);

    assert(isfinite(out[0]) && isfinite(out[1]));
    assert(fabsf(out[0]) <= 1e-6f);
    assert(fabsf(out[1]) <= 1e-6f);

    free(scratch);
    assert(elpis_dsv41_attention_state_destroy(&state)
           == ELPIS_DSV41_NATIVE_OK);
    assert(state == NULL);
    return 0;
}
