
#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

int main(void) {
    elpis_dsv41_attention_state *state = NULL;
    const size_t dimension = 2u;
    const size_t q_rank = 2u;
    const size_t heads = 1u;
    const size_t head_dim = 32u;
    const size_t rope_pairs = 1u;
    const size_t index_heads = 1u;
    const size_t index_dim = 32u;
    const size_t groups = 1u;
    const size_t rank = 1u;
    const size_t local_window = 2u;
    const size_t ratio = 1u;
    const size_t max_groups = 4u;
    const size_t index_topk = 1u;

    float x[2] = {1.0f, -1.0f};
    float wq_a[4] = {0};
    float q_norm[2] = {1, 1};
    float wq_b[64] = {0};
    float wkv[64] = {0};
    float kv_norm[32];
    float sink[1] = {0};
    float wo_a[32] = {0};
    float wo_b[2] = {0};

    float compressor_wkv[64] = {0};
    float compressor_norm[32];
    float indexer_wk[1024] = {0};
    float indexer_k_norm[32];
    float indexer_wq_b[64] = {0};
    float indexer_weights_proj[2] = {0};

    float freq[2] = {1.0f, 0.0f};
    uint32_t selected[1] = {99u};
    uint8_t candidates[4] = {0};
    size_t selected_count = 0u;
    size_t candidate_count = 0u;
    float out[2] = {1, 1};
    size_t scratch_floats;
    float *scratch;
    size_t i;

    assert((elpis_dsv41_native_capabilities() &
            ELPIS_DSV41_CAP_COMPRESSED_ATTN_APPLY_F32) != 0u);

    for (i = 0u; i < 32u; ++i) {
        kv_norm[i] = 1.0f;
        compressor_norm[i] = 1.0f;
        indexer_k_norm[i] = 1.0f;
    }

    assert(elpis_dsv41_attention_state_create(
        local_window, head_dim, index_dim, 4u, ratio, 1, &state)
        == ELPIS_DSV41_NATIVE_OK);

    scratch_floats = elpis_dsv41_compressed_attention_scratch_floats(
        q_rank, heads, head_dim, index_heads, index_dim,
        groups, rank, local_window, max_groups, index_topk);
    assert(scratch_floats > 0u);
    scratch = (float *)calloc(scratch_floats, sizeof(float));
    assert(scratch != NULL);

    assert(elpis_dsv41_compressed_attention_apply_f32(
        state, state, 0u, x,
        wq_a, q_norm, wq_b, wkv, kv_norm, sink, wo_a, wo_b,
        compressor_wkv, NULL, compressor_norm,
        indexer_wk, indexer_k_norm,
        indexer_wq_b, indexer_weights_proj,
        freq, freq, 1e-6f,
        selected, 1u, &selected_count,
        candidates, 4u, &candidate_count,
        scratch, scratch_floats, out,
        dimension, q_rank, heads, head_dim, rope_pairs,
        index_heads, index_dim, groups, rank, local_window,
        ratio, max_groups, index_topk,
        1, 1, 0, 0u, 0u) == ELPIS_DSV41_NATIVE_OK);

    assert(selected_count == 1u);
    assert(selected[0] == 0u);
    assert(elpis_dsv41_attention_state_count(state) == 1u);
    assert(isfinite(out[0]) && isfinite(out[1]));
    assert(fabsf(out[0]) <= 1e-6f);
    assert(fabsf(out[1]) <= 1e-6f);

    free(scratch);
    assert(elpis_dsv41_attention_state_destroy(&state)
           == ELPIS_DSV41_NATIVE_OK);
    return 0;
}
