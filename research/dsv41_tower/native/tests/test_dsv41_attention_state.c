#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int closef(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    elpis_dsv41_attention_state *s = NULL;
    float local[4], copied[12];
    size_t rows = 0u, p, i;

    assert((elpis_dsv41_native_capabilities() & ELPIS_DSV41_CAP_ATTN_STATE_V1) != 0u);
    assert(elpis_dsv41_attention_state_create(3, 4, 4, 8, 2, 1, &s) == ELPIS_DSV41_NATIVE_OK);
    assert(s != NULL);

    assert(elpis_dsv41_attention_state_bytes(s) ==
           (3u * 4u + 4u * (4u + 4u) + 2u * 2u * 4u) * sizeof(float));
    assert(elpis_dsv41_attention_state_count(s) == 0u);

    for (p = 0u; p < 5u; ++p) {
        for (i = 0u; i < 4u; ++i)
            local[i] = (float)(10u * p + i);
        assert(elpis_dsv41_attention_local_store_f32(s, p, local) == ELPIS_DSV41_NATIVE_OK);
    }

    assert(elpis_dsv41_attention_local_copy_f32(s, 4, copied, 3, &rows) == ELPIS_DSV41_NATIVE_OK);
    assert(rows == 3u);

    for (i = 0u; i < 4u; ++i) {
        assert(closef(copied[i], 20.0f + (float)i, 1e-6f));
        assert(closef(copied[4u + i], 30.0f + (float)i, 1e-6f));
        assert(closef(copied[8u + i], 40.0f + (float)i, 1e-6f));
    }

    {
        float kv0[4] = {1, 2, 3, 4};
        float kv1[4] = {5, 6, 7, 8};
        float gate0[4] = {0, 0, 0, 0};
        float gate1[4] = {0, 0, 0, 0};
        float latent[4] = {0};
        float compressed[4] = {9, 10, 11, 12};
        float index_key[4] = {13, 14, 15, 16};
        float indexes[16] = {0};
        float gathered[4] = {0};
        uint32_t selected[1] = {0};
        int ready = -1;

        assert(elpis_dsv41_attention_compress_push_f32(
            s, 0, kv0, gate0, latent, &ready) == ELPIS_DSV41_NATIVE_OK);
        assert(ready == 0);

        assert(elpis_dsv41_attention_compress_push_f32(
            s, 1, kv1, gate1, latent, &ready) == ELPIS_DSV41_NATIVE_OK);
        assert(ready == 1);

        for (i = 0u; i < 4u; ++i)
            assert(closef(latent[i], (kv0[i] + kv1[i]) * 0.5f, 1e-6f));

        assert(elpis_dsv41_attention_publish_group_f32(
            s, 1, compressed, index_key) == ELPIS_DSV41_NATIVE_OK);
        assert(elpis_dsv41_attention_state_count(s) == 1u);

        assert(elpis_dsv41_attention_index_copy_f32(
            s, indexes, 4, &rows) == ELPIS_DSV41_NATIVE_OK);
        assert(rows == 1u);

        for (i = 0u; i < 4u; ++i)
            assert(closef(indexes[i], index_key[i], 1e-6f));

        assert(elpis_dsv41_attention_gather_compressed_f32(
            s, selected, 1, gathered) == ELPIS_DSV41_NATIVE_OK);

        for (i = 0u; i < 4u; ++i)
            assert(closef(gathered[i], compressed[i], 1e-6f));
    }

    assert(elpis_dsv41_attention_state_destroy(&s) == ELPIS_DSV41_NATIVE_OK);
    assert(s == NULL);
    assert(elpis_dsv41_attention_state_destroy(&s) == ELPIS_DSV41_NATIVE_OK);
    assert(s == NULL);

    return 0;
}
