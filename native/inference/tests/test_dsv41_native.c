#include "elpis/dsv41_native.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int closef(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    float linear_x[2] = {1.0f, 2.0f};
    float linear_w[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    float linear_out[2] = {0.0f, 0.0f};

    float rms_x[2] = {3.0f, 4.0f};
    float rms_w[2] = {1.0f, 2.0f};
    float rms_out[2] = {0.0f, 0.0f};
    float inv = 1.0f / sqrtf((9.0f + 16.0f) / 2.0f + 1e-6f);

    float stream[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float pre[2] = {0.25f, 0.75f};
    float pre_out[2] = {0.0f, 0.0f};

    float post_x[2] = {10.0f, 20.0f};
    float residual[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float post[2] = {1.0f, 2.0f};
    float comb[4] = {0.5f, 0.25f, 0.5f, 0.75f};
    float post_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    assert(elpis_dsv41_native_abi_version() == 1u);
    assert((elpis_dsv41_native_capabilities() &
            (ELPIS_DSV41_CAP_LINEAR_F32 |
             ELPIS_DSV41_CAP_RMS_F32 |
             ELPIS_DSV41_CAP_HC_PRE_F32 |
             ELPIS_DSV41_CAP_HC_POST_F32 |
             ELPIS_DSV41_CAP_QUANT_DEQUANT_F32 |
             ELPIS_DSV41_CAP_ROPE_F32 |
             ELPIS_DSV41_CAP_CANDIDATE_MASK_F32 |
             ELPIS_DSV41_CAP_SELECT_POSITIONS_F32 |
             ELPIS_DSV41_CAP_SPARSE_ATTN_F32)) != 0u);

    assert(elpis_dsv41_linear_f32(linear_x, linear_w, linear_out, 2, 2) == ELPIS_DSV41_NATIVE_OK);
    assert(closef(linear_out[0], 11.0f, 1e-6f));
    assert(closef(linear_out[1], 17.0f, 1e-6f));

    assert(elpis_dsv41_rms_f32(rms_x, rms_w, 1e-6f, rms_out, 2) == ELPIS_DSV41_NATIVE_OK);
    assert(closef(rms_out[0], 3.0f * inv, 1e-6f));
    assert(closef(rms_out[1], 8.0f * inv, 1e-6f));

    assert(elpis_dsv41_hc_pre_f32(stream, pre, pre_out, 2, 2) == ELPIS_DSV41_NATIVE_OK);
    assert(closef(pre_out[0], 2.5f, 1e-6f));
    assert(closef(pre_out[1], 3.5f, 1e-6f));

    assert(elpis_dsv41_hc_post_f32(post_x, residual, post, comb, post_out, 2, 2) ==
           ELPIS_DSV41_NATIVE_OK);
    assert(closef(post_out[0], 12.0f, 1e-6f));
    assert(closef(post_out[1], 23.0f, 1e-6f));
    assert(closef(post_out[2], 22.5f, 1e-6f));
    assert(closef(post_out[3], 43.5f, 1e-6f));

    assert(elpis_dsv41_linear_f32(NULL, linear_w, linear_out, 2, 2) ==
           ELPIS_DSV41_NATIVE_INVALID);
    assert(elpis_dsv41_rms_f32(rms_x, rms_w, -1.0f, rms_out, 2) ==
           ELPIS_DSV41_NATIVE_INVALID);
    assert(elpis_dsv41_hc_pre_f32(stream, pre, pre_out, 0, 2) ==
           ELPIS_DSV41_NATIVE_INVALID);
    assert(elpis_dsv41_hc_post_f32(post_x, residual, post, comb, post_out, 2, 0) ==
           ELPIS_DSV41_NATIVE_INVALID);

    /* R1 attention primitive coverage */
    {
        float qin[32] = {0};
        float qout[32] = {0};
        size_t i;
        for (i = 0; i < 32; ++i) qin[i] = (float)i / 10.0f;
        assert(elpis_dsv41_quant_dequant_f32(qin, qout, 1, 32,
               ELPIS_DSV41_QUANT_LOCAL) == ELPIS_DSV41_NATIVE_OK);
        for (i = 0; i < 32; ++i) assert(isfinite(qout[i]));
    }

    {
        float rx[4] = {9.0f, 8.0f, 1.0f, 2.0f};
        float rf[2] = {0.0f, 1.0f};
        float ro[4] = {0};
        assert(elpis_dsv41_rope_f32(rx, rf, ro, 1, 4, 1, 0) == ELPIS_DSV41_NATIVE_OK);
        assert(closef(ro[0], 9.0f, 1e-6f) && closef(ro[1], 8.0f, 1e-6f));
        assert(closef(ro[2], -2.0f, 1e-6f) && closef(ro[3], 1.0f, 1e-6f));
    }

    {
        float logits[7] = {1.0f, 2.0f, 9.0f, 3.0f, 4.0f, 5.0f, 0.0f};
        uint8_t mask[7] = {0};
        assert(elpis_dsv41_candidate_mask_f32(logits, mask, 7, 2, 3) == ELPIS_DSV41_NATIVE_OK);
        assert(mask[0] == 1 && mask[1] == 1 && mask[2] == 1);
        assert(mask[3] == 0 && mask[4] == 0 && mask[5] == 0);
        assert(mask[6] == 1);
    }

    {
        float scores[5] = {2.0f, 7.0f, 7.0f, -1.0f, 5.0f};
        uint32_t pos[3] = {0};
        size_t count = 0;
        assert(elpis_dsv41_select_positions_f32(scores, pos, 5, 3, &count) == ELPIS_DSV41_NATIVE_OK);
        assert(count == 3 && pos[0] == 1 && pos[1] == 2 && pos[2] == 4);
    }

    {
        float query[2] = {1.0f, 0.0f};
        float kvs[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float sinkv[1] = {0.0f};
        float sout[2] = {0};
        float scratch[2] = {0};
        assert(elpis_dsv41_sparse_attention_f32(query, kvs, sinkv, sout, scratch, 1, 2, 2) ==
               ELPIS_DSV41_NATIVE_OK);
        assert(isfinite(sout[0]) && isfinite(sout[1]));
    }

    /* R2 layer-math coverage */
    {
        float stream2[4]={1.0f,-2.0f,3.0f,4.0f};
        float fn[32]={0}; float scale3[3]={0}; float base8[8]={0};
        float pre2[2]={0}, post2[2]={0}, comb2[4]={0}, mix8[8]={0};
        assert(elpis_dsv41_hc_mixes_f32(stream2,fn,scale3,base8,pre2,post2,comb2,mix8,
               2,2,3,1e-6f,1e-6f)==ELPIS_DSV41_NATIVE_OK);
        assert(closef(pre2[0],0.500001f,2e-6f));
        assert(closef(post2[0],1.0f,2e-6f));
    }
    {
        float x2[2]={1.0f,2.0f};
        float rw[6]={1.0f,0.0f,0.0f,1.0f,-1.0f,-1.0f}; float rb[3]={0};
        uint32_t chosen2[2]={0}; float values2[2]={0}; float scores3[3]={0};
        assert(elpis_dsv41_route_f32(x2,rw,rb,chosen2,values2,scores3,2,3,2,
               ELPIS_DSV41_SCORE_SOFTMAX,1.0f,1,1.0f)==ELPIS_DSV41_NATIVE_OK);
        assert(chosen2[0]==1u && chosen2[1]==0u);
    }
    {
        float x2[2]={1.0f,2.0f}; float w1[4]={1,0,0,1}; float w3[4]={1,0,0,1};
        float w2[4]={1,0,0,1}; float gate2[2]={0},up2[2]={0},out2[2]={0};
        assert(elpis_dsv41_expert_f32(x2,w1,w3,w2,0.5f,1,0.0f,gate2,up2,out2,2,2)
               ==ELPIS_DSV41_NATIVE_OK);
        assert(isfinite(out2[0]) && isfinite(out2[1]));
    }

    /* R4 Engram gated-write coverage */
    {
        float stream4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float rows4[2] = {1.0f, -1.0f};
        float wkv12[12] = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, -1.0f,
            0.5f, 0.25f,
            -0.5f, 0.75f
        };
        float qw4[4] = {1,1,1,1};
        float kw4[4] = {1,1,1,1};
        float scratch6[6] = {0};
        float out4[4] = {0};

        assert(elpis_dsv41_engram_gated_write_f32(
            stream4, rows4, wkv12, qw4, kw4, 1e-6f,
            scratch6, out4, 2, 2, 2) == ELPIS_DSV41_NATIVE_OK);
        assert(isfinite(out4[0]));
        assert(isfinite(out4[1]));
        assert(isfinite(out4[2]));
        assert(isfinite(out4[3]));
    }

    /* R5 attention math coverage */
    {
        float iq4[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float keys6[6] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f};
        float iw2[2] = {2.0f, 3.0f};
        float score3[3] = {0};
        assert(elpis_dsv41_index_scores_f32(
            iq4, keys6, iw2, score3, 2, 2, 3) == ELPIS_DSV41_NATIVE_OK);
        assert(closef(score3[0], 2.0f, 1e-6f));
        assert(closef(score3[1], 3.0f, 1e-6f));
        assert(closef(score3[2], 0.0f, 1e-6f));
    }

    {
        float a4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float woa4[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float wob4[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float scratch2[2] = {0};
        float out2[2] = {0};
        assert(elpis_dsv41_grouped_output_f32(
            a4, woa4, wob4, scratch2, out2,
            2, 2, 2, 1, 2) == ELPIS_DSV41_NATIVE_OK);
        assert(closef(out2[0], 1.0f, 1e-6f));
        assert(closef(out2[1], 4.0f, 1e-6f));
    }

    return 0;
}
