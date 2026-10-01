
#include "elpis/dsv41_native.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int r9_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int r9_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static size_t r9_mix_floats(size_t copies) {
    size_t rows;
    size_t out;
    if (!r9_add(copies, 2u, &rows)) return 0u;
    if (!r9_mul(rows, copies, &out)) return 0u;
    return out;
}

size_t
elpis_dsv41_layer_frame_scratch_floats(size_t copies,
                                       size_t dim,
                                       size_t row_values,
                                       size_t expert_count,
                                       int engram_enabled) {
    size_t mix;
    size_t begin_total = 0u;
    size_t after_total = 0u;
    size_t kv = 0u;

    if (copies == 0u || dim == 0u || expert_count == 0u ||
        (engram_enabled != 0 && engram_enabled != 1)) {
        return 0u;
    }

    mix = r9_mix_floats(copies);
    if (mix == 0u) return 0u;

    if (engram_enabled) {
        size_t copies_plus_one;
        if (row_values == 0u ||
            !r9_add(copies, 1u, &copies_plus_one) ||
            !r9_mul(copies_plus_one, dim, &kv)) {
            return 0u;
        }
    }

    if (!r9_add(kv, mix, &begin_total) ||
        !r9_add(begin_total, dim, &begin_total)) {
        return 0u;
    }

    if (!r9_add(mix, dim, &after_total) ||
        !r9_add(after_total, expert_count, &after_total)) {
        return 0u;
    }

    return begin_total > after_total ? begin_total : after_total;
}

elpis_dsv41_native_status
elpis_dsv41_layer_begin_f32(
    const float *stream,
    const float *pre_mix,
    const float *engram_rows,
    const float *engram_wkv,
    const float *engram_q_weight,
    const float *engram_k_weight,
    int engram_enabled,
    size_t row_values,
    const float *hc_attn_fn,
    const float *hc_attn_scale,
    const float *hc_attn_base,
    const float *attn_norm,
    float norm_eps,
    float hc_eps,
    size_t sinkhorn_iters,
    float *scratch,
    size_t scratch_floats,
    float *stream_work,
    float *ap,
    float *apost,
    float *ac,
    float *attention_x,
    size_t copies,
    size_t dim) {

    size_t mix_floats;
    size_t stream_floats;
    size_t kv_floats = 0u;
    size_t required;
    size_t at = 0u;
    float *kv_scratch = NULL;
    float *mix_scratch;
    float *vector_scratch;
    elpis_dsv41_native_status rc;

    if (stream == NULL || pre_mix == NULL ||
        hc_attn_fn == NULL || hc_attn_scale == NULL ||
        hc_attn_base == NULL || attn_norm == NULL ||
        scratch == NULL || stream_work == NULL ||
        ap == NULL || apost == NULL || ac == NULL ||
        attention_x == NULL ||
        copies == 0u || dim == 0u ||
        (engram_enabled != 0 && engram_enabled != 1)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (!r9_mul(copies, dim, &stream_floats)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    mix_floats = r9_mix_floats(copies);
    if (mix_floats == 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (engram_enabled) {
        size_t copies_plus_one;
        if (engram_rows == NULL || engram_wkv == NULL ||
            engram_q_weight == NULL || engram_k_weight == NULL ||
            row_values == 0u ||
            !r9_add(copies, 1u, &copies_plus_one) ||
            !r9_mul(copies_plus_one, dim, &kv_floats)) {
            return ELPIS_DSV41_NATIVE_INVALID;
        }
    }

    required = kv_floats;
    if (!r9_add(required, mix_floats, &required) ||
        !r9_add(required, dim, &required) ||
        scratch_floats < required) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (engram_enabled) {
        kv_scratch = scratch + at;
        at += kv_floats;
        rc = elpis_dsv41_engram_gated_write_f32(
            stream, engram_rows, engram_wkv,
            engram_q_weight, engram_k_weight,
            norm_eps, kv_scratch, stream_work,
            copies, dim, row_values);
        if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
    } else {
        memcpy(stream_work, stream, stream_floats * sizeof(float));
    }

    mix_scratch = scratch + at;
    at += mix_floats;
    vector_scratch = scratch + at;
    at += dim;

    if (at > scratch_floats) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    rc = elpis_dsv41_hc_mixes_f32(
        stream_work,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        ap,
        apost,
        ac,
        mix_scratch,
        copies,
        dim,
        sinkhorn_iters,
        norm_eps,
        hc_eps);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_hc_pre_f32(
        stream_work, pre_mix, vector_scratch, copies, dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    return elpis_dsv41_rms_f32(
        vector_scratch, attn_norm, norm_eps, attention_x, dim);
}

elpis_dsv41_native_status
elpis_dsv41_layer_after_attention_route_f32(
    const float *stream_work,
    const float *attention_out,
    const float *ap,
    const float *apost,
    const float *ac,
    const float *hc_ffn_fn,
    const float *hc_ffn_scale,
    const float *hc_ffn_base,
    const float *ffn_norm,
    const float *router_weight,
    const float *router_bias,
    float norm_eps,
    float hc_eps,
    size_t sinkhorn_iters,
    uint32_t score_mode,
    float gate_temp,
    int normalize_topk,
    float route_scale,
    float *scratch,
    size_t scratch_floats,
    float *stream_after_attention,
    float *fp,
    float *fpost,
    float *fc,
    float *moe_x,
    uint32_t *chosen,
    float *route_values,
    uint32_t *route_order,
    size_t copies,
    size_t dim,
    size_t expert_count,
    size_t active) {

    size_t mix_floats;
    size_t required;
    size_t at = 0u;
    float *mix_scratch;
    float *vector_scratch;
    float *route_scratch;
    elpis_dsv41_native_status rc;

    if (stream_work == NULL || attention_out == NULL ||
        ap == NULL || apost == NULL || ac == NULL ||
        hc_ffn_fn == NULL || hc_ffn_scale == NULL ||
        hc_ffn_base == NULL || ffn_norm == NULL ||
        router_weight == NULL || router_bias == NULL ||
        scratch == NULL || stream_after_attention == NULL ||
        fp == NULL || fpost == NULL || fc == NULL ||
        moe_x == NULL || chosen == NULL ||
        route_values == NULL || route_order == NULL ||
        copies == 0u || dim == 0u ||
        expert_count == 0u || active == 0u ||
        active > expert_count) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    mix_floats = r9_mix_floats(copies);
    if (mix_floats == 0u) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    required = mix_floats;
    if (!r9_add(required, dim, &required) ||
        !r9_add(required, expert_count, &required) ||
        scratch_floats < required) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    mix_scratch = scratch + at;
    at += mix_floats;
    vector_scratch = scratch + at;
    at += dim;
    route_scratch = scratch + at;
    at += expert_count;

    if (at > scratch_floats) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    rc = elpis_dsv41_hc_post_f32(
        attention_out,
        stream_work,
        apost,
        ac,
        stream_after_attention,
        copies,
        dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_hc_mixes_f32(
        stream_after_attention,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        fp,
        fpost,
        fc,
        mix_scratch,
        copies,
        dim,
        sinkhorn_iters,
        norm_eps,
        hc_eps);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_hc_pre_f32(
        stream_after_attention,
        ap,
        vector_scratch,
        copies,
        dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rms_f32(
        vector_scratch, ffn_norm, norm_eps, moe_x, dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_route_f32(
        moe_x,
        router_weight,
        router_bias,
        chosen,
        route_values,
        route_scratch,
        dim,
        expert_count,
        active,
        score_mode,
        gate_temp,
        normalize_topk,
        route_scale);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    return elpis_dsv41_route_order_u32(
        chosen, route_order, active);
}

elpis_dsv41_native_status
elpis_dsv41_layer_finish_f32(const float *moe_out,
                             const float *stream_after_attention,
                             const float *fpost,
                             const float *fc,
                             float *stream_out,
                             size_t copies,
                             size_t dim) {
    return elpis_dsv41_hc_post_f32(
        moe_out,
        stream_after_attention,
        fpost,
        fc,
        stream_out,
        copies,
        dim);
}
