
#include "elpis/dsv41_native.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

size_t
elpis_dsv41_local_attention_scratch_floats(size_t dimension,
                                           size_t q_rank,
                                           size_t heads,
                                           size_t head_dim,
                                           size_t o_groups,
                                           size_t o_rank,
                                           size_t local_window) {
    size_t hd, history, grouped, total = 0u;
    (void)dimension;

    if (q_rank == 0u || heads == 0u || head_dim == 0u ||
        o_groups == 0u || o_rank == 0u || local_window == 0u) {
        return 0u;
    }
    if (!mul_size(heads, head_dim, &hd) ||
        !mul_size(local_window, head_dim, &history) ||
        !mul_size(o_groups, o_rank, &grouped)) {
        return 0u;
    }

    /* qa + qnorm */
    if (!add_size(total, q_rank, &total) ||
        !add_size(total, q_rank, &total)) {
        return 0u;
    }
    /* q projection, rotated query, sparse output, inverse-RoPE output */
    if (hd > SIZE_MAX / 4u || !add_size(total, 4u * hd, &total)) {
        return 0u;
    }
    /* KV projection, RMS, RoPE, local quantized row */
    if (head_dim > SIZE_MAX / 4u ||
        !add_size(total, 4u * head_dim, &total)) {
        return 0u;
    }
    /* chronological local history + grouped projection scratch + score scratch */
    if (!add_size(total, history, &total) ||
        !add_size(total, grouped, &total) ||
        !add_size(total, local_window, &total)) {
        return 0u;
    }
    return total;
}

elpis_dsv41_native_status
elpis_dsv41_local_attention_apply_f32(
    elpis_dsv41_attention_state *state,
    size_t position,
    const float *x,
    const float *wq_a,
    const float *q_norm,
    const float *wq_b,
    const float *wkv,
    const float *kv_norm,
    const float *sink,
    const float *wo_a,
    const float *wo_b,
    const float *freq,
    float norm_eps,
    float *scratch,
    size_t scratch_floats,
    float *out,
    size_t dimension,
    size_t q_rank,
    size_t heads,
    size_t head_dim,
    size_t rope_pairs,
    size_t o_groups,
    size_t o_rank,
    size_t local_window) {

    size_t required, hd, grouped, at = 0u, rows = 0u;
    float *qa;
    float *qn;
    float *qproj;
    float *query;
    float *kv0;
    float *kv1;
    float *kv2;
    float *local;
    float *history;
    float *attn;
    float *inverse;
    float *projection;
    float *score_scratch;
    elpis_dsv41_native_status rc;

    if (state == NULL || x == NULL ||
        wq_a == NULL || q_norm == NULL || wq_b == NULL ||
        wkv == NULL || kv_norm == NULL || sink == NULL ||
        wo_a == NULL || wo_b == NULL || freq == NULL ||
        scratch == NULL || out == NULL ||
        dimension == 0u || q_rank == 0u || heads == 0u ||
        head_dim == 0u || rope_pairs == 0u ||
        rope_pairs > head_dim / 2u ||
        o_groups == 0u || o_rank == 0u || local_window == 0u ||
        heads % o_groups != 0u ||
        !isfinite(norm_eps) || norm_eps < 0.0f) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    required = elpis_dsv41_local_attention_scratch_floats(
        dimension, q_rank, heads, head_dim, o_groups, o_rank, local_window);
    if (required == 0u || scratch_floats < required) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    if (!mul_size(heads, head_dim, &hd) ||
        !mul_size(o_groups, o_rank, &grouped)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    qa = scratch + at; at += q_rank;
    qn = scratch + at; at += q_rank;
    qproj = scratch + at; at += hd;
    query = scratch + at; at += hd;
    kv0 = scratch + at; at += head_dim;
    kv1 = scratch + at; at += head_dim;
    kv2 = scratch + at; at += head_dim;
    local = scratch + at; at += head_dim;
    history = scratch + at; at += local_window * head_dim;
    attn = scratch + at; at += hd;
    inverse = scratch + at; at += hd;
    projection = scratch + at; at += grouped;
    score_scratch = scratch + at; at += local_window;

    if (at > scratch_floats) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    rc = elpis_dsv41_linear_f32(x, wq_a, qa, dimension, q_rank);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rms_f32(qa, q_norm, norm_eps, qn, q_rank);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_linear_f32(qn, wq_b, qproj, q_rank, hd);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rope_f32(
        qproj, freq, query, heads, head_dim, rope_pairs, 0);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_linear_f32(x, wkv, kv0, dimension, head_dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rms_f32(kv0, kv_norm, norm_eps, kv1, head_dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rope_f32(
        kv1, freq, kv2, 1u, head_dim, rope_pairs, 0);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_quant_dequant_f32(
        kv2, local, 1u, head_dim, ELPIS_DSV41_QUANT_LOCAL);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_attention_local_store_f32(state, position, local);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_attention_local_copy_f32(
        state, position, history, local_window, &rows);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
    if (rows == 0u || rows > local_window) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    rc = elpis_dsv41_sparse_attention_f32(
        query, history, sink, attn, score_scratch, heads, rows, head_dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rope_f32(
        attn, freq, inverse, heads, head_dim, rope_pairs, 1);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    return elpis_dsv41_grouped_output_f32(
        inverse, wo_a, wo_b, projection, out,
        heads, head_dim, o_groups, o_rank, dimension);
}
