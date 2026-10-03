
#include "elpis/dsv41_native.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int r7_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int r7_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int r7_add_n(size_t *total, size_t n) {
    size_t next;
    if (!r7_add(*total, n, &next)) return 0;
    *total = next;
    return 1;
}

size_t
elpis_dsv41_compressed_attention_scratch_floats(
    size_t q_rank,
    size_t heads,
    size_t head_dim,
    size_t index_heads,
    size_t index_dim,
    size_t o_groups,
    size_t o_rank,
    size_t local_window,
    size_t max_groups,
    size_t index_topk) {
    size_t hd, iq, combined_rows, combined, grouped, keys, total = 0u;

    if (q_rank == 0u || heads == 0u || head_dim == 0u ||
        index_heads == 0u || index_dim == 0u ||
        o_groups == 0u || o_rank == 0u ||
        local_window == 0u || max_groups == 0u || index_topk == 0u) {
        return 0u;
    }

    if (!r7_mul(heads, head_dim, &hd) ||
        !r7_mul(index_heads, index_dim, &iq) ||
        !r7_add(local_window, index_topk, &combined_rows) ||
        !r7_mul(combined_rows, head_dim, &combined) ||
        !r7_mul(o_groups, o_rank, &grouped) ||
        !r7_mul(max_groups, index_dim, &keys)) {
        return 0u;
    }

    /* qa, qn */
    if (!r7_add_n(&total, q_rank) || !r7_add_n(&total, q_rank)) return 0u;

    /* qproj, query, attn, inverse */
    if (hd > SIZE_MAX / 4u || !r7_add_n(&total, 4u * hd)) return 0u;

    /* local path kv0/kv1/kv2/local */
    if (head_dim > SIZE_MAX / 4u ||
        !r7_add_n(&total, 4u * head_dim)) return 0u;

    /* combined local+selected KV, grouped projection, attention score scratch */
    if (!r7_add_n(&total, combined) ||
        !r7_add_n(&total, grouped) ||
        !r7_add_n(&total, combined_rows)) return 0u;

    /* owner path: comp kv, gate, latent raw/norm, compressed rotate/q */
    if (head_dim > SIZE_MAX / 6u ||
        !r7_add_n(&total, 6u * head_dim)) return 0u;

    /* owner index key path: linear, rms, rotate, quant */
    if (index_dim > SIZE_MAX / 4u ||
        !r7_add_n(&total, 4u * index_dim)) return 0u;

    /* index source query path: linear, rotate, quant */
    if (iq > SIZE_MAX / 3u || !r7_add_n(&total, 3u * iq)) return 0u;

    /* index head weights, owner key copy, index scores */
    if (!r7_add_n(&total, index_heads) ||
        !r7_add_n(&total, keys) ||
        !r7_add_n(&total, max_groups)) return 0u;

    return total;
}

elpis_dsv41_native_status
elpis_dsv41_compressed_attention_apply_f32(
    elpis_dsv41_attention_state *state,
    elpis_dsv41_attention_state *owner_state,
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
    const float *compressor_wkv,
    const float *compressor_wgate,
    const float *compressor_norm,
    const float *indexer_wk,
    const float *indexer_k_norm,
    const float *indexer_wq_b,
    const float *indexer_weights_proj,
    const float *freq,
    const float *group_freq,
    float norm_eps,
    uint32_t *selected_io,
    size_t selected_capacity,
    size_t *selected_count_io,
    uint8_t *candidates_io,
    size_t candidate_capacity,
    size_t *candidate_count_io,
    float *scratch,
    size_t scratch_floats,
    float *out,
    size_t dimension,
    size_t q_rank,
    size_t heads,
    size_t head_dim,
    size_t rope_pairs,
    size_t index_heads,
    size_t index_dim,
    size_t o_groups,
    size_t o_rank,
    size_t local_window,
    size_t ratio,
    size_t max_groups,
    size_t index_topk,
    int kv_owner,
    int index_source,
    int candidate_mode,
    size_t candidate_blocks,
    size_t candidate_block_size) {

    size_t required, hd, iq, grouped, combined_rows, at = 0u;
    size_t local_rows = 0u, n, owner_rows = 0u, selected_count;
    float *qa, *qn, *qproj, *query;
    float *kv0, *kv1, *kv2, *local;
    float *combined_kv, *attn, *inverse, *projection, *attn_scores;
    float *comp_kv, *gate, *latent_raw, *latent_norm, *compressed_rot, *compressed_q;
    float *key0, *key1, *key2, *key_q;
    float *iq0, *iq1, *iq2, *iw, *keys, *scores;
    elpis_dsv41_native_status rc;
    int ready = 0;
    size_t i;

    if (state == NULL || owner_state == NULL ||
        x == NULL || wq_a == NULL || q_norm == NULL || wq_b == NULL ||
        wkv == NULL || kv_norm == NULL || sink == NULL ||
        wo_a == NULL || wo_b == NULL || freq == NULL ||
        selected_io == NULL || selected_count_io == NULL ||
        candidates_io == NULL || candidate_count_io == NULL ||
        scratch == NULL || out == NULL ||
        dimension == 0u || q_rank == 0u || heads == 0u || head_dim == 0u ||
        rope_pairs == 0u || rope_pairs > head_dim / 2u ||
        index_heads == 0u || index_dim == 0u || rope_pairs > index_dim / 2u ||
        o_groups == 0u || o_rank == 0u || local_window == 0u ||
        ratio == 0u || max_groups == 0u || index_topk == 0u ||
        heads % o_groups != 0u ||
        selected_capacity < index_topk ||
        (kv_owner != 0 && kv_owner != 1) ||
        (index_source != 0 && index_source != 1) ||
        candidate_mode < 0 || candidate_mode > 2 ||
        !isfinite(norm_eps) || norm_eps < 0.0f) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (kv_owner) {
        if (owner_state != state || !index_source ||
            compressor_wkv == NULL || compressor_norm == NULL ||
            indexer_wk == NULL || indexer_k_norm == NULL ||
            (ratio > 1u && compressor_wgate == NULL)) {
            return ELPIS_DSV41_NATIVE_INVALID;
        }
    }
    if (index_source) {
        if (indexer_wq_b == NULL || indexer_weights_proj == NULL) {
            return ELPIS_DSV41_NATIVE_INVALID;
        }
    } else if (candidate_mode != 0) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }
    if (candidate_mode != 0 &&
        (candidate_blocks == 0u || candidate_block_size == 0u)) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    n = (position + 1u) / ratio;
    if (n > max_groups) return ELPIS_DSV41_NATIVE_INVALID;

    required = elpis_dsv41_compressed_attention_scratch_floats(
        q_rank, heads, head_dim, index_heads, index_dim,
        o_groups, o_rank, local_window, max_groups, index_topk);
    if (required == 0u || scratch_floats < required) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (!r7_mul(heads, head_dim, &hd) ||
        !r7_mul(index_heads, index_dim, &iq) ||
        !r7_mul(o_groups, o_rank, &grouped) ||
        !r7_add(local_window, index_topk, &combined_rows)) {
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

    combined_kv = scratch + at; at += combined_rows * head_dim;
    attn = scratch + at; at += hd;
    inverse = scratch + at; at += hd;
    projection = scratch + at; at += grouped;
    attn_scores = scratch + at; at += combined_rows;

    comp_kv = scratch + at; at += head_dim;
    gate = scratch + at; at += head_dim;
    latent_raw = scratch + at; at += head_dim;
    latent_norm = scratch + at; at += head_dim;
    compressed_rot = scratch + at; at += head_dim;
    compressed_q = scratch + at; at += head_dim;

    key0 = scratch + at; at += index_dim;
    key1 = scratch + at; at += index_dim;
    key2 = scratch + at; at += index_dim;
    key_q = scratch + at; at += index_dim;

    iq0 = scratch + at; at += iq;
    iq1 = scratch + at; at += iq;
    iq2 = scratch + at; at += iq;
    iw = scratch + at; at += index_heads;
    keys = scratch + at; at += max_groups * index_dim;
    scores = scratch + at; at += max_groups;

    if (at > scratch_floats) return ELPIS_DSV41_NATIVE_INVALID;

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
        state, position, combined_kv, local_window, &local_rows);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    if (kv_owner) {
        rc = elpis_dsv41_linear_f32(
            x, compressor_wkv, comp_kv, dimension, head_dim);
        if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

        if (ratio > 1u) {
            rc = elpis_dsv41_linear_f32(
                x, compressor_wgate, gate, dimension, head_dim);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
        } else {
            memset(gate, 0, head_dim * sizeof(float));
        }

        rc = elpis_dsv41_attention_compress_push_f32(
            state, position, comp_kv, gate, latent_raw, &ready);
        if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

        if (ready) {
            if (group_freq == NULL) return ELPIS_DSV41_NATIVE_INVALID;

            rc = elpis_dsv41_rms_f32(
                latent_raw, compressor_norm, norm_eps, latent_norm, head_dim);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

            rc = elpis_dsv41_linear_f32(
                latent_norm, indexer_wk, key0, head_dim, index_dim);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_rms_f32(
                key0, indexer_k_norm, norm_eps, key1, index_dim);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_rope_f32(
                key1, group_freq, key2, 1u, index_dim, rope_pairs, 0);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_quant_dequant_f32(
                key2, key_q, 1u, index_dim, ELPIS_DSV41_QUANT_INDEX);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

            rc = elpis_dsv41_rope_f32(
                latent_norm, group_freq, compressed_rot,
                1u, head_dim, rope_pairs, 0);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_quant_dequant_f32(
                compressed_rot, compressed_q, 1u, head_dim,
                ELPIS_DSV41_QUANT_COMPRESSED);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

            /*
             * publish_group writes both key and compressed row. This remains
             * observationally equivalent to the Python ordering: scoring reads
             * only index_keys; compressed rows are consumed only after scoring.
             */
            rc = elpis_dsv41_attention_publish_group_f32(
                state, position, compressed_q, key_q);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
        }

        if (elpis_dsv41_attention_state_count(state) != n) {
            return ELPIS_DSV41_NATIVE_INVALID;
        }
    } else if (elpis_dsv41_attention_state_count(owner_state) != n) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (index_source) {
        if (n == 0u) {
            *selected_count_io = 0u;
        } else {
            float scale;

            rc = elpis_dsv41_linear_f32(
                qn, indexer_wq_b, iq0, q_rank, iq);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_rope_f32(
                iq0, freq, iq1, index_heads, index_dim, rope_pairs, 0);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            rc = elpis_dsv41_quant_dequant_f32(
                iq1, iq2, index_heads, index_dim, ELPIS_DSV41_QUANT_INDEX);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

            rc = elpis_dsv41_linear_f32(
                x, indexer_weights_proj, iw, dimension, index_heads);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            scale = (float)((1.0 / sqrt((double)index_dim)) *
                            (1.0 / sqrt((double)index_heads)));
            for (i = 0u; i < index_heads; ++i) iw[i] *= scale;

            rc = elpis_dsv41_attention_index_copy_f32(
                owner_state, keys, max_groups, &owner_rows);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
            if (owner_rows != n) return ELPIS_DSV41_NATIVE_INVALID;

            rc = elpis_dsv41_index_scores_f32(
                iq2, keys, iw, scores, index_heads, index_dim, n);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

            if (candidate_mode == 1) {
                if (candidate_capacity < n) return ELPIS_DSV41_NATIVE_INVALID;
                rc = elpis_dsv41_candidate_mask_f32(
                    scores, candidates_io, n,
                    candidate_blocks, candidate_block_size);
                if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
                *candidate_count_io = n;
            } else if (candidate_mode == 2) {
                if (*candidate_count_io != n || candidate_capacity < n) {
                    return ELPIS_DSV41_NATIVE_INVALID;
                }
                for (i = 0u; i < n; ++i) {
                    if (candidates_io[i] == 0u) scores[i] = -INFINITY;
                }
            }

            rc = elpis_dsv41_select_positions_f32(
                scores, selected_io, n, index_topk, selected_count_io);
            if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
        }
    }

    selected_count = *selected_count_io;
    if (selected_count > selected_capacity ||
        selected_count > index_topk ||
        selected_count > n) {
        return ELPIS_DSV41_NATIVE_INVALID;
    }

    if (selected_count != 0u) {
        rc = elpis_dsv41_attention_gather_compressed_f32(
            owner_state, selected_io, selected_count,
            combined_kv + local_rows * head_dim);
        if (rc != ELPIS_DSV41_NATIVE_OK) return rc;
    }

    rc = elpis_dsv41_sparse_attention_f32(
        query, combined_kv, sink, attn, attn_scores,
        heads, local_rows + selected_count, head_dim);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    rc = elpis_dsv41_rope_f32(
        attn, freq, inverse, heads, head_dim, rope_pairs, 1);
    if (rc != ELPIS_DSV41_NATIVE_OK) return rc;

    return elpis_dsv41_grouped_output_f32(
        inverse, wo_a, wo_b, projection, out,
        heads, head_dim, o_groups, o_rank, dimension);
}
