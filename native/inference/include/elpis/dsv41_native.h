#ifndef ELPIS_DSV41_NATIVE_H
#define ELPIS_DSV41_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ELPIS_DSV41_NATIVE_ABI_V1 = 1u
};

typedef enum {
    ELPIS_DSV41_NATIVE_OK = 0,
    ELPIS_DSV41_NATIVE_INVALID = -1,
    ELPIS_DSV41_NATIVE_NONFINITE = -2,
    ELPIS_DSV41_NATIVE_NOMEM = -3
} elpis_dsv41_native_status;

typedef struct elpis_dsv41_attention_state elpis_dsv41_attention_state;

enum {
    ELPIS_DSV41_CAP_LINEAR_F32 = UINT64_C(1) << 0,
    ELPIS_DSV41_CAP_RMS_F32 = UINT64_C(1) << 1,
    ELPIS_DSV41_CAP_HC_PRE_F32 = UINT64_C(1) << 2,
    ELPIS_DSV41_CAP_HC_POST_F32 = UINT64_C(1) << 3,
    ELPIS_DSV41_CAP_QUANT_DEQUANT_F32 = UINT64_C(1) << 4,
    ELPIS_DSV41_CAP_ROPE_F32 = UINT64_C(1) << 5,
    ELPIS_DSV41_CAP_CANDIDATE_MASK_F32 = UINT64_C(1) << 6,
    ELPIS_DSV41_CAP_SELECT_POSITIONS_F32 = UINT64_C(1) << 7,
    ELPIS_DSV41_CAP_SPARSE_ATTN_F32 = UINT64_C(1) << 8,
    ELPIS_DSV41_CAP_HC_MIXES_F32 = UINT64_C(1) << 9,
    ELPIS_DSV41_CAP_ROUTE_F32 = UINT64_C(1) << 10,
    ELPIS_DSV41_CAP_EXPERT_F32 = UINT64_C(1) << 11,
    ELPIS_DSV41_CAP_ATTN_STATE_V1 = UINT64_C(1) << 12,
    ELPIS_DSV41_CAP_ENGRAM_GATED_WRITE_F32 = UINT64_C(1) << 13,
    ELPIS_DSV41_CAP_INDEX_SCORES_F32 = UINT64_C(1) << 14,
    ELPIS_DSV41_CAP_GROUPED_OUTPUT_F32 = UINT64_C(1) << 15,
    ELPIS_DSV41_CAP_LOCAL_ATTN_APPLY_F32 = UINT64_C(1) << 16,
    ELPIS_DSV41_CAP_COMPRESSED_ATTN_APPLY_F32 = UINT64_C(1) << 17,
    ELPIS_DSV41_CAP_ROUTE_ORDER_U32 = UINT64_C(1) << 18,
    ELPIS_DSV41_CAP_EXPERT_ACCUMULATE_F32 = UINT64_C(1) << 19,
    ELPIS_DSV41_CAP_LAYER_BEGIN_F32 = UINT64_C(1) << 20,
    ELPIS_DSV41_CAP_LAYER_AFTER_ATTN_ROUTE_F32 = UINT64_C(1) << 21,
    ELPIS_DSV41_CAP_LAYER_FINISH_F32 = UINT64_C(1) << 22
};

typedef enum {
    ELPIS_DSV41_QUANT_LOCAL = 0,
    ELPIS_DSV41_QUANT_COMPRESSED = 1,
    ELPIS_DSV41_QUANT_INDEX = 2
} elpis_dsv41_quant_mode;

typedef enum {
    ELPIS_DSV41_SCORE_SOFTMAX = 0,
    ELPIS_DSV41_SCORE_SIGMOID = 1,
    ELPIS_DSV41_SCORE_SQRT_SOFTPLUS = 2
} elpis_dsv41_score_mode;

/*
 * DSV4.1 native arithmetic ABI R0.
 *
 * All arrays are contiguous IEEE-754 binary32 in host-native byte order.
 * Linear weights are row-major [out_dim, in_dim], matching the frozen
 * tower contract. Callers provide every output buffer. Functions allocate
 * nothing, create no threads, perform no I/O, retain no pointers and have
 * no global mutable state.
 *
 * Input and output regions must not overlap unless a later ABI explicitly
 * states otherwise. Dimensions must be nonzero. The implementation rejects
 * non-finite outputs rather than publishing them.
 */
uint32_t elpis_dsv41_native_abi_version(void);
uint64_t elpis_dsv41_native_capabilities(void);

elpis_dsv41_native_status
elpis_dsv41_linear_f32(const float *x,
                       const float *weight,
                       float *out,
                       size_t in_dim,
                       size_t out_dim);

elpis_dsv41_native_status
elpis_dsv41_rms_f32(const float *x,
                    const float *weight,
                    float eps,
                    float *out,
                    size_t dim);

elpis_dsv41_native_status
elpis_dsv41_hc_pre_f32(const float *stream,
                       const float *pre,
                       float *out,
                       size_t copies,
                       size_t dim);

elpis_dsv41_native_status
elpis_dsv41_hc_post_f32(const float *x,
                        const float *residual,
                        const float *post,
                        const float *comb,
                        float *out,
                        size_t copies,
                        size_t dim);

/* Quantize then dequantize contiguous rows using the frozen donor cache rules.
 * width is the final dimension; rows*width floats are consumed and produced.
 * LOCAL and INDEX require width divisible by 32; COMPRESSED by 16. */
elpis_dsv41_native_status
elpis_dsv41_quant_dequant_f32(const float *x,
                              float *out,
                              size_t rows,
                              size_t width,
                              uint32_t mode);

/* Apply RoPE to the final 2*pairs coordinates of each vector. freq is
 * [pairs,2] as (cos,sin); inverse negates sin. */
elpis_dsv41_native_status
elpis_dsv41_rope_f32(const float *x,
                     const float *freq,
                     float *out,
                     size_t vectors,
                     size_t dim,
                     size_t pairs,
                     int inverse);

/* Candidate block mask: block maxima, newest block pinned, descending stable
 * top-block selection. mask contains width bytes, each exactly 0 or 1. */
elpis_dsv41_native_status
elpis_dsv41_candidate_mask_f32(const float *logits,
                               uint8_t *mask,
                               size_t width,
                               size_t top_blocks,
                               size_t block_size);

/* Stable descending top-k by score (lower position wins ties), returned in
 * ascending position order. +/-infinity is allowed; NaN is rejected. */
elpis_dsv41_native_status
elpis_dsv41_select_positions_f32(const float *scores,
                                 uint32_t *positions,
                                 size_t width,
                                 size_t topk,
                                 size_t *selected);

/* Sink-augmented sparse attention. score_scratch is caller-owned storage for
 * at least kv_count floats; no allocation occurs in the library. */
elpis_dsv41_native_status
elpis_dsv41_sparse_attention_f32(const float *query,
                                 const float *kv,
                                 const float *sink,
                                 float *out,
                                 float *score_scratch,
                                 size_t heads,
                                 size_t kv_count,
                                 size_t dim);


/* Compute donor mHC pre/post/comb from stream [copies,dim]. */
elpis_dsv41_native_status
elpis_dsv41_hc_mixes_f32(const float *stream,
                         const float *fn,
                         const float *scale,
                         const float *base,
                         float *pre,
                         float *post,
                         float *comb,
                         float *mix_scratch,
                         size_t copies,
                         size_t dim,
                         size_t sinkhorn_iters,
                         float norm_eps,
                         float hc_eps);

elpis_dsv41_native_status
elpis_dsv41_route_f32(const float *x,
                      const float *weight,
                      const float *bias,
                      uint32_t *chosen,
                      float *values,
                      float *score_scratch,
                      size_t dim,
                      size_t expert_count,
                      size_t active,
                      uint32_t score_mode,
                      float gate_temp,
                      int normalize_topk,
                      float route_scale);

elpis_dsv41_native_status
elpis_dsv41_expert_f32(const float *x,
                       const float *w1,
                       const float *w3,
                       const float *w2,
                       float route_weight,
                       int weighted,
                       float swiglu_limit,
                       float *gate_scratch,
                       float *up_scratch,
                       float *out,
                       size_t dim,
                       size_t expert_dim);


/*
 * Sequence-local attention cache R3. One state corresponds to one layer of
 * one active sequence. Native code owns all cache storage and retains no caller
 * pointers. No threads, I/O or per-token allocation occur.
 */
elpis_dsv41_native_status
elpis_dsv41_attention_state_create(size_t local_window,
                                   size_t head_dim,
                                   size_t index_dim,
                                   size_t max_tokens,
                                   size_t ratio,
                                   int owner,
                                   elpis_dsv41_attention_state **out);

elpis_dsv41_native_status
elpis_dsv41_attention_state_destroy(elpis_dsv41_attention_state **state);

size_t
elpis_dsv41_attention_state_bytes(const elpis_dsv41_attention_state *state);

size_t
elpis_dsv41_attention_state_count(const elpis_dsv41_attention_state *state);

elpis_dsv41_native_status
elpis_dsv41_attention_local_store_f32(elpis_dsv41_attention_state *state,
                                      size_t position,
                                      const float *local);

elpis_dsv41_native_status
elpis_dsv41_attention_local_copy_f32(const elpis_dsv41_attention_state *state,
                                     size_t position,
                                     float *out,
                                     size_t row_capacity,
                                     size_t *rows);

elpis_dsv41_native_status
elpis_dsv41_attention_compress_push_f32(elpis_dsv41_attention_state *state,
                                        size_t position,
                                        const float *kv,
                                        const float *gate,
                                        float *out_latent,
                                        int *ready);

elpis_dsv41_native_status
elpis_dsv41_attention_publish_group_f32(elpis_dsv41_attention_state *state,
                                        size_t position,
                                        const float *compressed,
                                        const float *index_key);

elpis_dsv41_native_status
elpis_dsv41_attention_index_copy_f32(const elpis_dsv41_attention_state *state,
                                     float *out,
                                     size_t row_capacity,
                                     size_t *rows);

elpis_dsv41_native_status
elpis_dsv41_attention_gather_compressed_f32(
    const elpis_dsv41_attention_state *state,
    const uint32_t *positions,
    size_t selected,
    float *out);


/*
 * Engram gated associative write over already-materialized row values.
 *
 * stream/q_weight/k_weight/out are [copies,dim].
 * rows is a flattened vector of row_values floats.
 * wkv is row-major [((copies+1)*dim), row_values].
 * kv_scratch contains exactly (copies+1)*dim floats and is caller-owned.
 *
 * No row lookup, FMS access, token hashing, allocation, I/O or retained
 * pointer occurs in this function.
 */
elpis_dsv41_native_status
elpis_dsv41_engram_gated_write_f32(const float *stream,
                                   const float *rows,
                                   const float *wkv,
                                   const float *q_weight,
                                   const float *k_weight,
                                   float norm_eps,
                                   float *kv_scratch,
                                   float *out,
                                   size_t copies,
                                   size_t dim,
                                   size_t row_values);


/*
 * DSV4.1 sparse-index score aggregation.
 * iq:         [index_heads,index_dim]
 * index_keys: [count,index_dim]
 * iw:         [index_heads], already including the donor scale factor
 * out:        [count]
 *
 * For each position n:
 *   out[n] = sum_h max(dot(iq[h], index_keys[n]), 0) * iw[h]
 */
elpis_dsv41_native_status
elpis_dsv41_index_scores_f32(const float *iq,
                             const float *index_keys,
                             const float *iw,
                             float *out,
                             size_t index_heads,
                             size_t index_dim,
                             size_t count);

/*
 * DSV4.1 grouped attention output projection.
 * attn_out: [heads,head_dim] after inverse RoPE
 * wo_a:     [o_groups*o_rank, heads*head_dim/o_groups]
 * wo_b:     [dimension, o_groups*o_rank]
 * scratch:  [o_groups*o_rank]
 * out:      [dimension]
 */
elpis_dsv41_native_status
elpis_dsv41_grouped_output_f32(const float *attn_out,
                               const float *wo_a,
                               const float *wo_b,
                               float *scratch,
                               float *out,
                               size_t heads,
                               size_t head_dim,
                               size_t o_groups,
                               size_t o_rank,
                               size_t dimension);


/*
 * Pure-local/SWA Attention.apply orchestration.
 *
 * This is the ratio==0 path only. It composes the already-qualified native
 * linear/RMS/RoPE/local-quant/sparse-attention/grouped-output primitives with
 * one R3 sequence-local attention-state handle.
 *
 * Weight layout matches the frozen DSV4.1 tensor contract:
 *   wq_a    [q_rank, dimension]
 *   q_norm  [q_rank]
 *   wq_b    [heads*head_dim, q_rank]
 *   wkv     [head_dim, dimension]
 *   kv_norm [head_dim]
 *   sink    [heads]
 *   wo_a    [o_groups*o_rank, heads*head_dim/o_groups]
 *   wo_b    [dimension, o_groups*o_rank]
 *   freq    [rope_pairs,2] as (cos,sin)
 *
 * No allocation, I/O, FMS access, Python callback, retained caller pointer or
 * thread creation occurs. scratch is caller-owned and must contain at least
 * elpis_dsv41_local_attention_scratch_floats(...) floats.
 */
size_t
elpis_dsv41_local_attention_scratch_floats(size_t dimension,
                                           size_t q_rank,
                                           size_t heads,
                                           size_t head_dim,
                                           size_t o_groups,
                                           size_t o_rank,
                                           size_t local_window);

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
    size_t local_window);


/*
 * Compressed/global DSV4.1 Attention.apply orchestration for ratio>0 layers.
 *
 * state is the current layer's R3 attention state. owner_state is the current
 * shared KV owner; for kv_owner!=0 it must be identical to state.
 *
 * selected_io/selected_count_io and candidates_io/candidate_count_io are
 * caller-owned sequence-local SharedAttention storage. candidate_mode:
 *   0 = no candidate operation on this index source
 *   1 = this layer is candidate_source and publishes a block mask
 *   2 = downstream index source; apply the published mask before top-k
 *
 * Optional weight pointers are required exactly when their mechanism is active:
 *   kv_owner: compressor_wkv, compressor_norm, indexer_wk, indexer_k_norm
 *   kv_owner && ratio>1: compressor_wgate
 *   index_source: indexer_wq_b, indexer_weights_proj
 *
 * group_freq is required only when a kv_owner completes a compression group.
 *
 * No allocation, file/FMS access, Python callback, retained caller pointer or
 * thread creation occurs. All transient memory is caller-owned scratch.
 */
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
    size_t index_topk);

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
    size_t candidate_block_size);


/*
 * Convert the route result's score order into the donor's accumulation order.
 *
 * chosen[0:active] contains unique expert IDs in router-score order.
 * order[0:active] receives indices into chosen such that
 * chosen[order[0]], ..., chosen[order[active-1]] are ascending expert IDs.
 *
 * No allocation and no expert-count-sized scratch are required.
 */
elpis_dsv41_native_status
elpis_dsv41_route_order_u32(const uint32_t *chosen,
                            uint32_t *order,
                            size_t active);

/*
 * Compute one already-materialized expert and add its output to accumulator.
 *
 * This preserves the production FMS boundary: the caller may materialize one
 * selected expert at a time, invoke this function in route_order order, then
 * invoke it once more for the shared expert with weighted=0.
 *
 * gate_scratch/up_scratch contain expert_dim floats each.
 * expert_out_scratch and accumulator contain dim floats each.
 * accumulator is caller-owned persistent layer scratch and is updated in place.
 */
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
                                  size_t expert_dim);


/*
 * R9 split native Layer.apply frame.
 *
 * The split preserves the two existing external ownership boundaries:
 *
 *   layer_begin
 *     [optional already-materialized Engram rows]
 *       -> Engram gated write
 *       -> attention mHC mixes
 *       -> delayed mHC pre + attention RMS
 *
 *   caller invokes qualified R6/R7 native attention
 *
 *   layer_after_attention_route
 *       -> attention mHC post
 *       -> FFN mHC mixes
 *       -> delayed mHC pre + FFN RMS
 *       -> qualified native route + donor accumulation order
 *
 *   caller materializes one expert at a time and invokes R8 accumulation
 *
 *   layer_finish
 *       -> FFN mHC post
 *
 * No allocation, FMS access, row lookup, Python callback, retained caller
 * pointer or hidden thread is introduced.
 */
size_t
elpis_dsv41_layer_frame_scratch_floats(size_t copies,
                                       size_t dim,
                                       size_t row_values,
                                       size_t expert_count,
                                       int engram_enabled);

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
    size_t dim);

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
    size_t active);

elpis_dsv41_native_status
elpis_dsv41_layer_finish_f32(const float *moe_out,
                             const float *stream_after_attention,
                             const float *fpost,
                             const float *fc,
                             float *stream_out,
                             size_t copies,
                             size_t dim);

#ifdef __cplusplus
}
#endif
#endif
