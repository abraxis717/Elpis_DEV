#ifndef ELPIS_DSV41_STREAM_H
#define ELPIS_DSV41_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DSV4.1 Yielded Token Stream, R0 (YTS-R0).
 *
 * A DSV4.1-adapter protocol carried entirely in payloads of ONE adapter-defined
 * ELPIS_EXEC_BACKEND_ONLY operation of the existing generic execution port.
 * It adds nothing to the port. Vendor-neutral: no device pointer, stream,
 * event, pinned-memory or SDK concept appears here.
 *
 * A token is a chain of strictly alternating submissions:
 *
 *   TOKEN_BEGIN -> reply NEED | COMPLETE
 *   EXPERT_SUPPLY (one part) -> reply NEED | COMPLETE
 *
 * A materialization "yield" is the completed, successful result of a
 * submission (a NEED reply). Between submissions no port token is outstanding;
 * the host reads FMS on its own thread and the provider keeps the token's
 * continuation. The provider names only (layer, expert index); the host maps
 * that to admitted tensor bindings, reads and verifies the bytes through FMS
 * and supplies them with their binding digests. The provider never receives an
 * asset identifier, path, descriptor, offset, page map or authority.
 *
 * Every message is a fixed 128-byte little-endian header followed by a body of
 * exactly header.body_bytes bytes. Replies echo stream_id, epoch, seq,
 * position and manifest_digest. All integers are little-endian; all floats
 * are IEEE-754 binary32 little-endian.
 */

enum {
    ELPIS_DSV41_STREAM_PROTOCOL_VERSION = 1u,
    /* Adapter-defined capability bit (< 64) of the generic execution port. */
    ELPIS_DSV41_STREAM_OPERATION = 41u,
    ELPIS_DSV41_STREAM_HEADER_BYTES = 128u,
    ELPIS_DSV41_STREAM_MAGIC = 0x31535459 /* "YTS1" */
};
#define ELPIS_DSV41_STREAM_NONE 0xFFFFFFFFu /* position/layer/expert not applicable */

/* Message kinds. Requests are < 0x80; replies have bit 7 set. */
enum {
    ELPIS_DSV41_STREAM_MODEL_ADMIT_BEGIN = 0x01u,
    ELPIS_DSV41_STREAM_MODEL_ADMIT_TENSOR = 0x02u,
    ELPIS_DSV41_STREAM_MODEL_ADMIT_END = 0x03u,
    ELPIS_DSV41_STREAM_STREAM_OPEN = 0x04u,
    ELPIS_DSV41_STREAM_TOKEN_BEGIN = 0x05u,
    ELPIS_DSV41_STREAM_EXPERT_SUPPLY = 0x06u,
    ELPIS_DSV41_STREAM_STREAM_RELEASE = 0x07u,
    ELPIS_DSV41_STREAM_MODEL_RELEASE = 0x08u,

    ELPIS_DSV41_STREAM_ADMIT_PART_ACK = 0x81u,
    ELPIS_DSV41_STREAM_ADMIT_ACK = 0x82u,
    ELPIS_DSV41_STREAM_OPENED = 0x83u,
    ELPIS_DSV41_STREAM_NEED = 0x84u,
    ELPIS_DSV41_STREAM_COMPLETE = 0x85u,
    ELPIS_DSV41_STREAM_RELEASED = 0x86u,
    ELPIS_DSV41_STREAM_MODEL_RELEASED = 0x87u
};

/* Feature bits negotiated by MODEL_ADMIT_BEGIN / ADMIT_ACK. */
enum {
    ELPIS_DSV41_STREAM_FEATURE_CACHE = 1u << 0,
    ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS = 1u << 1
};

/* Tensor representation tag carried by every tensor and expert payload. */
enum { ELPIS_DSV41_STREAM_REPR_F32_LE_ROW_MAJOR_OUT_IN = 1u };

/* Tensor roles (MODEL_ADMIT_TENSOR.role_kind). Global roles carry layer=NONE;
 * expert roles carry an expert index where expert_count means the shared expert. */
enum {
    ELPIS_DSV41_ROLE_EMBED = 1, ELPIS_DSV41_ROLE_HEAD = 2, ELPIS_DSV41_ROLE_NORM = 3,
    ELPIS_DSV41_ROLE_ROPE_LOCAL = 4, ELPIS_DSV41_ROLE_ROPE_COMPRESSED = 5,

    ELPIS_DSV41_ROLE_ATTN_NORM = 16, ELPIS_DSV41_ROLE_FFN_NORM = 17,
    ELPIS_DSV41_ROLE_ATTN_WQ_A = 18, ELPIS_DSV41_ROLE_ATTN_Q_NORM = 19,
    ELPIS_DSV41_ROLE_ATTN_WQ_B = 20, ELPIS_DSV41_ROLE_ATTN_WKV = 21,
    ELPIS_DSV41_ROLE_ATTN_KV_NORM = 22, ELPIS_DSV41_ROLE_ATTN_SINK = 23,
    ELPIS_DSV41_ROLE_ATTN_WO_A = 24, ELPIS_DSV41_ROLE_ATTN_WO_B = 25,
    ELPIS_DSV41_ROLE_GATE_WEIGHT = 26, ELPIS_DSV41_ROLE_GATE_BIAS = 27,
    ELPIS_DSV41_ROLE_HC_ATTN_FN = 28, ELPIS_DSV41_ROLE_HC_ATTN_BASE = 29,
    ELPIS_DSV41_ROLE_HC_ATTN_SCALE = 30, ELPIS_DSV41_ROLE_HC_FFN_FN = 31,
    ELPIS_DSV41_ROLE_HC_FFN_BASE = 32, ELPIS_DSV41_ROLE_HC_FFN_SCALE = 33,
    ELPIS_DSV41_ROLE_COMP_WKV = 34, ELPIS_DSV41_ROLE_COMP_NORM = 35,
    ELPIS_DSV41_ROLE_COMP_WGATE = 36, ELPIS_DSV41_ROLE_IDX_WK = 37,
    ELPIS_DSV41_ROLE_IDX_K_NORM = 38, ELPIS_DSV41_ROLE_IDX_WQ_B = 39,
    ELPIS_DSV41_ROLE_IDX_WEIGHTS_PROJ = 40, ELPIS_DSV41_ROLE_ENGRAM_WKV = 41,
    ELPIS_DSV41_ROLE_ENGRAM_Q = 42, ELPIS_DSV41_ROLE_ENGRAM_K = 43,

    ELPIS_DSV41_ROLE_EXPERT_W1 = 64, ELPIS_DSV41_ROLE_EXPERT_W3 = 65,
    ELPIS_DSV41_ROLE_EXPERT_W2 = 66
};

/* Per-layer flags in MODEL_ADMIT_BEGIN. */
enum {
    ELPIS_DSV41_LAYER_KV_OWNER = 1u << 0,
    ELPIS_DSV41_LAYER_INDEX_SOURCE = 1u << 1,
    ELPIS_DSV41_LAYER_ENGRAM = 1u << 2
};

typedef struct {
    uint32_t magic;               /*   0 */
    uint16_t version;             /*   4 */
    uint16_t kind;                /*   6 */
    uint32_t header_bytes;        /*   8: 128 */
    uint32_t flags;               /*  12: zero in R0 */
    uint64_t stream_id;           /*  16: host-assigned, never reused; 0 = model level */
    uint64_t epoch;               /*  24: host nonce chosen at STREAM_OPEN */
    uint64_t seq;                 /*  32: host-assigned, +1 per request on the runtime */
    uint32_t position;            /*  40: token position or NONE */
    uint32_t layer;               /*  44: layer or NONE */
    uint64_t body_bytes;          /*  48 */
    uint8_t manifest_digest[32];  /*  56: admitted ParameterManifest digest */
    uint8_t reserved[40];         /*  88: zero */
} elpis_dsv41_stream_header;      /* 128 */

/*
 * Bodies (packed little-endian, no padding beyond what is listed):
 *
 * MODEL_ADMIT_BEGIN
 *   u32 vocab, dimension, layers, heads, head_dim, rope_dim, q_rank, o_groups,
 *       o_rank, local_window, max_tokens, index_heads, index_dim, index_topk,
 *       expert_count, active_experts, expert_dim, engram_order, engram_heads,
 *       engram_dim, hc_mult, hc_sinkhorn_iters, score_mode, norm_topk_prob,
 *       candidate_blocks, candidate_block_size, hash_columns            (27 x u32)
 *   i32 candidate_source
 *   f32 norm_eps, hc_eps, gate_temp, route_scale, swiglu_limit
 *   u32 requested_features, max_streams
 *   u64 part_bytes, expert_cache_bytes
 *   u8  config_digest[32]
 *   u32 compress_ratio[layers], layer_flags[layers]
 *
 * MODEL_ADMIT_TENSOR
 *   u32 role_kind, layer, expert, representation, ndim, dims[4]
 *   u8  binding_digest[32]
 *   u64 tensor_bytes, part_offset, part_len
 *   u8  data[part_len]
 *
 * MODEL_ADMIT_END      u32 tensor_count, u32 reserved
 * ADMIT_PART_ACK       (empty)
 * ADMIT_ACK            u8 provider_profile_digest[32]; u32 features_granted, reserved;
 *                      u64 reserved_bytes
 * STREAM_OPEN          u32 max_tokens, flags (requested observation features)
 * OPENED               u64 state_bytes_reserved
 *
 * TOKEN_BEGIN
 *   u32 token, engram_count
 *   engram_count x { u32 layer, rows, row_dim, reserved; f32 values[rows*row_dim] }
 *     (exactly the admitted Engram layers, ascending; decoded host-side rows)
 *
 * EXPERT_SUPPLY
 *   u32 layer, expert (expert_count = shared), cursor_index, representation
 *   u64 image_bytes, part_offset, part_len, reserved
 *   u8  binding_digest[3][32]  (w1, w3, w2)
 *   u8  data[part_len]          bytes [part_offset, part_offset+part_len) of
 *                               the canonical image w1 || w3 || w2
 *
 * NEED
 *   u32 layer, k, selected[k] (router-score order), need_count,
 *       need[need_count] (ascending routed IDs, shared = expert_count last),
 *       cursor_index, reserved
 *   u64 cursor_offset, cache_hits (bit j: ordered expert j served from cache;
 *       ordered = ascending selected, then shared)
 *
 * COMPLETE
 *   u32 vocab, layers, k, index_topk, flags (granted observation), reserved
 *   u64 cache_bytes_in_use, cache_entries, cache_evictions, cache_hits,
 *       expert_slot_high_water, supplied_experts
 *   f32 logits[vocab]
 *   layers x { u32 selected[k], elided, attention_count, positions_count,
 *              positions[index_topk] }
 *   f32 layer_streams[layers*hc_mult*dimension]   (only with OBSERVE granted)
 *
 * STREAM_RELEASE / MODEL_RELEASE / RELEASED / MODEL_RELEASED  (empty)
 */
enum {
    ELPIS_DSV41_STREAM_ADMIT_BEGIN_FIXED_BYTES = 27u * 4u + 4u + 5u * 4u + 2u * 4u + 2u * 8u + 32u,
    ELPIS_DSV41_STREAM_ADMIT_TENSOR_PREFIX_BYTES = 9u * 4u + 32u + 3u * 8u,
    ELPIS_DSV41_STREAM_SUPPLY_PREFIX_BYTES = 4u * 4u + 4u * 8u + 3u * 32u
};

/*
 * Provider attachment, DSV4.1-stream specific and deliberately narrow.
 *
 * Elpis seals and loads every native object with RTLD_LOCAL. A provider must
 * therefore never link its own copy of the execution runtime (execution.h:
 * private copies are unsupported). The host passes the runtime's buffer and
 * notify entry points instead. The provider fills a standard
 * elpis_exec_backend whose context it owns until detach.
 *
 * Lifecycle: attach -> elpis_exec_create(config.backend = filled backend)
 * -> bind_runtime -> submissions -> elpis_exec_shutdown/destroy (runs
 * backend.shutdown, which releases every model, stream, slot and cache entry)
 * -> detach (frees the context; also valid after a failed create).
 */
enum { ELPIS_DSV41_STREAM_PROVIDER_ABI_V1 = 1u };

typedef struct {
    uint32_t abi_version; /* ELPIS_DSV41_STREAM_PROVIDER_ABI_V1 */
    uint32_t reserved;
    elpis_exec_buffer *(*buffer_alloc)(size_t bytes);
    void *(*buffer_mutable_data)(elpis_exec_buffer *buffer);
    const void *(*buffer_data)(const elpis_exec_buffer *buffer);
    size_t (*buffer_size)(const elpis_exec_buffer *buffer);
    void (*buffer_release)(elpis_exec_buffer *buffer);
    void (*notify)(elpis_exec_runtime *runtime);
} elpis_dsv41_stream_host_v1;

/* Exported by a provider object. Return 0 on success. */
uint32_t elpis_dsv41_stream_provider_abi_version(void);
int elpis_dsv41_stream_provider_attach(const elpis_dsv41_stream_host_v1 *host,
                                       elpis_exec_backend *out);
int elpis_dsv41_stream_provider_bind_runtime(void *context, elpis_exec_runtime *runtime);
void elpis_dsv41_stream_provider_detach(void *context);

#ifdef __cplusplus
}
#endif
#endif
