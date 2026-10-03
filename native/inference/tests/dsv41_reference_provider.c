/*
 * TEST-ONLY YTS-R0 reference provider for DSV4.1.
 *
 * It implements the vendor-neutral DSV4.1 Yielded Token Stream (dsv41_stream.h)
 * behind the real elpis_exec_backend port, using the already-qualified CPU
 * DSV4.1 kernels in exactly the order DSV41NativeBackend.apply_layer uses them.
 * It exists to show that YTS-R0 preserves semantics through the generic port;
 * it is not a production accelerator provider and ships in no production target.
 *
 * Test controls: synchronous mode (work runs inside submit) or a fake
 * asynchronous "device" thread that optionally calls the host's notify;
 * fault injection by request index and kind; live-resource counters.
 */
#define _POSIX_C_SOURCE 200809L

#include "elpis/dsv41_native.h"
#include "elpis/dsv41_stream.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the YTS-R0 reference provider assumes a little-endian host"
#endif

#if defined(__GNUC__)
#define YTS_EXPORT __attribute__((visibility("default")))
#else
#define YTS_EXPORT
#endif

/* ------------------------------------------------------------------------- */
/* Test-only control surface                                                  */
/* ------------------------------------------------------------------------- */

enum {
    YTS_MODE_SYNC = 0,
    YTS_MODE_DEVICE = 1
};

enum {
    YTS_FAULT_NONE = 0,
    YTS_FAULT_SUBMIT_REJECT = 1, /* refuse at submit, before any processing */
    YTS_FAULT_POLL_FAIL = 2,     /* process (state advances), then fail at poll */
    YTS_FAULT_POLL_TIMEOUT = 3,  /* process, then never report completion */
    YTS_FAULT_BAD_ECHO = 4,      /* successful reply with a wrong seq echo */
    YTS_FAULT_BAD_NEED = 5,      /* NEED with an out-of-range selected expert */
    YTS_FAULT_NONFINITE = 6,     /* COMPLETE with a NaN logit */
    YTS_FAULT_BAD_CURSOR = 7,    /* NEED with an advanced cursor */
    YTS_FAULT_SLOW = 8           /* device mode: only this request takes device_delay_us */
};

typedef struct {
    int32_t mode;
    int32_t notify;
    uint32_t device_delay_us;
    uint32_t fault_kind;
    int64_t fault_message; /* request index on one context; -1 none */
} yts_ref_config;

typedef struct {
    /* live objects */
    int64_t contexts, models, streams, slots, cache_entries, cache_bytes, tokens, device_threads;
    /* cumulative */
    uint64_t messages, bytes_in, bytes_out, aborts, cache_hits, supplied_experts,
        slot_high_water, faults_fired, cache_evictions;
    /* device mode: delay from device completion to the poll that observes it */
    uint64_t observed, observe_ns, observe_max_ns;
} yts_ref_counters;

static pthread_mutex_t g_config_mu = PTHREAD_MUTEX_INITIALIZER;
static yts_ref_config g_config = {YTS_MODE_SYNC, 0, 0, YTS_FAULT_NONE, -1};

static atomic_llong c_contexts, c_models, c_streams, c_slots, c_cache_entries, c_cache_bytes,
    c_tokens, c_device_threads;
static atomic_ullong c_messages, c_bytes_in, c_bytes_out, c_aborts, c_cache_hits, c_supplied,
    c_slot_high_water, c_faults, c_evictions, c_observed, c_observe_ns, c_observe_max_ns;

YTS_EXPORT void elpis_dsv41_reference_provider_configure(const yts_ref_config *config) {
    pthread_mutex_lock(&g_config_mu);
    if (config) g_config = *config;
    pthread_mutex_unlock(&g_config_mu);
}

YTS_EXPORT void elpis_dsv41_reference_provider_counters(yts_ref_counters *out) {
    if (!out) return;
    out->contexts = atomic_load(&c_contexts);
    out->models = atomic_load(&c_models);
    out->streams = atomic_load(&c_streams);
    out->slots = atomic_load(&c_slots);
    out->cache_entries = atomic_load(&c_cache_entries);
    out->cache_bytes = atomic_load(&c_cache_bytes);
    out->tokens = atomic_load(&c_tokens);
    out->device_threads = atomic_load(&c_device_threads);
    out->messages = atomic_load(&c_messages);
    out->bytes_in = atomic_load(&c_bytes_in);
    out->bytes_out = atomic_load(&c_bytes_out);
    out->aborts = atomic_load(&c_aborts);
    out->cache_hits = atomic_load(&c_cache_hits);
    out->supplied_experts = atomic_load(&c_supplied);
    out->slot_high_water = atomic_load(&c_slot_high_water);
    out->faults_fired = atomic_load(&c_faults);
    out->cache_evictions = atomic_load(&c_evictions);
    out->observed = atomic_load(&c_observed);
    out->observe_ns = atomic_load(&c_observe_ns);
    out->observe_max_ns = atomic_load(&c_observe_max_ns);
}

YTS_EXPORT void elpis_dsv41_reference_provider_reset_cumulative(void) {
    atomic_store(&c_messages, 0);
    atomic_store(&c_bytes_in, 0);
    atomic_store(&c_bytes_out, 0);
    atomic_store(&c_aborts, 0);
    atomic_store(&c_cache_hits, 0);
    atomic_store(&c_supplied, 0);
    atomic_store(&c_slot_high_water, 0);
    atomic_store(&c_faults, 0);
    atomic_store(&c_evictions, 0);
    atomic_store(&c_observed, 0);
    atomic_store(&c_observe_ns, 0);
    atomic_store(&c_observe_max_ns, 0);
}

/* A fixed identity of this provider's kernel profile:
 * SHA-256("elpis.dsv41.stream.reference-provider.cpu-native-kernels.v1"). */
static const uint8_t PROFILE_DIGEST[32] = {
    0x46, 0x26, 0x7c, 0xbe, 0xa3, 0x9f, 0xc8, 0xe3, 0x62, 0xeb, 0x15, 0xb0, 0xe2, 0x72, 0xb8, 0xc5,
    0x9b, 0x70, 0xb8, 0x13, 0x88, 0x40, 0x03, 0x35, 0x40, 0xa3, 0x05, 0x64, 0xf7, 0xe0, 0x04, 0x5f};

/* ------------------------------------------------------------------------- */
/* Byte readers/writers (little-endian, unaligned-safe)                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    const uint8_t *p;
    size_t n, at;
    int bad;
} rd;

static void rd_bytes(rd *r, void *out, size_t n) {
    if (r->bad || n > r->n - r->at) {
        r->bad = 1;
        if (out) memset(out, 0, n);
        return;
    }
    if (out) memcpy(out, r->p + r->at, n);
    r->at += n;
}
static uint32_t rd_u32(rd *r) { uint32_t v; rd_bytes(r, &v, 4); return v; }
static int32_t rd_i32(rd *r) { int32_t v; rd_bytes(r, &v, 4); return v; }
static float rd_f32(rd *r) { float v; rd_bytes(r, &v, 4); return v; }
static uint64_t rd_u64(rd *r) { uint64_t v; rd_bytes(r, &v, 8); return v; }
static const uint8_t *rd_view(rd *r, size_t n) {
    if (r->bad || n > r->n - r->at) { r->bad = 1; return NULL; }
    const uint8_t *v = r->p + r->at;
    r->at += n;
    return v;
}

typedef struct {
    uint8_t *p;
    size_t at;
} wr;
static void wr_bytes(wr *w, const void *v, size_t n) { memcpy(w->p + w->at, v, n); w->at += n; }
static void wr_u32(wr *w, uint32_t v) { wr_bytes(w, &v, 4); }
static void wr_u64(wr *w, uint64_t v) { wr_bytes(w, &v, 8); }

static int mul_ok(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Model, cache, stream                                                       */
/* ------------------------------------------------------------------------- */

enum { LAYER_ROLES = 28, GLOBAL_ROLES = 6 };

typedef struct {
    float *data;
    size_t bytes, received;
    uint32_t ndim, dims[4];
    int complete;
} tensor;

typedef struct {
    float *data;
    size_t bytes;
    uint8_t digests[3][32];
    uint64_t last_use;
    int pinned;
} cache_entry;

typedef struct {
    /* geometry */
    size_t vocab, dim, layers, heads, head_dim, rope_dim, q_rank, o_groups, o_rank, local_window,
        max_tokens, index_heads, index_dim, index_topk, experts, active, expert_dim, engram_order,
        engram_heads, engram_dim, hc, sinkhorn, hash_columns, candidate_blocks, candidate_block_size;
    uint32_t score_mode;
    int norm_topk;
    int32_t candidate_source;
    float norm_eps, hc_eps, gate_temp, route_scale, swiglu_limit;
    uint32_t *ratio, *flags;
    uint32_t features, granted;
    uint64_t part_bytes, cache_budget;
    uint8_t manifest[32], config[32];
    /* tensors */
    tensor global[GLOBAL_ROLES];
    tensor *layer_role;  /* layers * LAYER_ROLES */
    tensor *expert_role; /* layers * (experts+1) * 3 */
    uint8_t *resident;   /* layers * (experts+1) */
    size_t tensor_bytes;
    uint32_t tensors_received;
    /* one bounded expert slot */
    float *slot;
    size_t image_bytes;
    /* optional cache */
    cache_entry **cache; /* layers * (experts+1) */
    size_t cache_bytes, cache_count;
    uint64_t cache_clock;
    int ready;
} model;

typedef struct {
    uint64_t id, epoch;
    uint32_t position;
    int observe;
    float *conditioning; /* frozen per-stream turn conditioning [dim], or NULL */
    elpis_dsv41_attention_state **attn;
    /* token continuation */
    int in_token, phase;
    uint32_t layer;
    float *cur, *pre, *stream_work, *ap, *apost, *ac, *attn_x, *attn_out, *stream_after, *fp, *fpost,
        *fc, *moe_x, *moe_out, *gate, *up, *expert_out, *stream_out, *route_values, *head_pre,
        *head_hidden, *logits, *layer_streams, *engram_rows, *frame_scratch, *local_scratch,
        *comp_scratch;
    size_t frame_floats, local_floats, comp_floats;
    uint32_t *chosen, *route_order;
    uint32_t *ordered, *need, *src;
    float *weights;
    cache_entry **pins;
    uint32_t need_count, cursor, j;
    uint64_t hits_mask;
    size_t slot_fill;
    int slot_expert_valid;
    uint32_t slot_expert;
    uint8_t slot_digests[3][32];
    int layer_needed;
    /* token-local shared attention */
    elpis_dsv41_attention_state *owner;
    uint32_t *shared_selected;
    size_t shared_selected_count;
    int has_selected;
    uint8_t *shared_candidates;
    size_t shared_candidate_count;
    int has_candidates;
    uint32_t *selected_buf;
    uint8_t *candidates_buf;
    /* trace */
    uint32_t *trace_selected, *trace_elided, *trace_count, *trace_positions, *trace_npositions;
    /* telemetry for the current token */
    uint64_t token_hits, token_supplied;
} stream;

enum { SRC_NEED = 0, SRC_RESIDENT = 1, SRC_CACHE = 2 };

typedef struct yts_token {
    struct yts_token *next;
    const elpis_exec_buffer *input;
    size_t max_output;
    elpis_exec_buffer *reply;
    elpis_exec_status status;
    uint32_t fault;
    int done, processing, aborted, never_done, device;
    uint64_t done_ns;
} yts_token;

typedef struct {
    elpis_dsv41_stream_host_v1 host;
    yts_ref_config config;
    pthread_mutex_t mu;
    pthread_cond_t device_cv, done_cv;
    pthread_t device;
    int device_started, stop;
    yts_token *queue_head, *queue_tail;
    elpis_exec_runtime *runtime;
    int64_t message_index;
    uint64_t last_seq, last_stream_id;
    int has_manifest;
    uint8_t manifest[32];
    model *m;
    stream *s;
    int released; /* shutdown released every resource */
} context;

static uint64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void note_high_water(size_t bytes) {
    unsigned long long prev = atomic_load(&c_slot_high_water);
    while (bytes > prev && !atomic_compare_exchange_weak(&c_slot_high_water, &prev, bytes)) {}
}

static void cache_free_entry(model *m, size_t at) {
    cache_entry *e = m->cache[at];
    if (!e) return;
    m->cache_bytes -= e->bytes;
    m->cache_count -= 1;
    atomic_fetch_sub(&c_cache_bytes, (long long)e->bytes);
    atomic_fetch_sub(&c_cache_entries, 1);
    free(e->data);
    free(e);
    m->cache[at] = NULL;
}

static void stream_free(stream *s, const model *m) {
    if (!s) return;
    if (s->attn) {
        for (size_t i = 0; i < m->layers; ++i)
            if (s->attn[i]) elpis_dsv41_attention_state_destroy(&s->attn[i]);
        free(s->attn);
    }
    float *floats[] = {s->cur, s->pre, s->stream_work, s->ap, s->apost, s->ac, s->attn_x, s->attn_out,
                       s->stream_after, s->fp, s->fpost, s->fc, s->moe_x, s->moe_out, s->gate, s->up,
                       s->expert_out, s->stream_out, s->route_values, s->head_pre, s->head_hidden,
                       s->logits, s->layer_streams, s->engram_rows, s->frame_scratch, s->local_scratch,
                       s->comp_scratch, s->weights, s->conditioning};
    for (size_t i = 0; i < sizeof(floats) / sizeof(floats[0]); ++i) free(floats[i]);
    uint32_t *words[] = {s->chosen, s->route_order, s->ordered, s->need, s->src, s->shared_selected,
                         s->selected_buf, s->trace_selected, s->trace_elided, s->trace_count,
                         s->trace_positions, s->trace_npositions};
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); ++i) free(words[i]);
    if (s->pins) {
        for (size_t i = 0; i <= m->active; ++i)
            if (s->pins[i]) s->pins[i]->pinned = 0;
        free(s->pins);
    }
    free(s->shared_candidates);
    free(s->candidates_buf);
    free(s);
    atomic_fetch_sub(&c_streams, 1);
}

static void model_free(model *m) {
    if (!m) return;
    for (size_t i = 0; i < GLOBAL_ROLES; ++i) free(m->global[i].data);
    if (m->layer_role) {
        for (size_t i = 0; i < m->layers * LAYER_ROLES; ++i) free(m->layer_role[i].data);
        free(m->layer_role);
    }
    if (m->expert_role) {
        for (size_t i = 0; i < m->layers * (m->experts + 1) * 3; ++i) free(m->expert_role[i].data);
        free(m->expert_role);
    }
    if (m->cache) {
        for (size_t i = 0; i < m->layers * (m->experts + 1); ++i) cache_free_entry(m, i);
        free(m->cache);
    }
    if (m->slot) {
        free(m->slot);
        atomic_fetch_sub(&c_slots, 1);
    }
    free(m->resident);
    free(m->ratio);
    free(m->flags);
    free(m);
    atomic_fetch_sub(&c_models, 1);
}

/* ------------------------------------------------------------------------- */
/* Tensor roles and expected geometry (mirrors config.tensor_shapes)          */
/* ------------------------------------------------------------------------- */

static tensor *role_slot(model *m, uint32_t kind, uint32_t layer, uint32_t expert) {
    if (kind >= ELPIS_DSV41_ROLE_EMBED && kind <= ELPIS_DSV41_ROLE_ROPE_COMPRESSED)
        return layer == ELPIS_DSV41_STREAM_NONE && expert == ELPIS_DSV41_STREAM_NONE ? &m->global[kind] : NULL;
    if (layer >= m->layers) return NULL;
    if (kind >= ELPIS_DSV41_ROLE_ATTN_NORM && kind <= ELPIS_DSV41_ROLE_ENGRAM_K)
        return expert == ELPIS_DSV41_STREAM_NONE ? &m->layer_role[layer * LAYER_ROLES + (kind - 16)] : NULL;
    if (kind >= ELPIS_DSV41_ROLE_EXPERT_W1 && kind <= ELPIS_DSV41_ROLE_EXPERT_W2 && expert <= m->experts)
        return &m->expert_role[(layer * (m->experts + 1) + expert) * 3 + (kind - 64)];
    return NULL;
}

/* Expected shape; returns 0 if the role does not exist for this geometry. */
static int expected_shape(const model *m, uint32_t kind, uint32_t layer, size_t dims[4], uint32_t *ndim) {
    size_t d = m->dim, hc = m->hc;
    uint32_t f = layer < m->layers ? m->flags[layer] : 0;
    uint32_t ratio = layer < m->layers ? m->ratio[layer] : 0;
    *ndim = 1;
    dims[1] = dims[2] = dims[3] = 0;
#define S1(a) do { *ndim = 1; dims[0] = (a); return 1; } while (0)
#define S2(a, b) do { *ndim = 2; dims[0] = (a); dims[1] = (b); return 1; } while (0)
    switch (kind) {
    case ELPIS_DSV41_ROLE_EMBED: S2(m->vocab, d);
    case ELPIS_DSV41_ROLE_HEAD: S2(m->vocab, d);
    case ELPIS_DSV41_ROLE_NORM: S1(d);
    case ELPIS_DSV41_ROLE_ROPE_LOCAL:
    case ELPIS_DSV41_ROLE_ROPE_COMPRESSED:
        *ndim = 3; dims[0] = m->max_tokens; dims[1] = m->rope_dim / 2; dims[2] = 2; return 1;
    case ELPIS_DSV41_ROLE_ATTN_NORM: S1(d);
    case ELPIS_DSV41_ROLE_FFN_NORM: S1(d);
    case ELPIS_DSV41_ROLE_ATTN_WQ_A: S2(m->q_rank, d);
    case ELPIS_DSV41_ROLE_ATTN_Q_NORM: S1(m->q_rank);
    case ELPIS_DSV41_ROLE_ATTN_WQ_B: S2(m->heads * m->head_dim, m->q_rank);
    case ELPIS_DSV41_ROLE_ATTN_WKV: S2(m->head_dim, d);
    case ELPIS_DSV41_ROLE_ATTN_KV_NORM: S1(m->head_dim);
    case ELPIS_DSV41_ROLE_ATTN_SINK: S1(m->heads);
    case ELPIS_DSV41_ROLE_ATTN_WO_A: S2(m->o_groups * m->o_rank, m->heads * m->head_dim / m->o_groups);
    case ELPIS_DSV41_ROLE_ATTN_WO_B: S2(d, m->o_groups * m->o_rank);
    case ELPIS_DSV41_ROLE_GATE_WEIGHT: S2(m->experts, d);
    case ELPIS_DSV41_ROLE_GATE_BIAS: S1(m->experts);
    case ELPIS_DSV41_ROLE_HC_ATTN_FN:
    case ELPIS_DSV41_ROLE_HC_FFN_FN: S2((2 + hc) * hc, hc * d);
    case ELPIS_DSV41_ROLE_HC_ATTN_BASE:
    case ELPIS_DSV41_ROLE_HC_FFN_BASE: S1((2 + hc) * hc);
    case ELPIS_DSV41_ROLE_HC_ATTN_SCALE:
    case ELPIS_DSV41_ROLE_HC_FFN_SCALE: S1(3);
    case ELPIS_DSV41_ROLE_COMP_WKV: if (!(f & ELPIS_DSV41_LAYER_KV_OWNER)) return 0; S2(m->head_dim, d);
    case ELPIS_DSV41_ROLE_COMP_NORM: if (!(f & ELPIS_DSV41_LAYER_KV_OWNER)) return 0; S1(m->head_dim);
    case ELPIS_DSV41_ROLE_COMP_WGATE:
        if (!(f & ELPIS_DSV41_LAYER_KV_OWNER) || ratio <= 1) return 0;
        S2(m->head_dim, d);
    case ELPIS_DSV41_ROLE_IDX_WK: if (!(f & ELPIS_DSV41_LAYER_KV_OWNER)) return 0; S2(m->index_dim, m->head_dim);
    case ELPIS_DSV41_ROLE_IDX_K_NORM: if (!(f & ELPIS_DSV41_LAYER_KV_OWNER)) return 0; S1(m->index_dim);
    case ELPIS_DSV41_ROLE_IDX_WQ_B:
        if (!(f & ELPIS_DSV41_LAYER_INDEX_SOURCE)) return 0;
        S2(m->index_heads * m->index_dim, m->q_rank);
    case ELPIS_DSV41_ROLE_IDX_WEIGHTS_PROJ:
        if (!(f & ELPIS_DSV41_LAYER_INDEX_SOURCE)) return 0;
        S2(m->index_heads, d);
    case ELPIS_DSV41_ROLE_ENGRAM_WKV:
        if (!(f & ELPIS_DSV41_LAYER_ENGRAM)) return 0;
        S2((hc + 1) * d, m->hash_columns * m->engram_dim);
    case ELPIS_DSV41_ROLE_ENGRAM_Q:
    case ELPIS_DSV41_ROLE_ENGRAM_K:
        if (!(f & ELPIS_DSV41_LAYER_ENGRAM)) return 0;
        S2(hc, d);
    case ELPIS_DSV41_ROLE_EXPERT_W1:
    case ELPIS_DSV41_ROLE_EXPERT_W3: S2(m->expert_dim, d);
    case ELPIS_DSV41_ROLE_EXPERT_W2: S2(d, m->expert_dim);
    default: return 0;
    }
#undef S1
#undef S2
}

static const float *W(const model *m, uint32_t kind, uint32_t layer) {
    const tensor *t = role_slot((model *)m, kind, layer, ELPIS_DSV41_STREAM_NONE);
    return t && t->complete ? t->data : NULL;
}
static const float *G(const model *m, uint32_t kind) {
    return m->global[kind].complete ? m->global[kind].data : NULL;
}

/* ------------------------------------------------------------------------- */
/* Reply construction                                                         */
/* ------------------------------------------------------------------------- */

typedef struct {
    elpis_exec_status status;
    elpis_exec_buffer *reply;
} outcome;

static uint8_t *begin_reply(context *c, const elpis_dsv41_stream_header *in, uint16_t kind, uint32_t position,
                            uint32_t layer, size_t body, size_t max_output, outcome *o) {
    size_t total = ELPIS_DSV41_STREAM_HEADER_BYTES + body;
    if (total > max_output) { o->status = ELPIS_EXEC_INTERNAL; return NULL; }
    o->reply = c->host.buffer_alloc(total);
    if (!o->reply) { o->status = ELPIS_EXEC_INTERNAL; return NULL; }
    uint8_t *p = c->host.buffer_mutable_data(o->reply);
    elpis_dsv41_stream_header h;
    memset(&h, 0, sizeof(h));
    h.magic = ELPIS_DSV41_STREAM_MAGIC;
    h.version = ELPIS_DSV41_STREAM_PROTOCOL_VERSION;
    h.kind = kind;
    h.header_bytes = ELPIS_DSV41_STREAM_HEADER_BYTES;
    h.stream_id = in->stream_id;
    h.epoch = in->epoch;
    h.seq = in->seq;
    h.position = position;
    h.layer = layer;
    h.body_bytes = body;
    memcpy(h.manifest_digest, in->manifest_digest, 32);
    memcpy(p, &h, sizeof(h));
    o->status = ELPIS_EXEC_OK;
    return p + ELPIS_DSV41_STREAM_HEADER_BYTES;
}

static elpis_exec_status fail(outcome *o, context *c, elpis_exec_status code) {
    if (o->reply) c->host.buffer_release(o->reply);
    o->reply = NULL;
    o->status = code;
    return code;
}

/* ------------------------------------------------------------------------- */
/* Admission                                                                  */
/* ------------------------------------------------------------------------- */

static elpis_exec_status admit_begin(context *c, const elpis_dsv41_stream_header *h, rd *r) {
    if (c->m) return ELPIS_EXEC_INVALID;
    model *m = calloc(1, sizeof(*m));
    if (!m) return ELPIS_EXEC_INTERNAL;
    atomic_fetch_add(&c_models, 1);
    uint32_t u[27];
    for (size_t i = 0; i < 27; ++i) u[i] = rd_u32(r);
    m->candidate_source = rd_i32(r);
    m->norm_eps = rd_f32(r);
    m->hc_eps = rd_f32(r);
    m->gate_temp = rd_f32(r);
    m->route_scale = rd_f32(r);
    m->swiglu_limit = rd_f32(r);
    m->features = rd_u32(r);
    uint32_t max_streams = rd_u32(r);
    m->part_bytes = rd_u64(r);
    m->cache_budget = rd_u64(r);
    rd_bytes(r, m->config, 32);
    m->vocab = u[0]; m->dim = u[1]; m->layers = u[2]; m->heads = u[3]; m->head_dim = u[4];
    m->rope_dim = u[5]; m->q_rank = u[6]; m->o_groups = u[7]; m->o_rank = u[8];
    m->local_window = u[9]; m->max_tokens = u[10]; m->index_heads = u[11]; m->index_dim = u[12];
    m->index_topk = u[13]; m->experts = u[14]; m->active = u[15]; m->expert_dim = u[16];
    m->engram_order = u[17]; m->engram_heads = u[18]; m->engram_dim = u[19]; m->hc = u[20];
    m->sinkhorn = u[21]; m->score_mode = u[22]; m->norm_topk = (int)u[23];
    m->candidate_blocks = u[24]; m->candidate_block_size = u[25]; m->hash_columns = u[26];
    int ok = !r->bad && max_streams == 1 && m->layers >= 1 && m->layers <= 4096;
    for (size_t i = 0; i < 22 && ok; ++i)
        if (i != 17 && u[i] == 0) ok = 0; /* every count except engram_order checked below */
    ok = ok && m->engram_order >= 2 && m->rope_dim % 2 == 0 && m->heads % m->o_groups == 0 &&
         m->active <= m->experts && m->active + 1 <= 64 && m->score_mode <= 2 && m->norm_topk <= 1 &&
         m->hash_columns == (m->engram_order - 1) * m->engram_heads && m->hc <= 64 &&
         m->candidate_source >= -1 && m->candidate_source < (int32_t)m->layers;
    if (ok) {
        m->ratio = calloc(m->layers, sizeof(uint32_t));
        m->flags = calloc(m->layers, sizeof(uint32_t));
        ok = m->ratio && m->flags;
    }
    for (size_t i = 0; ok && i < m->layers; ++i) m->ratio[i] = rd_u32(r);
    for (size_t i = 0; ok && i < m->layers; ++i) m->flags[i] = rd_u32(r);
    ok = ok && !r->bad && r->at == r->n;
    for (size_t i = 0; ok && i < m->layers; ++i) {
        if (m->ratio[i] > m->max_tokens || (m->flags[i] & ~7u)) ok = 0;
        if (!m->ratio[i] && (m->flags[i] & (ELPIS_DSV41_LAYER_KV_OWNER | ELPIS_DSV41_LAYER_INDEX_SOURCE))) ok = 0;
    }
    size_t layer_slots = 0, expert_slots = 0;
    ok = ok && mul_ok(m->layers, LAYER_ROLES, &layer_slots) &&
         mul_ok(m->layers, (m->experts + 1) * 3, &expert_slots);
    if (ok) {
        m->layer_role = calloc(layer_slots, sizeof(tensor));
        m->expert_role = calloc(expert_slots, sizeof(tensor));
        m->resident = calloc(m->layers * (m->experts + 1), 1);
        m->cache = calloc(m->layers * (m->experts + 1), sizeof(cache_entry *));
        ok = m->layer_role && m->expert_role && m->resident && m->cache;
    }
    if (!ok) { model_free(m); return ELPIS_EXEC_INVALID; }
    m->granted = m->features & (ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS |
                                ELPIS_DSV41_STREAM_FEATURE_CONDITIONING |
                                (m->cache_budget ? ELPIS_DSV41_STREAM_FEATURE_CACHE : 0u));
    memcpy(m->manifest, h->manifest_digest, 32);
    c->m = m;
    return ELPIS_EXEC_OK;
}

static elpis_exec_status admit_tensor(context *c, rd *r) {
    model *m = c->m;
    if (!m || m->ready) return ELPIS_EXEC_INVALID;
    uint32_t kind = rd_u32(r), layer = rd_u32(r), expert = rd_u32(r), repr = rd_u32(r), ndim = rd_u32(r);
    uint32_t dims[4];
    for (size_t i = 0; i < 4; ++i) dims[i] = rd_u32(r);
    uint8_t digest[32];
    rd_bytes(r, digest, 32);
    uint64_t bytes = rd_u64(r), offset = rd_u64(r), len = rd_u64(r);
    const uint8_t *data = rd_view(r, (size_t)len);
    if (r->bad || r->at != r->n || repr != ELPIS_DSV41_STREAM_REPR_F32_LE_ROW_MAJOR_OUT_IN || !len)
        return ELPIS_EXEC_INVALID;
    tensor *t = role_slot(m, kind, layer, expert);
    size_t want[4];
    uint32_t want_ndim = 0;
    if (!t || !expected_shape(m, kind, layer, want, &want_ndim) || ndim != want_ndim) return ELPIS_EXEC_INVALID;
    size_t count = 1;
    for (uint32_t i = 0; i < 4; ++i) {
        if (i < ndim) {
            if (dims[i] != want[i] || !mul_ok(count, want[i], &count)) return ELPIS_EXEC_INVALID;
        } else if (dims[i] != 0) {
            return ELPIS_EXEC_INVALID;
        }
    }
    size_t expected_bytes = 0;
    if (!mul_ok(count, 4, &expected_bytes) || bytes != expected_bytes || t->complete) return ELPIS_EXEC_INVALID;
    if (!t->data) {
        if (offset != 0) return ELPIS_EXEC_INVALID;
        t->data = malloc(expected_bytes);
        if (!t->data) return ELPIS_EXEC_INTERNAL;
        t->bytes = expected_bytes;
        t->ndim = ndim;
        memcpy(t->dims, dims, sizeof(dims));
    }
    if (offset != t->received || len > t->bytes - t->received) return ELPIS_EXEC_INVALID;
    memcpy((uint8_t *)t->data + t->received, data, (size_t)len);
    t->received += (size_t)len;
    if (t->received == t->bytes) {
        t->complete = 1;
        m->tensors_received += 1;
        m->tensor_bytes += t->bytes;
    }
    return ELPIS_EXEC_OK;
}

static elpis_exec_status admit_end(context *c, rd *r) {
    model *m = c->m;
    if (!m || m->ready) return ELPIS_EXEC_INVALID;
    uint32_t count = rd_u32(r);
    (void)rd_u32(r);
    if (r->bad || r->at != r->n || count != m->tensors_received) return ELPIS_EXEC_INVALID;
    for (uint32_t k = ELPIS_DSV41_ROLE_EMBED; k <= ELPIS_DSV41_ROLE_ROPE_COMPRESSED; ++k)
        if (!m->global[k].complete) return ELPIS_EXEC_INVALID;
    for (uint32_t layer = 0; layer < m->layers; ++layer) {
        for (uint32_t k = ELPIS_DSV41_ROLE_ATTN_NORM; k <= ELPIS_DSV41_ROLE_ENGRAM_K; ++k) {
            size_t dims[4];
            uint32_t nd;
            int required = expected_shape(m, k, layer, dims, &nd);
            tensor *t = role_slot(m, k, layer, ELPIS_DSV41_STREAM_NONE);
            if (required != t->complete || (!required && t->data)) return ELPIS_EXEC_INVALID;
        }
        for (uint32_t e = 0; e <= m->experts; ++e) {
            int have = 0;
            for (uint32_t k = 0; k < 3; ++k) {
                tensor *t = &m->expert_role[(layer * (m->experts + 1) + e) * 3 + k];
                if (t->data && !t->complete) return ELPIS_EXEC_INVALID;
                have += t->complete;
            }
            if (have != 0 && have != 3) return ELPIS_EXEC_INVALID; /* resident experts are whole */
            m->resident[layer * (m->experts + 1) + e] = have == 3;
        }
    }
    if (!mul_ok(m->expert_dim, m->dim, &m->image_bytes) || !mul_ok(m->image_bytes, 12, &m->image_bytes))
        return ELPIS_EXEC_INVALID;
    m->slot = malloc(m->image_bytes);
    if (!m->slot) return ELPIS_EXEC_INTERNAL;
    atomic_fetch_add(&c_slots, 1);
    m->ready = 1;
    return ELPIS_EXEC_OK;
}

/* ------------------------------------------------------------------------- */
/* Stream open/release                                                        */
/* ------------------------------------------------------------------------- */

#define ALLOC_N(field, n, size) do { size_t count_ = (n); (field) = calloc(count_ ? count_ : 1u, (size)); \
                                     if (!(field)) goto nomem; } while (0)
#define ALLOC_F(field, n) ALLOC_N(field, n, sizeof(float))
#define ALLOC_U(field, n) ALLOC_N(field, n, sizeof(uint32_t))

static elpis_exec_status stream_open(context *c, const elpis_dsv41_stream_header *h, rd *r, uint64_t *state_bytes) {
    model *m = c->m;
    if (!m || !m->ready || c->s || h->stream_id <= c->last_stream_id) return ELPIS_EXEC_INVALID;
    uint32_t max_tokens = rd_u32(r), flags = rd_u32(r);
    const uint32_t known = ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS | ELPIS_DSV41_STREAM_FEATURE_CONDITIONING;
    if (r->bad || max_tokens != m->max_tokens || (flags & ~known) || ((flags & known) & ~m->granted))
        return ELPIS_EXEC_INVALID;
    const uint8_t *conditioning = NULL;
    if (flags & ELPIS_DSV41_STREAM_FEATURE_CONDITIONING) {
        uint32_t count = rd_u32(r), reserved = rd_u32(r);
        if (r->bad || count != m->dim || reserved) return ELPIS_EXEC_INVALID;
        conditioning = rd_view(r, (size_t)count * 4);
    }
    if (r->bad || r->at != r->n) return ELPIS_EXEC_INVALID;
    stream *s = calloc(1, sizeof(*s));
    if (!s) return ELPIS_EXEC_INTERNAL;
    atomic_fetch_add(&c_streams, 1);
    if (conditioning) {
        s->conditioning = malloc(m->dim * sizeof(float));
        if (!s->conditioning) { stream_free(s, m); return ELPIS_EXEC_INTERNAL; }
        memcpy(s->conditioning, conditioning, m->dim * sizeof(float));
        for (size_t i = 0; i < m->dim; ++i)
            if (!isfinite(s->conditioning[i])) { stream_free(s, m); return ELPIS_EXEC_INVALID; }
    }
    s->id = h->stream_id;
    c->last_stream_id = h->stream_id; /* stream identities are never reused */
    s->epoch = h->epoch;
    s->observe = (flags & ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS) != 0;
    size_t cd = m->hc * m->dim, k1 = m->active + 1, max_groups_max = m->max_tokens;
    s->attn = calloc(m->layers, sizeof(*s->attn));
    if (!s->attn) goto nomem;
    *state_bytes = 0;
    for (size_t i = 0; i < m->layers; ++i) {
        int owner = (m->flags[i] & ELPIS_DSV41_LAYER_KV_OWNER) != 0;
        if (elpis_dsv41_attention_state_create(m->local_window, m->head_dim, m->index_dim, m->max_tokens,
                                               m->ratio[i], owner, &s->attn[i]) != ELPIS_DSV41_NATIVE_OK)
            goto nomem;
        *state_bytes += elpis_dsv41_attention_state_bytes(s->attn[i]);
    }
    ALLOC_F(s->cur, cd); ALLOC_F(s->pre, m->hc); ALLOC_F(s->stream_work, cd); ALLOC_F(s->ap, m->hc);
    ALLOC_F(s->apost, m->hc); ALLOC_F(s->ac, m->hc * m->hc); ALLOC_F(s->attn_x, m->dim);
    ALLOC_F(s->attn_out, m->dim); ALLOC_F(s->stream_after, cd); ALLOC_F(s->fp, m->hc);
    ALLOC_F(s->fpost, m->hc); ALLOC_F(s->fc, m->hc * m->hc); ALLOC_F(s->moe_x, m->dim);
    ALLOC_F(s->moe_out, m->dim); ALLOC_F(s->gate, m->expert_dim); ALLOC_F(s->up, m->expert_dim);
    ALLOC_F(s->expert_out, m->dim); ALLOC_F(s->stream_out, cd); ALLOC_F(s->route_values, m->active);
    ALLOC_F(s->head_pre, m->dim); ALLOC_F(s->head_hidden, m->dim); ALLOC_F(s->logits, m->vocab);
    ALLOC_F(s->layer_streams, m->layers * cd);
    ALLOC_F(s->engram_rows, m->layers * m->hash_columns * m->engram_dim);
    ALLOC_F(s->weights, k1);
    ALLOC_U(s->chosen, m->active); ALLOC_U(s->route_order, m->active); ALLOC_U(s->ordered, k1);
    ALLOC_U(s->need, k1); ALLOC_U(s->src, k1);
    ALLOC_U(s->shared_selected, m->index_topk); ALLOC_U(s->selected_buf, m->index_topk);
    ALLOC_U(s->trace_selected, m->layers * m->active); ALLOC_U(s->trace_elided, m->layers);
    ALLOC_U(s->trace_count, m->layers); ALLOC_U(s->trace_positions, m->layers * m->index_topk);
    ALLOC_U(s->trace_npositions, m->layers);
    s->pins = calloc(k1, sizeof(cache_entry *));
    s->shared_candidates = calloc(max_groups_max, 1);
    s->candidates_buf = calloc(max_groups_max, 1);
    if (!s->pins || !s->shared_candidates || !s->candidates_buf) goto nomem;
    s->frame_floats = elpis_dsv41_layer_frame_scratch_floats(m->hc, m->dim, m->hash_columns * m->engram_dim,
                                                             m->experts, 1);
    size_t frame_plain = elpis_dsv41_layer_frame_scratch_floats(m->hc, m->dim, 0, m->experts, 0);
    if (frame_plain > s->frame_floats) s->frame_floats = frame_plain;
    s->local_floats = elpis_dsv41_local_attention_scratch_floats(m->dim, m->q_rank, m->heads, m->head_dim,
                                                                 m->o_groups, m->o_rank, m->local_window);
    s->comp_floats = 0;
    for (size_t i = 0; i < m->layers; ++i) {
        if (!m->ratio[i]) continue;
        size_t n = elpis_dsv41_compressed_attention_scratch_floats(
            m->q_rank, m->heads, m->head_dim, m->index_heads, m->index_dim, m->o_groups, m->o_rank,
            m->local_window, m->max_tokens / m->ratio[i], m->index_topk);
        if (!n) goto nomem;
        if (n > s->comp_floats) s->comp_floats = n;
    }
    if (!s->frame_floats || !s->local_floats) goto nomem;
    ALLOC_F(s->frame_scratch, s->frame_floats);
    ALLOC_F(s->local_scratch, s->local_floats);
    ALLOC_F(s->comp_scratch, s->comp_floats);
    c->s = s;
    return ELPIS_EXEC_OK;
nomem:
    stream_free(s, m);
    return ELPIS_EXEC_INTERNAL;
}

static void unpin_layer(stream *s, const model *m) {
    for (size_t i = 0; i <= m->active; ++i) {
        if (s->pins[i]) s->pins[i]->pinned = 0;
        s->pins[i] = NULL;
    }
}

/* ------------------------------------------------------------------------- */
/* Token execution (DSV41NativeBackend.apply_layer order)                     */
/* ------------------------------------------------------------------------- */

static int native_ok(elpis_dsv41_native_status st) { return st == ELPIS_DSV41_NATIVE_OK; }

static int cache_insert(model *m, stream *s, uint32_t layer, uint32_t expert) {
    if (!(m->granted & ELPIS_DSV41_STREAM_FEATURE_CACHE) || m->image_bytes > m->cache_budget) return 1;
    size_t at = layer * (m->experts + 1) + expert;
    if (m->cache[at]) return 1;
    while (m->cache_bytes + m->image_bytes > m->cache_budget) {
        size_t victim = SIZE_MAX;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < m->layers * (m->experts + 1); ++i) {
            cache_entry *e = m->cache[i];
            if (e && !e->pinned && e->last_use < oldest) { oldest = e->last_use; victim = i; }
        }
        if (victim == SIZE_MAX) return 1; /* everything pinned: skip insertion */
        cache_free_entry(m, victim);
        atomic_fetch_add(&c_evictions, 1);
    }
    cache_entry *e = calloc(1, sizeof(*e));
    if (!e) return 0;
    e->data = malloc(m->image_bytes);
    if (!e->data) { free(e); return 0; }
    memcpy(e->data, m->slot, m->image_bytes);
    e->bytes = m->image_bytes;
    memcpy(e->digests, s->slot_digests, sizeof(e->digests));
    e->last_use = ++m->cache_clock;
    m->cache[at] = e;
    m->cache_bytes += e->bytes;
    m->cache_count += 1;
    atomic_fetch_add(&c_cache_entries, 1);
    atomic_fetch_add(&c_cache_bytes, (long long)e->bytes);
    return 1;
}

static int layer_begin_and_route(model *m, stream *s, uint32_t L, uint32_t position) {
    size_t D = m->dim, C = m->hc;
    int engram = (m->flags[L] & ELPIS_DSV41_LAYER_ENGRAM) != 0;
    size_t row_values = engram ? m->hash_columns * m->engram_dim : 0;
    size_t frame_n = elpis_dsv41_layer_frame_scratch_floats(C, D, row_values, m->experts, engram);
    if (!frame_n || frame_n > s->frame_floats) return 0;
    if (!native_ok(elpis_dsv41_layer_begin_f32(
            s->cur, s->pre, engram ? s->engram_rows + (size_t)L * row_values : NULL,
            engram ? W(m, ELPIS_DSV41_ROLE_ENGRAM_WKV, L) : NULL, engram ? W(m, ELPIS_DSV41_ROLE_ENGRAM_Q, L) : NULL,
            engram ? W(m, ELPIS_DSV41_ROLE_ENGRAM_K, L) : NULL, engram, row_values,
            W(m, ELPIS_DSV41_ROLE_HC_ATTN_FN, L), W(m, ELPIS_DSV41_ROLE_HC_ATTN_SCALE, L),
            W(m, ELPIS_DSV41_ROLE_HC_ATTN_BASE, L), W(m, ELPIS_DSV41_ROLE_ATTN_NORM, L), m->norm_eps, m->hc_eps,
            m->sinkhorn, s->frame_scratch, frame_n, s->stream_work, s->ap, s->apost, s->ac, s->attn_x, C, D)))
        return 0;

    uint32_t ratio = m->ratio[L];
    size_t pairs = m->rope_dim / 2;
    s->trace_npositions[L] = 0;
    if (!ratio) {
        const float *freq = G(m, ELPIS_DSV41_ROLE_ROPE_LOCAL) + (size_t)position * pairs * 2;
        if (!native_ok(elpis_dsv41_local_attention_apply_f32(
                s->attn[L], position, s->attn_x, W(m, ELPIS_DSV41_ROLE_ATTN_WQ_A, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_Q_NORM, L), W(m, ELPIS_DSV41_ROLE_ATTN_WQ_B, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_WKV, L), W(m, ELPIS_DSV41_ROLE_ATTN_KV_NORM, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_SINK, L), W(m, ELPIS_DSV41_ROLE_ATTN_WO_A, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_WO_B, L), freq, m->norm_eps, s->local_scratch, s->local_floats,
                s->attn_out, D, m->q_rank, m->heads, m->head_dim, pairs, m->o_groups, m->o_rank, m->local_window)))
            return 0;
    } else {
        int kv_owner = (m->flags[L] & ELPIS_DSV41_LAYER_KV_OWNER) != 0;
        int index_source = (m->flags[L] & ELPIS_DSV41_LAYER_INDEX_SOURCE) != 0;
        if (kv_owner) s->owner = s->attn[L];
        if (!s->owner) return 0;
        size_t max_groups = m->max_tokens / ratio;
        memset(s->selected_buf, 0, m->index_topk * sizeof(uint32_t));
        size_t selected_count = 0;
        if (s->has_selected) {
            if (s->shared_selected_count > m->index_topk) return 0;
            memcpy(s->selected_buf, s->shared_selected, s->shared_selected_count * sizeof(uint32_t));
            selected_count = s->shared_selected_count;
        }
        memset(s->candidates_buf, 0, max_groups);
        size_t candidate_count = 0;
        if (s->has_candidates) {
            if (s->shared_candidate_count > max_groups) return 0;
            memcpy(s->candidates_buf, s->shared_candidates, s->shared_candidate_count);
            candidate_count = s->shared_candidate_count;
        }
        int candidate_mode = 0;
        if (index_source && (int32_t)L == m->candidate_source) candidate_mode = 1;
        else if (index_source && m->candidate_source >= 0 && m->candidate_source < (int32_t)L) candidate_mode = 2;
        const float *table = G(m, ELPIS_DSV41_ROLE_ROPE_COMPRESSED);
        const float *freq = table + (size_t)position * pairs * 2;
        const float *group_freq = NULL;
        if (kv_owner && (position + 1) % ratio == 0) group_freq = table + (size_t)(position + 1 - ratio) * pairs * 2;
        size_t comp_n = elpis_dsv41_compressed_attention_scratch_floats(
            m->q_rank, m->heads, m->head_dim, m->index_heads, m->index_dim, m->o_groups, m->o_rank,
            m->local_window, max_groups, m->index_topk);
        if (!comp_n || comp_n > s->comp_floats) return 0;
        if (!native_ok(elpis_dsv41_compressed_attention_apply_f32(
                s->attn[L], s->owner, position, s->attn_x, W(m, ELPIS_DSV41_ROLE_ATTN_WQ_A, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_Q_NORM, L), W(m, ELPIS_DSV41_ROLE_ATTN_WQ_B, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_WKV, L), W(m, ELPIS_DSV41_ROLE_ATTN_KV_NORM, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_SINK, L), W(m, ELPIS_DSV41_ROLE_ATTN_WO_A, L),
                W(m, ELPIS_DSV41_ROLE_ATTN_WO_B, L), W(m, ELPIS_DSV41_ROLE_COMP_WKV, L),
                W(m, ELPIS_DSV41_ROLE_COMP_WGATE, L), W(m, ELPIS_DSV41_ROLE_COMP_NORM, L),
                W(m, ELPIS_DSV41_ROLE_IDX_WK, L), W(m, ELPIS_DSV41_ROLE_IDX_K_NORM, L),
                W(m, ELPIS_DSV41_ROLE_IDX_WQ_B, L), W(m, ELPIS_DSV41_ROLE_IDX_WEIGHTS_PROJ, L), freq, group_freq,
                m->norm_eps, s->selected_buf, m->index_topk, &selected_count, s->candidates_buf, max_groups,
                &candidate_count, s->comp_scratch, comp_n, s->attn_out, D, m->q_rank, m->heads, m->head_dim, pairs,
                m->index_heads, m->index_dim, m->o_groups, m->o_rank, m->local_window, ratio, max_groups,
                m->index_topk, kv_owner, index_source, candidate_mode, m->candidate_blocks,
                m->candidate_block_size)))
            return 0;
        if (selected_count > m->index_topk || candidate_count > max_groups) return 0;
        memcpy(s->trace_positions + (size_t)L * m->index_topk, s->selected_buf, selected_count * sizeof(uint32_t));
        s->trace_npositions[L] = (uint32_t)selected_count;
        if (index_source) {
            memcpy(s->shared_selected, s->selected_buf, selected_count * sizeof(uint32_t));
            s->shared_selected_count = selected_count;
            s->has_selected = 1;
        }
        if (candidate_mode == 1) {
            memcpy(s->shared_candidates, s->candidates_buf, candidate_count);
            s->shared_candidate_count = candidate_count;
            s->has_candidates = 1;
        }
    }
    s->trace_count[L] = (uint32_t)elpis_dsv41_attention_state_count(s->attn[L]);

    if (!native_ok(elpis_dsv41_layer_after_attention_route_f32(
            s->stream_work, s->attn_out, s->ap, s->apost, s->ac, W(m, ELPIS_DSV41_ROLE_HC_FFN_FN, L),
            W(m, ELPIS_DSV41_ROLE_HC_FFN_SCALE, L), W(m, ELPIS_DSV41_ROLE_HC_FFN_BASE, L),
            W(m, ELPIS_DSV41_ROLE_FFN_NORM, L), W(m, ELPIS_DSV41_ROLE_GATE_WEIGHT, L),
            W(m, ELPIS_DSV41_ROLE_GATE_BIAS, L), m->norm_eps, m->hc_eps, m->sinkhorn, m->score_mode, m->gate_temp,
            m->norm_topk, m->route_scale, s->frame_scratch, frame_n, s->stream_after, s->fp, s->fpost, s->fc,
            s->moe_x, s->chosen, s->route_values, s->route_order, C, D, m->experts, m->active)))
        return 0;
    memcpy(s->trace_selected + (size_t)L * m->active, s->chosen, m->active * sizeof(uint32_t));

    /* Canonical accumulation order: ascending routed expert ID, then shared. */
    s->need_count = 0;
    s->cursor = 0;
    s->j = 0;
    s->hits_mask = 0;
    for (size_t t = 0; t <= m->active; ++t) {
        uint32_t id = t < m->active ? s->chosen[s->route_order[t]] : (uint32_t)m->experts;
        if (t < m->active && (s->route_order[t] >= m->active || id >= m->experts)) return 0;
        s->ordered[t] = id;
        s->weights[t] = t < m->active ? s->route_values[s->route_order[t]] : 0.0f;
        size_t at = L * (m->experts + 1) + id;
        if (m->resident[at]) {
            s->src[t] = SRC_RESIDENT;
        } else if (m->cache[at]) {
            s->src[t] = SRC_CACHE;
            s->pins[t] = m->cache[at];
            m->cache[at]->pinned = 1;
            m->cache[at]->last_use = ++m->cache_clock;
            s->hits_mask |= UINT64_C(1) << t;
            s->token_hits += 1;
            atomic_fetch_add(&c_cache_hits, 1);
        } else {
            s->src[t] = SRC_NEED;
            s->need[s->need_count++] = id;
        }
    }
    s->layer_needed = s->need_count > 0;
    s->trace_elided[L] = s->need_count == 0;
    memset(s->moe_out, 0, D * sizeof(float));
    return 1;
}

/* Returns 1 when the layer finished, 0 when a NEED must be reported, -1 on failure. */
static int run_experts(model *m, stream *s, uint32_t L) {
    size_t D = m->dim, F = m->expert_dim, plane = F * D;
    while (s->j <= m->active) {
        uint32_t t = s->j, id = s->ordered[t];
        const float *w1, *w3, *w2;
        if (s->src[t] == SRC_RESIDENT) {
            const tensor *e = &m->expert_role[(L * (m->experts + 1) + id) * 3];
            w1 = e[0].data; w3 = e[1].data; w2 = e[2].data;
        } else if (s->src[t] == SRC_CACHE) {
            w1 = s->pins[t]->data; w3 = w1 + plane; w2 = w1 + 2 * plane;
        } else {
            if (!s->slot_expert_valid || s->slot_expert != id || s->slot_fill != m->image_bytes) return 0;
            w1 = m->slot; w3 = w1 + plane; w2 = w1 + 2 * plane;
        }
        int weighted = t < m->active;
        if (!native_ok(elpis_dsv41_expert_accumulate_f32(s->moe_x, w1, w3, w2, s->weights[t], weighted,
                                                         m->swiglu_limit, s->gate, s->up, s->expert_out,
                                                         s->moe_out, D, F)))
            return -1;
        if (s->src[t] == SRC_NEED) {
            if (!cache_insert(m, s, L, id)) return -1;
            s->slot_expert_valid = 0;
            s->slot_fill = 0;
            s->cursor += 1;
        }
        s->j += 1;
    }
    unpin_layer(s, m);
    if (!native_ok(elpis_dsv41_layer_finish_f32(s->moe_out, s->stream_after, s->fpost, s->fc, s->stream_out,
                                                m->hc, D)))
        return -1;
    memcpy(s->cur, s->stream_out, m->hc * D * sizeof(float));
    memcpy(s->pre, s->fp, m->hc * sizeof(float));
    if (s->observe) memcpy(s->layer_streams + (size_t)L * m->hc * D, s->cur, m->hc * D * sizeof(float));
    return 1;
}

/* Advance until a NEED or the token completes. Returns 1 complete, 0 need, -1 failure. */
static int advance(model *m, stream *s) {
    while (s->layer < m->layers) {
        if (s->phase == 0) {
            if (!layer_begin_and_route(m, s, s->layer, s->position)) return -1;
            s->phase = 1;
        }
        int r = run_experts(m, s, s->layer);
        if (r <= 0) return r;
        s->layer += 1;
        s->phase = 0;
    }
    if (!native_ok(elpis_dsv41_hc_pre_f32(s->cur, s->pre, s->head_pre, m->hc, m->dim)) ||
        !native_ok(elpis_dsv41_rms_f32(s->head_pre, G(m, ELPIS_DSV41_ROLE_NORM), m->norm_eps, s->head_hidden,
                                       m->dim)) ||
        !native_ok(elpis_dsv41_linear_f32(s->head_hidden, G(m, ELPIS_DSV41_ROLE_HEAD), s->logits, m->dim, m->vocab)))
        return -1;
    return 1;
}

static void write_need(context *c, const elpis_dsv41_stream_header *h, outcome *o, size_t max_output) {
    model *m = c->m;
    stream *s = c->s;
    size_t body = 8 + m->active * 4 + 4 + s->need_count * 4 + 8 + 16;
    uint8_t *p = begin_reply(c, h, ELPIS_DSV41_STREAM_NEED, s->position, s->layer, body, max_output, o);
    if (!p) return;
    wr w = {p, 0};
    wr_u32(&w, s->layer);
    wr_u32(&w, (uint32_t)m->active);
    for (size_t i = 0; i < m->active; ++i) wr_u32(&w, s->chosen[i]);
    wr_u32(&w, s->need_count);
    for (size_t i = 0; i < s->need_count; ++i) wr_u32(&w, s->need[i]);
    wr_u32(&w, s->cursor);
    wr_u32(&w, 0);
    wr_u64(&w, s->slot_fill);
    wr_u64(&w, s->hits_mask);
}

static void write_complete(context *c, const elpis_dsv41_stream_header *h, outcome *o, size_t max_output,
                           uint32_t position) {
    model *m = c->m;
    stream *s = c->s;
    size_t per_layer = m->active * 4 + 12 + m->index_topk * 4;
    size_t streams = s->observe ? m->layers * m->hc * m->dim * 4 : 0;
    size_t body = 24 + 48 + m->vocab * 4 + m->layers * per_layer + streams;
    uint8_t *p = begin_reply(c, h, ELPIS_DSV41_STREAM_COMPLETE, position, ELPIS_DSV41_STREAM_NONE, body,
                             max_output, o);
    if (!p) return;
    wr w = {p, 0};
    wr_u32(&w, (uint32_t)m->vocab);
    wr_u32(&w, (uint32_t)m->layers);
    wr_u32(&w, (uint32_t)m->active);
    wr_u32(&w, (uint32_t)m->index_topk);
    wr_u32(&w, s->observe ? ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS : 0u);
    wr_u32(&w, 0);
    wr_u64(&w, m->cache_bytes);
    wr_u64(&w, m->cache_count);
    wr_u64(&w, atomic_load(&c_evictions));
    wr_u64(&w, s->token_hits);
    wr_u64(&w, atomic_load(&c_slot_high_water));
    wr_u64(&w, s->token_supplied);
    wr_bytes(&w, s->logits, m->vocab * 4);
    for (size_t L = 0; L < m->layers; ++L) {
        wr_bytes(&w, s->trace_selected + L * m->active, m->active * 4);
        wr_u32(&w, s->trace_elided[L]);
        wr_u32(&w, s->trace_count[L]);
        wr_u32(&w, s->trace_npositions[L]);
        uint32_t *pos = s->trace_positions + L * m->index_topk;
        for (size_t i = 0; i < m->index_topk; ++i) wr_u32(&w, i < s->trace_npositions[L] ? pos[i] : 0u);
    }
    if (s->observe) wr_bytes(&w, s->layer_streams, streams);
}

static elpis_exec_status token_begin(context *c, const elpis_dsv41_stream_header *h, rd *r) {
    model *m = c->m;
    stream *s = c->s;
    if (!s || s->in_token || h->position != s->position || s->position >= m->max_tokens) return ELPIS_EXEC_INVALID;
    uint32_t token = rd_u32(r), count = rd_u32(r);
    size_t engram_layers = 0;
    for (size_t i = 0; i < m->layers; ++i) engram_layers += (m->flags[i] & ELPIS_DSV41_LAYER_ENGRAM) != 0;
    if (r->bad || token >= m->vocab || count != engram_layers) return ELPIS_EXEC_INVALID;
    size_t row_values = m->hash_columns * m->engram_dim;
    uint32_t prev = ELPIS_DSV41_STREAM_NONE;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t layer = rd_u32(r), rows = rd_u32(r), row_dim = rd_u32(r);
        (void)rd_u32(r);
        if (r->bad || layer >= m->layers || !(m->flags[layer] & ELPIS_DSV41_LAYER_ENGRAM) ||
            (prev != ELPIS_DSV41_STREAM_NONE && layer <= prev) || rows != m->hash_columns || row_dim != m->engram_dim)
            return ELPIS_EXEC_INVALID;
        rd_bytes(r, s->engram_rows + (size_t)layer * row_values, row_values * 4);
        prev = layer;
    }
    if (r->bad || r->at != r->n) return ELPIS_EXEC_INVALID;
    const float *embed = G(m, ELPIS_DSV41_ROLE_EMBED) + (size_t)token * m->dim;
    for (size_t copy = 0; copy < m->hc; ++copy) {
        float *row = s->cur + copy * m->dim;
        if (!s->conditioning) {
            memcpy(row, embed, m->dim * 4);
        } else {
            for (size_t i = 0; i < m->dim; ++i) row[i] = embed[i] + s->conditioning[i]; /* one F32 add */
        }
    }
    memset(s->pre, 0, m->hc * 4);
    s->pre[0] = 1.0f;
    s->in_token = 1;
    s->layer = 0;
    s->phase = 0;
    s->owner = NULL;
    s->has_selected = s->has_candidates = 0;
    s->shared_selected_count = s->shared_candidate_count = 0;
    s->slot_expert_valid = 0;
    s->slot_fill = 0;
    s->token_hits = s->token_supplied = 0;
    return ELPIS_EXEC_OK;
}

static elpis_exec_status expert_supply(context *c, const elpis_dsv41_stream_header *h, rd *r) {
    model *m = c->m;
    stream *s = c->s;
    if (!s || !s->in_token || h->position != s->position || h->layer != s->layer || s->phase != 1 ||
        s->cursor >= s->need_count)
        return ELPIS_EXEC_INVALID;
    uint32_t layer = rd_u32(r), expert = rd_u32(r), cursor = rd_u32(r), repr = rd_u32(r);
    uint64_t image = rd_u64(r), offset = rd_u64(r), len = rd_u64(r), reserved = rd_u64(r);
    uint8_t digests[3][32];
    rd_bytes(r, digests, sizeof(digests));
    const uint8_t *data = rd_view(r, (size_t)len);
    if (r->bad || r->at != r->n || layer != s->layer || cursor != s->cursor || expert != s->need[s->cursor] ||
        repr != ELPIS_DSV41_STREAM_REPR_F32_LE_ROW_MAJOR_OUT_IN || image != m->image_bytes || reserved ||
        offset != s->slot_fill || !len || len > image - offset)
        return ELPIS_EXEC_INVALID;
    if (offset == 0) {
        memcpy(s->slot_digests, digests, sizeof(digests));
        s->slot_expert = expert;
        s->slot_expert_valid = 1;
        size_t at = layer * (m->experts + 1) + expert;
        /* A cached copy of this expert must carry the same binding identity. */
        if (m->cache[at] && memcmp(m->cache[at]->digests, digests, sizeof(digests))) return ELPIS_EXEC_INVALID;
    } else if (!s->slot_expert_valid || s->slot_expert != expert ||
               memcmp(s->slot_digests, digests, sizeof(digests))) {
        return ELPIS_EXEC_INVALID;
    }
    memcpy((uint8_t *)m->slot + offset, data, (size_t)len);
    s->slot_fill += (size_t)len;
    note_high_water(s->slot_fill);
    if (s->slot_fill == m->image_bytes) {
        s->token_supplied += 1;
        atomic_fetch_add(&c_supplied, 1);
    }
    return ELPIS_EXEC_OK;
}

/* ------------------------------------------------------------------------- */
/* Message processing                                                         */
/* ------------------------------------------------------------------------- */

static void apply_reply_fault(context *c, outcome *o, uint32_t fault) {
    if (!o->reply || o->status != ELPIS_EXEC_OK) return;
    uint8_t *p = c->host.buffer_mutable_data(o->reply);
    elpis_dsv41_stream_header h;
    memcpy(&h, p, sizeof(h));
    if (fault == YTS_FAULT_BAD_ECHO) {
        h.seq += 1;
        memcpy(p, &h, sizeof(h));
    } else if (fault == YTS_FAULT_BAD_NEED && h.kind == ELPIS_DSV41_STREAM_NEED) {
        uint32_t bad = (uint32_t)c->m->experts + 7u;
        memcpy(p + ELPIS_DSV41_STREAM_HEADER_BYTES + 8, &bad, 4);
    } else if (fault == YTS_FAULT_BAD_CURSOR && h.kind == ELPIS_DSV41_STREAM_NEED) {
        size_t at = ELPIS_DSV41_STREAM_HEADER_BYTES + 8 + c->m->active * 4;
        uint32_t need_count;
        memcpy(&need_count, p + at, 4);
        at += 4 + (size_t)need_count * 4;
        uint32_t cursor;
        memcpy(&cursor, p + at, 4);
        cursor += 1;
        memcpy(p + at, &cursor, 4);
    } else if (fault == YTS_FAULT_NONFINITE && h.kind == ELPIS_DSV41_STREAM_COMPLETE) {
        float nan = NAN;
        memcpy(p + ELPIS_DSV41_STREAM_HEADER_BYTES + 72, &nan, 4);
    }
}

static elpis_exec_status process(context *c, const elpis_exec_buffer *input, size_t max_output, outcome *o) {
    o->reply = NULL;
    o->status = ELPIS_EXEC_OK;
    const uint8_t *in = c->host.buffer_data(input);
    size_t n = c->host.buffer_size(input);
    atomic_fetch_add(&c_bytes_in, n);
    elpis_dsv41_stream_header h;
    if (n < ELPIS_DSV41_STREAM_HEADER_BYTES) return fail(o, c, ELPIS_EXEC_INVALID);
    memcpy(&h, in, sizeof(h));
    static const uint8_t zero[40];
    if (h.magic != ELPIS_DSV41_STREAM_MAGIC || h.version != ELPIS_DSV41_STREAM_PROTOCOL_VERSION ||
        h.header_bytes != ELPIS_DSV41_STREAM_HEADER_BYTES || h.flags || memcmp(h.reserved, zero, 40) ||
        h.body_bytes != n - ELPIS_DSV41_STREAM_HEADER_BYTES || h.seq != c->last_seq + 1)
        return fail(o, c, ELPIS_EXEC_INVALID);
    c->last_seq = h.seq;
    if (h.kind == ELPIS_DSV41_STREAM_MODEL_ADMIT_BEGIN) {
        if (c->m) return fail(o, c, ELPIS_EXEC_INVALID);
    } else if (!c->m || memcmp(h.manifest_digest, c->m->manifest, 32)) {
        return fail(o, c, ELPIS_EXEC_INVALID);
    }
    int stream_message = h.kind >= ELPIS_DSV41_STREAM_TOKEN_BEGIN && h.kind <= ELPIS_DSV41_STREAM_STREAM_RELEASE;
    if (stream_message && (!c->s || h.stream_id != c->s->id || h.epoch != c->s->epoch))
        return fail(o, c, ELPIS_EXEC_INVALID);
    if (!stream_message && h.kind != ELPIS_DSV41_STREAM_STREAM_OPEN && (h.stream_id || h.epoch))
        return fail(o, c, ELPIS_EXEC_INVALID);
    rd r = {in + ELPIS_DSV41_STREAM_HEADER_BYTES, n - ELPIS_DSV41_STREAM_HEADER_BYTES, 0, 0};
    elpis_exec_status st;
    uint8_t *p;
    switch (h.kind) {
    case ELPIS_DSV41_STREAM_MODEL_ADMIT_BEGIN:
        if ((st = admit_begin(c, &h, &r)) != ELPIS_EXEC_OK) return fail(o, c, st);
        c->has_manifest = 1;
        begin_reply(c, &h, ELPIS_DSV41_STREAM_ADMIT_PART_ACK, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 0,
                    max_output, o);
        return o->status;
    case ELPIS_DSV41_STREAM_MODEL_ADMIT_TENSOR:
        if ((st = admit_tensor(c, &r)) != ELPIS_EXEC_OK) return fail(o, c, st);
        begin_reply(c, &h, ELPIS_DSV41_STREAM_ADMIT_PART_ACK, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 0,
                    max_output, o);
        return o->status;
    case ELPIS_DSV41_STREAM_MODEL_ADMIT_END: {
        if ((st = admit_end(c, &r)) != ELPIS_EXEC_OK) return fail(o, c, st);
        p = begin_reply(c, &h, ELPIS_DSV41_STREAM_ADMIT_ACK, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 48,
                        max_output, o);
        if (!p) return o->status;
        wr w = {p, 0};
        wr_bytes(&w, PROFILE_DIGEST, 32);
        wr_u32(&w, c->m->granted);
        wr_u32(&w, 0);
        wr_u64(&w, c->m->tensor_bytes + c->m->image_bytes + c->m->cache_budget);
        return o->status;
    }
    case ELPIS_DSV41_STREAM_STREAM_OPEN: {
        uint64_t state_bytes = 0;
        if (h.position != ELPIS_DSV41_STREAM_NONE || h.layer != ELPIS_DSV41_STREAM_NONE)
            return fail(o, c, ELPIS_EXEC_INVALID);
        if ((st = stream_open(c, &h, &r, &state_bytes)) != ELPIS_EXEC_OK) return fail(o, c, st);
        p = begin_reply(c, &h, ELPIS_DSV41_STREAM_OPENED, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 8,
                        max_output, o);
        if (!p) return o->status;
        wr w = {p, 0};
        wr_u64(&w, state_bytes);
        return o->status;
    }
    case ELPIS_DSV41_STREAM_TOKEN_BEGIN:
    case ELPIS_DSV41_STREAM_EXPERT_SUPPLY: {
        st = h.kind == ELPIS_DSV41_STREAM_TOKEN_BEGIN ? token_begin(c, &h, &r) : expert_supply(c, &h, &r);
        if (st != ELPIS_EXEC_OK) return fail(o, c, st);
        int rc = advance(c->m, c->s);
        if (rc < 0) return fail(o, c, ELPIS_EXEC_INTERNAL);
        if (rc == 0) {
            write_need(c, &h, o, max_output);
            return o->status;
        }
        uint32_t position = c->s->position;
        write_complete(c, &h, o, max_output, position);
        c->s->position += 1;
        c->s->in_token = 0;
        unpin_layer(c->s, c->m);
        return o->status;
    }
    case ELPIS_DSV41_STREAM_STREAM_RELEASE:
        if (r.n) return fail(o, c, ELPIS_EXEC_INVALID);
        stream_free(c->s, c->m);
        c->s = NULL;
        begin_reply(c, &h, ELPIS_DSV41_STREAM_RELEASED, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 0,
                    max_output, o);
        return o->status;
    case ELPIS_DSV41_STREAM_MODEL_RELEASE:
        if (r.n || c->s) return fail(o, c, ELPIS_EXEC_INVALID);
        model_free(c->m);
        c->m = NULL;
        c->has_manifest = 0;
        begin_reply(c, &h, ELPIS_DSV41_STREAM_MODEL_RELEASED, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 0,
                    max_output, o);
        return o->status;
    default:
        return fail(o, c, ELPIS_EXEC_INVALID);
    }
}

/* Process one request and apply any configured post-processing fault. */
static void run_token(context *c, yts_token *t, uint32_t fault) {
    outcome o;
    process(c, t->input, t->max_output, &o);
    if (fault == YTS_FAULT_POLL_FAIL) {
        if (o.reply) c->host.buffer_release(o.reply);
        o.reply = NULL;
        o.status = ELPIS_EXEC_INTERNAL;
    } else {
        apply_reply_fault(c, &o, fault);
    }
    if (o.reply) atomic_fetch_add(&c_bytes_out, c->host.buffer_size(o.reply));
    t->reply = o.reply;
    t->status = o.status;
    t->never_done = fault == YTS_FAULT_POLL_TIMEOUT;
}

/* ------------------------------------------------------------------------- */
/* Backend callbacks                                                          */
/* ------------------------------------------------------------------------- */

static void token_free(context *c, yts_token *t) {
    if (t->reply) c->host.buffer_release(t->reply);
    free(t);
    atomic_fetch_sub(&c_tokens, 1);
}

static void sleep_interruptible(context *c, yts_token *t, uint32_t us) {
    if (!us) return;
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += us / 1000000u;
    deadline.tv_nsec += (long)(us % 1000000u) * 1000L;
    if (deadline.tv_nsec >= 1000000000L) { deadline.tv_sec += 1; deadline.tv_nsec -= 1000000000L; }
    while (!t->aborted && !c->stop) {
        if (pthread_cond_timedwait(&c->device_cv, &c->mu, &deadline) == ETIMEDOUT) break;
    }
}

static void *device_main(void *arg) {
    context *c = arg;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        while (!c->stop && !c->queue_head) pthread_cond_wait(&c->device_cv, &c->mu);
        if (c->stop) break;
        yts_token *t = c->queue_head;
        c->queue_head = t->next;
        if (!c->queue_head) c->queue_tail = NULL;
        t->next = NULL;
        t->processing = 1;
        uint32_t delay = c->config.device_delay_us;
        if (c->config.fault_kind == YTS_FAULT_SLOW && t->fault != YTS_FAULT_SLOW) delay = 0;
        sleep_interruptible(c, t, delay);
        if (!t->aborted && !c->stop) {
            pthread_mutex_unlock(&c->mu);
            run_token(c, t, t->fault); /* only this thread touches provider state in device mode */
            pthread_mutex_lock(&c->mu);
        }
        t->processing = 0;
        t->done = !t->never_done;
        t->done_ns = mono_ns();
        elpis_exec_runtime *runtime = c->runtime;
        int notify = c->config.notify && t->done && !t->aborted;
        pthread_cond_broadcast(&c->done_cv);
        if (notify && runtime) {
            /* Notify outside the provider lock: the runtime takes its own pool lock.
             * backend.shutdown joins this thread before the runtime is freed. */
            pthread_mutex_unlock(&c->mu);
            c->host.notify(runtime);
            pthread_mutex_lock(&c->mu);
        }
    }
    pthread_mutex_unlock(&c->mu);
    return NULL;
}

static elpis_exec_status cb_init(void *ctx, uint64_t *caps) {
    context *c = ctx;
    *caps = UINT64_C(1) << ELPIS_DSV41_STREAM_OPERATION;
    if (c->config.mode == YTS_MODE_DEVICE && !c->device_started) {
        if (pthread_create(&c->device, NULL, device_main, c)) return ELPIS_EXEC_INTERNAL;
        c->device_started = 1;
        atomic_fetch_add(&c_device_threads, 1);
    }
    return ELPIS_EXEC_OK;
}

static uint32_t take_fault(context *c) {
    int64_t index = c->message_index++;
    atomic_fetch_add(&c_messages, 1);
    if (c->config.fault_message >= 0 && index == c->config.fault_message && c->config.fault_kind) {
        atomic_fetch_add(&c_faults, 1);
        return c->config.fault_kind;
    }
    return YTS_FAULT_NONE;
}

static elpis_exec_status cb_submit(void *ctx, const elpis_exec_task *task, const elpis_exec_buffer *input,
                                   size_t max_output, void **token) {
    context *c = ctx;
    if (!task || task->operation != ELPIS_DSV41_STREAM_OPERATION || task->flags != ELPIS_EXEC_BACKEND_ONLY ||
        task->compute || !input || !token)
        return ELPIS_EXEC_INVALID;
    pthread_mutex_lock(&c->mu);
    uint32_t fault = take_fault(c);
    pthread_mutex_unlock(&c->mu);
    if (fault == YTS_FAULT_SUBMIT_REJECT) return ELPIS_EXEC_BACKEND_REJECTED;
    yts_token *t = calloc(1, sizeof(*t));
    if (!t) return ELPIS_EXEC_INTERNAL;
    atomic_fetch_add(&c_tokens, 1);
    t->input = input;
    t->max_output = max_output;
    if (c->config.mode == YTS_MODE_SYNC) {
        pthread_mutex_lock(&c->mu); /* orders provider state between pool threads */
        run_token(c, t, fault);
        t->done = !t->never_done;
        pthread_mutex_unlock(&c->mu);
    } else {
        pthread_mutex_lock(&c->mu);
        t->fault = fault;
        t->device = 1;
        if (c->queue_tail) c->queue_tail->next = t; else c->queue_head = t;
        c->queue_tail = t;
        pthread_cond_broadcast(&c->device_cv);
        pthread_mutex_unlock(&c->mu);
    }
    *token = t;
    return ELPIS_EXEC_OK;
}

static elpis_exec_status cb_poll(void *ctx, void *token, elpis_exec_buffer **out) {
    context *c = ctx;
    yts_token *t = token;
    pthread_mutex_lock(&c->mu);
    if (!t->done) {
        pthread_mutex_unlock(&c->mu);
        return ELPIS_EXEC_WOULD_BLOCK;
    }
    uint64_t delay = t->device ? mono_ns() - t->done_ns : 0;
    pthread_mutex_unlock(&c->mu);
    if (t->device) {
        atomic_fetch_add(&c_observed, 1);
        atomic_fetch_add(&c_observe_ns, delay);
        unsigned long long prev = atomic_load(&c_observe_max_ns);
        while (delay > prev && !atomic_compare_exchange_weak(&c_observe_max_ns, &prev, delay)) {}
    }
    elpis_exec_status status = t->status;
    if (status == ELPIS_EXEC_OK) {
        *out = t->reply;
        t->reply = NULL;
    }
    token_free(c, t);
    return status;
}

static void cb_abort(void *ctx, void *token) {
    context *c = ctx;
    yts_token *t = token;
    pthread_mutex_lock(&c->mu);
    t->aborted = 1;
    pthread_cond_broadcast(&c->device_cv);
    while (t->processing) pthread_cond_wait(&c->done_cv, &c->mu);
    /* Unlink if still queued: quiescence means no later access to the input. */
    yts_token **at = &c->queue_head, *prev = NULL;
    while (*at && *at != t) { prev = *at; at = &(*at)->next; }
    if (*at == t) {
        *at = t->next;
        if (c->queue_tail == t) c->queue_tail = prev;
    }
    pthread_mutex_unlock(&c->mu);
    atomic_fetch_add(&c_aborts, 1);
    token_free(c, t);
}

static void release_all(context *c) {
    if (c->s) { stream_free(c->s, c->m); c->s = NULL; }
    if (c->m) { model_free(c->m); c->m = NULL; }
    c->has_manifest = 0;
}

static void cb_shutdown(void *ctx) {
    context *c = ctx;
    pthread_mutex_lock(&c->mu);
    c->stop = 1;
    pthread_cond_broadcast(&c->device_cv);
    pthread_mutex_unlock(&c->mu);
    if (c->device_started) {
        pthread_join(c->device, NULL);
        c->device_started = 0;
        atomic_fetch_sub(&c_device_threads, 1);
    }
    pthread_mutex_lock(&c->mu);
    while (c->queue_head) {
        yts_token *t = c->queue_head;
        c->queue_head = t->next;
        token_free(c, t);
    }
    c->queue_tail = NULL;
    c->runtime = NULL;
    release_all(c);
    c->released = 1;
    pthread_mutex_unlock(&c->mu);
}

/* ------------------------------------------------------------------------- */
/* Attachment ABI                                                             */
/* ------------------------------------------------------------------------- */

YTS_EXPORT uint32_t elpis_dsv41_stream_provider_abi_version(void) { return ELPIS_DSV41_STREAM_PROVIDER_ABI_V1; }

YTS_EXPORT int elpis_dsv41_stream_provider_attach(const elpis_dsv41_stream_host_v1 *host, elpis_exec_backend *out) {
    if (!host || !out || host->abi_version != ELPIS_DSV41_STREAM_PROVIDER_ABI_V1 || !host->buffer_alloc ||
        !host->buffer_mutable_data || !host->buffer_data || !host->buffer_size || !host->buffer_release ||
        !host->notify)
        return -1;
    context *c = calloc(1, sizeof(*c));
    if (!c) return -1;
    if (pthread_mutex_init(&c->mu, NULL)) { free(c); return -1; }
    if (pthread_cond_init(&c->device_cv, NULL)) { pthread_mutex_destroy(&c->mu); free(c); return -1; }
    if (pthread_cond_init(&c->done_cv, NULL)) {
        pthread_cond_destroy(&c->device_cv);
        pthread_mutex_destroy(&c->mu);
        free(c);
        return -1;
    }
    c->host = *host;
    pthread_mutex_lock(&g_config_mu);
    c->config = g_config;
    pthread_mutex_unlock(&g_config_mu);
    atomic_fetch_add(&c_contexts, 1);
    out->context = c;
    out->init = cb_init;
    out->submit = cb_submit;
    out->poll = cb_poll;
    out->abort = cb_abort;
    out->shutdown = cb_shutdown;
    return 0;
}

YTS_EXPORT int elpis_dsv41_stream_provider_bind_runtime(void *context_ptr, elpis_exec_runtime *runtime) {
    context *c = context_ptr;
    if (!c || !runtime) return -1;
    pthread_mutex_lock(&c->mu);
    c->runtime = runtime;
    pthread_mutex_unlock(&c->mu);
    return 0;
}

YTS_EXPORT void elpis_dsv41_stream_provider_detach(void *context_ptr) {
    context *c = context_ptr;
    if (!c) return;
    if (!c->released) cb_shutdown(c); /* create failed before init: nothing else ran */
    pthread_cond_destroy(&c->done_cv);
    pthread_cond_destroy(&c->device_cv);
    pthread_mutex_destroy(&c->mu);
    free(c);
    atomic_fetch_sub(&c_contexts, 1);
}
