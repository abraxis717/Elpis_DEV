/*
 * YTS-R0 protocol and lifecycle test for the test-only DSV4.1 reference provider,
 * driven through the real generic execution port (elpis_exec_* with an attached
 * BACKEND_ONLY backend). Numerical parity against DSV41NativeBackend lives in the
 * Python qualification; this test covers transport invariance (chunking, cache,
 * synchronous vs asynchronous device), notify, faults, abort and resource release
 * under the sanitizer matrix.
 */
#define _POSIX_C_SOURCE 200809L

#include "elpis/dsv41_stream.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test-only provider controls (dsv41_reference_provider.c). */
typedef struct {
    int32_t mode, notify;
    uint32_t device_delay_us, fault_kind;
    int64_t fault_message;
} yts_ref_config;
typedef struct {
    int64_t contexts, models, streams, slots, cache_entries, cache_bytes, tokens, device_threads;
    uint64_t messages, bytes_in, bytes_out, aborts, cache_hits, supplied_experts, slot_high_water, faults_fired,
        cache_evictions, observed, observe_ns, observe_max_ns;
} yts_ref_counters;
void elpis_dsv41_reference_provider_configure(const yts_ref_config *config);
void elpis_dsv41_reference_provider_counters(yts_ref_counters *out);
void elpis_dsv41_reference_provider_reset_cumulative(void);

enum { FAULT_NONE, FAULT_SUBMIT_REJECT, FAULT_POLL_FAIL, FAULT_POLL_TIMEOUT, FAULT_BAD_ECHO, FAULT_BAD_NEED,
       FAULT_NONFINITE, FAULT_BAD_CURSOR, FAULT_SLOW };

/* Tiny topology with every mechanism: SWA, ratio-2 owner + consumer, ratio-1 owner
 * that is also the candidate source, one Engram layer, resident experts on layer 2. */
enum { V = 32, D = 8, NL = 4, H = 2, A = 32, R = 8, Q = 8, OG = 2, OR = 4, WIN = 4, T = 16, J = 2, I = 32,
       K = 2, E = 4, ACT = 2, F = 12, EO = 3, EH = 2, ED = 32, HC = 4, HASH = (EO - 1) * EH, RESIDENT_LAYER = 2 };
static const uint32_t RATIO[NL] = {0, 2, 2, 1};
static const uint32_t FLAGS[NL] = {0, ELPIS_DSV41_LAYER_KV_OWNER | ELPIS_DSV41_LAYER_INDEX_SOURCE | ELPIS_DSV41_LAYER_ENGRAM,
                                   0, ELPIS_DSV41_LAYER_KV_OWNER | ELPIS_DSV41_LAYER_INDEX_SOURCE};
static const uint8_t MANIFEST[32] = {7, 7, 7, 7, 1, 2, 3, 4};
static const elpis_dsv41_stream_host_v1 HOST = {
    ELPIS_DSV41_STREAM_PROVIDER_ABI_V1, 0, elpis_exec_buffer_alloc, elpis_exec_buffer_mutable_data,
    elpis_exec_buffer_data, elpis_exec_buffer_size, elpis_exec_buffer_release, elpis_exec_notify};

static float unit(uint64_t *state) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(*state >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}

typedef struct {
    elpis_exec_runtime *r;
    void *context;
    uint64_t seq, stream_id, epoch;
    size_t part_bytes;
    uint32_t position;
} session;

static void put_u32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4); *p += 4; }
static void put_u64(uint8_t **p, uint64_t v) { memcpy(*p, &v, 8); *p += 8; }
static void put_f32(uint8_t **p, float v) { memcpy(*p, &v, 4); *p += 4; }
static uint32_t get_u32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t get_u64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }

static elpis_exec_buffer *request(session *s, uint16_t kind, uint64_t stream, uint64_t epoch, uint32_t position,
                                  uint32_t layer, size_t body, uint8_t **out) {
    elpis_exec_buffer *b = elpis_exec_buffer_alloc(ELPIS_DSV41_STREAM_HEADER_BYTES + body);
    assert(b);
    uint8_t *p = elpis_exec_buffer_mutable_data(b);
    elpis_dsv41_stream_header h;
    memset(&h, 0, sizeof(h));
    h.magic = ELPIS_DSV41_STREAM_MAGIC;
    h.version = ELPIS_DSV41_STREAM_PROTOCOL_VERSION;
    h.kind = kind;
    h.header_bytes = ELPIS_DSV41_STREAM_HEADER_BYTES;
    h.stream_id = stream;
    h.epoch = epoch;
    h.seq = ++s->seq;
    h.position = position;
    h.layer = layer;
    h.body_bytes = body;
    memcpy(h.manifest_digest, MANIFEST, 32);
    memcpy(p, &h, sizeof(h));
    *out = p + ELPIS_DSV41_STREAM_HEADER_BYTES;
    return b;
}

/* Submit, then take the ordered result. Returns the port status; *reply set on OK. */
static elpis_exec_status exchange(session *s, elpis_exec_buffer *b, elpis_exec_buffer **reply) {
    elpis_exec_task task = {ELPIS_DSV41_STREAM_OPERATION, 0, 0, ELPIS_EXEC_BACKEND_ONLY, s->seq, NULL};
    uint64_t sequence = 0;
    elpis_exec_status st = elpis_exec_submit(s->r, &task, &b, &sequence);
    assert(st == ELPIS_EXEC_OK && !b);
    elpis_exec_result res;
    while ((st = elpis_exec_take(s->r, 1000, &res)) == ELPIS_EXEC_WOULD_BLOCK) {}
    assert(st == ELPIS_EXEC_OK);
    *reply = res.output;
    if (res.status == ELPIS_EXEC_OK) {
        const elpis_dsv41_stream_header *h = elpis_exec_buffer_data(res.output);
        assert(h->magic == ELPIS_DSV41_STREAM_MAGIC && !memcmp(h->manifest_digest, MANIFEST, 32));
    }
    return res.status;
}

static const elpis_dsv41_stream_header *hdr(const elpis_exec_buffer *b) { return elpis_exec_buffer_data(b); }

static void attach(session *s, yts_ref_config cfg, unsigned polls) {
    memset(s, 0, sizeof(*s));
    elpis_dsv41_reference_provider_configure(&cfg);
    elpis_exec_backend backend;
    assert(elpis_dsv41_stream_provider_abi_version() == ELPIS_DSV41_STREAM_PROVIDER_ABI_V1);
    assert(elpis_dsv41_stream_provider_attach(&HOST, &backend) == 0);
    s->context = backend.context;
    elpis_exec_config c = {1, 1, 1u << 20, 1u << 20, polls, &backend};
    assert(elpis_exec_create(&c, &s->r) == ELPIS_EXEC_OK);
    assert(elpis_dsv41_stream_provider_bind_runtime(s->context, s->r) == 0);
}

static void detach(session *s) {
    elpis_exec_destroy(s->r);
    elpis_dsv41_stream_provider_detach(s->context);
    s->r = NULL;
    s->context = NULL;
}

/* Deterministic tensor contents for a role. */
static void fill(uint32_t kind, uint32_t layer, uint32_t expert, float *out, size_t n) {
    uint64_t st = 0x9e3779b97f4a7c15ull ^ ((uint64_t)kind << 40) ^ ((uint64_t)layer << 20) ^ expert;
    int norm = kind == ELPIS_DSV41_ROLE_NORM || kind == ELPIS_DSV41_ROLE_ATTN_NORM || kind == ELPIS_DSV41_ROLE_FFN_NORM ||
               kind == ELPIS_DSV41_ROLE_ATTN_Q_NORM || kind == ELPIS_DSV41_ROLE_ATTN_KV_NORM ||
               kind == ELPIS_DSV41_ROLE_COMP_NORM || kind == ELPIS_DSV41_ROLE_IDX_K_NORM ||
               kind == ELPIS_DSV41_ROLE_ENGRAM_Q || kind == ELPIS_DSV41_ROLE_ENGRAM_K;
    if (kind == ELPIS_DSV41_ROLE_HC_ATTN_SCALE || kind == ELPIS_DSV41_ROLE_HC_FFN_SCALE) {
        out[0] = .08f; out[1] = .09f; out[2] = .07f;
        return;
    }
    if (kind == ELPIS_DSV41_ROLE_ROPE_LOCAL || kind == ELPIS_DSV41_ROLE_ROPE_COMPRESSED) {
        for (size_t t = 0; t < T; ++t)
            for (size_t p = 0; p < R / 2; ++p) {
                double angle = (double)t / pow(kind == ELPIS_DSV41_ROLE_ROPE_LOCAL ? 1e4 : 4e4, (double)(2 * p) / R);
                out[(t * (R / 2) + p) * 2] = (float)cos(angle);
                out[(t * (R / 2) + p) * 2 + 1] = (float)sin(angle);
            }
        return;
    }
    for (size_t i = 0; i < n; ++i) out[i] = norm ? 1.0f + .04f * unit(&st) : .12f * unit(&st);
}

typedef struct { uint32_t kind, layer, expert, ndim, dims[4]; } role;

static size_t roles(role *out) {
    size_t n = 0;
#define ADD(k, l, e, nd, a, b, c) out[n++] = (role){(k), (l), (e), (nd), {(a), (b), (c), 0}}
    ADD(ELPIS_DSV41_ROLE_EMBED, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 2, V, D, 0);
    ADD(ELPIS_DSV41_ROLE_HEAD, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 2, V, D, 0);
    ADD(ELPIS_DSV41_ROLE_NORM, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 1, D, 0, 0);
    ADD(ELPIS_DSV41_ROLE_ROPE_LOCAL, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 3, T, R / 2, 2);
    ADD(ELPIS_DSV41_ROLE_ROPE_COMPRESSED, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 3, T, R / 2, 2);
    for (uint32_t l = 0; l < NL; ++l) {
        uint32_t x = ELPIS_DSV41_STREAM_NONE;
        ADD(ELPIS_DSV41_ROLE_ATTN_NORM, l, x, 1, D, 0, 0); ADD(ELPIS_DSV41_ROLE_FFN_NORM, l, x, 1, D, 0, 0);
        ADD(ELPIS_DSV41_ROLE_ATTN_WQ_A, l, x, 2, Q, D, 0); ADD(ELPIS_DSV41_ROLE_ATTN_Q_NORM, l, x, 1, Q, 0, 0);
        ADD(ELPIS_DSV41_ROLE_ATTN_WQ_B, l, x, 2, H * A, Q, 0); ADD(ELPIS_DSV41_ROLE_ATTN_WKV, l, x, 2, A, D, 0);
        ADD(ELPIS_DSV41_ROLE_ATTN_KV_NORM, l, x, 1, A, 0, 0); ADD(ELPIS_DSV41_ROLE_ATTN_SINK, l, x, 1, H, 0, 0);
        ADD(ELPIS_DSV41_ROLE_ATTN_WO_A, l, x, 2, OG * OR, H * A / OG, 0);
        ADD(ELPIS_DSV41_ROLE_ATTN_WO_B, l, x, 2, D, OG * OR, 0);
        ADD(ELPIS_DSV41_ROLE_GATE_WEIGHT, l, x, 2, E, D, 0); ADD(ELPIS_DSV41_ROLE_GATE_BIAS, l, x, 1, E, 0, 0);
        ADD(ELPIS_DSV41_ROLE_HC_ATTN_FN, l, x, 2, (2 + HC) * HC, HC * D, 0);
        ADD(ELPIS_DSV41_ROLE_HC_ATTN_BASE, l, x, 1, (2 + HC) * HC, 0, 0);
        ADD(ELPIS_DSV41_ROLE_HC_ATTN_SCALE, l, x, 1, 3, 0, 0);
        ADD(ELPIS_DSV41_ROLE_HC_FFN_FN, l, x, 2, (2 + HC) * HC, HC * D, 0);
        ADD(ELPIS_DSV41_ROLE_HC_FFN_BASE, l, x, 1, (2 + HC) * HC, 0, 0);
        ADD(ELPIS_DSV41_ROLE_HC_FFN_SCALE, l, x, 1, 3, 0, 0);
        if (FLAGS[l] & ELPIS_DSV41_LAYER_KV_OWNER) {
            ADD(ELPIS_DSV41_ROLE_COMP_WKV, l, x, 2, A, D, 0); ADD(ELPIS_DSV41_ROLE_COMP_NORM, l, x, 1, A, 0, 0);
            ADD(ELPIS_DSV41_ROLE_IDX_WK, l, x, 2, I, A, 0); ADD(ELPIS_DSV41_ROLE_IDX_K_NORM, l, x, 1, I, 0, 0);
            if (RATIO[l] > 1) ADD(ELPIS_DSV41_ROLE_COMP_WGATE, l, x, 2, A, D, 0);
        }
        if (FLAGS[l] & ELPIS_DSV41_LAYER_INDEX_SOURCE) {
            ADD(ELPIS_DSV41_ROLE_IDX_WQ_B, l, x, 2, J * I, Q, 0);
            ADD(ELPIS_DSV41_ROLE_IDX_WEIGHTS_PROJ, l, x, 2, J, D, 0);
        }
        if (FLAGS[l] & ELPIS_DSV41_LAYER_ENGRAM) {
            ADD(ELPIS_DSV41_ROLE_ENGRAM_WKV, l, x, 2, (HC + 1) * D, HASH * ED, 0);
            ADD(ELPIS_DSV41_ROLE_ENGRAM_Q, l, x, 2, HC, D, 0); ADD(ELPIS_DSV41_ROLE_ENGRAM_K, l, x, 2, HC, D, 0);
        }
        if (l == RESIDENT_LAYER)
            for (uint32_t e = 0; e <= E; ++e) {
                ADD(ELPIS_DSV41_ROLE_EXPERT_W1, l, e, 2, F, D, 0); ADD(ELPIS_DSV41_ROLE_EXPERT_W3, l, e, 2, F, D, 0);
                ADD(ELPIS_DSV41_ROLE_EXPERT_W2, l, e, 2, D, F, 0);
            }
    }
#undef ADD
    return n;
}

static void expert_image(uint32_t layer, uint32_t expert, float *image) {
    fill(ELPIS_DSV41_ROLE_EXPERT_W1, layer, expert, image, F * D);
    fill(ELPIS_DSV41_ROLE_EXPERT_W3, layer, expert, image + F * D, F * D);
    fill(ELPIS_DSV41_ROLE_EXPERT_W2, layer, expert, image + 2 * F * D, F * D);
}

static elpis_exec_status admit(session *s, uint32_t features, uint64_t cache_bytes, size_t tensor_part) {
    elpis_exec_buffer *reply;
    uint8_t *p;
    size_t body = ELPIS_DSV41_STREAM_ADMIT_BEGIN_FIXED_BYTES + 2 * 4 * NL;
    elpis_exec_buffer *b = request(s, ELPIS_DSV41_STREAM_MODEL_ADMIT_BEGIN, 0, 0, ELPIS_DSV41_STREAM_NONE,
                                   ELPIS_DSV41_STREAM_NONE, body, &p);
    uint32_t u[27] = {V, D, NL, H, A, R, Q, OG, OR, WIN, T, J, I, K, E, ACT, F, EO, EH, ED, HC, 20, 2, 1, 2, 2, HASH};
    for (size_t i = 0; i < 27; ++i) put_u32(&p, u[i]);
    put_u32(&p, 3); /* candidate source */
    put_f32(&p, 1e-20f); put_f32(&p, 1e-6f); put_f32(&p, 1.0f); put_f32(&p, 1.5f); put_f32(&p, 10.0f);
    put_u32(&p, features);
    put_u32(&p, 1);
    put_u64(&p, s->part_bytes);
    put_u64(&p, cache_bytes);
    memset(p, 0x11, 32); p += 32;
    for (size_t i = 0; i < NL; ++i) put_u32(&p, RATIO[i]);
    for (size_t i = 0; i < NL; ++i) put_u32(&p, FLAGS[i]);
    elpis_exec_status st = exchange(s, b, &reply);
    if (st != ELPIS_EXEC_OK) return st;
    assert(hdr(reply)->kind == ELPIS_DSV41_STREAM_ADMIT_PART_ACK);
    elpis_exec_buffer_release(reply);

    role all[256];
    size_t n = roles(all);
    for (size_t i = 0; i < n; ++i) {
        size_t count = 1;
        for (uint32_t d = 0; d < all[i].ndim; ++d) count *= all[i].dims[d];
        float *data = malloc(count * 4);
        assert(data);
        fill(all[i].kind, all[i].layer, all[i].expert, data, count);
        for (size_t at = 0; at < count * 4; at += tensor_part) {
            size_t len = count * 4 - at < tensor_part ? count * 4 - at : tensor_part;
            b = request(s, ELPIS_DSV41_STREAM_MODEL_ADMIT_TENSOR, 0, 0, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE,
                        ELPIS_DSV41_STREAM_ADMIT_TENSOR_PREFIX_BYTES + len, &p);
            put_u32(&p, all[i].kind); put_u32(&p, all[i].layer); put_u32(&p, all[i].expert);
            put_u32(&p, ELPIS_DSV41_STREAM_REPR_F32_LE_ROW_MAJOR_OUT_IN); put_u32(&p, all[i].ndim);
            for (int d = 0; d < 4; ++d) put_u32(&p, all[i].dims[d]);
            memset(p, (int)i, 32); p += 32;
            put_u64(&p, count * 4); put_u64(&p, at); put_u64(&p, len);
            memcpy(p, (uint8_t *)data + at, len);
            st = exchange(s, b, &reply);
            if (st != ELPIS_EXEC_OK) { free(data); return st; }
            assert(hdr(reply)->kind == ELPIS_DSV41_STREAM_ADMIT_PART_ACK);
            elpis_exec_buffer_release(reply);
        }
        free(data);
    }
    b = request(s, ELPIS_DSV41_STREAM_MODEL_ADMIT_END, 0, 0, ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 8, &p);
    put_u32(&p, (uint32_t)n);
    put_u32(&p, 0);
    st = exchange(s, b, &reply);
    if (st != ELPIS_EXEC_OK) return st;
    assert(hdr(reply)->kind == ELPIS_DSV41_STREAM_ADMIT_ACK);
    const uint8_t *q = (const uint8_t *)elpis_exec_buffer_data(reply) + ELPIS_DSV41_STREAM_HEADER_BYTES + 32;
    uint32_t granted = get_u32(&q);
    assert(granted == (features & (ELPIS_DSV41_STREAM_FEATURE_OBSERVE_LAYER_STREAMS |
                                   ELPIS_DSV41_STREAM_FEATURE_CONDITIONING |
                                   (cache_bytes ? ELPIS_DSV41_STREAM_FEATURE_CACHE : 0u))));
    elpis_exec_buffer_release(reply);
    return ELPIS_EXEC_OK;
}

static elpis_exec_status open_stream(session *s) {
    uint8_t *p;
    elpis_exec_buffer *reply;
    s->stream_id += 1;
    s->epoch = 0x5eed0000u + s->stream_id;
    s->position = 0;
    elpis_exec_buffer *b = request(s, ELPIS_DSV41_STREAM_STREAM_OPEN, s->stream_id, s->epoch, ELPIS_DSV41_STREAM_NONE,
                                   ELPIS_DSV41_STREAM_NONE, 8, &p);
    put_u32(&p, T);
    put_u32(&p, 0);
    elpis_exec_status st = exchange(s, b, &reply);
    if (st != ELPIS_EXEC_OK) return st;
    assert(hdr(reply)->kind == ELPIS_DSV41_STREAM_OPENED);
    elpis_exec_buffer_release(reply);
    return st;
}

static elpis_exec_status simple(session *s, uint16_t kind, int stream_level) {
    uint8_t *p;
    elpis_exec_buffer *reply;
    elpis_exec_buffer *b = request(s, kind, stream_level ? s->stream_id : 0, stream_level ? s->epoch : 0,
                                   ELPIS_DSV41_STREAM_NONE, ELPIS_DSV41_STREAM_NONE, 0, &p);
    elpis_exec_status st = exchange(s, b, &reply);
    if (st == ELPIS_EXEC_OK) elpis_exec_buffer_release(reply);
    return st;
}

/* Run one token; returns the port status of the failing exchange or OK, and
 * copies the logits on completion. *needs counts NEED replies. */
static elpis_exec_status token(session *s, uint32_t tok, float *logits, unsigned *needs, int *protocol_ok) {
    uint8_t *p;
    elpis_exec_buffer *reply;
    size_t body = 8 + 1 * (16 + HASH * ED * 4);
    elpis_exec_buffer *b = request(s, ELPIS_DSV41_STREAM_TOKEN_BEGIN, s->stream_id, s->epoch, s->position,
                                   ELPIS_DSV41_STREAM_NONE, body, &p);
    put_u32(&p, tok);
    put_u32(&p, 1);
    put_u32(&p, 1); put_u32(&p, HASH); put_u32(&p, ED); put_u32(&p, 0);
    uint64_t st_rng = 1000u + tok * 31u + s->position;
    for (size_t i = 0; i < HASH * ED; ++i) put_f32(&p, unit(&st_rng));
    elpis_exec_status st = exchange(s, b, &reply);
    static float image[3 * F * D];
    *protocol_ok = 1;
    while (st == ELPIS_EXEC_OK) {
        const elpis_dsv41_stream_header *h = hdr(reply);
        if (h->seq != s->seq || h->stream_id != s->stream_id || h->epoch != s->epoch) {
            *protocol_ok = 0;
            elpis_exec_buffer_release(reply);
            return ELPIS_EXEC_OK;
        }
        const uint8_t *q = (const uint8_t *)h + ELPIS_DSV41_STREAM_HEADER_BYTES;
        if (h->kind == ELPIS_DSV41_STREAM_COMPLETE) {
            uint32_t vocab = get_u32(&q), layers = get_u32(&q);
            assert(vocab == V && layers == NL);
            q += 4 * 4 + 6 * 8;
            memcpy(logits, q, V * 4);
            for (size_t i = 0; i < V; ++i)
                if (!isfinite(logits[i])) *protocol_ok = 0;
            q += V * 4;
            for (uint32_t l = 0; l < NL; ++l) {
                uint32_t sel[K];
                for (int i = 0; i < ACT; ++i) sel[i] = get_u32(&q);
                uint32_t elided = get_u32(&q);
                (void)get_u32(&q);
                uint32_t npos = get_u32(&q);
                q += 4 * K;
                assert(npos <= K && sel[0] < E && sel[1] < E && sel[0] != sel[1]);
                if (l == RESIDENT_LAYER) assert(elided == 1); /* every expert resident */
            }
            elpis_exec_buffer_release(reply);
            s->position += 1;
            return ELPIS_EXEC_OK;
        }
        assert(h->kind == ELPIS_DSV41_STREAM_NEED);
        *needs += 1;
        uint32_t layer = get_u32(&q), k = get_u32(&q);
        uint32_t selected[ACT];
        for (uint32_t i = 0; i < k; ++i) selected[i] = get_u32(&q);
        uint32_t need_count = get_u32(&q), need[ACT + 1];
        for (uint32_t i = 0; i < need_count; ++i) need[i] = get_u32(&q);
        uint32_t cursor = get_u32(&q);
        (void)get_u32(&q);
        uint64_t offset = get_u64(&q);
        elpis_exec_buffer_release(reply);
        if (k != ACT || selected[0] >= E || selected[1] >= E || cursor >= need_count || layer == RESIDENT_LAYER) {
            *protocol_ok = 0;
            return ELPIS_EXEC_OK;
        }
        for (uint32_t i = 1; i < need_count; ++i) assert(need[i] > need[i - 1]);
        uint32_t expert = need[cursor];
        expert_image(layer, expert, image);
        size_t image_bytes = sizeof(image), len = image_bytes - offset < s->part_bytes ? image_bytes - offset : s->part_bytes;
        b = request(s, ELPIS_DSV41_STREAM_EXPERT_SUPPLY, s->stream_id, s->epoch, s->position, layer,
                    ELPIS_DSV41_STREAM_SUPPLY_PREFIX_BYTES + len, &p);
        put_u32(&p, layer); put_u32(&p, expert); put_u32(&p, cursor);
        put_u32(&p, ELPIS_DSV41_STREAM_REPR_F32_LE_ROW_MAJOR_OUT_IN);
        put_u64(&p, image_bytes); put_u64(&p, offset); put_u64(&p, len); put_u64(&p, 0);
        for (int r = 0; r < 3; ++r) { memset(p, (int)(layer * 16 + expert * 3 + (uint32_t)r), 32); p += 32; }
        memcpy(p, (uint8_t *)image + offset, len);
        st = exchange(s, b, &reply);
    }
    return st;
}

static void counters_released(void) {
    yts_ref_counters c;
    elpis_dsv41_reference_provider_counters(&c);
    assert(c.contexts == 0 && c.models == 0 && c.streams == 0 && c.slots == 0 && c.cache_entries == 0 &&
           c.cache_bytes == 0 && c.tokens == 0 && c.device_threads == 0);
}

enum { TOKENS = 8 };
static const uint32_t TOKS[TOKENS] = {3, 9, 4, 4, 17, 30, 1, 22};

/* One full sequence; returns the logits of every token. */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void sequence(yts_ref_config cfg, size_t part, uint64_t cache, float out[TOKENS][V], double *polls_per_sub,
                     double *ns_per_sub) {
    session s;
    attach(&s, cfg, 1000);
    s.part_bytes = part;
    assert(admit(&s, cache ? ELPIS_DSV41_STREAM_FEATURE_CACHE : 0u, cache, 777) == ELPIS_EXEC_OK);
    assert(open_stream(&s) == ELPIS_EXEC_OK);
    elpis_exec_metrics before, after;
    elpis_exec_get_metrics(s.r, &before);
    uint64_t start = now_ns();
    for (int t = 0; t < TOKENS; ++t) {
        unsigned needs = 0;
        int ok;
        assert(token(&s, TOKS[t], out[t], &needs, &ok) == ELPIS_EXEC_OK && ok);
        if (!cache) assert(needs > 0); /* experts are only ever supplied by the host */
    }
    uint64_t elapsed = now_ns() - start;
    elpis_exec_get_metrics(s.r, &after);
    double submissions = (double)(after.backend_accepted - before.backend_accepted);
    if (polls_per_sub) *polls_per_sub = (double)(after.backend_polls - before.backend_polls) / submissions;
    if (ns_per_sub) *ns_per_sub = (double)elapsed / submissions;
    assert(after.backend_fallback == 0);
    assert(simple(&s, ELPIS_DSV41_STREAM_STREAM_RELEASE, 1) == ELPIS_EXEC_OK);
    if (cache) {
        yts_ref_counters c;
        elpis_dsv41_reference_provider_counters(&c);
        assert(c.cache_entries > 0);
    }
    assert(simple(&s, ELPIS_DSV41_STREAM_MODEL_RELEASE, 0) == ELPIS_EXEC_OK);
    yts_ref_counters c;
    elpis_dsv41_reference_provider_counters(&c);
    assert(c.cache_entries == 0 && c.models == 0 && c.streams == 0);
    detach(&s);
    counters_released();
}

static void transport_invariance(void) {
    static float base[TOKENS][V], other[TOKENS][V];
    yts_ref_config sync = {0, 0, 0, FAULT_NONE, -1}, device = {1, 1, 0, FAULT_NONE, -1};
    double polls = 0, notify_ns = 0, floor_ns = 0;
    sequence(sync, 3 * F * D * 4, 0, base, &polls, NULL);
    assert(polls == 1.0); /* synchronous provider: one poll per submission */
    size_t parts[] = {1, 100, 384, 1000};
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i) {
        sequence(sync, parts[i], 0, other, NULL, NULL);
        assert(!memcmp(base, other, sizeof(base)));
    }
    /* Cache budgets: one image, a few, unlimited. Results never change. */
    /* A one-image budget thrashes (LRU over a cyclic access larger than capacity):
     * evictions but no hits. A budget holding every non-resident expert hits. */
    uint64_t budgets[] = {3 * F * D * 4, 15 * 3 * F * D * 4, UINT64_C(1) << 30};
    for (size_t i = 0; i < 3; ++i) {
        elpis_dsv41_reference_provider_reset_cumulative();
        sequence(sync, 100, budgets[i], other, NULL, NULL);
        assert(!memcmp(base, other, sizeof(base)));
        yts_ref_counters c;
        elpis_dsv41_reference_provider_counters(&c);
        if (i == 0) assert(c.cache_evictions > 0);
        else assert(c.cache_hits > 0);
    }
    /* Asynchronous fake device. With notify a completion is observed at once; without
     * it an undecided token waits for the runtime's >= 1 ms re-poll grid. The provider
     * measures the delay from device completion to the poll that observes it. Results
     * are identical either way. */
    yts_ref_counters c;
    elpis_dsv41_reference_provider_reset_cumulative();
    sequence(device, 384, 0, other, &polls, &notify_ns);
    elpis_dsv41_reference_provider_counters(&c);
    double notify_observe = (double)c.observe_ns / (double)c.observed;
    assert(!memcmp(base, other, sizeof(base)));
    yts_ref_config floor_cfg = {1, 0, 0, FAULT_NONE, -1};
    elpis_dsv41_reference_provider_reset_cumulative();
    double floor_polls = 0;
    sequence(floor_cfg, 384, 0, other, &floor_polls, &floor_ns);
    elpis_dsv41_reference_provider_counters(&c);
    double floor_observe = (double)c.observe_ns / (double)c.observed;
    assert(!memcmp(base, other, sizeof(base)));
    assert(floor_observe >= 200000.0 && notify_observe * 4.0 < floor_observe);
    yts_ref_config slow_notify = {1, 1, 2000, FAULT_NONE, -1};
    sequence(slow_notify, 1 << 20, 0, other, NULL, NULL);
    assert(!memcmp(base, other, sizeof(base)));
    printf("notify: completion observed after %.1f us (floor %.1f us); %.2f vs %.2f polls and %.1f vs %.1f us per "
           "submission\n", notify_observe / 1e3, floor_observe / 1e3, polls, floor_polls, notify_ns / 1e3,
           floor_ns / 1e3);
    printf("transport invariance: chunking, cache budgets, sync/device and notify are bitwise identical\n");
}

/* Inject one fault at request index `at` (counted from the first admission message). */
static elpis_exec_status faulted(int mode, uint32_t kind, int64_t at, unsigned polls, uint32_t delay, int *protocol_ok) {
    yts_ref_config cfg = {mode, 1, delay, kind, at};
    session s;
    attach(&s, cfg, polls);
    s.part_bytes = 200;
    elpis_exec_status st = admit(&s, 0, 0, 1 << 20);
    *protocol_ok = 1;
    if (st == ELPIS_EXEC_OK) st = open_stream(&s);
    float logits[V];
    for (int t = 0; st == ELPIS_EXEC_OK && *protocol_ok && t < 4; ++t) {
        unsigned needs = 0;
        st = token(&s, TOKS[t], logits, &needs, protocol_ok);
    }
    elpis_exec_metrics m;
    elpis_exec_get_metrics(s.r, &m);
    assert(m.backend_fallback == 0);
    detach(&s); /* quarantine: shutdown releases every provider resource */
    counters_released();
    return st;
}

static void faults(void) {
    /* Admission is 1 + roles + 1 messages with 1 MiB tensor parts; then OPEN. Fault
     * well inside the token stream and at the very first token message. */
    role all[256];
    int64_t first_token = (int64_t)roles(all) + 3;
    int64_t points[] = {first_token, first_token + 1, first_token + 5, first_token + 17};
    for (int mode = 0; mode <= 1; ++mode) {
        for (size_t i = 0; i < sizeof(points) / sizeof(points[0]); ++i) {
            int ok;
            assert(faulted(mode, FAULT_SUBMIT_REJECT, points[i], 1000, 0, &ok) == ELPIS_EXEC_BACKEND_REJECTED);
            assert(faulted(mode, FAULT_POLL_FAIL, points[i], 1000, 0, &ok) == ELPIS_EXEC_INTERNAL);
            yts_ref_counters before, after;
            elpis_dsv41_reference_provider_counters(&before);
            /* The timed-out request never completes; asynchronous mode keeps a wider budget so
             * every other request completes inside it even on a loaded (sanitizer) host. */
            assert(faulted(mode, FAULT_POLL_TIMEOUT, points[i], mode ? 20u : 3u, 0, &ok) ==
                   ELPIS_EXEC_BACKEND_UNAVAILABLE);
            elpis_dsv41_reference_provider_counters(&after);
            assert(after.aborts == before.aborts + 1);
            assert(faulted(mode, FAULT_BAD_ECHO, points[i], 1000, 0, &ok) == ELPIS_EXEC_OK && !ok);
            assert(faulted(mode, FAULT_BAD_NEED, points[i] + 1, 1000, 0, &ok) == ELPIS_EXEC_OK && !ok);
        }
        int ok;
        /* NaN logits arrive in a successful COMPLETE; the host must reject them. */
        elpis_exec_status st = ELPIS_EXEC_OK;
        for (int64_t at = first_token; at < first_token + 64; ++at) {
            st = faulted(mode, FAULT_NONFINITE, at, 1000, 0, &ok);
            if (!ok) break;
        }
        assert(st == ELPIS_EXEC_OK && !ok);
        /* Only the first token message takes 2 s of device time: the 20-poll budget expires,
         * abort interrupts the step at once and the request ends BACKEND_UNAVAILABLE. */
        if (mode == 1) {
            uint64_t begin = now_ns();
            assert(faulted(mode, FAULT_SLOW, first_token, 20, 2000000, &ok) == ELPIS_EXEC_BACKEND_UNAVAILABLE);
            assert(now_ns() - begin < 1500000000ull);
        }
    }
    printf("faults: reject, poll failure, timeout+abort, malformed echo/need, nonfinite; resources released\n");
}

static void malformed_requests(void) {
    yts_ref_config cfg = {0, 0, 0, FAULT_NONE, -1};
    session s;
    attach(&s, cfg, 1000);
    s.part_bytes = 1000;
    assert(admit(&s, 0, 0, 1 << 20) == ELPIS_EXEC_OK);
    s.seq += 5; /* skipped sequence numbers are a protocol violation */
    assert(open_stream(&s) == ELPIS_EXEC_INVALID);
    detach(&s);
    counters_released();

    attach(&s, cfg, 1000);
    s.part_bytes = 1000;
    assert(admit(&s, 0, 0, 1 << 20) == ELPIS_EXEC_OK);
    assert(open_stream(&s) == ELPIS_EXEC_OK);
    s.epoch ^= 1; /* stale epoch: cross-stream message */
    float logits[V];
    unsigned needs = 0;
    int ok;
    assert(token(&s, 1, logits, &needs, &ok) == ELPIS_EXEC_INVALID);
    detach(&s);
    counters_released();

    /* Unsupported operation is refused before admission: nothing consumed. */
    attach(&s, cfg, 1000);
    elpis_exec_task task = {ELPIS_DSV41_STREAM_OPERATION + 1, 0, 0, ELPIS_EXEC_BACKEND_ONLY, 0, NULL};
    elpis_exec_buffer *b = elpis_exec_buffer_alloc(8);
    uint64_t sequence = UINT64_MAX;
    assert(elpis_exec_submit(s.r, &task, &b, &sequence) == ELPIS_EXEC_BACKEND_UNAVAILABLE && b);
    elpis_exec_buffer_release(b);
    detach(&s);
    counters_released();

    /* Detach after a create that failed validation (init never ran). */
    elpis_exec_backend backend;
    assert(elpis_dsv41_stream_provider_attach(&HOST, &backend) == 0);
    elpis_exec_config bad = {1, 0, 1u << 20, 1u << 20, 10, &backend};
    elpis_exec_runtime *r = NULL;
    assert(elpis_exec_create(&bad, &r) == ELPIS_EXEC_INVALID && !r);
    elpis_dsv41_stream_provider_detach(backend.context);
    counters_released();
    printf("malformed requests: sequence gap, stale epoch, unsupported op, failed create\n");
}

static void readmission_clears_cache(void) {
    yts_ref_config cfg = {0, 0, 0, FAULT_NONE, -1};
    session s;
    attach(&s, cfg, 1000);
    s.part_bytes = 1 << 20;
    float logits[V];
    unsigned needs = 0;
    int ok;
    for (int round = 0; round < 2; ++round) {
        assert(admit(&s, ELPIS_DSV41_STREAM_FEATURE_CACHE, UINT64_C(1) << 30, 1 << 20) == ELPIS_EXEC_OK);
        assert(open_stream(&s) == ELPIS_EXEC_OK);
        elpis_dsv41_reference_provider_reset_cumulative();
        assert(token(&s, 5, logits, &needs, &ok) == ELPIS_EXEC_OK && ok);
        yts_ref_counters c;
        elpis_dsv41_reference_provider_counters(&c);
        assert(c.cache_hits == 0); /* first token of an admission never hits */
        assert(simple(&s, ELPIS_DSV41_STREAM_STREAM_RELEASE, 1) == ELPIS_EXEC_OK);
        assert(simple(&s, ELPIS_DSV41_STREAM_MODEL_RELEASE, 0) == ELPIS_EXEC_OK);
        elpis_dsv41_reference_provider_counters(&c);
        assert(c.cache_entries == 0 && c.cache_bytes == 0);
    }
    detach(&s);
    counters_released();
    printf("re-admission on one runtime starts from an empty cache\n");
}

int main(void) {
    transport_invariance();
    faults();
    malformed_requests();
    readmission_clears_cache();
    puts("PASS dsv41 stream provider: YTS-R0 transport invariance, notify, faults, abort, release");
    return 0;
}
