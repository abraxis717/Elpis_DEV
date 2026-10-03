#define _POSIX_C_SOURCE 200809L
#include "elpis/dsv41_clock.h"
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NONE ELPIS_DSV41_STREAM_NONE
#define S(name) ELPIS_DSV41_STREAM_##name
#define MAX_CLOCKS 64
#define MEMORY_LIMIT (UINT64_C(512) << 20)
typedef struct {
    elpis_dsv41_clock id;
    int busy, opened, quiesced;
    atomic_int cancelled;
    elpis_dsv41_clock_config_v1 c;
    elpis_dsv41_clock_metrics_v1 m;
    float *conditioning; /* owned copy of c.conditioning, or NULL */
    uint8_t *request, *completes, *needed;
    uint64_t *rows, *times;
    uint32_t *tokens, *selected;
    int64_t tail[31];
    size_t columns, row_count, complete_bytes, request_bytes, total_tokens;
    uint32_t row_at, plan_layer, need_count, need[64], chosen[64], cursor;
    uint64_t offset, hits, replies, supplies, max_supplies, waiting_since, token_start;
} clock_state;

/* Handle identities only, not a backend registry or scheduler. */
static pthread_mutex_t handles_mu = PTHREAD_MUTEX_INITIALIZER;
static clock_state *handles[MAX_CLOCKS];
static uint64_t next_id;
static uint64_t mono(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t u64(const uint8_t *p) { return u32(p) | (uint64_t)u32(p + 4) << 32; }
static void w32(uint8_t *p, uint32_t v) { for (unsigned i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static void w64(uint8_t *p, uint64_t v) { for (unsigned i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static float f32(const uint8_t *p) { uint32_t bits = u32(p); float v; memcpy(&v, &bits, 4); return v; }
static clock_state *find(elpis_dsv41_clock id) {
    for (unsigned i = 0; i < MAX_CLOCKS; ++i) if (handles[i] && handles[i]->id == id) return handles[i];
    return NULL;
}
static elpis_clock_code pin(elpis_dsv41_clock id, clock_state **out, int cancel_busy) {
    pthread_mutex_lock(&handles_mu);
    clock_state *s = find(id);
    elpis_clock_code rc = !s ? ELPIS_CLOCK_STALE : s->busy ? ELPIS_CLOCK_BUSY : ELPIS_CLOCK_OK;
    if (s && s->busy && cancel_busy) atomic_store(&s->cancelled, 1);
    if (rc == ELPIS_CLOCK_OK) { s->busy = 1; *out = s; }
    pthread_mutex_unlock(&handles_mu);
    return rc;
}
static void unpin(clock_state *s) {
    pthread_mutex_lock(&handles_mu);
    s->busy = 0;
    pthread_mutex_unlock(&handles_mu);
}
static void quiesce(clock_state *s) {
    if (!s->quiesced) { s->c.materializer.quiesce(s->c.materializer.context); s->quiesced = 1; }
}
static elpis_clock_code quarantine(clock_state *s, elpis_clock_code rc) {
    if (s->m.state != ELPIS_CLOCK_QUARANTINED) {
        s->c.host.shutdown(s->c.runtime, 1);
        /* A clock deadline can precede ordered retirement. Shutdown quiesces
         * work but retains its result; retire/release that sole possible
         * outstanding result before returning ownership to the host. */
        elpis_exec_result pending = {0};
        if (s->c.host.take(s->c.runtime, 0, &pending) == ELPIS_EXEC_OK && pending.output) {
            s->m.bytes_p2h += s->c.host.buffer_size(pending.output);
            s->c.host.buffer_release(pending.output); s->m.outputs_released++;
        }
        s->m.quarantines++;
        s->m.state = ELPIS_CLOCK_QUARANTINED;
        s->opened = 0;
        quiesce(s);
    }
    s->m.provider_code = rc; s->m.code = rc; s->m.outcome = ELPIS_CLOCK_FAILED;
    return rc;
}
static void output_release(clock_state *s, elpis_exec_buffer *b) {
    if (b) { s->c.host.buffer_release(b); s->m.outputs_released++; }
}

/* Ordered BACKEND_ONLY exchange. Only the runtime polls the provider. */
static elpis_clock_code exchange(clock_state *s, uint32_t kind, uint32_t layer, size_t n,
                                 elpis_exec_buffer **out) {
    *out = NULL;
    if (s->m.sequence == UINT64_MAX) return quarantine(s, ELPIS_CLOCK_LIMIT);
    uint64_t seq = s->m.sequence + 1, port_sequence = 0;
    elpis_exec_buffer *in = s->c.host.buffer_alloc(128 + n);
    if (!in) return ELPIS_CLOCK_LIMIT;
    s->m.allocations++;
    uint8_t *p = s->c.host.buffer_mutable_data(in);
    if (!p) { s->c.host.buffer_release(in); s->m.requests_released++; return quarantine(s, ELPIS_CLOCK_DEVICE); }
    memset(p, 0, 128);
    w32(p, S(MAGIC)); p[4] = 1; p[6] = (uint8_t)kind;
    w32(p + 8, 128); w64(p + 16, s->c.stream_id); w64(p + 24, s->c.epoch); w64(p + 32, seq);
    uint32_t position = kind == S(TOKEN_BEGIN) || kind == S(EXPERT_SUPPLY) ? s->m.position : NONE;
    w32(p + 40, position); w32(p + 44, layer); w64(p + 48, n); memcpy(p + 56, s->c.manifest, 32);
    memcpy(p + 128, s->request, n);
    elpis_exec_task task = {S(OPERATION), kind, 0, ELPIS_EXEC_BACKEND_ONLY, seq, NULL};
    elpis_exec_metrics before, after;
    s->c.host.metrics(s->c.runtime, &before);
    elpis_exec_status st = s->c.host.submit(s->c.runtime, &task, &in, &port_sequence);
    if (st != ELPIS_EXEC_OK) {
        s->c.host.buffer_release(in);
        s->m.requests_released++;
        return quarantine(s, st == ELPIS_EXEC_CLOSED ? ELPIS_CLOCK_CLOSED : ELPIS_CLOCK_DEVICE);
    }
    s->m.sequence = seq; s->m.submissions++; s->m.consumed++; s->m.bytes_h2p += 128 + n;
    elpis_exec_result result = {0};
    uint64_t start = mono();
    do {
        st = s->c.host.take(s->c.runtime, 10, &result);
        if (st == ELPIS_EXEC_WOULD_BLOCK && mono() - start >= (uint64_t)s->c.exchange_timeout_ms * 1000000) {
            elpis_clock_code rc = quarantine(s, ELPIS_CLOCK_DEVICE);
            s->c.host.metrics(s->c.runtime, &after);
            s->m.polls += after.backend_polls - before.backend_polls;
            s->m.provider_ns += after.compute_ns - before.compute_ns;
            return rc;
        }
    } while (st == ELPIS_EXEC_WOULD_BLOCK);
    s->c.host.metrics(s->c.runtime, &after);
    s->m.polls += after.backend_polls - before.backend_polls;
    s->m.provider_ns += after.compute_ns - before.compute_ns;
    if (st != ELPIS_EXEC_OK) return quarantine(s, ELPIS_CLOCK_DEVICE);
    if (result.status != ELPIS_EXEC_OK || !result.output || result.sequence != port_sequence ||
        result.tag != seq || result.operation != S(OPERATION) || result.stage != kind) {
        output_release(s, result.output); return quarantine(s, ELPIS_CLOCK_DEVICE);
    }
    size_t size = s->c.host.buffer_size(result.output);
    const uint8_t *r = s->c.host.buffer_data(result.output);
    s->m.bytes_p2h += size;
    int good = r && size >= 128 && size <= s->c.max_output_bytes;
    if (good) {
        uint32_t rk = r[6] | (uint32_t)r[7] << 8;
        good = u32(r) == S(MAGIC) && r[4] == 1 && !r[5] && u32(r + 8) == 128 && !u32(r + 12) &&
            u64(r + 16) == s->c.stream_id && u64(r + 24) == s->c.epoch && u64(r + 32) == seq &&
            u32(r + 40) == position && u64(r + 48) == size - 128 && !memcmp(r + 56, s->c.manifest, 32);
        for (unsigned i = 88; i < 128; ++i) good = good && r[i] == 0;
        if (kind == S(STREAM_OPEN)) good = good && rk == S(OPENED) && size == 136 && u32(r + 44) == NONE;
        else if (kind == S(STREAM_RELEASE)) good = good && rk == S(RELEASED) && size == 128 && u32(r + 44) == NONE;
        else good = good && ((rk == S(NEED) && u32(r + 44) < NONE) ||
                                            (rk == S(COMPLETE) && u32(r + 44) == NONE));
    }
    if (!good) { output_release(s, result.output); return quarantine(s, ELPIS_CLOCK_INTEGRITY); }
    *out = result.output;
    return ELPIS_CLOCK_OK;
}
static void finish(clock_state *s, uint32_t outcome, elpis_clock_code code) {
    if (s->m.state == ELPIS_CLOCK_QUARANTINED) return;
    int discard = s->m.state == ELPIS_CLOCK_IN_TOKEN || s->m.state == ELPIS_CLOCK_PARKED;
    if (s->opened) {
        elpis_exec_buffer *reply = NULL;
        elpis_clock_code rc = exchange(s, S(STREAM_RELEASE), NONE, 0, &reply);
        output_release(s, reply);
        if (rc != ELPIS_CLOCK_OK) {
            quarantine(s, rc);
            /* Principal release_window never replaces the logical result.
             * Preserve completion/stop or the original host failure, while
             * recording the independent provider quarantine disposition. */
            s->m.outcome = outcome; s->m.code = code;
            return;
        }
        s->opened = 0;
        if (discard) s->m.discards++; else s->m.normal_releases++;
    }
    s->m.state = discard ? ELPIS_CLOCK_DISCARDED : ELPIS_CLOCK_RELEASED;
    s->m.outcome = outcome; s->m.code = code;
    quiesce(s);
}

elpis_clock_code elpis_dsv41_clock_hash(uint32_t current, const int64_t *tail, uint32_t order,
    uint32_t heads, uint32_t pad, const uint64_t *mult, const uint64_t *primes,
    const uint64_t *offsets, uint64_t *rows) {
    if (!tail || !mult || !primes || !offsets || !rows || order < 2 || order > 32 || !heads || heads > 1024)
        return ELPIS_CLOCK_INVALID;
    uint64_t mixed = 0; int blocked = 0;
    for (uint32_t i = 0; i < order; ++i) {
        int64_t t = i ? tail[i - 1] : (int64_t)current;
        if (t < -1 || t > INT32_MAX) return ELPIS_CLOCK_INVALID;
        blocked |= t == -1;
        uint64_t v = blocked ? pad : (uint64_t)t;
        if (!mult[i] || (v && mult[i] > INT64_MAX / v)) return ELPIS_CLOCK_INVALID;
        mixed ^= v * mult[i];
        if (i) for (uint32_t j = 0; j < heads; ++j) {
            size_t col = (size_t)(i - 1) * heads + j;
            if (!primes[col] || primes[col] > INT64_MAX || offsets[col] > INT64_MAX - primes[col])
                return ELPIS_CLOCK_INVALID;
            rows[col] = mixed % primes[col] + offsets[col];
        }
    }
    return ELPIS_CLOCK_OK;
}

static int selected_ok(clock_state *s, const uint32_t *chosen) {
    for (uint32_t i = 0; i < s->c.active_experts; ++i) {
        if (chosen[i] >= s->c.expert_count) return 0;
        for (uint32_t j = 0; j < i; ++j) if (chosen[j] == chosen[i]) return 0;
    }
    return 1;
}
static elpis_clock_code need_reply(clock_state *s, const uint8_t *r, size_t size, uint32_t header_layer) {
    uint32_t k = s->c.active_experts, chosen[64], ordered[64], need[64];
    if (size < 36 + 4 * k || u32(r + 4) != k) return ELPIS_CLOCK_INTEGRITY;
    uint32_t layer = u32(r), count = u32(r + 8 + 4 * k);
    if (layer != header_layer || layer >= s->c.layers || !count || count > k + 1 || size != 36 + 4 * k + 4 * count)
        return ELPIS_CLOCK_INTEGRITY;
    for (uint32_t i = 0; i < k; ++i) chosen[i] = ordered[i] = u32(r + 8 + 4 * i);
    if (!selected_ok(s, chosen)) return ELPIS_CLOCK_INTEGRITY;
    for (uint32_t i = 1; i < k; ++i) for (uint32_t j = i; j && ordered[j] < ordered[j - 1]; --j) {
        uint32_t tmp = ordered[j]; ordered[j] = ordered[j - 1]; ordered[j - 1] = tmp;
    }
    ordered[k] = s->c.expert_count;
    for (uint32_t i = 0; i < count; ++i) {
        need[i] = u32(r + 12 + 4 * k + 4 * i);
        if (i && need[i] <= need[i - 1]) return ELPIS_CLOCK_INTEGRITY;
    }
    const uint8_t *p = r + 12 + 4 * k + 4 * count;
    uint32_t cursor = u32(p); uint64_t offset = u64(p + 8), hits = u64(p + 16), expected_hits = 0;
    if (u32(p + 4)) return ELPIS_CLOCK_INTEGRITY;
    uint32_t found = 0;
    for (uint32_t i = 0; i <= k; ++i) {
        int requested = 0;
        for (uint32_t j = 0; j < count; ++j) requested |= need[j] == ordered[i];
        int resident = s->c.resident[(size_t)layer * (s->c.expert_count + 1) + ordered[i]] != 0;
        if (resident && requested) return ELPIS_CLOCK_INTEGRITY;
        found += (uint32_t)requested;
        if (!resident && !requested) expected_hits |= UINT64_C(1) << i;
    }
    if (found != count || hits != expected_hits || (!(s->c.features & S(FEATURE_CACHE)) && hits))
        return ELPIS_CLOCK_INTEGRITY;
    if (s->plan_layer == layer) {
        if (count != s->need_count || hits != s->hits || memcmp(chosen, s->chosen, k * 4) ||
            memcmp(need, s->need, count * 4)) return ELPIS_CLOCK_INTEGRITY;
    } else {
        if (s->plan_layer != NONE && (layer <= s->plan_layer || s->cursor != s->need_count || s->offset))
            return ELPIS_CLOCK_INTEGRITY;
        s->plan_layer = layer; s->need_count = count; s->hits = hits;
        memcpy(s->chosen, chosen, k * 4); memcpy(s->need, need, count * 4);
        memcpy(s->selected + (size_t)layer * k, chosen, k * 4); s->needed[layer] = 1;
        s->cursor = 0; s->offset = 0;
    }
    if (cursor != s->cursor || offset != s->offset || cursor >= count) return ELPIS_CLOCK_INTEGRITY;
    s->m.state = ELPIS_CLOCK_PARKED;
    return ELPIS_CLOCK_OK;
}
static elpis_clock_code complete_reply(clock_state *s, const uint8_t *p, size_t n) {
    const elpis_dsv41_clock_config_v1 *c = &s->c;
    uint32_t k = c->active_experts, owner_count = 0; int owner = 0;
    if (n != s->complete_bytes || u32(p) != c->vocab || u32(p + 4) != c->layers || u32(p + 8) != k ||
        u32(p + 12) != c->index_topk || u32(p + 16) != (c->features & S(FEATURE_OBSERVE_LAYER_STREAMS)) ||
        u32(p + 20) || (s->plan_layer != NONE && (s->cursor != s->need_count || s->offset)))
        return ELPIS_CLOCK_INTEGRITY;
    uint32_t best = 0; float best_value = f32(p + 72);
    for (uint32_t i = 0; i < c->vocab; ++i) {
        float v = f32(p + 72 + (size_t)i * 4);
        if (!isfinite(v)) return ELPIS_CLOCK_ENCODING;
        if (v > best_value) { best_value = v; best = i; }
    }
    size_t at = 72 + (size_t)c->vocab * 4;
    for (uint32_t layer = 0; layer < c->layers; ++layer) {
        uint32_t chosen[64];
        for (uint32_t i = 0; i < k; ++i) chosen[i] = u32(p + at + 4 * i);
        if (!selected_ok(s, chosen)) return ELPIS_CLOCK_INTEGRITY;
        uint32_t elided = u32(p + at + k * 4), count = u32(p + at + k * 4 + 4), npos = u32(p + at + k * 4 + 8);
        if (elided > 1 || npos > c->index_topk) return ELPIS_CLOCK_INTEGRITY;
        if (s->needed[layer]) {
            if (elided || memcmp(chosen, s->selected + (size_t)layer * k, k * 4)) return ELPIS_CLOCK_INTEGRITY;
        } else {
            if (!elided) return ELPIS_CLOCK_INTEGRITY;
            if (!(c->features & S(FEATURE_CACHE))) {
                for (uint32_t i = 0; i <= k; ++i)
                    if (!c->resident[(size_t)layer * (c->expert_count + 1) + (i == k ? c->expert_count : chosen[i])])
                        return ELPIS_CLOCK_INTEGRITY;
            }
        }
        if (c->layer_flags[layer] & ELPIS_DSV41_LAYER_KV_OWNER) { owner = 1; owner_count = count; }
        if ((!c->ratios[layer] && npos) || (c->ratios[layer] && !owner)) return ELPIS_CLOCK_INTEGRITY;
        uint32_t previous = 0;
        for (uint32_t i = 0; i < c->index_topk; ++i) {
            uint32_t pos = u32(p + at + k * 4 + 12 + 4 * i);
            if (i >= npos ? pos != 0 : pos >= owner_count || (i && pos <= previous)) return ELPIS_CLOCK_INTEGRITY;
            previous = pos;
        }
        at += k * 4 + 12 + (size_t)c->index_topk * 4;
    }
    for (; at < n; at += 4) if (!isfinite(f32(p + at))) return ELPIS_CLOCK_ENCODING;
    memcpy(s->completes + (size_t)s->m.position * n, p, n);
    s->m.argmax = best;
    return ELPIS_CLOCK_OK;
}

static elpis_clock_code service_result(clock_state *s, elpis_clock_code rc, elpis_clock_span *span,
                                       uint8_t *dest, size_t n, uint8_t *digests) {
    if (rc == ELPIS_CLOCK_OK) {
        s->m.acquires++;
        if (!span->data || span->bytes != n) rc = ELPIS_CLOCK_INTEGRITY;
        else { memcpy(dest, span->data, n); if (digests) memcpy(digests, span->digests, 96); }
        s->c.materializer.release(s->c.materializer.context, span); s->m.releases++;
    }
    if (rc == ELPIS_CLOCK_BUSY || rc == ELPIS_CLOCK_DEFER ||
        (rc == ELPIS_CLOCK_LIMIT && s->m.state == ELPIS_CLOCK_PARKED)) {
        if (!s->waiting_since) s->waiting_since = mono();
        if (mono() - s->waiting_since >= (uint64_t)s->c.materialization_timeout_ms * 1000000) return ELPIS_CLOCK_LIMIT;
        s->m.materialization_yields++; s->m.outcome = ELPIS_CLOCK_MATERIALIZATION_NEEDED;
        return ELPIS_CLOCK_DEFER;
    }
    s->waiting_since = 0;
    return rc;
}
static elpis_clock_code prepare(clock_state *s) {
    const elpis_dsv41_clock_config_v1 *c = &s->c;
    if (s->m.state == ELPIS_CLOCK_BOUNDARY) {
        s->m.token = s->m.position < c->prefill_count ? c->prefill[s->m.position] : s->m.argmax;
        s->token_start = mono(); s->row_at = 0;
        s->plan_layer = NONE; s->cursor = s->need_count = 0; s->offset = s->replies = s->supplies = 0;
        memset(s->needed, 0, c->layers);
        uint64_t *rows = s->rows + (size_t)s->m.position * s->row_count;
        for (uint32_t i = 0; i < c->engram_count; ++i) {
            elpis_clock_code rc = elpis_dsv41_clock_hash(c->token_map[s->m.token], s->tail, c->order, c->heads,
                c->pad, c->multipliers + (size_t)i * c->order, c->primes + i * s->columns,
                c->offsets + i * s->columns, rows + i * s->columns);
            if (rc != ELPIS_CLOCK_OK) return rc;
        }
        w32(s->request, s->m.token); w32(s->request + 4, c->engram_count);
        s->m.state = ELPIS_CLOCK_PREPARING;
    }
    size_t values = s->columns * c->row_dimension * 4;
    for (; s->row_at < c->engram_count; ++s->row_at) {
        uint32_t i = s->row_at;
        uint8_t *p = s->request + 8 + i * (16 + values);
        w32(p, c->engram_layers[i]); w32(p + 4, (uint32_t)s->columns); w32(p + 8, c->row_dimension); w32(p + 12, 0);
        elpis_clock_span span = {0};
        elpis_clock_code rc = c->materializer.rows(c->materializer.context, c->engram_layers[i], c->banks + i * 32,
            s->rows + (size_t)s->m.position * s->row_count + i * s->columns, s->columns, c->row_dimension, &span);
        rc = service_result(s, rc, &span, p + 16, values, NULL);
        if (rc != ELPIS_CLOCK_OK) return rc;
        for (size_t j = 0; j < values; j += 4) if (!isfinite(f32(p + 16 + j))) return ELPIS_CLOCK_ENCODING;
        if (atomic_load(&s->cancelled)) return ELPIS_CLOCK_CLOSED;
    }
    return ELPIS_CLOCK_OK;
}
static elpis_clock_code supply(clock_state *s, size_t *n) {
    size_t len = (size_t)(s->c.image_bytes - s->offset);
    if (len > s->c.part_bytes) len = (size_t)s->c.part_bytes;
    uint8_t *p = s->request;
    w32(p, s->plan_layer); w32(p + 4, s->need[s->cursor]); w32(p + 8, s->cursor); w32(p + 12, 1);
    w64(p + 16, s->c.image_bytes); w64(p + 24, s->offset); w64(p + 32, len); w64(p + 40, 0);
    elpis_clock_span span = {0};
    elpis_clock_code rc = s->c.materializer.expert(s->c.materializer.context, s->plan_layer,
        s->need[s->cursor], s->offset, len, &span);
    rc = service_result(s, rc, &span, p + S(SUPPLY_PREFIX_BYTES), len, p + 48);
    *n = S(SUPPLY_PREFIX_BYTES) + len;
    return rc;
}

static void run(clock_state *s, uint32_t budget) {
    uint32_t start_position = s->m.position;
    if (s->m.state == ELPIS_CLOCK_CREATED || s->m.state >= ELPIS_CLOCK_RELEASED) return;
    s->m.outcome = ELPIS_CLOCK_PROGRESS;
    while (s->m.position - start_position < budget) {
        if (atomic_load(&s->cancelled)) { finish(s, ELPIS_CLOCK_CANCELLED, ELPIS_CLOCK_CLOSED); return; }
        if (s->m.position == s->total_tokens) { finish(s, ELPIS_CLOCK_COMPLETE, ELPIS_CLOCK_OK); return; }
        elpis_clock_code rc; size_t n; uint32_t kind, layer = NONE;
        if (s->m.state == ELPIS_CLOCK_PARKED) {
            rc = supply(s, &n); kind = S(EXPERT_SUPPLY); layer = s->plan_layer;
        } else {
            rc = prepare(s); kind = S(TOKEN_BEGIN);
            n = 8 + s->c.engram_count * (16 + s->columns * s->c.row_dimension * 4);
        }
        if (rc == ELPIS_CLOCK_DEFER) return;
        if (rc != ELPIS_CLOCK_OK || atomic_load(&s->cancelled)) {
            finish(s, atomic_load(&s->cancelled) ? ELPIS_CLOCK_CANCELLED : ELPIS_CLOCK_FAILED,
                   atomic_load(&s->cancelled) ? ELPIS_CLOCK_CLOSED : rc); return;
        }
        elpis_exec_buffer *reply = NULL;
        rc = exchange(s, kind, layer, n, &reply);
        if (rc != ELPIS_CLOCK_OK) { finish(s, ELPIS_CLOCK_FAILED, rc); return; }
        s->m.state = ELPIS_CLOCK_IN_TOKEN;
        if (kind == S(EXPERT_SUPPLY)) {
            s->supplies++; s->offset += n - S(SUPPLY_PREFIX_BYTES);
            if (s->offset == s->c.image_bytes) { s->cursor++; s->offset = 0; }
        }
        const uint8_t *r = s->c.host.buffer_data(reply);
        size_t size = s->c.host.buffer_size(reply) - 128;
        uint32_t reply_kind = r[6];
        if (++s->replies > 1 + s->max_supplies || s->supplies > s->max_supplies) rc = ELPIS_CLOCK_INTEGRITY;
        else if (reply_kind == S(NEED)) rc = need_reply(s, r + 128, size, u32(r + 44));
        else rc = complete_reply(s, r + 128, size);
        output_release(s, reply);
        if (rc != ELPIS_CLOCK_OK) { quarantine(s, rc); return; }
        if (atomic_load(&s->cancelled)) { finish(s, ELPIS_CLOCK_CANCELLED, ELPIS_CLOCK_CLOSED); return; }
        if (reply_kind == S(NEED)) continue;
        s->tokens[s->m.position] = s->m.token;
        s->times[s->m.position] = mono() - s->token_start;
        memmove(s->tail + 1, s->tail, (s->c.order - 2) * sizeof(*s->tail));
        s->tail[0] = s->c.token_map[s->m.token];
        s->m.position++; s->m.state = ELPIS_CLOCK_BOUNDARY;
        if (s->m.position > s->c.prefill_count) {
            s->m.generated++;
            for (uint32_t i = 0; i < s->c.stop_count; ++i) if (s->c.stop_tokens[i] == s->m.token) {
                finish(s, ELPIS_CLOCK_STOP_TOKEN, ELPIS_CLOCK_OK); return;
            }
        }
        if (s->m.position == s->total_tokens) { finish(s, ELPIS_CLOCK_COMPLETE, ELPIS_CLOCK_OK); return; }
    }
}

static void free_clock(clock_state *s) {
    free(s->request); free(s->completes); free(s->needed); free(s->selected);
    free(s->rows); free(s->times); free(s->tokens); free(s->conditioning); free(s);
}
static int add_storage(uint64_t *n, uint64_t count, uint64_t unit) {
    if (count > MEMORY_LIMIT / unit || *n > MEMORY_LIMIT - count * unit) return 0;
    *n += count * unit; return 1;
}
uint32_t elpis_dsv41_clock_abi_version(void) { return ELPIS_DSV41_CLOCK_ABI_V1; }
elpis_clock_code elpis_dsv41_clock_create(const elpis_dsv41_clock_config_v1 *c, elpis_dsv41_clock *out) {
    if (!c || !out || *out) return ELPIS_CLOCK_INVALID;
    /* Never read past the layout the caller declared. */
    elpis_dsv41_clock_config_v1 v;
    memset(&v, 0, sizeof v);
    if (c->abi_version == ELPIS_DSV41_CLOCK_CONFIG_V1) memcpy(&v, c, offsetof(elpis_dsv41_clock_config_v1, conditioning));
    else if (c->abi_version == ELPIS_DSV41_CLOCK_CONFIG_V2) v = *c;
    else return ELPIS_CLOCK_INVALID;
    c = &v;
    if (c->reserved || !c->runtime ||
        c->host.abi_version != 1 || c->host.reserved || !c->host.buffer_alloc || !c->host.buffer_mutable_data ||
        !c->host.buffer_data || !c->host.buffer_size || !c->host.buffer_release || !c->host.submit ||
        !c->host.take || !c->host.metrics || !c->host.shutdown || c->materializer.abi_version != 1 ||
        c->materializer.reserved || !c->materializer.rows || !c->materializer.expert ||
        !c->materializer.release || !c->materializer.quiesce || !c->stream_id || !c->epoch ||
        !c->vocab || c->vocab > 1000000 || !c->layers || c->layers > 4096 ||
        !c->active_experts || c->active_experts > 63 || c->active_experts > c->expert_count ||
        c->expert_count > 65535 || !c->index_topk || c->index_topk > 65536 ||
        !c->dimension || c->dimension > 65536 || !c->hc_mult || c->hc_mult > 64 ||
        !c->max_tokens || c->max_tokens > 1000000 || (c->features & ~7u) || c->reserved2 ||
        (c->conditioning_count && (c->conditioning_count != c->dimension || !c->conditioning ||
                                   !(c->features & S(FEATURE_CONDITIONING)))) ||
        !c->engram_count || c->engram_count > c->layers || c->order < 2 || c->order > 32 ||
        !c->heads || c->heads > 1024 || !c->row_dimension || c->row_dimension > 65536 || c->pad > INT32_MAX ||
        !c->ratios || !c->layer_flags || !c->resident || !c->engram_layers || !c->banks || !c->token_map ||
        !c->multipliers || !c->primes || !c->offsets || !c->image_bytes || c->image_bytes > MEMORY_LIMIT ||
        !c->part_bytes || c->part_bytes > c->image_bytes || !c->memory_budget || c->memory_budget > MEMORY_LIMIT ||
        c->max_input_bytes > (64u << 20) || c->max_output_bytes > (64u << 20) ||
        !c->materialization_timeout_ms || !c->exchange_timeout_ms || !c->prefill || !c->prefill_count ||
        c->prefill_count > c->max_tokens || c->max_new_tokens > c->max_tokens - c->prefill_count ||
        c->stop_count > c->vocab || (c->stop_count && !c->stop_tokens)) return ELPIS_CLOCK_INVALID;
    for (uint32_t i = 0; i < c->conditioning_count; ++i) if (!isfinite(c->conditioning[i])) return ELPIS_CLOCK_INVALID;
    uint32_t max_id = c->pad;
    for (uint32_t i = 0; i < c->vocab; ++i) {
        if (c->token_map[i] > INT32_MAX) return ELPIS_CLOCK_INVALID;
        if (c->token_map[i] > max_id) max_id = c->token_map[i];
    }
    for (uint32_t i = 0; i < c->prefill_count; ++i) if (c->prefill[i] >= c->vocab) return ELPIS_CLOCK_INVALID;
    for (uint32_t i = 0; i < c->stop_count; ++i) if (c->stop_tokens[i] >= c->vocab) return ELPIS_CLOCK_INVALID;
    size_t columns = (size_t)(c->order - 1) * c->heads;
    uint32_t engram = 0; int owner = 0;
    for (uint32_t i = 0; i < c->layers; ++i) {
        if (c->layer_flags[i] & ~7u || c->ratios[i] > c->max_tokens) return ELPIS_CLOCK_INVALID;
        if (c->layer_flags[i] & ELPIS_DSV41_LAYER_KV_OWNER) owner = 1;
        if (c->ratios[i] && !owner) return ELPIS_CLOCK_INVALID;
        if (c->layer_flags[i] & ELPIS_DSV41_LAYER_ENGRAM) {
            if (engram == c->engram_count || c->engram_layers[engram++] != i) return ELPIS_CLOCK_INVALID;
        }
    }
    if (engram != c->engram_count) return ELPIS_CLOCK_INVALID;
    for (size_t i = 0; i < (size_t)c->engram_count * c->order; ++i)
        if (!c->multipliers[i] || !(c->multipliers[i] & 1) ||
            c->multipliers[i] > INT64_MAX / ((uint64_t)max_id + 1)) return ELPIS_CLOCK_INVALID;
    for (size_t i = 0; i < (size_t)c->engram_count * columns; ++i)
        if (!c->primes[i] || c->primes[i] > INT64_MAX || c->offsets[i] > INT64_MAX - c->primes[i])
            return ELPIS_CLOCK_INVALID;
    uint64_t complete64 = 72 + (uint64_t)c->vocab * 4 + c->layers *
        (4 * c->active_experts + 12 + (uint64_t)4 * c->index_topk);
    if (c->features & S(FEATURE_OBSERVE_LAYER_STREAMS)) complete64 += (uint64_t)4 * c->layers * c->hc_mult * c->dimension;
    uint64_t request64 = 8 + c->engram_count * (16 + (uint64_t)columns * c->row_dimension * 4);
    if (request64 < S(SUPPLY_PREFIX_BYTES) + c->part_bytes) request64 = S(SUPPLY_PREFIX_BYTES) + c->part_bytes;
    if (request64 < 16 + (uint64_t)4 * c->conditioning_count) request64 = 16 + (uint64_t)4 * c->conditioning_count;
    if (complete64 > MEMORY_LIMIT || request64 > MEMORY_LIMIT)
        return ELPIS_CLOCK_LIMIT;
    size_t complete = (size_t)complete64, request = (size_t)request64;
    size_t total = c->prefill_count + c->max_new_tokens, row_count = columns * c->engram_count;
    uint64_t bytes = sizeof(clock_state);
    if (request + 128 > c->max_input_bytes || complete + 128 > c->max_output_bytes ||
        !add_storage(&bytes, total, complete) || !add_storage(&bytes, total, row_count * 8) ||
        !add_storage(&bytes, total, 12) || !add_storage(&bytes, c->layers, 1 + 4 * c->active_experts) ||
        !add_storage(&bytes, 1, request) || !add_storage(&bytes, c->conditioning_count, 4) ||
        bytes > c->memory_budget) return ELPIS_CLOCK_LIMIT;
    clock_state *s = calloc(1, sizeof(*s));
    if (!s) return ELPIS_CLOCK_LIMIT;
    s->c = *c; s->c.conditioning = NULL; s->columns = columns; s->row_count = row_count; s->complete_bytes = complete;
    s->request_bytes = request; s->total_tokens = total;
    s->request = malloc(request); s->completes = malloc(total * complete); s->rows = calloc(total * row_count, 8);
    s->tokens = calloc(total, 4); s->times = calloc(total, 8);
    s->selected = calloc((size_t)c->layers * c->active_experts, 4); s->needed = calloc(c->layers, 1);
    if (c->conditioning_count) s->conditioning = malloc((size_t)c->conditioning_count * 4);
    if (!s->request || !s->completes || !s->rows || !s->tokens || !s->times || !s->selected || !s->needed ||
        (c->conditioning_count && !s->conditioning)) {
        free_clock(s); return ELPIS_CLOCK_LIMIT;
    }
    if (c->conditioning_count) memcpy(s->conditioning, c->conditioning, (size_t)c->conditioning_count * 4);
    for (uint32_t i = 0; i < c->order - 1; ++i) s->tail[i] = -1;
    s->m.sequence = c->sequence; s->m.storage_bytes = bytes;
    s->max_supplies = (uint64_t)c->layers * (c->active_experts + 1) * (1 + (c->image_bytes - 1) / c->part_bytes);
    atomic_init(&s->cancelled, 0);
    pthread_mutex_lock(&handles_mu);
    for (unsigned i = 0; i < MAX_CLOCKS; ++i) if (handles[i] && handles[i]->c.runtime == c->runtime) {
        pthread_mutex_unlock(&handles_mu); free_clock(s); return ELPIS_CLOCK_BUSY;
    }
    for (unsigned i = 0; i < MAX_CLOCKS && next_id < UINT64_MAX; ++i) if (!handles[i]) {
        s->id = ++next_id; handles[i] = s; *out = s->id;
        pthread_mutex_unlock(&handles_mu); return ELPIS_CLOCK_OK;
    }
    pthread_mutex_unlock(&handles_mu); free_clock(s); return ELPIS_CLOCK_LIMIT;
}
elpis_clock_code elpis_dsv41_clock_open(elpis_dsv41_clock id) {
    clock_state *s; elpis_clock_code rc = pin(id, &s, 0);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (s->m.state != ELPIS_CLOCK_CREATED) rc = ELPIS_CLOCK_STALE;
    else if (atomic_load(&s->cancelled)) finish(s, ELPIS_CLOCK_CANCELLED, ELPIS_CLOCK_CLOSED);
    else {
        uint32_t flags = s->c.features & S(FEATURE_OBSERVE_LAYER_STREAMS), body = 8, n = s->c.conditioning_count;
        if (n) {  /* frozen turn conditioning, once per stream */
            flags |= S(FEATURE_CONDITIONING);
            w32(s->request + 8, n); w32(s->request + 12, 0);
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t bits; memcpy(&bits, s->conditioning + i, 4); w32(s->request + 16 + 4 * (size_t)i, bits);
            }
            body = 16 + 4 * n;
        }
        w32(s->request, s->c.max_tokens); w32(s->request + 4, flags);
        elpis_exec_buffer *reply = NULL;
        rc = exchange(s, S(STREAM_OPEN), NONE, body, &reply);
        output_release(s, reply);
        if (rc == ELPIS_CLOCK_OK) { s->opened = 1; s->m.state = ELPIS_CLOCK_BOUNDARY; }
        else if (s->m.state != ELPIS_CLOCK_QUARANTINED) finish(s, ELPIS_CLOCK_FAILED, rc);
    }
    unpin(s); return rc;
}
elpis_clock_code elpis_dsv41_clock_advance(elpis_dsv41_clock id, uint32_t budget, elpis_dsv41_clock_metrics_v1 *out) {
    if (!out || !budget) return ELPIS_CLOCK_INVALID;
    clock_state *s; elpis_clock_code rc = pin(id, &s, 0);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (s->m.state == ELPIS_CLOCK_CREATED) rc = ELPIS_CLOCK_STALE;
    else { uint64_t start = mono(); run(s, budget); s->m.advance_ns += mono() - start; *out = s->m; }
    unpin(s); return rc;
}
elpis_clock_code elpis_dsv41_clock_cancel(elpis_dsv41_clock id) {
    pthread_mutex_lock(&handles_mu);
    clock_state *s = find(id);
    if (s) atomic_store(&s->cancelled, 1);
    pthread_mutex_unlock(&handles_mu);
    return s ? ELPIS_CLOCK_OK : ELPIS_CLOCK_STALE;
}
elpis_clock_code elpis_dsv41_clock_metrics(elpis_dsv41_clock id, elpis_dsv41_clock_metrics_v1 *out) {
    if (!out) return ELPIS_CLOCK_INVALID;
    clock_state *s; elpis_clock_code rc = pin(id, &s, 0);
    if (rc == ELPIS_CLOCK_OK) { *out = s->m; unpin(s); }
    return rc;
}
elpis_clock_code elpis_dsv41_clock_trace(elpis_dsv41_clock id, uint32_t pos, elpis_dsv41_clock_trace_v1 *out) {
    if (!out) return ELPIS_CLOCK_INVALID;
    clock_state *s; elpis_clock_code rc = pin(id, &s, 0);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (pos >= s->m.position) rc = ELPIS_CLOCK_INVALID;
    else *out = (elpis_dsv41_clock_trace_v1){s->tokens[pos], pos, s->rows + (size_t)pos * s->row_count,
        s->completes + (size_t)pos * s->complete_bytes, s->complete_bytes, s->times[pos]};
    unpin(s); return rc;
}
elpis_clock_code elpis_dsv41_clock_close(elpis_dsv41_clock id) {
    clock_state *s; elpis_clock_code rc = pin(id, &s, 1);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (s->m.state < ELPIS_CLOCK_RELEASED) finish(s, ELPIS_CLOCK_CANCELLED, ELPIS_CLOCK_CLOSED);
    unpin(s); return ELPIS_CLOCK_OK;
}
elpis_clock_code elpis_dsv41_clock_stop(elpis_dsv41_clock id) {
    clock_state *s; elpis_clock_code rc = pin(id, &s, 0);
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (s->m.state == ELPIS_CLOCK_BOUNDARY) finish(s, ELPIS_CLOCK_STOPPED, ELPIS_CLOCK_OK);
    else if (s->m.state < ELPIS_CLOCK_RELEASED) rc = ELPIS_CLOCK_BUSY;
    unpin(s); return rc;
}
elpis_clock_code elpis_dsv41_clock_destroy(elpis_dsv41_clock id) {
    clock_state *s; elpis_clock_code rc = pin(id, &s, 1);
    if (rc == ELPIS_CLOCK_STALE) {
        pthread_mutex_lock(&handles_mu); int old = id && id <= next_id; pthread_mutex_unlock(&handles_mu);
        return old ? ELPIS_CLOCK_OK : ELPIS_CLOCK_STALE;
    }
    if (rc != ELPIS_CLOCK_OK) return rc;
    if (s->m.state < ELPIS_CLOCK_RELEASED) finish(s, ELPIS_CLOCK_CANCELLED, ELPIS_CLOCK_CLOSED);
    pthread_mutex_lock(&handles_mu);
    for (unsigned i = 0; i < MAX_CLOCKS; ++i) if (handles[i] == s) handles[i] = NULL;
    pthread_mutex_unlock(&handles_mu); free_clock(s); return ELPIS_CLOCK_OK;
}
