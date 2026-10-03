/* Reuse the canonical reference-provider admission fixture verbatim. Its main
 * remains available but is not rerun here (it has its own CTest target). */
#define main yts_protocol_regression_main
#include "test_dsv41_stream_provider.c"
#undef main
#include "dsv41_clock_materializer.h"
#include <pthread.h>

static const elpis_dsv41_clock_host_v1 CLOCK_HOST = {1, 0, elpis_exec_buffer_alloc,
    elpis_exec_buffer_mutable_data, elpis_exec_buffer_data, elpis_exec_buffer_size, elpis_exec_buffer_release,
    elpis_exec_submit, elpis_exec_take, elpis_exec_get_metrics, elpis_exec_shutdown};
typedef struct {
    session provider;
    elpis_dsv41_clock clock;
    elpis_dsv41_clock_config_v1 c;
    clock_test_materializer mat;
    clock_test_bank bank;
    clock_test_expert experts[NL * (E + 1)];
    float images[NL * (E + 1)][3 * F * D], values[128][ED];
    uint8_t scratch[HASH * ED * 4], resident[NL * (E + 1)];
    uint32_t mapping[V], engram_layer;
    uint64_t multipliers[EO], primes[HASH], offsets[HASH];
} fixture;

static void setup(fixture *f, size_t part, yts_ref_config faults, uint64_t cache) {
    memset(f, 0, sizeof(*f));
    attach(&f->provider, faults, faults.fault_kind == FAULT_POLL_TIMEOUT ? 3 : 1000);
    f->provider.part_bytes = part;
    assert(admit(&f->provider, cache ? 1 : 0, cache, 777) == ELPIS_EXEC_OK);
    uint64_t rng = 128;
    for (unsigned r = 0; r < 128; ++r) for (unsigned d = 0; d < ED; ++d) f->values[r][d] = unit(&rng);
    f->bank = (clock_test_bank){.layer=1, .dimension=ED, .rows=128, .values=(uint8_t *)f->values};
    for (unsigned l = 0; l < NL; ++l) for (unsigned e = 0; e <= E; ++e) {
        unsigned i = l * (E + 1) + e;
        expert_image(l, e, f->images[i]);
        f->experts[i] = (clock_test_expert){.layer=l, .expert=e, .bytes=sizeof(f->images[i]),
                                          .image=(uint8_t *)f->images[i]};
        for (unsigned r = 0; r < 3; ++r) memset(f->experts[i].digests + r * 32, (int)(l * 16 + e * 3 + r), 32);
        f->resident[i] = l == RESIDENT_LAYER;
    }
    f->mat.banks = &f->bank; f->mat.bank_count = 1; f->mat.experts = f->experts;
    f->mat.expert_count = NL * (E + 1); f->mat.scratch = f->scratch; f->mat.scratch_bytes = sizeof(f->scratch);
    f->mat.cancel = elpis_dsv41_clock_cancel;
    for (unsigned i = 0; i < V; ++i) f->mapping[i] = i;
    f->engram_layer = 1;
    for (unsigned i = 0; i < EO; ++i) f->multipliers[i] = 11 + 2 * i;
    for (unsigned i = 0; i < HASH; ++i) { f->primes[i] = 17; f->offsets[i] = i * 17; }
    f->c = (elpis_dsv41_clock_config_v1){.abi_version=1, .runtime=f->provider.r, .host=CLOCK_HOST,
        .sequence=f->provider.seq, .stream_id=1, .epoch=17, .vocab=V, .layers=NL, .active_experts=ACT,
        .expert_count=E, .index_topk=K, .dimension=D, .hc_mult=HC, .max_tokens=T, .features=cache ? 1u : 0u,
        .engram_count=1, .order=EO, .heads=EH, .row_dimension=ED, .ratios=RATIO, .layer_flags=FLAGS,
        .resident=f->resident, .engram_layers=&f->engram_layer, .banks=f->bank.bank, .token_map=f->mapping,
        .multipliers=f->multipliers, .primes=f->primes, .offsets=f->offsets, .image_bytes=sizeof(f->images[0]),
        .part_bytes=part, .memory_budget=1 << 20, .max_input_bytes=1 << 20, .max_output_bytes=1 << 20,
        .materialization_timeout_ms=5, .exchange_timeout_ms=10000, .prefill=TOKS, .prefill_count=2,
        .max_new_tokens=6};
    memcpy(f->c.manifest, MANIFEST, 32);
    elpis_dsv41_clock_test_materializer(&f->mat, &f->c.materializer);
}
static void create_open(fixture *f) {
    assert(elpis_dsv41_clock_create(&f->c, &f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_open(f->clock) == ELPIS_CLOCK_OK);
}
static void teardown(fixture *f) {
    elpis_dsv41_clock_metrics_v1 m;
    assert(elpis_dsv41_clock_close(f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_close(f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_metrics(f->clock, &m) == ELPIS_CLOCK_OK);
    assert(m.allocations == m.consumed + m.requests_released);
    assert(elpis_dsv41_clock_destroy(f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_destroy(f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_metrics(f->clock, &m) == ELPIS_CLOCK_STALE);
    assert(elpis_dsv41_clock_cancel(f->clock) == ELPIS_CLOCK_STALE);
    assert(f->mat.live == 0 && f->mat.acquires == f->mat.releases && f->mat.quiesces == 1);
    elpis_exec_metrics port;
    elpis_exec_get_metrics(f->provider.r, &port);
    assert(port.backend_fallback == 0 && port.outstanding == 0);
    detach(&f->provider); counters_released();
}
static void check_recurrence(void) {
    float baseline[8][V]; uint32_t tokens[8];
    for (unsigned mode = 0; mode < 4; ++mode) {
        fixture f;
        setup(&f, mode == 1 ? 128 : 1152, (yts_ref_config){mode == 3, 1, 0, 0, -1}, mode == 2 ? 1000000 : 0);
        create_open(&f);
        elpis_dsv41_clock_metrics_v1 m;
        do { assert(elpis_dsv41_clock_advance(f.clock, mode == 1 ? 1 : T, &m) == ELPIS_CLOCK_OK); }
        while (m.outcome == ELPIS_CLOCK_PROGRESS);
        assert(m.outcome == ELPIS_CLOCK_COMPLETE && m.position == 8 && m.generated == 6);
        assert(m.acquires == m.releases && m.allocations == m.consumed && m.consumed == m.outputs_released);
        for (uint32_t i = 0; i < 8; ++i) {
            elpis_dsv41_clock_trace_v1 t;
            assert(elpis_dsv41_clock_trace(f.clock, i, &t) == ELPIS_CLOCK_OK);
            if (!mode) { memcpy(baseline[i], t.complete + 72, V * 4); tokens[i] = t.token; }
            else { assert(!memcmp(baseline[i], t.complete + 72, V * 4)); assert(tokens[i] == t.token); }
        }
        teardown(&f);
    }
}
static void check_host_boundaries(void) {
    for (unsigned scenario = 0; scenario < 9; ++scenario) {
        fixture f;
        setup(&f, 128, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        if (scenario == 0) f.c.max_new_tokens = 0;
        create_open(&f);
        f.mat.fault_call = scenario == 1 ? 1 : 2;
        if (scenario == 1 || scenario == 2) f.mat.fault_code = ELPIS_CLOCK_IO;
        if (scenario == 3 || scenario == 4) {
            f.mat.fault_code = ELPIS_CLOCK_DEFER; f.mat.fault_repeat = scenario == 4;
        }
        if (scenario == 5) assert(elpis_dsv41_clock_cancel(f.clock) == ELPIS_CLOCK_OK);
        if (scenario == 6 || scenario == 7) {
            f.mat.cancel_clock = f.clock; f.mat.fault_call = scenario == 6 ? 1 : 2;
        }
        elpis_dsv41_clock_metrics_v1 m;
        assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        if (scenario == 3 || scenario == 4) {
            assert(m.outcome == ELPIS_CLOCK_MATERIALIZATION_NEEDED && m.position == 0);
            elpis_exec_metrics port; elpis_exec_get_metrics(f.provider.r, &port); assert(!port.outstanding);
            if (scenario == 4) { struct timespec ts = {0, 10000000}; nanosleep(&ts, NULL); }
            assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        }
        if (scenario == 0) assert(m.position == 2 && m.generated == 0 && m.outcome == ELPIS_CLOCK_COMPLETE);
        if (scenario == 1) assert(m.state == ELPIS_CLOCK_RELEASED && m.code == ELPIS_CLOCK_IO);
        if (scenario == 2 || scenario == 4) assert(m.state == ELPIS_CLOCK_DISCARDED && m.outcome == ELPIS_CLOCK_FAILED);
        if (scenario == 3 || scenario == 8) assert(m.outcome == ELPIS_CLOCK_COMPLETE);
        if (scenario >= 5 && scenario <= 7) assert(m.outcome == ELPIS_CLOCK_CANCELLED && m.position == 0);
        if (scenario == 7) assert(m.state == ELPIS_CLOCK_DISCARDED);
        yts_ref_counters cnt; elpis_dsv41_reference_provider_counters(&cnt);
        assert(cnt.models == 1 && cnt.streams == 0 && !cnt.tokens);
        teardown(&f);
    }
}
static void check_faults(void) {
    fixture base;
    setup(&base, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
    uint64_t admission = base.provider.seq;
    create_open(&base); teardown(&base);
    /* Open, TOKEN_BEGIN, every one of the 9 supply messages, and release. */
    for (unsigned kind = 1; kind <= 7; ++kind) for (unsigned message = 0; message < 12; ++message) {
        if ((kind == FAULT_BAD_NEED || kind == FAULT_BAD_CURSOR) && (message == 0 || message >= 10)) continue;
        if (kind == FAULT_NONFINITE && message != 10) continue;
        fixture f;
        setup(&f, 1152, (yts_ref_config){0, 0, 0, kind, (int64_t)(admission + message)}, 0);
        f.c.prefill_count = 1; f.c.max_new_tokens = 0;
        assert(elpis_dsv41_clock_create(&f.c, &f.clock) == ELPIS_CLOCK_OK);
        elpis_clock_code rc = elpis_dsv41_clock_open(f.clock);
        elpis_dsv41_clock_metrics_v1 m;
        if (rc == ELPIS_CLOCK_OK) assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        assert(elpis_dsv41_clock_metrics(f.clock, &m) == ELPIS_CLOCK_OK);
        assert(m.state == ELPIS_CLOCK_QUARANTINED);
        assert(m.provider_code == (kind <= 3 ? ELPIS_CLOCK_DEVICE : kind == 6 ? ELPIS_CLOCK_ENCODING : ELPIS_CLOCK_INTEGRITY));
        if (message == 11) assert(m.outcome == ELPIS_CLOCK_COMPLETE && m.code == ELPIS_CLOCK_OK && m.position == 1);
        else assert(m.outcome == ELPIS_CLOCK_FAILED && m.code == m.provider_code);
        yts_ref_counters cnt; elpis_dsv41_reference_provider_counters(&cnt);
        assert(!cnt.models && !cnt.streams && !cnt.slots && !cnt.cache_entries && !cnt.tokens);
        teardown(&f);
    }
}
static void check_limits_stop(void) {
    fixture f;
    setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
    f.c.memory_budget = 1;
    assert(elpis_dsv41_clock_create(&f.c, &f.clock) == ELPIS_CLOCK_LIMIT && !f.clock);
    f.c.memory_budget = 1 << 20; f.c.max_output_bytes = 128;
    assert(elpis_dsv41_clock_create(&f.c, &f.clock) == ELPIS_CLOCK_LIMIT && !f.clock);
    f.c.max_output_bytes = 1 << 20; f.c.max_new_tokens = T;
    assert(elpis_dsv41_clock_create(&f.c, &f.clock) == ELPIS_CLOCK_INVALID && !f.clock);
    f.c.max_new_tokens = 6;
    /* All vocabulary tokens are stops: prefill still completes, then exactly
     * one generated token is fully advanced and published. */
    f.c.stop_tokens = f.mapping; f.c.stop_count = V;
    create_open(&f);
    elpis_dsv41_clock_metrics_v1 m;
    assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
    assert(m.outcome == ELPIS_CLOCK_STOP_TOKEN && m.position == 3 && m.generated == 1);
    teardown(&f);
}

/* Native-only boundary injection: mutate a COPY of a retired sealed reply.
 * This tests the host parser independently of reference-provider fault hooks. */
static unsigned corrupt_offset, corrupt_size, fail_alloc, corrupt_stage, corrupt_kind;
static elpis_exec_buffer *clock_alloc(size_t n) {
    if (fail_alloc) { --fail_alloc; return NULL; }
    return elpis_exec_buffer_alloc(n);
}
static elpis_exec_status corrupt_take(elpis_exec_runtime *r, unsigned ms, elpis_exec_result *out) {
    elpis_exec_status rc = elpis_exec_take(r, ms, out);
    if (rc == ELPIS_EXEC_OK && out->status == ELPIS_EXEC_OK && out->stage == corrupt_stage &&
        (!corrupt_kind || ((const uint8_t *)elpis_exec_buffer_data(out->output))[6] == corrupt_kind)) {
        size_t n = corrupt_size ? corrupt_size : elpis_exec_buffer_size(out->output);
        elpis_exec_buffer *copy = elpis_exec_buffer_alloc(n);
        assert(copy);
        uint8_t *p = elpis_exec_buffer_mutable_data(copy);
        memcpy(p, elpis_exec_buffer_data(out->output), n);
        if (!corrupt_size) p[corrupt_offset] ^= 0x80;
        elpis_exec_buffer_release(out->output); out->output = copy;
    }
    return rc;
}
static void check_parser_allocation(void) {
    const unsigned offsets[] = {0, 4, 6, 8, 12, 16, 24, 32, 40, 44, 48, 56, 88, 127,
                                128, 132, 136, 144, 160, 164, 168, 176};
    for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]) + 2; ++i) {
        fixture f;
        setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        corrupt_stage = ELPIS_DSV41_STREAM_TOKEN_BEGIN;
        corrupt_size = i >= sizeof(offsets) / sizeof(offsets[0]) ? (i & 1 ? 8 : 128) : 0;
        corrupt_offset = corrupt_size ? 0 : offsets[i];
        f.c.host.take = corrupt_take;
        create_open(&f);
        elpis_dsv41_clock_metrics_v1 m;
        assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        assert(m.state == ELPIS_CLOCK_QUARANTINED && m.code == ELPIS_CLOCK_INTEGRITY);
        teardown(&f);
    }
    const unsigned complete_offsets[] = {128, 132, 136, 140, 144, 148,
        128 + 72 + V * 4, 128 + 72 + V * 4 + ACT * 4,
        128 + 72 + V * 4 + ACT * 4 + 8};
    for (unsigned i = 0; i < sizeof(complete_offsets) / sizeof(complete_offsets[0]); ++i) {
        fixture f;
        setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        corrupt_stage = ELPIS_DSV41_STREAM_EXPERT_SUPPLY; corrupt_kind = ELPIS_DSV41_STREAM_COMPLETE;
        corrupt_size = 0; corrupt_offset = complete_offsets[i]; f.c.host.take = corrupt_take;
        create_open(&f);
        elpis_dsv41_clock_metrics_v1 m;
        assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        assert(m.state == ELPIS_CLOCK_QUARANTINED && m.code == ELPIS_CLOCK_INTEGRITY && m.position == 0);
        teardown(&f);
    }
    corrupt_kind = 0;
    for (unsigned phase = 0; phase < 2; ++phase) {
        fixture f;
        setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        f.c.host.buffer_alloc = clock_alloc;
        assert(elpis_dsv41_clock_create(&f.c, &f.clock) == ELPIS_CLOCK_OK);
        if (!phase) {
            fail_alloc = 1;
            assert(elpis_dsv41_clock_open(f.clock) == ELPIS_CLOCK_LIMIT);
        } else {
            assert(elpis_dsv41_clock_open(f.clock) == ELPIS_CLOCK_OK);
            fail_alloc = 1;
            elpis_dsv41_clock_metrics_v1 m;
            assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
            assert(m.state == ELPIS_CLOCK_RELEASED && m.code == ELPIS_CLOCK_LIMIT && !m.discards);
        }
        teardown(&f);
    }
}
typedef struct {
    elpis_dsv41_clock clock;
    elpis_dsv41_clock_metrics_v1 result;
} advancing;
static void *advance_thread(void *v) {
    advancing *a = v;
    assert(elpis_dsv41_clock_advance(a->clock, T, &a->result) == ELPIS_CLOCK_OK);
    return NULL;
}
static void check_concurrent_cancel(void) {
    fixture f, map;
    setup(&map, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
    uint64_t first_token = map.provider.seq + 1;
    create_open(&map); teardown(&map);
    setup(&f, 1152, (yts_ref_config){1, 1, 100000, FAULT_SLOW, (int64_t)first_token}, 0);
    create_open(&f);
    advancing a = {.clock=f.clock}; pthread_t thread;
    assert(!pthread_create(&thread, NULL, advance_thread, &a));
    yts_ref_counters cnt;
    uint64_t until = now_ns() + 1000000000;
    do {
        elpis_dsv41_reference_provider_counters(&cnt);
        if (cnt.tokens) break;
        struct timespec ts = {0, 100000}; nanosleep(&ts, NULL);
    } while (now_ns() < until);
    assert(cnt.tokens);
    assert(elpis_dsv41_clock_close(f.clock) == ELPIS_CLOCK_BUSY);
    assert(elpis_dsv41_clock_destroy(f.clock) == ELPIS_CLOCK_BUSY);
    assert(elpis_dsv41_clock_cancel(f.clock) == ELPIS_CLOCK_OK);
    assert(!pthread_join(thread, NULL));
    assert(a.result.outcome == ELPIS_CLOCK_CANCELLED && a.result.state == ELPIS_CLOCK_DISCARDED);
    assert(a.result.position == 0);
    teardown(&f);
    setup(&f, 1152, (yts_ref_config){1, 1, 50000, FAULT_SLOW, (int64_t)first_token}, 0);
    f.c.exchange_timeout_ms = 1;
    create_open(&f);
    elpis_dsv41_clock_metrics_v1 m;
    assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
    assert(m.state == ELPIS_CLOCK_QUARANTINED && m.code == ELPIS_CLOCK_DEVICE && m.position == 0);
    assert(m.polls > 0 && m.allocations == m.consumed && m.outputs_released == m.consumed);
    teardown(&f);
}
static void check_handle_capacity(void) {
    fixture f;
    setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
    elpis_exec_runtime *r[65] = {0}; elpis_dsv41_clock clocks[65] = {0};
    elpis_exec_config port = {1, 1, 1 << 20, 1 << 20, 0, NULL};
    for (unsigned i = 0; i < 65; ++i) {
        assert(elpis_exec_create(&port, &r[i]) == ELPIS_EXEC_OK);
        f.c.runtime = r[i];
        /* No open/admission: exercise cold storage and handle allocation only. */
        assert(elpis_dsv41_clock_create(&f.c, &clocks[i]) == (i < 64 ? ELPIS_CLOCK_OK : ELPIS_CLOCK_LIMIT));
    }
    for (unsigned i = 0; i < 65; ++i) {
        if (clocks[i]) assert(elpis_dsv41_clock_destroy(clocks[i]) == ELPIS_CLOCK_OK);
        elpis_exec_destroy(r[i]);
    }
    assert(f.mat.quiesces == 64 && !f.mat.live);
    detach(&f.provider); counters_released();
}
static elpis_exec_status rejection;
static elpis_exec_status reject_submit(elpis_exec_runtime *r, const elpis_exec_task *t,
                                      elpis_exec_buffer **b, uint64_t *seq) {
    if (t->stage == ELPIS_DSV41_STREAM_TOKEN_BEGIN) return rejection;
    return elpis_exec_submit(r, t, b, seq);
}
static void check_port_rejection(void) {
    const elpis_exec_status failures[] = {ELPIS_EXEC_CLOSED, ELPIS_EXEC_WOULD_BLOCK,
                                         ELPIS_EXEC_BACKEND_UNAVAILABLE, ELPIS_EXEC_INVALID};
    for (unsigned i = 0; i < sizeof(failures) / sizeof(failures[0]); ++i) {
        fixture f;
        setup(&f, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        f.c.host.submit = reject_submit; rejection = failures[i]; create_open(&f);
        elpis_dsv41_clock_metrics_v1 m;
        assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        assert(m.state == ELPIS_CLOCK_QUARANTINED && m.position == 0 && m.submissions == 1);
        assert(m.code == (rejection == ELPIS_EXEC_CLOSED ? ELPIS_CLOCK_CLOSED : ELPIS_CLOCK_DEVICE));
        assert(m.requests_released == 1);
        teardown(&f);
    }
}
/* test_dsv41_clock_production.c reuses this fixture with its own main. */
#ifdef DSV41_CLOCK_FIXTURE_ONLY
#define DSV41_CLOCK_MAIN native_clock_r0_main
#else
#define DSV41_CLOCK_MAIN main
#endif
int DSV41_CLOCK_MAIN(void) {
    check_recurrence(); check_host_boundaries(); check_faults(); check_limits_stop();
    check_parser_allocation(); check_concurrent_cancel(); check_handle_capacity();
    check_port_rejection();
    puts("PASS native clock: recurrence, chunk/cache/async invariance, host/provider faults, stop, bounds, stale handles");
    return 0;
}
