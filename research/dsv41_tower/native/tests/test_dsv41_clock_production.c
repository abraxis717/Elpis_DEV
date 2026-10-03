/* Native Clock + production Native Materializer R1, end to end over the real
 * execution runtime and the test-only reference provider. Every run is
 * compared bitwise with the R0 test materializer serving identical decoded
 * bytes from memory. Reuses the R0 clock fixture verbatim. */
#define DSV41_CLOCK_FIXTURE_ONLY 1
#include "test_dsv41_clock.c"
#include "elpis/dsv41_materializer.h"
#include <fcntl.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#define BANK_PAGE 512u
#define EXPERT_PAGE 256u
#define IMAGE_BYTES (3u * F * D * 4u)

static ino_t expert_inode;
static elpis_dsv41_clock cancel_target;
static int cancel_armed;
static long cancel_pread(int fd, void *buf, size_t n, int64_t off) {
    struct stat st;
    if (cancel_armed && fstat(fd, &st) == 0 && st.st_ino == expert_inode) {
        cancel_armed = 0; elpis_dsv41_clock_cancel(cancel_target); /* cancellation after acquisition began */
    }
    return (long)pread(fd, buf, n, (off_t)off);
}

static void write_all(const char *name, const void *data, size_t n) {
    FILE *f = fopen(name, "wb");
    assert(f && fwrite(data, 1, n, f) == n && fclose(f) == 0);
}
static uint32_t admit_file(elpis_dsv41_materializer id, const char *name, uint32_t page, int corrupt) {
    int fd = open(name, O_RDONLY | O_CLOEXEC);
    struct stat st;
    assert(fd >= 0 && fstat(fd, &st) == 0);
    uint64_t size = (uint64_t)st.st_size, count = (size + page - 1) / page;
    uint8_t *map = malloc(count * 32), *buf = malloc(page);
    for (uint64_t p = 0; p < count; ++p) {
        size_t n = (size_t)(p + 1 == count ? size - p * page : page);
        assert(pread(fd, buf, n, (off_t)(p * page)) == (ssize_t)n);
        elpis_file_service_raw_digest(buf, n, map + p * 32);
    }
    for (uint64_t p = 0; corrupt == -2 && p < count; ++p) map[p * 32 + 5] ^= 0x10; /* every pinned page wrong */
    elpis_file_asset_v1 a = {ELPIS_FILE_SERVICE_ABI_V1, page, fd, 0, size, count, {0}, map};
    a.stamp = (elpis_file_stamp){(uint64_t)st.st_dev, (uint64_t)st.st_ino, size,
        (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec,
        (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec};
    uint32_t asset;
    assert(elpis_dsv41_materializer_admit_asset(id, &a, &asset) == ELPIS_CLOCK_OK);
    close(fd); free(map); free(buf);
    return asset;
}

/* fp8: the bank is stored DS4_E4M3_E8M0_BF16 on disk; the R0 test bank holds
 * the natively decoded values so both services expose identical rows. */
static void prepare_files(fixture *f, int fp8) {
    if (fp8) {
        uint8_t row[ED + ED / 32], file[128 * sizeof row];
        uint64_t rng = 4141;
        for (unsigned r = 0; r < 128; ++r) {
            for (unsigned i = 0; i < ED; ++i) {
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                row[i] = (uint8_t)((rng >> 33) % 112) | (uint8_t)((rng >> 20) & 128);
            }
            row[ED] = (uint8_t)(118 + r % 6);
            memcpy(file + r * sizeof row, row, sizeof row);
            assert(elpis_dsv41_row_decode(ELPIS_DSV41_ROW_E4M3_E8M0_BF16, row, sizeof row, ED, f->values[r]) == ELPIS_CLOCK_OK);
        }
        write_all("bank.bin", file, sizeof file);
    } else write_all("bank.bin", f->values, sizeof f->values);
    static uint8_t experts[100 + NL * (E + 1) * IMAGE_BYTES];
    memset(experts, 0x5a, 100);
    for (unsigned i = 0; i < NL * (E + 1); ++i) memcpy(experts + 100 + i * IMAGE_BYTES, f->images[i], IMAGE_BYTES);
    write_all("experts.bin", experts, sizeof experts);
    struct stat st; assert(stat("experts.bin", &st) == 0); expert_inode = st.st_ino;
}
static elpis_dsv41_materializer production(fixture *f, int fp8, int corrupt_bank, int corrupt_expert, uint64_t staging) {
    elpis_dsv41_materializer_config_v1 c = {ELPIS_DSV41_MATERIALIZER_ABI_V1, 0,
        {ELPIS_FILE_SERVICE_ABI_V1, 0, 4096, 4 * BANK_PAGE, 1 << 20, 64, 4, 4, 0},
        NL, E, IMAGE_BYTES, staging, 1, HASH, ED, 0};
    elpis_dsv41_materializer id = 0;
    assert(elpis_dsv41_materializer_create(&c, &id) == ELPIS_CLOCK_OK);
    uint32_t bank = admit_file(id, "bank.bin", BANK_PAGE, corrupt_bank);
    uint32_t experts = admit_file(id, "experts.bin", EXPERT_PAGE, corrupt_expert);
    elpis_dsv41_bank_v1 b = {f->engram_layer, ED, fp8 ? ELPIS_DSV41_ROW_E4M3_E8M0_BF16 : ELPIS_DSV41_ROW_F32_LE,
                             bank, 128, 0, {0}, HASH, 0, (uint64_t)HASH * ED * 4};
    memcpy(b.bank, f->bank.bank, 32);
    assert(elpis_dsv41_materializer_add_bank(id, &b) == ELPIS_CLOCK_OK);
    for (unsigned i = 0; i < NL * (E + 1); ++i) {
        if (f->resident[i]) continue; /* resident experts are never materialized */
        const uint64_t base = 100 + (uint64_t)i * IMAGE_BYTES, role = IMAGE_BYTES / 3;
        elpis_dsv41_expert_v1 e = {i / (E + 1), i % (E + 1), {experts, experts, experts}, 0,
                                   {base, base + role, base + 2 * role}, {role, role, role}, {0}};
        memcpy(e.digests, f->experts[i].digests, 96);
        assert(elpis_dsv41_materializer_add_expert(id, &e) == ELPIS_CLOCK_OK);
    }
    assert(elpis_dsv41_materializer_seal(id) == ELPIS_CLOCK_OK);
    elpis_dsv41_materializer_bind((void *)(uintptr_t)id, &f->c.materializer);
    return id;
}
static elpis_dsv41_materializer_stats_v1 mstats(elpis_dsv41_materializer id) {
    elpis_dsv41_materializer_stats_v1 s;
    assert(elpis_dsv41_materializer_stats(id, &s) == ELPIS_CLOCK_OK);
    return s;
}
/* Resource closure: clock buffers, spans, FMS leases, provider stream. */
static void finish_production(fixture *f, elpis_dsv41_materializer id, unsigned quiesces) {
    elpis_dsv41_clock_metrics_v1 m;
    assert(elpis_dsv41_clock_close(f->clock) == ELPIS_CLOCK_OK && elpis_dsv41_clock_close(f->clock) == ELPIS_CLOCK_OK);
    assert(elpis_dsv41_clock_metrics(f->clock, &m) == ELPIS_CLOCK_OK);
    assert(m.allocations == m.consumed + m.requests_released && m.acquires == m.releases);
    assert(elpis_dsv41_clock_destroy(f->clock) == ELPIS_CLOCK_OK && elpis_dsv41_clock_destroy(f->clock) == ELPIS_CLOCK_OK);
    if (id) {
        elpis_dsv41_materializer_stats_v1 s = mstats(id);
        assert(s.live_spans == 0 && s.file.pinned_bytes == 0 && s.file.live_ranges == 0);
        assert(s.file.lease_acquires == s.file.lease_releases && s.quiesces == quiesces);
        assert(s.span_acquires == m.acquires);
        assert(s.file.resident_high_water <= 4096 && s.staging_high_water <= s.staging_budget);
        assert(elpis_dsv41_materializer_destroy(id) == ELPIS_CLOCK_OK);
        assert(elpis_dsv41_materializer_destroy(id) == ELPIS_CLOCK_OK);
    }
    elpis_exec_metrics port;
    elpis_exec_get_metrics(f->provider.r, &port);
    assert(port.backend_fallback == 0 && port.outstanding == 0);
    detach(&f->provider); counters_released();
}

typedef struct { uint32_t tokens[8]; uint8_t complete[8][4096]; uint64_t rows[8][HASH]; size_t bytes; } run_trace;
static void capture(fixture *f, run_trace *t) {
    for (uint32_t i = 0; i < 8; ++i) {
        elpis_dsv41_clock_trace_v1 x;
        assert(elpis_dsv41_clock_trace(f->clock, i, &x) == ELPIS_CLOCK_OK);
        assert(x.complete_bytes <= sizeof t->complete[i]);
        t->tokens[i] = x.token; t->bytes = x.complete_bytes;
        memcpy(t->complete[i], x.complete, x.complete_bytes); memcpy(t->rows[i], x.rows, sizeof t->rows[i]);
    }
}
static void advance_all(fixture *f, uint32_t budget, elpis_dsv41_clock_metrics_v1 *m) {
    do {
        assert(elpis_dsv41_clock_advance(f->clock, budget, m) == ELPIS_CLOCK_OK);
        if (m->outcome == ELPIS_CLOCK_MATERIALIZATION_NEEDED) { struct timespec ts = {0, 100000}; nanosleep(&ts, NULL); }
    } while (m->outcome == ELPIS_CLOCK_PROGRESS || m->outcome == ELPIS_CLOCK_MATERIALIZATION_NEEDED);
}

typedef struct { elpis_dsv41_materializer id; atomic_int stop; unsigned count; } quiescer;
static void *quiesce_loop(void *v) {
    quiescer *q = v;
    while (!atomic_load(&q->stop)) { assert(elpis_dsv41_materializer_quiesce(q->id) == ELPIS_CLOCK_OK); q->count++; sched_yield(); }
    return NULL;
}

static void check_bitwise(void) {
    for (unsigned mode = 0; mode < 10; ++mode) {
        int fp8 = mode & 1, part = (mode / 2) % 3, race = mode >= 6, cache = (mode / 2) == 2;
        size_t part_bytes = part == 0 ? 128 : part == 1 ? 1152 : 1152;
        run_trace *base = calloc(1, sizeof *base), *mine = calloc(1, sizeof *mine);
        fixture f;
        /* Baseline: R0 test materializer, identical decoded bytes in memory. */
        setup(&f, part_bytes, (yts_ref_config){0, 0, 0, 0, -1}, cache ? 1000000 : 0);
        prepare_files(&f, fp8);
        create_open(&f);
        elpis_dsv41_clock_metrics_v1 m0, m1;
        advance_all(&f, T, &m0);
        assert(m0.outcome == ELPIS_CLOCK_COMPLETE && m0.position == 8);
        capture(&f, base);
        teardown(&f);
        /* Production: files, native page verification, FMS leases, native decode. */
        setup(&f, part_bytes, (yts_ref_config){0, 0, 0, 0, -1}, cache ? 1000000 : 0);
        prepare_files(&f, fp8);
        elpis_dsv41_materializer id = production(&f, fp8, -1, -1, IMAGE_BYTES);
        f.c.materialization_timeout_ms = 5000;
        create_open(&f);
        quiescer q = {id, 0, 0};
        pthread_t thread;
        if (race) assert(!pthread_create(&thread, NULL, quiesce_loop, &q));
        advance_all(&f, mode == 2 ? 1 : T, &m1);
        if (race) { atomic_store(&q.stop, 1); assert(!pthread_join(thread, NULL)); }
        assert(m1.outcome == ELPIS_CLOCK_COMPLETE && m1.position == 8 && m1.generated == 6);
        capture(&f, mine);
        assert(!memcmp(base->tokens, mine->tokens, sizeof base->tokens));
        assert(!memcmp(base->rows, mine->rows, sizeof base->rows));
        for (unsigned i = 0; i < 8; ++i) assert(!memcmp(base->complete[i], mine->complete[i], base->bytes));
        assert(m1.submissions == m0.submissions && m1.bytes_h2p == m0.bytes_h2p && m1.bytes_p2h == m0.bytes_p2h);
        assert(m1.acquires == m0.acquires && m1.releases == m0.releases);
        elpis_dsv41_materializer_stats_v1 s = mstats(id);
        assert(s.row_calls >= 8 && s.file.misses > 0 && s.file.hits > 0);
        printf("production mode=%u fp8=%d part=%zu cache=%d race=%d tokens=8 yields=%llu row_bytes=%llu expert_bytes=%llu "
               "pread=%llu hits=%llu misses=%llu evictions=%llu leases=%llu forced=%llu\n",
               mode, fp8, part_bytes, cache, race, (unsigned long long)m1.materialization_yields,
               (unsigned long long)s.row_bytes, (unsigned long long)s.expert_bytes,
               (unsigned long long)s.file.pread_bytes, (unsigned long long)s.file.hits,
               (unsigned long long)s.file.misses, (unsigned long long)s.file.evictions,
               (unsigned long long)s.file.lease_acquires, (unsigned long long)s.forced_releases);
        finish_production(&f, id, 1 + q.count);
        free(base); free(mine);
    }
}

static void check_faults_through_clock(void) {
    elpis_dsv41_clock_metrics_v1 m;
    yts_ref_counters cnt;
    for (unsigned scenario = 0; scenario < 6; ++scenario) {
        fixture f;
        setup(&f, 128, (yts_ref_config){0, 0, 0, 0, -1}, 0);
        prepare_files(&f, scenario & 1);
        elpis_dsv41_materializer id = production(&f, scenario & 1,
            scenario == 0 ? -2 : -1,                      /* corrupt bank pages: before TOKEN_BEGIN */
            scenario == 1 ? -2 : -1,                      /* corrupt expert pages: after TOKEN_BEGIN */
            scenario == 2 ? 64 : IMAGE_BYTES);            /* staging budget below one part */
        f.c.materialization_timeout_ms = 5;
        create_open(&f);
        unsigned quiesces = 1;
        if (scenario == 3) {                              /* stale service: destroyed before advance */
            assert(elpis_dsv41_materializer_destroy(id) == ELPIS_CLOCK_OK);
            assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
            assert(m.outcome == ELPIS_CLOCK_FAILED && m.code == ELPIS_CLOCK_STALE && m.state == ELPIS_CLOCK_RELEASED);
            id = 0;
        } else if (scenario == 4) {                       /* cancellation before acquisition */
            assert(elpis_dsv41_clock_cancel(f.clock) == ELPIS_CLOCK_OK);
            assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
            assert(m.outcome == ELPIS_CLOCK_CANCELLED && m.position == 0 && m.acquires == 0);
        } else if (scenario == 5) {                       /* cancellation during expert acquisition */
            elpis_file_service_test_hooks h = {cancel_pread, 0};
            elpis_file_service_set_test_hooks(&h);
            cancel_target = f.clock; cancel_armed = 1;
            assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
            elpis_file_service_set_test_hooks(NULL);
            assert(m.outcome == ELPIS_CLOCK_CANCELLED && m.state == ELPIS_CLOCK_DISCARDED && m.position == 0);
        } else {
            advance_all(&f, T, &m);
            assert(m.outcome == ELPIS_CLOCK_FAILED);
            if (scenario == 0) assert(m.code == ELPIS_CLOCK_INTEGRITY && m.state == ELPIS_CLOCK_RELEASED && m.position == 0);
            if (scenario == 1) assert(m.code == ELPIS_CLOCK_INTEGRITY && m.state == ELPIS_CLOCK_DISCARDED && m.position == 0);
            if (scenario == 2) assert(m.code == ELPIS_CLOCK_LIMIT && m.state == ELPIS_CLOCK_DISCARDED && m.materialization_yields > 0);
        }
        elpis_dsv41_reference_provider_counters(&cnt);
        assert(cnt.models == 1 && cnt.streams == 0 && !cnt.tokens); /* model survives host failures */
        finish_production(&f, id, quiesces);
    }
    /* Provider-originated failures with no outstanding span: quarantine, no leak. */
    fixture map;
    setup(&map, 1152, (yts_ref_config){0, 0, 0, 0, -1}, 0);
    uint64_t admission = map.provider.seq;
    detach(&map.provider); counters_released();
    for (unsigned release = 0; release < 2; ++release) {
        fixture f;
        /* message index: first TOKEN_BEGIN, or the STREAM_RELEASE after one prefill token */
        setup(&f, 1152, (yts_ref_config){0, 0, 0, FAULT_POLL_FAIL, (int64_t)(admission + (release ? 11 : 1))}, 0);
        prepare_files(&f, 1);
        elpis_dsv41_materializer id = production(&f, 1, -1, -1, IMAGE_BYTES);
        f.c.prefill_count = 1; f.c.max_new_tokens = 0;
        create_open(&f);
        assert(elpis_dsv41_clock_advance(f.clock, T, &m) == ELPIS_CLOCK_OK);
        assert(m.state == ELPIS_CLOCK_QUARANTINED && m.provider_code == ELPIS_CLOCK_DEVICE);
        if (release) assert(m.outcome == ELPIS_CLOCK_COMPLETE && m.code == ELPIS_CLOCK_OK && m.position == 1);
        else assert(m.outcome == ELPIS_CLOCK_FAILED && m.position == 0);
        elpis_dsv41_reference_provider_counters(&cnt);
        assert(!cnt.models && !cnt.streams && !cnt.slots && !cnt.tokens);
        finish_production(&f, id, 1);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    check_bitwise();
    check_faults_through_clock();
    puts("PASS native clock + production materializer: bitwise parity, faults, cancellation, quiesce race, closure");
    return 0;
}
