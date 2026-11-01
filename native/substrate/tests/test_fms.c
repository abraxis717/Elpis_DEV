/* test_fms.c - FMS residency invariant suite. Any violation exits non-zero. */
#define _POSIX_C_SOURCE 200809L

#include "elpis/fms.h"
#include "elpis/fms_pal.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/fms_pal_test.h"
#include "elpis/sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB (1024ull * 1024ull)

static int fails = 0, checks = 0;
static const char *cur = "?";
#define CHECK(cond, ...) do { checks++; if (!(cond)) { \
        printf("  FAIL [%s] %s:%d ", cur, __FILE__, __LINE__); printf(__VA_ARGS__); \
        putchar('\n'); fails++; } } while (0)
#define CASE(name) do { cur = (name); printf("- %s\n", cur); } while (0)

/* FMS ABI v3: HOT and WARM only; no COLD residency, no storage domain, nothing ever written. */
static fms_config base_cfg(uint64_t hot, uint64_t warm, uint64_t ram_ceiling) {
    fms_config c;
    memset(&c, 0, sizeof c);
    c.tier_budget[FMS_HOT] = hot;
    c.tier_budget[FMS_WARM] = warm;
    c.domain_ceiling[FMS_DOM_RAM] = ram_ceiling;
    c.domain_ceiling[FMS_DOM_DEVICE] = hot ? hot : 1;
    c.high_wm = 0.90f; c.low_wm = 0.70f;
    c.max_objects = 64;
    c.hot_absent_policy = FMS_FOLD_DOWN;
    c.cold_absent_policy = FMS_REJECT;
    return c;
}

/* Nothing was ever charged to storage or moved through it. */
static void no_storage(fms_ctx *c, const char *where) {
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.tier_bytes[FMS_COLD] == 0 && st.domain_bytes[FMS_DOM_STORAGE] == 0, "%s: storage charged", where);
    CHECK(st.cold_writes == 0 && st.cold_reads == 0 && st.cold_replica_reuse == 0 && st.digest_failures == 0,
          "%s: cold traffic", where);
}

static void fill(unsigned char *p, size_t n, unsigned char v) { memset(p, v, n); }
static int verify(const unsigned char *p, size_t n, unsigned char v) {
    for (size_t i = 0; i < n; i += 997) if (p[i] != v) return 0;
    return p[n - 1] == v;
}

/* ---------------------------------------------------------------- cases --- */

static void case_budget_and_integrity(void) {
    CASE("the WARM budget is enforced without spill and placed data stays intact");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    fms_config cfg = base_cfg(0, 8 * MB, 8 * MB);
    fms_ctx *c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create");
    if (!c) return;

    fms_id id[16]; unsigned char exp[16]; int placed = 0;
    unsigned char *buf = malloc(MB);
    for (int i = 0; i < 16; i++) {
        exp[i] = (unsigned char)(i + 1);
        fill(buf, MB, exp[i]);
        int r = fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id[placed]);
        if (r >= 0) { CHECK(r == FMS_WARM, "placed in tier %d", r); exp[placed++] = exp[i]; }
        else CHECK(r == FMS_E_LIMIT, "beyond the budget: %s (expected a refusal, never a spill)", fms_strerror(r));
    }
    CHECK(placed == 8, "exactly the 8 MiB budget is placed, placed %d", placed);
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.tier_bytes[FMS_WARM] == 8 * MB && st.domain_bytes[FMS_DOM_RAM] == 8 * MB, "warm accounting");
    CHECK(st.demotions == 0, "something was demoted");
    for (int pass = 0; pass < 3; pass++)
        for (int i = 0; i < placed; i++) {
            void *p = NULL;
            int r = fms_acquire(c, id[i], FMS_WARM, FMS_READ | FMS_WRITE, &p);
            CHECK(r == FMS_WARM, "acquire %d: %s", i, fms_strerror(r));
            if (r >= 0) { CHECK(verify(p, MB, exp[i]), "corruption %d", i); fms_release(c, id[i]); }
        }
    no_storage(c, "budget");
    for (int i = 0; i < placed; i++) CHECK(fms_unregister(c, id[i]) == FMS_OK, "unregister %d", i);
    fms_get_stats(c, &st);
    CHECK(st.tier_bytes[0] + st.tier_bytes[1] + st.tier_bytes[2] == 0, "tier leak");
    CHECK(st.domain_bytes[0] + st.domain_bytes[1] + st.domain_bytes[2] == 0, "domain leak");
    free(buf);
    fms_destroy(c);
}

static void case_shared_ram_ceiling(void) {
    CASE("HOT+WARM shared physical ceiling (integrated GPU model)");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1;
    o.device_domain = FMS_DOM_RAM;      /* iGPU: HOT bytes are system RAM */
    o.zero_copy = 0;
    fms_config cfg = base_cfg(8 * MB, 8 * MB, 10 * MB);   /* 16 MiB logical, 10 MiB physical */
    fms_pal *pal = fms_pal_test_create(&o);
    fms_ctx *c = fms_create(&cfg, pal);
    CHECK(c != NULL, "create");
    if (!c) return;
    CHECK(fms_hot_available(c) == 1, "hot should be available");

    fms_id id[14]; int placed = 0; unsigned char *buf = malloc(MB);
    for (int i = 0; i < 14; i++) {
        fill(buf, MB, (unsigned char)i);
        int r = fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id[placed]);
        /* RAM is the one physical domain of both tiers and nothing lies below WARM: once it is full, a
         * registration is refused, never spilled. */
        if (r >= 0) placed++;
        else CHECK(r == FMS_E_LIMIT, "register %d: %s", i, fms_strerror(r));
        fms_stats st; fms_get_stats(c, &st);
        CHECK(st.domain_bytes[FMS_DOM_RAM] <= 10 * MB,
              "RAM ceiling breached at %d: %llu", i, (unsigned long long)st.domain_bytes[FMS_DOM_RAM]);
        CHECK(st.tier_bytes[FMS_HOT] <= 8 * MB, "hot tier over budget");
    }
    /* The PAL's own view must agree: no hidden duplicate allocations. */
    fms_stats st; fms_get_stats(c, &st);
    uint64_t pal_ram = fms_pal_test_host_bytes(pal) + fms_pal_test_device_bytes(pal);
    CHECK(pal_ram == st.domain_bytes[FMS_DOM_RAM],
          "accounting drift: pal=%llu fms=%llu", (unsigned long long)pal_ram,
          (unsigned long long)st.domain_bytes[FMS_DOM_RAM]);
    CHECK(placed == 10, "the 10 MiB physical ceiling holds exactly 10 objects, held %d", placed);
    no_storage(c, "shared ceiling");
    for (int i = 0; i < placed; i++) fms_unregister(c, id[i]);
    free(buf);
    fms_destroy(c);
}

static void case_zero_copy_accounting(void) {
    CASE("zero-copy placement is charged once, not twice");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_RAM; o.zero_copy = 1;
    fms_config cfg = base_cfg(8 * MB, 8 * MB, 8 * MB);
    fms_pal *pal = fms_pal_test_create(&o);
    fms_ctx *c = fms_create(&cfg, pal);
    CHECK(c != NULL, "create");
    if (!c) return;

    unsigned char *buf = malloc(MB);
    fill(buf, MB, 0xA5);
    fms_id id;
    int r = fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id);
    CHECK(r == FMS_HOT, "expected HOT placement, got %d (%s)", r, fms_strerror(r));

    fms_object_info info; fms_query(c, id, &info);
    fms_stats st; fms_get_stats(c, &st);
    CHECK(info.zero_copy == 1, "not marked zero-copy");
    CHECK(info.charge[FMS_DOM_RAM] == MB, "zero-copy charge should be exactly one copy: %llu",
          (unsigned long long)info.charge[FMS_DOM_RAM]);
    CHECK(st.domain_bytes[FMS_DOM_RAM] == MB, "double counted shared allocation");
    CHECK(fms_pal_test_device_bytes(pal) == 0, "aliased device buffer allocated real bytes");
    CHECK(st.zero_copy_placements == 1, "zero_copy_placements not recorded");

    void *p = NULL;
    r = fms_acquire(c, id, FMS_HOT, FMS_READ, &p);
    CHECK(r == FMS_HOT, "acquire hot: %s", fms_strerror(r));
    if (r >= 0) { CHECK(verify(p, MB, 0xA5), "zero-copy data mismatch"); fms_release(c, id); }
    fms_unregister(c, id);
    free(buf);
    fms_destroy(c);
}

static void case_tier_collapse(void) {
    CASE("tier collapse: REJECT vs FOLD_DOWN, WARM-only; COLD is refused (no writeback, ABI v3)");
    unsigned char *buf = malloc(MB);
    fill(buf, MB, 7);

    /* (a) No device, FOLD_DOWN: HOT request lands WARM. */
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    fms_config cfg = base_cfg(4 * MB, 8 * MB, 8 * MB);
    fms_ctx *c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create a");
    if (c) {
        CHECK(fms_hot_available(c) == 0, "device must not be emulated implicitly");
        fms_id id; int r = fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id);
        CHECK(r == FMS_WARM, "fold-down expected WARM, got %d", r);
        void *p = NULL;
        CHECK((int)fms_acquire(c, id, FMS_HOT, FMS_READ, &p) == FMS_WARM, "acquire fold-down");
        fms_release(c, id);
        fms_unregister(c, id);
        fms_destroy(c);
    }

    /* (b) No device, REJECT: HOT request is refused explicitly. */
    memset(&o, 0, sizeof o);
    cfg = base_cfg(4 * MB, 8 * MB, 8 * MB);
    cfg.hot_absent_policy = FMS_REJECT;
    c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create b");
    if (c) {
        fms_id id;
        CHECK((int)fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id) == FMS_E_UNSUPPORTED, "reject policy");
        CHECK((int)fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id) == FMS_WARM, "warm still works");
        fms_unregister(c, id);
        fms_destroy(c);
    }

    /* (c) WARM only. Over-budget registration fails loudly. */
    memset(&o, 0, sizeof o);
    cfg = base_cfg(0, 4 * MB, 4 * MB);
    cfg.cold_absent_policy = FMS_REJECT;
    c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create c");
    if (c) {
        fms_id id[8]; int placed = 0, refused = 0;
        for (int i = 0; i < 8; i++) {
            int r = fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id[i]);
            if (r >= 0) placed++; else { refused++; CHECK(r == FMS_E_LIMIT, "expected E_LIMIT, got %s",
                                                          fms_strerror(r)); }
        }
        CHECK(placed == 4, "warm-only should hold exactly 4 MiB, held %d", placed);
        CHECK(refused == 4, "over-budget registrations must fail loudly");
        fms_stats st; fms_get_stats(c, &st);
        CHECK(st.tier_bytes[FMS_COLD] == 0, "cold used while unavailable");
        for (int i = 0; i < placed; i++) fms_unregister(c, id[i]);
        fms_destroy(c);
    }

    /* (d) The legacy writeback is refused explicitly: a configuration budgeting COLD or storage is not created, and
     * a registration or acquisition that asks for COLD is refused under either absent policy. */
    memset(&o, 0, sizeof o);
    cfg = base_cfg(0, 4 * MB, 4 * MB);
    cfg.tier_budget[FMS_COLD] = 64 * MB;
    fms_pal *pal = fms_pal_test_create(&o);
    c = fms_create(&cfg, pal);
    CHECK(c == NULL, "a COLD budget must be refused");
    if (c) fms_destroy(c); else pal->destroy(pal->self);
    cfg = base_cfg(0, 4 * MB, 4 * MB);
    cfg.domain_ceiling[FMS_DOM_STORAGE] = 64 * MB;
    pal = fms_pal_test_create(&o);
    c = fms_create(&cfg, pal);
    CHECK(c == NULL, "a storage ceiling must be refused");
    if (c) fms_destroy(c); else pal->destroy(pal->self);
    for (int policy = FMS_FOLD_DOWN; policy <= FMS_REJECT; ++policy) {
        cfg = base_cfg(0, 4 * MB, 4 * MB);
        cfg.cold_absent_policy = (uint8_t)policy;
        c = fms_create(&cfg, fms_pal_test_create(&o));
        CHECK(c != NULL, "create d");
        if (!c) continue;
        fms_id id;
        CHECK((int)fms_register(c, 1, MB, FMS_COLD, 0.0f, buf, &id) == FMS_E_UNSUPPORTED, "COLD registration");
        CHECK((int)fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id) == FMS_WARM, "warm registration");
        void *p = NULL;
        CHECK((int)fms_acquire(c, id, FMS_COLD, FMS_READ, &p) == FMS_E_UNSUPPORTED, "COLD acquisition");
        fms_object_info info; fms_query(c, id, &info);
        CHECK(info.tier == FMS_WARM && info.cold_replica == 0, "no cold replica");
        no_storage(c, "cold refused");
        fms_unregister(c, id);
        fms_destroy(c);
    }
    free(buf);
}

static void case_pinning(void) {
    CASE("pinned objects are never demoted");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    fms_config cfg = base_cfg(0, 4 * MB, 4 * MB);
    fms_ctx *c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create");
    if (!c) return;
    unsigned char *buf = malloc(MB); fill(buf, MB, 0x11);
    fms_id pinned, other[8];
    CHECK(fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &pinned) >= 0, "register pinned");
    void *p = NULL;
    CHECK((int)fms_acquire(c, pinned, FMS_WARM, FMS_READ, &p) == FMS_WARM, "acquire");
    for (int i = 0; i < 8; i++) {
        int r = fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &other[i]);   /* refused once WARM is full */
        CHECK(r == FMS_WARM || r == FMS_E_LIMIT, "register other %d: %s", i, fms_strerror(r));
        fms_pump(c);
    }
    fms_object_info info; fms_query(c, pinned, &info);
    CHECK(info.tier == FMS_WARM, "pinned object was demoted to tier %u", info.tier);
    CHECK(info.pin_count == 1, "pin count %u", info.pin_count);
    CHECK(verify(p, MB, 0x11), "pinned pointer invalidated");
    CHECK(fms_unregister(c, pinned) == FMS_E_BUSY, "unregister of a pinned object must fail");
    fms_release(c, pinned);
    free(buf);
    fms_destroy(c);
}

static void case_lease_fence(void) {
    CASE("lease + fence: no demotion or release while a dispatch is in flight");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_DEVICE; o.fence_mode = FMS_FENCE_PENDING;
    fms_config cfg = base_cfg(4 * MB, 4 * MB, 8 * MB);
    cfg.fence_timeout_ns = 0;
    fms_pal *pal = fms_pal_test_create(&o);
    fms_ctx *c = fms_create(&cfg, pal);
    CHECK(c != NULL, "create");
    if (!c) return;

    unsigned char *buf = malloc(MB); fill(buf, MB, 0x33);
    fms_id id; CHECK((int)fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id) == FMS_HOT, "register hot");
    fms_lease *l = NULL;
    CHECK(fms_lease_acquire(c, id, FMS_HOT, FMS_READ, &l) == FMS_OK, "lease acquire");
    int dummy_fence = 1;
    CHECK(fms_lease_bind_fence(c, l, &dummy_fence) == FMS_OK, "bind fence");
    CHECK(fms_lease_tier(l) == FMS_HOT, "lease tier");
    CHECK(fms_lease_ptr(l) != NULL, "lease ptr");

    for (int i = 0; i < 8; i++) { fms_id t; fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &t); fms_pump(c); }
    fms_object_info info; fms_query(c, id, &info);
    CHECK(info.tier == FMS_HOT, "leased object demoted while fence pending (tier %u)", info.tier);
    CHECK(info.lease_count == 1, "lease count %u", info.lease_count);
    CHECK(fms_lease_release(c, l) == FMS_E_BUSY, "release must refuse a pending fence");
    CHECK(fms_reap(c) == FMS_OK, "reap with pending fence");
    fms_query(c, id, &info);
    CHECK(info.lease_count == 1, "reap released a pending lease");

    fms_pal_test_set_fence_mode(pal, FMS_FENCE_COMPLETE);
    CHECK(fms_reap(c) == FMS_OK, "reap after completion");
    fms_query(c, id, &info);
    CHECK(info.lease_count == 0 && info.pin_count == 0, "lease not reclaimed after fence completion");
    free(buf);
    fms_destroy(c);
}

static void case_device_loss(void) {
    CASE("device loss quarantines the object and forces CPU fallback");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_DEVICE; o.fence_mode = FMS_FENCE_PENDING;
    fms_config cfg = base_cfg(4 * MB, 4 * MB, 8 * MB);
    fms_pal *pal = fms_pal_test_create(&o);
    fms_ctx *c = fms_create(&cfg, pal);
    CHECK(c != NULL, "create");
    if (!c) return;
    unsigned char *buf = malloc(MB); fill(buf, MB, 0x44);
    fms_id id; fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id);
    fms_lease *l = NULL;
    fms_lease_acquire(c, id, FMS_HOT, FMS_READ, &l);
    int f = 1; fms_lease_bind_fence(c, l, &f);

    fms_pal_test_set_fence_mode(pal, FMS_FENCE_LOST);
    CHECK(fms_reap(c) == FMS_E_DEVICE, "reap must report device loss");
    CHECK(fms_hot_available(c) == 0, "hot must be closed after device loss");
    fms_object_info info; fms_query(c, id, &info);
    CHECK(info.state == FMS_ST_LOST, "lost object state %u", info.state);
    void *p = NULL;
    CHECK(fms_acquire(c, id, FMS_WARM, FMS_READ, &p) == FMS_E_STATE, "lost object must not be handed out");
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.device_failures >= 1 && st.forced_cpu_fallbacks >= 1, "device telemetry missing");

    /* New objects still work, on the CPU path. */
    fms_id id2;
    CHECK((int)fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id2) == FMS_WARM, "post-loss placement");
    free(buf);
    fms_destroy(c);
}

static void case_upload_failure_fallback(void) {
    CASE("device upload failure leaves data intact on the CPU path");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_DEVICE;
    fms_config cfg = base_cfg(4 * MB, 4 * MB, 8 * MB);
    fms_pal *pal = fms_pal_test_create(&o);
    fms_ctx *c = fms_create(&cfg, pal);
    CHECK(c != NULL, "create");
    if (!c) return;
    unsigned char *buf = malloc(MB); fill(buf, MB, 0x55);
    fms_id id; fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id);

    fms_pal_test_set_faults(pal, 0, 1, 0);          /* uploads fail */
    void *p = NULL;
    int r = fms_acquire(c, id, FMS_HOT, FMS_READ, &p);
    CHECK(r == FMS_WARM, "expected fallback to WARM, got %d (%s)", r, fms_strerror(r));
    CHECK(p && verify(p, MB, 0x55), "data lost on failed upload");
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.device_failures >= 1, "device failure not counted");
    CHECK(st.forced_cpu_fallbacks >= 1, "fallback not counted");
    CHECK(fms_pal_test_device_bytes(pal) == 0, "device buffer leaked after failed upload");
    fms_release(c, id);
    free(buf);
    fms_destroy(c);
}

static void case_two_contexts(void) {
    CASE("two contexts, two PALs, no shared global state");
    fms_pal_test_opts o1; memset(&o1, 0, sizeof o1);
    fms_pal_test_opts o2; memset(&o2, 0, sizeof o2);
    o2.emulate_device = 1; o2.device_domain = FMS_DOM_DEVICE;

    fms_config cfg1 = base_cfg(0, 2 * MB, 2 * MB);
    fms_config cfg2 = base_cfg(2 * MB, 2 * MB, 4 * MB);
    fms_ctx *a = fms_create(&cfg1, fms_pal_test_create(&o1));
    fms_ctx *b = fms_create(&cfg2, fms_pal_test_create(&o2));
    CHECK(a && b, "create both");
    if (!a || !b) return;
    CHECK(fms_hot_available(a) == 0 && fms_hot_available(b) == 1, "capabilities leaked between contexts");

    unsigned char *buf = malloc(MB);
    fms_id ia[4], ib[4];
    for (int i = 0; i < 4; i++) {
        fill(buf, MB, (unsigned char)(0x10 + i)); fms_register(a, 1, MB, FMS_WARM, 0.0f, buf, &ia[i]);
        fill(buf, MB, (unsigned char)(0x20 + i)); fms_register(b, 1, MB, FMS_HOT,  0.0f, buf, &ib[i]);
    }
    for (int i = 0; i < 4; i++) {
        void *p = NULL;
        if (fms_acquire(a, ia[i], FMS_WARM, FMS_READ, &p) >= 0) {
            CHECK(verify(p, MB, (unsigned char)(0x10 + i)), "ctx a corruption %d", i);
            fms_release(a, ia[i]);
        }
        if (fms_acquire(b, ib[i], FMS_WARM, FMS_READ, &p) >= 0) {
            CHECK(verify(p, MB, (unsigned char)(0x20 + i)), "ctx b corruption %d", i);
            fms_release(b, ib[i]);
        }
    }
    no_storage(a, "context a");
    no_storage(b, "context b");
    free(buf);
    fms_destroy(a);
    fms_destroy(b);
}

static void case_telemetry(void) {
    CASE("telemetry is populated and observational only (HOT <-> WARM movement; nothing below WARM)");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_DEVICE;
    fms_config cfg = base_cfg(4 * MB, 8 * MB, 8 * MB);
    fms_ctx *c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create");
    if (!c) return;
    unsigned char *buf = malloc(MB);
    fms_id id[8];
    for (int i = 0; i < 8; i++) { fill(buf, MB, (unsigned char)i);
                                  CHECK(fms_register(c, 1, MB, FMS_HOT, 0.0f, buf, &id[i]) >= 0, "register %d", i); }
    for (int i = 0; i < 8; i++) {
        void *p;
        if (fms_acquire(c, id[i], FMS_HOT, FMS_READ, &p) >= 0) {
            CHECK(verify(p, MB, (unsigned char)i), "data %d", i);
            fms_release(c, id[i]);
        }
    }
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.demotions > 0 && st.promotions > 0, "movement counters empty");
    CHECK(st.bytes_demoted > 0 && st.bytes_promoted > 0, "byte counters empty");
    CHECK(st.move_p50_ns > 0, "p50 latency not recorded");
    CHECK(st.move_p95_ns >= st.move_p50_ns, "p95 < p50");
    CHECK(st.objects == 8, "object count %llu", (unsigned long long)st.objects);
    CHECK(st.inflight_ops == 0, "in-flight ops leaked");
    no_storage(c, "telemetry");
    free(buf);
    fms_destroy(c);
}

static void case_a_held_object_never_moves(void) {
    CASE("an object another holder has pinned or leased is never moved: its bytes stay where the holder reads them");
    fms_pal_test_opts o; memset(&o, 0, sizeof o);
    o.emulate_device = 1; o.device_domain = FMS_DOM_DEVICE;   /* not zero-copy: a WARM -> HOT hop frees the WARM copy */
    fms_config cfg = base_cfg(4 * MB, 8 * MB, 8 * MB);
    fms_ctx *c = fms_create(&cfg, fms_pal_test_create(&o));
    CHECK(c != NULL, "create");
    if (!c) return;
    unsigned char *buf = malloc(MB);
    fill(buf, MB, 0x5a);
    fms_id id;
    CHECK((int)fms_register(c, 1, MB, FMS_WARM, 0.0f, buf, &id) == FMS_WARM, "register WARM");
    fms_stats before; fms_get_stats(c, &before);

    /* A reader holds the WARM copy; a second acquirer asks for HOT. The object must not move under the reader. */
    void *reader = NULL, *second = NULL;
    CHECK((int)fms_acquire(c, id, FMS_WARM, FMS_READ, &reader) == FMS_WARM, "the reader's pin is WARM");
    int r = fms_acquire(c, id, FMS_HOT, FMS_READ, &second);
    CHECK(r == FMS_WARM, "a held object is pinned where it is, got tier %d", r);
    CHECK(second == reader, "both holders read the same WARM bytes");
    CHECK(verify(reader, MB, 0x5a), "the first holder's bytes are intact");
    fms_stats st; fms_get_stats(c, &st);
    CHECK(st.promotions == before.promotions, "nothing was promoted while held");
    CHECK(st.forced_placements == before.forced_placements + 1, "the refused move is counted as a forced placement");
    CHECK(fms_release(c, id) == FMS_OK && fms_release(c, id) == FMS_OK, "release both pins");

    /* A lease holds the object the same way. */
    fms_lease *lease = NULL;
    CHECK(fms_lease_acquire(c, id, FMS_WARM, FMS_READ, &lease) == FMS_OK && fms_lease_tier(lease) == FMS_WARM,
          "a WARM lease");
    r = fms_acquire(c, id, FMS_HOT, FMS_READ, &second);
    CHECK(r == FMS_WARM && second == fms_lease_ptr(lease), "a leased object is pinned where it is, got tier %d", r);
    CHECK(verify(fms_lease_ptr(lease), MB, 0x5a), "the lease's bytes are intact");
    CHECK(fms_release(c, id) == FMS_OK, "release the pin");
    CHECK(fms_lease_release(c, lease) == FMS_OK, "release the lease");

    /* Nobody holds it now: the promotion happens, and the bytes move intact. */
    r = fms_acquire(c, id, FMS_HOT, FMS_READ, &second);
    CHECK(r == FMS_HOT, "an unheld object is promoted, got tier %d", r);
    CHECK(r == FMS_HOT && verify(second, MB, 0x5a), "the promoted bytes are intact");
    fms_get_stats(c, &st);
    CHECK(st.promotions == before.promotions + 1, "one promotion");
    if (r >= 0) fms_release(c, id);
    CHECK(fms_unregister(c, id) == FMS_OK, "unregister");
    no_storage(c, "held objects");
    free(buf);
    fms_destroy(c);
}

int main(void) {
    printf("FMS invariant suite (ABI v3, volatile: HOT and WARM only)\n");
    CHECK(FMS_ABI_VERSION == 3u, "ABI");

    case_budget_and_integrity();
    case_shared_ram_ceiling();
    case_zero_copy_accounting();
    case_tier_collapse();
    case_pinning();
    case_a_held_object_never_moves();
    case_lease_fence();
    case_device_loss();
    case_upload_failure_fallback();
    case_two_contexts();
    case_telemetry();

    printf("%d checks, %d failures\n", checks, fails);
    return fails != 0;
}
