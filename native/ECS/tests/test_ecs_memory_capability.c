/* Canonical ECS memory capability status: admitted operations work without any capability; LEARN and the FAST
 * channel are refused UNQUALIFIED (state, outputs and identity unchanged) unless RECURRENT is admitted; the SLOW
 * channel is never provided; restore never brings content into a handle that does not admit it (fast banks without
 * RECURRENT; slow banks or a moved slow cursor ever). */
#include "ecs_memory_test_util.h"

#include <stdint.h>

static elpis_ecs_mem_config cfg(uint32_t caps) {
    elpis_ecs_mem_config c = tu_config(64, 16, 32);
    c.capabilities = caps;
    return c;
}

static size_t snap(const elpis_ecs_mem *m, uint8_t **out) {
    const size_t n = elpis_ecs_mem_snapshot_size(m);
    size_t w = 0;
    *out = malloc(n);
    assert(*out && elpis_ecs_mem_snapshot(m, *out, n, &w) == ELPIS_ECS_MEM_OK && w == n);
    return n;
}

static void test_admitted_without_capabilities(void) {
    elpis_ecs_mem_config c;
    elpis_ecs_mem_config_default(&c);
    assert(c.capabilities == 0); /* the default admits the qualified operations only */
    elpis_ecs_mem *m = NULL;
    c = cfg(0);
    assert(elpis_ecs_mem_create(&c, &m) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation o = tu_bin(3, 1, 2, ELPIS_ECS_MEM_ASSERT, 1);
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_LOOKUP, 3, 2);
    tu_cue(&q, 1, 1);
    elpis_ecs_mem_answer a;
    assert(elpis_ecs_mem_query(m, &q, &a) == ELPIS_ECS_MEM_OK && a.found && tu_same(a.answer, tu_entity(2)));

    /* Unqualified operations: refused, nothing changes (identity, outputs, authoritative counters). */
    const elpis_ecs_mem_digest before = tu_identity(m);
    elpis_ecs_mem_stats s0, s1;
    assert(elpis_ecs_mem_inspect(m, &s0) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_work w;
    memset(&w, 0xA5, sizeof(w));
    elpis_ecs_mem_work w0 = w;
    assert(elpis_ecs_mem_learn(m, 0, &key, 2, &w) == ELPIS_ECS_MEM_UNQUALIFIED);
    assert(!memcmp(&w, &w0, sizeof(w)));
    elpis_ecs_mem_answer untouched;
    memset(&untouched, 0x5A, sizeof(untouched));
    for (uint32_t ch = ELPIS_ECS_MEM_FAST; ch <= ELPIS_ECS_MEM_SLOW; ++ch) { /* SLOW: never provided */
        elpis_ecs_mem_request f = tu_query(ch, 3, 2);
        tu_cue(&f, 1, 1);
        elpis_ecs_mem_answer b = untouched;
        assert(elpis_ecs_mem_query(m, &f, &b) == ELPIS_ECS_MEM_UNQUALIFIED);
        assert(!memcmp(&b, &untouched, sizeof(b)));
    }
    assert(tu_same(before, tu_identity(m)));
    assert(elpis_ecs_mem_inspect(m, &s1) == ELPIS_ECS_MEM_OK);
    assert(s1.learns == s0.learns && s1.clock == s0.clock);
    assert(s1.refusals_by_code[-ELPIS_ECS_MEM_UNQUALIFIED] == s0.refusals_by_code[-ELPIS_ECS_MEM_UNQUALIFIED] + 1);
    assert(!strcmp(elpis_ecs_mem_code_name(ELPIS_ECS_MEM_UNQUALIFIED), "ELPIS_ECS_MEM_UNQUALIFIED"));

    /* RECURRENT: LEARN and FAST, never SLOW. */
    elpis_ecs_mem *r = NULL;
    c = cfg(ELPIS_ECS_MEM_CAP_RECURRENT);
    assert(elpis_ecs_mem_create(&c, &r) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_observe(r, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_learn(r, 0, &key, 2, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_request f = tu_query(ELPIS_ECS_MEM_FAST, 3, 2);
    tu_cue(&f, 1, 1);
    assert(elpis_ecs_mem_query(r, &f, &a) == ELPIS_ECS_MEM_OK && a.found && tu_same(a.answer, tu_entity(2)));
    f.channel = ELPIS_ECS_MEM_SLOW;
    assert(elpis_ecs_mem_query(r, &f, &a) == ELPIS_ECS_MEM_UNQUALIFIED);
    elpis_ecs_mem_destroy(r);
    elpis_ecs_mem_destroy(m);
}

static void reseal(uint8_t *b, size_t n) { elpis_sha256(b, n - 32, b + n - 32); }

static void test_restore_admission(void) {
    elpis_ecs_mem *full = NULL;
    elpis_ecs_mem_config c = cfg(TU_CAPS);
    assert(elpis_ecs_mem_create(&c, &full) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_observation o = tu_bin(3, 1, 2, ELPIS_ECS_MEM_ASSERT, 1);
    elpis_ecs_mem_digest key;
    assert(elpis_ecs_mem_observe(full, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
    uint8_t *evidence_only = NULL;
    const size_t ne = snap(full, &evidence_only);
    assert(elpis_ecs_mem_learn(full, 0, &key, 2, NULL) == ELPIS_ECS_MEM_OK);
    uint8_t *fast_only = NULL;
    const size_t nf = snap(full, &fast_only);
    /* Slow-bank content: an integrity-valid envelope (resealed) that only the research reference could produce. */
    const size_t half = (size_t)c.banks * c.dim * c.dim * 4u;
    uint8_t *slow = malloc(nf);
    assert(slow);
    memcpy(slow, fast_only, nf);
    slow[128 + half] = 1; /* slow bank 0, weight 0 = 1 */
    reseal(slow, nf);

    const uint32_t caps[2] = {0, ELPIS_ECS_MEM_CAP_RECURRENT};
    struct { const uint8_t *b; size_t n; int need; } cases[3] = {{evidence_only, ne, 0}, {fast_only, nf, 1}, {slow, nf, 2}};
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 3; ++i) {
            elpis_ecs_mem *out = (elpis_ecs_mem *)(uintptr_t)1;
            const elpis_ecs_mem_status s = elpis_ecs_mem_restore(cases[i].b, cases[i].n, 1u << 26, caps[k], &out);
            if (cases[i].need <= k) {
                assert(s == ELPIS_ECS_MEM_OK);
                elpis_ecs_mem_digest d;
                assert(elpis_ecs_mem_identity(out, &d) == ELPIS_ECS_MEM_OK);
                assert(!memcmp(d.b, cases[i].b + cases[i].n - 32, 32)); /* identity preserved exactly */
                elpis_ecs_mem_destroy(out);
            } else {
                assert(s == ELPIS_ECS_MEM_UNQUALIFIED && out == (elpis_ecs_mem *)(uintptr_t)1);
            }
        }
    }
    /* A moved slow cursor (needs at least two resident slots to be in range). */
    elpis_ecs_mem_observation o2 = tu_bin(3, 5, 6, ELPIS_ECS_MEM_ASSERT, 3);
    assert(elpis_ecs_mem_observe(full, 0, &o2, NULL, NULL) == ELPIS_ECS_MEM_OK);
    uint8_t *two = NULL;
    const size_t n2 = snap(full, &two);
    two[72] = 1;
    reseal(two, n2);
    elpis_ecs_mem *out = (elpis_ecs_mem *)(uintptr_t)1;
    assert(elpis_ecs_mem_restore(two, n2, 1u << 26, TU_CAPS, &out) == ELPIS_ECS_MEM_UNQUALIFIED);
    two[72] = 2; /* beyond the slots: structurally corrupt, whatever the capabilities */
    reseal(two, n2);
    assert(elpis_ecs_mem_restore(two, n2, 1u << 26, TU_CAPS, &out) == ELPIS_ECS_MEM_CORRUPT);

    /* restore_into: the refusal leaves the target's state exactly as it was. */
    elpis_ecs_mem *plain = NULL;
    c = cfg(0);
    assert(elpis_ecs_mem_create(&c, &plain) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_restore_into(plain, 0, evidence_only, ne) == ELPIS_ECS_MEM_OK);
    const elpis_ecs_mem_digest before = tu_identity(plain);
    assert(elpis_ecs_mem_restore_into(plain, 0, fast_only, nf) == ELPIS_ECS_MEM_UNQUALIFIED);
    assert(tu_same(before, tu_identity(plain)));
    /* Integrity is checked before admission: a corrupt envelope is CORRUPT, never UNQUALIFIED. */
    uint8_t *bad = malloc(nf);
    assert(bad);
    memcpy(bad, fast_only, nf);
    bad[nf - 1] ^= 1;
    assert(elpis_ecs_mem_restore_into(plain, 0, bad, nf) == ELPIS_ECS_MEM_CORRUPT);
    free(bad);
    elpis_ecs_mem_destroy(plain);
    elpis_ecs_mem_destroy(full);
    free(evidence_only);
    free(fast_only);
    free(slow);
    free(two);
}

int main(void) {
    test_admitted_without_capabilities();
    test_restore_admission();
    puts("ecs memory capability: ok");
    return 0;
}
