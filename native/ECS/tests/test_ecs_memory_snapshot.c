/* ECS Memory MF-R0: ELPISEM1 snapshot / restore. Round trip, exact replay, integrity (every header and
 * trailer byte, sampled body bytes), semantic validation behind a resealed checksum, torn envelopes, version
 * and configuration refusal, budget enforcement and atomic restore_into. */
#include "ecs_memory_test_util.h"

#include <stdint.h>

static uint8_t *snap_of(const elpis_ecs_mem *m, size_t *len) {
    const size_t n = elpis_ecs_mem_snapshot_size(m);
    uint8_t *b = (uint8_t *)malloc(n);
    assert(b);
    size_t written = 0;
    assert(elpis_ecs_mem_snapshot(m, b, n, &written) == ELPIS_ECS_MEM_OK && written == n);
    *len = n;
    return b;
}
static void reseal(uint8_t *b, size_t len) {
    elpis_sha256(b, len - 32u, b + len - 32u);
}
static void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

/* A deterministic script exercising every state component. */
static void script(elpis_ecs_mem *m, uint64_t from, uint64_t to) {
    for (uint64_t i = from; i < to; ++i) {
        const uint32_t roles[3] = {1, 2, 3};
        const uint64_t ents[3] = {i % 9, 20 + i % 7, 40 + i % 5};
        elpis_ecs_mem_observation o = tu_obs(1 + (uint32_t)(i % 2), i % 6 == 5 ? ELPIS_ECS_MEM_NEGATE : ELPIS_ECS_MEM_ASSERT, i,
                                       2 + (uint32_t)(i % 2), roles, ents);
        elpis_ecs_mem_digest key;
        assert(elpis_ecs_mem_observe(m, 0, &o, &key, NULL) == ELPIS_ECS_MEM_OK);
        const elpis_ecs_mem_status s = elpis_ecs_mem_learn(m, 0, &key, i, NULL);
        assert(s == ELPIS_ECS_MEM_OK || s == ELPIS_ECS_MEM_NO_NET_EVIDENCE);
    }
}
static elpis_ecs_mem_config small_config(void) {
    elpis_ecs_mem_config c = tu_config(64, 6, 10);
    c.banks = 1;
    c.retain_q16 = 63000;
    return c;
}

static void test_round_trip_and_replay(void) {
    elpis_ecs_mem_config c = small_config();
    elpis_ecs_mem *a, *b;
    assert(elpis_ecs_mem_create(&c, &a) == ELPIS_ECS_MEM_OK);
    script(a, 0, 30);
    size_t len;
    uint8_t *bytes = snap_of(a, &len);
    elpis_ecs_mem_stats st;
    assert(elpis_ecs_mem_inspect(a, &st) == ELPIS_ECS_MEM_OK && st.state_bytes == len);
    assert(st.slots_used == 6 && st.codebook_used == 10); /* both full: eviction state is in the envelope */
    elpis_ecs_mem_digest id = tu_identity(a);
    assert(!memcmp(id.b, bytes + len - 32u, 32)); /* identity == trailer */
    assert(!memcmp(bytes, "ELPISEM1", 8));
    assert(elpis_ecs_mem_restore(bytes, len, c.byte_budget, TU_CAPS, &b) == ELPIS_ECS_MEM_OK);
    assert(tu_same(id, tu_identity(b)));
    size_t len2;
    uint8_t *again = snap_of(b, &len2);
    assert(len2 == len && !memcmp(again, bytes, len)); /* restore followed by snapshot is bitwise identical */
    free(again);
    /* Exact replay: the same future on both yields the same states, answers and evictions. */
    for (uint64_t i = 30; i < 60; ++i) {
        script(a, i, i + 1);
        script(b, i, i + 1);
        assert(tu_same(tu_identity(a), tu_identity(b)));
        elpis_ecs_mem_request q = tu_query(ELPIS_ECS_MEM_FAST, 1, 2);
        tu_cue(&q, 1, i % 9);
        elpis_ecs_mem_answer x, y;
        assert(elpis_ecs_mem_query(a, &q, &x) == ELPIS_ECS_MEM_OK && elpis_ecs_mem_query(b, &q, &y) == ELPIS_ECS_MEM_OK);
        assert(x.found == y.found && tu_same(x.answer, y.answer) && x.score == y.score && x.signal == y.signal);
    }
    /* Telemetry is not state: the restored handle's counters differ, its identity does not. */
    elpis_ecs_mem_stats sa, sb;
    assert(elpis_ecs_mem_inspect(a, &sa) == ELPIS_ECS_MEM_OK && elpis_ecs_mem_inspect(b, &sb) == ELPIS_ECS_MEM_OK);
    assert(sa.observes != sb.observes && sb.restores == 1);
    elpis_ecs_mem_destroy(b);

    /* restore_into an existing handle of the same configuration, then the same future. */
    elpis_ecs_mem *d;
    assert(elpis_ecs_mem_create(&c, &d) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_restore_into(d, 0, bytes, len) == ELPIS_ECS_MEM_OK);
    assert(tu_same(id, tu_identity(d)));
    elpis_ecs_mem_destroy(d);
    free(bytes);
    elpis_ecs_mem_destroy(a);
}

static void test_integrity(void) {
    elpis_ecs_mem_config c = small_config();
    elpis_ecs_mem *a, *t;
    assert(elpis_ecs_mem_create(&c, &a) == ELPIS_ECS_MEM_OK);
    script(a, 0, 20);
    assert(elpis_ecs_mem_create(&c, &t) == ELPIS_ECS_MEM_OK);
    script(t, 0, 3);
    const elpis_ecs_mem_digest target_id = tu_identity(t);
    size_t len;
    uint8_t *bytes = snap_of(a, &len);
    uint8_t *mut = (uint8_t *)malloc(len);
    assert(mut);
    unsigned long refused = 0;
    for (size_t i = 0; i < len; ++i) {
        const int header_or_trailer = i < ELPIS_ECS_MEM_HEADER_BYTES || i >= len - 32u;
        if (!header_or_trailer && i % 13u) continue;
        memcpy(mut, bytes, len);
        mut[i] ^= (uint8_t)(1u << (i % 8u));
        assert(elpis_ecs_mem_restore_into(t, 0, mut, len) == ELPIS_ECS_MEM_CORRUPT);
        ++refused;
    }
    assert(tu_same(target_id, tu_identity(t))); /* every refusal left the target intact */
    /* Truncation and trailing bytes at every length near the end, and short prefixes. */
    for (size_t cut = 1; cut <= 40; ++cut) assert(elpis_ecs_mem_restore_into(t, 0, bytes, len - cut) == ELPIS_ECS_MEM_CORRUPT);
    uint8_t *longer = (uint8_t *)malloc(len + 8);
    assert(longer);
    memcpy(longer, bytes, len);
    memset(longer + len, 0, 8);
    assert(elpis_ecs_mem_restore_into(t, 0, longer, len + 8) == ELPIS_ECS_MEM_CORRUPT);
    for (size_t n = 0; n < 200; n += 7) assert(elpis_ecs_mem_restore_into(t, 0, bytes, n) == ELPIS_ECS_MEM_CORRUPT);
    /* Torn envelope: the prefix of one valid state over the suffix of another. */
    size_t tlen;
    uint8_t *other = snap_of(t, &tlen);
    if (tlen == len) {
        memcpy(mut, bytes, len);
        memcpy(mut, other, len / 2);
        assert(elpis_ecs_mem_restore_into(t, 0, mut, len) == ELPIS_ECS_MEM_CORRUPT);
    }
    memcpy(mut, other, tlen < len ? tlen : len);
    if (tlen < len) assert(elpis_ecs_mem_restore_into(t, 0, mut, len) == ELPIS_ECS_MEM_CORRUPT);
    assert(tu_same(target_id, tu_identity(t)));
    printf("integrity: %lu single-bit corruptions refused, target unchanged\n", refused);
    free(other);
    free(longer);
    free(mut);
    free(bytes);
    elpis_ecs_mem_destroy(t);
    elpis_ecs_mem_destroy(a);
}

/* Checksums give integrity, not authenticity: a resealed envelope must still pass every semantic check. */
static void test_semantic_validation(void) {
    elpis_ecs_mem_config c = small_config();
    elpis_ecs_mem *a, *t;
    assert(elpis_ecs_mem_create(&c, &a) == ELPIS_ECS_MEM_OK);
    script(a, 0, 20);
    assert(elpis_ecs_mem_create(&c, &t) == ELPIS_ECS_MEM_OK);
    const elpis_ecs_mem_digest empty_id = tu_identity(t);
    size_t len;
    uint8_t *bytes = snap_of(a, &len);
    uint8_t *mut = (uint8_t *)malloc(len);
    assert(mut);
    const size_t banks = 2u * c.banks * c.dim * c.dim * 4u;
    const size_t slots_at = ELPIS_ECS_MEM_HEADER_BYTES + banks;
    const uint32_t nslots = 6, nents = 10;
    const size_t sorder_at = slots_at + nslots * ELPIS_ECS_MEM_SLOT_RECORD_BYTES;
    const size_t ents_at = sorder_at + 4u * nslots;
    const size_t eorder_at = ents_at + nents * ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES;
    assert(eorder_at + 4u * nents + 32u == len);
    struct {
        const char *what;
        size_t at;
        int width; /* 4 or 8: little-endian integer overwrite; 0, 1, 2: copy 32 bytes, a slot record or 4 bytes
                      from `from` */
        uint64_t value;
        size_t from;
    } cases[] = {
        {"version 2 (no migration exists)", 8, 4, 2, 0},
        {"abi 2", 12, 4, 2, 0},
        {"dim differs from the handle", 16, 4, 128, 0},
        {"law differs from the handle", 40, 4, 1, 0},
        {"invalid policy", 32, 4, 7, 0},
        {"reserved header word", 56, 4, 1, 0},
        {"reserved header tail", 100, 4, 1, 0},
        {"clock before resident ticks", 64, 8, 0, 0},
        {"cursor beyond slots", 72, 4, 6, 0},
        {"slot count beyond capacity", 76, 4, 7, 0},
        {"bank weight beyond the stored bound", ELPIS_ECS_MEM_HEADER_BYTES, 4, 0x7fffffffu, 0},
        {"slot participant_count 0", slots_at + 164, 4, 0, 0},
        {"slot participant_count 9", slots_at + 164, 4, 9, 0},
        {"slot without any observation", slots_at + 168, 8, 0, 0},
        {"slot last polarity invalid", slots_at + 180, 4, 9, 0},
        {"slot first tick after last tick", slots_at + 192, 8, UINT64_C(1) << 40, 0},
        {"slot key does not match its content", slots_at, 0, 0, slots_at + 32},
        {"slot participants not canonical", slots_at + 224, 4, 0xffffffffu, 0},
        {"data past participant_count", slots_at + 224 + 40 * 7, 4, 1, 0},
        {"duplicate slot records", slots_at + ELPIS_ECS_MEM_SLOT_RECORD_BYTES, 1, 0, slots_at},
        {"slot order not a permutation", sorder_at, 2, 0, sorder_at + 4},
        {"slot order index out of range", sorder_at, 4, 6, 0},
        {"duplicate codebook entities", ents_at + ELPIS_ECS_MEM_CODEBOOK_RECORD_BYTES, 0, 0, ents_at},
        {"codebook first tick after last", ents_at + 32, 8, UINT64_C(1) << 41, 0},
        {"codebook order not a permutation", eorder_at, 2, 0, eorder_at + 4},
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        memcpy(mut, bytes, len);
        if (cases[k].width == 4) put32(mut + cases[k].at, (uint32_t)cases[k].value);
        else if (cases[k].width == 8) put64(mut + cases[k].at, cases[k].value);
        else memcpy(mut + cases[k].at, bytes + cases[k].from,
                    cases[k].width == 0 ? 32u : cases[k].width == 1 ? ELPIS_ECS_MEM_SLOT_RECORD_BYTES : 4u);
        if (!memcmp(mut, bytes, len)) {
            fprintf(stderr, "case '%s' did not change the envelope\n", cases[k].what);
            assert(0);
        }
        reseal(mut, len);
        const elpis_ecs_mem_status s = elpis_ecs_mem_restore_into(t, 0, mut, len);
        if (s != ELPIS_ECS_MEM_CORRUPT) {
            fprintf(stderr, "resealed case '%s' was not refused (%s)\n", cases[k].what, elpis_ecs_mem_code_name(s));
            assert(0);
        }
    }
    assert(tu_same(empty_id, tu_identity(t)));
    /* A resealed in-bound edit is accepted: the checksum is not authentication (documented nonclaim). */
    memcpy(mut, bytes, len);
    put32(mut + ELPIS_ECS_MEM_HEADER_BYTES, 12345);
    reseal(mut, len);
    assert(elpis_ecs_mem_restore_into(t, 0, mut, len) == ELPIS_ECS_MEM_OK);
    assert(!tu_same(tu_identity(t), tu_identity(a)));
    free(mut);
    free(bytes);
    elpis_ecs_mem_destroy(t);
    elpis_ecs_mem_destroy(a);
}

static void test_budget_lease_and_capacity(void) {
    elpis_ecs_mem_config c = small_config();
    elpis_ecs_mem *a, *b = NULL;
    assert(elpis_ecs_mem_create(&c, &a) == ELPIS_ECS_MEM_OK);
    script(a, 0, 10);
    size_t len;
    uint8_t *bytes = snap_of(a, &len);
    uint64_t arena, maxstate;
    assert(elpis_ecs_mem_required_bytes(&c, &arena, &maxstate) == ELPIS_ECS_MEM_OK && len <= maxstate);
    assert(elpis_ecs_mem_restore(bytes, len, arena - 1, TU_CAPS, &b) == ELPIS_ECS_MEM_CAPACITY && b == NULL);
    assert(elpis_ecs_mem_restore(bytes, len, arena, TU_CAPS, &b) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(b);
    b = NULL;
    assert(elpis_ecs_mem_restore(NULL, len, arena, TU_CAPS, &b) == ELPIS_ECS_MEM_INVALID);
    assert(elpis_ecs_mem_restore(bytes, 10, arena, TU_CAPS, &b) == ELPIS_ECS_MEM_CORRUPT && b == NULL);
    /* restore_into: a different configuration and a foreign lease are refused. */
    elpis_ecs_mem_config other = c;
    other.slot_capacity = 7;
    elpis_ecs_mem *o;
    assert(elpis_ecs_mem_create(&other, &o) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_restore_into(o, 0, bytes, len) == ELPIS_ECS_MEM_CORRUPT);
    elpis_ecs_mem_destroy(o);
    elpis_ecs_mem *t;
    assert(elpis_ecs_mem_create(&c, &t) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_lease_claim(t, 5) == ELPIS_ECS_MEM_OK);
    assert(elpis_ecs_mem_restore_into(t, 0, bytes, len) == ELPIS_ECS_MEM_LEASED);
    assert(elpis_ecs_mem_restore_into(t, 5, bytes, len) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_destroy(t);
    /* snapshot into a short buffer: refused, buffer and length untouched. */
    uint8_t small[64];
    memset(small, 0xCD, sizeof(small));
    size_t written = 77;
    assert(elpis_ecs_mem_snapshot(a, small, sizeof(small), &written) == ELPIS_ECS_MEM_CAPACITY && written == 77);
    for (size_t i = 0; i < sizeof(small); ++i) assert(small[i] == 0xCD);
    free(bytes);
    elpis_ecs_mem_destroy(a);
}

int main(void) {
    test_round_trip_and_replay();
    test_integrity();
    test_semantic_validation();
    test_budget_lease_and_capacity();
    printf("PASS_ECS_MEM_SNAPSHOT\n");
    return 0;
}
