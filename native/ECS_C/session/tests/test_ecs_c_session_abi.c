/*
 * The ECS_C volatile session's C ABI, end to end (docs/ECS_C_SESSION.md). It never opens, creates or writes a file:
 * tests/boundary/test_volatile_runtime.py traces it for write-class system calls, and research/volatile_runtime_v0
 * runs the same binary unchanged under a containment that denies every filesystem write.
 *
 * Session lifecycle (create, evolve, destroy, create again: nothing carries over); every refusal is atomic (the status
 * and coordinate are unchanged but for the refusal counter); persistence-only operations are PERSISTENCE_RETIRED;
 * another session's coordinate is SESSION_MISMATCH; capacity is refused, never grown; kept receipts chain to the
 * receipt head; the heap does not grow over long runs or repeated sessions (glibc mallinfo2).
 */
#define _GNU_SOURCE 1
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

#include "elpis/ecs_c_session.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define RC(expr, code)                                                                                          \
    do {                                                                                                        \
        const int32_t rc_ = (expr);                                                                             \
        if (rc_ != (code)) {                                                                                    \
            const char *n_ = elpis_ecs_c_session_code_name(rc_);                                                \
            fprintf(stderr, "%s:%d: %s = %d (%s), expected %d\n", __FILE__, __LINE__, #expr, rc_, n_ ? n_ : "?", \
                    (int)(code));                                                                               \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define OK(expr) RC(expr, ELPIS_ECS_C_OK)

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void event(uint8_t e[512], uint16_t kind, const uint8_t sender[32], uint64_t seq, const uint8_t *receiver) {
    memset(e, 0, 512);
    e[0] = (uint8_t)kind;
    e[1] = (uint8_t)(kind >> 8);
    memcpy(e + 8, sender, 32);
    put_u64(e + 40, seq);
    if (receiver) memcpy(e + 48, receiver, 32);
}

static elpis_ecs_c_session *create(uint32_t capacity) {
    elpis_ecs_c_session_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_C_SESSION_ABI_VERSION;
    c.capacity = capacity;
    CHECK(getrandom(c.nonce, sizeof(c.nonce), 0) == (ssize_t)sizeof(c.nonce)); /* a system call, not a file */
    memset(c.operator_label, 1, 32);
    memset(c.runtime_label, 2, 32);
    elpis_ecs_c_session *s = NULL;
    OK(elpis_ecs_c_session_create(&c, &s));
    CHECK(s != NULL);
    return s;
}

static elpis_ecs_c_session_status status(const elpis_ecs_c_session *s) {
    elpis_ecs_c_session_status st;
    OK(elpis_ecs_c_session_read_status(s, &st));
    return st;
}

static uint64_t next_seq(const elpis_ecs_c_session *s, const uint8_t who[32]) {
    elpis_ecs_c_entity e;
    OK(elpis_ecs_c_session_entity_find(s, who, &e));
    return e.sender_watermark + 1;
}

static int32_t send(elpis_ecs_c_session *s, const uint8_t e[512], elpis_ecs_c_session_receipt *r) {
    elpis_ecs_c_session_coordinate c;
    OK(elpis_ecs_c_session_read_coordinate(s, &c));
    return elpis_ecs_c_session_transition(s, &c, e, r);
}

/* Everything observable but the refusal counter. */
static void same_but_refusal(const elpis_ecs_c_session_status *a, const elpis_ecs_c_session_status *b) {
    elpis_ecs_c_session_status x = *a, y = *b;
    CHECK(y.refusals == x.refusals + 1);
    x.refusals = y.refusals = 0;
    CHECK(!memcmp(&x, &y, sizeof(x)));
}

static uint8_t op[32], rt[32];
static void founders(const elpis_ecs_c_session *s) {
    elpis_ecs_c_entity e;
    OK(elpis_ecs_c_session_entity_at(s, 0, &e));
    CHECK(e.kind == ELPIS_ECS_C_OPERATOR && e.lifecycle == ELPIS_ECS_C_ACTIVE);
    memcpy(op, e.id, 32);
    OK(elpis_ecs_c_session_entity_at(s, 1, &e));
    CHECK(e.kind == ELPIS_ECS_C_RUNTIME && e.lifecycle == ELPIS_ECS_C_ACTIVE);
    memcpy(rt, e.id, 32);
}

static void found_source(elpis_ecs_c_session *s, uint8_t label_byte, uint32_t authority, uint8_t scope_byte,
                         uint8_t out[32]) {
    uint8_t label[32], e[512];
    memset(label, label_byte, 32);
    OK(elpis_ecs_c_session_next_entity_id(s, ELPIS_ECS_C_SOURCE, label, out));
    event(e, 2, op, next_seq(s, op), out);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 84, 8);
    put_u64(e + 88, 1);
    memcpy(e + 96, label, 32);
    memset(e + 128, scope_byte, 32);
    put_u32(e + 160, authority);
    OK(send(s, e, NULL));
}

static int32_t lifecycle(elpis_ecs_c_session *s, const uint8_t id[32], uint8_t from, uint8_t to) {
    uint8_t e[512];
    event(e, 3, op, next_seq(s, op), id);
    e[80] = from;
    e[81] = to;
    return send(s, e, NULL);
}

static size_t heap_in_use(void) {
    struct mallinfo2 m = mallinfo2();
    return m.uordblks;
}

int main(void) {
    /* Constants and names. */
    CHECK(elpis_ecs_c_session_abi_version() == ELPIS_ECS_C_SESSION_ABI_VERSION);
    CHECK(!strcmp(elpis_ecs_c_session_code_name(0), "ECS_C_OK"));
    CHECK(!strcmp(elpis_ecs_c_session_code_name(ELPIS_ECS_C_PERSISTENCE_RETIRED), "ECS_C_PERSISTENCE_RETIRED"));
    CHECK(!strcmp(elpis_ecs_c_session_code_name(ELPIS_ECS_C_SESSION_MISMATCH), "ECS_C_SESSION_MISMATCH"));
    CHECK(elpis_ecs_c_session_code_name(21) == NULL); /* a durable storage code: a session never returns it */
    CHECK(elpis_ecs_c_session_code_name(99) == NULL);
    CHECK(elpis_ecs_c_session_bytes(1) == 0 && elpis_ecs_c_session_bytes(ELPIS_ECS_C_MAX_CAPACITY + 1) == 0);
    CHECK(elpis_ecs_c_session_bytes(8) > 0 && elpis_ecs_c_session_bytes(4096) > elpis_ecs_c_session_bytes(8));

    /* Malformed creation: nothing is created. */
    elpis_ecs_c_session_config bad;
    memset(&bad, 0, sizeof(bad));
    bad.abi_version = 2;
    bad.capacity = 8;
    elpis_ecs_c_session *none = (elpis_ecs_c_session *)&bad;
    RC(elpis_ecs_c_session_create(&bad, &none), ELPIS_ECS_C_INVALID);
    CHECK(none == NULL);
    bad.abi_version = ELPIS_ECS_C_SESSION_ABI_VERSION;
    bad.capacity = 1;
    RC(elpis_ecs_c_session_create(&bad, &none), ELPIS_ECS_C_INVALID);
    RC(elpis_ecs_c_session_create(NULL, &none), ELPIS_ECS_C_INVALID);

    /* Genesis. */
    const uint32_t capacity = 6;
    elpis_ecs_c_session *s = create(capacity);
    founders(s);
    elpis_ecs_c_session_status st = status(s);
    CHECK(st.generation == 1 && st.entity_count == 2 && st.capacity == capacity && st.anchored == 0);
    CHECK(st.receipt_count == 1 && st.transitions == 0 && st.refusals == 0 && st.instance >= 1);
    CHECK(st.volatile_bytes == elpis_ecs_c_session_bytes(capacity));
    elpis_ecs_c_session_coordinate c0;
    OK(elpis_ecs_c_session_read_coordinate(s, &c0));
    CHECK(!memcmp(c0.session, st.session, 32) && c0.generation == 1 && !memcmp(c0.state_root, st.state_root, 32));

    /* Sources: founding, activation, admission, revision. Receipts are kept to check the chain head later. */
    uint8_t heads[32];
    memcpy(heads, st.receipt_head, 32);
    uint8_t src[32];
    found_source(s, 9, 7, 0xA0, src);
    uint8_t scope[32];
    memset(scope, 0xA0, 32);
    uint32_t limit = 0;
    RC(elpis_ecs_c_session_check_evidence(s, src, 1, scope, &limit), ELPIS_ECS_C_LIFECYCLE);
    OK(lifecycle(s, src, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE));
    OK(elpis_ecs_c_session_check_evidence(s, src, 1, scope, &limit));
    CHECK(limit == 8);
    elpis_ecs_c_entity row;
    OK(elpis_ecs_c_session_source_for_authority(s, 7, &row));
    CHECK(!memcmp(row.id, src, 32) && row.source_version == 1);
    uint8_t e[512];
    event(e, 4, op, next_seq(s, op), src);
    put_u64(e + 80, 1);
    put_u64(e + 88, 2);
    put_u32(e + 96, 4);
    memset(e + 104, 0xA1, 32);
    elpis_ecs_c_session_receipt receipt;
    OK(send(s, e, &receipt));
    CHECK(receipt.generation == 4 && receipt.predecessor_generation == 3);
    RC(elpis_ecs_c_session_check_evidence(s, src, 1, scope, &limit), ELPIS_ECS_C_SOURCE_STALE);
    memset(scope, 0xA1, 32);
    OK(elpis_ecs_c_session_check_evidence(s, src, 2, scope, &limit));
    CHECK(limit == 4);

    /* The in-session cognition lineage and evolution authority. */
    uint8_t k1[32], after[32], evidence[32];
    memset(k1, 0x11, 32);
    memset(after, 0x22, 32);
    memset(evidence, 0x33, 32);
    event(e, 5, rt, next_seq(s, rt), NULL);
    memcpy(e + 88, k1, 32);
    OK(send(s, e, NULL));
    event(e, 6, rt, next_seq(s, rt), NULL);
    memcpy(e + 88, k1, 32);
    memcpy(e + 120, after, 32);
    put_u64(e + 152, 3);
    memcpy(e + 160, evidence, 32);
    OK(send(s, e, NULL));
    st = status(s);
    CHECK(st.anchored == 1 && !memcmp(st.identity, after, 32) && st.publications == 2 && st.evidence_count == 3);
    event(e, 8, rt, next_seq(s, rt), NULL); /* EVOLUTION_RESERVE from the genesis evolution authority */
    memset(e + 80 + 80, 0x44, 32);
    OK(send(s, e, NULL));
    st = status(s);
    CHECK(st.evolution.pending == 1);
    uint8_t digest[32];
    OK(elpis_ecs_c_session_evolution_digest(&st.evolution, digest));
    CHECK(!memcmp(digest, st.evolution_digest, 32));

    /* Every refusal is atomic. */
    const elpis_ecs_c_session_status before_refusals = status(s);
    elpis_ecs_c_session_status prev = before_refusals, now;
    event(e, 3, op, next_seq(s, op) - 1, src); /* a replayed sequence */
    e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_DORMANT;
    RC(send(s, e, NULL), ELPIS_ECS_C_DUPLICATE);
    now = status(s), same_but_refusal(&prev, &now), prev = now;
    event(e, 3, op, next_seq(s, op) + 1, src); /* a gap */
    e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_DORMANT;
    RC(send(s, e, NULL), ELPIS_ECS_C_SEQUENCE_GAP);
    now = status(s), same_but_refusal(&prev, &now), prev = now;
    event(e, 3, op, next_seq(s, op), src); /* an illegal lifecycle step */
    e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_FOUNDED;
    RC(send(s, e, NULL), ELPIS_ECS_C_LIFECYCLE);
    now = status(s), same_but_refusal(&prev, &now), prev = now;
    event(e, 5, rt, next_seq(s, rt), NULL); /* already anchored */
    memset(e + 88, 0x55, 32);
    RC(send(s, e, NULL), ELPIS_ECS_C_ALREADY_ANCHORED);
    now = status(s), same_but_refusal(&prev, &now), prev = now;
    event(e, 5, op, next_seq(s, op), NULL); /* the operator may not anchor */
    memset(e + 88, 0x55, 32);
    RC(send(s, e, NULL), ELPIS_ECS_C_UNAUTHORIZED);
    now = status(s), same_but_refusal(&prev, &now), prev = now;
    uint8_t raw[512];
    event(raw, 3, op, next_seq(s, op), src);
    raw[3] = 1; /* a reserved byte */
    RC(send(s, raw, NULL), ELPIS_ECS_C_INVALID);
    CHECK(status(s).refusals == prev.refusals); /* malformed bytes never reach the session */
    elpis_ecs_c_session_coordinate stale = c0;
    event(e, 3, op, next_seq(s, op), src);
    e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_DORMANT;
    RC(elpis_ecs_c_session_transition(s, &stale, e, NULL), ELPIS_ECS_C_STALE);
    now = status(s), same_but_refusal(&prev, &now), prev = now;

    /* Persistence-only operations are retired, even well formed and correctly sequenced. */
    const uint16_t retired[4] = {10, 11, 12, 13};
    for (int i = 0; i < 4; ++i) {
        event(e, retired[i], op, next_seq(s, op), NULL);
        if (retired[i] == 11) put_u64(e + 80, st.generation + 5), memset(e + 96, 9, 32);
        if (retired[i] == 12) e[81] = 2, memset(e + 88, 9, 32);
        if (retired[i] == 13) memcpy(e + 88, st.commit, 32);
        RC(send(s, e, NULL), ELPIS_ECS_C_PERSISTENCE_RETIRED);
        now = status(s), same_but_refusal(&prev, &now), prev = now;
    }

    /* Capacity: refused before anything changes; never grown. */
    uint8_t more[32];
    found_source(s, 10, 8, 0xB0, more);
    found_source(s, 11, 9, 0xB0, more);
    found_source(s, 12, 10, 0xB0, more);
    CHECK(status(s).entity_count == capacity);
    uint8_t label[32];
    memset(label, 13, 32);
    OK(elpis_ecs_c_session_next_entity_id(s, ELPIS_ECS_C_SOURCE, label, more));
    event(e, 2, op, next_seq(s, op), more);
    e[80] = ELPIS_ECS_C_SOURCE, put_u32(e + 84, 8), put_u64(e + 88, 1), memcpy(e + 96, label, 32);
    memset(e + 128, 0xB0, 32), put_u32(e + 160, 11);
    prev = status(s);
    RC(send(s, e, NULL), ELPIS_ECS_C_CAPACITY);
    now = status(s), same_but_refusal(&prev, &now);
    CHECK(now.volatile_bytes == elpis_ecs_c_session_bytes(capacity));

    /* Another session's coordinate is refused by this one, and vice versa. */
    elpis_ecs_c_session *other = create(capacity);
    elpis_ecs_c_session_coordinate theirs;
    OK(elpis_ecs_c_session_read_coordinate(other, &theirs));
    event(e, 3, op, next_seq(s, op), src);
    e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_DORMANT;
    RC(elpis_ecs_c_session_transition(s, &theirs, e, NULL), ELPIS_ECS_C_SESSION_MISMATCH);
    elpis_ecs_c_session_coordinate mine;
    OK(elpis_ecs_c_session_read_coordinate(s, &mine));
    RC(elpis_ecs_c_session_transition(other, &mine, e, NULL), ELPIS_ECS_C_SESSION_MISMATCH);

    /* Kept receipts chain to the receipt head: replay a fresh session and keep every receipt. */
    elpis_ecs_c_session *chain = create(4);
    founders(chain);
    elpis_ecs_c_session_status cs = status(chain);
    uint8_t head[32];
    memcpy(head, cs.receipt_head, 32);
    for (int i = 0; i < 6; ++i) {
        event(e, 3, op, next_seq(chain, op), rt);
        e[80] = (i % 2) ? ELPIS_ECS_C_DORMANT : ELPIS_ECS_C_ACTIVE;
        e[81] = (i % 2) ? ELPIS_ECS_C_ACTIVE : ELPIS_ECS_C_DORMANT;
        OK(send(chain, e, &receipt));
        OK(elpis_ecs_c_session_receipt_chain(head, receipt.receipt, head));
        CHECK(!memcmp(head, receipt.receipt_head, 32));
    }
    cs = status(chain);
    CHECK(!memcmp(head, cs.receipt_head, 32) && cs.receipt_count == 7 && cs.generation == 7);
    (void)heads;

    /* A long run holds the heap constant: 100 000 transitions, accepted and refused. */
    const size_t heap_before = heap_in_use();
    for (int i = 0; i < 100000; ++i) {
        event(e, 3, op, next_seq(chain, op) + (uint64_t)(i % 3 == 0), rt); /* every third a sequence gap */
        e[80] = (i % 2) ? ELPIS_ECS_C_DORMANT : ELPIS_ECS_C_ACTIVE;
        e[81] = (i % 2) ? ELPIS_ECS_C_ACTIVE : ELPIS_ECS_C_DORMANT;
        (void)send(chain, e, NULL);
    }
    const size_t heap_after = heap_in_use();
    cs = status(chain);
    CHECK(cs.transitions + cs.refusals == 100006);
    CHECK(heap_after == heap_before);
    printf("  long run: %llu accepted, %llu refused, heap in use %zu -> %zu bytes\n",
           (unsigned long long)cs.transitions, (unsigned long long)cs.refusals, heap_before, heap_after);

    /* Session lifecycle: destroy, create anew. Nothing carries over; the old coordinate is refused. */
    elpis_ecs_c_session_coordinate last;
    OK(elpis_ecs_c_session_read_coordinate(s, &last));
    elpis_ecs_c_session_destroy(s);
    elpis_ecs_c_session *fresh = create(capacity);
    elpis_ecs_c_session_status fs = status(fresh);
    CHECK(fs.generation == 1 && fs.entity_count == 2 && fs.anchored == 0 && fs.receipt_count == 1);
    CHECK(fs.publications == 0 && fs.evidence_count == 0 && fs.evolution.revision == 0 && fs.evolution.pending == 0);
    RC(elpis_ecs_c_session_entity_find(fresh, src, &row), ELPIS_ECS_C_UNKNOWN_ENTITY);
    RC(elpis_ecs_c_session_source_for_authority(fresh, 7, &row), ELPIS_ECS_C_UNKNOWN_ENTITY);
    RC(elpis_ecs_c_session_transition(fresh, &last, e, NULL), ELPIS_ECS_C_SESSION_MISMATCH);
    CHECK(memcmp(fs.session, last.session, 32) != 0);

    /* 1 000 sessions created, used and destroyed: the heap returns to where it was. glibc counts a freed chunk held
     * in its per-thread cache (tcache) as in use, so one warm-up cycle first puts the allocator in its steady state;
     * after it, the measured growth must be exactly zero. */
    elpis_ecs_c_session_destroy(create(64));
    const size_t heap_sessions_before = heap_in_use();
    for (int i = 0; i < 1000; ++i) {
        elpis_ecs_c_session *t = create(64);
        founders(t);
        event(e, 3, op, 1, rt);
        e[80] = ELPIS_ECS_C_ACTIVE, e[81] = ELPIS_ECS_C_DORMANT;
        OK(send(t, e, NULL));
        elpis_ecs_c_session_destroy(t);
    }
    const size_t heap_sessions_after = heap_in_use();
    CHECK(heap_sessions_after == heap_sessions_before);
    printf("  1000 sessions of capacity 64 (%llu bytes each): heap in use %zu -> %zu bytes\n",
           (unsigned long long)elpis_ecs_c_session_bytes(64), heap_sessions_before, heap_sessions_after);

    elpis_ecs_c_session_destroy(fresh);
    elpis_ecs_c_session_destroy(other);
    elpis_ecs_c_session_destroy(chain);
    elpis_ecs_c_session_destroy(NULL);
    printf("ECS_C volatile session C ABI: PASS\n");
    return 0;
}
