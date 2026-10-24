/*
 * ECS_C relational view over the real ECS_C authority and the real ECS_C memory, end to end, through the
 * production libraries (libelpis_ecs_c_relational, libelpis_ecs_c_authority, libelpis_ecs_memory).
 *
 * 1. The gap, demonstrated on the memory: after SOURCE_VERSION, DORMANT, and reclamation with a reused 32-bit
 *    source_authority, memory LOOKUP keeps answering from evidence the authority's own admission law refuses, and
 *    attributes evidence to a source that never asserted it.
 * 2. The view under the same transitions: synced from elpis_ecs_c_read_status / elpis_ecs_c_entity_at only.
 * 3. Its admission law equals elpis_ecs_c_check_evidence on every (row, version, scope) probe, at every state.
 * 4. A real rollback: authority records copied back, the operator's RECONCILE_ROLLBACK, and the view's sync refused
 *    REGRESSION (a source version went down), until an explicit reset.
 *
 * Usage: test_ecs_c_relational_authority SCRATCH_DIR
 */
#define _XOPEN_SOURCE 700
#include <assert.h>
#include <fcntl.h>
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "elpis/ecs_c_relational.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define EOK(expr) CHECK((expr) == ELPIS_ECS_C_OK)
#define ROK(expr) CHECK((expr) == ELPIS_ECS_C_REL_OK)

static char state_dir[4096], authority_dir[4096];

static int remove_entry(const char *path, const struct stat *st, int flag, struct FTW *ftw) {
    (void)st, (void)flag, (void)ftw;
    chmod(path, 0700);
    return remove(path);
}
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
static int32_t open_authority(elpis_ecs_c_authority **out, elpis_ecs_c_report *report) {
    return elpis_ecs_c_open(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), out, report);
}
static int32_t send(elpis_ecs_c_authority *a, const uint8_t e[512]) {
    elpis_ecs_c_status s;
    EOK(elpis_ecs_c_read_status(a, &s));
    return elpis_ecs_c_transition(a, s.generation, s.state_root, e, NULL);
}
static uint64_t next_seq(elpis_ecs_c_authority *a, const uint8_t who[32]) {
    elpis_ecs_c_entity e;
    EOK(elpis_ecs_c_entity_find(a, who, &e));
    return e.sender_watermark + 1;
}
static void scope_of(uint8_t out[32], uint8_t tag) { memset(out, tag, 32); }

static void found_source(elpis_ecs_c_authority *a, const uint8_t op[32], uint8_t label, uint32_t authority,
                         uint8_t scope, uint8_t out[32]) {
    uint8_t l[32], e[512];
    memset(l, label, 32);
    EOK(elpis_ecs_c_next_entity_id(a, ELPIS_ECS_C_SOURCE, l, out));
    event(e, 2, op, next_seq(a, op), out);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 84, 8);
    put_u64(e + 88, 1);
    memcpy(e + 96, l, 32);
    memset(e + 128, scope, 32);
    put_u32(e + 160, authority);
    EOK(send(a, e));
}
static void lifecycle(elpis_ecs_c_authority *a, const uint8_t op[32], const uint8_t id[32], uint8_t from, uint8_t to) {
    uint8_t e[512];
    event(e, 3, op, next_seq(a, op), id);
    e[80] = from;
    e[81] = to;
    EOK(send(a, e));
}
static void source_version(elpis_ecs_c_authority *a, const uint8_t op[32], const uint8_t id[32], uint64_t expected,
                           uint64_t version, uint8_t scope) {
    uint8_t e[512];
    event(e, 4, op, next_seq(a, op), id);
    put_u64(e + 80, expected);
    put_u64(e + 88, version);
    put_u32(e + 96, 8);
    memset(e + 104, scope, 32);
    EOK(send(a, e));
}
static void reclaim(elpis_ecs_c_authority *a, const uint8_t op[32], const uint8_t id[32]) {
    uint8_t e[512];
    event(e, 14, op, next_seq(a, op), id);
    EOK(send(a, e));
}

/* The authority as the view consumes it: its status and every row, through its own C ABI. */
static elpis_ecs_c_entity rows[ELPIS_ECS_C_MAX_CAPACITY];
static elpis_ecs_c_rel_authority table_of(const elpis_ecs_c_authority *a, elpis_ecs_c_rel_coordinate *coord) {
    elpis_ecs_c_status s;
    EOK(elpis_ecs_c_read_status(a, &s));
    for (uint32_t i = 0; i < s.entity_count; ++i) EOK(elpis_ecs_c_entity_at(a, i, &rows[i]));
    elpis_ecs_c_rel_authority t;
    memset(&t, 0, sizeof(t));
    t.abi_version = ELPIS_ECS_C_REL_ABI_VERSION;
    t.entity_count = s.entity_count;
    t.generation = s.generation;
    t.next_founding_index = s.next_founding_index;
    t.reclaimed_count = s.reclaimed_count;
    memcpy(t.lineage, s.lineage, 32);
    memcpy(t.state_root, s.state_root, 32);
    t.entities = rows;
    if (coord) {
        coord->generation = s.generation;
        memcpy(coord->state_root, s.state_root, 32);
    }
    return t;
}
static int sync_to(elpis_ecs_c_rel *v, const elpis_ecs_c_authority *a, elpis_ecs_c_rel_sync_result *r) {
    const elpis_ecs_c_rel_authority t = table_of(a, NULL);
    return elpis_ecs_c_rel_sync(v, &t, r);
}

/* The view's admission law against the authority's, on every probe. */
static uint64_t law_probes, law_states;
static int mapped(int32_t code) {
    switch (code) {
    case ELPIS_ECS_C_OK: return ELPIS_ECS_C_REL_OK;
    case ELPIS_ECS_C_UNKNOWN_ENTITY: return ELPIS_ECS_C_REL_UNKNOWN_SOURCE;
    case ELPIS_ECS_C_UNAUTHORIZED: return ELPIS_ECS_C_REL_NOT_SOURCE;
    case ELPIS_ECS_C_LIFECYCLE: return ELPIS_ECS_C_REL_LIFECYCLE;
    case ELPIS_ECS_C_SOURCE_STALE: return ELPIS_ECS_C_REL_SOURCE_STALE;
    case ELPIS_ECS_C_SCOPE_CONFLICT: return ELPIS_ECS_C_REL_SCOPE_CONFLICT;
    default: return -1;
    }
}
static void same_law(const elpis_ecs_c_rel *v, const elpis_ecs_c_authority *a) {
    elpis_ecs_c_status s;
    EOK(elpis_ecs_c_read_status(a, &s));
    ++law_states;
    uint8_t unknown[32], scopes[5][32];
    memset(unknown, 0xEE, 32);
    for (int i = 0; i < 4; ++i) scope_of(scopes[i], (uint8_t)(0xA0 + i));
    for (uint32_t i = 0; i <= s.entity_count; ++i) {
        elpis_ecs_c_entity e;
        const uint8_t *id = unknown;
        if (i < s.entity_count) {
            EOK(elpis_ecs_c_entity_at(a, i, &e));
            id = e.id;
            memcpy(scopes[4], e.scope, 32);
        }
        for (uint64_t version = 0; version <= 4; ++version)
            for (int sc = 0; sc < 5; ++sc) {
                uint32_t limit = 0;
                const int want = mapped(elpis_ecs_c_check_evidence(a, id, version, scopes[sc], &limit));
                CHECK(want >= 0 && elpis_ecs_c_rel_check(v, id, version, scopes[sc]) == want);
                ++law_probes;
            }
    }
}

static elpis_ecs_mem_observation pair(uint32_t polarity, uint64_t tick, uint32_t authority, uint8_t a, uint8_t b) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 5;
    o.polarity = polarity;
    o.participant_count = 2;
    o.tick = tick;
    o.source_authority = authority;
    memset(o.provenance.b, (int)(tick & 0xFF), 32);
    o.participants[0].role = 1;
    memset(o.participants[0].entity.b, a, 32);
    o.participants[1].role = 2;
    memset(o.participants[1].entity.b, b, 32);
    return o;
}
static elpis_ecs_mem_request ask(uint8_t a) {
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof(q));
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.relation_type = 5;
    q.target_role = 2;
    q.cue_count = 1;
    q.cue[0].role = 1;
    memset(q.cue[0].entity.b, a, 32);
    return q;
}
static uint8_t answer_of(const elpis_ecs_mem_answer *ans) { return ans->found ? ans->answer.b[0] : 0; }

/* Admission as RuntimeCore's HOT lane performs it (the observation's source_authority names an ACTIVE source),
 * into the memory; and into the view under the binding the authority holds now. */
static void admit_both(elpis_ecs_mem *m, elpis_ecs_c_rel *v, const elpis_ecs_c_authority *a,
                       const elpis_ecs_mem_observation *o) {
    elpis_ecs_c_entity e;
    EOK(elpis_ecs_c_source_for_authority(a, o->source_authority, &e));
    CHECK(e.lifecycle == ELPIS_ECS_C_ACTIVE);
    CHECK(elpis_ecs_mem_observe(m, 0, o, NULL, NULL) == ELPIS_ECS_MEM_OK);
    elpis_ecs_c_rel_coordinate c;
    (void)table_of(a, &c);
    ROK(elpis_ecs_c_rel_admit(v, &c, e.id, e.source_version, e.scope, o, NULL, NULL));
}

static void copy_slots(int save, uint8_t *saved[2], size_t bytes) {
    char path[4300];
    for (int i = 0; i < 2; i++) {
        CHECK(snprintf(path, sizeof(path), "%s/authority.%c", state_dir, 'a' + i) < (int)sizeof(path));
        const int fd = open(path, save ? O_RDONLY : O_WRONLY);
        CHECK(fd >= 0);
        if (save) CHECK(pread(fd, saved[i], bytes, 0) == (ssize_t)bytes);
        else CHECK(pwrite(fd, saved[i], bytes, 0) == (ssize_t)bytes);
        CHECK(close(fd) == 0);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    snprintf(state_dir, sizeof state_dir, "%s/state", argv[1]);
    snprintf(authority_dir, sizeof authority_dir, "%s/authority", argv[1]);
    nftw(argv[1], remove_entry, 16, FTW_DEPTH | FTW_PHYS);
    CHECK(mkdir(argv[1], 0755) == 0);
    uint8_t opl[32], rtl[32];
    memset(opl, 1, 32);
    memset(rtl, 2, 32);
    const uint32_t capacity = 16;
    EOK(elpis_ecs_c_provision(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), capacity, opl, rtl,
                              NULL));
    elpis_ecs_c_authority *a = NULL;
    EOK(open_authority(&a, NULL));
    elpis_ecs_c_entity op;
    EOK(elpis_ecs_c_entity_at(a, 0, &op));
    uint8_t s7[32], s8[32], s9[32], s10[32];
    found_source(a, op.id, 7, 7, 0xA0, s7);
    found_source(a, op.id, 8, 8, 0xA0, s8);
    found_source(a, op.id, 9, 9, 0xA1, s9);
    lifecycle(a, op.id, s7, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    lifecycle(a, op.id, s8, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    lifecycle(a, op.id, s9, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);

    elpis_ecs_c_rel_config vc;
    elpis_ecs_c_rel_config_default(&vc);
    vc.entity_capacity = capacity;
    elpis_ecs_c_rel *v = NULL;
    ROK(elpis_ecs_c_rel_create(&vc, &v));
    ROK(sync_to(v, a, NULL));
    same_law(v, a);
    elpis_ecs_mem_config mc;
    elpis_ecs_mem_config_default(&mc);
    mc.dim = 64, mc.banks = 1, mc.slot_capacity = 32, mc.codebook_capacity = 64, mc.byte_budget = 1u << 26;
    elpis_ecs_mem *m = NULL;
    CHECK(elpis_ecs_mem_create(&mc, &m) == ELPIS_ECS_MEM_OK);

    /* s7 says 1->2 once; s8 says 1->3 twice; s9 says 4->5. */
    const elpis_ecs_mem_observation o72 = pair(ELPIS_ECS_MEM_ASSERT, 1, 7, 1, 2);
    const elpis_ecs_mem_observation o83 = pair(ELPIS_ECS_MEM_ASSERT, 2, 8, 1, 3);
    const elpis_ecs_mem_observation o95 = pair(ELPIS_ECS_MEM_ASSERT, 3, 9, 4, 5);
    admit_both(m, v, a, &o72);
    admit_both(m, v, a, &o83);
    admit_both(m, v, a, &o83);
    admit_both(m, v, a, &o95);
    const elpis_ecs_mem_request q1 = ask(1), q4 = ask(4);
    elpis_ecs_mem_answer am, av;
    CHECK(elpis_ecs_mem_query(m, &q1, &am) == ELPIS_ECS_MEM_OK);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&am) == 3 && answer_of(&av) == 3 && am.score == 2 && av.score == 2); /* agree while all valid */

    /* SOURCE_VERSION s8 v1 -> v2. */
    source_version(a, op.id, s8, 1, 2, 0xA0);
    uint8_t scope_a[32];
    scope_of(scope_a, 0xA0);
    uint32_t limit;
    CHECK(elpis_ecs_c_check_evidence(a, s8, 1, scope_a, &limit) == ELPIS_ECS_C_SOURCE_STALE);
    CHECK(elpis_ecs_mem_query(m, &q1, &am) == ELPIS_ECS_MEM_OK);
    CHECK(answer_of(&am) == 3 && am.score == 2); /* GAP: the memory answers from v1 evidence the law refuses */
    elpis_ecs_c_rel_sync_result r;
    ROK(sync_to(v, a, &r));
    CHECK(r.sources_changed == 1 && r.contributions_retired == 1 && r.relations_released == 1);
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&av) == 2 && av.score == 1 && av.conflicts == 0); /* only s7's admissible evidence */
    printf("  SOURCE_VERSION: memory LOOKUP 1->3 (score 2, from v1 evidence); view 1->2 (score 1): gap closed\n");

    /* DORMANT s7. */
    lifecycle(a, op.id, s7, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_DORMANT);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_flipped == 1);
    same_law(v, a);
    CHECK(elpis_ecs_mem_query(m, &q1, &am) == ELPIS_ECS_MEM_OK);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(answer_of(&am) == 3 && !av.found && !av.matches);
    printf("  DORMANT: memory still 1->3; view: no live evidence\n");

    /* TERMINATE s9, reclaim it, found s10 on the same 32-bit source_authority 9. */
    lifecycle(a, op.id, s9, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_TERMINATED);
    reclaim(a, op.id, s9);
    found_source(a, op.id, 10, 9, 0xA1, s10);
    lifecycle(a, op.id, s10, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    elpis_ecs_mem_digest k45;
    CHECK(elpis_ecs_mem_relation_key(&o95, &k45) == ELPIS_ECS_MEM_OK);
    elpis_ecs_mem_slot slot;
    CHECK(elpis_ecs_mem_slot_read(m, &k45, &slot) == ELPIS_ECS_MEM_OK);
    elpis_ecs_c_entity owner;
    EOK(elpis_ecs_c_source_for_authority(a, slot.last_authority, &owner));
    CHECK(!memcmp(owner.id, s10, 32)); /* GAP: the memory's evidence now resolves to a source that never said it */
    CHECK(elpis_ecs_mem_query(m, &q4, &am) == ELPIS_ECS_MEM_OK && answer_of(&am) == 5);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_retired == 1);
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q4, &av, NULL));
    CHECK(!av.found);
    CHECK(elpis_ecs_c_rel_relation_read(v, NULL, NULL, &k45, &slot) == ELPIS_ECS_C_REL_NOT_FOUND);
    size_t n = 7;
    ROK(elpis_ecs_c_rel_dependents(v, s10, NULL, 0, &n));
    CHECK(n == 0);
    const elpis_ecs_mem_observation o10 = pair(ELPIS_ECS_MEM_ASSERT, 4, 9, 6, 7);
    admit_both(m, v, a, &o10);
    elpis_ecs_c_rel_support sup[2];
    elpis_ecs_mem_digest k67;
    CHECK(elpis_ecs_mem_relation_key(&o10, &k67) == ELPIS_ECS_MEM_OK);
    ROK(elpis_ecs_c_rel_support_of(v, &k67, sup, 2, &n));
    CHECK(n == 1 && !memcmp(sup[0].source, s10, 32) && sup[0].version == 1 && sup[0].state == ELPIS_ECS_C_REL_LIVE);
    printf("  reclaim + reuse of source_authority 9: memory attributes 4->5 to the new source; view retired it\n");

    /* A real rollback. Evidence at s8 v2; the records saved; s8 -> v3; the records copied back. */
    const elpis_ecs_mem_observation o88 = pair(ELPIS_ECS_MEM_ASSERT, 5, 8, 1, 8);
    admit_both(m, v, a, &o88);
    const size_t bytes = (size_t)elpis_ecs_c_record_bytes(capacity);
    uint8_t *saved[2] = {malloc(bytes), malloc(bytes)};
    CHECK(saved[0] && saved[1]);
    copy_slots(1, saved, bytes);
    source_version(a, op.id, s8, 2, 3, 0xA2);
    ROK(sync_to(v, a, &r));
    CHECK(r.contributions_retired == 1);
    elpis_ecs_c_rel_status before;
    ROK(elpis_ecs_c_rel_read_status(v, &before));
    uint8_t id_before[32];
    ROK(elpis_ecs_c_rel_identity(v, id_before));
    elpis_ecs_c_close(a);
    copy_slots(0, saved, bytes);
    elpis_ecs_c_report report;
    EOK(open_authority(&a, &report));
    CHECK(report.disposition == ELPIS_ECS_C_ROLLBACK && report.mode == ELPIS_ECS_C_MODE_ROLLBACK);
    elpis_ecs_c_reconciliation rec;
    EOK(elpis_ecs_c_read_reconciliation(a, &rec));
    CHECK(rec.kind == 11);
    uint8_t e[512];
    event(e, 11, op.id, next_seq(a, op.id), NULL);
    put_u64(e + 80, rec.abandoned_generation);
    e[80 + 8] = rec.class_code;
    memcpy(e + 80 + 16, rec.digest, 32);
    EOK(send(a, e));
    elpis_ecs_c_close(a);
    EOK(open_authority(&a, &report));
    CHECK(report.disposition == ELPIS_ECS_C_CLEAN);
    elpis_ecs_c_entity e8;
    EOK(elpis_ecs_c_entity_find(a, s8, &e8));
    elpis_ecs_c_status st;
    EOK(elpis_ecs_c_read_status(a, &st));
    CHECK(e8.source_version == 2 && st.generation > before.generation); /* a higher generation, an older version */
    CHECK(sync_to(v, a, &r) == ELPIS_ECS_C_REL_REGRESSION);
    uint8_t id_after[32];
    ROK(elpis_ecs_c_rel_identity(v, id_after));
    CHECK(!memcmp(id_before, id_after, 32));
    /* Even an admission naming the authority's new coordinate is refused: the view has not reached it. */
    elpis_ecs_c_rel_coordinate c;
    (void)table_of(a, &c);
    scope_of(scope_a, 0xA0);
    CHECK(elpis_ecs_c_rel_admit(v, &c, s8, 2, scope_a, &o88, NULL, NULL) == ELPIS_ECS_C_REL_STALE_VIEW);
    ROK(elpis_ecs_c_rel_reset(v));
    ROK(sync_to(v, a, &r));
    same_law(v, a);
    ROK(elpis_ecs_c_rel_query(v, NULL, NULL, &q1, &av, NULL));
    CHECK(!av.found); /* a reset view holds nothing until evidence is admitted again */
    printf("  real rollback: generation %llu -> %llu with s8 v3 -> v2: sync refused REGRESSION; reset, resynced\n",
           (unsigned long long)before.generation, (unsigned long long)st.generation);

    free(saved[0]);
    free(saved[1]);
    elpis_ecs_mem_destroy(m);
    elpis_ecs_c_rel_destroy(v);
    elpis_ecs_c_close(a);
    nftw(argv[1], remove_entry, 16, FTW_DEPTH | FTW_PHYS);
    CHECK(mkdir(argv[1], 0755) == 0);
    printf("  admission law == elpis_ecs_c_check_evidence on %llu probes over %llu authority states\n",
           (unsigned long long)law_probes, (unsigned long long)law_states);
    printf("ECS_C relational view over the ECS_C authority: PASS\n");
    return 0;
}
