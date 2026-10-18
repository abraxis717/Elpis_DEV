/*
 * The ECS_C authority C ABI, end to end: provisioning, open and recovery reports, the typed events, compare-and-
 * publish, evidence admission, receipts, restart, refusals. Built twice: against the production library, and
 * (ELPIS_ECS_C_TESTING) against the testing library to drive one crash through the C fault hooks.
 *
 * Usage: test_ecs_c_authority_abi SCRATCH_DIR
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

#include "elpis/ecs_c_authority.h"

#define CHECK(expr)                                                                                             \
    do {                                                                                                        \
        if (!(expr)) {                                                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #expr);                            \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)
#define RC(expr, code)                                                                                          \
    do {                                                                                                        \
        int32_t rc_ = (expr);                                                                                   \
        if (rc_ != (code)) {                                                                                    \
            fprintf(stderr, "%s:%d: %s = %d (%s), expected %d\n", __FILE__, __LINE__, #expr, rc_,              \
                    elpis_ecs_c_code_name(rc_) ? elpis_ecs_c_code_name(rc_) : "?", (int)(code));                \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)

static char state_dir[4096], authority_dir[4096];

static int remove_entry(const char *path, const struct stat *st, int flag, struct FTW *ftw) {
    (void)st, (void)flag, (void)ftw;
    chmod(path, 0700);
    return remove(path);
}

static void wipe(const char *dir) { nftw(dir, remove_entry, 16, FTW_DEPTH | FTW_PHYS); }

static void label(uint8_t out[32], uint8_t fill) { memset(out, fill, 32); }

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* An event header: kind, sender, sequence, receiver. The payload is zero. */
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

static int32_t send(elpis_ecs_c_authority *a, const uint8_t e[512], elpis_ecs_c_receipt *receipt) {
    elpis_ecs_c_status s;
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    return elpis_ecs_c_transition(a, s.generation, s.state_root, e, receipt);
}

static uint64_t next_seq(elpis_ecs_c_authority *a, const uint8_t who[32]) {
    elpis_ecs_c_entity e;
    RC(elpis_ecs_c_entity_find(a, who, &e), ELPIS_ECS_C_OK);
    return e.sender_watermark + 1;
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    snprintf(state_dir, sizeof state_dir, "%s/state", argv[1]);
    snprintf(authority_dir, sizeof authority_dir, "%s/authority", argv[1]);
    wipe(argv[1]);
    CHECK(mkdir(argv[1], 0755) == 0);

    /* Constants and names. */
    CHECK(elpis_ecs_c_abi_version() == ELPIS_ECS_C_ABI_VERSION);
    CHECK(elpis_ecs_c_record_bytes(2) == 1344 + 2 * 160);
    CHECK(elpis_ecs_c_record_bytes(1) == 0 && elpis_ecs_c_record_bytes(ELPIS_ECS_C_MAX_CAPACITY + 1) == 0);
    CHECK(strcmp(elpis_ecs_c_code_name(0), "ECS_C_OK") == 0);
    CHECK(strcmp(elpis_ecs_c_code_name(ELPIS_ECS_C_RECOVERY_REFUSED), "ECS_C_RECOVERY_REFUSED") == 0);
    CHECK(elpis_ecs_c_code_name(77) == NULL);

    /* Provisioning. */
    uint8_t op_label[32], rt_label[32];
    label(op_label, 1);
    label(rt_label, 2);
    elpis_ecs_c_authority *a = NULL;
    elpis_ecs_c_report report;
    RC(open_authority(&a, &report), ELPIS_ECS_C_NOT_PROVISIONED);
    CHECK(a == NULL);
    elpis_ecs_c_provisioned p;
    RC(elpis_ecs_c_provision(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), 1, op_label,
                             rt_label, &p),
       ELPIS_ECS_C_INVALID);
    RC(elpis_ecs_c_provision(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), 8, op_label,
                             rt_label, &p),
       ELPIS_ECS_C_OK);
    CHECK(p.capacity == 8 && p.record_bytes == elpis_ecs_c_record_bytes(8));
    RC(elpis_ecs_c_provision(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), 8, op_label,
                             rt_label, NULL),
       ELPIS_ECS_C_ALREADY_PROVISIONED);
    struct stat st;
    CHECK(stat(authority_dir, &st) == 0 && (st.st_mode & 0777) == 0700);

    /* Open. */
    RC(open_authority(NULL, &report), ELPIS_ECS_C_INVALID);
    RC(open_authority(&a, &report), ELPIS_ECS_C_OK);
    CHECK(a != NULL && report.disposition == ELPIS_ECS_C_CLEAN && report.mode == ELPIS_ECS_C_MODE_NORMAL);
    CHECK(report.replay_verified == 1 && report.reads == 5 && report.replays == 1);
    elpis_ecs_c_authority *second = NULL;
    RC(open_authority(&second, NULL), ELPIS_ECS_C_LOCKED);
    CHECK(second == NULL);
    elpis_ecs_c_status s;
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    CHECK(s.generation == 1 && s.capacity == 8 && s.entity_count == 2 && s.anchored == 0 && s.receipt_count == 1);
    CHECK(memcmp(s.lineage, p.lineage, 32) == 0 && memcmp(s.record_digest, p.record_digest, 32) == 0);
    elpis_ecs_c_entity operator_, runtime_;
    RC(elpis_ecs_c_entity_at(a, 0, &operator_), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_entity_at(a, 1, &runtime_), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_entity_at(a, 2, &operator_), ELPIS_ECS_C_INVALID);
    RC(elpis_ecs_c_entity_at(a, 0, &operator_), ELPIS_ECS_C_OK);
    CHECK(operator_.kind == ELPIS_ECS_C_OPERATOR && operator_.lifecycle == ELPIS_ECS_C_ACTIVE);
    CHECK(runtime_.kind == ELPIS_ECS_C_RUNTIME && memcmp(runtime_.label, rt_label, 32) == 0);
    uint8_t head[32];
    memcpy(head, s.receipt_head, 32);

    /* Found and activate a source. */
    uint8_t src_label[32], scope[32], src[32], e[512];
    label(src_label, 3);
    label(scope, 4);
    RC(elpis_ecs_c_next_entity_id(a, ELPIS_ECS_C_SOURCE, src_label, src), ELPIS_ECS_C_OK);
    event(e, 2, operator_.id, next_seq(a, operator_.id), src);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 80 + 4, 16);
    put_u64(e + 80 + 8, 1);
    memcpy(e + 80 + 16, src_label, 32);
    memcpy(e + 80 + 48, scope, 32);
    elpis_ecs_c_receipt r;
    RC(send(a, e, &r), ELPIS_ECS_C_OK);
    CHECK(r.generation == 2 && r.predecessor_generation == 1 && r.founded_present == 1);
    CHECK(memcmp(r.founded, src, 32) == 0);
    RC(elpis_ecs_c_receipt_chain(head, r.receipt, head), ELPIS_ECS_C_OK);
    uint32_t limit = 0;
    RC(elpis_ecs_c_check_evidence(a, src, 1, scope, &limit), ELPIS_ECS_C_LIFECYCLE);
    event(e, 3, operator_.id, next_seq(a, operator_.id), src);
    e[80] = ELPIS_ECS_C_FOUNDED;
    e[81] = ELPIS_ECS_C_ACTIVE;
    RC(send(a, e, &r), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_receipt_chain(head, r.receipt, head), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_check_evidence(a, src, 1, scope, &limit), ELPIS_ECS_C_OK);
    CHECK(limit == 16);
    RC(elpis_ecs_c_check_evidence(a, src, 2, scope, &limit), ELPIS_ECS_C_SOURCE_STALE);
    RC(elpis_ecs_c_check_evidence(a, src, 1, src_label, &limit), ELPIS_ECS_C_SCOPE_CONFLICT);
    RC(elpis_ecs_c_check_evidence(a, operator_.id, 0, scope, &limit), ELPIS_ECS_C_UNAUTHORIZED);

    /* Cognition: anchor and publish (the runtime), refusals that write nothing. */
    uint8_t k1[32], k2[32], ev[32];
    label(k1, 5);
    label(k2, 6);
    label(ev, 7);
    event(e, 5, operator_.id, next_seq(a, operator_.id), NULL);
    memcpy(e + 80 + 8, k1, 32);
    RC(send(a, e, NULL), ELPIS_ECS_C_UNAUTHORIZED);
    event(e, 5, runtime_.id, next_seq(a, runtime_.id), NULL);
    memcpy(e + 80 + 8, k1, 32);
    RC(send(a, e, &r), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_receipt_chain(head, r.receipt, head), ELPIS_ECS_C_OK);
    RC(send(a, e, NULL), ELPIS_ECS_C_DUPLICATE);
    event(e, 6, runtime_.id, next_seq(a, runtime_.id), NULL);
    memcpy(e + 80 + 8, k1, 32);
    memcpy(e + 80 + 40, k2, 32);
    put_u64(e + 80 + 72, 3);
    memcpy(e + 80 + 80, ev, 32);
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_transition(a, s.generation - 1, s.state_root, e, NULL), ELPIS_ECS_C_STALE);
    RC(elpis_ecs_c_transition(a, s.generation, ev, e, NULL), ELPIS_ECS_C_STALE);
    RC(elpis_ecs_c_transition(a, s.generation, s.state_root, NULL, NULL), ELPIS_ECS_C_INVALID);
    e[3] = 1; /* reserved header byte */
    RC(send(a, e, NULL), ELPIS_ECS_C_INVALID);
    e[3] = 0;
    RC(send(a, e, &r), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_receipt_chain(head, r.receipt, head), ELPIS_ECS_C_OK);

    /* Evolution: reserve (an outstanding cross-domain operation), and the continuity-compatible digest. */
    event(e, 8, runtime_.id, next_seq(a, runtime_.id), NULL);
    memset(e + 80 + 80, 9, 32);
    RC(send(a, e, &r), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_receipt_chain(head, r.receipt, head), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    CHECK(s.anchored == 1 && s.state_format == 0 && memcmp(s.identity, k2, 32) == 0 && s.publications == 2);
    CHECK(s.evidence_count == 3 && s.evolution.pending == 1 && s.evolution.revision == 0);
    uint8_t digest[32];
    RC(elpis_ecs_c_evolution_digest(&s.evolution, digest), ELPIS_ECS_C_OK);
    CHECK(memcmp(digest, s.evolution_digest, 32) == 0);
    CHECK(memcmp(head, s.receipt_head, 32) == 0 && s.receipt_count == 6);
    elpis_ecs_c_status before = s;
    elpis_ecs_c_close(a);

    /* Restart: the exact authority, verified by replay, with an outstanding reservation reported. */
    RC(open_authority(&a, &report), ELPIS_ECS_C_OK);
    CHECK(report.disposition == ELPIS_ECS_C_CLEAN && report.replay_verified == 1 && report.evolution_pending == 1);
    CHECK(report.current_generation == before.generation && report.witness_generation == before.generation);
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    CHECK(memcmp(&s, &before, sizeof s) == 0);
    uint8_t legacy[ELPIS_ECS_C_LEGACY_RECORD_BYTES];
    RC(elpis_ecs_c_legacy_record("/nonexistent-elpis-continuity", 29, legacy), ELPIS_ECS_C_LEGACY_REFUSED);
    RC(elpis_ecs_c_legacy_record("relative", 8, legacy), ELPIS_ECS_C_PATH);

#ifdef ELPIS_ECS_C_TESTING
    /* One crash through the C fault hooks: death after the record sync, before the witness. */
    elpis_ecs_c_counters c;
    RC(elpis_ecs_c_testing_counters(a, 1, &c), ELPIS_ECS_C_OK);
    RC(elpis_ecs_c_testing_arm(a, 1, 1, 2), ELPIS_ECS_C_OK);
    event(e, 9, runtime_.id, next_seq(a, runtime_.id), NULL);
    put_u64(e + 80, s.evolution.revision);
    memcpy(e + 80 + 8, s.evolution.head, 32);
    e[80 + 40] = 1;
    memcpy(e + 80 + 48, s.evolution.assertion, 32);
    memset(e + 80 + 80, 10, 32);
    RC(send(a, e, NULL), ELPIS_ECS_C_TESTING_PROCESS_DEATH);
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_TESTING_PROCESS_DEATH);
    elpis_ecs_c_close(a);
    RC(open_authority(&a, &report), ELPIS_ECS_C_OK);
    CHECK(report.disposition == ELPIS_ECS_C_INTERRUPTED_COMPLETED && report.witness_completed == 1);
    RC(elpis_ecs_c_read_status(a, &s), ELPIS_ECS_C_OK);
    CHECK(s.generation == before.generation + 1 && s.evolution.pending == 0 && s.evolution.revision == 1);
    RC(elpis_ecs_c_testing_counters(a, 0, &c), ELPIS_ECS_C_OK);
    CHECK(c.creates == 0 && c.renames == 0 && c.unlinks == 0);
#endif
    elpis_ecs_c_close(a);

    /* A state directory whose records were destroyed: refused, with the disposition, and no handle. */
    char path[4200];
    size_t bytes = (size_t)elpis_ecs_c_record_bytes(8);
    uint8_t *zero = calloc(1, bytes);
    CHECK(zero != NULL);
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof path, "%s/authority.%c", state_dir, 'a' + i);
        int fd = open(path, O_WRONLY);
        CHECK(fd >= 0 && pwrite(fd, zero, bytes, 0) == (ssize_t)bytes && close(fd) == 0);
    }
    free(zero);
    a = (elpis_ecs_c_authority *)1;
    RC(open_authority(&a, &report), ELPIS_ECS_C_RECOVERY_REFUSED);
    CHECK(a == NULL && report.disposition == ELPIS_ECS_C_UNRECOVERABLE);
    CHECK(report.slot_class[0] == ELPIS_ECS_C_SLOT_EMPTY && report.slot_class[1] == ELPIS_ECS_C_SLOT_EMPTY);

    wipe(argv[1]);
    printf("ECS_C authority ABI: PASS\n");
    return 0;
}
