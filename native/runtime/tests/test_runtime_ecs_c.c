/* RuntimeCore ABI v6 over the ECS_C authority and the real ECS composite retained state (docs/RUNTIME_CORE.md, "The
 * durable authority (ABI v6)"), through the production libraries:
 *
 *   the operator provisions the authority and founds an ACTIVE source with the authority's own ABI, no runtime open;
 *   the runtime opens it (elpis_runtime_create_ecs_c) and anchors a composite lineage; typed observations are
 *   admitted against the authority's sources before the native call (an unknown source, and a source past its
 *   per-turn bound, are refused RUNTIME_OBSERVATION_REFUSED with the turn kept and nothing crossing); operator events
 *   reach the held authority only through elpis_runtime_authority_transition (refused while a turn is open, and for
 *   RuntimeCore's own kinds), and a DORMANT source is refused; every durable publication records the evidence summary
 *   of what it makes durable; a restart through the authority's ABI reads the same record; copied-back records refuse
 *   the runtime's open (RUNTIME_AUTHORITY, the disposition reported), the operator reconciles exactly what the
 *   authority names, and the runtime reopens at the abandoned generation + 1.
 *
 * Usage: test_runtime_ecs_c <scratch-dir>. Deterministic: no sleeps. */
#define _POSIX_C_SOURCE 200809L
#include "elpis/ecs_c_authority.h"
#include "elpis/ecs_composite.h"
#include "elpis/ecsg_k1.h"
#include "elpis/runtime.h"
#include "k1_tables.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OK(x) assert((x) == ELPIS_RUNTIME_OK)
#define EOK(x) assert((x) == ELPIS_ECS_C_OK)
enum { DIM = 6, WIDTH = 36, CAPACITY = 8, BOUND = 2 };
enum { SOURCE = 4242, UNKNOWN = 77 };
_Static_assert(sizeof(elpis_runtime_authority) == 216, "elpis_runtime_authority ABI size");

/* ---- the composite's function table (typed wrappers, as test_runtime_composite.c) ------------------------- */

static int c_digest(void *s, uint8_t out[32]) { return elpis_ecs_composite_state_digest(s, out); }
static int c_shape(void *s, size_t *dim, size_t *width) { return elpis_ecs_composite_shape(s, dim, width); }
static int c_claim(void *s, uint64_t lease) { return elpis_ecs_composite_lease_claim(s, lease); }
static int c_release(void *s, uint64_t lease) { return elpis_ecs_composite_lease_release(s, lease); }
static int c_reserve(void *s, uint64_t lease, size_t rows) { return elpis_ecs_composite_leased_reserve(s, lease, rows); }
static int c_identity(void *s, uint64_t lease, uint8_t digest[32], uint64_t *generation) {
    return elpis_ecs_composite_leased_identity(s, lease, digest, generation);
}
static int c_query(void *s, uint64_t lease, size_t dim, const double *x, size_t rows, double *out, uint64_t *g) {
    return elpis_ecs_composite_leased_query(s, lease, dim, x, rows, out, g);
}
static int c_begin(void *s, uint64_t lease, uint64_t *source, uint64_t *token) {
    return elpis_ecs_composite_leased_txn_begin_at(s, lease, source, token);
}
static int c_schedule(void *s, uint64_t lease, uint64_t token, const double *x, const double *y, size_t rows,
                      const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                      elpis_runtime_schedule_result *result) {
    return elpis_ecs_composite_leased_txn_run_schedule(s, lease, token, x, y, rows,
                                                       (const elpis_ecsg_k1_experience *)schedule, n, rate, s3, s3n,
                                                       (elpis_ecsg_k1_schedule_result *)result);
}
static int c_commit(void *s, uint64_t lease, uint64_t token, elpis_runtime_transition *t) {
    return elpis_ecs_composite_leased_txn_commit_at(s, lease, token, (elpis_ecsg_k1_transition *)t);
}
static int c_abort(void *s, uint64_t token) { return elpis_ecs_composite_txn_abort(s, token); }
static int c_capture(void *s, uint64_t lease, uint8_t *out, size_t size, uint64_t *generation) {
    return elpis_ecs_composite_leased_capture(s, lease, out, size, generation);
}
static int c_seal(uint8_t *envelope, size_t size) { return elpis_ecs_composite_envelope_seal(envelope, size); }
static int c_observe(void *s, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32], int *ms) {
    return elpis_ecs_composite_leased_txn_observe(s, lease, token, observation, (elpis_ecs_mem_digest *)key, ms);
}
static const elpis_runtime_composite_api COMPOSITE_API = {
    {c_digest, c_shape, c_claim, c_release, c_reserve, c_identity, c_query, c_begin, c_schedule, c_commit, c_abort,
     c_capture, c_seal},
    c_observe};

/* ---- fixtures ------------------------------------------------------------------------------------------------ */

static elpis_runtime_budget BUDGET;
static double W0[DIM * WIDTH];
static char scratch[4096], state_dir[4200], authority_dir[4200], ck_dir[4200], cmd[13000];

static elpis_ecs_composite_config config(void) {
    elpis_ecs_composite_config c;
    memset(&c, 0, sizeof(c));
    c.abi_version = ELPIS_ECS_COMPOSITE_ABI_V1;
    c.dim = DIM;
    c.width = WIDTH;
    c.max_rows = 8;
    elpis_ecs_mem_config_default(&c.memory);
    c.memory.dim = 64;
    c.memory.banks = 2;
    c.memory.slot_capacity = 16;
    c.memory.codebook_capacity = 32;
    return c; /* no unqualified capability */
}
static elpis_runtime_substrate csub(elpis_ecs_composite *x, uint64_t owner) {
    elpis_runtime_substrate d = {ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, x, 0, owner, DIM, &COMPOSITE_API};
    return d;
}
/* A typed observation recording memory source authority `source`. */
static elpis_ecs_mem_observation obs(uint32_t source, uint64_t tick) {
    elpis_ecs_mem_observation o;
    memset(&o, 0, sizeof(o));
    o.abi_version = ELPIS_ECS_MEM_ABI_V1;
    o.relation_type = 3;
    o.polarity = ELPIS_ECS_MEM_ASSERT;
    o.participant_count = 2;
    o.tick = tick;
    o.source_authority = source;
    o.participants[0].role = 1;
    o.participants[1].role = 2;
    for (int i = 0; i < 32; ++i) {
        o.participants[0].entity.b[i] = (uint8_t)(tick * 31 + (uint64_t)i);
        o.participants[1].entity.b[i] = (uint8_t)(tick * 17 + (uint64_t)i * 3);
        o.provenance.b[i] = (uint8_t)(tick + (uint64_t)i);
    }
    return o;
}
static void sh(const char *fmt, const char *a, const char *b) {
    assert(snprintf(cmd, sizeof(cmd), fmt, a, b) < (int)sizeof(cmd));
    assert(system(cmd) == 0);
}

/* ---- ECS_C events (elpis/ecs_c_authority.h, "Events") ------------------------------------------------------- */

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
static void lifecycle(uint8_t e[512], const uint8_t op[32], uint64_t seq, const uint8_t id[32], uint8_t from,
                      uint8_t to) {
    event(e, 3, op, seq, id);
    e[80] = from;
    e[81] = to;
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

static elpis_runtime *open_runtime(void) {
    elpis_runtime *rt = NULL;
    OK(elpis_runtime_create_ecs_c((const uint8_t *)state_dir, strlen(state_dir), (const uint8_t *)authority_dir,
                                  strlen(authority_dir), &rt));
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_open(rt, &snap));
    OK(elpis_runtime_checkpoint_attach(rt, (const uint8_t *)ck_dir, strlen(ck_dir)));
    return rt;
}

/* An observe-only turn of the given observations; returns the commit's status (each observation's in `codes`). */
static int observe_turn(elpis_runtime *rt, const elpis_runtime_substrate *d, const uint32_t *sources, int n,
                        uint64_t tick, int *codes) {
    elpis_runtime_observe_begin_result b;
    OK(elpis_runtime_turn_begin_observe(rt, d, &BUDGET, &b));
    for (int i = 0; i < n; ++i) {
        elpis_ecs_mem_observation o = obs(sources[i], tick + (uint64_t)i);
        elpis_runtime_observe_result r;
        codes[i] = elpis_runtime_turn_observe(rt, d, &o, &r);
    }
    elpis_runtime_volatile_commit vc;
    return elpis_runtime_turn_commit(rt, d, &vc);
}

static void publish(elpis_runtime *rt, const elpis_runtime_substrate *d) {
    const elpis_runtime_durable_ticket t = durable_publish(rt, d, ELPIS_RUNTIME_DURABLE_KIND_RECOVERABLE);
    assert(t.status == ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED);
}

static elpis_runtime_authority status_of(elpis_runtime *rt) {
    elpis_runtime_authority s;
    memset(&s, 0xA5, sizeof(s));
    OK(elpis_runtime_authority_status(rt, &s));
    return s;
}

static int zero32(const uint8_t *p) {
    for (int i = 0; i < 32; ++i)
        if (p[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    assert(elpis_runtime_abi_version() == ELPIS_RUNTIME_ABI_V6);
    assert(!strcmp(elpis_runtime_code_name(ELPIS_RUNTIME_AUTHORITY), "RUNTIME_AUTHORITY"));
    assert(!strcmp(elpis_runtime_code_name(ELPIS_RUNTIME_OBSERVATION_REFUSED), "RUNTIME_OBSERVATION_REFUSED"));
    OK(elpis_runtime_fuel_ceiling(&BUDGET));
    for (int i = 0; i < DIM * WIDTH; ++i) W0[i] = 0.01 * (double)((i * 7) % 13) - 0.05;
    assert(snprintf(scratch, sizeof(scratch), "%s", argv[1]) < (int)sizeof(scratch));
    assert(snprintf(state_dir, sizeof(state_dir), "%s/state", scratch) < (int)sizeof(state_dir));
    assert(snprintf(authority_dir, sizeof(authority_dir), "%s/authority", scratch) < (int)sizeof(authority_dir));
    assert(snprintf(ck_dir, sizeof(ck_dir), "%s/checkpoint", scratch) < (int)sizeof(ck_dir));
    sh("rm -rf '%s' && mkdir -p '%s'", scratch, scratch);

    /* The operator provisions the authority and founds one ACTIVE source (bound 2 per turn), no runtime open. */
    uint8_t op_label[32], rt_label[32], src_label[32], scope[32], op[32], src[32], e[512];
    memset(op_label, 1, 32);
    memset(rt_label, 2, 32);
    memset(src_label, 3, 32);
    memset(scope, 4, 32);
    EOK(elpis_ecs_c_provision(state_dir, strlen(state_dir), authority_dir, strlen(authority_dir), CAPACITY, op_label,
                              rt_label, NULL));
    elpis_ecs_c_authority *a = NULL;
    elpis_ecs_c_report report;
    EOK(open_authority(&a, &report));
    elpis_ecs_c_entity ent;
    EOK(elpis_ecs_c_entity_at(a, 0, &ent));
    memcpy(op, ent.id, 32);
    EOK(elpis_ecs_c_next_entity_id(a, ELPIS_ECS_C_SOURCE, src_label, src));
    event(e, 2, op, next_seq(a, op), src);
    e[80] = ELPIS_ECS_C_SOURCE;
    put_u32(e + 80 + 4, BOUND);
    put_u64(e + 80 + 8, 1);
    memcpy(e + 80 + 16, src_label, 32);
    memcpy(e + 80 + 48, scope, 32);
    put_u32(e + 80 + 80, SOURCE);
    EOK(send(a, e));
    lifecycle(e, op, next_seq(a, op), src, ELPIS_ECS_C_FOUNDED, ELPIS_ECS_C_ACTIVE);
    EOK(send(a, e));
    uint64_t op_seq = next_seq(a, op);
    elpis_ecs_c_close(a);

    /* The runtime opens the authority it does not own and anchors a composite lineage (checkpointed). */
    const elpis_ecs_composite_config cfg = config();
    OK(elpis_runtime_checkpoint_provision((const uint8_t *)ck_dir, strlen(ck_dir),
                                          elpis_ecs_composite_envelope_bytes(&cfg)));
    elpis_runtime *rt = open_runtime();
    elpis_runtime_authority s = status_of(rt);
    assert(s.backend == ELPIS_RUNTIME_BACKEND_ECS_C && s.ecs_c_code == 0 && s.disposition == ELPIS_ECS_C_CLEAN);
    assert(s.mode == ELPIS_ECS_C_MODE_NORMAL && s.replay_verified == 1 && s.capacity == CAPACITY);
    assert(s.entity_count == 3 && s.evidence_count == 0 && zero32(s.evidence_head));
    const uint64_t provisioned = s.generation;
    elpis_ecs_composite *x = NULL;
    assert(elpis_ecs_composite_create(&cfg, W0, &x) == ELPIS_ECS_COMPOSITE_OK);
    elpis_runtime_substrate d = csub(x, 7);
    elpis_continuity_snapshot snap;
    OK(elpis_runtime_anchor(rt, &d, &snap));
    assert(snap.generation == provisioned + 1 && status_of(rt).generation == provisioned + 1);

    /* Admission, before the native call: an unknown source, then the source past its per-turn bound. */
    OK(elpis_runtime_read_counters(rt, &(elpis_runtime_counters){0}, 1));
    const uint32_t first[4] = {SOURCE, UNKNOWN, SOURCE, SOURCE};
    int codes[4];
    OK(observe_turn(rt, &d, first, 4, 10, codes));
    assert(codes[0] == 0 && codes[1] == ELPIS_RUNTIME_OBSERVATION_REFUSED && codes[2] == 0);
    assert(codes[3] == ELPIS_RUNTIME_OBSERVATION_REFUSED);
    elpis_runtime_counters c;
    OK(elpis_runtime_read_counters(rt, &c, 0));
    assert(c.k1_observes == 2 && c.k1_commits == 1); /* the refused observations never crossed */
    assert(elpis_runtime_fault(rt) == 0);
    /* Volatile: the authority did not move. A durable publication records the summary of the two observations. */
    assert(status_of(rt).generation == provisioned + 1 && status_of(rt).evidence_count == 0);
    publish(rt, &d);
    s = status_of(rt);
    assert(s.generation == provisioned + 2 && s.evidence_count == 2 && !zero32(s.evidence_head));
    uint8_t head[32];
    memcpy(head, s.evidence_head, 32);
    /* The bound is per turn. */
    const uint32_t one[1] = {SOURCE};
    OK(observe_turn(rt, &d, one, 1, 20, codes));
    assert(codes[0] == 0);

    /* Operator events through the held authority: refused while a turn is open and for RuntimeCore's own kinds. */
    elpis_runtime_observe_begin_result b;
    OK(elpis_runtime_turn_begin_observe(rt, &d, &BUDGET, &b));
    lifecycle(e, op, op_seq, src, ELPIS_ECS_C_ACTIVE, ELPIS_ECS_C_DORMANT);
    uint64_t generation = 0;
    assert(elpis_runtime_authority_transition(rt, e, &generation) == ELPIS_RUNTIME_TURN_OPEN && generation == 0);
    OK(elpis_runtime_turn_abort(rt, &d));
    OK(elpis_runtime_authority_transition(rt, e, &generation));
    op_seq++;
    assert(generation == provisioned + 3 && status_of(rt).generation == generation);
    uint8_t anchor_event[512];
    event(anchor_event, 5, op, op_seq, NULL);
    assert(elpis_runtime_authority_transition(rt, anchor_event, NULL) == ELPIS_RUNTIME_INVALID);
    assert(elpis_runtime_authority_transition(rt, NULL, NULL) == ELPIS_RUNTIME_INVALID);
    /* A DORMANT source is refused; ACTIVE again, admitted. */
    assert(observe_turn(rt, &d, one, 1, 30, codes) == ELPIS_RUNTIME_INVALID); /* nothing observed: kept open */
    assert(codes[0] == ELPIS_RUNTIME_OBSERVATION_REFUSED);
    OK(elpis_runtime_turn_abort(rt, &d));
    lifecycle(e, op, op_seq++, src, ELPIS_ECS_C_DORMANT, ELPIS_ECS_C_ACTIVE);
    OK(elpis_runtime_authority_transition(rt, e, NULL));
    OK(observe_turn(rt, &d, one, 1, 40, codes));
    assert(codes[0] == 0);
    publish(rt, &d); /* two observations since the last publication (ticks 20 and 40) */
    s = status_of(rt);
    assert(s.evidence_count == 4 && memcmp(s.evidence_head, head, 32) != 0);

    /* Save the records, then two more publications move both slots past them. */
    const uint64_t saved_generation = s.generation;
    uint8_t saved_identity[32];
    OK(elpis_runtime_snapshot(rt, &snap));
    memcpy(saved_identity, snap.k1_state_digest, 32);
    const size_t bytes = (size_t)elpis_ecs_c_record_bytes(CAPACITY);
    uint8_t *saved[2];
    char path[4300];
    for (int i = 0; i < 2; i++) {
        saved[i] = malloc(bytes);
        assert(snprintf(path, sizeof(path), "%s/authority.%c", state_dir, 'a' + i) < (int)sizeof(path));
        const int fd = open(path, O_RDONLY);
        assert(saved[i] && fd >= 0 && pread(fd, saved[i], bytes, 0) == (ssize_t)bytes && close(fd) == 0);
    }
    for (uint64_t t = 50; t < 52; ++t) {
        OK(observe_turn(rt, &d, one, 1, t, codes));
        publish(rt, &d);
    }
    s = status_of(rt);
    const uint64_t acknowledged = s.generation;
    assert(acknowledged == saved_generation + 2 && s.evidence_count == 6);
    elpis_runtime_destroy(&rt);

    /* Restart through the authority's own ABI: the same record, verified. */
    EOK(open_authority(&a, &report));
    assert(report.disposition == ELPIS_ECS_C_CLEAN && report.current_generation == acknowledged);
    elpis_ecs_c_status es;
    EOK(elpis_ecs_c_read_status(a, &es));
    assert(es.evidence_count == 6 && es.anchored == 1 && es.state_format == 1 && es.publications == 5);
    EOK(elpis_ecs_c_source_for_authority(a, SOURCE, &ent));
    assert(ent.lifecycle == ELPIS_ECS_C_ACTIVE && !memcmp(ent.id, src, 32));
    elpis_ecs_c_close(a);

    /* Copied-back records: the runtime's open is refused, and the status says why. */
    for (int i = 0; i < 2; i++) {
        assert(snprintf(path, sizeof(path), "%s/authority.%c", state_dir, 'a' + i) < (int)sizeof(path));
        const int fd = open(path, O_WRONLY);
        assert(fd >= 0 && pwrite(fd, saved[i], bytes, 0) == (ssize_t)bytes && close(fd) == 0);
        free(saved[i]);
    }
    OK(elpis_runtime_create_ecs_c((const uint8_t *)state_dir, strlen(state_dir), (const uint8_t *)authority_dir,
                                  strlen(authority_dir), &rt));
    assert(elpis_runtime_open(rt, &snap) == ELPIS_RUNTIME_AUTHORITY);
    s = status_of(rt);
    assert(s.backend == ELPIS_RUNTIME_BACKEND_ECS_C && s.ecs_c_code == ELPIS_ECS_C_RECONCILE_REQUIRED);
    assert(s.disposition == ELPIS_ECS_C_ROLLBACK && s.mode == ELPIS_ECS_C_MODE_ROLLBACK);
    assert(s.generation == saved_generation && s.generation < acknowledged);
    assert(elpis_runtime_anchor(rt, &d, &snap) == ELPIS_RUNTIME_CLOSED);
    assert(elpis_runtime_authority_transition(rt, e, NULL) == ELPIS_RUNTIME_CLOSED);
    elpis_runtime_destroy(&rt);

    /* The operator reconciles exactly what the authority names; the runtime reopens past the abandoned generation. */
    EOK(open_authority(&a, &report));
    assert(report.mode == ELPIS_ECS_C_MODE_ROLLBACK);
    elpis_ecs_c_reconciliation rec;
    EOK(elpis_ecs_c_read_reconciliation(a, &rec));
    assert(rec.kind == 11 && rec.abandoned_generation == acknowledged && rec.class_code == 0);
    event(e, 11, op, next_seq(a, op), NULL);
    put_u64(e + 80, rec.abandoned_generation);
    e[80 + 8] = rec.class_code;
    memcpy(e + 80 + 16, rec.digest, 32);
    EOK(send(a, e));
    elpis_ecs_c_close(a);
    rt = open_runtime();
    OK(elpis_runtime_snapshot(rt, &snap));
    assert(snap.generation == acknowledged + 1 && !memcmp(snap.k1_state_digest, saved_identity, 32));
    s = status_of(rt);
    assert(s.disposition == ELPIS_ECS_C_CLEAN && s.mode == ELPIS_ECS_C_MODE_NORMAL && s.ecs_c_code == 0);
    assert(s.evidence_count == 4); /* the authority it continues from */
    elpis_runtime_destroy(&rt);
    elpis_ecs_composite_destroy(x);

    /* A continuity runtime reports its backend, and has no operator events. */
    char cont[4300];
    assert(snprintf(cont, sizeof(cont), "%s/continuity", scratch) < (int)sizeof(cont));
    OK(elpis_runtime_create((const uint8_t *)cont, strlen(cont), &rt));
    OK(elpis_runtime_open(rt, &snap));
    s = status_of(rt);
    assert(s.backend == ELPIS_RUNTIME_BACKEND_CONTINUITY && s.ecs_c_code == 0 && s.disposition == 0);
    assert(elpis_runtime_authority_transition(rt, e, NULL) == ELPIS_RUNTIME_INVALID);
    elpis_runtime_destroy(&rt);

    /* The scratch is the test's working directory: emptied, never removed (the next run starts in it). */
    sh("rm -rf '%s' && mkdir -p '%s'", scratch, scratch);
    printf("RuntimeCore over ECS_C: PASS\n");
    return 0;
}
