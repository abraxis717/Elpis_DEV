/* elpis: the native Elpis operator host (docs/NATIVE_HOST.md). One process, one RAM-only runtime; no interpreter, no
 * home, no state on disk (docs/ELPIS_MISSION.md, Amendment 3).
 *
 *   elpis (--libraries DIR | --native-build BUILD) catalog
 *   elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] session --script FILE|-
 *   elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] <command> [options]
 *
 * The libraries are read, never written: each is copied read-only into a sealed memfd, measured, and loaded from the
 * seal only when the catalog of this host binary and every library (`catalog` prints it) matches the operator's pin.
 * Session options: --dim N --width N --max-rows N --memory-dim N --banks N --slot-capacity N --codebook-capacity N
 * --seed N (the lineage every session starts from: its deterministic initial composite, built in memory),
 * --capacity N (the session's ECS_C entities, default 16), --nonce-seed HEX (64 hex: reproducible sessions for
 * differential runs; default: the kernel draws a fresh nonce for each session).
 *
 * A session is one volatile ECS_C session and one composite: `open` (or the first operation that needs it) starts
 * one at a fresh genesis, anchored at the lineage's initial identity; `close` ends it and discards everything it
 * learned or recorded. Nothing is published, checkpointed, recovered or resumed: the retired durable commands (init,
 * create, publish, reconcile, candidates, authority-reconcile, --home, --publish) are refused OPERATION_RETIRED.
 *
 * Containment is mandatory (host_contain.h): a privileged process, writable storage on a standard descriptor, an
 * unreadable or unlockable swap policy, or a kernel without Landlock v5 is refused before any operation runs; the
 * process then runs with no write permission to any file, no core dump and no child process.
 *
 * Every operation prints one JSON object on one line: {"op": ..., <its members>, "ok": true} or
 * {"op": ..., "ok": false, "code": ..., "detail": ...}; an operation that names "expect": CODE is met exactly when
 * it is refused with CODE. Exit status: 0 every operation succeeded or was refused as expected; 2 the command line,
 * an argument or a script is malformed, or names a retired command (nothing ran); 3 an operation was refused (or an
 * expectation unmet); 4 the host could not start: containment, admission or the catalog pin refused (nothing ran).
 *
 * The host computes no ECS mathematics and touches no ECS byte: it validates operator input, marshals it into the
 * admitted libraries' C ABIs and reports what they answer. RuntimeCore composes every turn and decides every session
 * transition; structure identifies every hyperedge. There is no text interface: no qualified ECS<->DSV4 codec map
 * exists (ECS_CODEC_UNQUALIFIED). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "elpis/ecs_c_types.h"
#include "elpis/sha256.h"
#include "host_contain.h"
#include "host_install.h"
#include "host_json.h"

#define MAX_INPUT ((size_t)8 << 20)  /* one operation's JSON, or a script */
#define ARENA_PER_BYTE 32            /* parser arena bytes per input byte */
/* One operation's output line: the largest is `authority` of a 4096-entity session (~1.9 MB). The host's memory stays
 * small enough to be locked whole under a common 8 MiB RLIMIT_MEMLOCK when swap is active (host_contain.h). */
#define MAX_OUTPUT ((size_t)2 << 20)
#define MAX_QUERY_ROWS 4096u
#define MAX_EXPERIENCES 64u
#define MAX_EXPERIENCE_ROWS 256u
#define MAX_STEPS (1u << 20)
#define MAX_TURN_OBSERVATIONS 64u
#define HYPEREDGE_NAMESPACE 0x20000000u
#define INCIDENCE_NAMESPACE 0x30000000u

/* ---- the process ---------------------------------------------------------------------------------------------- */

typedef struct {
    host_install d;
    host_containment contain;
    elpis_runtime *rt;       /* created at the first session; each open is a new session */
    int opened;              /* a session is open */
    elpis_ecs_composite *x;  /* the session's composite */
    uint64_t owner;
    elpis_runtime_substrate sub;
    uint8_t *envelope;       /* d.envelope_bytes of memory: `live` reads the component identities from it */
    double *s3;
    elpis_runtime_budget ceiling;
} host;

static const char *rt_code(const host *h, int rc) {
    const char *name = h->d.api.elpis_runtime_code_name(rc);
    return name ? name : "RUNTIME_UNKNOWN_CODE";
}
static const char *composite_code(const host *h, int rc) {
    const char *name = h->d.api.elpis_ecs_composite_code_name(rc);
    return name ? name : "ECS_COMPOSITE_UNKNOWN";
}
/* The session's exact refusal codes (elpis/ecs_c_types.h): the host loads no ECS_C library to name them. */
static const char *ecs_c_code(int code) {
    static const char *const LAW[] = {"ECS_C_OK", "ECS_C_INVALID", "ECS_C_STALE", "ECS_C_UNAUTHORIZED",
                                      "ECS_C_UNKNOWN_ENTITY", "ECS_C_LIFECYCLE", "ECS_C_DUPLICATE",
                                      "ECS_C_SEQUENCE_GAP", "ECS_C_CAPACITY", "ECS_C_EXHAUSTED", "ECS_C_SOURCE_STALE",
                                      "ECS_C_SCOPE_CONFLICT", "ECS_C_UNANCHORED", "ECS_C_ALREADY_ANCHORED",
                                      "ECS_C_STATE_MISMATCH", "ECS_C_EVOLUTION_MISMATCH", "ECS_C_EVOLUTION_PENDING",
                                      "ECS_C_EVOLUTION_NOT_PENDING", "ECS_C_LEGACY_REFUSED"};
    if (code >= 0 && code < (int)(sizeof LAW / sizeof *LAW)) return LAW[code];
    switch (code) {
    case ELPIS_ECS_C_CORRUPT: return "ECS_C_CORRUPT";
    case ELPIS_ECS_C_SESSION_MISMATCH: return "ECS_C_SESSION_MISMATCH";
    case ELPIS_ECS_C_PERSISTENCE_RETIRED: return "ECS_C_PERSISTENCE_RETIRED";
    case ELPIS_ECS_C_NOMEM: return "ECS_C_NOMEM";
    case ELPIS_ECS_C_ENTROPY_UNAVAILABLE: return "ECS_C_ENTROPY_UNAVAILABLE";
    default: return "ECS_C_UNKNOWN_CODE";
    }
}

static const char *const POLARITIES[] = {"NONE", "ASSERT", "NEGATE", "UNCERTAIN"};
static const char *const KINDS[] = {"NONE", "OPERATOR", "RUNTIME", "SOURCE"};
static const char *const LIFECYCLES[] = {"NONE", "FOUNDED", "ACTIVE", "DORMANT", "TERMINATED"};
#define NAME(table, i) ((size_t)(i) < sizeof(table) / sizeof(*(table)) ? (table)[i] : "UNKNOWN")

/* ---- the session: a volatile ECS_C session and the lineage's composite, both in memory ------------------------ */

static void session_close(host *h) {
    const host_api *a = &h->d.api;
    if (h->rt) a->elpis_runtime_close(h->rt); /* aborts an open turn natively; the session ends, nothing remains */
    if (h->x) a->elpis_ecs_composite_destroy(h->x);
    h->x = NULL, h->opened = 0;
}

/* Start a session: a fresh genesis, the lineage's initial composite built in memory, anchored. */
static int session_open(host *h, host_err *e) {
    const host_api *a = &h->d.api;
    if (h->opened) return 0;
    if (!h->rt) {
        const int rc = a->elpis_runtime_create(&h->d.session, &h->rt);
        if (rc) return host_fail(e, rt_code(h, rc), "the runtime could not be created");
    }
    const size_t f = a->elpis_runtime_features(h->d.lineage.dim);
    if (!h->s3 && !(h->s3 = malloc(sizeof(double) * (f ? f : 1)))) return host_fail(e, "HOST_RESOURCES", "out of memory");
    if (!h->envelope && !(h->envelope = malloc(h->d.envelope_bytes)))
        return host_fail(e, "HOST_RESOURCES", "out of memory");
    elpis_continuity_snapshot opened;
    int rc = a->elpis_runtime_open(h->rt, &opened);
    if (rc) return host_fail(e, rt_code(h, rc), "the session could not be opened");
    const elpis_ecs_composite_config cfg = host_composite_config(&h->d);
    double *w = malloc(sizeof(double) * h->d.lineage.dim * h->d.lineage.width);
    if (!w) {
        a->elpis_runtime_close(h->rt);
        return host_fail(e, "HOST_RESOURCES", "out of memory");
    }
    host_initial_weights(&h->d.lineage, w);
    rc = a->elpis_ecs_composite_create(&cfg, w, &h->x);
    free(w);
    uint8_t identity[32];
    if (!rc) rc = a->elpis_ecs_composite_state_digest(h->x, identity);
    if (rc || memcmp(identity, h->d.initial_identity, 32)) {
        session_close(h);
        return host_fail(e, "ECS_STATE", "the lineage's initial composite could not be built");
    }
    h->sub = (elpis_runtime_substrate){ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, h->x, 0, ++h->owner, h->d.lineage.dim,
                                       host_substrate_api()};
    elpis_continuity_snapshot anchored;
    rc = a->elpis_runtime_anchor(h->rt, &h->sub, &anchored);
    if (rc) {
        session_close(h);
        return host_fail(e, rt_code(h, rc), "the session could not anchor the lineage's initial state");
    }
    h->opened = 1;
    return 0;
}

static int require_state(host *h, host_err *e) { return session_open(h, e); }

/* ---- input --------------------------------------------------------------------------------------------------- */

static int input(host_err *e, const char *fmt, const char *what) { return host_fail(e, "CLI_INPUT", fmt, what); }

static int u64_in(const hj_value *v, uint64_t max, uint64_t *out) {
    if (!v || v->kind != HJ_NUMBER || !v->integral || v->integer < 0 || (uint64_t)v->integer > max) return -1;
    *out = (uint64_t)v->integer;
    return 0;
}

static int u64_member(const hj_value *op, const char *key, uint64_t max, uint64_t dflt, uint64_t *out, host_err *e) {
    const hj_value *v = hj_get(op, key);
    *out = dflt;
    if (!v || v->kind == HJ_NULL) return 0;
    if (u64_in(v, max, out)) return host_fail(e, "CLI_INPUT", "%s: an integer in 0..%llu", key, (unsigned long long)max);
    return 0;
}

static int hex_member(const hj_value *op, const char *key, uint8_t out[32], host_err *e) {
    const hj_value *v = hj_get(op, key);
    if (!v || v->kind != HJ_STRING || host_unhex32(v->string, out)) return input(e, "%s: 64 lowercase hex characters", key);
    return 0;
}

/* A HACF type or role id: an int (namespace included) or "<prefix>:N" (the namespace added explicitly). */
static int id_in(const hj_value *v, uint32_t ns, const char *prefix, const char *what, uint32_t *out, host_err *e) {
    uint64_t n;
    if (v && v->kind == HJ_NUMBER && !u64_in(v, UINT32_MAX, &n)) return *out = (uint32_t)n, 0;
    const size_t p = strlen(prefix);
    if (v && v->kind == HJ_STRING && !strncmp(v->string, prefix, p) && v->string[p] == ':' && v->string[p + 1]) {
        char *end;
        errno = 0;
        const unsigned long long k = strtoull(v->string + p + 1, &end, 10);
        if (!*end && !errno && v->string[p + 1] != '-' && v->string[p + 1] != '+' && (ns | k) <= UINT32_MAX)
            return *out = (uint32_t)(ns | k), 0;
    }
    return host_fail(e, "CLI_INPUT", "%s: a 32-bit int or '%s:N'", what, prefix);
}

/* Rows of `dim` numbers into a fresh buffer (the caller frees it). */
static int rows_in(const hj_value *v, uint64_t dim, size_t max_rows, double **out, size_t *rows, host_err *e) {
    if (!v || v->kind != HJ_ARRAY || !v->count || v->count > max_rows)
        return host_fail(e, "CLI_INPUT", "x: a non-empty list of at most %zu rows of %llu numbers", max_rows,
                         (unsigned long long)dim);
    double *x = malloc(sizeof(double) * v->count * dim);
    if (!x) return host_fail(e, "HOST_RESOURCES", "out of memory");
    for (size_t r = 0; r < v->count; ++r) {
        const hj_value *row = &v->items[r];
        if (row->kind != HJ_ARRAY || row->count != dim) {
            free(x);
            return host_fail(e, "CLI_INPUT", "x: a non-empty list of rows of %llu numbers", (unsigned long long)dim);
        }
        for (size_t i = 0; i < dim; ++i) {
            if (row->items[i].kind != HJ_NUMBER) {
                free(x);
                return input(e, "%s: numbers only", "x");
            }
            x[r * dim + i] = row->items[i].number;
        }
    }
    *out = x, *rows = v->count;
    return 0;
}

static const char *str_member(const hj_value *op, const char *key) {
    const hj_value *v = hj_get(op, key);
    return v && v->kind == HJ_STRING ? v->string : NULL;
}

/* A grant states who granted it and why: a non-empty, not only blank string. */
static int stated(const char *s) {
    for (; s && *s; ++s)
        if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r') return 1;
    return 0;
}

/* ---- output -------------------------------------------------------------------------------------------------- */

static int zero32(const uint8_t *p) {
    for (int i = 0; i < 32; ++i)
        if (p[i]) return 0;
    return 1;
}

static void hex_or_null(hjw *w, const char *key, const uint8_t *p) {
    hjw_key(w, key);
    if (zero32(p)) hjw_null(w);
    else hjw_hex(w, p, 32);
}

/* The open session (elpis_runtime_authority_status), or null. */
static void session_out(host *h, hjw *w) {
    elpis_runtime_authority s;
    hjw_key(w, "session");
    if (!h->opened || h->d.api.elpis_runtime_authority_status(h->rt, &s)) {
        hjw_null(w);
        return;
    }
    hjw_object(w);
    hjw_kstr(w, "backend", "ECS_C_SESSION");
    hjw_kstr(w, "durability", "VOLATILE");
    hjw_ku64(w, "ordinal", s.sessions);
    hjw_khex(w, "lineage", s.session, 32);
    hjw_ku64(w, "generation", s.generation);
    hjw_ku64(w, "entity_count", s.entity_count);
    hjw_ku64(w, "capacity", s.capacity);
    hjw_ku64(w, "receipt_count", s.receipt_count);
    hjw_ku64(w, "evidence_count", s.evidence_count);
    hjw_ku64(w, "volatile_bytes", s.volatile_bytes);
    hjw_khex(w, "commit", s.commit, 32);
    hjw_khex(w, "state_root", s.state_root, 32);
    hjw_khex(w, "receipt_head", s.receipt_head, 32);
    hex_or_null(w, "evidence_head", s.evidence_head);
    hjw_key(w, "ecs_c_code");
    if (s.ecs_c_code) hjw_str(w, ecs_c_code(s.ecs_c_code));
    else hjw_null(w);
    hjw_end(w, '}');
}

static void live_out(host *h, hjw *w) {
    uint8_t identity[32];
    uint64_t generation;
    hjw_key(w, "live");
    if (!h->opened || h->d.api.elpis_runtime_live_identity(h->rt, &h->sub, identity, &generation)) {
        hjw_null(w);
        return;
    }
    hjw_object(w);
    hjw_khex(w, "identity", identity, 32);
    hjw_ku64(w, "generation", generation);
    hjw_end(w, '}');
}

/* ---- inspection ---------------------------------------------------------------------------------------------- */

typedef int (*op_fn)(host *, const hj_value *, hjw *, host_err *);

static void budget_out(hjw *w, const elpis_runtime_budget *b) {
    hjw_key(w, "budget_ceiling");
    hjw_object(w);
    hjw_ku64(w, "max_experiences", b->max_experiences);
    hjw_ku64(w, "max_rows", b->max_rows);
    hjw_ku64(w, "max_experience_rows", b->max_experience_rows);
    hjw_ku64(w, "max_steps", b->max_steps);
    hjw_ku64(w, "max_work_units", b->max_work_units);
    hjw_ku64(w, "max_query_rows", b->max_query_rows);
    hjw_ku64(w, "max_observations", b->max_observations);
    hjw_end(w, '}');
}

static const char *const CAPABILITIES[][2] = {
    {"numeric_query", "WORKING, mechanically verified: f_W(x) of the live K1 state (read-only)"},
    {"numeric_learn", "WORKING, mechanically verified: the qualified K1 law (qualified only in its frozen regime) under "
                      "an explicit grant and learning rate; volatile: discarded when the session ends"},
    {"typed_evidence", "WORKING, mechanically verified: validated HACF hyperedge + assertion recorded by an "
                       "observe-only turn, admitted only from ACTIVE sources of the session; volatile"},
    {"lookup", "WORKING: exact relation LOOKUP by net evidence; a conflict between sources is reported, never resolved"},
    {"operator_authority", "WORKING: source founding, lifecycle, versions and reclamation as typed events of the "
                           "volatile ECS_C session"},
    {"persistence", "NONE: the host writes no file and its containment grants no write permission; every session "
                    "starts at a fresh genesis from the lineage's initial state"},
    {"recoverable_publication", "RETIRED (OPERATION_RETIRED): no persistent writer exists"},
    {"restart_recovery", "RETIRED (OPERATION_RETIRED): a restart is a new session"},
    {"text", "UNAVAILABLE: ECS_CODEC_UNQUALIFIED (no qualified ECS<->DSV4 codec map)"},
    {"memory_learn", "UNQUALIFIED: never enabled (no memory capability is ever set)"},
    {"memory_slow_consolidation", "NOT PROVIDED"},
    {"contextual_dynamics", "NOT IMPLEMENTED: no qualified law (the HCA substrate is gated research)"},
};
static const char *const NONCLAIMS[] = {"no language understanding", "no general intelligence",
                                        "no autonomous evolution", "no hierarchical superiority",
                                        "an observation is evidence storage, never truth or authority",
                                        "pins authenticate bytes, not authors",
                                        "containment covers this process: it does not make the operating system, "
                                        "its other processes or the reader of this output write nothing"};

static void lineage_out(hjw *w, const host_install *d) {
    const host_lineage *l = &d->lineage;
    hjw_key(w, "lineage");
    hjw_object(w);
    hjw_ku64(w, "dim", l->dim);
    hjw_ku64(w, "width", l->width);
    hjw_ku64(w, "max_rows", l->max_rows);
    hjw_ku64(w, "memory_dim", l->memory_dim);
    hjw_ku64(w, "banks", l->banks);
    hjw_ku64(w, "slot_capacity", l->slot_capacity);
    hjw_ku64(w, "codebook_capacity", l->codebook_capacity);
    hjw_ku64(w, "seed", l->seed);
    hjw_khex(w, "initial_identity", d->initial_identity, 32);
    hjw_khex(w, "initial_w_sha256", d->initial_w_sha256, 32);
    hjw_ku64(w, "envelope_bytes", d->envelope_bytes);
    hjw_ku64(w, "session_capacity", d->session.capacity);
    hjw_kbool(w, "reproducible_nonce", !zero32(d->session.nonce));
    hjw_end(w, '}');
}

static void containment_out(hjw *w, const host_containment *c) {
    hjw_key(w, "containment");
    hjw_object(w);
    hjw_kbool(w, "restricted", c->restricted);
    hjw_ku64(w, "landlock_abi", (uint64_t)c->landlock_abi);
    hjw_ku64(w, "write_grants", 0);
    hjw_ku64(w, "core_limit", c->core_limit);
    hjw_kbool(w, "dumpable", c->dumpable);
    hjw_ku64(w, "fsize_limit", c->fsize_limit);
    hjw_ku64(w, "swap_areas", c->swap_areas);
    hjw_kbool(w, "memory_locked", c->memory_locked);
    hjw_end(w, '}');
}

static int op_status(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op, (void)e;
    const host_install *d = &h->d;
    hjw_key(w, "host");
    hjw_object(w);
    hjw_kstr(w, "schema", "elpis.native-host.v2");
    hjw_kbool(w, "testing_build", d->testing);
    hjw_khex(w, "catalog_sha256", d->catalog_sha256, 32);
    hjw_khex(w, "host_sha256", d->host_sha256, 32);
    hjw_kstr(w, "persistence", "NONE");
    hjw_end(w, '}');
    hjw_key(w, "abi");
    hjw_object(w);
    hjw_ku64(w, "runtime", ELPIS_RUNTIME_ABI_V8);
    hjw_ku64(w, "composite", ELPIS_ECS_COMPOSITE_ABI_V1);
    hjw_end(w, '}');
    hjw_key(w, "libraries");
    hjw_object(w);
    for (int i = 0; i < HOST_LIBRARIES; ++i) {
        hjw_key(w, d->pins[i].id);
        hjw_object(w);
        hjw_khex(w, "sha256", d->pins[i].sha256, 32);
        hjw_ku64(w, "size", d->pins[i].size);
        hjw_end(w, '}');
    }
    hjw_end(w, '}');
    containment_out(w, &h->contain);
    budget_out(w, &h->ceiling);
    hjw_key(w, "capabilities");
    hjw_object(w);
    for (size_t i = 0; i < sizeof CAPABILITIES / sizeof *CAPABILITIES; ++i) hjw_kstr(w, CAPABILITIES[i][0], CAPABILITIES[i][1]);
    hjw_end(w, '}');
    hjw_key(w, "nonclaims");
    hjw_array(w);
    for (size_t i = 0; i < sizeof NONCLAIMS / sizeof *NONCLAIMS; ++i) hjw_str(w, NONCLAIMS[i]);
    hjw_end(w, ']');
    lineage_out(w, d);
    session_out(h, w);
    live_out(h, w);
    if (h->opened) {
        elpis_ecs_composite_stats st;
        if (!h->d.api.elpis_ecs_composite_inspect(h->x, &st)) {
            hjw_key(w, "footprint");
            hjw_object(w);
            hjw_ku64(w, "envelope_bytes", st.envelope_bytes);
            hjw_ku64(w, "allocated_bytes", st.allocated_bytes);
            hjw_ku64(w, "memory_state_bytes", st.memory_state_bytes);
            hjw_ku64(w, "memory_capacity_bytes", st.memory_capacity_bytes);
            hjw_ku64(w, "memory_slots_used", st.memory_slots_used);
            hjw_ku64(w, "memory_codebook_used", st.memory_codebook_used);
            hjw_ku64(w, "k1_epoch", st.k1_epoch);
            hjw_ku64(w, "memory_clock", st.memory_clock);
            hjw_ku64(w, "capabilities", st.capabilities);
            hjw_end(w, '}');
        }
    }
    const int fault = h->rt ? h->d.api.elpis_runtime_fault(h->rt) : 0;
    hjw_key(w, "fault");
    if (fault) hjw_str(w, rt_code(h, fault));
    else hjw_null(w);
    return 0;
}

static int op_open(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    if (session_open(h, e)) return -1;
    session_out(h, w);
    live_out(h, w);
    return 0;
}

/* End the session: everything it learned, recorded or founded is discarded (it was never anywhere but memory). */
static int op_close(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op, (void)e;
    const int was_open = h->opened;
    hjw_key(w, "discarded");
    if (was_open) {
        uint8_t identity[32];
        uint64_t generation = 0;
        elpis_runtime_authority s;
        const int live = h->d.api.elpis_runtime_live_identity(h->rt, &h->sub, identity, &generation) == 0;
        const int status = h->d.api.elpis_runtime_authority_status(h->rt, &s) == 0;
        hjw_object(w);
        hex_or_null(w, "live_identity", live ? identity : (const uint8_t[32]){0});
        hjw_ku64(w, "live_generation", live ? generation : 0);
        hjw_ku64(w, "session_generation", status ? s.generation : 0);
        hjw_kbool(w, "learned", live && memcmp(identity, h->d.initial_identity, 32) != 0);
        hjw_end(w, '}');
    } else {
        hjw_null(w);
    }
    session_close(h);
    hjw_kbool(w, "closed", was_open);
    return 0;
}

static int op_live(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    uint8_t identity[32], composite[32], k1[32], memory[32];
    uint64_t generation;
    int rc = a->elpis_runtime_live_identity(h->rt, &h->sub, identity, &generation);
    if (rc) return host_fail(e, rt_code(h, rc), "the live identity could not be read");
    /* The composite's envelope, written into this process's memory (never to a file), names its components. */
    rc = a->elpis_ecs_composite_snapshot_write(h->x, h->envelope, h->d.envelope_bytes);
    if (!rc) rc = a->elpis_ecs_composite_envelope_identities(h->envelope, h->d.envelope_bytes, composite, k1, memory);
    if (rc) return host_fail(e, composite_code(h, rc), "the live envelope could not be read");
    hjw_key(w, "live");
    hjw_object(w);
    hjw_khex(w, "identity", identity, 32);
    hjw_ku64(w, "generation", generation);
    hjw_khex(w, "k1_identity", k1, 32);
    hjw_khex(w, "memory_identity", memory, 32);
    hjw_end(w, '}');
    hjw_khex(w, "initial_identity", h->d.initial_identity, 32);
    return 0;
}

static int op_counters(host *h, const hj_value *op, hjw *w, host_err *e) {
    const hj_value *reset = hj_get(op, "reset");
    if (require_state(h, e)) return -1;
    elpis_runtime_counters c;
    const int rc = h->d.api.elpis_runtime_read_counters(h->rt, &c, reset && reset->kind == HJ_BOOL && reset->boolean);
    if (rc) return host_fail(e, rt_code(h, rc), "the counters could not be read");
    hjw_key(w, "counters");
    hjw_object(w);
#define C(f) hjw_ku64(w, #f, c.f);
    C(k1_state_digests) C(k1_reserves) C(k1_txn_begins) C(k1_run_schedules) C(k1_commits) C(k1_aborts)
    C(session_transitions) C(k1_queries) C(k1_shapes) C(k1_lease_claims) C(k1_identities) C(k1_observes)
#undef C
    hjw_end(w, '}');
    return 0;
}

/* ---- cognition: numeric QUERY and LEARN ------------------------------------------------------------------------ */

static void values_digest(const double *v, size_t n, uint8_t out[32]) {
    elpis_sha256_ctx s;
    elpis_sha256_init(&s);
    for (size_t i = 0; i < n; ++i) {
        uint64_t bits;
        uint8_t le[8];
        memcpy(&bits, &v[i], 8);
        for (int b = 0; b < 8; ++b) le[b] = (uint8_t)(bits >> (8 * b));
        elpis_sha256_update(&s, le, 8);
    }
    elpis_sha256_final(&s, out);
}

static int op_query(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    double *x;
    size_t rows;
    if (rows_in(hj_get(op, "x"), h->d.lineage.dim, MAX_QUERY_ROWS, &x, &rows, e)) return -1;
    double *out = malloc(sizeof(double) * rows);
    if (!out) {
        free(x);
        return host_fail(e, "HOST_RESOURCES", "out of memory");
    }
    elpis_runtime_query_result r;
    const int rc = a->elpis_runtime_query(h->rt, &h->sub, x, rows * h->d.lineage.dim, out, rows, &h->ceiling, &r);
    free(x);
    if (rc) {
        free(out);
        return host_fail(e, rt_code(h, rc), "QUERY refused%s%s", r.k1_status ? ": " : "",
                         r.k1_status ? composite_code(h, r.k1_status) : "");
    }
    uint8_t digest[32];
    values_digest(out, rows, digest);
    hjw_key(w, "values");
    hjw_array(w);
    for (size_t i = 0; i < rows; ++i) hjw_double(w, out[i]);
    hjw_end(w, ']');
    free(out);
    hjw_khex(w, "values_sha256", digest, 32);
    hjw_ku64(w, "live_generation", r.live_generation);
    hjw_ku64(w, "rows", rows);
    return 0;
}

static int op_learn(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    const uint64_t dim = h->d.lineage.dim;
    const hj_value *sched = hj_get(op, "schedule"), *yv = hj_get(op, "y");
    if (!sched || sched->kind != HJ_ARRAY) return input(e, "%s: a list of [rows, steps] pairs", "schedule");
    elpis_runtime_experience schedule[MAX_EXPERIENCES];
    uint64_t total = 0, steps = 0;
    for (size_t i = 0; i < sched->count; ++i) {
        const hj_value *p = &sched->items[i];
        uint64_t r, s;
        if (p->kind != HJ_ARRAY || p->count != 2 || u64_in(&p->items[0], INT64_MAX, &r) ||
            u64_in(&p->items[1], INT64_MAX, &s))
            return input(e, "%s: a list of [rows, steps] pairs", "schedule");
        if (i < MAX_EXPERIENCES) schedule[i] = (elpis_runtime_experience){r, s};
        if (r < 1 || r > MAX_EXPERIENCE_ROWS || s < 1 || s > MAX_STEPS)
            return host_fail(e, "STIMULUS", "every experience: 1..256 rows and 1..2^20 K1 steps");
        total += r, steps += s;
    }
    if (!yv || yv->kind != HJ_ARRAY) return input(e, "%s: a list of numbers", "y");
    for (size_t i = 0; i < yv->count; ++i)
        if (yv->items[i].kind != HJ_NUMBER) return input(e, "%s: a list of numbers", "y");
    double *x;
    size_t rows;
    if (rows_in(hj_get(op, "x"), dim, SIZE_MAX / (64 * sizeof(double)), &x, &rows, e)) return -1;
    if (!sched->count || sched->count > MAX_EXPERIENCES) {
        free(x);
        return host_fail(e, "STIMULUS", "1..64 experiences as (rows, steps) pairs");
    }
    if (yv->count != total || rows != total) {
        free(x);
        return host_fail(e, "STIMULUS", "x and y must hold exactly the scheduled rows");
    }
    /* The learning authority: an explicit positive finite rate and a stated grant. */
    const hj_value *grant = hj_get(op, "grant"), *rate = hj_get(op, "rate");
    const double lr = rate && rate->kind == HJ_NUMBER ? rate->number : 0.0;
    int refused = 0;
    if (grant && grant->kind != HJ_NULL) {
        if (!isfinite(lr) || lr <= 0)
            refused = host_fail(e, "LEARNING_RATE", "explicit positive finite learning rate");
        else if (grant->kind != HJ_STRING || !stated(grant->string))
            refused = host_fail(e, "LEARN_AUTHORITY", "a learning grant must state who granted it and why");
    } else {
        refused = host_fail(e, "LEARN_UNAUTHORIZED", "LEARN requires an explicit grant");
    }
    double *y = refused ? NULL : malloc(sizeof(double) * (total ? total : 1));
    if (!refused && !y) refused = host_fail(e, "HOST_RESOURCES", "out of memory");
    if (refused) {
        free(x);
        return -1;
    }
    for (size_t i = 0; i < total; ++i) y[i] = yv->items[i].number;
    elpis_runtime_turn_begin_result b;
    const size_t f = a->elpis_runtime_features(dim);
    int rc = a->elpis_runtime_turn_begin(h->rt, &h->sub, x, total * dim, y, total, schedule, sched->count, lr,
                                         &h->ceiling, h->s3, f, &b);
    free(x);
    free(y);
    if (rc)
        return host_fail(e, rt_code(h, rc), "LEARN refused before any commit%s%s", b.k1_status ? ": " : "",
                         b.k1_status ? composite_code(h, b.k1_status) : "");
    elpis_runtime_volatile_commit c;
    rc = a->elpis_runtime_turn_commit(h->rt, &h->sub, &c);
    if (rc) return host_fail(e, rt_code(h, rc), "the LEARN commit was refused; nothing was installed");
    uint8_t identity[32];
    a->elpis_ecs_composite_state_digest(h->x, identity);
    hjw_kstr(w, "durability", "VOLATILE");
    hjw_ku64(w, "experiences", sched->count);
    hjw_ku64(w, "rows", total);
    hjw_ku64(w, "steps", steps);
    hjw_key(w, "epoch");
    hjw_array(w);
    hjw_u64(w, c.transition.epoch_before);
    hjw_u64(w, c.transition.epoch_after);
    hjw_end(w, ']');
    hjw_key(w, "generation");
    hjw_array(w);
    hjw_u64(w, c.transition.generation_before);
    hjw_u64(w, c.transition.generation_after);
    hjw_end(w, ']');
    hjw_kstr(w, "grant", grant->string);
    hjw_khex(w, "identity", identity, 32);
    return 0;
}

/* ---- typed structural evidence ---------------------------------------------------------------------------------- */

typedef struct {
    elpis_semantic_hyperedge_v1 edge;
    elpis_semantic_assertion_v1 assertion;
    elpis_ecs_mem_observation observation;
    uint32_t polarity;
    uint64_t tick;
    uint8_t key[32];
} evidence_item;

static int polarity_of(const char *s) {
    for (uint32_t p = 1; p <= 3; ++p)
        if (s && !strcmp(s, POLARITIES[p])) return (int)p;
    return 0;
}

static int evidence_in(host *h, const hj_value *it, evidence_item *ev, host_err *e) {
    const host_api *a = &h->d.api;
    const hj_value *parts = hj_get(it, "participants");
    if (it->kind != HJ_OBJECT || !parts || parts->kind != HJ_ARRAY)
        return input(e, "%s: relation, participants, provenance, authority, polarity, tick", "evidence");
    if (!parts->count || parts->count > SEMANTIC_MAX_PARTICIPANTS)
        return host_fail(e, "OBSERVATION_INVALID", "participants: 1..%u", SEMANTIC_MAX_PARTICIPANTS);
    memset(ev, 0, sizeof *ev);
    elpis_semantic_hyperedge_v1 *edge = &ev->edge;
    edge->abi_version = 1;
    for (size_t i = 0; i < parts->count; ++i) {
        const hj_value *p = &parts->items[i];
        elpis_semantic_participant_descriptor *d = &edge->participants[i];
        uint64_t ordinal = 0;
        const hj_value *q = hj_get(p, "query_local");
        if (p->kind != HJ_OBJECT || hex_member(p, "node", d->node_identity.bytes, e) ||
            id_in(hj_get(p, "role"), INCIDENCE_NAMESPACE, "incidence", "participant role", &d->incidence_role, e) ||
            u64_member(p, "ordinal", UINT32_MAX, 0, &ordinal, e))
            return -1;
        if (q && q->kind != HJ_BOOL && q->kind != HJ_NULL) return input(e, "%s: a boolean", "query_local");
        d->ordinal = (uint32_t)ordinal;
        d->participant_flags = q && q->kind == HJ_BOOL && q->boolean ? SEMANTIC_PARTICIPANT_FLAG_QUERY : 0;
    }
    if (id_in(hj_get(it, "relation"), HYPEREDGE_NAMESPACE, "hyperedge", "relation", &edge->hyperedge_type, e)) return -1;
    edge->participant_count = (uint32_t)parts->count;
    /* Structure identifies and validates; nothing here canonicalizes or hashes a record. */
    int rc = a->elpis_semantic_canonicalize_participants(edge->participants, edge->participant_count);
    if (!rc) rc = a->elpis_semantic_hyperedge_identity(edge, &edge->hyperedge_identity);
    if (!rc) rc = a->elpis_semantic_hyperedge_validate(edge);
    if (rc) return host_fail(e, "OBSERVATION_STRUCTURE", "structure refused the hyperedge (%d)", rc);
    elpis_semantic_assertion_v1 *as = &ev->assertion;
    as->abi_version = 1;
    as->asserted_object_kind = SEMANTIC_OBJECT_KIND_HYPEREDGE;
    as->asserted_object_digest = edge->hyperedge_identity;
    uint64_t authority;
    if (hex_member(it, "provenance", as->provenance_digest.bytes, e)) return -1;
    if (u64_in(hj_get(it, "authority"), UINT32_MAX, &authority))
        return host_fail(e, "OBSERVATION_INVALID", "authority: a 32-bit unsigned integer");
    as->authority = (uint32_t)authority;
    rc = a->elpis_semantic_assertion_identity(as, &as->assertion_identity);
    if (!rc) rc = a->elpis_semantic_assertion_validate(as);
    if (rc) return host_fail(e, "OBSERVATION_STRUCTURE", "structure refused the assertion (%d)", rc);
    ev->polarity = (uint32_t)polarity_of(str_member(it, "polarity"));
    if (!ev->polarity)
        return host_fail(e, "OBSERVATION_INVALID", "polarity: ASSERT, NEGATE or UNCERTAIN (the caller's)");
    if (u64_in(hj_get(it, "tick"), INT64_MAX, &ev->tick))
        return host_fail(e, "OBSERVATION_INVALID", "tick: a 64-bit unsigned integer");
    static const char *const ADAPTER[] = {"OBSERVATION_INVALID", "OBSERVATION_INVALID", "OBSERVATION_STRUCTURE",
                                          "OBSERVATION_IDENTITY", "OBSERVATION_NONCANONICAL",
                                          "OBSERVATION_UNSUPPORTED"};
    rc = a->elpis_ecs_obs_hacf_observation(edge, as, ev->polarity, ev->tick, &ev->observation);
    if (rc) return host_fail(e, rc >= -5 ? ADAPTER[-rc] : "OBSERVATION_INVALID", "the HACF observation adapter refused (%d)", rc);
    return 0;
}

static int op_evidence(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    const hj_value *items = hj_get(op, "items");
    if (!items || items->kind != HJ_ARRAY || !items->count)
        return input(e, "%s: a non-empty list of evidence objects", "items");
    if (items->count > MAX_TURN_OBSERVATIONS)
        return host_fail(e, "OBSERVATION_AUTHORITY", "more observations than one turn admits (%u)", MAX_TURN_OBSERVATIONS);
    static evidence_item ev[MAX_TURN_OBSERVATIONS];
    for (size_t i = 0; i < items->count; ++i)
        if (evidence_in(h, &items->items[i], &ev[i], e)) return -1;
    /* The observation authority: a stated grant over an explicit set of sources, every item from one of them. */
    const hj_value *grant = hj_get(op, "grant"), *sources = hj_get(op, "sources");
    if (!grant || grant->kind == HJ_NULL) return host_fail(e, "OBSERVATION_UNAUTHORIZED", "OBSERVE requires an explicit grant");
    if (grant->kind != HJ_STRING || !stated(grant->string))
        return host_fail(e, "OBSERVATION_AUTHORITY", "an observation grant must state who granted it and why");
    if (!sources || sources->kind != HJ_ARRAY || !sources->count)
        return host_fail(e, "OBSERVATION_AUTHORITY", "sources: a non-empty list of 32-bit source ids");
    for (size_t i = 0; i < sources->count; ++i) {
        uint64_t s;
        if (u64_in(&sources->items[i], UINT32_MAX, &s))
            return host_fail(e, "OBSERVATION_AUTHORITY", "sources: a non-empty list of 32-bit source ids");
    }
    for (size_t i = 0; i < items->count; ++i) {
        int admitted = 0;
        for (size_t j = 0; j < sources->count; ++j)
            admitted |= (uint64_t)sources->items[j].integer == ev[i].observation.source_authority;
        if (!admitted)
            return host_fail(e, "OBSERVATION_AUTHORITY", "an observation from a source the grant does not admit (%u)",
                             ev[i].observation.source_authority);
    }
    /* One observe-only turn, all or nothing: any refused observation aborts it. */
    elpis_runtime_budget budget = h->ceiling;
    if (budget.max_observations > MAX_TURN_OBSERVATIONS) budget.max_observations = MAX_TURN_OBSERVATIONS;
    elpis_runtime_observe_begin_result b;
    int rc = a->elpis_runtime_turn_begin_observe(h->rt, &h->sub, &budget, &b);
    if (rc) return host_fail(e, rt_code(h, rc), "the observe-only turn was refused%s%s", b.k1_status ? ": " : "",
                             b.k1_status ? composite_code(h, b.k1_status) : "");
    for (size_t i = 0; i < items->count; ++i) {
        elpis_runtime_observe_result r;
        rc = a->elpis_runtime_turn_observe(h->rt, &h->sub, &ev[i].observation, &r);
        if (rc) {
            a->elpis_runtime_turn_abort(h->rt, &h->sub); /* all or nothing: no partial candidate commits */
            return host_fail(e, rt_code(h, rc), "observation %zu refused (memory status %d); nothing was installed", i,
                             r.memory_status);
        }
        memcpy(ev[i].key, r.relation_key, 32);
    }
    elpis_runtime_volatile_commit c;
    rc = a->elpis_runtime_turn_commit(h->rt, &h->sub, &c);
    if (rc) return host_fail(e, rt_code(h, rc), "the observe-only commit was refused; nothing was installed");
    if (c.transition.generation_before != b.live_generation)
        return host_fail(e, "RUNTIME_INVALID", "the observe-only commit does not continue the turn's generation");
    uint8_t identity[32];
    a->elpis_ecs_composite_state_digest(h->x, identity);
    hjw_kstr(w, "durability", "VOLATILE");
    hjw_ku64(w, "k1_steps", c.transition.steps);
    hjw_key(w, "generation");
    hjw_array(w);
    hjw_u64(w, c.transition.generation_before);
    hjw_u64(w, c.transition.generation_after);
    hjw_end(w, ']');
    hjw_key(w, "entries");
    hjw_array(w);
    for (size_t i = 0; i < items->count; ++i) {
        hjw_object(w);
        hjw_khex(w, "relation_key", ev[i].key, 32);
        hjw_khex(w, "hyperedge", ev[i].edge.hyperedge_identity.bytes, 32);
        hjw_khex(w, "assertion", ev[i].assertion.assertion_identity.bytes, 32);
        hjw_khex(w, "provenance", ev[i].assertion.provenance_digest.bytes, 32);
        hjw_ku64(w, "source_authority", ev[i].assertion.authority);
        hjw_kstr(w, "polarity", POLARITIES[ev[i].polarity]);
        hjw_ku64(w, "tick", ev[i].tick);
        hjw_end(w, '}');
    }
    hjw_end(w, ']');
    hjw_kstr(w, "grant", grant->string);
    hjw_khex(w, "identity", identity, 32);
    return 0;
}

static int op_lookup(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    const hj_value *cues = hj_get(op, "cues");
    if (!cues || cues->kind != HJ_ARRAY || !cues->count)
        return input(e, "%s: a non-empty list of {role, node, ordinal}", "cues");
    if (cues->count > ELPIS_ECS_MEM_MAX_PARTICIPANTS - 1)
        return host_fail(e, "CLI_INPUT", "cues: at most %u", ELPIS_ECS_MEM_MAX_PARTICIPANTS - 1);
    elpis_ecs_mem_request q;
    memset(&q, 0, sizeof q);
    q.abi_version = ELPIS_ECS_MEM_ABI_V1;
    q.channel = ELPIS_ECS_MEM_LOOKUP;
    q.candidate_source = ELPIS_ECS_MEM_CANDIDATES_CALLER;
    q.cue_count = (uint32_t)cues->count;
    for (size_t i = 0; i < cues->count; ++i) {
        const hj_value *c = &cues->items[i];
        uint64_t ordinal;
        if (c->kind != HJ_OBJECT || id_in(hj_get(c, "role"), INCIDENCE_NAMESPACE, "incidence", "cue role", &q.cue[i].role, e) ||
            u64_member(c, "ordinal", UINT32_MAX, 0, &ordinal, e) || hex_member(c, "node", q.cue[i].entity.b, e))
            return -1;
        q.cue[i].ordinal = (uint32_t)ordinal;
    }
    if (id_in(hj_get(op, "relation"), HYPEREDGE_NAMESPACE, "hyperedge", "relation", &q.relation_type, e) ||
        id_in(hj_get(op, "target_role"), INCIDENCE_NAMESPACE, "incidence", "target role", &q.target_role, e))
        return -1;
    elpis_ecs_mem_answer ans;
    uint8_t digest[32];
    int ms = 0;
    const int rc = a->elpis_ecs_composite_memory_query(h->x, &q, &ans, digest, &ms);
    if (rc) return host_fail(e, composite_code(h, rc), "LOOKUP refused (memory status %d)", ms);
    hjw_kbool(w, "found", (int)ans.found);
    hjw_key(w, "answer");
    if (ans.found) hjw_hex(w, ans.answer.b, 32);
    else hjw_null(w);
    hjw_ku64(w, "matches", ans.matches);
    hjw_ku64(w, "conflicts", ans.conflicts);
    hjw_ku64(w, "contradicted", ans.contradicted);
    hjw_ku64(w, "uncertain", ans.uncertain);
    hjw_khex(w, "state_identity", digest, 32);
    return 0;
}

static int op_slot(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    elpis_ecs_mem_digest key;
    if (hex_member(op, "key", key.b, e)) return -1;
    elpis_ecs_mem_slot s;
    int ms = 0;
    const int rc = a->elpis_ecs_composite_memory_slot_read(h->x, &key, &s, &ms);
    if (rc && ms == ELPIS_ECS_MEM_NOT_FOUND) return hjw_kbool(w, "found", 0), 0;
    if (rc) return host_fail(e, composite_code(h, rc), "the slot could not be read (memory status %d)", ms);
    elpis_semantic_hyperedge_v1 edge;
    const int inv = a->elpis_ecs_obs_hacf_hyperedge(&s, &edge);
    if (inv) return host_fail(e, "OBSERVATION_INVALID", "the HACF observation inverse refused (%d)", inv);
    hjw_kbool(w, "found", 1);
    hjw_ku64(w, "relation_type", s.relation_type);
    hjw_ku64(w, "assert_count", s.assert_count);
    hjw_ku64(w, "negate_count", s.negate_count);
    hjw_ku64(w, "uncertain_count", s.uncertain_count);
    hjw_kstr(w, "last_polarity", NAME(POLARITIES, s.last_polarity));
    hjw_ku64(w, "last_authority", s.last_authority);
    hjw_khex(w, "first_provenance", s.first_provenance.b, 32);
    hjw_khex(w, "last_provenance", s.last_provenance.b, 32);
    hjw_khex(w, "structure_ref", s.structure_ref.b, 32);
    hjw_khex(w, "hyperedge", edge.hyperedge_identity.bytes, 32);
    hjw_key(w, "participants");
    hjw_array(w);
    for (uint32_t i = 0; i < s.participant_count && i < ELPIS_ECS_MEM_MAX_PARTICIPANTS; ++i) {
        hjw_object(w);
        hjw_ku64(w, "role", s.participants[i].role);
        hjw_ku64(w, "ordinal", s.participants[i].ordinal);
        hjw_khex(w, "node", s.participants[i].entity.b, 32);
        hjw_end(w, '}');
    }
    hjw_end(w, ']');
    return 0;
}

/* ---- the session's authority: inspection and the operator's events --------------------------------------------- */

static int entity_at(host *h, uint32_t i, elpis_runtime_authority_entity *out, host_err *e) {
    const int rc = h->d.api.elpis_runtime_authority_entity_at(h->rt, i, out);
    if (rc) return host_fail(e, rt_code(h, rc), "entity %u could not be read", i);
    return 0;
}

static int entity_count(host *h, uint32_t *n, host_err *e) {
    elpis_runtime_authority s;
    const int rc = h->d.api.elpis_runtime_authority_status(h->rt, &s);
    if (rc) return host_fail(e, rt_code(h, rc), "the session's status could not be read");
    return *n = s.entity_count, 0;
}

/* One event through RuntimeCore, which validates it and sends it to the session at the session's coordinate. */
static int send_event(host *h, const uint8_t event[512], uint64_t *generation, host_err *e) {
    const host_api *a = &h->d.api;
    const int rc = a->elpis_runtime_authority_transition(h->rt, event, generation);
    if (!rc) return 0;
    /* A refusal of the session itself (the authority code space, or RUNTIME_AUTHORITY) is recorded by RuntimeCore as
     * its exact last code: report that one. RuntimeCore's own refusals (RUNTIME_*) are its own. */
    elpis_runtime_authority s;
    const int last = a->elpis_runtime_authority_status(h->rt, &s) ? 0 : s.ecs_c_code;
    if (last && (rc == ELPIS_RUNTIME_AUTHORITY || rc < 64))
        return host_fail(e, ecs_c_code(last), "the session refused the event; nothing changed (through RuntimeCore: "
                                              "%s)", rt_code(h, rc));
    return host_fail(e, rt_code(h, rc), "RuntimeCore refused the event before the session saw it");
}

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* An operator event (elpis/ecs_c_types.h): the session's founding operator (row 0) at its next sequence. */
static int operator_event(host *h, uint16_t kind, const uint8_t *receiver, uint8_t ev[512], host_err *e) {
    elpis_runtime_authority_entity op;
    if (entity_at(h, 0, &op, e)) return -1;
    if (op.kind != ELPIS_ECS_C_OPERATOR) return host_fail(e, "ECS_C_UNAUTHORIZED", "row 0 is not the operator");
    memset(ev, 0, 512);
    ev[0] = (uint8_t)kind, ev[1] = (uint8_t)(kind >> 8);
    memcpy(ev + 8, op.id, 32);
    put_u64(ev + 40, op.sender_watermark + 1);
    if (receiver) memcpy(ev + 48, receiver, 32);
    return 0;
}

static int find_source(host *h, uint32_t source_authority, elpis_runtime_authority_entity *out, host_err *e) {
    uint32_t n = 0;
    if (entity_count(h, &n, e)) return -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (entity_at(h, i, out, e)) return -1;
        if (out->kind == ELPIS_ECS_C_SOURCE && out->source_authority == source_authority) return 0;
    }
    return host_fail(e, "ECS_C_UNKNOWN_ENTITY", "no source owns source authority %u", source_authority);
}

static void entity_out(hjw *w, const elpis_runtime_authority_entity *x, uint32_t index) {
    hjw_object(w);
    hjw_ku64(w, "index", index);
    hjw_khex(w, "id", x->id, 32);
    hjw_kstr(w, "kind", NAME(KINDS, x->kind));
    hjw_kstr(w, "lifecycle", NAME(LIFECYCLES, x->lifecycle));
    hjw_ku64(w, "per_turn_limit", x->per_turn_limit);
    hjw_ku64(w, "founding_index", x->founding_index);
    hjw_ku64(w, "founded_generation", x->founded_generation);
    hjw_ku64(w, "last_generation", x->last_generation);
    hjw_ku64(w, "sender_watermark", x->sender_watermark);
    hjw_ku64(w, "source_version", x->source_version);
    hjw_khex(w, "label", x->label, 32);
    hjw_khex(w, "scope", x->scope, 32);
    hjw_ku64(w, "source_authority", x->source_authority);
    hjw_end(w, '}');
}

static int op_authority(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    if (session_open(h, e)) return -1;
    session_out(h, w);
    uint32_t n = 0;
    int rc = entity_count(h, &n, e);
    hjw_key(w, "entities");
    hjw_array(w);
    for (uint32_t i = 0; !rc && i < n; ++i) {
        elpis_runtime_authority_entity x;
        rc = entity_at(h, i, &x, e);
        if (!rc) entity_out(w, &x, i);
    }
    hjw_end(w, ']');
    return rc;
}

static int lifecycle_of(const char *s) {
    for (int l = 1; l <= 4; ++l)
        if (s && !strcmp(s, LIFECYCLES[l])) return l;
    return 0;
}

/* A source's scope: "scope" (64 lowercase hex) or "scope_label" (text, hashed as a label); exactly one, never zero. */
static int scope_in(const hj_value *op, int required, uint8_t scope[32], host_err *e) {
    const hj_value *raw = hj_get(op, "scope");
    const char *text = str_member(op, "scope_label");
    if (raw && text) return input(e, "%s: give scope or scope_label, not both", "scope");
    if (raw) return hex_member(op, "scope", scope, e);
    if (text && *text) return host_label(text, scope), 0;
    return required ? input(e, "%s: a source needs a scope (scope: 64 hex, or scope_label: text)", "scope") : 0;
}

static int op_source_found(host *h, const hj_value *op, hjw *w, host_err *e) {
    const char *text = str_member(op, "label");
    uint64_t authority, bound, version;
    uint8_t scope[32] = {0}, label[32], id[32];
    const hj_value *activate = hj_get(op, "activate");
    if (!text || !*text) return input(e, "%s: a non-empty string (hashed into the 32-byte label)", "label");
    if (u64_member(op, "source_authority", UINT32_MAX, UINT64_MAX, &authority, e) ||
        u64_member(op, "per_turn", ELPIS_ECS_C_MAX_PER_TURN, 1, &bound, e) ||
        u64_member(op, "version", INT64_MAX, 1, &version, e) || scope_in(op, 1, scope, e))
        return -1;
    if (authority == UINT64_MAX) return input(e, "%s: the memory source_authority its observations record", "source_authority");
    if (activate && activate->kind != HJ_BOOL) return input(e, "%s: a boolean", "activate");
    host_label(text, label);
    if (session_open(h, e)) return -1;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = h->d.api.elpis_runtime_authority_next_entity_id(h->rt, ELPIS_ECS_C_SOURCE, label, id);
    if (rc) return host_fail(e, rt_code(h, rc), "the next identity is unknown");
    rc = operator_event(h, 2, id, ev, e);
    if (!rc) {
        uint8_t *p = ev + ELPIS_ECS_C_PAYLOAD_AT;
        p[0] = ELPIS_ECS_C_SOURCE;
        put_u32(p + 4, (uint32_t)bound);
        put_u64(p + 8, version);
        memcpy(p + 16, label, 32);
        memcpy(p + 48, scope, 32);
        put_u32(p + 80, (uint32_t)authority);
        rc = send_event(h, ev, &generation, e);
    }
    const int active = !activate || activate->boolean;
    if (!rc && active && !(rc = operator_event(h, 3, id, ev, e))) {
        ev[ELPIS_ECS_C_PAYLOAD_AT] = ELPIS_ECS_C_FOUNDED, ev[ELPIS_ECS_C_PAYLOAD_AT + 1] = ELPIS_ECS_C_ACTIVE;
        rc = send_event(h, ev, &generation, e);
    }
    if (rc) return -1;
    hjw_khex(w, "id", id, 32);
    hjw_khex(w, "label", label, 32);
    hjw_ku64(w, "source_authority", authority);
    hjw_kstr(w, "lifecycle", active ? "ACTIVE" : "FOUNDED");
    hjw_ku64(w, "generation", generation);
    return 0;
}

static int op_source_lifecycle(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t authority;
    const int to = lifecycle_of(str_member(op, "to"));
    if (u64_member(op, "source_authority", UINT32_MAX, UINT64_MAX, &authority, e)) return -1;
    if (authority == UINT64_MAX) return input(e, "%s: the source's memory source_authority", "source_authority");
    if (to < (int)ELPIS_ECS_C_ACTIVE) return input(e, "%s: ACTIVE, DORMANT or TERMINATED", "to");
    if (session_open(h, e)) return -1;
    elpis_runtime_authority_entity s;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = find_source(h, (uint32_t)authority, &s, e);
    if (!rc && !(rc = operator_event(h, 3, s.id, ev, e))) {
        ev[ELPIS_ECS_C_PAYLOAD_AT] = s.lifecycle, ev[ELPIS_ECS_C_PAYLOAD_AT + 1] = (uint8_t)to;
        rc = send_event(h, ev, &generation, e);
    }
    if (rc) return -1;
    hjw_khex(w, "id", s.id, 32);
    hjw_kstr(w, "from", NAME(LIFECYCLES, s.lifecycle));
    hjw_kstr(w, "to", LIFECYCLES[to]);
    hjw_ku64(w, "generation", generation);
    return 0;
}

static int op_source_version(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t authority, expected, version, bound = 0;
    if (u64_member(op, "source_authority", UINT32_MAX, UINT64_MAX, &authority, e) ||
        u64_member(op, "expected", INT64_MAX, UINT64_MAX, &expected, e) ||
        u64_member(op, "version", INT64_MAX, UINT64_MAX, &version, e))
        return -1;
    if (authority == UINT64_MAX || expected == UINT64_MAX || version == UINT64_MAX)
        return input(e, "%s: source_authority, expected (the current version) and version are required", "source_version");
    if (session_open(h, e)) return -1;
    elpis_runtime_authority_entity s;
    uint8_t ev[512], scope[32];
    uint64_t generation = 0;
    int rc = find_source(h, (uint32_t)authority, &s, e);
    if (!rc) {
        memcpy(scope, s.scope, 32);
        if (u64_member(op, "per_turn", ELPIS_ECS_C_MAX_PER_TURN, s.per_turn_limit, &bound, e) || scope_in(op, 0, scope, e))
            rc = -1;
    }
    if (!rc && !(rc = operator_event(h, 4, s.id, ev, e))) {
        uint8_t *p = ev + ELPIS_ECS_C_PAYLOAD_AT;
        put_u64(p, expected); /* compare-and-set: the session refuses a stale expectation */
        put_u64(p + 8, version);
        put_u32(p + 16, (uint32_t)bound);
        memcpy(p + 24, scope, 32);
        rc = send_event(h, ev, &generation, e);
    }
    if (rc) return -1;
    hjw_khex(w, "id", s.id, 32);
    hjw_ku64(w, "version", version);
    hjw_ku64(w, "per_turn", bound);
    hjw_ku64(w, "generation", generation);
    return 0;
}

static int op_source_reclaim(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t authority;
    if (u64_member(op, "source_authority", UINT32_MAX, UINT64_MAX, &authority, e)) return -1;
    if (authority == UINT64_MAX) return input(e, "%s: the TERMINATED source's memory source_authority", "source_authority");
    if (session_open(h, e)) return -1;
    elpis_runtime_authority_entity s;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = find_source(h, (uint32_t)authority, &s, e);
    if (!rc && !(rc = operator_event(h, 14, s.id, ev, e))) rc = send_event(h, ev, &generation, e);
    if (rc) return -1;
    hjw_khex(w, "id", s.id, 32);
    hjw_ku64(w, "generation", generation);
    return 0;
}

static int op_text(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h, (void)op, (void)w;
    return host_fail(e, "ECS_CODEC_UNQUALIFIED", "no qualified ECS<->DSV4 codec map exists: there is no text "
                                                 "interface (QUERY and LEARN take admitted numeric stimuli)");
}

/* The durable operations of the retired persistent host: refused, nothing happens. */
static int op_retired(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h, (void)w;
    return host_fail(e, "OPERATION_RETIRED", "'%s' belonged to the retired persistent runtime: Elpis writes no state "
                                             "(docs/ELPIS_MISSION.md, Amendment 3); a session ends with the process "
                                             "or `close`", str_member(op, "op"));
}

#ifdef ELPIS_HOST_TESTING
/* ---- testing host only: the negative controls ------------------------------------------------------------------- */

/* Pause at an operation boundary: print one line, then wait (bounded) for one byte on standard input. A harness
 * inspects the live process meanwhile (its mappings, its descriptors) or kills it. */
static int op_testing_pause(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h;
    uint64_t timeout;
    if (u64_member(op, "timeout_ms", 600000, 60000, &timeout, e)) return -1;
    fputs("{\"op\":\"testing_pause\",\"paused\":true}\n", stdout);
    fflush(stdout);
    struct pollfd p = {0, POLLIN, 0};
    char byte;
    if (poll(&p, 1, (int)timeout) != 1 || read(0, &byte, 1) != 1)
        return host_fail(e, "TESTING_NOT_RESUMED", "no byte arrived on standard input");
    hjw_kbool(w, "resumed", 1);
    return 0;
}

/* A deliberate write attempt (the negative control of the containment): "create" a new file, "write" an existing one,
 * "extend" one past `offset`, "remove" or "rename" one, or "mkdir". Reports the errno (0: it happened). */
static int op_testing_write_probe(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h;
    const char *path = str_member(op, "path"), *kind = str_member(op, "kind");
    uint64_t offset;
    if (!path || path[0] != '/' || !kind || u64_member(op, "offset", INT64_MAX, 0, &offset, e))
        return input(e, "%s: an absolute path, a kind (create, write, extend, remove, rename, mkdir) and an offset",
                     "testing_write_probe");
    signal(SIGXFSZ, SIG_IGN); /* past the file size limit a write fails EFBIG instead of killing the process */
    errno = 0;
    int rc;
    if (!strcmp(kind, "create")) {
        const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        rc = fd < 0 ? errno : 0;
        if (fd >= 0) close(fd);
    } else if (!strcmp(kind, "write") || !strcmp(kind, "extend")) {
        const int fd = open(path, O_RDWR | O_CLOEXEC); /* read-write: "write" must read the byte it rewrites */
        rc = fd < 0 ? errno : 0;
        if (fd >= 0) {
            const off_t at = !strcmp(kind, "extend") ? (off_t)offset : 0;
            char byte = 0;
            if (at == 0 && pread(fd, &byte, 1, 0) != 1) rc = errno ? errno : EIO; /* never write a guessed byte */
            else rc = pwrite(fd, &byte, 1, at) == 1 ? 0 : errno;
            close(fd);
        }
    } else if (!strcmp(kind, "remove")) {
        rc = unlink(path) ? errno : 0;
    } else if (!strcmp(kind, "rename")) {
        char to[4200];
        snprintf(to, sizeof to, "%s.renamed", path);
        rc = rename(path, to) ? errno : 0;
    } else if (!strcmp(kind, "mkdir")) {
        rc = mkdir(path, 0700) ? errno : 0;
    } else {
        return input(e, "%s: kind is create, write, extend, remove, rename or mkdir", "testing_write_probe");
    }
    hjw_ki64(w, "errno", rc);
    return 0;
}

/* Process death at an operation boundary: no close, no destructor, nothing flushed but what was printed. */
static int op_testing_exit(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h, (void)op, (void)w, (void)e;
    _exit(99);
}
#endif

/* ---- dispatch, the command line and the script ----------------------------------------------------------------- */

static const struct {
    const char *name;
    op_fn fn;
} OPS[] = {
    {"status", op_status},
    {"open", op_open},
    {"close", op_close},
    {"live", op_live},
    {"counters", op_counters},
    {"query", op_query},
    {"learn", op_learn},
    {"evidence", op_evidence},
    {"lookup", op_lookup},
    {"slot", op_slot},
    {"authority", op_authority},
    {"source_found", op_source_found},
    {"source_lifecycle", op_source_lifecycle},
    {"source_version", op_source_version},
    {"source_reclaim", op_source_reclaim},
    {"text", op_text},
    {"publish", op_retired},
    {"reconcile", op_retired},
    {"candidates", op_retired},
    {"authority_reconcile", op_retired},
    {"init", op_retired},
    {"create", op_retired},
#ifdef ELPIS_HOST_TESTING
    {"testing_pause", op_testing_pause},
    {"testing_write_probe", op_testing_write_probe},
    {"testing_exit", op_testing_exit},
#endif
};

static char OUT[MAX_OUTPUT];

static void emit(const hjw *w) {
    fwrite(w->buf, 1, w->len, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static void emit_failure(const char *name, const char *code, const char *detail) {
    hjw w;
    hjw_init(&w, OUT, sizeof OUT);
    hjw_object(&w);
    hjw_key(&w, "op");
    if (name) hjw_str(&w, name);
    else hjw_null(&w);
    hjw_kbool(&w, "ok", 0);
    hjw_kstr(&w, "code", code);
    hjw_kstr(&w, "detail", detail);
    hjw_end(&w, '}');
    emit(&w);
}

/* Run one operation and print its line; 1 when it succeeded or was refused exactly as it expected. */
static int run(host *h, const hj_value *op) {
    const char *name = op && op->kind == HJ_OBJECT ? str_member(op, "op") : NULL;
    const hj_value *expect = op && op->kind == HJ_OBJECT ? hj_get(op, "expect") : NULL;
    hjw w;
    hjw_init(&w, OUT, sizeof OUT);
    hjw_object(&w);
    hjw_key(&w, "op");
    if (name) hjw_str(&w, name);
    else hjw_null(&w);
    const hjw mark = w;
    host_err e = {{0}, {0}, 0};
    op_fn fn = NULL;
    for (size_t i = 0; name && i < sizeof OPS / sizeof *OPS; ++i)
        if (!strcmp(OPS[i].name, name)) fn = OPS[i].fn;
    int ok;
    if (!name) ok = !host_fail(&e, "CLI_INPUT", "each operation is an object with an 'op' name");
    else if (!fn) ok = !host_fail(&e, "CLI_INPUT", "unknown operation '%s'", name);
    else ok = fn(h, op, &w, &e) == 0;
    if (!ok && !e.keep) w = mark;
    if (!hjw_ok(&w)) {
        w = mark;
        ok = !host_fail(&e, "HOST_OUTPUT", "the operation's output exceeds %zu bytes", MAX_OUTPUT);
        e.keep = 0;
    }
    hjw_kbool(&w, "ok", ok);
    if (!ok) {
        hjw_kstr(&w, "code", e.code);
        hjw_kstr(&w, "detail", e.detail);
    }
    int met = ok;
    if (expect && expect->kind == HJ_STRING) {
        met = !ok && !strcmp(e.code, expect->string);
        hjw_kstr(&w, "expected", expect->string);
        hjw_kbool(&w, "met", met);
    }
    hjw_end(&w, '}');
    emit(&w);
    return met;
}

static hj_arena ARENA;

static int arena_for(size_t len) {
    const size_t need = len * ARENA_PER_BYTE + ((size_t)1 << 18);
    if (ARENA.cap_bytes < need) {
        free(ARENA.bytes);
        ARENA.bytes = malloc(need);
        ARENA.cap_bytes = ARENA.bytes ? need : 0;
    }
    ARENA.used_bytes = ARENA.used_values = 0;
    ARENA.max_values = len + 64;
    return ARENA.bytes ? 0 : -1;
}

static int parse_json(const char *text, size_t len, hj_value **out, char *err, size_t errlen) {
    if (len > MAX_INPUT) return snprintf(err, errlen, "more than %zu bytes", MAX_INPUT), -1;
    if (arena_for(len)) return snprintf(err, errlen, "out of memory"), -1;
    return hj_parse(text, len, &ARENA, out, err, errlen);
}

/* A script: a JSON array of operations, or JSON lines (blank lines and lines starting with '#' skipped). Every line
 * is parsed before anything runs; a malformed script runs nothing. */
static char *read_script(const char *path, size_t *len, char *err, size_t errlen) {
    FILE *f = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!f) return snprintf(err, errlen, "%s: %s", path, strerror(errno)), NULL;
    /* A buffer that grows with the script (never the 8 MiB bound up front: the host's memory is kept small enough to
     * be locked whole when swap is active). */
    size_t cap = (size_t)1 << 16, n = 0;
    char *buf = malloc(cap + 1);
    while (buf) {
        n += fread(buf + n, 1, cap - n, f);
        if (n < cap || cap > MAX_INPUT) break;
        char *grown = realloc(buf, 2 * cap + 1);
        if (!grown) {
            free(buf);
            buf = NULL;
            break;
        }
        buf = grown, cap *= 2;
    }
    const int failed = ferror(f);
    if (f != stdin) fclose(f);
    if (!buf || failed || n > MAX_INPUT) {
        free(buf);
        return snprintf(err, errlen, "a script is at most %zu bytes and must be readable", MAX_INPUT), NULL;
    }
    buf[n] = 0;
    *len = n;
    return buf;
}

static int script_lines(char *text, size_t len, int run_them, host *h, int *all_ok, char *err, size_t errlen) {
    for (size_t at = 0, line = 1; at < len; ++line) {
        size_t end = at;
        while (end < len && text[end] != '\n') ++end;
        size_t b = at;
        while (b < end && (text[b] == ' ' || text[b] == '\t' || text[b] == '\r')) ++b;
        if (b < end && text[b] != '#') {
            hj_value *op;
            char perr[160];
            if (parse_json(text + at, end - at, &op, perr, sizeof perr))
                return snprintf(err, errlen, "script line %zu: %s", line, perr), -1;
            if (run_them) *all_ok &= run(h, op);
        }
        at = end + 1;
    }
    return 0;
}

/* ---- one-shot commands: the options each command maps onto its operation ---------------------------------------- */

typedef enum { A_JSON, A_STRING, A_U64, A_DOUBLE, A_FLAG, A_ID, A_SOURCES } arg_kind;
typedef struct {
    const char *option, *member;
    arg_kind kind;
} arg_spec;
typedef struct {
    const char *command, *op;
    arg_spec args[10];
} command_spec;

static const command_spec COMMANDS[] = {
    {"status", "status", {{0}}},
    {"live", "live", {{0}}},
    {"counters", "counters", {{"--reset", "reset", A_FLAG}}},
    {"query", "query", {{"--x", "x", A_JSON}}},
    {"learn",
     "learn",
     {{"--x", "x", A_JSON}, {"--y", "y", A_JSON}, {"--schedule", "schedule", A_JSON}, {"--rate", "rate", A_DOUBLE},
      {"--grant", "grant", A_STRING}}},
    {"evidence",
     "evidence",
     {{"--items", "items", A_JSON}, {"--grant", "grant", A_STRING}, {"--sources", "sources", A_SOURCES}}},
    {"lookup",
     "lookup",
     {{"--relation", "relation", A_ID}, {"--target-role", "target_role", A_ID}, {"--cues", "cues", A_JSON}}},
    {"slot", "slot", {{"--key", "key", A_STRING}}},
    {"authority", "authority", {{0}}},
    {"source-found",
     "source_found",
     {{"--label", "label", A_STRING}, {"--source-authority", "source_authority", A_U64}, {"--per-turn", "per_turn", A_U64},
      {"--version", "version", A_U64}, {"--scope", "scope", A_STRING}, {"--scope-label", "scope_label", A_STRING},
      {"--dormant", "dormant", A_FLAG}}},
    {"source-lifecycle",
     "source_lifecycle",
     {{"--source-authority", "source_authority", A_U64}, {"--to", "to", A_STRING}}},
    {"source-version",
     "source_version",
     {{"--source-authority", "source_authority", A_U64}, {"--expected", "expected", A_U64},
      {"--version", "version", A_U64}, {"--per-turn", "per_turn", A_U64}, {"--scope", "scope", A_STRING},
      {"--scope-label", "scope_label", A_STRING}}},
    {"source-reclaim", "source_reclaim", {{"--source-authority", "source_authority", A_U64}}},
    {"text", "text", {{"--input", "input", A_STRING}}},
};

/* The commands and options of the retired persistent host. */
static const char *const RETIRED_COMMANDS[] = {"init", "create", "publish", "reconcile", "candidates",
                                               "authority-reconcile", "--home", "--contained", "--publish"};

static void usage(void) {
    fprintf(stderr, "usage: elpis (--libraries DIR | --native-build BUILD) catalog\n"
                    "       elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] "
                    "session --script FILE|-\n"
                    "       elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] "
                    "<command> [options]   (docs/NATIVE_HOST.md)\n"
                    "session options: --dim N --width N --max-rows N --memory-dim N --banks N --slot-capacity N "
                    "--codebook-capacity N --seed N --capacity N --nonce-seed HEX\n");
}

/* Build the operation object of a one-shot command (members parsed one argument at a time, never concatenated). */
static int command_op(const command_spec *c, int argc, char **argv, hj_value *op, hj_value *items, const char **keys,
                      char *err, size_t errlen) {
    size_t n = 0;
    items[n] = (hj_value){.kind = HJ_STRING, .string = c->op, .length = strlen(c->op)};
    keys[n++] = "op";
    int dormant = 0;
    for (int i = 0; i < argc; ++i) {
        const arg_spec *s = NULL;
        for (int j = 0; j < 10 && c->args[j].option; ++j)
            if (!strcmp(argv[i], c->args[j].option)) s = &c->args[j];
        if (!s) return snprintf(err, errlen, "%s: unknown option %s", c->command, argv[i]), -1;
        for (size_t k = 1; k < n; ++k)
            if (!strcmp(keys[k], s->member)) return snprintf(err, errlen, "%s given twice", s->option), -1;
        if (s->kind == A_FLAG) {
            if (!strcmp(s->member, "dormant")) {
                dormant = 1;
                continue;
            }
            items[n] = (hj_value){.kind = HJ_BOOL, .boolean = 1};
            keys[n++] = s->member;
            continue;
        }
        if (++i >= argc) return snprintf(err, errlen, "%s needs a value", s->option), -1;
        const char *v = argv[i];
        hj_value *parsed;
        char *end;
        switch (s->kind) {
        case A_JSON: {
            if (hj_parse(v, strlen(v), &ARENA, &parsed, err, errlen)) return -1;
            items[n] = *parsed;
            break;
        }
        case A_STRING: items[n] = (hj_value){.kind = HJ_STRING, .string = v, .length = strlen(v)}; break;
        case A_U64: {
            errno = 0;
            const unsigned long long u = strtoull(v, &end, 10);
            if (!*v || *end || errno || v[0] == '-' || u > (1ull << 53))
                return snprintf(err, errlen, "%s: an integer in 0..2^53", s->option), -1;
            items[n] = (hj_value){.kind = HJ_NUMBER, .number = (double)u, .integral = 1, .integer = (int64_t)u};
            break;
        }
        case A_DOUBLE: {
            errno = 0;
            const double d = strtod(v, &end);
            if (!*v || *end || errno) return snprintf(err, errlen, "%s: a number", s->option), -1;
            items[n] = (hj_value){.kind = HJ_NUMBER, .number = d};
            break;
        }
        case A_ID: {
            if (!strncmp(v, "hyperedge:", 10) || !strncmp(v, "incidence:", 10)) {
                items[n] = (hj_value){.kind = HJ_STRING, .string = v, .length = strlen(v)};
                break;
            }
            errno = 0;
            const unsigned long long u = strtoull(v, &end, 0);
            if (!*v || *end || errno || v[0] == '-' || u > UINT32_MAX)
                return snprintf(err, errlen, "%s: an int or 'hyperedge:N' / 'incidence:N'", s->option), -1;
            items[n] = (hj_value){.kind = HJ_NUMBER, .number = (double)u, .integral = 1, .integer = (int64_t)u};
            break;
        }
        case A_SOURCES: {
            static hj_value list[256];
            size_t m = 0;
            for (const char *p = v; *p;) {
                errno = 0;
                const unsigned long long u = strtoull(p, &end, 10);
                if (end == p || errno || u > UINT32_MAX || m == 256 || (*end && *end != ','))
                    return snprintf(err, errlen, "%s: comma-separated 32-bit source ids", s->option), -1;
                list[m++] = (hj_value){.kind = HJ_NUMBER, .number = (double)u, .integral = 1, .integer = (int64_t)u};
                p = *end ? end + 1 : end;
            }
            items[n] = (hj_value){.kind = HJ_ARRAY, .items = list, .count = m};
            break;
        }
        default: return -1;
        }
        keys[n++] = s->member;
    }
    if (!strcmp(c->op, "source_found") && dormant) {
        items[n] = (hj_value){.kind = HJ_BOOL, .boolean = 0};
        keys[n++] = "activate";
    }
    *op = (hj_value){.kind = HJ_OBJECT, .items = items, .keys = keys, .count = n};
    return 0;
}

static int parse_u64_arg(const char *v, uint64_t *out) {
    char *end;
    errno = 0;
    const unsigned long long u = strtoull(v, &end, 10);
    if (!*v || *end || errno || v[0] == '-') return -1;
    *out = u;
    return 0;
}

/* The global options: where the libraries are, the catalog pin, and every session's configuration. */
typedef struct {
    host_source source;
    int pinned;
    uint8_t pin[32];
} host_options;

static int global_options(int argc, char **argv, int *at, host_options *o, host_install *d, char *err, size_t errlen) {
    static const char *const names[] = {"--dim", "--width", "--max-rows", "--memory-dim", "--banks",
                                        "--slot-capacity", "--codebook-capacity", "--seed"};
    host_lineage *l = &d->lineage;
    *l = (host_lineage){6, 36, 64, 64, 2, 64, 128, 0}; /* the Python LineageConfig's defaults */
    uint64_t *fields[] = {&l->dim, &l->width, &l->max_rows, &l->memory_dim, &l->banks, &l->slot_capacity,
                          &l->codebook_capacity, &l->seed};
    uint64_t capacity = 16;
    memset(&d->session, 0, sizeof d->session);
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i += 2) {
        for (size_t r = 0; r < sizeof RETIRED_COMMANDS / sizeof *RETIRED_COMMANDS; ++r)
            if (!strcmp(argv[i], RETIRED_COMMANDS[r])) return *at = i, 1;
        if (i + 1 >= argc) return snprintf(err, errlen, "%s needs a value", argv[i]), -1;
        const char *v = argv[i + 1];
        int known = 0;
        for (int j = 0; j < 8; ++j)
            if (!strcmp(argv[i], names[j])) {
                known = 1;
                if (parse_u64_arg(v, fields[j])) return snprintf(err, errlen, "%s: an unsigned integer", argv[i]), -1;
            }
        if (known) continue;
        if (!strcmp(argv[i], "--libraries") || !strcmp(argv[i], "--native-build")) {
            if (o->source.dir) return snprintf(err, errlen, "exactly one of --libraries or --native-build"), -1;
            o->source = (host_source){v, !strcmp(argv[i], "--native-build")};
        } else if (!strcmp(argv[i], "--catalog-sha256")) {
            if (host_unhex32(v, o->pin)) return snprintf(err, errlen, "--catalog-sha256: 64 lowercase hex characters"), -1;
            o->pinned = 1;
        } else if (!strcmp(argv[i], "--capacity")) {
            if (parse_u64_arg(v, &capacity) || capacity < ELPIS_ECS_C_MIN_CAPACITY || capacity > ELPIS_ECS_C_MAX_CAPACITY)
                return snprintf(err, errlen, "--capacity: %u..%u entities", ELPIS_ECS_C_MIN_CAPACITY,
                                ELPIS_ECS_C_MAX_CAPACITY), -1;
        } else if (!strcmp(argv[i], "--nonce-seed")) {
            if (host_unhex32(v, d->session.nonce)) return snprintf(err, errlen, "--nonce-seed: 64 lowercase hex"), -1;
        } else {
            return snprintf(err, errlen, "unknown option %s", argv[i]), -1;
        }
    }
    if (!o->source.dir) return snprintf(err, errlen, "the libraries: --libraries DIR or --native-build BUILD"), -1;
    d->source = o->source;
    d->session.abi_version = ELPIS_RUNTIME_ABI_V8;
    d->session.capacity = (uint32_t)capacity;
    host_label("operator", d->session.operator_label);
    host_label("runtime", d->session.runtime_label);
    *at = i;
    return 0;
}

static void emit_catalog(const char *catalog, const uint8_t digest[32]) {
    hjw w;
    hjw_init(&w, OUT, sizeof OUT);
    hjw_object(&w);
    hjw_kstr(&w, "op", "catalog");
    hjw_key(&w, "catalog_sha256");
    hjw_hex(&w, digest, 32);
    hjw_kbool(&w, "ok", 1);
    hjw_end(&w, '}');
    /* {"op":"catalog","catalog_sha256":...,"ok":true} with the catalog object spliced in before "ok". */
    const char *ok = strstr(OUT, ",\"ok\":");
    fwrite(OUT, 1, (size_t)(ok - OUT), stdout);
    fputs(",\"catalog\":", stdout);
    fputs(catalog, stdout);
    fputs(ok, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    static host h;
    host_options o;
    memset(&o, 0, sizeof o);
#ifdef ELPIS_HOST_TESTING
    h.d.testing = 1;
#endif
    char err[256] = {0};
    int at = 0;
    const int g = global_options(argc, argv, &at, &o, &h.d, err, sizeof err);
    if (g < 0) return usage(), emit_failure(NULL, "CLI_INPUT", err), 2;
    if (g > 0 || at >= argc) {
        if (at >= argc) return usage(), 2;
    }
    const char *command = argv[at];
    for (size_t r = 0; r < sizeof RETIRED_COMMANDS / sizeof *RETIRED_COMMANDS; ++r)
        if (!strcmp(command, RETIRED_COMMANDS[r]))
            return emit_failure(command, "OPERATION_RETIRED", "the persistent host is retired: there is no home, no "
                                                              "deployment state, no publication and no recovery "
                                                              "(docs/NATIVE_HOST.md)"), 2;
    for (int i = at + 1; i < argc; ++i)
        if (!strcmp(argv[i], "--publish"))
            return emit_failure(command, "OPERATION_RETIRED", "--publish: no persistent writer exists"), 2;
    const int catalog = !strcmp(command, "catalog");
    if (catalog && at + 1 != argc) return usage(), 2;
    if (!catalog && !o.pinned)
        return emit_failure(command, "CLI_INPUT", "--catalog-sha256: the pin of the catalog (`catalog` prints it)"), 2;
    /* Validate the command line (and read the script) before anything is admitted. */
    const command_spec *spec = NULL;
    for (size_t i = 0; i < sizeof COMMANDS / sizeof *COMMANDS; ++i)
        if (!strcmp(COMMANDS[i].command, command)) spec = &COMMANDS[i];
    static hj_value items[16], one_shot;
    static const char *keys[16];
    char *script = NULL;
    size_t script_len = 0;
    hj_value *script_doc = NULL;
    if (!strcmp(command, "session")) {
        if (argc != at + 3 || strcmp(argv[at + 1], "--script")) return usage(), 2;
        if (!(script = read_script(argv[at + 2], &script_len, err, sizeof err))) return emit_failure("session", "CLI_INPUT", err), 2;
        size_t b = 0;
        while (b < script_len && (script[b] == ' ' || script[b] == '\n' || script[b] == '\t' || script[b] == '\r')) ++b;
        const int malformed = b < script_len && script[b] == '['
                                  ? parse_json(script, script_len, &script_doc, err, sizeof err)
                                  : script_lines(script, script_len, 0, NULL, NULL, err, sizeof err);
        if (malformed || (script_doc && script_doc->kind != HJ_ARRAY)) {
            if (!malformed) snprintf(err, sizeof err, "a script is a JSON array or JSON lines of operations");
            free(script);
            return emit_failure("session", "CLI_INPUT", err), 2;
        }
    } else if (spec) {
        size_t args = 0; /* the arena holds exactly what the arguments can parse into */
        for (int i = at + 1; i < argc; ++i) args += strlen(argv[i]) + 1;
        if (args > MAX_INPUT) return emit_failure(command, "CLI_INPUT", "the arguments exceed 8 MiB"), 2;
        if (arena_for(args)) return emit_failure(command, "HOST_RESOURCES", "out of memory"), 2;
        if (command_op(spec, argc - at - 1, argv + at + 1, &one_shot, items, keys, err, sizeof err))
            return emit_failure(spec->op, "CLI_INPUT", err), 2;
    } else if (!catalog) {
        return usage(), 2;
    }
    /* Containment of the process, then admission, then the filesystem: nothing runs uncontained. */
    host_err e = {{0}, {0}, 0};
    const char *name = spec ? spec->op : command;
    static char catalog_text[1 << 14];
    size_t catalog_len = 0;
    if (host_contain_process(&h.contain, &e) ||
        host_measure(&h.d, catalog ? catalog_text : NULL, sizeof catalog_text, &catalog_len, &e)) {
        free(script);
        return emit_failure(name, e.code, e.detail), 4;
    }
    if (!catalog && memcmp(h.d.catalog_sha256, o.pin, 32)) {
        free(script);
        return emit_failure(name, "INSTALL_PIN", "the measured catalog (this host binary and its libraries) does not "
                                                 "match --catalog-sha256"), 4;
    }
    if (host_load_libraries(&h.d, &e) || host_lineage_check(&h.d, &e) || host_contain_files(&h.contain, &e)) {
        free(script);
        return emit_failure(name, e.code, e.detail), 4;
    }
    if (catalog) return emit_catalog(catalog_text, h.d.catalog_sha256), 0;
    h.d.api.elpis_runtime_fuel_ceiling(&h.ceiling);
    int all_ok = 1;
    if (script_doc) {
        for (size_t i = 0; i < script_doc->count; ++i) all_ok &= run(&h, &script_doc->items[i]);
    } else if (script) {
        script_lines(script, script_len, 1, &h, &all_ok, err, sizeof err);
    } else {
        all_ok = run(&h, &one_shot);
    }
    session_close(&h);
    if (h.rt) h.d.api.elpis_runtime_destroy(&h.rt);
    free(h.envelope);
    free(h.s3);
    free(script);
    free(ARENA.bytes);
    return all_ok ? 0 : 3;
}
