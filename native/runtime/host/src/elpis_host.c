/* elpis: the native Elpis operator host (docs/NATIVE_HOST.md). One process over one deployment; no interpreter.
 *
 *   elpis --home DIR init --native-build BUILD [--capacity N] [--authority-dir DIR] [--source-commit SHA]
 *   elpis --home DIR <command> [options]          one operation (learn/evidence --publish: two)
 *   elpis --home DIR session --script FILE|-      many operations in one process (volatile work lives here)
 *
 * Every operation prints one JSON object on one line: {"op": ..., <its members>, "ok": true} or
 * {"op": ..., "ok": false, "code": ..., "detail": ...}; an operation that names "expect": CODE is met exactly when
 * it is refused with CODE. Exit status: 0 every operation succeeded or was refused as expected; 2 the command line,
 * an argument or a script is malformed (nothing ran); 3 an operation was refused (or an expectation unmet); 4 the
 * deployment could not be opened (nothing ran).
 *
 * The host computes no ECS mathematics and touches no ECS byte: it validates operator input, marshals it into the
 * admitted libraries' C ABIs and reports what they answer. RuntimeCore composes every turn; the ECS_C authority
 * decides every durable transition; structure identifies every hyperedge. There is no text interface: no qualified
 * ECS<->DSV4 codec map exists (ECS_CODEC_UNQUALIFIED). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "elpis/sha256.h"
#include "host_contain.h"
#include "host_deploy.h"
#include "host_json.h"

#define MAX_INPUT ((size_t)8 << 20)  /* one operation's JSON, or a script */
#define ARENA_PER_BYTE 32            /* parser arena bytes per input byte */
#define MAX_OUTPUT ((size_t)8 << 20) /* one operation's output line */
#define MAX_QUERY_ROWS 4096u
#define MAX_EXPERIENCES 64u
#define MAX_EXPERIENCE_ROWS 256u
#define MAX_STEPS (1u << 20)
#define MAX_TURN_OBSERVATIONS 64u
#define HYPEREDGE_NAMESPACE 0x20000000u
#define INCIDENCE_NAMESPACE 0x30000000u

/* ---- the process ---------------------------------------------------------------------------------------------- */

typedef struct {
    host_deployment d;
    elpis_runtime *rt;  /* the session's runtime (created at the first operation that needs it) */
    int opened;         /* rt is open */
    int recovered;      /* the lineage's recovery disposition was read */
    elpis_runtime_recovery recovery;
    elpis_ecs_composite *x; /* the restored composite (RESUMABLE) */
    uint64_t owner;
    elpis_runtime_substrate sub;
    uint8_t *envelope; /* d.envelope_bytes */
    double *s3;
    elpis_runtime_budget ceiling;
} host;

static const char *rt_code(const host *h, int rc) {
    const char *name = h->d.api.elpis_runtime_code_name(rc);
    return name ? name : "RUNTIME_UNKNOWN_CODE";
}
static const char *ecs_c_code(const host *h, int rc) {
    const char *name = h->d.api.elpis_ecs_c_code_name(rc);
    return name ? name : "ECS_C_UNKNOWN_CODE";
}
static const char *composite_code(const host *h, int rc) {
    const char *name = h->d.api.elpis_ecs_composite_code_name(rc);
    return name ? name : "ECS_COMPOSITE_UNKNOWN";
}

static const char *const DISPOSITIONS[] = {"NONE", "NOTHING_TO_RECOVER", "RESUMABLE", "CANDIDATE_UNRESOLVED",
                                           "CHECKPOINT_MISSING"};
static const char *const DURABLE[] = {"NONE",       "PENDING",   "ACKNOWLEDGED", "FAILED",           "UNCERTAIN",
                                      "SUPERSEDED", "ABANDONED", "UNKNOWN",      "IDENTITY_PUBLISHED"};
static const char *const POLARITIES[] = {"NONE", "ASSERT", "NEGATE", "UNCERTAIN"};
static const char *const KINDS[] = {"NONE", "OPERATOR", "RUNTIME", "SOURCE"};
static const char *const LIFECYCLES[] = {"NONE", "FOUNDED", "ACTIVE", "DORMANT", "TERMINATED"};
static const char *const MODES[] = {"NORMAL", "RESTRICTED", "ROLLBACK", "WITNESS_MISSING"};
static const char *const SLOT_CLASSES[] = {"EMPTY", "TORN", "UNAUTHENTICATED", "INVALID", "VALID"};
#define NAME(table, i) ((size_t)(i) < sizeof(table) / sizeof(*(table)) ? (table)[i] : "UNKNOWN")

static const char *disposition_name(uint32_t d) {
    switch (d) {
    case 1: return "CLEAN";
    case 2: return "TRAILING_INCOMPLETE";
    case 3: return "INTERRUPTED_COMPLETED";
    case 4: return "RESTRICTED_UNAUTHENTICATED";
    case 5: return "PREDECESSOR_UNAVAILABLE";
    case 10: return "ROLLBACK";
    case 11: return "CORRUPT_ACKNOWLEDGED";
    case 12: return "UNAUTHENTICATED_ACKNOWLEDGED";
    case 13: return "WITNESS_MISSING";
    case 20: return "CONFLICT";
    case 21: return "WITNESS_INCONSISTENT";
    case 22: return "REPLAY_MISMATCH";
    case 23: return "UNRECOVERABLE";
    default: return d ? "UNKNOWN" : "NONE";
    }
}

/* ---- the session: the runtime over the ECS_C authority, the attached store, the restored composite ------------ */

static int runtime_open(host *h, host_err *e) {
    const host_api *a = &h->d.api;
    if (h->opened) return 0;
    if (!h->rt) {
        const int rc = a->elpis_runtime_create_ecs_c((const uint8_t *)h->d.state_dir, strlen(h->d.state_dir),
                                                     (const uint8_t *)h->d.authority_dir, strlen(h->d.authority_dir),
                                                     &h->rt);
        if (rc) return host_fail(e, rt_code(h, rc), "the runtime could not be created over the ECS_C authority");
    }
    elpis_continuity_snapshot snap;
    const int rc = a->elpis_runtime_open(h->rt, &snap);
    if (rc == ELPIS_RUNTIME_AUTHORITY)
        return host_fail(e, rt_code(h, rc), "the ECS_C authority is not in its normal mode: `authority` reports why; "
                                            "the operator reconciles it with `authority-reconcile`");
    if (rc) return host_fail(e, rt_code(h, rc), "the runtime could not open its authority");
    h->opened = 1;
    if (h->d.has_lineage) {
        const int at = a->elpis_runtime_checkpoint_attach(h->rt, (const uint8_t *)h->d.checkpoint_dir,
                                                          strlen(h->d.checkpoint_dir));
        if (at) return host_fail(e, rt_code(h, at), "the checkpoint store could not be attached");
    }
    return 0;
}

static void session_close(host *h) {
    const host_api *a = &h->d.api;
    if (h->rt) a->elpis_runtime_destroy(&h->rt); /* aborts an open turn natively; never touches the state */
    if (h->x) a->elpis_ecs_composite_destroy(h->x);
    free(h->envelope);
    free(h->s3);
    h->rt = NULL, h->x = NULL, h->envelope = NULL, h->s3 = NULL, h->opened = h->recovered = 0;
}

/* Open the lineage: the runtime, the recovery disposition and, when RESUMABLE, exactly the authorized composite. */
static int lineage(host *h, host_err *e) {
    const host_api *a = &h->d.api;
    if (!h->d.has_lineage) return host_fail(e, "LINEAGE_MISSING", "no lineage: run create");
    if (runtime_open(h, e)) return -1;
    if (h->recovered) return 0;
    if (!h->envelope && !(h->envelope = malloc(h->d.envelope_bytes)))
        return host_fail(e, "HOST_RESOURCES", "out of memory");
    const int rc = a->elpis_runtime_checkpoint_recover(h->rt, &h->recovery, h->envelope, h->d.envelope_bytes);
    if (rc) return host_fail(e, rt_code(h, rc), "the recovery disposition could not be read");
    h->recovered = 1;
    if (h->recovery.disposition != ELPIS_RUNTIME_RECOVERY_RESUMABLE) return 0;
    if (h->recovery.envelope_bytes != h->d.envelope_bytes)
        return host_fail(e, "LINEAGE_MISMATCH", "the authorized envelope is not the lineage's size");
    const elpis_ecs_composite_config cfg = host_composite_config(&h->d, &h->d.lineage);
    const int r = a->elpis_ecs_composite_restore(&cfg, h->envelope, h->recovery.envelope_bytes, &h->x);
    if (r) return host_fail(e, "LINEAGE_MISMATCH", "the authorized envelope does not restore: %s", composite_code(h, r));
    uint8_t digest[32];
    if (a->elpis_ecs_composite_state_digest(h->x, digest) || memcmp(digest, h->recovery.authorized, 32))
        return host_fail(e, "LINEAGE_MISMATCH", "the restored composite is not the authority's identity");
    h->sub = (elpis_runtime_substrate){ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, h->x, 0, ++h->owner, h->d.lineage.dim,
                                       host_substrate_api()};
    const size_t f = a->elpis_runtime_features(h->d.lineage.dim);
    if (!h->s3 && !(h->s3 = malloc(sizeof(double) * (f ? f : 1)))) return host_fail(e, "HOST_RESOURCES", "out of memory");
    return 0;
}

static int require_state(host *h, host_err *e) {
    if (lineage(h, e)) return -1;
    if (!h->x)
        return host_fail(e, "LINEAGE_NOT_RESUMABLE", "the lineage is %s: see `status`; an unresolved candidate needs "
                                                     "`reconcile`", NAME(DISPOSITIONS, h->recovery.disposition));
    return 0;
}

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

static void snapshot_out(hjw *w, const char *key, const elpis_continuity_snapshot *s) {
    hjw_key(w, key);
    if (!s) {
        hjw_null(w);
        return;
    }
    hjw_object(w);
    hjw_kstr(w, "state_format", s->state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE ? "COMPOSITE" : "K1");
    hjw_key(w, "identity");
    if (s->anchored) hjw_hex(w, s->k1_state_digest, 32);
    else hjw_null(w);
    hjw_ku64(w, "generation", s->generation);
    hjw_end(w, '}');
}

static void durable_out(host *h, hjw *w) {
    elpis_continuity_snapshot s;
    snapshot_out(w, "durable", h->opened && !h->d.api.elpis_runtime_snapshot(h->rt, &s) ? &s : NULL);
}

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

static void ticket_out(host *h, hjw *w, const char *key, const elpis_runtime_durable_ticket *t) {
    hjw_key(w, key);
    hjw_object(w);
    hjw_ku64(w, "sequence", t->sequence);
    hjw_kstr(w, "status", NAME(DURABLE, t->status));
    hjw_key(w, "kind");
    if (t->kind == ELPIS_RUNTIME_DURABLE_KIND_RECOVERABLE) hjw_str(w, "RECOVERABLE");
    else if (t->kind == ELPIS_RUNTIME_DURABLE_KIND_IDENTITY) hjw_str(w, "IDENTITY");
    else hjw_null(w);
    hjw_kbool(w, "recoverable", (int)t->recoverable);
    hjw_kbool(w, "acknowledged", t->status == ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED);
    hex_or_null(w, "identity", t->identity);
    hjw_ku64(w, "generation", t->generation);
    hjw_key(w, "code");
    if (t->code) hjw_str(w, rt_code(h, t->code));
    else hjw_null(w);
    hjw_end(w, '}');
}

static void recovery_out(hjw *w, const elpis_runtime_recovery *r) {
    hjw_key(w, "recovery");
    hjw_object(w);
    hjw_kstr(w, "disposition", NAME(DISPOSITIONS, r->disposition));
    hex_or_null(w, "authorized", r->authorized);
    hex_or_null(w, "candidate", r->candidate);
    hjw_ku64(w, "envelope_bytes", r->envelope_bytes);
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
                      "an explicit grant and learning rate; volatile until published"},
    {"typed_evidence", "WORKING, mechanically verified: validated HACF hyperedge + assertion recorded by an "
                       "observe-only turn, admitted only from ACTIVE sources of the ECS_C authority"},
    {"lookup", "WORKING: exact relation LOOKUP by net evidence; a conflict between sources is reported, never resolved"},
    {"recoverable_publication", "WORKING: checkpoint slot read back verified, then one authenticated ECS_C record "
                                "and its rollback witness (ACKNOWLEDGED)"},
    {"restart_recovery", "WORKING: exactly the last acknowledged state, replay-verified; an unresolved candidate "
                         "needs reconcile, a non-normal authority authority-reconcile"},
    {"operator_authority", "WORKING: source founding, lifecycle, versions and reclamation as typed ECS_C events"},
    {"text", "UNAVAILABLE: ECS_CODEC_UNQUALIFIED (no qualified ECS<->DSV4 codec map)"},
    {"memory_learn", "UNQUALIFIED: never enabled (no memory capability is ever set)"},
    {"memory_slow_consolidation", "NOT PROVIDED"},
    {"contextual_dynamics", "NOT IMPLEMENTED: no qualified law (the HCA substrate is gated research)"},
    {"power_loss_durability", "NOT CLAIMED: crash consistency is established for process death only"},
};
static const char *const NONCLAIMS[] = {"no language understanding", "no general intelligence",
                                        "no autonomous evolution", "no hierarchical superiority",
                                        "an observation is evidence storage, never truth or authority",
                                        "pins authenticate bytes, not authors",
                                        "the deployment key authenticates records, not the operator"};

static void lineage_config_out(hjw *w, const host_deployment *d) {
    const host_lineage *l = &d->lineage;
    hjw_key(w, "lineage");
    if (!d->has_lineage) {
        hjw_null(w);
        return;
    }
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
    hjw_end(w, '}');
}

static void authority_summary_out(host *h, hjw *w) {
    elpis_runtime_authority s;
    hjw_key(w, "authority");
    if (!h->rt || h->d.api.elpis_runtime_authority_status(h->rt, &s)) {
        hjw_null(w);
        return;
    }
    hjw_object(w);
    hjw_kbool(w, "held", h->opened);
    hjw_kstr(w, "disposition", disposition_name(s.disposition));
    hjw_kstr(w, "mode", NAME(MODES, s.mode));
    hjw_key(w, "ecs_c_code");
    if (s.ecs_c_code) hjw_str(w, ecs_c_code(h, s.ecs_c_code));
    else hjw_null(w);
    hjw_ku64(w, "generation", s.generation);
    hjw_kbool(w, "replay_verified", s.replay_verified);
    hjw_kbool(w, "witness_completed", s.witness_completed);
    hjw_ku64(w, "receipt_count", s.receipt_count);
    hjw_ku64(w, "evidence_count", s.evidence_count);
    hjw_ku64(w, "entity_count", s.entity_count);
    hjw_ku64(w, "capacity", s.capacity);
    hjw_khex(w, "lineage", s.lineage, 32);
    hjw_khex(w, "record_digest", s.record_digest, 32);
    hjw_khex(w, "state_root", s.state_root, 32);
    hjw_khex(w, "receipt_head", s.receipt_head, 32);
    hjw_khex(w, "evidence_head", s.evidence_head, 32);
    hjw_end(w, '}');
}

static int op_status(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    const host_deployment *d = &h->d;
    hjw_key(w, "deployment");
    hjw_object(w);
    hjw_kstr(w, "home", d->home);
    hjw_kstr(w, "schema", "elpis.native-host.v1");
    hjw_kbool(w, "testing_build", d->testing);
    hjw_key(w, "source_commit");
    if (d->source_commit[0]) hjw_str(w, d->source_commit);
    else hjw_null(w);
    hjw_khex(w, "catalog_sha256", d->catalog_sha256, 32);
    hjw_khex(w, "host_sha256", d->host_sha256, 32);
    hjw_kstr(w, "state_dir", d->state_dir);
    hjw_kstr(w, "authority_dir", d->authority_dir);
    hjw_end(w, '}');
    hjw_key(w, "abi");
    hjw_object(w);
    hjw_ku64(w, "runtime", ELPIS_RUNTIME_ABI_V6);
    hjw_ku64(w, "composite", ELPIS_ECS_COMPOSITE_ABI_V1);
    hjw_ku64(w, "ecs_c", ELPIS_ECS_C_ABI_VERSION);
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
    budget_out(w, &h->ceiling);
    hjw_key(w, "capabilities");
    hjw_object(w);
    for (size_t i = 0; i < sizeof CAPABILITIES / sizeof *CAPABILITIES; ++i) hjw_kstr(w, CAPABILITIES[i][0], CAPABILITIES[i][1]);
    hjw_end(w, '}');
    hjw_key(w, "nonclaims");
    hjw_array(w);
    for (size_t i = 0; i < sizeof NONCLAIMS / sizeof *NONCLAIMS; ++i) hjw_str(w, NONCLAIMS[i]);
    hjw_end(w, ']');
    lineage_config_out(w, d);
    /* The session's state; a session that cannot open is reported, not an error of status. */
    host_err why = {{0}, {0}, 0};
    const int session = d->has_lineage ? lineage(h, &why) : runtime_open(h, &why);
    authority_summary_out(h, w);
    hjw_key(w, "session_refusal");
    if (session) {
        hjw_object(w);
        hjw_kstr(w, "code", why.code);
        hjw_kstr(w, "detail", why.detail);
        hjw_end(w, '}');
    } else {
        hjw_null(w);
    }
    if (!d->has_lineage || session) return 0;
    const host_api *a = &d->api;
    recovery_out(w, &h->recovery);
    durable_out(h, w);
    if (h->x) {
        uint8_t identity[32];
        uint64_t generation;
        const int rc = a->elpis_runtime_live_identity(h->rt, &h->sub, identity, &generation);
        if (rc) return host_fail(e, rt_code(h, rc), "the live identity could not be read");
        hjw_key(w, "live");
        hjw_object(w);
        hjw_khex(w, "identity", identity, 32);
        hjw_ku64(w, "generation", generation);
        hjw_end(w, '}');
        elpis_ecs_composite_stats st;
        if (!a->elpis_ecs_composite_inspect(h->x, &st)) {
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
    elpis_runtime_durability du;
    if (!a->elpis_runtime_durable_status(h->rt, &du)) {
        hjw_key(w, "durability");
        hjw_object(w);
        hjw_ku64(w, "live_generation", du.live_generation);
        hjw_kbool(w, "pending", (int)du.pending);
        hjw_key(w, "fault");
        if (du.fault) hjw_str(w, rt_code(h, du.fault));
        else hjw_null(w);
        ticket_out(h, w, "last_acknowledged", &du.last_acknowledged);
        hjw_ku64(w, "checkpoint_capacity", du.checkpoint_capacity);
        hjw_end(w, '}');
    }
    const int fault = a->elpis_runtime_fault(h->rt);
    hjw_key(w, "fault");
    if (fault) hjw_str(w, rt_code(h, fault));
    else hjw_null(w);
    return 0;
}

static int op_open(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    if (lineage(h, e)) return -1;
    recovery_out(w, &h->recovery);
    durable_out(h, w);
    authority_summary_out(h, w);
    return 0;
}

static int op_close(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op, (void)e;
    int volatile_work = 0;
    elpis_runtime_durability du;
    if (h->opened && h->x && !h->d.api.elpis_runtime_durable_status(h->rt, &du))
        volatile_work = !du.durable_known || du.durable.generation == 0 ||
                        du.live_generation != du.last_acknowledged.generation;
    session_close(h);
    hjw_kbool(w, "closed", 1);
    hjw_kbool(w, "volatile_work_discarded", volatile_work);
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
    durable_out(h, w);
    return 0;
}

static int op_counters(host *h, const hj_value *op, hjw *w, host_err *e) {
    const hj_value *reset = hj_get(op, "reset");
    if (lineage(h, e)) return -1;
    elpis_runtime_counters c;
    const int rc = h->d.api.elpis_runtime_read_counters(h->rt, &c, reset && reset->kind == HJ_BOOL && reset->boolean);
    if (rc) return host_fail(e, rt_code(h, rc), "the counters could not be read");
    hjw_key(w, "counters");
    hjw_object(w);
#define C(f) hjw_ku64(w, #f, c.f);
    C(k1_state_digests) C(k1_reserves) C(k1_txn_begins) C(k1_run_schedules) C(k1_commits) C(k1_aborts)
    C(publications) C(k1_queries) C(k1_shapes) C(k1_lease_claims) C(k1_identities) C(k1_captures)
    C(durable_requests) C(durable_superseded) C(durable_acknowledged) C(durable_failed) C(k1_observes)
#undef C
    hjw_end(w, '}');
    return 0;
}

static int op_candidates(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    if (lineage(h, e)) return -1;
    elpis_runtime_recovery r;
    const int rc = h->d.api.elpis_runtime_checkpoint_recover(h->rt, &r, NULL, 0);
    if (rc) return host_fail(e, rt_code(h, rc), "the recovery disposition could not be read");
    recovery_out(w, &r);
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

/* ---- durability ---------------------------------------------------------------------------------------------- */

static int op_publish(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (require_state(h, e)) return -1;
    const hj_value *wait = hj_get(op, "wait");
    uint64_t timeout;
    if (u64_member(op, "timeout_ms", INT64_MAX, UINT64_MAX, &timeout, e)) return -1;
    elpis_runtime_durable_ticket t;
    int rc = a->elpis_runtime_recoverable_request(h->rt, &h->sub, &t);
    if (rc) return host_fail(e, rt_code(h, rc), "the recoverable publication was refused before capture");
    if (!(wait && wait->kind == HJ_BOOL && !wait->boolean)) {
        rc = a->elpis_runtime_durable_wait(h->rt, t.sequence, timeout, &t);
        if (rc && rc != ELPIS_RUNTIME_DURABLE_PENDING)
            return host_fail(e, rt_code(h, rc), "the publication's outcome could not be read");
    }
    /* The acknowledgment law, re-checked: recoverable exactly when ACKNOWLEDGED, and only a recoverable request's. */
    const int acknowledged = t.status == ELPIS_RUNTIME_DURABLE_ACKNOWLEDGED;
    if ((t.recoverable != 0) != acknowledged || (acknowledged && t.kind != ELPIS_RUNTIME_DURABLE_KIND_RECOVERABLE))
        return host_fail(e, "RUNTIME_INVALID", "the durable ticket violates the acknowledgment law");
    ticket_out(h, w, "ticket", &t);
    durable_out(h, w);
    if (acknowledged || t.status == ELPIS_RUNTIME_DURABLE_REQUESTED) return 0;
    e->keep = 1; /* the ticket says what happened */
    return host_fail(e, t.status == ELPIS_RUNTIME_DURABLE_UNCERTAIN ? "DURABLE_UNCERTAIN"
                        : t.status == ELPIS_RUNTIME_DURABLE_SUPERSEDED ? "DURABLE_SUPERSEDED"
                        : t.status == ELPIS_RUNTIME_DURABLE_ABANDONED  ? "DURABLE_ABANDONED"
                                                                       : "DURABLE_FAILED",
                     "the publication was not acknowledged%s%s", t.code ? ": " : "", t.code ? rt_code(h, t.code) : "");
}

static int op_reconcile(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (lineage(h, e)) return -1;
    if (h->x) return host_fail(e, "LINEAGE_BOUND", "nothing to reconcile: the lineage is resumable");
    const char *action = str_member(op, "action");
    uint8_t candidate[32];
    if (hex_member(op, "candidate", candidate, e)) return -1;
    int rc;
    if (action && !strcmp(action, "discard")) rc = a->elpis_runtime_checkpoint_discard(h->rt, candidate);
    else if (action && !strcmp(action, "adopt")) rc = a->elpis_runtime_checkpoint_adopt(h->rt, candidate, &(elpis_continuity_snapshot){0});
    else return input(e, "%s: discard or adopt", "action");
    if (rc) return host_fail(e, rt_code(h, rc), "the reconciliation of exactly this candidate was refused");
    h->recovered = 0;
    if (lineage(h, e)) return -1;
    hjw_kstr(w, "action", action);
    recovery_out(w, &h->recovery);
    durable_out(h, w);
    return 0;
}

/* ---- the ECS_C authority: inspection, operator events, reconciliation ---------------------------------------- */

/* The authority an operator event reaches: through the runtime that holds it, or opened directly (no runtime open). */
typedef struct {
    host *h;
    elpis_ecs_c_authority *direct;
    elpis_ecs_c_report report;
} held;

static int held_begin(host *h, held *t, host_err *e) {
    memset(t, 0, sizeof *t);
    t->h = h;
    if (h->opened) return 0;
    if (h->rt) h->d.api.elpis_runtime_destroy(&h->rt); /* a runtime whose open was refused holds nothing */
    const int rc = h->d.api.elpis_ecs_c_open(h->d.state_dir, strlen(h->d.state_dir), h->d.authority_dir,
                                             strlen(h->d.authority_dir), &t->direct, &t->report);
    if (rc) return host_fail(e, ecs_c_code(h, rc), "the ECS_C authority refused to open (disposition %s)",
                             disposition_name(t->report.disposition));
    return 0;
}

static void held_end(held *t) {
    if (t->direct) t->h->d.api.elpis_ecs_c_close(t->direct);
    t->direct = NULL;
}

static int held_count(held *t, uint32_t *n, host_err *e) {
    const host_api *a = &t->h->d.api;
    if (t->direct) {
        elpis_ecs_c_status s;
        const int rc = a->elpis_ecs_c_read_status(t->direct, &s);
        if (rc) return host_fail(e, ecs_c_code(t->h, rc), "the authority's status could not be read");
        return *n = s.entity_count, 0;
    }
    elpis_runtime_authority s;
    const int rc = a->elpis_runtime_authority_status(t->h->rt, &s);
    if (rc) return host_fail(e, rt_code(t->h, rc), "the authority's status could not be read");
    return *n = s.entity_count, 0;
}

static int held_entity(held *t, uint32_t i, elpis_ecs_c_entity *out, host_err *e) {
    const host_api *a = &t->h->d.api;
    _Static_assert(sizeof(elpis_runtime_authority_entity) == sizeof(elpis_ecs_c_entity), "entity layout");
    const int rc = t->direct ? a->elpis_ecs_c_entity_at(t->direct, i, out)
                             : a->elpis_runtime_authority_entity_at(t->h->rt, i, (elpis_runtime_authority_entity *)out);
    if (rc) return host_fail(e, t->direct ? ecs_c_code(t->h, rc) : rt_code(t->h, rc), "entity %u could not be read", i);
    return 0;
}

static int held_next_id(held *t, uint8_t kind, const uint8_t label[32], uint8_t out[32], host_err *e) {
    const host_api *a = &t->h->d.api;
    const int rc = t->direct ? a->elpis_ecs_c_next_entity_id(t->direct, kind, label, out)
                             : a->elpis_runtime_authority_next_entity_id(t->h->rt, kind, label, out);
    if (rc) return host_fail(e, t->direct ? ecs_c_code(t->h, rc) : rt_code(t->h, rc), "the next identity is unknown");
    return 0;
}

static int held_send(held *t, const uint8_t event[512], uint64_t *generation, host_err *e) {
    const host_api *a = &t->h->d.api;
    if (t->direct) {
        elpis_ecs_c_status s;
        elpis_ecs_c_receipt r;
        int rc = a->elpis_ecs_c_read_status(t->direct, &s);
        if (!rc) rc = a->elpis_ecs_c_transition(t->direct, s.generation, s.state_root, event, &r);
        if (rc) return host_fail(e, ecs_c_code(t->h, rc), "the authority refused the event; nothing changed");
        return *generation = r.generation, 0;
    }
    const int rc = a->elpis_runtime_authority_transition(t->h->rt, event, generation);
    if (rc) {
        /* RUNTIME_AUTHORITY and CONTINUITY_INVALID come back from a transition only when the authority itself
         * refused, which RuntimeCore records as its exact last code: report that one, as the direct path does. */
        elpis_runtime_authority s;
        const int last = a->elpis_runtime_authority_status(t->h->rt, &s) ? 0 : s.ecs_c_code;
        if (last && (rc == ELPIS_RUNTIME_AUTHORITY || rc == ELPIS_CONTINUITY_INVALID))
            return host_fail(e, ecs_c_code(t->h, last), "the authority refused the event; nothing changed (through "
                                                        "RuntimeCore: %s)", rt_code(t->h, rc));
        return host_fail(e, rt_code(t->h, rc), "RuntimeCore refused the event before the authority saw it");
    }
    return 0;
}

static void put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* An operator event (elpis/ecs_c_authority.h, "Events"): the operator (row 0) at its next sequence. */
static int operator_event(held *t, uint16_t kind, const uint8_t *receiver, uint8_t ev[512], host_err *e) {
    elpis_ecs_c_entity op;
    if (held_entity(t, 0, &op, e)) return -1;
    if (op.kind != ELPIS_ECS_C_OPERATOR) return host_fail(e, "ECS_C_UNAUTHORIZED", "row 0 is not the operator");
    memset(ev, 0, 512);
    ev[0] = (uint8_t)kind, ev[1] = (uint8_t)(kind >> 8);
    memcpy(ev + 8, op.id, 32);
    put_u64(ev + 40, op.sender_watermark + 1);
    if (receiver) memcpy(ev + 48, receiver, 32);
    return 0;
}

static int find_source(held *t, uint32_t source_authority, elpis_ecs_c_entity *out, host_err *e) {
    uint32_t n;
    if (held_count(t, &n, e)) return -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (held_entity(t, i, out, e)) return -1;
        if (out->kind == ELPIS_ECS_C_SOURCE && out->source_authority == source_authority) return 0;
    }
    return host_fail(e, "ECS_C_UNKNOWN_ENTITY", "no source owns source authority %u", source_authority);
}

static void entity_out(hjw *w, const elpis_ecs_c_entity *x, uint32_t index) {
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

static const char *const RECONCILE_KINDS[] = {"ROLLBACK", "UNAUTHENTICATED", "REPIN"};

static void reconciliation_out(hjw *w, const elpis_ecs_c_reconciliation *r) {
    hjw_key(w, "reconciliation");
    if (r->kind < 11 || r->kind > 13) {
        hjw_null(w);
        return;
    }
    hjw_object(w);
    hjw_kstr(w, "kind", RECONCILE_KINDS[r->kind - 11]);
    hjw_khex(w, "digest", r->digest, 32);
    if (r->kind == 11) {
        hjw_ku64(w, "abandoned_generation", r->abandoned_generation);
        hjw_ku64(w, "class", r->class_code);
    } else if (r->kind == 12) {
        hjw_ku64(w, "slot", r->slot);
        hjw_ku64(w, "class", r->class_code);
    } else {
        hjw_key(w, "witness_classes");
        hjw_array(w);
        hjw_str(w, NAME(SLOT_CLASSES, r->witness_classes[0]));
        hjw_str(w, NAME(SLOT_CLASSES, r->witness_classes[1]));
        hjw_end(w, ']');
    }
    hjw_end(w, '}');
}

static void report_out(hjw *w, const elpis_ecs_c_report *r) {
    hjw_key(w, "report");
    hjw_object(w);
    hjw_kstr(w, "disposition", disposition_name(r->disposition));
    hjw_kstr(w, "mode", NAME(MODES, r->mode));
    hjw_key(w, "slot_classes");
    hjw_array(w);
    hjw_str(w, NAME(SLOT_CLASSES, r->slot_class[0]));
    hjw_str(w, NAME(SLOT_CLASSES, r->slot_class[1]));
    hjw_end(w, ']');
    hjw_key(w, "witness_classes");
    hjw_array(w);
    hjw_str(w, NAME(SLOT_CLASSES, r->witness_class[0]));
    hjw_str(w, NAME(SLOT_CLASSES, r->witness_class[1]));
    hjw_end(w, ']');
    hjw_key(w, "slot_generations"); /* the slots' untrusted header claims (0 when unreadable) */
    hjw_array(w);
    hjw_u64(w, r->slot_generation[0]);
    hjw_u64(w, r->slot_generation[1]);
    hjw_end(w, ']');
    hjw_ku64(w, "witness_generation", r->witness_generation);
    hjw_kbool(w, "replay_verified", r->replay_verified);
    hjw_kbool(w, "witness_completed", r->witness_completed);
    hjw_kbool(w, "evolution_pending", r->evolution_pending);
    hjw_ku64(w, "current_generation", r->current_generation);
    hjw_ku64(w, "reads", r->reads);
    hjw_ku64(w, "read_bytes", r->read_bytes);
    hjw_ku64(w, "replays", r->replays);
    hjw_end(w, '}');
}

static int op_authority(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)op;
    held t;
    if (h->opened) {
        authority_summary_out(h, w);
    } else {
        const int rc = held_begin(h, &t, e);
        if (rc && !strcmp(e->code, "ECS_C_RECOVERY_REFUSED")) {
            hjw_kbool(w, "held", 0);
            report_out(w, &t.report);
            e->keep = 1; /* the report says why */
            return -1;
        }
        if (rc) return -1;
        elpis_ecs_c_status s;
        elpis_ecs_c_reconciliation r;
        const host_api *a = &h->d.api;
        if (a->elpis_ecs_c_read_status(t.direct, &s) || a->elpis_ecs_c_read_reconciliation(t.direct, &r)) {
            held_end(&t);
            return host_fail(e, "ECS_C_CLOSED", "the authority could not be read");
        }
        hjw_key(w, "authority");
        hjw_object(w);
        hjw_kbool(w, "held", 0);
        hjw_ku64(w, "generation", s.generation);
        hjw_kstr(w, "mode", NAME(MODES, s.mode));
        hjw_kbool(w, "anchored", s.anchored);
        hjw_ku64(w, "receipt_count", s.receipt_count);
        hjw_ku64(w, "evidence_count", s.evidence_count);
        hjw_ku64(w, "entity_count", s.entity_count);
        hjw_ku64(w, "capacity", s.capacity);
        hjw_ku64(w, "publications", s.publications);
        hjw_ku64(w, "reconcile_count", s.reconcile_count);
        hjw_ku64(w, "reclaimed_count", s.reclaimed_count);
        hjw_khex(w, "lineage", s.lineage, 32);
        hjw_khex(w, "deployment", s.deployment, 32);
        hjw_khex(w, "record_digest", s.record_digest, 32);
        hjw_khex(w, "state_root", s.state_root, 32);
        hjw_khex(w, "receipt_head", s.receipt_head, 32);
        hjw_khex(w, "evidence_head", s.evidence_head, 32);
        hjw_end(w, '}');
        report_out(w, &t.report);
        reconciliation_out(w, &r);
        held_end(&t);
    }
    if (held_begin(h, &t, e)) return -1;
    uint32_t n;
    int rc = held_count(&t, &n, e);
    hjw_key(w, "entities");
    hjw_array(w);
    for (uint32_t i = 0; !rc && i < n; ++i) {
        elpis_ecs_c_entity x;
        rc = held_entity(&t, i, &x, e);
        if (!rc) entity_out(w, &x, i);
    }
    hjw_end(w, ']');
    held_end(&t);
    return rc;
}

static int op_authority_reconcile(host *h, const hj_value *op, hjw *w, host_err *e) {
    const host_api *a = &h->d.api;
    if (h->opened)
        return host_fail(e, "ECS_C_NOT_RECONCILABLE", "the runtime holds the authority in its normal mode: nothing to "
                                                      "reconcile");
    const char *kind = str_member(op, "kind");
    uint8_t digest[32];
    if (hex_member(op, "digest", digest, e)) return -1;
    int k = 0;
    for (int i = 0; i < 3; ++i)
        if (kind && !strcmp(kind, RECONCILE_KINDS[i])) k = 11 + i;
    if (!k) return input(e, "%s: ROLLBACK, UNAUTHENTICATED or REPIN (as `authority` reports it)", "kind");
    held t;
    if (held_begin(h, &t, e)) return -1;
    elpis_ecs_c_reconciliation r;
    int rc = a->elpis_ecs_c_read_reconciliation(t.direct, &r);
    if (rc || !r.kind) {
        held_end(&t);
        return host_fail(e, rc ? ecs_c_code(h, rc) : "ECS_C_NOT_RECONCILABLE", "the authority accepts no reconciliation");
    }
    /* The operator names exactly what the authority reported; anything else is refused before an event exists. */
    if (r.kind != k || memcmp(r.digest, digest, 32)) {
        held_end(&t);
        return host_fail(e, "AUTHORITY_RECONCILE_MISMATCH", "the authority accepts only the %s reconciliation of the "
                                                            "digest `authority` reports", RECONCILE_KINDS[r.kind - 11]);
    }
    uint8_t ev[512];
    rc = operator_event(&t, (uint16_t)k, NULL, ev, e);
    uint8_t *p = ev + ELPIS_ECS_C_PAYLOAD_AT;
    if (k == 11) put_u64(p, r.abandoned_generation), p[8] = r.class_code, memcpy(p + 16, r.digest, 32);
    else if (k == 12) p[0] = r.slot, p[1] = r.class_code, memcpy(p + 8, r.digest, 32);
    else p[0] = r.witness_classes[0], p[1] = r.witness_classes[1], memcpy(p + 8, r.digest, 32);
    uint64_t generation = 0;
    if (!rc) rc = held_send(&t, ev, &generation, e);
    held_end(&t);
    if (rc) return -1;
    hjw_kstr(w, "kind", kind);
    hjw_ku64(w, "generation", generation);
    return 0;
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
    held t;
    if (held_begin(h, &t, e)) return -1;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = held_next_id(&t, ELPIS_ECS_C_SOURCE, label, id, e);
    if (!rc) rc = operator_event(&t, 2, id, ev, e);
    if (!rc) {
        uint8_t *p = ev + ELPIS_ECS_C_PAYLOAD_AT;
        p[0] = ELPIS_ECS_C_SOURCE;
        put_u32(p + 4, (uint32_t)bound);
        put_u64(p + 8, version);
        memcpy(p + 16, label, 32);
        memcpy(p + 48, scope, 32);
        put_u32(p + 80, (uint32_t)authority);
        rc = held_send(&t, ev, &generation, e);
    }
    const int active = !activate || activate->boolean;
    if (!rc && active && !(rc = operator_event(&t, 3, id, ev, e))) {
        ev[ELPIS_ECS_C_PAYLOAD_AT] = ELPIS_ECS_C_FOUNDED, ev[ELPIS_ECS_C_PAYLOAD_AT + 1] = ELPIS_ECS_C_ACTIVE;
        rc = held_send(&t, ev, &generation, e);
    }
    held_end(&t);
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
    held t;
    if (held_begin(h, &t, e)) return -1;
    elpis_ecs_c_entity s;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = find_source(&t, (uint32_t)authority, &s, e);
    if (!rc && !(rc = operator_event(&t, 3, s.id, ev, e))) {
        ev[ELPIS_ECS_C_PAYLOAD_AT] = s.lifecycle, ev[ELPIS_ECS_C_PAYLOAD_AT + 1] = (uint8_t)to;
        rc = held_send(&t, ev, &generation, e);
    }
    held_end(&t);
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
    held t;
    if (held_begin(h, &t, e)) return -1;
    elpis_ecs_c_entity s;
    uint8_t ev[512], scope[32];
    uint64_t generation = 0;
    int rc = find_source(&t, (uint32_t)authority, &s, e);
    if (!rc) {
        memcpy(scope, s.scope, 32);
        if (u64_member(op, "per_turn", ELPIS_ECS_C_MAX_PER_TURN, s.per_turn_limit, &bound, e) || scope_in(op, 0, scope, e))
            rc = -1;
    }
    if (!rc && !(rc = operator_event(&t, 4, s.id, ev, e))) {
        uint8_t *p = ev + ELPIS_ECS_C_PAYLOAD_AT;
        put_u64(p, expected); /* compare-and-set: the authority refuses a stale expectation */
        put_u64(p + 8, version);
        put_u32(p + 16, (uint32_t)bound);
        memcpy(p + 24, scope, 32);
        rc = held_send(&t, ev, &generation, e);
    }
    held_end(&t);
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
    held t;
    if (held_begin(h, &t, e)) return -1;
    elpis_ecs_c_entity s;
    uint8_t ev[512];
    uint64_t generation = 0;
    int rc = find_source(&t, (uint32_t)authority, &s, e);
    if (!rc && !(rc = operator_event(&t, 14, s.id, ev, e))) rc = held_send(&t, ev, &generation, e);
    held_end(&t);
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

#ifdef ELPIS_HOST_TESTING
/* ---- testing host only: the runtime's fault plans and a crash at an operation boundary ------------------------ */

static int op_testing_durable_fault(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t action, step, arg;
    if (lineage(h, e) || u64_member(op, "action", 10, 0, &action, e) || u64_member(op, "step", 9, 0, &step, e) ||
        u64_member(op, "arg", INT64_MAX, 0, &arg, e))
        return -1;
    const int rc = h->d.api.elpis_runtime_testing_durable_fault(h->rt, (uint32_t)action, (uint32_t)step, arg);
    if (rc) return host_fail(e, rt_code(h, rc), "the durable fault plan was refused");
    hjw_kbool(w, "armed", action != 0);
    return 0;
}

static int op_testing_control_fault(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t kind, action, point, arg;
    if (lineage(h, e) || u64_member(op, "kind", 4, 0, &kind, e) || u64_member(op, "action", 2, 0, &action, e) ||
        u64_member(op, "point", 6, 0, &point, e) || u64_member(op, "arg", INT64_MAX, 0, &arg, e))
        return -1;
    const int rc = h->d.api.elpis_runtime_testing_control_fault(h->rt, (uint32_t)kind, (uint32_t)action,
                                                                (uint32_t)point, arg);
    if (rc) return host_fail(e, rt_code(h, rc), "the control fault plan was refused");
    hjw_kbool(w, "armed", action != 0);
    return 0;
}

/* Wait (bounded) until the durability worker is parked at `parked`: the moment a harness kills the process. */
static int op_testing_wait_parked(host *h, const hj_value *op, hjw *w, host_err *e) {
    uint64_t parked, timeout;
    if (lineage(h, e) || u64_member(op, "parked", 1000, 0, &parked, e) ||
        u64_member(op, "timeout_ms", 600000, 30000, &timeout, e))
        return -1;
    for (uint64_t waited = 0;; ++waited) {
        elpis_runtime_durability du;
        const int rc = h->d.api.elpis_runtime_durable_status(h->rt, &du);
        if (rc) return host_fail(e, rt_code(h, rc), "the durability status could not be read");
        if (du.parked == parked) break;
        if (waited >= timeout) return host_fail(e, "TESTING_NOT_PARKED", "the worker did not park at %llu",
                                                (unsigned long long)parked);
        nanosleep(&(struct timespec){0, 1000000}, NULL);
    }
    hjw_ku64(w, "parked", parked);
    return 0;
}

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

/* What a write would do here (under --contained: what the containment allows): "create" a new file, "write" an
 * existing one, or "extend" one past `offset`. Reports the errno (0: it happened). */
static int op_testing_write_probe(host *h, const hj_value *op, hjw *w, host_err *e) {
    (void)h;
    const char *path = str_member(op, "path"), *kind = str_member(op, "kind");
    uint64_t offset;
    if (!path || path[0] != '/' || !kind || u64_member(op, "offset", INT64_MAX, 0, &offset, e))
        return input(e, "%s: an absolute path, a kind (create, write, extend) and an offset", "testing_write_probe");
    signal(SIGXFSZ, SIG_IGN); /* past the file size limit a write fails EFBIG instead of killing the process */
    errno = 0;
    int rc;
    if (!strcmp(kind, "create")) {
        const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        rc = fd < 0 ? errno : 0;
        if (fd >= 0) close(fd), unlink(path);
    } else if (!strcmp(kind, "write") || !strcmp(kind, "extend")) {
        const int fd = open(path, O_WRONLY | O_CLOEXEC);
        rc = fd < 0 ? errno : 0;
        if (fd >= 0) {
            /* "write" rewrites the first byte with itself; "extend" writes one byte at `offset`. */
            const off_t at = !strcmp(kind, "extend") ? (off_t)offset : 0;
            char byte = 0;
            if (at == 0 && pread(fd, &byte, 1, 0) != 1) byte = 0;
            rc = pwrite(fd, &byte, 1, at) == 1 ? 0 : errno;
            close(fd);
        }
    } else {
        return input(e, "%s: kind is create, write or extend", "testing_write_probe");
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
    {"candidates", op_candidates},
    {"query", op_query},
    {"learn", op_learn},
    {"evidence", op_evidence},
    {"lookup", op_lookup},
    {"slot", op_slot},
    {"publish", op_publish},
    {"reconcile", op_reconcile},
    {"authority", op_authority},
    {"authority_reconcile", op_authority_reconcile},
    {"source_found", op_source_found},
    {"source_lifecycle", op_source_lifecycle},
    {"source_version", op_source_version},
    {"source_reclaim", op_source_reclaim},
    {"text", op_text},
#ifdef ELPIS_HOST_TESTING
    {"testing_durable_fault", op_testing_durable_fault},
    {"testing_control_fault", op_testing_control_fault},
    {"testing_wait_parked", op_testing_wait_parked},
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
    const size_t need = len * ARENA_PER_BYTE + ((size_t)1 << 20);
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
    char *buf = malloc(MAX_INPUT + 1);
    size_t n = buf ? fread(buf, 1, MAX_INPUT + 1, f) : 0;
    if (f != stdin) fclose(f);
    if (!buf || n > MAX_INPUT) {
        free(buf);
        return snprintf(err, errlen, "a script is at most %zu bytes", MAX_INPUT), NULL;
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
    int publish; /* --publish appends a publication */
} command_spec;

static const command_spec COMMANDS[] = {
    {"status", "status", {{0}}, 0},
    {"live", "live", {{0}}, 0},
    {"counters", "counters", {{"--reset", "reset", A_FLAG}}, 0},
    {"candidates", "candidates", {{0}}, 0},
    {"query", "query", {{"--x", "x", A_JSON}}, 0},
    {"learn",
     "learn",
     {{"--x", "x", A_JSON}, {"--y", "y", A_JSON}, {"--schedule", "schedule", A_JSON}, {"--rate", "rate", A_DOUBLE},
      {"--grant", "grant", A_STRING}},
     1},
    {"evidence",
     "evidence",
     {{"--items", "items", A_JSON}, {"--grant", "grant", A_STRING}, {"--sources", "sources", A_SOURCES}},
     1},
    {"lookup",
     "lookup",
     {{"--relation", "relation", A_ID}, {"--target-role", "target_role", A_ID}, {"--cues", "cues", A_JSON}},
     0},
    {"slot", "slot", {{"--key", "key", A_STRING}}, 0},
    {"publish", "publish", {{0}}, 0},
    {"reconcile", "reconcile", {{"--discard", "discard", A_STRING}, {"--adopt", "adopt", A_STRING}}, 0},
    {"authority", "authority", {{0}}, 0},
    {"authority-reconcile", "authority_reconcile", {{"--kind", "kind", A_STRING}, {"--digest", "digest", A_STRING}}, 0},
    {"source-found",
     "source_found",
     {{"--label", "label", A_STRING}, {"--source-authority", "source_authority", A_U64}, {"--per-turn", "per_turn", A_U64},
      {"--version", "version", A_U64}, {"--scope", "scope", A_STRING}, {"--scope-label", "scope_label", A_STRING},
      {"--dormant", "dormant", A_FLAG}},
     0},
    {"source-lifecycle",
     "source_lifecycle",
     {{"--source-authority", "source_authority", A_U64}, {"--to", "to", A_STRING}},
     0},
    {"source-version",
     "source_version",
     {{"--source-authority", "source_authority", A_U64}, {"--expected", "expected", A_U64},
      {"--version", "version", A_U64}, {"--per-turn", "per_turn", A_U64}, {"--scope", "scope", A_STRING},
      {"--scope-label", "scope_label", A_STRING}},
     0},
    {"source-reclaim", "source_reclaim", {{"--source-authority", "source_authority", A_U64}}, 0},
    {"text", "text", {{"--input", "input", A_STRING}}, 0},
};

static void usage(void) {
    fprintf(stderr, "usage: elpis --home DIR init (--native-build BUILD | --libraries DIR) [--capacity N] "
                    "[--authority-dir DIR] [--source-commit SHA]\n"
                    "       elpis --home DIR create [--dim N --width N --max-rows N --memory-dim N --banks N "
                    "--slot-capacity N --codebook-capacity N --seed N]\n"
                    "       elpis --home DIR [--contained] session --script FILE|-\n"
                    "       elpis --home DIR [--contained] <command> [options]   (docs/NATIVE_HOST.md)\n");
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
        if (c->publish && !strcmp(argv[i], "--publish")) continue;
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
    if (!strcmp(c->op, "reconcile")) { /* --discard ID | --adopt ID */
        if (n != 2) return snprintf(err, errlen, "reconcile: exactly one of --discard or --adopt"), -1;
        items[2] = items[1];
        keys[2] = "candidate";
        items[1] = (hj_value){.kind = HJ_STRING, .string = keys[1], .length = strlen(keys[1])};
        keys[1] = "action";
        n = 3;
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

static int cmd_create(host *h, int argc, char **argv) {
    host_lineage l = {6, 36, 64, 64, 2, 64, 128, 0}; /* the Python LineageConfig's defaults */
    static const char *const names[] = {"--dim", "--width", "--max-rows", "--memory-dim", "--banks",
                                        "--slot-capacity", "--codebook-capacity", "--seed"};
    uint64_t *fields[] = {&l.dim, &l.width, &l.max_rows, &l.memory_dim, &l.banks, &l.slot_capacity,
                          &l.codebook_capacity, &l.seed};
    for (int i = 0; i < argc; i += 2) {
        int known = 0;
        for (int j = 0; j < 8; ++j)
            if (!strcmp(argv[i], names[j])) {
                known = 1;
                if (i + 1 >= argc || parse_u64_arg(argv[i + 1], fields[j])) {
                    emit_failure("create", "CLI_INPUT", "every create option takes an unsigned integer");
                    return 2;
                }
            }
        if (!known) {
            emit_failure("create", "CLI_INPUT", "create: unknown option");
            return 2;
        }
    }
    host_err e = {{0}, {0}, 0};
    uint8_t identity[32];
    elpis_continuity_snapshot anchored;
    if (host_create_lineage(&h->d, &l, identity, &anchored, &e)) {
        emit_failure("create", e.code, e.detail);
        return 3;
    }
    hjw w;
    hjw_init(&w, OUT, sizeof OUT);
    hjw_object(&w);
    hjw_kstr(&w, "op", "create");
    hjw_khex(&w, "identity", identity, 32);
    hjw_kstr(&w, "state_format", anchored.state_format == ELPIS_CONTINUITY_FORMAT_COMPOSITE ? "COMPOSITE" : "K1");
    hjw_ku64(&w, "authority_generation", anchored.generation);
    hjw_khex(&w, "initial_w_sha256", h->d.initial_w_sha256, 32);
    hjw_ku64(&w, "envelope_bytes", h->d.envelope_bytes);
    hjw_kbool(&w, "ok", 1);
    hjw_end(&w, '}');
    emit(&w);
    return 0;
}

static int cmd_init(int argc, char **argv, const char *home) {
    const char *build = NULL, *libraries = NULL, *authority = NULL, *commit = NULL;
    uint64_t capacity = 16;
    for (int i = 0; i < argc; i += 2) {
        if (i + 1 >= argc) return emit_failure("init", "CLI_INPUT", "every init option takes a value"), 2;
        if (!strcmp(argv[i], "--native-build")) build = argv[i + 1];
        else if (!strcmp(argv[i], "--libraries")) libraries = argv[i + 1];
        else if (!strcmp(argv[i], "--authority-dir")) authority = argv[i + 1];
        else if (!strcmp(argv[i], "--source-commit")) commit = argv[i + 1];
        else if (!strcmp(argv[i], "--capacity") && !parse_u64_arg(argv[i + 1], &capacity) && capacity <= UINT32_MAX) continue;
        else return emit_failure("init", "CLI_INPUT", "init: unknown or malformed option"), 2;
    }
    if (!build == !libraries)
        return emit_failure("init", "CLI_INPUT", "init needs exactly one of --native-build or --libraries"), 2;
    static host_deployment d;
    host_err e = {{0}, {0}, 0};
    if (host_init(home, build ? build : libraries, !build, authority, (uint32_t)capacity, commit, &d, &e)) {
        emit_failure("init", e.code, e.detail);
        return 4;
    }
    hjw w;
    hjw_init(&w, OUT, sizeof OUT);
    hjw_object(&w);
    hjw_kstr(&w, "op", "init");
    hjw_kstr(&w, "home", d.home);
    hjw_khex(&w, "catalog_sha256", d.catalog_sha256, 32);
    hjw_khex(&w, "host_sha256", d.host_sha256, 32);
    hjw_kstr(&w, "authority_dir", d.authority_dir);
    hjw_ku64(&w, "capacity", d.capacity);
    hjw_key(&w, "libraries");
    hjw_object(&w);
    for (int i = 0; i < HOST_LIBRARIES; ++i) {
        hjw_key(&w, d.pins[i].id);
        hjw_object(&w);
        hjw_khex(&w, "sha256", d.pins[i].sha256, 32);
        hjw_ku64(&w, "size", d.pins[i].size);
        hjw_end(&w, '}');
    }
    hjw_end(&w, '}');
    hjw_kbool(&w, "testing_build", d.testing);
    hjw_kbool(&w, "ok", 1);
    hjw_end(&w, '}');
    emit(&w);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || strcmp(argv[1], "--home")) return usage(), 2;
    const char *home = argv[2];
    /* --contained: the host restricts itself after admission (docs/NATIVE_HOST.md, "Containment"). */
    const int contained = !strcmp(argv[3], "--contained");
    if (contained) {
        argv += 1, argc -= 1;
        if (argc < 4) return usage(), 2;
        host_err ce = {{0}, {0}, 0};
        if (host_contain_stdio(&ce)) return emit_failure(argv[3], ce.code, ce.detail), 4;
    }
    const char *command = argv[3];
    if (contained && (!strcmp(command, "init") || !strcmp(command, "create")))
        return emit_failure(command, "CLI_INPUT", "provisioning creates files: init and create are never contained"), 2;
    if (!strcmp(command, "init")) return cmd_init(argc - 4, argv + 4, home);
    /* Validate the command line before the deployment is touched. */
    const command_spec *spec = NULL;
    for (size_t i = 0; i < sizeof COMMANDS / sizeof *COMMANDS; ++i)
        if (!strcmp(COMMANDS[i].command, command)) spec = &COMMANDS[i];
    char err[256] = {0};
    static hj_value items[16], one_shot;
    static const char *keys[16];
    char *script = NULL;
    size_t script_len = 0;
    hj_value *script_doc = NULL;
    int publish = 0;
    if (!strcmp(command, "session")) {
        if (argc != 6 || strcmp(argv[4], "--script")) return usage(), 2;
        if (!(script = read_script(argv[5], &script_len, err, sizeof err))) return emit_failure("session", "CLI_INPUT", err), 2;
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
        if (arena_for(MAX_INPUT)) return emit_failure(command, "HOST_RESOURCES", "out of memory"), 2;
        if (command_op(spec, argc - 4, argv + 4, &one_shot, items, keys, err, sizeof err))
            return emit_failure(spec->op, "CLI_INPUT", err), 2;
        for (int i = 4; i < argc; ++i) publish |= spec->publish && !strcmp(argv[i], "--publish");
    } else if (strcmp(command, "create")) {
        return usage(), 2;
    }
    static host h;
    host_err e = {{0}, {0}, 0};
    if (host_open(home, &h.d, &e) || (contained && host_contain(&h.d, &e))) {
        free(script);
        return emit_failure(spec ? spec->op : command, e.code, e.detail), 4;
    }
    h.d.api.elpis_runtime_fuel_ceiling(&h.ceiling);
    int all_ok = 1;
    if (!strcmp(command, "create")) {
        all_ok = cmd_create(&h, argc - 4, argv + 4) == 0;
    } else if (script_doc) {
        for (size_t i = 0; i < script_doc->count; ++i) all_ok &= run(&h, &script_doc->items[i]);
    } else if (script) {
        script_lines(script, script_len, 1, &h, &all_ok, err, sizeof err);
    } else {
        all_ok = run(&h, &one_shot);
        if (publish) {
            static hj_value p_items[1];
            static const char *p_keys[1] = {"op"};
            p_items[0] = (hj_value){.kind = HJ_STRING, .string = "publish", .length = 7};
            const hj_value p = {.kind = HJ_OBJECT, .items = p_items, .keys = p_keys, .count = 1};
            all_ok &= run(&h, &p);
        }
    }
    session_close(&h);
    free(script);
    free(ARENA.bytes);
    return all_ok ? 0 : 3;
}
