/* One native Elpis deployment (docs/NATIVE_HOST.md, "The home"): an operator-owned home holding the pinned
 * libraries, the ECS_C authority and one ECS composite lineage, every file fixed in number and size.
 *
 *   <home>/host.cfg        the configuration and trust root (write-once): the host binary's and the catalog's pins
 *   <home>/catalog         the pinned libraries, in admission order (write-once)
 *   <home>/lib/            the pinned libraries, copied from an operator build at init (write-once, 0500)
 *   <home>/state/          the ECS_C state directory (authority.a, authority.b: fixed-size records)
 *   <authority dir>        the ECS_C authority directory (deployment.key, witness.a, witness.b); <home>/authority
 *                          unless init names another protected directory
 *   <home>/lineage.cfg     the lineage configuration and the provenance of its initial state (write-once, at create)
 *   <home>/checkpoint/     the K1 Recovery R0 store: two fixed slot files sized for the composite envelope
 *
 * Nothing here keeps logs, history or caches: init writes the first three, create the last two, and afterwards
 * every transition rewrites fixed slots in place. */
#ifndef ELPIS_HOST_DEPLOY_H
#define ELPIS_HOST_DEPLOY_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecs_c_authority.h"
#include "elpis/ecs_composite.h"
#include "elpis/ecs_observation_hacf.h"
#include "elpis/runtime.h"
#include "host_admit.h"

#ifdef ELPIS_HOST_TESTING
/* libelpis_runtime_testing.so only (native/runtime/src/ffi.rs, mod testing). */
int elpis_runtime_testing_durable_fault(elpis_runtime *runtime, uint32_t action, uint32_t step, uint64_t arg);
int elpis_runtime_testing_control_fault(elpis_runtime *runtime, uint32_t op, uint32_t action, uint32_t point,
                                        uint64_t arg);
#define HOST_TESTING_FUNCTIONS(X) X(elpis_runtime_testing_durable_fault) X(elpis_runtime_testing_control_fault)
#else
#define HOST_TESTING_FUNCTIONS(X)
#endif

enum { HOST_LIBRARIES = 7, HOST_PATH = 4096 };
/* Admission order (dependencies first); RuntimeCore last. */
enum { LIB_MATH, LIB_K1, LIB_MEMORY, LIB_COMPOSITE, LIB_HACF, LIB_ECS_C, LIB_RUNTIME };

#define HOST_RUNTIME_FUNCTIONS(X)                                                                                  \
    X(elpis_runtime_abi_version) X(elpis_runtime_features) X(elpis_runtime_fuel_ceiling)                          \
    X(elpis_runtime_code_name) X(elpis_runtime_create_ecs_c) X(elpis_runtime_destroy) X(elpis_runtime_open)       \
    X(elpis_runtime_close) X(elpis_runtime_fault) X(elpis_runtime_snapshot) X(elpis_runtime_read_counters)        \
    X(elpis_runtime_anchor) X(elpis_runtime_query) X(elpis_runtime_turn_begin) X(elpis_runtime_turn_commit)       \
    X(elpis_runtime_turn_abort) X(elpis_runtime_turn_observe) X(elpis_runtime_turn_begin_observe)                 \
    X(elpis_runtime_recoverable_request) X(elpis_runtime_durable_wait) X(elpis_runtime_durable_status)            \
    X(elpis_runtime_live_identity) X(elpis_runtime_checkpoint_provision) X(elpis_runtime_checkpoint_attach)       \
    X(elpis_runtime_checkpoint_recover) X(elpis_runtime_checkpoint_discard) X(elpis_runtime_checkpoint_adopt)     \
    X(elpis_runtime_authority_status) X(elpis_runtime_authority_transition) X(elpis_runtime_authority_entity_at)  \
    X(elpis_runtime_authority_next_entity_id)

#define HOST_COMPOSITE_FUNCTIONS(X)                                                                                \
    X(elpis_ecs_composite_code_name) X(elpis_ecs_composite_envelope_bytes) X(elpis_ecs_composite_create)          \
    X(elpis_ecs_composite_restore) X(elpis_ecs_composite_destroy) X(elpis_ecs_composite_state_digest)             \
    X(elpis_ecs_composite_shape) X(elpis_ecs_composite_memory_query) X(elpis_ecs_composite_memory_slot_read)      \
    X(elpis_ecs_composite_snapshot_write) X(elpis_ecs_composite_inspect) X(elpis_ecs_composite_lease_claim)       \
    X(elpis_ecs_composite_lease_release) X(elpis_ecs_composite_leased_reserve)                                    \
    X(elpis_ecs_composite_leased_identity) X(elpis_ecs_composite_leased_query)                                    \
    X(elpis_ecs_composite_leased_txn_begin_at) X(elpis_ecs_composite_leased_txn_run_schedule)                     \
    X(elpis_ecs_composite_leased_txn_commit_at) X(elpis_ecs_composite_txn_abort)                                  \
    X(elpis_ecs_composite_leased_capture) X(elpis_ecs_composite_envelope_seal)                                    \
    X(elpis_ecs_composite_leased_txn_observe) X(elpis_ecs_composite_envelope_identities)

#define HOST_MEMORY_FUNCTIONS(X) X(elpis_ecs_mem_config_default)

#define HOST_HACF_FUNCTIONS(X)                                                                                     \
    X(elpis_semantic_canonicalize_participants) X(elpis_semantic_hyperedge_identity)                              \
    X(elpis_semantic_hyperedge_validate) X(elpis_semantic_assertion_identity) X(elpis_semantic_assertion_validate) \
    X(elpis_ecs_obs_hacf_observation) X(elpis_ecs_obs_hacf_hyperedge)

#define HOST_ECS_C_FUNCTIONS(X)                                                                                    \
    X(elpis_ecs_c_abi_version) X(elpis_ecs_c_code_name) X(elpis_ecs_c_record_bytes) X(elpis_ecs_c_provision)    \
    X(elpis_ecs_c_open)                                                                                            \
    X(elpis_ecs_c_close) X(elpis_ecs_c_read_status) X(elpis_ecs_c_read_report) X(elpis_ecs_c_read_reconciliation) \
    X(elpis_ecs_c_entity_at) X(elpis_ecs_c_next_entity_id) X(elpis_ecs_c_transition)

/* Every native function the host calls, resolved from the admitted libraries only (never linked). */
typedef struct {
#define HOST_FIELD(name) __typeof__(name) *name;
    HOST_RUNTIME_FUNCTIONS(HOST_FIELD)
    HOST_TESTING_FUNCTIONS(HOST_FIELD)
    HOST_COMPOSITE_FUNCTIONS(HOST_FIELD)
    HOST_MEMORY_FUNCTIONS(HOST_FIELD)
    HOST_HACF_FUNCTIONS(HOST_FIELD)
    HOST_ECS_C_FUNCTIONS(HOST_FIELD)
#undef HOST_FIELD
} host_api;

/* A refusal: a stable code and a human-readable detail. */
typedef struct {
    char code[64];
    char detail[768];
    int keep; /* the operation's members written before the refusal stay in its output */
} host_err;

int host_fail(host_err *e, const char *code, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

/* The lineage's configuration (lineage.cfg; the LineageConfig of the Python deployment, same defaults). */
typedef struct {
    uint64_t dim, width, max_rows, memory_dim, banks, slot_capacity, codebook_capacity, seed;
} host_lineage;

typedef struct {
    char home[HOST_PATH];
    char state_dir[HOST_PATH], authority_dir[HOST_PATH], checkpoint_dir[HOST_PATH];
    int home_fd, lib_fd;
    int testing;
    char source_commit[41]; /* empty: none recorded */
    uint32_t capacity;      /* the ECS_C authority's entity capacity */
    uint8_t host_sha256[32], catalog_sha256[32];
    host_pin pins[HOST_LIBRARIES];
    void *handles[HOST_LIBRARIES];
    int sealed[HOST_LIBRARIES]; /* the admitted libraries' sealed descriptors, open for the life of the process */
    host_api api;
    int has_lineage;
    host_lineage lineage;
    uint8_t initial_identity[32], initial_w_sha256[32];
    uint64_t envelope_bytes;
#ifdef ELPIS_HOST_TESTING
    uint32_t testing_anchor_point; /* nonzero: create parks its anchor there (control fault 1, action 2) */
    uint64_t testing_anchor_arg;
#endif
} host_deployment;

/* The library set this host variant admits: catalog id, file name beneath lib/, SONAME, path in a build tree. */
typedef struct {
    const char *id, *file, *soname, *build_path;
} host_library;
const host_library *host_libraries(void);

/* init: a new deployment at `home` from the libraries of the operator build tree `build`, or (`flat`) of an installed
 * library directory holding them by file name (both absolute). `authority_dir` NULL: <home>/authority.
 * `source_commit` NULL or 40 lowercase hex. On success `d` is open (as host_open). */
int host_init(const char *home, const char *build, int flat, const char *authority_dir, uint32_t capacity,
              const char *source_commit, host_deployment *d, host_err *e);
/* Open an existing deployment: configuration, pins (host binary, catalog, every library), admission, lineage. */
int host_open(const char *home, host_deployment *d, host_err *e);
/* The composite's RuntimeCore function table (typed wrappers over the admitted entries of the open deployment). */
const elpis_runtime_composite_api *host_substrate_api(void);
/* The ELPISEC1 composite configuration of the lineage (no memory capability, ever). */
elpis_ecs_composite_config host_composite_config(const host_deployment *d, const host_lineage *l);
/* The deterministic initial K1 weights (elpis.deployment.initial-w.v1): dim * width values into `w`. */
void host_initial_weights(const host_lineage *l, double *w);
/* create: publish lineage.cfg, provision the checkpoint store, create the composite and anchor it (durable first:
 * the checkpoint slot, then the ECS_C authority). Each file is published under a temporary name and renamed into
 * place, so a create interrupted at any point leaves either nothing of a step or all of it. Another create of exactly
 * the same lineage (lineage.cfg byte-identical) completes an interrupted one: what is missing is made, an unanchored
 * lineage is anchored, and a lineage the authority already holds at its initial identity is reported (`resumed`),
 * with nothing written. Any other existing lineage is refused (LINEAGE_EXISTS). Fills `out_identity`. */
int host_create_lineage(host_deployment *d, const host_lineage *l, uint8_t out_identity[32],
                        elpis_continuity_snapshot *anchored, int *resumed, host_err *e);

#ifdef ELPIS_HOST_TESTING
/* Testing hosts: a detached thread that waits (bounded, 300 s) until the durability status of `rt` reports `parked`
 * and then writes one line, {"op":"testing_parked","parked":N}, to standard output: the moment a harness kills a
 * process whose caller is parked inside a control operation (which cannot report it itself). */
int host_testing_watch(const host_api *a, elpis_runtime *rt, uint32_t parked);
#endif

/* A 32-byte label or scope of operator text: SHA-256("elpis.host.label.v1" || 0x00 || text). */
void host_label(const char *text, uint8_t out[32]);
/* Hex helpers: 64 lowercase hex characters <-> 32 bytes. */
int host_unhex32(const char *text, uint8_t out[32]);

#endif
