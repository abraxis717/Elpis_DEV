/* The native host's installation (docs/NATIVE_HOST.md, "The installation"): the immutable libraries it runs, admitted
 * read-only into sealed memory against the operator's catalog pin, and the configuration of the sessions it opens.
 * RAM-only (docs/ELPIS_MISSION.md, Amendment 3): there is no home, no deployment state, no configuration file and no
 * lineage on disk. Every session starts from a deterministic initial state computed in memory and ends with the
 * process (or `close`); nothing is ever written.
 *
 * The catalog: one canonical JSON object naming this host binary's SHA-256 and, for each library in admission order,
 * its id, file, SONAME, size and SHA-256 (all measured from the sealed copies). `elpis catalog` prints it and its
 * SHA-256; every other command requires that SHA-256 (--catalog-sha256), so the operator's one pin authenticates the
 * host binary and every library byte that runs. */
#ifndef ELPIS_HOST_INSTALL_H
#define ELPIS_HOST_INSTALL_H

#include <stddef.h>
#include <stdint.h>

#include "elpis/ecs_composite.h"
#include "elpis/ecs_observation_hacf.h"
#include "elpis/runtime.h"
#include "host_admit.h"

enum { HOST_LIBRARIES = 6, HOST_PATH = 4096 };
/* Admission order (dependencies first); RuntimeCore last. */
enum { LIB_MATH, LIB_K1, LIB_MEMORY, LIB_COMPOSITE, LIB_HACF, LIB_RUNTIME };

#define HOST_RUNTIME_FUNCTIONS(X)                                                                                  \
    X(elpis_runtime_abi_version) X(elpis_runtime_features) X(elpis_runtime_fuel_ceiling)                          \
    X(elpis_runtime_code_name) X(elpis_runtime_create) X(elpis_runtime_destroy) X(elpis_runtime_open)             \
    X(elpis_runtime_close) X(elpis_runtime_fault) X(elpis_runtime_snapshot) X(elpis_runtime_read_counters)        \
    X(elpis_runtime_anchor) X(elpis_runtime_query) X(elpis_runtime_turn_begin) X(elpis_runtime_turn_commit)       \
    X(elpis_runtime_turn_abort) X(elpis_runtime_turn_observe) X(elpis_runtime_turn_begin_observe)                 \
    X(elpis_runtime_live_identity) X(elpis_runtime_authority_status) X(elpis_runtime_authority_transition)        \
    X(elpis_runtime_authority_entity_at) X(elpis_runtime_authority_next_entity_id)

#define HOST_COMPOSITE_FUNCTIONS(X)                                                                                \
    X(elpis_ecs_composite_code_name) X(elpis_ecs_composite_envelope_bytes) X(elpis_ecs_composite_create)          \
    X(elpis_ecs_composite_destroy) X(elpis_ecs_composite_state_digest) X(elpis_ecs_composite_shape)               \
    X(elpis_ecs_composite_memory_query) X(elpis_ecs_composite_memory_slot_read)                                   \
    X(elpis_ecs_composite_snapshot_write) X(elpis_ecs_composite_inspect) X(elpis_ecs_composite_lease_claim)       \
    X(elpis_ecs_composite_lease_release) X(elpis_ecs_composite_leased_reserve)                                    \
    X(elpis_ecs_composite_leased_identity) X(elpis_ecs_composite_leased_query)                                    \
    X(elpis_ecs_composite_leased_txn_begin_at) X(elpis_ecs_composite_leased_txn_run_schedule)                     \
    X(elpis_ecs_composite_leased_txn_commit_at) X(elpis_ecs_composite_txn_abort)                                  \
    X(elpis_ecs_composite_leased_txn_observe) X(elpis_ecs_composite_envelope_identities)

#define HOST_MEMORY_FUNCTIONS(X) X(elpis_ecs_mem_config_default)

#define HOST_HACF_FUNCTIONS(X)                                                                                     \
    X(elpis_semantic_canonicalize_participants) X(elpis_semantic_hyperedge_identity)                              \
    X(elpis_semantic_hyperedge_validate) X(elpis_semantic_assertion_identity) X(elpis_semantic_assertion_validate) \
    X(elpis_ecs_obs_hacf_observation) X(elpis_ecs_obs_hacf_hyperedge)

/* Every native function the host calls, resolved from the admitted libraries only (never linked). */
typedef struct {
#define HOST_FIELD(name) __typeof__(name) *name;
    HOST_RUNTIME_FUNCTIONS(HOST_FIELD)
    HOST_COMPOSITE_FUNCTIONS(HOST_FIELD)
    HOST_MEMORY_FUNCTIONS(HOST_FIELD)
    HOST_HACF_FUNCTIONS(HOST_FIELD)
#undef HOST_FIELD
} host_api;

/* A refusal: a stable code and a human-readable detail. */
typedef struct {
    char code[64];
    char detail[768];
    int keep; /* the operation's members written before the refusal stay in its output */
} host_err;

int host_fail(host_err *e, const char *code, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

/* The lineage every session starts (the LineageConfig of the Python runtime, same defaults): the composite's shape
 * and the seed of its deterministic initial weights. */
typedef struct {
    uint64_t dim, width, max_rows, memory_dim, banks, slot_capacity, codebook_capacity, seed;
} host_lineage;

/* Where the libraries are read from: an installed library directory (flat file names) or an operator build tree
 * (each library at its build path). Read-only either way. */
typedef struct {
    const char *dir;
    int build_tree;
} host_source;

typedef struct {
    host_source source;
    uint8_t host_sha256[32], catalog_sha256[32];
    host_pin pins[HOST_LIBRARIES];
    void *handles[HOST_LIBRARIES];
    int sealed[HOST_LIBRARIES]; /* the admitted libraries' sealed descriptors, open for the life of the process */
    host_api api;
    host_lineage lineage;
    elpis_runtime_config session; /* every session's configuration (capacity, nonce seed, founder labels) */
    uint8_t initial_identity[32], initial_w_sha256[32];
    uint64_t envelope_bytes;
    int testing;
} host_install;

/* The library set: catalog id, file name, SONAME, build path beneath a build tree. */
typedef struct {
    const char *id, *file, *soname, *build_path;
} host_library;
const host_library *host_libraries(void);

/* Seal and measure every library and the host binary; compute the catalog and its SHA-256 (`catalog` NULL: not
 * written out). Nothing is loaded. */
int host_measure(host_install *d, char *catalog, size_t catalog_cap, size_t *catalog_len, host_err *e);
/* Load the measured libraries (after the catalog pin matched), resolve every function the host calls and check the
 * ABI versions (RuntimeCore 8, composite 1). */
int host_load_libraries(host_install *d, host_err *e);
/* The lineage's derived values: the composite configuration's envelope size, the initial weights' SHA-256 and the
 * initial composite identity (a composite created and destroyed in memory). */
int host_lineage_check(host_install *d, host_err *e);
/* The composite's RuntimeCore function table (typed wrappers over the admitted entries). */
const elpis_runtime_composite_api *host_substrate_api(void);
/* The ELPISEC1 composite configuration of the lineage (no memory capability, ever). */
elpis_ecs_composite_config host_composite_config(const host_install *d);
/* The deterministic initial K1 weights (elpis.deployment.initial-w.v1): dim * width values into `w`. */
void host_initial_weights(const host_lineage *l, double *w);
/* A 32-byte label or scope of operator text: SHA-256("elpis.host.label.v1" || 0x00 || text). */
void host_label(const char *text, uint8_t out[32]);
/* Hex helpers: 64 lowercase hex characters <-> 32 bytes. */
int host_unhex32(const char *text, uint8_t out[32]);

#endif
