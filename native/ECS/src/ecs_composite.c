/* ECS composite retained state (ELPISEC1 v1): the K1 state and the ECS memory under one owner, one transaction
 * and one identity. Contract: elpis/ecs_composite.h and docs/ECS_COMPOSITE_STATE.md. Owns no law of its own: K1
 * and the memory compute everything; this file stages, binds, serializes and commits. */
#include "elpis/ecs_composite.h"

#include "elpis/sha256.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t MAGIC[8] = {'E', 'L', 'P', 'I', 'S', 'E', 'C', '1'};
/* The private lease under which the composite holds its K1 state: every unmanaged K1 mutation refuses LEASED. */
static const uint64_t K1_LEASE = UINT64_C(0x454353434F4D5031); /* "ECSCOMP1" */

struct elpis_ecs_composite {
    atomic_int guard;                 /* 0 free, -1 one writer, n > 0 readers */
    _Atomic uint64_t lease;           /* 0: unmanaged */
    elpis_ecs_composite_config cfg;
    size_t k1_bytes, memory_capacity, envelope_bytes;
    elpis_ecsg_k1 *k1;
    elpis_ecs_mem *memory;            /* committed */
    elpis_ecs_mem *candidate;         /* the transaction's memory candidate (valid while candidate_ready) */
    uint8_t *staging;                 /* envelope_bytes: the commit's identity and the candidate copy */
    uint8_t identity[32], k1_identity[32], memory_identity[32];
    int poisoned;                     /* an internal invariant failed: every call refuses CORRUPT */
    /* The open transaction (one at most). */
    int txn_open, candidate_ready;
    uint64_t token, k1_token, next_token;
    /* Non-authoritative telemetry. */
    uint64_t commits, aborts, refusals, observes, learns, consolidations;
};

/* ---- little-endian fields -------------------------------------------------------------------------------- */

static void wr32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void wr64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint32_t rd32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static uint64_t rd64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static int is_zero(const uint8_t *p, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= p[i];
    return acc == 0;
}

/* ---- guard ------------------------------------------------------------------------------------------------ */

static int enter_write(elpis_ecs_composite *c) {
    int expected = 0;
    return atomic_compare_exchange_strong(&c->guard, &expected, -1);
}
static void leave_write(elpis_ecs_composite *c) { atomic_store(&c->guard, 0); }
static int enter_read(elpis_ecs_composite *c) {
    int v = atomic_load(&c->guard);
    while (v >= 0) {
        if (atomic_compare_exchange_weak(&c->guard, &v, v + 1)) return 1;
    }
    return 0;
}
static void leave_read(elpis_ecs_composite *c) { atomic_fetch_sub(&c->guard, 1); }

static elpis_ecs_composite_status refuse(elpis_ecs_composite *c, elpis_ecs_composite_status s) {
    if (s != ELPIS_ECS_COMPOSITE_OK) c->refusals++;
    return s;
}

static elpis_ecs_composite_status from_memory(elpis_ecs_mem_status s) {
    switch (s) {
    case ELPIS_ECS_MEM_OK: return ELPIS_ECS_COMPOSITE_OK;
    case ELPIS_ECS_MEM_INVALID: return ELPIS_ECS_COMPOSITE_INVALID;
    case ELPIS_ECS_MEM_CAPACITY: return ELPIS_ECS_COMPOSITE_CAPACITY;
    case ELPIS_ECS_MEM_CORRUPT: return ELPIS_ECS_COMPOSITE_CORRUPT;
    case ELPIS_ECS_MEM_NOMEM: return ELPIS_ECS_COMPOSITE_NOMEM;
    case ELPIS_ECS_MEM_UNQUALIFIED: return ELPIS_ECS_COMPOSITE_UNQUALIFIED;
    default: return ELPIS_ECS_COMPOSITE_REFUSED;
    }
}
/* K1 statuses share the composite numbers by construction. */
static elpis_ecs_composite_status from_k1(elpis_ecsg_k1_status s) { return (elpis_ecs_composite_status)s; }
/* A K1 restore refusal: the envelope is not a valid K1 component unless the refusal is a resource limit. */
static elpis_ecs_composite_status from_k1_restore(elpis_ecsg_k1_status s) {
    return s == ELPIS_ECSG_K1_NOMEM || s == ELPIS_ECSG_K1_CAPACITY ? from_k1(s) : ELPIS_ECS_COMPOSITE_CORRUPT;
}

const char *elpis_ecs_composite_code_name(int status) {
    switch (status) {
    case ELPIS_ECS_COMPOSITE_OK: return "ELPIS_ECS_COMPOSITE_OK";
    case ELPIS_ECS_COMPOSITE_INVALID: return "ELPIS_ECS_COMPOSITE_INVALID";
    case ELPIS_ECS_COMPOSITE_NONFINITE: return "ELPIS_ECS_COMPOSITE_NONFINITE";
    case ELPIS_ECS_COMPOSITE_STALE: return "ELPIS_ECS_COMPOSITE_STALE";
    case ELPIS_ECS_COMPOSITE_BUSY: return "ELPIS_ECS_COMPOSITE_BUSY";
    case ELPIS_ECS_COMPOSITE_CAPACITY: return "ELPIS_ECS_COMPOSITE_CAPACITY";
    case ELPIS_ECS_COMPOSITE_NOMEM: return "ELPIS_ECS_COMPOSITE_NOMEM";
    case ELPIS_ECS_COMPOSITE_CORRUPT: return "ELPIS_ECS_COMPOSITE_CORRUPT";
    case ELPIS_ECS_COMPOSITE_LEASED: return "ELPIS_ECS_COMPOSITE_LEASED";
    case ELPIS_ECS_COMPOSITE_UNQUALIFIED: return "ELPIS_ECS_COMPOSITE_UNQUALIFIED";
    case ELPIS_ECS_COMPOSITE_REFUSED: return "ELPIS_ECS_COMPOSITE_REFUSED";
    default: return "ELPIS_ECS_COMPOSITE_UNKNOWN";
    }
}

/* ---- configuration and sizes ------------------------------------------------------------------------------ */

typedef struct {
    size_t k1_bytes, memory_capacity, envelope_bytes;
} sizes;

/* 0: valid (sizes filled); INVALID or CAPACITY otherwise. */
static elpis_ecs_composite_status config_sizes(const elpis_ecs_composite_config *c, sizes *out) {
    if (!c || c->abi_version != ELPIS_ECS_COMPOSITE_ABI_V1 || c->reserved || !c->dim || !c->width || !c->max_rows)
        return ELPIS_ECS_COMPOSITE_INVALID;
    if (c->dim > ELPIS_ECSG_K1_MAX_DIM || (uint64_t)(size_t)c->width != c->width ||
        (uint64_t)(size_t)c->max_rows != c->max_rows)
        return ELPIS_ECS_COMPOSITE_INVALID;
    const size_t k1 = elpis_ecsg_k1_envelope_bytes((size_t)c->dim, (size_t)c->width);
    if (!k1) return ELPIS_ECS_COMPOSITE_CAPACITY; /* beyond the K1 image budget */
    uint64_t arena = 0, max_state = 0;
    const elpis_ecs_mem_status ms = elpis_ecs_mem_required_bytes(&c->memory, &arena, &max_state);
    if (ms != ELPIS_ECS_MEM_OK) return ms == ELPIS_ECS_MEM_CAPACITY ? ELPIS_ECS_COMPOSITE_CAPACITY
                                                                     : ELPIS_ECS_COMPOSITE_INVALID;
    if (max_state > ELPIS_ECS_COMPOSITE_MAX_MEMORY_BYTES) return ELPIS_ECS_COMPOSITE_CAPACITY;
    const uint64_t total = (uint64_t)ELPIS_ECS_COMPOSITE_HEADER_BYTES + k1 + max_state + ELPIS_ECS_COMPOSITE_DIGEST_BYTES;
    if (total > ELPIS_ECS_COMPOSITE_MAX_ENVELOPE_BYTES || total > SIZE_MAX) return ELPIS_ECS_COMPOSITE_CAPACITY;
    out->k1_bytes = k1;
    out->memory_capacity = (size_t)max_state;
    out->envelope_bytes = (size_t)total;
    return ELPIS_ECS_COMPOSITE_OK;
}

size_t elpis_ecs_composite_envelope_bytes(const elpis_ecs_composite_config *config) {
    sizes z;
    return config_sizes(config, &z) == ELPIS_ECS_COMPOSITE_OK ? z.envelope_bytes : 0;
}

/* ---- envelope ----------------------------------------------------------------------------------------------- */

/* Assemble the complete ELPISEC1 envelope of (K1 committed or candidate, memory) into out (envelope_bytes). */
static elpis_ecs_composite_status assemble(elpis_ecs_composite *c, int candidate, const elpis_ecs_mem *memory,
                                           uint8_t *out) {
    uint8_t *k1 = out + ELPIS_ECS_COMPOSITE_HEADER_BYTES;
    uint8_t *mem = k1 + c->k1_bytes;
    const elpis_ecsg_k1_status ks = candidate ? elpis_ecsg_k1_txn_snapshot_write(c->k1, c->k1_token, k1, c->k1_bytes)
                                              : elpis_ecsg_k1_snapshot_write(c->k1, k1, c->k1_bytes);
    if (ks != ELPIS_ECSG_K1_OK) return from_k1(ks);
    size_t written = 0;
    const elpis_ecs_mem_status ms = elpis_ecs_mem_snapshot(memory, mem, c->memory_capacity, &written);
    if (ms != ELPIS_ECS_MEM_OK) return from_memory(ms);
    memset(mem + written, 0, c->memory_capacity - written);
    memset(out, 0, ELPIS_ECS_COMPOSITE_HEADER_BYTES);
    memcpy(out, MAGIC, 8);
    wr32(out + 8, ELPIS_ECS_COMPOSITE_FORMAT_VERSION);
    wr64(out + 16, c->k1_bytes);
    wr64(out + 24, c->memory_capacity);
    wr64(out + 32, written);
    memcpy(out + 48, k1 + c->k1_bytes - 32, 32);
    memcpy(out + 80, mem + written - 32, 32);
    elpis_sha256(out, c->envelope_bytes - 32, out + c->envelope_bytes - 32);
    return ELPIS_ECS_COMPOSITE_OK;
}

typedef struct {
    size_t k1_bytes, memory_capacity, memory_bytes;
} layout;

/* Structural verification of an ELPISEC1 envelope: header, sizes, reserved fields, padding, recorded component
 * identities against the component trailers, and the composite trailer. Components are not validated here. */
static int verify_structure(const uint8_t *e, size_t length, layout *out) {
    if (!e || length < ELPIS_ECS_COMPOSITE_HEADER_BYTES + 2u * ELPIS_ECS_COMPOSITE_DIGEST_BYTES) return 0;
    if (memcmp(e, MAGIC, 8) || rd32(e + 8) != ELPIS_ECS_COMPOSITE_FORMAT_VERSION || rd32(e + 12) != 0) return 0;
    const uint64_t k1 = rd64(e + 16), cap = rd64(e + 24), mem = rd64(e + 32);
    if (rd64(e + 40) != 0 || !is_zero(e + 112, 16)) return 0;
    if (k1 < ELPIS_ECSG_K1_HEADER_BYTES + ELPIS_ECSG_K1_DIGEST_BYTES || k1 > length) return 0;
    if (cap > ELPIS_ECS_COMPOSITE_MAX_MEMORY_BYTES || mem > cap ||
        mem < (uint64_t)ELPIS_ECS_MEM_HEADER_BYTES + ELPIS_ECS_MEM_DIGEST_BYTES)
        return 0;
    if ((uint64_t)ELPIS_ECS_COMPOSITE_HEADER_BYTES + k1 + cap + ELPIS_ECS_COMPOSITE_DIGEST_BYTES != (uint64_t)length)
        return 0;
    const uint8_t *k1p = e + ELPIS_ECS_COMPOSITE_HEADER_BYTES, *memp = k1p + k1;
    if (memcmp(e + 48, k1p + k1 - 32, 32) || memcmp(e + 80, memp + mem - 32, 32)) return 0;
    if (!is_zero(memp + mem, (size_t)(cap - mem))) return 0;
    uint8_t digest[32];
    elpis_sha256(e, length - 32, digest);
    if (memcmp(digest, e + length - 32, 32)) return 0;
    out->k1_bytes = (size_t)k1;
    out->memory_capacity = (size_t)cap;
    out->memory_bytes = (size_t)mem;
    return 1;
}

elpis_ecs_composite_status elpis_ecs_composite_envelope_identities(const uint8_t *envelope, size_t length,
                                                                   uint8_t composite[32], uint8_t k1[32],
                                                                   uint8_t memory[32]) {
    layout l;
    if (!envelope) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!verify_structure(envelope, length, &l)) return ELPIS_ECS_COMPOSITE_CORRUPT;
    if (composite) memcpy(composite, envelope + length - 32, 32);
    if (k1) memcpy(k1, envelope + 48, 32);
    if (memory) memcpy(memory, envelope + 80, 32);
    return ELPIS_ECS_COMPOSITE_OK;
}

/* ---- lifecycle ----------------------------------------------------------------------------------------------- */

/* Allocate the object, both memory handles and the staging buffer around an already-built K1 state (adopted; it
 * is destroyed on failure). */
static elpis_ecs_composite_status build(const elpis_ecs_composite_config *config, const sizes *z, elpis_ecsg_k1 *k1,
                                        elpis_ecs_composite **out) {
    elpis_ecs_composite *c = calloc(1, sizeof(*c));
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_NOMEM;
    if (!c) goto fail_k1;
    c->cfg = *config;
    c->k1_bytes = z->k1_bytes;
    c->memory_capacity = z->memory_capacity;
    c->envelope_bytes = z->envelope_bytes;
    c->k1 = k1;
    c->next_token = 1;
    atomic_init(&c->guard, 0);
    atomic_init(&c->lease, 0);
    elpis_ecs_mem_status ms = elpis_ecs_mem_create(&config->memory, &c->memory);
    if (ms == ELPIS_ECS_MEM_OK) ms = elpis_ecs_mem_create(&config->memory, &c->candidate);
    if (ms != ELPIS_ECS_MEM_OK) {
        s = from_memory(ms);
        goto fail;
    }
    c->staging = malloc(c->envelope_bytes);
    if (!c->staging) goto fail;
    if (elpis_ecsg_k1_lease_claim(k1, K1_LEASE) != ELPIS_ECSG_K1_OK) {
        s = ELPIS_ECS_COMPOSITE_INVALID;
        goto fail;
    }
    *out = c;
    return ELPIS_ECS_COMPOSITE_OK;
fail:
    elpis_ecs_mem_destroy(c->memory);
    elpis_ecs_mem_destroy(c->candidate);
    free(c->staging);
    free(c);
fail_k1:
    elpis_ecsg_k1_destroy(&k1);
    return s;
}

/* Compute and cache the committed identities (after create, restore, import). */
static elpis_ecs_composite_status settle(elpis_ecs_composite *c) {
    const elpis_ecs_composite_status s = assemble(c, 0, c->memory, c->staging);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    memcpy(c->identity, c->staging + c->envelope_bytes - 32, 32);
    memcpy(c->k1_identity, c->staging + 48, 32);
    memcpy(c->memory_identity, c->staging + 80, 32);
    return ELPIS_ECS_COMPOSITE_OK;
}

static int k1_shape_is(elpis_ecsg_k1 *k1, const elpis_ecs_composite_config *config) {
    size_t dim = 0, width = 0;
    return elpis_ecsg_k1_shape(k1, &dim, &width) == ELPIS_ECSG_K1_OK && dim == config->dim && width == config->width;
}

elpis_ecs_composite_status elpis_ecs_composite_create(const elpis_ecs_composite_config *config, const double *initial_w,
                                                      elpis_ecs_composite **out) {
    sizes z;
    if (!out || !initial_w) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = config_sizes(config, &z);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    elpis_ecsg_k1 *k1 = NULL;
    const elpis_ecsg_k1_status ks =
        elpis_ecsg_k1_create((size_t)config->dim, (size_t)config->width, (size_t)config->max_rows, initial_w, &k1);
    if (ks != ELPIS_ECSG_K1_OK) return from_k1(ks);
    elpis_ecs_composite *c = NULL;
    if ((s = build(config, &z, k1, &c)) != ELPIS_ECS_COMPOSITE_OK) return s;
    if ((s = settle(c)) != ELPIS_ECS_COMPOSITE_OK) {
        elpis_ecs_composite_destroy(c);
        return s;
    }
    *out = c;
    return ELPIS_ECS_COMPOSITE_OK;
}

elpis_ecs_composite_status elpis_ecs_composite_restore(const elpis_ecs_composite_config *config, const uint8_t *envelope,
                                                       size_t length, elpis_ecs_composite **out) {
    sizes z;
    layout l;
    if (!out || !envelope) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = config_sizes(config, &z);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (length != z.envelope_bytes || !verify_structure(envelope, length, &l) || l.k1_bytes != z.k1_bytes ||
        l.memory_capacity != z.memory_capacity)
        return ELPIS_ECS_COMPOSITE_CORRUPT;
    elpis_ecsg_k1 *k1 = NULL;
    const elpis_ecsg_k1_status ks = elpis_ecsg_k1_restore(envelope + ELPIS_ECS_COMPOSITE_HEADER_BYTES, l.k1_bytes,
                                                          (size_t)config->max_rows, &k1);
    if (ks != ELPIS_ECSG_K1_OK) return from_k1_restore(ks);
    if (!k1_shape_is(k1, config)) {
        elpis_ecsg_k1_destroy(&k1);
        return ELPIS_ECS_COMPOSITE_CORRUPT;
    }
    elpis_ecs_composite *c = NULL;
    if ((s = build(config, &z, k1, &c)) != ELPIS_ECS_COMPOSITE_OK) return s;
    const elpis_ecs_mem_status ms = elpis_ecs_mem_restore_into(
        c->memory, 0, envelope + ELPIS_ECS_COMPOSITE_HEADER_BYTES + l.k1_bytes, l.memory_bytes);
    if (ms != ELPIS_ECS_MEM_OK) {
        elpis_ecs_composite_destroy(c);
        return ms == ELPIS_ECS_MEM_UNQUALIFIED ? ELPIS_ECS_COMPOSITE_UNQUALIFIED : ELPIS_ECS_COMPOSITE_CORRUPT;
    }
    if ((s = settle(c)) != ELPIS_ECS_COMPOSITE_OK) {
        elpis_ecs_composite_destroy(c);
        return s;
    }
    /* Exact restore: the state re-serializes to the very bytes it came from. */
    if (memcmp(c->staging, envelope, length)) {
        elpis_ecs_composite_destroy(c);
        return ELPIS_ECS_COMPOSITE_CORRUPT;
    }
    *out = c;
    return ELPIS_ECS_COMPOSITE_OK;
}

elpis_ecs_composite_status elpis_ecs_composite_import_k1(const elpis_ecs_composite_config *config,
                                                         const uint8_t *k1_envelope, size_t length,
                                                         elpis_ecs_composite **out) {
    sizes z;
    if (!out || !k1_envelope) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = config_sizes(config, &z);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (length != z.k1_bytes) return ELPIS_ECS_COMPOSITE_CORRUPT;
    elpis_ecsg_k1 *k1 = NULL;
    const elpis_ecsg_k1_status ks = elpis_ecsg_k1_restore(k1_envelope, length, (size_t)config->max_rows, &k1);
    if (ks != ELPIS_ECSG_K1_OK) return from_k1_restore(ks);
    if (!k1_shape_is(k1, config)) {
        elpis_ecsg_k1_destroy(&k1);
        return ELPIS_ECS_COMPOSITE_CORRUPT;
    }
    elpis_ecs_composite *c = NULL;
    if ((s = build(config, &z, k1, &c)) != ELPIS_ECS_COMPOSITE_OK) return s;
    if ((s = settle(c)) != ELPIS_ECS_COMPOSITE_OK) {
        elpis_ecs_composite_destroy(c);
        return s;
    }
    *out = c;
    return ELPIS_ECS_COMPOSITE_OK;
}

void elpis_ecs_composite_destroy(elpis_ecs_composite *c) {
    if (!c) return;
    if (c->txn_open) (void)elpis_ecsg_k1_txn_abort(c->k1, c->k1_token);
    elpis_ecsg_k1_destroy(&c->k1);
    elpis_ecs_mem_destroy(c->memory);
    elpis_ecs_mem_destroy(c->candidate);
    free(c->staging);
    free(c);
}

/* ---- reads --------------------------------------------------------------------------------------------------- */

elpis_ecs_composite_status elpis_ecs_composite_state_digest(elpis_ecs_composite *c, uint8_t out[32]) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = c->poisoned ? ELPIS_ECS_COMPOSITE_CORRUPT : ELPIS_ECS_COMPOSITE_OK;
    if (s == ELPIS_ECS_COMPOSITE_OK) memcpy(out, c->identity, 32);
    leave_read(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_shape(elpis_ecs_composite *c, size_t *dim, size_t *width) {
    if (!c || !dim || !width) return ELPIS_ECS_COMPOSITE_INVALID;
    *dim = (size_t)c->cfg.dim; /* immutable */
    *width = (size_t)c->cfg.width;
    return ELPIS_ECS_COMPOSITE_OK;
}

elpis_ecs_composite_status elpis_ecs_composite_query_identity(elpis_ecs_composite *c, size_t dim, const double *x,
                                                              size_t rows, double *out, uint8_t digest[32]) {
    if (!c || !digest) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_CORRUPT;
    if (!c->poisoned) {
        uint8_t k1[32];
        s = from_k1(elpis_ecsg_k1_query_identity(c->k1, dim, x, rows, out, k1));
        if (s == ELPIS_ECS_COMPOSITE_OK) {
            if (memcmp(k1, c->k1_identity, 32)) s = ELPIS_ECS_COMPOSITE_CORRUPT; /* K1 is private: never */
            else memcpy(digest, c->identity, 32);
        }
    }
    leave_read(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_memory_query(elpis_ecs_composite *c, const elpis_ecs_mem_request *request,
                                                            elpis_ecs_mem_answer *answer, uint8_t digest[32],
                                                            int *memory_status) {
    if (!c || !digest) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_CORRUPT;
    if (!c->poisoned) {
        const elpis_ecs_mem_status ms = elpis_ecs_mem_query(c->memory, request, answer);
        if (memory_status) *memory_status = ms;
        s = from_memory(ms);
        if (s == ELPIS_ECS_COMPOSITE_OK) memcpy(digest, c->identity, 32);
    }
    leave_read(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_memory_slot_read(elpis_ecs_composite *c, const elpis_ecs_mem_digest *key,
                                                                elpis_ecs_mem_slot *out, int *memory_status) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_CORRUPT;
    if (!c->poisoned) {
        const elpis_ecs_mem_status ms = elpis_ecs_mem_slot_read(c->memory, key, out);
        if (memory_status) *memory_status = ms;
        s = from_memory(ms);
    }
    leave_read(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_snapshot_write(elpis_ecs_composite *c, uint8_t *out, size_t size) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    if (size < c->envelope_bytes) return ELPIS_ECS_COMPOSITE_CAPACITY;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_CORRUPT;
    if (!c->poisoned) {
        s = assemble(c, 0, c->memory, out);
        if (s == ELPIS_ECS_COMPOSITE_OK && memcmp(out + c->envelope_bytes - 32, c->identity, 32))
            s = ELPIS_ECS_COMPOSITE_CORRUPT;
    }
    leave_read(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_inspect(elpis_ecs_composite *c, elpis_ecs_composite_stats *out) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_read(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecsg_k1_counters kc;
    elpis_ecs_mem_stats ms;
    elpis_ecs_composite_status s = from_k1(elpis_ecsg_k1_stats(c->k1, &kc));
    if (s == ELPIS_ECS_COMPOSITE_OK) s = from_memory(elpis_ecs_mem_inspect(c->memory, &ms));
    if (s == ELPIS_ECS_COMPOSITE_OK) {
        memset(out, 0, sizeof(*out));
        out->envelope_bytes = c->envelope_bytes;
        out->k1_envelope_bytes = c->k1_bytes;
        out->memory_capacity_bytes = c->memory_capacity;
        out->memory_state_bytes = ms.state_bytes;
        out->k1_workspace_bytes = kc.workspace_bytes;
        out->memory_arena_bytes = ms.arena_bytes;
        out->staging_bytes = c->envelope_bytes;
        out->allocated_bytes = sizeof(*c) + kc.workspace_bytes + 2u * ms.allocated_bytes + c->envelope_bytes;
        out->k1_epoch = elpis_ecsg_k1_epoch(c->k1);
        out->memory_clock = ms.clock;
        out->memory_slots_used = ms.slots_used;
        out->memory_codebook_used = ms.codebook_used;
        out->capabilities = c->cfg.memory.capabilities;
        out->txn_open = (uint32_t)c->txn_open;
        out->lease_held = atomic_load(&c->lease) != 0;
        out->commits = c->commits;
        out->aborts = c->aborts;
        out->refusals = c->refusals;
        out->memory_observes = c->observes;
        out->memory_learns = c->learns;
        out->memory_consolidations = c->consolidations;
    }
    leave_read(c);
    return s;
}

/* ---- managed ownership --------------------------------------------------------------------------------------- */

elpis_ecs_composite_status elpis_ecs_composite_lease_claim(elpis_ecs_composite *c, uint64_t lease) {
    if (!c || !lease) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_write(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_BUSY;
    if (!c->txn_open) {
        atomic_store(&c->lease, lease);
        s = ELPIS_ECS_COMPOSITE_OK;
    }
    s = refuse(c, s);
    leave_write(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_lease_release(elpis_ecs_composite *c, uint64_t lease) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_write(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s;
    if (!lease || atomic_load(&c->lease) != lease) s = ELPIS_ECS_COMPOSITE_LEASED;
    else if (c->txn_open) s = ELPIS_ECS_COMPOSITE_BUSY;
    else {
        atomic_store(&c->lease, 0);
        s = ELPIS_ECS_COMPOSITE_OK;
    }
    s = refuse(c, s);
    leave_write(c);
    return s;
}

uint64_t elpis_ecs_composite_lease_of(const elpis_ecs_composite *c) {
    return c ? atomic_load(&((elpis_ecs_composite *)c)->lease) : 0;
}

/* Writer entry for a mutation: guard, poison and lease checks. Returns OK with the guard held. */
static elpis_ecs_composite_status enter_mutation(elpis_ecs_composite *c, uint64_t lease) {
    if (!enter_write(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_OK;
    if (c->poisoned) s = ELPIS_ECS_COMPOSITE_CORRUPT;
    else if (atomic_load(&c->lease) != lease) s = ELPIS_ECS_COMPOSITE_LEASED;
    if (s != ELPIS_ECS_COMPOSITE_OK) {
        refuse(c, s);
        leave_write(c);
    }
    return s;
}
/* As enter_mutation, and the open transaction must be `token`. */
static elpis_ecs_composite_status enter_txn(elpis_ecs_composite *c, uint64_t lease, uint64_t token) {
    elpis_ecs_composite_status s = enter_mutation(c, lease);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (!c->txn_open || token != c->token) {
        refuse(c, ELPIS_ECS_COMPOSITE_INVALID);
        leave_write(c);
        return ELPIS_ECS_COMPOSITE_INVALID;
    }
    return ELPIS_ECS_COMPOSITE_OK;
}
static elpis_ecs_composite_status leave_with(elpis_ecs_composite *c, elpis_ecs_composite_status s) {
    s = refuse(c, s);
    leave_write(c);
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_leased_reserve(elpis_ecs_composite *c, uint64_t lease, size_t max_rows) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_mutation(c, lease);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (c->txn_open) return leave_with(c, ELPIS_ECS_COMPOSITE_BUSY);
    s = from_k1(elpis_ecsg_k1_leased_reserve(c->k1, K1_LEASE, max_rows));
    if (s == ELPIS_ECS_COMPOSITE_OK) c->cfg.max_rows = max_rows;
    return leave_with(c, s);
}

/* ---- transactions -------------------------------------------------------------------------------------------- */

static void close_txn(elpis_ecs_composite *c) {
    c->txn_open = 0;
    c->candidate_ready = 0;
    c->token = 0;
    c->k1_token = 0;
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_begin(elpis_ecs_composite *c, uint64_t lease,
                                                                uint8_t source_digest[32], uint64_t *token) {
    if (!c || !source_digest || !token) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_mutation(c, lease);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (c->txn_open) return leave_with(c, ELPIS_ECS_COMPOSITE_BUSY);
    uint8_t k1_source[32];
    uint64_t k1_token = 0;
    s = from_k1(elpis_ecsg_k1_leased_txn_begin(c->k1, K1_LEASE, k1_source, &k1_token));
    if (s != ELPIS_ECS_COMPOSITE_OK) return leave_with(c, s);
    if (memcmp(k1_source, c->k1_identity, 32)) { /* K1 is private: its state cannot have moved */
        (void)elpis_ecsg_k1_txn_abort(c->k1, k1_token);
        c->poisoned = 1;
        return leave_with(c, ELPIS_ECS_COMPOSITE_CORRUPT);
    }
    c->txn_open = 1;
    c->candidate_ready = 0;
    c->k1_token = k1_token;
    c->token = c->next_token++;
    if (!c->next_token) c->next_token = 1;
    memcpy(source_digest, c->identity, 32);
    *token = c->token;
    return leave_with(c, ELPIS_ECS_COMPOSITE_OK);
}

elpis_ecs_composite_status
elpis_ecs_composite_leased_txn_run_schedule(elpis_ecs_composite *c, uint64_t lease, uint64_t token, const double *x,
                                            const double *y, size_t total_rows, const elpis_ecsg_k1_experience *schedule,
                                            size_t experiences, double learning_rate, double *s3_out, size_t s3_count,
                                            elpis_ecsg_k1_schedule_result *result) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_txn(c, lease, token);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    s = from_k1(elpis_ecsg_k1_leased_txn_run_schedule(c->k1, K1_LEASE, c->k1_token, x, y, total_rows, schedule,
                                                      experiences, learning_rate, s3_out, s3_count, result));
    if (s == ELPIS_ECS_COMPOSITE_STALE || s == ELPIS_ECS_COMPOSITE_NONFINITE) close_txn(c); /* K1 discarded it */
    return leave_with(c, s);
}

/* The memory candidate: a copy of the committed memory, made once per transaction. */
static elpis_ecs_composite_status ready_candidate(elpis_ecs_composite *c) {
    if (c->candidate_ready) return ELPIS_ECS_COMPOSITE_OK;
    size_t written = 0;
    elpis_ecs_mem_status ms = elpis_ecs_mem_snapshot(c->memory, c->staging, c->envelope_bytes, &written);
    if (ms == ELPIS_ECS_MEM_OK) ms = elpis_ecs_mem_restore_into(c->candidate, 0, c->staging, written);
    if (ms != ELPIS_ECS_MEM_OK) return from_memory(ms);
    c->candidate_ready = 1;
    return ELPIS_ECS_COMPOSITE_OK;
}

static elpis_ecs_composite_status memory_result(elpis_ecs_mem_status ms, int *memory_status) {
    if (memory_status) *memory_status = ms;
    return from_memory(ms);
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_observe(elpis_ecs_composite *c, uint64_t lease, uint64_t token,
                                                                  const elpis_ecs_mem_observation *observation,
                                                                  elpis_ecs_mem_digest *out_key, int *memory_status) {
    if (!c || !observation) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_txn(c, lease, token);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if ((s = ready_candidate(c)) != ELPIS_ECS_COMPOSITE_OK) return leave_with(c, s);
    s = memory_result(elpis_ecs_mem_observe(c->candidate, 0, observation, out_key, NULL), memory_status);
    if (s == ELPIS_ECS_COMPOSITE_OK) c->observes++;
    return leave_with(c, s);
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_memory_learn(elpis_ecs_composite *c, uint64_t lease,
                                                                       uint64_t token, const elpis_ecs_mem_digest *key,
                                                                       uint64_t tick, int *memory_status) {
    if (!c || !key) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_txn(c, lease, token);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (!(c->cfg.memory.capabilities & ELPIS_ECS_MEM_CAP_RECURRENT)) {
        if (memory_status) *memory_status = ELPIS_ECS_MEM_UNQUALIFIED;
        return leave_with(c, ELPIS_ECS_COMPOSITE_UNQUALIFIED);
    }
    if ((s = ready_candidate(c)) != ELPIS_ECS_COMPOSITE_OK) return leave_with(c, s);
    s = memory_result(elpis_ecs_mem_learn(c->candidate, 0, key, tick, NULL), memory_status);
    if (s == ELPIS_ECS_COMPOSITE_OK) c->learns++;
    return leave_with(c, s);
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_memory_consolidate(elpis_ecs_composite *c, uint64_t lease,
                                                                             uint64_t token, uint64_t tick,
                                                                             uint32_t max_items, uint32_t *replayed,
                                                                             int *memory_status) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_status s = enter_txn(c, lease, token);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    if (!(c->cfg.memory.capabilities & ELPIS_ECS_MEM_CAP_CONSOLIDATE)) {
        if (memory_status) *memory_status = ELPIS_ECS_MEM_UNQUALIFIED;
        return leave_with(c, ELPIS_ECS_COMPOSITE_UNQUALIFIED);
    }
    if ((s = ready_candidate(c)) != ELPIS_ECS_COMPOSITE_OK) return leave_with(c, s);
    s = memory_result(elpis_ecs_mem_consolidate(c->candidate, 0, tick, max_items, replayed, NULL), memory_status);
    if (s == ELPIS_ECS_COMPOSITE_OK) c->consolidations++;
    return leave_with(c, s);
}

elpis_ecs_composite_status elpis_ecs_composite_txn_snapshot_write(elpis_ecs_composite *c, uint64_t token, uint8_t *out,
                                                                  size_t size) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    if (size < c->envelope_bytes) return ELPIS_ECS_COMPOSITE_CAPACITY;
    if (!enter_write(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_INVALID;
    if (c->poisoned) s = ELPIS_ECS_COMPOSITE_CORRUPT;
    else if (c->txn_open && token == c->token) {
        s = assemble(c, 1, c->candidate_ready ? c->candidate : c->memory, out);
        if (s == ELPIS_ECS_COMPOSITE_STALE) close_txn(c);
    }
    return leave_with(c, s);
}

static elpis_ecs_composite_status commit(elpis_ecs_composite *c, uint64_t lease, uint64_t token,
                                         elpis_ecs_composite_commit *out) {
    elpis_ecs_composite_status s = enter_txn(c, lease, token);
    if (s != ELPIS_ECS_COMPOSITE_OK) return s;
    const elpis_ecs_mem *memory = c->candidate_ready ? c->candidate : c->memory;
    s = assemble(c, 1, memory, c->staging); /* the candidate's identity, before anything commits */
    if (s != ELPIS_ECS_COMPOSITE_OK) {
        if (s == ELPIS_ECS_COMPOSITE_STALE) close_txn(c);
        return leave_with(c, s);
    }
    elpis_ecsg_k1_commit_identity k1;
    s = from_k1(elpis_ecsg_k1_leased_txn_commit_identity(c->k1, K1_LEASE, c->k1_token, &k1));
    if (s != ELPIS_ECS_COMPOSITE_OK) {
        if (s == ELPIS_ECS_COMPOSITE_STALE) close_txn(c); /* K1 discarded it; other refusals leave it open */
        return leave_with(c, s);
    }
    /* Committed: from here nothing may fail. The memory install is a pointer exchange. */
    memset(out, 0, sizeof(*out));
    out->k1 = k1.transition;
    memcpy(out->before, c->identity, 32);
    memcpy(out->k1_before, c->k1_identity, 32);
    memcpy(out->memory_before, c->memory_identity, 32);
    if (c->candidate_ready) {
        elpis_ecs_mem *previous = c->memory;
        c->memory = c->candidate;
        c->candidate = previous;
    }
    out->memory_changed = (uint32_t)(memcmp(c->staging + 80, c->memory_identity, 32) != 0);
    memcpy(c->identity, c->staging + c->envelope_bytes - 32, 32);
    memcpy(c->k1_identity, c->staging + 48, 32);
    memcpy(c->memory_identity, c->staging + 80, 32);
    memcpy(out->after, c->identity, 32);
    memcpy(out->k1_after, c->k1_identity, 32);
    memcpy(out->memory_after, c->memory_identity, 32);
    /* K1's own report must name exactly the K1 state the identity binds (K1 is private: never otherwise). */
    if (memcmp(k1.state_after_digest, c->k1_identity, 32) || memcmp(k1.state_before_digest, out->k1_before, 32))
        c->poisoned = 1;
    close_txn(c);
    c->commits++;
    return leave_with(c, c->poisoned ? ELPIS_ECS_COMPOSITE_CORRUPT : ELPIS_ECS_COMPOSITE_OK);
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_commit(elpis_ecs_composite *c, uint64_t lease, uint64_t token,
                                                                 elpis_ecs_composite_commit *out) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    return commit(c, lease, token, out);
}

elpis_ecs_composite_status elpis_ecs_composite_leased_txn_commit_identity(elpis_ecs_composite *c, uint64_t lease,
                                                                          uint64_t token,
                                                                          elpis_ecsg_k1_commit_identity *out) {
    if (!c || !out) return ELPIS_ECS_COMPOSITE_INVALID;
    elpis_ecs_composite_commit full;
    const elpis_ecs_composite_status s = commit(c, lease, token, &full);
    if (s == ELPIS_ECS_COMPOSITE_OK) {
        out->transition = full.k1;
        memcpy(out->state_before_digest, full.before, 32);
        memcpy(out->state_after_digest, full.after, 32);
    }
    return s;
}

elpis_ecs_composite_status elpis_ecs_composite_txn_abort(elpis_ecs_composite *c, uint64_t token) {
    if (!c) return ELPIS_ECS_COMPOSITE_INVALID;
    if (!enter_write(c)) return ELPIS_ECS_COMPOSITE_BUSY;
    elpis_ecs_composite_status s = ELPIS_ECS_COMPOSITE_OK;
    if (c->txn_open) {
        if (token != c->token) s = ELPIS_ECS_COMPOSITE_INVALID;
        else {
            (void)elpis_ecsg_k1_txn_abort(c->k1, c->k1_token);
            close_txn(c);
            c->aborts++;
        }
    }
    return leave_with(c, s);
}
