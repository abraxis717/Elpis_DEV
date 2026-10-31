/* The native host's installation (host_install.h; docs/NATIVE_HOST.md). It opens files read-only and writes none: no
 * function here creates, writes, renames, truncates or removes anything. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "host_install.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "elpis/sha256.h"
#include "host_json.h"

#define CATALOG_SCHEMA "elpis.native-host-catalog.v2"
#define W_DOMAIN "elpis.deployment.initial-w.v1"
#define LABEL_DOMAIN "elpis.host.label.v1"

int host_fail(host_err *e, const char *code, const char *fmt, ...) {
    if (e) {
        snprintf(e->code, sizeof e->code, "%s", code);
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(e->detail, sizeof e->detail, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ---- the library set ------------------------------------------------------------------------------------------ */

#define LIB(id, dir) {#id, "lib" #id ".so", "lib" #id ".so", dir "/lib" #id ".so"}
static const host_library LIBRARIES[HOST_LIBRARIES] = {
    LIB(elpis_ecsg_math, "native/ECS_G"),
    LIB(elpis_ecsg_k1, "native/ECS_G"),
    LIB(elpis_ecs_memory, "native/ECS_C"),
    LIB(elpis_ecs_composite, "native/runtime/composite"),
    LIB(elpis_hacf_observation, "native/runtime"),
    LIB(elpis_runtime, "native/runtime"),
};
#undef LIB

const host_library *host_libraries(void) { return LIBRARIES; }

/* ---- small helpers -------------------------------------------------------------------------------------------- */

static int lower_hex(const char *text, size_t n) {
    if (!text || strlen(text) != n) return 0;
    for (size_t i = 0; i < n; ++i)
        if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f'))) return 0;
    return 1;
}

int host_unhex32(const char *text, uint8_t out[32]) {
    if (!lower_hex(text, 64)) return -1; /* lowercase only: one spelling per digest */
    for (int i = 0; i < 32; ++i) {
        const char h = text[2 * i], l = text[2 * i + 1];
        out[i] = (uint8_t)(((h <= '9' ? h - '0' : h - 'a' + 10) << 4) | (l <= '9' ? l - '0' : l - 'a' + 10));
    }
    return 0;
}

void host_label(const char *text, uint8_t out[32]) {
    elpis_sha256_ctx h;
    elpis_sha256_init(&h);
    elpis_sha256_update(&h, LABEL_DOMAIN, sizeof LABEL_DOMAIN); /* the domain and its NUL */
    elpis_sha256_update(&h, text, strlen(text));
    elpis_sha256_final(&h, out);
}

/* ---- measurement, the catalog and loading ---------------------------------------------------------------------- */

static void write_catalog(hjw *w, const host_install *d) {
    hjw_object(w);
    hjw_kstr(w, "schema", CATALOG_SCHEMA);
    hjw_khex(w, "host_sha256", d->host_sha256, 32);
    hjw_key(w, "abi");
    hjw_object(w);
    hjw_ku64(w, "runtime", ELPIS_RUNTIME_ABI_V8);
    hjw_ku64(w, "composite", ELPIS_ECS_COMPOSITE_ABI_V1);
    hjw_end(w, '}');
    hjw_key(w, "libraries");
    hjw_array(w);
    for (int i = 0; i < HOST_LIBRARIES; ++i) {
        hjw_object(w);
        hjw_kstr(w, "id", d->pins[i].id);
        hjw_kstr(w, "file", d->pins[i].file);
        hjw_kstr(w, "soname", d->pins[i].soname);
        hjw_ku64(w, "size", d->pins[i].size);
        hjw_khex(w, "sha256", d->pins[i].sha256, 32);
        hjw_end(w, '}');
    }
    hjw_end(w, ']');
    hjw_end(w, '}');
}

int host_measure(host_install *d, char *catalog, size_t catalog_cap, size_t *catalog_len, host_err *e) {
    for (int i = 0; i < HOST_LIBRARIES; ++i) d->sealed[i] = -1;
    if (!d->source.dir || d->source.dir[0] != '/')
        return host_fail(e, "INSTALL_PATH", "the library directory must be an absolute path");
    const int dirfd = open(d->source.dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirfd < 0) return host_fail(e, "INSTALL_PATH", "%s: %s", d->source.dir, strerror(errno));
    char err[512];
    int rc = host_digest_self(d->host_sha256, err, sizeof err) ? host_fail(e, "INSTALL_PIN", "%s", err) : 0;
    for (int i = 0; !rc && i < HOST_LIBRARIES; ++i) {
        const host_library *l = &LIBRARIES[i];
        d->pins[i] = (host_pin){l->id, l->file, l->soname, 0, {0}};
        if (host_seal(dirfd, d->source.build_tree ? l->build_path : l->file, &d->pins[i], &d->sealed[i], err,
                      sizeof err))
            rc = host_fail(e, "INSTALL_PIN", "native library admission refused: %s", err);
    }
    close(dirfd);
    if (rc) return rc;
    static char text[1 << 14];
    hjw w;
    hjw_init(&w, text, sizeof text);
    write_catalog(&w, d);
    if (!hjw_ok(&w)) return host_fail(e, "INSTALL_PIN", "catalog too large");
    elpis_sha256(text, w.len, d->catalog_sha256);
    if (catalog) {
        if (w.len + 1 > catalog_cap) return host_fail(e, "INSTALL_PIN", "catalog too large");
        memcpy(catalog, text, w.len);
        catalog[w.len] = 0;
        *catalog_len = w.len;
    }
    return 0;
}

static const host_api *ACTIVE; /* the installation whose functions the substrate table calls (one per process) */

int host_load_libraries(host_install *d, host_err *e) {
    char err[512];
    if (host_load(d->pins, d->sealed, HOST_LIBRARIES, d->handles, err, sizeof err))
        return host_fail(e, "INSTALL_PIN", "native library admission refused: %s", err);
    host_api *a = &d->api;
#define RESOLVE(lib, name)                                                                                         \
    if (!(a->name = (__typeof__(name) *)host_symbol(d->handles[lib], #name)))                                      \
        return host_fail(e, "INSTALL_PIN", "%s does not export %s", d->pins[lib].file, #name);
#define R_RUNTIME(name) RESOLVE(LIB_RUNTIME, name)
#define R_COMPOSITE(name) RESOLVE(LIB_COMPOSITE, name)
#define R_MEMORY(name) RESOLVE(LIB_MEMORY, name)
#define R_HACF(name) RESOLVE(LIB_HACF, name)
    HOST_RUNTIME_FUNCTIONS(R_RUNTIME)
    HOST_COMPOSITE_FUNCTIONS(R_COMPOSITE)
    HOST_MEMORY_FUNCTIONS(R_MEMORY)
    HOST_HACF_FUNCTIONS(R_HACF)
#undef RESOLVE
    if (a->elpis_runtime_abi_version() != ELPIS_RUNTIME_ABI_V8)
        return host_fail(e, "INSTALL_ABI", "RuntimeCore ABI %u is not this host's (8)", a->elpis_runtime_abi_version());
    ACTIVE = a;
    return 0;
}

/* ---- the composite's function table (typed wrappers over the admitted entries) --------------------------------- */

static int c_digest(void *s, uint8_t out[32]) { return ACTIVE->elpis_ecs_composite_state_digest(s, out); }
static int c_shape(void *s, size_t *dim, size_t *width) { return ACTIVE->elpis_ecs_composite_shape(s, dim, width); }
static int c_claim(void *s, uint64_t lease) { return ACTIVE->elpis_ecs_composite_lease_claim(s, lease); }
static int c_release(void *s, uint64_t lease) { return ACTIVE->elpis_ecs_composite_lease_release(s, lease); }
static int c_reserve(void *s, uint64_t lease, size_t rows) {
    return ACTIVE->elpis_ecs_composite_leased_reserve(s, lease, rows);
}
static int c_identity(void *s, uint64_t lease, uint8_t digest[32], uint64_t *generation) {
    return ACTIVE->elpis_ecs_composite_leased_identity(s, lease, digest, generation);
}
static int c_query(void *s, uint64_t lease, size_t dim, const double *x, size_t rows, double *out, uint64_t *g) {
    return ACTIVE->elpis_ecs_composite_leased_query(s, lease, dim, x, rows, out, g);
}
static int c_begin(void *s, uint64_t lease, uint64_t *source, uint64_t *token) {
    return ACTIVE->elpis_ecs_composite_leased_txn_begin_at(s, lease, source, token);
}
static int c_schedule(void *s, uint64_t lease, uint64_t token, const double *x, const double *y, size_t rows,
                      const elpis_runtime_experience *schedule, size_t n, double rate, double *s3, size_t s3n,
                      elpis_runtime_schedule_result *result) {
    /* Layout-identical types (runtime.h): the experience, schedule result and transition records. */
    return ACTIVE->elpis_ecs_composite_leased_txn_run_schedule(s, lease, token, x, y, rows,
                                                                (const elpis_ecsg_k1_experience *)schedule, n, rate,
                                                                s3, s3n, (elpis_ecsg_k1_schedule_result *)result);
}
static int c_commit(void *s, uint64_t lease, uint64_t token, elpis_runtime_transition *t) {
    return ACTIVE->elpis_ecs_composite_leased_txn_commit_at(s, lease, token, (elpis_ecsg_k1_transition *)t);
}
static int c_abort(void *s, uint64_t token) { return ACTIVE->elpis_ecs_composite_txn_abort(s, token); }
static int c_observe(void *s, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32], int *ms) {
    return ACTIVE->elpis_ecs_composite_leased_txn_observe(s, lease, token, observation, (elpis_ecs_mem_digest *)key,
                                                          ms);
}
static const elpis_runtime_composite_api SUBSTRATE_API = {
    {c_digest, c_shape, c_claim, c_release, c_reserve, c_identity, c_query, c_begin, c_schedule, c_commit, c_abort},
    c_observe};

const elpis_runtime_composite_api *host_substrate_api(void) { return &SUBSTRATE_API; }

/* ---- the lineage ---------------------------------------------------------------------------------------------- */

elpis_ecs_composite_config host_composite_config(const host_install *d) {
    const host_lineage *l = &d->lineage;
    elpis_ecs_composite_config c;
    memset(&c, 0, sizeof c);
    c.abi_version = ELPIS_ECS_COMPOSITE_ABI_V1;
    c.dim = l->dim, c.width = l->width, c.max_rows = l->max_rows;
    d->api.elpis_ecs_mem_config_default(&c.memory);
    c.memory.abi_version = ELPIS_ECS_MEM_ABI_V1;
    c.memory.dim = (uint32_t)l->memory_dim, c.memory.banks = (uint32_t)l->banks;
    c.memory.slot_capacity = (uint32_t)l->slot_capacity, c.memory.codebook_capacity = (uint32_t)l->codebook_capacity;
    c.memory.slot_policy = ELPIS_ECS_MEM_REFUSE, c.memory.codebook_policy = ELPIS_ECS_MEM_EVICT_LRU;
    c.memory.capabilities = 0; /* admitted operations only: never an unqualified capability */
    return c;
}

void host_initial_weights(const host_lineage *l, double *w) {
    for (uint64_t i = 0; i < l->dim * l->width; ++i) {
        uint8_t in[sizeof W_DOMAIN - 1 + 16], h[32];
        memcpy(in, W_DOMAIN, sizeof W_DOMAIN - 1);
        for (int b = 0; b < 8; ++b) {
            in[sizeof W_DOMAIN - 1 + b] = (uint8_t)(l->seed >> (8 * b));
            in[sizeof W_DOMAIN - 1 + 8 + b] = (uint8_t)(i >> (8 * b));
        }
        elpis_sha256(in, sizeof in, h);
        uint64_t u = 0;
        for (int b = 7; b >= 0; --b) u = (u << 8) | h[b];
        w[i] = ((double)u / 18446744073709551616.0 - 0.5) * 0.36;
    }
}

static int bounded(uint64_t v, uint64_t lo, uint64_t hi) { return v >= lo && v <= hi; }

int host_lineage_check(host_install *d, host_err *e) {
    const host_api *a = &d->api;
    const host_lineage *l = &d->lineage;
    if (!bounded(l->dim, 1, 64) || !bounded(l->width, 1, 1u << 16) || !bounded(l->max_rows, 1, 1u << 16) ||
        !bounded(l->memory_dim, 1, UINT32_MAX) || !bounded(l->banks, 1, UINT32_MAX) ||
        !bounded(l->slot_capacity, 1, UINT32_MAX) || !bounded(l->codebook_capacity, 1, UINT32_MAX))
        return host_fail(e, "LINEAGE_CONFIG", "every lineage dimension is a positive integer (dim 1..64)");
    const elpis_ecs_composite_config cfg = host_composite_config(d);
    d->envelope_bytes = a->elpis_ecs_composite_envelope_bytes(&cfg);
    if (!d->envelope_bytes) return host_fail(e, "LINEAGE_CONFIG", "configuration beyond the composite's declared bounds");
    double *w = malloc(sizeof(double) * l->dim * l->width);
    if (!w) return host_fail(e, "HOST_RESOURCES", "out of memory");
    host_initial_weights(l, w);
    elpis_sha256(w, sizeof(double) * l->dim * l->width, d->initial_w_sha256); /* binary64, little-endian hosts */
    elpis_ecs_composite *x = NULL;
    const int created = a->elpis_ecs_composite_create(&cfg, w, &x);
    free(w);
    if (created)
        return host_fail(e, "LINEAGE_CONFIG", "composite create refused: %s", a->elpis_ecs_composite_code_name(created));
    const int rc = a->elpis_ecs_composite_state_digest(x, d->initial_identity);
    a->elpis_ecs_composite_destroy(x);
    return rc ? host_fail(e, "ECS_STATE", "the initial composite has no identity") : 0;
}
