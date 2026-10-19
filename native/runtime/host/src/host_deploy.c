/* One native Elpis deployment (host_deploy.h; docs/NATIVE_HOST.md). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "host_deploy.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "elpis/sha256.h"
#include "host_json.h"

#define HOST_SCHEMA "elpis.native-host.v1"
#define CATALOG_SCHEMA "elpis.native-host-catalog.v1"
#define LINEAGE_SCHEMA "elpis.native-lineage.v1"
#define W_DOMAIN "elpis.deployment.initial-w.v1"
#define LABEL_DOMAIN "elpis.host.label.v1"
#define CONFIG_LIMIT ((size_t)1 << 16)
#define LIBRARY_LIMIT ((uint64_t)1 << 28)

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
    LIB(elpis_ecs_c_authority, "native/ECS_C/authority"),
#ifdef ELPIS_HOST_TESTING
    /* The testing RuntimeCore build: the same source with its fault hooks; its SONAME is the production one. */
    {"elpis_runtime_testing", "libelpis_runtime_testing.so", "libelpis_runtime.so",
     "native/runtime/libelpis_runtime_testing.so"},
#else
    LIB(elpis_runtime, "native/runtime"),
#endif
};
#undef LIB

const host_library *host_libraries(void) { return LIBRARIES; }

/* ---- small helpers -------------------------------------------------------------------------------------------- */

/* `n` lowercase hex characters exactly. */
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

static int join(char *out, const char *dir, const char *name) {
    const int n = snprintf(out, HOST_PATH, "%s/%s", dir, name);
    return n > 0 && n < HOST_PATH ? 0 : -1;
}

/* An absolute path without "." or ".." components, a doubled or trailing slash. */
static int clean_path(const char *p) {
    if (!p || p[0] != '/' || strlen(p) >= HOST_PATH - 64) return 0;
    for (const char *s = p; *s;) {
        const char *next = strchr(s + 1, '/');
        const size_t n = next ? (size_t)(next - s - 1) : strlen(s + 1);
        if (n == 0 && (next || s != p)) return 0;
        if ((n == 1 && s[1] == '.') || (n == 2 && s[1] == '.' && s[2] == '.')) return 0;
        if (!next) break;
        s = next;
    }
    return 1;
}

/* An existing home must be a real directory owned by this user and writable by nobody else. */
static int check_home(const char *home, int must_exist, host_err *e) {
    if (!clean_path(home))
        return host_fail(e, "DEPLOYMENT_PATH", "the home must be an absolute path without '.', '..' or empty parts");
    struct stat st;
    if (lstat(home, &st)) {
        if (errno == ENOENT && !must_exist) return 0;
        if (errno == ENOENT) return host_fail(e, "DEPLOYMENT_MISSING", "no deployment at %s (run init)", home);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: %s", home, strerror(errno));
    }
    if (!S_ISDIR(st.st_mode))
        return host_fail(e, "DEPLOYMENT_PATH", "the home must be a real directory, not a symlink or a file");
    if (st.st_uid != geteuid()) return host_fail(e, "DEPLOYMENT_PATH", "the home must be owned by the current user");
    if (st.st_mode & 022) return host_fail(e, "DEPLOYMENT_PATH", "the home must not be writable by group or others");
    return 0;
}

static int read_beneath(int dirfd, const char *name, size_t limit, char **out, size_t *len, host_err *e) {
    const int fd = host_open_beneath(dirfd, name);
    if (fd < 0) {
        if (errno == ENOENT) return host_fail(e, "DEPLOYMENT_MISSING", "%s is missing", name);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: %s", name, strerror(errno));
    }
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size > limit) {
        close(fd);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: a regular file of at most %zu bytes is required", name, limit);
    }
    char *buf = malloc((size_t)st.st_size + 1);
    size_t done = 0;
    while (buf && done < (size_t)st.st_size) {
        const ssize_t r = read(fd, buf + done, (size_t)st.st_size - done);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    if (!buf || done != (size_t)st.st_size) {
        free(buf);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: cannot be read", name);
    }
    buf[done] = 0;
    *out = buf, *len = done;
    return 0;
}

static int sync_dir(int dirfd) { return fsync(dirfd); }

/* Create `name` beneath `dirfd` with exactly `data` (never replacing a file), synced with its directory. */
static int write_once(int dirfd, const char *name, const void *data, size_t len, mode_t mode, host_err *e) {
    const int fd = openat(dirfd, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
    if (fd < 0) return host_fail(e, "DEPLOYMENT_PATH", "cannot create %s: %s", name, strerror(errno));
    size_t done = 0;
    while (done < len) {
        const ssize_t w = write(fd, (const char *)data + done, len - done);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        done += (size_t)w;
    }
    const int synced = done == len && fsync(fd) == 0;
    close(fd);
    if (!synced || sync_dir(dirfd)) return host_fail(e, "DEPLOYMENT_PATH", "cannot write %s: %s", name, strerror(errno));
    return 0;
}

static int exists_beneath(int dirfd, const char *name) {
    struct stat st;
    return fstatat(dirfd, name, &st, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT;
}

/* ---- resolution and ABI checks -------------------------------------------------------------------------------- */

static const host_api *ACTIVE; /* the deployment whose functions the substrate table calls (one per process) */

static int resolve(host_deployment *d, host_err *e) {
    host_api *a = &d->api;
#define RESOLVE(lib, name)                                                                                         \
    if (!(a->name = (__typeof__(name) *)host_symbol(d->handles[lib], #name)))                                      \
        return host_fail(e, "DEPLOYMENT_PIN", "%s does not export %s", d->pins[lib].file, #name);
#define R_RUNTIME(name) RESOLVE(LIB_RUNTIME, name)
#define R_COMPOSITE(name) RESOLVE(LIB_COMPOSITE, name)
#define R_MEMORY(name) RESOLVE(LIB_MEMORY, name)
#define R_HACF(name) RESOLVE(LIB_HACF, name)
#define R_ECS_C(name) RESOLVE(LIB_ECS_C, name)
    HOST_RUNTIME_FUNCTIONS(R_RUNTIME)
    HOST_TESTING_FUNCTIONS(R_RUNTIME)
    HOST_COMPOSITE_FUNCTIONS(R_COMPOSITE)
    HOST_MEMORY_FUNCTIONS(R_MEMORY)
    HOST_HACF_FUNCTIONS(R_HACF)
    HOST_ECS_C_FUNCTIONS(R_ECS_C)
#undef RESOLVE
    if (a->elpis_runtime_abi_version() != ELPIS_RUNTIME_ABI_V6)
        return host_fail(e, "DEPLOYMENT_ABI", "RuntimeCore ABI %u is not this host's (6)", a->elpis_runtime_abi_version());
    if (a->elpis_ecs_c_abi_version() != ELPIS_ECS_C_ABI_VERSION)
        return host_fail(e, "DEPLOYMENT_ABI", "ECS_C ABI %u is not this host's (%u)", a->elpis_ecs_c_abi_version(),
                         ELPIS_ECS_C_ABI_VERSION);
    ACTIVE = a;
    return 0;
}

static int admit(host_deployment *d, host_err *e) {
    char err[512];
    if (host_admit(d->lib_fd, d->pins, HOST_LIBRARIES, d->handles, d->sealed, err, sizeof err))
        return host_fail(e, "DEPLOYMENT_PIN", "native library admission refused: %s", err);
    return resolve(d, e);
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
static int c_capture(void *s, uint64_t lease, uint8_t *out, size_t size, uint64_t *generation) {
    return ACTIVE->elpis_ecs_composite_leased_capture(s, lease, out, size, generation);
}
static int c_seal(uint8_t *envelope, size_t size) { return ACTIVE->elpis_ecs_composite_envelope_seal(envelope, size); }
static int c_observe(void *s, uint64_t lease, uint64_t token, const void *observation, uint8_t key[32], int *ms) {
    return ACTIVE->elpis_ecs_composite_leased_txn_observe(s, lease, token, observation, (elpis_ecs_mem_digest *)key,
                                                          ms);
}
static const elpis_runtime_composite_api SUBSTRATE_API = {
    {c_digest, c_shape, c_claim, c_release, c_reserve, c_identity, c_query, c_begin, c_schedule, c_commit, c_abort,
     c_capture, c_seal},
    c_observe};

const elpis_runtime_composite_api *host_substrate_api(void) { return &SUBSTRATE_API; }

/* ---- configuration files -------------------------------------------------------------------------------------- */

static int parse(char *text, size_t len, hj_arena *arena, char *bytes, size_t cap, hj_value **root, const char *name,
                 host_err *e) {
    char err[128];
    *arena = (hj_arena){4096, 0, bytes, cap, 0};
    if (hj_parse(text, len, arena, root, err, sizeof err) || (*root)->kind != HJ_OBJECT)
        return host_fail(e, "DEPLOYMENT_CONFIG", "%s: not a JSON object (%s)", name, err);
    return 0;
}

static const char *str_of(const hj_value *o, const char *key) {
    const hj_value *v = hj_get(o, key);
    return v && v->kind == HJ_STRING ? v->string : NULL;
}

static int u64_of(const hj_value *o, const char *key, uint64_t *out) {
    const hj_value *v = hj_get(o, key);
    if (!v || v->kind != HJ_NUMBER || !v->integral || v->integer < 0) return -1;
    *out = (uint64_t)v->integer;
    return 0;
}

static int hex_of(const hj_value *o, const char *key, uint8_t out[32]) { return host_unhex32(str_of(o, key), out); }

static void write_catalog(hjw *w, const host_deployment *d) {
    hjw_object(w);
    hjw_kstr(w, "schema", CATALOG_SCHEMA);
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

static int read_catalog(host_deployment *d, host_err *e) {
    char *text;
    size_t len;
    if (read_beneath(d->home_fd, "catalog", CONFIG_LIMIT, &text, &len, e)) return -1;
    uint8_t digest[32];
    elpis_sha256(text, len, digest);
    if (memcmp(digest, d->catalog_sha256, 32)) {
        free(text);
        return host_fail(e, "DEPLOYMENT_PIN", "the native catalog does not match its pin");
    }
    static char bytes[1 << 20];
    hj_arena arena;
    hj_value *root;
    int rc = parse(text, len, &arena, bytes, sizeof bytes, &root, "catalog", e);
    const hj_value *libs = rc ? NULL : hj_get(root, "libraries");
    const char *schema = rc ? NULL : str_of(root, "schema");
    if (!rc && (!schema || strcmp(schema, CATALOG_SCHEMA) || !libs || libs->kind != HJ_ARRAY ||
                libs->count != HOST_LIBRARIES))
        rc = host_fail(e, "DEPLOYMENT_CONFIG", "catalog: schema %s with this host's %d libraries required",
                       CATALOG_SCHEMA, HOST_LIBRARIES);
    for (int i = 0; !rc && i < HOST_LIBRARIES; ++i) {
        const hj_value *l = &libs->items[i];
        const char *id = str_of(l, "id"), *file = str_of(l, "file"), *soname = str_of(l, "soname");
        if (l->kind != HJ_OBJECT || !id || !file || !soname || strcmp(id, LIBRARIES[i].id) ||
            strcmp(file, LIBRARIES[i].file) || strcmp(soname, LIBRARIES[i].soname) || u64_of(l, "size", &d->pins[i].size) ||
            hex_of(l, "sha256", d->pins[i].sha256))
            rc = host_fail(e, "DEPLOYMENT_CONFIG", "catalog entry %d is not this host's %s", i, LIBRARIES[i].id);
    }
    free(text);
    return rc;
}

static void write_host_cfg(hjw *w, const host_deployment *d) {
    hjw_object(w);
    hjw_kstr(w, "schema", HOST_SCHEMA);
    hjw_khex(w, "host_sha256", d->host_sha256, 32);
    hjw_kstr(w, "catalog", "catalog");
    hjw_khex(w, "catalog_sha256", d->catalog_sha256, 32);
    hjw_kbool(w, "testing", d->testing);
    hjw_key(w, "source_commit");
    if (d->source_commit[0]) hjw_str(w, d->source_commit);
    else hjw_null(w);
    hjw_key(w, "abi");
    hjw_object(w);
    hjw_ku64(w, "runtime", ELPIS_RUNTIME_ABI_V6);
    hjw_ku64(w, "composite", ELPIS_ECS_COMPOSITE_ABI_V1);
    hjw_ku64(w, "ecs_c", ELPIS_ECS_C_ABI_VERSION);
    hjw_end(w, '}');
    hjw_kstr(w, "authority_dir", d->authority_dir);
    hjw_ku64(w, "capacity", d->capacity);
    hjw_end(w, '}');
}

static int read_host_cfg(host_deployment *d, host_err *e) {
    char *text;
    size_t len;
    if (read_beneath(d->home_fd, "host.cfg", CONFIG_LIMIT, &text, &len, e)) {
        if (!strcmp(e->code, "DEPLOYMENT_MISSING")) host_fail(e, "DEPLOYMENT_MISSING", "no deployment at %s (run init)", d->home);
        return -1;
    }
    static char bytes[1 << 18];
    hj_arena arena;
    hj_value *root;
    int rc = parse(text, len, &arena, bytes, sizeof bytes, &root, "host.cfg", e);
    const char *schema = rc ? NULL : str_of(root, "schema");
    if (!rc && (!schema || strcmp(schema, HOST_SCHEMA)))
        rc = host_fail(e, "DEPLOYMENT_CONFIG", "host.cfg: schema %s required", HOST_SCHEMA);
    if (!rc) {
        const hj_value *abi = hj_get(root, "abi");
        uint64_t r = 0, c = 0, x = 0;
        if (!abi || abi->kind != HJ_OBJECT || abi->count != 3 || u64_of(abi, "runtime", &r) ||
            u64_of(abi, "composite", &c) || u64_of(abi, "ecs_c", &x) || r != ELPIS_RUNTIME_ABI_V6 ||
            c != ELPIS_ECS_COMPOSITE_ABI_V1 || x != ELPIS_ECS_C_ABI_VERSION)
            rc = host_fail(e, "DEPLOYMENT_ABI", "the deployment's ABI is not this host's (runtime 6, composite 1, ECS_C 1)");
    }
    if (!rc) {
        const hj_value *t = hj_get(root, "testing");
#ifdef ELPIS_HOST_TESTING
        const int testing = 1;
#else
        const int testing = 0;
#endif
        if (!t || t->kind != HJ_BOOL) rc = host_fail(e, "DEPLOYMENT_CONFIG", "host.cfg: testing must be a boolean");
        else if (t->boolean != testing)
            rc = host_fail(e, "DEPLOYMENT_CONFIG", t->boolean ? "a testing deployment runs only under the testing host"
                                                            : "the testing host runs only a testing deployment");
        d->testing = testing;
    }
    uint64_t capacity = 0;
    const char *authority = rc ? NULL : str_of(root, "authority_dir");
    if (!rc && (hex_of(root, "host_sha256", d->host_sha256) || hex_of(root, "catalog_sha256", d->catalog_sha256) ||
                !authority || !clean_path(authority) || u64_of(root, "capacity", &capacity) || !capacity ||
                capacity > ELPIS_ECS_C_MAX_CAPACITY))
        rc = host_fail(e, "DEPLOYMENT_CONFIG", "host.cfg: pins, authority_dir and capacity required");
    if (!rc) {
        snprintf(d->authority_dir, sizeof d->authority_dir, "%s", authority);
        d->capacity = (uint32_t)capacity;
        const hj_value *commit = hj_get(root, "source_commit");
        if (commit && commit->kind == HJ_STRING && lower_hex(commit->string, 40))
            snprintf(d->source_commit, sizeof d->source_commit, "%s", commit->string);
        else if (!commit || commit->kind != HJ_NULL)
            rc = host_fail(e, "DEPLOYMENT_CONFIG", "host.cfg: source_commit is null or 40 lowercase hex");
    }
    free(text);
    if (rc) return rc;
    uint8_t self[32];
    char err[256];
    if (host_digest_self(self, err, sizeof err)) return host_fail(e, "DEPLOYMENT_PIN", "%s", err);
    if (memcmp(self, d->host_sha256, 32))
        return host_fail(e, "DEPLOYMENT_PIN", "this host binary is not the one the deployment pinned at init");
    return 0;
}

static void write_lineage(hjw *w, const host_deployment *d) {
    const host_lineage *l = &d->lineage;
    hjw_object(w);
    hjw_kstr(w, "schema", LINEAGE_SCHEMA);
    hjw_key(w, "config");
    hjw_object(w);
    hjw_ku64(w, "dim", l->dim);
    hjw_ku64(w, "width", l->width);
    hjw_ku64(w, "max_rows", l->max_rows);
    hjw_ku64(w, "memory_dim", l->memory_dim);
    hjw_ku64(w, "banks", l->banks);
    hjw_ku64(w, "slot_capacity", l->slot_capacity);
    hjw_ku64(w, "codebook_capacity", l->codebook_capacity);
    hjw_ku64(w, "seed", l->seed);
    hjw_end(w, '}');
    hjw_khex(w, "initial_w_sha256", d->initial_w_sha256, 32);
    hjw_khex(w, "initial_identity", d->initial_identity, 32);
    hjw_kstr(w, "initialization", W_DOMAIN);
    hjw_ku64(w, "envelope_bytes", d->envelope_bytes);
    hjw_end(w, '}');
}

static int read_lineage(host_deployment *d, host_err *e) {
    if (!exists_beneath(d->home_fd, "lineage.cfg")) return 0;
    char *text;
    size_t len;
    if (read_beneath(d->home_fd, "lineage.cfg", CONFIG_LIMIT, &text, &len, e)) return -1;
    static char bytes[1 << 18];
    hj_arena arena;
    hj_value *root;
    int rc = parse(text, len, &arena, bytes, sizeof bytes, &root, "lineage.cfg", e);
    const char *schema = rc ? NULL : str_of(root, "schema");
    const hj_value *c = rc ? NULL : hj_get(root, "config");
    host_lineage *l = &d->lineage;
    if (!rc && (!schema || strcmp(schema, LINEAGE_SCHEMA) || !c || c->kind != HJ_OBJECT || u64_of(c, "dim", &l->dim) ||
                u64_of(c, "width", &l->width) || u64_of(c, "max_rows", &l->max_rows) ||
                u64_of(c, "memory_dim", &l->memory_dim) || u64_of(c, "banks", &l->banks) ||
                u64_of(c, "slot_capacity", &l->slot_capacity) || u64_of(c, "codebook_capacity", &l->codebook_capacity) ||
                u64_of(c, "seed", &l->seed) || hex_of(root, "initial_w_sha256", d->initial_w_sha256) ||
                hex_of(root, "initial_identity", d->initial_identity) || u64_of(root, "envelope_bytes", &d->envelope_bytes)))
        rc = host_fail(e, "LINEAGE_CONFIG", "lineage.cfg: schema %s with its configuration required", LINEAGE_SCHEMA);
    free(text);
    if (rc) return rc;
    const elpis_ecs_composite_config cfg = host_composite_config(d, l);
    if (d->api.elpis_ecs_composite_envelope_bytes(&cfg) != d->envelope_bytes)
        return host_fail(e, "LINEAGE_CONFIG", "lineage.cfg: the envelope size is not its configuration's");
    d->has_lineage = 1;
    return 0;
}

/* ---- init and open -------------------------------------------------------------------------------------------- */

static int open_dirs(host_deployment *d, const char *home, host_err *e) {
    snprintf(d->home, sizeof d->home, "%s", home);
    if (join(d->state_dir, home, "state") || join(d->checkpoint_dir, home, "checkpoint"))
        return host_fail(e, "DEPLOYMENT_PATH", "the home path is too long");
    d->home_fd = open(home, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (d->home_fd < 0) return host_fail(e, "DEPLOYMENT_PATH", "%s: %s", home, strerror(errno));
    return 0;
}

static int empty_dir(int fd) {
    const int dup_fd = dup(fd);
    DIR *dir = dup_fd >= 0 ? fdopendir(dup_fd) : NULL;
    if (!dir) return 0;
    int empty = 1;
    for (struct dirent *ent; (ent = readdir(dir));)
        if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) empty = 0;
    closedir(dir);
    return empty;
}

static int copy_library(int lib_fd, const char *build, const host_library *l, host_pin *pin, host_err *e) {
    char path[HOST_PATH];
    if (join(path, build, l->build_path)) return host_fail(e, "DEPLOYMENT_PATH", "the build path is too long");
    const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size > LIBRARY_LIMIT) {
        if (fd >= 0) close(fd);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: a built library is required", path);
    }
    char *data = malloc((size_t)st.st_size ? (size_t)st.st_size : 1);
    size_t done = 0;
    while (data && done < (size_t)st.st_size) {
        const ssize_t r = read(fd, data + done, (size_t)st.st_size - done);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    if (!data || done != (size_t)st.st_size) {
        free(data);
        return host_fail(e, "DEPLOYMENT_PATH", "%s: cannot be read", path);
    }
    *pin = (host_pin){l->id, l->file, l->soname, done, {0}};
    elpis_sha256(data, done, pin->sha256);
    const int rc = write_once(lib_fd, l->file, data, done, 0500, e);
    free(data);
    return rc;
}

int host_init(const char *home, const char *build, const char *authority_dir, uint32_t capacity,
              const char *source_commit, host_deployment *d, host_err *e) {
    memset(d, 0, sizeof *d);
    d->home_fd = d->lib_fd = -1;
#ifdef ELPIS_HOST_TESTING
    d->testing = 1;
#endif
    if (check_home(home, 0, e)) return -1;
    if (!clean_path(build)) return host_fail(e, "DEPLOYMENT_PATH", "the native build must be an absolute path");
    if (source_commit) {
        if (!lower_hex(source_commit, 40))
            return host_fail(e, "DEPLOYMENT_CONFIG", "source_commit: a 40-character lowercase hex commit id");
        snprintf(d->source_commit, sizeof d->source_commit, "%s", source_commit);
    }
    if (capacity < ELPIS_ECS_C_MIN_CAPACITY || capacity > ELPIS_ECS_C_MAX_CAPACITY)
        return host_fail(e, "DEPLOYMENT_CONFIG", "capacity: %u..%u entities", ELPIS_ECS_C_MIN_CAPACITY,
                         ELPIS_ECS_C_MAX_CAPACITY);
    d->capacity = capacity;
    struct stat st;
    if (lstat(home, &st) && mkdir(home, 0700)) return host_fail(e, "DEPLOYMENT_PATH", "%s: %s", home, strerror(errno));
    if (open_dirs(d, home, e)) return -1;
    if (!empty_dir(d->home_fd))
        return host_fail(e, "DEPLOYMENT_EXISTS", "%s is not empty: an existing deployment is never overwritten", home);
    if (authority_dir) {
        if (!clean_path(authority_dir)) return host_fail(e, "DEPLOYMENT_PATH", "the authority directory must be absolute");
        snprintf(d->authority_dir, sizeof d->authority_dir, "%s", authority_dir);
    } else if (join(d->authority_dir, home, "authority")) {
        return host_fail(e, "DEPLOYMENT_PATH", "the home path is too long");
    }
    if (mkdirat(d->home_fd, "lib", 0700) ||
        (d->lib_fd = openat(d->home_fd, "lib", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)) < 0)
        return host_fail(e, "DEPLOYMENT_PATH", "lib: %s", strerror(errno));
    for (int i = 0; i < HOST_LIBRARIES; ++i)
        if (copy_library(d->lib_fd, build, &LIBRARIES[i], &d->pins[i], e)) return -1;
    if (admit(d, e)) return -1; /* verified before anything names it */
    static char text[1 << 16];
    hjw w;
    hjw_init(&w, text, sizeof text);
    write_catalog(&w, d);
    if (!hjw_ok(&w)) return host_fail(e, "DEPLOYMENT_CONFIG", "catalog too large");
    elpis_sha256(text, w.len, d->catalog_sha256);
    if (write_once(d->home_fd, "catalog", text, w.len, 0400, e)) return -1;
    uint8_t op[32], rt[32];
    host_label("operator", op);
    host_label("runtime", rt);
    const int32_t p = d->api.elpis_ecs_c_provision(d->state_dir, strlen(d->state_dir), d->authority_dir,
                                                   strlen(d->authority_dir), capacity, op, rt, NULL);
    if (p) return host_fail(e, d->api.elpis_ecs_c_code_name(p), "the ECS_C authority could not be provisioned");
    char err[256];
    if (host_digest_self(d->host_sha256, err, sizeof err)) return host_fail(e, "DEPLOYMENT_PIN", "%s", err);
    hjw_init(&w, text, sizeof text);
    write_host_cfg(&w, d);
    if (!hjw_ok(&w)) return host_fail(e, "DEPLOYMENT_CONFIG", "host.cfg too large");
    return write_once(d->home_fd, "host.cfg", text, w.len, 0600, e); /* the publication point of init */
}

int host_open(const char *home, host_deployment *d, host_err *e) {
    memset(d, 0, sizeof *d);
    d->home_fd = d->lib_fd = -1;
    if (check_home(home, 1, e) || open_dirs(d, home, e) || read_host_cfg(d, e) || read_catalog(d, e)) return -1;
    d->lib_fd = openat(d->home_fd, "lib", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (d->lib_fd < 0) return host_fail(e, "DEPLOYMENT_PATH", "lib: %s", strerror(errno));
    for (int i = 0; i < HOST_LIBRARIES; ++i)
        d->pins[i].id = LIBRARIES[i].id, d->pins[i].file = LIBRARIES[i].file, d->pins[i].soname = LIBRARIES[i].soname;
    if (admit(d, e)) return -1;
    return read_lineage(d, e);
}

/* ---- the lineage ---------------------------------------------------------------------------------------------- */

elpis_ecs_composite_config host_composite_config(const host_deployment *d, const host_lineage *l) {
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

int host_create_lineage(host_deployment *d, const host_lineage *l, uint8_t out_identity[32],
                        elpis_continuity_snapshot *anchored, host_err *e) {
    const host_api *a = &d->api;
    if (d->has_lineage || exists_beneath(d->home_fd, "lineage.cfg") || exists_beneath(d->home_fd, "checkpoint"))
        return host_fail(e, "LINEAGE_EXISTS", "a lineage (or a partial one) exists: lineage data is never overwritten");
    if (!bounded(l->dim, 1, 64) || !bounded(l->width, 1, 1u << 16) || !bounded(l->max_rows, 1, 1u << 16) ||
        !bounded(l->memory_dim, 1, UINT32_MAX) || !bounded(l->banks, 1, UINT32_MAX) ||
        !bounded(l->slot_capacity, 1, UINT32_MAX) || !bounded(l->codebook_capacity, 1, UINT32_MAX))
        return host_fail(e, "LINEAGE_CONFIG", "every lineage dimension is a positive integer (dim 1..64)");
    const elpis_ecs_composite_config cfg = host_composite_config(d, l);
    const uint64_t envelope = a->elpis_ecs_composite_envelope_bytes(&cfg);
    if (!envelope) return host_fail(e, "LINEAGE_CONFIG", "configuration beyond the composite's declared bounds");
    double *w = malloc(sizeof(double) * l->dim * l->width);
    if (!w) return host_fail(e, "HOST_RESOURCES", "out of memory");
    host_initial_weights(l, w);
    elpis_ecs_composite *x = NULL;
    const int created = a->elpis_ecs_composite_create(&cfg, w, &x);
    d->lineage = *l;
    d->envelope_bytes = envelope;
    elpis_sha256(w, sizeof(double) * l->dim * l->width, d->initial_w_sha256); /* binary64, little-endian hosts */
    free(w);
    if (created) return host_fail(e, "LINEAGE_CONFIG", "composite create refused: %s", a->elpis_ecs_composite_code_name(created));
    int rc = a->elpis_ecs_composite_state_digest(x, d->initial_identity) ? host_fail(e, "ECS_STATE", "no identity") : 0;
    static char text[1 << 14];
    hjw jw;
    hjw_init(&jw, text, sizeof text);
    write_lineage(&jw, d);
    if (!rc) rc = write_once(d->home_fd, "lineage.cfg", text, jw.len, 0400, e);
    if (!rc) {
        const int p = a->elpis_runtime_checkpoint_provision((const uint8_t *)d->checkpoint_dir, strlen(d->checkpoint_dir),
                                                            envelope);
        if (p) rc = host_fail(e, a->elpis_runtime_code_name(p), "the checkpoint store could not be provisioned");
    }
    elpis_runtime *rt = NULL;
    if (!rc) {
        int r = a->elpis_runtime_create_ecs_c((const uint8_t *)d->state_dir, strlen(d->state_dir),
                                              (const uint8_t *)d->authority_dir, strlen(d->authority_dir), &rt);
        elpis_continuity_snapshot opened;
        if (!r) r = a->elpis_runtime_open(rt, &opened);
        if (!r) r = a->elpis_runtime_checkpoint_attach(rt, (const uint8_t *)d->checkpoint_dir, strlen(d->checkpoint_dir));
        const elpis_runtime_substrate s = {ELPIS_RUNTIME_SUBSTRATE_COMPOSITE, 0, x, 0, 1, l->dim, &SUBSTRATE_API};
        if (!r) r = a->elpis_runtime_anchor(rt, &s, anchored);
        if (r) rc = host_fail(e, a->elpis_runtime_code_name(r), "the lineage could not be anchored");
    }
    if (rt) a->elpis_runtime_destroy(&rt);
    a->elpis_ecs_composite_destroy(x);
    if (rc) return rc;
    memcpy(out_identity, d->initial_identity, 32);
    d->has_lineage = 1;
    return 0;
}
