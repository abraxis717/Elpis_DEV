#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_k1_fms.h"
#include "ecsg_k1_internal.h"
#include "elpis/sha256.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* One slot per logical K1 state: the FMS object holding its resident image, the pin while an operation (or an
 * open transaction) holds it, and one persistent native workspace that the adapter binds to the resident bytes
 * for each operation. */
typedef struct {
    uint64_t id;
    fms_id object;
    int pinned;
    uint8_t *image;
    elpis_ecsg_k1 *k1;
    int txn_open;
    uint64_t txn_external_token;
    uint64_t txn_native_token;
    uint64_t txn_tokens_issued;
    int busy;
    elpis_ecsg_k1_fms_info info;
} slot;

struct elpis_ecsg_k1_fms {
    fms_ctx *fms;
    slot *slots;
    size_t capacity;
    uint64_t issued;
    uint64_t high_water;
    uint64_t demotion_ns;
    pthread_mutex_t mu;
};

static uint64_t now_ns(void)
{
    struct timespec t;
    (void)clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static int fm(int code)
{
    return code < 0 ? ELPIS_ECSG_K1_FMS_ERROR_BASE + code : 0;
}

static void put_u64(uint8_t *p, uint64_t v)
{
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)(v >> (8u * i));
    }
}

static void logical_identity(const uint8_t key[32], size_t dim, size_t width, uint8_t out[32])
{
    static const char domain[] = "elpis.ecsg.k1.logical.v1";
    uint8_t framed[sizeof(domain) + 48u];
    memcpy(framed, domain, sizeof(domain));
    memcpy(framed + sizeof(domain), key, 32u);
    put_u64(framed + sizeof(domain) + 32u, (uint64_t)dim);
    put_u64(framed + sizeof(domain) + 40u, (uint64_t)width);
    elpis_sha256(framed, sizeof(framed), out);
}

static slot *find(elpis_ecsg_k1_fms *r, uint64_t id)
{
    size_t i;
    for (i = 0u; i < r->capacity; ++i) {
        if (id != 0u && r->slots[i].id == id) {
            return &r->slots[i];
        }
    }
    return NULL;
}

static int take(elpis_ecsg_k1_fms *r, uint64_t id, slot **out)
{
    slot *s;
    int rc;
    if (r == NULL || out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    pthread_mutex_lock(&r->mu);
    s = find(r, id);
    rc = s == NULL ? ELPIS_ECSG_K1_INVALID : s->busy ? ELPIS_ECSG_K1_BUSY : ELPIS_ECSG_K1_OK;
    if (rc == ELPIS_ECSG_K1_OK) {
        s->busy = 1;
        *out = s;
    }
    pthread_mutex_unlock(&r->mu);
    return rc;
}

static void give(elpis_ecsg_k1_fms *r, slot *s)
{
    pthread_mutex_lock(&r->mu);
    s->busy = 0;
    pthread_mutex_unlock(&r->mu);
}

static void high_water(elpis_ecsg_k1_fms *r)
{
    fms_stats st;
    fms_get_stats(r->fms, &st);
    pthread_mutex_lock(&r->mu);
    if (st.domain_bytes[FMS_DOM_RAM] > r->high_water) {
        r->high_water = st.domain_bytes[FMS_DOM_RAM];
    }
    pthread_mutex_unlock(&r->mu);
}

/* Pin the resident image WARM (FMS materializes COLD -> WARM when needed; that is the only allocating path) and
 * bind the workspace to it. A CPU operation needs no accelerator fence, so the unfenced pin is used: on a WARM
 * object it allocates nothing. */
static int acquire(elpis_ecsg_k1_fms *r, slot *s, unsigned mode)
{
    void *ptr = NULL;
    uint64_t start;
    int rc;
    if (s->pinned) {   /* an open transaction holds a WRITE pin: the image cannot move */
        return s->image != NULL ? ELPIS_ECSG_K1_OK : fm(FMS_E_STATE);
    }
    start = now_ns();
    rc = fms_acquire(r->fms, s->object, FMS_WARM, mode, &ptr);
    s->info.materialization_ns += now_ns() - start;
    if (rc < 0) {
        s->info.lease_failures += 1u;
        return fm(rc);
    }
    s->pinned = 1;
    s->image = (uint8_t *)ptr;
    if (rc != FMS_WARM || s->image == NULL) {
        (void)fms_release(r->fms, s->object);
        s->pinned = 0;
        s->image = NULL;
        s->info.lease_failures += 1u;
        return fm(FMS_E_STATE);
    }
    s->info.acquisitions += 1u;
    ecsg_k1_internal_bind(s->k1, s->image);
    return ELPIS_ECSG_K1_OK;
}

static int release(elpis_ecsg_k1_fms *r, slot *s)
{
    int rc;
    ecsg_k1_internal_bind(s->k1, NULL);
    s->image = NULL;
    if (!s->pinned) {
        return ELPIS_ECSG_K1_OK;
    }
    rc = fms_release(r->fms, s->object);
    s->pinned = 0;
    return fm(rc);
}

static void observe(slot *s)
{
    s->info.epoch = elpis_ecsg_k1_epoch(s->k1);
    s->info.provenance = elpis_ecsg_k1_provenance_of(s->k1);
    s->info.generation = elpis_ecsg_k1_generation(s->k1);
}

static void clear_transaction(elpis_ecsg_k1_fms *r, slot *s, int aborted)
{
    if (s->txn_open && aborted) {
        (void)elpis_ecsg_k1_txn_abort(s->k1, s->txn_native_token);
        s->info.aborts += 1u;
    }
    s->txn_open = 0;
    s->txn_external_token = 0u;
    s->txn_native_token = 0u;
    (void)release(r, s);
}

/* --- lifecycle ----------------------------------------------------------------------------------------------- */

uint32_t elpis_ecsg_k1_fms_abi_version(void)
{
    return ELPIS_ECSG_K1_FMS_ABI_V1;
}

int elpis_ecsg_k1_fms_create(fms_ctx *owned, size_t capacity, elpis_ecsg_k1_fms **out)
{
    fms_stats existing;
    elpis_ecsg_k1_fms *r;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = NULL;
    if (owned == NULL || capacity == 0u || capacity > SIZE_MAX / sizeof(slot)) {
        return ELPIS_ECSG_K1_INVALID;
    }
    fms_get_stats(owned, &existing);
    if (existing.objects != 0u || existing.inflight_ops != 0u || existing.pinned_bytes != 0u) {
        return ELPIS_ECSG_K1_INVALID;
    }
    r = (elpis_ecsg_k1_fms *)calloc(1u, sizeof(*r));
    if (r == NULL) {
        return ELPIS_ECSG_K1_NOMEM;
    }
    r->slots = (slot *)calloc(capacity, sizeof(*r->slots));
    if (r->slots == NULL || pthread_mutex_init(&r->mu, NULL) != 0) {
        free(r->slots);
        free(r);
        return ELPIS_ECSG_K1_NOMEM;
    }
    r->fms = owned;
    r->capacity = capacity;
    *out = r;
    return ELPIS_ECSG_K1_OK;
}

int elpis_ecsg_k1_fms_destroy(elpis_ecsg_k1_fms **runtime)
{
    elpis_ecsg_k1_fms *r;
    size_t i;
    if (runtime == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    r = *runtime;
    if (r == NULL) {
        return ELPIS_ECSG_K1_OK;
    }
    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].id != 0u || r->slots[i].busy) {
            pthread_mutex_unlock(&r->mu);
            return ELPIS_ECSG_K1_BUSY;
        }
    }
    pthread_mutex_unlock(&r->mu);
    fms_destroy(r->fms);
    pthread_mutex_destroy(&r->mu);
    free(r->slots);
    free(r);
    *runtime = NULL;
    return ELPIS_ECSG_K1_OK;
}

/* Registers the image of a fully built standalone state `src` as a new FMS object with its own workspace. */
static int install(elpis_ecsg_k1_fms *r, const uint8_t key[32], elpis_ecsg_k1 *src, size_t max_rows, uint64_t *out)
{
    uint8_t logical[32];
    const size_t dim = elpis_ecsg_k1_dim(src);
    const size_t width = elpis_ecsg_k1_width(src);
    const size_t bytes = ecsg_k1_internal_image_bytes(src);
    elpis_ecsg_k1 *workspace = NULL;
    slot *s = NULL;
    fms_id object = 0u;
    size_t i;
    int rc;
    if (key == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    logical_identity(key, dim, width, logical);
    rc = ecsg_k1_internal_create_bound(dim, width, max_rows, &workspace);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].id != 0u && memcmp(logical, r->slots[i].info.logical_identity, 32u) == 0) {
            pthread_mutex_unlock(&r->mu);
            (void)elpis_ecsg_k1_destroy(&workspace);
            return ELPIS_ECSG_K1_INVALID;
        }
        if (r->slots[i].id == 0u && s == NULL) {
            s = &r->slots[i];
        }
    }
    if (s == NULL || r->issued == UINT64_MAX) {
        pthread_mutex_unlock(&r->mu);
        (void)elpis_ecsg_k1_destroy(&workspace);
        return ELPIS_ECSG_K1_CAPACITY;
    }
    rc = fms_register(r->fms, UINT32_C(0x45434b31), (uint64_t)bytes, FMS_WARM, 0.0f,
                      ecsg_k1_internal_image(src), &object);
    if (rc < 0) {
        pthread_mutex_unlock(&r->mu);
        (void)elpis_ecsg_k1_destroy(&workspace);
        return fm(rc);
    }
    memset(s, 0, sizeof(*s));
    s->id = ++r->issued;
    s->object = object;
    s->k1 = workspace;
    s->info.dim = dim;
    s->info.width = width;
    s->info.max_rows = max_rows;
    s->info.image_bytes = bytes;
    s->info.envelope_bytes = elpis_ecsg_k1_envelope_bytes(dim, width);
    s->info.workspace_bytes = elpis_ecsg_k1_workspace_bytes(dim, width, max_rows);
    s->info.epoch = elpis_ecsg_k1_epoch(src);
    s->info.provenance = elpis_ecsg_k1_provenance_of(src);
    memcpy(s->info.logical_identity, logical, 32u);
    *out = s->id;
    pthread_mutex_unlock(&r->mu);
    high_water(r);
    return ELPIS_ECSG_K1_OK;
}

static int install_from(elpis_ecsg_k1_fms *r, const uint8_t key[32], elpis_ecsg_k1_status built,
                        elpis_ecsg_k1 *src, size_t max_rows, uint64_t *out)
{
    int rc = built;
    if (rc == ELPIS_ECSG_K1_OK) {
        rc = install(r, key, src, max_rows, out);
    }
    if (src != NULL) {
        (void)elpis_ecsg_k1_destroy(&src);
    }
    return rc;
}

int elpis_ecsg_k1_fms_register(elpis_ecsg_k1_fms *r, const uint8_t key[32], size_t dim, size_t width,
                               size_t max_rows, const double *w, uint64_t *out)
{
    elpis_ecsg_k1 *src = NULL;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = 0u;
    if (r == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    const elpis_ecsg_k1_status built = elpis_ecsg_k1_create(dim, width, max_rows, w, &src);   /* sequenced before src is read */
    return install_from(r, key, built, src, max_rows, out);
}

int elpis_ecsg_k1_fms_restore(elpis_ecsg_k1_fms *r, const uint8_t key[32], const uint8_t *envelope, size_t bytes,
                              size_t max_rows, uint64_t *out)
{
    elpis_ecsg_k1 *src = NULL;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = 0u;
    if (r == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    const elpis_ecsg_k1_status built = elpis_ecsg_k1_restore(envelope, bytes, max_rows, &src);   /* sequenced before src is read */
    return install_from(r, key, built, src, max_rows, out);
}

int elpis_ecsg_k1_fms_import_w_only(elpis_ecsg_k1_fms *r, const uint8_t key[32], const uint8_t *snapshot,
                                    size_t bytes, size_t max_rows, uint64_t *out)
{
    elpis_ecsg_k1 *src = NULL;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *out = 0u;
    if (r == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    const elpis_ecsg_k1_status built = elpis_ecsg_k1_import_w_only(snapshot, bytes, max_rows, &src);   /* sequenced before src is read */
    return install_from(r, key, built, src, max_rows, out);
}

int elpis_ecsg_k1_fms_close(elpis_ecsg_k1_fms *r, uint64_t *id)
{
    slot *s;
    int rc;
    if (r == NULL || id == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    if (*id == 0u) {
        return ELPIS_ECSG_K1_OK;
    }
    rc = take(r, *id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (s->pinned || s->txn_open) {
        give(r, s);
        return ELPIS_ECSG_K1_BUSY;
    }
    rc = fms_unregister(r->fms, s->object);
    if (rc != FMS_OK) {
        give(r, s);
        return fm(rc);
    }
    (void)elpis_ecsg_k1_destroy(&s->k1);
    pthread_mutex_lock(&r->mu);
    memset(s, 0, sizeof(*s));
    pthread_mutex_unlock(&r->mu);
    *id = 0u;
    return ELPIS_ECSG_K1_OK;
}

int elpis_ecsg_k1_fms_inspect(elpis_ecsg_k1_fms *r, uint64_t id, elpis_ecsg_k1_fms_info *out)
{
    fms_object_info o;
    slot *s;
    int rc;
    if (out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = fms_query(r->fms, s->object, &o);
    if (rc == FMS_OK) {
        *out = s->info;
        out->object_handle = s->object;
        out->generation = elpis_ecsg_k1_generation(s->k1);
        out->tier = o.tier;
        out->residency_state = o.state;
        out->cold_replica = o.cold_replica;
        out->lease_count = o.pin_count + o.lease_count;   /* pins and fenced leases */
        out->transaction_open = s->txn_open ? 1u : 0u;
    }
    give(r, s);
    return fm(rc);
}

int elpis_ecsg_k1_fms_k1_stats(elpis_ecsg_k1_fms *r, uint64_t id, elpis_ecsg_k1_counters *out)
{
    slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = elpis_ecsg_k1_stats(s->k1, out);
    give(r, s);
    return rc;
}

int elpis_ecsg_k1_fms_stats(elpis_ecsg_k1_fms *r, elpis_ecsg_k1_fms_metrics *out)
{
    size_t i;
    if (r == NULL || out == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].busy) {
            pthread_mutex_unlock(&r->mu);
            return ELPIS_ECSG_K1_BUSY;
        }
    }
    memset(out, 0, sizeof(*out));
    for (i = 0u; i < r->capacity; ++i) {
        const slot *s = &r->slots[i];
        if (s->id == 0u) {
            continue;
        }
        out->states += 1u;
        out->logical_bytes += s->info.image_bytes;
        out->workspace_bytes += s->info.workspace_bytes;
        out->leases += s->pinned ? 1u : 0u;
    }
    fms_get_stats(r->fms, &out->residency);
    out->resident_authoritative_bytes = out->residency.tier_bytes[FMS_WARM] + out->residency.tier_bytes[FMS_HOT];
    out->resident_high_water = r->high_water;
    out->demotion_ns = r->demotion_ns;
    pthread_mutex_unlock(&r->mu);
    return ELPIS_ECSG_K1_OK;
}

int elpis_ecsg_k1_fms_pump(elpis_ecsg_k1_fms *r)
{
    uint64_t start;
    int rc;
    if (r == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    start = now_ns();
    rc = fms_pump(r->fms);
    pthread_mutex_lock(&r->mu);
    r->demotion_ns += now_ns() - start;
    pthread_mutex_unlock(&r->mu);
    return fm(rc);
}

int elpis_ecsg_k1_fms_reserve(elpis_ecsg_k1_fms *r, uint64_t id, size_t rows)
{
    slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = s->txn_open ? ELPIS_ECSG_K1_BUSY : elpis_ecsg_k1_reserve(s->k1, rows);
    if (rc == ELPIS_ECSG_K1_OK) {
        s->info.max_rows = elpis_ecsg_k1_max_rows(s->k1);
        s->info.workspace_bytes = elpis_ecsg_k1_workspace_bytes(s->info.dim, s->info.width, s->info.max_rows);
    }
    give(r, s);
    return rc;
}

/* --- direct operations: pin -> native operation over the resident bytes -> unpin --------------------- */

typedef enum { OP_FORWARD, OP_LEARN, OP_CONSOLIDATE, OP_RESET, OP_COPY_W, OP_COPY_H, OP_COPY_A, OP_SNAPSHOT } op_kind;

typedef struct {
    op_kind kind;
    const double *x;
    const double *y;
    size_t rows;
    double rate;
    uint64_t steps;
    double *out;
    size_t count;
    uint8_t *bytes;
    elpis_ecsg_k1_transition *transition;
} op_args;

static int run_op(elpis_ecsg_k1 *k1, const op_args *a)
{
    switch (a->kind) {
    case OP_FORWARD:
        return elpis_ecsg_k1_forward(k1, a->x, a->rows, a->out);
    case OP_LEARN:
        return elpis_ecsg_k1_learn(k1, a->x, a->y, a->rows, a->rate, a->steps, a->transition);
    case OP_CONSOLIDATE:
        return elpis_ecsg_k1_consolidate(k1, a->x, a->rows, a->transition);
    case OP_RESET:
        return elpis_ecsg_k1_reset(k1, a->transition);
    case OP_COPY_W:
        return elpis_ecsg_k1_copy_w(k1, a->out, a->count);
    case OP_COPY_H:
        return elpis_ecsg_k1_copy_h_packed(k1, a->out, a->count);
    case OP_COPY_A:
        return elpis_ecsg_k1_copy_a(k1, a->out, a->count);
    case OP_SNAPSHOT:
        return elpis_ecsg_k1_snapshot_write(k1, a->bytes, a->count);
    }
    return ELPIS_ECSG_K1_INVALID;
}

static int direct(elpis_ecsg_k1_fms *r, uint64_t id, const op_args *a)
{
    const int writes = a->kind == OP_LEARN || a->kind == OP_CONSOLIDATE || a->kind == OP_RESET;
    uint64_t start;
    uint64_t *timer;
    slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = acquire(r, s, writes ? FMS_WRITE : FMS_READ);
    if (rc != ELPIS_ECSG_K1_OK) {
        give(r, s);
        return rc;
    }
    timer = a->kind == OP_FORWARD ? &s->info.query_ns : a->kind == OP_LEARN ? &s->info.learn_ns
          : a->kind == OP_CONSOLIDATE ? &s->info.consolidate_ns : &s->info.commit_ns;
    start = now_ns();
    rc = run_op(s->k1, a);
    *timer += now_ns() - start;
    if (writes && rc == ELPIS_ECSG_K1_OK) {
        s->info.commits += 1u;
        observe(s);
    }
    if (!s->txn_open) {
        (void)release(r, s);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_k1_fms_forward(elpis_ecsg_k1_fms *r, uint64_t id, const double *x, size_t rows, double *out)
{
    op_args a = {OP_FORWARD, x, NULL, rows, 0.0, 0u, out, 0u, NULL, NULL};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_learn(elpis_ecsg_k1_fms *r, uint64_t id, const double *x, const double *y, size_t rows,
                            double rate, uint64_t steps, elpis_ecsg_k1_transition *t)
{
    op_args a = {OP_LEARN, x, y, rows, rate, steps, NULL, 0u, NULL, t};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_consolidate(elpis_ecsg_k1_fms *r, uint64_t id, const double *x, size_t rows,
                                  elpis_ecsg_k1_transition *t)
{
    op_args a = {OP_CONSOLIDATE, x, NULL, rows, 0.0, 0u, NULL, 0u, NULL, t};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_reset(elpis_ecsg_k1_fms *r, uint64_t id, elpis_ecsg_k1_transition *t)
{
    op_args a = {OP_RESET, NULL, NULL, 0u, 0.0, 0u, NULL, 0u, NULL, t};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_copy_w(elpis_ecsg_k1_fms *r, uint64_t id, double *out, size_t count)
{
    op_args a = {OP_COPY_W, NULL, NULL, 0u, 0.0, 0u, out, count, NULL, NULL};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_copy_h_packed(elpis_ecsg_k1_fms *r, uint64_t id, double *out, size_t count)
{
    op_args a = {OP_COPY_H, NULL, NULL, 0u, 0.0, 0u, out, count, NULL, NULL};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_copy_a(elpis_ecsg_k1_fms *r, uint64_t id, double *out, size_t count)
{
    op_args a = {OP_COPY_A, NULL, NULL, 0u, 0.0, 0u, out, count, NULL, NULL};
    return direct(r, id, &a);
}

int elpis_ecsg_k1_fms_snapshot_write(elpis_ecsg_k1_fms *r, uint64_t id, uint8_t *out, size_t size)
{
    op_args a = {OP_SNAPSHOT, NULL, NULL, 0u, 0.0, 0u, NULL, size, out, NULL};
    return direct(r, id, &a);
}

/* --- transactions: the WRITE pin is held from begin to commit/abort ---------------------------------------- */

static int txn_take(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, slot **out)
{
    int rc = take(r, id, out);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (!(*out)->txn_open || token == 0u || token != (*out)->txn_external_token) {
        give(r, *out);
        return ELPIS_ECSG_K1_INVALID;
    }
    return ELPIS_ECSG_K1_OK;
}

/* After a candidate operation: a stale or refused candidate is discarded by K1; release the pin with it. */
static int txn_settle(elpis_ecsg_k1_fms *r, slot *s, int rc)
{
    if (rc != ELPIS_ECSG_K1_OK && rc != ELPIS_ECSG_K1_INVALID && rc != ELPIS_ECSG_K1_CAPACITY &&
        rc != ELPIS_ECSG_K1_BUSY) {
        s->info.aborts += 1u;
        s->txn_open = 0;
        clear_transaction(r, s, 0);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_k1_fms_txn_begin(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t *token)
{
    slot *s;
    int rc;
    if (token == NULL) {
        return ELPIS_ECSG_K1_INVALID;
    }
    *token = 0u;
    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (s->txn_open || s->txn_tokens_issued == UINT64_MAX) {
        give(r, s);
        return s->txn_open ? ELPIS_ECSG_K1_BUSY : ELPIS_ECSG_K1_CAPACITY;
    }
    rc = acquire(r, s, FMS_WRITE);
    if (rc == ELPIS_ECSG_K1_OK) {
        rc = elpis_ecsg_k1_txn_begin(s->k1, &s->txn_native_token);
        if (rc != ELPIS_ECSG_K1_OK) {
            (void)release(r, s);
        }
    }
    if (rc == ELPIS_ECSG_K1_OK) {
        s->txn_open = 1;
        s->txn_external_token = ++s->txn_tokens_issued;
        *token = s->txn_external_token;
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_k1_fms_txn_learn(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, const double *x, const double *y,
                                size_t rows, double rate, uint64_t steps, elpis_ecsg_k1_transition *t)
{
    slot *s;
    uint64_t start;
    int rc = txn_take(r, id, token, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    start = now_ns();
    rc = elpis_ecsg_k1_txn_learn(s->k1, s->txn_native_token, x, y, rows, rate, steps, t);
    s->info.learn_ns += now_ns() - start;
    return txn_settle(r, s, rc);
}

int elpis_ecsg_k1_fms_txn_consolidate(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, const double *x,
                                      size_t rows)
{
    slot *s;
    uint64_t start;
    int rc = txn_take(r, id, token, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    start = now_ns();
    rc = elpis_ecsg_k1_txn_consolidate(s->k1, s->txn_native_token, x, rows);
    s->info.consolidate_ns += now_ns() - start;
    return txn_settle(r, s, rc);
}

int elpis_ecsg_k1_fms_txn_forward(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, const double *x, size_t rows,
                                  double *out)
{
    slot *s;
    uint64_t start;
    int rc = txn_take(r, id, token, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    start = now_ns();
    rc = elpis_ecsg_k1_txn_forward(s->k1, s->txn_native_token, x, rows, out);
    s->info.query_ns += now_ns() - start;
    if (rc == ELPIS_ECSG_K1_NONFINITE) {   /* a refused query does not discard the candidate */
        give(r, s);
        return rc;
    }
    return txn_settle(r, s, rc);
}

int elpis_ecsg_k1_fms_txn_epoch(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, uint64_t *epoch)
{
    slot *s;
    int rc = txn_take(r, id, token, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    rc = elpis_ecsg_k1_txn_epoch(s->k1, s->txn_native_token, epoch);
    return txn_settle(r, s, rc);
}

int elpis_ecsg_k1_fms_txn_commit(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token, elpis_ecsg_k1_transition *t)
{
    slot *s;
    uint64_t start;
    int rc = txn_take(r, id, token, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    start = now_ns();
    rc = elpis_ecsg_k1_txn_commit(s->k1, s->txn_native_token, t);   /* one copy of W, epoch, H, a */
    s->info.commit_ns += now_ns() - start;
    if (rc == ELPIS_ECSG_K1_OK) {
        s->info.commits += 1u;
        observe(s);
    } else {
        s->info.aborts += 1u;
    }
    s->txn_open = 0;
    clear_transaction(r, s, 0);
    give(r, s);
    return rc;
}

int elpis_ecsg_k1_fms_txn_abort(elpis_ecsg_k1_fms *r, uint64_t id, uint64_t token)
{
    slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_K1_OK) {
        return rc;
    }
    if (!s->txn_open) {
        give(r, s);
        return ELPIS_ECSG_K1_OK;
    }
    if (token != s->txn_external_token) {
        give(r, s);
        return ELPIS_ECSG_K1_INVALID;
    }
    clear_transaction(r, s, 1);
    give(r, s);
    return ELPIS_ECSG_K1_OK;
}
