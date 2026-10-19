#define _POSIX_C_SOURCE 200809L
#include "elpis/ecsg_fms.h"
#include "elpis/ecsg_state.h"
#include "elpis/sha256.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    elpis_ecsg_fms_id id;
    fms_id object;
    fms_lease *lease;
    void *image;

    elpis_ecsg_executor *txn_exec;
    uint8_t *txn_publish;
    uint64_t txn_external_token;
    uint64_t txn_native_token;
    uint64_t txn_tokens_issued;
    uint64_t txn_source_generation;

    uint64_t generation;
    int busy;
    elpis_ecsg_fms_info info;
    elpis_ecsg_exec_stats accumulated;
} state_slot;

struct elpis_ecsg_fms {
    fms_ctx *fms;
    state_slot *slots;
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
    return code < 0 ? ELPIS_ECSG_FMS_ERROR_BASE + code : 0;
}

static void write_u64_le(uint8_t *p, uint64_t x)
{
    unsigned i;
    for (i = 0u; i < 8u; ++i) {
        p[i] = (uint8_t)(x >> (i * 8u));
    }
}

static void logical_identity(const uint8_t key[32], size_t dim, size_t width, uint8_t out[32])
{
    static const char domain[] = "elpis.ecsg.logical.v1";
    uint8_t framed[sizeof(domain) + 48u];

    memcpy(framed, domain, sizeof(domain)); /* includes the NUL domain frame */
    memcpy(framed + sizeof(domain), key, 32u);
    write_u64_le(framed + sizeof(domain) + 32u, (uint64_t)dim);
    write_u64_le(framed + sizeof(domain) + 40u, (uint64_t)width);
    elpis_sha256(framed, sizeof(framed), out);
}

static state_slot *find_slot(elpis_ecsg_fms *r, uint64_t id)
{
    size_t i;
    for (i = 0u; i < r->capacity; ++i) {
        if (id != 0u && r->slots[i].id == id) {
            return &r->slots[i];
        }
    }
    return NULL;
}

static int take(elpis_ecsg_fms *r, uint64_t id, state_slot **out)
{
    state_slot *s;
    int rc;

    if (r == NULL || out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    pthread_mutex_lock(&r->mu);
    s = find_slot(r, id);
    rc = s == NULL ? ELPIS_ECSG_EXEC_INVALID
         : s->busy ? ELPIS_ECSG_EXEC_BUSY : ELPIS_ECSG_EXEC_OK;
    if (s != NULL && s->busy) {
        s->accumulated.busy_refusals += 1u;
    }
    if (rc == ELPIS_ECSG_EXEC_OK) {
        s->busy = 1;
        *out = s;
    }
    pthread_mutex_unlock(&r->mu);
    return rc;
}

static void give(elpis_ecsg_fms *r, state_slot *s)
{
    pthread_mutex_lock(&r->mu);
    s->busy = 0;
    pthread_mutex_unlock(&r->mu);
}

static void update_high_water(elpis_ecsg_fms *r)
{
    fms_stats st;
    fms_get_stats(r->fms, &st);
    pthread_mutex_lock(&r->mu);
    if (st.domain_bytes[FMS_DOM_RAM] > r->high_water) {
        r->high_water = st.domain_bytes[FMS_DOM_RAM];
    }
    pthread_mutex_unlock(&r->mu);
}

static int acquire_state(elpis_ecsg_fms *r, state_slot *s, unsigned mode)
{
    uint64_t start;
    int rc;

    if (s->lease != NULL) {
        return s->image != NULL ? ELPIS_ECSG_EXEC_OK : fm(FMS_E_STATE);
    }
    start = now_ns();
    rc = fms_lease_acquire(r->fms, s->object, FMS_WARM, mode, &s->lease);
    s->info.materialization_ns += now_ns() - start;
    if (rc != FMS_OK) {
        s->info.lease_failures += 1u;
        return fm(rc);
    }
    if (fms_lease_tier(s->lease) != FMS_WARM) {
        (void)fms_lease_release(r->fms, s->lease);
        s->lease = NULL;
        s->info.lease_failures += 1u;
        return fm(FMS_E_UNSUPPORTED);
    }
    s->image = fms_lease_ptr(s->lease);
    if (s->image == NULL) {
        (void)fms_lease_release(r->fms, s->lease);
        s->lease = NULL;
        s->info.lease_failures += 1u;
        return fm(FMS_E_STATE);
    }
    s->info.acquisitions += 1u;
    update_high_water(r);
    return ELPIS_ECSG_EXEC_OK;
}

static int release_state(elpis_ecsg_fms *r, state_slot *s)
{
    int rc;

    if (s->lease == NULL) {
        s->image = NULL;
        return ELPIS_ECSG_EXEC_OK;
    }
    s->image = NULL;
    rc = fms_lease_release(r->fms, s->lease);
    if (rc != FMS_E_BUSY) {
        s->lease = NULL;
    }
    return fm(rc);
}

static void merge_executor_stats(elpis_ecsg_fms *r, state_slot *s, elpis_ecsg_executor *e)
{
    elpis_ecsg_exec_stats x;

    if (e == NULL || elpis_ecsg_executor_stats(e, &x) != ELPIS_ECSG_EXEC_OK) {
        return;
    }

    /*
     * accumulated is shared accounting state. A concurrent caller that is
     * refused by take() increments accumulated.busy_refusals while holding
     * r->mu, so every merge into the same structure must use that mutex too.
     * The executor statistics snapshot itself is taken before the lock.
     */
    pthread_mutex_lock(&r->mu);
    s->accumulated.heap_allocations += x.heap_allocations;
    s->accumulated.forward_calls += x.forward_calls;
    s->accumulated.learn_calls += x.learn_calls;
    s->accumulated.steps_executed += x.steps_executed;
    s->accumulated.commits += x.commits;
    s->accumulated.refusals += x.refusals;
    s->accumulated.txn_begins += x.txn_begins;
    s->accumulated.txn_aborts += x.txn_aborts;
    s->accumulated.stale_refusals += x.stale_refusals;
    s->accumulated.busy_refusals += x.busy_refusals;
    pthread_mutex_unlock(&r->mu);
}

static void combined_executor_stats(state_slot *s, elpis_ecsg_exec_stats *out)
{
    elpis_ecsg_exec_stats active;

    *out = s->accumulated;
    out->workspace_bytes = 0u;
    out->max_rows = s->info.max_rows;
    if (s->txn_exec != NULL && elpis_ecsg_executor_stats(s->txn_exec, &active) == ELPIS_ECSG_EXEC_OK) {
        out->workspace_bytes = active.workspace_bytes;
        out->heap_allocations += active.heap_allocations;
        out->forward_calls += active.forward_calls;
        out->learn_calls += active.learn_calls;
        out->steps_executed += active.steps_executed;
        out->commits += active.commits;
        out->refusals += active.refusals;
        out->txn_begins += active.txn_begins;
        out->txn_aborts += active.txn_aborts;
        out->stale_refusals += active.stale_refusals;
        out->busy_refusals += active.busy_refusals;
    }
}

static int open_executor(state_slot *s, elpis_ecsg_executor **out)
{
    int rc;

    if (s->txn_exec != NULL) {
        *out = s->txn_exec;
        return ELPIS_ECSG_EXEC_OK;
    }
    rc = elpis_ecsg_executor_restore((const uint8_t *)s->image,
                                     s->info.snapshot_bytes,
                                     s->info.max_rows,
                                     out);
    return rc;
}

static int publish_executor(state_slot *s, elpis_ecsg_executor *e, uint8_t *buffer)
{
    int rc;

    rc = elpis_ecsg_executor_snapshot_write(e, buffer, s->info.snapshot_bytes);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        return rc;
    }
    memcpy(s->image, buffer, s->info.snapshot_bytes);
    s->info.epoch = elpis_ecsg_executor_epoch(e);
    return ELPIS_ECSG_EXEC_OK;
}

static void clear_transaction(elpis_ecsg_fms *r, state_slot *s, int count_abort)
{
    if (s->txn_exec != NULL) {
        merge_executor_stats(r, s, s->txn_exec);
        (void)elpis_ecsg_executor_destroy(&s->txn_exec);
    }
    free(s->txn_publish);
    s->txn_publish = NULL;
    s->txn_external_token = 0u;
    s->txn_native_token = 0u;
    s->txn_source_generation = 0u;
    if (count_abort) {
        s->info.aborts += 1u;
    }
    (void)release_state(r, s);
}

uint32_t elpis_ecsg_fms_abi_version(void)
{
    return ELPIS_ECSG_FMS_ABI_V1;
}

int elpis_ecsg_fms_create(fms_ctx *owned, size_t capacity, elpis_ecsg_fms **out)
{
    fms_stats existing;
    elpis_ecsg_fms *r;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    *out = NULL;
    if (owned == NULL || capacity == 0u || capacity > SIZE_MAX / sizeof(state_slot)) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    fms_get_stats(owned, &existing);
    if (existing.objects != 0u || existing.inflight_ops != 0u || existing.pinned_bytes != 0u) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    r = (elpis_ecsg_fms *)calloc(1u, sizeof(*r));
    if (r == NULL) {
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    r->slots = (state_slot *)calloc(capacity, sizeof(*r->slots));
    if (r->slots == NULL || pthread_mutex_init(&r->mu, NULL) != 0) {
        free(r->slots);
        free(r);
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    r->fms = owned;
    r->capacity = capacity;
    *out = r;
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_destroy(elpis_ecsg_fms **runtime)
{
    elpis_ecsg_fms *r;
    size_t i;

    if (runtime == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    r = *runtime;
    if (r == NULL) {
        return ELPIS_ECSG_EXEC_OK;
    }
    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].id != 0u || r->slots[i].busy) {
            pthread_mutex_unlock(&r->mu);
            return ELPIS_ECSG_EXEC_BUSY;
        }
    }
    pthread_mutex_unlock(&r->mu);
    fms_destroy(r->fms);
    pthread_mutex_destroy(&r->mu);
    free(r->slots);
    free(r);
    *runtime = NULL;
    return ELPIS_ECSG_EXEC_OK;
}

static int install_snapshot(elpis_ecsg_fms *r,
                            const uint8_t key[32],
                            size_t dim,
                            size_t width,
                            size_t max_rows,
                            const uint8_t *snapshot,
                            size_t snapshot_bytes,
                            uint64_t epoch,
                            uint64_t *out)
{
    uint8_t logical[32];
    state_slot *s = NULL;
    fms_id object = 0u;
    size_t i;
    int rc;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    *out = 0u;
    if (r == NULL || key == NULL || snapshot == NULL || dim == 0u || width == 0u || max_rows == 0u ||
        elpis_ecsg_executor_workspace_bytes(dim, width, max_rows) == 0u) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    logical_identity(key, dim, width, logical);

    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].id != 0u && memcmp(logical, r->slots[i].info.logical_identity, 32u) == 0) {
            pthread_mutex_unlock(&r->mu);
            return ELPIS_ECSG_EXEC_INVALID;
        }
        if (r->slots[i].id == 0u && s == NULL) {
            s = &r->slots[i];
        }
    }
    if (s == NULL || r->issued == UINT64_MAX) {
        pthread_mutex_unlock(&r->mu);
        return ELPIS_ECSG_EXEC_CAPACITY;
    }

    rc = fms_register(r->fms, UINT32_C(0x45435347), (uint64_t)snapshot_bytes,
                      FMS_WARM, 0.0f, snapshot, &object);
    if (rc < 0) {
        pthread_mutex_unlock(&r->mu);
        return fm(rc);
    }

    memset(s, 0, sizeof(*s));
    s->id = ++r->issued;
    s->object = object;
    s->info.dim = dim;
    s->info.width = width;
    s->info.max_rows = max_rows;
    s->info.snapshot_bytes = snapshot_bytes;
    s->info.logical_bytes = snapshot_bytes;
    s->info.epoch = epoch;
    s->accumulated.max_rows = max_rows;
    memcpy(s->info.logical_identity, logical, 32u);
    *out = s->id;
    pthread_mutex_unlock(&r->mu);
    update_high_water(r);
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_register(elpis_ecsg_fms *r, const uint8_t key[32],
                            size_t dim, size_t width, size_t max_rows,
                            const double *w, uint64_t *out)
{
    elpis_ecsg_state *state = NULL;
    uint8_t *snapshot = NULL;
    size_t bytes;
    int rc;

    if (w == NULL) {
        if (out != NULL) *out = 0u;
        return ELPIS_ECSG_EXEC_INVALID;
    }
    rc = elpis_ecsg_state_create(dim, width, w, &state);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        if (out != NULL) *out = 0u;
        return rc;
    }
    bytes = elpis_ecsg_state_snapshot_size(state);
    snapshot = (uint8_t *)malloc(bytes);
    if (snapshot == NULL) {
        (void)elpis_ecsg_state_destroy(&state);
        if (out != NULL) *out = 0u;
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    rc = elpis_ecsg_state_snapshot_write(state, snapshot, bytes);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        rc = install_snapshot(r, key, dim, width, max_rows, snapshot, bytes, 0u, out);
    }
    free(snapshot);
    (void)elpis_ecsg_state_destroy(&state);
    return rc;
}

int elpis_ecsg_fms_restore(elpis_ecsg_fms *r, const uint8_t key[32],
                           const uint8_t *snapshot, size_t bytes,
                           size_t max_rows, uint64_t *out)
{
    elpis_ecsg_state *state = NULL;
    size_t dim;
    size_t width;
    uint64_t epoch;
    int rc;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    *out = 0u;
    rc = elpis_ecsg_state_snapshot_restore(snapshot, bytes, &state);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        return rc;
    }
    dim = elpis_ecsg_state_dim(state);
    width = elpis_ecsg_state_width(state);
    epoch = elpis_ecsg_state_epoch(state);
    rc = install_snapshot(r, key, dim, width, max_rows, snapshot, bytes, epoch, out);
    (void)elpis_ecsg_state_destroy(&state);
    return rc;
}

int elpis_ecsg_fms_close(elpis_ecsg_fms *r, uint64_t *id)
{
    state_slot *s;
    int rc;

    if (r == NULL || id == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (*id == 0u) {
        return ELPIS_ECSG_EXEC_OK;
    }
    rc = take(r, *id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        return rc;
    }
    if (s->lease != NULL || s->txn_exec != NULL) {
        give(r, s);
        return ELPIS_ECSG_EXEC_BUSY;
    }
    rc = fms_unregister(r->fms, s->object);
    if (rc != FMS_OK) {
        give(r, s);
        return fm(rc);
    }
    pthread_mutex_lock(&r->mu);
    memset(s, 0, sizeof(*s));
    pthread_mutex_unlock(&r->mu);
    *id = 0u;
    return ELPIS_ECSG_EXEC_OK;
}

static int fill_info(elpis_ecsg_fms *r, state_slot *s, elpis_ecsg_fms_info *out)
{
    fms_object_info o;
    elpis_ecsg_exec_stats active;
    int rc;

    rc = fms_query(r->fms, s->object, &o);
    if (rc != FMS_OK) {
        return fm(rc);
    }
    *out = s->info;
    out->object_handle = s->object;
    out->generation = s->generation;
    out->resident_authoritative_bytes = o.tier == FMS_COLD ? 0u : o.size_bytes;
    out->tier = o.tier;
    out->residency_state = o.state;
    out->cold_replica = o.cold_replica;
    out->lease_count = o.lease_count;
    out->transaction_open = s->txn_exec != NULL ? 1u : 0u;
    out->active_workspace_bytes = 0u;
    if (s->txn_exec != NULL && elpis_ecsg_executor_stats(s->txn_exec, &active) == ELPIS_ECSG_EXEC_OK) {
        out->active_workspace_bytes = active.workspace_bytes;
    }
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_inspect(elpis_ecsg_fms *r, uint64_t id, elpis_ecsg_fms_info *out)
{
    state_slot *s;
    int rc;

    if (out == NULL) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = fill_info(r, s, out);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_exec_stats(elpis_ecsg_fms *r, uint64_t id, elpis_ecsg_exec_stats *out)
{
    state_slot *s;
    int rc;

    if (out == NULL) return ELPIS_ECSG_EXEC_INVALID;
    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    combined_executor_stats(s, out);
    give(r, s);
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_stats(elpis_ecsg_fms *r, elpis_ecsg_fms_metrics *out)
{
    size_t i;

    if (r == NULL || out == NULL) return ELPIS_ECSG_EXEC_INVALID;
    pthread_mutex_lock(&r->mu);
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].busy) {
            pthread_mutex_unlock(&r->mu);
            return ELPIS_ECSG_EXEC_BUSY;
        }
    }
    memset(out, 0, sizeof(*out));
    for (i = 0u; i < r->capacity; ++i) {
        state_slot *s = &r->slots[i];
        if (s->id == 0u) continue;
        out->states += 1u;
        out->logical_bytes += s->info.logical_bytes;
        if (s->txn_exec != NULL) {
            elpis_ecsg_exec_stats x;
            if (elpis_ecsg_executor_stats(s->txn_exec, &x) == ELPIS_ECSG_EXEC_OK) {
                out->active_workspace_bytes += x.workspace_bytes;
            }
        }
    }
    fms_get_stats(r->fms, &out->residency);
    out->resident_authoritative_bytes = out->residency.tier_bytes[FMS_WARM] + out->residency.tier_bytes[FMS_HOT];
    out->resident_high_water = r->high_water;
    /* Exact lease count is collected from the state slots, not inferred from bytes. */
    out->leases = 0u;
    for (i = 0u; i < r->capacity; ++i) {
        if (r->slots[i].id != 0u && r->slots[i].lease != NULL) out->leases += 1u;
    }
    out->demotion_ns = r->demotion_ns;
    pthread_mutex_unlock(&r->mu);
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_pump(elpis_ecsg_fms *r)
{
    uint64_t start;
    int rc;

    if (r == NULL) return ELPIS_ECSG_EXEC_INVALID;
    start = now_ns();
    rc = fms_pump(r->fms);
    pthread_mutex_lock(&r->mu);
    r->demotion_ns += now_ns() - start;
    pthread_mutex_unlock(&r->mu);
    return fm(rc);
}

static int direct_begin(elpis_ecsg_fms *r, state_slot *s, unsigned mode,
                        elpis_ecsg_executor **e, int *transient)
{
    int rc = acquire_state(r, s, mode);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    if (s->txn_exec != NULL) {
        *e = s->txn_exec;
        *transient = 0;
        return ELPIS_ECSG_EXEC_OK;
    }
    rc = open_executor(s, e);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        (void)release_state(r, s);
        return rc;
    }
    *transient = 1;
    return ELPIS_ECSG_EXEC_OK;
}

static void direct_finish(elpis_ecsg_fms *r, state_slot *s,
                          elpis_ecsg_executor **e, int transient)
{
    if (transient && e != NULL && *e != NULL) {
        merge_executor_stats(r, s, *e);
        (void)elpis_ecsg_executor_destroy(e);
    }
    if (s->txn_exec == NULL) {
        (void)release_state(r, s);
    }
}

int elpis_ecsg_fms_reserve(elpis_ecsg_fms *r, uint64_t id, size_t rows)
{
    state_slot *s;
    int rc;

    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    if (s->txn_exec != NULL) {
        give(r, s);
        return ELPIS_ECSG_EXEC_BUSY;
    }
    if (rows == 0u || elpis_ecsg_executor_workspace_bytes(s->info.dim, s->info.width, rows) == 0u) {
        give(r, s);
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (rows > s->info.max_rows) {
        s->info.max_rows = rows;
        s->accumulated.max_rows = rows;
    }
    give(r, s);
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_forward(elpis_ecsg_fms *r, uint64_t id,
                           const double *x, size_t rows, double *out)
{
    state_slot *s;
    elpis_ecsg_executor *e = NULL;
    uint64_t start;
    int transient = 0;
    int rc;

    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = direct_begin(r, s, FMS_READ, &e, &transient);
    start = now_ns();
    if (rc == ELPIS_ECSG_EXEC_OK) rc = elpis_ecsg_executor_forward(e, x, rows, out);
    s->info.query_ns += now_ns() - start;
    direct_finish(r, s, &e, transient);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_copy_w(elpis_ecsg_fms *r, uint64_t id, double *out, size_t count)
{
    state_slot *s;
    elpis_ecsg_executor *e = NULL;
    int transient = 0;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = direct_begin(r, s, FMS_READ, &e, &transient);
    if (rc == ELPIS_ECSG_EXEC_OK) rc = elpis_ecsg_executor_copy_w(e, out, count);
    direct_finish(r, s, &e, transient);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_project_s3(elpis_ecsg_fms *r, uint64_t id,
                              double *mu, double *m, double *t3)
{
    state_slot *s;
    elpis_ecsg_executor *e = NULL;
    int transient = 0;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = direct_begin(r, s, FMS_READ, &e, &transient);
    if (rc == ELPIS_ECSG_EXEC_OK) rc = elpis_ecsg_executor_project_s3(e, mu, m, t3);
    direct_finish(r, s, &e, transient);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_snapshot_write(elpis_ecsg_fms *r, uint64_t id, uint8_t *out, size_t size)
{
    state_slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    if (out == NULL || size < s->info.snapshot_bytes) {
        give(r, s);
        return ELPIS_ECSG_EXEC_INVALID;
    }
    rc = acquire_state(r, s, FMS_READ);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        memcpy(out, s->image, s->info.snapshot_bytes);
    }
    if (s->txn_exec == NULL) (void)release_state(r, s);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_learn_schedule(elpis_ecsg_fms *r, uint64_t id,
                                  const double *x, const double *y,
                                  const elpis_ecsg_drive *drives, size_t count,
                                  double rate, elpis_ecsg_exec_transition *transition)
{
    state_slot *s;
    elpis_ecsg_executor *e = NULL;
    uint8_t *publish = NULL;
    uint64_t generation_before;
    uint64_t start;
    uint64_t commit_start;
    int transient = 0;
    int rc;

    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = direct_begin(r, s, FMS_WRITE, &e, &transient);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        give(r, s);
        return rc;
    }
    publish = s->txn_exec != NULL ? s->txn_publish : (uint8_t *)malloc(s->info.snapshot_bytes);
    if (publish == NULL) {
        direct_finish(r, s, &e, transient);
        give(r, s);
        return ELPIS_ECSG_EXEC_NOMEM;
    }
    generation_before = s->generation;
    start = now_ns();
    rc = elpis_ecsg_executor_learn_schedule(e, x, y, drives, count, rate, transition);
    s->info.learn_ns += now_ns() - start;
    if (rc == ELPIS_ECSG_EXEC_OK) {
        commit_start = now_ns();
        rc = publish_executor(s, e, publish);
        s->info.commit_ns += now_ns() - commit_start;
        if (rc == ELPIS_ECSG_EXEC_OK) {
            s->generation += 1u;
            s->info.commits += 1u;
            if (transition != NULL) {
                transition->generation_before = generation_before;
                transition->generation_after = s->generation;
            }
        }
    }
    if (s->txn_exec == NULL) free(publish);
    direct_finish(r, s, &e, transient);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_learn(elpis_ecsg_fms *r, uint64_t id,
                         const double *x, const double *y, size_t rows,
                         double rate, uint64_t steps,
                         elpis_ecsg_exec_transition *transition)
{
    elpis_ecsg_drive drive;
    drive.rows = rows;
    drive.steps = steps;
    return elpis_ecsg_fms_learn_schedule(r, id, x, y, &drive, 1u, rate, transition);
}

static int txn_validate(state_slot *s, uint64_t token)
{
    if (s->txn_exec == NULL || token == 0u || token != s->txn_external_token) {
        return ELPIS_ECSG_EXEC_INVALID;
    }
    if (s->generation != s->txn_source_generation) {
        (void)elpis_ecsg_executor_txn_abort(s->txn_exec, s->txn_native_token);
        return ELPIS_ECSG_EXEC_STALE;
    }
    return ELPIS_ECSG_EXEC_OK;
}

int elpis_ecsg_fms_txn_begin(elpis_ecsg_fms *r, uint64_t id, uint64_t *token)
{
    state_slot *s;
    elpis_ecsg_executor *e = NULL;
    uint64_t native_token = 0u;
    int rc;

    if (token == NULL) return ELPIS_ECSG_EXEC_INVALID;
    *token = 0u;
    rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    if (s->txn_exec != NULL || s->txn_tokens_issued == UINT64_MAX) {
        give(r, s);
        return s->txn_exec != NULL ? ELPIS_ECSG_EXEC_BUSY : ELPIS_ECSG_EXEC_CAPACITY;
    }
    rc = acquire_state(r, s, FMS_WRITE);
    if (rc == ELPIS_ECSG_EXEC_OK) rc = open_executor(s, &e);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        s->txn_publish = (uint8_t *)malloc(s->info.snapshot_bytes);
        if (s->txn_publish == NULL) rc = ELPIS_ECSG_EXEC_NOMEM;
    }
    if (rc == ELPIS_ECSG_EXEC_OK) rc = elpis_ecsg_executor_txn_begin(e, &native_token);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        if (e != NULL) {
            merge_executor_stats(r, s, e);
            (void)elpis_ecsg_executor_destroy(&e);
        }
        free(s->txn_publish);
        s->txn_publish = NULL;
        (void)release_state(r, s);
        give(r, s);
        return rc;
    }
    s->txn_exec = e;
    s->txn_native_token = native_token;
    s->txn_external_token = ++s->txn_tokens_issued;
    s->txn_source_generation = s->generation;
    *token = s->txn_external_token;
    give(r, s);
    return ELPIS_ECSG_EXEC_OK;
}

static int txn_stale_cleanup(elpis_ecsg_fms *r, state_slot *s, int rc)
{
    if (rc == ELPIS_ECSG_EXEC_STALE) {
        s->accumulated.stale_refusals += 1u;
        clear_transaction(r, s, 1);
    }
    return rc;
}

int elpis_ecsg_fms_txn_learn_schedule(elpis_ecsg_fms *r, uint64_t id, uint64_t token,
                                      const double *x, const double *y,
                                      const elpis_ecsg_drive *drives, size_t count,
                                      double rate, elpis_ecsg_exec_transition *transition)
{
    state_slot *s;
    uint64_t start;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = txn_validate(s, token);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        rc = txn_stale_cleanup(r, s, rc);
        give(r, s);
        return rc;
    }
    start = now_ns();
    rc = elpis_ecsg_executor_txn_learn_schedule(s->txn_exec, s->txn_native_token,
                                                x, y, drives, count, rate, transition);
    s->info.learn_ns += now_ns() - start;
    if (transition != NULL) {
        transition->generation_before = s->generation;
        transition->generation_after = s->generation;
    }
    if (rc == ELPIS_ECSG_EXEC_NONFINITE) {
        clear_transaction(r, s, 1);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_txn_learn(elpis_ecsg_fms *r, uint64_t id, uint64_t token,
                             const double *x, const double *y, size_t rows,
                             double rate, uint64_t steps,
                             elpis_ecsg_exec_transition *transition)
{
    elpis_ecsg_drive drive;
    drive.rows = rows;
    drive.steps = steps;
    return elpis_ecsg_fms_txn_learn_schedule(r, id, token, x, y, &drive, 1u, rate, transition);
}

int elpis_ecsg_fms_txn_forward(elpis_ecsg_fms *r, uint64_t id, uint64_t token,
                               const double *x, size_t rows, double *out)
{
    state_slot *s;
    uint64_t start;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = txn_validate(s, token);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        start = now_ns();
        rc = elpis_ecsg_executor_txn_forward(s->txn_exec, s->txn_native_token, x, rows, out);
        s->info.query_ns += now_ns() - start;
    } else {
        rc = txn_stale_cleanup(r, s, rc);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_txn_project_s3(elpis_ecsg_fms *r, uint64_t id, uint64_t token,
                                  double *mu, double *m, double *t3)
{
    state_slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = txn_validate(s, token);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        rc = elpis_ecsg_executor_txn_project_s3(s->txn_exec, s->txn_native_token, mu, m, t3);
    } else {
        rc = txn_stale_cleanup(r, s, rc);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_txn_epoch(elpis_ecsg_fms *r, uint64_t id, uint64_t token, uint64_t *epoch)
{
    state_slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = txn_validate(s, token);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        rc = elpis_ecsg_executor_txn_epoch(s->txn_exec, s->txn_native_token, epoch);
    } else {
        rc = txn_stale_cleanup(r, s, rc);
    }
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_txn_commit(elpis_ecsg_fms *r, uint64_t id, uint64_t token,
                              elpis_ecsg_exec_transition *transition)
{
    state_slot *s;
    elpis_ecsg_exec_transition t;
    uint64_t generation_before;
    uint64_t start;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    rc = txn_validate(s, token);
    if (rc != ELPIS_ECSG_EXEC_OK) {
        rc = txn_stale_cleanup(r, s, rc);
        give(r, s);
        return rc;
    }
    memset(&t, 0, sizeof(t));
    generation_before = s->generation;
    start = now_ns();
    rc = elpis_ecsg_executor_txn_commit(s->txn_exec, s->txn_native_token, &t);
    if (rc == ELPIS_ECSG_EXEC_OK && t.steps != 0u) {
        rc = publish_executor(s, s->txn_exec, s->txn_publish);
        if (rc == ELPIS_ECSG_EXEC_OK) {
            s->generation += 1u;
            s->info.commits += 1u;
            t.generation_before = generation_before;
            t.generation_after = s->generation;
        }
    } else if (rc == ELPIS_ECSG_EXEC_OK) {
        t.generation_before = generation_before;
        t.generation_after = generation_before;
    }
    if (transition != NULL) *transition = t;
    s->info.commit_ns += now_ns() - start;
    clear_transaction(r, s, rc == ELPIS_ECSG_EXEC_OK ? 0 : 1);
    give(r, s);
    return rc;
}

int elpis_ecsg_fms_txn_abort(elpis_ecsg_fms *r, uint64_t id, uint64_t token)
{
    state_slot *s;
    int rc = take(r, id, &s);
    if (rc != ELPIS_ECSG_EXEC_OK) return rc;
    if (s->txn_exec == NULL) {
        give(r, s);
        return ELPIS_ECSG_EXEC_OK;
    }
    if (token != s->txn_external_token) {
        give(r, s);
        return ELPIS_ECSG_EXEC_INVALID;
    }
    rc = elpis_ecsg_executor_txn_abort(s->txn_exec, s->txn_native_token);
    if (rc == ELPIS_ECSG_EXEC_OK) {
        clear_transaction(r, s, 1);
    }
    give(r, s);
    return rc;
}
