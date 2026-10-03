/* Native verified file-asset page service (elpis/fms_file_service.h).
 *
 * Runtime mirror of elpis.substrate.file_assets.FMSFileAssets._load/acquire/
 * evict: identical stamp checks, bounded pread loop, page digest verification
 * before registration, LRU victim choice among unleased pages, lease rollback
 * and telemetry definitions. Cold admission/authority stays in Python.
 */
#define _POSIX_C_SOURCE 200809L
#include "elpis/fms_file_service.h"
#include "elpis/fms.h"
#include "elpis/fms_pal_posix.h"
#include "elpis/sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PAGE_KIND 0x50414745u /* same object kind as the Python bridge */
#define NIL UINT32_MAX

typedef struct {
    int fd;
    uint32_t page_size;
    uint64_t size, page_count;
    elpis_file_stamp stamp;
    uint8_t *digests;
} asset_t;

typedef struct {
    uint32_t asset, pins, prev, next;
    uint64_t page;
    fms_id oid;
    int used;
} page_t;

typedef struct { fms_lease *lease; const uint8_t *ptr; uint32_t slot; uint64_t start, size; } part_t;

typedef struct {
    uint64_t generation, length;
    part_t *parts;
    uint32_t count;
    int live;
} range_t;

struct elpis_file_service {
    pthread_mutex_t mu;
    atomic_uint_fast64_t interrupts;
    elpis_file_service_config_v1 cfg;
    fms_ctx *fms;
    asset_t *assets;
    uint32_t asset_count;
    page_t *pages;          /* max_pages slots */
    uint32_t *table;        /* open addressing: slot + 1, 0 empty */
    uint64_t mask;
    uint32_t lru_head, lru_tail, free_head, resident;
    range_t *ranges;
    uint8_t *scratch;       /* one page of pread staging */
    uint64_t scratch_bytes;
    elpis_file_service_stats_v1 st;
};

#ifdef ELPIS_FILE_SERVICE_TESTING
static elpis_file_service_test_hooks hooks;
void elpis_file_service_set_test_hooks(const elpis_file_service_test_hooks *h) {
    if (h) hooks = *h; else memset(&hooks, 0, sizeof hooks);
}
static void *svc_alloc(size_t n) {
    if (hooks.fail_alloc_after && --hooks.fail_alloc_after == 0) return NULL;
    return malloc(n);
}
static long svc_pread(int fd, void *buf, size_t n, int64_t off) {
    return hooks.pread ? hooks.pread(fd, buf, n, off) : (long)pread(fd, buf, n, (off_t)off);
}
#else
#define svc_alloc malloc
static long svc_pread(int fd, void *buf, size_t n, int64_t off) { return (long)pread(fd, buf, n, (off_t)off); }
#endif

static uint64_t mono(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static void stamp_of(const struct stat *s, elpis_file_stamp *out) {
    out->dev = (uint64_t)s->st_dev; out->ino = (uint64_t)s->st_ino; out->size = (uint64_t)s->st_size;
    out->mtime_ns = (int64_t)s->st_mtim.tv_sec * 1000000000 + s->st_mtim.tv_nsec;
    out->ctime_ns = (int64_t)s->st_ctim.tv_sec * 1000000000 + s->st_ctim.tv_nsec;
}
static int stamp_equal(const elpis_file_stamp *a, const elpis_file_stamp *b) {
    return a->dev == b->dev && a->ino == b->ino && a->size == b->size &&
           a->mtime_ns == b->mtime_ns && a->ctime_ns == b->ctime_ns;
}
static int stamp_current(const asset_t *a) {
    struct stat s; elpis_file_stamp now;
    if (fstat(a->fd, &s) != 0) return 0;
    stamp_of(&s, &now);
    return stamp_equal(&now, &a->stamp);
}

/* elpis.inference.raw-bytes.r0: SHA-256 of the canonical JSON byte wrapper. */
void elpis_file_service_raw_digest(const void *data, size_t n, uint8_t out[32]) {
    static const char prefix[] = "elpis.inference.raw-bytes.r0\0{\"__bytes__\":\"";
    static const char hex[] = "0123456789abcdef";
    const uint8_t *p = data;
    uint8_t buf[4096];
    elpis_sha256_ctx ctx;
    elpis_sha256_init(&ctx);
    elpis_sha256_update(&ctx, prefix, sizeof prefix - 1);
    for (size_t i = 0; i < n;) {
        size_t k = 0;
        for (; i < n && k < sizeof buf; ++i) { buf[k++] = (uint8_t)hex[p[i] >> 4]; buf[k++] = (uint8_t)hex[p[i] & 15]; }
        elpis_sha256_update(&ctx, buf, k);
    }
    elpis_sha256_update(&ctx, "\"}", 2);
    elpis_sha256_final(&ctx, out);
}

static elpis_file_status from_fms(int rc) {
    switch (rc) {
    case FMS_E_NOMEM: case FMS_E_LIMIT: return ELPIS_FILE_LIMIT;
    case FMS_E_NOTFOUND: return ELPIS_FILE_MISSING;
    case FMS_E_BUSY: return ELPIS_FILE_BUSY;
    case FMS_E_UNSUPPORTED: return ELPIS_FILE_UNSUPPORTED;
    case FMS_E_IO: return ELPIS_FILE_IO;
    case FMS_E_DIGEST: return ELPIS_FILE_INTEGRITY;
    case FMS_E_DEVICE: return ELPIS_FILE_DEVICE;
    default: return ELPIS_FILE_INVALID;
    }
}

/* ---- page table: hash (asset, page) -> slot, plus intrusive LRU list ------ */
static uint64_t hash_key(uint32_t asset, uint64_t page) {
    uint64_t h = page * UINT64_C(0x9e3779b97f4a7c15) ^ ((uint64_t)asset * UINT64_C(0xc2b2ae3d27d4eb4f));
    return h ^ (h >> 29);
}
static uint32_t find_slot(const elpis_file_service *s, uint32_t asset, uint64_t page) {
    for (uint64_t i = hash_key(asset, page) & s->mask;; i = (i + 1) & s->mask) {
        uint32_t v = s->table[i];
        if (!v) return NIL;
        const page_t *p = &s->pages[v - 1];
        if (p->asset == asset && p->page == page) return v - 1;
    }
}
static void table_insert(elpis_file_service *s, uint32_t slot) {
    uint64_t i = hash_key(s->pages[slot].asset, s->pages[slot].page) & s->mask;
    while (s->table[i]) i = (i + 1) & s->mask;
    s->table[i] = slot + 1;
}
static void table_remove(elpis_file_service *s, uint32_t slot) {
    uint64_t i = hash_key(s->pages[slot].asset, s->pages[slot].page) & s->mask;
    while (s->table[i] != slot + 1) i = (i + 1) & s->mask;
    s->table[i] = 0;
    /* Backward-shift deletion keeps every probe chain contiguous. */
    for (uint64_t j = (i + 1) & s->mask; s->table[j]; j = (j + 1) & s->mask) {
        const page_t *p = &s->pages[s->table[j] - 1];
        uint64_t home = hash_key(p->asset, p->page) & s->mask;
        if (((j - home) & s->mask) >= ((j - i) & s->mask)) { s->table[i] = s->table[j]; s->table[j] = 0; i = j; }
    }
}
static void lru_unlink(elpis_file_service *s, uint32_t slot) {
    page_t *p = &s->pages[slot];
    if (p->prev != NIL) s->pages[p->prev].next = p->next; else s->lru_head = p->next;
    if (p->next != NIL) s->pages[p->next].prev = p->prev; else s->lru_tail = p->prev;
    p->prev = p->next = NIL;
}
static void lru_append(elpis_file_service *s, uint32_t slot) {
    page_t *p = &s->pages[slot];
    p->prev = s->lru_tail; p->next = NIL;
    if (s->lru_tail != NIL) s->pages[s->lru_tail].next = slot; else s->lru_head = slot;
    s->lru_tail = slot;
}
static void refresh_residency(elpis_file_service *s) {
    fms_stats f;
    fms_get_stats(s->fms, &f);
    s->st.resident_bytes = f.tier_bytes[FMS_HOT] + f.tier_bytes[FMS_WARM];
    s->st.pinned_bytes = f.pinned_bytes;
    s->st.resident_pages = s->resident;
    if (s->st.resident_bytes > s->st.resident_high_water) s->st.resident_high_water = s->st.resident_bytes;
}
static elpis_file_status drop_page(elpis_file_service *s, uint32_t slot) {
    int rc = fms_unregister(s->fms, s->pages[slot].oid);
    if (rc != FMS_OK) return from_fms(rc);
    table_remove(s, slot);
    lru_unlink(s, slot);
    s->pages[slot].used = 0;
    s->pages[slot].next = s->free_head; s->free_head = slot;
    s->resident--; s->st.evictions++;
    return ELPIS_FILE_OK;
}

/* Caller holds the lock. Mirrors FMSFileAssets._load exactly. */
static elpis_file_status load(elpis_file_service *s, uint32_t asset, uint64_t page, uint32_t *out) {
    asset_t *a = &s->assets[asset];
    if (!stamp_current(a)) { s->st.integrity_failures++; return ELPIS_FILE_INTEGRITY; } /* changed backing file */
    uint32_t slot = find_slot(s, asset, page);
    if (slot != NIL) {
        s->st.hits++;
        lru_unlink(s, slot); lru_append(s, slot);
        *out = slot;
        return ELPIS_FILE_OK;
    }
    s->st.misses++;
    uint64_t start = page * a->page_size, count = a->size - start < a->page_size ? a->size - start : a->page_size;
    uint64_t begin = mono(), got = 0;
    while (got < count) {
        long n = svc_pread(a->fd, s->scratch + got, (size_t)(count - got), (int64_t)(start + got));
        if (n < 0 && errno == EINTR) { s->st.interrupted_reads++; continue; }
        if (n <= 0) { s->st.io_failures++; s->st.read_ns += mono() - begin; return ELPIS_FILE_IO; } /* short read / error */
        s->st.pread_bytes += (uint64_t)n; s->st.reads++;
        got += (uint64_t)n;
    }
    s->st.read_ns += mono() - begin;
    begin = mono();
    uint8_t digest[32];
    elpis_file_service_raw_digest(s->scratch, (size_t)count, digest);
    int good = !memcmp(digest, a->digests + page * 32, 32) && stamp_current(a);
    s->st.integrity_ns += mono() - begin;
    if (!good) { s->st.integrity_failures++; return ELPIS_FILE_INTEGRITY; }
    if (4 * count > s->st.staging_high_water) s->st.staging_high_water = 4 * count;
    begin = mono();
    fms_id oid = 0;
    int rc;
    while ((rc = fms_register(s->fms, PAGE_KIND, count, FMS_WARM, 0.0f, s->scratch, &oid)) < 0) {
        if (rc != FMS_E_NOMEM && rc != FMS_E_LIMIT && rc != FMS_E_UNSUPPORTED) {
            s->st.staging_ns += mono() - begin; return from_fms(rc);
        }
        uint32_t victim = s->lru_head;
        while (victim != NIL && s->pages[victim].pins) victim = s->pages[victim].next;
        if (victim == NIL) { s->st.staging_ns += mono() - begin; return ELPIS_FILE_LIMIT; } /* all leased */
        elpis_file_status ev = drop_page(s, victim);
        if (ev != ELPIS_FILE_OK) { s->st.staging_ns += mono() - begin; return ev; }
    }
    s->st.staging_ns += mono() - begin;
    if (s->free_head == NIL) { /* FMS max_objects equals max_pages; cannot happen */
        fms_unregister(s->fms, oid); return ELPIS_FILE_LIMIT;
    }
    slot = s->free_head; s->free_head = s->pages[slot].next;
    s->pages[slot] = (page_t){asset, 0, NIL, NIL, page, oid, 1};
    table_insert(s, slot); lru_append(s, slot);
    s->resident++;
    refresh_residency(s);
    *out = slot;
    return ELPIS_FILE_OK;
}

static void unlease(elpis_file_service *s, part_t *parts, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        fms_lease_release(s->fms, parts[i].lease);
        s->pages[parts[i].slot].pins--;
        s->st.lease_releases++;
    }
}

uint32_t elpis_file_service_abi_version(void) { return ELPIS_FILE_SERVICE_ABI_V1; }

elpis_file_status elpis_file_service_create(const elpis_file_service_config_v1 *c, elpis_file_service **out) {
    if (!c || !out || c->abi_version != ELPIS_FILE_SERVICE_ABI_V1 || c->reserved || !c->warm_bytes ||
        !c->staging_bytes || !c->map_bytes || !c->max_pages || c->max_pages > (1u << 24) || !c->max_assets ||
        c->max_assets > 65536 || !c->max_ranges || c->max_ranges > 4096 ||
        (c->absent_policy != FMS_FOLD_DOWN && c->absent_policy != FMS_REJECT)) return ELPIS_FILE_INVALID;
    *out = NULL;
    uint64_t scratch = c->staging_bytes / 4 < c->warm_bytes ? c->staging_bytes / 4 : c->warm_bytes;
    if (scratch > (UINT64_C(16) << 20)) scratch = UINT64_C(16) << 20;
    if (!scratch) return ELPIS_FILE_LIMIT;
    uint64_t cap = 1;
    while (cap < 2 * (uint64_t)c->max_pages) cap <<= 1;
    elpis_file_service *s = calloc(1, sizeof *s);
    if (!s) return ELPIS_FILE_LIMIT;
    s->cfg = *c; s->mask = cap - 1; s->scratch_bytes = scratch;
    s->assets = calloc(c->max_assets, sizeof *s->assets);
    s->pages = calloc(c->max_pages, sizeof *s->pages);
    s->table = calloc(cap, sizeof *s->table);
    s->ranges = calloc(c->max_ranges, sizeof *s->ranges);
    s->scratch = svc_alloc((size_t)scratch);
    fms_pal *pal = s->assets && s->pages && s->table && s->ranges && s->scratch ? fms_pal_posix_create_ram_only() : NULL;
    if (pal) {
        fms_config cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.tier_budget[FMS_WARM] = c->warm_bytes;
        cfg.domain_ceiling[FMS_DOM_RAM] = c->warm_bytes;
        cfg.high_wm = 0.95f; cfg.low_wm = 0.75f;
        cfg.max_objects = c->max_pages;
        cfg.hot_absent_policy = (uint8_t)c->absent_policy;
        cfg.cold_absent_policy = FMS_REJECT;
        s->fms = fms_create(&cfg, pal);
        if (!s->fms) pal->destroy(pal->self);
    }
    if (!s->fms || pthread_mutex_init(&s->mu, NULL) != 0) {
        if (s->fms) fms_destroy(s->fms);
        free(s->assets); free(s->pages); free(s->table); free(s->ranges); free(s->scratch); free(s);
        return ELPIS_FILE_LIMIT;
    }
    atomic_init(&s->interrupts, 0);
    s->lru_head = s->lru_tail = NIL;
    for (uint32_t i = 0; i < c->max_pages; ++i) s->pages[i].next = i + 1 < c->max_pages ? i + 1 : NIL;
    s->free_head = 0;
    s->st.warm_budget = c->warm_bytes; s->st.staging_budget = c->staging_bytes; s->st.max_pages = c->max_pages;
    *out = s;
    return ELPIS_FILE_OK;
}

elpis_file_status elpis_file_service_admit(elpis_file_service *s, const elpis_file_asset_v1 *a, uint32_t *out) {
    if (!s || !a || !out || a->abi_version != ELPIS_FILE_SERVICE_ABI_V1 || a->reserved || a->fd < 0 ||
        !a->size || !a->page_size || a->page_size > (16u << 20) || !a->page_digests ||
        a->page_count != (a->size + a->page_size - 1) / a->page_size || a->stamp.size != a->size)
        return ELPIS_FILE_INVALID;
    pthread_mutex_lock(&s->mu);
    elpis_file_status rc = ELPIS_FILE_OK;
    uint64_t map = a->page_count * 32;
    if (a->page_size > s->scratch_bytes || a->page_size > s->cfg.warm_bytes) rc = ELPIS_FILE_LIMIT; /* four-copy bound */
    else if (s->asset_count == s->cfg.max_assets || a->page_count > s->cfg.map_bytes / 32 ||
             s->st.map_bytes > s->cfg.map_bytes - map) rc = ELPIS_FILE_LIMIT;
    int fd = -1;
    uint8_t *digests = NULL;
    if (rc == ELPIS_FILE_OK) {
        struct stat st; elpis_file_stamp now;
        int flags = fcntl(a->fd, F_GETFL);
        if (flags < 0) rc = ELPIS_FILE_INVALID; /* bad cold descriptor */
        else if ((flags & O_ACCMODE) != O_RDONLY) rc = ELPIS_FILE_UNSUPPORTED; /* never a writable replica */
        else if ((fd = fcntl(a->fd, F_DUPFD_CLOEXEC, 3)) < 0) rc = ELPIS_FILE_IO;
        else if (fstat(fd, &st) != 0) rc = ELPIS_FILE_IO;
        else if (!S_ISREG(st.st_mode)) rc = ELPIS_FILE_INTEGRITY; /* regular file required */
        else {
            stamp_of(&st, &now);
            if (!stamp_equal(&now, &a->stamp)) rc = ELPIS_FILE_INTEGRITY; /* changed since admission */
        }
    }
    if (rc == ELPIS_FILE_OK && !(digests = svc_alloc((size_t)map))) rc = ELPIS_FILE_LIMIT;
    if (rc == ELPIS_FILE_OK) {
        memcpy(digests, a->page_digests, (size_t)map);
        s->assets[s->asset_count] = (asset_t){fd, a->page_size, a->size, a->page_count, a->stamp, digests};
        *out = s->asset_count++;
        s->st.assets = s->asset_count; s->st.map_bytes += map;
    } else if (fd >= 0) close(fd);
    pthread_mutex_unlock(&s->mu);
    return rc;
}

elpis_file_status elpis_file_service_acquire(elpis_file_service *s, uint32_t asset, uint64_t offset,
                                             uint64_t length, uint64_t *range) {
    if (!s || !range) return ELPIS_FILE_INVALID;
    uint64_t epoch = atomic_load(&s->interrupts);
    pthread_mutex_lock(&s->mu);
    elpis_file_status rc = ELPIS_FILE_OK;
    if (asset >= s->asset_count) { pthread_mutex_unlock(&s->mu); return ELPIS_FILE_MISSING; }
    asset_t *a = &s->assets[asset];
    if (!length || offset > a->size || length > a->size - offset) { pthread_mutex_unlock(&s->mu); return ELPIS_FILE_INVALID; }
    uint64_t first = offset / a->page_size, last = (offset + length - 1) / a->page_size, n = last - first + 1;
    if (n > s->cfg.warm_bytes / a->page_size) { pthread_mutex_unlock(&s->mu); return ELPIS_FILE_LIMIT; }
    s->st.semantic_bytes += length;
    uint32_t r = 0;
    while (r < s->cfg.max_ranges && s->ranges[r].live) ++r;
    part_t *parts = r < s->cfg.max_ranges ? svc_alloc((size_t)n * sizeof *parts) : NULL;
    if (!parts) { pthread_mutex_unlock(&s->mu); return ELPIS_FILE_LIMIT; }
    uint32_t count = 0;
    for (uint64_t page = first; page <= last && rc == ELPIS_FILE_OK; ++page) {
        if (atomic_load(&s->interrupts) != epoch) { rc = ELPIS_FILE_BUSY; break; }
        uint32_t slot;
        rc = load(s, asset, page, &slot);
        if (rc != ELPIS_FILE_OK) break;
        fms_lease *lease = NULL;
        int got = fms_lease_acquire(s->fms, s->pages[slot].oid, FMS_WARM, FMS_READ, &lease);
        if (got != FMS_OK) { rc = from_fms(got); break; }
        int tier = fms_lease_tier(lease);
        if (tier != FMS_HOT && tier != FMS_WARM) {
            fms_lease_release(s->fms, lease); rc = ELPIS_FILE_INTEGRITY; break;
        }
        s->pages[slot].pins++; s->st.lease_acquires++;
        uint64_t base = page * a->page_size;
        uint64_t lo = offset > base ? offset - base : 0;
        uint64_t hi = (offset + length < base + a->page_size ? offset + length : base + a->page_size) - base;
        parts[count++] = (part_t){lease, fms_lease_ptr(lease), slot, lo, hi - lo};
    }
    if (rc != ELPIS_FILE_OK) { unlease(s, parts, count); free(parts); }
    else {
        range_t *x = &s->ranges[r];
        x->generation++; x->parts = parts; x->count = count; x->length = length; x->live = 1;
        *range = x->generation << 12 | r;
        s->st.range_acquires++; s->st.live_ranges++;
    }
    refresh_residency(s);
    pthread_mutex_unlock(&s->mu);
    return rc;
}

static range_t *range_find(elpis_file_service *s, uint64_t handle) {
    uint64_t r = handle & 4095;
    if (r >= s->cfg.max_ranges) return NULL;
    range_t *x = &s->ranges[r];
    return x->live && x->generation == handle >> 12 ? x : NULL;
}

elpis_file_status elpis_file_service_read(elpis_file_service *s, uint64_t handle, void *dst, uint64_t bytes) {
    if (!s || !dst) return ELPIS_FILE_INVALID;
    pthread_mutex_lock(&s->mu);
    range_t *x = range_find(s, handle);
    elpis_file_status rc = !x ? ELPIS_FILE_STALE : bytes != x->length ? ELPIS_FILE_INVALID : ELPIS_FILE_OK;
    if (rc == ELPIS_FILE_OK) {
        uint8_t *d = dst;
        for (uint32_t i = 0; i < x->count; ++i) { memcpy(d, x->parts[i].ptr + x->parts[i].start, x->parts[i].size); d += x->parts[i].size; }
    }
    pthread_mutex_unlock(&s->mu);
    return rc;
}

static void range_drop(elpis_file_service *s, range_t *x) {
    unlease(s, x->parts, x->count);
    free(x->parts); x->parts = NULL; x->count = 0; x->live = 0;
    s->st.range_releases++; s->st.live_ranges--;
}

elpis_file_status elpis_file_service_release(elpis_file_service *s, uint64_t handle) {
    if (!s) return ELPIS_FILE_INVALID;
    pthread_mutex_lock(&s->mu);
    range_t *x = range_find(s, handle);
    if (x) { range_drop(s, x); refresh_residency(s); }
    pthread_mutex_unlock(&s->mu);
    return x ? ELPIS_FILE_OK : ELPIS_FILE_STALE;
}

elpis_file_status elpis_file_service_copy(elpis_file_service *s, uint32_t asset, uint64_t offset,
                                          uint64_t length, void *dst) {
    uint64_t handle;
    elpis_file_status rc = elpis_file_service_acquire(s, asset, offset, length, &handle);
    if (rc != ELPIS_FILE_OK) return rc;
    rc = elpis_file_service_read(s, handle, dst, length);
    elpis_file_status released = elpis_file_service_release(s, handle);
    return rc != ELPIS_FILE_OK ? rc : released;
}

elpis_file_status elpis_file_service_evict(elpis_file_service *s, uint32_t asset) {
    if (!s) return ELPIS_FILE_INVALID;
    pthread_mutex_lock(&s->mu);
    elpis_file_status rc = ELPIS_FILE_OK;
    for (uint32_t i = s->lru_head; i != NIL; i = s->pages[i].next)
        if ((asset == UINT32_MAX || s->pages[i].asset == asset) && s->pages[i].pins) rc = ELPIS_FILE_BUSY;
    for (uint32_t i = s->lru_head; rc == ELPIS_FILE_OK && i != NIL;) {
        uint32_t next = s->pages[i].next;
        if (asset == UINT32_MAX || s->pages[i].asset == asset) rc = drop_page(s, i);
        i = next;
    }
    refresh_residency(s);
    pthread_mutex_unlock(&s->mu);
    return rc;
}

void elpis_file_service_interrupt(elpis_file_service *s) { if (s) atomic_fetch_add(&s->interrupts, 1); }

uint64_t elpis_file_service_release_all(elpis_file_service *s) {
    if (!s) return 0;
    pthread_mutex_lock(&s->mu);
    uint64_t n = 0;
    for (uint32_t r = 0; r < s->cfg.max_ranges; ++r) if (s->ranges[r].live) { range_drop(s, &s->ranges[r]); ++n; }
    s->st.forced_releases += n;
    refresh_residency(s);
    pthread_mutex_unlock(&s->mu);
    return n;
}

void elpis_file_service_stats(elpis_file_service *s, elpis_file_service_stats_v1 *out) {
    if (!s || !out) return;
    pthread_mutex_lock(&s->mu);
    refresh_residency(s);
    *out = s->st;
    pthread_mutex_unlock(&s->mu);
}

size_t elpis_file_service_pages(elpis_file_service *s, uint32_t *assets, uint64_t *pages, size_t capacity) {
    if (!s) return 0;
    pthread_mutex_lock(&s->mu);
    size_t n = 0;
    for (uint32_t i = s->lru_head; i != NIL; i = s->pages[i].next, ++n)
        if (n < capacity && assets && pages) { assets[n] = s->pages[i].asset; pages[n] = s->pages[i].page; }
    pthread_mutex_unlock(&s->mu);
    return n;
}

void elpis_file_service_destroy(elpis_file_service *s) {
    if (!s) return;
    elpis_file_service_release_all(s);
    pthread_mutex_lock(&s->mu);
    for (uint32_t i = s->lru_head; i != NIL;) { uint32_t next = s->pages[i].next; drop_page(s, i); i = next; }
    for (uint32_t i = 0; i < s->asset_count; ++i) { close(s->assets[i].fd); free(s->assets[i].digests); }
    pthread_mutex_unlock(&s->mu);
    fms_destroy(s->fms);
    pthread_mutex_destroy(&s->mu);
    free(s->assets); free(s->pages); free(s->table); free(s->ranges); free(s->scratch); free(s);
}
