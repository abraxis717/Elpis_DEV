/* vector_index.cpp - exact dense index over FMS-resident immutable shards.
 *
 * FMS contract enforced here:
 *   - shard bytes are verified BEFORE fms_register(), never after
 *   - the resident copy is the FMS object; this layer keeps no shadow payload
 *   - scoring holds fms_lease_acquire(FMS_WARM, FMS_READ) for its whole duration
 *   - every exit path releases the lease, including error paths
 *   - FMS_HOT is never passed to any FMS entry point in this translation unit
 *   - a promotion failure is a structured error, never an empty result
 *   - FMS_E_DIGEST from a corrupt cold replica surfaces as an integrity failure */

#include "elpis/vector_index.h"
#include "elpis/vector_execution.h"
#include "elpis/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <new>
#include <exception>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

namespace {

struct ShardEntry {
    fms_id   id = 0;
    size_t   bytes = 0;
    char     digest[65] = {0};
    elpis_vshard_header hdr{};
    /* Serialises residency work for THIS shard only. Without it, several
     * readers promote the same shard at once, each seeing the others'
     * FMS_ST_MOVING as FMS_E_BUSY, and under memory pressure a reader can
     * starve on the retry path. It covers lease acquisition only: the READ
     * lease then pins the shard WARM, so scoring runs outside the gate and
     * concurrent scans of one shard proceed in parallel. Lock order is always
     * ix->mu, then this. */
    std::shared_ptr<std::mutex> gate = std::make_shared<std::mutex>();
};

/* Two FMS statuses are transient under concurrency and are retried under a hard
 * bound; everything else, above all FMS_E_DIGEST, is returned immediately so a
 * real failure can never be hidden by waiting.
 *
 *   FMS_E_BUSY   the object is mid-move: another reader is promoting it, or the
 *                background pump is demoting it.
 *   FMS_E_LIMIT  no headroom right now because concurrent readers hold pins on
 *                other shards. Those pins are released when their scans finish.
 *
 * Sizing rule this implies, stated in docs/R2_MEMORY_ACCOUNTING.md: the WARM
 * ceiling must exceed (concurrent queries x largest shard), otherwise readers
 * serialise on the retry path and, in the limit, fail. A ceiling smaller than a
 * single shard fails immediately at admission instead, in fms_register. */
/* The wait is bounded by wall time, not by an attempt count: a transient move
 * costs one cold I/O round trip, and that is orders of magnitude slower under a
 * sanitizer or on the MacBook target than on an unsanitized workstation. A
 * fixed attempt count silently becomes a different timeout on every host. Five
 * seconds is long enough for a slow cold write and short enough that a genuine
 * deadlock still surfaces as a structured failure. */
const int      kBusyDeadlineMs = 5000;
const unsigned kBusyBackoffUs = 200;
/* On the shared execution pool a shard task never sleeps for this: it makes one
 * attempt per run and returns ELPIS_EXEC_DEFER, so the pool thread runs other
 * work until the task is re-run (>=1 ms later). The deadline is the same. */

/* Lease guard: releases on every path, including exceptions and early returns. */
class LeaseGuard {
public:
    LeaseGuard(fms_ctx *c, fms_id id) : ctx_(c), id_(id) {}
    ~LeaseGuard() { reset(); }
    LeaseGuard(const LeaseGuard &) = delete;
    LeaseGuard &operator=(const LeaseGuard &) = delete;

    /* One attempt. FMS_WARM only. HACF vector never requests FMS_HOT. */
    fms_status try_acquire(void **ptr) {
        fms_status r = fms_lease_acquire(ctx_, id_, FMS_WARM, FMS_READ, &lease_);
        if (r != FMS_OK) { lease_ = nullptr; return r; }
        *ptr = fms_lease_ptr(lease_);
        tier_ = fms_lease_tier(lease_);
        return FMS_OK;
    }
    /* Caller-thread path: retries BUSY/LIMIT with a short sleep until the deadline. */
    fms_status acquire(void **ptr) {
        fms_status r = FMS_E_BUSY;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(kBusyDeadlineMs);
        for (;;) {
            r = try_acquire(ptr);
            if (r != FMS_E_BUSY && r != FMS_E_LIMIT) break;
            if (std::chrono::steady_clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::microseconds(kBusyBackoffUs));
        }
        return r;
    }
    int tier() const { return tier_; }
    void reset() {
        if (lease_) { fms_lease_release(ctx_, lease_); lease_ = nullptr; }
    }
private:
    fms_ctx   *ctx_;
    fms_id     id_;
    fms_lease *lease_ = nullptr;
    int        tier_ = -1;
};

} // namespace

struct elpis_vector_index {
    fms_ctx                *fms = nullptr;
    elpis_embedding_profile profile{};
    char                    profile_digest[65] = {0};
    char                    corpus_digest[65] = {0};
    bool                    bind_corpus = false;
    std::vector<ShardEntry> shards;
    /* Readers (search, inspect, verify, list) take a shared lock; admission and
     * close take it exclusively. FMS keeps its own lock for residency. */
    mutable std::shared_mutex mu;   /* mutable: const read accessors still lock */
};

struct elpis_vector_executor {
    elpis_vector_index *index = nullptr;       /* borrowed; outlives executor */
    elpis_exec_runtime *runtime = nullptr;
    std::mutex search_mu;                      /* one ordered consumer */
    bool broken = false;
};

namespace {

/* Canonical digest text: exactly 64 lowercase hex characters then NUL, read
 * with a hard bound so an unterminated fixed-width array is never over-read. */
int canonical_digest(const char *d) {
    if (!d) return -1;
    for (int i = 0; i < 64; i++) {
        char c = d[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return -1;
    }
    return d[64] == '\0' ? 0 : -1;
}

/* Last-error state is per thread, errno-style. Searches run concurrently under
 * a shared lock, so a single slot inside the index would be a data race and
 * would also let one thread overwrite another thread's diagnosis. */
thread_local elpis_vec_error t_err = {ELPIS_VEC_OK, 0, {0}};

/* Records a structured failure. cause preserves the underlying fms_status so an
 * integrity failure remains identifiable as one all the way out. */
int set_err(elpis_vector_index *ix, int status, int cause, const char *detail) {
    (void)ix;
    t_err.status = status;
    t_err.cause = cause;
    std::snprintf(t_err.detail, sizeof t_err.detail, "%s", detail ? detail : "");
    return status;
}

int set_errf(elpis_vector_index *ix, int status, int cause, const char *fmt, ...) {
    (void)ix;
    t_err.status = status;
    t_err.cause = cause;
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(t_err.detail, sizeof t_err.detail, fmt ? fmt : "", ap);
    va_end(ap);
    return status;
}

void clear_err(elpis_vector_index *ix) {
    (void)ix;
    t_err.status = ELPIS_VEC_OK;
    t_err.cause = 0;
    t_err.detail[0] = 0;
}

/* No C++ exception may cross the C ABI. Every entry point below that allocates
 * a container or a string runs inside this guard, so std::bad_alloc becomes a
 * named vector-layer failure instead of std::terminate. Integrity errors are
 * returned as status codes, never thrown, so nothing here can swallow one. */
template <typename F>
int guard(elpis_vector_index *ix, F &&fn) {
    try {
        return fn();
    } catch (const std::bad_alloc &) {
        return set_err(ix, ELPIS_VEC_E_INTERNAL, 0, "allocation failed inside the vector layer");
    } catch (const std::exception &e) {
        return set_errf(ix, ELPIS_VEC_E_INTERNAL, 0, "internal exception: %s", e.what());
    } catch (...) {
        return set_err(ix, ELPIS_VEC_E_INTERNAL, 0, "internal exception: unknown type");
    }
}

/* Index search is deliberately restricted to L2-normalized profiles. This
 * bounds DOT and cosine scores to the canonical [-1,1] domain used by
 * score_key. The shard verifier admits a documented 1e-4 norm tolerance, so a
 * DOT product can overshoot by a similarly tiny amount; clamp only that
 * declared numerical envelope and reject anything larger. */
const double kScoreOvershoot = 2.0e-4;

bool canonical_index_score(double raw, double *out) {
    if (!out || !std::isfinite(raw)) return false;
    if (raw > 1.0) {
        if (raw > 1.0 + kScoreOvershoot) return false;
        raw = 1.0;
    } else if (raw < -1.0) {
        if (raw < -1.0 - kScoreOvershoot) return false;
        raw = -1.0;
    }
    *out = raw;
    return true;
}

/* Namespace and authority filters follow the corpus policy. An unknown class is
 * a caller error and is refused, never answered with an empty result. */
bool filter_valid_ns(const char *s) {
    if (!s) return true;
    size_t n = 0;
    while (n < 96 && s[n]) n++;
    if (n == 96 || n == 0) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch < 0x20 || ch == 0x7f) return false;
    }
    return true;
}

bool filter_valid_authority(const char *s) {
    return !s || !std::strcmp(s, "canonical") || !std::strcmp(s, "reference") ||
           !std::strcmp(s, "advisory") || !std::strcmp(s, "provisional");
}

/* An FMS residency failure keeps its identity: FMS_E_DIGEST is an integrity
 * failure, never "no results", "not found", "out of memory" or "try later". */
int residency_status(fms_status r) {
    return (r == FMS_E_DIGEST) ? ELPIS_VEC_E_INTEGRITY : ELPIS_VEC_E_RESIDENCY;
}

/* Streaming comparison of two sorted digest runs, both under lease. */
bool shares_chunk_digest(const uint8_t *a, uint64_t na, const uint8_t *b, uint64_t nb,
                         char dup_out[65]) {
    uint64_t i = 0, j = 0;
    while (i < na && j < nb) {
        const uint8_t *da = a + i * ELPIS_VSHARD_RECORD_BYTES;
        const uint8_t *db = b + j * ELPIS_VSHARD_RECORD_BYTES;
        int c = std::memcmp(da, db, 32);
        if (c == 0) { if (dup_out) elpis_hex32(da, dup_out); return true; }
        if (c < 0) i++; else j++;
    }
    return false;
}


struct ExecShardEnvelope {
    int32_t status;
    int32_t cause;
    uint32_t count;
    uint32_t reserved;
    char detail[192];
};

struct ExecShardContext {
    elpis_vector_index *index;
    const ShardEntry *shard;
    const float *query;
    uint32_t dimensions;
    uint32_t k;
    const char *ns_filter;
    const char *authority_filter;
    /* Set when a lease attempt is first refused (BUSY/LIMIT); the task is re-run
     * by the pool, one attempt per run, until this passes. Only the one thread
     * running the task touches it; the pool mutex orders successive runs. */
    std::chrono::steady_clock::time_point busy_deadline;
    bool busy;
};

int emit_exec_envelope(int status, int cause, const char *detail,
                       const std::vector<elpis_vector_hit> &hits,
                       size_t cap, elpis_exec_buffer **out) {
    if (!out) return ELPIS_EXEC_INVALID;
    *out = nullptr;
    size_t bytes = sizeof(ExecShardEnvelope) + hits.size() * sizeof(elpis_vector_hit);
    if (bytes > cap) return ELPIS_EXEC_INVALID;
    elpis_exec_buffer *b = elpis_exec_buffer_alloc(bytes);
    if (!b) return ELPIS_EXEC_INTERNAL;
    unsigned char *dst = static_cast<unsigned char *>(elpis_exec_buffer_mutable_data(b));
    if (!dst) { elpis_exec_buffer_release(b); return ELPIS_EXEC_INTERNAL; }
    ExecShardEnvelope env{};
    env.status = status;
    env.cause = cause;
    env.count = static_cast<uint32_t>(hits.size());
    std::snprintf(env.detail, sizeof env.detail, "%s", detail ? detail : "");
    std::memcpy(dst, &env, sizeof env);
    if (!hits.empty())
        std::memcpy(dst + sizeof env, hits.data(), hits.size() * sizeof(elpis_vector_hit));
    *out = b;
    return ELPIS_EXEC_OK;
}

elpis_exec_status vector_shard_compute(void *opaque, const elpis_exec_buffer *,
                                       size_t cap, elpis_exec_buffer **out) {
    ExecShardContext *ctx = static_cast<ExecShardContext *>(opaque);
    if (!ctx || !ctx->index || !ctx->shard || !ctx->query || !out)
        return ELPIS_EXEC_INVALID;
    try {
        elpis_vector_index *ix = ctx->index;
        const ShardEntry &s = *ctx->shard;
        void *p = nullptr;
        LeaseGuard g(ix->fms, s.id);
        fms_status r;
        {
            /* One promoter per shard. Never block a pool thread on it: another
             * task is promoting this shard (or verify is hashing it), run later. */
            std::unique_lock<std::mutex> shard_lk(*s.gate, std::try_to_lock);
            if (!shard_lk.owns_lock()) return ELPIS_EXEC_DEFER;
            r = g.try_acquire(&p);
        }
        if (r == FMS_E_BUSY || r == FMS_E_LIMIT) {
            const auto now = std::chrono::steady_clock::now();
            if (!ctx->busy) {
                ctx->busy = true;
                ctx->busy_deadline = now + std::chrono::milliseconds(kBusyDeadlineMs);
            }
            if (now < ctx->busy_deadline) return ELPIS_EXEC_DEFER;
        }
        if (r != FMS_OK) {
            std::vector<elpis_vector_hit> empty;
            char detail[192];
            std::snprintf(detail, sizeof detail,
                          "shard could not be made WARM-resident: %s%s", fms_strerror(r),
                          r == FMS_E_LIMIT
                              ? " (WARM ceiling must exceed concurrent queries x largest shard)"
                              : "");
            return static_cast<elpis_exec_status>(
                emit_exec_envelope(residency_status(r), r, detail, empty, cap, out));
        }
        if (g.tier() != FMS_WARM) {
            std::vector<elpis_vector_hit> empty;
            return static_cast<elpis_exec_status>(
                emit_exec_envelope(ELPIS_VEC_E_RESIDENCY, 0,
                                   "shard resident at an unexpected tier",
                                   empty, cap, out));
        }

        const uint8_t *recs = static_cast<const uint8_t *>(p) + ELPIS_VSHARD_HEADER_BYTES;
        const uint64_t n = s.hdr.vector_count;
        std::vector<double> scores(n);
        if (elpis_vector_score_block(recs, n, s.hdr.dimensions, ctx->query,
                                     s.hdr.metric, scores.data()) != 0) {
            std::vector<elpis_vector_hit> empty;
            return static_cast<elpis_exec_status>(
                emit_exec_envelope(ELPIS_VEC_E_QUERY, 0,
                                   "scoring rejected the query", empty, cap, out));
        }

        /* Every record is checked in index order exactly as the serial path does
         * (the first unreadable metadata entry or out-of-domain score fails the
         * shard), but only the local top-k are materialized as hits. The hit
         * order is elpis_vector_hit_compare: canonical key descending, then chunk
         * digest ascending. Lowercase hex preserves byte order, so comparing the
         * raw 32-byte digests yields the same total order. */
        struct Candidate { int64_t key; double score; uint64_t i; const char *ns, *au; };
        std::vector<Candidate> found;
        found.reserve(static_cast<size_t>(n));
        for (uint64_t i = 0; i < n; ++i) {
            const char *ns = "", *au = "";
            if (elpis_vshard_record_meta(p, s.bytes, i, &ns, &au) != 0) {
                std::vector<elpis_vector_hit> empty;
                return static_cast<elpis_exec_status>(
                    emit_exec_envelope(ELPIS_VEC_E_INTEGRITY, 0,
                                       "shard metadata map unreadable", empty, cap, out));
            }
            if (ctx->ns_filter && std::strcmp(ns, ctx->ns_filter) != 0) continue;
            if (ctx->authority_filter && std::strcmp(au, ctx->authority_filter) != 0) continue;
            double canonical_score = 0.0;
            if (!canonical_index_score(scores[i], &canonical_score)) {
                std::vector<elpis_vector_hit> empty;
                return static_cast<elpis_exec_status>(
                    emit_exec_envelope(ELPIS_VEC_E_INTEGRITY, 0,
                                       "scoring produced a non-finite or out-of-domain value",
                                       empty, cap, out));
            }
            found.push_back({elpis_vector_score_key(canonical_score), canonical_score, i, ns, au});
        }
        const size_t keep = std::min<size_t>(found.size(), ctx->k);
        std::partial_sort(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(keep), found.end(),
                          [recs](const Candidate &a, const Candidate &b) {
                              if (a.key != b.key) return a.key > b.key;
                              return std::memcmp(recs + a.i * ELPIS_VSHARD_RECORD_BYTES,
                                                 recs + b.i * ELPIS_VSHARD_RECORD_BYTES, 32) < 0;
                          });
        std::vector<elpis_vector_hit> local(keep);
        for (size_t j = 0; j < keep; ++j) {
            const Candidate &c = found[j];
            elpis_vector_hit &h = local[j];
            elpis_hex32(recs + c.i * ELPIS_VSHARD_RECORD_BYTES, h.chunk_digest);
            elpis_hex32(recs + c.i * ELPIS_VSHARD_RECORD_BYTES + 32, h.doc_digest);
            std::snprintf(h.shard_digest, sizeof h.shard_digest, "%s", s.digest);
            std::snprintf(h.embedding_profile_digest, sizeof h.embedding_profile_digest,
                          "%s", ix->profile_digest);
            std::snprintf(h.ns, sizeof h.ns, "%s", c.ns);
            std::snprintf(h.authority, sizeof h.authority, "%s", c.au);
            h.score = c.score;
            h.score_key = c.key;
        }
        return static_cast<elpis_exec_status>(
            emit_exec_envelope(ELPIS_VEC_OK, 0, "ok", local, cap, out));
    } catch (...) {
        return ELPIS_EXEC_INTERNAL;
    }
}

} // namespace

extern "C" {

int elpis_vector_index_create(fms_ctx *fms, const elpis_embedding_profile *profile,
                              const char *corpus_manifest_digest, elpis_vector_index **out) {
    if (!fms || !profile || !out) return ELPIS_VEC_E_INVAL;
    *out = nullptr;
    if (elpis_embedding_profile_validate(profile) != 0) return ELPIS_VEC_E_PROFILE;
    if (profile->normalization != ELPIS_NORM_L2) return ELPIS_VEC_E_PROFILE;
    elpis_vector_index *ix = nullptr;
    try {
        ix = new elpis_vector_index();
    } catch (...) {
        return ELPIS_VEC_E_INTERNAL;                 /* no exception crosses the C ABI */
    }
    ix->fms = fms;
    ix->profile = *profile;
    if (elpis_embedding_profile_digest(profile, ix->profile_digest) != 0) { delete ix; return ELPIS_VEC_E_PROFILE; }
    if (corpus_manifest_digest) {
        if (canonical_digest(corpus_manifest_digest) != 0) { delete ix; return ELPIS_VEC_E_INVAL; }
        std::snprintf(ix->corpus_digest, sizeof ix->corpus_digest, "%s", corpus_manifest_digest);
        ix->bind_corpus = true;
    }
    *out = ix;
    return 0;
}

void elpis_vector_index_destroy(elpis_vector_index *ix) {
    if (!ix) return;
    for (const ShardEntry &s : ix->shards) fms_unregister(ix->fms, s.id);
    delete ix;
}

const char *elpis_vector_index_error(const elpis_vector_index *ix) {
    return ix ? t_err.detail : "null index";
}

const elpis_vec_error *elpis_vector_index_last_error(const elpis_vector_index *ix) {
    return ix ? &t_err : nullptr;
}

static int add_shard_bytes_impl(elpis_vector_index *ix, const void *bytes, size_t len,
                                char shard_digest_out[65]) {
    if (!ix || !bytes) return ELPIS_VEC_E_INVAL;
    std::unique_lock<std::shared_mutex> lk(ix->mu);
    clear_err(ix);
    if (ix->shards.size() >= ELPIS_VINDEX_MAX_SHARDS)
        return set_err(ix, ELPIS_VEC_E_LIMIT, 0, "shard limit reached");

    /* 1. verify before admission */
    elpis_vshard_header hdr;
    char reason[64];
    if (elpis_vshard_verify(bytes, len, &hdr, reason, sizeof reason) != 0) {
        /* Structural and digest rejections are distinguished: a mangled header
         * is a format error, a bad digest is an integrity failure. */
        bool integrity = std::strstr(reason, "digest") != nullptr ||
                         std::strstr(reason, "non_finite") != nullptr ||
                         std::strstr(reason, "duplicate") != nullptr;
        return set_errf(ix, integrity ? ELPIS_VEC_E_INTEGRITY : ELPIS_VEC_E_FORMAT, 0,
                        "shard rejected: %s", reason);
    }
    if (std::strcmp(hdr.embedding_profile_digest, ix->profile_digest) != 0)
        return set_err(ix, ELPIS_VEC_E_PROFILE, 0, "shard rejected: embedding_profile_mismatch");
    if (ix->bind_corpus && std::strcmp(hdr.corpus_manifest_digest, ix->corpus_digest) != 0)
        return set_err(ix, ELPIS_VEC_E_PROFILE, 0, "shard rejected: corpus_manifest_mismatch");
    for (const ShardEntry &s : ix->shards)
        if (std::strcmp(s.digest, hdr.shard_digest) == 0)
            return set_err(ix, ELPIS_VEC_E_DUPLICATE, 0, "shard rejected: already_admitted");

    /* 2. duplicate chunk digests across shards are rejected, not deduplicated.
     * The comparison runs against the FMS-resident copies under lease. */
    for (const ShardEntry &s : ix->shards) {
        void *p = nullptr;
        std::lock_guard<std::mutex> shard_lk(*s.gate);
        LeaseGuard g(ix->fms, s.id);
        fms_status r = g.acquire(&p);
        if (r != FMS_OK)
            return set_errf(ix, residency_status(r), r,
                            "cannot inspect admitted shard: %s", fms_strerror(r));
        char dup[65];
        if (shares_chunk_digest((const uint8_t *)bytes + ELPIS_VSHARD_HEADER_BYTES, hdr.vector_count,
                                (const uint8_t *)p + ELPIS_VSHARD_HEADER_BYTES, s.hdr.vector_count,
                                dup))
            return set_errf(ix, ELPIS_VEC_E_DUPLICATE, 0,
                            "shard rejected: duplicate_chunk_digest %s", dup);
    }

    /* 3. hand the verified bytes to FMS; the index keeps no second copy */
    ShardEntry e;
    e.bytes = len;
    e.hdr = hdr;
    std::snprintf(e.digest, sizeof e.digest, "%s", hdr.shard_digest);
    fms_status r = fms_register(ix->fms, ELPIS_FMS_KIND_VECTOR_SHARD, (uint64_t)len,
                                FMS_WARM, 0.0f, bytes, &e.id);
    if (r < 0)
        return set_errf(ix, residency_status(r), r,
                        "fms_register failed: %s", fms_strerror(r));
    ix->shards.push_back(e);
    if (shard_digest_out) std::snprintf(shard_digest_out, 65, "%s", hdr.shard_digest);
    return ELPIS_VEC_OK;
}

static int add_shard_file_impl(elpis_vector_index *ix, const char *path,
                               char shard_digest_out[65]) {
    if (!ix || !path) return ELPIS_VEC_E_INVAL;
    void *bytes = nullptr;
    size_t len = 0;
    if (elpis_vshard_read_file(path, &bytes, &len) != 0)
        return set_err(ix, ELPIS_VEC_E_FORMAT, 0, "cannot read shard file");
    int rc = elpis_vector_index_add_shard_bytes(ix, bytes, len, shard_digest_out);
    std::free(bytes);
    return rc;
}

static int close_shard_impl(elpis_vector_index *ix, const char *shard_digest) {
    if (!ix || !shard_digest) return ELPIS_VEC_E_INVAL;
    if (canonical_digest(shard_digest) != 0)
        return set_err(ix, ELPIS_VEC_E_INVAL, 0, "close: malformed shard digest");
    std::unique_lock<std::shared_mutex> lk(ix->mu);
    for (size_t i = 0; i < ix->shards.size(); i++) {
        if (std::strcmp(ix->shards[i].digest, shard_digest) != 0) continue;
        fms_status r = fms_unregister(ix->fms, ix->shards[i].id);
        if (r != FMS_OK)
            return set_errf(ix, residency_status(r), r,
                            "fms_unregister: %s", fms_strerror(r));
        ix->shards.erase(ix->shards.begin() + (long)i);
        return ELPIS_VEC_OK;
    }
    return set_err(ix, ELPIS_VEC_E_NOTFOUND, 0, "no such shard");
}

int elpis_vector_index_close_shard(elpis_vector_index *ix, const char *shard_digest) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&] { return close_shard_impl(ix, shard_digest); });
}

int elpis_vector_index_profile_digest(const elpis_vector_index *ix, char profile_digest_out[65],
                                      char corpus_digest_out[65]) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(const_cast<elpis_vector_index *>(ix), [&] {
        std::shared_lock<std::shared_mutex> lk(ix->mu);
        if (profile_digest_out) std::snprintf(profile_digest_out, 65, "%s", ix->profile_digest);
        if (corpus_digest_out)
            std::snprintf(corpus_digest_out, 65, "%s",
                          ix->bind_corpus ? ix->corpus_digest
                                          : "0000000000000000000000000000000000000000000000000000000000000000");
        return ELPIS_VEC_OK;
    });
}

uint32_t elpis_vector_index_shard_count(const elpis_vector_index *ix) {
    if (!ix) return 0u;
    try {
        std::shared_lock<std::shared_mutex> lk(ix->mu);
        return (uint32_t)ix->shards.size();
    } catch (...) {
        return 0u;
    }
}

/* Caller must already hold ix->mu (shared or exclusive). Used by the manifest
 * writer, which takes the lock once instead of re-entering the public
 * accessors: std::shared_mutex is not recursive, so a public accessor calling
 * another public accessor would deadlock. */
int list_shards_locked(const elpis_vector_index *ix, char (*out)[65], uint32_t cap,
                              uint32_t *n_out) {
    std::vector<std::string> d;
    d.reserve(ix->shards.size());
    for (const ShardEntry &s : ix->shards) d.emplace_back(s.digest);
    std::sort(d.begin(), d.end());                    /* deterministic listing */
    uint32_t n = (uint32_t)d.size();
    if (out) {
        if (cap < n) return ELPIS_VEC_E_LIMIT;
        for (uint32_t i = 0; i < n; i++) std::snprintf(out[i], 65, "%s", d[i].c_str());
    }
    *n_out = n;
    return ELPIS_VEC_OK;
}

static int list_shards_impl(const elpis_vector_index *ix, char (*out)[65], uint32_t cap,
                            uint32_t *n_out) {
    if (!ix || !n_out) return ELPIS_VEC_E_INVAL;
    std::shared_lock<std::shared_mutex> lk(ix->mu);
    return list_shards_locked(ix, out, cap, n_out);
}

/* Caller must already hold ix->mu. */
int inspect_locked(const elpis_vector_index *ix, const char *shard_digest,
                          elpis_vshard_header *out) {
    for (const ShardEntry &s : ix->shards)
        if (std::strcmp(s.digest, shard_digest) == 0) { *out = s.hdr; return ELPIS_VEC_OK; }
    return ELPIS_VEC_E_NOTFOUND;
}

int elpis_vector_index_inspect(elpis_vector_index *ix, const char *shard_digest,
                               elpis_vshard_header *out) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&]() -> int {
        if (!shard_digest || !out) return ELPIS_VEC_E_INVAL;
        if (canonical_digest(shard_digest) != 0)
            return set_err(ix, ELPIS_VEC_E_INVAL, 0, "inspect: malformed shard digest");
        std::shared_lock<std::shared_mutex> lk(ix->mu);
        for (const ShardEntry &s : ix->shards)
            if (std::strcmp(s.digest, shard_digest) == 0) { *out = s.hdr; return ELPIS_VEC_OK; }
        return set_err(ix, ELPIS_VEC_E_NOTFOUND, 0, "no such shard");
    });
}

int elpis_vector_index_shard_object(const elpis_vector_index *ix, const char *shard_digest,
                                    fms_id *out) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(const_cast<elpis_vector_index *>(ix), [&]() -> int {
        if (!shard_digest || !out) return ELPIS_VEC_E_INVAL;
        if (canonical_digest(shard_digest) != 0)
            return set_err(const_cast<elpis_vector_index *>(ix), ELPIS_VEC_E_INVAL, 0,
                           "shard_object: malformed shard digest");
        std::shared_lock<std::shared_mutex> lk(ix->mu);
        for (const ShardEntry &s : ix->shards)
            if (std::strcmp(s.digest, shard_digest) == 0) { *out = s.id; return ELPIS_VEC_OK; }
        return set_err(const_cast<elpis_vector_index *>(ix), ELPIS_VEC_E_NOTFOUND, 0,
                       "no such shard");
    });
}

static int verify_impl(elpis_vector_index *ix, const char *shard_digest) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    /* NULL      -> verify every admitted shard; an empty index is vacuously OK.
     * found     -> verify it.
     * absent    -> ELPIS_VEC_E_NOTFOUND, never a silent OK.
     * malformed -> ELPIS_VEC_E_INVAL, checked before any lookup. */
    if (shard_digest && canonical_digest(shard_digest) != 0)
        return set_err(ix, ELPIS_VEC_E_INVAL, 0, "verify: malformed shard digest");
    std::shared_lock<std::shared_mutex> lk(ix->mu);
    bool found = false;
    for (const ShardEntry &s : ix->shards) {
        if (shard_digest && std::strcmp(s.digest, shard_digest) != 0) continue;
        found = true;
        void *p = nullptr;
        std::lock_guard<std::mutex> shard_lk(*s.gate);
        LeaseGuard g(ix->fms, s.id);
        fms_status r = g.acquire(&p);
        if (r != FMS_OK) {
            /* A corrupt cold replica arrives as FMS_E_DIGEST and stays an
             * integrity failure, with the FMS cause preserved. */
            return set_errf(ix, residency_status(r), r,
                            "shard residency failure: %s", fms_strerror(r));
        }
        char reason[64];
        elpis_vshard_header hdr;
        if (elpis_vshard_verify(p, s.bytes, &hdr, reason, sizeof reason) != 0)
            return set_errf(ix, ELPIS_VEC_E_INTEGRITY, 0,
                            "shard verify failed: %s", reason);
        if (std::strcmp(hdr.shard_digest, s.digest) != 0)
            return set_err(ix, ELPIS_VEC_E_INTEGRITY, 0, "shard digest changed while resident");
    }
    if (shard_digest && !found)
        return set_err(ix, ELPIS_VEC_E_NOTFOUND, 0, "verify: no such shard in this index");
    return ELPIS_VEC_OK;
}

static int search_impl(elpis_vector_index *ix, const elpis_vector_query *q,
                       elpis_vector_hit *hits, uint32_t *n_out) {
    if (!ix || !q || !n_out) return ELPIS_VEC_E_INVAL;
    *n_out = 0;
    clear_err(ix);
    if (q->k == 0) return ELPIS_VEC_OK;               /* k = 0 is a valid empty request */
    if (!q->vector || q->dimensions != ix->profile.dimensions)
        return set_err(ix, ELPIS_VEC_E_QUERY, 0, "query dimension mismatch");
    if (!elpis_vector_all_finite(q->vector, q->dimensions))
        return set_err(ix, ELPIS_VEC_E_QUERY, 0, "query vector is not finite");
    if (!hits) return ELPIS_VEC_E_INVAL;
    if (!filter_valid_ns(q->ns_filter))
        return set_err(ix, ELPIS_VEC_E_INVAL, 0, "search: invalid namespace filter");
    if (!filter_valid_authority(q->authority_filter))
        return set_err(ix, ELPIS_VEC_E_INVAL, 0, "search: invalid authority class filter");

    /* Every searchable index is L2-bound. Normalize the query privately so
     * DOT and cosine both stay inside the canonical score domain. */
    std::vector<float> qv(q->vector, q->vector + q->dimensions);
    if (elpis_vector_l2_normalize(qv.data(), q->dimensions) != 0)
        return set_err(ix, ELPIS_VEC_E_QUERY, 0, "zero-norm query under an L2 profile");

    std::shared_lock<std::shared_mutex> lk(ix->mu);   /* concurrent searches run in parallel */
    std::vector<elpis_vector_hit> pool;
    pool.reserve(q->k * (ix->shards.size() + 1));

    for (const ShardEntry &s : ix->shards) {
        void *p = nullptr;
        LeaseGuard g(ix->fms, s.id);
        fms_status r;
        {
            std::lock_guard<std::mutex> shard_lk(*s.gate);   /* one promoter per shard */
            r = g.acquire(&p);
        }
        if (r != FMS_OK) {
            /* Structured error, never an empty result. FMS_E_DIGEST stays an
             * integrity failure with its cause intact. */
            return set_errf(ix, residency_status(r), r,
                            "shard could not be made WARM-resident: %s%s", fms_strerror(r),
                            r == FMS_E_LIMIT
                                ? " (WARM ceiling must exceed concurrent queries x largest shard)"
                                : "");
        }
        if (g.tier() != FMS_WARM)
            return set_err(ix, ELPIS_VEC_E_RESIDENCY, 0, "shard resident at an unexpected tier");

        const uint8_t *recs = (const uint8_t *)p + ELPIS_VSHARD_HEADER_BYTES;
        const uint64_t n = s.hdr.vector_count;
        std::vector<double> scores(n);
        if (elpis_vector_score_block(recs, n, s.hdr.dimensions, qv.data(), s.hdr.metric,
                                     scores.data()) != 0)
            return set_err(ix, ELPIS_VEC_E_QUERY, 0, "scoring rejected the query");

        for (uint64_t i = 0; i < n; i++) {
            const char *ns = "", *au = "";
            if (elpis_vshard_record_meta(p, s.bytes, i, &ns, &au) != 0)
                return set_err(ix, ELPIS_VEC_E_INTEGRITY, 0, "shard metadata map unreadable");
            /* Filter before admission: an excluded hit never enters the pool. */
            if (q->ns_filter && std::strcmp(ns, q->ns_filter) != 0) continue;
            if (q->authority_filter && std::strcmp(au, q->authority_filter) != 0) continue;

            elpis_vector_hit h;
            std::memset(&h, 0, sizeof h);
            elpis_hex32(recs + i * ELPIS_VSHARD_RECORD_BYTES, h.chunk_digest);
            elpis_hex32(recs + i * ELPIS_VSHARD_RECORD_BYTES + 32, h.doc_digest);
            std::snprintf(h.shard_digest, sizeof h.shard_digest, "%s", s.digest);
            std::snprintf(h.embedding_profile_digest, sizeof h.embedding_profile_digest, "%s",
                          ix->profile_digest);
            std::snprintf(h.ns, sizeof h.ns, "%s", ns);
            std::snprintf(h.authority, sizeof h.authority, "%s", au);
            double canonical_score = 0.0;
            if (!canonical_index_score(scores[i], &canonical_score))
                return set_err(ix, ELPIS_VEC_E_INTEGRITY, 0,
                               "scoring produced a non-finite or out-of-domain value");
            h.score = canonical_score;
            h.score_key = elpis_vector_score_key(canonical_score);
            pool.push_back(h);
        }
        /* Lease released here by LeaseGuard: the shard is demotable again. */
    }

    std::sort(pool.begin(), pool.end(), [](const elpis_vector_hit &a, const elpis_vector_hit &b) {
        return elpis_vector_hit_compare(&a, &b) < 0;
    });

    uint32_t n = (uint32_t)std::min<size_t>(pool.size(), q->k);
    for (uint32_t i = 0; i < n; i++) {
        hits[i] = pool[i];
        hits[i].rank = i;
    }
    *n_out = n;
    return ELPIS_VEC_OK;
}

int elpis_vector_index_add_shard_bytes(elpis_vector_index *ix, const void *bytes, size_t len,
                                       char shard_digest_out[65]) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&] { return add_shard_bytes_impl(ix, bytes, len, shard_digest_out); });
}

int elpis_vector_index_add_shard_file(elpis_vector_index *ix, const char *path,
                                      char shard_digest_out[65]) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&] { return add_shard_file_impl(ix, path, shard_digest_out); });
}

int elpis_vector_index_search(elpis_vector_index *ix, const elpis_vector_query *q,
                              elpis_vector_hit *hits, uint32_t *n_out) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&] { return search_impl(ix, q, hits, n_out); });
}

int elpis_vector_executor_create(elpis_vector_index *ix, unsigned workers,
                                 elpis_vector_executor **out) {
    if (!ix || !out || workers > 4) return ELPIS_VEC_E_INVAL;
    *out = nullptr;
    clear_err(ix);
    try {
        std::unique_ptr<elpis_vector_executor> vx(new elpis_vector_executor());
        vx->index = ix;
        const size_t max_output =
            sizeof(ExecShardEnvelope) +
            static_cast<size_t>(ELPIS_VECTOR_EXEC_MAX_K) * sizeof(elpis_vector_hit);
        elpis_exec_config cfg{};
        cfg.workers = workers;
        cfg.capacity = ELPIS_VINDEX_MAX_SHARDS;
        cfg.max_input_bytes = 1;
        cfg.max_output_bytes = max_output;
        elpis_exec_status erc = elpis_exec_create(&cfg, &vx->runtime);
        if (erc != ELPIS_EXEC_OK) {
            int status = erc == ELPIS_EXEC_WOULD_BLOCK ? ELPIS_VEC_E_LIMIT
                                                       : ELPIS_VEC_E_INTERNAL;
            return set_errf(ix, status, 0, "execution runtime create failed: %d",
                            static_cast<int>(erc));
        }
        *out = vx.release();
        return ELPIS_VEC_OK;
    } catch (...) {
        return set_err(ix, ELPIS_VEC_E_INTERNAL, 0,
                       "allocation failed while creating vector executor");
    }
}

void elpis_vector_executor_destroy(elpis_vector_executor *vx) {
    if (!vx) return;
    elpis_exec_destroy(vx->runtime);
    delete vx;
}

void elpis_vector_executor_metrics(elpis_vector_executor *vx, elpis_exec_metrics *out) {
    if (!vx || !out) return;
    elpis_exec_get_metrics(vx->runtime, out);
}

int elpis_vector_executor_search(elpis_vector_executor *vx,
                                 const elpis_vector_query *q,
                                 elpis_vector_hit *hits, uint32_t *n_out) {
    if (!vx || !vx->index) return ELPIS_VEC_E_INVAL;
    elpis_vector_index *ix = vx->index;
    return guard(ix, [&]() -> int {
        if (!q || !n_out) return ELPIS_VEC_E_INVAL;
        std::lock_guard<std::mutex> search_lk(vx->search_mu);
        *n_out = 0;
        clear_err(ix);
        if (vx->broken)
            return set_err(ix, ELPIS_VEC_E_INTERNAL, 0, "vector executor is closed after a transport failure");
        if (q->k == 0) return ELPIS_VEC_OK;
        if (q->k > ELPIS_VECTOR_EXEC_MAX_K)
            return set_err(ix, ELPIS_VEC_E_LIMIT, 0, "parallel search k exceeds ELPIS_VECTOR_EXEC_MAX_K");
        if (!q->vector || q->dimensions != ix->profile.dimensions)
            return set_err(ix, ELPIS_VEC_E_QUERY, 0, "query dimension mismatch");
        if (!elpis_vector_all_finite(q->vector, q->dimensions))
            return set_err(ix, ELPIS_VEC_E_QUERY, 0, "query vector is not finite");
        if (!hits) return ELPIS_VEC_E_INVAL;
        if (!filter_valid_ns(q->ns_filter))
            return set_err(ix, ELPIS_VEC_E_INVAL, 0, "search: invalid namespace filter");
        if (!filter_valid_authority(q->authority_filter))
            return set_err(ix, ELPIS_VEC_E_INVAL, 0, "search: invalid authority class filter");

        std::vector<float> qv(q->vector, q->vector + q->dimensions);
        if (elpis_vector_l2_normalize(qv.data(), q->dimensions) != 0)
            return set_err(ix, ELPIS_VEC_E_QUERY, 0, "zero-norm query under an L2 profile");

        std::shared_lock<std::shared_mutex> lk(ix->mu);
        const size_t shard_count = ix->shards.size();
        if (!shard_count) return ELPIS_VEC_OK;

        std::vector<ExecShardContext> contexts(shard_count);
        std::vector<elpis_exec_buffer *> inputs(shard_count, nullptr);
        for (size_t i = 0; i < shard_count; ++i) {
            inputs[i] = elpis_exec_buffer_alloc(1);
            if (!inputs[i]) {
                for (elpis_exec_buffer *b : inputs) elpis_exec_buffer_release(b);
                return set_err(ix, ELPIS_VEC_E_INTERNAL, 0,
                               "execution input allocation failed");
            }
            contexts[i] = ExecShardContext{
                ix, &ix->shards[i], qv.data(), q->dimensions, q->k,
                q->ns_filter, q->authority_filter, {}, false
            };
        }

        size_t submitted = 0;
        for (; submitted < shard_count; ++submitted) {
            elpis_exec_bound_task task{
                ELPIS_EXEC_VECTOR_SHARD, 1u, static_cast<uint32_t>(submitted),
                ELPIS_EXEC_PURE, static_cast<uint64_t>(submitted),
                vector_shard_compute, &contexts[submitted]
            };
            uint64_t sequence = 0;
            elpis_exec_status erc =
                elpis_exec_submit_bound(vx->runtime, &task, &inputs[submitted], &sequence);
            if (erc != ELPIS_EXEC_OK) {
                for (size_t j = submitted; j < shard_count; ++j)
                    elpis_exec_buffer_release(inputs[j]);
                elpis_exec_shutdown(vx->runtime, 1);
                vx->broken = true;
                return set_errf(ix, ELPIS_VEC_E_INTERNAL, 0,
                                "execution submit failed after %zu shards: %d",
                                submitted, static_cast<int>(erc));
            }
        }

        std::vector<elpis_vector_hit> pool;
        pool.reserve(static_cast<size_t>(q->k) * shard_count);
        int first_status = ELPIS_VEC_OK;
        int first_cause = 0;
        char first_detail[192] = {0};

        for (size_t i = 0; i < shard_count; ++i) {
            elpis_exec_result result{};
            elpis_exec_status trc = elpis_exec_take(vx->runtime, 60000, &result);
            if (trc != ELPIS_EXEC_OK) {
                if (first_status == ELPIS_VEC_OK) {
                    first_status = ELPIS_VEC_E_INTERNAL;
                    std::snprintf(first_detail, sizeof first_detail,
                                  "execution retirement failed: %d", static_cast<int>(trc));
                }
                elpis_exec_shutdown(vx->runtime, 1);
                vx->broken = true;
                break;
            }
            if (result.tag != i || result.operation != ELPIS_EXEC_VECTOR_SHARD ||
                result.status != ELPIS_EXEC_OK || !result.output) {
                if (first_status == ELPIS_VEC_OK) {
                    first_status = ELPIS_VEC_E_INTERNAL;
                    std::snprintf(first_detail, sizeof first_detail,
                                  "execution result metadata/status invalid at shard %zu", i);
                }
                elpis_exec_buffer_release(result.output);
                continue;
            }

            const size_t bytes = elpis_exec_buffer_size(result.output);
            if (bytes < sizeof(ExecShardEnvelope)) {
                if (first_status == ELPIS_VEC_OK) {
                    first_status = ELPIS_VEC_E_INTERNAL;
                    std::snprintf(first_detail, sizeof first_detail,
                                  "execution result truncated at shard %zu", i);
                }
                elpis_exec_buffer_release(result.output);
                continue;
            }

            const unsigned char *src =
                static_cast<const unsigned char *>(elpis_exec_buffer_data(result.output));
            ExecShardEnvelope env{};
            std::memcpy(&env, src, sizeof env);
            const size_t expected =
                sizeof env + static_cast<size_t>(env.count) * sizeof(elpis_vector_hit);
            if (env.count > q->k || expected != bytes) {
                if (first_status == ELPIS_VEC_OK) {
                    first_status = ELPIS_VEC_E_INTERNAL;
                    std::snprintf(first_detail, sizeof first_detail,
                                  "execution result envelope invalid at shard %zu", i);
                }
                elpis_exec_buffer_release(result.output);
                continue;
            }
            if (env.status != ELPIS_VEC_OK) {
                if (first_status == ELPIS_VEC_OK) {
                    first_status = env.status;
                    first_cause = env.cause;
                    std::snprintf(first_detail, sizeof first_detail, "%s", env.detail);
                }
            } else if (env.count) {
                const elpis_vector_hit *local =
                    reinterpret_cast<const elpis_vector_hit *>(src + sizeof env);
                pool.insert(pool.end(), local, local + env.count);
            }
            elpis_exec_buffer_release(result.output);
        }

        if (first_status != ELPIS_VEC_OK)
            return set_err(ix, first_status, first_cause, first_detail);

        std::sort(pool.begin(), pool.end(),
                  [](const elpis_vector_hit &a, const elpis_vector_hit &b) {
                      return elpis_vector_hit_compare(&a, &b) < 0;
                  });
        uint32_t n = static_cast<uint32_t>(std::min<size_t>(pool.size(), q->k));
        for (uint32_t i = 0; i < n; ++i) {
            hits[i] = pool[i];
            hits[i].rank = i;
        }
        *n_out = n;
        return ELPIS_VEC_OK;
    });
}

int elpis_vector_index_verify(elpis_vector_index *ix, const char *shard_digest) {
    if (!ix) return ELPIS_VEC_E_INVAL;
    return guard(ix, [&] { return verify_impl(ix, shard_digest); });
}

int elpis_vector_index_list_shards(const elpis_vector_index *ix, char (*out)[65], uint32_t cap,
                                   uint32_t *n_out) {
    if (!ix || !n_out) return ELPIS_VEC_E_INVAL;
    return guard(const_cast<elpis_vector_index *>(ix),
                 [&] { return list_shards_impl(ix, out, cap, n_out); });
}

/* Internal, used by vector_manifest.cpp. Takes the shared lock exactly once. */
int elpis_vector_index_manifest_locked(elpis_vector_index *ix, char (*ids)[65], uint32_t cap,
                                       uint32_t *n_out, elpis_vshard_header *hdrs,
                                       char profile_digest_out[65], char corpus_digest_out[65]) {
    if (!ix || !n_out) return ELPIS_VEC_E_INVAL;
    std::shared_lock<std::shared_mutex> lk(ix->mu);
    int rc = list_shards_locked(ix, ids, cap, n_out);
    if (rc != ELPIS_VEC_OK) return rc;
    if (ids && hdrs)
        for (uint32_t i = 0; i < *n_out; i++) {
            rc = inspect_locked(ix, ids[i], &hdrs[i]);
            if (rc != ELPIS_VEC_OK) return rc;
        }
    if (profile_digest_out) std::snprintf(profile_digest_out, 65, "%s", ix->profile_digest);
    if (corpus_digest_out)
        std::snprintf(corpus_digest_out, 65, "%s",
                      ix->bind_corpus ? ix->corpus_digest
                                      : "0000000000000000000000000000000000000000000000000000000000000000");
    return ELPIS_VEC_OK;
}

} // extern "C"
