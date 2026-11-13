/* ELPIS_YM_AWR_R120 incremental geometric update kernel. RESEARCH_ONLY.
 *
 * One native call performs: event validation -> CGC coefficients -> in-place bounded
 * geometric update of existing relation rows -> affected-window discovery through the
 * existing binding lists -> shared frozen HCA88 passes -> MEAN reconciliation -> commit.
 *
 * The HCA arithmetic is NOT re-implemented here: this translation unit includes the frozen
 * R119 kernel source verbatim and calls its static instance_pass() and reconcile(), so every
 * pass is the R119 EXACT_GUARDED arithmetic by construction (same compiler flags).
 *
 * Window cache. instance_pass() reads a registry through (rel_off, members, frames) only.
 * For a cached instance those three pointers address one contiguous block that holds the
 * member lists and frames of its (at most four) relations, with local relation index = slot.
 * The values are byte copies of the registry rows, so results are bit-identical; only the
 * memory locality changes.
 *
 * Parallel Jacobi (opt-in, threads > 1). Within one synchronous pass every window reads the
 * same pre-pass buffer and writes its own candidate block, so windows are independent; running
 * them on several threads executes the identical per-window arithmetic and is bit-identical to
 * the serial pass. Reconciliation and commit stay serial.
 *
 * Transaction. All work happens in compact scratch buffers. The pool, the per-row event
 * sequence and the pending list are written only after every stage succeeded.
 */
#include "awr_kernel_r119.c"
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

enum {
    R120_OK = 0,                 /* committed and propagated */
    R120_COMMITTED_DEFERRED = 1, /* updates committed, propagation recorded as pending */
    R120_REFUSED_ARGUMENT = 10,
    R120_REFUSED_INVALID_VALUE = 11,
    R120_REFUSED_UNKNOWN_RELATION = 12,
    R120_REFUSED_NOT_MATERIALIZED = 13,
    R120_REFUSED_STALE_RELATION = 14,
    R120_REFUSED_NOT_INCIDENT = 15,
    R120_REFUSED_CONFLICT = 16,
    R120_REFUSED_EVENT_BUDGET = 17,
    R120_REFUSED_INSTANCE_BUDGET = 18,
    R120_REFUSED_ROW_BUDGET = 19,
    R120_REFUSED_LINK_BUDGET = 20,
    R120_REFUSED_NONFINITE = 21,
    R120_REFUSED_KERNEL = 22,
    R120_REFUSED_PENDING_CAPACITY = 23
};

#define R120_FLAG_PROPAGATE 1      /* run the HCA passes on the affected windows */
#define R120_FLAG_DEFER_ON_BUDGET 2 /* a propagation budget refusal commits the updates and records pending rows */
#define R120_FLAG_PROFILE 4         /* time the coefficient + update stage per event (adds two clock reads per event) */

/* report slots (int64) */
enum { RP_STATUS, RP_FAIL_INDEX, RP_APPLIED, RP_SKIPPED_DUPLICATE, RP_TOUCHED_ROWS, RP_INSTANCES, RP_ROWS,
       RP_LINKS, RP_PENDING_ROWS, RP_KERNEL_RC, RP_NS_VALIDATE_UPDATE, RP_NS_SELECT, RP_NS_HCA, RP_NS_RECONCILE,
       RP_NS_COMMIT, RP_NS_TOTAL, RP_CACHED_INSTANCES, RP_NS_CGC, RP_COUNT };

typedef struct {
    /* registry (immutable) */
    const uint64_t *relation_ids; const int64_t *rel_off; const int32_t *members; const double *frames;
    /* volatile AWR state (R118 AWRState arrays) */
    double *pool; const int32_t *rel_row; const uint8_t *inst_live; const int32_t *slot_rel; const int32_t *slot_row;
    const int32_t *row_head; const int32_t *bind_next; int32_t *mark; const int32_t *inst_links;
    /* R120 side tables */
    uint64_t *row_seq; const uint32_t *row_gen; uint8_t *row_pending; int32_t *pending_list; uint8_t *inst_mark;
    const int64_t *wc_off; const uint8_t *wc_data;
    /* frozen operators */
    const double *W; const double *bias; const float *cgc_w;
    /* scratch (preallocated by the wrapper) */
    int32_t *handles; int32_t *s_slot_rel; int32_t *s_slot_cmp; int32_t *rows; int32_t *count; int32_t *off;
    int32_t *list; int32_t *last_event; double *buf_a; double *buf_b; double *cand; uint64_t *seq_new; int64_t *gc;
    /* scalars */
    int64_t n_rel; int64_t max_links; int64_t pending_count;
    double alpha, beta, theta, eta, gamma_v, rho_b, kappa;
    int32_t inst_capacity, row_capacity, max_events, max_instances, max_rows, correct, mode, pending_capacity;
    uint32_t gmask; int32_t threads; int32_t parallel_min; int32_t pad_;
} awr120_ctx;

int awr120_abi_version(void) { return 120; }
int awr120_ctx_bytes(void) { return (int)sizeof(awr120_ctx); }

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Frozen Q4E5 cubic geometric operator: float32 weights [3][5][8];
 * out_o = 0.5 * sum_n (z + z^2 + z^3),  z = sum_d x_d w[o][d][n]  (float32, fixed order). */
static void cgc3_forward(const float *w, const float *x, float *out) {
    for (int o = 0; o < 3; o++) {
        float acc = 0.0f;
        for (int n = 0; n < 8; n++) {
            float z = 0.0f;
            for (int d = 0; d < 5; d++) z = z + x[d] * w[(o * 5 + d) * 8 + n];
            float z2 = z * z;
            acc = acc + ((z + z2) + z2 * z);
        }
        out[o] = 0.5f * acc;
    }
}

void awr120_cgc_probe(const float *w, const float *x, float *out) { cgc3_forward(w, x, out); }

/* The R120 update law on one relation state s = (A[4], B[4]) in its own gauge frame.
 *   v = weight * (U o)                       transported, signed observation
 *   x = (tanh A.v, tanh B.v, tanh A.B, tanh |v - A|^2, aux)      gauge invariants
 *   (a, b, c) = kappa * tanh(CGC3(x))                           bounded coefficients
 *   A' = clamp4(A + eta (a A + b B) + gamma v)
 *   B' = clamp4(B + eta (c A - a B) + gamma rho v)
 * frame == NULL means "no observation" (v = 0). */
static int update_row(const awr120_ctx *c, double *s, const double *frame, const double *obs, double weight,
                      double aux) {
    double v[4] = {0.0, 0.0, 0.0, 0.0};
    if (frame) {
        double t[4];
        apply_link(frame, obs, t);
        for (int k = 0; k < 4; k++) v[k] = weight * t[k];
    }
    const double *A = s, *B = s + 4;
    double dv[4];
    for (int k = 0; k < 4; k++) dv[k] = v[k] - A[k];
    float x[5], y[3];
    x[0] = (float)tanh(dot4(A, v));
    x[1] = (float)tanh(dot4(B, v));
    x[2] = (float)tanh(dot4(A, B));
    x[3] = (float)tanh(dot4(dv, dv));
    x[4] = (float)aux;
    cgc3_forward(c->cgc_w, x, y);
    const double a = c->kappa * tanh((double)y[0]);
    const double b = c->kappa * tanh((double)y[1]);
    const double g = c->kappa * tanh((double)y[2]);
    double An[4], Bn[4];
    for (int k = 0; k < 4; k++) {
        An[k] = (A[k] + c->eta * (a * A[k] + b * B[k])) + c->gamma_v * v[k];
        Bn[k] = (B[k] + c->eta * (g * A[k] - a * B[k])) + (c->gamma_v * c->rho_b) * v[k];
    }
    clamp4(An);
    clamp4(Bn);
    for (int k = 0; k < 4; k++) {
        if (!isfinite(An[k]) || !isfinite(Bn[k])) return 1;
        s[k] = An[k];
        s[4 + k] = Bn[k];
    }
    return 0;
}

void awr120_update_probe(const awr120_ctx *c, double *s, const double *frame, const double *obs, double weight,
                         double aux, int32_t *rc) {
    *rc = update_row(c, s, frame, obs, weight, aux);
}

/* ---- window cache ------------------------------------------------------------------------ */
/* block: int64 rel_off[5]; int32 members[M] (padded to 8 bytes); double frames[8 M] */
static inline int64_t wc_block_bytes(int64_t m) { return 40 + 8 * ((m + 1) / 2) + 64 * m; }

int64_t awr120_cache_build(const awr120_ctx *c, int32_t n, const int32_t *handles, int64_t *wc_off,
                           uint8_t *wc_data, int64_t capacity, int64_t used, int32_t *built) {
    int32_t done = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t h = handles[i];
        if (h < 0 || h >= c->inst_capacity || !c->inst_live[h]) continue;
        int64_t m = 0;
        for (int d = 0; d < AWR_SLOTS; d++) {
            int32_t r = c->slot_rel[(int64_t)h * AWR_SLOTS + d];
            if (r >= 0) m += c->rel_off[r + 1] - c->rel_off[r];
        }
        int64_t bytes = wc_block_bytes(m);
        if (used + bytes > capacity) break;
        uint8_t *blk = wc_data + used;
        int64_t *lo = (int64_t *)blk;
        int32_t *lm = (int32_t *)(blk + 40);
        double *lf = (double *)(blk + 40 + 8 * ((m + 1) / 2));
        int64_t pos = 0;
        for (int d = 0; d < AWR_SLOTS; d++) {
            lo[d] = pos;
            int32_t r = c->slot_rel[(int64_t)h * AWR_SLOTS + d];
            if (r < 0) continue;
            for (int64_t q = c->rel_off[r]; q < c->rel_off[r + 1]; q++, pos++) {
                lm[pos] = c->members[q];
                memcpy(lf + 8 * pos, c->frames + 8 * q, 64);
            }
        }
        lo[AWR_SLOTS] = pos;
        if (m & 1) lm[m] = 0;
        wc_off[h] = used;
        used += bytes;
        done++;
    }
    *built = done;
    return used;
}

/* ---- reconciliation ----------------------------------------------------------------------------
 * Bit-identical to the frozen R119 `reconcile` (which stays the reference and is compared against
 * in every verification): per row and coordinate the candidates are summed in ascending order and
 * divided by their number. Only the sorting algorithm differs (Shell sort above 24 values instead
 * of insertion sort). The ascending sequence is unique up to the order of equal values; equal
 * values are bit-identical except +0 / -0, whose relative order cannot change a sum (they follow
 * every negative value, and (+0) + (-0) = (-0) + (+0) = +0). Rows are independent, so the row loop
 * may run in parallel without changing any result. */
static inline void sort_ascending(double *v, int32_t m) {
    static const int32_t gaps[] = {701, 301, 132, 57, 23, 10, 4};
    if (m > 24)
        for (int g = 0; g < 7; g++) {
            const int32_t gap = gaps[g];
            for (int32_t i = gap; i < m; i++) {
                const double val = v[i];
                int32_t t = i;
                while (t >= gap && v[t - gap] > val) { v[t] = v[t - gap]; t -= gap; }
                v[t] = val;
            }
        }
    for (int32_t i = 1; i < m; i++) {
        const double val = v[i];
        int32_t t = i;
        while (t > 0 && v[t - 1] > val) { v[t] = v[t - 1]; t--; }
        v[t] = val;
    }
}

static int reconcile120(const awr120_ctx *c, int32_t n, const double *cand, int32_t nr, double *cout, int par) {
    const int32_t *slot_cmp = c->s_slot_cmp;
    int32_t *count = c->count, *off = c->off, *list = c->list;
    for (int32_t r = 0; r < nr; r++) count[r] = 0;
    for (int32_t i = 0; i < n * AWR_SLOTS; i++) if (slot_cmp[i] >= 0) count[slot_cmp[i]]++;
    off[0] = 0;
    for (int32_t r = 0; r < nr; r++) {
        if (count[r] < 1) return AWR_ERR_ARGUMENT;
        if (count[r] > AWR_MAX_BIND) return AWR_ERR_BINDINGS;
        if (c->mode == 1 && count[r] > 1) return AWR_ERR_OVERLAP;
        off[r + 1] = off[r] + count[r];
        count[r] = 0;
    }
    for (int32_t i = 0; i < n * AWR_SLOTS; i++) {
        int32_t r = slot_cmp[i];
        if (r >= 0) list[off[r] + count[r]++] = i;
    }
    (void)par;
#ifdef _OPENMP
#pragma omp parallel for num_threads(c->threads) schedule(static) if (par)
#endif
    for (int32_t r = 0; r < nr; r++) {
        const int32_t m = off[r + 1] - off[r];
        const int32_t *ids = list + off[r];
        double *dst = cout + (int64_t)r * AWR_SC;
        if (m == 1) { memcpy(dst, cand + (int64_t)ids[0] * AWR_SC, sizeof(double) * AWR_SC); continue; }
        double v[AWR_MAX_BIND];
        for (int k = 0; k < AWR_SC; k++) {
            v[0] = cand[(int64_t)ids[0] * AWR_SC + k];
            for (int32_t q = 1; q < m; q++) v[q] = cand[(int64_t)ids[q] * AWR_SC + k];
            sort_ascending(v, m);
            double acc = v[0];
            for (int32_t q = 1; q < m; q++) acc += v[q];
            dst[k] = acc / (double)m;
        }
    }
    return AWR_OK;
}

/* ---- shared propagation core ----------------------------------------------------------------
 * pass_loop: `passes` synchronous Jacobi passes of the frozen R119 instance pass over the
 * selected instances (handles[0..ni), compact slots in s_slot_cmp), MEAN / EXCLUSIVE
 * reconciliation after each pass. Every compact row must be bound by a selected instance. */
static inline int one_window(awr120_ctx *c, int32_t i, int32_t ni, const double *cur, uint64_t theta_bits,
                             int64_t *links, int64_t *cached, int64_t *gc) {
    static const int32_t local_rel[AWR_SLOTS] = {0, 1, 2, 3};
    const int32_t h = c->handles[i];
    const int32_t *cmp = c->s_slot_cmp + i * AWR_SLOTS;
    const int64_t o = c->wc_off ? c->wc_off[h] : -1;
    if (c->wc_off && i + 2 < ni) {
        const int64_t o2 = c->wc_off[c->handles[i + 2]];
        if (o2 >= 0) {
            __builtin_prefetch(c->wc_data + o2);
            __builtin_prefetch(c->wc_data + o2 + 64);
            __builtin_prefetch(c->wc_data + o2 + 128);
        }
    }
    double *out = c->cand + (int64_t)i * AWR_SLOTS * AWR_SC;
    if (o >= 0) {
        const uint8_t *blk = c->wc_data + o;
        const int64_t *lo = (const int64_t *)blk;
        const int64_t m = lo[AWR_SLOTS];
        int32_t lrel[AWR_SLOTS];
        for (int d = 0; d < AWR_SLOTS; d++) lrel[d] = cmp[d] >= 0 ? local_rel[d] : -1;
        (*cached)++;
        return instance_pass(lrel, cmp, cur, lo, (const int32_t *)(blk + 40),
                             (const double *)(blk + 40 + 8 * ((m + 1) / 2)), c->W, c->bias, c->correct, c->alpha,
                             c->beta, out, links, c->gmask, theta_bits, gc);
    }
    return instance_pass(c->slot_rel + (int64_t)h * AWR_SLOTS, cmp, cur, c->rel_off, c->members, c->frames, c->W,
                         c->bias, c->correct, c->alpha, c->beta, out, links, c->gmask, theta_bits, gc);
}

static int pass_loop(awr120_ctx *c, int32_t ni, int32_t nr, int32_t passes, int64_t *rep, double **cur_out) {
    const uint64_t theta_bits = absbits(c->theta);
    double *cur = c->buf_a, *nxt = c->buf_b;
    int64_t links = 0, ns_hca = 0, ns_rec = 0, cached = 0;
    int rc = AWR_OK;
    for (int32_t p = 0; p < passes && rc == AWR_OK; p++) {
        const int64_t ta = now_ns();
#ifdef _OPENMP
        if (c->threads > 1 && ni >= c->parallel_min) {
            int prc = AWR_OK;
            const double *src = cur;
#pragma omp parallel num_threads(c->threads) reduction(+ : links, cached) reduction(max : prc)
            {
                int64_t gcl[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma omp for schedule(static) nowait
                for (int32_t i = 0; i < ni; i++) {
                    int r1 = one_window(c, i, ni, src, theta_bits, &links, &cached, gcl);
                    if (r1 > prc) prc = r1;
                }
#pragma omp critical
                for (int q = 0; q < 8; q++) c->gc[q] += gcl[q];
            }
            rc = prc;
        } else
#endif
        {
            for (int32_t i = 0; i < ni && rc == AWR_OK; i++)
                rc = one_window(c, i, ni, cur, theta_bits, &links, &cached, c->gc);
        }
        const int64_t tb = now_ns();
        ns_hca += tb - ta;
        if (rc == AWR_OK) rc = reconcile120(c, ni, c->cand, nr, nxt, c->threads > 1 && ni >= c->parallel_min);
        ns_rec += now_ns() - tb;
        if (rc == AWR_OK) { double *t = cur; cur = nxt; nxt = t; }
    }
    rep[RP_LINKS] = links;
    rep[RP_NS_HCA] = ns_hca;
    rep[RP_NS_RECONCILE] = ns_rec;
    rep[RP_CACHED_INSTANCES] = passes > 0 ? cached / passes : 0;
    if (rc != AWR_OK) {
        rep[RP_KERNEL_RC] = rc;
        return rc == AWR_ERR_NONFINITE ? R120_REFUSED_NONFINITE : R120_REFUSED_KERNEL;
    }
    *cur_out = cur;
    return R120_OK;
}

int awr120_openmp(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/* add one instance to the working set: compact its bound rows (loading new ones from the pool) */
static int add_instance(awr120_ctx *c, int32_t h, int32_t i, int32_t *nr) {
    for (int d = 0; d < AWR_SLOTS; d++) {
        const int32_t r = c->slot_rel[(int64_t)h * AWR_SLOTS + d];
        const int32_t row = c->slot_row[(int64_t)h * AWR_SLOTS + d];
        if (r < 0) { c->s_slot_cmp[i * AWR_SLOTS + d] = -1; continue; }
        if (c->mark[row] < 0) {
            if (*nr >= c->max_rows) return R120_REFUSED_ROW_BUDGET;
            c->mark[row] = *nr;
            c->rows[*nr] = row;
            memcpy(c->buf_a + (int64_t)*nr * AWR_SC, c->pool + (int64_t)row * AWR_SC, sizeof(double) * AWR_SC);
            (*nr)++;
        }
        c->s_slot_cmp[i * AWR_SLOTS + d] = c->mark[row];
    }
    return R120_OK;
}

/* run_windows. On entry compact rows 0..nr-1 are the seed rows (loaded in buf_a, marked).
 * Level 1 selects every live instance bound to a seed row through the binding lists; each
 * further level selects the instances bound to rows added by the previous level. The cost is
 * proportional to the affected materialized neighbourhood, never to the registry. */
static int run_windows(awr120_ctx *c, int32_t *nr_io, int32_t radius, int32_t passes, int64_t *rep,
                       double **cur_out, int32_t *ni_out) {
    int32_t nr = *nr_io, ni = 0, lo = 0;
    int64_t links_budget = 0;
    int status = R120_OK;
    const int64_t t0 = now_ns();
    for (int32_t level = 0; level < radius && status == R120_OK; level++) {
        const int32_t hi = nr, first_new = ni;
        for (int32_t q = lo; q < hi && status == R120_OK; q++) {
            for (int32_t code = c->row_head[c->rows[q]]; code >= 0; code = c->bind_next[code]) {
                const int32_t h = code >> 2;
                if (c->inst_mark[h]) continue;
                if (ni >= c->max_instances) { status = R120_REFUSED_INSTANCE_BUDGET; break; }
                c->inst_mark[h] = 1;
                c->handles[ni++] = h;
            }
        }
        for (int32_t i = first_new; i < ni && status == R120_OK; i++) {
            links_budget += c->inst_links[c->handles[i]];
            status = add_instance(c, c->handles[i], i, &nr);
        }
        if (status == R120_OK && links_budget * passes > c->max_links) status = R120_REFUSED_LINK_BUDGET;
        lo = hi;
    }
    *nr_io = nr;
    *ni_out = ni;
    rep[RP_INSTANCES] = ni;
    rep[RP_ROWS] = nr;
    rep[RP_NS_SELECT] = now_ns() - t0;
    if (status != R120_OK) return status;
    return pass_loop(c, ni, nr, passes, rep, cur_out);
}

static void release(awr120_ctx *c, int32_t nr, int32_t ni) {
    for (int32_t k = 0; k < nr; k++) c->mark[c->rows[k]] = -1;
    for (int32_t i = 0; i < ni; i++) c->inst_mark[c->handles[i]] = 0;
}

static int64_t lookup_relation(const awr120_ctx *c, uint64_t id) {
    int64_t lo = 0, hi = c->n_rel;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (c->relation_ids[mid] < id) lo = mid + 1; else hi = mid;
    }
    return (lo < c->n_rel && c->relation_ids[lo] == id) ? lo : -1;
}

/* Batched update and selective propagation. Event arrays (length n):
 *   rel_id   stable 64-bit relation identity
 *   seq      per-relation event sequence; an event with seq <= the last applied sequence of its
 *            relation is a duplicate and is skipped (idempotent redelivery); two events of one
 *            batch with the same (relation, seq) and different payloads refuse the batch
 *   exp_gen  expected materialization generation of the relation row (0 = unchecked)
 *   part     participant index anchoring the observation frame, or -1 for "no observation"
 *   obs[4]   observation in that participant's frame;  weight in [-1, 1];  aux in [0, 1]
 * Events of one relation are applied in batch order (their sequence must increase). */
int awr120_update_propagate(awr120_ctx *c, int32_t n, const uint64_t *rel_id, const uint64_t *seq,
                            const uint32_t *exp_gen, const int32_t *part, const double *obs, const double *weight,
                            const double *aux, int32_t flags, int32_t radius, int32_t passes, int64_t *rep) {
    for (int i = 0; i < RP_COUNT; i++) rep[i] = 0;
    rep[RP_FAIL_INDEX] = -1;
    const int64_t t0 = now_ns();
    int32_t nt = 0, ni = 0, applied = 0, skipped = 0;
    int status = R120_OK;
    if (n < 0 || radius < 1 || passes < 1) status = R120_REFUSED_ARGUMENT;
    else if (n > c->max_events) status = R120_REFUSED_EVENT_BUDGET;
    int64_t ns_cgc = 0;
    for (int32_t e = 0; e < n && status == R120_OK; e++) {
        const double *o = obs + 4 * (int64_t)e;
        if (!isfinite(weight[e]) || !isfinite(aux[e]) || weight[e] < -1.0 || weight[e] > 1.0 || aux[e] < 0.0 ||
            aux[e] > 1.0 || !isfinite(o[0]) || !isfinite(o[1]) || !isfinite(o[2]) || !isfinite(o[3])) {
            status = R120_REFUSED_INVALID_VALUE; rep[RP_FAIL_INDEX] = e; break;
        }
        int64_t rel = lookup_relation(c, rel_id[e]);
        if (rel < 0) { status = R120_REFUSED_UNKNOWN_RELATION; rep[RP_FAIL_INDEX] = e; break; }
        int32_t row = c->rel_row[rel];
        if (row < 0) { status = R120_REFUSED_NOT_MATERIALIZED; rep[RP_FAIL_INDEX] = e; break; }
        if (exp_gen[e] != 0 && exp_gen[e] != c->row_gen[row]) {
            status = R120_REFUSED_STALE_RELATION; rep[RP_FAIL_INDEX] = e; break;
        }
        const double *frame = 0;
        if (part[e] >= 0) {
            for (int64_t q = c->rel_off[rel]; q < c->rel_off[rel + 1]; q++)
                if (c->members[q] == part[e]) { frame = c->frames + 8 * q; break; }
            if (!frame) { status = R120_REFUSED_NOT_INCIDENT; rep[RP_FAIL_INDEX] = e; break; }
        } else if (o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0 || o[3] != 0.0) {
            /* an observation without a frame anchor cannot be transported covariantly */
            status = R120_REFUSED_NOT_INCIDENT; rep[RP_FAIL_INDEX] = e; break;
        }
        int32_t k = c->mark[row];
        uint64_t last = k >= 0 ? c->seq_new[k] : c->row_seq[row];
        if (seq[e] <= last) {
            if (k >= 0 && seq[e] == last && c->last_event[k] >= 0) {
                int32_t f = c->last_event[k];           /* same (relation, seq) inside this batch */
                if (part[f] != part[e] || weight[f] != weight[e] || aux[f] != aux[e] ||
                    memcmp(obs + 4 * (int64_t)f, o, 32) != 0) {
                    status = R120_REFUSED_CONFLICT; rep[RP_FAIL_INDEX] = e; break;
                }
            }
            skipped++;
            continue;
        }
        if (k < 0) {
            if (nt >= c->max_rows) { status = R120_REFUSED_ROW_BUDGET; rep[RP_FAIL_INDEX] = e; break; }
            k = nt++;
            c->mark[row] = k;
            c->rows[k] = row;
            memcpy(c->buf_a + (int64_t)k * AWR_SC, c->pool + (int64_t)row * AWR_SC, sizeof(double) * AWR_SC);
            c->last_event[k] = -1;
        }
        const int64_t tc = (flags & R120_FLAG_PROFILE) ? now_ns() : 0;
        if (update_row(c, c->buf_a + (int64_t)k * AWR_SC, frame, o, weight[e], aux[e])) {
            status = R120_REFUSED_NONFINITE; rep[RP_FAIL_INDEX] = e; break;
        }
        if (flags & R120_FLAG_PROFILE) ns_cgc += now_ns() - tc;
        c->seq_new[k] = seq[e];
        c->last_event[k] = e;
        applied++;
    }
    rep[RP_APPLIED] = applied;
    rep[RP_SKIPPED_DUPLICATE] = skipped;
    rep[RP_TOUCHED_ROWS] = nt;
    rep[RP_NS_CGC] = ns_cgc;
    const int64_t t1 = now_ns();
    rep[RP_NS_VALIDATE_UPDATE] = t1 - t0;
    int32_t nr = nt;
    double *cur = c->buf_a;
    int deferred = 0;
    if (status == R120_OK && nt > 0) {
        /* an updated relation always has at least one live binding (it is materialized) */
        if (flags & R120_FLAG_PROPAGATE) {
            int st2 = run_windows(c, &nr, radius, passes, rep, &cur, &ni);
            if (st2 != R120_OK) {
                int budget = st2 == R120_REFUSED_INSTANCE_BUDGET || st2 == R120_REFUSED_ROW_BUDGET ||
                             st2 == R120_REFUSED_LINK_BUDGET;
                if (budget && (flags & R120_FLAG_DEFER_ON_BUDGET)) { deferred = 1; cur = c->buf_a; }
                else status = st2;
            }
        } else {
            deferred = 1;
        }
    }
    if (status == R120_OK && deferred) {
        /* pending capacity is checked before any write */
        int64_t add = 0;
        for (int32_t k = 0; k < nt; k++) add += !c->row_pending[c->rows[k]];
        if (c->pending_count + add > c->pending_capacity) status = R120_REFUSED_PENDING_CAPACITY;
    }
    const int64_t t2 = now_ns();
    if (status == R120_OK) {
        int32_t ncommit = deferred ? nt : nr;
        for (int32_t k = 0; k < ncommit; k++)
            memcpy(c->pool + (int64_t)c->rows[k] * AWR_SC, cur + (int64_t)k * AWR_SC, sizeof(double) * AWR_SC);
        for (int32_t k = 0; k < nt; k++) {
            int32_t row = c->rows[k];
            c->row_seq[row] = c->seq_new[k];
            if (deferred && !c->row_pending[row]) {
                c->row_pending[row] = 1;
                c->pending_list[c->pending_count++] = row;
            }
        }
        status = deferred ? R120_COMMITTED_DEFERRED : R120_OK;
    }
    release(c, nr, ni);
    const int64_t t3 = now_ns();
    rep[RP_PENDING_ROWS] = c->pending_count;
    rep[RP_NS_COMMIT] = t3 - t2;
    rep[RP_NS_TOTAL] = t3 - t0;
    rep[RP_STATUS] = status;
    return status;
}

/* Propagate up to `limit` pending rows (recorded by deferred updates). Remaining rows stay pending. */
int awr120_propagate_pending(awr120_ctx *c, int32_t limit, int32_t radius, int32_t passes, int64_t *rep) {
    for (int i = 0; i < RP_COUNT; i++) rep[i] = 0;
    rep[RP_FAIL_INDEX] = -1;
    const int64_t t0 = now_ns();
    int status = R120_OK;
    int32_t nt = 0, ni = 0;
    if (limit < 1 || radius < 1 || passes < 1) status = R120_REFUSED_ARGUMENT;
    int64_t take = c->pending_count < limit ? c->pending_count : limit;
    if (take > c->max_rows) take = c->max_rows;
    for (int64_t q = 0; q < take && status == R120_OK; q++) {
        int32_t row = c->pending_list[q];
        c->mark[row] = nt;
        c->rows[nt] = row;
        memcpy(c->buf_a + (int64_t)nt * AWR_SC, c->pool + (int64_t)row * AWR_SC, sizeof(double) * AWR_SC);
        nt++;
    }
    int32_t nr = nt;
    double *cur = c->buf_a;
    if (status == R120_OK && nt > 0) status = run_windows(c, &nr, radius, passes, rep, &cur, &ni);
    const int64_t t2 = now_ns();
    if (status == R120_OK && nt > 0) {
        for (int32_t k = 0; k < nr; k++)
            memcpy(c->pool + (int64_t)c->rows[k] * AWR_SC, cur + (int64_t)k * AWR_SC, sizeof(double) * AWR_SC);
        for (int32_t k = 0; k < nt; k++) c->row_pending[c->rows[k]] = 0;
        int64_t rest = c->pending_count - nt;
        memmove(c->pending_list, c->pending_list + nt, sizeof(int32_t) * (size_t)rest);
        c->pending_count = rest;
    }
    release(c, nr, ni);
    const int64_t t3 = now_ns();
    rep[RP_TOUCHED_ROWS] = nt;
    rep[RP_PENDING_ROWS] = c->pending_count;
    rep[RP_NS_COMMIT] = t3 - t2;
    rep[RP_NS_TOTAL] = t3 - t0;
    rep[RP_STATUS] = status;
    return status;
}

/* Plain synchronous step over explicit instance handles (the R119 step semantics) through the
 * window cache: used for the HCA-only arm and as the like-for-like latency baseline. */
int awr120_step(awr120_ctx *c, int32_t n, const int32_t *handles, int32_t passes, int64_t *rep) {
    for (int i = 0; i < RP_COUNT; i++) rep[i] = 0;
    rep[RP_FAIL_INDEX] = -1;
    const int64_t t0 = now_ns();
    int status = R120_OK;
    int32_t nr = 0, ni = 0;
    int64_t links_budget = 0;
    if (n < 0 || passes < 1) status = R120_REFUSED_ARGUMENT;
    for (int32_t i = 0; i < n && status == R120_OK; i++) {
        const int32_t h = handles[i];
        if (h < 0 || h >= c->inst_capacity || !c->inst_live[h]) {
            status = R120_REFUSED_ARGUMENT; rep[RP_FAIL_INDEX] = i; break;
        }
        if (c->inst_mark[h]) continue;                       /* duplicates in the request are idempotent */
        if (ni >= c->max_instances) { status = R120_REFUSED_INSTANCE_BUDGET; break; }
        c->inst_mark[h] = 1;
        c->handles[ni] = h;
        links_budget += c->inst_links[h];
        status = add_instance(c, h, ni, &nr);
        ni++;
    }
    if (status == R120_OK && links_budget * passes > c->max_links) status = R120_REFUSED_LINK_BUDGET;
    rep[RP_NS_SELECT] = now_ns() - t0;
    double *cur = c->buf_a;
    if (status == R120_OK && ni > 0) status = pass_loop(c, ni, nr, passes, rep, &cur);
    const int64_t t2 = now_ns();
    if (status == R120_OK)
        for (int32_t k = 0; k < nr; k++)
            memcpy(c->pool + (int64_t)c->rows[k] * AWR_SC, cur + (int64_t)k * AWR_SC, sizeof(double) * AWR_SC);
    release(c, nr, ni);
    const int64_t t3 = now_ns();
    rep[RP_INSTANCES] = ni;
    rep[RP_ROWS] = nr;
    rep[RP_NS_COMMIT] = t3 - t2;
    rep[RP_NS_TOTAL] = t3 - t0;
    rep[RP_STATUS] = status;
    return status;
}
