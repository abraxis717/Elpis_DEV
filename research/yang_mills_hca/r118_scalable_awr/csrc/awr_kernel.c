/* R118 native backend: the frozen 88-parameter HCA over sparse 4-slot windows.
 *
 * RESEARCH_ONLY. Compiled at runtime into volatile scratch by native.py; no
 * binary is ever committed. Build flags: -std=c11 -O2 -ffp-contract=off
 * (no fused multiply-add contraction, IEEE double semantics, glibc libm).
 *
 * Mathematics (frozen sources r62_source.graph_step_split, r63_source.hybrid_step,
 * r30_source.invariants; controller research/yang_mills_hca/reference.py):
 *
 *   per slot d with in-links e = (src s, shared participant p), canonical order
 *   (src ascending, participant index ascending):
 *     L_e   = U[r_d,p] U[r_s,p]^T              (SO(2)xSO(2) blocks)
 *     w_e   = L_e A_s ;  wc_e = L_e B_s
 *     sal_d = sum_e softmax_e(8 |w_e|) w_e     (0 if no in-links)
 *     col_d = mean_e wc_e                      (0 if no in-links)
 *     f     = squash(|A|^2,|B|^2,|sal|^2,|col|^2,A.sal,A.col,B.sal,B.col,sal.col,0)
 *     k     = [sig, tanh x3, sig, tanh x3](W f + b)
 *     A'    = clamp4(k0 A + k1 sal + k2 col + k3 B)
 *     B'    = clamp4(k4 B + k5 sal + k6 col + k7 A)
 *   controller over the instance's 32 coordinates (exactly rounded sums = math.fsum):
 *     d = p - s; r = s * fsum(s.d) / max(fsum(s.s), 1e-24); p' = p + alpha r + beta (d - r)
 *
 * Reconciliation (MEAN): each compact relation row receives the mean of its
 * candidates; per coordinate the candidates are sorted ascending and summed in
 * that order, so the result is independent of the order of active instances.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__) || defined(__SSE2__)
#include <xmmintrin.h>
#define AWR_HAVE_MXCSR 1
#else
#define AWR_HAVE_MXCSR 0
#endif

#define AWR_SLOTS 4
#define AWR_SC 8            /* coordinates per relation node: A[4], B[4] */
#define AWR_MAX_ARITY 64
#define AWR_MAX_IN (3 * AWR_MAX_ARITY)
#define AWR_MAX_BIND 1024

enum {
    AWR_OK = 0,
    AWR_ERR_NONFINITE = 1,
    AWR_ERR_INDEGREE = 2,
    AWR_ERR_ARGUMENT = 3,
    AWR_ERR_OVERLAP = 4,
    AWR_ERR_BINDINGS = 5,
    AWR_ERR_DEAD_HANDLE = 6,
    AWR_ERR_DUPLICATE_HANDLE = 7,
    AWR_ERR_ROW_CAPACITY = 8
};

int awr_abi_version(void) { return 4; }

/* ---- exactly rounded summation: a port of CPython's math.fsum (finite inputs) ---- */
static double awr_fsum(const double *v, int n) {
    double p[2 * 32 + 8];
    int np = 0;
    for (int k = 0; k < n; k++) {
        double x = v[k];
        int i = 0;
        for (int j = 0; j < np; j++) {
            double y = p[j];
            if (fabs(x) < fabs(y)) { double t = x; x = y; y = t; }
            double hi = x + y;
            double yr = hi - x;
            double lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        np = i;
        if (x != 0.0) p[np++] = x;
    }
    double hi = 0.0, lo = 0.0;
    if (np > 0) {
        hi = p[--np];
        while (np > 0) {
            double x = hi;
            double y = p[--np];
            hi = x + y;
            double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (np > 0 && ((lo < 0.0 && p[np - 1] < 0.0) || (lo > 0.0 && p[np - 1] > 0.0))) {
            double y = lo * 2.0;
            double x = hi + y;
            double yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

double awr_fsum_probe(const double *v, int32_t n) { return (n >= 0 && n <= 32) ? awr_fsum(v, n) : NAN; }

static inline double squash(double x) { return x / (1.0 + fabs(x)); }
static inline double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double dot4(const double *x, const double *y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

/* y = blockdiag(L) x, L given as two row-major 2x2 blocks (8 doubles). */
static inline void apply_link(const double *L, const double *x, double *y) {
    y[0] = L[0] * x[0] + L[1] * x[1];
    y[1] = L[2] * x[0] + L[3] * x[1];
    y[2] = L[4] * x[2] + L[5] * x[3];
    y[3] = L[6] * x[2] + L[7] * x[3];
}

/* L = Ud Us^T blockwise: (Bd Bs^T)[i][j] = sum_k Bd[i][k] Bs[j][k]. */
static inline void link_op(const double *Ud, const double *Us, double *L) {
    for (int b = 0; b < 2; b++) {
        const double *d = Ud + 4 * b, *s = Us + 4 * b;
        double *o = L + 4 * b;
        o[0] = d[0] * s[0] + d[1] * s[1];
        o[1] = d[0] * s[2] + d[1] * s[3];
        o[2] = d[2] * s[0] + d[3] * s[1];
        o[3] = d[2] * s[2] + d[3] * s[3];
    }
}

static inline void clamp4(double *v) {
    double n = sqrt(dot4(v, v));
    double scale = 4.0 / (n + 1e-30);
    if (scale > 1.0) scale = 1.0;
    for (int k = 0; k < 4; k++) v[k] = v[k] * scale;
}

/* One instance: gather (from compact rows), transport, HCA, controller -> cand[4][8]. */
static int instance_pass(const int32_t *rel, const int32_t *cmp, const double *cin,
                         const int64_t *rel_off, const int32_t *members, const double *frames,
                         const double *W, const double *bias, int32_t correct, double alpha, double beta,
                         double *cand, int64_t *links) {
    double A[AWR_SLOTS][4], B[AWR_SLOTS][4], S[AWR_SLOTS][4], Cc[AWR_SLOTS][4];
    double P[AWR_SLOTS][AWR_SC];
    double w[AWR_MAX_IN][4], wc[AWR_MAX_IN][4], x[AWR_MAX_IN];
    for (int d = 0; d < AWR_SLOTS; d++) {
        if (cmp[d] >= 0) {
            const double *src = cin + (int64_t)cmp[d] * AWR_SC;
            for (int k = 0; k < 4; k++) { A[d][k] = src[k]; B[d][k] = src[4 + k]; }
        } else {
            for (int k = 0; k < 4; k++) { A[d][k] = 0.0; B[d][k] = 0.0; }
        }
    }
    for (int d = 0; d < AWR_SLOTS; d++) {
        for (int k = 0; k < 4; k++) { S[d][k] = 0.0; Cc[d][k] = 0.0; }
        if (rel[d] < 0) continue;
        int ne = 0;
        const int64_t d0 = rel_off[rel[d]], d1 = rel_off[rel[d] + 1];
        for (int s = 0; s < AWR_SLOTS; s++) {
            if (s == d || rel[s] < 0) continue;
            int64_t i = d0, j = rel_off[rel[s]];
            const int64_t j1 = rel_off[rel[s] + 1];
            while (i < d1 && j < j1) {
                if (members[i] < members[j]) { i++; }
                else if (members[i] > members[j]) { j++; }
                else {
                    if (ne >= AWR_MAX_IN) return AWR_ERR_INDEGREE;
                    double L[8];
                    link_op(frames + 8 * i, frames + 8 * j, L);
                    apply_link(L, A[s], w[ne]);
                    apply_link(L, B[s], wc[ne]);
                    x[ne] = 8.0 * sqrt(dot4(w[ne], w[ne]));
                    ne++;
                    i++;
                    j++;
                }
            }
        }
        *links += ne;
        if (ne == 0) continue;
        double mx = x[0];
        for (int e = 1; e < ne; e++) if (x[e] > mx) mx = x[e];
        double tot = 0.0;
        for (int e = 0; e < ne; e++) { x[e] = exp(x[e] - mx); tot += x[e]; }
        for (int e = 0; e < ne; e++) {
            double pe = x[e] / tot;
            for (int k = 0; k < 4; k++) S[d][k] += pe * w[e][k];
        }
        for (int k = 0; k < 4; k++) {
            double acc = 0.0;
            for (int e = 0; e < ne; e++) acc += wc[e][k];
            Cc[d][k] = acc / (double)ne;
        }
    }
    for (int d = 0; d < AWR_SLOTS; d++) {
        if (rel[d] < 0) { for (int k = 0; k < AWR_SC; k++) P[d][k] = 0.0; continue; }
        const double *a = A[d], *b = B[d], *s = S[d], *c = Cc[d];
        double f[10] = {dot4(a, a), dot4(b, b), dot4(s, s), dot4(c, c), dot4(a, s),
                        dot4(a, c), dot4(b, s), dot4(b, c), dot4(s, c), 0.0};
        for (int j = 0; j < 10; j++) f[j] = squash(f[j]);
        double kk[8];
        for (int q = 0; q < 8; q++) {
            double z = 0.0;
            for (int j = 0; j < 10; j++) z += W[q * 10 + j] * f[j];
            z = z + bias[q];
            kk[q] = (q == 0 || q == 4) ? sigm(z) : tanh(z);
        }
        double ua[4], ub[4];
        for (int k = 0; k < 4; k++) {
            ua[k] = kk[0] * a[k] + kk[1] * s[k] + kk[2] * c[k] + kk[3] * b[k];
            ub[k] = kk[4] * b[k] + kk[5] * s[k] + kk[6] * c[k] + kk[7] * a[k];
        }
        clamp4(ua);
        clamp4(ub);
        for (int k = 0; k < 4; k++) { P[d][k] = ua[k]; P[d][4 + k] = ub[k]; }
    }
    if (correct) {
        double s32[32], dd[32], prod[32];
        for (int d = 0; d < AWR_SLOTS; d++)
            for (int k = 0; k < 4; k++) {
                s32[d * 4 + k] = A[d][k];
                s32[16 + d * 4 + k] = B[d][k];
            }
        double p32[32];
        for (int d = 0; d < AWR_SLOTS; d++)
            for (int k = 0; k < 4; k++) {
                p32[d * 4 + k] = P[d][k];
                p32[16 + d * 4 + k] = P[d][4 + k];
            }
        for (int i = 0; i < 32; i++) dd[i] = p32[i] - s32[i];
        for (int i = 0; i < 32; i++) prod[i] = s32[i] * s32[i];
        double norm = awr_fsum(prod, 32);
        for (int i = 0; i < 32; i++) prod[i] = s32[i] * dd[i];
        double dt = awr_fsum(prod, 32);
        double den = (1e-24 > norm) ? 1e-24 : norm;
        for (int i = 0; i < 32; i++) {
            double r = s32[i] * dt / den;
            p32[i] = p32[i] + alpha * r + beta * (dd[i] - r);
        }
        for (int d = 0; d < AWR_SLOTS; d++)
            for (int k = 0; k < 4; k++) {
                P[d][k] = p32[d * 4 + k];
                P[d][4 + k] = p32[16 + d * 4 + k];
            }
    }
    for (int d = 0; d < AWR_SLOTS; d++)
        for (int k = 0; k < AWR_SC; k++) {
            double v = P[d][k];
            if (!isfinite(v)) return AWR_ERR_NONFINITE;
            cand[d * AWR_SC + k] = v;
        }
    return AWR_OK;
}

/* Active-set compaction: validates handles (sorted, unique, live) and maps every
 * bound slot to a compact row. mark[] is a pool-sized table that must be -1 on
 * entry for every row; touched entries are reset by awr_release_marks. */
int awr_compact(int32_t n, const int32_t *handles, const uint8_t *live, int32_t instance_capacity,
                const int32_t *slot_rel_table, const int32_t *slot_row_table, int32_t *mark,
                int32_t *slot_rel, int32_t *slot_cmp, int32_t *rows, int32_t *nr_out, int32_t max_rows) {
    int32_t nr = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t h = handles[i];
        if (h < 0 || h >= instance_capacity) { *nr_out = nr; return AWR_ERR_ARGUMENT; }
        if (i > 0 && handles[i - 1] >= h) { *nr_out = nr; return AWR_ERR_DUPLICATE_HANDLE; }
        if (!live[h]) { *nr_out = nr; return AWR_ERR_DEAD_HANDLE; }
        for (int d = 0; d < AWR_SLOTS; d++) {
            int32_t r = slot_rel_table[(int64_t)h * AWR_SLOTS + d];
            int32_t row = slot_row_table[(int64_t)h * AWR_SLOTS + d];
            slot_rel[i * AWR_SLOTS + d] = r;
            if (r < 0) { slot_cmp[i * AWR_SLOTS + d] = -1; continue; }
            if (mark[row] < 0) {
                if (nr >= max_rows) { *nr_out = nr; return AWR_ERR_ROW_CAPACITY; }
                mark[row] = nr;
                rows[nr++] = row;
            }
            slot_cmp[i * AWR_SLOTS + d] = mark[row];
        }
    }
    *nr_out = nr;
    return AWR_OK;
}

void awr_release_marks(int32_t nr, const int32_t *rows, int32_t *mark) {
    for (int32_t k = 0; k < nr; k++) mark[rows[k]] = -1;
}

/* Order-independent reconciliation of candidates into compact rows. */
static int reconcile(int32_t n, const int32_t *slot_cmp, const double *cand, int32_t nr, double *cout,
                     int32_t mode, int32_t *count, int32_t *off, int32_t *list) {
    for (int32_t r = 0; r < nr; r++) count[r] = 0;
    for (int32_t i = 0; i < n * AWR_SLOTS; i++) if (slot_cmp[i] >= 0) count[slot_cmp[i]]++;
    off[0] = 0;
    for (int32_t r = 0; r < nr; r++) {
        if (count[r] < 1) return AWR_ERR_ARGUMENT;
        if (count[r] > AWR_MAX_BIND) return AWR_ERR_BINDINGS;
        if (mode == 1 && count[r] > 1) return AWR_ERR_OVERLAP;
        off[r + 1] = off[r] + count[r];
        count[r] = 0;
    }
    for (int32_t i = 0; i < n * AWR_SLOTS; i++) {
        int32_t r = slot_cmp[i];
        if (r >= 0) list[off[r] + count[r]++] = i;
    }
    double v[AWR_MAX_BIND];
    for (int32_t r = 0; r < nr; r++) {
        int32_t m = off[r + 1] - off[r];
        const int32_t *ids = list + off[r];
        for (int k = 0; k < AWR_SC; k++) {
            if (m == 1) { cout[(int64_t)r * AWR_SC + k] = cand[(int64_t)ids[0] * AWR_SC + k]; continue; }
            for (int32_t q = 0; q < m; q++) {
                double val = cand[(int64_t)ids[q] * AWR_SC + k];
                int32_t t = q;
                while (t > 0 && v[t - 1] > val) { v[t] = v[t - 1]; t--; }
                v[t] = val;
            }
            double acc = v[0];
            for (int32_t q = 1; q < m; q++) acc += v[q];
            cout[(int64_t)r * AWR_SC + k] = acc / (double)m;
        }
    }
    return AWR_OK;
}

/* Gather -> passes x (HCA + controller + reconciliation) -> commit on success only.
 * buf_a/buf_b: compact-state buffers [nr*8]; cand: [n*4*8]. Pool rows are
 * written only if every pass succeeded (transactional). */
int awr_execute(int32_t n, const int32_t *slot_rel, const int32_t *slot_cmp, int32_t nr, const int32_t *rows,
                double *pool, double *buf_a, double *buf_b, double *cand,
                const int64_t *rel_off, const int32_t *members, const double *frames,
                const double *W, const double *bias, int32_t correct, double alpha, double beta,
                int32_t passes, int32_t mode, int32_t *count, int32_t *off, int32_t *list,
                int64_t *links_out, int32_t flush_subnormals) {
    /* Optional flush-to-zero / denormals-are-zero for this call only (opt-in, default off):
     * the caller's floating-point environment is restored on every exit path. */
#if AWR_HAVE_MXCSR
    const unsigned int csr = _mm_getcsr();
    if (flush_subnormals) _mm_setcsr(csr | 0x8040u);
#else
    if (flush_subnormals) { *links_out = 0; return AWR_ERR_ARGUMENT; }
#endif
    int rc = AWR_OK;
    for (int32_t r = 0; r < nr; r++)
        memcpy(buf_a + (int64_t)r * AWR_SC, pool + (int64_t)rows[r] * AWR_SC, sizeof(double) * AWR_SC);
    double *cur = buf_a, *nxt = buf_b;
    int64_t links = 0;
    for (int32_t pss = 0; pss < passes && rc == AWR_OK; pss++) {
        for (int32_t i = 0; i < n && rc == AWR_OK; i++)
            rc = instance_pass(slot_rel + i * AWR_SLOTS, slot_cmp + i * AWR_SLOTS, cur, rel_off, members,
                               frames, W, bias, correct, alpha, beta, cand + (int64_t)i * AWR_SLOTS * AWR_SC,
                               &links);
        if (rc == AWR_OK) rc = reconcile(n, slot_cmp, cand, nr, nxt, mode, count, off, list);
        if (rc == AWR_OK) { double *t = cur; cur = nxt; nxt = t; }
    }
    if (rc == AWR_OK)
        for (int32_t r = 0; r < nr; r++)
            memcpy(pool + (int64_t)rows[r] * AWR_SC, cur + (int64_t)r * AWR_SC, sizeof(double) * AWR_SC);
    *links_out = links;
#if AWR_HAVE_MXCSR
    if (flush_subnormals) _mm_setcsr(csr);
#endif
    return rc;
}

unsigned int awr_mxcsr_probe(void) {
#if AWR_HAVE_MXCSR
    return _mm_getcsr();
#else
    return 0u;
#endif
}

/* Candidate-only entry for parity tests: one pass, no reconciliation, no commit. */
int awr_candidates(int32_t n, const int32_t *slot_rel, const int32_t *slot_cmp, const double *cin, double *cand,
                   const int64_t *rel_off, const int32_t *members, const double *frames,
                   const double *W, const double *bias, int32_t correct, double alpha, double beta,
                   int64_t *links_out) {
    int64_t links = 0;
    for (int32_t i = 0; i < n; i++) {
        int rc = instance_pass(slot_rel + i * AWR_SLOTS, slot_cmp + i * AWR_SLOTS, cin, rel_off, members, frames,
                               W, bias, correct, alpha, beta, cand + (int64_t)i * AWR_SLOTS * AWR_SC, &links);
        if (rc != AWR_OK) { *links_out = links; return rc; }
    }
    *links_out = links;
    return AWR_OK;
}
