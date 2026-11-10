/* R119 native backend: the R118 kernel (csrc/awr_kernel.c, SHA-256 87b53392...) plus
 * GUARDED EXACT fast paths for the subnormal regime. RESEARCH_ONLY.
 *
 * Every guard replaces a computation by one that is proven to give bit-identical
 * IEEE-754 binary64 results (round-to-nearest-even, no FMA contraction, no FTZ), and is
 * taken only when a cheap, assist-free sufficient condition holds (integer comparisons
 * on bit patterns). Otherwise the R118 arithmetic below runs verbatim. Guards (bit mask):
 *   G_SOFTMAX 1: all transported A-states |w| < 2^-60  => every exp(x_e - max) == 1.0,
 *                softmax weights == 1/ne exactly (needs libm property checked by
 *                awr119_libm_selfcheck: exp(t) == 1.0 for t in [-2^-56, 0]).
 *   G_GATES   2: all slot inputs |a|,|b|,|sal|,|col| < theta (exact rational bound,
 *                engine_r119/guards.py) => fl(W.f-sum + bias) == bias for every gate.
 *   G_CLAMP   4: max |v| < 1.9375 => |v| < 3.875 < 4 => clamp scale == 1 exactly.
 *   G_RADIAL  8: integer exponent bound proves |alpha r| and |r| below a quarter-ulp of
 *                p and of d for every coordinate => p + alpha r + beta (d - r) == p + beta d.
 * The forensic census (forensics.py) attributes the R118-D exception to exactly the
 * computations these guards skip. Compiled at runtime into volatile scratch; no
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

int awr119_abi_version(void) { return 119; }

#define AWR_G_SOFTMAX 1u
#define AWR_G_GATES 2u
#define AWR_G_CLAMP 4u
#define AWR_G_RADIAL 8u
#define AWR_BITS_2P_M60 0x3C30000000000000ULL   /* 2^-60 */
#define AWR_BITS_1P9375 0x3FFF000000000000ULL   /* 1.9375 = 31/16 */
/* guard counters: [0..3] hits (softmax, gates, clamp, radial), [4..7] evaluations */

static inline uint64_t dbits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }
/* For finite doubles, |x| <= |y| iff (bits(x) & ~sign) <= (bits(y) & ~sign). */
static inline uint64_t absbits(double x) { return dbits(x) & 0x7fffffffffffffffULL; }
static inline uint64_t maxabs4(const double *v) {
    uint64_t m = absbits(v[0]);
    for (int k = 1; k < 4; k++) { uint64_t b = absbits(v[k]); if (b > m) m = b; }
    return m;
}
static inline int bexp(double x) { return (int)((dbits(x) >> 52) & 0x7ff); }   /* biased exponent */

/* libm precondition of G_SOFTMAX: exp(t) == 1.0 for every t in [-2^-56, 0]. Checked on the
 * boundary, on every binade down to the smallest subnormal, and on +-0. */
int awr119_libm_selfcheck(void) {
    if (exp(0.0) != 1.0 || exp(-0.0) != 1.0) return 0;
    for (int e = -56; e >= -1074; e--) {                 /* t = -2^e: boundary and every binade */
        if (exp(-ldexp(1.0, e)) != 1.0) return 0;
    }
    for (int e = -57; e >= -1021; e--) {                 /* t = -1.5 * 2^e, strictly inside */
        if (exp(-ldexp(1.5, e)) != 1.0) return 0;
    }
    return 1;
}

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

double awr119_fsum_probe(const double *v, int32_t n) { return (n >= 0 && n <= 32) ? awr_fsum(v, n) : NAN; }

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

/* G_RADIAL condition (integer arithmetic only; no floating-point operation on tiny values).
 * Original: r_i = fl(fl(s_i*dt)/den), dt = fsum(fl(s_j*d_j)), den >= 1e-24 > 2^-80;
 *           out_i = fl(fl(p_i + fl(alpha r_i)) + fl(beta * fl(d_i - r_i))).
 * With |s_j| < 2^(Es+1), |d_j| < 2^(Ed+1): |fl(s_j d_j)| <= 2^max(Es+Ed+2, -1074), so
 * |dt| <= 2^Edt, Edt = max(Es+Ed+2, -1074) + 5 (32 terms, exactly rounded sum), and
 * |r_i| <= 2^Er_i, Er_i = max(E(s_i)+1+Edt, -1074) + 80; alpha, beta <= 1.
 * If p_i, d_i are normal and Er_i <= E(p_i) - 55 and Er_i <= E(d_i) - 55, then
 * |fl(alpha r_i)| < 2^(E(p_i)-54) (half the smallest gap at p_i) => fl(p_i + .) == p_i, and
 * likewise fl(d_i - r_i) == d_i, hence out_i == fl(p_i + fl(beta d_i)) bit for bit.
 * s_i == 0 gives r_i == +-0 and the same identities for nonzero p_i, d_i. A coordinate with
 * s_i, p_i, d_i all +0.0 (empty slot) gives +0 on both paths. Anything else: no guard. */
static int radial_negligible(const double *s, const double *p, const double *d, double alpha, double beta) {
    if (!(alpha <= 1.0) || !(beta <= 1.0) || alpha < 0.0 || beta < 0.0) return 0;
    int es = -2000, ed = -2000;
    for (int i = 0; i < 32; i++) {
        uint64_t as = absbits(s[i]), ad = absbits(d[i]);
        if (as) { int e = bexp(s[i]); e = (e == 0) ? -1022 : e - 1023; if (e > es) es = e; }
        if (ad) { int e = bexp(d[i]); e = (e == 0) ? -1022 : e - 1023; if (e > ed) ed = e; }
    }
    int edt = ((es + ed + 2) > -1074 ? (es + ed + 2) : -1074) + 5;
    for (int i = 0; i < 32; i++) {
        const uint64_t bs = dbits(s[i]), bp = dbits(p[i]), bd = dbits(d[i]);
        if (bs == 0 && bp == 0 && bd == 0) continue;              /* +0, +0, +0 */
        const int xp = bexp(p[i]), xd = bexp(d[i]);
        if (xp == 0 || xd == 0) return 0;                       /* zero or subnormal p/d */
        if (absbits(s[i]) == 0) continue;                       /* r_i == +-0, p_i, d_i normal */
        int xs = bexp(s[i]);
        int esi = (xs == 0) ? -1022 : xs - 1023;
        int t = esi + 1 + edt;
        int er = (t > -1074 ? t : -1074) + 80;
        if (er > (xp - 1023) - 55 || er > (xd - 1023) - 55) return 0;
    }
    return 1;
}

/* One instance: gather (from compact rows), transport, HCA, controller -> cand[4][8]. */
static int instance_pass(const int32_t *rel, const int32_t *cmp, const double *cin,
                         const int64_t *rel_off, const int32_t *members, const double *frames,
                         const double *W, const double *bias, int32_t correct, double alpha, double beta,
                         double *cand, int64_t *links, uint32_t gmask, uint64_t theta_bits, int64_t *gc) {
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
                    ne++;
                    i++;
                    j++;
                }
            }
        }
        *links += ne;
        if (ne == 0) continue;
        int fast_sm = 0;
        if (gmask & AWR_G_SOFTMAX) {
            uint64_t m = 0;
            for (int e = 0; e < ne; e++) { uint64_t t = maxabs4(w[e]); if (t > m) m = t; }
            fast_sm = m < AWR_BITS_2P_M60;
            gc[4]++;
        }
        if (fast_sm) {
            /* |w| < 2^-60 => x_e = 8 sqrt(w.w) <= 2^-56 => x_e - max in [-2^-56, 0] => exp == 1.0,
             * tot == ne exactly, x_e / tot == 1.0 / ne: identical to the branch below. */
            gc[0]++;
            const double pe = 1.0 / (double)ne;
            for (int e = 0; e < ne; e++)
                for (int k = 0; k < 4; k++) S[d][k] += pe * w[e][k];
        } else {
            for (int e = 0; e < ne; e++) x[e] = 8.0 * sqrt(dot4(w[e], w[e]));
            double mx = x[0];
            for (int e = 1; e < ne; e++) if (x[e] > mx) mx = x[e];
            double tot = 0.0;
            for (int e = 0; e < ne; e++) { x[e] = exp(x[e] - mx); tot += x[e]; }
            for (int e = 0; e < ne; e++) {
                double pe = x[e] / tot;
                for (int k = 0; k < 4; k++) S[d][k] += pe * w[e][k];
            }
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
        double kk[8];
        int fast_g = 0;
        if (gmask & AWR_G_GATES) {
            uint64_t m = maxabs4(a), t;
            if ((t = maxabs4(b)) > m) m = t;
            if ((t = maxabs4(s)) > m) m = t;
            if ((t = maxabs4(c)) > m) m = t;
            fast_g = m < theta_bits;
            gc[5]++;
        }
        if (fast_g) {
            /* max input < theta => |sum_j W_qj f_j| <= gap(bias_q)/4 => z == bias_q exactly. */
            gc[1]++;
            for (int q = 0; q < 8; q++) kk[q] = (q == 0 || q == 4) ? sigm(bias[q]) : tanh(bias[q]);
        } else {
            double f[10] = {dot4(a, a), dot4(b, b), dot4(s, s), dot4(c, c), dot4(a, s),
                            dot4(a, c), dot4(b, s), dot4(b, c), dot4(s, c), 0.0};
            for (int j = 0; j < 10; j++) f[j] = squash(f[j]);
            for (int q = 0; q < 8; q++) {
                double z = 0.0;
                for (int j = 0; j < 10; j++) z += W[q * 10 + j] * f[j];
                z = z + bias[q];
                kk[q] = (q == 0 || q == 4) ? sigm(z) : tanh(z);
            }
        }
        double ua[4], ub[4];
        for (int k = 0; k < 4; k++) {
            ua[k] = kk[0] * a[k] + kk[1] * s[k] + kk[2] * c[k] + kk[3] * b[k];
            ub[k] = kk[4] * b[k] + kk[5] * s[k] + kk[6] * c[k] + kk[7] * a[k];
        }
        if (gmask & AWR_G_CLAMP) {
            /* max|v| < 1.9375 => |v| <= 3.875 => 4/(|v|+1e-30) > 1 => scale == 1, v*1 == v. */
            gc[6] += 2;
            if (maxabs4(ua) < AWR_BITS_1P9375) gc[2]++; else clamp4(ua);
            if (maxabs4(ub) < AWR_BITS_1P9375) gc[2]++; else clamp4(ub);
        } else {
            clamp4(ua);
            clamp4(ub);
        }
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
        int fast_r = 0;
        if (gmask & AWR_G_RADIAL) {
            gc[7]++;
            fast_r = radial_negligible(s32, p32, dd, alpha, beta);
        }
        if (fast_r) {
            gc[3]++;
            for (int i = 0; i < 32; i++) p32[i] = p32[i] + beta * dd[i];
        } else {
            for (int i = 0; i < 32; i++) prod[i] = s32[i] * s32[i];
            double norm = awr_fsum(prod, 32);
            for (int i = 0; i < 32; i++) prod[i] = s32[i] * dd[i];
            double dt = awr_fsum(prod, 32);
            double den = (1e-24 > norm) ? 1e-24 : norm;
            for (int i = 0; i < 32; i++) {
                double r = s32[i] * dt / den;
                p32[i] = p32[i] + alpha * r + beta * (dd[i] - r);
            }
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
 * entry for every row; touched entries are reset by awr119_release_marks. */
int awr119_compact(int32_t n, const int32_t *handles, const uint8_t *live, int32_t instance_capacity,
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

void awr119_release_marks(int32_t nr, const int32_t *rows, int32_t *mark) {
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
int awr119_execute(int32_t n, const int32_t *slot_rel, const int32_t *slot_cmp, int32_t nr, const int32_t *rows,
                double *pool, double *buf_a, double *buf_b, double *cand,
                const int64_t *rel_off, const int32_t *members, const double *frames,
                const double *W, const double *bias, int32_t correct, double alpha, double beta,
                int32_t passes, int32_t mode, int32_t *count, int32_t *off, int32_t *list,
                int64_t *links_out, int32_t flush_subnormals, uint32_t gmask, double theta, int64_t *gc) {
    const uint64_t theta_bits = absbits(theta);
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
                               &links, gmask, theta_bits, gc);
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
int awr119_candidates(int32_t n, const int32_t *slot_rel, const int32_t *slot_cmp, const double *cin, double *cand,
                   const int64_t *rel_off, const int32_t *members, const double *frames,
                   const double *W, const double *bias, int32_t correct, double alpha, double beta,
                   int64_t *links_out, uint32_t gmask, double theta, int64_t *gc) {
    const uint64_t theta_bits = absbits(theta);
    int64_t links = 0;
    for (int32_t i = 0; i < n; i++) {
        int rc = instance_pass(slot_rel + i * AWR_SLOTS, slot_cmp + i * AWR_SLOTS, cin, rel_off, members, frames,
                               W, bias, correct, alpha, beta, cand + (int64_t)i * AWR_SLOTS * AWR_SC, &links,
                               gmask, theta_bits, gc);
        if (rc != AWR_OK) { *links_out = links; return rc; }
    }
    *links_out = links;
    return AWR_OK;
}
