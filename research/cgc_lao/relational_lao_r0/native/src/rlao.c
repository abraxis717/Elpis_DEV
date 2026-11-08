/*
 * Relational LAO R0: native C11 relational adapter. See rlao.h.
 *
 * A statement-for-statement port of src/relational_lao/scalar.py. Every sum is a left-to-right loop; the log
 * evidence of the slots is summed over sorted terms; transcendentals are the C library's exp, log, sqrt and pow.
 * Build with -ffp-contract=off (no fused multiply-add) so that it is bitwise equal to the reference.
 * No heap, no recursion, no I/O, no global mutable state; every loop is bounded by a compile-time capacity.
 */
#include "rlao.h"

#include <math.h>
#include <string.h>

#define CONFIG_MAGIC 0x52434647u   /* "RCFG" */
#define STATE_MAGIC 0x52535441u    /* "RSTA" */
#define STATE_POISONED 0x504f4953u /* "POIS" */

static const double LW_FLOOR = -1.0e4;
static const double V_FLOOR = 1.0e-6;
static const double L_FLOOR = 1e-300;

/* 16-node probabilists' Gauss-Hermite rule normalised to N(0, 1): the literals of cgc_lao_lab.filters. */
static const double GH_X[RLAO_GH] = {
    -6.630878198393129, -5.472225705949343, -4.492955302520011, -3.6008736241715487,
    -2.7602450476307014, -1.9519803457163334, -1.1638291005549648, -0.3867606045005574,
    0.3867606045005574, 1.1638291005549648, 1.9519803457163334, 2.7602450476307014,
    3.6008736241715487, 4.492955302520011, 5.472225705949343, 6.630878198393129};
static const double GH_W[RLAO_GH] = {
    1.4978147231618412e-10, 1.309473216286817e-07, 1.530003216248732e-05, 0.0005259849265739087,
    0.007266937601184749, 0.04728475235401406, 0.1583383727509497, 0.286568521238012,
    0.286568521238012, 0.1583383727509497, 0.04728475235401406, 0.007266937601184749,
    0.0005259849265739087, 1.530003216248732e-05, 1.309473216286817e-07, 1.4978147231618412e-10};

size_t rlao_state_size(void) { return sizeof(rlao_state); }
size_t rlao_config_size(void) { return sizeof(rlao_config); }

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t config_digest(const rlao_config *c) {
    uint64_t h = 14695981039346656037ull;
    h = fnv(h, &c->hypotheses, sizeof c->hypotheses);
    h = fnv(h, &c->slots, sizeof c->slots);
    h = fnv(h, &c->readout, sizeof c->readout);
    h = fnv(h, c->q, sizeof c->q);
    h = fnv(h, c->rho, sizeof c->rho);
    h = fnv(h, c->log_prior, sizeof c->log_prior);
    h = fnv(h, &c->bias_var, sizeof c->bias_var);
    return h;
}

static int config_ok(const rlao_config *c) {
    return c != NULL && c->magic == CONFIG_MAGIC && c->digest == config_digest(c);
}

static int state_ok(const rlao_state *s, const rlao_config *c) {
    return s != NULL && s->magic == STATE_MAGIC && s->config_digest == c->digest && s->slots == c->slots &&
           s->hypotheses == c->hypotheses && s->buffered <= RLAO_BUFFER;
}

int rlao_config_init(rlao_config *cfg, uint32_t hypotheses, const double *q, const double *rho, const double *prior,
                     uint32_t slots, double bias_var, uint32_t readout) {
    if (!cfg || !q || !rho || !prior) return RLAO_E_NULL;
    if (hypotheses < 1u || hypotheses > RLAO_MAX_HYP || slots < 1u || slots > RLAO_MAX_SLOTS ||
        readout > RLAO_READOUT_INDEPENDENT || !(bias_var > 0.0) || !isfinite(bias_var))
        return RLAO_E_CONFIG;
    double total = 0.0;
    for (uint32_t m = 0; m < hypotheses; ++m) {
        if (!(q[m] >= 0.0 && q[m] < 0.5) || !(rho[m] >= 0.0 && rho[m] <= 1.0) || !(prior[m] > 0.0) ||
            !isfinite(prior[m]))
            return RLAO_E_CONFIG;
        total += prior[m];
    }
    rlao_config c;
    memset(&c, 0, sizeof c);
    c.hypotheses = hypotheses;
    c.slots = slots;
    c.readout = readout;
    c.bias_var = bias_var;
    for (uint32_t m = 0; m < hypotheses; ++m) {
        c.q[m] = q[m];
        c.rho[m] = rho[m];
        c.log_prior[m] = log(prior[m] / total);
    }
    c.magic = CONFIG_MAGIC;
    c.digest = config_digest(&c);
    *cfg = c;
    return RLAO_OK;
}

static void reset_arrays(rlao_state *st, const rlao_config *cfg) {
    for (uint32_t m = 0; m < RLAO_MAX_HYP; ++m)
        for (uint32_t r = 0; r < RLAO_MAX_SLOTS; ++r) {
            st->mu[m][r] = 0.0;
            st->v[m][r] = (m < cfg->hypotheses && r < cfg->slots) ? cfg->bias_var : 0.0;
            st->lam[m][r] = 0.0;
        }
    for (uint32_t r = 0; r < RLAO_MAX_SLOTS; ++r) st->counts[r] = 0u;
}

int rlao_reset(rlao_state *st, const rlao_config *cfg) {
    if (!st || !cfg) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    memset(st, 0, sizeof *st);
    st->slots = cfg->slots;
    st->hypotheses = cfg->hypotheses;
    st->config_digest = cfg->digest;
    reset_arrays(st, cfg);
    st->magic = STATE_MAGIC;
    return RLAO_OK;
}

static double sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    const double e = exp(z);
    return e / (1.0 + e);
}

static void evidence(const rlao_config *cfg, uint32_t m, const uint8_t *reads, uint32_t n, double *a1, double *a0) {
    const double q = cfg->q[m];
    for (int k = 0; k < 2; ++k) {
        const unsigned c = k == 0 ? 1u : 0u;
        double p;
        if (n == 1u) {
            const double f = (double)(reads[0] ^ c);
            p = pow(q, f) * pow(1.0 - q, 1.0 - f);
        } else {
            const double f0 = (double)(reads[0] ^ c), f1 = (double)(reads[1] ^ c);
            const double s = f0 + f1;
            const double indep = pow(q, s) * pow(1.0 - q, 2.0 - s);
            const double shared = s == 1.0 ? 0.0 : pow(q, f0) * pow(1.0 - q, 1.0 - f0);
            p = (1.0 - cfg->rho[m]) * indep + cfg->rho[m] * shared;
        }
        if (k == 0) *a1 = p; else *a0 = p;
    }
}

/* The RMF3 law on one slot (scalar.ScalarRelational.update); returns 0 or a numerical failure. */
static int apply(rlao_state *st, const rlao_config *cfg, uint32_t r, double z, const uint8_t *reads, uint32_t n) {
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
        double a1, a0;
        evidence(cfg, m, reads, n, &a1, &a0);
        const double mu = st->mu[m][r], sd = sqrt(st->v[m][r]);
        double L = 0.0, m1 = 0.0, m2 = 0.0;
        for (uint32_t g = 0; g < RLAO_GH; ++g) {
            const double b = mu + sd * GH_X[g];
            const double p = sigmoid(z + b);
            const double lik = (p * a1 + (1.0 - p) * a0) * GH_W[g];
            L += lik;
            m1 += lik * b;
            m2 += lik * b * b;
        }
        if (L > L_FLOOR) {
            m1 = m1 / L;
            m2 = m2 / L;
            const double var = m2 - m1 * m1;
            st->mu[m][r] = m1;
            st->v[m][r] = var > V_FLOOR ? var : V_FLOOR;
            st->lam[m][r] += log(L);
        } else {
            st->lam[m][r] += LW_FLOOR;
        }
        if (!isfinite(st->mu[m][r]) || !isfinite(st->v[m][r]) || !isfinite(st->lam[m][r])) return 1;
    }
    st->counts[r] += 1u;
    return 0;
}

static int reads_ok(const uint8_t *reads, uint32_t n) {
    if (!reads || n < 1u || n > RLAO_MAX_READS) return 0;
    for (uint32_t k = 0; k < n; ++k)
        if (reads[k] > 1u) return 0;
    return 1;
}

int rlao_observe(rlao_state *st, const rlao_config *cfg, uint32_t slot, double z, const uint8_t *reads,
                 uint32_t n_reads, uint32_t step, uint32_t binding) {
    if (!st || !cfg) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    if (slot >= cfg->slots || !isfinite(z) || !reads_ok(reads, n_reads)) return RLAO_E_INPUT;
    if (st->buffered >= RLAO_BUFFER) return RLAO_E_CAPACITY;   /* refused before anything changes */
    if (apply(st, cfg, slot, z, reads, n_reads)) {
        st->magic = STATE_POISONED;
        return RLAO_E_NUMERIC;
    }
    rlao_event *e = &st->buffer[st->buffered++];
    memset(e, 0, sizeof *e);
    e->step = step;
    e->slot = (uint8_t)slot;
    e->n_reads = (uint8_t)n_reads;
    for (uint32_t k = 0; k < n_reads; ++k) e->reads[k] = reads[k];
    e->binding = binding;
    e->z = z;
    return RLAO_OK;
}

static double log_weight(const rlao_state *st, const rlao_config *cfg, uint32_t m) {
    double s[RLAO_MAX_SLOTS] = {0.0};
    const uint32_t n = cfg->slots;
    for (uint32_t r = 0; r < n; ++r) {          /* insertion sort, ascending: the order of Python's sorted() */
        const double x = st->lam[m][r];
        uint32_t i = r;
        while (i > 0u && s[i - 1u] > x) {
            s[i] = s[i - 1u];
            --i;
        }
        s[i] = x;
    }
    double acc = s[0];
    for (uint32_t r = 1; r < n; ++r) acc += s[r];
    return cfg->log_prior[m] + acc;
}

static void weights(const rlao_state *st, const rlao_config *cfg, double *w) {
    double lw[RLAO_MAX_HYP];
    double top = 0.0;
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
        lw[m] = log_weight(st, cfg, m);
        if (m == 0u || lw[m] > top) top = lw[m];
    }
    double s = 0.0;
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
        w[m] = exp(lw[m] - top);
        s += w[m];
    }
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) w[m] = w[m] / s;
}

static double mean(const rlao_state *st, uint32_t m, uint32_t r, double z) {
    const double sd = sqrt(st->v[m][r]);
    double acc = 0.0;
    for (uint32_t g = 0; g < RLAO_GH; ++g) acc += GH_W[g] * sigmoid(z + (st->mu[m][r] + sd * GH_X[g]));
    return acc;
}

int rlao_weights(const rlao_state *st, const rlao_config *cfg, double *out) {
    if (!st || !cfg || !out) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    weights(st, cfg, out);
    return RLAO_OK;
}

static void marginal(const rlao_state *st, const rlao_config *cfg, const double *z_row, double *out) {
    double w[RLAO_MAX_HYP];
    weights(st, cfg, w);
    for (uint32_t r = 0; r < cfg->slots; ++r) {
        double acc = 0.0;
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) acc += w[m] * mean(st, m, r, z_row[r]);
        out[r] = acc;
    }
}

static void independent(const rlao_state *st, const rlao_config *cfg, const double *z_row, double *out) {
    for (uint32_t r = 0; r < cfg->slots; ++r) {
        double lw[RLAO_MAX_HYP], w[RLAO_MAX_HYP];
        double top = 0.0, s = 0.0, acc = 0.0;
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
            lw[m] = cfg->log_prior[m] + st->lam[m][r];
            if (m == 0u || lw[m] > top) top = lw[m];
        }
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
            w[m] = exp(lw[m] - top);
            s += w[m];
        }
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) acc += (w[m] / s) * mean(st, m, r, z_row[r]);
        out[r] = acc;
    }
}

static int z_ok(const rlao_config *cfg, const double *z_row) {
    for (uint32_t r = 0; r < cfg->slots; ++r)
        if (!isfinite(z_row[r])) return 0;
    return 1;
}

int rlao_predictive(const rlao_state *st, const rlao_config *cfg, const double *z_row, double *out) {
    if (!st || !cfg || !z_row || !out) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    if (!z_ok(cfg, z_row)) return RLAO_E_INPUT;
    marginal(st, cfg, z_row, out);
    return RLAO_OK;
}

static int alternative_ok(const rlao_config *cfg, const rlao_alternative *a) {
    uint32_t used = 0u;
    if (a->paths < 1u || a->paths > RLAO_MAX_PATHS) return 0;
    for (uint32_t p = 0; p < a->paths; ++p) {
        if (a->length[p] < 1u || a->length[p] > RLAO_MAX_PATH_LEN) return 0;
        for (uint32_t k = 0; k < a->length[p]; ++k) {
            const uint32_t r = a->slot[p][k];
            if (r >= cfg->slots || (used & (1u << r))) return 0;   /* distinct relations, edge-disjoint paths */
            used |= 1u << r;
        }
    }
    return 1;
}

static double compose(const rlao_alternative *a, const double *p) {
    double miss = 1.0;
    for (uint32_t i = 0; i < a->paths; ++i) {
        double prod = 1.0;
        for (uint32_t k = 0; k < a->length[i]; ++k) prod *= p[a->slot[i][k]];
        miss *= 1.0 - prod;
    }
    return 1.0 - miss;
}

int rlao_expected_utility(const rlao_state *st, const rlao_config *cfg, const rlao_alternative *alt,
                          const double *z_row, double *out) {
    if (!st || !cfg || !alt || !z_row || !out) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    if (!z_ok(cfg, z_row) || !alternative_ok(cfg, alt)) return RLAO_E_INPUT;
    if (cfg->readout == RLAO_READOUT_COUPLED) {   /* sum_h w_h U(E_h[p]) (scalar.ScalarRelational.expected_utility) */
        double w[RLAO_MAX_HYP];
        weights(st, cfg, w);
        double total = 0.0;
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
            double miss = 1.0;
            for (uint32_t i = 0; i < alt->paths; ++i) {
                double prod = 1.0;
                for (uint32_t k = 0; k < alt->length[i]; ++k) {
                    const uint32_t r = alt->slot[i][k];
                    prod *= mean(st, m, r, z_row[r]);
                }
                miss *= 1.0 - prod;
            }
            total += w[m] * (1.0 - miss);
        }
        *out = total;
    } else {
        double p[RLAO_MAX_SLOTS];
        if (cfg->readout == RLAO_READOUT_MARGINAL) marginal(st, cfg, z_row, p);
        else independent(st, cfg, z_row, p);
        *out = compose(alt, p);
    }
    return RLAO_OK;
}

int rlao_rebuild(rlao_state *st, const rlao_config *cfg, uint32_t keep) {
    if (!st || !cfg) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    reset_arrays(st, cfg);
    for (uint32_t i = 0; i < st->buffered; ++i) {
        if (!(keep & (1u << i))) continue;
        const rlao_event *e = &st->buffer[i];
        if (apply(st, cfg, e->slot, e->z, e->reads, e->n_reads)) {
            st->magic = STATE_POISONED;
            return RLAO_E_NUMERIC;
        }
    }
    return RLAO_OK;
}

int rlao_drop(rlao_state *st, const rlao_config *cfg, uint32_t drop) {
    if (!st || !cfg) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    uint32_t n = 0u;
    for (uint32_t i = 0; i < st->buffered; ++i)
        if (!(drop & (1u << i))) st->buffer[n++] = st->buffer[i];
    for (uint32_t i = n; i < st->buffered; ++i) memset(&st->buffer[i], 0, sizeof st->buffer[i]);
    st->buffered = n;
    return RLAO_OK;
}

int rlao_reset_slot(rlao_state *st, const rlao_config *cfg, uint32_t slot) {
    if (!st || !cfg) return RLAO_E_NULL;
    if (!config_ok(cfg)) return RLAO_E_CONFIG;
    if (!state_ok(st, cfg)) return RLAO_E_STATE;
    if (slot >= cfg->slots) return RLAO_E_INPUT;
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
        st->mu[m][slot] = 0.0;
        st->v[m][slot] = cfg->bias_var;
        st->lam[m][slot] = 0.0;
    }
    st->counts[slot] = 0u;
    uint32_t drop = 0u;
    for (uint32_t i = 0; i < st->buffered; ++i)
        if (st->buffer[i].slot == slot) drop |= 1u << i;
    return rlao_drop(st, cfg, drop);
}

void rlao_zeroize(rlao_state *st) {
    if (!st) return;
    volatile unsigned char *p = (volatile unsigned char *)st;
    for (size_t i = 0; i < sizeof *st; ++i) p[i] = 0u;
}

uint32_t rlao_logical_scalars(const rlao_config *cfg) {
    if (!config_ok(cfg)) return 0u;
    return 3u * cfg->hypotheses * cfg->slots + cfg->slots;
}
