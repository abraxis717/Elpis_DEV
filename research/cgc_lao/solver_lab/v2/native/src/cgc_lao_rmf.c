/*
 * CGC-LAO bounded reliability-mixture learner: native C11 reference. See cgc_lao_rmf.h.
 *
 * Statement-for-statement port of v2/src/cgc_lao_v2/scalar2.py: every sum is a left-to-right loop from 0.0,
 * transcendentals are the C library's exp/log/sqrt/pow, branches are identical. Build with
 * -ffp-contract=off so that no multiply-add is fused (the Python reference cannot fuse).
 * No heap, no recursion, no I/O, no global mutable state; all loops are bounded by compile-time capacities.
 */
#include "cgc_lao_rmf.h"

#include <math.h>
#include <string.h>

#define CONFIG_MAGIC 0x434c524du   /* "CLRM" */
#define STATE_MAGIC 0x534c524du    /* "SLRM" */
#define STATE_POISONED 0x504f4953u /* "POIS" */

static const double LW_FLOOR = -1.0e4;
static const double V_FLOOR = 1.0e-6;
static const double STATE_CAP = 1.5;
static const double MASKED = -1.0e9;
static const double L_FLOOR = 1e-300;

/* 16-node probabilists' Gauss-Hermite rule normalised to N(0, 1): the literals of cgc_lao_lab.filters. */
static const double GH_X[CGC_LAO_RMF_GH] = {
    -6.630878198393129, -5.472225705949343, -4.492955302520011, -3.6008736241715487,
    -2.7602450476307014, -1.9519803457163334, -1.1638291005549648, -0.3867606045005574,
    0.3867606045005574, 1.1638291005549648, 1.9519803457163334, 2.7602450476307014,
    3.6008736241715487, 4.492955302520011, 5.472225705949343, 6.630878198393129};
static const double GH_W[CGC_LAO_RMF_GH] = {
    1.4978147231618412e-10, 1.309473216286817e-07, 1.530003216248732e-05, 0.0005259849265739087,
    0.007266937601184749, 0.04728475235401406, 0.1583383727509497, 0.286568521238012,
    0.286568521238012, 0.1583383727509497, 0.04728475235401406, 0.007266937601184749,
    0.0005259849265739087, 1.530003216248732e-05, 1.309473216286817e-07, 1.4978147231618412e-10};

size_t cgc_lao_rmf_state_size(void) { return sizeof(cgc_lao_rmf_state); }
size_t cgc_lao_rmf_config_size(void) { return sizeof(cgc_lao_rmf_config); }

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t config_digest(const cgc_lao_rmf_config *c) {
    uint64_t h = 14695981039346656037ull;
    h = fnv(h, &c->hypotheses, sizeof c->hypotheses);
    h = fnv(h, &c->horizon, sizeof c->horizon);
    h = fnv(h, &c->acquisition, sizeof c->acquisition);
    h = fnv(h, c->nodes, sizeof c->nodes);
    h = fnv(h, c->q, sizeof c->q);
    h = fnv(h, c->node_weight, sizeof c->node_weight);
    h = fnv(h, c->rho, sizeof c->rho);
    h = fnv(h, c->log_prior, sizeof c->log_prior);
    h = fnv(h, &c->bias_var, sizeof c->bias_var);
    h = fnv(h, &c->stop_payoff, sizeof c->stop_payoff);
    return h;
}

static int config_ok(const cgc_lao_rmf_config *c) {
    return c != NULL && c->magic == CONFIG_MAGIC && c->hypotheses >= 1u && c->hypotheses <= CGC_LAO_RMF_MAX_HYP &&
           c->digest == config_digest(c);
}

static int state_ok(const cgc_lao_rmf_state *s, const cgc_lao_rmf_config *c) {
    return s != NULL && s->magic == STATE_MAGIC && s->config_digest == c->digest && s->hypotheses == c->hypotheses &&
           s->t <= c->horizon;
}

uint32_t cgc_lao_rmf_logical_scalars(const cgc_lao_rmf_config *cfg) {
    if (!config_ok(cfg)) {
        return 0u;
    }
    return cfg->hypotheses * (1u + 2u * CGC_LAO_RMF_EXPERTS) + CGC_LAO_RMF_EXPERTS;
}

cgc_lao_rmf_status cgc_lao_rmf_config_init(cgc_lao_rmf_config *cfg, uint32_t hypotheses, const uint32_t *nodes,
                                           const double *q, const double *rho, const double *prior, double bias_var,
                                           double stop_payoff, uint32_t horizon, uint32_t acquisition) {
    if (cfg == NULL || nodes == NULL || q == NULL || rho == NULL || prior == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (hypotheses < 1u || hypotheses > CGC_LAO_RMF_MAX_HYP || !(isfinite(bias_var) && bias_var > 0.0) ||
        !(isfinite(stop_payoff) && stop_payoff >= 0.0 && stop_payoff <= 1.0) || horizon < 1u ||
        horizon > CGC_LAO_RMF_MAX_HORIZON || acquisition > CGC_LAO_RMF_GREEDY) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    double total = 0.0;
    for (uint32_t m = 0; m < hypotheses; ++m) {
        if (nodes[m] < 1u || nodes[m] > CGC_LAO_RMF_MAX_NODES || !(isfinite(rho[m]) && rho[m] >= 0.0 && rho[m] <= 1.0) ||
            !(isfinite(prior[m]) && prior[m] > 0.0)) {
            return CGC_LAO_RMF_E_CONFIG;
        }
        for (uint32_t j = 0; j < nodes[m]; ++j) {
            const double v = q[m * CGC_LAO_RMF_MAX_NODES + j];
            if (!(isfinite(v) && v >= 0.0 && v < 0.5)) {
                return CGC_LAO_RMF_E_CONFIG;
            }
        }
        total += prior[m];
    }
    cgc_lao_rmf_config c;
    memset(&c, 0, sizeof c);
    c.hypotheses = hypotheses;
    c.horizon = horizon;
    c.acquisition = acquisition;
    for (uint32_t m = 0; m < hypotheses; ++m) {
        c.nodes[m] = nodes[m];
        for (uint32_t j = 0; j < nodes[m]; ++j) {
            c.q[m][j] = q[m * CGC_LAO_RMF_MAX_NODES + j];
        }
        c.node_weight[m] = 1.0 / (double)nodes[m];
        c.rho[m] = rho[m];
        c.log_prior[m] = log(prior[m] / total);
    }
    c.bias_var = bias_var;
    c.stop_payoff = stop_payoff;
    c.digest = config_digest(&c);
    c.magic = CONFIG_MAGIC;
    *cfg = c;
    return CGC_LAO_RMF_OK;
}

cgc_lao_rmf_status cgc_lao_rmf_reset(cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg) {
    if (st == NULL || cfg == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (!config_ok(cfg)) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    cgc_lao_rmf_state s;
    memset(&s, 0, sizeof s);
    s.hypotheses = cfg->hypotheses;
    s.config_digest = cfg->digest;
    for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
        s.log_weight[m] = cfg->log_prior[m];
        for (uint32_t a = 0; a < CGC_LAO_RMF_EXPERTS; ++a) {
            s.mean[m][a] = 0.0;
            s.var[m][a] = cfg->bias_var;
        }
    }
    s.magic = STATE_MAGIC;
    *st = s;
    return CGC_LAO_RMF_OK;
}

static double sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + exp(-z));
    }
    const double e = exp(z);
    return e / (1.0 + e);
}

static void weights_of(const cgc_lao_rmf_state *s, uint32_t hyp, double *w) {
    double top = s->log_weight[0];
    for (uint32_t m = 1; m < hyp; ++m) {
        if (s->log_weight[m] > top) {
            top = s->log_weight[m];
        }
    }
    double sum = 0.0;
    for (uint32_t m = 0; m < hyp; ++m) {
        w[m] = exp(s->log_weight[m] - top);
    }
    for (uint32_t m = 0; m < hyp; ++m) {
        sum += w[m];
    }
    for (uint32_t m = 0; m < hyp; ++m) {
        w[m] = w[m] / sum;
    }
}

static double success(const cgc_lao_rmf_state *s, uint32_t m, uint32_t a, double z) {
    const double sd = sqrt(s->var[m][a]);
    double acc = 0.0;
    for (uint32_t g = 0; g < CGC_LAO_RMF_GH; ++g) {
        acc += GH_W[g] * sigmoid(z + (s->mean[m][a] + sd * GH_X[g]));
    }
    return acc;
}

static void predictive_of(const cgc_lao_rmf_state *s, uint32_t hyp, const double z[CGC_LAO_RMF_EXPERTS],
                          double out[CGC_LAO_RMF_EXPERTS]) {
    double w[CGC_LAO_RMF_MAX_HYP];
    weights_of(s, hyp, w);
    for (uint32_t a = 0; a < CGC_LAO_RMF_EXPERTS; ++a) {
        double acc = 0.0;
        for (uint32_t m = 0; m < hyp; ++m) {
            acc += success(s, m, a, z[a]) * w[m];
        }
        out[a] = acc;
    }
}

static int finite3(const double z[CGC_LAO_RMF_EXPERTS]) {
    return isfinite(z[0]) && isfinite(z[1]) && isfinite(z[2]);
}

static uint32_t first_argmax(const double v[CGC_LAO_RMF_EXPERTS], const int *mask) {
    uint32_t best = 0u;
    double best_v = (mask == NULL || mask[0]) ? v[0] : MASKED;
    for (uint32_t a = 1; a < CGC_LAO_RMF_EXPERTS; ++a) {
        const double x = (mask == NULL || mask[a]) ? v[a] : MASKED;
        if (x > best_v) {
            best = a;
            best_v = x;
        }
    }
    return best;
}

cgc_lao_rmf_status cgc_lao_rmf_predictive(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                          const double z[CGC_LAO_RMF_EXPERTS], double out[CGC_LAO_RMF_EXPERTS]) {
    if (st == NULL || cfg == NULL || z == NULL || out == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (!config_ok(cfg)) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    if (!state_ok(st, cfg)) {
        return CGC_LAO_RMF_E_STATE;
    }
    if (!finite3(z)) {
        return CGC_LAO_RMF_E_INPUT;
    }
    double p[CGC_LAO_RMF_EXPERTS];
    predictive_of(st, cfg->hypotheses, z, p);
    if (!finite3(p)) {
        return CGC_LAO_RMF_E_NUMERIC;
    }
    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    return CGC_LAO_RMF_OK;
}

cgc_lao_rmf_status cgc_lao_rmf_propose(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                       const double z[CGC_LAO_RMF_EXPERTS], uint32_t *expert) {
    if (expert == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    double pred[CGC_LAO_RMF_EXPERTS];
    const cgc_lao_rmf_status rc = cgc_lao_rmf_predictive(st, cfg, z, pred);
    if (rc != CGC_LAO_RMF_OK) {
        return rc;
    }
    if (st->t >= cfg->horizon) {
        return CGC_LAO_RMF_E_HORIZON;
    }
    int unseen[CGC_LAO_RMF_EXPERTS];
    uint32_t need = 0u;
    for (uint32_t a = 0; a < CGC_LAO_RMF_EXPERTS; ++a) {
        unseen[a] = st->counts[a] == 0u;
        need += unseen[a] ? 1u : 0u;
    }
    const uint32_t best = first_argmax(pred, NULL);
    const uint32_t unseen_best = first_argmax(pred, unseen);
    const int force = need >= cfg->horizon - st->t;
    const int probe = cfg->acquisition == CGC_LAO_RMF_PROBE_EARLY && (st->t == 1u || st->t == 2u) && need > 0u;
    *expert = (force || probe) ? unseen_best : best;
    return CGC_LAO_RMF_OK;
}

static void evidence(const cgc_lao_rmf_config *c, uint32_t m, const uint8_t *reads, uint32_t n, double *a1,
                     double *a0) {
    for (int cl = 1; cl >= 0; --cl) {
        double acc = 0.0;
        for (uint32_t j = 0; j < c->nodes[m]; ++j) {
            const double q = c->q[m][j];
            double p;
            if (n == 1u) {
                const double f = (double)(reads[0] ^ cl);
                p = pow(q, f) * pow(1.0 - q, 1.0 - f);
            } else {
                const double f0 = (double)(reads[0] ^ cl);
                const double f1 = (double)(reads[1] ^ cl);
                const double s = f0 + f1;
                const double indep = pow(q, s) * pow(1.0 - q, 2.0 - s);
                const double shared = (s == 1.0) ? 0.0 : pow(q, f0) * pow(1.0 - q, 1.0 - f0);
                p = (1.0 - c->rho[m]) * indep + c->rho[m] * shared;
            }
            acc += p * c->node_weight[m];
        }
        if (cl == 1) {
            *a1 = acc;
        } else {
            *a0 = acc;
        }
    }
}

cgc_lao_rmf_status cgc_lao_rmf_observe(cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg, uint32_t expert,
                                       double z, const uint8_t *reads, uint32_t n_reads, uint32_t advance) {
    if (st == NULL || cfg == NULL || reads == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (!config_ok(cfg)) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    if (!state_ok(st, cfg)) {
        return CGC_LAO_RMF_E_STATE;
    }
    if (expert >= CGC_LAO_RMF_EXPERTS || !isfinite(z) || n_reads < 1u || n_reads > CGC_LAO_RMF_MAX_READS ||
        advance > 1u) {
        return CGC_LAO_RMF_E_INPUT;
    }
    for (uint32_t r = 0; r < n_reads; ++r) {
        if (reads[r] > 1u) {
            return CGC_LAO_RMF_E_INPUT;
        }
    }
    if (advance && st->t >= cfg->horizon) {
        return CGC_LAO_RMF_E_HORIZON;
    }
    const uint32_t hyp = cfg->hypotheses;
    double new_mean[CGC_LAO_RMF_MAX_HYP], new_var[CGC_LAO_RMF_MAX_HYP], new_lw[CGC_LAO_RMF_MAX_HYP];
    for (uint32_t m = 0; m < hyp; ++m) {
        double a1 = 0.0, a0 = 0.0;
        evidence(cfg, m, reads, n_reads, &a1, &a0);
        const double mu = st->mean[m][expert];
        const double sd = sqrt(st->var[m][expert]);
        double L = 0.0, m1 = 0.0, m2 = 0.0;
        for (uint32_t g = 0; g < CGC_LAO_RMF_GH; ++g) {
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
            const double v = m2 - m1 * m1;
            new_mean[m] = m1;
            new_var[m] = v > V_FLOOR ? v : V_FLOOR;
            new_lw[m] = st->log_weight[m] + log(L);
        } else {
            new_mean[m] = mu;
            new_var[m] = st->var[m][expert];
            new_lw[m] = st->log_weight[m] + LW_FLOOR;
        }
    }
    double top = new_lw[0];
    for (uint32_t m = 1; m < hyp; ++m) {
        if (new_lw[m] > top) {
            top = new_lw[m];
        }
    }
    int finite = isfinite(top);
    for (uint32_t m = 0; m < hyp; ++m) {
        const double x = new_lw[m] - top;
        new_lw[m] = x > LW_FLOOR ? x : LW_FLOOR;
        finite = finite && isfinite(new_lw[m]) && isfinite(new_mean[m]) && isfinite(new_var[m]) && new_var[m] > 0.0;
    }
    if (!finite) {
        st->magic = STATE_POISONED;
        return CGC_LAO_RMF_E_NUMERIC;
    }
    for (uint32_t m = 0; m < hyp; ++m) {
        st->mean[m][expert] = new_mean[m];
        st->var[m][expert] = new_var[m];
        st->log_weight[m] = new_lw[m];
    }
    if (advance) {
        st->counts[expert] += 1u;
        st->t += 1u;
    }
    return CGC_LAO_RMF_OK;
}

cgc_lao_rmf_status cgc_lao_rmf_weights(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg, double *out) {
    if (st == NULL || cfg == NULL || out == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (!config_ok(cfg)) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    if (!state_ok(st, cfg)) {
        return CGC_LAO_RMF_E_STATE;
    }
    weights_of(st, cfg->hypotheses, out);
    return CGC_LAO_RMF_OK;
}

cgc_lao_rmf_status cgc_lao_rmf_state_correction(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                                double out[CGC_LAO_RMF_EXPERTS]) {
    if (st == NULL || cfg == NULL || out == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    if (!config_ok(cfg)) {
        return CGC_LAO_RMF_E_CONFIG;
    }
    if (!state_ok(st, cfg)) {
        return CGC_LAO_RMF_E_STATE;
    }
    double w[CGC_LAO_RMF_MAX_HYP];
    weights_of(st, cfg->hypotheses, w);
    for (uint32_t a = 0; a < CGC_LAO_RMF_EXPERTS; ++a) {
        double acc = 0.0;
        for (uint32_t m = 0; m < cfg->hypotheses; ++m) {
            acc += w[m] * st->mean[m][a];
        }
        out[a] = acc > STATE_CAP ? STATE_CAP : (acc < -STATE_CAP ? -STATE_CAP : acc);
    }
    return CGC_LAO_RMF_OK;
}

cgc_lao_rmf_status cgc_lao_rmf_decide(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                      const double zq[CGC_LAO_RMF_EXPERTS], uint32_t *decision) {
    if (decision == NULL) {
        return CGC_LAO_RMF_E_NULL;
    }
    double pred[CGC_LAO_RMF_EXPERTS];
    const cgc_lao_rmf_status rc = cgc_lao_rmf_predictive(st, cfg, zq, pred);
    if (rc != CGC_LAO_RMF_OK) {
        return rc;
    }
    const uint32_t best = first_argmax(pred, NULL);
    *decision = pred[best] < cfg->stop_payoff ? CGC_LAO_RMF_STOP : best;
    return CGC_LAO_RMF_OK;
}

void cgc_lao_rmf_zeroize(cgc_lao_rmf_state *st) {
    if (st == NULL) {
        return;
    }
    volatile unsigned char *p = (volatile unsigned char *)st;
    for (size_t i = 0; i < sizeof *st; ++i) {
        p[i] = 0u;
    }
}
