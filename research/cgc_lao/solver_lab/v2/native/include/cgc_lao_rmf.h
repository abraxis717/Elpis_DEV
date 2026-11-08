/*
 * CGC-LAO bounded reliability-mixture learner: native C11 reference (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).
 *
 * A host-admitted, episode-local learner for three experts with selected binary feedback. The host owns an
 * immutable configuration (hypotheses, prior, bias variance, STOP payoff, horizon, acquisition rule) and one
 * volatile state per episode. Every function validates its inputs before writing, writes results only when
 * every intermediate is finite, allocates nothing, performs no I/O and runs a fixed number of operations
 * bounded by the compile-time capacities below. A non-finite intermediate poisons the state, after which
 * every call refuses it until the host resets it. The algorithm is the explicit-loop scalar reference
 * research/cgc_lao/solver_lab/v2/src/cgc_lao_v2/scalar2.py, statement for statement.
 */
#ifndef CGC_LAO_RMF_H
#define CGC_LAO_RMF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CGC_LAO_RMF_EXPERTS 3u
#define CGC_LAO_RMF_MAX_HYP 8u       /* hypothesis capacity of a configuration and a state */
#define CGC_LAO_RMF_MAX_NODES 4u     /* corruption-rate nodes per hypothesis (1 = point, >1 = band) */
#define CGC_LAO_RMF_MAX_READS 2u     /* readings of one event per observation */
#define CGC_LAO_RMF_GH 16u           /* Gauss-Hermite nodes */
#define CGC_LAO_RMF_MAX_HORIZON 64u
#define CGC_LAO_RMF_STOP 3u          /* terminal decision code for STOP */

typedef enum {
    CGC_LAO_RMF_OK = 0,
    CGC_LAO_RMF_E_NULL = 1,      /* a required pointer is NULL */
    CGC_LAO_RMF_E_CONFIG = 2,    /* invalid or uninitialised configuration */
    CGC_LAO_RMF_E_STATE = 3,     /* state not reset for this configuration, zeroised, or poisoned */
    CGC_LAO_RMF_E_INPUT = 4,     /* invalid expert, reading, reading count or non-finite logit */
    CGC_LAO_RMF_E_HORIZON = 5,   /* an advancing observation beyond the horizon */
    CGC_LAO_RMF_E_NUMERIC = 6    /* non-finite intermediate: the state is now poisoned */
} cgc_lao_rmf_status;

enum { CGC_LAO_RMF_PROBE_EARLY = 0, CGC_LAO_RMF_GREEDY = 1 };

/* Immutable coefficients. Built only by cgc_lao_rmf_config_init; never written afterwards. */
typedef struct {
    uint32_t magic;
    uint32_t hypotheses;
    uint32_t horizon;
    uint32_t acquisition;
    uint32_t nodes[CGC_LAO_RMF_MAX_HYP];
    double q[CGC_LAO_RMF_MAX_HYP][CGC_LAO_RMF_MAX_NODES];
    double node_weight[CGC_LAO_RMF_MAX_HYP];
    double rho[CGC_LAO_RMF_MAX_HYP];
    double log_prior[CGC_LAO_RMF_MAX_HYP];
    double bias_var;
    double stop_payoff;
    uint64_t digest;             /* FNV-1a 64 of the coefficients; binds states to this configuration */
} cgc_lao_rmf_config;

/* Volatile episode state (float64). Logical scalars: hypotheses * 7 + 3 counts. */
typedef struct {
    uint32_t magic;
    uint32_t t;                  /* advancing observations so far */
    uint32_t counts[CGC_LAO_RMF_EXPERTS];
    uint32_t hypotheses;
    uint64_t config_digest;
    double log_weight[CGC_LAO_RMF_MAX_HYP];
    double mean[CGC_LAO_RMF_MAX_HYP][CGC_LAO_RMF_EXPERTS];
    double var[CGC_LAO_RMF_MAX_HYP][CGC_LAO_RMF_EXPERTS];
} cgc_lao_rmf_state;

size_t cgc_lao_rmf_state_size(void);
size_t cgc_lao_rmf_config_size(void);
uint32_t cgc_lao_rmf_logical_scalars(const cgc_lao_rmf_config *cfg);

/* q: hypotheses x CGC_LAO_RMF_MAX_NODES row-major (unused node slots ignored); rho, prior: per hypothesis. */
cgc_lao_rmf_status cgc_lao_rmf_config_init(cgc_lao_rmf_config *cfg, uint32_t hypotheses, const uint32_t *nodes,
                                           const double *q, const double *rho, const double *prior, double bias_var,
                                           double stop_payoff, uint32_t horizon, uint32_t acquisition);
cgc_lao_rmf_status cgc_lao_rmf_reset(cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg);
cgc_lao_rmf_status cgc_lao_rmf_predictive(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                          const double z[CGC_LAO_RMF_EXPERTS], double out[CGC_LAO_RMF_EXPERTS]);
cgc_lao_rmf_status cgc_lao_rmf_propose(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                       const double z[CGC_LAO_RMF_EXPERTS], uint32_t *expert);
/* advance = 1: the step's selected event; advance = 0: an extra independent event at the same step. */
cgc_lao_rmf_status cgc_lao_rmf_observe(cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg, uint32_t expert,
                                       double z, const uint8_t *reads, uint32_t n_reads, uint32_t advance);
cgc_lao_rmf_status cgc_lao_rmf_weights(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg, double *out);
cgc_lao_rmf_status cgc_lao_rmf_state_correction(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                                double out[CGC_LAO_RMF_EXPERTS]);
cgc_lao_rmf_status cgc_lao_rmf_decide(const cgc_lao_rmf_state *st, const cgc_lao_rmf_config *cfg,
                                      const double zq[CGC_LAO_RMF_EXPERTS], uint32_t *decision);
void cgc_lao_rmf_zeroize(cgc_lao_rmf_state *st);

#ifdef __cplusplus
}
#endif
#endif
