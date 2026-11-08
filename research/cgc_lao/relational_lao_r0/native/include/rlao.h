#ifndef ELPIS_RLAO_H
#define ELPIS_RLAO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Relational LAO R0: the native C11 relational adapter (RESEARCH ONLY, NO RUNTIME AUTHORITY;
 * research/cgc_lao/relational_lao_r0/SPEC.md). It is a bounded, volatile relational belief state over at most
 * 8 relation slots and 4 reliability hypotheses. It ports the explicit-loop scalar reference
 * (src/relational_lao/scalar.py) statement for statement and is bitwise equal to it on one machine
 * (build with -ffp-contract=off).
 *
 * State, fixed capacity, one value and no heap:
 *   mu, v, lam [H][S]  Gaussian bias belief and summed log evidence of each slot under each hypothesis
 *   counts [S]         applied events per slot
 *   buffer [8]         the provenance-indexed events of the episode (horizon bound: 8 events, 12 readings)
 * The shared reliability posterior is log w_h = log prior_h + sum_r lam[h][r]; the sum runs over sorted terms,
 * so slot order does not change it.
 *
 * Invalidation, both exact:
 *   rebuild    re-applies exactly the buffered events the host keeps (strategy B, bounded replay)
 *   reset_slot returns a slot to its prior and removes its evidence term, without replay (strategy A)
 * Capacity is refused, never grown. A numerical failure poisons the state until it is reset.
 * The host owns admission, bindings and authority; the adapter never reads or writes ECS_C.
 */

#define RLAO_MAX_HYP 4u
#define RLAO_MAX_SLOTS 8u
#define RLAO_GH 16u
#define RLAO_BUFFER 8u
#define RLAO_MAX_READS 2u
#define RLAO_MAX_PATHS 3u
#define RLAO_MAX_PATH_LEN 3u

#define RLAO_OK 0
#define RLAO_E_NULL 1
#define RLAO_E_CONFIG 2
#define RLAO_E_STATE 3
#define RLAO_E_INPUT 4
#define RLAO_E_CAPACITY 5
#define RLAO_E_NUMERIC 6

#define RLAO_READOUT_COUPLED 0u
#define RLAO_READOUT_MARGINAL 1u
#define RLAO_READOUT_INDEPENDENT 2u

/* Immutable coefficients (built only by rlao_config_init). */
typedef struct {
    uint32_t magic;
    uint32_t hypotheses;
    uint32_t slots;
    uint32_t readout;
    double q[RLAO_MAX_HYP];
    double rho[RLAO_MAX_HYP];
    double log_prior[RLAO_MAX_HYP];
    double bias_var;
    uint64_t digest;
} rlao_config;

/* One buffered event: the step, the slot, the readings and the host's opaque binding index. */
typedef struct {
    uint32_t step;
    uint8_t slot;
    uint8_t n_reads;
    uint8_t reads[RLAO_MAX_READS];
    uint32_t binding;
    uint32_t reserved;
    double z;
} rlao_event;

typedef struct {
    uint32_t magic;
    uint32_t slots;
    uint32_t hypotheses;
    uint32_t buffered;
    uint64_t config_digest;
    uint32_t counts[RLAO_MAX_SLOTS];
    double mu[RLAO_MAX_HYP][RLAO_MAX_SLOTS];
    double v[RLAO_MAX_HYP][RLAO_MAX_SLOTS];
    double lam[RLAO_MAX_HYP][RLAO_MAX_SLOTS];
    rlao_event buffer[RLAO_BUFFER];
} rlao_state;

/* An alternative of a query: an OR of edge-disjoint paths, each an AND of distinct slots. */
typedef struct {
    uint32_t paths;
    uint32_t length[RLAO_MAX_PATHS];
    uint32_t slot[RLAO_MAX_PATHS][RLAO_MAX_PATH_LEN];
} rlao_alternative;

int rlao_config_init(rlao_config *cfg, uint32_t hypotheses, const double *q, const double *rho, const double *prior,
                     uint32_t slots, double bias_var, uint32_t readout);
int rlao_reset(rlao_state *st, const rlao_config *cfg);
/* Apply one event of `slot` (readings at logit z) and buffer it; CAPACITY (nothing changed) when the buffer is full. */
int rlao_observe(rlao_state *st, const rlao_config *cfg, uint32_t slot, double z, const uint8_t *reads,
                 uint32_t n_reads, uint32_t step, uint32_t binding);
int rlao_weights(const rlao_state *st, const rlao_config *cfg, double *out);
int rlao_predictive(const rlao_state *st, const rlao_config *cfg, const double *z_row, double *out);
int rlao_expected_utility(const rlao_state *st, const rlao_config *cfg, const rlao_alternative *alt,
                          const double *z_row, double *out);
/* Strategy B: reset the state and re-apply, in order, the buffered events whose bit is set in `keep`. */
int rlao_rebuild(rlao_state *st, const rlao_config *cfg, uint32_t keep);
/* Remove the buffered events whose bit is set in `drop` (retired); the applied state is unchanged. */
int rlao_drop(rlao_state *st, const rlao_config *cfg, uint32_t drop);
/* Strategy A: slot back to its prior, its evidence term removed, its buffered events discarded. */
int rlao_reset_slot(rlao_state *st, const rlao_config *cfg, uint32_t slot);
void rlao_zeroize(rlao_state *st);
size_t rlao_state_size(void);
size_t rlao_config_size(void);
uint32_t rlao_logical_scalars(const rlao_config *cfg);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rlao_event) == 24, "rlao_event layout");
_Static_assert(sizeof(rlao_state) == 1016, "rlao_state layout");
_Static_assert(sizeof(rlao_config) == 128, "rlao_config layout");
#endif

#ifdef __cplusplus
}
#endif

#endif
