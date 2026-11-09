# Relational LAO R0: development protocol

Written before any TRAIN or DEV record of this study, and committed before both. Exploratory smoke runs are
disclosed in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Arms

Built by `study.arms_for`. Every arm sees the same admitted readings of one shared log.

**Candidates** (Contract I, DUPLICATE, PROBE_R acquisition):

| Arm | Hypotheses | Prior | Readout | Invalidation |
|---|---|---|---|---|
| RRMF3 | {0, 0.4, 0.48} | uniform | coupled | replay (B) |
| RRMF3_M | {0, 0.4, 0.48} | uniform | marginal | replay |
| RRMF3_A | {0, 0.4, 0.48} | uniform | coupled | slot reset (A) |
| RRMF4 | {0, 0.2, 0.4, 0.48} | uniform | coupled | replay |
| RRMF4_M | {0, 0.2, 0.4, 0.48} | uniform | marginal | replay |
| RRMF4_W | {0, 0.2, 0.4, 0.48} | (1, 0.5, 1, 2) | coupled | replay |
| RRMF4_W_M | {0, 0.2, 0.4, 0.48} | (1, 0.5, 1, 2) | marginal | replay |

**Per-candidate counterparts:**
- `:IND`: the same filter with a separate reliability posterior per slot (INDEPENDENT_RMF).
- `:HCA`: LAO_PLUS_HCA.
- `:PERM`: identity-permuted.
- `:STALE`: no invalidation; authority regimes only.

**Controls:** NOOP, EXACT_RELATIONAL, STATIC_COUNTS, HCA_ONLY, and KNOWN_RELIABILITY (privileged, a reference only).

**Contract II fallbacks** (first reading only):
- SAFE4_P8: {0, 0.2, 0.4, 0.48}, prior (1, 1, 8, 8).
- SAFE3_P4: {0, 0.4, 0.48}, prior (1, 4, 4).

## Shared log

The DRIVER is NOOP with PROBE_R. Its acquisition depends only on the public priors and on coverage, so the log is
independent of every learner, and every arm is a shadow with matched information.

## Rounds

| Round | Partition, size | Regimes | Purpose |
|---|---|---|---|
| TRAIN | TRAIN 0–31 × 2048 | DESIGN | Fixes the scales (below). |
| DEV (single round) | DEV 0–31 × 2048 | DESIGN, ROBUSTNESS, AUTHORITY (16 regimes) | Selection. |
| DEV on-policy | DEV 32–63 × 2048 | DESIGN | Each candidate drives itself with PROBE_R, giving RG9. RRMF3 and the selected candidate also drive with VOI_R, on DESIGN and FLIP45 (acquisition characterisation). |

The gate scales, fixed by TRAIN:
- `δ_clean` = 0.25 × the mean over TRAIN seeds of CLEAN compositional `[NOOP − KNOWN_RELIABILITY]`. A candidate keeps at least 75% of the oracle's clean gain.
- `online_budget` = 1.10 × the TRAIN mean online regret of the NOOP driver on ALL.
- `τ_safe` = 0.0005, as in LAO v2–v4.

## Gates

The metric is compositional terminal regret per episode, and the per-seed means are paired over world seeds. ALL is the
mean of CLEAN, FLIP20 and FLIP40. REVISION is the mean of CLEAN and FLIP20 under REV_DRIFT and RECLAIM_DRIFT.

| Gate | Requirement (upper bound of the paired interval) |
|---|---|
| RG1 | ALL: C − NOOP < 0 |
| RG2 | CLEAN: C − NOOP < 0, and C − KNOWN_RELIABILITY ≤ δ_clean |
| RG3 | ALL: C − X < 0 for every matched-information control X: EXACT_RELATIONAL, STATIC_COUNTS, HCA_ONLY, C`:IND`, C`:HCA` |
| RG4 | FLIP40 and FLIP45: C − NOOP ≤ τ_safe |
| RG5 | QUNIF or BURST: C − NOOP < 0, at 97.5% or at the round's level if that is higher |
| RG6 | ALL: C − C`:PERM` < 0 (the relational binding matters) |
| RG7 | REVISION: C − C`:STALE` < 0, and C − NOOP < 0 |
| RG8 | On VALIDATION and HOLDOUT only, the held-out-topology seeds: ALL C − NOOP < 0 and C − C`:IND` < 0 |
| RG9 | ALL online regret, on-policy: the mean's upper bound ≤ `online_budget` |
| Mechanical | The R1 and R2 suites pass in CI: exact invalidation, revoked sources unreachable, the host differential over the real view, and the R2 parity |

**Selection.** A candidate is eligible when every DEV gate holds at **99.9%** (RG8 does not apply on DEV). Among
eligible candidates, the smallest mechanism is chosen, by this key:
1. logical state scalars;
2. likelihood evaluations per update;
3. readout products per alternative (marginal 1, coupled M);
4. DEV ALL compositional regret;
5. DEV REVISION regret;
6. name.

If none is eligible, the primary is NOT_QUALIFIED on this round. A diagnosed and improved candidate set may be tried in a
new DEV round on fresh DEV seeds (64 and up), under a protocol written before that round.

**Acquisition adoption.** VOI_R replaces PROBE_R for the selected candidate only when both hold: on-policy ALL
compositional regret with VOI_R is lower than with PROBE_R at 99.9% (paired over seeds), and VOI_R satisfies RG9.

**Fallback.** F1–F4 at 99.9% (`study.fallback_gates`): FLIP40, FLIP45 and FLIP48 non-harm, a CLEAN gain, and
EVENT40 non-harm. Among eligible fallbacks, the lowest DEV CLEAN regret is chosen.

## After selection

1. `PREREGISTRATION_R6.md`: these gates at 95% (RG5 at 97.5%), RG8, the scales, and the plans.
2. `FROZEN_R6.json`: sources, specifications and plans.
3. VALIDATION, once: 32 × 4096 with every arm, and held-out topology V_LADDER on its even seeds.
4. HOLDOUT, once, and only if every VALIDATION gate passes: 64 × 4096, with H_RANDOM topologies on even seeds.
5. OOD, once, after the HOLDOUT. It contains:
   - the ρ sweep that fixes this workload's own admissible correlation, by the rule: the largest grid ρ whose every
     ρ′ ≤ ρ has `C − NOOP ≤ τ_safe` on compositional regret;
   - OOD corruption;
   - every topology.
