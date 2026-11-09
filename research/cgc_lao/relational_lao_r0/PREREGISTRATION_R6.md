# Relational LAO R0: preregistration of the R6 qualification

Written after [SELECTION_R6.json](SELECTION_R6.json) and before any VALIDATION, HOLDOUT or OOD episode.
[FROZEN_R6.json](FROZEN_R6.json) binds this file, the selection, the gate scales, the sources and the plans by digest.
The analysis code is frozen with the sources (`study.gate_parts`, `families`, `claims`, `primary_status`,
`fallback_status`, `rho_admission`, `ood_characterisation`), so no analysis choice is made after the data.

## What was seen before this file

- The TRAIN round, the single DEV round, the DEV on-policy runs (every record under `evidence/train` and
  `evidence/dev`), and the exploratory smoke runs on DEV seeds 200–203 disclosed in [RESEARCH_LOG.md](RESEARCH_LOG.md).
- Dry runs of the frozen code paths on TRAIN worlds, with the V_LADDER template and generated H_RANDOM topologies
  substituted in-process (disclosed in the research log). No VALIDATION, HOLDOUT or OOD world was generated, and no
  HOLDOUT or OOD topology was evaluated with a learner.

## Frozen mechanism

**Primary: RRMF3_M** (Contract I, DUPLICATE readings on the schedule (2, 2, 2, 2, 1, 1, 1, 1)).
- Hypotheses: corruption rates {0, 0.4, 0.48}, uniform prior.
- State: 8 relation slots at fixed capacity. For each hypothesis and slot, a Gaussian bias belief (μ, v) and a
  log-evidence Λ, plus 8 counts: 80 logical scalars. The reliability weights are shared across slots:
  `log w_h = log π_h + Σ_r Λ[h, r]`.
- Readout: marginal. A path's success is the product over its relations of the mixture predictive `Σ_h w_h E_h[p_r]`.
  An alternative of edge-disjoint paths succeeds by noisy-OR.
- Invalidation: replay (strategy B) from a bounded buffer of 8 events. Every buffered event is bound to its source
  binding, and only LIVE evidence is replayed.
- Acquisition: **PROBE_R**, by the DEV adoption rule (`SELECTION_R6.json`).
  - With VOI_R, RRMF3_M's on-policy ALL compositional regret was lower: −0.000562 [−0.000791, −0.000334] at 99.9%.
  - But its online regret, 0.327887 [0.319017, 0.336756], exceeds the budget 0.21609027, so VOI_R is not adopted.

**Fallback: SAFE3_P4** (Contract II, first reading only).
- Hypotheses {0, 0.4, 0.48}, prior (1, 4, 4), coupled readout, replay invalidation.

**Gate scales** (from TRAIN, `evidence/train/scales.json`):
- `δ_clean` = 0.00140298;
- `online_budget` = 0.21609027;
- `τ_safe` = 0.0005.

## Plans

The frozen primary drives the log with its frozen acquisition. Every other arm is a shadow fed the same admitted
readings (matched information). The arms are `study.qualification_arms`:
- RRMF3_M, NOOP;
- RRMF3`:IND` (independent per-slot reliability, same filter and readout family);
- RRMF3`:HCA` (LAO_PLUS_HCA);
- RRMF3_M`:PERM` (identity-permuted);
- RRMF3_M`:STALE` (no invalidation; authority regimes only);
- SAFE3_P4;
- EXACT_RELATIONAL, STATIC_COUNTS, HCA_ONLY;
- KNOWN_RELIABILITY, a privileged reference used for scoring only.

| Phase | Partition, seeds × episodes | Regimes | Templates |
|---|---|---|---|
| VALIDATION | VALIDATION 0–31 × 4096 | DESIGN, ROBUSTNESS, AUTHORITY (16) | V_LADDER on even seeds; D_DIAMOND, D_FAN on odd seeds |
| HOLDOUT | HOLDOUT 0–63 × 4096 | the same 16 | H_RANDOM (one per world seed) on even seeds; D_* on odd seeds |
| OOD | OOD 0–63 × 2048 | CLEAN, FLIP20, FLIP40, CORR10–CORR90, EVENT40, SHARED_SRC, SECOND_EVENT40, ASYM40, CTXQ40, FLIP30 | seed mod 4: D_DIAMOND, D_FAN, V_LADDER, H_RANDOM |

Each phase runs once (`study qualify` refuses a second run). The HOLDOUT opens only if every VALIDATION gate part
passes, and the OOD only after the HOLDOUT. `study qualify` enforces both, and refuses sources that differ from
`FROZEN_R6.json`. Neither the HOLDOUT nor the OOD is opened to diagnose a failure.

## Primary gates

The metric is compositional terminal regret per episode (PATH, CONJ and REACH queries). Intervals are two-sided paired
Student-t over world seeds at **95%**, except RG5 at 97.5%. ALL is the mean of CLEAN, FLIP20 and FLIP40. REVISION is
the mean of CLEAN and FLIP20 under REV_DRIFT and RECLAIM_DRIFT. C is the primary.

| Part | Requirement (upper bound of the interval) |
|---|---|
| RG1 | ALL: C − NOOP < 0 |
| RG2_NOOP, RG2_KNOWN | CLEAN: C − NOOP < 0; C − KNOWN_RELIABILITY ≤ `δ_clean` |
| RG3_EXACT_RELATIONAL, RG3_STATIC_COUNTS, RG3_HCA_ONLY | ALL: C − X < 0 |
| RG3_IND | ALL: C − RRMF3`:IND` < 0 (the coupling matters) |
| RG3_HCA | ALL: C − RRMF3`:HCA` < 0 |
| RG4_FLIP40, RG4_FLIP45 | C − NOOP ≤ `τ_safe` |
| RG5 | QUNIF or BURST: C − NOOP < 0, at 97.5% |
| RG6 | ALL: C − RRMF3_M`:PERM` < 0 (the relational binding matters) |
| RG7_STALE, RG7_NOOP | REVISION: C − RRMF3_M`:STALE` < 0; C − NOOP < 0 |
| RG8_NOOP, RG8_IND | ALL on the held-out-topology seeds (V_LADDER on VALIDATION, 16 seeds; H_RANDOM on HOLDOUT, 32 seeds): C − NOOP < 0; C − RRMF3`:IND` < 0 |
| RG9 | ALL online regret of the primary driving on-policy (the phase's own record): mean upper ≤ `online_budget` |
| Mechanical | The R1, R2 and R5 suites pass in CI on the exact head: exact invalidation, revoked sources unreachable, the host differential over the real ECS_C view, the R2 parity, the HCA differential |

**Multiplicity.** Qualification is an intersection–union test: every part must pass, so no adjustment is applied.
Family claims (below) are Bonferroni-adjusted.

## Fallback gates (95%)

- F1: FLIP40, C − NOOP ≤ `τ_safe`.
- F2: CLEAN, C − NOOP < 0.
- F3: FLIP45 and FLIP48, C − NOOP ≤ `τ_safe`.
- F4: EVENT40, C − NOOP ≤ `τ_safe`.

The fallback is a shadow on the primary's log and reads only the first reading of each step.

## Dispositions (`study.primary_status`, `study.fallback_status`)

**Primary:**
- **QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_I**: every part passes on VALIDATION and on HOLDOUT, and the mechanical
  gate passes.
- **INDEPENDENT_RELATIONAL_FILTER_ADAPTER**: every part passes on VALIDATION, and on HOLDOUT the only failing parts
  are coupling parts (RG3_IND, RG8_IND). The gain is then reproducible by independent per-edge filters with an
  unchanged readout, so no contribution of relational coupling is claimed.
- **NOT_QUALIFIED** otherwise, with the failing parts named per phase. If VALIDATION fails, the HOLDOUT and OOD stay
  sealed.

**Fallback:** **QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_II_CONSERVATIVE** if F1–F4 pass on VALIDATION and on
HOLDOUT; otherwise NOT_QUALIFIED.

**Scope.** Synthetic RLB1 worlds only. No real provider is certified, nothing is admitted to a runtime, and every HCA
output stays SALIENCE_ONLY.

## Family claims (`study.families`)

For CLEAN, FLIP20, FLIP40, FLIP45, FLIP48, QUNIF, BURST and EVENT40, with Bonferroni-adjusted two-sided 99.375%
intervals of C − NOOP:
- USEFUL if the upper bound < 0;
- SAFE if the upper bound ≤ `τ_safe`;
- NOT_QUALIFIED otherwise.

The qualified scope is the families labelled USEFUL or SAFE on both VALIDATION and HOLDOUT.

## Secondary estimates (`study.claims`, 95%, reported and not gates)

The estimates:
- direct-relation gain;
- compositional gain vs NOOP, and vs the best matched-information control;
- the coupling vs `:IND`;
- the identity ablation;
- the geometric projection (RRMF3`:HCA` − C);
- the contradiction regimes (FLIP45, FLIP48, EVENT40);
- STOP: stop rates of C, NOOP and KNOWN, and boundary regret vs NOOP;
- online regret;
- CLEAN vs KNOWN, and clean retention;
- edge MSE;
- per-kind compositional gains;
- source revision vs `:STALE` and vs NOOP;
- the held-out topologies.

Interpretation, fixed now:
- **Relational learning** is claimed only if RG6 holds on both phases.
- **The coupling** is claimed to contribute only if RG3_IND and RG8_IND hold on both phases. Otherwise the result is
  an independent relational-filter adapter at most.
- **The HCA projection.** Unless the upper bound of RRMF3`:HCA` − C is below 0, the projection adds no advantage over
  the exact compositional readout. The simpler rule is then preferred, and HCA stays a salience-only view.
- **STOP.** The stopping problem is not declared solved. Stop rates are compared with KNOWN_RELIABILITY, and boundary
  regret (regret where the STOP choice disagrees with the true optimum) with NOOP.

## Correlation admission (OOD; `study.rho_admission`)

- **Grid** at q = 0.4: ρ ∈ {0 (FLIP40), 0.1, 0.25, 0.5, 0.75, 0.9, 1 (EVENT40)}.
- **ρ_admit** is the largest grid ρ such that every grid ρ′ ≤ ρ has compositional C − NOOP with a 95% upper bound
  ≤ `τ_safe`. It is none if ρ = 0 fails.
- **Host use.** The relational host can use min(ρ_admit, `MAX_QUALIFIED_RHO` = 0.25), because the strict host
  refuses more. The SLB2 threshold is not reused.
- **Other OOD results.** The OOD corruption regimes and the per-topology results (ALL, C − NOOP and C − `:IND`) are
  reported pass or fail. There is no OOD gate.

## Native adapter and host parity (only if a disposition qualifies)

- **Native vs scalar.** The native C11 adapter must be **bitwise** equal to the frozen scalar reference on every
  operation the host performs: the state after every operation, the weights, the predictive, every readout of every
  alternative, rebuild and slot reset.
  - Samples, through the host over the real ECS_C view: VALIDATION seeds 0–7 × 256 episodes per VALIDATION regime,
    and the same on HOLDOUT.
  - Configurations: the primary (Contract I) and the fallback (Contract II).
- **Host vs vectorised.** On the same episodes, the host reproduces the vectorised frozen arm's selections and
  terminal decisions, except documented near-ties: a gap under 1e-9 at the decision.
- **CI.** The adapter builds and is tested in every native job, including the sanitizer builds and the
  allocation-free test.

## Resources

On 4 workers: VALIDATION about 1 hour, HOLDOUT about 2 hours, OOD about 1 hour. A record is written only when its
phase completes. A phase interrupted by an execution boundary is re-run from the start under the same frozen sources.
The run is deterministic and nothing partial is inspected.

## Continuation if not qualified

If VALIDATION fails:
- the HOLDOUT and OOD stay sealed, and the disposition is recorded;
- diagnosis uses fresh DEV seeds (64 and up) under a protocol written first;
- a new qualification needs new VALIDATION, HOLDOUT and OOD partitions under a new RNG root.
