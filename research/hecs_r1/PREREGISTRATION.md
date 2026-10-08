# H-ECS R1: hierarchical ECS world models — preregistration

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

The machine-readable authority is `specs/hecs-r1.v1.spec.json`, rendered by `hecs1 spec` and frozen in the
same commit as this document. Every evidence file embeds it. Where this text and the specification differ, the
specification decides. Both are write-once: no parameter, seed, condition, measurement, gate or stop law below
may change after any DEV or QUAL observation exists.

## 0. Relation to R0

H-ECS R0 (`research/hecs_r0`, `docs/research/HECS_R0_RESULTS.md`) is closed with `TASK_INVALID_ON_DEV`. Its
absolute validity gate V1 (FLAT_N36 level-1 one-step NMSE ≤ 0.25) sat at the SEPARATED world's irreducible
floor. The hierarchy questions were **not measured**; R0 is not a negative hierarchy result, and nothing here
reinterprets or edits it.

R1 asks R0's question with R0's mechanics. Its laboratory (`research/hecs_r1/rust`) is a copy of R0's; every
file is either byte-identical to R0's or listed as modified, with the reason, in `SOURCE_EQUIVALENCE.json`
(checked by `tests/research/hecs_r1`). The R0 sources and evidence are not touched.

What R1 changes, all fixed before any R1 DEV observation:

1. **The V1 validity law** (section 5): an analytic floor, a held-out best-linear reference and ECS adequacy
   relative to that reference, instead of an absolute threshold.
2. **Seeds** (section 4): fresh DEV and QUAL seeds by SHA-256 derivation, disjoint from R0's; DESIGN-only seeds.
3. **V4's MATCHED bound**: 1.5 → 2.0 (section 6). DESIGN showed R0's bound at the generator's own typical value.
4. **The planner's goal cost**: summed path distance instead of endpoint distance (section 6). DESIGN showed the
   oracle could not meet V2 with the endpoint cost.

Everything else is R0's: worlds, data, ECS primitive and training, encoders, conditions, widths, depths, strides,
windows, compute budgets, measurements and the V2, V3, V5, H1, H2 and depth thresholds.

## 1. Question

> Width scaling inside one ECS keeps one representation and one update regime. Does a hierarchy of canonical
> ECS states at distinct temporal scales, optionally with distinct learned representation spaces between
> levels, behave differently from width scaling of one ECS?

The conceptual reference is H-JEPA (arXiv:2610.06805): per-level latent spaces at coarser strides, top-down
subgoals, and higher levels discarding fast detail only when timescales are separated. R1 borrows those
principles, not the architecture. **ECS is not a JEPA**: there is no end-to-end trained neural encoder, no
SIGReg and no transformer predictor.

## 2. What is fixed and not touched

- **The ECS primitive.** Each level owns exactly one native K1 state `(W, epoch, H, a)`, created and driven
  through the existing C ABI (`create`, `forward`, `learn`, `state_digest`). The qualified cubic equations are
  unchanged. Learning is LEARN only, so H stays zero and every step is exactly G1. `S3` is an observable, never
  retained state; `W` is authoritative.
- **One state per level.** No shared microscopic `W`; no single wide ECS standing in for levels. A level sees
  only its lower level's latents and actions.
- **Research only.** No canonical inference path, no ECS↔DSV codec, no HACF edge, no continuity use.

## 3. Design (R0's, restated)

- **World.** Slow factor `s' = clip(s + 0.04 a + 0.005 n)`; second factor `f' = ρ f + sqrt(1 − ρ²) 0.5 n`, never
  influenced by the action. SEPARATED: ρ = −0.7 (fast). MATCHED control: ρ = 0.995 (timescale close to `s`).
  Observation: a world-specific rotation of `(s, f)` plus noise 0.01. Exploration: 16 training and 8 held-out
  episodes of 256 steps; uniform actions held 1–8 steps.
- **Levels.** Level 1 whitens the observation (clock = environment step). Level l > 1 ticks every 4 lower steps;
  its latent encodes the window of the last `window` lower latents ending at the tick (past only); its action is
  the mean lower action over the stride. ECS input `[z0, z1, b, sel0, sel1, 1]` (d = 6), target `z'_k`.
  Training: a fixed K1-step budget on shuffled mini-batches of 128 transitions, 25 steps per call, rate
  `0.002 · 36 / width`, `W ~ N(0, 0.18²)`.
- **Conditions** (total width matched):

  | Condition | Levels (width) | Upper representation |
  |---|---|---|
  | FLAT_N36 / N72 / N108 | 1 (36 / 72 / 108) | none: the width baseline |
  | TEMPORAL_SHARED_L2 / L3 | 2 / 3 (36 each), stride 4, window 1 | none: the lower latent at the tick |
  | HIERARCHICAL_DISTINCT_L2 / L3 | 2 / 3 (36 each), stride 4, window 2 | learned affine window (4) → 2 |

  The distinct encoder keeps the two most predictable unit-variance directions of the window: the closed-form
  generalized eigenproblem `R p = λ S p` against a linear one-tick model (a linear predictability proxy, not
  end-to-end training through the ECS).
- **Measurements** (every level of every condition): prediction NMSE at horizons 1, 2, 4, 8 and the diverged
  fraction; slow and fast recoverability (ridge probes to `s` and `f` at the window end); the training trajectory
  (epoch, training MSE, `‖W‖_F` at 0/25/50/75/100% and the final digest); attractors (free run at `b = 0` from a
  5 × 5 grid); compute (training multiply-adds, planning forward rows and multiply-adds); memory (K1 image,
  workspace, encoder parameters, controller resident bytes); and, new in R1 as a report, each level's one-step
  adequacy (ECS vs that level's own linear reference, constant and persistence baselines).
- **Planning.** 24 paired episodes per world; reach the goal slow value within 64 steps (`|s − s_goal| ≤ 0.1`),
  goal given as an observation rendered from `(s_goal, f_goal)` with `f_goal` from the second factor's law.
  Deterministic random shooting over the levels' own ECS models: FLAT plans 16 steps (blocks of 4);
  hierarchies plan top-down (the top plans 4 of its steps, each lower level plans one upper tick to its
  subgoal). Every condition executes 4 steps per replan, from the encoding of each level's most recent window.
  Compute is matched in native forward rows per replan, ≈ {256, 1024, 4096}: K = {8, 32, 128} (FLAT),
  {16, 64, 256} (two levels), {11, 43, 171} (three levels, within 4%). Subgoal tracking: the normalized squared
  distance between the level-2 latent reached and level 1's subgoal. Oracle: the flat planner over the true
  slow dynamics (validity only).

## 4. Seeds and phases

- **Derivation.** Seed `i` of phase `p` is the first 8 bytes (big-endian) of
  `SHA-256("elpis.hecs-r1.<p>.v1:<i>")`: DEV `i = 0..3`, QUAL `i = 0..7`, DESIGN `i = 0..3`. They were fixed
  before any R1 observation, are not chosen and none equals an R0 seed (0–3, 100–107, 9999).
- **DESIGN** (`hecs1 design`, DESIGN seeds only): level-1 validity diagnostics, level-1 probes, spectral
  centroids and the oracle at every grid budget. It trains no upper level and plans with no ECS: no hypothesis
  quantity exists in DESIGN evidence. DESIGN informed sections 5 and 6 and is committed
  (`evidence/design/hecs-r1.v1.design.json`); its seeds never enter DEV or QUAL.
- **DEV calibration** (`hecs1 calibrate`): budgets {6000, 24000, 96000} K1 steps; choose the smallest budget at
  which V1 (A, B and C) holds in every DEV instance (both worlds). That is the only criterion consulted. None:
  `TASK_INVALID_ON_DEV`, stop.
- **DEV run** (`hecs1 run --phase dev`) at the chosen budget. Any of V1–V5 failing: `TASK_INVALID_ON_DEV`,
  stop. DEV hypothesis values are reported and non-binding.
- **QUAL run** (`hecs1 run --phase qual`), authorized by a valid DEV and run **once** on the QUAL seeds at the
  DEV budget. Its disposition is final.

## 5. The R1 validity law (V1)

For each world instance, on exactly the held-out one-step examples the ECS is scored on, with `horizon_error`'s
normalizer (mean held-out per-coordinate latent variance), FLAT_N36 level 1:

- **F, analytic floor.** `tr(P Q Pᵀ) / 2 / normalizer`, with `P` the instance's whitening map and
  `Q = R diag(0.005², (1 − ρ²) 0.5²) Rᵀ + n² I + n² A Aᵀ`, `A = R diag(1, ρ) Rᵀ`, `n = 0.01`: factor innovations,
  next-observation noise, and the current observation's noise carried by the dynamics (unclipped).
- **L, linear reference.** Least squares `z' ~ [z0, z1, b, 1]` on the training partition only, scored on the
  held-out examples. It never sees evaluation targets.
- **C, constant baseline** (training mean of `z'`) and **P, persistence** (`z' = z`), reported.
- **E, ECS.** One-step NMSE; a diverged prediction makes it infinite.

| Gate | Requirement | What it rules out |
|---|---|---|
| **V1A REFERENCE_PREDICTABILITY** | `L ≤ (1 − 0.5) · C` | a bad task: less than half the one-step error is predictable at all |
| **V1B REFERENCE_AT_FLOOR** | `|L − F| ≤ 0.005 + 0.15 · F` | a world that is not what its generator says (leakage, mis-construction, a mis-specified floor) |
| **V1C ECS_ADEQUACY** | no divergence and `(C − E) ≥ 0.98 · (C − L)` | bad ECS training: the ECS leaves more than 2% of the linearly removable error unlearned |

Irreducible stochasticity is **not** a failure: a high floor is valid when `L` sits at it and the ECS captures what
`L` captures. That is exactly the case R0's absolute gate rejected.

**Why these margins (DESIGN evidence, DESIGN seeds only).**
- A: the linear reference removed 0.71–0.83 (SEPARATED) and 0.99 (MATCHED) of the constant error; 0.5 leaves
  a wide margin for real predictable structure and fails a world without it.
- B: `L/F` was 0.93–1.07, `|L − F|` ≤ 0.015 (SEPARATED, F ≈ 0.22–0.29) and ≤ 0.0006 (MATCHED, F ≈ 0.008);
  the bound allows ≈ 0.04 and ≈ 0.006 respectively: roughly 2.6× and 10× the observed deviation. The
  relative term keeps B meaningful at MATCHED's small floor; R0's own post-hoc diagnosis (ECS within 0.002 of
  the linear predictor) agrees.
- C: the ECS captured ≥ 0.994 of the linearly removable error at every budget and instance (≥ 0.9996 in
  MATCHED from 24000 steps). An untrained or failing ECS captures far less; 0.98 separates the two without
  touching the converged regime.

**V2–V5** (R0's, unchanged except V4's MATCHED bound): V2 oracle success ≥ 0.8 in each world (mean); V3
level-1 probe NMSE ≤ 0.1 for `s` and `f`, every instance; V4 mean spectral-centroid ratio `f/s` ≥ 4
(SEPARATED) and ≤ 2.0 (MATCHED); V5 no K1 refusal in any training.

## 6. Design corrections inherited from R0 (before any DEV data)

- **V4 MATCHED: 1.5 → 2.0.** R0 chose ρ = 0.995 from one design instance with ratio 1.36 and set the bound at
  1.5. DESIGN instances measured 1.34, 1.47, 1.51 and 1.65 (mean 1.49): the bound sat on the generator's
  typical value, R0's V1 mistake again. 2.0 is still an order of magnitude below SEPARATED (30–33) and keeps the
  control's meaning, no separated fast timescale.
- **Planner cost: endpoint → summed path distance.** R0 scored a candidate by its 16-step endpoint and executed
  only its first block, so near the goal the executed block was arbitrary among candidates sharing the best
  endpoint and the state wandered ±0.16 around the goal. The oracle succeeded in 0.50–0.75 of DESIGN episodes,
  below V2's 0.8; the planning task was invalid for every condition alike. Scoring every predicted state's
  squared distance to the goal (standard trajectory cost) gives oracle success 1.0 on DESIGN. Subgoal tracking
  (one upper tick, scored at its end) is unchanged. This applies identically to the oracle, FLAT and the
  hierarchies' top level.

## 7. Hypotheses (R0's, unchanged; QUAL decides; means over seeds unless stated)

**H1, selective abstraction (HIERARCHICAL_DISTINCT_L2).** `dF = NMSE_f(L2) − NMSE_f(L1)`.
- **H1:** in SEPARATED, mean `dF ≥ 0.3`, `NMSE_s(L2) ≤ 0.2`, and `dF ≥ 0.3` in at least 75% of seeds.
- **H1C (data dependence):** mean `dF` in MATCHED ≤ mean `dF` in SEPARATED − 0.2.
- **H1T (temporal control):** TEMPORAL_SHARED_L2 mean `dF` in SEPARATED < 0.1.

**H2, planning (SEPARATED; success means over seeds at each compute point).**
- **H2A, beyond width:** HIERARCHICAL_DISTINCT_L2 − FLAT_N72 ≥ 0.15 at ≥ 2 of 3 compute points.
- **H2B, beyond temporal decomposition:** HIERARCHICAL_DISTINCT_L2 − TEMPORAL_SHARED_L2 ≥ 0.10 at ≥ 2 of 3.

**Reported, not gated:** width (FLAT_N36 / N72 / N108 success); depth L1 → L2 → L3 per representation at the
middle compute point (IMPROVES ≥ +0.05, NO_CHANGE, DEGRADES ≤ −0.05); every measurement in MATCHED; per-level
adequacy, horizon errors, attractors, trajectories, compute and memory. A result where both worlds show the same
"abstraction" is to be investigated for metric artifacts, not celebrated.

**Disposition.** `TASK_INVALID` if any of V1–V5 fails; else `OUTCOME_A` (H1, H1C, H1T, H2A, H2B all hold),
`OUTCOME_B` (the H1 family holds, H2 not), `OUTCOME_C` (H2 holds, the H1 family not), `OUTCOME_D` (neither).
The interpretation keeps separate the temporal-decomposition effect (TS vs FLAT), the distinct-representation
effect (HD vs TS) and the raw width effect (FLAT widths). Every one of them may be negative; a valid negative
result is preserved as such.

## 8. Integration law

No disposition makes H-ECS canonical. Only QUAL `OUTCOME_A` authorizes an **experimental**, explicitly
noncanonical integration: a Rust hierarchy controller over the same K1 C ABI consuming already-defined
native-ready ECS inputs and producing bounded hierarchical readouts and subgoals; no Python scheduling, no
change to `Runtime.run_turn`, no semantic codec, no HACF edge, no continuity use, no trajectory log. Any other
outcome authorizes no integration.

## 9. What would not rescue a failure

No gate, threshold, world parameter, budget, seed or condition changes after DEV or QUAL data exist. The
calibration grid is the only DEV-time choice and its rule is fixed above. A failed validity gate is reported as
`TASK_INVALID_ON_DEV` (or `TASK_INVALID` in QUAL); a failed hypothesis is reported as failed.
