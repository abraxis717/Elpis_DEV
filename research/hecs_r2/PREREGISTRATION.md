# H-ECS R2: multi-step validity of ECS world models, then hierarchy — preregistration

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

The machine-readable authority is `specs/hecs-r2.v1.spec.json`, rendered by `hecs2 spec` and frozen in the same
commit as this document and the DESIGN evidence it cites (`evidence/design/hecs-r2.v1.design.json`). That commit
is the freeze: before it, R2 was DESIGN/MECHANICS and not frozen. Every evidence file embeds the specification;
where this text and the specification differ, the specification decides. No parameter, seed, horizon, condition,
measurement, gate or stop law below may change after any DEV or QUAL observation exists.

## 0. Relation to R0 and R1

Both are closed and are not edited, rerun or reinterpreted here.

- **R0** (`research/hecs_r0`): `TASK_INVALID_ON_DEV`. Its absolute one-step validity gate sat at the SEPARATED
  world's irreducible floor.
- **R1** (`research/hecs_r1`, preregistration frozen in `5c57dec`): `TASK_INVALID`. DEV calibrated at 6000 K1
  steps and passed V1–V5; the single QUAL run failed V1C in 1 of 16 instances (MATCHED, capture 0.9785 < 0.98;
  ECS 0.0290, linear 0.0063, floor 0.0063). The hypotheses were not adjudicated. The one permitted R1 edit is a
  documentation correction (`docs/research/HECS_R1_RESULTS.md`, "Corrections"): R1's results had attributed the
  poor distinct-hierarchy planning to a cause the planner code contradicts; the cause is now stated as
  unresolved. No R1 evidence, threshold, disposition or nonclaim changed.

Two things R1 left open motivate R2:

1. **R1's calibration had no margin**, and its adequacy bar measured absolute quality, not training
   convergence (section 8).
2. **R1 never measured the object the planners consume.** Every planner rolls a level model forward recursively
   (FLAT: 16 level-1 steps; hierarchies: 4 steps per level), but training and every R1 gate are one step,
   teacher-forced. In R1's recorded QUAL evidence, 29 of 112 level models had a horizon-8 NMSE above 10³ (up to
   ~10²²⁴; DEV: 12 of 56) while the recorded "diverged" fraction was 0 for every one of them: R0/R1 counted only
   non-finite native results. (A descriptive reading of R1's evidence, not a reinterpretation of its
   disposition.) A hierarchy comparison is meaningless if the world models it plans with are not valid at the
   horizons it uses.

## 1. Questions and their order

**Primary (M).** Can the qualified ECS/K1 world-model primitive, trained one step ahead exactly as in R0/R1,
remain a useful predictive object over the recursive horizons the H-ECS planners actually consume?

**Conditional (H2).** Only if it can: does a hierarchy of ECS states plan better than width scaling at matched
compute, and is any advantage attributable to the distinct representation beyond temporal decomposition?

The logical order is fixed:

```
task / world valid                                  (T1, T3, V2-V5: section 7)
  -> one-step constituent models valid              (S1: section 9)
  -> multi-step constituent models valid at the
     consumed horizons                              (M: section 10)
  -> planning task oracle-valid                     (V2)
  -> only then the hierarchy hypotheses             (H2: section 12)
```

A failure of S1 or M is a result about the ECS world model, **not** a hierarchy result: it is never reported as
evidence against (or for) hierarchy. Equally, if M holds, tangent expansion alone is not a hierarchy failure.

## 2. What is fixed and not touched

- **The ECS primitive.** Each level owns exactly one native K1 state `(W, epoch, H, a)`, created and driven
  through the existing C ABI. The qualified cubic equations are unchanged (`k1.rs` and `model.rs` are R1's
  bytes). Learning is LEARN only (H stays zero: exact G1), one step ahead on teacher-forced transitions. `S3` is an
  observable, never retained state; `W` is authoritative. R2 reads `W` (existing `copy_w`) to compute exact state
  Jacobians; it adds no native API.
- **One state per level.** No shared `W`; no single wide ECS standing in for levels.
- **Research only.** No canonical inference path, no ECS↔DSV codec, no HACF→ECS edge, no continuity use, no
  language claim.

## 3. Design (R1's, unchanged)

Worlds, data, level construction, encoders, the seven conditions, widths, depths, strides, windows, training rule,
planner, compute budgets and oracle are R1's (`SOURCE_EQUIVALENCE.json`; `tests/research/hecs_r2` checks the
specification sections against R1's):

- **World.** Slow `s' = clip(s + 0.04 a + 0.005 n, −1, 1)`; second factor `f' = ρ f + sqrt(1 − ρ²) 0.5 n`, never
  influenced by the action. SEPARATED ρ = −0.7, MATCHED ρ = 0.995. Observation: a world-specific rotation of
  `(s, f)` plus noise 0.01. 16 training and 8 held-out episodes of 256 steps; uniform actions held 1–8 steps.
- **Levels.** Level 1 whitens the observation. Level l > 1 ticks every 4 lower steps; TEMPORAL_SHARED (TS) keeps
  the lower latent at the tick (window 1); HIERARCHICAL_DISTINCT (HD) encodes a window of 2 lower latents with a
  closed-form affine "most predictable directions" map fixed from lower training data before the ECS trains.
  **Any abstraction HD shows is that fixed encoder's, not something the ECS learned.**
- **Conditions.** FLAT_N36 / N72 / N108; TEMPORAL_SHARED_L2 / L3; HIERARCHICAL_DISTINCT_L2 / L3 (36 per level;
  FLAT_N72 matches the two-level total width, FLAT_N108 the three-level).
- **Planning.** 24 paired episodes, 64 steps, replan every 4, summed path cost, success `|s − s_goal| ≤ 0.1`;
  compute matched in native forward rows per replan (K = {8, 32, 128} FLAT, {16, 64, 256} two levels,
  {11, 43, 171} three levels); oracle over the true slow dynamics.

**Consumed horizons** (what each planner rolls each level out over, in that level's own steps; the mechanics
tests check them against the planners' native row counts):

| Condition | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| FLAT_N36 / N72 / N108 | 16 | – | – |
| TEMPORAL_SHARED_L2, HIERARCHICAL_DISTINCT_L2 | 4 (one level-2 tick) | 4 (its horizon) | – |
| TEMPORAL_SHARED_L3, HIERARCHICAL_DISTINCT_L3 | 4 | 4 (one level-3 tick) | 4 |

**Binding pairs.** The planning comparison's conditions (FLAT_N72, TEMPORAL_SHARED_L2, HIERARCHICAL_DISTINCT_L2)
bind task validity T3, S1 and M through their (level, consumed horizon) pairs: **(L1/N72, 16), (L1/N36, 4),
(TS/L2, 4), (HD/L2, 4)**. Every other condition and level (FLAT_N36 / N108, the three-level conditions) is
measured and reported identically but gates nothing. Measured horizons at every level: 1, 2, 4, 8, 16.

## 4. Seeds and phases

- **Derivation.** Seed `i` of phase `p` is the first 8 bytes (big-endian, unsigned) of
  `SHA-256("elpis.hecs-r2.<p>.v1:<i>")`: DESIGN `i = 0..3`, DEV `i = 0..3`, QUAL `i = 0..7`. Fixed before any R2
  observation; disjoint from each other, from R0's and from R1's.
- **Provenance.** Every seed is recorded as an exact decimal string (R1 recorded u64 seeds through a signed
  integer). Every evidence block that needs no ECS (`references`: references, clip accounting, T3, oracle, probes,
  centroids) carries its world and seed string, and the mechanics tests recompute it verbatim from that recorded
  seed.
- **DESIGN** (`hecs2 design`, DESIGN seeds only): references at every level and horizon, T3, the oracle, probes,
  centroids, clip accounting, the references' own tangent growth and domain excursion; one-step adequacy of every
  level untrained and at every grid budget. **No ECS prediction beyond one step and no ECS planning**: no M,
  tangent, excursion or hypothesis quantity of an ECS exists in DESIGN evidence. Section 13 records what DESIGN
  decided.
- **DEV calibration** (`hecs2 calibrate`, section 8). No converged budget: `TASK_INVALID_ON_DEV`, stop.
- **DEV run** at the chosen budget. Task validity fails: `TASK_INVALID_ON_DEV`, stop. Otherwise QUAL is
  authorized; DEV's S1, M and H2 values are recorded and non-binding.
- **QUAL run**, once, on the QUAL seeds at the DEV budget. Its disposition is final. There is no second QUAL.

## 5. Measurements

All errors are NMSE: squared error averaged over the two latent coordinates, divided by the mean held-out
per-coordinate latent variance (R1's normalizer), on exactly the same held-out starts and targets.

**One step** (R1's): ECS `E`, linear reference `L` (least squares `z' ~ [z, b, 1]`, training data only),
constant `C` (training mean), persistence, analytic floor `F` (level 1). Capture `(C − E) / (C − L)`.

**Multi-step** (`multistep.rs`, every trained level, h ∈ {1, 2, 4, 8, 16}), from every held-out start with the
true level actions:

- **ECS free-running** `E_h`: the open-loop rollout of the ECS map, fed its own predictions.
- **NUMERICAL_ESCAPE**: a rollout state that is non-finite or has any |z| > 10³ (whitened units; true latents
  stay within a few units). A catastrophic-divergence sentinel: the rollout stops and is counted; it is not scored.
- **DOMAIN_EXCURSION**: a rollout state outside the level's **training envelope** (the axis-aligned box of every
  training latent of that level, no margin; training data only) at some step ≤ h. It says the model left the
  region its training data supports, not that it diverged. Reported with the truth's own excursion rate and the
  linear reference's, and for planner-candidate rollouts (the planner's own action generator) as well.
- **ECS teacher-forced** `T_h`: the one-step prediction of the same target `z_{t+h}` from the true `z_{t+h−1}`.
- **References**: the iterated linear reference `L_h` (R1's one-step map, iterated open loop); the generator
  plug-in `G_h` (level 1 and TS levels, whose latent is a whitened observation: the true dynamics, clipping
  included at level-step granularity, run from the factor estimate the latent implies); constant `C_h`;
  persistence `P_h` (`z_{t+h} = z_t`); the unclipped analytic floor `F_h` (a consistency diagnostic only,
  section 7).
- **Capture** at h: `(C_h − E_h) / (C_h − L_h)`.
- **Finite-horizon tangent diagnostics** (section 11): exact state Jacobians, ordered derivative products on the
  teacher and free-running paths, `σ_max(P_h)` and `γ_h = log σ_max(P_h) / h`; the linear map's `σ_max(A^h)`; the
  generator's product; direct perturbation amplification (ε = 10⁻⁴); passive (`b = 0`) escape; a first-order error
  recursion along the free path; the one-step spectral radius (descriptive only).

## 6. The generic rollout-error recurrence (mathematics, not an ECS claim)

Let the true and model transitions under the **same** registered action sequence be `x_{t+1} = F_t(x_t)` and
`y_{t+1} = G_t(y_t)`, from the same state `x_0 = y_0`, and `e_t = ‖x_t − y_t‖`. Assume, on a domain `D`:

1. both trajectories remain in `D` for `t < h`;
2. a **uniform** one-step discrepancy bound: `‖F_t(x) − G_t(x)‖ ≤ ε_t` for every `x ∈ D`;
3. `L_t` is a **valid Lipschitz bound** of the model map on `D`: `‖G_t(x) − G_t(y)‖ ≤ L_t ‖x − y‖`.

Then `e_{t+1} = ‖F_t(x_t) − G_t(x_t) + G_t(x_t) − G_t(y_t)‖ ≤ ε_t + L_t e_t`, and by induction

```
e_h <= sum_{i=0}^{h-1} eps_i * prod_{j=i+1}^{h-1} L_j
```

For constant `ε`, `L`: `e_h ≤ ε (L^h − 1) / (L − 1)` if `L ≠ 1`, and `e_h ≤ h ε` if `L = 1`. The bound is attained
by `G(x) = L x`, `F(x) = L x + c` with `c` fixed in direction (`multistep::recurrence_bound`,
`constant_bound`; tests: closed forms, the sum form, tight and loose linear constructions, a nonlinear
contraction).

**What it explains.** Excellent one-step prediction (small `ε`) does not imply planner-usable recursion: under an
expanding model map the one-step discrepancy is amplified geometrically. With `ε = 10⁻³` and `L = 1.5`, a
16-step error above 1 is attained. Under a contracting map, errors saturate instead.

**What it does not license.** R2's empirical derivative products are *local*: they are measured along particular
paths at particular states. They are not certified uniform Lipschitz bounds on a domain containing both
trajectories, and the measured one-step residuals are not a uniform `ε`. So R2 never evaluates this bound as a
theorem about the ECS; agreement between tangent growth and observed free-running error is reported as
descriptive evidence only (section 11).

## 7. Task validity (every instance, both worlds)

| Gate | Requirement | What it rules out |
|---|---|---|
| **T1A** (R1 V1A) | level-1 one-step `L ≤ (1 − 0.5) C` | a task with little predictable structure |
| **T1B** (R1 V1B) | level-1 one-step `|L − F| ≤ 0.005 + 0.15 F` | a world that is not what its generator says |
| **T3A** | every binding pair: `L_h ≤ (1 − 0.25) C_h` | a consumed horizon with no predictable structure left |
| **T3B** | every binding observation pair: `L_h ≤ 1.20 · G_h` on the same targets | a linear reference far from what the true dynamics achieve, which would make capture relative to it weak |
| **V2–V5** (R1's) | oracle success ≥ 0.8 per world; level-1 probes ≤ 0.1; centroid ratio ≥ 4 (SEPARATED), ≤ 2 (MATCHED); no K1 refusal in any level | an unsolvable planning task; an unrecoverable level-1 latent; worlds that are not what they claim; refused training |

**The analytic h-step floor: audit and decision (DIAGNOSTIC_ONLY).** The world clips the slow factor at ±1 each
environment step; `F_h` uses the unclipped dynamics. DESIGN measured how often the trajectories meet the
boundary: at the binding pairs 7.9–27.5% (SEPARATED) and 5.7–15.2% (MATCHED) of the consumed windows contain a
clip event, up to 56% at TS/L3 (h = 4). So option A (clipping negligible) is **not** defensible beyond one step.
Mathematically, clipping is a 1-Lipschitz contraction applied after independent noise, so at level 1 (per-step
actions known) it can only lower the plug-in predictor's slow-factor error relative to `F_h` (Efron–Stein); at
coarser levels the within-stride action order is unknown and clipping can raise it. And `F_h` is an expectation
compared with finite-sample estimates: on DESIGN, `L_h/F_h` spans 0.94–1.13 at the binding pairs and 0.82–1.41
at TS/L3 (104 held-out starts), and the spread is the same on the unclipped windows alone. A tight `|L_h − F_h|`
tolerance therefore fails on sampling noise: the provisional floor-based T3B failed in 4 of 8 DESIGN instances
(TS/L3), which is R0's mistake again. A loose one would test nothing. R2 therefore takes **option C**, with the
measured clipping-aware generator (option B) as the reference. T3B compares the iterated linear reference with the
generator plug-in **on the same targets**: `L_h/G_h` is 1.0005–1.0244 (SEPARATED) and 1.016–1.079 (MATCHED) at
the binding pairs on DESIGN, and the bound 1.20 leaves 2.5× the largest observed excess. `F_h`, its ratios and the
clip fractions are reported as consistency diagnostics. At one step (T1B, inherited) the floor remains a gate:
there, clipping touches 3.8–10.9% of single steps with an overshoot of at most one action step, and R1's tolerance
covers the DESIGN spread (`L/F` 0.96–1.05).

Irreducible noise does not invalidate the task: every multi-step criterion is relative to what is predictable at
that horizon on the same targets.

## 8. Calibration: training convergence, with a margin

DESIGN showed that one-step capture **plateaus per instance** at a level set by the data and the model's bias,
not by the training budget. From 96000 to 384000 steps no binding level moves by more than 0.0005, yet the plateaus
differ: TS/L2 (SEPARATED) converges at 0.980 in one DESIGN instance and at ≥ 0.99 in the others; L1/N36 at 0.991;
the level-3 models at 0.951–0.965. R1's "smallest budget at which capture ≥ 0.98" and the provisional R2 rule
"capture ≥ 0.99 at every level, then one grid step up" both test absolute quality, not convergence: the latter
would never have been met on DESIGN, at any budget, for a reason unrelated to R2's question.

**Rule.** Over the grid (6000, 24000, 96000, 384000 K1 steps per level), `B*` is the smallest budget `b_i` such
that, from `b_i` to `b_{i+1}` (4× the training), **no binding level of any DEV instance improves its one-step
capture by more than 0.005**. A refusal or divergence is never converged. The chosen budget is **one grid step
above `B*`**, so it lies past measured convergence. No such `B*`: `TASK_INVALID_ON_DEV`.

**Why 0.005 (DESIGN).** The largest binding improvement was 0.0154 from 6000 to 24000, 0.0060 from 24000 to 96000
and 0.0005 from 96000 to 384000. Whichever of the last two steps DEV finds converged, the chosen budget is one
whose next 4× step moved no binding level by more than 0.0005 on DESIGN. Absolute adequacy is a separate, scientific
stage (S1), not a calibration criterion.

## 9. S1: one-step constituent validity (QUAL decides, SEPARATED binds)

Every binding level of every seed: trained without refusal, no divergence, one-step capture **≥ 0.95**. Fails in
SEPARATED: `ONE_STEP_MODEL_INVALID`. M is still reported, and the hierarchy is not adjudicated.

**Why 0.95 (DESIGN).** S1 must separate a trained model from an untrained or broken one without cutting into the
converged regime. On DESIGN, untrained one-step capture was −249 to −1.05 and the lowest converged binding value
was 0.9802 (TS/L2, SEPARATED). 0.95 sits 0.03 below the converged minimum. That is twice R1's margin, whose
QUAL miss came from an unconverged budget, which the calibration now excludes. The inherited 0.98 would cut into
the converged regime, as R1's own justification forbade. S1 is reported for MATCHED and for the non-binding levels.

## 10. M: multi-step validity (the primary gate; QUAL decides, SEPARATED binds)

**Definition.** M holds in a world iff **every seed** meets, at **every binding pair** and at **every measured
horizon h ≤ that pair's consumed horizon**:

- free-running capture `(C_h − E_h) / (C_h − L_h) ≥ 0.90`, and
- NUMERICAL_ESCAPE ≤ 0.01 of the rollouts.

M fails in SEPARATED: `WORLD_MODEL_INVALID`. That is the primary question answered negatively, and it is not a
hierarchy result.

**Audit (DESIGN; the constants were not kept merely because they existed):**

- *What it measures.* The planner consumes free-running predictions at every step up to the consumed horizon
  (its path cost sums every predicted state), so M is evaluated at every measured h up to it, not only at its end.
  Teacher-forced error is excluded from M on purpose: it is not the object the planner consumes. It is reported, and
  the gap between teacher-forced and free-running error diagnoses compounding.
- *Usefulness.* With T3A, an M-valid level's free-running prediction removes at least 0.90 × 0.25 of the
  constant baseline's error at its consumed horizon. A correct predictor clears the bound by a wide margin: the
  generator plug-in's capture at the binding pairs is ≥ 1.0003. A trivial predictor does not: persistence reaches
  at most 0.6275 in SEPARATED. **In MATCHED persistence reaches 0.936–0.993**, so M = 0.90 is not informative
  there. MATCHED M is reported with that flag (`M_informative_MATCHED`) and never interpreted as usefulness. This is
  one reason M binds in SEPARATED, the world in which the planning comparison is adjudicated.
- *Catastrophe.* An escaped rollout counts against M directly; one escaped seed beyond 1% fails M. A large but
  finite runaway error drags capture far below 0.90. Capture is computed over unescaped rollouts, a bias bounded
  by the 1% escape allowance.
- *Seed aggregation: every seed.* The claim is "the primitive, trained this way, yields a planner-valid
  constituent model in fresh worlds". A model that is invalid at its consumed horizon in a quarter of fresh worlds
  would plan badly in those worlds, so the provisional 75% rule made the claim weaker than its wording. Per-seed
  sampling variation of a near-correct predictor's capture is small: the generator's binding captures span
  1.0003–1.0135 on DESIGN, against a margin of 0.10. So a per-seed failure reflects the model, not sampling.
  `M_SEED_FRACTION = 1`.
- *Excluded from M, with reasons.* **DOMAIN_EXCURSION**: on DESIGN the truth itself leaves its training box in up to 9.4% of
  binding windows (HD/L2, MATCHED) and the valid linear reference in up to 18.4% (HD/L2, SEPARATED), so a box gate
  would reject valid predictors; excursion that harms prediction already lowers capture. **Tangent growth and perturbation
  amplification**: the true process has `σ_max(P_h)` = 1.0000–1.0030 at the binding pairs in whitened coordinates (a neutral slow
  factor) and
  the linear reference stays below 1. Neither gives a principled threshold, and growth that does not harm
  prediction over the consumed horizon is not invalidity. All are diagnostics.

## 11. Finite-horizon tangent diagnostics (never gates)

For a level map `z_{t+1} = G(z_t, b_t)`, `J_t = D_z G(z_t, b_t)` at fixed action. R2 computes `J_t` **exactly** from
the level's W: `y_k = Σ_i φ(u_ki)`, `u_ki = Σ_a x_a W[a, i]`, `φ(u) = u/2 + u²/2 + u³/2`, so
`∂y_k/∂z_j = Σ_i φ'(u_ki) W[j, i]`. It is tested against native FORWARD (value) and its centered differences. The
ordered product `P_h = J_{t+h−1} ⋯ J_{t+1} J_t` is the derivative of the h-step open-loop map (chain rule), and
`σ_max(P_h)`, its induced 2-norm, is the worst-case local amplification of a small state error over h steps. It is
measured on two paths that answer different questions and are never conflated:

- **teacher path**: `J_t` at the true held-out `z_t`, the local sensitivity around states of the real process;
- **free-running path**: `J_t` at the model's own predictions `ẑ_t` (same actions), the sensitivity on the state
  distribution the planner's model creates for itself; defined only for rollouts not escaped by step h − 1.

Reported per level and horizon: median, 90th percentile and fraction > 1 of `σ_max(P_h)` on each path, the median
`γ_h = log σ_max(P_h) / h`, the linear map's `σ_max(A^h)`, the generator's product, direct perturbation
amplification, and the median ratio of a first-order error recursion (`e_{k+1} = r_k + J_k e_k`, `r_k` the true
one-step residual) to the actual free-running error. The one-step spectral radius is reported descriptively only:
for non-normal Jacobians `ρ(J)` does not control `σ_max` of a product (`spectral_radius_does_not_control_...`:
`ρ(A) = ρ(B) = 0.5` yet `σ_max(BA) > 100`).

**Interpretation limits.** `γ_h` is a *finite-time tangent growth rate*, not a Lyapunov exponent. An exponent is
an asymptotic limit; the donor mathematics establishes it only with a measure-preserving map, integrability,
invariance and an almost-everywhere convergence argument (subadditive ergodic reasoning). None of these holds for a
learned, action-driven map over 16 steps. `σ_max` also depends on coordinates: a change of basis C moves
`log σ_max` by at most `log cond(C)`, which vanishes only as h → ∞, so every number is stated in the level's
whitened latent coordinates (the planner's). No claim of positive entropy, hyperbolicity, a Lyapunov spectrum or
chaos follows from any of these quantities.

**Donor.** github.com/openai/math at `fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`,
`lean/OAI/Dynamics/StandardMap/Lyapunov/{Definitions, DerivativeGrowth, ActualLyapunov, Subadditive}.lean`. Used for
mathematical structure only: derivative products, operator norms and their submultiplicativity, the coordinate
bound, and the separation of finite-time growth from asymptotic claims. No Standard Map conclusion is transferred.
The file digests are in `SOURCE_EQUIVALENCE.json`.

## 12. Hierarchy hypotheses (conditional on task validity, S1 and M in SEPARATED)

Requiring FLAT's 16-step model to be M-valid too means a hierarchy cannot be credited merely because the flat
planner's longer rollout is invalid ("fewer planning steps"). Success means over SEPARATED seeds at each compute
point:

- **H2A, distinct over width:** HD_L2 − FLAT_N72 ≥ 0.15 at ≥ 2 of 3 compute points.
- **H2B, distinct over temporal:** HD_L2 − TS_L2 ≥ 0.10 at ≥ 2 of 3.
- **H2C, temporal over width:** TS_L2 − FLAT_N72 ≥ 0.10 at ≥ 2 of 3.

**Hierarchy outcome** (only when adjudicated): `DISTINCT_ADVANTAGE` (H2A and H2B), else `TEMPORAL_ADVANTAGE`
(H2C), else `UNATTRIBUTED_ADVANTAGE` (H2A alone), else `NO_ADVANTAGE_OVER_WIDTH`.

**Credit rules.** TS has HD's exact temporal structure, so H2C is the temporal-decomposition effect and H2B the
representation effect beyond it. H2A alone is attributed to neither. A distinct-representation effect is credited
to the fixed closed-form encoder plus the ECS, never to abstraction learned by the ECS. Width (FLAT_N36 / N72 /
N108) is hypothesis-generating only. Depth (L2 → L3, ±0.05 at the middle compute point) is reported, not gated,
and the level-3 models' one-step plateau (0.951–0.965 on DESIGN, SEPARATED) is stated with it.

## 13. DESIGN record (DESIGN seeds only; no ECS multi-step quantity)

DESIGN ran twice, both times on DESIGN seeds only. The committed evidence is round 2 (source commit `b4b84ed`,
the frozen specification embedded).

- **Round 1** (provisional law: floor-based T3B, every-level 0.98 adequacy, capture ≥ 0.99 calibration margin, M
  with 75% of seeds). Findings: the floor-based T3B failed in 4 of 8 instances (TS/L3, sampling noise); one-step
  capture plateaued per instance (TS/L3 0.951, HD/L3 0.965, TS/L2 0.980, L1/N36 0.991, identical from 24000 to
  384000 steps), so the provisional calibration margin was unattainable at any budget. Both defects are in the
  provisional law, not the task. They led to the sample-matched T3B, the convergence calibration, S1, and the
  binding pairs.
- **Round 2** (the law above). Oracle success 1.0 in every instance. Level-1 probes ≤ 0.0006. Centroid ratio
  30.2–37.5 (SEPARATED) and 1.24–1.52 (MATCHED). T3A ≥ 0.4434 and T3B `L/G` ≤ 1.0787 at every binding pair. Clip
  interaction, persistence and generator captures, untrained and converged one-step capture, and convergence as
  quoted in sections 7–10.

DESIGN chose definitions and thresholds. DEV and QUAL cannot change them.

## 14. Dispositions

| Phase | Condition | Disposition |
|---|---|---|
| DEV calibration | no converged budget | `TASK_INVALID_ON_DEV` (stop) |
| DEV run | task validity fails | `TASK_INVALID_ON_DEV` (stop) |
| DEV run | task validity holds | `QUAL_AUTHORIZED` (S1, M, H2 non-binding) |
| QUAL | task validity fails | `TASK_INVALID` |
| QUAL | valid; S1 fails in SEPARATED | `ONE_STEP_MODEL_INVALID` (hierarchy not adjudicated) |
| QUAL | valid; S1 holds; M fails in SEPARATED | `WORLD_MODEL_INVALID` (hierarchy not adjudicated) |
| QUAL | valid; S1 and M hold in SEPARATED | `HIERARCHY_ADJUDICATED` with its hierarchy outcome |

`ONE_STEP_MODEL_INVALID` and `WORLD_MODEL_INVALID` are valid negative answers about the ECS world model, preserved
as such. A result "one-step valid but finite-horizon unusable" is a clean answer to the primary question.

## 15. Integration law

No disposition makes H-ECS canonical. Only QUAL `HIERARCHY_ADJUDICATED` with H2A **and** H2B authorizes an
**experimental**, explicitly noncanonical integration (a Rust hierarchy controller over the same K1 C ABI; no
Python scheduling, no change to `Runtime.run_turn`, no semantic codec, no HACF edge, no continuity use). Anything
else authorizes no integration.

## 16. What would not rescue a failure

No gate, threshold, world parameter, budget, grid, seed, horizon or condition changes after DEV data exist. The
calibration budget is the only DEV-time choice, and its rule is fixed above. A mechanics defect found after the
freeze is classified `MECHANICS_FAILURE` and repaired in the smallest mechanical layer, never in the scientific
object, its data or its law, and the repair is recorded with its reason. A defect that stops a phase before its
evidence is written is repaired, and that phase is run from its recorded seeds. A defect found in written evidence
is reported against that evidence and never rescued by a rerun (there is no second QUAL). Failures are reported
as the table in section 14 names them.
