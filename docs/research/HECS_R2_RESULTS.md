# H-ECS R2 results: the ECS/K1 world model is not valid at the planners' consumed horizons in every fresh world

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

**Disposition: `WORLD_MODEL_INVALID`** (QUAL, once, under the frozen law). The hierarchy hypotheses were **not
adjudicated**. Integration is **NOT_AUTHORIZED** (`NO_CANONICAL_PROMOTION`). This is a result about the ECS/K1
constituent world model at the horizons the planners consume. It is **not** a hierarchy result.

Preregistration: [`research/hecs_r2/PREREGISTRATION.md`](../../research/hecs_r2/PREREGISTRATION.md), frozen in
commit `6f9cb6b` (HECS-R2-B) with its DESIGN record before any DEV observation. Specification:
`research/hecs_r2/specs/hecs-r2.v1.spec.json` (sha256 `2d942f162f938aaa58878afeb7745603f0994e5b1352edd074c4f2d0091935fa`).
Evidence (write-once): one compact record per phase, `research/hecs_r2/evidence/{design,dev,qual}/RECORD.json`.
Each attests the raw runner output by SHA-256 and byte count (raw output is telemetry and is not committed), names
the source the `hecs2` binary was built from (with the SHA-256 of every source file), the K1 library and `hecs2`
digests and the seeds, and carries the decisive numbers from which `tests/research/hecs_r2` recomputes each
disposition. Every number below is in those records.

| Phase | Commit | Built from (lineage) | Raw output (sha256, bytes; not committed) | Disposition |
|---|---|---|---|---|
| Freeze (DESIGN authority) | `6f9cb6b` | `b4b84ed` | `hecs-r2.v1.design.json` `fb786f4f…95bd`, 457980 | — |
| DEV calibration | `cd5e587` | `aa918b3` (pre-cleanup) | `hecs-r2.v1.calibration.json` `db4f375d…0c00`, 80322 | `CALIBRATED`, 96000 K1 steps |
| DEV run | `cd5e587` | `aa918b3` (pre-cleanup) | `hecs-r2.v1.dev.json` `5dac78e9…35b8`, 1409992 | `QUAL_AUTHORIZED` |
| QUAL run (once) | `77b69bc` | `c1d38d2` (pre-cleanup) | `hecs-r2.v1.qual.json` `53776376…d9ac`, 2690878 | `WORLD_MODEL_INVALID` |

The phases first landed as `05be551`, `aa918b3`/`c1d38d2` and `82bb08a`, a lineage that committed the raw output.
It was rewritten as the commits above so that no raw telemetry stays in Git. No phase was rerun. The `hecs2`
sources at `aa918b3` and `c1d38d2` are byte-identical to the final ones (per-file SHA-256 in each record), and a
rebuild from the final sources reproduces the executed `hecs2` binary (`6130aff5…`).

R0 (`TASK_INVALID_ON_DEV`) and R1 (`TASK_INVALID`) remain closed and byte-for-byte unchanged
(`tests/research/hecs_r2` pins their preregistrations, specifications and evidence).

## The question and the order in which it was answered

> Can the qualified ECS/K1 world model, trained one step ahead, remain a useful predictive object over the
> recursive horizons the H-ECS planners actually consume?

The frozen order is task validity → one-step constituent validity (S1) → multi-step validity at every consumed
horizon (M) → planning-task validity (V2) → hierarchy (H2). R2 answers the first three and stops at M.

## Results by stage

| Field | Result |
|---|---|
| **TASK_VALIDITY** | **VALID** in every DEV and QUAL instance. T1A/T1B (level 1, one step); T3A at every binding pair (explained ≥ 0.320 SEPARATED, ≥ 0.865 MATCHED in QUAL); T3B, the iterated linear reference against the measured clipping-aware generator on the same targets (`L_h/G_h` ≤ 1.032 SEPARATED, ≤ 1.162 MATCHED; bound 1.20); V2 oracle success 1.0 in both worlds; V3 level-1 probes ≤ 0.0009; V4 centroid ratio 34.85 (SEPARATED) / 1.47 (MATCHED); V5 no K1 refusal. |
| **CALIBRATION_RESULT** | `CALIBRATED`. Under the frozen convergence rule, the largest binding-level one-step capture improvement was 0.0323 from 6000 to 24000 steps and 0.0022 from 24000 to 96000 (≤ 0.005). B* = 24000; the chosen budget is one grid step above: **96000**. No grid point, tolerance or rule changed. |
| **ONE_STEP_VALIDITY** | **S1 holds for every seed in both worlds** (QUAL binding minimum: SEPARATED 0.9765 at TS/L2, MATCHED 0.9891 at HD/L2; S1 bound 0.95). The constituent models learn the one-step map about as well as the linear reference does. Non-binding: TS/L3 (0.9266) and HD/L3 (0.7919) fall below 0.95 in some SEPARATED seeds. |
| **MULTISTEP_VALIDITY** | **M fails in SEPARATED** (the informative world). 6 of 8 seeds meet every bound; the frozen law requires all 8. |
| **PLANNING_TASK_VALIDITY** | Valid (V2: the oracle solved every episode in both worlds). The planning comparison was not adjudicated because M failed. |
| **SEPARATED_M_RESULT** | **FAIL.** Seed `2761248050979241786`: HD/L2 free-running capture 0.894 at its consumed horizon h = 4. Seed `17712258514216225143`: L1/N72 (FLAT_N72) capture 0.884 at h = 16. Every other seed meets every bound at every binding pair and horizon. Minimum capture per binding pair: L1/N72 (h ≤ 16) 0.884; L1/N36 (h ≤ 4) 0.994; TS/L2 (h ≤ 4) 0.972; HD/L2 (h ≤ 4) 0.894. Numerical escape is 0 everywhere. |
| **MATCHED_M_RESULT** | Nominally holds for every seed (binding minimum 0.924, HD/L2). |
| **M_informative_MATCHED** | **false.** The trivial persistence predictor reaches capture 0.997 at MATCHED's binding pairs, so MATCHED M is **not** evidence that the learned model is useful (DESIGN anticipated this; MATCHED is a control). In SEPARATED, persistence reaches at most 0.610, below M's 0.90, so M is informative there. |
| **TANGENT_GROWTH_RESULT** | Diagnostic only; see below. The two M failures have opposite tangent signatures: **expansion** at the HD/L2 failure and **over-contraction** at the L1/N72 failure. |
| **DOMAIN_EXCURSION_RESULT** | Descriptive only. At the binding pairs in SEPARATED, the level-1 and TS/L2 rollouts never left their training envelope (the truth itself left it in up to 3.4% of windows). HD/L2 rollouts left it in 0.2–15.8% of windows (the truth: 0.8–3.3%), including 15.8% at the failing-adjacent seed `2136955805524040833` (capture 0.907) and 2.0% at the failing seed. Non-binding horizons go further (up to 50% at HD/L3, h = 16, on very few starts). |
| **NUMERICAL_ESCAPE_RESULT** | **0 at every binding pair and every horizon ≤ its consumed horizon, in every seed of both worlds.** Beyond the consumed horizons (descriptive): HD/L2 at h = 16 escaped in up to 14.8% of SEPARATED rollouts. |
| **HIERARCHY_HYPOTHESES_ADJUDICATED** | **NO.** |
| **FLAT_WIDTH_RESULT** | Not adjudicated (width was hypothesis-generating only). Descriptive QUAL planning success, SEPARATED, middle compute point: FLAT_N36 0.68, FLAT_N72 0.72, FLAT_N108 0.76. M per FLAT condition (descriptive): N36 fails, N72 fails (the seed above), N108 holds. |
| **TEMPORAL_SHARED_RESULT** | Not adjudicated. TEMPORAL_SHARED_L2's own pairs met M in every SEPARATED seed. Descriptive success 0.62 / 0.63 / 0.67 at the three compute points (H2C's TS − FLAT difference was negative: −0.09 / −0.09 / −0.04). |
| **HIERARCHICAL_DISTINCT_RESULT** | Not adjudicated. HIERARCHICAL_DISTINCT_L2's distinct level failed M (seed above). Descriptive success 0.01 / 0.22 / 0.27. Any abstraction its fixed closed-form encoder introduces is the encoder's, not the ECS's. |
| **DEPTH_RESULT** | Not adjudicated (descriptive): L1 → L2 → L3 success at the middle compute point fell for both representations (TS 0.68 → 0.63 → 0.25; HD 0.68 → 0.22 → 0.08). The level-3 models fail S1 in some SEPARATED seeds, so these numbers do not isolate depth. |
| **INTEGRATION_AUTHORIZED** | **NO** (only `HIERARCHY_ADJUDICATED` with H2A and H2B authorizes an experimental integration). |

The planning numbers above are recorded by the frozen measurement and are reported for completeness. Because M
failed, no hierarchy, width or depth conclusion follows from them, positive or negative.

## What the diagnostics say (descriptive, not gates, not theorems)

All quantities are in each level's whitened latent coordinates (the planner's). `σ_max(P_h)` is the induced
2-norm of the ordered derivative product of the level map along a path, computed from exact state Jacobians.
`γ_h = log σ_max(P_h) / h` is a **finite-time tangent growth rate**, not a Lyapunov exponent.

- **One step is not the problem.** At both failing pairs the one-step ECS error matches the one-step linear
  reference (HD/L2, seed `2761…`: 0.3413 vs 0.3440, capture 1.004; L1/N72, seed `1771…`: 0.2305 vs 0.2268,
  capture 0.995). The degradation appears only when the model is fed its own predictions, exactly the separation
  the rollout-error recurrence (preregistration section 6) describes.
- **HD/L2, seed `2761248050979241786` (capture 0.894): expansion.** Teacher-path `σ_max(P_4)` median 1.49 (p90
  2.05; 70% of starts > 1; `γ_4` = +0.10); free-path median 1.16 (58% > 1). The linear reference contracts
  (`σ_max` 0.95). The free-running error (0.406) exceeds the linear reference's (0.336) while nothing escapes:
  local expansion amplifies small one-step residuals over four steps.
- **L1/N72, seed `17712258514216225143` (capture 0.884): over-contraction.** `σ_max(P_16)` median 0.60 on both
  paths (no start > 1; `γ_16` = −0.032; direct perturbation amplification 0.39), against the true process's
  neutral 1.000 and the linear reference's 0.93. The model damps the slow, marginally stable factor that a 16-step
  plan must carry. Its free-running error (0.506) exceeds the linear reference's (0.437). A *more* stable map
  failed here: stability in the tangent sense is not validity.
- **Agreement with the first-order picture.** Propagating the true one-step residuals along the model's own
  free-path Jacobians (`e_{k+1} = r_k + J_k e_k`) reproduces the actual free-running error at the consumed
  horizons to within about 1.5% (median ratio 0.986–1.006 over every SEPARATED binding pair and seed). The errors stay in a locally linear regime. That is
  descriptive agreement, not a certified bound: the measured Jacobians are local, not Lyapunov-certified
  Lipschitz constants.
- **The binding pairs that met M in every seed** (L1/N36 and TS/L2, at h = 4) contract mildly (median
  `σ_max(P_4)` 0.96–0.99 on both paths) and keep capture ≥ 0.97.
- **DEV (non-binding)** showed the same pattern more sharply. HD/L2's free-running prediction collapsed at h = 4
  in three of four SEPARATED seeds (capture −0.83, −2.66, 0.49) with expanding tangents (mean one-step Jacobian
  spectral radius 1.09–1.10 in those seeds, 0.97 in the fourth), and one MATCHED L1/N72 rollout set degraded at
  h = 16 (capture 0.15). DEV values did not change
  anything: the law was frozen before them.

## Scientific claims (conservative)

1. Under the frozen synthetic regime and training recipe (R0/R1's), the ECS/K1 level models are valid **one step**
   ahead: S1 held in every QUAL instance of both worlds.
2. They are **not** valid at the horizons the H-ECS planners consume **in every fresh world**. In SEPARATED, 2 of 8
   QUAL worlds fell below the frozen free-running capture bound (0.90) at a binding pair: once for the flat
   16-step model (0.884) and once for the distinct-hierarchy level at 4 steps (0.894). Hence
   `WORLD_MODEL_INVALID`: the constituent world models did not earn a hierarchy adjudication.
3. The failures are near the bound, not catastrophic, inside the consumed horizons: no numerical escape, and 6
   of 8 worlds passed every bound. Beyond the consumed horizons the distinct level degrades sharply.
4. Excellent one-step prediction did not imply planner-usable recursion, in both directions of tangent behaviour
   (expansion and over-contraction).

## Nonclaims

- **No hierarchy claim**, positive or negative. H2A, H2B and H2C were not adjudicated; the planning success
  numbers support no width, temporal, representation or depth conclusion. `WORLD_MODEL_INVALID` is not
  `HIERARCHY_FAIL`.
- **No claim that the ECS learned an abstraction.** HIERARCHICAL_DISTINCT's encoder is a fixed closed-form linear
  map.
- **No usefulness claim from MATCHED.** M there is nominal: persistence already reaches it.
- **No dynamical-systems claim.** `γ_h` is a finite-time tangent growth rate over at most 16 steps. Nothing here
  is a Lyapunov exponent or spectrum, positive entropy, hyperbolicity, chaos or asymptotic instability. The
  mathematical donor (openai/math `fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`, Standard Map Lyapunov files) supplied
  the finite-horizon derivative-product object and the conditions an asymptotic claim would need. No Standard Map
  conclusion is transferred to ECS.
- **No theorem about the ECS.** The rollout-error recurrence is generic mathematics. The empirical tangent
  products are local measurements, not certified Lipschitz bounds.
- **No generality.** One synthetic action-conditioned 2-D world family, one training recipe and one grid. Nothing
  extends to language, DSV, HACF or the canonical runtime. ECS is not claimed to be a JEPA.
- **No runtime authority.** H-ECS is not canonical, nothing in `src/` uses it, and no integration was authorized.
- R0 and R1 are unchanged; nothing here reinterprets their dispositions.

## Open problems

- Why the distinct level expands in some worlds (DEV: severely; QUAL: one world) while the temporal-shared level,
  with the same strides and horizons, contracts. Candidates: the fixed encoder's predictable-components basis, the
  pooled-action input at that level, and cubic-map curvature on a learned window encoding. None is tested here.
- Why the flat 16-step model can over-damp the marginally stable slow factor while matching the linear one-step
  error. One-step teacher-forced training does not constrain the tangent of the neutral direction.
- Whether multi-step (rollout) training objectives, tangent-aware regularisation, or a different budget/grid would
  change M. Each would be a new preregistered lineage, not a rescue of R2.

## Next scientific step

A successor experiment (a new lineage, not R2) that keeps R2's validity contract (task → S1 → M at every consumed
horizon → planning → hierarchy) and tests one preregistered intervention on the **constituent model's recursive
validity**. For example, a training objective that includes free-running multi-step error, or one that constrains
the neutral-direction tangent. Hierarchy should be adjudicated only if M then holds in every fresh world.
