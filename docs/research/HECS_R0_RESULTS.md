# H-ECS R0 results: `TASK_INVALID_ON_DEV`

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `NO_CANONICAL_PROMOTION`

Preregistration: [`research/hecs_r0/PREREGISTRATION.md`](../../research/hecs_r0/PREREGISTRATION.md).
Frozen specification: `research/hecs_r0/specs/hecs-r0.v1.spec.json` (commit `HECS-R0-A`).
Evidence: `research/hecs_r0/evidence/dev/hecs-r0.v1.calibration.json`.

## Disposition

**TASK_INVALID_ON_DEV.** The study stopped at its first preregistered step, DEV
calibration. No DEV run and no QUAL run took place, as the preregistration requires.
Nothing is claimed about hierarchy, selective abstraction, planning, width or depth:
those questions remain **open**, not answered negatively.

## What ran

DEV calibration (`hecs calibrate`, seeds 0-3, both worlds) trained FLAT_N36's level-1
K1 state at each grid budget. It then measured the held-out one-step NMSE: V1, the only
criterion calibration consults. The preregistered rule takes the smallest budget whose
NMSE is ≤ 0.25 with no divergence in every DEV world, and declares
`TASK_INVALID_ON_DEV` if none qualifies.

| Budget (K1 steps) | SEPARATED seeds 0 / 1 / 2 / 3 | MATCHED seeds 0 / 1 / 2 / 3 | Worst |
|---|---|---|---|
| 6000 | 0.2945 / 0.3164 / 0.1911 / 0.2456 | 0.0070 / 0.0123 / 0.0122 / 0.0094 | 0.3164 |
| 24000 | 0.2901 / 0.3117 / 0.1900 / 0.2445 | 0.0054 / 0.0090 / 0.0075 / 0.0070 | 0.3117 |
| 96000 | 0.2897 / 0.3114 / 0.1899 / 0.2447 | 0.0052 / 0.0083 / 0.0071 / 0.0069 | 0.3114 |

No budget meets V1, so no budget was chosen. No divergence and no K1 refusal occurred.
Calibration took 30 s of wall time.

## Diagnosis (post hoc; it changes nothing above)

The ECS is not the failing part: the gate was. Two observations support this.

1. **Converged.** Quadrupling the budget from 24,000 to 96,000 steps changes no value by
   more than 0.0007. The level-1 model is at its plateau.
2. **At the data's floor.** On the same DEV data, the best linear one-step predictor of
   the whitened latent from (latent, action) reaches NMSE 0.2895 / 0.3103 / 0.1895 /
   0.2423 (SEPARATED) and 0.0051 / 0.0080 / 0.0070 / 0.0069 (MATCHED). The ECS is within
   0.002 of that in every instance. This was computed with a throwaway program outside
   the frozen laboratory, not committed.

The cause is the world's construction:
- the SEPARATED second factor is AR(1) with ρ = −0.7;
- by design its unpredictable innovation is 1 − ρ² = 0.51 of its variance;
- whitening gives that factor unit variance, so the irreducible one-step error alone is
  about 0.25 of the two-coordinate average latent variance;
- sample variation in each world's mixing and held-out variance puts two instances above
  0.25.

V1 was written as an absolute threshold, 0.25. It should have been written relative to
the world's achievable error. As preregistered, the threshold fails the task in two
instances regardless of the model. That is a preregistration error, and it is reported
here as such.

## What R0 does not rescue

The preregistration forbids changing gates, thresholds, world parameters, budgets or
seeds after data exist. R0 therefore ends here. A successor (R1) would need:
- a **new** preregistration, frozen before any data;
- a V1 stated relative to each world's achievable floor, for example excess one-step
  NMSE over the best linear predictor ≤ 0.05;
- **fresh** DEV and QUAL seeds, because R0's DEV seeds are spent.

The R0 code, specification and evidence stay as written.

## Phase B report

| Field | Value |
|---|---|
| BASE_SHA | `a54e00536e306ef934e6b7f0e345e45abe694593` (main after PR #37, post-merge CI green) |
| HIERARCHY_SPEC | Declarative `HierarchySpec`: levels of (stride, window, width, horizon), ≤ 3 levels, strides 2-8, window ≤ stride; validated before use |
| LEVEL_STATE_AUTHORITY | One native K1 state `(W, epoch, H, a)` per level, created and driven through the existing C ABI; no shared `W`; a level sees only its lower level's latents and actions (mechanics test: learning in one level leaves another's digest and epoch unchanged) |
| STRIDES_WINDOWS | Level 1: stride 1, window 1. Upper levels: stride 4; window 1 (TEMPORAL_SHARED) or 2 (HIERARCHICAL_DISTINCT); past-only windows ending at the tick; bounded streaming controller, equal to the offline lift (mechanics test) |
| ENCODER_DEFINITION | Level 1: observation whitening. Distinct upper levels: affine window (4) → 2, the two most predictable unit-variance directions (closed-form generalized eigenproblem against a linear one-tick model); 12 parameters per boundary. Shared: none |
| FLAT_PARAMETER_BUDGET | `W`: 216 / 432 / 648 (N36 / N72 / N108); K1 image 30,344 / 32,072 / 33,800 bytes |
| TEMPORAL_SHARED_PARAMETER_BUDGET | `W`: 432 (L2) / 648 (L3), encoders 0; K1 images 60,688 / 91,032 bytes |
| HIERARCHICAL_DISTINCT_PARAMETER_BUDGET | `W`: 432 + 12 (L2) / 648 + 24 (L3); K1 images 60,688 / 91,032 bytes |
| COMPUTE_ACCOUNTING | Planner compute matched in native forward rows per replan, about {256, 1024, 4096} for every condition (three-level within 4%). Training multiply-adds counted per level. **Measured values: not run** |
| FAST_SLOW_WORLD_RESULTS | **Not run** (stopped at DEV calibration) |
| MATCHED_TIMESCALE_CONTROL_RESULTS | **Not run** |
| WIDTH_BASELINE_RESULTS | **Not run** |
| DEPTH_ABLATION_RESULTS | **Not run** |
| REPRESENTATION_PROBES | **Not run** |
| PLANNING_RESULTS | **Not run** |
| SCIENTIFIC_DISPOSITION | `TASK_INVALID_ON_DEV`: the preregistered V1 threshold lies below the SEPARATED world's achievable one-step error in two of four DEV instances. The ECS reached the best linear predictor's floor everywhere |
| NONCLAIMS | See below |

## Non-claims

- **Hierarchy.** Nothing is claimed about whether an ECS hierarchy differs from width
  scaling, abstracts selectively, plans better or benefits from depth. These were not
  measured.
- **Floor, not learning ability.** The one-step values show only that one level-1 K1
  state learned this synthetic linear-Gaussian dynamics to the best linear predictor's
  floor. That says nothing about learning in general.
- **Research only.** No canonical promotion, no runtime authority, no codec, no HACF
  edge and no language claim.
- **Not a JEPA.** ECS is not claimed to be a JEPA, and nothing here bears on H-JEPA's
  results.
