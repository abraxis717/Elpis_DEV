# H-ECS R1 results: `TASK_INVALID` (QUAL)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `NO_CANONICAL_PROMOTION`

Preregistration: [`research/hecs_r1/PREREGISTRATION.md`](../../research/hecs_r1/PREREGISTRATION.md), frozen in
commit `5c57dec` (HECS-R1-A) before any DEV observation. Specification: `research/hecs_r1/specs/hecs-r1.v1.spec.json`.
Evidence (write-once): `research/hecs_r1/evidence/{design,dev,qual}/`.

## Disposition

| Phase | Result |
|---|---|
| R0 (closed, unchanged) | `TASK_INVALID_ON_DEV` |
| R1 DESIGN (DESIGN seeds) | informed the V1 margins; found two inherited flaws (V4 MATCHED bound, endpoint planner cost), corrected before freeze |
| R1 DEV calibration | `CALIBRATED` at 6000 K1 steps (V1 A, B, C hold in all 8 DEV instances) |
| R1 DEV run (6000) | valid: V1–V5 hold. Hypothesis values non-binding |
| R1 QUAL run (6000, once, 8 fresh seeds) | **`TASK_INVALID`**: V1C `ECS_ADEQUACY` fails in 1 of 16 instances |
| Hierarchy pipeline integration | **`NOT_AUTHORIZED`** (only QUAL `OUTCOME_A` authorizes it) |

The QUAL disposition is final. The hypotheses are **not adjudicated**: the preregistered law forbids drawing them
from an invalid QUAL run, and nothing is re-run, re-tuned or rescued.

## Why QUAL is invalid

In QUAL instance MATCHED / QUAL seed index 4 (`15236446594267265809`), the FLAT_N36 level-1 ECS removed 0.9785
of the one-step error that the held-out linear reference removes from the constant baseline. V1C requires
0.98. Its one-step NMSE was 0.0290 against a linear reference of 0.0063 (analytic floor 0.0063). Every other
instance passed A, B and C, and V2–V5 held (oracle success 1.0 / 1.0, centroid ratio 31.1 / 1.51, no refusal).

The law worked as designed. Unlike R0's absolute gate, which failed at the world's irreducible floor, V1B
confirmed the world sits exactly at its generator floor (|L − F| ≤ 0.0004 in MATCHED). V1C then detected a level-1 ECS
that, at the smallest budget DEV calibration admitted (6000 steps), had not learned the predictable structure in
one fresh instance. The calibration rule chose the minimum passing budget with no safety margin; DESIGN and DEV
minimum captures at 6000 steps were 0.994 and 0.9935, so a fresh instance below 0.98 was possible and occurred. This is
recorded as the cause, not repaired.

## Task-validity diagnostics (FLAT_N36, level 1)

| | DESIGN (4 + 4 inst., 6000) | DEV (4 + 4) | QUAL (8 + 8) |
|---|---|---|---|
| Analytic floor F, SEPARATED | 0.215–0.287 | 0.210–0.262 | 0.188–0.301 |
| Linear reference L, SEPARATED | 0.227–0.291 | 0.212–0.255 | 0.186–0.283 |
| ECS E, SEPARATED | 0.231–0.294 | 0.218–0.259 | 0.191–0.296 |
| Analytic floor F, MATCHED | 0.0081–0.0089 | 0.0041–0.0077 | 0.0053–0.0097 |
| Linear reference L, MATCHED | 0.0075–0.0090 | 0.0041–0.0076 | 0.0052–0.0101 |
| ECS E, MATCHED | 0.0094–0.0117 | 0.0060–0.0133 | 0.0075–0.0290 |
| V1A explained by L | 0.71–0.99 | ≥ 0.75 | ≥ 0.72 |
| V1C ECS capture (min) | 0.994 | 0.9935 | **0.9785** (one instance) |

`ANALYTIC_ERROR_FLOORS` and `LINEAR_REFERENCE_RESULTS` agree everywhere within V1B. The SEPARATED world's
irreducible one-step error is about 0.19–0.30 of the held-out latent variance: R0's 0.25 threshold sat in the
middle of it.

## Non-binding observations (DEV and QUAL; not adjudicated)

Means over seeds. Probes are held-out ridge NMSE to the ground-truth factor at each level (0 = exact, 1 = no
information). Planning success at the three compute points (≈ 256 / 1024 / 4096 forward rows per replan).

**Representation (SEPARATED).**

| Condition | Slow probe by level | Fast probe by level |
|---|---|---|
| FLAT_N36/72/108 | 0.0002–0.0003 | 0.0004 |
| TEMPORAL_SHARED_L2/L3 | 0.0002–0.0003 at every level | 0.0004 at every level |
| HIERARCHICAL_DISTINCT_L2 | 0.0002 → 0.0003 | 0.0004 → **1.00** |
| HIERARCHICAL_DISTINCT_L3 | 0.0002 → 0.0003 → 0.0004 | 0.0004 → **1.00** → 1.03 |

In MATCHED the distinct levels keep the second factor (fast probe at L2 0.008 DEV, 0.013 QUAL). The selective-
abstraction pattern H1/H1C/H1T describe appears in both DEV and QUAL, but it is a property of the **closed-form
linear predictable-components encoder** fixed before any ECS training. In SEPARATED the fast factor is
unpredictable over a stride of 4, so the encoder drops it. The ECS levels did not learn it. The temporal
hierarchy without a distinct representation abstracts nothing (by construction it shares level 1's space).

**Planning (SEPARATED; QUAL / DEV).**

| Condition | 256 | 1024 | 4096 |
|---|---|---|---|
| FLAT_N36 | 0.078 / 0.062 | 0.135 / 0.146 | 0.161 / 0.083 |
| FLAT_N72 | 0.016 / 0.021 | 0.078 / 0.125 | 0.078 / 0.250 |
| FLAT_N108 | 0.000 / 0.000 | 0.052 / 0.021 | 0.089 / 0.031 |
| TEMPORAL_SHARED_L2 | 0.125 / 0.167 | 0.208 / 0.156 | 0.224 / 0.146 |
| TEMPORAL_SHARED_L3 | 0.068 / 0.062 | 0.104 / 0.146 | 0.161 / 0.125 |
| HIERARCHICAL_DISTINCT_L2 | 0.016 / 0.021 | 0.005 / 0.010 | 0.005 / 0.031 |
| HIERARCHICAL_DISTINCT_L3 | 0.026 / 0.042 | 0.000 / 0.021 | 0.010 / 0.000 |
| Oracle (true slow dynamics) | 1.000 / 0.990 | | |

- **Every learned planner is far below the oracle** (≤ 0.25 vs 1.0). The level ECS models are accurate one step
  ahead but their open-loop rollouts blow up by horizon 8: the horizon-8 NMSE is astronomically large at most
  levels, and the attractor census shows 14–78% of free runs diverging. The planners therefore fail mostly
  because the models are unusable at the rollout horizons planning needs, not because of the hierarchy.
- **Width** (raw width benefit): wider single ECS states plan worse (N36 ≥ N72 ≥ N108 at most points) and
  have larger multi-step errors.
- **Temporal decomposition**: TEMPORAL_SHARED_L2 is the best learned planner in QUAL at every compute point.
- **Distinct representation**: HIERARCHICAL_DISTINCT plans worst. Its level-1 subgoal tracking error is larger
  (normalized squared distance 1.2–1.4, against 0.5 for the temporal hierarchy). A subgoal in a space without
  the fast factor is matched by a level-1 planner whose goal distance still counts it.
- **Depth**: TEMPORAL_SHARED L1 → L2 IMPROVES in QUAL and is NO_CHANGE in DEV; L2 → L3 DEGRADES in QUAL.
  HIERARCHICAL_DISTINCT L1 → L2 DEGRADES in both.

None of these is a scientific claim. They describe one invalid QUAL run and its valid DEV run.

## Compute accounting

Matched as preregistered. Native forward rows per replan: FLAT ≈ 251–255 / 1002–1021 / 4019–4087 (divergent
candidates stop early); two levels exactly 256 / 1024 / 4096; three levels 264 / 1032 / 4104 (within the 4%
tolerance). Training multiply-adds per condition: 6.6e8 (N36), 1.33e9 (N72 and every two-level hierarchy), 1.99e9
(N108 and every three-level hierarchy). `W` parameters 216 / 432 / 648; distinct encoders add 12 per boundary.
K1 image bytes 30,344 / 60,688 / 91,032 for 1 / 2 / 3 levels of width 36 (FLAT_N72 32,072; FLAT_N108 33,800).
Controller resident bytes 24 (flat) to 136 (HD_L3), constant per episode.

## Evidence provenance

The DEV and QUAL run files record each seed with the laboratory's integer JSON type, a signed 64-bit value
(`seed as i64`). Seeds at or above 2^63 therefore appear negative, for example `-2260723101071832941` for DEV
seed `16186020972637718675`. The computation used the unsigned seed; only the record is affected, and the exact
seed is the recorded value modulo 2^64 (`tests/research/hecs_r1` decodes and checks it). The calibration and
DESIGN files record seeds as decimal strings. Classified `EVIDENCE_PROVENANCE_DEFECT`; the write-once evidence is
not rewritten.

## Phase B report fields

| Field | Value |
|---|---|
| HECS_R0_DISPOSITION | `TASK_INVALID_ON_DEV` (unchanged) |
| HECS_R1_PREREGISTRATION_SHA | `5c57dec` |
| HECS_R1_DEV_DISPOSITION | valid (calibrated 6000; V1–V5 hold); non-binding pattern OUTCOME_B |
| HECS_R1_QUAL_DISPOSITION | `TASK_INVALID` (V1C, 1 of 16 instances) |
| HIERARCHY_PIPELINE_INTEGRATION | `NOT_AUTHORIZED` |

## What R1 does not rescue, and what a successor would need

R1 ends here. A successor (R2) would need a new preregistration, frozen before any data and with fresh seeds. Its
calibration rule should choose a budget with a margin on V1C (for example the smallest budget whose minimum DEV
capture is at least 0.99, or one grid step above the smallest passing budget). Given what R1 observed it should
also reconsider whether one-step-trained cubic ECS world models can serve multi-step planning at all. That second
question is the larger obstacle to any planning claim. Neither change may be applied to R1.

## Non-claims

- **Hierarchy.** Nothing is claimed about whether an ECS hierarchy differs from width scaling, abstracts
  selectively, plans better or benefits from depth. QUAL was invalid and nothing was adjudicated.
- **Abstraction.** The observed loss of the fast factor at distinct upper levels comes from a fixed linear
  encoder chosen by a closed-form predictability criterion, not from ECS learning.
- **Planning.** Low learned-planner success reflects the models' open-loop instability under this protocol. It
  is not evidence against hierarchy, and it is not evidence for it.
- **Research only.** No canonical promotion, no runtime authority, no codec, no HACF edge, no continuity use and
  no language claim. ECS is not a JEPA; nothing here bears on H-JEPA's results.
