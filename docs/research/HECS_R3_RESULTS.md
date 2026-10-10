# H-ECS R3 results: a qualified constituent world-model law; the hierarchy hypothesis is adjudicated negative

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_RUNTIME_READINESS_CLAIM` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

| Claim | Disposition |
|---|---|
| Constituent world-model learning law (C2/C2M) | **QUALIFIED UNDER THE FROZEN REGIME** (`WORLD_MODEL_VALID`) |
| Hierarchy hypothesis (R2 H2 law over matched constituents) | **ADJUDICATED NEGATIVE** (`NO_ADVANTAGE_OVER_WIDTH`) |
| Hierarchy integration | **NOT AUTHORIZED** |
| Runtime readiness | **NOT CLAIMED** |

No H-ECS hierarchical runtime is canonical. Nothing in this lineage is on the canonical runtime path.

This document summarizes the frozen evidence under `research/hecs_r3/`. That evidence is the authority; this summary does not modify it. Every number below is in the committed records named in each section.

## The question

R2 (`HECS_R2_RESULTS.md`) closed `WORLD_MODEL_INVALID`: the ECS/K1 world model, trained one step ahead, was not valid at every consumed horizon in every fresh world. R3 therefore asked a constituent question first and the hierarchy question second:

1. **Constituent.** Is there a learning law under which every constituent the H2 planners consume passes the frozen validity hierarchy?
   - Validity hierarchy: TASK → S1 ≥ 0.95 → M capture ≥ 0.90 and escape ≤ 0.01.
   - Scope: every seed, binding and bound horizon, on fresh DEV and then fresh QUAL.
   - Bindings: L1/N72 h16, L1/N36 h4, TS/L2 h4, HD/L2 h4.
2. **Hierarchy.** If so, what does the unchanged R2 H2 law adjudicate?

## Lineage

| Stage | Record | Commit(s) | Disposition |
|---|---|---|---|
| Successor DEV, C0 (historical R3 rollout) vs C1 | `research/hecs_r3/unresolved/successor_dev_3_of_8/` | `eb5be79`..`6776cd4` | DEV incomplete at 3/8; both fail M at seed 2, HD/L2 h4 |
| Successor DEV closure 8/8 | `research/hecs_r3/closure/successor_dev_8_of_8/` | `d26c78b` | C0 and C1 both `WORLD_MODEL_INVALID_ON_DEV` (one cell: seed 2, HD/L2 h4) |
| C2 freeze (true-target terminal anchor at HD/L2 only) | `research/hecs_r3/successor_c2/freeze/` | `8a98d9c` | pre-registered |
| C2 fresh DEV | `.../successor_c2/dev/` | `b64e32f` | 8/8, `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE` |
| C2 QUAL Q1 | `.../successor_c2/qual_q1_incomplete/` | `24fed33` | infrastructure incident (container restart); mechanically incomplete, no scientific verdict; all 12 seeds retired |
| C2-Q2 freeze + correction R1 | `.../successor_c2/q2_freeze/` | `39e8c33`, `39ea422` | fresh 12-seed registry; candidate unchanged |
| C2-Q2 QUAL | `.../successor_c2/qual_q2/` | `3d8b793` | 12/12 `WORLD_MODEL_VALID` |
| C2M freeze (matched family) | `research/hecs_r3/successor_c2m/freeze/` | `c14318e` | pre-registered |
| C2M fresh DEV | `.../successor_c2m/dev/` | `b89c474` | 8/8, QUAL eligible; H2 non-binding |
| C2M fresh QUAL | `.../successor_c2m/qual/` | `93545ca` | 12/12 `WORLD_MODEL_VALID`; `HIERARCHY_ADJUDICATED`; `NO_ADVANTAGE_OVER_WIDTH` |

Every phase ran once under a write-once supervisor (START/COMMIT chain, FAULT with `REPLAY_PERMITTED=NO`). Raw telemetry is not in Git; each `COMMIT.txt` binds its SHA-256.

## 1. The constituent learning law is qualified

**C2.** At HD/L2, the ROLLOUT arm is trained from the native G1 W:
- 4096 updates;
- six C0 window picks per update;
- the unchanged R3 rollout objective (λ = 0.37);
- plus a true-target terminal anchor `w_T · 0.5‖ẑ_h − z_h‖²/(LATENT·var)`, w_T = 1.0;
- exact reverse-mode gradient, unchanged rate and clip.

It removed the HD/L2 h4 lower-tail failure:

| Phase | Result | Min bound M capture (all bindings) | Max escape |
|---|---|---|---|
| DEV | 8/8 | 0.9830 | 0 |
| QUAL (C2-Q2) | 12/12 | 0.9742 | 0 |

In C2 only HD/L2 used this law. The other bindings kept the R3 paired law.

**C2M (matched).** The same law is applied to **every** compared constituent at its consumed horizon, with terminal weight `w_T(h) = (4/h)^1.5`:
- 1.0 at h4 (exactly C2);
- 0.125 at h16.

DESIGN showed that the unchanged weight fails at h16: 38/40 worlds, min M16 0.8843, from clip-saturating terminal gradients. The exponent was chosen by a rule declared before the grid ran, and confirmed on 48 held-out DESIGN worlds at every binding.

| Phase | Result | L1/N72 h16 | L1/N36 | TS/L2 | HD/L2 | Escape |
|---|---|---|---|---|---|---|
| DEV | 8/8, 32/32 gates | 0.9941 | 0.9957 | 0.9807 | 0.9764 | 0 |
| QUAL | 12/12, 48/48 gates | 0.9958 | 0.9948 | 0.9760 | 0.9661 | 0 |

The values are minimum bound ROLLOUT M capture over seeds. The minimum S1 on QUAL is 0.9767 (TS/L2).

**Limitation.** The true-target terminal anchor uses true future latents during training. C2/C2M is a **learning law from observed trajectories** (offline LEARN). It is not a query-time method for building a world model from novel context. One scientific seed (96,000 G1 steps and 4096 C2 updates per binding) is an independent qualification replicate, never per-turn work (`research/hecs_r3/successor_c2/ARCHITECTURE_CONSTRAINTS.md`).

## 2. The hierarchy hypothesis is adjudicated negative

Every constituent the H2 comparison consumes is a qualified world model trained by the same learning law. The R2 H2 comparison was therefore **admissible**, and it is attributable to hierarchy rather than to unequal training.

The comparison used the unchanged R2 H2 law (`hecs_r2 spec.rs` thresholds, `experiment::evaluate` arithmetic). Planning is QUERY-only native forward computation over the materialized trained weights, ported verbatim from R2 and verified bitwise against `hecs_r2::experiment::run_seed`.

Mean SEPARATED planning success over the 12 QUAL seeds, at matched planning compute (256 / 1024 / 4096 forward rows per replan):

| Condition | b0 | b1 | b2 |
|---|---|---|---|
| FLAT_N72 (L1/N72) | 0.854 | 0.837 | 0.840 |
| TEMPORAL_SHARED_L2 | 0.646 | 0.628 | 0.698 |
| HIERARCHICAL_DISTINCT_L2 | 0.017 | 0.132 | 0.264 |

| Comparison | Requirement | Result |
|---|---|---|
| H2A (HD − FLAT) | ≥ 0.15 | fails at every budget |
| H2B (HD − TS) | ≥ 0.10 | fails at every budget |
| H2C (TS − FLAT) | ≥ 0.10 | fails at every budget |

HD exceeded FLAT in 0 of 36 seed × budget cells.

**Outcome: `NO_ADVANTAGE_OVER_WIDTH`.** The flat width-72 model substantially outperforms both temporal hierarchies under the frozen planning-compute comparison. The distinct-latent hierarchy is the worst at every budget. The direction was disclosed in the C2M protocol from DESIGN and from R2's own planning, and it was never a selection input.

## What this establishes and what it does not

| Statement | Status |
|---|---|
| Constituent world-model law | `QUALIFIED_UNDER_FROZEN_REGIME` |
| Hierarchy hypothesis | `ADJUDICATED_NEGATIVE` |
| Hierarchy integration | `NOT_AUTHORIZED` |

Under R2, integration requires `DISTINCT_ADVANTAGE`. It is not authorized, and the H-ECS persistent-runtime milestone, which was conditional on the hierarchy passing, is not opened.

The following are **not** established, and must not be claimed:
- "H-ECS qualified";
- a hierarchy qualified for integration;
- an EDEN or any other canonical hierarchy;
- runtime readiness;
- a language or semantic claim.

The C2/C2M models are synthetic-world research objects. They are not wired into `src/elpis/runtime` or `native/runtime`.

No further R3 scientific execution is authorized by this record. Any future hierarchy work would be a new, separately pre-registered family with fresh registries.
