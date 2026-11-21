# Elpis Torch R12 — amendment P5: the representation seam (diagnosis only)

**Status.** FROZEN before any P5 data.
- **The plan.** `research/elpis_torch/results/r12/amendment_p5.json`, SHA-256 `e655c3fc0f46063b4a5c42622276004539628deac04cd212863307203705fdb7`.
- **The guard.** `run_r12_p5.frozen_p5()` refuses to run unless all of these hold:
  - the amendment names the R12 study that triggered it, by digest;
  - that study's reading is `NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED`;
  - the R12 plan is unchanged;
  - the Stage B plan equals R11's frozen plan.
- **Provenance.** Every P5 record carries the amendment's digest and the digest of every source it ran. P5's
  `combine` refuses any record whose digests differ.

**Why it exists.** R12's frozen reading was `NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED` (`EVIDENCE_R12.md` §6):
no arm qualified in either regime. `PREREGISTRATION_R12.md` §7 then allows the representation stage of directive
P5, but only under its own amendment, frozen before its data.

**What it can and cannot do.**
- **Diagnosis only.** P5 qualifies nothing. It changes no R12 record, rule or disposition.
- **The oracle stays in the lab.** Its Stage B condition reads the exact task coordinates, an oracle by
  construction. That condition is never a claimed inference path.

## 1. Questions

- **Directive P5.** Does the last block's 120-parameter specialist receive the information its adaptation protocol
  needs?
- **Directive P8.** Is the failure explained by initialization, task-feature access, routing, or a demonstrated
  absence of benefit?

## 2. Conditions

Every comparison pairs the same (family, seed, episode). P5 runs on the R12 records' own evaluation episodes: T1's
groups (R11C, seeds 301–306, split `R12TEST`) and T2's groups (R12C, seeds 401–406, split `TEST`), 12 episodes each.

| Condition | What it is | Directive P5 item |
|---|---|---|
| **Stage B** (`stage_b`) | R11 Stage B's procedure, unchanged: `run_r11.train_stage_b` with R11's frozen plan and the `BOUNDED_SPECIALIST` profile, then `specialist_tasks.evaluate_arm`. It is trained on the group's own tasks, so the Stage B and R12 episodes are the same ones. The specialist reads `u = Q_task x + Q_nuisance ξ`: the exact coordinates through a fixed orthonormal embedding, with unit nuisance and no context. Arms: G0 (FFN alone), G2 (CGC, `g = 1`) and G3 (matched GELU) | R11 Stage B's task-coordinate access; the controlled oracle feature exposure |
| **Stack** (from the R12 records) | The R12 G2 `fast8` with the support in context, and the R12 G0 frozen at k = 32 | The actual Stage C projected representation |
| **Rebuilt** (`stack`, T2) | T2's G2 retrained by `train_fast` exactly as in R12, and verified against the T2 record: rows identical, selected step and DEV trace equal | (integrity) |
| **Fixed projection** (`stack`, T2) | The same training with the last block's `P_in` never learning: it keeps its initial value, and nothing else changes | Learned `P_in` against a fixed projection |
| **Misrouted branch** (`stack`, T2) | This family's rebuilt G2, with the other family's last-block specialist branch (`P_in`, core, `P_out`) substituted, then the standard 8-step evaluation | Correct expert identity against intentional misrouting |
| **Misrouted expert** (`stack`, T2; `stage_b`) | The stack: the other family's whole last-block expert (FFN and specialist) substituted. Stage B: R11's own control, the other family's G2 expert on these episodes | The same |
| **Decodability** (`stack`, T2) | Out-of-sample linear R² of the 4 task coordinates `z`, from the specialist's code `q = P_in h` (5 values) and from its input `h` (64 values). The decoder is fitted on 8 DEV episodes and scored per TEST episode, for context-free probes and for queries in context | What information reaches the specialist |

**Restricted to G2** (the fixed gate, the arm R11 Stage B qualified). The learned gate was `GATE_NOT_LIMITING` in R11
and adds nothing to the representation question. G1 is not rerun.

## 3. Readings (frozen)

**"Better"** is R12's frozen rule (`PREREGISTRATION_R12.md` §5). All three conditions must hold:
- a pooled geometric-mean ratio ≤ 0.95;
- a one-sided exact sign test p < 0.05;
- at least 2/3 of the 12 groups below 1.

| Reading | Rule |
|---|---|
| `stage_b_assay` (per regime) | `REPRODUCED` if Stage B G2 is better than Stage B G0. This checks that R11 Stage B's specialization holds on these groups |
| `benefit_ceiling` (per regime) | `HEADROOM_UNDER_COORDINATE_ACCESS` if Stage B G2 is better than the regime's FFN-only stack in context. `IN_CONTEXT_INFERENCE_BETTER` if that stack is better than Stage B G2. Otherwise `INDETERMINATE` |
| `task_feature_access` (per regime; T2 attributes) | `COORDINATE_ACCESS_BETTER` if Stage B G2 is better than the stack's G2 `fast8`. `STACK_REPRESENTATION_BETTER` if the reverse holds. Otherwise `NO_DIFFERENCE_ESTABLISHED` |
| `projection` (T2) | `LEARNED_PROJECTION_BETTER`, `FIXED_PROJECTION_BETTER` or `NO_DIFFERENCE_ESTABLISHED` |
| `identity` (T2 stack; Stage B) | `IDENTITY_MATTERS` if the correct specialist or expert is better than the misrouted one; else `NOT_ESTABLISHED` |
| Integrity | Three conditions: every rebuild is identical to its T2 record; the fixed projection is unchanged; every R12 gate holds in every stack evaluation. Otherwise `NONQUAL_INTEGRITY` |

**Attribution (P8), read mechanically.**
- **Initialization.** `EXPLAINS_THE_T1_CURVE` if G2's adaptation-curve criterion fails in T1 and holds in T2
  (`study_r12.json`).
- **Task-feature access.** The T2 `task_feature_access` reading.
- **Routing.** The expert is fixed (one expert, routing weight exactly 1), so routing cannot cause the failure. The
  identity readings say whether the trained specialist's identity matters.
- **Absence of benefit.** The `benefit_ceiling` readings.

## 4. Development, budget and stopping

**Development** (records `p5dev_*.json`, named with digests in the amendment).
- `run_stage_b` rebuilt R11's own Stage B group (R11B, seed 201: G0, G2, G3 and G2's misrouted control) and matched
  R11's record row for row: all 8 comparisons identical.
- The stack mechanics ran in `R12C_DEV` at a few training steps.
- No P5 group, namespace or seed was used. Nothing in the plan changed.

**Budget.** About 100 CPU-minutes at 2 threads:
- 24 G2 stack trainings (12 rebuilt, 12 with a fixed projection);
- 72 Stage B trainings.

**Stopping.**
- No P5 record is rerun, and no seed is added.
- No rule or measure changes after a result.
- No new architecture: the oracle access is R11's own harness, not a new path into the stack.
- A failed run is recorded as failed.
