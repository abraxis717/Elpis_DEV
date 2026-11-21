# Elpis Torch R12 — evidence: the micro-CGC volatile fast state inside the Stage C stack

**Principal question.** Inside R11's actual Stage C ERSA stack, does restoring the exact episode-local update of the
last block's 120-parameter CGC core recover CGC's specialist advantage?
- **The update.** R11 Stage B's: 8 clipped SGD steps on the observed support, through a private, volatile, bounded
  fast state.
- **The baselines.** The FFN-only stack, and a matched fast-adapted GELU specialist.

**Answer: no, not under the tested conditions.**
- **The verdict.** Every arm is `NOT_ESTABLISHED`, in both training regimes and for both gate modes. The frozen
  reading is `NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED`.
- **The mechanism itself is exact and coherent.**
  - The cached path equals full recomputation (0.0), and the analytic gradient equals autograd (at most 6.9e-17
    in float64).
  - In every one of 864 evaluation episodes, reset and disabled learning restore the frozen output bit for
    bit, the recurrent states stay unchanged, and no target leaks.
  - Earlier blocks are refused.
  - It costs 960 bytes, and 15–25 ms for 8 updates on one CPU thread.
- **T1, R11's Stage C models adapted at test time: the update does not help.**
  - These models never learned an adaptation-compatible initialization. The support-loss gradient is small (median
    step-1 norm 0.04–0.16), so 8 steps move `ΔW` by only 0.03–0.09 against a bound of 1.28.
  - The query loss stays flat (learned gate G1: 1.05× its frozen value) or gets worse (fixed gate G2: 1.60×).
  - Fast G1 beats the fast matched GELU (0.68), but not the FFN-only stack (0.92, p = 0.40).
- **T2, the same stack trained with the update in the loop: the update works, but only matches the FFN-only
  stack.**
  - The adaptation is steep: G2's 8-step query loss is 0.044× its frozen value.
  - It is target-driven: 0.116× the loss after the same update on shuffled targets.
  - CGC beats the matched fast GELU by 48% (G2, 0.519) and 36% (G1, 0.643).
  - But fast G2 only equals the FFN-only stack with the support in context (0.982, p = 0.66), and fast G1 is worse
    (1.22).
  - The trained stacks rely on the update instead of the context. Frozen, with the support in context, they are 20
    to 22× worse than the FFN-only stack, at about the context-free level.
- **P5, the representation seam (diagnosis only).**
  - The task coordinates reach the specialist: R² 0.998 from its input and 0.97 from its 5-value code. A fixed
    `P_in` works as well as a learned one.
  - On the same episodes, the stack's adapted specialist beats R11 Stage B's exact coordinate access (0.42). So
    task-feature access is not the limit, and initialization explains T1's flat curve.
  - But even with exact coordinates, the 8-step CGC is 1.85 to 2.34× worse than the FFN-only stack's in-context
    inference. On these tasks the restored update has no advantage to recover.

**What R12 changed.**
- Nothing under `src/` or `native/`, no CI policy, no R0–R11 record and no default.
- No source that any earlier record pins: R12 adds new files only.
- No checkpoint parameter at inference: every update is a volatile `ΔW` over a frozen model, verified by digest.

**Documents.**
- Plans:
  - `PREREGISTRATION_R12.md`, frozen at `e01b2b2`;
  - the P5 amendment, `PREREGISTRATION_R12_P5.md`, frozen at `4561a9a` before any P5 data.
- Design: `DESIGN_R12.md`, with the P0 audit and the dependency diagram.
- Contract: `CGC_FAST_STATE_CONTRACT_R12.md`.
- Ledger: `CLAIM_AUDIT_R9.md`, section "R12 result".

## 0. Verdicts, by kind

| Kind | Verdict | Where |
|---|---|---|
| Mechanical and numerical qualification | **PASS**, under five checks: (1) cached = full recomputation (0.0, output and gradient); (2) analytic = autograd gradient (at most 6.9e-17 in float64); (3) batched = transactional update; (4) float32 within 1.3e-6 of float64; (5) states unchanged, earlier blocks refused. **Engineering, not a specialist result** | §3 |
| T1 rebuild of R11's Stage C models | **Exact**: 48 of 48 arms identical to R11's records (rows, trace, selected step) | §3 |
| T1, G1 (learned gate) | **`NOT_ESTABLISHED`**: fails beats-FFN (0.918, p = 0.40) and the adaptation curve (1.049) | §4 |
| T1, G2 (fixed gate) | **`NOT_ESTABLISHED`**: 8 steps raise the query loss 1.60×; 2.30× the FFN-only stack; 1.70× the fast GELU | §4 |
| T2, G1 | **`NOT_ESTABLISHED`**: fails beats-FFN (1.219). It meets the other five criteria | §5 |
| T2, G2 | **`NOT_ESTABLISHED`**: fails beats-FFN (0.982, p = 0.66, 7/12 groups). It meets the other five criteria, beating the fast GELU at 0.519 | §5 |
| Integrity gates (every episode of every specialist arm, both regimes) | **PASS** | §4, §5 |
| The reading of the two regimes | **`NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED`** | §6.1 |
| P5, the representation seam (diagnosis only; amendment frozen before its data) | Four findings: (1) initialization explains T1's curve; (2) task-feature access is **not limiting** (`STACK_REPRESENTATION_BETTER`, 0.421; a fixed `P_in` equals a learned one); (3) identity matters (misrouting is 3.6 to 4.4× worse); (4) **`IN_CONTEXT_INFERENCE_BETTER`** in both regimes' groups (0.540, 0.428). Integrity held | §6.2 |
| Attribution (directive P8) | **`NOT_ESTABLISHED`, explained.** Initialization explains T1 only. Task-feature access and routing are not the cause. The absence of benefit over in-context inference on these tasks is demonstrated | §6.3 |
| Cache and state coherence | **COHERENT** for the last block; every earlier block refused; no HOLD needed | §7 |
| Cost (d64 stack, one CPU thread) | 960 fast-state bytes; 8 updates in 15–25 ms on the cached path, against 58–63 ms by full recomputation | §8 |
| Sidecar and manifest | **NOT PROPOSED.** Not justified, since nothing qualified. The requirements are recorded; no GGUF code or claim | §9 |
| R0–R11 records and verdicts | Unchanged | `CLAIM_AUDIT_R9.md` |

## 1. Provenance

**Base.** `main` at `72b66b1f2c069ff331bf39992bec47f31efdc377`, the R11 merge; post-merge CI run 44850567035
succeeded. Branch `claude/funny-turing-obbdy5`.

**The session branch.** The branch already existed and carried an old R5 commit (`fe82cc8`). R6 had reconciled its
content into main:
- its five records are byte-identical there;
- `run_r5.py` evolved after it.

`main` was merged into the branch, resolving `run_r5.py` to main's version, so the branch's tree equalled main's
exactly before R12 began. The PR diff therefore contains only R12.

**The freezes.**

| Plan | File | SHA-256 | Frozen at | Before |
|---|---|---|---|---|
| R12 | `results/r12/preregistration.json` | `f0f106067f1672a3dd4b5fa17f0f51543d2eca4753dc4e48080cdeed4552d44c` | `e01b2b2` | any confirmation record |
| P5 amendment | `results/r12/amendment_p5.json` | `e655c3fc0f46063b4a5c42622276004539628deac04cd212863307203705fdb7` | `4561a9a` | any P5 record; it names `study_r12.json` by digest |

**Integrity of the records.**
- **Digests.** Every confirmation record carries the plan's SHA-256 and the SHA-256 of every source it ran. Every P5
  record also carries the amendment's digest, the R12 study's, and those of the P5 and R11 runners.
- **Refusals.** Neither `combine` refused a record.
- **No dirty tree.** No record is `+dirty`. Each names the HEAD at the moment it was written (§11).
- **Independent recomputation.** `test_r12_records.py` re-implements every rule on the JSON alone, R12's and P5's.

## 2. What ran

| Record | What | Count |
|---|---|---|
| `t1_<family>_s<seed>.json` | T1: R11's Stage C models rebuilt (R11C, seeds 301–306, both families; all four arms) and verified against R11's records, then evaluated with the fast update on 12 fresh episodes (split `R12TEST`) | 12 |
| `t2_<family>_s<seed>.json` | T2: the same stack trained with the update in the loop (R12C, seeds 401–406, both families; all four arms), evaluated on 12 TEST episodes | 12 |
| `study_r12.json` | `combine`: the frozen rules over the 24 records | 1 |
| `parity.json`, `audit.json`, `latency.json`, `latency_repeat.json` | Engineering: numerical parity, exact accounting, measured cost (measured twice) | 4 |
| `dev_*.json` | Development only, named with digests in the plan | 2 |
| `amendment_p5.json` | P5: the plan, frozen before any P5 record | 1 |
| `p5_b_<regime>_s<seed>.json` | P5: R11 Stage B's procedure on each R12 group's own tasks (G0, G2, G3, and G2's misrouted control), evaluated on that group's R12 episodes | 12 |
| `p5_c_s<seed>.json` | P5: T2's G2 rebuilt and verified; with a fixed `P_in`; misrouted (specialist branch, whole expert); the decodability of the task coordinates | 6 |
| `study_p5.json` | P5's `combine`: the amendment's frozen readings | 1 |
| `p5dev_*.json` | P5 development only, named with digests in the amendment | 2 |

**Per comparison.** 144 paired episodes (12 groups × 12 episodes), grouped by (family, seed).

**Per evaluation episode of a specialist arm.**
- The curve at 0, 1, 2, 4 and 8 steps.
- Every control: shuffled targets, reset, updates disabled, the previous episode's core.
- The secondary conditions: context-free queries and OOD queries.
- Every integrity gate, and the update trace.

**Compute.**
- The T1 records took 73 minutes in total and the T2 records 103 minutes, each at 2 threads.
- P5 took 72 more minutes: 53 for the stack records and 19 for Stage B.
- Everything ran in two parallel queues on a 4-core CPU container. There was no GPU.

## 3. Mechanical and numerical qualification

**Parity.** These are engineering results; they qualify the mechanism's implementation, not its usefulness.

Numerical parity: maximum absolute differences (parity.json).

| Arm, precision | Output: full vs cached | Gradient: full vs cached | Gradient: analytic vs autograd | Update: batched vs transactional | States unchanged | Bound reached |
|---|---|---|---|---|---|---|
| G1, float32 | 0 | 0 | 1.5e-08 | 1.5e-08 | yes | no |
| G1, float64 | 0 | 0 | 3.5e-17 | 5.6e-17 | yes | no |
| G2, float32 | 0 | 0 | 3.0e-08 | 3.0e-08 | yes | no |
| G2, float64 | 0 | 0 | 6.9e-17 | 5.6e-17 | yes | no |
| G3, float32 | 0 | 0 | — | 3.0e-08 | yes | no |
| G3, float64 | 0 | 0 | — | 5.6e-17 | yes | no |

**ERSA LM with window attention** (two chunks): the logits change under a changed last core, while every ERSA state, the window K/V ring and its token history stay bit-identical. **An earlier block** is refused.

| Arm | Adapted output: float32 vs float64 | Output RMS | `ΔW`: float32 vs float64 |
|---|---|---|---|
| G1 | 1.3e-06 | 1.16 | 5.5e-09 |
| G2 | 7.3e-07 | 0.73 | 2.9e-08 |
| G3 | 9.5e-07 | 1.11 | 2.5e-08 |

**T1's rebuild of R11's models.**
48 of 48 rebuilt arms (12 groups × 4 arms) are identical to the committed R11 records: every TEST row (largest difference 0.0), the training trace and the selected step.

## 4. T1: R11's Stage C models, adapted at test time (the decisive test as specified)

**The protocol.**
- **The models.** R11's own Stage C models, rebuilt exactly. They were trained end to end for frozen in-context
  inference and never for adaptation.
- **The episodes.** Each evaluation episode is fresh: split `R12TEST`, never seen by R11 or by development. Its 32
  support pairs stay in the context.
- **The update.** The last block's core takes 8 steps on the same 32 pairs, presented as context-free probes. The
  48 queries are then scored.

**Adaptation curve.**

Query MSE, support in context (k = 32): mean over 12 groups of the per-group median.

| Arm | 0 steps (frozen) | 1 | 2 | 4 | 8 | shuffled targets, 8 | reset before queries | updates disabled | zero predictor |
|---|---|---|---|---|---|---|---|---|---|
| G0 FFN only | 0.0063 | — | — | — | — | — | — | — | 0.1497 |
| G1 CGC, learned gate | 0.0053 | 0.0053 | 0.0053 | 0.0053 | 0.0054 | 0.0059 | 0.0053 | 0.0053 | 0.1497 |
| G2 CGC, fixed gate | 0.0080 | 0.0082 | 0.0084 | 0.0091 | 0.0109 | 0.0166 | 0.0080 | 0.0080 | 0.1497 |
| G3 GELU (matched) | 0.0063 | 0.0063 | 0.0063 | 0.0065 | 0.0070 | 0.0075 | 0.0063 | 0.0063 | 0.1497 |

**Paired comparisons.**

Paired comparisons (the better rule; 144 paired episodes).

| Arm | Comparison | ratio | sign-test p | groups below 1 | better |
|---|---|---|---|---|---|
| G1 | fast_vs_fast_gelu | 0.679 | 2.2e-09 | 11/12 | yes |
| G1 | fast_vs_ffn | 0.918 | 0.401 | 5/12 | no |
| G1 | fast_vs_frozen_gelu | 0.850 | 0.018 | 9/12 | yes |
| G1 | fast_vs_frozen_self | 1.049 | 0.533 | 7/12 | no |
| G1 | fast_vs_shuffled | 0.823 | 8.8e-06 | 11/12 | yes |
| G1 | frozen_vs_ffn | 0.875 | 0.040 | 6/12 | no |
| G1 | secondary_context_free_fast_vs_fast_gelu | 1.099 | 1.000 | 1/12 | no |
| G1 | secondary_ood_fast_vs_fast_gelu | 0.814 | 9.7e-04 | 10/12 | yes |
| G1 | secondary_ood_fast_vs_ffn | 0.912 | 0.280 | 6/12 | no |
| G1 | secondary_previous_core_vs_frozen | 1.053 | 0.992 | 3/12 | no |
| G2 | fast_vs_fast_gelu | 1.700 | 1.000 | 3/12 | no |
| G2 | fast_vs_ffn | 2.299 | 1.000 | 1/12 | no |
| G2 | fast_vs_frozen_gelu | 2.130 | 1.000 | 1/12 | no |
| G2 | fast_vs_frozen_self | 1.603 | 1.000 | 1/12 | no |
| G2 | fast_vs_shuffled | 0.650 | 6.4e-09 | 10/12 | yes |
| G2 | frozen_vs_ffn | 1.434 | 0.982 | 6/12 | no |
| G2 | secondary_context_free_fast_vs_fast_gelu | 0.917 | 1.9e-05 | 10/12 | yes |
| G2 | secondary_ood_fast_vs_fast_gelu | 1.331 | 0.998 | 4/12 | no |
| G2 | secondary_ood_fast_vs_ffn | 1.490 | 1.000 | 3/12 | no |
| G2 | secondary_previous_core_vs_frozen | 1.631 | 1.000 | 0/12 | no |
| G3 | fast_vs_ffn | 1.353 | 1.000 | 3/12 | no |
| G3 | fast_vs_frozen_self | 1.253 | 1.000 | 3/12 | no |

**Criteria and disposition.**

The six criteria and the disposition.

| Arm | assay | beats FFN | beats fast GELU | adaptation curve | target-driven | integrity | disposition |
|---|---|---|---|---|---|---|---|
| G1 CGC, learned gate | yes | **no** | yes | **no** | yes | yes | **`NOT_ESTABLISHED`** |
| G2 CGC, fixed gate | yes | **no** | **no** | **no** | yes | yes | **`NOT_ESTABLISHED`** |

**Update diagnostics.**

Update diagnostics (medians over all episodes of the arm).

| Arm | gradient norm, step 1 | gradient norm, step 8 | clipped steps | support loss before / after 8 | delta norm after 8 | cosine of successive updates | R² of P_in(h) on the probe inputs |
|---|---|---|---|---|---|---|---|
| G1 CGC, learned gate | 0.043 | 0.040 | 0/1152 | 0.0915 / 0.0889 | 0.026 | 1.000 | 0.948 |
| G2 CGC, fixed gate | 0.158 | 0.126 | 0/1152 | 0.0913 / 0.0774 | 0.090 | 0.999 | 0.870 |
| G3 GELU (matched) | 0.099 | 0.093 | 0/1152 | 0.0920 / 0.0836 | 0.061 | 1.000 | 0.949 |

**Secondary conditions.**

Secondary conditions (mean over groups of the per-group median).

| Arm | context-free, frozen | context-free, fast 8 | OOD, frozen | OOD, fast 8 | previous episode's core | last specialist ablated (frozen) |
|---|---|---|---|---|---|---|
| G0 FFN only | 0.0920 | — | 0.0153 | — | — | 0.0063 |
| G1 CGC, learned gate | 0.0937 | 0.0917 | 0.0141 | 0.0142 | 0.0054 | 0.0368 |
| G2 CGC, fixed gate | 0.0922 | 0.0778 | 0.0186 | 0.0215 | 0.0117 | 0.7033 |
| G3 GELU (matched) | 0.0929 | 0.0847 | 0.0157 | 0.0162 | 0.0071 | 0.0301 |

**Integrity.**

Integrity gates, every episode.

| Arm | all | bound_never_binds | disabled_exact | disabled_refused | every_update_applied | finite | full_vs_cached_parity | no_leakage | reset_exact | states_unchanged_by_the_core |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| G2 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| G3 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |

**Per-group ratios.**

Per-group geometric-mean ratios of the primary comparisons.

| Group | G1 fast/FFN | G1 fast/fast GELU | G2 fast/FFN | G2 fast/fast GELU | G3 fast/FFN |
|---|---|---|---|---|---|
| smooth 301 | 1.07 | 0.98 | 1.09 | 1.01 | 1.09 |
| conditional 301 | 0.37 | 0.39 | 1.37 | 1.45 | 0.95 |
| smooth 302 | 1.22 | 0.29 | 3.71 | 0.88 | 4.20 |
| conditional 302 | 0.83 | 0.54 | 3.95 | 2.56 | 1.55 |
| smooth 303 | 1.60 | 0.91 | 1.21 | 0.69 | 1.76 |
| conditional 303 | 0.57 | 0.72 | 1.50 | 1.90 | 0.79 |
| smooth 304 | 1.05 | 0.75 | 4.28 | 3.07 | 1.40 |
| conditional 304 | 0.46 | 0.83 | 0.54 | 0.99 | 0.55 |
| smooth 305 | 1.19 | 0.56 | 5.54 | 2.62 | 2.11 |
| conditional 305 | 1.08 | 0.44 | 3.85 | 1.57 | 2.46 |
| smooth 306 | 0.97 | 0.96 | 1.82 | 1.82 | 1.00 |
| conditional 306 | 1.67 | 1.67 | 6.03 | 6.02 | 1.00 |

**What T1 shows.**
- **The update is too weak to matter.** The support-loss gradients are small (median step-1 norm 0.04 to 0.16) and
  never clipped. Successive steps point the same way (cosine ≈ 1), and 8 steps move `ΔW` by 0.03 to 0.09. The
  context-free support loss falls by only 3% (G1), 15% (G2) and 9% (G3).
- **What little it does works against the context.** The update fits context-free probes, so it corrects the error
  a prediction makes WITHOUT context. ERSA has already removed that error for the in-context queries, so the
  correction double-counts it. For the fixed gate the query loss rises 1.60× in 8 steps, and the OOD and
  previous-core conditions move the same way. Shuffled targets are worse still (0.65), so the move is
  target-driven, but in the wrong direction for in-context queries.
- **The learned gate stays near 0.13 (block 1).** It keeps G1's branch small, so its curve is flat. Fast G1 beats the
  fast matched GELU (0.679, 11/12) mostly because the GELU arm gets worse under the update (1.25×) while G1 barely
  moves. Against the frozen GELU the margin is 0.850.
- **Reading.** The integrated mechanism is not at fault: it is exact (§3), and T2 shows it can adapt. These Stage C
  models simply lack an adaptation-compatible initialization, which the directive's training caveat anticipated.

## 5. T2: the same stack, trained with the update in the loop

**The protocol.**
- **The training.** The same architecture, adaptation algorithm and training plan as T1. The one change: during
  training, a specialist arm's query pass runs under its last-block core adapted on that step's support, with a
  first-order outer gradient, and DEV selection scores after adaptation. The objective therefore differs from R11
  Stage C's, which is disclosed. G0 trains with R11's `train_stack`.
- **The seeds.** Fresh: R12C, 401–406.

**Adaptation curve.**

Query MSE, support in context (k = 32): mean over 12 groups of the per-group median.

| Arm | 0 steps (frozen) | 1 | 2 | 4 | 8 | shuffled targets, 8 | reset before queries | updates disabled | zero predictor |
|---|---|---|---|---|---|---|---|---|---|
| G0 FFN only | 0.0052 | — | — | — | — | — | — | — | 0.1489 |
| G1 CGC, learned gate | 0.0777 | 0.0261 | 0.0140 | 0.0085 | 0.0059 | 0.0234 | 0.0777 | 0.0777 | 0.1489 |
| G2 CGC, fixed gate | 0.0916 | 0.0259 | 0.0133 | 0.0078 | 0.0046 | 0.0367 | 0.0916 | 0.0916 | 0.1489 |
| G3 GELU (matched) | 0.0781 | 0.0343 | 0.0214 | 0.0117 | 0.0086 | 0.0144 | 0.0781 | 0.0781 | 0.1489 |

**Paired comparisons.**

Paired comparisons (the better rule; 144 paired episodes).

| Arm | Comparison | ratio | sign-test p | groups below 1 | better |
|---|---|---|---|---|---|
| G1 | fast_vs_fast_gelu | 0.643 | 7.5e-17 | 11/12 | yes |
| G1 | fast_vs_ffn | 1.219 | 1.000 | 3/12 | no |
| G1 | fast_vs_frozen_gelu | 0.064 | 4.5e-44 | 12/12 | yes |
| G1 | fast_vs_frozen_self | 0.062 | 4.5e-44 | 12/12 | yes |
| G1 | fast_vs_shuffled | 0.243 | 7.9e-37 | 12/12 | yes |
| G1 | frozen_vs_ffn | 19.538 | 1.000 | 0/12 | no |
| G1 | secondary_context_free_fast_vs_fast_gelu | 0.587 | 7.5e-10 | 9/12 | yes |
| G1 | secondary_ood_fast_vs_fast_gelu | 0.731 | 7.5e-17 | 11/12 | yes |
| G1 | secondary_ood_fast_vs_ffn | 1.183 | 1.000 | 3/12 | no |
| G1 | secondary_previous_core_vs_frozen | 1.556 | 1.000 | 1/12 | no |
| G2 | fast_vs_fast_gelu | 0.519 | 7.4e-15 | 10/12 | yes |
| G2 | fast_vs_ffn | 0.982 | 0.661 | 7/12 | no |
| G2 | fast_vs_frozen_gelu | 0.052 | 4.5e-44 | 12/12 | yes |
| G2 | fast_vs_frozen_self | 0.044 | 4.5e-44 | 12/12 | yes |
| G2 | fast_vs_shuffled | 0.116 | 4.5e-44 | 12/12 | yes |
| G2 | frozen_vs_ffn | 22.227 | 1.000 | 0/12 | no |
| G2 | secondary_context_free_fast_vs_fast_gelu | 0.378 | 5.2e-26 | 11/12 | yes |
| G2 | secondary_ood_fast_vs_fast_gelu | 0.576 | 1.5e-17 | 11/12 | yes |
| G2 | secondary_ood_fast_vs_ffn | 0.932 | 0.040 | 7/12 | no |
| G2 | secondary_previous_core_vs_frozen | 1.543 | 1.000 | 1/12 | no |
| G3 | fast_vs_ffn | 1.895 | 1.000 | 3/12 | no |
| G3 | fast_vs_frozen_self | 0.100 | 4.5e-44 | 12/12 | yes |

**Criteria and disposition.**

The six criteria and the disposition.

| Arm | assay | beats FFN | beats fast GELU | adaptation curve | target-driven | integrity | disposition |
|---|---|---|---|---|---|---|---|
| G1 CGC, learned gate | yes | **no** | yes | yes | yes | yes | **`NOT_ESTABLISHED`** |
| G2 CGC, fixed gate | yes | **no** | yes | yes | yes | yes | **`NOT_ESTABLISHED`** |

**Update diagnostics.**

Update diagnostics (medians over all episodes of the arm).

| Arm | gradient norm, step 1 | gradient norm, step 8 | clipped steps | support loss before / after 8 | delta norm after 8 | cosine of successive updates | R² of P_in(h) on the probe inputs |
|---|---|---|---|---|---|---|---|
| G1 CGC, learned gate | 1.228 | 0.092 | 14/1152 | 0.0954 / 0.0061 | 0.157 | 0.978 | 0.983 |
| G2 CGC, fixed gate | 1.563 | 0.081 | 27/1152 | 0.0975 / 0.0036 | 0.149 | 0.669 | 0.981 |
| G3 GELU (matched) | 0.782 | 0.081 | 0/1152 | 0.0952 / 0.0121 | 0.189 | 0.996 | 0.977 |

**Secondary conditions.**

Secondary conditions (mean over groups of the per-group median).

| Arm | context-free, frozen | context-free, fast 8 | OOD, frozen | OOD, fast 8 | previous episode's core | last specialist ablated (frozen) |
|---|---|---|---|---|---|---|
| G0 FFN only | 0.0913 | — | 0.0129 | — | — | 0.0052 |
| G1 CGC, learned gate | 0.0896 | 0.0102 | 0.0917 | 0.0145 | 0.1669 | 0.2750 |
| G2 CGC, fixed gate | 0.0978 | 0.0054 | 0.1064 | 0.0110 | 0.1779 | 1.0246 |
| G3 GELU (matched) | 0.0924 | 0.0147 | 0.0909 | 0.0194 | 0.1589 | 0.1281 |

**Training.**

Training: selected step (median) and seconds.

| Arm | selected step (median; range) | training seconds (median) | learned gates (median, block 0 / 1) |
|---|---|---|---|
| G0 FFN only | 2000 (1250–2000) | 87 | — |
| G1 CGC, learned gate | 2000 (1250–2000) | 136 | 0.078 / 0.207 |
| G2 CGC, fixed gate | 1750 (1000–2000) | 142 | — |
| G3 GELU (matched) | 2000 (1500–2000) | 142 | — |

**Integrity.**

Integrity gates, every episode.

| Arm | all | bound_never_binds | disabled_exact | disabled_refused | every_update_applied | finite | full_vs_cached_parity | no_leakage | reset_exact | states_unchanged_by_the_core |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| G2 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| G3 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass |

**Per-group ratios.**

Per-group geometric-mean ratios of the primary comparisons.

| Group | G1 fast/FFN | G1 fast/fast GELU | G2 fast/FFN | G2 fast/fast GELU | G3 fast/FFN |
|---|---|---|---|---|---|
| smooth 401 | 0.27 | 0.55 | 0.37 | 0.75 | 0.49 |
| conditional 401 | 2.27 | 0.43 | 2.69 | 0.51 | 5.26 |
| smooth 402 | 1.42 | 0.62 | 0.57 | 0.25 | 2.29 |
| conditional 402 | 1.96 | 0.85 | 0.48 | 0.21 | 2.32 |
| smooth 403 | 0.36 | 0.97 | 0.42 | 1.15 | 0.37 |
| conditional 403 | 2.38 | 0.66 | 2.34 | 0.65 | 3.61 |
| smooth 404 | 2.63 | 3.32 | 2.14 | 2.70 | 0.79 |
| conditional 404 | 1.59 | 0.55 | 2.44 | 0.84 | 2.92 |
| smooth 405 | 1.92 | 0.88 | 0.69 | 0.32 | 2.17 |
| conditional 405 | 1.33 | 0.35 | 0.93 | 0.24 | 3.81 |
| smooth 406 | 1.07 | 0.20 | 1.61 | 0.30 | 5.42 |
| conditional 406 | 0.64 | 0.57 | 0.55 | 0.49 | 1.12 |

**What T2 shows.**
- **An adaptation-compatible initialization exists in this stack, and the restored update then works.**
  - It is steep: G2 goes from 0.0916 frozen to 0.0046 after 8 steps.
  - It is target-driven: 0.116× the shuffled-target update.
  - It is the CGC-specific advantage R97W and R11 Stage B found: against the matched 120-parameter GELU under the
    identical update, 0.519 (G2) and 0.643 (G1).
  - The gradients are 8 to 29× T1's (median step-1 norm 0.8 to 1.6); the clip binds on 2% of G2's steps.
- **But the trained stack substitutes the update for the context instead of adding to it.**
  - Frozen, with the 32 support pairs in context, every specialist arm is about as bad as a context-free predictor
    (0.078 to 0.092, against 0.091 for G0 without context). The FFN-only stack, trained by the same plan, reaches
    0.0052.
  - After the update, the context adds little: fast G2 is 0.0046 in context and 0.0054 without it.
  - The fast specialist therefore competes with ERSA's in-context inference rather than refining it, and only draws
    level: fast G2 against G0 is 0.982 (p = 0.66, 7/12 groups). The 12 groups split widely, from 0.37 to 2.69.
- **A stale core is harmful.** Under the previous episode's core, the query loss is 1.54× the frozen loss. The
  contract's one-state-per-episode reset is therefore necessary, not cosmetic.

## 6. The reading of the two regimes, and the representation seam

### 6.1 The frozen reading

- **No arm qualifies in either regime.** The reading is `NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED`
  (`study_r12.json`).
- **The two regimes separate two failures.**
  - In T1 the mechanism is present but the initialization is not adaptation-compatible. The update barely moves,
    and where it moves it fights the context.
  - In T2 the initialization is compatible and the update is strong and CGC-specific. But the stack learns to rely
    on it in place of in-context inference, and ends level with the FFN-only stack.
- **The trigger.** The plan sent this reading to the representation stage (directive P5), under its own amendment.

### 6.2 P5: the representation seam (diagnosis only)

**The plan.** The amendment, `PREREGISTRATION_R12_P5.md`, was frozen at `4561a9a` before any P5 data. P5 asks
whether the specialist receives the information its adaptation protocol needs. It runs on the R12 records' own
evaluation episodes, so every comparison is paired.

**What runs.**
- **R11 Stage B's procedure, unchanged**, trained on each R12 group's own tasks (12 T1 and 12 T2 groups). Its
  specialist reads the exact task coordinates with no context: an oracle by construction, used for diagnosis only.
- **T2's G2, rebuilt.** Every rebuild is identical to its T2 record. Three variants follow:
  - the same training with `P_in` fixed at its random initialization;
  - the other family's specialist branch substituted;
  - the other family's whole last-block expert substituted.
- **The decodability of the task coordinates**, from the specialist's input and from its 5-value code.

**Development.** R11's own Stage B group (R11B, seed 201) was rebuilt through the new path and matched R11's
record in all 8 comparisons.

**Levels.**

Query MSE after 8 steps (geometric mean over the 144 paired episodes; frozen = 0 steps).

| Condition | T1 groups | T2 groups |
|---|---|---|
| Stack G0, frozen, support in context (R12 record) | 0.0043 | 0.0039 |
| Stack G2, fast 8, support in context (R12 record) | 0.0098 | 0.0039 |
| Stage B G0: FFN alone, coordinate access, no context | 0.0990 | 0.1016 |
| Stage B G2: CGC, coordinate access, 8 steps | 0.0079 | 0.0092 |
| Stage B G3: GELU, coordinate access, 8 steps | 0.0114 | 0.0115 |
| Stage B G2, the other family's expert (misrouted), 8 steps | 0.0613 | 0.0664 |
| Stage B G2, 0 steps | 0.4203 | 0.1359 |
| Stack G2 rebuilt, fast 8 (= the R12 record) | — | 0.0039 |
| Stack G2, fixed `P_in`, fast 8 | — | 0.0038 |
| Stack G2, fixed `P_in`, frozen | — | 0.0831 |
| Stack G2, the other family's specialist branch, fast 8 | — | 0.0140 |
| Stack G2, the other family's whole last-block expert, fast 8 | — | 0.0172 |

**Paired comparisons.**

Paired comparisons (R12's better rule; 144 paired episodes each).

| Groups | Comparison (X vs Y) | ratio X/Y | sign-test p | groups below 1 | X better |
|---|---|---|---|---|---|
| T1 | in_context_g0_vs_stage_b_g2 | 0.540 | 3.1e-07 | 9/12 | yes |
| T1 | stack_g2_vs_stage_b_g2 | 1.243 | 0.992 | 3/12 | no |
| T1 | stage_b_g2_vs_in_context_g0 | 1.850 | 1.000 | 3/12 | no |
| T1 | stage_b_g2_vs_stack_g2 | 0.805 | 0.012 | 9/12 | yes |
| T1 | stage_b_g2_vs_stage_b_g0 | 0.079 | 4.5e-44 | 12/12 | yes |
| T1 | stage_b_g2_vs_stage_b_g3 | 0.690 | 1.5e-17 | 11/12 | yes |
| T1 | stage_b_g2_vs_stage_b_misrouted | 0.128 | 4.7e-40 | 12/12 | yes |
| T2 | in_context_g0_vs_stage_b_g2 | 0.428 | 1.9e-21 | 10/12 | yes |
| T2 | stack_g2_vs_stage_b_g2 | 0.421 | 1.3e-20 | 10/12 | yes |
| T2 | stage_b_g2_vs_in_context_g0 | 2.335 | 1.000 | 2/12 | no |
| T2 | stage_b_g2_vs_stack_g2 | 2.377 | 1.000 | 2/12 | no |
| T2 | stage_b_g2_vs_stage_b_g0 | 0.091 | 4.5e-44 | 12/12 | yes |
| T2 | stage_b_g2_vs_stage_b_g3 | 0.800 | 1.9e-12 | 11/12 | yes |
| T2 | stage_b_g2_vs_stage_b_misrouted | 0.139 | 6.5e-42 | 12/12 | yes |
| T2_stack | correct_vs_misrouted_branch | 0.277 | 7.9e-37 | 12/12 | yes |
| T2_stack | correct_vs_misrouted_unit | 0.226 | 4.7e-40 | 12/12 | yes |
| T2_stack | fixed_vs_learned_projection | 0.968 | 0.720 | 5/12 | no |
| T2_stack | learned_vs_fixed_projection | 1.033 | 0.339 | 7/12 | no |

**Readings.**

The frozen readings.

| Groups | Reading | Value |
|---|---|---|
| T1 | benefit_ceiling | `IN_CONTEXT_INFERENCE_BETTER` |
| T1 | stage_b_assay | `REPRODUCED` |
| T1 | stage_b_identity | `IDENTITY_MATTERS` |
| T1 | task_feature_access | `COORDINATE_ACCESS_BETTER` |
| T2 | benefit_ceiling | `IN_CONTEXT_INFERENCE_BETTER` |
| T2 | stage_b_assay | `REPRODUCED` |
| T2 | stage_b_identity | `IDENTITY_MATTERS` |
| T2 | task_feature_access | `STACK_REPRESENTATION_BETTER` |
| T2_stack | identity_branch | `IDENTITY_MATTERS` |
| T2_stack | identity_unit | `IDENTITY_MATTERS` |
| T2_stack | projection | `NO_DIFFERENCE_ESTABLISHED` |

**Decodability of the task coordinates.**

Linear decodability of the task coordinates (out-of-sample R², median over T2 TEST episodes).

| Model | from the code `P_in h`, probes | from the input `h`, probes | from the code, queries in context | from the input, queries in context |
|---|---|---|---|---|
| learned `P_in` (rebuilt) | 0.975 | 0.998 | 0.971 | 0.998 |
| fixed `P_in` | 0.946 | 0.998 | 0.930 | 0.998 |

**Integrity and attribution.**

Integrity and attribution.

| Item | Value |
|---|---|
| integrity: all | pass |
| integrity: fixed projection unchanged | pass |
| integrity: gates | pass |
| integrity: rebuilt identical | pass |
| attribution: absence of benefit | T1: `IN_CONTEXT_INFERENCE_BETTER`; T2: `IN_CONTEXT_INFERENCE_BETTER` |
| attribution: initialization | `EXPLAINS_THE_T1_CURVE` |
| attribution: routing | `FIXED_SINGLE_EXPERT_NOT_A_FACTOR; identity IDENTITY_MATTERS` |
| attribution: task feature access | `STACK_REPRESENTATION_BETTER` |

**What P5 shows.**
- **The task information reaches the specialist.** Out of sample, the 4 task coordinates are decodable linearly:
  - from the last block's channel input, with median R² 0.998;
  - from the specialist's 5-value code, with R² 0.97 (learned `P_in`).

  This holds for context-free probes and for queries in context alike.
- **R11 Stage B's specialization is real on these very groups.** With exact coordinate access and no context:
  - CGC after 8 steps is 0.079 to 0.091× the FFN alone;
  - it beats the matched GELU (0.69 to 0.80);
  - it needs its own expert: the other family's is 7 to 8× worse.
- **The stack's representation is not the bottleneck.** On the same episodes, T2's adapted stack G2 is 0.421× Stage
  B's coordinate-access G2. T1's models, which were not trained to adapt, are 1.24× worse than Stage B: that is the
  initialization, not the representation. Stage B's plan (R11's: 1,000 outer steps of 4 episodes) is smaller than
  the stack's (2,000 of 16), so part of this margin is training budget.
- **A learned `P_in` is not required.** A `P_in` fixed at its random initialization gives the same adapted loss:
  0.0038 against 0.0039 (learned/fixed 1.033, p = 0.34). Its code still carries the coordinates (R² 0.93 to 0.95).
- **The specialist's identity matters.** Substituting the other family's specialist branch makes the adapted loss
  3.6× worse (correct/misrouted 0.277). Substituting its whole expert makes it 4.4× worse (0.226). Both hold even
  though the 8 steps then adapt the substituted core.
- **There is no headroom over in-context inference.** Even with exact coordinates, the 8-step CGC is 1.85× (T1
  groups) and 2.34× (T2 groups) worse than the FFN-only stack with the support in context. The reverse comparison
  is "better" by the frozen rule in both: 0.540 and 0.428. The best adapted specialist anywhere in R12, T2's stack
  G2, only draws level with the FFN-only stack (0.982).

### 6.3 Attribution (directive P8)

| P8 factor | Finding |
|---|---|
| Initialization | **It explains T1's flat curve.** G2's adaptation criterion fails in T1 and holds in T2 (0.044× the frozen loss). It does not explain the missing advantage over the FFN-only stack: in T2 the initialization is adaptation-compatible, and the advantage is still absent |
| Task-feature access | **Not limiting.** The coordinates reach the specialist (R² 0.998 from its input, 0.97 from its code). The stack's adapted specialist beats exact coordinate access (0.421). A fixed `P_in` is as good as a learned one |
| Routing | **Not a factor in this laboratory condition**, which has one expert at weight 1. But the trained identity matters (misrouting costs 3.6 to 4.4×), so routing would matter wherever experts are learned and routed. R11 found its learned router does not separate the families |
| Demonstrated absence of benefit | **Yes, within these conditions.** The FFN-only stack's in-context inference beats the 8-step, 120-parameter CGC even under exact coordinate access (0.540 and 0.428). The best integrated specialist only draws level (0.982). On these tasks, at this scale, ERSA's in-context inference already captures what the restored update could add |

**The verdict R12 reports: `NOT_ESTABLISHED`, with the failure explained.**
- **The restored mechanism is sound.** It is exact and coherent, and it is CGC-specific: it beats the matched GELU
  wherever it was trained to adapt (0.52 to 0.64) and under exact coordinate access (0.69 to 0.80).
- **But there is no advantage to recover.** On these tasks, the FFN-only ERSA stack's in-context inference is at
  least as good as the best adaptation the 120-parameter specialist achieves.
- **Nothing was bent to get here.** No architecture was added, and no rule changed.

## 7. Cache and state coherence

**Disposition: COHERENT for the last block. Every earlier block is refused (fail closed); no HOLD was needed.**

| Check | Where | Result |
|---|---|---|
| The adapted core changes no recurrent state | Every evaluation episode of every specialist arm in all 24 groups (864 episodes): a full recomputation under the adapted core leaves the ERSA cells and convolution tails bit-identical to the frozen pass | Held in every episode |
| The cached last-block path equals full recomputation | The same episodes: maximum absolute output difference | 0.0 in every episode (the gate allows 1e-5) |
| An ERSA LM with window attention | `parity.json`: two chunks; the window K/V ring, its token history and the ERSA states under a changed last core | Bit-identical; the outputs change |
| Any earlier block | `SpecialistFastState(block=0)` | `FastStateError` (refused) |
| State versions never mix | A state carries its model digest and a version counter; `close()` re-verifies the digest; one state per episode | Tested; the interference control measures what a stale state costs |

**Why the last block is coherent.**
- Its channel runs after that block's ERSA write (and, in the LM, after the window K/V write and the evidence read).
- It feeds only `final_norm` and the readout or head.
- So nothing that any later token reads depends on the core, and the cached `pre` of the last block is exact for
  every core value.

**Why an earlier block is refused.** Block l's channel feeds every later block's input. A changed core would
change every later recurrent state for the whole history, which only an unbounded replay could rebuild. The
directive's rule applies: fail closed rather than mix state versions.

## 8. Runtime cost: latency and memory

Measured CPU wall-clock, one thread, median µs (latency.json; latency_repeat.json).

| Operation | G2 (run 1 / run 2) | G3 (run 1 / run 2) |
|---|---|---|
| One support evaluation (32 probes, cached) | 662 / 682 | 783 / 734 |
| One gradient (32 probes, cached) | 1,121 / 1,259 | 1,859 / 1,265 |
| One bounded update | 1,996 / 2,204 | 3,527 / 2,430 |
| Eight bounded updates | 25,315 / 15,485 | 18,924 / 19,270 |
| Reset | 2.3 / 2.3 | 7.2 / 6.4 |
| One adapted output (80-token episode, cached path) | 742 / 744 | 855 / 827 |
| The same, by full recomputation | 4,852 / 4,947 | 5,261 / 5,049 |
| Eight updates, each step recomputing the whole stack | 62,532 / 57,749 | 63,245 / 62,866 |
| Reference: one frozen stack forward (80 tokens) | 4,934 / 4,710 | 4,894 / 4,774 |

Bytes (float32).

| Item | G2 | G3 |
|---|---|---|
| Fast state: `ΔW` | 480 | 480 |
| Fast state: velocity buffer (stays zero; momentum 0) | 480 | 480 |
| Fast state: total | 960 | 960 |
| Support activations: the last block's `pre` for 32 probes (64 values each) | 8192 | 8192 |
| Support targets (32 × 3) | 384 | 384 |

**Notes on cost.**
- **The cached path.** It recomputes only the last block's channel and the head, so 8 updates take 15–25 ms on one
  thread. Recomputing the whole stack at every step takes 58–63 ms. One adapted output costs 0.7–0.9 ms on the cached
  path, against 4.9–5.3 ms by full recomputation.
- **Repeatability.** The two runs agree within 12% for most operations. Three differ by 30 to 40%: G2's eight
  updates, and G3's gradient and single update, on a shared CPU container. Treat them as orders of magnitude.
- **What the bytes cover.**
  - The fast state is 960 bytes: `ΔW` (480) and `bounded_update`'s velocity buffer (480, zero because momentum is
    0).
  - Not counted in those 960 bytes:
    - the support activations, 8,192 bytes per 32-probe set (the last block's `pre`);
    - the targets, 384 bytes;
    - one transient gradient, 480 bytes;
    - the identity metadata, a few strings.
  - There is no replay buffer, router state or inference cache: the routing is fixed and the cached `pre` is
    recomputed per input.
- **Training cost (T2).** A specialist arm trains in 136–142 s, against 87 s for G0: the in-loop update costs about
  1.6×.
- **No throughput claim.** This is a d64, two-block CPU harness. Nothing here predicts full-model or GPU throughput.

## 9. Packaging: sidecar and manifest

**Status: NOT PROPOSED.** The directive allows a packaging interface only for a qualified mechanism ("then—and only
then"; P9 item 12, "only if justified"). Nothing qualified, so R12 proposes no sidecar, no manifest and no
externalization. What follows is the feasibility analysis that P6 asks for. It records what a future, qualified
phase would have to specify; it is not an interface.

**What R12 implements.**
- **Nothing outside PyTorch.** There is no GGUF, llama.cpp or native code.
- **No compatibility claim.** No GGUF compatibility follows from the PyTorch tests.

**Feasibility, item by item.**

| Item | Finding |
|---|---|
| Authority | The frozen PyTorch checkpoint stays authoritative. In PyTorch, the specialist's static tensors are IN the checkpoint, and an external copy would add nothing. A sidecar would be justified only for a runtime whose base file cannot carry or execute them, such as a GGUF converted without the expert-local CGC. Then one rule applies: exactly one authority per tensor, and a runtime refuses to load both |
| Static content per adapted expert | `W0` (the base or meta-initialization) `[5, 3, 8]`, `P_in [5, H]`, `P_out [H, 3]` and, for a learned gate, `[1]`. That is `8H + 120` (+1) values: 632 float32 values (2,528 bytes) at H = 64. Static and read-only: safetensors is suitable |
| Dynamic content | `ΔW` only: 120 float32 values (480 bytes), with a zero velocity buffer (480 bytes). It lives in RAM or VRAM as `W_active = W0 + ΔW`, starts at zero, stays bounded, and is discarded on reset or close. It must never be a file: no mutable safetensors, export, periodic save, or per-token or per-update artifact (the contract, §7) |
| Manifest fields that R12's tested configuration would fill | Schema and version; base-model id and digest; adapter file and digest, if external; the tensor-name map (`blocks.{l}.sparse.experts.{e}.cgc.{core,proj_in,proj_out}.weight`); layer `l` = the last block, expert `e` = 0; H = 64, geometry 5 / 3 / 8; profile `BOUNDED_SPECIALIST` (b = 3, φ(u) = (u + u² + u³)/2); float32 base and fast state, float64 reference; gate fixed at 1, or learned; SGD with global-norm clip 2.0, lr 0.08, 8 steps, momentum 0; delta bound 1.28 per tensor; observed (input, target) pairs with a named source, context-free probes, queries never teaching; reset = exact zero; private per session; last block only, earlier blocks refused; parity: cached = full (0.0), float32 within 1e-5 of float64 |
| Validation a runtime would need | Refuse, never guess: a missing field; an unknown schema version; a digest mismatch; a shape or dtype that differs from the geometry; an expert id absent from the routing table; a non-final block; a duplicated authority. No silent reshape, cast or renumbering |
| What a GGUF runtime would still lack | A GGUF file plus a JSON manifest implements none of the mechanism. A runtime would have to provide each of these, and qualify each separately (parity with the float64 reference, latency on the target hardware, isolation): the bounded cubic operator in the declared expert; a private float32 `ΔW` per session; an exact bounded update from legitimate observed pairs; a stable routing identity; state and cache coherence (last block only); provenance, reset and zero-write enforcement |

**Verdict.** A static sidecar plus a manifest is technically straightforward, since the static part is 632 values
per expert. But it is not justified. R12 found no expert-local advantage to package:
- T1 and T2 are both `NOT_ESTABLISHED` (§6).
- The fixed-expert laboratory condition says nothing about routing identity under learned routing.

A future phase should revisit packaging only after a qualified mechanism exists.

## 10. Outstanding research gaps

1. **The integrated combination.** Neither training regime produced a stack that uses both in-context inference
   and the fast update: T1's models use only the context, and T2's lean on the update. Whether a training procedure
   could make the two complementary is untested. P5 bounds the prize: on these tasks, the update alone, even with
   exact coordinates, is about 2× worse than in-context inference.
2. **Tasks where in-context inference is weak.** R12's tasks are R97W's low-dimensional regression families, whose
   32 support pairs ERSA reads well. The restored update might matter where the context cannot carry the episode:
   longer or higher-dimensional episodes, or support that exceeds the context. This is untested.
3. **The teaching condition.** By the leakage contract, the probes are context-free. A leakage-free in-context
   teaching condition, such as leave-one-out contexts, was not built. T1 shows that the context-free correction
   conflicts with in-context inference.
4. **Learned routing among adapted experts.** It is out of scope: the laboratory condition fixes one expert. Since
   identity matters (P5), learned routing would matter too.
5. **Earlier blocks.** They are refused. Adapting them would need a replay of the recurrent state (HOLD).
6. **Language modelling.** There is no next-token teaching interface, no provenance contract for one, and no
   language-model result.
7. **Scale and hardware.** Everything is d64, two blocks, on CPU. There is no 500M, 1B, GPU or quantized-surroundings
   result.
8. **Packaging and runtime.** No sidecar is justified, and no GGUF or native runtime exists. There is nothing
   qualified to package.
9. **Statistics.**
   - Each regime has 12 groups.
   - The four primary judgements are reported uncorrected.
   - The P5 readings are diagnostic and uncorrected.
   - Stage B's training budget is smaller than the stack's.

## 11. Incidents and deviations

1. **The session branch was not at main.** It carried an old R5 commit (`fe82cc8`). `main` was merged into it, with
   `run_r5.py` resolved to main's version, before any R12 work. The branch's tree then equalled main's exactly
   (§1), so the PR contains only R12.
2. **A development record was first written outside the repository.** A background launch line lost a shell
   variable to operator precedence, so the development T2 run (`R12C_DEV`, seed 41) wrote its record and log to the
   filesystem root. Both were moved, unchanged, before the freeze. The record's digest is in the plan's development
   section, and `test_r12_records.py` checks it. No confirmation record was affected.
3. **The records test was committed with the study, not before it.** `test_r12_records.py` reads
   `study_r12.json`, which exists only after `combine`. Committing the test earlier would have made every
   intermediate commit's CI fail. It was written before the confirmation runs and added unchanged when the study
   existed, together with its P5 checks.
4. **The recorded git revision is the HEAD when each record was written.** `provenance()` runs at the end of a
   run. Records were committed one at a time while other runs were in flight, so a record names the commit of the
   record committed before it, never its own. Every such commit added only a result file. Every record carries the
   same source digests, every one of them equal to the frozen sources, and no record is `+dirty`.
5. **Development fixes, all before the freeze.**
   - The window-attention parity model first used a window smaller than one chunk plus a block, which `ErsaLM`
     refuses. It now uses chunk 16, window 24 and two chunks.
   - A float conversion of a tensor that required grad raised a warning; the value is now detached first.
   - A redundant `inference_mode(False)` wrapper was removed from `adapted_query_mse`.
6. **Nothing was rerun.** No confirmation group was repeated, and no seed was added. P5 ran once, under its frozen amendment, and every P5 record was written at the freeze commit.
7. **The session's worker process restarted after the P5 runs had finished.** All 18 P5 records had been written.
   Each was checked before it was committed: its status, its clean provenance at the freeze commit, and its
   integrity. Nothing was rerun.

## 12. Reproduction

```sh
ELPIS_REQUIRE_TORCH=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider -q \
  tests/research/elpis_torch/test_r12.py tests/research/elpis_torch/test_r12_p5.py \
  tests/research/elpis_torch/test_r12_records.py
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12 parity --out /tmp/parity.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12 reproduce --family smooth --seed 301 --out /tmp/t1.json   # about 6 min, 2 threads
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12 episodic --family smooth --seed 401 --out /tmp/t2.json    # about 9 min, 2 threads
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12 combine --out /tmp/study_r12.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12_p5 stage_b --regime T2 --seed 401 --out /tmp/p5b.json   # about 2 min
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12_p5 stack --seed 401 --out /tmp/p5c.json                 # about 12 min
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r12_p5 combine --out /tmp/study_p5.json
```

A rerun of a recorded group writes an identical record apart from its timings and git revision; the T1 records
themselves show R11's models rebuilt bit for bit.
