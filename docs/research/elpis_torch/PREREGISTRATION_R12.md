# Elpis Torch R12 — preregistration: restoring the CGC fast update inside the Stage C stack

**Status.** FROZEN before any confirmation record.
- **The plan.** `research/elpis_torch/results/r12/preregistration.json`, SHA-256
  `f0f106067f1672a3dd4b5fa17f0f51543d2eca4753dc4e48080cdeed4552d44c`. `run_r12.frozen()` refuses a plan that
  drifted from the runner, the fast-state defaults, the stack constants or R11's Stage C amendment.
- **Every confirmation record carries** the SHA-256 of the plan and of every source it ran. These are the R12
  modules, `session.py`, `lastblock.py` and the R11 stack, tasks, profile, channel, CGC, model, config and ERSA
  sources, plus R11's Stage C runner.
- **`combine` refuses** any record whose digests differ.

**Base.** `main` at `72b66b1f2c069ff331bf39992bec47f31efdc377` (R11 merged; post-merge CI run 44850567035
succeeded). Branch `claude/funny-turing-obbdy5`.

## 1. Question

Inside R11's actual Stage C ERSA stack, does restoring the exact episode-local update of the LAST block's
120-parameter specialist core recover CGC's specialist advantage?
- **The update.** R11 Stage B's: 8 clipped SGD steps on the observed support.
- **The baselines.** The FFN-only stack, and a matched fast-adapted GELU specialist.

## 2. What is held fixed

| Element | Value |
|---|---|
| Stack | R11's `RegressionStack`, unchanged: d 64, 2 ERSA blocks (4 heads, 16 × 16 state, convolution 4), one routed expert per block (FFN 86 + specialist), linear input map and readout. Arms G0 (FFN only), G1 (CGC, learned gate, init 0.1), G2 (CGC, fixed `g = 1`), G3 (matched GELU, `g = 1`) |
| Episode | R11's teacher families (smooth, conditional). 32 support pairs in context as `[x, y, 0]`, then 48 queries `[x, 0, 1]` |
| Fast update | `fast_state.SpecialistFastState` on the LAST block's specialist core only. The update is `session.bounded_update`: lr 0.08, global gradient-norm clip 2.0, 8 steps, no momentum, delta bound 1.28 (unreachable within 8 steps), no loss-increase rejection |
| Teaching signal | The episode's 32 observed support pairs, presented as context-free probes `[x, 0, 1]`. A probe carries no target and has no context from which to read one. The targets enter only the support loss. Queries never teach |
| Budget match | G3 adapts its own 120 core values under the identical update |

## 3. Two training regimes, never pooled

| Regime | Models | Seeds | Evaluation episodes |
|---|---|---|---|
| **T1** (the decisive test as specified: "use the existing Stage C stack and checkpoint") | R11's Stage C models, rebuilt with R11's own code and arm order. Each is verified row by row against the committed R11 record; a mismatch makes that group NONQUAL. These models were never trained for adaptation | R11C, 301–306 (R11's recorded seeds; no checkpoint was archived) | 12 FRESH episodes per group (split `R12TEST`, never used by R11 or by development) |
| **T2** (does an adaptation-compatible initialization exist in this stack?) | The same stack and R11's training plan (Adam 3e-3, cosine, 2,000 steps of 16 episodes, k from the grid, DEV selection at k = 32 every 250 steps), with ONE change: a specialist arm's query pass runs under its last-block core adapted on that step's probes. The outer gradient is first order (R11 Stage B), and DEV selection scores after adaptation. G0 trains with R11's `train_stack` | R12C, 401–406 (fresh) | 12 TEST episodes per group |

The training objective of T2 differs from R11 Stage C's; this is disclosed. The architecture and the adaptation
algorithm are identical across regimes.

## 4. Conditions and controls (per evaluation episode)

All of the following use the 32 support pairs in context unless stated:

| Measure | Definition |
|---|---|
| FFN-only | G0, frozen |
| Frozen CGC (or GELU) | The specialist arm with no core update (0 steps) |
| **Fast CGC (or GELU)** | After 1, 2, 4 and **8** bounded steps on the probes |
| Shuffled targets | 8 steps on the same probes with the support targets permuted. The context is unchanged |
| Reset before queries | 8 steps, then `reset()`. Must equal frozen bit for bit |
| Updates disabled | The identical pipeline with learning disabled. Must equal frozen bit for bit, every attempt refused |
| Context-free queries | Fast-only: the queries without the support context (secondary) |
| OOD queries | Inputs × 1.4 (secondary) |
| Interference | This episode's queries under the PREVIOUS episode's adapted core (secondary) |
| Reliance | The last block's specialist ablated, frozen (secondary) |
| Diagnostics | Gradient norms, clipping, successive-update cosine, support loss per step, R² of `P_in h` on the probe inputs |

## 5. Decision rules (frozen)

**"X better than Y"** has three conditions:
- the pooled geometric-mean ratio of X/Y over all 144 paired episodes (12 groups × 12) is ≤ 0.95;
- a one-sided exact sign test on the paired log ratios gives p < 0.05;
- at least 2/3 of the (family, seed) groups have a geometric mean below 1.

**Primary endpoint.** Query MSE with the support in context after 8 fast steps (`fast8`).

**`INTEGRATED_FAST_CGC_SPECIALIST_QUALIFIED`.** Judged per regime and per gate mode: G2 and G1 separately, with no
pooling. All six criteria must hold:

| Criterion | Rule |
|---|---|
| Assay learned | G0's median frozen query MSE ≤ 0.5 × the zero predictor's, in every group |
| Beats FFN-only | arm `fast8` better than G0 |
| Beats fast GELU | arm `fast8` better than G3 `fast8` |
| Adaptation curve | arm `fast8` better than the same arm frozen |
| Target-driven | arm `fast8` better than the same arm on shuffled targets |
| Integrity | every gate of §6 holds for the arm AND for G3 |

**Otherwise:**
- `NOT_ESTABLISHED`, with the failing criteria recorded;
- `ASSAY_NOT_LEARNED`;
- `NONQUAL_INTEGRITY`, if a gate fails;
- `NONQUAL`, if a record is missing.

**Multiplicity.** Four primary judgements (2 regimes × 2 gate modes), reported separately and uncorrected. A
qualification is stated only for the arm and regime that meet every criterion, and only within these conditions.

## 6. Integrity gates (every evaluation episode of every specialist arm)

1. Reset before the queries restores the frozen output bit for bit.
2. Updates disabled leave the frozen output bit for bit, and every attempt is `REFUSED_LEARNING_DISABLED`.
3. No probe token carries a target, and no query input equals a support input.
4. The recurrent states of a full recomputation under the adapted core equal the frozen states bit for bit.
5. Cached and full-recomputation outputs agree within 1e-5.
6. Every update is APPLIED, the bound never binds, and every value is finite.

## 7. Reading the two regimes

| Reading | Condition |
|---|---|
| `DIRECT_RESTORATION_QUALIFIED` | A T1 arm qualifies: restoring the update suffices |
| `QUALIFIED_ONLY_WITH_ADAPTATION_TRAINED_INITIALIZATION` | No T1 arm qualifies, but a T2 arm does. The mechanism works in the stack, but the Stage C models lacked an adaptation-compatible initialization |
| `NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED` | No arm qualifies in either regime. The representation stage (directive P5) then runs only under its own amendment, frozen before its data. Oracle task coordinates are diagnosis only, never a claimed path |
| `NONQUAL` | Integrity failure or a missing record |

## 8. Development (labelled DEVELOPMENT ONLY; kept as `results/r12/dev_*.json`)

**What was run.**
- **T1** on `R11C_DEV` seed 31, smooth, a Stage C model trained by R11's code, evaluated on split `R12DEV`.
- **T2** on `R12C_DEV` seed 41, smooth.
- No confirmation namespace, seed or split was used.

**Before that, two checks:**
- the runner rebuilt R11's smooth/301 group bit for bit (rows, traces, selected steps, gates);
- the parity checks passed (`run_r12 parity`, recorded at confirmation).

**What was observed** (one group each; not evidence):
- **T1, direct adaptation, barely moves the models.** G2 goes 0.0121 → 0.0122 at 8 steps, G1 0.0119 → 0.0117 and
  G3 0.0073 → 0.0072. The support-loss gradients are small (norm 0.02–0.08, never clipped).
- **T2 shows adaptation curves.**
  - G1 goes 0.093 → 0.0029, G2 0.078 → 0.0050 and G3 0.067 → 0.0131.
  - FFN-only scores 0.0029.
  - Shuffled targets give 0.043 for G1 and 0.022 for G2.
- **All integrity gates held.**
- **Cost.** About 5.5 minutes per T1 group and 8 minutes per T2 group, at 2 threads.

**What it changed.** Nothing in the plan. It checked mechanics, integrity and cost only.

## 9. Stopping rules

- No confirmation record is rerun, and no seed is added.
- No threshold or endpoint changes after a result.
- The fast-update protocol, the bound and the geometry are not changed to rescue a result.
- No K1, LAO, RALPH, larger core, additional expert, memory tier, or native or GGUF code.
- A failed run is recorded as failed.
- If a blocker would require changing ERSA's recurrence or replaying unbounded history, the disposition is HOLD.
