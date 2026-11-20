# Elpis Torch R11 — CGC capability contract: what each piece of evidence establishes, and under which conditions

This is the audit R11 builds on. Every row names the conditions under which a result holds. A result is not
evidence outside those conditions, and nothing here transfers one mechanism's evidence to another.

| Mechanism | Experience-dependent inference state | Qualified behaviour | Not shown |
|---|---|---|---|
| **E0 / `CGCCore`** | None intrinsic | Microscopic differentiable cubic computation | Memory, meta-learning without a procedure, retention, attention, retrieval, reasoning, specialization without training |
| **`CGCTransform`** | None intrinsic | A projected cubic transformation; its mathematics and mechanics (tests) | Any capability beyond its trained function |
| **R97W** | Episode-local adapted weights, under an explicit gradient procedure | Bounded microscopic meta-learning | Frozen-weight in-context learning; a property of the stateless forward pass |
| **K1** | Explicit adaptive state `(W, epoch, H, a)` | Synthetic conflicting-experience retention and consolidation | Anything about a stateless CGC expert |

## 1. ECS_G E0: the mathematical primitive

**The operator.**
- `f_W(x) = Σ_i φ(xᵀ w_i)`, with `φ(z) = (z + z² + z³)/2` and `φ′(z) = 1/2 + z + 3z²/2`.
- One scalar cell with input dimension `d` and `N` units has `d × N` core parameters.
- Several independent cells may share one input.

**The reference.**
- Upstream: `abraxis717/CGC` `src/cgc/ecsg_e0.py`, SHA-256
  `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084`.
- Verified in this phase against the clone at commit `0895c2b`.
- Vendored byte for byte as `research/elpis_torch/upstream/cgc_ecsg_e0.py`, with its digest checked by the R0
  tests.

**Qualification in this repository.**
- `CGCCore` with `preact_bound=None` equals E0 cell by cell. On R11's micro geometry (5 inputs, 3 cells, 8 units)
  the R11 parity record measures:
  - 1.4e-14 forward;
  - 7e-15 between autograd and the analytic gradient;
  - within the 1e-12 tolerance.
- With a learned input projection, a cell remains a factorized E0 operator on the original hidden input (tested
  since R0).

**Not E0.**
- The output projection `P_out` is new architecture.
- With `preact_bound` set, the operator is **E0-derived, not E0**. R11 quantifies the departure at `b = 3`: a 0.6%
  relative cell gap at pre-activation RMS 0.08, 29% at 0.4, 71% at 0.8.

**What E0 does not provide by itself:** episodic memory, meta-learning without an explicit learning procedure,
retention across conflicting experiences, attention, context retrieval, autonomous reasoning, or expert
specialization without training.

## 2. R97O: differentiable multistep computation

**Source.** Upstream `evidence/phase97o_ecsg_e0_differentiable_unroll/run_YwHfIlCG/result.json`, SHA-256
`edc2aebc97de29d8fc6208eceee725643f4aaca5f44f3e6a5e8ad09743824207`.

**Result.** Status `ECSG_E0_DIFFERENTIABLE_MULTISTEP_UNROLL_PARITY_PASS_NO_NATIVE_STATE_OR_EXPERT_CLAIM`.
- 72/72 test cases were within tolerance: 27 at 2 steps, 27 at 4, 18 at 8.
- They span four groups: negative, random, sparse and zero.
- The largest errors are 6.1e-12 on the meta-gradient and 4.4e-16 on the trajectory forward.

**Supported.** Differentiable microscopic adaptation mechanics: the exact E0 gradient-descent update, unrolled.

**Not supported.** Transformer-scale learning, runtime consolidation, or language competence. The claim bound
excludes stateful C, K1, expert routing and persistence.

## 3. R97W: microscopic meta-learning

**Source.**
- Upstream `research/benchmarks/r97w/evidence/run_20261025T191037_af40e7f2/result.json`, SHA-256
  `576922cb74ba4aa13f824ac399a389bb215e12cc4e4b936315a279386169c0db`.
- The portable port is `research/benchmarks/r97w/portable`.

**Conditions.**

| Property | CGC | FFN control |
|---|---|---|
| Inputs | 4 features + the constant 1 = 5 | 5 |
| Outputs | 3 | 3 |
| Units | 8 per output | width 13 (GELU) |
| Trainable parameters | 3 × 5 × 8 = 120 | 13·5 + 13 + 3·13 + 3 = 120 |
| Learning | first-order Reptile (interpolation 0.28) | the same |
| Episode adaptation | 8 SGD steps, lr 0.08, global L2 clip 2.0 | the same |

- Smooth and conditional task families.
- 64 training episodes per (family, seed).
- Seeds 17 and 83.
- 12 held-out episodes per group.
- Evaluation after 0, 1, 2, 4 and 8 steps.
- Cold, shuffled-history, continual-learning and fast-state-reset controls.
- Float64.

**Result.** Disposition `SCIENTIFIC_PASS_BOUNDED_POST_FORGETTING_META_LEARNING_ECS_NO_CAUSAL_EMERGENCE_CLAIM`.
- CGC beat the matched FFN in **48/48** paired held-out evaluations.
- Median final query MSE, CGC against FFN:

  | Group | CGC | FFN |
  |---|---|---|
  | smooth, seed 17 | 0.042 | 0.074 |
  | smooth, seed 83 | 0.051 | 0.098 |
  | conditional, seed 17 | 0.030 | 0.059 |
  | conditional, seed 83 | 0.045 | 0.080 |

  That is 43–48% lower.

**Supported.** Parameter-efficient episodic meta-learning under explicit gradient-based adaptation.

**Not supported.**
- An attention-free language model.
- Meta-learning from a stateless forward pass.
- Universal superiority over FFN.
- Matched GPU latency or FLOPs.
- Automatic transfer to a projected transformer `CGCTransform`.
- Frozen-weight in-context adaptation.

**Provenance.** The outer Reptile initialization is learned offline; held-out adaptation uses explicit inner
gradient steps. The portable benchmark is a derived reproduction, not the frozen original package, whose spec,
source and dev records were not published. R11 re-implements the teacher from the portable specification under its
own namespaces: a new experiment, never a re-run of R97W's held-out episodes.

## 4. K1: separately qualified stateful retention

**Sources.** `docs/research/ECS_RETENTION_R3_RESULTS.md` and `docs/research/ECS_K1_NATIVE_RESULTS.md`.

**State.** K1 carries explicit state `(W, epoch, H, a)`, in which `W` is mutable adaptive state.

**The R3 result.**
- 32/32 conflicting-experience worlds retained, against 0/32 for plain G1.
- Median earlier-experience NMSE 0.0013.
- A bounded extra state of 28,552 bytes in that experiment's configuration.
- Removing the state causes forgetting, a reset causes the registered degradation, and a transplant restores the
  continuation bitwise.

**The native qualification.** K1N-v2 qualified the native implementation under its registered decision-level and
local numerical contracts; K1N-v1 stays NOT_QUALIFIED.

**The boundary.**
- K1 is not the E0 forward pass and not `CGCTransform`.
- Its retention depends on its complete state transition and consolidation law.
- It is not attributable to any CGC expert.
- **R11 does not implement K1 anywhere.** A future K1 integration is a separate phase. It would have to specify:
  - where `(W, epoch, H, a)` lives;
  - which observations are admitted;
  - how the qualified update executes;
  - whether a mutable volatile `W` is permitted under the frozen-checkpoint runtime contract;
  - the state size at transformer width;
  - how K1's output reaches expert computation and future recurrent state;
  - whether its retention survives that interface;
  - causal controls against ERSA and attention.

## 5. Elpis R0/R1: the existing session-local fast weights

**The mechanism** (`research/elpis_torch/session.py`).
- `W_active = W_base + ΔW_session`, applied to every `cgc.core.weight` through `torch.func.functional_call`.
- The checkpoint is never written.

**R0 established** the mechanics:
- bounded updates (gradient clip and a delta-norm ball);
- refusal and rollback;
- session isolation;
- provenance-aware targets (only observed tokens teach);
- cache coherence;
- no persistent checkpoint writes;
- positive toy continuation-adaptation tests.

**R1 compared full-stack and last-block CGC updates.** On its transduction task, with 16 observed support
trajectories and 16 updates at lr 0.2:
- last-block variants reached 0.67–0.76 accuracy on changed transitions;
- full-stack variants reached 0.90–0.92;
- the two differ in interference and cache cost (`EVIDENCE_R1.md` §5).

**How R11 uses it.** The inner loop of R11's Stage B harness adapts only the specialist's core through
`functional_call` with base weights untouched: the R0 convention.

**What it is not.**
- It is not R97W's Reptile protocol.
- It is not K1's consolidation law.
- It is not a qualified sparse-expert adaptation path; it targets the earlier dense model.
- It does not authorize inference-time weight mutation in the intended runtime.

## 6. R9 and R10: what the transformer-scale CGC experiments found

**R9** (`EVIDENCE_R9.md`, `CLAIM_AUDIT_R9.md` L13).
- **Verdict.** `CGC_NOT_ESTABLISHED` for static next-byte prediction at d 64 against parameter- and
  compute-matched FFN width.
- **Scope.** `ASSAY_CANNOT_TEST_META_LEARNING_CLAIM`: R9 neither establishes nor refutes an adaptation benefit.

**R10** (`EVIDENCE_R10.md`, L16).
- **Adaptation:** `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`. The CGC arms adapted worse.
- **Retention:** `ASSAY_NOT_LEARNED`. That disposition stands and is not re-scored here.
- **The gate confound (A1).**
  - R10's trained CGC gates settled near 0.05–0.08, and zeroing them moved accuracy by at most 0.028.
  - R10 therefore established that its trained CGC configurations did not help.
  - It did not isolate the value of an engaged cubic branch. R11's fixed-gate arm (G2) exists for that.

**The retention finding, stated plainly (A2).** At d 64 the tested transformers **learned regime-conditioned
bindings but did not reliably preserve earlier bindings through conflicting interference**:
- Pre-interference accuracy was often above 0.90.
- The best aggregate post-interference score was 0.173, below the 0.20 admission floor.
- An attention reset removed most within-window retention.
- An ERSA reset removed much of the surviving behind-gap retention.
- Ablating CGC had little effect.

This identifies a weakness of the tested architectures, not a failure to learn the task. It does not show that
every attention/ERSA configuration forgets, nor that K1 would solve the task if integrated. The retention that
existed is causally attributable to attention or ERSA, not to CGC.

## 7. What follows for R11

- The positive evidence is microscopic, explicit-adaptation evidence: E0 and R97W.
- Expert-local CGC is the place to test whether that behaviour survives the transformer adapter's projections and
  bound (Stage A), and its integration inside the real expert with its FFN (Stage B).
- Frozen-weight inference (Mode B and Stage C) is a separate question. A gain under offline inner updates does not
  establish it.
- Retention under conflicting interference belongs to K1. It is outside R11.
