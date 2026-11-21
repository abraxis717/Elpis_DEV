# Elpis Torch R11 — CGC specialization: objective, training interface and causal path

**The specialty.** A compact, rapidly adaptable correction to a learned expert transformation, on one small task
family. This is the capability R97W evidences at microscopic scale. It is not episodic memory, and nothing here
claims CGC has memory.

**The expert.** `Expert(u) = SwiGLU(u) + g · CGCTransform(u)` (the repository's `ChannelExpert`).
- The FFN keeps general learned computation.
- The CGC branch reads a 5-dimensional projection of the expert's input and writes a 3-cell correction through
  `P_out`.

## 1. Two modes, never conflated

| Mode | What may change | Status |
|---|---|---|
| **A: explicit adaptation** (research only) | Within an episode, the 120 core parameters, by explicit SGD on observed support examples (8 steps, lr 0.08, global L2 clip 2.0). Base weights are untouched: the R0 session's `W_base + ΔW` through `functional_call`. The adapted tensors are transient and discarded after each episode | An OFFLINE scientific procedure. It authorizes no runtime weight update |
| **B: frozen-weight inference** | Nothing. Trained weights are fixed; the branch can respond to context only through its input activations | The k = 0 value in Stages A and B; Stage C in the real stack |

If a Mode A benefit disappears in Mode B, the result is `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES`. That is
information, not a failure of the microscopic result. No runtime learning mechanism is added to avoid it.

## 2. The training interface (Stage B)

| Question | R11's answer |
|---|---|
| The activation given to `P_in` | The expert's input `u = Q_task·x + Q_nuisance·ξ`, a 64-dimensional hidden state. `x = [z, 1]` (the task, 5-dimensional) occupies a 5-dimensional subspace; per-example Gaussian nuisance fills the rest (unit RMS on average) |
| The support examples | 32 (input, target) pairs of one episode, from R97W's teacher (smooth or conditional family) under R11's own namespaces |
| The observed targets | The episode's 3-dimensional targets, read from the expert output through a fixed orthonormal readout `R` (64 → 3) that stands for the rest of the model |
| Adapted in the inner loop | The specialist core only: `cgc.core.weight` (120); for the GELU control, its 120 network parameters |
| Fixed within an episode | FFN, `P_in`, `P_out`, gate, readout |
| How the adapted specialist changes the output | `out = FFN(u) + g · P_out(core_adapted(P_in u))`, then `R · out` |
| The residual-scale bound | The `b = 3` pre-activation bound in `BOUNDED_SPECIALIST`; `P_out` initialized to unit readout gain (§4) |
| State persisting between tokens | None: the expert is a per-token map; ERSA owns recurrent state |
| How routing decides evaluation | Stage B uses a fixed, known assignment (one expert per family): a LABORATORY control. Learned-router selection is tested only if a specialist qualifies, and never with oracle routing at inference |

**The outer (offline) training.**
- The objective is first-order MAML: the query loss after the inner loop trains every parameter, the core through
  its adapted value.
- Adam, lr 3e-3, cosine to 10%, 1,000 outer steps of 4 episodes.
- Every arm keeps its best checkpoint by DEVELOPMENT episodes, under the same rule.
- Learned projections are disclosed capacity: the adapter's 8H parameters beyond the 120 core parameters.

## 3. The arms and what each one isolates (A1)

| Arm | Branch | Gate | Objective | Isolates |
|---|---|---|---|---|
| G0 | none | — | pooled supervised | the FFN alone (reference) |
| G1 | micro CGC | learned scalar, init 0.1 (historical) | specialist (FOMAML) | the historical mechanism |
| G2 | micro CGC | fixed `g = 1`, no parameter | specialist | an engaged cubic branch |
| G3 | GELU 5 → 13 → 3 (120) between the same projections | fixed `g = 1` | specialist | cubic-specific utility against a matched ordinary specialist |
| G2_POOLED | micro CGC | fixed `g = 1` | pooled supervised (no inner loop in training) | whether the specialist OBJECTIVE matters |

**Further controls.**
- **Causal ablation.** Remove the adapted branch after training.
- **Specialist swap.** The other family's specialist on this expert's FFN.
- **Expert misrouting.** The other family's whole expert.
- **Frozen value.** k = 0.

**Diagnostics.**
- The gate's trajectory.
- Branch RMS against FFN RMS.
- The effective contribution.
- Gradient norms of core, `P_in`, `P_out`, gate and FFN.
- `‖P_out‖`, which can suppress a branch even with `g ≡ 1`.
- Pre-activation saturation.
- Non-finite counts.
- Relative parameter change.

## 4. The residual-scale calibration (A1), fixed before any Stage B data

**Why the default does not work.** With the repository's default initialization, `P_out` has the residual-write
rule (std 0.014 at one layer). The specialist's inner-loop gain is then about 0.1, and the cell offsets dominate.
In development, k = 8 equalled k = 0: the branch could not adapt at all.

**The rule adopted.** The branch takes Stage A's projected-adapter initialization:
- `P_in` std H^-1/2;
- the core at R97W's scale (cubic: 0.17/√5; GELU: R97W's FFN rule);
- `P_out` std 3^-1/2, which gives unit gain through the orthonormal readout.

**Scope.** The rule applies identically to G1, G2, G3 and G2_POOLED. G1's historical gate initialization (0.1) is
kept, so G1 against G2 differs only in the gate. No other knob was tuned on Stage B data.

## 5. The causal path in the full stack

**Within a block.** The channel runs after the block's ERSA and attention (`CGC_BOUNDING_R11.md` §6). An
expert-local CGC in block `l` therefore:
- can shape the inputs of blocks `l+1 …`;
- cannot shape block `l`'s own recurrent write;
- in the last block, can shape only the readout.

**Consequence for Stage C.** A Stage C test of frozen inference must therefore place the specialist where its output
reaches the prediction, and must credit the ERSA and attention state to ERSA and attention, never to the specialist.

## 6. What makes the specialization testable

A branch is not specialized because it is named CGC. Specialization has to be visible in all of these:
- **Its trained function.** It adapts within an episode.
- **Its input distribution.** `P_in` reads the task subspace out of the hidden state.
- **Its objective.** G2 against G2_POOLED.
- **Its task-conditioned behaviour.** The specialist swap and expert misrouting.
- **Its causal effect.** The ablation.
- **Its measured cost.** Exact parameters and measured dispatch time.

If the chosen task does not teach a distinctive function, the claim is not established.
