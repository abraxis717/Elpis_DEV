# Elpis Torch R11 — preregistration: a bounded, specialized expert-local CGC

**Status.** FROZEN before any confirmation run.
- **The plan.** `research/elpis_torch/results/r11/preregistration.json` (SHA-256 `a7df440a779b0e5154dabacec4968e45588cb3beeb0cd1be2a42c339a2edc198`) is the
  machine-readable plan; `run_r11.frozen()` refuses a plan that drifted from the runner.
- **Every confirmation record carries** the SHA-256 of the plan and of the runner, the task and harness module, the
  profile module, `sparse_channel.py`, `cgc.py`, `model.py`, `config.py` and the vendored upstream E0.
- **`combine` refuses** a record whose digests differ.
- **Before the freeze,** only development records exist: namespaces `R11A_DEV` and `R11B_DEV`, seeds 11, 12 and 21
  (§7).

**Base.** `main` at `d2d683ad874dda6312e27f87a8a2f1b0a64f6096`, post-merge CI run 44795525346 at 16/16. Branch
`claude/r11-cgc-specialist`.

## 1. Question

Does a correctly bounded, explicitly specialized expert-local `CGCTransform`, inside the existing FFN expert:
- keep R97W's microscopic adaptive behaviour under the transformer adapter's projections and bound;
- make a causal, specialized and cubic-specific contribution to its expert;
- and transfer any of that to frozen-weight inference?

## 2. The specialist profile

- **Geometry.** Bottleneck 5, 3 cells, 8 units: a 120-parameter core, inside an adapter of `8H + 121` parameters.
- **Budget.** At most 5% of the expert's SwiGLU parameters.
- **Profiles.**
  - `E0_REFERENCE` (no bound: exact E0).
  - `BOUNDED_SPECIALIST` (b = 3: E0-derived).
- See `CGC_BOUNDING_R11.md`.

## 3. Stage A: the microscopic reference (fresh R97W-protocol episodes)

**Protocol.** R97W's portable protocol, re-implemented under R11's own namespace `R11A`:
- support 32, query 48, OOD 32 (inputs ×1.4);
- 8 inner SGD steps, lr 0.08, global L2 clip 2.0;
- first-order Reptile at interpolation 0.28 over 64 training episodes;
- 12 held-out TEST episodes per group, evaluated at k = 0, 1, 2, 4 and 8;
- a forget probe on the last 8 training episodes;
- float64.

**Groups.** Smooth and conditional families × seeds 101–106: 12 groups, 144 paired episodes per comparison.

**Arms.** All share each group's episodes:

| Arm | Core | Projections (H = 64) | Bound | Outer |
|---|---|---|---|---|
| E0_META / E0_COLD / E0_SHUFFLED | E0, 120 | none | none | Reptile / none / Reptile on shuffled support targets |
| GELU_META / GELU_COLD | GELU 5 → 13 → 3, 120 | none | — | Reptile / none |
| PROJ_E0_FIXED_META, PROJ_E0_COLD | E0 | fixed random | none | Reptile / none |
| PROJ_E0_LEARNED_META | E0 | learned (one clipped SGD step per episode, lr 0.1) | none | Reptile |
| PROJ_GELU_FIXED_META, PROJ_GELU_COLD | GELU | fixed random | — | Reptile / none |
| PROJ_GELU_LEARNED_META | GELU | learned (lr 0.3) | — | Reptile |
| PROJ_B3_FIXED_META, PROJ_B3_COLD | E0 | fixed random | 3 | Reptile / none |
| PROJ_B3_LEARNED_META | E0 | learned (lr 0.3) | 3 | Reptile |

**What adapts where.**
- **Projected adapter.** `y = R · P_out · core(P_in · E · x)`, with a fixed embedding `E` and readout `R`.
- **Inner loop.** It adapts ONLY the 120 core parameters.
- **Projection learning rates.** These were chosen by the development rule (`results/r11/calibration.json`): per
  core type, the grid value in {0.01, 0.03, 0.1, 0.3} with the lowest development k = 8 query loss.
- **Extra capacity.** Learned projections add 8H outer-loop parameters, accounted separately in every record.

**Primary metric.** Held-out query MSE after 8 inner steps.

## 4. Stage B: the specialist inside the real `ChannelExpert`

**The arms.** G0, G1, G2, G3 and G2_POOLED, as in `CGC_SPECIALIZATION_R11.md` §3.

**The setting.**
- d 64, FFN 86, the micro specialist.
- A hidden state carrying a 5-dimensional task subspace plus nuisance.
- A fixed readout.

**Groups.** Seeds 201–206 × both families, under namespace `R11B`.

**Training.** First-order MAML (G1, G2, G3) or pooled supervised (G0, G2_POOLED):
- Adam lr 3e-3, cosine to 10%, 1,000 outer steps of 4 episodes;
- per-arm checkpoint selection on 5 DEV episodes every 100 steps;
- the branch initialized by Stage A's adapter rule (the A1 residual-scale calibration).

**Profile.** `BOUNDED_SPECIALIST`. If Stage A finds that the bound destroys the behaviour (§6), Stage B uses
`E0_REFERENCE`. `stage_b` reads Stage A's records to apply this rule.

**Evaluations,** on 12 TEST episodes per group:
- k = 0 … 8 (Mode A), with k = 0 as Mode B;
- the branch ablated;
- the specialist swapped (the other family's);
- the expert misrouted (the other family's).

**Diagnostics.**
- The gate.
- Branch RMS against FFN RMS.
- Gradient norms by component.
- `‖P_out‖`.
- Saturation.
- Parameter change.

## 5. Stage C: frozen-weight transfer into the real stack

- **Trigger.** `EXPERT_SPECIALIZATION_QUALIFIED` for G1 or G2 in Stage B.
- **Procedure.** If triggered, Stage C gets its own frozen amendment before any Stage C data. Otherwise it is
  NOT_RUN, with the reason.
- **Routing.** Learned-router selection is tested only if a specialist qualifies. Oracle routing never counts as
  learned routing.

## 6. Decision rules (frozen)

**"X better than Y"** must meet all three conditions:
- the pooled geometric-mean ratio of X/Y over all paired episodes is ≤ 0.95;
- a one-sided exact sign test on the paired log ratios gives p < 0.05;
- at least 2/3 of the (family, seed) groups have a geometric mean below 1.

**Stage A.**
- **`MICRO_REFERENCE_QUALIFIED`.**
  - E0_META against E0_COLD meets R97W's four utility criteria in EVERY group: geometric mean ≤ 0.95, at least
    2/3 of episodes better, median AUC ratio ≤ 0.95, and median forget-probe reset/fast ratio ≥ 1.1.
  - The parity record shows E0 ≤ 1e-12.
- **`PROJECTED_ADAPTATION_QUALIFIED`,** per projected CGC arm: the same utility, against its own cold arm, in every
  group.
- **Comparative findings** (reported, the "better" rule):
  - E0 against GELU;
  - projected E0 or bounded against projected GELU, with the same projection procedure;
  - projected against microscopic.
- **`bound_destroys_behavior`.** The bounded arm fails utility where the unbounded passes, or its pooled k = 8 loss
  exceeds the unbounded's by more than 10%.

**Stage B,** for an arm X ∈ {G1, G2}:

| Criterion | Definition |
|---|---|
| assay valid | G0's median k = 0 query MSE ≤ 0.90 × the zero predictor's, in every group. Otherwise `ASSAY_NOT_LEARNED` |
| adaptive benefit | X better than G0, with pooled ratio ≤ 0.80 |
| causal | pooled geomean(ablated / intact at k = 8) ≥ 1.25 |
| specialist objective | X better than G2_POOLED |
| CGC-specific | X better than G3 |

The criteria combine into these dispositions:

| Condition | Disposition |
|---|---|
| All five criteria met, within the budget | **`EXPERT_SPECIALIZATION_QUALIFIED`** |
| Benefit, causal and objective met, but not better than G3 | `SPECIALIST_SLOT_FUNCTIONAL_NO_CGC_SPECIFIC_ADVANTAGE` |
| Otherwise | `NOT_ESTABLISHED` |

**Gate (A1).**

| Condition | Disposition |
|---|---|
| G2 slot-functional and G1 not | `GATE_IDENTIFIABILITY_FAILURE` |
| Both | `GATE_NOT_LIMITING` |
| Neither | `SPECIALIST_OPERATION_NOT_ESTABLISHED` |

**Mode B.**

| Condition | Disposition |
|---|---|
| X better than G0 at k = 0 | `FROZEN_BENEFIT` |
| Otherwise, if slot-functional | `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES` |

**Independence of stages.** Passing an earlier stage never passes a later one. A mechanical PASS is never promoted
into a specialist PASS.

## 7. Development before the freeze (kept, labelled DEVELOPMENT)

**What was run.** Namespaces `R11A_DEV` and `R11B_DEV`, seeds 11, 12 and 21. No confirmation seed, namespace or
episode was used.

**What it decided.**
- **The projection learning rates.** By the rule in §3.
- **The A1 calibration** (`CGC_SPECIALIZATION_R11.md` §4). It was found necessary because the default `P_out`
  residual initialization gave the specialist an inner-loop gain near 0.1, so it could not adapt.
- **Per-arm checkpoint selection.** Without it, the training length alone decided the cubic-against-GELU
  comparison: at 300 steps the adaptive arms were about equal; at 1,000 the GELU specialist degraded.

**Observed in development, kept and not evidence.**
- The fixed-gate CGC, the learned-gate CGC (once its gate had grown to about 0.46) and the GELU specialist all
  adapted strongly.
- The pooled-objective CGC barely adapted.
- At k = 0 the meta-learned CGC initializations were far worse than the FFN alone.

The records are `results/r11/dev_*.json` and `results/r11/calibration.json`.

## 8. Stopping rules

The following are not allowed:
- a rerun of a confirmation record;
- added seeds;
- a threshold or endpoint changed after results;
- retuning the bound or the geometry to rescue a result;
- a second specialist if the first fails;
- a runtime weight update.

A failed run is recorded as failed. If the bound destroys the behaviour, R11 reports it and keeps the unbounded
reference. If routing or training semantics prevent specialization, R11 names the failure and stops there.
