# Elpis Torch R11 — CGC bounding: parameters, numerics and placement

**Source of every number.** Exact counts come from the tensors (`run_r11 audit`, meta tensors, nothing allocated;
record `results/r11/audit.json`). Numerical figures come from `run_r11 parity` (`results/r11/parity.json`).

## 1. The problem: the CGC branch was a second large network, not a specialist

**What the R6 scale layouts contain.** These are the `run_r6.scale_model` configurations. CGC sits in every dense
`ffn+cgc` channel, in every routed and shared expert, and, unlisted until now, in the R5 evidence encoder.

| Layout | CGC instances (dense / routed / shared) | Core-model CGC parameters (gates included) | Without gates | R5 evidence-encoder CGC | Share of the model |
|---|---|---|---|---|---|
| R6 500M | 90 (10 / 80 / 0) | 144,506,970 | 144,506,880 | 851,970 | 28.6% of 507,683,413 |
| R6 1B | 162 (9 / 144 / 9) | 422,314,146 | 422,313,984 | 851,970 | 42.5% of 994,692,069 |

**By component.** Including the encoder, the parameters split as follows:

| Layout | `P_in` | Cubic core | `P_out` | Gates |
|---|---|---|---|---|
| 500M | 14,811,136 | 71,303,168 | 59,244,544 | 92 |
| 1B | 44,892,160 | 198,705,152 | 179,568,640 | 164 |

**Active per token.** CGC is active for 64,880,670 parameters per token at 500M and 125,042,724 at 1B: every dense
CGC plus the two selected experts' CGC, plus the shared expert at 1B.

The directive's preliminary figures (144,506,880 and 422,313,984) are reproduced exactly. The 851,970 encoder CGC
parameters are an additional placement that those figures excluded.

## 2. The micro profile (`research/elpis_torch/cgc_specialist.py`)

**Geometry.** R97W's: bottleneck `d = 5`, 3 cells, 8 units, so the core holds 5 × 3 × 8 = **120** parameters.

**The adapter around it.**
- `P_in`: hidden → 5, giving 5H parameters.
- `P_out`: 3 → hidden, giving 3H parameters.
- The gate: 1 parameter.

| Hidden width | Adapter parameters (`8H + 121`) | Core |
|---|---|---|
| 64 | 633 | 120 |
| 768 | 6,265 | 120 |
| 1024 | 8,313 | 120 |

The projections are counted in every total: the adapter is not "120 parameters".

| Profile | Pre-activation | Status |
|---|---|---|
| `E0_REFERENCE` | `preact_bound=None` | Every cell is exactly E0 (factorized through `P_in`). Parity with the pinned upstream E0: 1.4e-14 forward, 7e-15 gradient |
| `BOUNDED_SPECIALIST` | `u → 3·tanh(u/3)` before φ | **E0-derived, not E0.** The pre-activation gap obeys `\|u\|³/(3b²)` (tested). The cell gap is 0.6% at pre-activation RMS 0.08, 29% at 0.4 and 71% at 0.8 (maximum-relative, `parity.json`) |

## 3. The budget

**The rule.** The total adapter, projections and gate included, is at most 5% of the parameters of the expert's
SwiGLU. This is an engineering design budget R11 introduces, not a historical threshold. Deviations need owner
approval; none was taken.

| Host | SwiGLU parameters | Micro adapter | Fraction |
|---|---|---|---|
| d64 expert (R10, FFN 86) | 16,512 | 633 | 3.83% |
| R6 500M routed expert (FFN 1,024) | 2,359,296 | 6,265 | 0.27% |
| R6 500M dense channel (FFN 2,048) | 4,718,592 | 6,265 | 0.13% |
| R6 1B routed/shared expert (FFN 683) | 2,098,176 | 8,313 | 0.40% |
| R6 1B dense channel (FFN 2,731) | 8,389,632 | 8,313 | 0.10% |

## 4. The micro scale layouts: NEW configurations with new totals

Expert count, top-k, FFN widths, ERSA, attention, PLE and R5 are unchanged. Only the CGC geometry changes.

| Layout | Core-model CGC parameters | Total, with R5's encoder at its legacy CGC | Model active per token |
|---|---|---|---|
| R6 500M, micro | 563,850 (90 × 6,265) | 363,740,293 | 222,330,921 |
| R6 1B, micro | 1,346,706 (162 × 8,313) | 573,724,629 | 308,831,303 |

The active-per-token figures are for the model as built, its encoder included.

**An interface finding: the R5 encoder's CGC is coupled to the core config.**
- The evidence encoder derives its `ModelConfig` from the core's. It overrides width, bottleneck and channels, but
  inherits `cgc_units` and the bound.
- A micro core profile therefore silently changes the encoder's CGC from 4 to 8 units: 1,376,258 parameters
  instead of 851,970.
- R11 leaves the R5 evidence path alone. It reports totals with the encoder at its legacy geometry, and records the
  coupling for a future decoupling decision. No code was changed for it.

## 5. Operator cost per token

| | Micro (H = 768) | Micro (H = 1024) | Legacy routed (R6 500M: 192 / 768 / 4) |
|---|---|---|---|
| Weight multiply-adds | 6,264 | 8,312 | 1,327,104 |
| Horner multiplies (3 per unit) | 72 | 72 | 9,216 |
| tanh evaluations (bounded) | 24 | 24 | 3,072 |
| Gate multiplies | 768 | 1,024 | 768 |

Measured CPU dispatch costs are in `EVIDENCE_R11.md`. A microscopic kernel is dominated by framework dispatch and
the projections; that overhead is a cost to report, not a reason to scale it up.

## 6. Placement: the causal path

`ErsaLM.forward` runs, per block:

```
u = norm(x) -> ERSA (+ attention, + evidence) -> pre = x + mixed -> x = pre + channel(norm(pre))
```

A block's channel, and so its CGC, runs AFTER that block's recurrent write. So:
- A block's CGC cannot change that block's ERSA state, for this chunk or any later one.
- It changes the input of every later block, and so their states.
- A last-block CGC affects only the readout.

All three are tested (`test_r11.py`, `test_a_blocks_cgc_cannot_change_that_blocks_ersa_state_only_later_blocks`).
No claim that CGC controls an ERSA write follows from the modules sharing a block.

## 7. Backward compatibility

Every R11 option is opt-in. The defaults reproduce the historical expert parameter for parameter and bit for bit
(tested), as well as the R6 accounting (507,683,413 / 994,692,069) and the R10 arms. The R11 options are:
- `SparseConfig.cgc_gate_mode="fixed"`;
- `cgc_preact_bound` overriding the model's bound;
- `kind="ffn+mlp"`, a control arm only.

`CGCCore`, `CGCTransform`, expert-local placement, `ChannelExpert`'s anatomy, the scalar gate, checkpoint schemas
and parameter identities are unchanged. The legacy geometries remain constructible.
