# Elpis Torch R11 — design: a correctly bounded CGC–FFN expert

**What R11 corrects.** The PyTorch expert gives CGC an enormous, unspecialized nonlinear branch. In R6's 500M and
1B layouts, CGC is 28.6% and 42.5% of the model (`CGC_BOUNDING_R11.md` §1), and nothing trains it for a narrow
specialty.

**What R11 does.**
- **It keeps the anatomy:** `Expert(u) = SwiGLU(u) + g · CGCTransform(u)`, with CGC inside the expert. There is no
  sidecar, no controller and no new memory tier.
- **It bounds the branch** to R97W's microscopic geometry.
- **It names its mathematics** honestly: `E0_REFERENCE` against `BOUNDED_SPECIALIST`.
- **It gives the branch an explicit specialist objective.**
- **It measures** what the branch then contributes to the expert.

R11 changes nothing under `src/` or `native/`, no CI policy, no R0–R10 record, no K1 authority, and nothing in
RALPH or Yang–Mills/HCA.

## 1. Current and corrected anatomy

| | Current (R6–R10 defaults) | Corrected (R11 micro profile) |
|---|---|---|
| Expert | `SwiGLU(u) + g·P_out(core(P_in u))` | the same |
| Bottleneck / cells / units | e.g. 192 / 768 / 4 (R6 500M expert); 16 / 48 / 4 (R10 d64) | **5 / 3 / 8** (R97W) |
| Core parameters | 589,824 (R6 500M expert) | **120** |
| Adapter parameters | 1,327,105 (R6 500M expert); 7,169 (R10 d64) | **8H + 121**: 6,265 at H = 768; 633 at H = 64 |
| Share of the expert's SwiGLU | 56% (R6 500M); 43% (R10 d64) | 0.27% (R6 500M); 3.8% (d64), within the 5% budget |
| Mathematics | E0-derived (model bound 3) | `E0_REFERENCE` (exact) or `BOUNDED_SPECIALIST` (E0-derived, b = 3), named |
| Gate | learned scalar | learned scalar (historical), or a fixed `g = 1` control |
| Objective | the shared language-model loss only | an explicit specialist objective (Stage B), offline |

## 2. Code

| File | Change |
|---|---|
| `research/elpis_torch/cgc_specialist.py` (new) | Everything R11 adds around the expert: <br>• the profile: geometry, `PROFILES`, `micro_sparse`, `micro_model`; <br>• the budget; <br>• exact accounting: `adapter_parameters`, `cgc_census`, `check_adapter`, `operator_costs`; <br>• diagnostics: `branch_diagnostics`, `gradient_norms`; <br>• the matched `GeluSpecialist` control |
| `research/elpis_torch/sparse_channel.py` | Opt-in options: <br>• `cgc_gate_mode` (`learned` by default, or `fixed`); <br>• `cgc_preact_bound` (the model's bound by default); <br>• kind `ffn+mlp` (control only); <br>• `ChannelExpert.specialist()`. <br>Defaults are bit-identical to before (tested) |
| `research/elpis_torch/model.py`, `ersa_model.py` | Initialization rules for the control's new parameter names only |
| `research/elpis_torch/specialist_tasks.py` (new) | The R97W-form teacher under R11's namespaces; the Stage A learners; the Stage B harness around the real `ChannelExpert` |
| `research/elpis_torch/experiments/run_r11.py` (new) | `audit`, `parity`, `calibrate`, `stage_a`, `stage_b`, `latency`, `combine`, `manifest` |
| `research/elpis_torch/specialist_stack.py` (new) | Stage C: the regression stack of the repository's `ErsaBlock`s (built and initialized by `ErsaLM`) with the micro specialist in each block's expert; in-context episodes; frozen ablations, ERSA reset, routing purity |
| `research/elpis_torch/experiments/run_r11_stage_c.py` (new) | `run`, `router`, `combine`, `dev`, under the frozen Stage C amendment |
| `research/elpis_torch/experiments/bench_r11_operators.py` (new) | PERFORMANCE_ONLY supplement: operators in rotated order, the bound alone, and the experts' measured memory |
| `tests/research/elpis_torch/test_r11_records.py` (new) | JSON only: digests, chronology, planned groups, development labels, every verdict recomputed independently, the manifest |
| `tests/research/elpis_torch/test_r11.py` (new) | Compatibility, profile, E0 parity, bound, gates, zero-gate starvation, routing-conditional gradients, the causal path, harness contracts, decision rules |

## 3. The experiment

| Stage | What it tests | Plan |
|---|---|---|
| A | Whether microscopic adaptation survives the transformer adapter's projections and bound: faithful E0, projected (fixed or learned projections), bounded, and a 120-parameter GELU, with cold, shuffled-support and k = 0 controls | `PREREGISTRATION_R11.md` §3 |
| B | Whether the specialist, inside the real `ChannelExpert` beside its FFN, makes a causal, specialized, cubic-specific contribution | `PREREGISTRATION_R11.md` §4 |
| C | Whether a qualified specialist contributes to frozen-weight inference in the actual ERSA stack | `PREREGISTRATION_R11.md` §5 |

Stage B runs G0–G3 and a no-objective control, plus ablation, swap and misrouting. Stage C is conditional on Stage B
and needs a separate frozen amendment before it runs. Stage B triggered it, and it ran under
`PREREGISTRATION_R11_STAGE_C.md` (frozen at `369e333`). Results: `EVIDENCE_R11.md`.

## 4. Runtime contract

- Checkpoint weights stay frozen at inference.
- There are no runtime persistent writes.
- Inner loops exist only in the offline research harness, and are discarded after each episode.
- A proposal that requires runtime fast adaptation is a separate, unapproved architectural choice. R11 does not make
  it.
