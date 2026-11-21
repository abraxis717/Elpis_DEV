# Elpis Torch R11 — evidence: a bounded, specialized expert-local CGC

**Principal question.** Does a correctly bounded, explicitly specialized expert-local `CGCTransform`, inside the
existing FFN expert:
- keep R97W's microscopic adaptive behaviour under the transformer adapter's projections and bound (Stage A);
- make a causal, specialized and cubic-specific contribution to its expert (Stage B);
- and transfer any of that to frozen-weight inference in the actual stack (Stage C)?

**Answer.**
- **Stage A: yes, at the microscopic scale and through a learned adapter.**
  - On fresh R97W-protocol episodes, faithful E0 meta-learns and beats the matched GELU by 46%
    (`MICRO_REFERENCE_QUALIFIED`).
  - The bounded adapter with learned projections keeps that behaviour (`PROJECTED_ADAPTATION_QUALIFIED`).
  - Fixed random projections lose most of it.
- **Stage B: yes, but only under explicit offline inner updates.** Inside the real `ChannelExpert`, the fixed-gate
  bounded micro CGC is `EXPERT_SPECIALIZATION_QUALIFIED`:
  - its adapted loss is 0.113 of the FFN alone's;
  - it is causal: ablation multiplies the loss by 10.2;
  - it is made by its specialist objective;
  - it is 15% better than a matched GELU specialist.

  The historical learned gate is slot-functional but without the cubic-specific advantage (`GATE_NOT_LIMITING`).
  Without inner updates, both specialists are worse than the FFN alone: `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES`.
- **Stage C: no.**
  - Trained end to end in the actual ERSA stack and then frozen, with the support examples in context, neither CGC
    specialist beats the FFN alone: `NOT_ESTABLISHED` for G1 and G2.
  - All in-context information is carried by ERSA.
  - A learned router does not separate the task families: `INTEGRATION_GAP_LEARNED_ROUTING`.
- **The footprint is corrected.**
  - The micro profile takes CGC from 28.6% of the R6 500M layout to 0.39%, and from 42.5% of the 1B layout to 0.38%.
    These are exact counts; the profile is opt-in and the defaults are unchanged.
  - On CPU the micro branch still doubles a d64 expert's time, through dispatch overhead.

The specialist's value therefore lives in explicit adaptation, which the frozen-weight inference contract does not
provide. R11 reports that gap and implements no runtime learning to close it.

R11 changes nothing under `src/` or `native/`, no CI policy, no R0–R10 record, no K1 authority, and nothing in
RALPH or Yang–Mills/HCA. Every new option is opt-in, and the defaults are bit-identical to the historical expert
(§3).
- Plans: `PREREGISTRATION_R11.md`, frozen at `47537ef`; `PREREGISTRATION_R11_STAGE_C.md`, frozen at `369e333`.
- Design: `DESIGN_R11.md`, `CGC_BOUNDING_R11.md`, `CGC_SPECIALIZATION_R11.md`.
- Capability audit: `CGC_CAPABILITY_R11.md`.
- Ledger: `CLAIM_AUDIT_R9.md`, section "R11 result".

## 0. Verdicts, by kind

| Kind | Verdict | Where |
|---|---|---|
| Mechanical qualification (A4) | PASS for compatibility, geometry, budget, E0 parity, the quantified bound, the gates, the causal path and the harness contracts. **Not a specialist result** | §3 |
| Stage A, faithful E0 | **`MICRO_REFERENCE_QUALIFIED`**: 12/12 groups; E0 / GELU 0.543 | §4 |
| Stage A, bounded adapter with learned projections | **`PROJECTED_ADAPTATION_QUALIFIED`**: 12/12 groups. Unbounded learned: 11/12. Both fixed-projection arms: 8/12. All three are NOT_ESTABLISHED | §4 |
| Stage A, the bound | `bound_destroys_behavior = False`, so the Stage B profile is `BOUNDED_SPECIALIST` | §4 |
| Stage B, G2 (fixed gate) | **`EXPERT_SPECIALIZATION_QUALIFIED`**, under explicit inner updates (Mode A) | §5 |
| Stage B, G1 (learned gate) | `SPECIALIST_SLOT_FUNCTIONAL_NO_CGC_SPECIFIC_ADVANTAGE` | §5 |
| Gate identifiability (A1) | `GATE_NOT_LIMITING` | §5 |
| Mode B (k = 0), G1 and G2 | `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES` | §5 |
| Stage C, G1 and G2 (frozen, support in context, actual stack) | **`NOT_ESTABLISHED`**: `FROZEN_INFERENCE_TRANSFER_QUALIFIED` not reached | §6 |
| Stage C, learned routing | **`INTEGRATION_GAP_LEARNED_ROUTING`**: 2 of 12 blocks separate the families; mean purity 0.69 | §6.4 |
| Runtime-weight immutability | IMMUTABLE in every R11 path | §8 |
| Footprint (R6 500M / 1B) | CGC 28.63% → 0.39% and 42.54% → 0.38%. Exact counts; not built | §7.1 |
| Cost (d64 expert, CPU) | 2.02× the FFN-only expert's time; +21% autograd bytes per token; 2,528 parameter bytes | §7 |
| R9 and R10 verdicts | Unchanged | `CLAIM_AUDIT_R9.md` |
| K1 (A3) | A separate mechanism, not implemented in ERSA or the expert | §9 |

## 1. Provenance

**Base.** `main` at `d2d683ad874dda6312e27f87a8a2f1b0a64f6096` (the R10 merge), post-merge CI run 44795525346 at
16/16. Branch `claude/r11-cgc-specialist`.

**Freezes.**

| Plan | Machine-readable form | SHA-256 | Frozen at | Before |
|---|---|---|---|---|
| `PREREGISTRATION_R11.md` | `results/r11/preregistration.json` | `a7df440a779b0e5154dabacec4968e45588cb3beeb0cd1be2a42c339a2edc198` | `47537ef` (20:17 UTC) | any confirmation record |
| `PREREGISTRATION_R11_STAGE_C.md` | `results/r11/amendment_stage_c.json` | `ede13d3d982963fa341f4e46ccf9d2d6ad87670a61561b38bad8298e3dd51a20` | `369e333` (20:46 UTC) | any Stage C record |

**Integrity.**
- Every confirmation record carries the SHA-256 of its plan and of every source it ran: the runner, the task and
  harness module, the profile module, `sparse_channel.py`, `cgc.py`, `model.py`, `config.py` and the vendored
  upstream E0. Stage C records add the amendment, the Stage C runner and `specialist_stack.py`.
- `run_r11 combine` and `run_r11_stage_c combine` refuse a record whose digests differ. Neither refused one.
- `tests/research/elpis_torch/test_r11_records.py` reads the records as JSON and re-implements every decision rule
  independently. It also checks that each plan predates the records it governs, by `git show` at the recorded
  revision.

**Upstream evidence read** (public CGC repository, read-only clone at `0895c2b`): the pinned E0 cell, R97O's and
R97W's archived results (`CGC_CAPABILITY_R11.md`). Nothing was copied (`PROVENANCE.md`, R11 section).

## 2. What ran

| Step | Command | Records | Scale | Time (4 CPU cores) |
|---|---|---|---|---|
| Audit (exact counts, meta tensors) | `run_r11 audit` | `audit.json` | R6 500M and 1B layouts, d64 expert | seconds; nothing allocated |
| Parity | `run_r11 parity` | `parity.json` | micro geometry, float64 | seconds |
| Calibration (DEVELOPMENT) | `run_r11 calibrate` | `calibration.json` | `R11A_DEV`, seeds 11, 12 | minutes |
| Development | `run_r11 stage_a --dev`, `stage_b --dev`, `run_r11_stage_c dev` | `dev_*.json` | `R11A_DEV`, `R11B_DEV`, `R11C_DEV` | — |
| **Stage A** | `run_r11 stage_a` | `stage_a_<family>_s<seed>.json` × 12 | 14 arms × 12 groups × 12 TEST episodes | about 2 s per group |
| **Stage B** | `run_r11 stage_b` | `stage_b_s<seed>.json` × 6 | 5 arms × 12 groups × 12 TEST episodes, plus swap and misrouting | 165–174 s per seed |
| Combined A/B analysis | `run_r11 combine` | `study_r11.json` | — | — |
| **Stage C** | `run_r11_stage_c run` | `c_<family>_s<seed>.json` × 12 | 4 arms × 12 groups × 12 TEST episodes × 7 support sizes | 359–388 s per group (two threads, two parallel queues) |
| Routing test | `run_r11_stage_c router` | `c_router_s<seed>.json` × 6 | G2, two experts per block | 105–106 s per seed |
| Stage C analysis | `run_r11_stage_c combine` | `c_study.json` | — | — |
| Latency (PERFORMANCE_ONLY) | `run_r11 latency`; `bench_r11_operators` | `latency.json`; `operator_bench_rotated.json` | one CPU thread | minutes |
| Manifest | `run_r11 manifest` | `MANIFEST.json` | every R11 record with its digest | — |

Every planned group has exactly one record. No confirmation record was rerun, no seed was added, and no threshold
or endpoint changed after a result (the stopping rules, `PREREGISTRATION_R11.md` §8).

## 3. Mechanical qualification (A4): the expert, the profile and the harness work as specified

These are engineering results, established by tests and exact records. **They are not specialist evidence**, and
nothing in §4–§6 is inferred from them.

| Property | Result | Where |
|---|---|---|
| Backward compatibility | The default expert is the historical one, parameter for parameter, in order and bit for bit; the R6 totals (507,683,413 / 994,692,069) and active counts are unchanged | `test_r11.py` (2 tests) |
| Micro geometry | Core 120 = 5 × 3 × 8; adapter `8H + 121` (633 / 6,265 / 8,313 at H = 64 / 768 / 1024), counted from the tensors | `audit.json`; `test_r11.py` |
| Budget | Within 5% of the expert's SwiGLU in every R6 placement: 0.10%–0.40%; 3.83% at d64 | `audit.json` |
| `E0_REFERENCE` parity | Forward 1.4e-14, gradient (autograd against analytic) 7.1e-15, tolerance 1e-12, against the pinned upstream E0 | `parity.json` |
| `BOUNDED_SPECIALIST` | E0-derived, not E0. The pre-activation gap obeys `\|u\|³/(3b²)` at every scale tested. The maximum-relative cell gap is 0.6% at pre-activation RMS 0.08, 29% at 0.39, 71% at 0.79, 83% at 1.58 and 97% at 3.16 | `parity.json` |
| float32 against float64 (micro adapter) | Maximum absolute difference 1.1e-5 / 1.9e-5 / 3.3e-5 at H = 64 / 768 / 1024, against output RMS 3.0–4.2 | `parity.json` |
| Fixed gate | No parameter; contributes the full branch | `test_r11.py` |
| GELU control | 5 → 13 → 3 (120 parameters) between the same projections; the same adapter budget | `test_r11.py` |
| Zero-gate starvation | With `g = 0`, every branch parameter but the gate receives exactly zero gradient | `test_r11.py` |
| Routing-conditional gradients | An expert receives gradient only from the tokens routed to it | `test_r11.py` |
| Causal path | A block's CGC cannot change that block's ERSA state; it changes every later block's | `test_r11.py` |
| Inner loop | Moves only the 120 core parameters; evaluation adapts a copy and leaves every weight byte-identical (Stage B and Stage C) | `test_r11.py` (3 tests) |
| Stage C stack | The repository's own `ErsaBlock`s, built and initialized by `ErsaLM`; causal; its ablation removes only the specialist | `test_r11.py` (3 tests) |

**Disposition:** mechanical PASS for the expert, the profiles, the budget and the harnesses. By A4 this is not
promoted into a specialist PASS.

## 4. Stage A: the microscopic reference under the adapter

**The run.** 12 groups (smooth and conditional × seeds 101–106), 12 held-out TEST episodes each: 144 paired
episodes per comparison. Every arm trained and was evaluated on the same episodes, under R11's own namespace
`R11A` (a new experiment under R97W's protocol, not a re-run of R97W).

**Held-out query MSE after k inner steps.** Mean over groups of the per-group median.

zero predictor: 0.150

| Arm | k=0 | k=1 | k=2 | k=4 | k=8 | OOD, k=8 | inner / outer-trained / resident |
|---|---|---|---|---|---|---|---|
| E0_COLD | 0.209 | 0.162 | 0.131 | 0.093 | 0.062 | 0.131 | 120 / 120 / 120 |
| E0_META | 0.126 | 0.102 | 0.086 | 0.064 | 0.045 | 0.100 | 120 / 120 / 120 |
| E0_SHUFFLED | 0.159 | 0.131 | 0.109 | 0.079 | 0.053 | 0.090 | 120 / 120 / 120 |
| GELU_COLD | 0.151 | 0.142 | 0.134 | 0.120 | 0.101 | 0.133 | 120 / 120 / 120 |
| GELU_META | 0.135 | 0.125 | 0.116 | 0.101 | 0.080 | 0.114 | 120 / 120 / 120 |
| PROJ_B3_COLD | 0.195 | 0.153 | 0.139 | 0.128 | 0.116 | 0.168 | 120 / 120 / 632 |
| PROJ_B3_FIXED_META | 0.131 | 0.117 | 0.111 | 0.103 | 0.096 | 0.143 | 120 / 120 / 632 |
| PROJ_B3_LEARNED_META | 0.136 | 0.078 | 0.058 | 0.045 | 0.040 | 0.100 | 120 / 632 / 632 |
| PROJ_E0_COLD | 0.198 | 0.153 | 0.139 | 0.128 | 0.116 | 0.170 | 120 / 120 / 632 |
| PROJ_E0_FIXED_META | 0.131 | 0.117 | 0.111 | 0.103 | 0.096 | 0.143 | 120 / 120 / 632 |
| PROJ_E0_LEARNED_META | 0.135 | 0.088 | 0.075 | 0.060 | 0.049 | 0.099 | 120 / 632 / 632 |
| PROJ_GELU_COLD | 0.150 | 0.142 | 0.135 | 0.126 | 0.117 | 0.149 | 120 / 120 / 632 |
| PROJ_GELU_FIXED_META | 0.145 | 0.133 | 0.125 | 0.116 | 0.107 | 0.138 | 120 / 120 / 632 |
| PROJ_GELU_LEARNED_META | 0.136 | 0.118 | 0.107 | 0.095 | 0.082 | 0.119 | 120 / 632 / 632 |

**Utility** (R97W's four criteria, against the arm's own cold control, in EVERY group):

| Arm | Groups passing | Disposition |
|---|---|---|
| E0_META | 12/12 | MICRO_REFERENCE_QUALIFIED |
| GELU_META | 9/12 | control |
| PROJ_B3_FIXED_META | 8/12 | NOT_ESTABLISHED |
| PROJ_B3_LEARNED_META | 12/12 | PROJECTED_ADAPTATION_QUALIFIED |
| PROJ_E0_FIXED_META | 8/12 | NOT_ESTABLISHED |
| PROJ_E0_LEARNED_META | 11/12 | NOT_ESTABLISHED |
| PROJ_GELU_FIXED_META | 7/12 | control |
| PROJ_GELU_LEARNED_META | 12/12 | control |

`PROJ_E0_LEARNED_META` misses in one group: conditional seed 105 has 7 of 12 episodes better (8 required), though
its geometric mean there is 0.593.

**Paired comparisons at k = 8** (the "better" rule):

| Comparison | Pooled geometric-mean ratio | Sign test p | Groups below 1 | Better |
|---|---|---|---|---|
| E0_META/E0_SHUFFLED | 0.838 | 4.6e-28 | 12/12 | yes |
| E0_META/GELU_META | 0.543 | 4.5e-44 | 12/12 | yes |
| PROJ_B3_FIXED_META/PROJ_E0_FIXED_META | 1.001 | 1.000 | 3/12 | no |
| PROJ_B3_FIXED_META/PROJ_GELU_FIXED_META | 0.924 | 8.8e-06 | 9/12 | yes |
| PROJ_B3_LEARNED_META/PROJ_E0_LEARNED_META | 0.831 | 5.0e-13 | 9/12 | yes |
| PROJ_B3_LEARNED_META/PROJ_GELU_LEARNED_META | 0.520 | 1.9e-21 | 11/12 | yes |
| PROJ_E0_FIXED_META/E0_META | 2.183 | 1.000 | 0/12 | no |
| PROJ_E0_FIXED_META/PROJ_GELU_FIXED_META | 0.923 | 8.8e-06 | 9/12 | yes |
| PROJ_E0_LEARNED_META/PROJ_GELU_LEARNED_META | 0.626 | 4.9e-25 | 12/12 | yes |

**Outer-loop learning curve.** DEV-episode k = 8 query MSE (per-group median, mean over groups), by the number of
training episodes seen:

| Arm | 0 | 16 | 32 | 64 |
|---|---|---|---|---|
| E0_META | 0.066 | 0.055 | 0.052 | 0.050 |
| E0_SHUFFLED | 0.066 | 0.059 | 0.058 | 0.056 |
| GELU_META | 0.105 | 0.103 | 0.096 | 0.087 |
| PROJ_B3_FIXED_META | 0.122 | 0.110 | 0.104 | 0.104 |
| PROJ_B3_LEARNED_META | 0.122 | 0.067 | 0.070 | 0.048 |
| PROJ_E0_FIXED_META | 0.121 | 0.109 | 0.104 | 0.104 |
| PROJ_E0_LEARNED_META | 0.121 | 0.084 | 0.067 | 0.059 |
| PROJ_GELU_FIXED_META | 0.118 | 0.117 | 0.113 | 0.110 |
| PROJ_GELU_LEARNED_META | 0.118 | 0.112 | 0.102 | 0.088 |

**What Stage A establishes.**
- **The microscopic result reproduces (`MICRO_REFERENCE_QUALIFIED`).** On fresh seeds and episodes, faithful E0
  meta-learns in every group and beats the 120-parameter GELU by 46% at k = 8 (ratio 0.543, 12/12 groups).
- **The cubic core's advantage over the matched GELU survives the adapter** in all four projected settings: 0.92
  with fixed projections, 0.63 (E0) and 0.52 (bounded) with learned projections.
- **How much adaptation survives depends on the projections.**
  - Fixed random projections cost 2.2× in loss and leave meta-learning a small, unreliable gain over cold (8/12
    groups).
  - Learned projections recover the microscopic level. They add 512 outer-trained parameters at H = 64,
    disclosed; the inner loop stays at 120.
- **The bound does not destroy the behaviour.** With learned projections, `BOUNDED_SPECIALIST` is the only projected
  CGC that qualifies (`PROJECTED_ADAPTATION_QUALIFIED`), and it is 17% better than unbounded (0.831), consistent with
  the calibration record, where the bound allowed a larger projection learning rate (0.3 against 0.1). So
  `bound_destroys_behavior = False`, and Stage B used `BOUNDED_SPECIALIST`, by the frozen rule.
- **It is still E0-derived, not E0.** Its competence here is measured, not inherited from R97W.

## 5. Stage B: the specialist inside the real `ChannelExpert`

**The setting.** One `ChannelExpert` (d 64, FFN 86) per family, the micro specialist beside its FFN, a 64-wide
hidden state carrying the 5-dimensional task subspace plus unit nuisance, and a fixed orthonormal readout
(`CGC_SPECIALIZATION_R11.md` §2). 12 groups (seeds 201–206 × both families), 12 TEST episodes each. The expert
assignment is fixed and known: a LABORATORY control, not routing.

**Adaptation curves.** Query MSE after k inner steps, mean over groups of the per-group median. k = 0 is Mode B
(no update); k = 8 is Mode A.

| Arm | k=0 (Mode B) | k=1 | k=2 | k=4 | k=8 (Mode A) | OOD, k=8 | branch ablated | zero predictor | selected step (median) |
|---|---|---|---|---|---|---|---|---|---|
| G0 | 0.1005 | 0.1005 | 0.1005 | 0.1005 | 0.1005 | 0.1457 | 0.1005 | 0.1533 | 600 |
| G1 | 1.3428 | 0.3451 | 0.0552 | 0.0177 | 0.0136 | 0.0485 | 0.1142 | 0.1533 | 800 |
| G2 | 0.6697 | 0.1882 | 0.0410 | 0.0146 | 0.0113 | 0.0464 | 0.1215 | 0.1533 | 700 |
| G3 | 0.1137 | 0.0278 | 0.0189 | 0.0154 | 0.0140 | 0.0503 | 0.1225 | 0.1533 | 950 |
| G2_POOLED | 0.0996 | 0.0992 | 0.0989 | 0.0985 | 0.0979 | 0.1430 | 0.1012 | 0.1533 | 600 |

**Paired comparisons** (the "better" rule; 144 paired episodes):

| Comparison | k=8 ratio | p | groups below 1 | better | k=0 ratio | better at k=0 |
|---|---|---|---|---|---|---|
| G1/G0 | 0.130 | 4.5e-44 | 12/12 | yes | 6.101 | no |
| G1/G2_POOLED | 0.132 | 4.5e-44 | 12/12 | yes | 6.105 | no |
| G1/G3 | 0.973 | 0.078 | 8/12 | no | 5.857 | no |
| G2/G0 | 0.113 | 4.5e-44 | 12/12 | yes | 2.762 | no |
| G2/G1 | 0.875 | 1.8e-08 | 9/12 | yes | 0.453 | yes |
| G2/G2_POOLED | 0.115 | 4.5e-44 | 12/12 | yes | 2.764 | no |
| G2/G3 | 0.851 | 4.7e-08 | 10/12 | yes | 2.652 | no |
| G2_POOLED/G0 | 0.986 | 1.5e-04 | 11/12 | no | 0.999 | no |
| G3/G0 | 0.133 | 4.5e-44 | 12/12 | yes | 1.042 | no |

**Controls at k = 8** (pooled geometric mean over 144 episodes):

| Arm | ablated / intact | causal (>= 1.25) | other family's specialist / own | other family's whole expert / own |
|---|---|---|---|---|
| G1 | 8.561 | yes | 1.532 | 5.796 |
| G2 | 10.196 | yes | 1.563 | 6.033 |
| G2_POOLED | 1.027 | no | 1.031 | 2.092 |
| G3 | 8.908 | yes | 1.819 | 5.548 |

**The criteria** (`PREREGISTRATION_R11.md` §6):

| Criterion | G1 (learned gate) | G2 (fixed gate) |
|---|---|---|
| Assay valid (G0 k = 0 ≤ 0.90 × zero predictor, every group) | yes | yes |
| Adaptive benefit (better than G0, ratio ≤ 0.80) | yes (0.130) | yes (0.113) |
| Causal (ablated / intact ≥ 1.25) | yes (8.56) | yes (10.20) |
| Specialist objective (better than G2_POOLED) | yes (0.132) | yes (0.115) |
| CGC-specific (better than G3) | **no** (0.973, p = 0.078, 8/12) | **yes** (0.851, p = 4.7e-8, 10/12) |
| Within budget | yes (3.83%) | yes (3.83%) |
| **Disposition** | `SPECIALIST_SLOT_FUNCTIONAL_NO_CGC_SPECIFIC_ADVANTAGE` | **`EXPERT_SPECIALIZATION_QUALIFIED`** |
| Mode B (better than G0 at k = 0?) | no (6.10× worse): `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES` | no (2.76× worse): `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES` |

Gate (A1): both G1 and G2 are slot-functional, so **`GATE_NOT_LIMITING`** by the frozen rule.

**Training curves.** DEV k = 8 query MSE, median over groups, every 100 outer steps; G1's learned gate:

| Outer step | G0 | G1 | G2 | G3 | G2_POOLED | G1 gate |
|---|---|---|---|---|---|---|
| 0 | 0.1561 | 0.1561 | 0.1680 | 0.1214 | 0.1680 | 0.100 |
| 100 | 0.1095 | 0.1124 | 0.0353 | 0.0309 | 0.1082 | 0.115 |
| 200 | 0.1083 | 0.0995 | 0.0242 | 0.0245 | 0.1054 | 0.185 |
| 300 | 0.1067 | 0.0268 | 0.0185 | 0.0217 | 0.1052 | 0.374 |
| 400 | 0.1039 | 0.0181 | 0.0145 | 0.0187 | 0.1020 | 0.394 |
| 500 | 0.1015 | 0.0165 | 0.0145 | 0.0197 | 0.0989 | 0.420 |
| 600 | 0.1007 | 0.0163 | 0.0130 | 0.0183 | 0.0991 | 0.430 |
| 700 | 0.1024 | 0.0158 | 0.0129 | 0.0180 | 0.1001 | 0.447 |
| 800 | 0.1008 | 0.0157 | 0.0135 | 0.0192 | 0.0997 | 0.446 |
| 900 | 0.1017 | 0.0156 | 0.0156 | 0.0179 | 0.0995 | 0.451 |
| 1000 | 0.0995 | 0.0148 | 0.0129 | 0.0177 | 0.0982 | 0.452 |

**Branch diagnostics at the end of training** (median over groups, on each group's DIAGNOSE episode):

| Arm | gate | branch / FFN RMS | pre-activation RMS (max) | saturated | non-finite | ‖P_out‖ init → end | relative change: core / P_in / P_out / FFN out |
|---|---|---|---|---|---|---|---|
| G1 | 0.452 (0.41–0.53) | 17.30 | 0.368 (1.19) | 0 | 0 | 8.21 → 8.99 | 0.69 / 1.75 / 0.36 / 1.62 |
| G2 | 1 (fixed) | 19.50 | 0.221 (0.93) | 0 | 0 | 8.21 → 8.56 | 0.56 / 1.46 / 0.23 / 1.58 |
| G3 | 1 (fixed) | 2.09 | — | — | 0 | 8.21 → 9.21 | 1.10 / 1.39 / 0.45 / 1.49 |
| G2_POOLED | 1 (fixed) | 2.50 | 0.086 (0.54) | 0 | 0 | 8.21 → 8.11 | 0.48 / 0.48 / 0.11 / 1.67 |

**Gradient norms by component** (median over groups):

| Arm | outer step | core | P_in | P_out | gate | FFN |
|---|---|---|---|---|---|---|
| G1 | 0 | 0.0149 | 0.0094 | 0.0043 | 0.0167 | 0.0071 |
| G1 | 300 | 0.0876 | 0.0220 | 0.0058 | 0.0175 | 0.0613 |
| G1 | 900 | 0.1439 | 0.0202 | 0.0076 | 0.0155 | 0.0404 |
| G2 | 0 | 0.2014 | 0.0674 | 0.0471 | — | 0.0070 |
| G2 | 300 | 0.0866 | 0.0275 | 0.0068 | — | 0.0492 |
| G2 | 900 | 0.1666 | 0.0234 | 0.0108 | — | 0.0484 |
| G3 | 0 | 0.0807 | 0.0061 | 0.0216 | — | 0.0063 |
| G3 | 300 | 0.0758 | 0.0128 | 0.0040 | — | 0.0505 |
| G3 | 900 | 0.1039 | 0.0131 | 0.0046 | — | 0.0474 |

seconds per seed: [173.7, 165.3, 169.7, 165.9, 164.5, 167.5]

**What Stage B establishes.**
- **Under explicit offline inner updates (Mode A), the fixed-gate bounded micro CGC is a qualified expert
  specialist** (`EXPERT_SPECIALIZATION_QUALIFIED`, G2):
  - it cuts the expert's adapted query loss to 0.113 of the FFN alone;
  - it is causal: removing the adapted branch multiplies the loss by 10.2;
  - the specialist objective is what makes it: the same branch trained with the pooled objective (G2_POOLED)
    adapts barely at all (0.986 of G0) and is not causal (1.03);
  - it beats the matched GELU specialist by 15% (0.851, 10/12 groups). The cubic-specific advantage is real but
    much smaller than the 46% of the bare microscopic comparison (§4), and smaller than the specialist's advantage
    over the FFN.
- **Specialization is partial.** The other family's specialist, adapted from its own initialization on this
  family's support, is 1.56× worse than this family's own (G2), and still 0.17 of G0's loss. The other family's whole
  expert is 6.0× worse: most of the task-conditioned behaviour lives in the trained FFN and projections the
  specialist works with, not in the 120 core parameters alone.
- **The learned gate is not limiting, but it costs the CGC-specific advantage.**
  - At initialization the 0.1 gate scales the branch's gradients: the core gradient is 0.015 in G1 against 0.20 in
    G2 (13×), and G1's training curve lags G2's by 100–200 outer steps.
  - The gate grows from 0.10 to about 0.39 by step 400 and 0.45 by step 700, then stays there; it never approaches 1.
  - G2's adapted loss ends 12.5% below G1's (G2/G1 = 0.875, p = 1.8e-8), and G1 loses the advantage over GELU (0.973,
    p = 0.078).
  - `‖P_out‖` grows slightly in every arm trained with the specialist objective (8.21 → 8.56–9.21): the output
    projection does not suppress the branch.
- **Mode A only.** Without inner updates the meta-learned initializations are far worse than the FFN alone (G2
  2.76×, G1 6.10× at k = 0; their k = 0 loss is 4–9× the zero predictor's). The branch output is 17–20× the FFN's
  RMS on the diagnostic inputs: the specialist is trained as a starting point for adaptation, not as a predictor.
  The k = 0 value is therefore `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES` for both arms, as the plan anticipated
  (`CGC_SPECIALIZATION_R11.md` §1). No runtime learning mechanism was added to avoid it.
- **The bound is engaged only lightly.** Trained pre-activation RMS is 0.22 (G2) and 0.37 (G1), with maxima below
  1.2 and no saturation: the specialist operates where the bounded cell stays close to E0 (§3).

## 6. Stage C: frozen-weight transfer into the actual ERSA stack

### 6.1 The setting

**Why it ran.**
- Stage B found `EXPERT_SPECIALIZATION_QUALIFIED` for G2, which is the preregistered trigger
  (`PREREGISTRATION_R11.md` §5).
- Stage C ran under its own amendment, frozen at `369e333` before any Stage C data.

**The stack.** The repository's own `ErsaBlock`s, built and initialized by `ErsaLM`:
- d 64, 2 blocks;
- ERSA with 4 heads, a 16 × 16 state per head and a convolution of 4;
- no attention;
- each block's expert channel (FFN 86 + specialist) placed after that block's ERSA.

They sit between a linear input map (9 → 64) and a linear readout (64 → 3), a disclosed harness difference from the
byte language model.

**The episodes.**
- A sequence holds `k` support tokens `[x, y, 0]`, then 48 query tokens `[x, 0, 1]`.
- The loss is on the queries.
- 12 groups (seeds 301–306 × both families), 12 TEST episodes each.

**Training, then freezing.**
- Every arm trains end to end: Adam 3e-3, cosine schedule, 2,000 steps of 16 episodes, with `k` drawn per step from
  {0, 1, 2, 4, 8, 16, 32}.
- Each keeps its best DEV checkpoint (k = 32, scored every 250 steps).
- It is then FROZEN: no inner loop and no weight update. The support examples reach the model only as tokens.

### 6.2 Results

**In-context curves.** Query MSE with `k` support examples in the sequence, mean over groups of the per-group median:

| Arm | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | ablate all blocks (k=32) | ERSA reset after support (k=32) | zero predictor | selected step (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 0.0943 | 0.0209 | 0.0166 | 0.0115 | 0.0086 | 0.0064 | 0.0056 | 0.0056 | 0.0943 | 0.1503 | 2000 |
| G1 | 0.0952 | 0.0209 | 0.0170 | 0.0113 | 0.0084 | 0.0062 | 0.0050 | 0.0436 | 0.0952 | 0.1503 | 2000 |
| G2 | 0.0941 | 0.0218 | 0.0157 | 0.0127 | 0.0100 | 0.0084 | 0.0074 | 1.5943 | 0.0941 | 0.1503 | 2000 |
| G3 | 0.0933 | 0.0217 | 0.0172 | 0.0120 | 0.0090 | 0.0067 | 0.0059 | 0.0499 | 0.0933 | 0.1503 | 2000 |

Smooth family:

| Arm | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|
| G0 | 0.0953 | 0.0208 | 0.0166 | 0.0122 | 0.0095 | 0.0073 | 0.0066 |
| G1 | 0.0971 | 0.0209 | 0.0167 | 0.0125 | 0.0096 | 0.0077 | 0.0069 |
| G2 | 0.0955 | 0.0223 | 0.0165 | 0.0136 | 0.0111 | 0.0094 | 0.0085 |
| G3 | 0.0934 | 0.0208 | 0.0167 | 0.0125 | 0.0105 | 0.0084 | 0.0077 |

Conditional family:

| Arm | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|
| G0 | 0.0933 | 0.0209 | 0.0165 | 0.0107 | 0.0077 | 0.0055 | 0.0047 |
| G1 | 0.0932 | 0.0210 | 0.0174 | 0.0102 | 0.0073 | 0.0047 | 0.0031 |
| G2 | 0.0928 | 0.0213 | 0.0149 | 0.0118 | 0.0090 | 0.0074 | 0.0062 |
| G3 | 0.0933 | 0.0226 | 0.0176 | 0.0115 | 0.0075 | 0.0050 | 0.0040 |

**Paired comparisons** (the "better" rule; 144 paired episodes):

| Comparison | k | ratio | p | groups below 1 | better |
|---|---|---|---|---|---|
| G0/G1 | 32 | 1.147 | 0.960 | 5/12 | no |
| G0/G2 | 32 | 0.701 | 9.7e-04 | 7/12 | no |
| G0/G3 | 32 | 0.910 | 0.599 | 7/12 | no |
| G1/G0 | 32 | 0.872 | 0.057 | 7/12 | no |
| G1/G3 | 32 | 0.793 | 0.002 | 7/12 | no |
| G2/G0 | 32 | 1.427 | 0.999 | 5/12 | no |
| G2/G1 | 32 | 1.637 | 1.000 | 4/12 | no |
| G2/G3 | 32 | 1.299 | 0.894 | 7/12 | no |
| G3/G0 | 32 | 1.098 | 0.467 | 5/12 | no |
| G0/G2 | 1 | 0.998 | 0.988 | 4/12 | no |
| G2/G0 | 1 | 1.002 | 0.018 | 8/12 | no |
| G2/G3 | 1 | 0.975 | 9.7e-04 | 9/12 | no |

**Frozen controls at k = 32** (pooled geometric mean of intervention / intact):

| Arm | ablate every block's specialist | block 0 only | block 1 only | ERSA reset after the support |
|---|---|---|---|---|
| G0 | 1.000 | 1.000 | 1.000 | 22.147 |
| G1 | 4.028 | 1.446 | 3.577 | 25.523 |
| G2 | 200.484 | 4.909 | 77.560 | 15.286 |
| G3 | 4.361 | 2.051 | 2.539 | 19.865 |

**The criteria** (amendment §3):

| Criterion | G1 (learned gate) | G2 (fixed gate) |
|---|---|---|
| Assay (G0 k = 32 ≤ 0.5 × zero predictor, every group) | yes | yes |
| Better than G0 at k = 32 | no (0.872, p = 0.057, 7/12) | no (1.427, 5/12) |
| Better than G3 at k = 32 | no (0.793, p = 0.0017, 7/12) | no (1.299, 7/12) |
| Ablating every block's specialist raises the loss ≥ 1.10× | yes (4.03) | yes (200.5) |
| G0 better than the arm | no (1.147) | no: 0.701, p = 9.7e-4, but 7/12 groups, below 2/3 |
| **Disposition** | **`NOT_ESTABLISHED`** | **`NOT_ESTABLISHED`** |

**Training curves.** DEV k = 32 query MSE, median over groups:

| Step | G0 | G1 | G2 | G3 |
|---|---|---|---|---|
| 0 | 1.2010 | 1.1914 | 1.0748 | 1.1626 |
| 250 | 0.0198 | 0.0224 | 0.0295 | 0.0240 |
| 500 | 0.0137 | 0.0142 | 0.0180 | 0.0121 |
| 750 | 0.0126 | 0.0105 | 0.0149 | 0.0113 |
| 1000 | 0.0080 | 0.0080 | 0.0102 | 0.0080 |
| 1250 | 0.0061 | 0.0053 | 0.0081 | 0.0066 |
| 1500 | 0.0042 | 0.0044 | 0.0073 | 0.0053 |
| 1750 | 0.0033 | 0.0034 | 0.0065 | 0.0042 |
| 2000 | 0.0032 | 0.0033 | 0.0069 | 0.0037 |

**G1's learned gates** at the selected checkpoint:

| Block | median | min | max |
|---|---|---|---|
| 0 | 0.088 | 0.024 | 0.135 |
| 1 | 0.129 | 0.096 | 0.177 |

**Parameters:**

| Arm | stack parameters | specialist parameters (both blocks) |
|---|---|---|
| G0 | 69,083 | 0 |
| G1 | 70,349 | 1,266 |
| G2 | 70,347 | 1,264 |
| G3 | 70,347 | 1,264 |

**The run-to-run spread.** Per-group geometric-mean ratios at k = 32:

| Group | G1/G0 | G2/G0 | G3/G0 | G2/G3 | G1/G3 |
|---|---|---|---|---|---|
| smooth 301 | 1.07 | 0.95 | 1.02 | 0.93 | 1.04 |
| conditional 301 | 0.39 | 1.02 | 1.06 | 0.96 | 0.37 |
| smooth 302 | 0.96 | 1.69 | 4.23 | 0.40 | 0.23 |
| conditional 302 | 0.73 | 2.67 | 1.52 | 1.76 | 0.48 |
| smooth 303 | 1.35 | 0.58 | 1.48 | 0.39 | 0.91 |
| conditional 303 | 0.53 | 1.20 | 0.69 | 1.76 | 0.77 |
| smooth 304 | 1.07 | 3.26 | 0.83 | 3.92 | 1.29 |
| conditional 304 | 0.50 | 0.52 | 0.55 | 0.96 | 0.91 |
| smooth 305 | 0.98 | 2.53 | 0.94 | 2.70 | 1.04 |
| conditional 305 | 1.04 | 0.88 | 1.01 | 0.87 | 1.02 |
| smooth 306 | 0.97 | 0.97 | 1.02 | 0.95 | 0.96 |
| conditional 306 | 1.74 | 6.28 | 0.99 | 6.35 | 1.76 |

### 6.3 What Stage C establishes

- **The assay is learned, and every arm learns in context.** With 32 support examples in the sequence, every arm's
  query loss falls from about 0.094 at k = 0 to 0.005–0.007, against 0.150 for the zero predictor.
- **The in-context information is ERSA's.** Resetting the ERSA state after the support prefix returns every arm
  exactly to its k = 0 loss. The stack has no attention, and the expert channel holds no state.
- **No frozen specialist transfers a benefit:** `NOT_ESTABLISHED` for both G1 and G2.
  - **G2** (fixed gate, qualified in Stage B) has 1.43× G0's loss at k = 32, pooled, and is better in only 5 of 12
    groups.
    - G0's advantage is significant (p = 9.7e-4) but holds in only 7 of 12 groups. That is below the rule's 2/3, so
      the frozen rule does not declare G2 worse than the FFN either.
  - **G1** (learned gate) has 0.872 of G0's loss, with p = 0.057 and 7 of 12 groups: not better, by the rule.
  - **The matched GELU specialist (G3)** has 1.10× G0's loss.
  - **With one support example** (k = 1), G2 against G0 is 1.002 and G2 against G3 is 0.975: within 3%, below the
    rule's 5%.
- **Reliance is not benefit.**
  - Frozen G2 leans heavily on its specialists: removing them from both blocks multiplies its k = 32 loss by 200
    (block 1's alone by 78).
  - Ablation costs G1 4.0× and G3 4.4×.
  - Yet G0, with no specialist at all, does as well or better. A trained network uses whatever branch it has; that
    is not a contribution the FFN could not make (amendment §3, attribution).
- **Without the specialist objective, the gate does not open.** G1's gates end at a median of 0.09 (block 0) and
  0.13 (block 1), from 0.1 at initialization. In Stage B the specialist objective drove the same gate to 0.45.
- **Run-to-run spread is large.** Per-group G2/G0 ratios range from 0.52 to 6.28, and G3/G0 from 0.55 to 4.23. At
  this budget, with one training run per arm and group, the runs vary more than the arms differ.
- **The budget was binding.**
  - Most arms selected their final checkpoint (median step 2,000). DEV loss was still falling, or had only just
    flattened, between 1,750 and 2,000 steps.
  - The comparison holds at the amendment's fixed budget. Longer training was not tested, and testing it would need
    a new plan.

**Conclusion.** Stage B's qualified specialist does not transfer to frozen-weight inference in the actual stack.
- Its Stage B advantage required explicit inner updates (Mode A).
- Trained end to end and then frozen, the same branch adds nothing measurable over the FFN alone.
- This is the gap the directive anticipated: a missing inference-time adaptation mechanism. R11 reports it and
  leaves it unimplemented (§9).

### 6.4 Learned routing

**The test.** G2 with two experts per block and a learned top-1 router, the repository's `SparseChannel`.
- **Training.** The same budget, on mixed families: each batch alternates them. The task loss is the only training
  signal, as frozen. The router's load-balance and z-losses have weight 0 in this configuration, and the Stage C
  trainer does not add them.
- **What is counted.** The share of each family's tokens routed to that family's majority expert, per block. The
  counts cover all 80 tokens of each of 12 TEST sequences at k = 32: support and query.
  - The amendment says "tokens".
  - The frozen module's docstring says "query tokens": a wording slip in the docstring. The code counts every token.

| Seed | block | smooth → expert (purity) | conditional → expert (purity) | separated | k=32 MSE smooth / conditional |
|---|---|---|---|---|---|
| 301 | 0 | 1 (0.515) | 0 (0.518) | yes | 0.0091 / 0.0097 |
| 301 | 1 | 1 (0.964) | 1 (0.986) | no | 0.0091 / 0.0097 |
| 302 | 0 | 1 (0.559) | 1 (0.653) | no | 0.0074 / 0.0064 |
| 302 | 1 | 1 (0.693) | 1 (0.647) | no | 0.0074 / 0.0064 |
| 303 | 0 | 0 (0.542) | 0 (0.547) | no | 0.0188 / 0.0141 |
| 303 | 1 | 1 (0.681) | 1 (0.684) | no | 0.0188 / 0.0141 |
| 304 | 0 | 1 (0.806) | 0 (0.658) | yes | 0.0105 / 0.0108 |
| 304 | 1 | 0 (0.984) | 0 (0.931) | no | 0.0105 / 0.0108 |
| 305 | 0 | 0 (0.559) | 0 (0.610) | no | 0.0109 / 0.0084 |
| 305 | 1 | 1 (0.530) | 1 (0.586) | no | 0.0109 / 0.0084 |
| 306 | 0 | 0 (0.624) | 0 (0.948) | no | 0.0066 / 0.0056 |
| 306 | 1 | 0 (0.675) | 0 (0.702) | no | 0.0066 / 0.0056 |

**What it shows** (purities as smooth / conditional):
- **Separation is rare.** Only 2 of the 12 blocks send the two families to different majority experts.
  - In one of them (seed 301, block 0) the split is at chance (0.52 / 0.52).
  - In the other (seed 304, block 0) it is partial (0.81 / 0.66).
- **Collapse is common.** Several blocks send both families mostly to one expert, for example seed 301 block 1
  (0.96 / 0.99) and seed 304 block 1 (0.98 / 0.93).
- **The headline purity flatters the result.** The mean purity of 0.69 counts that collapse as purity.
- **The router does not route by task.** Trained by the task loss alone, the learned top-1 router does not make its
  specialists family specialists.

So Stage B's arrangement (one specialist per family, by assignment) has no learned-routing counterpart in this
stack: **`INTEGRATION_GAP_LEARNED_ROUTING`**. Oracle task identity was never used at inference.

## 7. Costs: parameters, latency, memory

### 7.1 The scale models' CGC footprint, before and after (exact, `audit.json`)

| | R6 500M, legacy | R6 500M, micro | R6 1B, legacy | R6 1B, micro |
|---|---|---|---|---|
| Core-model CGC parameters (gates included) | 144,506,970 | 563,850 | 422,314,146 | 1,346,706 |
| R5 evidence-encoder CGC (legacy geometry) | 851,970 | 851,970 | 851,970 | 851,970 |
| Resident CGC | 145,358,940 | 1,415,820 | 423,166,116 | 2,198,676 |
| Model total | 507,683,413 | 363,740,293 | 994,692,069 | 573,724,629 |
| CGC share of the model | 28.63% | 0.39% | 42.54% | 0.38% |
| CGC active per token | 64,880,670 | 187,950 | 125,042,724 | 299,268 |
| Model active per token (as built) | 286,499,353 | 222,330,921 | 433,050,471 | 308,831,303 |

The micro layouts are NEW configurations with new totals: expert count, top-k, FFN widths, ERSA, attention, PLE and R5
are unchanged, and nothing at 500M or 1B was allocated or trained. "As built", the micro core config also changes the
encoder's CGC to 8 units (1,376,258 parameters; §9): the active-per-token figures include that encoder.

### 7.2 Latency (MEASURED, CPU, one thread; PERFORMANCE_ONLY; no GPU claim)

**The expert's forward**, d 64, 64 tokens (frozen `run_r11 latency`, repetitions in rotated order; median µs):

| Expert | median µs | / G0 |
|---|---|---|
| G0 | 47.7 | 1.00 |
| G1 | 99.7 | 2.09 |
| G2 | 96.7 | 2.02 |
| G2_POOLED | 100.4 | 2.10 |
| G3 | 104.4 | 2.19 |
| LEGACY_R10_A | 137.3 | 2.88 |

- At d64 the micro specialist **doubles** the expert's CPU time (2.02× for G2), although it adds 3.8% of the
  expert's parameters. Its work is tiny (about 630 multiply-adds per token); the time is framework dispatch of
  several small operators. The legacy R10 CGC costs 2.88×.
- A fused kernel would remove most of the dispatch; R11 did not write one, and these figures are not GPU
  throughput.

**The CGC operator alone**, 64 tokens (frozen `run_r11 latency`, FIXED order: micro bounded, micro E0, legacy; median µs):

| H | micro bounded | micro E0 | legacy routed | parameters micro / legacy |
|---|---|---|---|---|
| 64 | 38.7 | 31.0 | 81.5 | 632 / 9,216 |
| 768 | 213.0 | 74.1 | 3637.2 | 6,264 / 1,327,104 |
| 1024 | 237.7 | 85.9 | 7388.8 | 8,312 / 2,359,296 |

**The same operators in rotated order**, with the bound's elementwise work timed alone (supplement `bench_r11_operators`; median µs):

| H | micro bounded | micro E0 | legacy routed | the bound alone (24 units) |
|---|---|---|---|---|
| 64 | 44.2 | 35.1 | 90.2 | 8.6 |
| 768 | 170.3 | 82.9 | 3615.5 | 55.6 |
| 1024 | 183.7 | 100.1 | 7482.4 | 57.8 |

- **The micro operator is 17–41× cheaper than the legacy routed CGC** in CPU time at H = 768–1024, in both orders.
  It has 212–284× fewer parameters, so a microscopic kernel is dispatch-bound.
- **The bound's cost.**
  - At H = 64, where no operator in the loop is large, the bounded micro operator takes 8–9 µs more per 64 tokens
    than the unbounded one. That is what the bound alone costs (8.6 µs).
  - Beside the large legacy operator (H ≥ 768), every small operation slows: the bound alone then takes 56–58 µs.
    That is an effect of this CPU's caches and allocator, not of the arithmetic (24 tanh evaluations per token).
- **No GPU claim.**

### 7.3 Memory (MEASURED parameter and autograd bytes; exact accounting)

**The Stage B experts**, d 64, float32 (supplement `bench_r11_operators`):

| Arm | parameter bytes | specialist parameter bytes | saved for backward, 64 tokens | per token | Mode A transient state |
|---|---|---|---|---|---|
| G0 | 66,048 | 0 | 170,496 | 2664 | 0 |
| G1 | 68,580 | 2,532 | 222,180 | 3472 | 480 |
| G2 | 68,576 | 2,528 | 205,792 | 3216 | 480 |
| G2_POOLED | 68,576 | 2,528 | 205,792 | 3216 | 480 |
| G3 | 68,576 | 2,528 | 181,664 | 2838 | 480 |

- **Inference.** The trained specialist adds 2,528 bytes of float32 parameters at d64 (6,265 and 8,313 parameters per
  expert at H = 768 and 1024). Its per-token working set is 32 values (5 inputs, 24 unit pre-activations, 3 cells). A
  frozen forward saves nothing for backward.
- **Training.** Autograd saves 21% more per token for G2 than for G0 (3,216 against 2,664 bytes), 30% more for G1 and
  7% more for G3.
- **Mode A (research only).** The transient adapted core is 120 values per episode (480 bytes in float32), discarded
  after the episode.
- **At scale (bf16, resident).** CGC falls from 290.7 MB to 2.8 MB at 500M and from 846.3 MB to 4.4 MB at 1B
  (§7.1 × 2 bytes).

## 8. Runtime-weight immutability

- **Mode A** exists only in the offline research harness. Inner loops adapt a copy of the 120 core parameters through
  `functional_call` overrides and are discarded after each episode. Evaluation leaves every weight byte-identical
  (tested for Stage B and Stage C).
- **Stage C** evaluates frozen models under `torch.no_grad()`, with no optimizer; the support examples enter only as
  tokens in the sequence.
- **No runner saves weights.** The single write site of each R11 runner is its `--out` JSON record (tested).
- **No runtime persistent write, no inference-time parameter mutation, no new memory tier** was added. The R0/R1
  session fast weights are untouched and unused by R11.

**Status:** IMMUTABLE at inference, in every R11 path.

## 9. Unresolved interface requirements

None of these was resolved in R11, and none was worked around. Each is a decision for the owner.

1. **Frozen-weight inference needs a mechanism R11 does not have.**
   - The qualified specialist (Stage B) works by inner-loop updates of its 120 core parameters on observed support
     examples.
   - The target inference architecture keeps checkpoint weights frozen.
   - Stage C measured what end-to-end training with the support examples in context transfers without updates (§6).
   - A runtime adaptation mechanism would be a separate architectural decision: session fast weights, or explicit
     adaptive state like K1's. R11 leaves the runtime behaviour unimplemented, as directed.
2. **A training interface for the specialist objective in the language model.**
   - The specialist objective is episodic: support and query sets, with first-order MAML in an offline outer loop.
   - `lm_training.py` trains next-byte prediction only.
   - Integration would need an episodic data contract, and a rule for how that outer loop coexists with the
     language-model loss. Not built.
3. **Routing.**
   - Stage B assigned each family its expert: a laboratory control.
   - Stage C's routing test asked whether a learned top-1 router separates the families by itself (§6.4).
   - Oracle task identity was never used at inference.
4. **The R5 evidence encoder's CGC is coupled to the core configuration.**
   - The encoder inherits `cgc_units` and the bound, so a micro core profile silently changes the encoder's CGC
     from 4 to 8 units (851,970 → 1,376,258 parameters).
   - A decoupling decision is needed before any micro scale model is built. R11 left the encoder alone
     (`CGC_BOUNDING_R11.md` §4).
5. **A fused kernel.**
   - On CPU the micro branch doubles the d64 expert's time, through dispatch of small operators (§7.2).
   - A fused kernel would be needed before any throughput claim: projection, 24 bounded units, Horner evaluation
     and readout in one pass.
   - GPU behaviour is unmeasured.
6. **The gate.**
   - The historical initialization (0.1) starves the branch's gradients 13× at the start, and the learned gate
     settles near 0.45.
   - The fixed `g = 1` performed better (§5).
   - An integration must choose the gate deliberately. R11 changes no default.
7. **The K1 boundary (A3).**
   - Retention under conflicting interference is K1's qualified capability: a separate mechanism with explicit
     state.
   - R11 is not authorized to implement K1 inside ERSA or the FFN expert, and did not (`CGC_CAPABILITY_R11.md` §4,
     §6).

## 10. What R11 does not establish

- **No language-model evidence.** No language model was trained with the micro specialist, so there is no
  bits-per-byte or generation claim.
- **No scale evidence.** The 500M and 1B micro layouts were counted exactly from meta tensors, never allocated or
  trained.
- **No GPU throughput.** Every timing is CPU dispatch on one thread.
- **Narrow tasks.** The evidence covers two synthetic families of R97W's teacher (smooth and conditional). Nothing
  is shown about other task families.
- **Stage B is a laboratory harness.** It has one expert per family, a synthetic hidden state, a fixed readout and a
  known assignment. Its qualification is of the expert-local mechanism under explicit offline adaptation, and
  nothing more.
- **The bounded operator is E0-derived, not E0, and it is not K1.** It is not the native 120-parameter CGC either,
  though its core has that geometry.
- **R10's retention weakness is untouched.** R11 did not test retention under conflicting interference, which
  belongs to K1.
- **No historical result changed.** R9's `CGC_NOT_ESTABLISHED`, R10's
  `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED` and R10's `ASSAY_NOT_LEARNED` all stand. They concern the legacy
  geometry inside language models, and R11 neither re-tested nor reinterpreted them.

## 11. Execution incidents and deviations

1. **Two Stage C records carry `+dirty`.** These are `c_smooth_s301.json` and `c_conditional_s301.json`.
   - **Why.** Provenance is stamped when a run finishes. At that moment the working tree held uncommitted edits to
     `PROVENANCE.md`, the benchmark supplement and the new records test.
   - **Why they stand.**
     - None of those files is imported by the Stage C runner.
     - Every imported source's digest in both records matches the frozen sources.
     - No package module changed between the recorded revision and the final head, except the benchmark supplement
       (tested: `test_a_record_written_beside_uncommitted_edits_ran_unchanged_code`).
   - **What followed.** The stopping rules forbid a rerun, so they were not rerun. The tree was kept clean for every
     later record, and all of those are clean.
2. **The frozen latency command timed its operators in a fixed order:** micro bounded, then micro E0, then legacy.
   Each micro measurement therefore follows the legacy operator's large working set. The performance-only supplement
   `bench_r11_operators.py` rotates the order and times the bound alone (§7.2). The frozen record is kept as it is.
3. **The supplement was extended before its first run.** Its memory measurement (parameter bytes and bytes saved for
   backward) was added because the report requires observed memory. It feeds no decision rule.
4. **Development decisions before the freezes** (labelled DEVELOPMENT and kept):
   - **The projection learning rates,** from the calibration record.
   - **The A1 residual-scale calibration.** The default `P_out` residual initialization gave an inner-loop gain
     near 0.1, so k = 8 equalled k = 0.
   - **Per-arm checkpoint selection.** Without it, training length alone decided the cubic-against-GELU comparison.
   - **Stage C's training length and selection cadence,** fixed from one development seed (`R11C_DEV`, 31). This
     included one ad hoc 2,000-step check that was not kept as a file; the amendment discloses it.
5. **The queues.** Stage C ran in two background queues of two threads each, within the amendment's two-hour
   budget. Stage C took 43 minutes of wall-clock time (20:46–21:29 UTC). No run failed, timed out or was retried.
6. **Naming.** `R11_PERFORMANCE_NOTE.md` is R10's performance handoff, written before this phase. It is not R11
   evidence.

## 12. Reproduction

```sh
# Tests (the R11 suites; JSON-only record tests need no PyTorch)
ELPIS_REQUIRE_TORCH=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider -q \
  tests/research/elpis_torch/test_r11.py tests/research/elpis_torch/test_r11_records.py

# Audit, parity, latency (seconds to minutes)
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11 audit --out /tmp/audit.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11 parity --out /tmp/parity.json

# One Stage A group (about 2 s), one Stage B seed (about 3 min), one Stage C group (about 6.5 min, 2 threads)
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11 stage_a --family smooth --seed 101 --out /tmp/a.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11 stage_b --seed 201 --out /tmp/b.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11_stage_c run --family smooth --seed 301 --out /tmp/c.json

# The analyses, from the committed records
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11 combine --out /tmp/study_r11.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r11_stage_c combine --out /tmp/c_study.json
```

Each confirmation command refuses a seed outside its plan and a plan that drifted from its runner. A rerun writes
only where `--out` says.
