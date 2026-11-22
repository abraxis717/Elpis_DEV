# Elpis Torch R9 — claim audit

**Purpose.** This ledger records what each piece of evidence in this repository supports, against the evidence itself.
For each claim it gives:
- the exact record behind it;
- its scope;
- the control it lacks;
- the wording the evidence permits;
- the extrapolations it does not support.

**Historical records are not edited.**
- Where a record is imprecise, the correction is made here, with a link to the original.
- Where a research README states more than its evidence supports, the README is corrected and points here.

**Statuses:**
- **SUPPORTED**: the wording below holds within the stated scope.
- **IMPRECISE**: a record or summary says more, or something different, than its own data; the correction is given.
- **NOT ESTABLISHED**: no evidence either way at the stated scope.
- **CONTRADICTED**: the evidence says otherwise.

## Ledger

### L1. Native K1 retention (Retention R3) — SUPPORTED within its scope

**Evidence.** `docs/research/ECS_RETENTION_R3_RESULTS.md`, `research/ecs_retention_r3/` (preregistration, spec RET3A).

**Result.** On 32 fresh QUAL worlds:
- K1 held the sequence in 32/32 worlds.
- Median earlier-experience NMSE at D: 0.0013 for K1, against 0.690 for plain G1 and 0.0030 for rehearsal.
- K1's extra state is 28,552 B at every stage.
- Removing the state is bitwise G1.
- Resetting it degraded 31/32 worlds.
- Transplanting it gives bitwise continuation in 32/32.

**Scope.**
- A synthetic aliasing-conflict task family at fixed `d` (`CAPACITY_SCOPE`).
- `RESEARCH_ONLY`, `NO_LANGUAGE_CLAIM`, `SEMANTICS=NONE`.

**Missing control.** Capacity-limited regimes, other task families, larger `d`, language. The record itself names
all four.

**Permissible wording.** "Under the frozen synthetic aliasing-conflict regime, K1's fixed 28,552-byte state caused
sequential retention: removing, resetting or transplanting it changes the outcome as predicted."

**Unsupported extrapolation.**
- Elpis, or any Elpis language model, holds 28,552 bytes of state.
- K1 retention transfers to language modelling.
- K1 solves continual learning.
- The PyTorch CGC inherits these results.

### L2. Native K1 qualification (K1N) — SUPPORTED with the v2 contract's limits

**Evidence.** `docs/research/ECS_K1_NATIVE_RESULTS.md`; attestation
`research/ecs_k1_native/evidence/ecsg-k1-native.v2.attestation.json`.

**Result.**
- **K1N-v1: NOT_QUALIFIED.** Worlds r3qual-0019 and r3qual-0028 fell outside the planned float tolerances, for W
  relative error up to 8.8e-2; diagnosed as trajectory sensitivity.
- **K1N-v2: QUALIFIED.** It passed:
  - E1–E5;
  - L1, local numerical agreement over 16 registered steps;
  - L2, consolidation numerical agreement;
  - D1_Q, exact decision-record equivalence on all 32 R3 worlds, preserving OUTCOME_A;
  - D1_F, the same on 32 fresh worlds.

**Scope.** Decision-level and registered local-numerical contracts.

**Missing control.** Full-trajectory numerical identity across the R3 worlds was never the v2 contract, and v1 shows
it does not hold on 2/32 worlds.

**Permissible wording.** "Native K1 reproduces every R3 decision record and the OUTCOME_A disposition, with
registered local and consolidation numerical agreement."

**Unsupported extrapolation.** Native v2 reproduces the original R3 trajectories numerically.

### L3. The PyTorch CGC is "the ECS_G E0 cubic cell" — IMPRECISE (README corrected)

**Evidence.** `research/elpis_torch/cgc.py`:
- `CGCCore` is E0 only with `preact_bound=None`, and is tested to float64 round-off against `e0.py`.
- `ModelConfig.cgc_preact_bound` defaults to **3.0**, applying `b·tanh(u/b)` before the cubic ("extension, not E0").
- R6/R7 experts wrap the cells in learned `P_in`/`P_out` projections (bottleneck 16, readout 48→64).

**Scope.** Every PyTorch model since R0 that keeps the default bound.

**Missing control.** None needed; this is a definition.

**Permissible wording.** "An E0-derived, bounded cubic operator with learned input and output projections."

**Unsupported extrapolation.**
- The PyTorch models contain the E0 cell, or K1.
- L1 and L2 say anything about the PyTorch channel.

### L4. R0: FFN against CGC against FFN + CGC — SUPPORTED as a null

**Evidence.** `docs/research/elpis_torch/EVIDENCE_R0.md`.

**Result.** Toy transduction task, three seeds, parameter-matched: "There is no evidence here that CGC is better than
an FFN, and none that it is worse."

**Scope.** Toy task, tiny model.

**Missing control.** Compute matching; language data.

**Permissible wording.** As recorded.

**Unsupported extrapolation.** CGC improves models.

### L5. R2: ERSA has bounded recurrent state — SUPPORTED, for the mixer only

**Evidence.** `docs/research/elpis_torch/EVIDENCE_R2.md`: kernel correctness, gates and bounded state PASS (tests).

**Scope.** The ERSA state per layer, which is fixed in size.

**Missing control.** Whole-model accounting: weights, PLE tables, history ring, index, scratch.

**Permissible wording.** "ERSA's per-layer recurrent state has a fixed size."

**Unsupported extrapolation.** The whole architecture is O(1) or fixed-memory. R9 P3 gives the whole-model
accounting (`EVIDENCE_R9.md`).

### L6. R1 and R5: "constant memory" or "constant state bytes at every length" — IMPRECISE wording, accurate data

**Evidence.**
- `EVIDENCE_R1.md`: "constant memory", 276 MiB per stream.
- `EVIDENCE_R5.md` §4: "constant state bytes at every length".

**What the data show.**
- The state is a **fixed allocation sized by capacity**: the R5 ring holds 2^20 token ids, plus the descriptor index.
- It is constant across live history only because it is preallocated.
- It grows linearly with capacity.

**Missing control.** Live-growth accounting and whole-model memory (R9 P3).

**Permissible wording.** "The history state is a fixed allocation sized for its capacity; it does not grow with the
stream inside that capacity."

**Unsupported extrapolation.**
- Memory independent of context length in general.
- An O(1) system.

### L7. R5: "R5 did not solve autonomous recall" — IMPRECISE (R6–R8 summaries; repeated in `EVIDENCE_R8.md` §0/§6 and `PREREGISTRATION_R9.md` §8)

**Evidence.** `docs/research/elpis_torch/EVIDENCE_R5.md` §4 verdict: **PARTIALLY QUALIFIED**. At 1M tokens, on
synthetic tasks, with learned (label-free at inference) addressing:

| Measure | Result |
|---|---|
| Coarse recall@64 | ≥ 0.997 |
| Fine recall@8 | ≥ 0.995 |
| Records accuracy | 0.97–0.98 |
| Lexical accuracy | 0.96 |
| Revision-heavy latest binding | 0.83 — FAIL |
| Revision-heavy end-to-end | 0.85 — FAIL |
| Scoped symbols | 0.88–0.94 — FAIL |

Training used training-only retrieval supervision.

**Scope.** Synthetic streams, SEMANTICS=NONE.

**Missing control.** Natural-language retrieval; models trained without retrieval supervision.

**Permissible wording.** "R5 learned single-hop recall on synthetic records at 1M tokens, but failed its
revision-heavy and scoped-symbol gates; it is a synthetic retrieval substrate, not a language context."

**Unsupported extrapolation.**
- R5 provides million-token comprehension.
- The opposite, that R5 retrieves nothing autonomously.

### L8. R6: routed FFN + CGC experts — SUPPORTED as mechanics; usefulness not established

**Evidence.** `docs/research/elpis_torch/EVIDENCE_R6.md` §3. At 400 steps, d 64, held-out bits/byte:

| Variant | Bits/byte |
|---|---|
| Routed FFN only | 2.593 ± 0.027 |
| Routed FFN + CGC | 2.581 ± 0.025 |
| The same with a frozen random router | 2.562 ± 0.023 |

**Scope.** "Differences between routed variants are within one standard deviation"; the gain over dense is
attributed to capacity.

**Missing control.** A parameter- or compute-matched FFN-only arm (the FFN-only arm had 26% fewer parameters).

**Permissible wording.** "Routing and CGC experts work mechanically; no CGC- or routing-specific benefit was
measured."

**Unsupported extrapolation.**
- CGC experts improve language modelling.
- Routing specializes.

### L9. R7: the PLE improves small-scale bits/byte — SUPPORTED within scope; step count IMPRECISE

**Evidence.** `docs/research/elpis_torch/EVIDENCE_R7.md` §4 and `results/r7/study.json`.

**Result.** Three seeds, d 64, R7's corpus:

| Variant | Held-out code, bits/byte | Held-out prose, bits/byte |
|---|---|---|
| R6 | 1.827 | 2.284 |
| R6 + PLE | 1.732 | 2.203 |

The PLE won against random rows and against a 97-expert capacity control, on every seed and both domains.

**Correction.** The text says "1,500 steps"; the records show **1,432** completed steps. The epoch ran out, and the
1,500-step schedule stopped at 95.5% (`CLOSEOUT_R8.md`).

**Scope.**
- About 4.6 M parameters, 4.2 M of them in the table.
- About 2.9 MB of this repository's own text.
- A small held-out set: 15 code files and 5 prose files.

**Missing control.** Scale; other corpora; downstream tasks.

**Permissible wording.** "At this scale the tested PLE configuration lowered held-out bits/byte against R6, against
content-free rows and against the same parameters spent on experts."

**Unsupported extrapolation.**
- Any CGC benefit: every arm contained CGC, and the effect is the PLE's.
- Gains at scale.
- Reasoning or code-generation ability.

### L10. R8: stored fine keys — SUPPORTED as a computation result only

**Evidence.**
- `docs/research/elpis_torch/EVIDENCE_R8.md` and `CLOSEOUT_R8.md`.
- The R5 read costs 1.8–8.5× less per decoded token on CPU.
- Prefill reading at every position is 5–32× faster from 64K.
- Selections are bitwise identical in the float64 tests.
- The tier costs +128 or +256 MiB per stream at 1M.

**Scope.** CPU only, untrained timing models; CUDA NOT_RUN.

**Missing control.** GPU; trained models.

**Permissible wording.** "Stored fine keys preserve the R5 reference selections while removing most of its
re-encoding cost, at a fixed memory cost."

**Unsupported extrapolation.**
- Retrieval accuracy, semantic understanding or reasoning.
- GPU throughput.
- The 60–100 tokens/s target.

### L11. "28,552 bytes of state" — SUPPORTED only for K1 (see L1)

**Evidence.** Retention R0/R2/R3 records only.

**Unsupported extrapolation.** The number describes Elpis, ERSA, R5, R8 or the PyTorch models. It describes none of
them (R9 P3).

### L12. Yang–Mills / HCA (`research/yang_mills_hca/`) — frozen shadow evidence, NOT production-admitted

**Evidence.** `research/yang_mills_hca/README.md`, `RESULTS.md`, `ARCHITECTURE.md`, `PROVENANCE.md`.

**What the line contains.**
- An 88-parameter equivariant recurrent HCA: six frozen R77/R78 checkpoints, two 4×4 state channels per node, and
  SO(2)×SO(2) gauge transport.
- R118/R119 research engineering.

**Open holds.**
- `R93_GATE=ENERGY_NORMALIZED_DENOMINATOR_HOLD`.
- "Teacher-forced local transition fidelity remains poor."
- The listed unqualified seams include frame transport and end-to-end gauge covariance.
- Every frozen protocol marks CGC `SHADOW_ONLY`.

**Permissible wording.** "A frozen research line with historical equivariance checks on its evaluated graphs."

**Unsupported extrapolation.**
- Transport of CGC or K1 information has been shown.
- A production geometric memory exists.

R9 Gate A is HOLD (`GATE_A_PROTOCOL_R9.md`).

### L13. R9: CGC attribution in the R7 architecture — NOT ESTABLISHED

**Evidence.**
- `EVIDENCE_R9.md` §2.
- `results/r9/study_cgc_attribution.json` and the 21 run records `results/r9/cgc_*.json`.
- All of these were produced under `PREREGISTRATION_R9.md` (frozen at `3bdd449`).

**Result.**
- **Primary verdict: `CGC_NOT_ESTABLISHED`.** A did not beat the parameter-matched FFN arm C or the multiply-add-matched
  arm C_mac on either domain.
  - C was lower than A on every seed and both domains, by 0.014 (code) and 0.009 (prose) bits/byte on average. That
    is below the preregistered 0.020, so the class is `NOT_DISTINGUISHED`, not `X_BETTER`.
- **Secondary: `SMALLER_FFN_NOT_WORSE`.** Removing CGC and its parameters (B) was not distinguished from A.
- **Reliance.** Zeroing the trained A's gates at evaluation (D) raised the loss by 0.030 (code) and 0.033 (prose)
  bits/byte. The trained model uses its CGC path. That shows reliance, not superiority.
- **Without the PLE (factorial).**
  - B0 was worse than A0 (`A_BETTER`, 0.022 and 0.032).
  - The parameter-matched C0 was not distinguished from A0.
  - The CGC parameters are useful as parameters. Nothing shows they are better spent on CGC than on FFN width.
- **Latency.** A decoded 1.11× slower than C at p50, above the preregistered 1.10, so the difference is material.

**Scope.**
- R7's d 64 study model.
- 1,500 steps, 3 seeds.
- R7's corpus and split, on CPU.

**Missing control.** Larger models, longer training, other data. R9 does not run them, by its own rules: no rescue by
scaling.

**Permissible wording.** "In R7's architecture, under a preregistered parameter- and compute-matched comparison, CGC
showed no measurable benefit over FFN width, and cost 11% decode latency at this scale."

**Unsupported extrapolation.**
- "CGC is useless", or "CGC is harmful at scale": neither was tested.
- "The FFN is better" as a qualified result: the direction favoured C, but not by the preregistered margin.

### L14. Per-token memory traffic of the scale configurations (R9 P3) — corrected figures

**Evidence.** `EVIDENCE_R9.md` §3, `results/r9/p3_account.json` and `results/r9/p3_trace.json`.

**What was corrected.**
- An R9 working draft reported 614 MiB per decoded token for `r6_500m` at 1M context (stored fine keys, bfloat16
  weights). It counted the evidence encoder's 65,536 × 256 lookup table (32 MiB) as matrix weights read every token.
  A decoded token reads 200 of its rows (2,600 when every candidate is reread).
- The corrected compulsory figure is 583 MiB (647 MiB once the R5 ring has wrapped).
- The executed operand traffic of the same step, in float32 as run, is 2,050 MiB (2,114 MiB wrapped).

**Permissible wording.**
- "Per-token traffic of r6_500m is dominated by weights (about 515 MiB at bfloat16)."
- "The only per-token term that grows with context is the R5 descriptor scan (1 KiB per 32-token block), plus a
  wrapped-index copy once the ring has wrapped."
- "ERSA reads no history."

**Unsupported extrapolation.**
- Measured DRAM traffic: no hardware counters in this VM.
- GPU behaviour: CUDA NOT_RUN.
- Decoding in bfloat16 on CPU: this implementation does not.

## README corrections made in R9

`research/elpis_torch/README.md`, with pointers back to this ledger:

1. **The introduction (L3).** It called the operator "the ECS_G E0 cubic cell … inside every block". It now
   describes an E0-derived bounded operator with learned projections, in every block's dense channel from R0 to R5
   and inside routed experts from R6 on.
2. **Claim status.** A pointer to this ledger was added, with R9's verdict.
3. **"Not here, on purpose".** It still listed "MoE routing" after R6 admitted token-routed channel experts by the
   owner's directive. It now reads "MoE routing beyond R6's token-routed channel experts".

## R9 result

The verdict wording follows `PREREGISTRATION_R9.md` §5 exactly.
- Primary: `CGC_NOT_ESTABLISHED`.
- Secondary (B): `SMALLER_FFN_NOT_WORSE`.

R9 adds no claim that CGC improves the PyTorch architecture. It keeps every adverse result (`EVIDENCE_R9.md`).

## R10 additions (the R9 verdict above is unchanged)

### Scope of L13: `ASSAY_CANNOT_TEST_META_LEARNING_CLAIM`

R9's executed verdict stays **`CGC_NOT_ESTABLISHED`**. Its scope is static predictive efficiency:
- **What it established:** static held-out bits/byte at d 64 did not show an incremental CGC benefit over matched
  FFN width.
- **What it did not test:** whether CGC helps a model adapt, or meta-learn. Its assay (next-byte prediction on a
  fixed corpus) gives the model nothing new to adapt to within a sequence. So it neither establishes nor refutes a
  CGC-specific adaptation or meta-learning benefit.
- **The complementary-component hypothesis is untested.** The claim is that CGC complements the FFN, rather than
  replacing it, in adapting to new relationships and resisting interference. R9 did not test it in the current
  transformer.

`ASSAY_CANNOT_TEST_META_LEARNING_CLAIM` is a scope assessment of R9's assay. It does not replace R9's verdict
retroactively. R10 tests the hypothesis directly (`PREREGISTRATION_R10.md`).

### L15. Three CGC-related mechanisms that must not be conflated

| Mechanism | What it is | What its evidence shows | What it does not show |
|---|---|---|---|
| **Microscopic R97W** (historical; the separate CGC repository, summarized in `research/cgc_lao/EVIDENCE_LEDGER.md`) | A three-output, 120-parameter CGC against a 120-parameter GELU network, with first-order Reptile meta-learning: eight gradient-based inner adaptation steps per episode (episode-local fast weights) | 48/48 paired held-out adaptation comparisons won (4 groups × 12) | Inference-time in-context learning: its adaptation is gradient descent on the weights |
| **Native K1** (Retention R3; L1) | A bounded additional state (28,552 B) carrying conflicting experiences under a frozen synthetic regime | Causal retention: removal, reset and transplant change the outcome as predicted | Anything about the PyTorch CGCTransform, which has no such state |
| **PyTorch `CGCTransform`** (`research/elpis_torch/cgc.py`; L3) | A stateless cubic feed-forward transformation with learned input and output projections, inside routed experts beside an FFN | R9: no static-prediction benefit over matched FFN width at d 64 (L13) | — |

**Where in-context adaptation can come from.** `CGCTransform` owns no addressable or mutable temporal memory. Any
in-context adaptation in the Elpis transformer must be carried by:
- model activations;
- ERSA's recurrent state;
- window attention;
- R5 history, where enabled.

The question R10 asks is therefore whether CGC improves the use of those mechanisms. A result that credits CGC with
their capability is not admissible. Neither R97W nor K1 is evidence about it.

## R10 result

### L16. R10: CGC and in-context adaptation and retention — NOT ESTABLISHED (adaptation); ASSAY_NOT_LEARNED (retention)

**Plan.** `PREREGISTRATION_R10.md`, frozen at `34cdbf5`.

**Setup.** R9's d 64 backbone with a 64-token attention window. Arm A (FFN + CGC) was compared with:
- C: parameter-matched FFN;
- C_mac: multiply-add-matched FFN;
- B: the same FFN without CGC;
- an attention-free pair, D_A against D_C.

Every arm trained 3,000 steps on the same episode streams. There were three screening seeds.

**Supported.**
- **Adaptation: `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`.** The assay was learned (best arm 0.80; floor
  0.30), and screening was not favorable:
  - A − C ADAPT −0.143;
  - A − C_mac −0.155;
  - D_A − D_C −0.054, negative on every seed.

  The CGC arms adapted worse to new in-context bindings.
- **Retention: `ASSAY_NOT_LEARNED`.** The best arm reached 0.173 against a floor of 0.20. No arm kept earlier
  conflicting bindings through interference. So no CGC retention benefit is established, and R10 cannot say
  whether CGC would help a model that does retain them.
- **Attribution, by state resets.**
  - In-window retention is attention's.
  - Retention behind a gap is the ERSA state's.
  - Zeroing CGC at evaluation changes accuracy by at most 0.028. The CGC arms' deficit therefore arises in
    training, not in what CGC computes at inference.
- **Cost.** CGC costs 10% decode latency (A/C 1.101; D_A/D_C 1.104) and about 22% training time.

**Not supported.**
- Any claim that CGC improves in-context adaptation or retention in this architecture at this scale.
- Any claim credited to CGC from the exploratory patterns in `EVIDENCE_R10.md` §8, which were not preregistered:
  - retention behind a gap in the window arms, which is ERSA-carried and reverses in the attention-free pair;
  - held-out shift learning, where D_A is ahead of D_C on every seed but at low levels.

**Unchanged.**
- R9's `CGC_NOT_ESTABLISHED`.
- L15's separation of R97W, native K1 and `CGCTransform`.
- Gate A HOLD.


## R11 result (the R9 and R10 verdicts above are unchanged)

### L17. R11: a bounded, specialized expert-local CGC — QUALIFIED under explicit offline adaptation only; NOT_ESTABLISHED for frozen-weight transfer (Stage C)

**Plans.**
- `PREREGISTRATION_R11.md`, frozen at `47537ef`.
- The Stage C amendment, `PREREGISTRATION_R11_STAGE_C.md`, frozen at `369e333` before any Stage C data.

**Evidence.** `EVIDENCE_R11.md`, from records whose verdicts `test_r11_records.py` recomputes independently.

**Supported.**
- **The footprint correction** (exact; `CGC_BOUNDING_R11.md`).
  - The legacy geometry makes CGC 28.6% of the R6 500M layout and 42.5% of the 1B layout. That is a second large
    network, not a specialist.
  - The opt-in micro profile (R97W's 5 / 3 / 8 core, 120 parameters, adapter `8H + 121`) takes CGC to 0.39% and
    0.38%, within a 5% per-expert budget.
  - The defaults, and every historical count, are unchanged.
- **Naming** (consistent with L3).
  - `E0_REFERENCE` equals E0 within 1e-12.
  - `BOUNDED_SPECIALIST` (b = 3) is E0-derived, not E0, and its departure is quantified.
- **Stage A.**
  - On fresh R97W-protocol episodes, faithful E0 meta-learns in every group and beats the 120-parameter GELU by 46%:
    `MICRO_REFERENCE_QUALIFIED`.
  - With learned projections, the bounded adapter qualifies: `PROJECTED_ADAPTATION_QUALIFIED`.
  - Fixed random projections cost 2.2× and do not qualify.
- **Stage B, inside the real `ChannelExpert`, in a laboratory harness with a known expert assignment.** The
  fixed-gate bounded micro CGC is `EXPERT_SPECIALIZATION_QUALIFIED` under explicit offline inner updates:
  - its adapted loss is 0.113 of the FFN alone's;
  - it is causal: ablation multiplies the loss by 10.2;
  - its specialist objective is necessary: the pooled objective gives 0.986 of the FFN alone;
  - it beats a matched GELU specialist: 0.851, 10/12 groups.
- **The learned gate (historical, initialized at 0.1).**
  - It is slot-functional, but without a CGC-specific advantage (0.973 against GELU, p = 0.078).
  - The gate is not limiting by the frozen rule. It starves early gradients 13× and settles near 0.45.
- **Mode B.** Without inner updates, both CGC specialists are worse than the FFN alone (2.76× and 6.10× at k = 0):
  `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES`.
- **Stage C.**
  - Trained end to end in the actual ERSA stack and then frozen, with the support examples in context, neither CGC
    specialist beats the FFN alone at k = 32: `NOT_ESTABLISHED` for G1 and G2. G2's loss is 1.43× G0's, pooled.
  - All in-context information is ERSA's: an ERSA reset after the support returns every arm to its k = 0 loss.
  - Ablating the frozen G2 specialist multiplies its loss by 200. That is reliance, not benefit.
  - A learned top-1 router does not separate the task families: `INTEGRATION_GAP_LEARNED_ROUTING` (2 of 12 blocks;
    mean purity 0.69).

**Not supported.**
- Any claim that CGC adapts or learns at inference with frozen weights (Stage C: `NOT_ESTABLISHED`).
- Any language-model, scale (500M / 1B) or GPU-throughput claim.
- Any claim that Stage B's qualification transfers outside its harness:
  - one expert per family;
  - a synthetic hidden state;
  - a fixed readout;
  - two R97W-form task families.
- Promotion of the mechanical qualification (tests and exact records) into a specialist result (A4).
- Any claim about retention under conflicting interference: that belongs to K1, which R11 did not touch (A3).

**The R10 retention finding, restated (A2; R10's verdict is unchanged).**
- R10's verdict stays `ASSAY_NOT_LEARNED`.
- The tested d 64 transformers learned regime-conditioned bindings (pre-interference accuracy often above 0.90).
- They did not reliably preserve earlier bindings through conflicting interference: best aggregate 0.173 against
  the 0.20 floor.
- This identifies a weakness of the tested architectures, not a failure to learn the task.
- It shows neither that every attention or ERSA configuration forgets, nor that K1 would solve the task if
  integrated.

**Unchanged.**
- R9's `CGC_NOT_ESTABLISHED`.
- R10's verdicts.
- L15's separation of R97W, native K1 and `CGCTransform`.
- Gate A HOLD.

## R12 result (the R9, R10 and R11 verdicts above are unchanged)

### L18. R12: the micro-CGC volatile fast state in the Stage C stack — mechanism EXACT; specialist advantage NOT_ESTABLISHED in both regimes; P5: no headroom over in-context inference

**Plans.**
- `PREREGISTRATION_R12.md`, frozen at `e01b2b2`.
- The P5 amendment, `PREREGISTRATION_R12_P5.md`, frozen at `4561a9a` before any P5 data.

**Evidence.** `EVIDENCE_R12.md`, from records whose verdicts `test_r12_records.py` recomputes independently.

**Supported.**
- **The volatile fast state** (mechanical and numerical; engineering, not a specialist result).
  - It is exact: cached = full recomputation, and the analytic gradient equals autograd to at most 6.9e-17.
  - It is coherent: last block only, with the states unchanged.
  - It is isolated: private per session, never serialized, and the checkpoint is never written.
  - It is 960 bytes. 8 updates take 15–25 ms on one CPU thread.
- **R11's Stage C models rebuild bit for bit** from their recorded seeds: 48 of 48 arms.
- **T2.** With an adaptation-trained initialization, the restored update is steep, target-driven and CGC-specific.
  G2 after 8 steps is 0.044× its frozen loss, 0.116× the shuffled-target update, and 0.519× the matched fast GELU.
- **P5, diagnosis only.**
  - R11 Stage B's specialization reproduces on the R12 groups: 0.079 to 0.091× the FFN alone, and 0.69 to 0.80× the
    GELU.
  - The task coordinates reach the specialist (R² 0.998 from its input).
  - A fixed `P_in` suffices, and the specialist's identity matters.

**Not supported.**
- **Any claim that restoring the update recovers a specialist advantage in the Stage C stack.** It is
  `NOT_ESTABLISHED` for G1 and G2 in T1 and T2. Fast G2 against the FFN-only stack is 2.30 in T1, and 0.982
  (p = 0.66) in T2.
- **Any claim that the CGC fast update beats ERSA's in-context inference on these tasks.** Even under exact
  coordinate access it is 1.85 to 2.34× worse.
- **Any claim the harness cannot carry.** That covers any language-model, scale, GPU, sidecar or GGUF claim, any
  learned-routing claim, and any claim outside R97W's regression families.

**Unchanged.**
- R9's `CGC_NOT_ESTABLISHED`.
- R10's verdicts.
- R11's verdicts and records. R11 remains historical authority: R12 rebuilt its models and changed none of its
  conclusions.
- L15's separation of R97W, native K1 and `CGCTransform`.
- Gate A HOLD.

## R13 result (the R9 to R12 verdicts above are unchanged)

### L19. R13: the Geodome cognitive substrate admission — P0–P4 mechanically PASS; distributed competence HOLD; no advantage claimed

**Plans.**
- R13 ran no preregistered competence study. The owner asked for the P0–P4 results first.
- `PREREGISTRATION_R13_P5.md` is a DRAFT for review. It is not frozen, and no data was collected under it.

**Evidence.** `EVIDENCE_R13.md`, from records (`results/r13/`) whose gates and dispositions `test_r13_records.py`
recomputes from the raw rows.

**Supported (mechanical and numerical; engineering, not a cognitive result).**
- **Geodome Reference API v1.0.0 is admitted as a frozen dependency.** HexaFlake `4e5fe566` is vendored byte for
  byte and verified against its pins and its own `PUBLIC.sha256`, and imported in isolation. Its identity
  invariants hold: 144002 vertices, 432000 edges, 864000 arcs, and the mapping and edge digests.
- **The tensor executor equals Geodome's own scheduler on matched semantics.** Round records, statuses, rounds and
  visits are equal, and rows agree within 6.1e-16, in dense and sparse residency. All four stop statuses were
  exercised.
- **The synchronous schedule equals the scheduler on the class where the two coincide** (4.4e-16). Outside it they
  differ, and are not claimed equal.
- **Gradients.**
  - They reach all 19 kernel tensors through 5 hops.
  - They are exactly zero beyond the hop bound.
  - float64 gradcheck passes end to end through the actual routed final channel.
  - The cognitive path makes no host extraction, and one call is one bounded graph.
- **The bridge to the actual EvidenceLM**, in 18 variants:
  - The disabled path and the zero gate are bit-exact.
  - Reasoning never moves the external clock.
  - Admission is causal, canonical, live and an exact reread.
  - All 31 refusals are atomic.
  - Sessions are isolated, the backbone is immutable, and reset is clean.

**Not supported.**
- **Any claim that the substrate computes anything useful.** No task was trained or evaluated;
  `DISTRIBUTED_COMPETENCE` is `HOLD`.
- **Any cognitive, geometric or semantic advantage.** `COGNITIVE_ADVANTAGE` is `NOT_CLAIMED`.
- **Learned addressing** (P6, `NOT_STARTED`), **CGC admission** (P7, `NOT_STARTED`) and **HCA88**
  (`NOT_ADMITTED`).
- **Any language-model, scale, GPU or throughput claim.** The latencies are one CPU container's measurement.

**Unchanged.**
- R9's `CGC_NOT_ESTABLISHED`.
- The R10–R12 verdicts.
- Gate A HOLD.
- No earlier record or existing source changed; `identity.json` checks this against `d73cdde`.
