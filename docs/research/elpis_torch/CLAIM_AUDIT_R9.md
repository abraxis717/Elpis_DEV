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
