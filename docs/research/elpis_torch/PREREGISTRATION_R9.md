# Elpis Torch R9 — preregistration: CGC attribution, memory and bandwidth accounting

**Status: FROZEN before any confirmatory run.**

- **Machine-readable authority:** `research/elpis_torch/results/r9/preregistration.json`.
- **Runner:** `research/elpis_torch/experiments/run_r9.py`.
- **Commit:** both files are committed and pushed in the same commit as this document, before any confirmatory run
  starts.
- **Digests:** every confirmatory record carries the SHA-256 of both files. `run_r9 combine` refuses a record whose
  digests differ from the files it runs with. A later edit of either file therefore makes the existing records
  non-confirmatory.

Nothing below may change after this commit: arms, widths, data, steps, endpoints, `delta_min`, classes, verdict
rules and stopping rules.

## 1. Question and hypotheses

**Question.** Does CGC contribute measurable language-modelling benefit to the R7 architecture beyond an ordinary
FFN given the same resources?

**Hypotheses** (lower bits/byte is better):

| | Claim under test | Predicted if CGC is useful | Falsified by |
|---|---|---|---|
| H1 (decisive) | CGC-specific benefit | A < C and A < C_mac on held-out code and prose | ties within `delta_min`, mixed signs, or C ≤ A |
| H2 | CGC adds over the smaller FFN-only model | A < B on code and prose | no domain where A is better |
| D | Reliance (diagnostic only) | — | — |

D is not a hypothesis test. A trained model's reliance on a component says nothing about whether the component beats
an alternative trained in its place.

A negative result does not invalidate the mathematical CGC operator, nor the native K1 retention evidence (a different
mechanism and task). It would mean only that CGC's incremental value **in this autoregressive architecture at this
scale** is not established.

## 2. Arms

All arms use R7's study model unchanged except for the routed channel:

- d 64, 2 ERSA blocks, both channels routed: 8 experts, top-2, normalized weights, no auxiliary losses;
- the R7 PLE (orders 2 and 3, 4 heads each, 65,537-row primes, 8-wide rows, fused before block 1);
- float32, byte tokenizer (vocab 259).

| Arm | Routed channel | PLE | Role |
|---|---|---|---|
| A | SwiGLU FFN 86 + gated CGC (48 channels, bottleneck 16, 4 units, preactivation bound 3.0, gate init 0.1) | yes | CGC-positive reference (R7's PLE model) |
| B | SwiGLU FFN 86, no CGC | yes | CGC absent from training |
| C | SwiGLU FFN **123**, no CGC | yes | **Decisive control:** closest total parameter count |
| C_mac | SwiGLU FFN **126**, no CGC | yes | Resource control: closest multiply-add count |
| D | Trained A, every CGC gate set to 0 at evaluation | yes | Reliance diagnostic |
| A0, B0, C0 | As A, B, C | **no** | Secondary factorial: CGC × PLE interaction |

**Resources** (EXACT; `run_r9 accounting`, meta device; computed before any training):

| Arm | Total parameters | vs A | Active parameters / token | vs A | Per-expert parameters | Per-expert multiply-adds | vs A |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4,635,576 | | 155,868 | | 23,681 | 24,256 | |
| B | 4,520,872 | -2.47% | 127,192 | -18.40% | 16,512 | 16,512 | -31.9% |
| C | 4,634,536 | **-0.022%** | 155,608 | -0.17% | 23,616 | 23,616 | **-2.64%** |
| C_mac | 4,643,752 | +0.18% | 157,912 | +1.31% | 24,192 | 24,192 | **-0.26%** |

A's multiply-adds include the CGC's 576 cubic terms per expert.

No single FFN width matches A on parameters and on multiply-adds at the same time:

- **C** is matched on parameters and spends 2.6% fewer multiply-adds in the routed channel.
- **C_mac** is matched on multiply-adds and holds 2.2% more parameters per expert.

Neither control is called matched on a resource it is not matched on. A must beat **both** for H1 to pass, so a CGC
"win" that only buys more compute does not count.

D keeps every CGC parameter and still computes the CGC; its accounting is A's.

## 3. Data and training

**Corpus.** R7's corpus:

- **Source:** read from git at `c05adff` — `src/` and `research/elpis_torch/` (`*.py`, without `upstream/`) and
  `docs/` (`*.md`).
- **Split:** a file is held out when SHA-256 of its path is 0 mod 10.
- **Digest:** `ab516f5f3db9f231…`. Any other digest makes a run NONQUAL.
- **Sizes:** train 275 files, 2,915,823 bytes; held-out code 15 files; held-out prose 5 files.

**Training:**
- **Optimizer:** AdamW (0.9, 0.95), learning rate 3e-3, linear warmup over 50 steps, then cosine decay to 0.1 × at
  step 1,500.
- **Regularization:** weight decay 0.01 on matrices; gradient clip 1.0.
- **PLE tables:** SparseAdam at 5 × the learning rate, no decay.
- **Batches:** 16 × 128 bytes, fed in chunks of 64.
- **Seeds:** 0, 1 and 2.
- **Pairing:** every parameter two arms share is initialized identically, from the seed and its name, and every arm of
  a seed sees the same batches in the same order.

**Exactly 1,500 steps, and what R7 actually ran.** R7's records show **1,432** completed steps, not 1,500. Its
seeded epoch holds 1,432 full batches for this corpus (22,922 rows); the iterator ran out, so its 1,500-step schedule
was cut at 95.5%.

R9 runs exactly 1,500 steps, as directed:
- the batch stream is R7's seeded epoch, continued by a second seeded epoch (seed + 1000);
- the first 68 batches of that second epoch are used;
- the stream is identical for every arm of a seed.

R7's protocol as executed is reproduced separately, as an integrity check (section 6).

**Evaluation.**
- **Batches:** 48 held-out batches per domain (fixed by seed 99), as in R7.
- **Primary endpoint:** held-out bits/byte at step 1,500.
- **Learning curve only:** evaluations at steps 250, 500, … 1,500. Nothing is selected, stopped or decided on them.

**Execution.** 2 threads per process and 2 processes at once, as R7 ran. Order:
1. the integrity run and the 12 primary training runs (A, B, C and C_mac × 3 seeds);
2. latency alone on the idle machine;
3. the 9 factorial runs;
4. combine.

## 4. Endpoints

**Primary.** Held-out code bits/byte and held-out prose bits/byte at step 1,500, for every seed. These are reported
individually and as paired differences against A.

**Secondary.** All recorded and reported; none decides the verdict.

- **CGC gates:** values at every evaluation step.
- **Gradient norms by parameter group:** training gradients after clipping at every evaluation step; one held-out batch
  at the end.
- **Routing on held-out batches:** shares, maximum over mean, coefficient of variation, router entropy, channels under
  1%.
- **Training stability:** status, and the loss at every evaluation step.
- **Free-running generation mechanics:** greedy continuation, 96 bytes, from fixed 64-byte held-out prompts —
  printable fraction, distinct 4-grams, byte runs, agreement with the source. Mechanics, not quality.
- **Quality versus cumulative training compute:** 3 × the measured forward matmul FLOPs per token × 2,048 tokens per
  step.
- **Decode and prefill latency:** one thread, batch 1, arms interleaved.
- **Quality per unit of measured cost.**
- **Peak RSS, and matmul FLOPs per token** counted by `FlopCounterMode`.

Measurement labels: EXACT (formula, or `nbytes`), MEASURED (counted or timed here), PROJECTED (EXACT quantities under
stated assumptions), BOUND (by construction).

## 5. Analysis plan

For a comparison of arm X against A on a domain, the paired difference is `bpb(X, s) − bpb(A, s)` for each seed s.
A positive difference means A is better. `delta_min` = **0.020 bits/byte**: about 1.2% of R7's reference code loss,
and more than twice the spread (0.018) of R7's paired PLE-versus-capacity differences on code.

**Classes**, per comparison and domain:

| Class | Condition |
|---|---|
| A_BETTER | every seed's difference > 0, and the mean ≥ `delta_min` |
| X_BETTER | every seed's difference < 0, and the mean ≤ −`delta_min` |
| NOT_DISTINGUISHED | anything else |

**Primary verdict (H1):**

| Verdict | Condition |
|---|---|
| CGC_BENEFIT_QUALIFIED | A_BETTER against C on code **and** prose, **and** A_BETTER against C_mac on code **and** prose |
| FFN_PREFERRED | C is X_BETTER on at least one domain, and A is A_BETTER against C on none |
| CGC_NOT_ESTABLISHED | every other complete outcome: ties, mixed domains, a win against C but not against C_mac |
| NONQUAL | any run of A, C or C_mac not completed within its limit; a corpus digest mismatch; a record with other preregistration or runner digests |

**Secondary verdict (H2, A against B):**
- **CGC_ADDS_OVER_SMALLER_FFN:** A is A_BETTER on both domains.
- **SMALLER_FFN_NOT_WORSE:** A is A_BETTER on no domain.
- **MIXED:** otherwise.

**Reliance (D).** `bpb(D) − bpb(A)` per seed and mean. Descriptive.

**Factorial** (secondary, descriptive):
- the same classes for A0 against C0 and B0;
- interaction per seed and domain: `(bpb(C) − bpb(A)) − (bpb(C0) − bpb(A0))`;
- no qualification decision.

**Intervals.** 95% t-intervals (df 2) are reported for transparency; with three seeds they decide nothing. No p-values.

**Latency.**
- If A's measured decode p50 exceeds C's by more than 10%, this is reported as a material latency cost of CGC.
- It does not change the quality classes.
- Latency is MEASURED on this CPU at d 64 and is not a hardware-independent cost.

## 6. Integrity checks

These are pipeline checks, not scientific endpoints.

- **R7 reproduction.** The integrity run trains `run_r7.study_model("ple", 0)` with `run_r7._train`, exactly as R7
  ran (one epoch, 1,432 steps). It must land within 0.005 bits/byte of R7's recorded seed 0: code 1.71834, prose
  2.19864. A FAIL is reported and investigated; it does not enter the verdict, because every arm runs in the same
  pipeline.
- **Construction identity.** `build("A", s)` must equal `run_r7.study_model("ple", s)` parameter by parameter for every
  seed.

## 7. Limits and stopping rules

**Limits:**
- 1,800 s per run;
- 3 hours for the primary runs;
- 2 hours for the factorial.

R7 measured about 200 s per run under the same thread layout.

**Stopping rules:**
- No early stopping and no step selection: the endpoint is step 1,500.
- A run that diverges or exceeds its limit is recorded as such and is not repeated; its arm cannot qualify.
- No seed is added and no confirmatory run is repeated after any outcome has been seen.
- Whatever the verdict, no larger model is trained in R9 to change it.
- The factorial runs only after every primary run has finished, and only within its budget.

**NONQUAL outcomes:**
- a corpus digest mismatch;
- a primary run that is missing, diverged or timed out;
- refused digests.

## 8. P3 — memory and bandwidth accounting (methodology frozen here)

**Context lengths.**
- 1,024, 8,192, 65,536, 262,144 and 1,048,576 tokens.
- History capacity is fixed at 1,048,576.
- Each length is reported both as allocated capacity and as live history.

**Configurations.**
- **The R9 study arms (d 64):** no history head, so their state does not depend on context.
- **The R8 timing geometry `enc256`:**
  - d 256, 4 ERSA blocks, 8 experts top-2 in blocks 1 and 3, the PLE before block 1;
  - an R5 head with a 256-wide encoder;
  - measured with R5 off, the reference reread and stored fine keys, PLE on and off.
- **`r6_500m` and `r6_1b`:** on the meta device only, never allocated.

**Labels.** EXACT, MEASURED and PROJECTED are kept in separate fields. A formula is never reported as a measurement.

**DDR3 scenario (ASSUMED platform, not this machine).**

| Assumption | Values |
|---|---|
| RAM | 8 GiB |
| Peak bandwidth | 10, 15 and 20 GB/s |
| Achieved fraction | 0.5 and 0.8 |
| Compute | 50, 100 and 200 GFLOP/s |
| Weight bytes per parameter | 4 (float32), 2 (bfloat16), 0.5625 (int4, group 32 with scales) |
| Batch | 1 and 4 |
| Cache-resident fraction of active weights | 0 and 0.5 |

- **Lower bound per token:** the larger of traffic / bandwidth and FLOPs / compute.
- **Active path only.** The bound counts only the active path: routed experts that do not run are not read. Resident
  bytes are reported separately.

**GPU scenario.** PROJECTED only, for 300 and 900 GB/s and 20 and 80 TFLOP/s. CUDA is NOT_RUN.

**Throughput target.** The 60–100 tokens/s objective is compared against the projections. It is never claimed.

**Retrieval quality.** The metric is frozen here even though R9 does not run it:
- **Metric:** exact-answer accuracy at answer positions, plus recall@64 (coarse) and recall@4 (fine) of the gold
  block.
- **Cases:**
  - latest binding: records with 1–3 revisions;
  - revision-sensitive: superseded binds and stale questions;
  - contradiction: negated binds;
  - provenance: scoped definitions with wrong-scope declarations.
- **Status: NOT_RUN in R9.**
  - No model trained for retrieval exists in R9's scope: the study arms have no history head, and R5's trained models
    were not kept as checkpoints.
  - Training one is outside R9's priority order.
  - R5's result stands: it did not solve autonomous recall.

## 9. What this study cannot show

- It covers one scale (d 64, about 4.6 M parameters with the PLE), one corpus of about 2.9 MB of this repository's
  own code and prose, and 1,500 steps. A result here is a result at this scale.
- It does not test the native K1 retention mechanism or Retention R3's task. Those are different mechanisms and
  different evidence.
- Bits/byte is not understanding, reasoning or code-generation ability.
- Decode latency on this CPU at d 64 is dominated by fixed overheads. It is reported, not generalized.
