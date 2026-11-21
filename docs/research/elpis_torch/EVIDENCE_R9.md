# Elpis Torch R9 — evidence: Boardy adversarial qualification

**Principal question.** What measurable benefit, if any, does CGC contribute to the current Elpis PyTorch
architecture beyond a parameter-matched FFN, and does that benefit survive honest computation, memory and
evidence-retrieval accounting?

**Answer.** None was measured.
- Under a preregistered comparison that matches parameters and compute, R7's model with CGC did not beat the same
  model with the CGC's resources spent on FFN width. The FFN arm was ahead on every seed and both domains, by less
  than the preregistered margin.
- The trained models do use their CGC path, but that is reliance, not benefit.
- CGC costs 11% decode latency at this scale.
- Accounting at the scale configurations shows that per-token traffic is dominated by weights. Nothing in it depends
  on CGC beyond its share of the weights.

R9 is a falsification phase. It changes no model code and touches nothing under `src/` or `native/`.
- Plan: `PREREGISTRATION_R9.md` (frozen at `3bdd449` before any confirmatory run).
- Design: `DESIGN_R9.md`.
- Ledger of what each earlier record supports: `CLAIM_AUDIT_R9.md`.
- Gate A: `GATE_A_PROTOCOL_R9.md`.

## 0. Verdicts, by kind

| Kind | Verdict | Where |
|---|---|---|
| Correctness | R7 reproduced exactly; construction identity for seeds 0–2; frozen-plan drift refused; digest tampering and missing runs give NONQUAL; zeroed gates remove CGC exactly | §2.1, §5 |
| Scientific usefulness of CGC | **`CGC_NOT_ESTABLISHED`** (primary); **`SMALLER_FFN_NOT_WORSE`** (secondary, B) | §2 |
| CGC reliance | Zeroing trained A's gates raises the loss (`A_BETTER` for D, both domains) | §2.3 |
| R7 PLE | Attributable: −0.08 to −0.13 bits/byte in every arm; no consistent interaction with CGC | §2.5 |
| Memory | EXACT allocations agree with measured RSS. History state is a fixed allocation at capacity: 84 MiB (reread) or 340 MiB (stored keys) at 1M for `r6_500m`, beside 968 MiB of bfloat16 weights | §3 |
| Performance (CPU, measured) | A decodes 1.11× slower than C (material). R5/R8 decode costs by length are in §3.4 | §2.6, §3.4 |
| Performance (projected) | `r6_500m` at 1M, central DDR3 assumptions: at most 19.6 tokens/s (stored keys) and 6.5 (reread). The 60–100 target is inside the frozen ranges only at their most optimistic corner, and only with stored keys | §3.6 |
| Retrieval quality | NOT_RUN: the metric is frozen, no confirmatory retrieval evaluation in R9 | §3.7 |
| Integration (Gate A) | **HOLD**: no executable CGC/K1-to-transport seam exists, and none was invented | §4 |
| GPU | CUDA NOT_RUN | — |

## 1. P0 — R8 closeout

`CLOSEOUT_R8.md` records the post-merge facts. They were verified on GitHub against `main` at `4534fbd`:
- PR #78 is merged, and CI run 44553578427 on the merged SHA passed 16/16 jobs.
- R6, R7 and R8 are on `main` with no stranded commits.
- `src/` and `native/` are unchanged.
- Stored fine keys are opt-in; `fine_keys="reread"` stays the default.
- CUDA is NOT_RUN.
- Two native questions (GCC 15 warnings; clang 21 with Landlock ABI 8) stay open outside R9.
- R7's "1,500 steps" is corrected there to the 1,432 steps its records show.

R9 branches from that `main` and is one PR.

## 2. P2 — CGC attribution

**What ran.** Every run below completed, in the preregistered order (§7 of the preregistration):
- integrity;
- 12 primary runs (A, B, C, C_mac × seeds 0, 1, 2);
- latency, alone on the machine;
- 9 factorial runs (A0, B0, C0 × seeds).

Each training run is exactly 1,500 steps. Every record carries the SHA-256 of the preregistration (`9974d72d…`) and of
the runner (`47b46a9e…`). `combine` refused nothing and found every run (`results/r9/study_cgc_attribution.json`).

### 2.1 Integrity

- **R7 reproduced.** R7's seed 0 was rerun under R7's own protocol as executed (one seeded epoch, 1,432 steps). It
  matched R7's recorded losses to 3.4e-6 (code) and 4.3e-6 (prose) bits/byte, inside the 0.005 tolerance: **PASS**.
- **Construction identity.** For seeds 0, 1 and 2, arm A's construction is bitwise R7's PLE model. Every arm shares
  every non-expert parameter with A at initialization.

### 2.2 Final held-out loss, bits/byte (lower is better)

| Arm | Routed expert | Code s0 | s1 | s2 | Mean | Prose s0 | s1 | s2 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | FFN 86 + CGC (R7) | 1.7118 | 1.7323 | 1.7310 | 1.7250 | 2.1960 | 2.1938 | 2.2119 | 2.2006 |
| B | FFN 86 | 1.7346 | 1.7214 | 1.7460 | 1.7340 | 2.2291 | 2.2129 | 2.2175 | 2.2198 |
| C | FFN 123 (parameters matched) | 1.7104 | 1.7067 | 1.7167 | 1.7113 | 2.1933 | 2.1882 | 2.1926 | 2.1914 |
| C_mac | FFN 126 (multiply-adds matched) | 1.7156 | 1.7169 | 1.7248 | 1.7191 | 2.1924 | 2.1922 | 2.2033 | 2.1960 |
| D | A, gates 0 at evaluation | 1.7405 | 1.7610 | 1.7625 | 1.7547 | 2.2313 | 2.2247 | 2.2434 | 2.2331 |

### 2.3 Paired differences against A (other − A; positive: A better), and the frozen classes

| Comparison | Code per seed | Mean | Class | Prose per seed | Mean | Class |
|---|---|---:|---|---|---:|---|
| C − A | −0.0013, −0.0256, −0.0143 | −0.0138 | NOT_DISTINGUISHED | −0.0027, −0.0056, −0.0193 | −0.0092 | NOT_DISTINGUISHED |
| C_mac − A | +0.0038, −0.0154, −0.0062 | −0.0059 | NOT_DISTINGUISHED | −0.0036, −0.0016, −0.0087 | −0.0046 | NOT_DISTINGUISHED |
| B − A | +0.0228, −0.0109, +0.0150 | +0.0090 | NOT_DISTINGUISHED | +0.0330, +0.0190, +0.0055 | +0.0192 | NOT_DISTINGUISHED |
| D − A | +0.0287, +0.0287, +0.0315 | +0.0296 | A_BETTER | +0.0353, +0.0309, +0.0315 | +0.0325 | A_BETTER |

**Primary verdict: `CGC_NOT_ESTABLISHED`.**
- CGC qualifies only if A beats both C and C_mac on both domains (`PREREGISTRATION_R9.md` §5). It beat neither.
- The direction is adverse to CGC. C was lower than A on all six seed-domain pairs, and C_mac on five of six.
- The margins (0.014 and 0.009 bits/byte for C) are below `delta_min` = 0.020, so the frozen class is
  NOT_DISTINGUISHED, not X_BETTER. `FFN_PREFERRED` is therefore not the verdict, and it is not claimed.

**Secondary verdict (B): `SMALLER_FFN_NOT_WORSE`.** Removing CGC and its parameters was not distinguished from A at
this budget. CGC is 30% of each routed expert's parameters and 2.5% of the model's.

**Reliance (D).** The trained A uses its CGC path: removing it at evaluation raises the loss by 0.030 to 0.033
bits/byte on every seed. This is the expected effect of deleting a trained component. It says nothing about whether
the component is better than the alternative use of its resources, which is what C and C_mac test.

### 2.4 Matching, parameters and compute (EXACT, `results/r9/accounting.json`; MEASURED FLOPs from the runs)

| Arm | Total parameters | Active per token | Per-expert multiply-adds | Matmul FLOPs per token (measured) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| A | 4,635,576 | 155,868 | 24,256 | 372,352 | 556–558 |
| B | 4,520,872 | 127,192 | 16,512 | 315,008 | 521–523 |
| C | 4,634,536 (−0.022%) | 155,608 | 23,616 (−2.6%) | 371,840 | 526–531 |
| C_mac | 4,643,752 (+0.18%) | 157,912 | 24,192 (−0.26%) | 376,448 | 529 |

### 2.5 The no-PLE factorial and R7's PLE

| Arm | Code s0 | s1 | s2 | Mean | Prose s0 | s1 | s2 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1.8461 | 1.8110 | 1.8090 | 1.8220 | 2.3046 | 2.2724 | 2.2734 | 2.2835 |
| B0 | 1.8559 | 1.8508 | 1.8240 | 1.8436 | 2.3417 | 2.3116 | 2.2937 | 2.3157 |
| C0 | 1.8402 | 1.8815 | 1.7947 | 1.8388 | 2.2855 | 2.3190 | 2.2563 | 2.2869 |

Without the PLE:
- C0 − A0 is NOT_DISTINGUISHED on both domains (seed 1 is an outlier where A0 led by 0.07).
- B0 − A0 is **A_BETTER** on both: +0.022 code, +0.032 prose.
- Zeroing A0's gates costs 0.085 to 0.088 bits/byte (A_BETTER).

Without the PLE, then, CGC's parameters are worth having *as parameters*. Removing them without replacement hurts.
Spending them on FFN width does as well.

**R7's PLE is attributable, with or without CGC.** It lowers the loss in every arm:

| Arm | Code | Prose |
|---|---:|---:|
| A0 → A | −0.0970 | −0.0829 |
| B0 → B | −0.1096 | −0.0959 |
| C0 → C | −0.1275 | −0.0956 |

R7 reported 1.827 → 1.732 over 1,432 steps. The preregistered interaction, `(C − A) − (C0 − A0)`, has no consistent
sign:

| Seed | Code | Prose |
|---|---:|---:|
| 0 | +0.005 | +0.016 |
| 1 | −0.096 | −0.052 |
| 2 | +0.000 | −0.002 |

No CGC × PLE interaction is claimed.

### 2.6 Secondary measurements (decide nothing)

- **Gates.** CGC gates start at 0.1 and end lower: their mean is 0.059 to 0.072 for A and 0.081 to 0.088 for A0.
- **Gradients.** On held-out batches (seed 0), the CGC gradient norm is 0.042 and the gate's 0.050. The routed FFN's
  is 0.229 and the embedding's 0.250.
- **Routing.** Experts route unevenly in every arm: entropy about 1.0–1.2 of a 2.08 maximum, with some experts
  below 1% of tokens. No arm is distinguished by it.
- **Generation.** Mechanics of greedy continuations (printable fraction, distinct 4-grams, byte runs) do not
  separate the arms at this size.
- **Latency** (MEASURED; one thread; untrained seed-0 models; interleaved, rotating order):

  | Arm | Decode p50 (ms) |
  |---|---:|
  | A | 3.574 |
  | B | 3.279 |
  | C | 3.208 |
  | C_mac | 3.245 |
  | A0 | 3.022 |
  | B0 | 2.699 |
  | C0 | 2.745 |

  **A / C = 1.114, above the preregistered 1.10: material.** CGC's cubic cells and projections cost decode time
  beyond their multiply-add count at this width.

### 2.7 Provenance notes

- **`+dirty` records.** Two records name `3bdd449…+dirty`: `integrity.json` and `cgc_B_s0.json`. During the first
  minutes of the runs, untracked draft documents sat in the working tree; they were moved out before any other run
  started. The preregistration and runner digests of both records equal the committed files, which is what `combine`
  checks. The dirt was documentation, not code.
- **Clean records.** Every other record names `3bdd449` clean.
- **No reruns.** No run was repeated, no seed added, no threshold changed.

## 3. P3 — whole-model memory and bandwidth (Boardy Gate B)

The records are `results/r9/p3_*.json`, produced by `experiments/run_r9_p3.py` (`DESIGN_R9.md` §3).

**Four kinds of number, kept apart:**
- **EXACT:** allocations and parameter counts.
- **PROJECTED:** compulsory traffic and FLOPs under the frozen assumptions.
- **EXECUTED:** operand bytes of real decode steps.
- **MEASURED:** time, RSS and copy bandwidth on this VM.

**Not measured:**
- DRAM traffic: this VM exposes no hardware performance counters. Its event sources are breakpoint, msr, power,
  software, tracepoint and uprobe, and `perf_event_paranoid` is 2.
- HBM traffic: CUDA is NOT_RUN.

### 3.1 Persistent footprint at capacity 1M (EXACT; RSS-verified)

| Configuration | PLE | Resident parameters | Weights | ERSA state + tails | Window KV | Ring tokens | Descriptors + ids | Fine keys | History state total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r9_arm_A_d64` | on | 4,635,576 | 17.7 MiB fp32 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | **0.01** |
| `r8_enc256_reread` | on | 18,184,981 | 69.4 MiB fp32 | 0.03 | 0.00 | 4.00 | 32.25 | 0.00 | **36.28** |
| `r8_enc256_stored` | on | 18,184,981 | 69.4 MiB fp32 | 0.03 | 0.00 | 4.00 | 32.25 | 256.00 | **292.28** |
| `r8_enc256_reread` | off | 9,658,069 | 36.8 MiB fp32 | 0.03 | 0.00 | 4.00 | 32.25 | 0.00 | **36.28** |
| `r8_enc256_stored` | off | 9,658,069 | 36.8 MiB fp32 | 0.03 | 0.00 | 4.00 | 32.25 | 256.00 | **292.28** |
| `r6_500m_reread` | off | 507,683,413 | 968.3 MiB bf16 | 7.68 | 40.00 | 4.00 | 32.25 | 0.00 | **83.94** |
| `r6_500m_stored` | off | 507,683,413 | 968.3 MiB bf16 | 7.68 | 40.00 | 4.00 | 32.25 | 256.00 | **339.94** |
| `r6_1b_reread` | off | 994,692,069 | 1,897.2 MiB bf16 | 9.21 | 32.00 | 4.00 | 32.25 | 0.00 | **77.48** |
| `r6_1b_stored` | off | 994,692,069 | 1,897.2 MiB bf16 | 9.21 | 32.00 | 4.00 | 32.25 | 256.00 | **333.48** |

MiB unless stated. The `r6` weights are counted at bfloat16, as the projection assumes. Their state tensors are float32 as implemented. The `r6` scale models carry no PLE.

**Fixed allocation.** History state is allocated once, at capacity, whatever the live length:
- the R5 token ring holds 2^20 token ids;
- the descriptor index holds 32,768 blocks × 4 descriptors × 64 dimensions;
- the R8 fine-key tier, when stored, holds 32 tokens × 64 dimensions per block.

Within capacity, the live history fills the allocation and never grows it.

**A stopped first run.** The first `rss` run (reread) filled its 64K tokens with an R5 read at every position. It
was stopped after 35 minutes, before writing any record. Since `9873ba8` the fill ingests as R8's admission prefill
does (no read), and the measured decode step still reads.

**Measured RSS agrees.** On `enc256` (PLE on), creating a 1M-capacity state grew RSS by:
- 36.10 MiB against 36.28 MiB EXACT with reread keys (0.995);
- 292.10 MiB against 292.28 MiB with stored keys (0.999).

**Scratch the allocation excludes.** This is allocator scratch, outside the EXACT figure:

| Mode | Filling 64K tokens retained | One decode step retained | Process peak |
|---|---:|---:|---:|
| Reread | 24 MiB | 28 MiB | 461 MiB |
| Stored | 30 MiB | nothing | 634 MiB |

**Not any of these.** The 28,552-byte figure of K1 retention describes none of these configurations (L11).

### 3.2 Compulsory traffic per decoded token (PROJECTED), by length

| Configuration | PLE | 1K | 8K | 64K | 256K | 1M | 1M + 4K wrapped | GFLOP per token at 1M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `r9_arm_A_d64` | on | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | — | 0.000 |
| `r8_enc256_reread` | on | 25.1 | 27.9 | 29.6 | 35.6 | 59.6 | 123.6 | 10.769 |
| `r8_enc256_stored` | on | 23.2 | 23.7 | 25.4 | 31.4 | 55.4 | 119.4 | 0.851 |
| `r8_enc256_reread` | off | 24.6 | 27.4 | 29.1 | 35.1 | 59.1 | 123.1 | 10.769 |
| `r8_enc256_stored` | off | 22.7 | 23.2 | 24.9 | 30.9 | 54.9 | 118.9 | 0.851 |
| `r6_500m_reread` | off | 537.4 | 555.1 | 556.9 | 562.9 | 586.9 | 650.9 | 15.277 |
| `r6_500m_stored` | off | 535.5 | 550.9 | 552.7 | 558.7 | 582.7 | 646.7 | 1.742 |
| `r6_1b_reread` | off | 819.7 | 835.0 | 836.8 | 842.8 | 866.8 | 930.8 | 15.915 |
| `r6_1b_stored` | off | 817.2 | 829.6 | 831.4 | 837.4 | 861.4 | 925.4 | 2.065 |

MiB per decoded token. This is the compulsory traffic: each persistent byte the token needs, once, and no intermediates. Weights and window KV are at bfloat16 for `r6`, float32 otherwise.

### 3.3 The decode-size diagnosis: what a decoded token actually moves

A working draft reported 614 MiB per decoded token for `r6_500m` at 1M context (stored fine keys, bfloat16 weights).
That figure was decomposed and verified against the executed decode path (`p3_trace.json`) before any P3 record was
produced.

**Decomposition of the 614 MiB (draft, bfloat16 weights, float32 descriptors and ERSA state):**

| Component | MiB | Verdict |
|---|---:|---|
| Matrix weights on the active path | 514.45 | correct: about 270M parameters × 2 B, including the tied 65,536 × 768 output head (96 MiB) |
| Evidence encoder's `delta` table, counted as matrix weights | 32.00 | **error**: a 65,536 × 256 lookup table read by row; a decoded token reads 200 rows (0.10 MiB) |
| R5 descriptor scan | 32.00 | correct: 32,768 live blocks × 4 × 64 × 4 B, every token |
| Window KV (5 layers, 4,096 tokens, at bfloat16) | 20.00 | correct for bfloat16 KV (40 MiB as executed in float32) |
| ERSA state and convolution tails, read and written once | 15.35 | correct as compulsory traffic (executed: read three times, §3.3.3) |
| R8 fine keys of 64 candidates | 0.50 | correct |
| Token rereads | 0.001 | correct |
| Core-embedding rows read by the encoder | — | **omitted**: 0.29 MiB |

- **Corrected compulsory figure: 582.7 MiB at 1M.**
- **646.7 MiB in the steady state of an endless stream**, once the ring has wrapped (§3.3.4).

#### 3.3.1 What the step executes

`r6_500m`, stored keys, float32 as executed. Operand bytes per decoded token, by data class; the classes are disjoint
and sum to the total:

| Class / item | Stored keys, 1M | Stored keys, 1M + 4K wrapped | Reread, 1M |
|---|---:|---:|---:|
| **Weight traffic** | **1,019.56** | **1,019.56** | **1,050.07** |
| &nbsp;&nbsp;dense channels | 326.25 | 326.25 | 326.25 |
| &nbsp;&nbsp;routed experts top k | 281.25 | 281.25 | 281.25 |
| &nbsp;&nbsp;tied lm head | 192.00 | 192.00 | 192.00 |
| &nbsp;&nbsp;ersa projections | 181.29 | 181.29 | 181.29 |
| &nbsp;&nbsp;window attention projections | 30.00 | 30.00 | 30.00 |
| &nbsp;&nbsp;r5 head projections | 8.42 | 8.42 | 38.92 |
| &nbsp;&nbsp;routing matrices | 0.23 | 0.23 | 0.23 |
| &nbsp;&nbsp;norms | 0.12 | 0.12 | 0.12 |
| **KV / history reads** | **72.50** | **72.50** | **72.01** |
| &nbsp;&nbsp;window kv | 40.00 | 40.00 | 40.00 |
| &nbsp;&nbsp;r5 descriptor scan | 32.00 | 32.00 | 32.00 |
| &nbsp;&nbsp;r8 fine keys | 0.50 | 0.50 | 0.00 |
| &nbsp;&nbsp;r5 ring tokens | 0.00 | 0.00 | 0.01 |
| **ERSA recurrent state** | **95.08** | **95.08** | **95.08** |
| &nbsp;&nbsp;state sized intermediates | 64.72 | 64.72 | 64.72 |
| &nbsp;&nbsp;ersa state read | 22.50 | 22.50 | 22.50 |
| &nbsp;&nbsp;ersa state write | 7.50 | 7.50 | 7.50 |
| &nbsp;&nbsp;conv tail write | 0.18 | 0.18 | 0.18 |
| &nbsp;&nbsp;conv tail read | 0.18 | 0.18 | 0.18 |
| **Evidence re-encoding** | **145.22** | **145.22** | **1,640.57** |
| &nbsp;&nbsp;encoder intermediates | 122.84 | 122.84 | 1,608.82 |
| &nbsp;&nbsp;encoder and key projection weights | 21.59 | 21.59 | 21.59 |
| &nbsp;&nbsp;core embedding rows | 0.59 | 0.59 | 7.62 |
| &nbsp;&nbsp;encoder delta rows | 0.20 | 0.20 | 2.54 |
| **Embedding lookups** | **0.00** | **0.00** | **0.00** |
| **Other** | **717.63** | **781.63** | **759.46** |
| &nbsp;&nbsp;window attention recompute | 688.04 | 688.04 | 688.04 |
| &nbsp;&nbsp;channel ple norm intermediates | 14.10 | 14.10 | 14.10 |
| &nbsp;&nbsp;r5 per block bookkeeping and wrapped index copy | 8.27 | 40.27 | 8.27 |
| &nbsp;&nbsp;r5 rerank readout inject intermediates | 4.91 | 4.91 | 46.74 |
| &nbsp;&nbsp;embedding output and logits | 1.17 | 1.17 | 1.17 |
| &nbsp;&nbsp;r5 coarse scan intermediates | 1.13 | 33.12 | 1.13 |
| &nbsp;&nbsp;state commits | 0.01 | 0.01 | 0.01 |
| **Total, executed operand bytes** | **2,050.0** | **2,114.0** | **3,617.2** |
| Unique persistent bytes touched | 1,117.5 | 1,117.5 | 1,126.4 |

MiB per decoded token. In the stored-keys columns the step decodes at offset 16 inside a block; a step that completes a block adds 0.45 MiB (descriptor and fine-key writes).

**Weight traffic (1,020 MiB in float32):**

| Part | MiB |
|---|---:|
| Dense FFN + CGC channels | 326 |
| Routed experts (top-2 of 8) | 281 |
| Tied output head | 192 |
| ERSA projections | 181 |
| Window-attention projections | 30 |
| R5 head | 8 |

**ERSA recurrent state (95 MiB):**
- The state is read 22.5 MiB, three times per layer: S k, S q and retain·S.
- It is written 7.5 MiB.
- State-sized temporaries account for 65 MiB.

**Evidence re-encoding (145 MiB, stored keys):**
- The encoder runs twice per decoded token: the open block's 40-token window, then the 4 injected candidates.
- It reads its weights each time: 21.6 MiB.
- Its intermediates are 123 MiB.
- In reread mode it re-encodes all 64 candidates: 1,641 MiB.

**Window attention recompute (688 MiB).** This is the largest non-weight term, and it is bounded by the window, not
by the context. Rotary positions are relative to the window (`ScrollAttention.forward`), so every decoded token
re-rotates all 4,096 held keys of each of the 5 attention layers. It then:
- expands K and V threefold for grouped-query attention (`repeat_interleave`);
- concatenates them with the new token;
- computes logits over the window.

The KV bytes themselves are 40 MiB of this.

#### 3.3.2 Executed against compulsory, and DRAM

**Three figures per decoded token at 1M (stored keys, float32 as executed):**

| Figure | MiB |
|---|---:|
| Executed operand total | 2,050 |
| Unique persistent bytes touched (each weight once, gathered rows, each state tier's reads and writes once) | 1,117 |
| Compulsory projection at bfloat16 weights and KV (the corrected draft figure) | 583 |

The compulsory projection equals the unique figure after halving the weights and the KV. The trace test checks this
agreement to within the 17,408 weights of the descriptor projection, which runs only when a block completes.

**Logical bytes are not DRAM transfers.** The executed operand bytes count every read, including re-reads of
intermediates that a cache serves. The unique persistent bytes are the traffic a token cannot avoid when nothing
survives in cache between tokens. That holds here: per-token weights are 1,029 MiB in float32, and this host's L3 is
260 MiB, shared by the whole host.

**Real DRAM traffic lies between the two figures:**

| Configuration | Reread keys | Stored keys |
|---|---:|---:|
| `r6_500m`, float32 | 1.1–3.5 GiB | 1.1–2.0 GiB |
| Weights and KV halved to bfloat16 | 0.6 GiB and up | 0.6 GiB and up |

Where it lies depends on how much of the 0.7 GiB of window-attention intermediates and the encoder's intermediates
spill from cache. A DDR3-era CPU, with a few MiB of last-level cache, spills most of it. Neither end is measured.

#### 3.3.3 What scales with context length

Traced at 1K, 8K, 64K, 256K, 1M and the wrapped 1M + 4K. Totals in MiB per decoded token:

| Length | Total | Weights | Window KV | Descriptor scan | ERSA (state + temporaries) | Evidence re-encoding | Window recompute | R5 bookkeeping + wrapped copy | Unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,464.5 | 1,019.6 | 10.2 | 0.03 | 95.1 | 145.2 | 175.1 | 0.02 | 1,055.4 |
| 8K | 2,008.9 | 1,019.6 | 40.0 | 0.25 | 95.1 | 145.2 | 688.0 | 0.10 | 1,085.7 |
| 64K | 2,011.2 | 1,019.6 | 40.0 | 2.00 | 95.1 | 145.2 | 688.0 | 0.61 | 1,087.5 |
| 256K | 2,019.0 | 1,019.6 | 40.0 | 8.00 | 95.1 | 145.2 | 688.0 | 2.37 | 1,093.5 |
| 1M | 2,050.0 | 1,019.6 | 40.0 | 32.00 | 95.1 | 145.2 | 688.0 | 9.40 | 1,117.5 |
| 1M + 4K wrapped | 2,114.0 | 1,019.6 | 40.0 | 32.00 | 95.1 | 145.2 | 688.0 | 73.40 | 1,117.5 |

`r6_500m`, stored keys, executed float32, MiB per decoded token. Reread mode, same shape: 1K 2,196.6, 8K 3,576.1, 64K 3,578.4, 256K 3,586.2, 1M 3,617.2, 1M + 4K wrapped 3,681.2.

**Fill validation.** At 2,064 tokens, a real prefill and the synthetic fill gave identical totals in both modes (stored 1,727,830,521 B, reread 3,371,158,783 B) and identical counters. The only differing entries are the identities of the routed experts selected: 238 and 182 keys, every one a routed expert's weight.

**`enc256`** (d 256, the R8 timing geometry), executed float32 totals in MiB per decoded token:

| PLE | Keys | 1K | 8K | 64K | 256K | 1M | 1M + 4K wrapped |
|---|---|---:|---:|---:|---:|---:|---:|
| on | stored | 123.1 | 124.7 | 127.0 | 134.8 | 165.8 | 229.8 |
| on | reread | 645.2 | 1,241.9 | 1,244.2 | 1,252.0 | 1,283.0 | 1,347.0 |
| off | stored | 122.6 | 124.2 | 126.5 | 134.2 | 165.3 | 229.3 |
| off | reread | 644.7 | 1,241.4 | 1,243.7 | 1,251.4 | 1,282.5 | 1,346.5 |

**Constant in context:**
- All weight traffic.
- The ERSA mixer: byte-identical at every length, 261.0 MiB with its projections.
- The ERSA state and its temporaries.
- Evidence re-encoding: in reread mode, saturated once 64 candidate blocks exist (2,048 tokens).
- Embedding lookups, channels, readout.

**Grows until the 4,096-token window is full, then constant:**
- The window KV read: 10.2 → 40.0 MiB.
- The window-attention recompute: 175 → 688 MiB.

**Grows linearly with context:**
- Only the R5 coarse path.
- The descriptor scan reads 1 KiB per live 32-token block: 0.03 MiB at 1K, 32.0 MiB at 1M.
- Per-block bookkeeping over the score vector adds 8.3 MiB at 1M: recency, eligibility, top-K, the table of block ids.
- From 8K to 1M the step grows by 41 MiB. The test accounts for that growth exactly from these terms.

**Steps up once the ring has wrapped: +64 MiB per token.**
- Past capacity, the live blocks wrap around the ring, and `EvidenceLM._retrieve` joins two views with `torch.cat`
  (`evidence_model.py`, the "wrapped: two views joined" branch).
- Every decoded token therefore copies the whole 32 MiB live index, then scans the copy.
- An endless stream lives in this regime.

**ERSA reads no history.** Its operations touch only its fixed per-layer state (6 heads × 128 × 128 float32), its
convolution tail and the current token. Their bytes do not change from 1K to the wrapped ring. The chunked kernel
(`chunked_scan`) evaluates a one-token chunk as 1 × 1 systems over that state. The only historical scan in the
decode path is the R5 head's coarse descriptor scan: an exact, vectorized scoring of every live block, by design
(R5).

**The bfloat16 premise.** The draft figure's bfloat16 weights describe a precision this implementation cannot
decode on CPU: the chunked scan's triangular solve has no bfloat16 CPU kernel (tested). Had it run, the float32
descriptor index (key dtype `auto` is float32 off CUDA) would be converted to bfloat16 at every read
(`res_keys.to(dtype=...)`): another O(L) copy per token.

### 3.4 Measured decode cost by length (CPU, this VM)

The measurement is R8's unchanged admission run (`run_r8 admission`, 4 threads, untrained timing models,
interleaved), on `enc256` with the PLE on and off, at R9's lengths.

| Length | PLE | No R5 read p50 | Stored keys p50 / p95 | Reread p50 / p95 | Stored + cache p50 | Reread + cache p50 | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1K | on | 7.11 | 19.94 / 29.52 | 58.10 / 88.89 | 14.33 | 15.16 | 4,112 |
| 8K | on | 7.02 | 19.01 / 26.20 | 89.29 / 111.90 | 17.70 | 17.90 | 5,478 |
| 64K | on | 7.39 | 20.74 / 28.20 | 95.84 / 114.08 | 19.97 | 70.49 | 6,851 |
| 256K | on | 7.23 | 21.26 / 27.22 | 90.61 / 119.68 | 20.51 | 82.94 | 7,640 |
| 1M | on | 7.30 | 25.93 / 35.77 | 96.31 / 114.13 | 25.02 | 91.40 | 8,368 |
| 1K | off | 6.10 | 18.11 / 25.27 | 53.24 / 70.53 | 12.56 | 13.99 | 4,112 |
| 8K | off | 6.19 | 17.92 / 23.58 | 90.46 / 119.48 | 16.49 | 17.07 | 5,473 |
| 64K | off | 6.84 | 20.71 / 31.28 | 100.31 / 130.05 | 20.24 | 71.33 | 6,839 |
| 256K | off | 6.30 | 19.10 / 28.06 | 90.86 / 112.48 | 18.90 | 81.48 | 7,632 |
| 1M | off | 6.71 | 24.56 / 37.61 | 92.01 / 113.24 | 23.00 | 90.09 | 8,353 |

Decode step milliseconds, MEASURED: `enc256`, capacity 1M, 4 threads, 64 decode steps per configuration, untrained weights, this VM. Peak RSS is the measuring process's, which holds every configuration's model and state snapshot at once. R8's own records measured the same configurations at their lengths on another day, with lower absolute times (for example 58 ms reread and 21 ms stored at 1M, PLE on). Absolute times vary between VMs and are PERFORMANCE_ONLY.

### 3.5 Measured against projected bandwidth

The traffic a decoded token moves, divided by its measured decode time, gives an implied bandwidth. Compare it with
this VM's measured copy bandwidth: 46.4 GB/s read + write (1 GiB copies, 4 threads; 27.7 GB/s at 256 MiB).

| PLE | Keys | Length | Decode p50 (ms) | Compulsory MiB | Implied GB/s | Executed MiB | Implied GB/s |
|---|---|---|---:|---:|---:|---:|---:|
| on | stored | 1K | 19.94 | 23.2 | 1.22 | 123.1 | 6.5 |
| on | stored | 64K | 20.74 | 25.4 | 1.29 | 127.0 | 6.4 |
| on | stored | 1M | 25.93 | 55.4 | 2.24 | 165.8 | 6.7 |
| on | reread | 1K | 58.10 | 25.1 | 0.45 | 645.2 | 11.6 |
| on | reread | 64K | 95.84 | 29.6 | 0.32 | 1,244.2 | 13.6 |
| on | reread | 1M | 96.31 | 59.6 | 0.65 | 1,283.0 | 14.0 |
| off | stored | 1K | 18.11 | 22.7 | 1.31 | 122.6 | 7.1 |
| off | stored | 64K | 20.71 | 24.9 | 1.26 | 126.5 | 6.4 |
| off | stored | 1M | 24.56 | 54.9 | 2.34 | 165.3 | 7.1 |
| off | reread | 1K | 53.24 | 24.6 | 0.49 | 644.7 | 12.7 |
| off | reread | 64K | 100.31 | 29.1 | 0.30 | 1,243.7 | 13.0 |
| off | reread | 1M | 92.01 | 59.1 | 0.67 | 1,282.5 | 14.6 |

This VM's MEASURED copy bandwidth (read + write, 4 threads): 46.4 GB/s for 1 GiB copies and 27.7 GB/s for 256 MiB.

**At this size the decode is not bandwidth-bound.** The implied rates sit far below the copy bandwidth (28–46 GB/s):

| Bytes counted | Implied rate |
|---|---:|
| Compulsory | 0.3–2.3 GB/s |
| Executed | 6–15 GB/s | The `enc256` model fits in this host's L3, so its
decode time is dominated by per-op overheads and the R5 read's serial stages, not by DRAM. This is why the
projections for the DDR3 scenario use the scale configurations, whose weights do not fit any cache.

### 3.6 Roofline for the frozen scenarios (PROJECTED; GPU NOT_RUN)

Ceilings are upper bounds on tokens/s per stream, computed from the compulsory bytes and FLOPs (optimistic, §3.3.2).
The 60–100 tokens/s target is compared, never claimed. The executed decode moves more than the compulsory bytes: 1.8×
with stored keys and 3.2× with reread, in float32.

**`r6_500m` at 1M, central DDR3 assumptions** (bfloat16 weights, batch 1, nothing cache-resident, 15 GB/s at 80%,
100 GFLOP/s):

| Path | Ceiling at 1M | Wrapped | Bound by | GFLOP per token |
|---|---:|---:|---|---:|
| Stored keys | 19.6 tok/s | 17.7 tok/s | memory | — |
| Reread | 6.5 tok/s | — | compute | 15.3 |

**At the frozen ranges' most optimistic corner** (int4 weights with half of them cache-resident, 20 GB/s at 80%,
200 GFLOP/s):

| Configuration | Ceiling |
|---|---:|
| `r6_500m`, stored keys | 108.5 tok/s at 1M; 74.6 once the ring has wrapped (101.5 at batch 4) |
| `r6_500m`, reread | never above 13.1 tok/s |
| `r6_1b`, stored keys, central | 13.3 tok/s |
| `r6_1b`, stored keys, corner | 85.2 tok/s (96.8 at batch 4) |

**The target is therefore reachable only with stored keys, at the optimistic corner.** For a single stream with
bfloat16 weights it is 3–5× away, and it is out of reach on the reread path.

| Configuration | Context | Weights | Batch | Bytes per token (MiB) | Memory bound at 15 GB/s × 0.8 | At 20 GB/s × 0.8 | Compute bound at 100 GFLOP/s | Ceiling at 15 GB/s, 100 GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `r6_500m_reread` | 1M | bfloat16 | 1 | 586.9 | 19.5 | 26.0 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M | bfloat16 | 4 | 201.1 | 56.9 | 75.9 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M | int4_group32 | 1 | 217.1 | 52.7 | 70.3 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M | int4_group32 | 4 | 108.6 | 105.4 | 140.5 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M + 4K wrapped | bfloat16 | 1 | 650.9 | 17.6 | 23.4 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M + 4K wrapped | bfloat16 | 4 | 265.1 | 43.2 | 57.6 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M + 4K wrapped | int4_group32 | 1 | 281.1 | 40.7 | 54.3 | 6.5 | **6.5** |
| `r6_500m_reread` | 1M + 4K wrapped | int4_group32 | 4 | 172.6 | 66.3 | 88.4 | 6.5 | **6.5** |
| `r6_500m_stored` | 1M | bfloat16 | 1 | 582.7 | 19.6 | 26.2 | 57.4 | **19.6** |
| `r6_500m_stored` | 1M | bfloat16 | 4 | 196.9 | 58.1 | 77.5 | 57.4 | **57.4** |
| `r6_500m_stored` | 1M | int4_group32 | 1 | 212.9 | 53.8 | 71.7 | 57.4 | **53.8** |
| `r6_500m_stored` | 1M | int4_group32 | 4 | 104.4 | 109.6 | 146.1 | 57.4 | **57.4** |
| `r6_500m_stored` | 1M + 4K wrapped | bfloat16 | 1 | 646.7 | 17.7 | 23.6 | 57.4 | **17.7** |
| `r6_500m_stored` | 1M + 4K wrapped | bfloat16 | 4 | 260.9 | 43.9 | 58.5 | 57.4 | **43.9** |
| `r6_500m_stored` | 1M + 4K wrapped | int4_group32 | 1 | 276.9 | 41.3 | 55.1 | 57.4 | **41.3** |
| `r6_500m_stored` | 1M + 4K wrapped | int4_group32 | 4 | 168.4 | 68.0 | 90.6 | 57.4 | **57.4** |

Tokens per second per stream, PROJECTED from the compulsory bytes and FLOPs under the frozen DDR3 scenario (8 GiB RAM, nothing cache-resident). At batch 4, the weights are amortized over four streams. bfloat16 does not run on CPU in this implementation (§3.3.3), so these rows are projections of a configuration this code cannot execute.

GPU scenario (bfloat16, batch 1, 0.8 of peak bandwidth; CUDA NOT_RUN): `r6_500m_reread` 1M: memory bound 390 (300 GB/s) / 1170 (900 GB/s), compute bound 1,309 (20 TFLOP/s); `r6_500m_reread` 1M + 4K wrapped: memory bound 352 (300 GB/s) / 1055 (900 GB/s), compute bound 1,309 (20 TFLOP/s); `r6_500m_stored` 1M: memory bound 393 (300 GB/s) / 1178 (900 GB/s), compute bound 11,483 (20 TFLOP/s); `r6_500m_stored` 1M + 4K wrapped: memory bound 354 (300 GB/s) / 1062 (900 GB/s), compute bound 11,483 (20 TFLOP/s).

### 3.7 Retrieval quality

The retrieval-quality metric is frozen in `results/r9/preregistration.json` (key `p3`). No confirmatory retrieval
evaluation runs in R9, so its status is **NOT_RUN**. R8 established computation, not accuracy (L10).

### 3.8 The R5 reread against R8 stored keys, with the PLE on and off

| Path | Memory at 1M | Compulsory traffic per token | Compute per token |
|---|---|---|---|
| Stored keys | +256 MiB per stream at 1M (`enc256` and `r6_500m`: 32 tokens × 64 × 4 B per block) | +0.5 MiB of gathered candidate keys | re-encodes 4 blocks instead of 64: a 0.9 GFLOP R5 read in place of 14.8 for `r6_500m` |
| Reread | nothing extra | — | the full 64-block re-encoding |

The PLE adds its table to resident memory and 8 rows per token to the traffic. It changes neither path's R5 costs
(§3.4).

## 4. P4 — Gate A (projection and transport): HOLD

`GATE_A_PROTOCOL_R9.md` holds the audit and the frozen protocol.

**Audit findings:**
- Native K1's state has no export.
- The HCA line (`research/yang_mills_hca/`) carries relation rows of 8 floats, with SO(2)×SO(2) block rotations.
- Every frozen HCA protocol marks CGC `SHADOW_ONLY`.
- The HCA line lists frame transport and end-to-end gauge covariance among its unqualified seams.

No map φ from K1 or CGC state to relation rows exists, nor a re-import, nor a task that consumes transported K1
information. Building one would invent the seam the gate is meant to test.

**The frozen protocol:**
- preconditions P-1 to P-3;
- tests A1–A7, with thresholds fixed in advance. These cover collisions, equivariance, conditioning, capacity, and
  non-inferiority and superiority against untransported and null arms, plus a stranger-node arm.

**Disposition: HOLD.** Every test is NOT_RUN. Gate A reopens only when a seam exists that the owner admits.

## 5. P5 — workflow controls and negative controls

**Fault injection and negative controls** (tests in `tests/research/elpis_torch/test_r9.py` and `test_r9_records.py`;
all pass):

| Control | Expected | Observed |
|---|---|---|
| A plan that drifted from R7 (one CGC width changed) | refused before anything runs | refused (`ValueError`) |
| A run record with a tampered runner digest | `combine` refuses it; verdict NONQUAL | refused; NONQUAL |
| A missing primary run | NONQUAL | NONQUAL, run marked MISSING |
| Zeroed gates | exactly the FFN; restoring them restores the model bitwise | exact (float64) |
| Fabricated records for each verdict | the classifier gives each preregistered verdict | all three |
| The committed study record | equals a fresh `combine` of the committed runs | equal |
| The verdict, recomputed without PyTorch from the JSON records | equals the recorded verdict | equal |
| The synthetic history fill (P3) | the same decode-step bytes as a real prefill | equal totals; only routed-expert identities differ |
| ERSA reads history | its traffic grows with context | **does not**: byte-identical from 1K to the wrapped ring |

No negative control failed.
- Two records carry a `+dirty` git revision (§2.7). This is a provenance blemish, kept and explained, not a failed
  control.

**Workflow:**
- Work:
  - One branch from the verified `main` and one PR; no stacking.
  - The preregistration and runner were committed and pushed before the first confirmatory run.
- Analysis discipline:
  - No run was repeated, no seed added and no threshold changed.
  - Adverse results are kept: C ahead of A on every seed, CGC's latency cost, the corrected 614 MiB figure.
- Scope:
  - No scope expansion followed the null result: no larger model, no new architecture.
  - No model code changed; `src/`, `native/`, CI runner selection and every R0–R8 record are untouched.
  - No external model, dataset or private file was used. The corpus is this repository's text at `c05adff`.

## 6. Limitations

**CGC attribution:**
- One small model (d 64), 1,500 steps, 3 seeds, one corpus, CPU only.
- The null is a null at this scale and budget. It is not evidence that CGC fails at scale, and not evidence that it
  succeeds there.
- The FFN's lead (0.009 to 0.014 bits/byte) is below the preregistered margin, so "FFN preferred" is not claimed.

**Memory and bandwidth:**
- Projections assume the frozen platforms.
- Executed bytes are logical operand bytes; DRAM traffic was not measured, because this VM exposes no hardware
  counters.
- bfloat16 decode is a projection: this implementation does not decode in bfloat16 on CPU.
- Measured latencies are of untrained timing models on one shared VM.

**Not done:**
- No GPU (CUDA NOT_RUN).
- No confirmatory retrieval evaluation (the metric is frozen, NOT_RUN).
- Gate A is HOLD.

## 7. Historical boundaries and changed files

**Unchanged:**
- `src/`, `native/`, `.github/workflows/ci.yml`.
- Every R0–R8 record and document, and the model code in `research/elpis_torch/*.py`.
- Imprecise historical wording is corrected in `CLAIM_AUDIT_R9.md`, with links. The R9 preregistration's own phrase
  "R5 … did not solve autonomous recall" is frozen and corrected in L7.

**Added:**
- `docs/research/elpis_torch/{CLOSEOUT_R8, PREREGISTRATION_R9, CLAIM_AUDIT_R9, DESIGN_R9, EVIDENCE_R9,
  GATE_A_PROTOCOL_R9}.md`.
- `research/elpis_torch/experiments/run_r9.py` and `run_r9_p3.py`.
- `research/elpis_torch/results/r9/*.json`, with `MANIFEST.json` giving every record's SHA-256.
- `tests/research/elpis_torch/test_r9.py` and `test_r9_records.py`.

**Edited:**
- `research/elpis_torch/README.md` (claim corrections and R9 pointers).
- `tests/research/elpis_torch/test_isolation.py` (the two R9 runners' single write sites).
