# Elpis Torch R7 — parametric lookup embeddings (PLE): evidence

Research only; CPU (4 cores, no CUDA); small models trained here, a timing model untrained. Design and every
divergence from the Engram reference: `DESIGN_R7.md`. Records: `research/elpis_torch/results/r7/`.

Status words: PASS / PARTIAL / FAIL / HOLD / NOT_RUN / PROJECTED / NONQUAL, and MEASURED (this CPU) or EXACT.
Wall-clock numbers are PERFORMANCE_ONLY.

## 0. Base

R7 starts from the R6 head `c05adffafbd5d8a94ee40c285031d1cd219f1c4c` on `claude/r6-expert-backbone` (PR
abraxis717/Elpis_DEV#76). Its exact-head CI is green on both runs: push run 44306588067 and pull-request run
44306596921, 16 of 16 jobs. The research job passed twice: 443 passed and 2 skipped on the runner's own kernels,
then the same with the AVX2 paths forced.

The R6 closeout is `EVIDENCE_R6.md` section 10. Two CI failures were diagnosed as numerical ordering, and the tests'
contracts were repaired; the model was unchanged. A record-writer rounding defect was corrected. R7's branch,
`claude/r7-ple-admission`, descends from that head.

## 1. P1 — sources and design

The sources are `deepseek-ai/Engram` at `fb7f84a` (demo and paper), arXiv:2601.07372 v1 and v2, and the vLLM
Engram documentation. `DESIGN_R7.md` sorts each step into integer addressing, row gathering or neural computation
and lists every divergence. No Engram code, weights or dependencies are used. **PASS.**

## 2. P2 — the mechanism (tests, all PASS: `test_r7.py`, 29 tests)

| Requirement | Evidence |
|---|---|
| deterministic, seeded addressing | equals a Python-integer reference over boundaries and the largest token id; the addressing digest is pinned (golden value), changes with the seed and the vocabulary; no `hash()` |
| distinct head ranges, valid addresses | distinct primes per (order, head), cumulative offsets, every address inside its own range |
| overflow | every multiplier odd and below `2^63 / (V + 1)`; an impossible vocabulary is refused |
| causality | changing tokens from position 50 on leaves earlier addresses identical |
| chunk boundaries | addresses through 1-, 2- and 7-token calls equal one call, boundaries across chunk edges included |
| documents, BOS, padding | a BOS restarts n-grams (the second document's addresses equal those of that document alone); trailing padding changes no address |
| exact tokens | "Ap" and "ap" address differently (no normalization) |
| invalid inputs | bad configurations, out-of-range ids, history shape or dtype, and a boundary outside the vocabulary are all refused |
| collisions | two bigrams sharing a row in one head are told apart by the other head |
| content-free control | random addresses are redrawn per call, stay in range and are reproducible from the seed |
| gradients | the table's sparse gradient covers exactly the addressed rows, all non-zero; key, value and both norms receive gradient |
| lazy updates | inside `train_lm`, every row a step changes was addressed by that step, and more than 90% of the addressed rows change |
| initialization, checkpoints | the same seed gives identical weights. Schema `elpis-torch-r7-lm/1` carries the PLE configuration and addressing digest. A stale digest, other addressing with the same shapes, or another tokenizer is refused. An R6 checkpoint still loads |
| zero contribution | `PLE.enabled = False`, and a zero value projection, each give logits bit for bit equal to the model without PLE (ERSA with experts, and the R5 evidence model) |
| gate | it reads the hidden state and lies strictly inside (0, 1) |
| one fusion point, one addressing per call | rows enter the configured block's input; addressing runs once per forward call and not at all when disabled |
| state | history `batch x (max_order - 1)` int64, constant over the stream, reset to the sentinel, independent per session, EXACT in `nbytes` and in both interfaces' fixed bytes |
| with experts and R5 | causal: bit for bit with the reference dispatch, within 1e-12 with the grouped one. Chunk 29 against one call agrees within 1e-9 |
| generation | free-running generation is independent of how the prompt is split and agrees with teacher forcing |

## 3. P3 — read-only stores (tests, all PASS)

- **TensorStore** in float32 reproduces the trainable table's logits bit for bit. In bfloat16 its rows are the
  table's values cast to bfloat16, and its logits differ by less than 1% of their largest magnitude.
- **FileStore**, float32 and bfloat16: rows equal the reference store's bit for bit, and so do the model's logits
  with the file attached.
  - It reads exactly the distinct rows each call needs: in the test, 5 rows in 3 positioned reads, never the table.
  - It counts requested rows, distinct rows, reads and logical bytes.
- **Refusals:** a wrong magic, a truncated file, an extended file, other addressing, another shape and an
  out-of-range row are refused. A flipped data byte passes opening but fails the explicit offline `verify()`.
- **No training, no writes.** A store refuses training. In a separate interpreter with an audit hook, file-backed
  inference (open, prefill, decode, reset, a fault) produced no write-class event. A deliberate write is detected.
- **Isolation:** `ple.py` contains no file access. In `ple_store.py`, the isolation test allows exactly one call,
  `open(self.path, os.O_RDONLY)`, and no write, map, truncate or rename.
- **Cold device reads: NOT_RUN.** The file is read back from the page cache in this container.

**PASS.**

## 4. P4 — is it useful?

**Corpus.** This repository read from git at `c05adff`, so it never changes with the working tree and holds none of
R7's own files:

- Code: `src/` and `research/elpis_torch/`, the `*.py` files, without `upstream/`.
- Prose: `docs/`, the `*.md` files.
- A file is held out when SHA-256 of its path is 0 mod 10.
- Train: 275 files, 2,915,823 bytes. Held-out code: 15 files, 151,683 bytes. Held-out prose: 5 files, 70,745 bytes.
  Corpus digest `ab516f5f…`.

**Training.** The model: d 64, 2 ERSA blocks, 8 routed FFN + CGC experts top-2 in both blocks (R6's controls).

- 1,500 steps of 16 x 128 bytes, about one epoch.
- AdamW at 3e-3 with warmup and cosine decay.
- Seeds 0, 1, 2: the same data order and the same initial weights for every shared parameter, so comparisons are
  paired.
- Evaluation: 48 held-out batches per domain.

**The PLE**, sized before any training from the training corpus alone:

- Orders 2 and 3, 4 heads each, primes from 65,537 rows, 8-wide rows, fused before block 1, boundary BOS.
- Tables: lazy Adam at 5 x the learning rate, no weight decay.
- The corpus has 44,079 distinct trigrams, so a trigram shares a row in all four heads 5.7% of the time.

| variant (3 seeds) | held-out code, bits/byte | held-out prose, bits/byte | total parameters | active per token |
|---|---:|---:|---:|---:|
| R6 (no PLE) | 1.827 ± 0.024 | 2.284 ± 0.018 | 431,656 | 147,484 |
| R6 + PLE | **1.732 ± 0.012** | **2.203 ± 0.010** | 4,635,576 | 155,868 |
| R6 + PLE, contribution removed at evaluation | 4.460 ± 0.068 | 4.278 ± 0.060 | (same model) | |
| R6 + PLE with random rows (CONTROL) | 1.850 ± 0.026 | 2.319 ± 0.022 | 4,635,576 | 155,868 |
| the same, contribution removed | 1.848 ± 0.026 | 2.317 ± 0.022 | (same model) | |
| R6 with 97 experts per block (CAPACITY CONTROL) | 1.780 ± 0.007 | 2.236 ± 0.007 | 4,658,266 | 158,876 |

Paired differences, PLE minus control, seeds 0 / 1 / 2:

| against | code | prose |
|---|---|---|
| R6 | -0.136, -0.078, -0.070 (mean -0.094) | -0.106, -0.078, -0.060 (mean -0.081) |
| random rows | -0.121, -0.140, -0.092 (mean -0.118) | -0.108, -0.148, -0.091 (mean -0.116) |
| the same total parameters as experts | -0.056, -0.038, -0.050 (mean -0.048) | -0.045, -0.034, -0.020 (mean -0.033) |

What these show:

- **The PLE model wins every seed against every control, on both held-out domains.** It costs 5.7% more active
  parameters per token than R6, and 1.9% fewer than the capacity control.
- **The gain needs content addressing.** Random rows trained the same way, with the same optimizer, learning rate and
  table, end slightly worse than R6, and removing them changes nothing. Their gate sits at 0.50 with no saturation.
- **The trained model relies on its rows.** Removing them raises the loss by 2.73 bits/byte on code and 2.07 on
  prose, to 4.46 and 4.28. This is the training-inference inconsistency the Engram paper also reports. It means the PLE carries information the backbone
  stopped computing, not that removal is a safe ablation.
- **Training cost.** Training time per run, on 2 threads with two processes in parallel: PLE 194-204 s, R6 175-177 s,
  the 97-expert control 314-323 s (the R6 and 97-expert seed-0 runs were slowed by a concurrent test run).

**Use of the table.**
- Gates on held-out code, seeds 0-2: mean 0.65-0.69, median 0.71-0.80, 5th percentile 0.13-0.14, 95th 0.94-0.96.
  3.7-9.7% are above 0.95 and 0.9-1.4% below 0.05: context-dependent, not saturated.
- Rows used: 28.8% of the 524,450 rows during training, 8.0% on held-out code.
- Gradient norms on one held-out batch:

| parameters | gradient norm |
|---|---|
| table | 0.022-0.023 (6,354 rows) |
| value | 0.21-0.23 |
| key | 0.045-0.068 |
| PLE norms | 0.004-0.009 |
| embeddings | 0.26-0.28 |
| ERSA | 0.39-0.42 |
| experts | 0.23-0.25 |

**Collisions** on the training corpus (`collisions.json`):

| order | distinct n-grams | rows used per head | share a row in one head | share a row in every head |
|---|---:|---|---|---:|
| 2 | 4,918 | about 4,720 | 7.3-8.3% | 0% |
| 3 | 44,079 | about 32,100 | 48.8-49.0% | 5.7% |

**Positive control** (`local.json`): 512 fixed lowercase words in uniform sequences, 400 steps, seeds 0 and 1.

| variant | bits/byte | inside words | contribution removed |
|---|---:|---:|---:|
| R6 | 3.546 / 3.525 | 3.318 / 3.295 | |
| PLE | 1.665 / 1.690 | 1.066 / 1.097 | 5.95 / 5.74 |
| random rows | 3.960 / 4.106 | 3.815 / 3.991 | 3.95 / 4.10 |

Where local patterns decide the next byte, the PLE more than halves the loss. Content-free rows add noise at this
short horizon.

**Revision-heavy control** (`revision.json`): R5 record streams, revisions 1-3, 2,048 tokens, 400 steps, seeds 0
and 1. There are 367 answer probes, 270 of them on revised keys.

| variant | all positions | answers | revised answers |
|---|---|---|---|
| R6 | 3.742 / 3.718 | 3.148 / 2.986 | 3.161 / 2.968 |
| PLE | 3.722 / 3.717 | 2.914 / 3.076 | 2.918 / 3.056 |

(nats per token)

- Removing the PLE changes answer loss by at most 0.006 nats: the PLE is not used for answers.
- The paired answer differences have opposite signs (-0.23, +0.09).
- Neither model learns the retrieval in this budget. **NONQUAL** for revision-dependent capability. No harm or gain
  is attributable to the PLE.

**P4: PASS at this scale.** The PLE adds useful predictive capacity over R6, over content-free rows and over the
same parameters spent on experts, with a small increase in active parameters. It does not establish better
reasoning, code generation or long-range retrieval (see section 7).

## 5. P5 — what it costs

**Measured cost** (`latency.json`): CPU, 4 threads, paired and interleaved. One untrained model (d 256, 4 blocks,
8 experts top-2 in blocks 1 and 3) serves three identical streams fed the same tokens in a rotating order: PLE off,
resident rows and file-backed rows. The PLE has 2 orders x 4 heads x 32,771 rows x 32 and sits before block 1.
Samples: 400 decode steps and 20 prefills of 512 tokens per R5 setting.

| R5 | PLE | decode p50 (p95), ms per token | difference vs PLE off, p50 (p95) | 512-token prefill p50 (p95), ms | difference p50 (p95) |
|---|---|---:|---:|---:|---:|
| off | off | 4.41 (6.21) | | 42.8 (52.2) | |
| off | resident | | +0.49 (+1.36) | | +1.3 (+4.3) |
| off | file-backed | | +0.59 (+2.03) | | +5.9 (+17.9) |
| on | off | 17.21 (24.06) | | 301.5 (424.3) | |
| on | resident | | +0.84 (+7.91) | | -4.3 (+61.9), within noise |
| on | file-backed | | +0.80 (+5.54) | | +5.6 (+74.3) |

The PLE's parts alone, resident, p50:

| part | 1 token | 512 tokens |
|---|---:|---:|
| addressing | 0.19 ms | 0.24 ms |
| row gather | 0.004 ms | 0.012 ms |
| fusion including the gather | 0.10 ms | 0.71 ms |

- **Bytes.** 8 rows of 128 bytes, 1,024 logical bytes per token. During decode the file store made 3,200 positioned
  reads of 409,600 bytes, one per row.
- **The PLE's decode cost** at this small width is about 0.5-0.8 ms (11% without R5, 5% with it). Its measured
  parts are addressing, 0.19 ms (about 20 small tensor operations per call, a cost that hardly depends on length),
  and fusion, 0.10 ms. The row gather (0.004 ms) and memory traffic barely register.
- **R5 dominates:** +12.8 ms per decoded token (+290%) and +259 ms per 512-token prefill when it reads at every
  position.
- **File-backed rows** cost one positioned read per row through a Python loop: +5.9 ms per 512-token prefill.
  Cold-device reads: NOT_RUN.

**Projected cost for the R6 configurations** (`accounting.json`, formulas, nothing allocated): a PLE of 2 orders x 8
heads x 64 dims before block 1.

| rows per head | table parameters | bfloat16 bytes |
|---:|---:|---:|
| 65,536 | 67.2M | 0.13 GB |
| 1,048,576 | 1.07B | 2.15 GB |
| 16,777,216 | 17.2B | 34.4 GB |

- **Per token,** whatever the table size: 16 rows, 2,048 bytes in bfloat16 and about 80 integer addressing
  operations.
- **Fusion:** 1.57M (500M) or 2.10M (1B) multiply-adds per token. That is 0.67% / 0.57% of the active weights'
  236M / 366M, and about 0.02% of one R5 evidence read (7.2-7.4G for re-encoding candidates at 1M).
- These are PROJECTED. A 1B-row table was not allocated.

**Expert dispatch.** The analysis in `DESIGN_R7.md` section 4 stands: the grouped path synchronizes the host once per
routed block per call (`bincount(...).tolist()`) and loops over every expert in Python. The 97-expert control's
slower training here (about 1.8 x R6 on 2 threads) shows that loop on a CPU. No CUDA claim is made.

**P5: PASS for the measurements, which are CPU-only.** The PLE's incremental cost is small and measured. The
60-100 tokens/s GPU target is a future objective, not assessed: CUDA NOT_RUN.

## 6. Regression and isolation

- **Research suite on the R7 head:** 470 passed, 5 skipped, 0 failed. The skips are 2 CUDA tests and 3
  native-kernel tests, whose library is not built in this container.
- **Isolation gate:**
  - Writers are registered: `run_r7.py` has 2, the store export and `--out`.
  - `ple.py` is in the writer-free runtime list.
  - The read-only check on `ple_store.py` passes.
  - The R7 admission is recorded, while the names `engram`, `memory` and `retrieval` stay forbidden.
- **CI** runs `run_r7 smoke`: 79 s on two threads here, nothing kept.
- Nothing under `src/` or `native/` changes. R0-R6 behaviour without PLE is bit for bit unchanged, and R6
  checkpoints load.

## 7. Limitations

- **Scale.** Small models (d 64, 2 blocks), one epoch of 2.9 MB of bytes, three seeds, one PLE configuration chosen
  a priori. The held-out prose is 5 files. A larger model may need its PLE differently, and the Engram paper's
  allocation curve suggests an optimum, not a monotone gain. The gap to the capacity control could change with scale.
- **Perplexity only.** Held-out bits per byte, not code generation, reasoning or long-range retrieval. The revision
  control learned nothing in its budget.
- **The PLE makes the trained model depend on it:** removing it is catastrophic, not neutral.
- **CPU only.** Timing is single-machine, untrained and PERFORMANCE_ONLY. No GPU, no cold storage reads, no
  prefetch.
- **What R7 leaves out:** tokenizer compression, the short convolution and multi-branch keys. The study does not
  tell whether they would add or subtract here.

## 8. Reproduce

```bash
python -B -m pytest tests/research/elpis_torch
python -B -m research.elpis_torch.experiments.run_r7 study --variants r6,ple,ple_random --threads 2 --out a.json
python -B -m research.elpis_torch.experiments.run_r7 study --variants more_experts --threads 2 --out b.json
python -B -m research.elpis_torch.experiments.run_r7 combine --inputs a.json,b.json --out study.json
python -B -m research.elpis_torch.experiments.run_r7 local --out local.json
python -B -m research.elpis_torch.experiments.run_r7 revision --out revision.json
python -B -m research.elpis_torch.experiments.run_r7 collisions --out collisions.json
python -B -m research.elpis_torch.experiments.run_r7 latency --decode-steps 400 --prefill-runs 20 --out latency.json
python -B -m research.elpis_torch.experiments.run_r7 accounting --out accounting.json
```
