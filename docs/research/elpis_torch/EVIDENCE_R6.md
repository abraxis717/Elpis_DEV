# Elpis Torch R6 — expertized generative backbone and evidence inference integration: evidence

Research only; CPU (4 cores, no CUDA); tiny trained models; the 500M / 1B configurations are instantiated and run
untrained. Design: `DESIGN_R6.md`. Records: `research/elpis_torch/results/r6/`. Base: `main` at
`bdbc690f9edab950cb260a78309700784c1bd9b5`; branch `claude/r6-expert-backbone`.

Status words: PASS / FAIL / PARTIAL, MEASURED (this CPU), EXACT (tensor sizes), PROJECTED (formula), NOT_RUN,
NOT_REPRODUCED, HISTORICAL.

## 1. P0 — R5 reconciliation

* Post-merge commit `fe82cc8` audited against `main`: its five result records are strict supersets of the
  versions in `main` (every shared row byte-identical, same checkpoint digests) and were carried forward. Its
  loader edit was NOT carried: it extended rank embeddings whenever a model was built wider, without permission,
  shrink protection or a record of the override.
* Replacement: `evidence_model.evaluation_state_dict` allows only listed evaluation overrides, refuses fewer
  evidence slots than trained, extends the readout's rank table by repeating the oldest trained rank (exactly the
  trained model's clamping rule) and keeps learned rows bit for bit; `run_r5.load(allow_slot_extension=...)`
  verifies the parameter digest and records `trained_evidence` and `evaluation_override`. Tests: extension
  rules, validation, functional identity when no more evidence exists, digest refusal.
* Reproducibility: checkpoint schema `elpis-torch-r5-checkpoint/2` (model, ERSA and evidence configurations with
  digests, vocabulary layout digest, SHA-256 of the task generators' source); a two-run training smoke gives
  identical digests and a verified round trip (test). `EVIDENCE_R5.md` archives R5 as HISTORICAL: the trained R5
  checkpoints are not repository artifacts, nothing was re-trained, and every incomplete cell is listed.

**R5_MERGE_RECONCILIATION: PASS.**

## 2. P1 — R5 as a first-class context-session model

Audit: `ContextSession` decided "has history" with `isinstance(model, ErsaHistoryLM)`. For `EvidenceLM` this was
false, so a session silently ignored origin and placement, reported no counters, and `projected_bytes()` omitted
the R5 ring and index. Now every family supplies `context_interface()` (core default, R3 `HistoryInterface`,
R5 `EvidenceInterface`); the session never inspects types.

Tests (all PASS): direct execution equals the session through chunked prefill (chunk 37), two prefill calls,
probe-only logits and 80 single-token steps, across the ring's wraparound (600 tokens, capacity 256), to 1e-9 in
float64; reset in place and independent sessions; EXACT accounting (`projected state == nbytes`, ring tiers
included); options that do not apply are refused; the reread cache equals the uncached path to 1e-10, stays
within its bound, never holds a block below the live window, empties on reset, and a model change refuses the
session. Address identity (session, epoch, block, slot, generation, completion, eviction, rollover) remains
covered by the R5 tests.

Reread cache, MEASURED on an UNTRAINED R5 model (mechanics, not retrieval quality; 70 queries):

| tokens | cache | blocks re-encoded | hits / misses | reread ms per query |
|---:|---:|---:|---|---:|
| 65,536 | none | 2,777 | — | 13.1 |
| 65,536 | 16 MiB | 989 | 1,788 / 989 | 3.1 |
| 262,144 | none | 4,038 | — | 26.2 |
| 262,144 | 16 MiB | 1,898 | 2,140 / 1,898 | 11.2 |

A 64 MiB cache gave the same hits as 16 MiB. **R5_UNIFIED_INFERENCE: PASS.**

## 3. P2 / P3 — sparse FFN + CGC experts

Mechanism (`sparse_channel.py`): per-token top-K router (float32 or wider; ties to the lower expert), experts
owning SwiGLU + local CGC + gate, optional shared expert, grouped dispatch of selected experts only, no dropping,
per-call statistics and optional balance / z losses. Tests (all PASS): grouped equals the masked reference in
value (1e-12) and gradient (1e-10) for 4-16 experts, K = 1-3, non-contiguous inputs and batch shapes; N x K
assignments exactly (no token dropped); no aliased parameters and expert-local CGC effects confined to that
expert's tokens; gradients reach the router, active experts' FFN and CGC, the shared expert, ERSA, embeddings and
the tied head; a zero gate is exactly the FFN and its gate learns first, then the CGC; finite under a saturated
router and bfloat16 autocast; `sparse=None` is the old model; upcycling reproduces dense models with CGC, FFN and
FFN + CGC channels (clones to 1e-12 in float64; shared copy and routing-disabled control bitwise).

Grouped dispatch, MEASURED (forward + backward, d 256, top-2, float32, two threads). CORRECTED at the closeout
(section 10): the first record said "identical outputs, max |difference| 0.0" because its writer rounded every float
to six decimal places; the two paths agree to float32 rounding, not bit for bit. Re-measured:

| experts | tokens | expert evaluations (grouped / reference) | grouped s | reference s | speedup | max \|difference\| |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 512 | 1,024 / 4,096 | 0.076 | 0.160 | 2.1 | 6.0e-8 |
| 8 | 4,096 | 8,192 / 32,768 | 0.366 | 0.932 | 2.5 | 3.0e-8 |
| 16 | 4,096 | 8,192 / 65,536 | 0.372 | 2.066 | 5.6 | 6.0e-8 |
| 64 | 4,096 | 8,192 / 262,144 | 0.576 | 8.065 | 14.0 | 1.2e-7 |

The first run measured speedups of 2.4, 3.0, 6.5 and 19.6; wall-clock on this shared CPU varies between runs.

Training controls, MEASURED (d 64, 2 blocks, both blocks sparse for routed variants; 400 steps of 16 x 128 bytes
of this repository's source code; held-out: 4 other source files, 25,096 bytes; three seeds):

| variant | held-out bits per byte | total parameters | active per token |
|---|---|---:|---:|
| dense FFN + CGC | 2.628 +- 0.017 | 158,362 | 158,362 |
| routed FFN only (8 experts, top-2) | 2.593 +- 0.027 | 316,952 | 118,808 |
| routed FFN + CGC | 2.581 +- 0.025 | 431,656 | 147,484 |
| routed FFN + CGC + shared expert | 2.580 +- 0.028 | 330,410 | 145,310 |
| routed FFN + CGC, FROZEN random router | 2.562 +- 0.023 | 431,656 | 147,484 |

Every run completed (no divergence); no expert was inactive; learned routers concentrated (top-1 probability
0.41-0.52, load imbalance max/mean up to 3.4) where the frozen router stayed near uniform (0.16-0.17, 1.4-1.5).
**The frozen random router is at least as good as the learned one: at this scale and budget there is no evidence
that routing learns useful distinctions; the measured gain of the sparse variants over dense is attributable to
capacity.** Expert specialization is not claimed. Differences between routed variants are within one standard
deviation.

**SPARSE_FFN_CGC_EXPERTS: PASS. EXPERT_GRADIENT_AND_EQUIVALENCE: PASS. Router stability: PASS (mechanics);
learned-routing usefulness: NOT DEMONSTRATED.**

## 4. P4 — experts with the R5 evidence path

Tests (all PASS): with experts and R5 together the model is causal (future tokens change no earlier logit) and
chunk-invariant (chunk 29 against one call, 1e-9); R5 off equals the routed core; labels change neither logits
nor routes; expert selection adds no persistent state (state bytes equal to the dense evidence model's); a
router change refuses the live session; injected (top-4/top-8) and exposed evidence counts configure
independently and every rank embedding of a top-8 model trains; R5 + window attention + experts runs and is
chunk-invariant (regression for a defect found here: the core's window-slot map and the R5 ring shared the state
attribute `ring`, so any R5 model with window attention crashed; the core's map is now `window_slots`). The
narrower evidence encoder (below) keeps chunk parity.

Top-4 versus top-8 injection with the R5-trained models: NOT_REPRODUCED (checkpoints unavailable); the
mechanism is tested independently. **ERSA_EXPERT_R5_INTEGRATION: PASS.**

## 5. P5 — autoregressive training

`lm_training.py` (tests all PASS): rows never cross documents and padding targets are ignored; the loss is
normalized over valid targets and is the same for chunks of 40, 16 and 7 (1e-9); truncated backpropagation
detaches between chunks (gradients differ) and equals the full graph for one chunk; routers' auxiliary losses
enter with their weights; a non-finite loss is refused with nothing applied; identical seeds give identical
digests; a byte-level code fixture is learned (held-out loss down by more than 1.5 nats in 40 steps; a longer
run: 5.53 -> 2.85 nats/token in 60 steps of 16 x 128 bytes, 8 experts top-2); teacher-forced predictions equal
free-running generation; generation after several prefill calls equals one prefill, with experts and with R5;
checkpoints (`elpis-torch-r6-lm/1`) round-trip with digest and tokenizer verification; the byte tokenizer
satisfies both tokenizer protocols. This is trainability, not language competence.

With R5 enabled and no labels the coarse index and the hard selection receive no direct next-token gradient;
learning addressing without labels is NOT claimed. **AUTOREGRESSIVE_EXPERT_TRAINING: PASS (small scale).**

## 6. P6 — 500M and 1B sparse configurations

Instantiated on `meta` (EXACT counts); vocabulary 65,536; R5 in the last block with a 256-wide encoder and
32-query tiles.

| | `r6_500m` | `r6_1b` |
|---|---|---|
| geometry | d 768, 20 blocks, 8 experts top-2 (odd blocks) | d 1,024, 18 blocks, 16 experts top-2 + shared (odd blocks) |
| total = resident parameters | 507.7M | 994.7M |
| active parameters per token | 286.5M | 433.1M |
| of which: routed experts / shared / routers | 294.9M / — / 0.1M | 641.9M / 40.1M / 0.1M |
| dense channels / ERSA / attention / embeddings / evidence path | 85.5 / 47.5 / 7.9 / 50.3 / 21.4M | 136.8 / 76.0 / 10.5 / 67.1 / 22.1M |
| weights (resident) | 1.89 GiB fp32, 0.95 GiB bf16 | 3.71 GiB fp32, 1.85 GiB bf16 |
| per-token multiply-adds (active weights + head) | 236M + 50M | 366M + 67M |
| per evidence query at 1M: coarse scan / candidate re-encoding | 8.4M / 7.24G | 8.4M / 7.41G |

Sparse activation reduces compute per token, not resident weight memory. Re-encoding the 64 candidates is the
dominant per-query cost; with an encoder as wide as the model it was 60-107 G multiply-adds per query, which is
why the configurations use a 256-wide encoder (a down-projection added in R6, as in R3); the reread cache
removes about half of it in the measurement above.

Context memory per stream (state EXACT with a bfloat16 index; scratch PROJECTED in float32, shown for `r6_500m`;
`r6_1b`'s scratch is within 0.02 GiB of it):

| tokens | `r6_500m` state | of which R5 ring + index | `r6_1b` state | decode scratch | 512-token prefill reading at every position: scratch | coarse scan per query |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 47.9 MiB | 0.16 MiB | 41.4 MiB | 20.4 MiB | 0.14 GiB | 65.5K |
| 32,768 | 48.3 MiB | 0.63 MiB | 41.9 MiB | 21.2 MiB | 0.44 GiB | 262K |
| 131,072 | 50.2 MiB | 2.53 MiB | 43.8 MiB | 24.3 MiB | 0.84 GiB | 1.05M |
| 262,144 | 52.8 MiB | 5.06 MiB | 46.3 MiB | 28.4 MiB | 0.85 GiB | 2.10M |
| 1,048,576 | 67.9 MiB | 20.25 MiB | 61.5 MiB | 53.2 MiB | 0.90 GiB | 8.39M |

The rest of the state is the ERSA matrices, convolution tails and the 4,096-token window caches; no per-token KV
cache and no per-expert index. Per stream, state plus scratch stays within the 1 GiB target at every reported length
in both read modes (a test pins it): about 0.12 GiB for a decode step at 1M, and at most 0.97 GiB (`r6_1b` at 1M)
for a 512-token prefill that reads at every position. The second needs 32-query tiles: with R5's default 128, one
tile can re-encode 8,192 distinct candidate blocks, projecting 2.82 GiB at 128K and 5.59 GiB at 262K, over the
2 GiB ceiling (the projection for tiles of 128, 64, 32 and 16 is in `scale.json`). The tile only partitions
independent queries and changes no logit (tested); generation reads only where logits are requested.

Bounded real forwards (untrained weights, capacity 16,384, two CPU threads, MEASURED): `r6_500m` finite,
128-token prefill 0.65 s, 0.091 s per decoded token, peak RSS 3.1 GB; `r6_1b` finite, 1.06 s, 0.112 s, 5.3 GB (the run
recorded at the previous head: 0.89 s / 0.112 s and 1.31 s / 0.128 s; wall-clock is PERFORMANCE_ONLY). Training at these sizes is NOT_RUN; PROJECTED
from the EXACT counts, float32 weights, gradients and two AdamW moments take 16 bytes per parameter, 7.6 GiB
(`r6_500m`) and 14.8 GiB (`r6_1b`), before activations. **SPARSE_SCALING_ACCOUNTING: PASS.** No large model was
trained; no capability is implied.

## 7. Regression

* Elpis Torch research suite on the final head: 440 passed, 5 skipped, 0 failed (`python -B -m pytest
  tests/research/elpis_torch`, 236 s; the same with MKL's AVX2 kernels, the CI-like path of section 10); the skips are two CUDA tests (NOT_RUN) and three native-kernel parity tests
  whose library is not built in this container (CI's research job builds it and requires it). R6 adds 43 tests
  (`test_r6.py`), R5 has 18. CI's research job now also runs the R5 and R6 experiment smoke commands (4.9 s and
  12.3 s on two threads here; nothing is kept).
* Repository FAST lane (`PYTHONPATH=src pytest`): 1,200 passed, 712 skipped, 5 failed and 10 errors, all in
  native-backed canonical tests (K1 boundary, HACF, structural evidence, context substrate); the same 5 failures
  and 10 errors occur on `main` (`bdbc690`) in this container (native libraries are not built here). R6 changes
  nothing under `src/` or `native/`.
* Zero runtime storage writes: the isolation gate registers the explicit offline writers (`lm_training.py` 1,
  `run_r6.py` 1) and keeps `sparse_channel.py` and every runtime module writer-free; the audit-hook test of the
  evidence stream passes. CUDA: NOT_RUN (no GPU).

## 8. Owner decisions recorded here

* Token-routed channel experts and their router left the isolation test's deferred list by the R6 directive;
  agentic selectors, policies, planners, halting and controllers stay forbidden.
* R6 changed R5's evidence encoder to allow a narrower width (default unchanged: equal to `d_model`).

## 9. Reproduce

```bash
python -B -m pytest tests/research/elpis_torch
python -B -m research.elpis_torch.experiments.run_r6 controls --threads 2 --out controls.json
python -B -m research.elpis_torch.experiments.run_r6 dispatch --out dispatch.json
python -B -m research.elpis_torch.experiments.run_r6 scale --out scale.json
python -B -m research.elpis_torch.experiments.run_r6 cache --out cache.json
python -B -m research.elpis_torch.experiments.run_r6 smoke
```

## 10. Closeout: the exact-head CI failure at `eba8646`

CI run 44207120535 (`eba8646`): 15 of 16 jobs passed; `Elpis Torch research (PyTorch CPU)` reported 1 failed, 440
passed, 2 skipped. `test_experts_with_r5_are_causal_and_chunk_invariant` asserted that the first 301 positions'
logits are bit-for-bit equal for two streams that differ only from position 301 on. CI used the same torch
(2.15.0+cpu) on Python 3.12.16. The test had passed here (AVX-512, 4 threads).

Reproduced here by forcing MKL's AVX2 kernels (`MKL_ENABLE_INSTRUCTIONS=AVX2`). Diagnosed with `run_r6 causality`
(records `results/r6/causality_native.json` and `causality_avx2.json`; float64; MEASURED):

| check | native (AVX-512, 4 threads) | AVX2 kernels, 4 threads |
|---|---|---|
| grouped dispatch: past logits bit for bit | yes | no: 77,258 of 388,892 differ, at most 2.2e-16 (4.1e-16 of the largest logit), from position 6 |
| past routes (selected experts) | identical | identical; block 1's routing weights within 1.1e-16 |
| first module output that differs | none | block 0's routed channel (6.9e-18); block 0's ERSA output is identical |
| reference dispatch (fixed shapes) | bit for bit | bit for bit |
| grouped routing, but every expert's GEMMs on all rows | bit for bit | bit for bit |
| R5 off | bit for bit | differs the same way (R5 is not involved) |
| token-serial causal control: distance from each full run | 7.5e-11, 7.5e-11 | 7.5e-11, 7.5e-11 |
| row counts M in [500, 800] that change one expert's rows [0, 500) | 13 of 301 | 299 of 301 |
| planted leak: past logits moved by 1e-10 x the call's mean over time | 2.2e-10 | 2.2e-10 |

**Diagnosis: numerical ordering, not leakage.**
- The grouped dispatch runs each expert's GEMMs on the rows routed to it. A row count depends on every route in the
  call, including future tokens': block 0's segments of 641 / 599 / 548 / 612 rows become 614 / 582 / 577 / 627.
- MKL's double-precision GEMM rounds a given row differently for some row counts: most of them under AVX2 kernels
  with several threads, a few natively. An exploratory probe with AVX2 kernels on one thread was bit for bit.
- No route and no value of a future token enters a past computation. The past routes are identical, and the
  fixed-shape reference and the all-rows control are bit for bit. The token-serial control, which cannot see the
  future, is equally far from both runs, 340,000 times the difference between them.
- The grouped path is causal to rounding.

**Repair (the test's contract, not the model):**
- Past routes must be identical.
- With the reference dispatch, past logits must be equal bit for bit.
- With the grouped dispatch, past logits must be within 1e-12 of each other and of the reference: 4,500 times the
  largest difference measured.
- New test `test_the_causality_contract_detects_a_planted_future_leak` plants a 1e-7 future dependence and must see
  it exceed 100 x 1e-12 under both dispatches.

Both tests pass natively and with AVX2 kernels, and so does the whole research suite. The exact-head CI result is
reported in the pull request, since a commit cannot record the CI run that qualifies it.

**Record writer:**
- `run_r6`'s writer rounded every float to six decimal places. That erased the dispatch difference (section 3, now
  corrected and re-measured) and would have erased this section's values. It now keeps six significant digits.
- The R1–R5 drivers still round to six decimal places. Their zero-valued difference fields bound those values below
  5e-7 only: `r1/adapt.json` `stream_logits_error_vs_recompute`, the `max_abs_logit_difference*` fields of
  `r4/parity_32k*.json`, `r4_1/parity_32k.json` and `r4_1/cache_workloads_*.json`, and `r4/gradients.json`
  `max_gradient_difference_*`. Their tests assert directly. The historical records are left unchanged.

```bash
python -B -m research.elpis_torch.experiments.run_r6 causality --threads 4 --out causality_native.json
MKL_ENABLE_INSTRUCTIONS=AVX2 python -B -m research.elpis_torch.experiments.run_r6 causality --threads 4 --out causality_avx2.json
MKL_ENABLE_INSTRUCTIONS=AVX2 python -B -m pytest tests/research/elpis_torch
```
