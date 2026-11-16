# Elpis Torch R4 — million-token context runtime: engineering evidence

Engineering only. R4 changes no learning objective, selection rule, key or query mathematics, and makes no
retrieval or reasoning claim: the scientific reference for retrieval is `EVIDENCE_R3.md`, which is not rewritten.
CPU only (PyTorch 2.15 CPU, 4 cores); every CUDA gate is NOT_RUN. Records: `research/elpis_torch/results/r4/`.

Stacked on R3: source `bdfe0dafcf89d8eb626dab3d8cb5baf6a4121fec` (PR #74, open). Status words: MEASURED (this
CPU), INSTANTIATED (exact tensor shapes on `meta`), PROJECTED (formula), PASS / FAIL / NOT_RUN.

## 1. Spark reference: what a "1M context" costs

Published configuration of XHToken Spark-X2.5-4B and X2.6-4B (`config.json` of both fetched 2026-11-16, byte-identical,
SHA-256 `767161ece8ed2891e44345afdf98d29c1cacb964116cb6db7d20dc1c757490ad`; not committed): 36 layers = 27 sliding-window + 9 full-attention (3:1), window 512, 16 query / 4 KV heads, head dim
256, hidden 2,560, vocab 131,072, `max_position_embeddings` 1,048,576, head-wise sigmoid attention-output gate,
partial rotary 0.25 (theta 5e6) on full layers and full rotary (theta 1e4) on sliding layers.

KV cache of one stream, bf16 (`spark_reference.spark_kv_bytes`, PROJECTED from the published geometry):

| tokens | 9 full layers | 27 sliding layers | total |
|---:|---:|---:|---:|
| 4,096 | 0.14 GiB | 54 MiB | 0.20 GiB |
| 32,768 | 1.13 GiB | 54 MiB | 1.18 GiB |
| 131,072 | 4.50 GiB | 54 MiB | 4.55 GiB |
| 262,144 | 9.00 GiB | 54 MiB | 9.05 GiB |
| 1,048,576 | **36.0 GiB** | 54 MiB | 36.05 GiB |

Each token costs 36 KiB of full-layer KV, forever. Distinctions that the headline "1M context" blurs:

| | Spark-style | Elpis R3/R4 |
|---|---|---|
| local window KV | 54 MiB, constant | 20-24 MiB (500M/1B configs), constant |
| full-history KV | grows 36 KiB/token; 36 GiB at 1M | none: an int32 token ring (4 MiB at 1M) and a learned block index (8.25 MiB) |
| 1M positional limit | a configuration limit; beyond it the cache cannot grow | positions are relative (ERSA has none; readout uses log-distance and ranks); streams continue past 1M (tested across 2^20) |
| genuine retention | every token is attended (lossless in principle) | lossy: only selected blocks are reread; R3 measured recall@2 0.31-0.51 at 1M (FAIL of its gate) |
| scrolling past the horizon | not possible without dropping KV (StreamingLLM-style eviction would change the model) | built in: deterministic eviction of tokens and index records |

Transferable principles (and their status here): sparse placement of expensive attention (already: 3:1 in the
scaling configs); very short local windows (compatible); attention-output gating (compatible, not adopted: it
would change the trained model); few KV heads (already 1-2); separate local/long-range position treatment
(already: rotary only in the window, log-distance and recency rank in the readout). Incompatible with a 1-2 GiB
context budget: any layer that keeps full-history KV. Even ONE Spark-geometry full layer is 4 GiB at 1M.
The cards state that X2.6 improves long context by post-training on an unchanged architecture: capacity to
address 1M positions is not the same as using them, which is also R3's finding.

Measured comparator (`spark_reference.SparkStyleLM`: untrained, independent implementation of the published
geometry scaled to the Elpis experiment width: hidden 64, 4 layers in the 3:1 pattern, 4 query / 1 KV heads, head
dim 16, window 512; MEASURED, PERFORMANCE_ONLY, `results/r4/spark.json`):

| tokens | prefill tok/s | decode ms/token | full + sliding KV allocated (exact = formula) |
|---:|---:|---:|---:|
| 4,096 | 11,244 | 2.85 | 0.73 MB |
| 32,768 | 3,404 | 5.16 | 4.40 MB |
| 131,072 | 902 | 23.30 | 16.98 MB |
| 262,144 / 1,048,576 | NOT_RUN (quadratic prefill on CPU) | | 33.8 / 134.4 MB (formula) |

Decode cost grows linearly with the context and prefill quadratically. The comparator's peak RSS (0.4 / 1.1 /
3.7 GB) is dominated by its naive dense score matrix and is not a property of Spark.

## 2. P0: audit of the R3 context runtime (before any change)

Read from `history_model.py` at `bdfe0da` and measured with work counters on the trained R3 seed-0 checkpoint
(`results/r4/bench_fixed_tile.json`, path A; horizon 2^20, block 32, two index keys per block, top-2).

| # | Path | What R3 does | Cost / risk found |
|---|---|---|---|
| 1 | one-token decode | per read layer: mask the held index slots (`bool(held.all())`, a device sync), copy them if the ring is not full, CONCATENATE ring keys with the chunk's new keys (a full copy of the index every token), build `distance`/eligibility `[T, N]`, score all `N` records, `topk`, then RE-ENCODE every selected resident block (40 tokens through the encoder), allocate a `compact [N]` table | measured 1.97 blocks re-encoded per decoded token; 8 MiB index copy per token at 1M (500M config); about 8 host syncs per call |
| 2 | chunked prefill | the same for `T` queries at once | scratch `[B, T, N]` x (pooled match x per_block, score, masked copy) + `[T, N]` int64 distance + float log-distance + bool eligibility: about 0.9 GiB per read layer per 1,024-token chunk at N = 32,768 (fp32); measured peak RSS 828 MiB at 1.1M tokens with 512-token chunks vs 388 MiB after the change |
| 3 | new block complete | encoded once from token ids (history + chunk), keys per read layer, written DETACHED into the ring at commit | correct; detached keys are the cross-chunk gradient gap (section 6) |
| 4 | index insert / evict | `index_copy_` into slot `b mod slots`; the oldest record is overwritten | deterministic, O(new blocks); no defect |
| 5 | top-K | `torch.topk` over the whole table | tie order implementation-defined (fixed in R4: newer block first) |
| 6 | original-token reread | `TokenHistory.gather` with two validation syncs, `int(where[old].min())` | syncs per call; tokens and positions stay on one device (no transfer) |
| 7 | block re-encoding | every selected resident block, every call | measured 82,255 re-encodings over a 1.1M-token prefill (A) |
| 8 | repeated access | no reuse: the same block is re-encoded each time it is selected | ~2 per decoded token |
| 9 | session reset | no reset: a new state must be allocated | allocation churn |
| 10 | past the 1M horizon | ring overwrite and absolute block ids; nothing special | works; R3 never measured beyond exactly 2^20 (R4 streams 2^20 + 2^16) |

Also: logits `[B, T, vocab]` are computed for every chunk position, and R3's `ErsaLM.run` concatenates them for
the whole input (1M x 593 x 4 B = 2.3 GiB per stream for the experiment vocabulary; 256 GiB at a 65,536
vocabulary). Device-state movement: none in R3 (all tensors on the model device); the R4 placement options add
optional ones (section 3). Cache-invalidation risk: the index keys are computed with the encoder weights of the
moment and are never invalidated, so changing an encoder or reader weight mid-stream would leave stale keys; the
only adaptation R3 supports (last-block CGC fast weights) does not touch them. R4 keeps that contract and its new
cache does invalidate (section 5).

## 3. What R4 built

| Piece | Where | Contract |
|---|---|---|
| deterministic exact top-K | `context_runtime.exact_topk` | R3's ranking; exact ties resolved newer block first; used by the reference AND the tiled path |
| tiled exact scoring | `context_runtime.tiled_select`, `ErsaHistoryLM._tiled_read` | R3's score, eligibility and ranking over the occupied index records only, as zero-copy contiguous views, in tiles of `max(tile, tile_elements / (B T))` records (defaults 4,096 and 2^21) with a running exact top-K merge; scratch independent of the horizon |
| encoded-block cache | `context_runtime.BlockCache` | session-local, hard byte bound, LRU eviction, purge of blocks outside every future horizon, invalidation on any embedding/encoder parameter change (pointer, version counter, dtype, device), bypassed whenever gradients are enabled |
| streaming session | `context_runtime.ContextSession` | frozen model only; chunked prefill keeping only the final logits and requested probes (`ErsaLM.forward(logit_positions=...)`); incremental decode; continuation over calls; in-place `reset`; `close`; budget refusal before allocation; not picklable or copyable; no storage writes |
| placement | `model.new_state(history_device=..., index_device=...)` | the token ring and/or the index may live off the model device (CPU RAM); tiles and rereads are copied to the model device when used |
| key replay | `RuntimeOptions(key_grad_blocks=G)` (reference path, training) | resident index keys of the last `G` blocks (`-1`: all) recomputed from their original tokens WITH gradient instead of the detached stored keys; same values, same scoring, same eligibility |
| exact budget | `scaling.fixed_state_bytes`, `session_scratch_bytes`, `r4_context_budget` | fixed state equal to the allocated bytes (tested); scratch PROJECTED |
| comparator | `spark_reference.py` | section 1 |

R3's default path is unchanged except the tie rule. Re-evaluating the committed R3 seed-0 records (1K/4K/16K,
learned and oracle modes) reproduces every probe statistic to the records' 6-decimal rounding (< 5e-7) with
identical recall@K and evidence-read rates.

## 4. P1/P2: streaming parity, exact top-K, rollover

| Gate | Result | Evidence |
|---|---|---|
| tiled selection = reference selection | PASS | tests: float64, chunk 64/13/1, tiles 1..1000, partially filled index, past the horizon, with and without window attention: identical block ids, logits <= 1e-10; float32: identical ids, <= 1e-4. Trained checkpoint, 32K tokens (`results/r4/parity_32k.json`): 65,536 selections identical, logit difference 0.0 for B and C |
| exact ties | PASS within the stated tolerance | bitwise ties are resolved newer-first in every path (brute-force-checked `exact_topk`; repeated-block fixture). Across DIFFERENT chunkings or tilings a block may be encoded in a different batch and its score differ in the last bits; such near-ties can be ordered differently (this is true of the R3 reference itself); the selected scores then agree to 1e-12 (tested) |
| streaming API | PASS | prefill probes and final logits equal `model.run` (<= 1e-10), continuation over calls, token-by-token decode, in-place reset reproduces a fresh stream bitwise with the same buffers, plain `ErsaLM` too |
| logits not materialized | PASS | the head runs only at requested positions (`logit_positions`; equal to slicing the full logits to rounding, <= 1e-12 float64); B/C keep one `[B, vocab]` per call |
| crossing absolute position 2^20 | PASS | a stream from origin 2^20 - 160 equals the same stream from 0 to rounding (<= 1e-12 float64), with block ids shifted exactly |
| eviction, horizon | PASS | selections never older than the query's horizon or incomplete; the ring holds exactly the last `horizon/block` blocks; repeat runs select identically |
| 1M stream past the horizon | MEASURED | 1,114,368 tokens (2^20 + 2^16 + 256 decoded) in A/B/C: final position 1,114,368, oldest token held 65,752, state bytes constant, buffers never reallocated |
| invalid addresses / evicted positions | PASS | refused (`EvictedError`, `ValueError`) |
| session isolation, zero writes | PASS | separate caches and states; interleaved sessions give the single-session result; no writer in the new modules (isolation gate), sessions and caches refuse pickling and copying |

## 5. P3: encoded-block cache

| Check | Result |
|---|---|
| cold/warm parity (capacities 1 B, 3 entries, 1 MiB; window and none) | PASS: identical selections, logits <= 1e-12 (float64) |
| byte bound | PASS: never exceeded (tests); at 1M the 32 MiB cache held 33,547,008 bytes |
| invalidation on an encoder weight change | PASS: cleared once; the continuation equals an uncached run with the new weights |
| purge beyond the horizon | PASS: no entry older than the oldest eligible block |
| training | PASS: bypassed with gradients enabled (no detached tensor reused) |

Measured with the trained checkpoint (`results/r4/bench_final.json`, 512-token chunks, 256 decoded tokens; MEASURED,
PERFORMANCE_ONLY):

| tokens | blocks re-encoded during prefill A / C | cache hits / misses / evictions (C) | decode re-encodes per token B / C |
|---:|---:|---|---:|
| 4,352 | 578 / 0 | 578 / 0 / 0 (+504 in decode) | 1.97 / 0 |
| 33,024 | 1,904 / 0 | 1,904 / 0 / 0 | 1.97 / 0 |
| 131,328 | 7,362 / 0 | 7,362 / 0 / 133 | 1.97 / 0 |
| 262,400 | 15,914 / 0 | 15,914 / 0 / 4,229 | 1.97 / 0 |
| 1,114,368 | 82,633 / 3 | 82,630 / 3 / 30,856 | 1.97 / 0 |

## 6. P4: cross-chunk gradients

Tests (float64): with `key_grad_blocks=-1` the gradients of a 200-token stream trained in 40-token chunks equal the
one-chunk gradients for every parameter (<= 1e-9); with the R3 behaviour (detached keys) the encoder gradients
differ (> 1e-6), so the test detects the gap. `results/r4/gradients.json` (480 tokens, 60-token chunks, 8-token
blocks, largest gradient entry 537.9):

| replay horizon (blocks) | max gradient difference vs unchunked | blocks replayed | replay activations (PROJECTED scalars) |
|---:|---:|---:|---:|
| 0 (R3: detached) | 1.39 | 0 | 0 |
| 2 | 1.33 | 14 | 39,424 |
| 8 | 1.75 | 55 | 154,880 |
| 32 | 0.62 | 170 | 478,720 |
| all resident | **0.0** | 208 | 585,728 |

Only full replay restores the exact gradient. A bounded horizon is truncated backpropagation: it delivers the
gradient of the replayed keys exactly but omits the rest, and the residual is not monotone in the horizon. Memory
is bounded by the horizon: `G` blocks x span x encoder width per read layer. This makes the existing objective
correctly trainable across chunks (e.g. with a window that forces short chunks); it does not claim to fix learned
addressing or long-context generalization. No training run was made.

## 7. P5: GPU deployment memory at 1M (500M / 1B)

`results/r4/scale.json` (`scaling.r4_context_budget`; chunk 512, tiles per section 3, cache 32 MiB, bf16
activations):

| | `r3_500m` | `r3_1b` |
|---|---:|---:|
| instantiated parameters | 492,765,430 | 980,182,246 |
| weights bf16 / fp32 | 0.92 / 1.84 GiB | 1.83 / 3.65 GiB |
| ERSA cells (fp32 accumulation) | 15.0 MiB | 20.0 MiB |
| conv tails | 0.18 MiB | 0.23 MiB |
| window KV (bf16) + its token ring | 24.0 + 0.02 MiB | 20.0 + 0.02 MiB |
| token history (int32, 2^20 + 80 slots) | 4.0 MiB | 4.0 MiB |
| block index (2 read layers) | 8.25 MiB | 8.25 MiB |
| **exact state total** | **51.4 MiB** | **52.5 MiB** |
| block cache (configured bound) | 32 MiB | 32 MiB |
| prefill scratch, tiled (PROJECTED) | 767.5 MiB | 785.5 MiB |
| prefill scratch, R3 reference scorer (PROJECTED) | 915.7 MiB | 933.7 MiB |
| **GPU context total, all on GPU** | **851 MiB** (target 1 GiB: within) | **870 MiB** (within) |
| token history moved to CPU RAM | GPU 847 MiB + CPU 4.0 MiB | GPU 866 MiB + CPU 4.0 MiB |
| GPU total incl. bf16 weights | 1.75 GiB | 2.68 GiB |

The state is exact; the scratch dominates and scales with the prefill chunk (decoding, `T = 1`, needs a few MiB).
These numbers supersede R3's projection (619.5 / 688.6 MiB), whose scratch model omitted the readout gathers.
GPU execution: NOT_RUN. `run_r4 gpu` (and the CUDA tests) measure fp32 parity, CPU-resident history, cache
behaviour, bf16 autocast, budget refusal and the 500M configuration's peak context memory streamed past 2^20.

## 8. P6: performance comparison (CPU, MEASURED, PERFORMANCE_ONLY)

Trained R3 seed-0 checkpoint (SHA-256 prefix `bbc0a697329fed00`), random token streams (runtime only; NOT recall
evidence), 512-token chunks, 4 threads, one process per run (`results/r4/bench_final.json`):

| tokens | prefill tok/s A / B / C | decode ms/token, median A / B / C | decode p90 A / B / C | peak RSS MiB A / B / C |
|---:|---|---|---|---|
| 4K | 26,321 / 22,077 / 27,194 | 3.74 / 3.78 / 3.11 | 4.51 / 5.02 / 3.78 | 270 / 268 / 269 |
| 32K | 20,591 / 19,497 / 23,767 | 4.12 / 3.74 / 2.54 | 5.87 / 4.32 / 3.40 | 280 / 278 / 285 |
| 128K | 14,325 / 13,436 / 13,704 | 3.94 / 3.72 / 3.50 | 5.48 / 4.98 / 3.77 | 338 / 345 / 382 |
| 256K | 10,342 / 9,735 / 10,623 | 3.98 / 4.88 / 3.10 | 5.81 / 5.87 / 4.14 | 409 / 367 / 417 |
| 1M (+64K past) | 3,084 / 4,070 / 4,498 | 4.90 / 4.96 / 4.25 | 6.85 / 6.09 / 5.17 | **823 / 385 / 444** |

At 1M: the runtime halves the peak memory (823 -> 385 MiB without, 444 MiB with the 32 MiB cache) and raises
prefill throughput by 32 % (B) and 46 % (C); with the cache, decoding is the fastest at every length. Below 1M,
B's prefill is 5-16 % slower than A (the per-tile merge costs more than it saves when the index is small); C
recovers it by avoiding re-encodings. Decode cost of all three paths is nearly flat in the context
length (index scoring is linear in `horizon/block` but small); the Spark-style comparator's decode grows 8x from
4K to 128K (section 1). Index lookup work at 1M: 32,768 records (65,536 dot products of 32 dimensions) per
decoded token for the model's one read layer (`records_scored` 8,388,616 over 256 decoded tokens).

## 9. Gates

| Gate | Status |
|---|---|
| P0 context-runtime audit | PASS (section 2) |
| P1 bounded streaming, logit parity, rollover, deterministic eviction | PASS (section 4; CPU) |
| P2 optimized scoring returns the reference candidates and outputs | PASS (exact on non-degenerate scores; rounding-level near-ties documented) |
| P3 cache invalidation, bounds, cold/warm parity | PASS |
| P4 cross-chunk gradient delivery, bounded memory | PASS (exact with full replay; bounded replay = truncated gradient, measured) |
| P5 exact accounting 500M/1B at 1M, executable GPU qualification | PASS (accounting); GPU qualification NOT_RUN |
| P6 reproducible comparisons with negative results kept | PASS |
| future-token leakage | PASS (R3 tests unchanged; tiled/cached paths use the same eligibility and are compared position by position) |
| session isolation | PASS |
| zero runtime writes, no checkpoint mutation | PASS (isolation gate; frozen-model requirement; base parameters never assigned) |
| no uncontrolled memory growth | PASS on CPU (state constant to 1.11M tokens, buffers never reallocated, cache within its bound) |
| no unexplained numerical drift | PASS (0.0 on the trained float32 checkpoint at 32K; float64 <= 1e-10) |
| CUDA gates | NOT_RUN |

## 10. Negative results and unresolved items

1. **First tiled scorer was 4x slower in prefill** (5.9K vs 23K tok/s at 64K): it scored every allocated slot
   (32,768 at a 1M horizon) including empty ones. Fixed by scoring only occupied records.
2. **Fixed 4,096-record tiles made decoding 2x slower at 1M** (10.3 vs 5.2 ms/token): one-token queries paid the
   per-tile merge 8 times and copied the index through `index_select`. Fixed with zero-copy ring views and tiles
   sized by an element budget (`results/r4/bench_fixed_tile.json`).
3. **A 2^22 element budget made prefill slower and larger at 1M** (3,119 tok/s, 498 MiB) than 2^21 (4,070
   tok/s, 385 MiB): kept as `results/r4/bench_tile22.json`; the default is 2^21.
4. **B is not faster than A below 1M** (5-16 % slower prefill); its value is bounded memory.
5. Decode timings vary by +-1 ms between runs on this shared CPU (64-token samples ranked B and C differently in
   two rounds); the final round uses 256 tokens and reports median and p90.
6. **Tie order across different chunkings is only defined up to rounding** (section 4), for the reference too.
7. **Bounded key replay is not a smooth approximation** (section 6).
8. Remaining host synchronizations per call: token-range validation (2), occupied-slot bounds (2), the
   boundary-tie check of `exact_topk` (1 per tile), cache bookkeeping (Python dictionary). Not measured on a GPU.
9. The CPU-index placement copies index tiles to the model device for every call: with `index_device="cpu"`
   decoding at 1M would move 8 MiB per token per read layer. Implemented for memory-constrained placements, not
   recommended; not measured on a GPU.
10. The comparator is untrained and tiny; its numbers compare context costs, not models.
11. **CI found two over-strict assertions (TEST-CONTRACT DEFECT, docs/CI_POLICY.md).** Push run 435 on head
    `166548b` failed `test_logit_positions_equal_slices_of_the_full_logits` (8.3e-17) and
    `test_streams_continue_across_absolute_position_2_pow_20_with_unchanged_logits` (1.7e-16), while pull-request
    run 436 on the same head passed: same tree, different verdicts, i.e. ENVIRONMENT-SENSITIVE. Cause: both tests
    demanded bitwise equality between computations with different matrix shapes (BLAS rounding depends on shape
    and CPU). The asserted law is unchanged and now carries the documented float64 tolerance (1e-12); selections
    are still required to be identical. Nothing was re-run to obtain a green result.
12. Not built: a hierarchical or approximate index (would change selections: a separate scientific experiment),
    a fused GPU kernel, any learned-addressing change (owned elsewhere).

