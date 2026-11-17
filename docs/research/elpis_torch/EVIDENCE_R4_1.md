# Elpis Torch R4.1 — context runtime correction and hardening

Corrective engineering on R4 (`EVIDENCE_R4.md`, not rewritten; R4 records in `results/r4/` unchanged). No change to
ERSA, CGC, the history query/key definitions, the readout, the selection rule or any training objective; no
training. CPU only (PyTorch 2.15 CPU, 4 cores); every CUDA item is NOT_RUN. Records: `research/elpis_torch/results/r4_1/`.
Tests: `tests/research/elpis_torch/test_r4_1.py` (33). Entering head: `8b5de9b` (PR #75, stacked on #74).

## 1. Defects found and fixed

| Gate | Defect (reproduced by a test that failed on `8b5de9b`) | Fix | Status |
|---|---|---|---|
| `CURRENT_POSITION_LOGITS` | `prefill(..., keep_last=False)` advanced the state but kept the previous call's logits; `generate()` then continued from a prediction made for an earlier position (`Failed: DID NOT RAISE`) | the session keeps a prediction only together with the position it belongs to (`current_logits()` is `None` once the state moved, also through direct model calls); `generate` refuses without a current prediction; empty prefill = no-op; `generate(0)` = empty `[B, 0]`, no change; negative, boolean, float or string counts refused; reset and close invalidate | PASS |
| `ABSOLUTE_ORIGIN_INDEX` | `ring_segments` counted `min(next_block, slots)` records as occupied: a stream opened at a large origin scanned the whole empty capacity (`assert 512 == 5`) | `BlockIndex.count` (records written and still held), maintained on write and reset; `ring_segments` uses it. Invariant tested after every write: `count == (ids >= 0).sum()`, the views hold exactly the blocks they claim | PASS |
| `SESSION_MODEL_REVISION` | a live session would continue with index keys and recurrent state computed under a different model revision (only the block cache noticed encoder changes) | `RevisionWatch` (below); continuation refused with `ModelRevisionError` until `reset`, which rebinds to the model as it is; opt-in SHA-256 digest | PASS |

Correctness of the occupancy fix (tests): origins 0, 2^17, 2^17 + 3, 2^20 and 2^20 - 512; empty, partial, exactly
full, wrapped and over-full writes; continuation past 2^20 with eviction; two sessions at different origins; reset
and reuse of the same buffers; tiled selections and logits equal to the R3 reference path (which masks empty slots
explicitly) at every origin; scoring work per call bounded by (held + in-chunk) records x queries, never by the
capacity; every selected block complete at its query and inside its horizon.

## 2. The model-revision contract

A running `ContextSession` may continue only with the model revision its state, index and cache were computed
with. What is checked, and at what cost (`results/r4_1` timings; MEASURED on this CPU):

| Change | Detected by | Cost per call |
|---|---|---|
| in-place op on any parameter or buffer (embedding, block encoder, index-key projection, reader, ERSA, final-block CGC: all tested) | PyTorch version counters of the tensors cached at bind | fast path: 25 us (small model), 190 us (500M configuration, 404 tensors, on `meta`) |
| replaced parameter, buffer or submodule (even with equal values) | global PyTorch registration hooks (a counter) -> full `model_revision` comparison | one slow check (~0.65 ms at 500M) when any registration happened anywhere |
| storage moved (`.to(dtype/device)`), reshaped, unfrozen, `train()` | storage pointer, dtype, device, shape, `requires_grad`, training flag | in the fast path |
| writes that bypass version tracking (`param.data.mul_`, foreign views via NumPy/DLPack, raw storage) | NOT detected by the cheap check (tested); detected by the opt-in `verify_digest()` (SHA-256 of every parameter, `record_digest=True`) | a full read of the weights per verification |

Checked once per `prefill` and `step` call and once per `generate` call (not per generated token). An unrelated
registration elsewhere costs one slow check and never a false refusal (tested). Sessions sharing a frozen model are
independent and each notices a change. The check is not cryptographic. The final-block CGC fast-weight interface
envisaged by R1/R3 is a different, explicitly authorized mechanism and is not implemented here; under this
contract an adapted final-block weight would be a new revision.

## 3. Adversarial encoded-block cache

Cache mechanics (`results/r4_1/cache_patterns.json`; entries of the 500M configuration, 33,280 bytes; a 1M horizon of
32,768 blocks; 20,000 accesses):

| Access pattern | 32 MiB (1,008 entries) | 64 entries | < 1 entry |
|---|---:|---:|---:|
| uniform over the horizon | **2.9 %** hits | 0.2 % | 0 % (all rejected) |
| alternating oldest / newest | 100 % | 100 % | 0 % |
| cyclic scan of 2x capacity | **0 %** (LRU worst case) | 0 % | 0 % |
| hot set (10 blocks, 90 %) | 90.3 % | 90.1 % | 0 % |
| cold, never repeated | 0 % | 0 % | 0 % |

Bookkeeping: 0.15-0.4 us per hit, 1.2-1.3 us per miss with eviction; the byte bound held in every run.

Model-level workloads (`results/r4_1/cache_workloads_65k.json`, `..._262k.json`): the trained R3 seed-0 checkpoint
decodes 256 teacher-forced tokens after a 65,536- or 262,144-token history whose 32 bound keys are spread evenly;
the question pattern steers the model's OWN selections (filler, uniform, alternating oldest/newest, hot, cyclic
scan, identical block contents at every address). Capacities: none, 32 MiB, 7 entries, less than one entry.

* **Parity: exact.** Cached and uncached decode logits are identical (max difference 0.0, float32) in all 23 cached
  runs, including identical-content blocks (exact score ties) and capacities smaller than one entry.
* **Hits.** 32 MiB: 99.2-100 %. 7 entries: 39 % (cyclic scan), 42 % (identical content), 45 % (uniform),
  92 % (alternating), 99 % (hot, filler). Less than one entry: 0 % (every insertion rejected).
* **Latency (median ms per decoded token, B / C).** 32 MiB at 262K: uniform 4.50 / 3.89, alternating 4.60 / 4.56,
  hot 4.27 / 3.77, cyclic 4.79 / 3.88, identical 4.54 / 3.55. Small capacities give NO gain and are sometimes
  slower (65K, 7 entries: filler 4.15 / 5.30, identical 4.28 / 4.51; < 1 entry: uniform 4.24 / 5.15). Run-to-run
  noise on this shared CPU is about +-1 ms.
* **Overhead.** Cache management 15-22 ms per 256 decoded tokens (60-90 us per token, including purges of
  entries that left the horizon).
* **Limitation of the model-level fixture.** Decode-time accesses are bounded by the 32 bound keys (at most a few
  dozen distinct blocks), so the 32 MiB cache cannot be overflowed by this task; capacity pressure is measured with
  the small capacities and with the cache-mechanics patterns above, where uniform and scanning access defeat LRU.

Further tests: a reused ring slot never serves the evicted block's encoding (keyed by absolute block, purged);
interleaved sessions with small separate caches equal their solo runs; nothing older than the horizon survives in
the cache.

## 4. Memory accounting: what is exact, what is projected, what is measured

| Class | What | How established |
|---|---|---|
| A. exact persistent tensor storage | ERSA cells, convolution tails, window rings, token history, block index (51.4 / 52.5 MiB for the 500M / 1B configurations at 1M) | `scaling.fixed_state_bytes`, equal to the allocated bytes (tested); unchanged by R4.1 (`BlockIndex.count` is a Python integer) |
| configured bound | encoded-block cache, 32 MiB | enforced at every insertion (tested) |
| B. projected working memory | per-call scratch (767.5 / 785.5 MiB at a 512-token prefill chunk) | `scaling.session_scratch_bytes`: components summed as if all were live at once; NOT an allocator ceiling |
| C. measured allocator peak | | CUDA: NOT_RUN (`run_r4 gpu-horizon`). CPU, small model, 1.11M tokens (R4 `bench_final.json`): RSS growth 587 MiB measured vs 594 MiB projected for the R3 reference path, 149 vs 235 MiB for the tiled path, 208 vs 267 MiB with the cache |

So the R4 figures of 851 / 870 MiB GPU context are PROJECTED: exact state + configured cache + a scratch
estimate. On CPU the estimate bounded what was measured; on CUDA it is unverified. Not modelled: the caching
allocator's reserved-but-free blocks and fragmentation, autocast's cached bf16 casts of fp32 weights (about half the
fp32 weight size, counted with the weights, not the context), library workspaces. Decoding (`T = 1`) needs only a few
MiB of scratch; the estimate scales with the prefill chunk.

`ContextSession.projected_bytes()` now states this (`status`), reports the exact state by tier and the scratch by
component, and splits the total by device according to placement: the token history on `history_device`, the index
on `index_device`, everything else on the model's device (tested). Transient transfers of the placements: a
CPU-resident token ring moves only the token windows of reread blocks (`blocks x span` int64 per call); a
CPU-resident index copies every scored tile to the model device, i.e. the whole index (8.25 MiB at 1M for the
scaling configurations) per decoded token: valid for memory-starved devices, costly in bandwidth, unmeasured on a GPU.
The budget refusal at construction refuses clearly over-budget configurations before allocating anything; passing
it guarantees nothing about the measured peak (stated in the code).

## 5. GPU qualification harness (`run_r4`)

| Requirement | Implementation |
|---|---|
| no GPU work at import | nothing allocates at import; registration hooks are installed lazily by the first session and only count |
| no interference with other processes | no process control, driver, toolkit or system setting is touched; nothing is downloaded |
| explicit device | `--device cuda:N` is required by `gpu` and `gpu-horizon`; `smoke` never starts CUDA work without it; CUDA tests use `ELPIS_TORCH_CUDA_DEVICE` (default `cuda:0`) |
| admission before allocation | `torch.cuda.mem_get_info` free memory must exceed the requirement + 512 MiB, else `REFUSED_ADMISSION` (small model: 256 MiB; 500M: fp32 weights + projected context + half the weights for autocast casts, about 3.6 GiB) |
| progressive | `gpu`: small model only (FP32 GPU-history, FP32 CPU-history, FP32 cache cold+warm, then BF16 autocast). `gpu-horizon` runs those first and stops (`STOPPED_AFTER_SMALL_MODEL`) unless they pass |
| explicit full horizon | `gpu-horizon` refuses without `--confirm-full-horizon`; `--config r3_500m|r3_1b`, `--tokens` |
| measurement | memory relative to a baseline taken after `synchronize` (`memory_allocated`), peaks with `max_memory_allocated` and `max_memory_reserved` after `reset_peak_memory_stats`, at 2^16, 2^18, 2^19, 2^20 and the end; every latency bracketed by `synchronize` |
| failure classes | `ALLOCATION_FAILURE` (`torch.cuda.OutOfMemoryError`), `NUMERICAL_FAILURE` (FP32 parity > 1e-3 or non-finite BF16), `REFUSED_ADMISSION`, `NOT_RUN` |
| cleanup | `try/finally`: sessions closed, references dropped, `gc.collect`, `torch.cuda.empty_cache` |

Cost of `gpu-horizon` for the 500M configuration (PROJECTED, unmeasured): about 0.9 G multiply-accumulates per
token (R4 accounting) x 1.11M tokens = 2 PFLOP of arithmetic, minutes at tens of TFLOP/s, plus Python and kernel
launch overhead for about 2,200 chunks; GPU memory about 3.6 GiB (fp32 weights, autocast casts, context). The
small-model `gpu` command takes seconds and under 256 MiB.

Owner commands (workstation; nothing here was executed on a GPU):

```bash
ELPIS_REQUIRE_TORCH=1 ELPIS_TORCH_CUDA_DEVICE=cuda:0 python -B -m pytest -p no:cacheprovider -q tests/research/elpis_torch
python -B -m research.elpis_torch.experiments.run_r4 gpu --device cuda:0 --out gpu_smoke.json
python -B -m research.elpis_torch.experiments.run_r4 gpu-horizon --device cuda:0 --confirm-full-horizon --out gpu_horizon.json
```

Add `--ckpt <R3 checkpoint>` to the first two for trained weights (none ships with the repository). Run them when
the GPU has the free memory the admission check reports; they do not stop or reconfigure anything already running.

## 6. Numerical and scientific equivalence (R3, R4)

| Check | Result |
|---|---|
| R3 records (seed 0, 1K / 4K / 16K; learned, oracle-override, history-off) re-evaluated | every probe statistic within the records' 6-decimal rounding (< 5e-7); recall@K and evidence-read identical |
| trained checkpoint, 32K tokens, A/B/C (`results/r4_1/parity_32k.json`) | 65,536 of 65,536 selections identical; logits difference 0.0 (B and C) |
| R4 test suite (tiled top-K, ties, cache cold/warm, chunked/incremental, full and bounded replay, rollover, eviction, isolation, zero writes) | PASS, unchanged |
| R4.1 changes to outputs | none: no correction changed a logit on any compared path (the defects concerned which logits a session returns, what work is done, and what is refused) |

No reliable 1M-token retrieval is claimed: R3's gate G7b (learned retrieval at 1M) remains FAIL.

## 7. Register

* Not built: the final-block fast-weight interface, any cryptographic integrity scheme beyond the opt-in digest, a
  cache policy other than LRU (scan-resistant policies would be a separate engineering experiment), GPU kernels.
* NOT_RUN: every CUDA measurement.
