# Elpis Torch R8 — exact evidence read-path acceleration: evidence

Research only. CPU: 4 cores, AVX-512 with AMX, torch 2.15, Python 3.13. No CUDA. The timing models are untrained.
Design: `DESIGN_R8.md`. Records: `research/elpis_torch/results/r8/`. Runner: `experiments/run_r8.py`.

Status words: PASS / PARTIAL / FAIL / HOLD / NOT_RUN / NONQUAL / PROJECTED, and MEASURED (this CPU) or EXACT.
Wall-clock numbers are PERFORMANCE_ONLY. No faster run was kept in place of a slower one: every configuration of a
table was measured in the same process, interleaved step by step with a rotating order, and every row measured is
reported.

## 0. Base

Neither earlier PR is merged: PR abraxis717/Elpis_DEV#76 (R6, base `main`) and PR abraxis717/Elpis_DEV#77 (R7,
stacked on R6) are both open. As the directive requires, R8 therefore branches from the R7 head
`b41842c1e9a6ba5be96806b15d9412ec143f919b` (`claude/r7-ple-admission`) as `claude/r8-evidence-read-path`. Its PR is
stacked on the R7 branch.

The R7 head's exact-head CI is green: push run 44318126199 and pull-request run 44318200660, 16 of 16 jobs each. The
research job passed with 473 passed and 2 skipped, both on the runner's own kernels and with the AVX2 paths forced.

The conclusions and limits of `EVIDENCE_R6.md` and `EVIDENCE_R7.md` are preserved:

- R5 did not solve autonomous recall.
- Label-free hard coarse selection has no gradient.
- The PLE improved only small-scale perplexity.
- Router specialization is unproven.
- GPU behaviour is unmeasured.

The session's original branch, `claude/funny-turing-obbdy5` (PR abraxis717/Elpis_DEV#74, merged), was not touched.
Its one later commit, `fe82cc8`, was reconciled into R6: its records are byte-identical in this tree, and R6
replaced its loader change with the explicit slot-extension rule.

## 1. P0 — where the R5 read's time goes, and what the R6 cache does about it

Records: `results/r8/baseline_enc64.json`, `baseline_enc256_ple_off.json`, `baseline_enc256_ple_on.json`
(`run_r8 baseline`, runner committed at `d3e5662` before any measurement).

**Setup (matched).**

- One model per head geometry: the R7 latency core (d 256, 4 blocks, 8 experts top-2 in blocks 1 and 3, the PLE
  before block 1) with an R5 head:
  - `enc64`: R5's defaults, a 64-wide encoder and query tile 128;
  - `enc256`: the R6 500M / 1B head, a 256-wide encoder, 8 read heads of 128 and query tile 32.
- Capacity 1,048,576 tokens, untrained, float32, 4 threads.
- One seeded byte stream per (geometry, PLE) is ingested once, without reads, to 4,096 / 65,536 / 262,144 /
  1,048,576 tokens. At each occupancy every configuration starts from its own copy of that snapshot.
- Configurations:
  - A, `off`: the read disabled; the index is still maintained.
  - B, `uncached`.
  - C, the R6 cache through `ContextSession(cache_bytes=...)`.
  - D, the cache attached directly, at 64, 512 and 4,096 encoded blocks (2.1 / 16.5 / 132 MiB for `enc256`;
    0.5 / 4.1 / 33 MiB for `enc64`).
  - E, each with the PLE off and on.
- Decode: 64 steps of the same token feed for every configuration, interleaved step by step.
  - `natural`: the cache starts empty and warms; steps 0-7 are reported as cold, 8-63 as warm.
  - `replay`: the same 64 steps again, with each cache as the natural pass left it. These are repeated queries,
    the warm bound. It is UNREALISTIC WARMING and is reported separately.
  - `novel`: the cache is emptied before every step.
- Prefill: 512 tokens with logits at every position (a cold chunk, then a warm chunk, 2 runs), and with logits at
  the last position only.
- Stage times: the model's own laps (`coarse_scan`, `exact_reread`, `rerank`, `readout`), plus measurement-only
  wrappers that split out validation, token rereads, re-encoding, the rest of the reread, the local block's encoding
  and the top-K.

**Decode, ms per token, p50 / p95** (hit rate in brackets). `overlap` is the fraction of a step's candidates that
the previous step also asked for. `first-time` is the fraction of requested blocks never requested before in the
64 steps.

| head | PLE | tokens | R5 off | uncached | cache 512 | cache 4,096 | session, cache 512 | replay, cache 4,096 | novel, cache 512 | overlap | first-time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enc64 | off | 4K | 5.53 / 7.39 | 29.9 / 43.1 † | 9.90 / 14.0 (97%) | 9.87 / 13.5 (97%) | 10.0 / 14.9 | 10.3 / 14.9 | 30.5 / 38.8 | 52.2% | 3% |
| enc64 | off | 64K | 5.65 / 9.17 | 20.0 / 35.6 | 17.7 / 33.1 (31%) | 14.6 / 31.2 (63%) | 18.3 / 30.6 | 10.1 / 14.6 | 20.1 / 33.5 | 5.3% | 37% |
| enc64 | off | 262K | 5.53 / 6.85 | 20.9 / 26.3 | 19.5 / 27.5 (12%) | 18.1 / 26.7 (35%) | 20.1 / 27.1 | 10.0 / 13.8 | 18.7 / 22.9 | 2.2% | 65% |
| enc64 | off | 1M | 5.62 / 8.70 | 21.2 / 28.6 | 21.3 / 26.7 (6%) | 20.5 / 26.9 (18%) | 21.8 / 28.4 | 13.2 / 19.2 | 23.2 / 34.9 | 1.3% | 82% |
| enc64 | on | 4K | 5.57 / 7.47 | 18.8 / 22.2 | 9.40 / 16.9 (97%) | 9.63 / 14.8 (97%) | 9.65 / 11.9 | 9.74 / 13.5 | 19.5 / 27.6 | 49.3% | 3% |
| enc64 | on | 64K | 5.67 / 7.25 | 18.5 / 22.7 | 16.5 / 23.4 (31%) | 14.0 / 23.6 (62%) | 16.6 / 23.0 | 10.4 / 15.6 | 19.3 / 25.8 | 3.8% | 38% |
| enc64 | on | 262K | 5.53 / 9.26 | 17.7 / 23.7 | 17.4 / 24.1 (12%) | 16.1 / 24.2 (33%) | 17.8 / 26.3 | 10.9 / 14.2 | 19.4 / 27.6 | 1.3% | 67% |
| enc64 | on | 1M | 5.92 / 7.67 | 20.4 / 26.1 | 20.6 / 25.5 (4%) | 19.9 / 27.0 (16%) | 21.2 / 30.6 | 12.4 / 15.2 | 20.2 / 24.9 | 0.4% | 84% |
| enc256 | off | 4K | 5.51 / 7.82 | 82.4 / 110 † | 14.7 / 21.5 (97%) | 13.9 / 26.8 (97%) | 13.5 / 23.8 | 14.0 / 19.0 | 86.8 / 108 | 51.0% | 3% |
| enc256 | off | 64K | 6.27 / 9.31 | 65.5 / 84.2 | 49.6 / 71.7 (29%) | 33.8 / 66.4 (60%) | 48.5 / 70.9 | 14.2 / 20.3 | 71.6 / 88.6 | 4.7% | 40% |
| enc256 | off | 262K | 6.13 / 8.89 | 62.2 / 87.0 | 58.0 / 73.0 (10%) | 45.1 / 65.8 (29%) | 53.2 / 69.7 | 14.6 / 24.2 | 59.6 / 86.1 | 1.8% | 71% |
| enc256 | off | 1M | 6.03 / 9.63 | 59.3 / 83.9 | 58.1 / 80.7 (3%) | 54.5 / 80.3 (12%) | 58.5 / 77.7 | 17.7 / 27.9 | 66.7 / 96.3 | 0.9% | 88% |
| enc256 | on | 4K | 5.99 / 9.17 | 84.1 / 128 † | 14.5 / 47.2 (97%) | 14.8 / 32.9 (97%) | 14.8 / 23.6 | 14.3 / 19.7 | 89.4 / 109 | 49.4% | 3% |
| enc256 | on | 64K | 7.75 / 11.7 | 88.3 / 116 | 63.5 / 86.7 (26%) | 40.4 / 89.2 (59%) | 63.9 / 92.2 | 15.3 / 23.1 | 77.5 / 94.7 | 3.0% | 41% |
| enc256 | on | 262K | 6.96 / 10.9 | 59.1 / 80.2 | 57.7 / 83.2 (8%) | 47.4 / 82.0 (27%) | 56.5 / 82.2 | 16.0 / 21.2 | 67.5 / 88.0 | 1.0% | 73% |
| enc256 | on | 1M | 6.86 / 9.78 | 56.1 / 87.8 | 55.5 / 76.0 (3%) | 53.2 / 80.2 (11%) | 55.6 / 94.3 | 20.9 / 26.8 | 66.3 / 91.7 | 0.4% | 89% |

† The first occupancy measured in a process. See finding 6.

**Where the uncached step goes** (ms per step, mean): coarse scan (top-K included), top-K alone, validation, token
rereads, re-encoding, the rest of the reread, fine reranking (its key projections included), readout, the local
block's encoding, and everything outside the laps (the core, about 5.6 ms, included).

| head | PLE | tokens | step | coarse scan | top-K | validation | token rereads | re-encoding | reread, rest | rerank | readout | local block | outside the laps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enc64 | off | 64K | 22.5 | 0.60 | 0.33 | 0.14 | 0.09 | 11.6 | 0.32 | 1.14 | 0.57 | 1.33 | 6.49 |
| enc64 | off | 1M | 22.1 | 2.30 | 0.57 | 0.15 | 0.10 | 9.06 | 0.29 | 1.07 | 0.51 | 1.39 | 7.09 |
| enc64 | on | 64K | 18.7 | 0.55 | 0.28 | 0.12 | 0.08 | 8.83 | 0.27 | 0.98 | 0.48 | 1.15 | 6.12 |
| enc64 | on | 1M | 20.9 | 2.13 | 0.51 | 0.12 | 0.10 | 8.34 | 0.27 | 1.01 | 0.48 | 1.20 | 7.09 |
| enc256 | off | 64K | 66.4 | 0.64 | 0.37 | 0.14 | 0.10 | 50.2 | 0.65 | 3.44 | 0.82 | 2.41 | 7.75 |
| enc256 | off | 1M | 61.5 | 2.39 | 0.60 | 0.14 | 0.10 | 41.2 | 0.66 | 3.14 | 0.75 | 2.35 | 10.4 |
| enc256 | on | 64K | 86.9 | 0.72 | 0.42 | 0.16 | 0.11 | 67.9 | 0.73 | 4.21 | 1.09 | 2.91 | 8.80 |
| enc256 | on | 1M | 60.7 | 2.46 | 0.61 | 0.14 | 0.11 | 41.5 | 0.69 | 3.52 | 0.83 | 2.40 | 8.72 |

The top-K is also inside the coarse scan and rerank laps. The validation, token rereads, re-encoding and rest of
the reread are inside the `exact_reread` lap. The 4K and 262K rows are in the records.

**Prefill of 512 tokens, ms p50** (cold chunk, warm chunk; blocks re-encoded per chunk in brackets):

| head | PLE | tokens | R5 off, every position | uncached, every position | cache 512, every position | R5 off, last | uncached, last | cache 512, last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| enc64 | off | 4K | 44.5, 43.0 | 389, 266 (512, 576) | 180, 115 (128, 16) | 52.6 | 87.1 | 85.3 |
| enc64 | off | 64K | 61.4, 57.2 | 4,337, 4,447 (7,228, 7,316) | 3,511, 2,952 (5,827, 5,453) | 79.1 | 74.4 | 77.6 |
| enc64 | off | 262K | 52.6, 47.9 | 9,828, 9,855 (15,912, 16,225) | 9,220, 9,743 (14,877, 14,839) | 57.2 | 61.2 | 61.3 |
| enc64 | off | 1M | 55.9, 50.4 | 15,302, 14,709 (29,048, 29,346) | 13,465, 13,676 (27,696, 27,937) | 45.1 | 71.3 | 67.1 |
| enc64 | on | 1M | 58.1, 43.0 | 12,612, 13,089 (29,835, 29,887) | 13,021, 12,923 (28,518, 28,534) | 44.6 | 55.1 | 56.5 |
| enc256 | off | 64K | 65.7, 47.3 | 33,089, 34,596 (18,876, 19,099) | 25,042, 22,890 (14,105, 14,067) | 56.4 | 90.5 | 93.2 |
| enc256 | off | 1M | 52.3, 48.2 | 56,056, 54,699 (30,125, 30,322) | 52,379, 51,528 (29,166, 29,352) | 48.1 | 90.4 | 86.5 |
| enc256 | on | 1M | 76.5, 64.2 | 61,141, 59,541 (30,816, 30,809) | 56,334, 58,829 (30,011, 29,957) | 71.5 | 112 | 122 |

The other rows are in the records.

**Findings.**

1. **R5 off costs 5.5-7.8 ms per decoded token.** That is the core with experts and the PLE.
2. **Re-encoding the candidates is the read's largest stage.** From 64K on it takes 8.2-11.6 ms per step for
   `enc64` (40-52% of the uncached step) and 41-68 ms for `enc256` (67-78%). The other stages:
   - The coarse scan grows with occupancy, to 2.1-2.5 ms at 1M, plus 0.5-0.6 ms of top-K.
   - Validation and token rereads together stay under 0.3 ms: exact source identity is cheap.
   - The local block's encoding is 1.1-2.9 ms.
   - Fine reranking, with the projections of every candidate's keys, is about 1-4.2 ms.
   - The readout is 0.5-1.2 ms.
3. **The R6 cache helps only where candidates repeat, and at long occupancies they rarely do.**
   - Successive steps share 49-52% of their candidates at 4K (128 blocks exist), 3-5% at 64K, 1-2% at 262K and
     0.4-1.3% at 1M.
   - At 1M, 82-89% of the blocks requested in 64 steps had not been requested before.
   - At 1M a 4,096-block cache hits 11-18% and saves at most about 1 ms (`enc64`) or 3-5 ms (`enc256`) per step. A
     512-block cache hits 3-6% and saves nothing measurable.
   - The 1 GiB target leaves room for about 500-1,200 `enc256` blocks beside a 512-token prefill reading at every
     position (section 2).
   - Through `ContextSession(cache_bytes=...)` the cache measures within noise of the same cache attached directly.
4. **The warm bound is not reachable by warming.** Replaying the same 64 steps, with a cache already holding their
   candidates, gives 10-21 ms per step: 12.4-13.2 ms (`enc64`) and 17.7-20.9 ms (`enc256`) at 1M. That is what
   perfect reuse would buy. It is UNREALISTIC WARMING: a stream does not repeat its queries.
   - The novel pass, with the cache emptied before every step, ran in the same second pass as the replay. It is no
     faster than uncached.
5. **Prefill reading at every position is dominated by re-encoding.** A 512-token chunk costs:
   - 0.14-15 s with `enc64` and 2.6-61 s with `enc256`, against 41-77 ms with R5 off;
   - with 7,000-31,000 blocks re-encoded per chunk from 64K on.

   The 512-block cache (4.1 MiB for `enc64`, 16.5 MiB for `enc256`) saves 34-91% at 4K, 19-34% at 64K, and
   between -3% and 12% from 262K on. Reading only at the last position costs between -5 and +83 ms relative to
   R5 off; 2 runs each, noisy.
6. **Measurement notes.**
   - The first occupancy measured in a process was slow for every configuration with R5 on. With `enc64` and the
     PLE off at 4K, re-encoding ran at 0.34 ms per block, against 0.16 ms later in the same process. Its uncached
     step was 29.9 ms, against 18.8 ms at 4K in the later PLE-on pass. The `enc256` 4K rows were each first in
     their process too (marked †). P4 adds a warm-up pass first.
   - Absolute levels drift between processes: R5 off measured 5.5-6.3 ms in two processes and 6.0-7.8 ms in a
     third. Comparisons are made only within a process.
   - The records' `source_revision` was read at each write. The `enc256` records name later commits, which differ
     from the runner commit `d3e5662` only by result files; their `+dirty` is the record file itself. The P4
     runner reads the revision once, at start, and ignores the records directory.

**P0 diagnosis.** The existing cache is not sufficient. Re-encoding remains the dominant cost at every occupancy
from 64K on, and most of it is FIRST-TIME re-encoding of blocks whose encodings were computed once already, when
they completed. Hence P1.

## 2. P1 — query-independent representations

Record: `results/r8/represent.json` (`run_r8 represent`: formulas over the configurations; nothing is allocated).
EXACT byte counts per stored block, and per stream at 1,048,576 tokens (32,768 blocks), float32:

| Tensor | `enc64` per block | `enc64` at 1M | `enc256` per block | `enc256` at 1M | depends on the query? | computable at block completion? |
|---|---:|---:|---:|---:|---|---|
| coarse descriptors `[4, 64]` (stored by R5) | 1 KiB | 32 MiB | 1 KiB | 32 MiB | no | yes (R5 does) |
| encoding `[33, w]` | 8.25 KiB | 264 MiB | 33 KiB | 1,056 MiB | no | yes (R5 computes it, then drops it) |
| fine keys `[32, f]` | 4 KiB | 128 MiB | 8 KiB | 256 MiB | no | yes |
| read keys and values `[2, 32, r]` | 8 KiB | 256 MiB | 32 KiB | 1,024 MiB | no | yes |
| coarse, fine and read queries; scores | — | — | — | — | yes | no |

bfloat16 halves every figure but is not exact. It is not used.

The 1 GiB per-stream target at 1M (`r6_1b`, bfloat16 index on the accelerator; `r6_500m`'s totals are within
7 MiB of these):

| Read path, query tile | state | decode: scratch / total | 512-token prefill reading at every position: scratch / total | room left beside that prefill |
|---|---:|---:|---:|---:|
| R5 reference, tile 32 (R6's setting) | 61.5 MiB | 53.2 MiB / 0.11 GiB | 929.4 MiB / 0.97 GiB | 33 MiB: 1,028 encoded blocks of 32,768 |
| R5 reference, tile 16 | 61.5 MiB | 53.2 MiB / 0.11 GiB | 457.5 MiB / 0.51 GiB | 505 MiB: 15,671 encoded blocks |
| R8 stored fine keys, tile 32 | 317.5 MiB | 35.6 MiB / 0.34 GiB | 192.5 MiB / 0.50 GiB | 514 MiB |
| R8 stored fine keys, tile 16 | 317.5 MiB | 35.6 MiB / 0.34 GiB | 134.1 MiB / 0.44 GiB | 572 MiB |

Findings:

- Every block-side tensor of the read is query-independent and exists, at block completion, in the call that
  completes the block. R5 keeps only the descriptors.
- Storing every encoding does not fit beside a reading prefill: 1,056 MiB for `enc256` (DESIGN_R8 section 3).
- The fine keys are 4.1x (`enc256`) or 2.1x (`enc64`) smaller than the encodings. They are what EVERY candidate
  needs. Read keys and values are needed only for the 4 injected blocks.
- With the fine keys stored, re-encoding is needed only for injected blocks. The worst-case scratch then falls
  more than the tier adds.
- Validity rules (completion equals reread, wraparound, precision, invalidation): `DESIGN_R8.md` section 2. They are
  tested in section 4 below.

**P1: PASS.** The fine keys are the one representation that is query-independent, computable at completion, exact
in the model's dtype, and within the budget at 1M in both read modes.

## 3. P2 — the acceleration built

One acceleration, as allowed: **stored fine keys, with the exact reread kept only for injected blocks**
(`EvidenceConfig.fine_keys = "stored"`; `DESIGN_R8.md` sections 3-4).

- At block completion, the encoding that produces the block's descriptors also produces its fine keys. They are
  written by the ring's `write`, into the same slots.
- A read reranks its candidates from the stored keys, after the same validation of every (candidate, query) pair.
  It then rereads and re-encodes only the blocks it injects, at most `evidence` per query.

The R5 path (`"reread"`) is the default and is unchanged: every new statement is under a `stored` branch or writes
the tier. The R6 cache, the CGC mathematics, the experts, the PLE, the coarse scan, the selection rules and the
readout are untouched.

Excluded by construction:

- persistent writes: the tier is volatile state;
- a second memory authority: the tier lives in the ring's slots under the ring's identity rules;
- unbounded growth: one allocation, `B x slots x block x fine_dim`;
- stale features: a fingerprint retires the tier;
- lost gradients: training never uses the tier.

## 4. P3 — qualification tests

`tests/research/elpis_torch/test_r8.py`: 32 tests, all PASS, both natively and with the AVX2 kernel paths forced.

The models are float64, with routed experts (grouped dispatch unless stated) and a 256-token ring. The streams
are 600-token R5 record streams, read at every position in calls of 70 tokens, so earlier calls' blocks come from
the ring and the ring wraps.

- "Identical" means bit for bit. That covers candidates, eligible counts, the exposed fine top-K, the selected
  blocks, the read block and position, and the local block.
- "Within ROUNDING" means within `1e-12`, the R6 bound for comparisons whose tensor shapes differ.

| Requirement (directive) | Test | Result |
|---|---|---|
| original vs accelerated outputs | batches 1, 2 and 3: identical identities; fine scores, null weights and logits within ROUNDING; fewer blocks re-encoded; tier reads counted; no stale rejection; the ring wrapped | PASS |
| PLE on / off; grouped vs reference dispatch | the same parity for all four combinations | PASS |
| every reranking variant | `rerank` coarse and all, `supersession` sum and off, a wider configuration (`coarse_top` 12, `evidence` 4) | PASS |
| revision-heavy streams and stale controls | 3-5 revisions per key and a stale question for every superseded bind. Identical coarse and fine ranks of the labelled gold, gold blocks and historical masks; logits within ROUNDING | PASS |
| oracle diagnosis | the forced gold is reread and injected; selections identical, logits within ROUNDING | PASS |
| deterministic candidate ids and rankings | two runs identical in logits and every exposed field | PASS |
| chunk vs incremental | the R5 reference in one 600-token call vs stored keys in a session: a 300-token prefill in 37-token chunks with probes at the first, middle and last positions of blocks, then 300 single-token steps; within `1e-9` (R6's bound) | PASS |
| same-token and cross-block queries | logits at chosen positions only (first, second, last of blocks, across call edges) equal the every-position run's within ROUNDING | PASS |
| future-token mutation | changing every token from position 301 on changes nothing before it. Identities identical under both dispatches. Logits, fine scores and null weights identical with the fixed-shape dispatch, within ROUNDING with the grouped one | PASS |
| a leakage perturbation must fail | a fine-key or read-key projection that adds `1e-7` of the mean over a call's blocks, later blocks included, moves the past far beyond `100 x ROUNDING` | PASS (detected) |
| an incorrect block must not be accepted | `read_fine` refuses evicted, edge, unwritten and never-existing blocks (4 refusals, counted). Keys served for the wrong blocks (rolled by one) change the selections or the logits by more than `1e6 x ROUNDING` | PASS (detected) |
| exact source identification; wraparound | every live block's stored keys equal the keys recomputed from that block's own token ids reread from the ring (within ROUNDING); the slot of an overwritten block holds its successor and the overwritten address is refused | PASS |
| reset, session isolation | two sessions hold distinct, different tiers. Reset zeros the tier, clears the recorded revision and starts epoch 1. A reset session equals a fresh one bit for bit | PASS |
| invalidation after revision; no stale features | an in-place encoder change mid-stream retires the tier (1 invalidation, no tier read afterwards); outputs equal the reference model's under the same change within ROUNDING. Reset restores a valid tier. NEGATIVE CONTROL: with a blind fingerprint the stale keys move the logits by more than `1e6 x ROUNDING`. A session refuses the changed model (`ModelRevisionError`) | PASS |
| preserved gradients in training | with gradients the tier is never read (bypass counted). Losses and every gradient, the fine projections' included, equal the reference model's bit for bit | PASS |
| with the R6 cache | stored keys with a 32 KiB cache equal stored keys without one (within ROUNDING), within the bound, nothing below the live window, fewer blocks re-encoded | PASS |
| accounting | `fixed_bytes == nbytes` with the tier at capacities 256 and 4,096; `index_fine_keys = B x slots x 16 x 8 x 8 B`; projected state equals held state in a session | PASS |
| 1M accounting | `r6_500m` and `r6_1b` at 1M with stored keys: the tier is exactly 256 MiB; state plus decode scratch and state plus 512-token-prefill scratch are both within 1 GiB; the state grows and the prefill scratch shrinks against R5 | PASS |
| configuration, checkpoints | an unknown value is refused; the default is `"reread"`; an evaluation override changes no weight; `lm_training` records carry the option; a record without it builds the reference | PASS |
| ring contract | a tiered ring refuses a write without keys or with the wrong width; a plain ring refuses keys and `read_fine`; reset zeros and invalidates | PASS |
| zero-write | in a separate interpreter with an audit hook, a stored-keys stream to 3.5 ring capacities with reads, a reset and a fault records no write event; the deliberate-write control is detected | PASS |

The full research suite: 507 tests, the 475 of the R7 head plus these 32. Locally 502 passed and 5 skipped, 0
failed, natively and with the AVX2 paths forced. The skips are 3 native-library tests (library not built here) and
2 CUDA tests (NOT_RUN). **P3: PASS.**

## 5. P4 — admission

<!-- P4 -->

## 6. Limits

<!-- LIMITS -->

## 7. Regression, CI, reproduction

<!-- CI -->
