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

<!-- P0_DECODE -->

**Where the uncached step goes** (ms per step, mean): coarse scan (top-K included), top-K alone, validation, token
rereads, re-encoding, the rest of the reread, fine reranking (its key projections included), readout, the local
block's encoding, and everything outside the laps (the core, about 5.6 ms, included).

<!-- P0_STAGES -->

**Prefill of 512 tokens, ms p50** (cold chunk, warm chunk; blocks re-encoded per chunk in brackets):

<!-- P0_PREFILL -->

**Findings.**

<!-- P0_FINDINGS -->

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

<!-- P3 -->

## 5. P4 — admission

<!-- P4 -->

## 6. Limits

<!-- LIMITS -->

## 7. Regression, CI, reproduction

<!-- CI -->
