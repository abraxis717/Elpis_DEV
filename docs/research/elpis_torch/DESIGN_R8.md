# Elpis Torch R8: the R5 evidence read with stored fine keys

Research only (`research/elpis_torch`). Status: ADMISSION CANDIDATE, opt-in. The R5 reference read stays the
default and the yardstick. Measurements and qualification: `EVIDENCE_R8.md`. Code: `evidence_ring.py` (the tier),
`evidence_model.py` (`EvidenceConfig.fine_keys`, the read path, accounting), `experiments/run_r8.py` (P0 baseline,
P1 representations, P4 admission). Tests: `tests/research/elpis_torch/test_r8.py`.

Question (the owner's): can redundant historical encoding work be removed by reusing bounded, validated,
query-independent representations, without giving up causality, source identity, numerical integrity or the
existing context-memory budget?

Answer in one line: yes for the fine reranker's keys. Each block's fine keys are computed once, when the block
completes, from the encoding its coarse descriptors already come from, and kept beside the descriptors in the ring's
index. Candidates are reranked from them, and only the blocks the working context injects are reread from their
exact token ids and re-encoded. Per query the encoder then runs on at most `evidence` blocks instead of
`coarse_top`: 4 instead of 64 in every configuration here.

## 1. Where the read's time goes (P0, measured in `EVIDENCE_R8.md` section 1)

The R5 read per query:

1. A coarse scan of every live block's stored descriptors, then a deterministic top-64.
2. Validation of every (candidate, query) address, then the exact reread of every candidate's token ids, with 8
   tokens of left context.
3. Re-encoding of every candidate (the causal block encoder: 40 tokens, 2 layers).
4. Fine keys of every candidate, then token-level matching, supersession and the top-4.
5. Read keys and values of every candidate, then the readout of the 4 selected blocks plus the local block.

Step 3 dominates. A bounded cache does not shrink it when successive queries ask for new blocks, and at long
occupancies they mostly do. With an untrained model:

- successive decode steps share at most about 5% of their candidates from 65,536 tokens on, and about 1% at 1M;
- over 64 steps, more than 80% of the blocks requested at 1M had not been requested before;
- candidate ages spread uniformly over the history.

A trained model's queries may repeat more. That is not measured here.

## 2. The query-independent tensors (P1)

Everything on the block side of the read depends only on the block's own exact token ids and the weights. Only the
query projections and the scores depend on the query.

Per stored block (block 32, overlap 8), with float32 bytes and per stream at 1,048,576 tokens (32,768 blocks):

| Tensor | Shape | `enc64` (R5 default) | `enc256` (R6 scale head) | Used by |
|---|---|---|---|---|
| coarse descriptors | `[4, 64]` | 1 KiB; 32 MiB | 1 KiB; 32 MiB | the coarse scan (already stored at completion) |
| encoding | `[33, w]` | 8.25 KiB; 264 MiB | 33 KiB; 1,056 MiB | everything below |
| fine keys | `[32, f]` | 4 KiB; 128 MiB | 8 KiB; 256 MiB | the fine reranker, for EVERY candidate |
| read keys and values | `[2, 32, r]` | 8 KiB; 256 MiB | 32 KiB; 1,024 MiB | the readout, for the SELECTED blocks only |

Here `w`, `f` and `r` are 64, 32 and 32 for `enc64`, and 256, 64 and 128 for `enc256`. In bfloat16 every figure
halves, but a bfloat16 copy would not be exact.

**Computed at block completion?** Yes. A completed block's descriptors are computed from its encoding in the call
that completes it (`_chunk_context` encodes every touched block). The fine keys are one more projection of that
encoding.

**The same values as a later reread?** Yes, mathematically. The ring declares a block live only while its tokens
and its left context are still held. So the window and validity mask that encode a block at completion are those a
later reread rebuilds; positions before the stream origin are invalid in both. In floating point, the completion
encoding runs in a batch of the chunk's touched blocks and the reread in a batch of the call's candidates. BLAS and
vectorized reductions may round rows differently for different batch sizes (`EVIDENCE_R6` section 10), so equality
holds to rounding. The R6 cache has the same property.

**Wraparound?** A tier in the ring's own slots is overwritten when the slot is reused, together with the
descriptors. The ring's identity check, by slot generation and liveness, refuses any stale address before the read.

**Weight updates?** Stored keys are valid only for the weights they were computed from. A fingerprint of every
parameter they depend on decides that: embeddings, the encoder, and `fine_k`, `fine_prev` and `fine_position`.

**Budget?** Section 5. A tier of all fine keys fits within the 1 GiB per-stream target at 1M in both read modes.
Its state cost is offset in the worst case: the 512-token prefill's scratch shrinks by more than the tier adds.

## 3. Options weighed. One is built.

| Option | Removes | Costs | Verdict |
|---|---|---|---|
| a larger R6 cache | repeated candidates within its bound | 33 KiB per block (`enc256`). At 1M the 1 GiB target leaves 17-39 MiB beside a 512-token prefill reading at every position (query tile 32): 532-1,200 blocks of 32,768 | P0: it helps only repeats, and at 1M it barely helps |
| completion-time admission into the cache | first reads of RECENT blocks | none beyond the bound | P0: candidate ages are uniform, so few would hit. Not built |
| every encoding, stored | all re-encoding | 1,056 MiB (`enc256`) or 264 MiB (`enc64`) per stream at 1M; 528 MiB in bfloat16, not exact | over budget (`enc256`). Not built |
| read keys and values, stored | the readout's projections | larger than the encodings | Not built |
| **fine keys, stored; reread only what is injected** | re-encoding of every candidate that is not injected (60 of 64) | 256 MiB (`enc256`) or 128 MiB (`enc64`) per stream at 1M, float32 | **built**, opt-in |
| bfloat16 / AMX arithmetic, quantized storage | constant factors | not exact | Not built (the directive: an approximation is not exact) |

## 4. The mechanism

```
completion of block b (any call):   tokens ─► encode ─┬─► describe ─► ring.keys[slot(b)]       (R5)
                                                      └─► fine_keys ─► ring.fine[slot(b)]       (R8, same write)
query at p:   coarse scan (descriptors) ─► top-64 candidates ─► validate (block, query)
              ─► stored fine keys of the candidates ─► fine rerank (unchanged arithmetic) ─► top-4
              ─► EXACT reread + re-encode of the injected blocks only ─► read keys / values ─► readout
```

Configuration: `EvidenceConfig.fine_keys`, `"reread"` (default, the R5 reference) or `"stored"`. It is an
execution option: no parameter changes meaning or shape. It is in `EVALUATION_OVERRIDES`, carried by checkpoints,
and an R5 / R6 / R7 record without it builds the reference.

| Property | Rule (tested in `test_r8.py`) |
|---|---|
| authority | The ring's token ids remain the only authority. The tier is derived from them. It lives in the ring's slots, is written by the ring's own `write` with the descriptors, is zeroed by its `reset` and is read only through its `_check`. Every injected block is reread from its tokens and re-encoded. |
| identity | `read_fine` refuses evicted, overwritten, incomplete, never-written and reset addresses (`StaleAddressError`, counted), like `reread`. Candidate pairs are validated against their queries first, as in R5. |
| wraparound | The tier slot of block `b` is overwritten when block `b + slots` is written. The overwritten address is refused. |
| sessions, reset | One tier per state. `reset` zeros it and starts a new epoch. Two sessions never share one. A reset session equals a fresh one bit for bit. |
| invalidation | The first tier write after a reset records the fingerprint. Any later read or write under other weights makes the tier stale: the reference path runs until reset (`fine_tier_invalidations`). A `ContextSession` refuses a changed model anyway (R4.1). |
| gradients | Never used with gradients enabled: training runs the reference path (`fine_tier_bypassed`), and its losses and gradients equal the reference model's bit for bit. |
| precision | The tier is in the model's dtype (float32 here, float64 in the tests). No cast, so no approximation. |
| numerical equivalence | Candidates, rankings, selected and read blocks identical; scores and logits within rounding (float64: `1e-12`). |
| causality | A query reads only blocks completed before its own block. The tier is written after a call's tokens are all known and read only by later calls. Bit for bit with the fixed-shape expert dispatch. |
| R6 cache | Still used, now only for the injected blocks' encodings. |
| PLE, experts | Unchanged. The tier depends on the embeddings and the evidence encoder only. |
| accounting | `index_fine_keys = B x slots x block x fine_dim x dtype bytes`, EXACT in `nbytes` and `fixed_bytes` and placed with the ring (`history_device`). The scratch projection re-encodes `evidence` instead of `coarse_top` blocks per query and adds the gathered keys. |

The reference path's code is unchanged when `fine_keys == "reread"`. The new statements are under `stored` branches
in `_retrieve_tile`, plus the tier write in `_commit_context`.

## 5. Cost model (PROJECTED; measured in `EVIDENCE_R8.md`)

Per query at 1M (block 32, `coarse_top` 64, `evidence` 4):

| | R5 reference | R8 stored keys |
|---|---|---|
| blocks reread (token ids) | 64 x 40 tokens | 4 x 40 tokens |
| blocks re-encoded | 64 (minus R6 cache hits) | 4 (minus R6 cache hits) |
| fine keys | recomputed for 64 blocks | 64 blocks gathered: 512 KiB (`enc256`) |
| at block completion | descriptors | descriptors + fine keys (one `w x f` projection of 32 rows, twice) |

Context memory per stream at 1M, `r6_1b` with a bfloat16 index (EXACT state, PROJECTED scratch, query tile 32;
`run_r8 represent`):

| | state | decode scratch | total, decode | 512-token prefill reading at every position: scratch | total, prefill |
|---|---:|---:|---:|---:|---:|
| R5 reference | 61.5 MiB | 53.2 MiB | 0.11 GiB | 929.4 MiB | 0.97 GiB |
| R8 stored keys | 317.5 MiB | 35.6 MiB | 0.34 GiB | 192.5 MiB | 0.50 GiB |

`r6_500m`'s totals are within 7 MiB of these. The tier costs 256 MiB of state. Re-encoding 4 blocks per query
instead of 64 saves 737 MiB of prefill scratch. The worst case therefore falls from 0.97 to 0.50 GiB, and the
decode-only footprint rises from 0.11 to 0.34 GiB.

## 6. Unchanged, and not built

- The R5 mathematics: coarse score, fine score, supersession, selection rule, readout and injection. The CGC
  experts, the PLE, the ERSA core.
- No persistent write: the tier is volatile state, like the ring.
- No expert paging. No new retrieval oracle. No agentic control or stopping policy (Hermes owns agentic learning).
- No unbounded growth: one fixed allocation, sized by the capacity.
- Nothing under `src/` or `native/`.
- No CUDA kernel, Triton, dependency or download. CUDA timing: NOT_RUN.
