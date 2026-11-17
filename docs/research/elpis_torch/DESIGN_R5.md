# Elpis Torch R5: learned coarse-to-fine evidence retrieval over a 1,048,576-token exact history

Research only. Code: `research/elpis_torch/evidence_ring.py` (exact history with identity),
`evidence_model.py` (the retrieval path beside the unchanged core), `evidence_tasks.py` (tasks and labels),
`experiments/run_r5.py` (training, evaluation, controls). Evidence: `EVIDENCE_R5.md`. Baseline audit:
`BASELINE_R5.md`.

## 1. The question and its decomposition

Can the model find the right historical evidence in a bounded 1,048,576-token context, reread it exactly, and
answer from it, while short-context behaviour is preserved? R5 measures the four failure sources separately:

| Stage | What can fail | Measured by |
|---|---|---|
| addressing | the evidence block is not among the coarse candidates | coarse recall@16/32/64/128, source rank, MRR |
| evidence (rerank) | it is a candidate but not selected | fine recall@1/4/8, fine rank, stale / wrong-scope top-1 rate |
| readout and reasoning | it is selected but the answer is wrong | conditional accuracy (gold selected), oracle-source accuracy |
| generation | the answer is right teacher-forced but not free-running | not measured in R5 ("egress only later") |

Failure localization: poor recall@64 means addressing is broken; good recall@64 with poor recall@4/8 means the
reranker is broken; good evidence recall with poor conditional accuracy means readout or reasoning is broken.

## 2. Topology

```
tokens ──► ERSA/CGC core (unchanged, ersa_model.ErsaLM) ──► last block: u = RMSNorm(x)
                │                                                   │ query state
                │ every completed 32-token block                    ▼
                ├──► causal block encoder ──► 4 descriptors ──► coarse index (ring of 32,768 records)
                │                                                   │ scan of all live blocks, top-64
                ▼                                                   ▼
          exact token ring (2^20, int32) ──── exact reread of the 64 candidates (validated addresses)
                                                                    │ re-encoded
                                                                    ▼
                                         fine reranker: token-level match + supersession + coarse score
                                                                    │ top-4 (top-8 exposed)
                                                                    ▼
             working context = 4 evidence blocks (by position) + the query's own block (tokens <= query)
                                                                    │ readout attention with a null option
                                                                    ▼
                                   added to the last block's mixer output ──► channel ──► logits
```

**Exact history (`EvidenceRing`).** A FIFO ring of exactly `capacity = 2^20` int32 token ids per stream, and an
index ring of `slots = capacity / block = 32,768` records (`per_block x key_dim` descriptors, absolute block id per
slot). Every address carries `(session, epoch, block, slot = b mod slots, generation = b // slots)`. `reset`
increments the epoch; every ring object has a unique session. A reread validates each (block, query) pair: the
slot must still carry the block, the block must be complete at the query, and its whole span (its 32 tokens and
the encoder's 8 tokens of left context) must still be held. Anything else raises `StaleAddressError` and is
counted; nothing is read from a reused slot. Requiring the left context makes a reread bit-for-bit the encoding
the descriptor was computed from, whatever the chunking; it costs one block at the ring's edge (32,768 live
blocks exactly when the stream fills the ring, 32,766-32,767 in steady state).

**Coarse index.** When a block completes, a small causal block encoder (the core's token embeddings plus a
zero-initialized learned correction, three learned predecessor maps, a learned position, two causal attention +
CGC layers) encodes it from its exact tokens, and four softmax poolings over its tokens give four unit
descriptors of 64 dimensions. They are stored once (detached; `key_dtype` "auto" = bfloat16 on CUDA, float32 on
CPU). The coarse query is `normalize(W u)` at the query position; a block's coarse score is
`exp(scale) * max_j <q, d_j> - softplus(r) * log(1 + distance)`. One vectorized product scans every live
block; the top 64 are kept by R4's deterministic rule (score descending, ties to the newer block). Eligible:
blocks complete before the query's own block, with their span held. No oracle input exists on this path.

**Exact reread and fine reranker.** The 64 candidates' original token ids are reread from the ring and
re-encoded (no per-token key/value cache exists anywhere). The fine score of a candidate is

```
fine = m + a * coarse - softplus(g) * sum_{newer candidates j} sigmoid(k (m_j - t))
m    = log-mean-exp over the block's tokens of <W_f u, F h_t + P h_{t-1} + pos_t>
```

The last term is the learned supersession feature: a candidate is penalized by how strongly NEWER candidates
(and the query's own block) also match the query. It is what lets a latest-binding rule be learned from relative
order, without absolute distance. The top `evidence = 4` are injected; the top `fine_top = 8` are exposed with
absolute ranges, scores and (when labels are passed for scoring) hit/miss.

**Bounded deterministic injection.** The working context of the read is the 4 selected blocks ordered by
absolute position, plus the query's own block with its exact tokens up to the query (recent local context and
partially filled tail blocks; the causal encoder makes these states equal to the complete block's on the shared
prefix). The query attends over these `5 x 32` tokens and a null slot, with predecessor keys, a recency-rank
embedding (newest 0; no absolute distance, so the readout is length-invariant) and the fine scores as a
log-softmax selection prior. The output is added to the last block's mixer output.

**Placement.** The read sits in the last block, after that block's ERSA update, so it changes only the logits of
its own position: no recurrent state depends on it. Computing it only where logits are requested
(`logit_positions`) is therefore exactly the dense computation restricted to those positions; this is tested.
Evaluation computes it at the probe positions; the dense mode computes it everywhere.

## 3. Modes, labels and losses

`learned` (default, autonomous), `off` (core only), `oracle` (DIAGNOSIS: the gold block is forced into the
selected evidence). Reranking ablations: `coarse` (the coarse order picks the 4 blocks; reread kept) and `all`
(all 64 coarse candidates are reread and injected; no reranker). Labels (`evidence`, `stale`) are accepted for
scoring and training; in `learned` mode they cannot change an output (tested), `teacher` insertion of the gold
into the candidates is refused without gradients, and `oracle` refuses to run without labels.

Losses (training only): `coarse_source_loss` = cross-entropy of the gold block over every eligible block
(superseded evidence of the same query is excluded from the negatives: ordering revisions is the reranker's
job); `fine_evidence_loss` = cross-entropy of the gold over the candidates (teacher insertion when the coarse
stage missed it); the answer loss = cross-entropy at probe positions plus 0.05 x every position. Candidate
descriptors are recomputed with gradient (replay) so hard negatives train the index; other stored descriptors
are detached.

Training: Phase A, 2,000 steps of 16 x 1,024-token mixtures (records, scoped symbols, R3 facts), every
parameter. Phase B, the curriculum 4K -> 16K -> 64K -> 262K -> 1M with the core frozen (only the evidence path
learns), each long stream processed chunk by chunk with an immediate backward per chunk (exact: no graph crosses a
chunk), and a 1,024-token rehearsal batch per step. Fact density, revision count and hard-negative rates are
drawn independently per batch.

## 4. Tasks (`evidence_tasks.py`)

The vocabulary extends R3's facts vocabulary (R3 streams are valid inputs). Families: **records** (`KEY_k VAL_v`
binds with revisions; questions `ASK KEY_k` or the associative `ASSOC_c`), **scoped** program symbols
(`DEF SCOPE_s SYM_x VAL_v`; question `GET SCOPE_s SYM_x`), **r3facts** (the R3 generator). Hard negatives,
generated explicitly and recorded per probe: superseded value / older definition, negated statement
(`NOT KEY_k VAL_x`, a verbatim bind pattern), reference / call (`USE KEY_k`, `CALL SCOPE_s SYM_x`), lexical overlap
(`USE KEY_k KEY_j VAL_x`), same value under another key or symbol, stale direct question (duplicate-like:
`ASK KEY_k VAL_old`), stale associative question (semantic: `ASSOC_c VAL_old`), wrong-scope declaration (placed
before or after the gold). Gold distances are log-uniform or uniform up to the stream length.

## 5. Memory and compute (per stream, default geometry)

| Category | Bytes | Note |
|---|---:|---|
| exact token ring | 4,194,304 | 4 B/token |
| index descriptors | 33,554,432 (float32) / 16,777,216 (bfloat16) | 1,024 / 512 B per block |
| index ids | 262,144 | 8 B per slot |
| persistent retrieval state | 38.0 MB (float32) / 21.2 MB (bfloat16) | 36.25 / 20.25 B per token |
| core working state | 9,728 | ERSA matrices and convolution tail |
| weights | 1.07 MB float32 (266,563 parameters, 149,099 in the evidence path) | |

Per query: one scan of 32,768 x 4 x 64 multiply-adds (8.4M), 64 block re-encodings (span 40, width 64), and the
fine and readout products over 65 x 32 tokens. Transient scratch is bounded per query tile (`tile_elements`).

## 6. Not built (by directive)

No DeepSeek-V4.1-Flash or Spark-X2.6 reproduction; no VGGT-Omega code, weights or dependencies; no importance
eviction (FIFO only); no RALPH or autonomous loop; multi-hop retrieval only after single-hop qualifies (see
`EVIDENCE_R5.md`); no Triton kernels; no change to the canonical runtime; no runtime storage
writes (the isolation and zero-write tests cover the new modules).
