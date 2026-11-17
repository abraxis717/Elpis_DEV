# Elpis Torch R3: long-context addressing

Status: RESEARCH_ONLY, NO_RUNTIME_AUTHORITY, NO_LANGUAGE_CLAIM. Builds on R0 (`DESIGN.md`), R1 (`DESIGN_R1.md`) and
R2 (`EVIDENCE_R2.md`), which are preserved. Code: `research/elpis_torch/history.py`, `history_model.py`,
`longrange_tasks.py`, `program_tasks.py`, `scaling.py`, `experiments/run_r3.py`; changes to `ersa.py`,
`ersa_model.py`, `scroll.py`, `tokenizer.py`. Tests: `tests/research/elpis_torch/test_r3.py`. Results:
`EVIDENCE_R3.md`. Nothing in `src/`, `native/`, the mission or `ELPIS_SYSTEM.json` is touched.

## 1. The question and the three functions

A bounded recurrent state cannot hold unlimited independent facts (R2: recall of random associations is reliable
only to about a quarter to a half of `key_dim` per head; retrieval decayed to 0.35 at 131K tokens). R3 asks
whether a compact **learned index** plus **selective rereading of the original tokens** lets the model use
information its state no longer holds, at small context-memory cost. Three functions are kept apart so that a
failure can be attributed:

| Function | Mechanism | Bounded by |
|---|---|---|
| working memory | ERSA (gated delta rule) + short causal convolution + CGC channel, optional window attention | fixed state per layer |
| addressing | a learned query from the model's hidden state scores one learned key set per completed block; top-`K` | `horizon / block` index records |
| readout | the selected blocks are REREAD from the exact token history, re-encoded, attended by the query | `K` blocks per query |

No host-side retriever: the host stores tokens and index records, evicts, and gathers what the model selected.

## 2. Topology

```
tokens -> embed -> [ block 0 ] -> ... -> [ block L-1 ] -> RMSNorm -> tied head -> logits
block l:   u = RMSNorm(x)
           x = x + ERSA_l(u) [+ WindowAttention_l(u) if l in attn_layers] [+ HistoryRead_l(u) if l in history.layers]
           x = x + Channel_l(RMSNorm(x))            Channel = CGC (FFN control available)
```

`HistoryRead_l` (one `HistoryReader` per read layer, one shared `BlockEncoder`):

1. **Index (bounded).** Every completed block of `s` tokens is encoded once (`BlockEncoder` over the block and
   `overlap` tokens before it: shared embeddings, a learned span position, one bidirectional attention layer, one
   channel layer of the model's kind). `per_block` attention-pooled vectors are projected to unit `index_dim`
   keys and stored with the absolute block number in a ring of `horizon / s` records (`history.BlockIndex`).
2. **Addressing (learned).** `q = normalize(W_iq u_t)`; block score
   `max_j <q, k_{b,j}> * exp(index_scale) - softplus(rho) * log(1 + distance)`, over the eligible blocks; top-`K`.
3. **Rereading (exact).** Selected blocks that are already in the index are re-encoded from their original token
   ids (`history.TokenHistory`); blocks completed in the current chunk are encoded from the chunk.
4. **Readout.** Keys `W_k h_j + W_kp h_{j-1} + position_j`, values `W_v h_j`; logits get the block's recency rank
   embedding (newest first), a per-head slope on `log(1 + distance)`, and the selection prior
   `select_scale * log softmax(selected scores)`; a learned null logit lets the query read nothing; one softmax
   over `K s + 1` entries; output projected into the residual stream beside ERSA.

The predecessor term `W_kp h_{j-1}` is an R3 finding (`EVIDENCE_R3.md`, P0): without it the readout learned to
attend to the token that MATCHES the query (`KEY_k`) and read its content, which answers alias hops (whose
answer is a key) but never values. With it, matching what came before a token reads that token: the induction
pattern built into the readout's keys.

### Causality and eligibility

Block `b` (positions `[b s, (b + 1) s)`) is eligible for the query at `p` iff it is complete at `p`
(`(b + 1) s - 1 <= p`) and among the `horizon / s` most recent complete blocks of `p`. The encoder only ever
sees tokens at or before the block's end. Blocks completing inside a chunk are scored from the chunk itself,
so the computation does not depend on how the stream is chunked (tested: chunk 64, 13 and 1 give equal logits
and equal selections, past the horizon, with and without window attention).

### Source identity

A historical token enters only through the readout, never through ERSA, the window or the token stream. Its key
carries its position inside its block, its predecessor, its block's recency rank among the selection and the
query-relative log-distance (all length-invariant, so nothing has to extrapolate a trained position table).

## 3. Modes and labels

| Mode | Use | Labels in the model |
|---|---|---|
| `off` | recurrent only, same weights | none |
| `learned` | autonomous inference | none |
| `oracle` | DIAGNOSIS: the fixture's source block is read where given, nothing elsewhere | the ADDRESS, never the answer |
| `oracle_override` | DIAGNOSIS: the oracle address where given, the learned selection elsewhere | the address |

Training-only labels: the generator's source position of each probe (`FactStreams.query_block`). They feed the
optional addressing loss (`addressing_loss`: cross-entropy of the index scores over eligible blocks) of the
`history` architecture, and the address input of the `history_oracle` architecture. Evaluation in `learned`
mode passes none (tested: learned-mode logits are identical with and without the label tensor present in the
caller). Train, validation and test streams use disjoint seed domains (0.., 600,000,000.., 700,000,000..).

## 4. Gradients

* The language-model loss reaches the scores of the SELECTED blocks through the selection prior in the readout
  logits, and through them the query map, the index keys, the pooling and the encoder.
* A block that was not selected gets a gradient only from `addressing_loss` (training-only labels).
* Index records written to the ring are detached. Within one training chunk every block is "new" and fully
  differentiable; across chunks (truncated backpropagation, e.g. with a 128-token attention window) stored index
  keys receive no gradient, while reread blocks are re-encoded with the current weights and do.
* The hard top-`K` itself has no gradient. Nothing claims otherwise.

## 5. ERSA execution (P3)

`chunked_scan` evaluates the R2 recurrence `S_t = r_t S_{t-1} - r_t e_t (S_{t-1} k_t) k_t^T + w_t v_t k_t^T`
block-parallel. With `u_t = w_t v_t - r_t e_t S_{t-1} k_t` and `G[t, i] = r_{i+1} ... r_t`:

    (I + L) U = diag(w) V - diag(e g) K S_0^T,    L[t, i] = e_t G[t, i] (k_t . k_i)  (i < t)
    o_t = g_t S_0 q_t + sum_{i <= t} G[t, i] (k_i . q_t) u_i
    S_end = g_end S_0 + sum_i G[end, i] u_i k_i^T

`G` is a cumulative product of masked gates (no ratio of cumulative products), so an exact wipe (`r = 0`)
stays exact; `I + L` is unit lower triangular (forward substitution). It is the same recurrence, tested equal
to the serial reference to 1e-12 in float64 (values and gradients, with wipes) and to 1e-4 in float32. The
serial loop remains the default and the reference (`ersa_kernel="serial"`).

Measured on this CPU (4 threads, B=24, T=384, 4 heads, 16x16): forward + backward 0.79 s serial, 0.068 s chunked
(block 64). PERFORMANCE_ONLY.

Device and precision: every state tensor (ERSA cells, convolution tails, window rings, token history, block
index) is allocated on the model's parameter device (tested on PyTorch's `meta` device standing in for a second
device). Under autocast the recurrence runs with autocast disabled in the state's dtype (float32 or float64):
16-bit projections, 32-bit accumulation (tested with bfloat16 on CPU). CUDA is NOT_RUN here
(`run_r3 gpu` is the workstation qualification).

Last-block fast weights: the history read happens before the last block's channel, and index keys depend only
on token ids and the encoder, so a last-block CGC change invalidates no ERSA state, window ring or index record
(tested). Adapting the encoder or a reader would invalidate the index; no session adapts them. No `ErsaLM`
fast-weight session class is added in R3.

## 6. Configuration surface

`ErsaConfig`: `ersa_kernel`, `scan_block`, `attn_layers` (placement), `attn_heads`, `attn_kv_heads`,
`attn_window`, `chunk`. `HistoryConfig`: `block`, `horizon`, `topk`, `index_dim`, `per_block`, `read_heads`,
`read_kv_heads`, `read_head_dim`, `overlap`, `layers`, `encoder_heads`, `encoder_dim`, `mode`.

## 7. Memory and compute (per stream)

| Item | Size | Grows with stream |
|---|---|---|
| ERSA cells | `L * heads * key_dim * value_dim` (accumulation dtype) | no |
| window ring | `attn_layers * 2 * kv_heads * head_dim * window` | no |
| token history | `(horizon + block + overlap) * 4` bytes | no |
| block index | `read_layers * horizon/block * (per_block * index_dim * dtype + 8)` | no |
| per-chunk scratch | scores `chunk * horizon/block`; rereads `chunk * K * (block + 1) * encoder_dim` | no |

Per decoded token: weights + ERSA update + window + `read_layers * horizon/block * per_block * index_dim`
scoring (linear in the horizon; a coarse-to-fine index would make it sublinear and is NOT built: R3's gates
did not reach the point where it is justified) + rereading `K * span` tokens through the encoder. Rereading is
the dominant term at scale unless the encoder is narrow (`encoder_dim`) or encoded blocks are cached (not
built). `scaling.py` gives the numbers for the 500M and 1B configurations.

## 8. Not built

A hierarchical index, an encoded-block cache, a fast-weight session for `ErsaLM`, any controller, re-ranker,
supervisor or external retriever; persistence of any state; a subword tokenizer (only its interface,
`tokenizer.SubwordTokenizerProtocol`).
