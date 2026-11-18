# Elpis Torch R6: an expertized ERSA backbone with shared evidence access

Research only (`research/elpis_torch`); nothing in the canonical runtime changes. Code: `sparse_channel.py`
(routed FFN + CGC experts), `ersa_model.py` (`ErsaConfig.sparse`), `context_runtime.py` (the session interface),
`evidence_model.py` (R5 as a session family, reread cache, evaluation-only slot overrides), `lm_training.py`
(autoregressive training and checkpoints), `experiments/run_r6.py` (measurements). Evidence: `EVIDENCE_R6.md`.

Ownership: model-native agentic learning (action decisions, tool calls, observation-conditioned actions,
recovery, latent recurrence, adaptive computation and stopping, RALPH, policy-learning comparisons) belongs to
Hermes. R6 adds no action policy, tool executor, agent benchmark, reasoning controller or stopping rule; the
isolation tests keep those names out of the package.

## 1. The model, as implemented

```
token ids ──► embeddings (tied with the output head)
                │
   ┌────────────┴──────────────────────────────── block b (b = 0 .. L-1) ─────────────┐
   │  u = RMSNorm(x)                                                                  │
   │  m = ERSA(u)                      recurrent mixing; the only owner of volatile    │
   │      [+ window attention(u)]      state (ERSA matrices, conv tail, window ring)   │
   │      [+ R5 evidence read(u)]      LAST block only, after that block's ERSA update │
   │  x = x + m                                                                        │
   │  v = RMSNorm(x)                                                                   │
   │  x = x + channel(v)                                                               │
   │        dense:  FFN(v) | CGC(v) | FFN(v) + g CGC(v)                                │
   │        sparse (blocks in ErsaConfig.sparse.layers):                               │
   │           p = softmax(W_r v)  (float32 or wider) ; S = top-K of E (ties: low index)│
   │           shared(v) + sum_{i in S} w_i [FFN_i(v) + g_i CGC_i(v)]                  │
   └───────────────────────────────────────────────────────────────────────────────────┘
                │
        RMSNorm ──► tied head ──► logits (only at requested positions)
```

One R5 exact ring (2^20 tokens) and one coarse index per stream serve the whole model: no expert owns a history,
an index or a recurrent state. The evidence read stays where R5 qualified it (the last block, computed from that
block's input after its ERSA update), so it changes only its own position's logits and no recurrent state; the
routed channel of the same block then transforms `x + m` like any channel. Moving retrieval into more blocks is
not done: it has not been shown to pay for its cost.

## 2. Sparse FFN + CGC experts (`sparse_channel.py`)

* Each routed expert owns a SwiGLU, a CGC transformation (its own geometry, default the model's) and a learnable
  gate `g_i` (default 0.1, so CGC parameters receive gradient from the first step; 0.0 is an exact FFN control
  whose gate learns first). `kind="ffn"` gives FFN-only experts. Parameters are never aliased (tested).
* Router: a bias-free linear map of the hidden state, softmax in float32 (or the model's wider precision), top-K
  by a stable sort (ties to the lower expert index), selected probabilities renormalized to sum to 1 (optional).
  No labels or keywords enter routing.
* Dispatch: every token goes to exactly its K experts (no capacity limit, no dropping). The grouped path sorts the
  N x K assignments by expert once and runs each expert on its contiguous segment, scattering weighted outputs;
  the masked reference path (every expert on every token) exists only for tests.
* Per call, each sparse channel reports the load-balance loss `E sum_e f_e P_e`, the router z-loss, their weighted
  sum (the configured `balance_weight`, `z_weight`; zero by default), and detached statistics: counts,
  frequencies, imbalance (max/mean, CV), router entropy, top-1 probability, co-activation, inactive fraction and
  active multiply-adds per token. The report is overwritten each call (no growth).
* Placement: `ErsaConfig.sparse.layers` names the blocks whose dense channel is replaced; others keep the dense
  channel. `sparse=None` is exactly the old model (same parameters, same logits; R0-R5 checkpoints load).
* Equivalence controls: `upcycle()` builds a sparse copy of a dense model with the same function: K cloned experts
  with renormalized weights (exact algebraically; within rounding numerically) or a shared copy with zero routed
  outputs (bitwise); `routed_enabled=False` disables every routed contribution.

## 3. One session interface for every family (`context_runtime.py`)

`ContextSession` asks the model for `context_interface()`: state construction with origin, placement and
options; the cache it carries; counters; EXACT fixed bytes; a PROJECTED scratch estimate. The core default refuses
options that do not apply; `HistoryInterface` is R3/R4 unchanged; `EvidenceInterface` builds the R5 state (ring
and index together on `history_device`), refuses the R4 scorer options, reports ring and retrieval counters, and
offers a bounded reread-encoding cache that is used only for blocks just validated live for their queries,
purged below the live window, cleared on reset and invalidated by any embedding or encoder change.

## 4. Autoregressive training (`lm_training.py`)

Documents never share a training row (`BOS doc EOS`, padded, padding targets ignored); rows are fed in chunks
through the explicit state with full or truncated backpropagation; the loss is normalized over valid targets in
at least float32, plus the routers' configured auxiliary losses; non-finite losses are refused; checkpoints
record architecture (expert topology and evidence configuration included), tokenizer identity, parameter digest,
step, seed and optionally the optimizer, never live state. With R5 enabled and no labels the coarse index and the
hard selection receive no direct next-token gradient; label-free addressing is not claimed.

## 5. Scaling configurations (`experiments/run_r6.py`)

| Name | Widths | Experts | Window attention | Evidence |
|---|---|---|---|---|
| `r6_500m` | d 768, 20 blocks, vocabulary 65,536 | 8 routed, top-2, in blocks 1, 3, ..., 19 | 4,096 tokens in blocks 3, 7, ..., 19 | R5 defaults, encoder width 256, 32-query tiles |
| `r6_1b` | d 1,024, 18 blocks | 16 routed, top-2, plus a shared expert | 4,096 tokens in blocks 3, 7, 11, 15 | same |

Counts are instantiated on the `meta` device; total, resident and active parameters are reported separately
(resident = total: every expert must be in memory to serve an arbitrary route). Context memory is EXACT for the
state tensors and PROJECTED for scratch; the R5 coarse scan is linear in the live blocks per query. The query
tile is an execution option (it partitions independent queries and changes no logit); 32 instead of R5's 128
keeps a prefill that reads at every position within the 1 GiB target at every reported length.

## 6. Not built

No action policy, tool executor, agent benchmark, reasoning controller, adaptive stopping or RALPH (Hermes); no
per-expert history or index; no retrieval in every block; no approximate search; no custom kernels; no
importance eviction; no change to the mission or the canonical runtime.
