# Elpis Torch R7: parametric lookup embeddings (PLE), an admission candidate

Research only (`research/elpis_torch`). Status: ADMISSION CANDIDATE. The mechanism is built and tested; whether it
is useful is decided by `EVIDENCE_R7.md`. Code: `ple.py` (addressing, table, fusion), `ple_store.py` (read-only
inference stores), `ersa_model.py` (`ErsaConfig.ple`), `lm_training.py` (lazy table updates, checkpoint contract),
`experiments/run_r7.py` (study, controls, measurements, the offline store writer).

Question: can deterministic n-gram-addressed parametric memory increase useful predictive capacity without
proportionally increasing active computation, recurrent state or decode latency?

Hypothesis: PLE supplies learned local-pattern information through a few deterministic row reads, while ERSA and
the CGC / FFN experts keep all dynamic computation.

## 1. Primary sources (read for this design)

| Source | Version read | What it gave |
|---|---|---|
| github.com/deepseek-ai/Engram, `README.md`, `engram_demo_v1.py` (422 lines), `Engram_paper.pdf` | commit `fb7f84a21f91223715394a33a1dc24bbfb7f788e` (2026-01-14), Apache-2.0 | the implemented hashing, tables, gate and convolution; paper pp. 1-18 and Appendix A / C |
| arXiv:2601.07372, "Conditional Memory via Scalable Lookup" | v1 (12 Jan 2026) and v2 (12 Jul 2026); v2 sections 1-3 and appendices read verbatim, sections 4-6 only as a summary | v2 keeps v1's equations, settings and placement |
| docs.vllm.ai/en/latest/features/engram/ | read 2026-11-18 | serving: hashing, lookup and gate fused on GPU; tables in pinned host memory (UVA) by default; side-stream prefetch; optional disk offload with `preadv`; Qwen4Exp calls the mechanism "PLE" |

No code was copied. The Elpis module is written from the published description, and every difference is listed in
section 3. The demo's dependencies (NumPy, SymPy, Transformers, tokenizer downloads) are not used, and no external
weights are used.

## 2. The reference mechanism, by kind of operation

| Step | Reference (paper Eqs. 1-5; demo) | Kind |
|---|---|---|
| tokenizer compression | a surjective map: NFKC, NFD, accent stripping, lowercasing, whitespace collapse (23.43% fewer ids) | table lookup |
| suffix n-grams | `g[t, n] = (x'[t-n+1] .. x'[t])`, n = 2 .. N (N = 3); `pad_id` before the sequence start | integer |
| hash | `XOR_k x'[t-k] * m_k` with odd 64-bit multipliers per layer (seeded NumPy generator, bounded against overflow), shared by the heads of an order; head j: `mod P_j`, distinct primes | integer |
| tables | one embedding with per-head offsets; rows concatenated: `e[t]` | row gather |
| key, value | `k = W_K e`, `v = W_V e` | neural |
| gate | `alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d))` (paper Eq. 4); the demo applies a signed square root before the sigmoid | neural |
| refinement | `Y = SiLU(Conv1D(RMSNorm(alpha v))) + alpha v`, depthwise, causal, kernel 4, dilation N, zero init | neural, with (kernel - 1) x N = 9 positions of state |
| multi-branch | 4 mHC branches: one table and `W_V`, a key projection per branch | neural |
| placement | residual before attention and MoE, layers 2 and 15 (27B), 2 and 6 (ablation); best single layer: the second | |
| training | tables: Adam, learning rate x5, no weight decay; backbone: Muon, weight decay 0.1 | |
| serving | addresses known before the forward pass: prefetch from host memory overlapped with earlier layers (<= 2.8% throughput loss for a 100B table, H800); multi-level cache by Zipf frequency (proposed) | system |

## 3. The Elpis PLE, as implemented, and every divergence

```
tokens ─┬─► embed ─► x ─► block 0 ─► ... ─► x ──(+)──► block L (mixer, channel) ─► ... ─► head
        │                                       ▲
        │                                     alpha · W_V e        alpha = sigmoid(RMSNorm(x) · RMSNorm(W_K e) / sqrt(d))
        │                                       ▲
        └─► addresses (history suffix ⊕ tokens) ─► e = ‖ E[a(n, h)]  (8 rows of the table, gathered)
```

| Step | Elpis R7 | Divergence and reason |
|---|---|---|
| token ids | the byte tokenizer's exact ids | no compression: the directive forbids lossy normalization without a separate experiment |
| before the stream, before a boundary | SENTINEL = vocabulary size (never a token). `PLEConfig.boundary` (BOS for byte data) cuts the history: tokens before the latest boundary inside an n-gram become SENTINEL | the demo pads with `pad_id`, and nothing in the sources defines streaming or document boundaries |
| hash | `XOR_k x'[t-k] * m[n,h,k]`, with an odd multiplier per (order, head, offset) | per-head multipliers (the demo shares them across heads). They come from SHA-256 of a versioned string instead of NumPy (absent from CI; stability across versions not promised). The bound `m < 2^63 / (V + 1)` rules out int64 overflow (tested against Python integers) |
| tables | distinct primes >= `rows` per (order, head); one `nn.Embedding` with offsets; sparse gradients | same structure |
| gate | paper Eq. 4, one scalar per position, query = the residual stream entering the fusion block | the demo's signed square root is not adopted (not in either paper version) |
| refinement | none | the paper reports that removing the convolution "only marginally degrades". It would add 9 positions x d of per-stream state; the fused block's ERSA mixer has its own causal convolution |
| branches | one stream | Elpis has no multi-branch residual |
| placement | one configured block (`PLEConfig.layer`), before its mixer | the directive: one layer first. The study fuses before block 1 of 2, after one mixing round |
| training | tables: `SparseAdam` at `lr_multiplier` (default 5) x the learning rate, betas (0.9, 0.95), no weight decay; the rest: the model's AdamW | lazy Adam makes a row that a step does not address untouched by that step (tested) |
| serving | a resident trainable table, or an immutable `TensorStore` / read-only `FileStore` | no prefetch, cache or offload framework before usefulness is shown |

**Addressing contract (tested).**
- Addresses are a pure function of the token ids at or before each position and of the stream's last
  `max_order - 1` ids. They are computed once per forward call and reused (`ErsaLM.forward`).
- They are equal whether a sequence arrives in one call, in chunks or token by token.
- They are pinned by a digest: `NgramAddresser.digest()` hashes the version string, orders, heads, boundary, mode,
  vocabulary, primes and multipliers. Checkpoints and stores carry it, and a mismatch is refused.
- `addressing="random"` is the content-free CONTROL: rows redrawn on every call from a seeded generator.

**State.** `ErsaState.ple_history` holds `[batch, max_order - 1]` int64 token ids. It is bounded, volatile, reset
with the session, accounted in `nbytes()` and `fixed_state_bytes()`, and never serialized. A checkpoint holds the
configuration, the addressing digest and the parameters, never live state (schema `elpis-torch-r7-lm/1`). R6
checkpoints (`elpis-torch-r6-lm/1`) still load, as models without PLE.

**Controls.**
- `PLE.enabled = False` removes the contribution exactly: the logits equal, bit for bit, the same model without PLE
  (tested for ERSA with experts and for R5). A zero value projection gives the same result.
- `ErsaConfig.ple = None` is the R6 model, and R0-R6 checkpoints load unchanged.

**Read-only stores (`ple_store.py`).**
- `TensorStore` is an immutable copy at the stored precision (float32 reference; bfloat16 optional, tested within its
  precision).
- `FileStore` reads a fixed-header, row-major file opened `O_RDONLY`. Each call reads exactly its distinct rows, one
  positioned read per run of consecutive rows. There is no memory map, no cache and no full load.
- Both stores count requested rows, distinct rows, reads and logical bytes. Opening checks magic, version, dtype,
  shape and size.
- The content digest is verified only by an explicit offline `verify()`. Attaching checks shape and addressing.
- A store refuses training. Files are written only by `run_r7.export_store`, an explicit offline operation.
- A read from the page cache is not a device read. Cold-device timing is NOT_RUN in this container.

**Recurrent compatibility.** The addresses of a position depend only on the token ids, so any future multi-pass use
of the same positions can reuse them. Nothing here performs multiple passes.

## 4. Cost model

Per token, a PLE reads `lookups = |orders| x heads` rows of `head_dim` values. That is `lookups x head_dim x
bytes` logical bytes, independent of the table's size. It adds `2 x width x d` multiply-adds for the key and value
projections and 2 RMSNorms. Addressing costs about `2 x sum(orders) x heads` integer operations. The table is
resident, total-but-not-active parameters: `active_parameters` counts only the rows read.

At decode, the lookup's addresses need the newest token, so they are known only one step ahead, before the forward
pass of that step. A host-memory or file-backed table can therefore overlap its reads with the blocks before the
fusion block, as Engram's serving does, but not across steps.

Two other paths dominate R6's measured cost, and R7 measures them beside the PLE (`run_r7 latency`):

- **R5 evidence read.** Per query at 1M tokens, the R6 accounting projects 8.39M multiply-adds for the coarse scan
  and 7.2-7.4G for re-encoding the 64 candidates. That is many times the per-token active weight cost (236M / 366M
  for the 500M / 1B configurations). The bounded reread cache removes a part, and semantics are not traded for
  speed.
- **Grouped expert dispatch.**
  - `torch.bincount(...).tolist()` synchronizes the host once per routed block per call: on a GPU, once per routed
    block per decoded token.
  - A Python loop over all experts launches each selected expert's GEMMs separately, followed by an `index_add`.
  - At batch 1, two experts run per routed block, but the loop still iterates every expert.
  - A grouped-GEMM kernel and device-side counts would remove the loop and the synchronization. This is analysis
    only; no CUDA claim is made.

## 5. Not built

- tokenizer compression;
- the short convolution;
- multi-branch keys;
- a PLE in every block;
- prefetch, cache hierarchy, offload scheduler, sharding or quantized rows;
- any donor weights;
- any change to R5 semantics, CGC, the experts or their residency;
- adaptive computation, agentic control or RALPH (Hermes owns agentic learning);
- any change under `src/` or `native/`.
