# Elpis Torch R1: scrolling-context architecture

Status: RESEARCH_ONLY, NO_RUNTIME_AUTHORITY, NO_LANGUAGE_CLAIM. Builds on R0 (`DESIGN.md`, baseline head
`320a6e4`, preserved unchanged). Code: `research/elpis_torch/scroll.py`, `scroll_session.py`, `lastblock.py`,
`scroll_data.py`, `experiments/run_r1.py`. Tests: `tests/research/elpis_torch/test_scroll*.py`. Results and
the recommendation: `EVIDENCE_R1.md`. Nothing in `src/`, `native/` or the mission is touched.

**What it is not.** Old context is reached through compressed blocks and a few selected blocks per query.
That is lossy. Nothing here is equivalent to attention over the full context, at any length.

## 1. Comparators, and what was taken from each

| Source | Its mechanism | Taken | Not taken |
|---|---|---|---|
| Native Sparse Attention (arXiv 2502.11089) | three branches per query: compressed blocks, selected raw blocks, sliding window; block selection reuses the compressed attention scores; trained end to end | the three kinds of key, selection by the query's own attention logits (no auxiliary selector), optionally separate key maps for memory (`memory_keys="separate"`) | three separate softmaxes with a learned gate (one joint softmax here); its MLP compressor (softmax-gated pooling here); its kernels and its results |
| StreamingLLM attention sinks (arXiv 2309.17453) | keep the first tokens' keys and values forever beside a rolling window; assign positions inside the cache, not in the text; cache keys before the rotary transform | pinned first tokens (`sinks`); rotary phase computed relative to the window from unrotated cached keys | its use on a pretrained dense model |
| Kimi Linear (arXiv 2510.26692) | interleave linear-attention layers (Kimi Delta Attention, a gated delta rule) with full-attention layers 3:1; constant-size recurrent state in the linear layers | the hybrid layout: only some layers carry long-range state (`global_layers`); the 3:1 ratio is the default of the capacity projection | linear attention itself. KDA is a recurrent fast-weight state, not an addressable history, and was not implemented or measured |
| `research/dsv41_tower` (this repository) | local ring window, compressed keys at a ratio, top-k index selection, a learned sink | read as a second description of the same family | no code, no numerics, no weights |

## 2. Four coordinate systems

| | Symbol | Range | Meaning |
|---|---|---|---|
| absolute stream position | `p` | unbounded integer | token `p` is the `p`-th token of the stream |
| resident window slot | `p mod W` | `[0, W)` | where token `p`'s local key and value live while `p` is among the last `W` |
| memory address | block `b = p // s` in slot `b mod M`; token `p` in slot `p mod (H + s)` | | compressed tier; fine tier and token history |
| logical horizon | `H = M s` | tokens | how far back a query can reach through memory |

`W` window, `s` block, `C` chunk (most tokens per call), `H` horizon, `M = H / s` compressed slots. Required:
`W >= C + s`, `H` a multiple of `s`, stream origin a multiple of `s`. `StreamState.coordinates()` reports all
of them; `token_at(p)` returns a token by absolute position or raises `EvictedError`.

## 3. One attention, four kinds of key

For a query at absolute position `p`, in one softmax:

| Keys | Which | Logit | Position signal |
|---|---|---|---|
| local | tokens `p - W + 1 .. p`, raw | `<R(i) q, R(j) k> / sqrt(e)`, `i, j` window-relative | rotary, relative to the window |
| sinks | the stream's first `sinks` tokens, once they have left the window | `<R(D_j) q, k_j> / sqrt(e)`, `D_j = W + sinks - j` | a fixed offset behind the window |
| compressed | one pooled key/value per block, for selected blocks read coarsely | `<gq, ck_b> / sqrt(e) - softplus(rho_h) log(p - end_b)` | learned slope on log-distance |
| fine | the raw tokens of the best `fine_blocks` selected blocks | `<gq, gk_t> / sqrt(e) - softplus(rho_h) log(p - t)` | the same slope |

* **Reachability is defined per query.** Block `b` is reachable from `p` iff it ends at or before `p - W`
  (never both in the window and in memory, and nothing falls between) and `b >= p // s - M` (inside the
  horizon). The chunk size is therefore an execution detail: any chunking gives the same logits.
* **Selection is the attention itself.** A block's score is the query's largest head logit against the
  block's compressed key. The top `global_topk` reachable blocks are read: the first `fine_blocks` as raw
  tokens, the rest as their single compressed token. No controller, ranker, index model or second network.
* **Compression.** `ck_b = sum_t softmax(gate_t) gk_t`, `cv_b = sum_t softmax(gate_t) v_t` over the block's
  `s` tokens; `gate` is a linear map of the layer input. Memory keys carry no rotary phase.
* **Memory keys.** `memory_keys="shared"` (default): `gq = q`, `gk = k`, the window's own maps, unrotated, so
  whatever the window learns about matching content is what addresses memory. `"separate"`: memory has its own
  `gq`, `gk` maps, as in NSA. Both are implemented and tested against the oracle; `EVIDENCE_R1.md` reports
  what each did when trained.
* **Layers.** `global_layers` lists the layers that carry memory; the others are local only. With memory in
  more than one layer information persists beyond one horizon (layer `l + 1` remembers what layer `l`
  remembered): tested, and a reason to prefer few memory layers.

### Positions after scrolling

Local keys are cached unrotated and rotated by their offset inside the window at use, so the rotary angle
never exceeds `W + C`. Memory distances are differences of integers. Consequently the logits do not depend
on the absolute position at all: a stream that starts at `2^20 - 160`, `2^20`, `2^40 + 8` or `2^62 - 8000`
produces bitwise the logits of the same stream started at 0 (tested). There is no position at which the
arithmetic changes regime, in particular not at one million.

### Eviction

Every buffer is allocated once in `new_state` and written by index. A new block overwrites slot `b mod M`,
which held block `b - M`: first in, first out, by construction. A new token overwrites window slot
`p mod W` and fine slot `p mod (H + s)`. Nothing else removes anything; nothing is concatenated; a chunk
larger than `C` is refused whole (`ChunkError`) and the state is left as it was.

## 4. Correctness method

`tests/research/elpis_torch/test_scroll.py` contains an oracle that shares no ring arithmetic with the
implementation: one token at a time, every key and value of the whole stream kept in dictionaries by
absolute position, reachability decided from the written rule. The chunked, preallocated, wrapping
implementation equals it to `1e-9` (float64) over a stream of 2.3 horizons, with memory in one or two
layers, with and without sinks, and from an origin of `2^20`. Further gates: equality with the R0 dense model
when everything fits in the window; chunk-size invariance; bitwise origin invariance; state size and tensor
addresses constant over 5.5 horizons; every selected block inside its query's horizon and outside its window.

## 5. Last-block fast weights and cache validity

Claim: if only the final block's CGC core weight `W_L` changes, every cached key, value, sink, compressed
block and fine token is unchanged, exactly.

Proof. Let `x_l` be the stream entering block `l`. Block `l` writes to its caches only functions of
`norm(x_l)` (`k`, `v`, `gk`, `gate` and their pooled forms). `x_{l+1} = h_l + channel_l(norm(h_l))` with
`h_l = x_l + attention_l(...)`. `W_L` appears only in `channel_L`, the last operation before the final norm
and the head. So `x_0 .. x_L` and `h_L` do not depend on `W_L`, hence no cache entry of any layer does, and

    logits(W_L) = head(final_norm(h_L + channel_{W_L}(norm(h_L))))

Changing the CGC of any earlier block `l < L` changes `x_{l+1}` and therefore the caches of every block
above `l`. Both directions are tested numerically: the R0 KV cache is bitwise identical under a last-block
change and differs in layer 1 under a block-0 change; a scrolling session's state after adapting is bitwise
the state a model holding `W_L + dW` from the first token would have written.

Consequences, implemented:

* `SessionConfig(fast_scope="last")` in the R0 session keeps its KV cache across updates (R0's default
  `"all"` still drops and rebuilds it).
* `ScrollSession` accepts only `"last"`: a full-stack update would require recomputing a history whose
  tokens beyond the horizon no longer exist.
* An update needs no forward pass through the model. The session keeps a bounded ring (`learn_window`) of
  `h_L` and the following token with its provenance; the update is the R0 bounded transaction (clip,
  project, validate, atomic commit, refusal statuses) on the pairs whose follower was OBSERVED.
* Cost of an update: `learn_window` evaluations of one CGC transform and the head, forward and backward.
  It does not depend on the stream length.

What the restriction costs is leverage: only what the last block's channel can express is adaptable.
Measured in `EVIDENCE_R1.md`.

## 6. Compute and memory

Per stream, from the configuration alone (`accounting.scroll_state_bytes`; equal to the allocated bytes,
tested), independent of the number of tokens streamed:

| Tier | Scalars |
|---|---|
| window | `L * 2 * g e * W` |
| sinks | `L * 2 * g e * sinks` |
| compressed | `L_mem * 2 * g e * M` |
| fine | `L_mem * 2 * g e * (H + s)` |
| open block | `L_mem * (2 g e + g) * s` |
| token history | `(H + s)` int32 |

Attention mixing per token (`accounting.scroll_compute_per_token`): a local layer costs `O(W)`. A memory
layer adds one score per resident block, `h e M`, **linear in the horizon** with constant `1 / s`, plus a
constant number of selected keys. Dense attention at the horizon costs `2 h e H`. For the experiment model
at `H = 2^20`: 8,704 (local), 4,194,304 (block scoring), 4,608 (selected) against 134,217,728 (dense).
Block scoring is the term that grows; a second level of blocks would make it logarithmic and is not built.

### Projected placement on a 24 GB GPU (PROJECTED, not measured)

`accounting.deployment_projection()`: `d_model` 2048, 24 layers, 16 heads, 4 key/value heads, 6 memory layers
(every fourth), window 8,192, block 64, horizon `2^20`, 16-bit scalars, one stream, about 1.29e9 parameters.

| Class | Item | Fine tier in CPU RAM | Fine tier on GPU |
|---|---|---:|---:|
| GPU resident | weights | 2.41 GiB | 2.41 GiB |
| | window, all layers | 0.38 GiB | 0.38 GiB |
| | compressed tier | 0.19 GiB | 0.19 GiB |
| | block scores per chunk of 512 | 0.25 GiB | 0.25 GiB |
| | chunk activations (estimate) | 0.12 GiB | 0.12 GiB |
| | fine tier | | 12.00 GiB |
| | **total** | **3.35 GiB** | **15.35 GiB** |
| CPU RAM | fine tier | 12.00 GiB | |
| | token history | 4 MiB | 4 MiB |
| Immutable persistent (read-only) | checkpoint | 2.41 GiB | 2.41 GiB |

A dense KV cache of the same model at `2^20` tokens would be 48 GiB. Either placement fits 24 GiB on paper;
with memory in all 24 layers the GPU placement does not (tested as a property of the formula). With the fine
tier in CPU RAM each chunk fetches `C * fine_blocks * s` raw keys and values per memory layer across the bus;
that transfer was not measured. No session state belongs to the persistent class, and nothing writes there.

## 7. Limits

* **Lossy.** A fact survives only if its block is selected. Selection is top-k of one head's logits against a
  pooled key; `EVIDENCE_R1.md` reports how often the right block is selected at each length.
* **Selection is not differentiable.** Blocks that are never selected receive no gradient through the fine
  path; the compressed path is the only training signal for the keys that decide selection.
* **Block scoring is linear in the horizon.**
* **Recency is a single learned slope on log-distance.** Two matching facts at large, similar distances are
  barely separated: contradictory evidence far back is an expected weakness.
* **Trained short, run long.** The models are trained on 512-token streams and evaluated to `2^20`. No model
  was trained at long length here.
* **CPU, float32/float64, tens of thousands of parameters.** No GPU run, no mixed precision.
* **One stream per state.** Batches are independent streams of equal length; no padding, no variable length.

## 8. Not built

Linear attention (KDA), hierarchical blocks, learned block sizes, any retrieval controller or re-ranker,
RALPH in any role, persistence of memory across sessions, memory shared between sessions, summarization by
generation, anything on the canonical runtime path.

## 9. Ouroboros qualification protocol for the one-million-token run

To be run on the owner's hardware; this environment has no GPU. Each step states its pass condition.

1. Environment record: GPU model and memory, driver, PyTorch build, commit SHA, `nvidia-smi` before the run.
2. `pytest tests/research/elpis_torch` with `ELPIS_REQUIRE_TORCH=1`: all pass on that machine.
3. Train `sparse`, `local` and `dense` with `run_r1 train` at the committed settings, three seeds each;
   keep the logs. Pass: each `COMPLETED`.
4. `run_r1 ladder` for each checkpoint, all stages including `1,310,720`. Pass conditions, all mechanical:
   `state_bytes_constant` true at every stage; `state_bytes == state_bytes_formula`; final `position` equals
   the stage length; `evicted_blocks == 16,384` at the last stage; no non-finite logit.
5. Move the model and the state to the GPU (a device option is not implemented; adding it is part of this step) and repeat step 4 with
   `torch.cuda.max_memory_allocated()` recorded per stage. Pass: the peak is flat from 256K to 1.31M tokens
   and within 10% of the capacity model for that configuration.
6. Accuracy is reported, not gated: by kind, distance bucket and contradiction, with the fraction of probes
   whose fact block was selected. Preregister any threshold before running.
7. Zero-write: run the scrolling scenario under `strace` (as `test_session.py` does for R0) and under a
   read-only root. Pass: no write-class system call between the session markers.
8. Record every failure. A failed stage is kept; a repair is a new run.
