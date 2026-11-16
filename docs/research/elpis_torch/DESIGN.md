# Elpis Torch R0: architecture of the CGC-native generative substrate

Status: RESEARCH_ONLY, NO_RUNTIME_AUTHORITY, NO_LANGUAGE_CLAIM. Code: `research/elpis_torch`. Tests:
`tests/research/elpis_torch`. Results: `EVIDENCE_R0.md`. Provenance of the CGC mathematics:
`research/elpis_torch/PROVENANCE.md`. This document describes what is implemented. It is not canonical
architecture (`docs/ARCHITECTURE.md` is), and it changes nothing in `src/`, `native/` or the mission.

## 1. What it is

A decoder-only transformer in PyTorch that trains by backpropagation and generates autoregressively, in
which the Cubic Geometric Core (CGC, the ECS_G E0 cell) is a neural operator inside every block. CGC is not
a module beside the model, not an oracle consulted at inference and not a controller around another model:
its weights are ordinary parameters, trained jointly with the attention weights by the same loss.

The three block configurations are alternative architectures of one model, compared under one harness.
They are not three agents.

## 2. Tensors and topology

Symbols: `B` batch, `T` positions, `H = d_model`, `V = vocab_size`, `L = n_layers`, `h` query heads,
`g` key/value heads (`h % g == 0`; `g = h` is ordinary multi-head attention, `g < h` grouped-query),
`e = H / h` head dimension (even).

| Stage | Shape | Notes |
|---|---|---|
| tokens | `[B, T]` int64 | byte tokenizer: ids 0..255 are bytes, 256..258 BOS/EOS/PAD |
| embedding | `[B, T, H]` | `embed.weight` `[V, H]`; no absolute position table |
| block input | `[B, T, H]` | the residual stream |
| queries | `[B, h, T, e]` | `q` `[H -> h e]`, rotary applied |
| keys, values | `[B, g, S, e]` | `k`, `v` `[H -> g e]`; rotary on keys; `S = T` batched, `S = cached + T` incremental |
| attention scores | `[B, h, T, S]` | `q k^T / sqrt(e)`, causal mask, softmax |
| channel input | `[B, T, H]` | RMSNorm of the stream |
| logits | `[B, T, V]` | final RMSNorm, then `embed.weight^T` (tied) or a separate head |

Block (pre-norm, the same in every configuration):

    x = x + Attention(RMSNorm(x))
    x = x + Channel(RMSNorm(x))

Channel transformations:

| | Channel | Parameters per block |
|---|---|---|
| A `ffn` | `W_out (silu(W_gate x) * W_up x)` (SwiGLU), hidden width `f` | `3 H f` |
| B `cgc` | `P_out core(P_in x)` | `H d + d C N + C H` (bottleneck `d`) or `H C N + C H` (dense) |
| C `ffn+cgc` | `SwiGLU(x) + gate * P_out core(P_in x)` | sum of the two, plus one scalar gate |

`core` is `C` E0 cells of `N` cubic units each over one shared input of dimension `d` (`d = H` in the dense
variant). Its weight is one tensor `[d, C, N]`:

    u[..., c, i] = x[..., :] . W[:, c, i]            # one matrix product: [.., d] @ [d, C N]
    y[..., c]    = sum_i phi(u[..., c, i])           # phi(u) = (u + u^2 + u^3) / 2

Intermediate shapes are `[B, T, d]` (bottleneck code), `[B, T, C, N]` (pre-activations) and `[B, T, C]`
(cell outputs).

### Faithful E0 and extensions, kept apart

* **Faithful.** With `cgc_preact_bound=None`, every output channel of `CGCCore` equals the float64 E0
  reference on that channel's `[d, N]` weight (tested to `1e-12`). With a bottleneck and no bound, cell `c`
  is still exactly E0 on the full hidden input with the weight `P_in^T W_c`: the bottleneck is a rank
  constraint on `W`, not another nonlinearity.
* **Extension: the pre-activation bound.** `u -> b tanh(u / b)` before `phi` (default `b = 3`). Not E0. It
  changes the pre-activation by at most `|u|^3 / (3 b^2)` and converges to E0 as `b -> inf`.
* **Architecture: `P_in`, `P_out`, the gate, RMSNorm.** E0 has no readout; the readout is needed because a
  sum of `phi` has a non-negative quadratic part (`sum u^2 >= 0`), so a signed mixture of cells is required
  to express negative curvature.

### Stability of a cubic in a stack

* The block's RMSNorm gives the CGC input unit RMS, so `|u| <= ||x|| ||w|| = sqrt(d) ||gamma||_inf ||w||`:
  in a pre-norm stack the pre-activation scale is set by the weights of that block, not by the output of the
  cubic below it. Blocks do not compose into an unnormalized degree-`3^L` polynomial of the activations.
* Initialization puts the pre-activations at standard deviation `cgc_init_gain` (0.5): `P_in ~ N(0, 1/H)`,
  `W ~ N(0, gain^2 / d)`. Every write into the residual stream (`attn.out`, `ffn.out`, `cgc.proj_out`)
  starts at `init_std / sqrt(2 L)`. A cell's mean output `N gain^2 / 2` is a constant offset `P_out` carries.
* `phi' = 1/2 + u + (3/2) u^2 >= 1/3`: a cubic unit never dies, but its gradient grows quadratically with
  `|u|`. Unbounded, large weights give large gradients and, in float32, overflow (`u^3`). Bounded, every
  cell is capped at `N phi(b)` whatever the weights are; the price is saturation (`sech^2`), which
  `CGCCore.diagnostics` reports as the fraction of pre-activations beyond `b`.
* Weight decay applies to every matrix, including `W`. Gradient clipping is global.
* Measured: twelve stacked CGC blocks, dense and bottleneck, bounded and unbounded, are finite at
  initialization with a residual stream below RMS 1 (`test_model.py`). Trained behaviour: `EVIDENCE_R0.md`.

### Initialization

Each parameter is drawn from a generator seeded by `sha256(seed, parameter name)`, in float64, then cast.
It therefore depends on nothing but the seed and the name: not on construction order, not on the dtype. A
and C share their attention and FFN values bit for bit, and C with `cgc_gate_init = 0` computes A's logits
bit for bit at initialization (`test_the_augmented_block_starts_as_exactly_the_ffn_block`).

## 3. Execution

| Mode | Call | Cost for `T` positions |
|---|---|---|
| batched (training, teacher forcing) | `model(tokens)` | one pass, attention `O(T^2)` |
| prefill | `model(tokens, cache)` on an empty cache | the same pass, cache filled |
| incremental | `model(new_tokens, cache)` | attention over `cached + new` keys only |

`KVCache` is an explicit object the caller owns: per layer, the rotated keys and values
`[B, g, length, e]`, a `length`, the model's `max_seq_len` as capacity and a `weights_tag`. Nothing is
cached inside the module. A forward pass that would exceed `max_seq_len` raises `ContextLengthError` and
leaves the cache as it was; nothing is truncated or slid silently. The cache grows by concatenation
(`O(T^2)` copying over a generation of `T` tokens): simple and exact, to be replaced by a preallocated
ring when it matters. Prefill and incremental decoding agree with the batched pass to `1e-10` (float64) and
`2e-5` (float32), the declared tolerances.

Training is teacher-forced next-token cross-entropy (`next_token_loss`), AdamW, warmup and cosine decay,
global gradient clipping. It is deterministic for a fixed seed and thread count (bitwise, tested). A
non-finite loss or gradient stops training before the step is applied and is reported (`DIVERGED`).

## 4. Volatile CGC fast weights

    W_active = W_base + dW_session            for every `blocks.*.cgc.core.weight`, nothing else

* `W_base` is the frozen model's parameter tensor. A session opens only over a frozen model
  (`training.freeze` or `load_checkpoint`): evaluation mode, no parameter requires a gradient.
* `dW_session` is allocated by the session at open, all zeros, the same shape as `W_base`. It is not a
  parameter or buffer of the model, so it cannot appear in `state_dict()`.
* Every forward pass of a session is `torch.func.functional_call(model, {name: W_base + dW}, ...)`. The
  module's own parameters are never assigned.
* Reset allocates fresh zero deltas and drops the momentum buffers, the stream and its cache. Close drops
  everything and re-verifies the SHA-256 digest of all base parameters taken at open.
* Two sessions over one model share only the read-only base tensors.

### What may teach

Only observed next tokens. Two entrances:

* `observe(tokens)` appends caller-supplied evidence to the session's stream (provenance OBSERVED) and
  returns its prequential loss (each token scored by the prediction made before it was seen).
  `adapt_on_stream` then uses, as targets, only stream positions whose token is OBSERVED.
* `adapt(SupportSet.observed(tokens, mask, source=...))` takes an explicit supervised support set whose
  provenance the caller declares.

`generate` records its samples as GENERATED. They stay in the stream as context and are never a target:
the stream update is bit for bit the update computed with a mask over the observed targets only
(`test_generated_tokens_are_context_and_never_targets`). There is no self-supervised signal, no confidence
score, no self-evaluation. For an explicit support set the session cannot verify where the tokens came
from; it refuses only the direct output of `generate`.

### One update (atomic)

1. Refuse if learning is disabled, the model has no CGC core, or `max_steps` updates were already committed.
2. Loss of the support under a trial copy of `dW`; refuse if not finite.
3. Gradient with respect to `dW` only; refuse if not finite.
4. Clip to global norm `grad_clip`; momentum (optional); candidate `dW' = dW - lr * v`.
5. Per tensor, refuse a non-finite norm; project onto `||dW'|| <= max_delta_ratio * ||W_base||`.
6. Refuse a non-finite candidate.
7. Evaluate the support under the candidate; roll back if not finite, or (optionally) if the loss rose.
8. Commit `dW'` and the momentum together, count the step, advance the weights version, drop the cache.

Until step 8 nothing of the session has changed, so every refusal leaves it exactly as it was (tested for
each status; a non-finite gradient has no natural trigger at this scale and is exercised by fault injection).

### Attention-cache coherence

A change of `W` in block `l` changes the stream leaving block `l`, hence the keys and values of every
block above it. The session therefore treats the whole cache as invalid after a committed update:

* the stream's cache is dropped at commit and rebuilt from the stream's tokens under the new weights before
  the next read (`cache_rebuilds` counts it). After the rebuild, incremental logits equal a from-scratch
  pass (tested to `2e-5`);
* every cache carries the tag `(session identity, weights version)` it was written under, and
  `ElpisTorchLM.forward` raises `StaleCacheError` for any other tag. A cache from before an update, from
  before a reset, from another session or from the bare model is refused, not silently used. The negative
  control forges the tag and gets wrong logits.

Block 0's keys and values do not depend on any fast weight (they are computed from the embeddings) and
could be kept. That optimization is not implemented.

### Zero-write properties

| Property | How it holds | Verified |
|---|---|---|
| Base parameters are not mutated by a session | functional override; no assignment | digest and tensor equality after adaptation; digest re-check at close; a deliberate mutation is detected |
| No session state in the model | deltas are not module state | `state_dict()` keys and values unchanged |
| No serialization of a session | `__getstate__`, `__reduce_ex__`, `__deepcopy__` raise; no save/export API | pickle, copy, deepcopy, `torch.save` refused |
| No filesystem write during a session lifetime, including an exception inside it | no code path writes | Python audit hook, and `strace` over write-class system calls, each with a negative control that must be caught |
| A checkpoint file is not written back | no path to the file exists in a session | digest, size, mtime and inode unchanged; directory listing unchanged |
| Restart reproduces the checkpoint | load is read-only (`weights_only=True`) | same digest and logits after a session adapted and closed |
| The only storage writers in the package | `save_checkpoint` (offline) and the experiment's `--out` | source scan in `test_isolation.py` |

Not verifiable from inside the process, and required of the host exactly as for the native runtime
(`docs/VOLATILE_RUNTIME_CENSUS.md`, section 6): no swap or locked memory, no core dumps, read-only
filesystems, no crash collectors or tracing that persist process memory. PyTorch tensors are ordinary
anonymous memory: nothing here locks it, and nothing prevents another thread of the same process from
writing a base tensor in place (the digest detects it; it does not prevent it). The strace gate covers the
scenario it runs, on one host; it is not a proof over all inputs. Python's own bytecode cache is disabled by
`-B` / `PYTHONDONTWRITEBYTECODE` in the tests, and must be by the deployment.

### Boundary to read-only memory (not implemented)

The session's only inputs are token sequences with a provenance and an optional target mask. A read-only
resource (an H-gram association read, a HACF retrieval result rendered through the codec) can enter in
exactly two ways without any new mechanism: as OBSERVED context in the stream, or as a `SupportSet` whose
`source` names the resource. Nothing flows back: a session has no write path to any resource. No retrieval
controller, ranking or memory agent is implemented, and `Provenance` has only the two values that are used.

## 5. Compute and memory

`accounting.py` computes these from the configuration. Three quantities, never interchangeable:

* **Parameters.** Per block: attention `H e (2h + 2g)`; channel as in the table of section 2; two norms `2H`.
  Model: `L` blocks + `V H` embedding (+ `V H` head if untied) + `H`.
* **Multiply-accumulates per generated token** at context `T` (cached execution): per block
  `H e (2h + 2g)` projections + `2 h e T` attention mixing + the channel; plus `V H` for the logits.
  A linear map touches every weight once per token, so the channel costs its parameter count, plus `f` for
  SwiGLU's product, plus `3 C N` for the cubic. Exponentials, `tanh`, square roots and additions are not
  counted (listed separately as `transcendentals`).
* **Activation memory.** The channel's widest intermediate is `f` for SwiGLU (three tensors of it) and
  `C N` for CGC (two, three with the bound). At equal parameters a bottleneck buys many more cells than a
  dense core does, so its activations are wider: at `H = 64` the matched configurations hold 172 (FFN),
  430 (dense) and 726 (bottleneck) scalars per token. Wall-clock follows memory traffic as well as
  multiply-accumulates; see the measured rates in `EVIDENCE_R0.md`.

Consequently "parameter-matched" in this work means equal parameter counts and nothing more. The matched
configurations have nearly equal multiply-accumulate counts (the channel differs only by the activation
terms), which is a property of these particular layers, and they are not matched in memory or time.

Scaling laws:

| Quantity | Law |
|---|---|
| attention, training | time and memory `O(L h T^2)` per sequence |
| KV cache | `2 L g e T` scalars per sequence |
| channel A | `3 H f` parameters and multiply-accumulates; `f ~ 8H/3` gives `8 H^2` |
| channel B dense | `H C (N + 1)`; matching `8 H^2` needs `C (N + 1) = 8 H`, e.g. `C = H`, `N = 7` |
| channel B bottleneck | `H d + d C N + C H` |
| session fast weights | `L d C N` scalars per session, twice that with momentum, and about three times that transiently during an update (trial copy and gradient) |
| one fast-weight update | one forward and backward pass over the support plus one validation pass, through the whole model, without the cache |
| cache rebuild after an update | one forward pass over the stream so far |

## 6. Limits known now

* **Session memory at scale.** A dense core at `H = 1024`, `C = 1024`, `N = 7`, `L = 24` holds about
  `1.8e8` scalars, so one session's delta would be about 0.7 GB in float32. Full-rank deltas for every cell
  do not scale to many concurrent sessions. Low-rank or per-layer-subset deltas are the obvious next
  design; neither is implemented or tested.
* **Update cost.** An update costs a full backward pass and invalidates the cache, so adapting after every
  token is quadratic in the stream length. R0 adapts at explicit points only.
* **Context.** The stream is bounded by `max_seq_len` and refuses beyond it. No sliding window, no
  compression, no retrieval.
* **Precision.** float32 and float64 on CPU. No mixed precision, no quantization, no GPU run was made.
* **Concurrency.** A session is single threaded. Sessions are isolated in data, not synchronized.
* **Interference.** Adapting to changed evidence degrades predictions the evidence did not change
  (measured, `EVIDENCE_R0.md`). No consolidation or retention law (K1 or other) is applied to fast weights.
* **Attention is conventional.** Only the channel transformation is CGC. Whether attention itself should be
  geometric is open.

## 7. Deliberately absent

No RALPH in any role (supervisor, expert selector, oracle, halting agent, reasoning module); no MoE routing,
Engram, mHC, HCA, AWR, GDN or ERSA; no persistent inference learning; no energy function; no
metacognitive controller; no retrieval controller. With learning disabled, or with no session at all, the
model is an ordinary transformer that runs forward passes and generates. The block and the session are the
two places a later mechanism would attach, and neither contains a placeholder for one.

## 8. Assumptions

1. A byte vocabulary is enough to exercise the substrate; no compatibility with any pretrained tokenizer or
   model is implied, and the DSV4 codec is not used.
2. The transduction task has no semantics. Passing it shows that the mechanics work and that the model can
   fit a sequential dependency, not that it reasons.
3. Small models on CPU (tens of thousands of parameters). Nothing was measured at language-model scale.
4. Fast weights are restricted to the CGC cores because the mission of this work says so, not because it
   was shown to be the best set.
5. Evidence handed to `observe` or to a `SupportSet` is trusted to be observed. Provenance is enforced for
   the session's own samples only.
6. Determinism is asserted for one process, one thread count and one PyTorch build.
