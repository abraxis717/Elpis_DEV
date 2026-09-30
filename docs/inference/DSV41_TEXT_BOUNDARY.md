# V4.1 text boundary — MECHANICS_NONQUAL for verbal inference

The real tokenizer boundary is implemented. A compatible production arithmetic
driver is not. **Do not set
`VERBAL_INFERENCE_BLOCKER=PRODUCTION_PARAMETER_ARTIFACT_ONLY`.** Supplying weights
to the current `CompactTarget` cannot fix the architecture mismatch.

## Authority and provenance

The fresh checkout was clean on `main` / `origin/main`:

- Elpis commit `774c04e33654d136b8abbd4bd1dc90e9eb4980c4`;
  tree `be2159b7781cdb7ef619a8596bc3af66992a0917` (includes PR #7).
- Recipe commit `8cadfede7063c896b944e7bae05daa3549ae97ea`;
  tree `3ed5db20847e09a9f3d157660e27c8d2c3141fc3`.
- Remote `ls-remote` failed on unavailable DNS. These are freshly inspected
  clone refs, not a claim that the remote was independently refreshed.
- [Starting authority](evidence/starting_authority.json) records SHA-256 for
  the starting system authority, nonclaims, complete inference/runtime sources
  and inference/integration tests. No abandoned Astra branch was used.

The tokenizer is caller-supplied, not downloaded or bundled with the package.
The extracted artifact and the fresh recipe's
`static/tokenizers/v41/tokenizer.json` are byte-identical:

`81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`

The recipe provenance identifies a comparison to
[`DeepSeek-V4-Flash-Vision-Exp` revision 6821d6a](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/tokenizer.json).
It changes ID 129264 from `<｜deepseek_image｜>` (`special=false`) to
`<｜image｜>` (`special=true`). The donor did not record its original import
revision. The comparison revision must not be mislabeled as the import revision.
See [preserved license and provenance](../../LICENSES/PROVENANCE.md).

Prompt semantics come from the pinned recipe's `v4/mod.rs` (SHA-256
`0a4577a216bcbddabd5bcf969cabd75060deb79e643548f8a04bda1d76e4a982`)
and `v4/dsv41.rs` (`3408554e8a4ade05e034231cab8f87efcee7459d7fdd28fbb926f652e3b56786`).
Only a narrow Elpis-owned text adapter was added. No recipe application framework,
model code, heavyweight inference package or production weights were imported.

## Exact identities and semantics

- Tokenizer domain: `elpis.inference.tokenizer.deepseek-recipe-v41.v1`.
- Admitted tokenizer identity:
  `a4ff8fefe87f25992ad0d20def12cc29d4fcbb06549d35e6f3329778e7ea16e1`.
  It binds artifact SHA-256, engine `tokenizers-0.23.2`, renderer, vocabulary,
  stop IDs and UTF-8 decoder policy.
- Renderer: `elpis.inference.render.deepseek-recipe-v41-text.v1`.
- Vocabulary: 128,000 BPE model entries; 129,280 total contiguous IDs including
  added tokens. IDs are derived from the artifact, not hard-coded in the adapter.
- BOS = 0; EOS = 1; pad = 2; System = 128799; User = 128803;
  Assistant = 128804; think = 128821; end-think = 128822.
  **Role/thinking markers have `special=false` in this artifact.** They remain
  chat control tokens. `special_tokens` exposes actual flags;
  `control_tokens` exposes all added-token spellings.
- Encoding never inserts implicit special tokens. Chat rendering adds one BOS,
  recipe role markers, EOS after historical assistant messages and the assistant
  suffix. `thinking=False` is an explicit Elpis API default and appends
  `</think>`; the recipe's conversation default is thinking enabled.
- Thinking inserts the exact first-message system effort text: low 50,
  default/high/xhigh 75, max 100. No additional inferred prompt template.
- Generation stops on artifact-derived EOS. `<|EOT|>` is not guessed to be EOS.
  Model-specific extensions require a future qualified model contract.
- Plain chat/HACF content rejects added/control-token spellings and invalid UTF-8.
  `encode(rendered_prompt)` supports explicitly caller-rendered recipe prompts.
  Tool, multimodal and response-schema chat templates are outside the adapter.
- Incremental decode visits only the new token's precomputed byte material.
  At most three pending UTF-8 bytes are retained. Invalid or terminal incomplete
  bytes use U+FFFD, consistent with the official ByteLevel decoder. EOS is
  consumed, not displayed. Other markers are preserved by default so an output
  parser can distinguish them. This is text transport, not tool execution.
- Synthetic identity `synthetic-raw16-DSV41_ENGRAM` and renderer
  `elpis.inference.render.synthetic-nibble16.r0` are unchanged.
- Production model/target identity: **NONE ADMITTED**.

## Calling the boundary

Install the optional inference and text extras (`.[inference,text]`). Supply a
local artifact under a deployment-selected substrate root:

```python
from elpis.inference.text import ChatMessage, V41Tokenizer, RECIPE_SHA256
from elpis.substrate.boundary import RootCapability

with RootCapability(tokenizer_root) as root:
    tokenizer = V41Tokenizer.load(root, "tokenizer.json",
                                  expected_sha256=RECIPE_SHA256)
ids = tokenizer.encode_chat((ChatMessage("user", "Hello, Elpis."),))
# (0, 128803, 19923, 14, 3909, 62848, 16, 128804, 128822)
```

`ChatMessage` is also exported from `elpis.inference.text`. With a compatible
future target, the composition call is:

```python
result, record = runtime.run_text(
    engine, state, "Hello, Elpis.", admission,
    tokenizer=tokenizer, request_id="turn-1", max_new_tokens=64,
    expected_state=state.digest, emit=print_chunk,
)
```

`admission` comes from `Runtime.admit_context(..., text_tokenizer=tokenizer)` or
an explicitly empty `admit_context` under the production renderer. The existing
Regex ingress accepts its bounded task grammar; arbitrary chat is not silently
sent through that grammar. Retrieval task, corpus pins and context snapshot stay
explicit. The principal's prefill order stays `admission.tokens + request.prompt`.
Whether a trained target supports that prefix order is part of its qualification.

`run_text` validates model/tokenizer/vocabulary before begin. During a sequence,
only model `next`, incremental byte decode and caller emission run. It measures
encoding, prefill/admission, model steps, decode, emitter and finalization
separately. Emission exceptions finalize the stopped sequence but record no
successful receipt; already streamed text remains an uncommitted proposal.
After success it records the existing principal commit and an inert text-output
digest in ECS. It retains the exact PrincipalRequest for replay.

## Production intake gap — exact mapping, not a rename

`Target` mixes legacy transaction/stream contracts with the principal methods.
`PrincipalEngine` itself only needs config/model/numerical identity,
`admit_stream`, `window_initial`, and `window_step`. Its sequencing and bounded
committed state are model independent. `WindowState`'s single history/window and
the compact driver's equations are fixture assumptions, not a production tower.

| Required production binding | Current compact principal path | Missing production compatibility |
| --- | --- | --- |
| Model identity | Binds config, tensors, all projections, scheme, bank, experts | Exact trained architecture/profile and per-artifact pins |
| Embedding / vocabulary output | `(v,d)` embedding, `(d,v)` output, F32 row-major | Actual checkpoint names/layout; vocabulary and tokenizer digest agreement |
| Attention | One `(d,d)` Q/K/V, no O projection, tanh residual | Layer stack, normalization, positions/RoPE, latent Q/KV, grouped/factorized O, sinks and hyperconnections where required |
| Local geometry | One window and one hidden vector | Per-layer window/head/latent geometry; no durable activation state |
| Router / experts | Softmax top-k, F32 SwiGLU, shared sum | Actual router score function, scaling, correction bias, expert layouts and complete expert coverage |
| DSV4.1 addressing | Frozen synthetic compressed map, primes/multipliers and mean row projection | Trained compressed map and exact per-layer table/layout/gating/projection; digest/tokenizer match |
| Numerical profile | Explicit CPU NumPy F32 | Qualified BF16/FP8/FP4 and scale layouts if present, or an independently qualified F32 artifact; no silent casting |
| Projections/channels | Principal retains model-owned M; G/X/R and compress unused | Exact learned associative projection; no external G/X/R activation inputs |
| Stop / replay | PrincipalRequest stop IDs; identity trace at finalize | Model stop profile bound to actual tokenizer and numerical replay evidence |

`inspect_parameters` accepts a caller-pinned artifact path, byte size and SHA-256
under `RootCapability`, streams verification, validates safetensors tensor
names/shapes/dtypes/spans and returns a read-only inventory. It rejects digest
changes, out-of-budget files/headers, duplicate keys, unsupported storage types,
shape/span inconsistencies, overlaps, holes and symlinks. Storage inspection is
not numerical support. `bind_production_target` explicitly returns
`UNSUPPORTED:PRODUCTION_DRIVER_UNIMPLEMENTED`, never `CompactTarget`.

This preflight is deliberately **not described as complete production intake**.
No invented trained parameters, guessed architecture conversion or pseudo-trained
replacement model was used to hide this gap.

## Local assets, qualification and next work

[Local asset inventory](evidence/local_assets.json) contains exact locations,
hashes, every tensor's shape/type/offset, configuration, expert layouts and the
reference-only V4.1 geometry. Large assets remained in place and were never
mutated. The most relevant checkpoint is
`/mnt/primesauce/models/DeepSeek/DeepSeek.V4Pro_1B/model.safetensors`:
2,013,819,938 bytes; 310 tensors (307 BF16, three I64 routing tables); four layers,
dimension 2048, 16 routed experts/top-2 plus one shared expert, window 128.
It has **no engram tensors**, a different tokenizer SHA-256 and different role/
image tokens. Its README says it was initialized then trained on five
copypastas, not reduced from trained full-model weights. It is incompatible.

The quarantined ElpisPrimeConsolidated and Dimple initializer are also inventoried;
neither is a V4.1 language tower. V4.1 donor files under `models/ElpisDonors` are
reference code/configuration, not trained parameters. Other model families are
not inferred compatible from names.

See [qualification results](QUALIFICATION.md) and
[tokenizer measurements](evidence/tokenizer_performance.json). No model generation
transcript, neural tokens/s, neural prefill latency, model working-set measurement
or real-model FMS/expert activity exists. Null metrics are intentional.

The next task is to establish and implement the exact principal-compatible
trained DSV4.1 tower contract: per-layer arithmetic/layout, learned addressing,
retained projection and numerical profile, with donor differentials. Then admit
its trained artifacts through substrate/FMS and qualify several GREEDY verbal
completions. Merely copying a weight file is insufficient today.

Generated text remains output/proposal. Explicit post-sequence artifact admission
into HACF is deferred. Canonical ECS topology projection into structural/HACF
addresses and bounded context is also deferred; callers still supply snapshot
identity. Neither edge belongs in the active token loop, and none of this work
establishes geodesic intelligence.
