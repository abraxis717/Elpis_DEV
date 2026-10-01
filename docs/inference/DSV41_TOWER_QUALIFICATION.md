# DSV4.1 tower mechanics qualification

Disposition: `PASS_DSV41_TOWER_ARCHITECTURE`. This qualifies the arithmetic of the
production-shaped driver `elpis.inference.drivers.dsv41` against pinned donor
behavior. It is **not** a trained model. Only the deterministic
`PRODUCTION_SHAPED_FIXTURE` (`TRAINING=NONE`) is exercised, so no language-quality
or intelligence claim follows (see `docs/NONCLAIMS.md`).

## Authority

- **Branch base.** WIP tower `a8b5caa` (`feat/dsv41-tower-wip-2`, merged by PR #9),
  on the accepted text-boundary tranche `5c62d35`.
- **Donor files.** `docs/inference/dsv41_donor_authority.json` pins 31 files under
  `ElpisDonors`. All 31 were verified by SHA-256 and byte length before use. The
  oracle executes `DeepSeek4.1/00_V41_REFERENCE/inference/{model,engram}.py`.
- **Tokenizer.** It is caller supplied:
  `deepseek-ai/deepseek-recipe@8cadfede…/static/tokenizers/v41/tokenizer.json`,
  SHA-256 `81f64d12…e8d99`.

## Reproducing (no skip can pass)

Torch is a qualification-only dependency of the donor oracle. It is not in
`pyproject.toml`, `src/elpis/**`, the runtime or the driver. Install it in a
separate environment:

```sh
python -m venv /tmp/oracle && /tmp/oracle/bin/pip install -e ".[test]" "torch>=2.10" sympy
ELPIS_DSV41_QUALIFY=1 ELPIS_REQUIRE_NATIVE=1 ELPIS_NATIVE_BUILD=build \
ELPIS_V41_TOKENIZER=/path/tokenizer.json ELPIS_TOWER_DONORS=/path/ElpisDonors \
/tmp/oracle/bin/python -m pytest tests/inference/dsv41
```

With `ELPIS_DSV41_QUALIFY=1`, the run enforces three rules:

- both reference paths must be bound;
- both `test_differential.py` and `test_tower.py` must be collected;
- any skip, including a missing `torch`, is reported as a failure.

Recorded run: torch 2.10.0 (CPU, one thread), NumPy 1.26.4, tokenizers 0.23.2,
sympy 1.14.0. Result: 17 differential and 8 tower tests passed, 0 skipped.

## Differential results by mechanism

All Elpis/donor checks compare against definitions AST-extracted from the pinned
donor. Device leaves (FP8/FP4 casts, the sparse kernel, Sinkhorn) are independent
Torch CPU F32 equivalents.

- **Exact equality:**
  - tokenizer compressed map, Engram prime layout, offsets and multipliers;
  - streamed n-gram hashes, including across arbitrary chunking;
  - cache quantize/dequantize in the local E4M3, index FP4 and compressed
    FP4/E4M3 modes;
  - router expert ordering for softmax, sigmoid and sqrtsoftplus with k ∈ {1, 2},
    normalized and unnormalized.
- **Within tolerance:**
  - route weights and SwiGLU expert arithmetic: `rtol=2e-6`;
  - mHC pre/post/comb orientation, `hc_post` and RMSNorm: `rtol=3e-6`.
- **Full seven-layer tower.** The fixture covers pure SWA, a ratio-2 compressor
  owner, ratio-2 consumers, a ratio-1 owner that is also the candidate source,
  candidate-masked consumers, two Engram layers, and routed plus shared MoE in
  every layer.
  - `test_full_tower_matches_unmodified_donor_prefill_at_every_position` runs Elpis
    incremental decode against the **unmodified** donor prefill path on every
    prefix. It compares every sublayer output (Engram, attention, MoE), every
    layer's residual stream, the logits and the argmax.
  - `test_full_seven_layer_donor_prefill_decode_and_per_layer_streams` compares
    against the donor's own prefill plus incremental decode, with one documented
    harness correction (below).
- **Tolerance.** `rtol=1e-4`, `atol=1e-5`. The operands are identical F32, but
  reductions run in a different order, so bitwise equality is not attainable. The
  observed worst case over 24 positions × 7 layers is 3.0e-7 on streams and 8.9e-7
  on logits, with identical argmax at every position.

### Donor decode defect

`Indexer.forward` rebinds `shared_attn.index_k` only when a KV owner completes a
group. During decode, a ratio>1 owner whose group is still filling therefore
scores against the previously published keys. In the production config (ratio-2
owners 2, 8, 14 and ratio-1 owner 20), those are layer 20's per-token keys from the
previous step.

The unmodified donor's prefill and decode paths disagree at every group-starting
position, by max |Δlogit| 0.85 to 0.92. At completing positions they agree to
about 3e-7. `test_donor_decode_index_slot_defect_against_its_own_prefill` pins
this. Elpis implements the prefill and documented semantics, where each owner
reads its own keys, and needed no change. The harness correction
(`owner_binds_index_keys`) only makes donor decode match donor prefill.

## Lifecycle

Sequence-local tower caches are released on every exit path:

- successful `finalize()`;
- typed failure, released immediately;
- an unexpected exception in `begin()` or `next()`, released before the raise and
  with the sequence marked FAILED;
- explicit idempotent `PrincipalSequence.close()`, also usable as a context manager.

Garbage collection is not part of the contract. `PrincipalState` carries no
attention, history, logit or stream state.

## FMS intake

`TensorStore` uses the public `FMSFileAssets.manifest()` and `RangeLease.readinto()`
instead of the provider's private `_assets`. Digest, role, shape, dtype,
byte-range and identity binding checks are unchanged.

## Measured cost (fixture, 1 thread, 200 decode tokens)

- **Decode:** median 10.9 ms per token, p95 16.6 ms. Prefill is about 10.5 ms per
  token.
- **Per-token breakdown:** mHC and norms 4.0 ms, attention 2.25 ms, MoE 2.24 ms,
  Engram 1.38 ms, output head 0.76 ms.
- **MoE detail:**
  - expert materialization: 1.10 ms, of which leases take 0.78 ms and views
    0.32 ms;
  - expert arithmetic: 0.57 ms;
  - routing: 0.20 ms.
- **Materialization change.** One copy of the 24,192 selected-expert bytes per
  token. Leases per token dropped from 79 to 33 (21 experts plus 12 Engram rows).
  Logits and streams are bitwise identical before and after.
- **Memory and FMS:**
  - sequence state is a constant 620,416 bytes, with a scratch bound of
    151,744 bytes;
  - staging high-water is 1,152 bytes against a 64 KiB budget;
  - FMS misses occur only on first use.
- **Admission.** Every expert byte is read twice: once by FMS registration and
  again by tensor verification (40,320 bytes on the fixture). The Engram token map
  takes about 0.55 s to build. Integrity checks are kept, and the second pass is
  recorded rather than removed.

## Limitations

- Single-sequence, single-rank NumPy F32 reference. Lowering to native C/C++
  kernels behind a stable C ABI, with native == NumPy == donor parity, is the next
  phase.
- No learned parameter artifact and no trained Engram address table are admitted.
- Vision, DSpark/MTP and tensor parallelism are excluded, as is bitwise GPU,
  BF16 or FP8 equivalence.
