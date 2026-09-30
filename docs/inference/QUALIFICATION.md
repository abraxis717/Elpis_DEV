# Verbal boundary qualification — 2026-09-30

**Disposition: `MECHANICS_NONQUAL`.** The real tokenizer and text composition
are qualified. Production driver/intake is incomplete. No compatible trained
model was instantiated and no neural language-generation claim is made.

## Required evidence

1. **Starting Elpis main:** `774c04e33654d136b8abbd4bd1dc90e9eb4980c4`;
   tree `be2159b7781cdb7ef619a8596bc3af66992a0917`.
   Fresh clean `main == origin/main`, including PR #7. Remote DNS verification
   was unavailable. [Source inventory](evidence/starting_authority.json).
2. **Fresh DeepSeek recipe:** `8cadfede7063c896b944e7bae05daa3549ae97ea`;
   tree `3ed5db20847e09a9f3d157660e27c8d2c3141fc3`. Checkout remains unmodified.
3. **Tokenizer SHA-256:**
   `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
   Extracted and fresh recipe copies were independently hashed and match.
4. **Provenance:** exact recipe revision above; recipe records upstream
   `DeepSeek-V4-Flash-Vision-Exp` comparison revision
   `6821d6ad3681a4b137b066b76094fa82ebd0a380`, not an original import revision.
   The only recorded change is the image token reassignment at 129264.
   MIT notices retained; [full details](DSV41_TEXT_BOUNDARY.md).
5. **Files changed:** [exact file list](evidence/changed_files.txt). Inference `text.py`, `production.py`, `admission.py`;
   runtime `text.py`, `composition.py`; optional dependency metadata;
   four new test modules and one benchmark; system authority/nonclaims;
   provenance, two MIT notices, this documentation and evidence records.
   No changes to `principal.py`, synthetic driver/fixtures, DSV4/Qwen address
   arithmetic, HACF, ECS, native executor or other native source.
6. **Model assets:** [complete observed inventories](evidence/local_assets.json),
   with shapes, offsets, dtypes, geometry, hashes and binding refusals.
   DeepSeek 1B has a different tokenizer, 307 BF16 + three I64 tensors, four
   layers, 16 routed + one shared expert per layer and no engram tensors.
   ElpisPrimeConsolidated (60 tensors) and Dimple initializer (20 tensors) are
   not V4.1 language towers. All inspected assets remained in place/read-only.
7. **Identities:** tokenizer
   `a4ff8fefe87f25992ad0d20def12cc29d4fcbb06549d35e6f3329778e7ea16e1`;
   renderer `elpis.inference.render.deepseek-recipe-v41-text.v1`.
   Production target/model: **NONE ADMITTED**. Synthetic tokenizer remains
   `synthetic-raw16-DSV41_ENGRAM`; no identity was reinterpreted.
8. **Tokenizer differentials:** five frozen prompt fixtures matched unchanged
   Rust `render_message`/`reasoning_effort_template` bodies extracted from the
   pinned recipe in a dependency-free text-only harness. Shell types and the
   unused tool path are stubbed; this is not a full framework differential.
   Their token IDs matched official `tokenizers 0.23.2` over the exact artifact.
   [Both ID sequences](evidence/tokenizer_fixtures.json) are retained.
   All 129,279 non-stop vocabulary entries matched official single-token decode;
   300 deterministic random sequences of 40 IDs matched whole-sequence decode.
   Unicode, byte splits, invalid/final partial UTF-8, BOS/EOS, role/control flags,
   stop handling, content rejection and long incremental streams passed.
   Full Cargo recipe build was attempted offline but tokenizers 0.23.2 was not
   cached; the full application/framework differential remains unrun.
9. **Regression:** full Python run: **2,587 passed, 10 skipped, one deselected**
   in 233.02 seconds. The ten donor tests subsequently **all passed** against
   the exact, clean migration-basis checkout. Existing synthetic DSV4/Qwen,
   principal, transaction, sequence, steering, replay and integration tests pass.
   The deselected existing network-guard test failed in the first run because
   this sandbox denies AF_INET socket creation before the guard can be tested.
   Its source was not weakened. Raw initial failure and final logs are retained.
   Native: **92/92 passed**, fresh Release build, native libraries required.
   Final focused tokenizer/intake/text suite: **32 passed, zero skipped**.
10. **Real-model intake:** pinned DeepSeek 1B safetensors passed read-only
    SHA-256/storage-span inspection and production binding refused
    `UNSUPPORTED:PRODUCTION_DRIVER_UNIMPLEMENTED: DeepseekV4ForCausalLM`.
    No weights were cast, reshaped, adapted, copied or executed. Generic storage
    validation is not a complete tensor/architecture/numerical model contract.
11. **Generation transcript:** none. The integration suite's forced output
    `Transport probe: 你好 👋🏽` comes from a named `TokenEmissionTestDouble` with
    no neural parameters, exclusively for text transport/finalization/replay.
    It must not be reported as model output. Real prompt `Hello, Elpis.` encodes:
    `[0, 128803, 19923, 14, 3909, 62848, 16, 128804, 128822]`.
12. **Performance:** [raw tokenizer measurements](evidence/tokenizer_performance.json).
    Median tokenizer load/verify/precompute 621.69 ms (boundary only);
    1,600-byte, 608-token chat encode 1.105 ms; 600-token incremental decode
    0.449 ms total / 748 ns per token. Per-token timed p95 790 ns.
    Byte-material tuple plus bytes objects 6,175,790 bytes (not total RSS).
    UTF-8 carry ≤3 bytes. No neural tokens/s, prefill latency, working-set,
    FMS hit/miss or expert/page measurements exist for a production model.
    PR #7 native source is byte-identical; native tests passed.
13. **Hot loop:** the new integration test traps existing HACF, Regex, ECS
    receipt-history, structural resolution, RootCapability opens, canonical
    and raw identity functions, tokenizer loading/encoding and file reads after
    sequence begin; `Recorder` observes zero identity/hash calls during token
    generation plus incremental decoding. Existing resident synthetic neural
    hot-loop proof also passes. Decoder has no token history/full-prefix decode,
    global lock or authority source. Runtime does decoding alongside output,
    and records principal/text identities only after finalization. Arbitrary
    caller-supplied Python emitters remain the caller's trusted output boundary.
14. **Development branch:** `feat/dsv41-text-boundary`, based only on the starting
    public main. Final commit/tree are given in the delivery message; this report
    cannot contain its own Git commit hash. No remote push/publication occurred.
15. **Blocker:** the public single-layer synthetic equations are not a compatible
    trained production tower. Architecture-specific multi-layer arithmetic,
    numerical/layout intake and trained addressing/parameter artifacts are
    still required. This is not yet a parameters-only blocker, nor a model
    quality failure. No canonical intended trained tower manifest was located.
16. **Exact next task:** establish the intended principal-compatible DSV4.1 tower
    manifest and implement/qualify its layer, attention/O, router/shared-expert,
    addressing/projection and numerical mapping against the reference; bind its
    compatible trained artifacts via substrate/FMS, then run several deterministic
    GREEDY prompts with full latency/residency/activity and language transcripts.

## Reproduction

All persistent input assets were read-only. Build/test/log files were under
`/mnt/primesauce/ChatGPT_Sandbox/Astra/work/verbal-boundary` and
`/mnt/primesauce/ChatGPT_Sandbox/Astra/logs/verbal-boundary`.
Qualification used Python 3.11.9/NumPy 1.26.4/pytest 9.0.2 from
`/mnt/primesauce/Elpis_DEV/toolchain/venvs/elpis-release-py3119` and a local copy
of tokenizers 0.23.2 from an established `/mnt/primesauce/models/...` runtime.
The delivered package uses optional dependencies; no `/home/joe` path or
external framework runtime is required by the implementation.

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j 8
ctest --test-dir "$BUILD" --output-on-failure -j 4
export ELPIS_NATIVE_BUILD="$BUILD" ELPIS_REQUIRE_NATIVE=1
export ELPIS_V41_TOKENIZER=/absolute/path/to/tokenizer.json
export ELPIS_DEEPSEEK_RECIPE=/absolute/path/to/pinned/recipe
python -m pytest -k 'not test_network_guard_is_active' -rs
ELPIS_DONOR_ROOT=/absolute/path/to/migration-basis/donor \
  ELPIS_REQUIRE_DONOR=1 python -m pytest tests/boundary/test_donor_parity.py
python -m tests.inference.text_boundary_benchmark "$ELPIS_V41_TOKENIZER"
```

The socket exclusion is local to this sandbox run, not a changed repository
test/CI policy. Normal environments should run the test too. The tokenizer and
reference tests explicitly skip without their external artifacts; an unconfigured
CI run is not evidence of tokenizer qualification.

Generated-content admission into HACF and canonical ECS projection into context
remain explicit post-sequence/pre-sequence tasks. This work grants no model
output semantic/mutation authority and establishes no geodesic-intelligence claim.
