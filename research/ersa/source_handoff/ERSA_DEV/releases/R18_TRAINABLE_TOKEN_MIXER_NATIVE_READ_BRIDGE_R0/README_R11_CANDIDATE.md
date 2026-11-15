# ERSA R11 — Typed native SSC state machine (candidate)

**Qualification boundary:** `PASS_TYPED_C_NATIVE_14OP_STATE_TRANSITIONS_AND_FROZEN_SSC_ORACLE_DIFFERENTIAL_ONLY` if the entire pinned package reproduces on Ouroboros.

Direct continuation of frozen ERSA R10. It is **not** a complete GDN replacement, autonomous selector, GPU neural runtime, or production system.

## Difference from R10

- R10 kept SSC session state in a fixed C-owned opaque JSON image, but decoded and recomputed *every* SSC transition in Python.
- R11 creates separate bounded native C arrays for entities, relations, control records, exact token/KV windows, convolution slots, and issued/committed proposal tokens.
- **All 14 SSC state operations execute typed state transitions in C** (not Python callback transition, and not whole-image JSON serialization).
- C owns the state, bounded replacement/eviction, Hebbian utility updates, logical clocks, proposal liveness and native norm-softmax calculation. The read/commit path is generation-checked under a pthread mutex. All potentially failing operation conditions are checked before irreversible mutation. SSC-compatible rejection telemetry is retained; rejecting a request may legitimately change the bounded rejection/control counters.
- The host still validates the frozen SSC Python contract objects, the immutable admission registry, write-authority string, and frame/transport mapping; it converts them into bounded native value arguments. **A malicious host or raw C-ABI caller is outside the demonstrated trust boundary.** The original SSC reference backend and OracleModel are never imported into the R11 runtime path; they appear only in independent tests.
- `begin_session` and `reset` clear C data without resetting the monotonic C generation; proposal object strong references are RAM-only and bounded.
- R11's C-owned typed arrays and `struct` layout are an experimental ABI; no stable binary ABI is claimed.

## Local validation

- Frozen R10 package and all older source manifests are retained and recomputed before and after execution.
- 604 inherited reference, SSC, adversarial and native-shadow tests are rerun without alteration.
- 68 additional R11 tests: full 14-operation oracle comparisons over 18 seeds, 12 randomized 45-step baselines, 12 randomized 90-step relation-lifecycle traces, all policies, malformed requests, clock rejection, stale/duplicate proposal tokens, capacity 8192 + 64 inserts, concurrent owner calls and reject-on-nonfinite aggregate output.
- CPU read microbenchmark compares R10 full-owned-state exact reads to R11 typed C exact reads at N=256/1024; at N=8192 only R11 is timed to avoid an unqualified R10 full-state load cost. Results are reference diagnostics, not trained attention throughput.

## Remaining HOLDs

1. SSC's `transport_reference.py` historical file was not frozen; official fixture suite remains unqualified.
2. The host-controlled immutable registry, trusted write-authority, input validation, and transport computation remain external dependencies. Not full freestanding C-native SSC ingestion or verified security attestation.
3. Long-horizon learned/semantic retrieval utility is unqualified; R5's open-world relevance failures remain.
4. The neural recurrent ERSA model, training, GPU execution, and model-scale decode/prefill tradeoffs are still unqualified.
5. Native state memory bounds are enforced; **OS-level prevention of all persistent writes is not attested**. Execution package is designed to run from tmpfs and never modify the canonical Elpis tree.
6. Proposal identities remain bounded to 8192 per session, with no autonomous compaction prior to reset.

## Scope of the next phase

Complete the native runtime admission/conformance seam by freezing the missing official transport, then address learned semantic selection and a recurrent neural benchmark under fixed compute and RAM budgets. Do not revive the MemNet provenance digression.
