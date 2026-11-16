# ERSA R6 — stateless native ranking / signed-aggregate execution seam

**Purpose:** Demonstrate exact deterministic top-K selection from *prevalidated, host-canonically-rounded* scores and a C-native signed R4 reduction, directly compared against the frozen ERSA R5 reference. This is the first narrow native CPU operator, **not a full neural runtime or a GDN-drop-in**.

## What changed

- `native/ersa_native_r0.c`: bounded C11 ABI for 8-feature scoring, stable top-K ranking, signed R4 aggregate with mean fallback. N <= 8192, no global state, no storage, and output buffers committed only upon successful validation. C source is project-original and uses no NVIDIA GDN source.
- `native/ersa_native.py`: stricter NumPy/ctypes boundary with CPU-only compilation into `/dev/shm`. Never copies compiled binaries into persistent model or project directories.
- `tests/test_r6_native_operator.py`: 50 new pytest variants including all 12 frozen R2 semantic fixtures replayed via R5 reference, randomized scoring and signed reductions, invalid-input rejection, duplicates, capacity, sign retention, gauge-covariant reduction, rank tie-breaking and a deliberately-wrong magnitude-only mutant.
- `tools/r6_native_bench.py`: end-to-end-neutral, kernel-only cost diagnostics for 8,192 candidates, precomputed canonical scores, and 4D reductions.
- `tools/qualify_ersa_r6.py`: fail-closed source pinning, 296 total tests, R5 metric and capacity replay, native benchmark, SHA-256 result and compact SEND BACK.

## Invariants and limitations

- Source authority / admission / stale-revision checks remain in the frozen R5 Python bank; native operators are invoked **after** those checks and confer **no authority** to read a relation.
- The ranking ABI accepts *already canonicalized scores*; R5 uses Python `round(score,12)` as its total-order contract. C's 8-feature scorer has demonstrated numerical agreement but is not yet certified for bitwise rounding parity at boundary ties. Do not substitute raw C scores for canonical scores without a new gate.
- Native ranking and reduction are **kernel-level**; full Python bank admission, index traversal, feature extraction, scope authorization and SSC mapping are not native. Native end-to-end throughput improvement is therefore **not established**.
- The original R2 fixture still exhibits unscoped semantic mismatch; no change to test data or oracles. Complete candidate coverage is not proof of autonomous relevance; SSC predecessor parity requires a verified source not contained in the uploaded source archive.
- Builds only `.so` files in temporary `/dev/shm`, no persistent runtime state, no GPU, no model training. OS write-prevention attestation is not claimed.

**Qualification ceiling:** `PASS_NATIVE_STATELESS_RANK_AND_SIGNED_REDUCE_DIFFERENTIAL_ONLY`

**Next integration gate:** real SSC predecessor namespace contract + native bounded feature extraction + trained recurrent selection, with negative controls preserved. Do not promote R6 to a complete neural substrate.
