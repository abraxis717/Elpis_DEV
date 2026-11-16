# ERSA R8 — Genuine SSC R0 read-contract seam, native dual reducer

**Qualification ceiling:** `PASS_SSC_PINNED_READ_PROJECTION_AND_NATIVE_DUAL_REDUCER_ONLY`.

This is *not* complete SSC state-bank conformance, autonomous semantic selection,
trained recurrence, or a GDN replacement. It is a verified prerequisite with
explicit unmapped surfaces and two discovered predecessor defects.

## What is real

- The exact SSC predecessor source bundle supplied by Ouroboros (seven files)
  is included under `ssc_predecessor/src/elpis_ssc/` with frozen SHA-256 pins.
  Original SSC `contract_r0.py`, `reference_backend.py`, `oracle_r0.py`,
  `harness_r0.py` and `geometry_r0.py` are used directly in the R8 tests.
- The R7 ERSA implementation is preserved verbatim under the original paths;
  `FROZEN_R7.sha256` pins 43 predecessor files. R2's four original pins are
  checked against the read-only host source tree.
- `integration/ssc_r8.py` creates an *explicit, bounded, one-to-one* mapping
  from admitted SSC string relation IDs to ERSA numeric IDs. It projects only
  already-committed, canonically transported SSC relation vectors. Source
  provenance, ticket, participant, role, ordinal, flags and protected status
  are verified before projection. No implicit string hashing, geometric
  identity inference, unsolicited writes, or stale snapshot reuse.
- `MonitoredSscSession` invalidates a captured projection after any SSC
  operation, including reset with bitwise-identical contents. It is an
  in-process coherency protocol, not isolation from malicious same-process
  callers or an external trust root.
- A separate C-native stateless `ersa_ssc_softmax_r8` implements SSC's
  **norm-softmax salient view plus uniform collective mean** over already
  canonicalized vectors. This is *not* ERSA's signed L1 reducer, which remains
  frozen in R7 and is deliberately tested as non-equivalent.
- End-to-end bounded source projection/native single-relation reads are
  checked against the SSC predecessor reference and independent oracle.
  Multi-relation SSC reducer differentials include randomized source vectors.
- All 377 R7 tests plus 88 new SSC/bridge/adversarial tests are run, together
  with the 8,192-slot retained-protection stress.

## Known blocking incompatibilities — do not turn into PASS

1. The SSC frozen `reference_backend.step(action_budget)` clips a too-large
   budget, returning OK, whereas the independent SSC `oracle_r0.step()`
   rejects it with STEP_BUDGET. R8 reproduces this divergence. Original source
   is not mutated; full predecessor parity is still HOLD.
2. Both frozen SSC reference and oracle accept `decay(-2)`, decrementing tick.
   Agreement is not valid monotone-time semantics. R8 preserves that failure
   as an explicit negative control; any future production adapter must reject
   negative time advances by its *versioned* stronger contract.
3. The seven-file collector omitted `transport_reference.py`, imported by the
   original `fixtures_r0.py`. R8's small test-only identity transport exercises
   the real SSC contract and oracle but is **not** the original gauge fixture
   source. Full official SSC fixture replay cannot be claimed yet.
4. SSC `read_exact` and `upsert_exact` operate on entity addresses that do not
   equal ERSA relation IDs. SSC observe/rolling KV, convolution state,
   proposal ticket lifecycle, SSC mutation and eviction, and full SSC trace
   behavior are *not* implemented by the ERSA read projection. The SSC
   predecessor reference remains state authority in tests.
5. An SSC live record whose dynamic protection deviates from immutable
   admission protection is refused rather than falsely mapped.
6. Foreign aliases that mutate the SSC backend without the monitored wrapper
   cannot guarantee detection of an identical-state reset. The test harness
   uses the monitored owner exclusively; no process isolation claim.

## Run on Ouroboros

Extract this ZIP into `/dev/shm` (never use it as a persistent project install).
Export:

    PYTHONDONTWRITEBYTECODE=1
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    CUDA_VISIBLE_DEVICES=-1
    TMPDIR=/dev/shm
    HOME=<tmpfs extraction directory>
    XDG_CACHE_HOME=<tmpfs extraction directory>
    PYTHONPATH=<tmpfs root>:<tmpfs root>/src:<tmpfs root>/native:<tmpfs root>/ssc_predecessor/src

Then run `python3 -B tools/qualify_ersa_r8.py` from the extraction root.
The runner verifies both host predecessor source sets, all bundle manifests,
executes the 465 tests, runs the capacity stress and prints `=== SEND BACK ===`.
No host source files or weights are deliberately changed, but an OS-level
write-prevention attestation has not been performed.

## Next direct engineering work

Close the remaining SSC state-operation semantics with a versioned adapter
and the *actual* `transport_reference.py`, then implement the native
bounded state-update path; maintain independent oracle comparison and strict
separation between SSC norm-softmax and ERSA signed L1. Only after complete
state/operator semantics should learned semantic selection or a neural runtime
be promoted. No renewed M5X/provenance side branch.
