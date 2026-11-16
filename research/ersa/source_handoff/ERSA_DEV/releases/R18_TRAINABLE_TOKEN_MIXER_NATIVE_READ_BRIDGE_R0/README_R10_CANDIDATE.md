# ERSA R10 — Native RAM-state authority, host-operated SSC transition bank

**Local qualification scope:** `PASS_NATIVE_CAS_STATE_AUTHORITY_AND_14OP_HOST_TRANSITION_DIFFERENTIAL_ONLY` (subject to host reproduction).

This is a **direct advancement of the ERSA substrate**. No M5X or source-authentication expansion. No trained model or GDN dependence.

## What changed from R9

- R9's SSC `ReferenceBank` no longer owns the R10 candidate's live state.
- A new native C image owns the **entire per-session state**: entities, admitted relations, controls, token/KV window, conv placeholders, utility, logical time, pending proposal identities, and counters. The image is a preallocated, bounded C RAM buffer; its generation is a monotonic CAS counter. A commit is one atomic replacement under a C mutex. Failed writes leave all native state bytes and the generation unchanged. Reset clears the logical image, not the native CAS generation.
- All **14 SSC method names** are available on `NativeSscAuthorityR10`, and an independently authored Python **host transition implementation** operates on ephemeral decoded copies of the native image before committing. **Native authority does not imply native C execution**. The host is still used for validation, policy and transport operations; a fully C-native SSC state machine remains UNQUALIFIED.
- Strong references to in-flight proposal objects exist outside the C image only to prevent Python object-ID ABA reuse; their authoritative issued/committed status lives in the bounded native image. The references are capped at 8,192 and reset-cleared. This explicitly extends the frozen SSC reference with bounded pending-identity admission; a true long-running proposal reclamation protocol is still open.
- A pthread mutex protects C-native CAS and read copies. Direct `NativeAuthority.close()` must not race a direct foreign-API use-after-close; owner-level method calls serialize `close()` against operations.
- Reject over-cap `step` and negative/nonintegral `decay` without advancing time, while invalidating read generations.
- SSC `ReferenceBank` is **not imported by the candidate runtime**. The independent `OracleModel` is test-only and compared after every operation across 18 full 14-op traces and 12 randomized episodes. Original R9 and earlier test suites remain unchanged and are rerun.

## Capacity and performance tradeoff

- C state-image allocation is fixed to a configured maximum (default 16 MiB; hard maximum 32 MiB), with independent SSC slot ceilings up to 8,192 entities and 8,192 relations. The entire image is decoded and re-encoded for each operation. This is **NOT** a high-performance native neural memory runtime, and large-capacity per-operation latency is not qualified. R9's fixed-slot native stress test still runs separately.
- Token and control windows have hard bounds; proposal identities are capped at 8,192. No unbounded per-session ledger is admitted.
- External registry, transport and write-authority are host-owned fixed contract inputs, not independently attested. Python host metadata and temporary copies necessarily use additional volatile memory beyond the fixed C image.
- No persistent runtime state, checkpoint, GPU, network, or training. No source changes to SSC or frozen R8/R9.

## Known limitations / exact next gate

1. R10's **native authority** is the C session image, but transitions are Python-host-executed and serialize the entire image per operation. **Not a fully native SSC state-operation implementation**. Implement in-place native C entity/relation/ticket/clock/KV/conv transitions, then directly differential-qualify all 14 operations.
2. The original `transport_reference.py` has not been frozen; official SSC fixture suite remains HOLD. R10 uses the actual seven pinned SSC contract/oracle files and the R8 test-only transport.
3. Full long-horizon parity is limited by explicit bounded proposal identity admission and not yet checked on model-scale episodes; 8192 *native image capacity* does not establish performance at that size.
4. Unscoped semantic selection and neural recurrent integration remain HOLD. Production trust, rights, and OS NVMe-write attestation remain HOLD.

## Local reproducibility

Unzip the ZIP under `/dev/shm`, verify `FILES.sha256`, run `python3 -B tools/qualify_ersa_r10.py`. The runner independently verifies frozen R9/R8/R7, original SSC/R2 pins (including external source files), runs **all** tests and rechecks source hashes afterward. All C shared objects are compiled under `/dev/shm`.

No installation, no runtime background services, no modification of `/mnt/primesauce` project state.
