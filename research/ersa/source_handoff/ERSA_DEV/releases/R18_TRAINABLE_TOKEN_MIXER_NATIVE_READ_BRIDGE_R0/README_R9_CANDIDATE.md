# ERSA R9 — Versioned SSC state-operation guard and native exact-state shadow

**Scientific ceiling:** `PASS_SSC_VERSIONED_REFERENCE_GUARD_AND_NATIVE_EXACT_SHADOW_ONLY`.

This phase deliberately targets the unfinished **ERSA deterministic substrate**. No
M5X work, learned controller claims, signatures, source attestation, or semantic
selection claims are added.

## Implemented

1. **Frozen predecessor preservation.** All 59 R8 own files, 43 earlier R7
   files, seven real SSC source files, and four original ERSA R2 source paths
   are hash-verified. Frozen source files are unchanged.
2. **14-operation monitored reference adapter.** `integration/ssc_r9.py`
   forwards actual SSC operations through the R8 monitored owner, checks
   frozen registry/spec/transport identity and source snapshot, increments
   its view epoch for every attempted operation, and fails closed on out-of-
   band mutations. SSC's genuine `ReferenceBank` remains authoritative.
3. **Two concrete predecessor corrections at the new boundary.** Rejects
   `step > max_steps_per_call` instead of reference clipping and rejects
   negative or nonintegral `decay` before reaching either defective
   predecessor. Invalid attempts invalidate captured R8 projections.
4. **Native fixed-capacity exact-address shadow.** `native/ersa_state_r9.c`
   stores finite signed 4-vectors in preallocated memory with a maximum
   8192 slots, validated writes, delete/reinsert generation increments,
   stale-generation rejection, full-capacity rejection, and strict address
   namespace. No external/native authority or background execution.
5. **Native snapshot publication is all-or-nothing.** Rows from the SSC
   authoritative snapshot are staged in a new bounded C instance, verified,
   then swapped. The old native state remains untouched on rejected rebuild.
   **This is NOT a two-phase atomic commit across SSC and the shadow.** A
   failure after a successful SSC op holds the guard; it cannot undo SSC.
6. **Independent checks.** 12 valid 14-op SSC reference-versus-oracle trace
   seeds, 8192-entry C-native admission/replacement, randomized dictionary
   differential, ABA protection, failure atomicity, frozen predecessor bug
   reproductions, external alias detection, and all 465 inherited R8 tests.

## Unqualified boundaries

- The C shadow is not the SSC source of truth. SSC's entity state, relation
  lifecycle, KV/conv state, tickets, utility, and eviction are still executed
  by the original Python reference backend. No full native SSC state-bank
  parity or full GDN replacement is claimed.
- 14 SSC operations are guarded/forwarded; **14 C-native SSC operations are
  not implemented**. The shadow supports only exact entity address/value
  operations (no native relation lifecycle).
- SSC's true `transport_reference.py` is not present in the frozen SSC ZIP.
  An Ouroboros run reported its SHA without freezing the file in this package.
  The original SSC fixture suite is therefore not certified.
- Runtime provenance, learned autonomous relevance, full neural recurrent
  execution, rights, and production trust remain HOLD.
- Guards assume exclusive in-process ownership of the SSC reference object.
  A hostile alias can mutate it between checks; OS isolation is not certified.
- Fixed-capacity C slots make memory bounded, not necessarily latency-optimal:
  upsert uses bounded linear search, and guard snapshot rebuild is not an
  optimized incremental native state store.

## Qualification on Ouroboros

Extract to `/dev/shm`, verify `FILES.sha256`, and run:

`python3 -B tools/qualify_ersa_r9.py`

Environment: `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`TMPDIR=/dev/shm`, `HOME=<extract dir>`, `XDG_CACHE_HOME=<extract dir>`,
`CUDA_VISIBLE_DEVICES=-1`. Compilation uses `cc` and writes `.so` only under
`/dev/shm`. Never unzip into canonical ERSA or SSC source directories.

## Next direct work

The *next* architecture gate must replace the Python SSC state bank with an
independent bounded native backend covering entity/relation lifecycle, proposal
operations, eviction, clock, KV/conv state and receipts. Compare all 14
operations to the SSC independent oracle, keep the two corrected semantics
versioned and disqualify any divergence. Then qualify autonomous selection and
neural runtime against GDN on matched tests. No additional metadata branch.
