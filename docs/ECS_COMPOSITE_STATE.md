# ECS composite retained state (ELPISEC1 v1)

Authority: `docs/ELPIS_MISSION.md` (one ECS), `docs/ECS_K1_RUNTIME.md` (K1, unchanged), `docs/ECS_MEMORY.md`
(the memory, unchanged). Contract: `native/ECS/include/elpis/ecs_composite.h` (normative). Library:
`libelpis_ecs_composite` (`native/ECS/src/ecs_composite.c`).

The one ECS retained state is the qualified K1 state `(W, epoch, H, a)` **and** the ECS memory. The composite is
the single object that owns both, mutates both only inside one transaction, serializes both into one fixed-size
envelope, and identifies both with one digest. It computes no law of its own: K1 and the memory compute
everything; the composite stages, binds, serializes and commits.

## Decision: a composite envelope (option B), not a K1 extension (option A)

| | A: extend K1 (ELPISGK1 v2 with memory inside) | **B: composite envelope over unchanged components** |
|---|---|---|
| K1 sources | change `ecsg_k1.c/.h` (digest-pinned by the mission gate, bound by the K1N-v2 attestation) | **unchanged**; K1 used through its public leased ABI |
| ELPISGK1 v1 meaning | a v2 beside it, with K1 identity now covering memory | **unchanged**: the K1 component is a byte-identical ELPISGK1 v1 envelope |
| ELPISEM1 v1 meaning | re-encoded inside K1 | **unchanged**: the memory component is a byte-identical ELPISEM1 v1 envelope |
| K1 qualification | would need re-attestation (numerical code touched) | **untouched**: the K1 law runs exactly as qualified |
| memory without K1 | impossible | possible (free-standing handles keep no authority) |
| one identity, one owner, one commit | yes | **yes** (the composite object) |

B keeps every frozen authority bit-for-bit and adds exactly one new versioned thing: the ELPISEC1 binding.

## ELPISEC1 v1

Little-endian; the size is fixed by the configuration (it never depends on how full the memory is), so a
checkpoint slot can be provisioned once.

| Offset | Field |
|---|---|
| 0 | magic `ELPISEC1` |
| 8 | format version (u32) = 1 |
| 12 | flags (u32) = 0 |
| 16 | K1 envelope bytes (u64) = `elpis_ecsg_k1_envelope_bytes(dim, width)` |
| 24 | memory capacity (u64) = the configuration's largest ELPISEM1 size (`elpis_ecs_mem_required_bytes`) |
| 32 | memory envelope bytes (u64), ≤ capacity |
| 40 | reserved (u64) = 0 |
| 48 | K1 retained-state identity (32): the ELPISGK1 trailer |
| 80 | memory identity (32): the ELPISEM1 trailer |
| 112 | reserved (16) = 0 |
| 128 | the ELPISGK1 v1 envelope, unchanged |
| … | the ELPISEM1 v1 envelope, unchanged, then zero padding to the memory capacity |
| end | SHA-256 of every preceding byte: the **composite retained-state identity** |

The composite identity is a new identity. It is never a K1 identity and never a memory identity (tested).

**Restore** validates completely before returning anything: exact length for the caller's configuration, magic,
version, flags, reserved fields, size fields, recorded component identities against the component trailers,
zero padding, the composite trailer, then each component through its own complete validator (`ELPISGK1` by K1
restore, with the K1 shape checked against the configuration; `ELPISEM1` by memory `restore_into`, which
requires the memory configuration to equal the caller's and refuses unadmitted learned content UNQUALIFIED).
Finally the restored state must re-serialize to the exact input bytes (exact restore). Any other version is
refused (`CORRUPT`): there is no implicit migration.

**Legacy.** A bare `ELPISGK1` v1 envelope is a K1-only retained state. `restore` refuses it (`CORRUPT`).
`import_k1` admits it explicitly as a composite with an empty memory and a new composite identity; its K1
component bytes are preserved exactly. Moving a durable lineage from a K1 identity to a composite identity is
an explicit, authority-checked publication, never a reinterpretation (see "Authority").

**Resource limits.** Memory capacity ≤ `ELPIS_ECS_COMPOSITE_MAX_MEMORY_BYTES` (64 MiB); K1 within its own image
budget; total ≤ `ELPIS_ECS_COMPOSITE_MAX_ENVELOPE_BYTES`. A configuration beyond them is refused `CAPACITY`
(`elpis_ecs_composite_envelope_bytes` returns 0). Allocation happens at create/restore/import (object, K1 arena,
two memory arenas, one envelope staging buffer) and at an explicit `leased_reserve`; `inspect` reports every
figure (see "Resource ledger").

## One owner, one transaction

The composite holds its K1 state under a private lease, so every unmanaged K1 mutation of the component refuses
`LEASED`; the memory handles are private. The only mutation path is the composite's leased transaction:

`leased_txn_begin` → any of `leased_txn_run_schedule` (the qualified K1 experience law on the K1 candidate),
`leased_txn_observe`, `leased_txn_memory_learn` (memory candidate: a copy of the committed memory made at the
first memory operation; LEARN is UNQUALIFIED unless the configuration admits RECURRENT) → `txn_snapshot_write` (the candidate's complete
envelope; its trailer is the identity commit will report) → `leased_txn_commit` (computes the candidate's
identity, commits K1, then installs the memory candidate by pointer exchange, which cannot fail) or
`txn_abort` (installs nothing).

* **No partial publication.** Nothing is visible before commit; reads between transaction calls see the
  committed state. A commit installs both components and the new identity together, or nothing.
* **Refusals.** INVALID, CAPACITY, BUSY, LEASED, UNQUALIFIED and REFUSED (a memory refusal, its status reported)
  are recoverable: both candidates unchanged. STALE and NONFINITE from the K1 schedule discard the whole
  composite transaction (K1's own contract). No refusal changes committed state.
* **Status numbers** extend K1's (-1..-8 identical), so a RuntimeCore substrate table drives a composite exactly
  as it drives K1; `leased_txn_commit_identity` reports in the K1 commit-identity layout with composite
  identities as the state digests.
* **Invariant checks.** The K1 identity reported by K1 at begin/commit/query must equal the component identity
  the composite recorded; otherwise the object poisons itself and refuses everything (`CORRUPT`). K1 is private,
  so this cannot happen without memory corruption.
* **Concurrency.** One writer XOR many readers, non-blocking, no fairness (as K1 and the memory).

## Authority

The composite object provides process-local atomicity only. Durable authority is the established path's, now
format-aware (C ABI and laws in `docs/RUNTIME_CORE.md`, "Composite substrates"):

1. **Continuity** names the authoritative composite identity in a v3 record (`docs/CONTINUITY.md`, "Record v3").
   It holds the identity and its format, never memory content. A lineage keeps its format; only the explicit
   legacy import moves a K1 lineage to a composite one.
2. **Checkpoint (K1 Recovery R0)** holds the complete `ELPISEC1` envelope in a store provisioned for its size
   (`docs/K1_RECOVERY_R0.md`, "Composite envelopes"). Slots are evidence; they never authorize themselves.
3. **RuntimeCore** is the one mutation owner: it holds the composite's lease, so every unmanaged mutation is
   refused `LEASED`; the memory changes only through `turn_observe` inside a managed turn (a LEARN turn, or since
   RuntimeCore ABI v5 an observe-only turn that runs no K1 schedule), whose native commit is volatile (RuntimeCore
   ABI v4 onward: the live coordinate advances; nothing is checkpointed or published). It becomes
   durable only through an explicit durable publication: the composite is captured under the lease
   (`elpis_ecs_composite_leased_capture`: the K1 component unsealed, the memory component sealed by its own
   format), sealed by the pure `elpis_ecs_composite_envelope_seal`, checkpointed, then published. A memory
   transition is therefore durably authoritative under exactly K1's discipline, and never otherwise. (Through ABI
   v3 every turn's candidate envelope was checkpointed before the native commit and published after it; that
   per-turn durability is superseded.)
4. **Recovery**: restart reads continuity's identity, the checkpoint returns the matching envelope,
   `elpis_ecs_composite_restore` rebuilds the state exactly, and binding verifies its identity against
   continuity before anything is built on it. An interrupted durable publication leaves an unresolved candidate
   slot that the operator discards (resume the acknowledged state) or adopts (the captured state): a transition is
   either absent or present once, never duplicated (tested with process death after the slot sync and with the
   continuity record torn). Volatile transitions since the last acknowledgment are lost by a crash.

What this does **not** claim: that RuntimeCore admits memory LEARN or CONSOLIDATE (it does not; unqualified), or
that any memory operation has scientific qualification. A snapshot round trip alone is not crash-consistent
recovery; the recovery claim rests on the checkpoint + continuity tests above.

## Python composition (mechanics)

`elpis.ECS.composite` binds the composite as `elpis.ECS.k1` binds K1 (one native call per method, no ECS law in
Python); its configuration never sets a memory capability, so OBSERVE and LOOKUP are reachable and memory LEARN /
QUERY FAST are not. The managed runtime admits it under the library identifier `elpis_ecs_composite` (math, K1, the
memory and the composite are admitted in dependency order and bound by SONAME to the pinned objects) and drives it
through the existing RuntimeCore substrate kind 3:

* `Runtime.anchor_cognition`, `run_query`, `run_learn`, `publish_recoverable` (a checkpoint store provisioned for
  the composite envelope), `recover_k1` (the ELPISEC1 envelope; `CompositeState.restore`) and `release` work on a
  `CompositeState` exactly as on K1; `ContinuitySnapshot.state_format` is `"COMPOSITE"` for its lineage.
* `run_learn(..., observations=..., observation_authority=ObservationAuthority(...))` carries typed
  `MemoryObservation` values into the same managed turn: admitted at the boundary before anything native (an
  explicit grant, the source identifiers it admits, a per-turn bound of at most 64), observed natively on the memory
  candidate, then committed with the K1 candidate as one volatile transition. All or nothing: a refused observation
  (a stale tick, capacity, a malformed record) aborts the whole turn.
* `run_observe(substrate, observations, observation_authority)` (RuntimeCore ABI v5) records typed evidence with
  **no K1 learning**: the same admission, then an observe-only managed turn (no experience schedule), one volatile
  commit in which the K1 component is byte-identical and only the memory moves (`docs/RUNTIME_CORE.md`, "Observe-only
  turns (ABI v5)"). All or nothing, as above; the native per-turn bound is the authority's.
* `Runtime.import_composite` (an operator operation) is the explicit legacy move of a K1 lineage to a composite one.
* The direct continuity adapter (`elpis.continuity.ContinuityStore`) still binds K1 lineages only: it refuses to
  read, or encode, a composite record.

Tests: `tests/integration/test_composite_runtime.py` (ownership and lease refusal, atomic observe + commit, every
refusal installing nothing, exact snapshot / restore, bounded allocation and checkpoint footprint, restart losing
exactly the volatile work, K1 / composite non-substitution, the explicit import, unadmitted libraries, format-aware
readers) and `tests/integration/test_composite_process_kill.py` (real SIGKILL inside a composite publication: the
memory transition is present exactly once when continuity authorizes it, never the volatile one). Mechanics only:
nothing decides what to observe, and an observation is evidence, never truth or a learning instruction.

## Tests

`ECS.test_ecs_composite` (every native job, ASan+UBSan, TSan): layout and identity (components byte-identical to
K1 and the memory alone); K1-only, memory-only and joint transitions (each component equal to the same law
applied alone); interrupted transitions (abort, wrong tokens, recoverable refusals leave the candidate envelope
bitwise unchanged, NONFINITE discards, destroy with an open transaction); exact restore and deterministic replay;
corruption (every header byte and a stride of all bytes flipped; per-field flips with a resealed trailer),
mismatched component identity, unsupported version, flags, magic, truncation, another configuration, the old
format (refused by restore, admitted by `import_k1`), capacity, capability admission (UNQUALIFIED restore of
learned content), leases, and readers racing a committing writer (no torn envelope).
