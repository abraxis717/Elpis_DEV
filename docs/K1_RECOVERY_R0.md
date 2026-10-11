# K1 Recovery R0: a bounded, continuity-authorized K1 checkpoint owner

**CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.**

Status: implemented systems mechanism (`native/runtime/src/checkpoint.rs`, RuntimeCore ABI v3, `elpis.runtime.recovery`).
UNQUALIFIED: this document specifies the mechanism and its tests; it does not qualify it. Qualification
specification: `docs/qualification/K1_RECOVERY_R0.md` (NOT RUN).

## Why a distinct owner

Before R0 a restarted runtime knew *which* K1 retained-state identity was authoritative (continuity holds a 32-byte
digest) but had no durable copy of the state itself: an in-memory K1 lost with its process was unrecoverable. The
two existing durable owners cannot hold it:

* **continuity** (`native/continuity`, docs/CONTINUITY.md) is the current-authority register: two fixed 176-byte
  slots holding identities, not state. Turning it into a state store or an event log is prohibited.
* **FMS COLD** is scratch residency: it reclaims orphans of a crashed owner on open. Scratch is not a checkpoint
  and must never become checkpoint authority.

R0 is therefore a third, distinct owner with exactly one job: keep the complete `(W, epoch, H, a)` envelope
(`ELPISGK1`, its SHA-256 trailer = the retained-state identity) of the authorized state and of at most one newer
candidate.

## Physical bound

| Property | Value |
|---|---|
| Files | exactly `k1-checkpoint.a` and `k1-checkpoint.b` in one directory |
| Size | each `128 + E` bytes, where `E` is the admitted K1 shape's envelope size (`K1Library.envelope_bytes(dim, width)`) |
| Created by | the operator only (`elpis_runtime_checkpoint_provision`, `elpis.runtime.recovery.provision_k1_checkpoint`): a new directory, both files fully written with zeros and synced; an existing directory is refused |
| Written by | RuntimeCore only, in place: never created, grown, shrunk, renamed, multiplied or unlinked by the runtime |
| History | none: two slots, the authorized one and the one being (or last) written |
| Ownership | one runtime at a time (exclusive `flock` on the directory); a second attach is refused |

A state of another shape is refused (`CHECKPOINT_SHAPE`), never resized for. The persistence registry classifies
the slots as `FIXED_CAPACITY_AUTONOMOUS` and provisioning as `OPERATOR_EXPLICIT_BOUNDED`
(`elpis.runtime.persistence`). ZERO AUTONOMOUSLY EVER-EXPANDING ELPIS-OWNED PERSISTENCE holds: the number of slot
files and their bytes are independent of the number of anchors, LEARNs, QUERYs and restarts.

## Slot format

```text
  0   magic "ELPK1CK\x01"
  8   format version (u16) = 1, then 6 reserved zero bytes
  16  slot generation (u64, >= 1; the newest slot has the highest)
  24  envelope length (u64) = E
  32  retained-state identity (32): the envelope's own SHA-256 trailer
  64  reserved, zero (32)
  96  SHA-256 over "elpis.k1-checkpoint.slot.v1\0" || bytes 0..96 || envelope
  128 envelope (E bytes)
```

A slot is valid only if every field checks, the slot checksum matches, and the envelope's trailer is the SHA-256 of
the envelope before it and equals the header identity. An all-zero header is an empty slot; anything else (a torn
or corrupted write) is invalid and is simply not a checkpoint. Validity is integrity, never authority.

## Protocol

With a store attached (`RuntimeConfig.k1_checkpoint_dir`; `elpis_runtime_checkpoint_attach`):

* **Anchor.** Before continuity anchors the lineage, the anchored state's complete envelope
  (`snapshot_write`, read-only) is verified (its identity equals the identity being anchored) and written into a
  slot and synced. A refusal anchors nothing.
* **LEARN.** After the schedule ran on the native candidate and before the native commit, the candidate's complete
  envelope (`txn_snapshot_write`, read-only, under the transaction) is written into the slot that does **not** hold
  the authorized identity, and synced. Then the native commit; its committed identity must equal the checkpointed
  one, or the runtime fail-stops (`CONTINUITY_STATE_MISMATCH`) and publishes nothing. Then the one continuity
  publication. A commit refused natively withdraws the candidate slot (best effort). The authorized slot is never
  touched by a LEARN.
* **Refusal while unresolved.** A LEARN is refused (`CHECKPOINT_UNRESOLVED`, no fail-stop) while an unresolved
  candidate exists. QUERY stays available (read-only, verified against continuity).

The checkpoint adds one envelope copy and one slot write + `fdatasync` to every LEARN. That is the explicit
price of recoverability; a runtime without a configured store keeps the hot path of docs/RUNTIME_CORE.md.

## Recovery dispositions

`Runtime.recover_k1()` / `elpis_runtime_checkpoint_recover` is read-only. Continuity decides; the slots are
evidence.

| Continuity | Slots | Disposition | What may happen |
|---|---|---|---|
| unanchored | anything | `NOTHING_TO_RECOVER` | nothing; a slot never creates a lineage |
| anchored at D | a verified slot holds D, no newer slot of another identity | `RESUMABLE` | restore the returned envelope (`K1State.restore`); the next QUERY/LEARN binds it after verifying it against D |
| anchored at D | a verified slot holds D and a newer verified slot holds C != D | `CANDIDATE_UNRESOLVED` | nothing implicit; LEARN is refused until an operator discards C (resume D) or adopts C (continuity D -> C) |
| anchored at D | no verified slot holds D | `CHECKPOINT_MISSING` | the authorized state is not recoverable from this store; no other slot is a candidate |

There is no silent roll forward (a candidate never becomes authority by being newer or valid) and no silent roll
back (an authorized identity without its envelope is reported missing, never replaced by an older or other slot).

### Operator reconciliation

* `discard_k1_candidate(C)` / `elpis_runtime_checkpoint_discard`: withdraws exactly the unresolved candidate `C`
  (its header is zeroed and synced). Any other identity is `CHECKPOINT_INVALID`.
* `adopt_k1_candidate(C)` / `elpis_runtime_checkpoint_adopt`: re-reads and re-verifies `C`'s complete envelope,
  then continuity publishes `D -> C`. Refused while a K1 state is bound (`RUNTIME_TURN_OPEN`); a publication
  failure fail-stops. Adoption is the operator's decision that `C` is the lineage; the checkpoint never makes it.

Both are `OPERATOR_OPERATIONS` in `elpis.runtime.persistence`.

## Crash matrix

Deterministic fault points (testing library: `elpis_runtime_testing_checkpoint_crash`, continuity fault injection)
are proven in `native/runtime/src/tests.rs` (`mod recovery`, in-memory K1 stand-in) and over real native K1 in
`tests/integration/test_k1_recovery.py`.

| Process dies / fails ... | Durable state | After restart |
|---|---|---|
| during the candidate slot write (torn) | authorized slot intact; the other slot invalid | `RESUMABLE` at D |
| after the candidate checkpoint, before the native commit | D authorized; C checkpointed | `CANDIDATE_UNRESOLVED` (C) |
| after the native commit, before publication | D authorized; C checkpointed (and committed in the lost memory) | `CANDIDATE_UNRESOLVED` (C) |
| publication refused / uncertain and lost | D authorized; C checkpointed | `CANDIDATE_UNRESOLVED` (C) |
| publication uncertain but durable | C authorized; C's slot is the newest | `RESUMABLE` at C |
| after publication | C authorized | `RESUMABLE` at C |

## Non-claims

* The checkpoint is integrity-checked, not authenticated: anyone able to write the directory can replace slots.
  Replacement can only produce `CHECKPOINT_MISSING`, `CANDIDATE_UNRESOLVED` or a refusal, because a slot is
  resumable only when its identity is continuity's; it cannot make foreign bytes authoritative without an explicit
  operator adopt. Continuity itself is not authenticated either (docs/CONTINUITY.md).
* No media-level durability beyond `fdatasync` is claimed; no protection against a failing disk that returns old
  data with valid checksums for both slots.
* R0 does not checkpoint FMS COLD scratch, codec state, evolution state or any history, and does not make H-ECS or
  any hierarchy canonical. It changes no K1 or S3 mathematics: envelopes are written by K1's own read-only
  `snapshot_write` / `txn_snapshot_write`.
* Recovery is unqualified: no qualification run with recorded evidence exists.
