# K1 Recovery R0: a bounded, continuity-authorized K1 checkpoint owner

**CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.**

Status: implemented systems mechanism (`native/runtime/src/checkpoint.rs`, written by the durability lane of
`native/runtime/src/durable.rs`, RuntimeCore ABI v4, `elpis.runtime.recovery`).
UNQUALIFIED here: this document specifies the mechanism and its tests and qualifies nothing by itself. Its ABI v4
qualification specification, `docs/qualification/K1_RECOVERY_R0_V4.md`, is RUN PASS with in-repository evidence for
commit `18f2f4d` only, within the scope stated there (crash consistency under `SIGKILL`, torn writes and lost-sync
faults; not power loss). The ABI v3 pre-registration `docs/qualification/K1_RECOVERY_R0.md` (NOT RUN) is kept as
written.

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

### Composite envelopes

A slot holds one self-identifying retained-state envelope: `ELPISGK1` for a K1 lineage, or `ELPISEC1`
(docs/ECS_COMPOSITE_STATE.md) for an ECS composite lineage. The slot format, protocol, dispositions and
crash matrix are unchanged; only the admitted envelope magic is extended, and the size bound grows to the
composite maximum (`ELPIS_ECS_COMPOSITE_MAX_ENVELOPE_BYTES`). A store is provisioned for exactly one envelope
size, so a composite lineage needs a store provisioned for its composite envelope size; a K1-sized store
refuses a composite snapshot (`ECS_REFUSED`, capacity) before anything is written. Because the composite
envelope carries the memory, a memory transition is resumable only from a slot exactly as K1's is: it is
durable only through an explicit durable publication (slot first) and authoritative only once continuity names its
identity.

## Protocol

With a store attached (`RuntimeConfig.k1_checkpoint_dir`; `elpis_runtime_checkpoint_attach`):

* **Anchor.** Before continuity anchors the lineage, the anchored state's complete envelope
  (`snapshot_write`, read-only) is verified (its identity equals the identity being anchored) and written into a
  slot and synced. A refusal anchors nothing.
* **LEARN.** Volatile (RuntimeCore ABI v4): a LEARN never touches the store. It advances the live state only.
* **Recoverable publication** (explicit: `Runtime.publish_recoverable`, `elpis_runtime_recoverable_request`;
  refused `RUNTIME_RECOVERY_UNAVAILABLE` without an attached store, never downgraded). The live state's envelope is
  captured unsealed under the managed lease (`leased_capture`) and handed to the durability lane, which seals it
  (`envelope_seal`, pure), verifies its identity and format, writes it into the slot that does **not** hold the
  authorized identity and syncs it, **reads the slot back and verifies it is exactly the captured envelope**, then
  has continuity publish `D -> C`. The request is `ACKNOWLEDGED` (always `recoverable`) only after both are durable
  and the slot verified. A failed slot write, torn write, sync or read-back is `FAILED` (`CHECKPOINT_IO`) and the
  slot is withdrawn (best effort); the authorized slot is never touched. A capture that already is the authority
  publishes no record; its slot is still read back and verified (and written first if no slot held it).
* **Identity-only publication** (`Runtime.publish_identity`, `elpis_runtime_identity_request`) is refused
  (`RUNTIME_CHECKPOINT_ATTACHED`) while a store is attached: with a store, continuity never names an identity the
  store does not hold. Without one it resolves `IDENTITY_PUBLISHED`, never `ACKNOWLEDGED` (docs/RUNTIME_CORE.md,
  "Recoverable and identity-only publication").
* **Refusal while unresolved.** A durable publication is refused (`FAILED`, `CHECKPOINT_UNRESOLVED`, no fail-stop)
  while an unresolved candidate exists. LEARN and QUERY stay available on the live state.

**Superseded (through ABI v3):** every LEARN wrote and synced its candidate's complete envelope before the native
commit, so recoverability cost one envelope copy and one slot write + `fdatasync` per LEARN on the HOT path. Now
that price is paid only per explicit recoverable publication, on the durability lane; restart resumes the last
completed publication (which may be one whose acknowledgment the crash prevented), and volatile work since then is
lost by a crash (the API reports it as volatile).

## Recovery dispositions

`Runtime.recover_k1()` / `elpis_runtime_checkpoint_recover` is read-only. Continuity decides; the slots are
evidence.

| Continuity | Slots | Disposition | What may happen |
|---|---|---|---|
| unanchored | anything | `NOTHING_TO_RECOVER` | nothing; a slot never creates a lineage |
| anchored at D | a verified slot holds D, no newer slot of another identity | `RESUMABLE` | restore the returned envelope (`K1State.restore`); the next QUERY/LEARN binds it after verifying it against D |
| anchored at D | a verified slot holds D and a newer verified slot holds C != D | `CANDIDATE_UNRESOLVED` | nothing implicit; durable publication is refused until an operator discards C (resume D) or adopts C (continuity D -> C) |
| anchored at D | no verified slot holds D | `CHECKPOINT_MISSING` | the authorized state is not recoverable from this store; no other slot is a candidate |

There is no silent roll forward (a candidate never becomes authority by being newer or valid) and no silent roll
back (an authorized identity without its envelope is reported missing, never replaced by an older or other slot).

### Operator reconciliation

* `discard_k1_candidate(C)` / `elpis_runtime_checkpoint_discard`: withdraws exactly the unresolved candidate `C`
  (its header is zeroed and synced). Any other identity is `CHECKPOINT_INVALID`.
* `adopt_k1_candidate(C)` / `elpis_runtime_checkpoint_adopt`: re-reads and re-verifies `C`'s complete envelope,
  then continuity publishes `D -> C`. Refused while a K1 state is bound (`RUNTIME_TURN_OPEN`) and while a durable
  publication is pending or running (`RUNTIME_DURABLE_BUSY`, as discard is); a publication failure fail-stops.
  Adoption is the operator's decision that `C` is the lineage; the checkpoint never makes it.

Both are `OPERATOR_OPERATIONS` in `elpis.runtime.persistence`.

## Crash matrix

Deterministic fault points of a durable publication (testing library: `elpis_runtime_testing_durable_fault`:
die or park at each step of `durable.rs`, slot/record write, torn-write and sync failures) are proven in
`native/runtime/src/tests.rs` (in-memory K1 stand-in), over real native K1 in `tests/integration/test_k1_recovery.py`
and the composite in `native/runtime/tests/test_runtime_composite.c`; with real `SIGKILL` of a process parked at
every step and after the acknowledgment was delivered in `tests/integration/test_durable_process_kill.py`, plus
randomized kills while learning and publishing. Restart recovers only a complete, authorized durable state, which
may include a publication completed immediately before the caller lost its acknowledgment: a missing
acknowledgment is not evidence that nothing became durable (docs/RUNTIME_CORE.md, "Crash law").

| Process dies / fails ... | Durable state | After restart |
|---|---|---|
| any time between durable publications (volatile LEARNs) | D authorized | `RESUMABLE` at D: the volatile work is lost |
| before the capture is sealed or written | D authorized | `RESUMABLE` at D |
| during the candidate slot write (torn) | authorized slot intact; the other slot invalid | `RESUMABLE` at D |
| after the candidate slot is complete (synced or not), before the record | D authorized; C checkpointed | `CANDIDATE_UNRESOLVED` (C) |
| during the continuity record write (torn) | D authorized; C checkpointed | `CANDIDATE_UNRESOLVED` (C) |
| record written (process death; the page cache survives), synced, or acknowledged (the caller may never have received the acknowledgment) | C authorized | `RESUMABLE` at C |
| record written, then power loss before its sync (not exercised by `SIGKILL`) | D or C authorized (the record may be lost) | `CANDIDATE_UNRESOLVED` (C) / `RESUMABLE` at C |
| slot read back unequal to the capture (`FAILED`, `CHECKPOINT_IO`) | D authorized; the slot withdrawn | `RESUMABLE` at D |
| publication refused (`FAILED`) | D authorized; the candidate slot withdrawn | `RESUMABLE` at D |
| publication uncertain (`UNCERTAIN`), record lost / durable | D / C authorized | `CANDIDATE_UNRESOLVED` (C) / `RESUMABLE` at C |

## Non-claims

* The checkpoint is integrity-checked, not authenticated: anyone able to write the directory can replace slots.
  Replacement can only produce `CHECKPOINT_MISSING`, `CANDIDATE_UNRESOLVED` or a refusal, because a slot is
  resumable only when its identity is continuity's; it cannot make foreign bytes authoritative without an explicit
  operator adopt. Continuity itself is not authenticated either (docs/CONTINUITY.md).
* No media-level durability beyond `fdatasync` is claimed; no protection against a failing disk that returns old
  data with valid checksums for both slots.
* R0 does not checkpoint FMS COLD scratch, codec state, evolution state or any history, and does not make H-ECS or
  any hierarchy canonical. It changes no K1 or S3 mathematics: envelopes are K1's own (`snapshot_write` at the
  anchor; the leased capture plus the pure seal for a durable publication, byte-identical to `snapshot_write`).
* Recovery is unqualified: no qualification run with recorded evidence exists.
