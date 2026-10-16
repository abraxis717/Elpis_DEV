# K1 Recovery R0 under RuntimeCore ABI v4: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED**

CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

Successor of [K1_RECOVERY_R0.md](K1_RECOVERY_R0.md), which was pre-registered against RuntimeCore ABI v3, where every
LEARN wrote its candidate's envelope and published continuity before returning. Under ABI v4 a LEARN is volatile and
durability is an explicit request served by the durability lane, so the v3 crash points (a checkpointed LEARN) no
longer exist and that specification cannot be inherited. It stays as written, historical authority for ABI v3.

## Question

Under ABI v4, after a process death at any step of an anchor, a recoverable publication, an identity-only publication,
the legacy composite import or an operator reconciliation, does restart resume exactly the state continuity
authorizes (the last completed durable publication, possibly one whose acknowledgment the caller never received), or
report an explicit disposition that needs an operator, never rolling silently forward or back and never recovering
volatile work that was never requested, while the checkpoint store stays two fixed slots, for bare K1 and ECS
composite lineages alike?

## What is kept apart (each its own observable)

| Concept | Observable |
|---|---|
| Volatile LEARN | the live generation advances; continuity's record and both slots are byte-identical |
| Capture / publication request | a `PENDING` ticket at once, at the captured live generation; the HOT lane never waits |
| Durable checkpoint identity | the slot's envelope trailer equals the capture's identity, read back and verified |
| Caller acknowledgment | an `ACKNOWLEDGED` (recoverable) or `IDENTITY_PUBLISHED` ticket delivered to the caller: not the same event as the synced record |
| Recovery after process death | the disposition and the authorized identity seen by a fresh process |
| Uncertain outcome | an `UNCERTAIN` ticket, the sticky lane fault until reopen, and restart's resolution to exactly one complete record |
| Operator reconciliation | `discard` / `adopt` of exactly the named candidate, refused while a publication is in flight |
| Fixed-size storage | exactly two slot files of the provisioned size, inodes unchanged |
| Composite vs bare K1 lineage | the record's format (v2 K1 / v3 composite), ELPISGK1 vs ELPISEC1 envelopes, no substitution either way |

## Subject

`native/runtime/src/checkpoint.rs`, `durable.rs` and `core.rs` (ABI v4); `elpis.runtime.recovery`; `Runtime.recover_k1`,
`discard_k1_candidate`, `adopt_k1_candidate`, `publish_recoverable`, `publish_identity`, `import_composite`.
Mechanics: `native/runtime/src/tests.rs` (`recovery`, `crashes`), `tests/integration/test_k1_recovery.py`,
`test_durable_process_kill.py`, `test_composite_process_kill.py`, `test_durable_ticket_law.py`,
`test_autonomous_no_growth.py`, `runtime.test_runtime_composite`.

## Inputs

* The qualifying commit and library digests (the testing RuntimeCore build, whose worker can be parked).
* A pre-registered crash-point list, written before the run: the durability lane's steps 1-9 for a recoverable
  publication (taken; sealed; slot torn at pre-registered byte offsets; slot written; slot synced; record torn at
  pre-registered offsets; record written; record synced; acknowledged) and after the acknowledgment was delivered; the
  identity-only publication's steps (sealed; record torn, written, synced; acknowledged); the anchor (the slot written
  before continuity anchors); the legacy import (composite slot synced, record torn); operator `discard` and `adopt`.
  The anchor, the import and the reconciliations are control operations outside the lane's step machine: where the
  testing library has no fault hook for one of their points today, adding and reviewing that hook is a prerequisite of
  the freeze, not something the run may improvise.
* Two lineages: bare K1 (d = 6, N = 36) and composite (the same K1 shape; memory D = 64, B = 2, S = 16, E = 32), each
  LEARN of the composite carrying one typed observation.
* Real `SIGKILL` of a child process parked at each point, plus randomized kills while learning and publishing. A
  power-loss model only through the lane's in-process lost-sync faults (`RecordSyncLost`, `RecordSyncDurable`); no
  power-loss claim is ever drawn from `SIGKILL`.
* Bounded scale: at most 64 LEARNs per lineage, at most 16 runs per parked point (they are deterministic; repetitions
  detect nondeterminism), at most 64 randomized kills per lineage.

## Procedure

For every point, lineage and run: a child anchors with an attached store (composite-sized for the composite), learns
(volatile), publishes recoverably (acknowledged), learns again (the candidate), parks the worker at the point (or
receives that acknowledgment), learns once more (volatile, never requested), reports the identities and relation keys,
and is killed. A fresh process restarts over the same directories and records continuity's format and identity, the
recovery disposition, the authorized and candidate identities, the resumed envelope's identity, which relation keys
are resident in the restored composite, the slot files' names, sizes and inodes, and the outcome of the pre-registered
operator action for that point; it then binds the resumed state, completes one LEARN and one recoverable publication.

## Gates (pre-registered)

* G1: every disposition is the one the crash matrices of docs/RUNTIME_CORE.md ("Crash law") and docs/K1_RECOVERY_R0.md
  prescribe for its point and lineage.
* G2: every `RESUMABLE` envelope restores (`K1State.restore` / `CompositeState.restore`) to a state whose identity equals
  continuity's; a composite's memory transition is present exactly once when its publication became authoritative and
  absent otherwise.
* G3: volatile work never requested is never recovered; no run adopts, discards or resumes anything without the
  explicit call; no recoverable publication succeeds over an unresolved candidate (`FAILED`, `CHECKPOINT_UNRESOLVED`).
* G4: an identity-only lineage never yields a recoverable acknowledgment (restart reports `CHECKPOINT_MISSING` or
  rebinds only caller-supplied bytes); no ticket anywhere violates the acknowledgment law.
* G5: an `UNCERTAIN` publication faults the lane until reopen while the HOT lane keeps running, and restart resolves it to
  exactly one complete record.
* G6: the checkpoint directory holds exactly two files of the provisioned size and inodes after every run.

## Would establish / would not establish

Passing would establish crash consistency of the ABI v4 durability protocol on the qualifying platform and filesystem
under the listed faults, for both lineage formats. It would **not** establish media durability beyond `fdatasync`
(nor power-loss behavior from `SIGKILL`), protection against a disk returning stale but checksum-valid data in both
slots, authentication of the checkpoint or continuity directory, or anything about H-ECS or the meaning of any memory
content.

## Evidence

`research/qualification/k1_recovery_r0_v4/evidence/` (write-once; not present: the run has not happened).
