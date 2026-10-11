# K1 Recovery R0: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED**

CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

## Question

After a crash at any boundary of an anchor or a checkpointed LEARN, does restart either resume exactly the state
continuity authorizes, or report an explicit disposition that needs an operator, never rolling silently forward or
back, while the checkpoint store stays two fixed slots?

## Subject

`native/runtime/src/checkpoint.rs` and its use in `native/runtime/src/core.rs` (RuntimeCore ABI v3),
`elpis.runtime.recovery`, `Runtime.recover_k1` / `discard_k1_candidate` / `adopt_k1_candidate`
(docs/K1_RECOVERY_R0.md). Mechanics: `native/runtime/src/tests.rs` (`mod recovery`),
`native/runtime/tests/test_runtime_abi.c`, `tests/integration/test_k1_recovery.py`,
`tests/integration/test_autonomous_no_growth.py`.

## Inputs

* The qualifying commit and library digests.
* A pre-registered crash-point list: during the candidate slot write (before / after each `pwrite` byte range and
  the `fdatasync`), after the checkpoint, after the native commit, during and after the continuity publication, and
  during operator `discard` / `adopt`.
* Real process death (`SIGKILL` of a child process at each point, via the testing library's fault hooks), not only
  simulated death; and a power-loss model by dropping unsynced writes (a bounded loopback filesystem or the
  continuity testing library's lost-sync faults).
* Bounded scale: one K1 shape (d = 6, N = 36), at most 64 LEARNs per lineage, at most 32 crash runs per point.

## Procedure

For every crash point and run: start a child runtime with an attached store, anchor, LEARN to the point, kill; then
restart in a fresh process and record the disposition, the authorized identity, the candidate identity, whether the
resumed envelope's digest equals continuity's identity, the slot files' names, sizes and inodes, and the outcome of
the pre-registered operator action for that point.

## Gates (pre-registered)

* G1: every disposition is the one the crash matrix in docs/K1_RECOVERY_R0.md prescribes for its point.
* G2: every `RESUMABLE` envelope restores to a state whose retained-state identity equals continuity's.
* G3: no run adopts, discards or resumes anything without the explicit call; no LEARN runs over an unresolved
  candidate.
* G4: the checkpoint directory holds exactly two files of the provisioned size and inodes after every run.

## Would establish / would not establish

Passing would establish crash-consistency of the protocol on the qualifying platform and filesystem under the listed
faults. It would **not** establish media durability beyond `fdatasync`, protection against a disk returning stale
but checksum-valid data in both slots, authentication of the checkpoint directory, or anything about H-ECS.

## Evidence

`research/qualification/k1_recovery_r0/evidence/` (write-once; not present: the run has not happened).
