# K1 Recovery R0 under RuntimeCore ABI v4: qualification specification

Status: **RUN PASS** · Subject: **QUALIFIED**

CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

Successor of [K1_RECOVERY_R0.md](K1_RECOVERY_R0.md), which was pre-registered against RuntimeCore ABI v3, where every
LEARN wrote its candidate's envelope and published continuity before returning. Under ABI v4 a LEARN is volatile and
durability is an explicit request served by the durability lane, so the v3 crash points (a checkpointed LEARN) no
longer exist and that specification cannot be inherited. It stays as written, historical authority for ABI v3.

## Qualified scope

Run once, on QUAL inputs, under the freeze record published before the run (`FREEZE.json`, commit a2a84e2), with the
verdict recomputed by the independent checker: **RUN PASS**, G1-G6 over all 312 planned trials (G1 2576 checks, G2 908,
G3 1363, G4 476, G5 64, G6 332; no failure). What is qualified, and only that:

* **Subject:** the K1 Recovery R0 durability protocol (the durability lane, the two-slot checkpoint store and
  continuity as the authority register) for bare K1 and composite lineages, with the ABI v4 durability law unchanged
  in the ABI v5 library (premise P1-P4 against 65287ad, verified mechanically at the freeze).
* **Commit:** `18f2f4de3c1bacef4f5b6bc61ac5999ecd133038` (tree `a6de5ce`), the merge commit of PR #52, from a fresh
  clone, a fresh Release build and a fresh venv with a non-editable install. No other commit is qualified by this
  run, including the later commits that only add this record and its evidence.
* **Binary:** the testing RuntimeCore build of that commit (`libelpis_runtime_testing.so`, SHA-256
  `3c16135e4c3d665cc716a656440c1c4b090a03e1e82bd7f768f28bdd94c81b9a`), because the crash points exist only there,
  with the K1, composite and HACF libraries whose digests the freeze record lists, deployed as `home/lib` copies,
  pinned and admitted at ABI runtime 5, composite 1. That the production build carries the same protocol rests on
  premise P4 (the production view of `durable.rs` and `checkpoint.rs`) and on the review of the control operations
  in `core.rs` (identical exported symbols and call targets). That is a source argument, not a fault run of the
  production binary.
* **Environment:** Linux 6.18.44 x86_64, glibc 2.39, GCC 13.3.0, rustc 1.97.0, CPython 3.12.3 (isolated); the work
  root on an ext2/ext3-family filesystem (statfs magic); one machine.
* **Faults:** `SIGKILL` at every registered lane step and control-operation point, torn slot and record writes at the
  registered offsets, and the in-process lost-sync faults (`uncertain`). `SIGKILL` keeps the page cache, so nothing
  here bears on power loss or media durability ("Would establish / would not establish" holds as written).

## Amendment before any run (RuntimeCore ABI v5; freeze prerequisites)

Made before the freeze, on DEV inputs only; nothing of it was chosen after observing a QUAL result.

* **The ABI under test.** The library is RuntimeCore ABI v5. v5 changed only the managed turn (the observe-only
  turn, `budget.max_observations`, `counters.k1_observes`); the durability lane, the checkpoint store, continuity's
  records, the envelopes and every crash law this specification measures are ABI v4's, unchanged. The question and
  the gates below are therefore asked of the v4 durability law as the v5 library implements it.
* **The unchanged-law premise, bound to source** (`premise.py`; `plan.json` `premise.abi_v4_commit` = 65287ad, the last
  ABI v4 commit). The freeze is refused, and so is every run, unless all hold against the qualifying checkout: P1
  `native/continuity` byte-identical; P2 the ABI v4 durability-law test files (`test_k1_recovery.py`,
  `test_durable_process_kill.py`, `test_durable_ticket_law.py`, `test_composite_process_kill.py`,
  `test_autonomous_no_growth.py`) byte-identical; P3 all 21 ABI v4 recovery and crash tests of `tests.rs` present with
  identical bodies; P4 `durable.rs` and `checkpoint.rs` identical once their testing-only items are removed. Not
  mechanical, so recorded with a digest rather than claimed: the restructured control operations in `core.rs`
  (production equivalence per the 2026-10-18 review: identical exported symbols and call targets).
* **The control-operation hooks** this specification made a prerequisite exist:
  `elpis_runtime_testing_control_fault` (testing library only; docs/RUNTIME_CORE.md, "Control-operation crash
  points"): anchor, import, discard and adopt at begun / slot written / slot synced / record torn / record written /
  record synced, simulated (in-process tests) or parked for a real `SIGKILL` (this run).
* **Instrument.** `research/qualification/k1_recovery_r0_v4/`: `plan.json` (every trial's coordinate rule, point,
  lineage, expectation, operator action, torn offset, envelope size, bound and input seed: the pre-registered
  crash-point list), `freeze.py` (measures the executing object and compares it with the freeze record, fail
  closed), `runner.py` (`freeze` writes the freeze record once; `run` drives real K1 and composite lineages through
  the public runtime API over a pinned deployment of the testing build, parks each child at its point and `SIGKILL`s
  it, then observes restart in a fresh process; `seal` writes the evidence directory once), `check.py` (the gates and
  the evidence validation, from the plan and the records alone; it imports no Elpis code).
* **Procedure as registered.** Families: `publish` (lane steps 1-9 and after delivery; both lineages), `identity`
  (sealed, record torn / written / synced, acknowledged; bare K1 only: a composite's envelope size is stated only by
  a provisioned store, so its identity-only publication is refused `CHECKPOINT_INVALID`, registered as
  `identity_refused`), `anchor` (points 1-6, both lineages, store attached), `import` (points 1-6: a K1 lineage with its
  own K1-sized store, a freshly provisioned composite store, restart observed through both stores), `discard`
  (points 1-3) and `adopt` (points 1, 4-6) of a candidate left by a `SIGKILL` at lane step 5 (both lineages),
  `uncertain` (the in-process lost-sync faults 8 and 9, no kill), `random` (32 randomized kills per lineage while
  learning and publishing). 4 runs per parked point. After the first acknowledgment the caller keeps a copy of the
  acknowledged state's bytes (release, snapshot, rebind): restart uses it only to show that a recoverable publication
  over an unresolved candidate fails (`CHECKPOINT_UNRESOLVED`) and, for the identity-only lineage, that only the
  caller bytes of continuity's identity rebind. An identity-only restart attaches a freshly provisioned store. The
  operator action for an unresolved candidate is pre-registered: adopt on even run indices, discard on odd ones (the
  discard and adopt families repeat their own reconciliation).
* **Plan corrections made on DEV inputs** (before the freeze): a record sync that fails with the bytes lost leaves the
  synced candidate slot, so restart reports `CANDIDATE_UNRESOLVED`, not `RESUMABLE` (the crash law already said so:
  `test_k1_recovery.py`); composite identity-only publication is a registered refusal, not a point.
* **Evidence schema.** `elpis.k1-recovery-r0-v4.freeze.v2` (the freeze record), `.record.v2` (one record per trial),
  `.run.v2` (the run summary: what was observed before, the deployment, what was observed after), `.verdict.v2` (the
  checker's output), and `SHA256SUMS` over exactly those files (`runner.py seal`; `check.py evidence` validates). The
  evidence is digest-bound to the qualifying commit and environment it measured; it is not cryptographically signed
  (its custody is this repository's history).
* **Freeze on the accepted source.** The freeze binds one exact commit. It is made only on the source finally
  accepted (the merge commit of the integration into `main`, built and installed from a clean checkout of it), never
  on a branch head that the merge would replace; a qualification of one commit is not claimed for any other.

## Prerequisite review (2026-10-18) and its resolution

An independent read-only review of the hooks and the first instrument (commits 785faf9, f2e199f) found the hooks
correct and production-safe (identical exported symbols and call targets with the feature off; every point's
on-disk effect as documented) and the instrument not freezable (B1, S1-S4, N1-N6). Every finding is resolved before
any freeze, on DEV inputs only:

* **B1, freeze verification.** `freeze.py` measures the executing object itself and the run refuses before it creates
  or opens anything unless every measure equals the freeze record: the checkout's commit and tree and that it is
  clean (again after the run), every instrument file (the freeze producer included) and the specification, the
  subject source files, the interpreter (path, version, isolated mode) and the `elpis` package it actually imported
  (its root, every loaded module's origin, and its content against the checkout's `src/elpis`), the build-tree
  libraries, the deployed `home/lib` copies, their catalog pins and the ABI the admitted libraries report (against a
  trial deployment made by the freeze producer), the toolchain, the platform and the work root's filesystem. Under
  `python -I` the `PYTHON*` variables (`PYTHONPATH` included) are ignored and no script directory is added: the
  package is whatever the interpreter's own `site-packages` resolves, which is why it is measured; children verify
  they run isolated over the same package. `tests/research/qualification_k1_recovery_r0_v4/test_freeze.py` changes
  each bound item alone and shows the refusal naming it, and that a refused QUAL run creates nothing.
* **S1, independent checking.** Records carry no expectation. `check.py` derives every trial's coordinate (index,
  family, lineage, point, run index), seed, fault argument, random-family draws, expected disposition, authority,
  candidate, operator action and continuity generations from the plan; a record must sit at exactly its coordinate,
  once, in order.
* **S2.** `identity_refused` registers D0 as the authority; the checker proves the refused operations moved nothing
  (continuity's raw records unchanged, generation unchanged, no acknowledgment, no slot and no candidate in a store
  attached afterwards).
* **S3, torn writes.** Offsets slot 17 / 127 / 128 + c/2 / 128 + c - 32 and record 24 / 50 / 100 / 144 (each writes a
  differing byte and leaves the old checksum). Restart records both continuity records in full and every slot's
  difference range against a copy taken before the point; a tear counts only when exactly one file changed, only
  below the offset, by at least one byte, and the result does not verify.
* **S4, format.** For every family the checker parses continuity's raw current record (magic, format byte, digest,
  generation, checksum under the format's domain) and requires the lineage's format; restart offers a state of the
  other format from the same seed and it must be refused (`CONTINUITY_STATE_MISMATCH`) with continuity unchanged.
* **N1-N3** fixed (bd0f637). **N4:** a slot point its operation never writes is consumed and leaves nothing armed
  (native test, with its mutant); every import point in process over the real libraries. **N5:** an imported
  composite's memory is checked empty (0 slots, then 1 after the continuation's observation); the PENDING ticket check
  stays: a request that waited for the worker would fail it. **N6:** children run `python -I -B` with no `PYTHON*`
  variable; unawaited random-family tickets report their resolutions, and a superseded or failed request may never
  become the authority; `runner.py seal` and `check.py evidence` produce and validate the manifest.
* **Generations.** Exactly the expected continuity generation at restart, after the operator action and after the
  continuation (one record): no retry and no duplicate publication.
* **Adversarial check.** `test_instrument.py` generates a DEV run (one run per point), and every listed mutant fails
  its gate (the list is in the test).

* **Rehearsal under a freeze (DEV inputs).** The whole pipeline (clean checkout, Release build, non-editable install,
  `freeze`, `run --freeze`, `verdict`, `seal`, `evidence`) was rehearsed on DEV inputs. At dffb5b0 it judged one
  random-family trial G3 FAIL (`random/composite/random/27`: an unresolved candidate that looked never requested). The
  defect was the runner's, not the subject's: the child flushes its request event before it publishes, but that event
  reached the runner in the same pipe chunk as the LEARN the runner was waiting for, and the reader discarded the
  rest of the chunk. The runner now reads each child's output through one buffer for the child's life (c81a223, with
  a regression test the old reader fails). The rehearsal at c81a223 passes G1-G6 over 312 trials, and its sealed
  directory validates (`verify_evidence(..., inputs="dev")` finds no problem; as QUAL evidence it is refused, as it
  must be). An earlier full DEV run (4c996c6) passed only because no chunk split there; the checker caught the loss
  when it occurred.

DEV records are development records, never evidence. The subject stays `UNQUALIFIED` until its QUAL evidence says
otherwise.

## Question

Under ABI v4 (as the ABI v5 library implements it), after a process death at any step of an anchor, a recoverable publication, an identity-only publication,
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

`native/runtime/src/checkpoint.rs`, `durable.rs` and `core.rs` (the ABI v4 durability law in the ABI v5 library);
`elpis.runtime.recovery`; `Runtime.recover_k1`,
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

`research/qualification/k1_recovery_r0_v4/evidence/` holds the directory written once by the QUAL run on 18f2f4d
and sealed by `runner.py seal`. The repository keeps no file over 2 MiB, so `records.jsonl` (2408621 bytes) is
committed gzip-compressed as `records.jsonl.gz` (SHA-256 `ea26994633e50d21994791768c76ce72437bc18109e6e3ab2a33fe48761effae`);
every other file is committed as sealed. To validate, restore the sealed directory, then run the frozen validator,
which checks exactly its five files, every SHA-256 (so the restored records are the written bytes), the run bound to
its freeze record and records, the observed checkout, instrument, subject and libraries equal to the freeze, and the
verdict recomputed from the records:

```sh
mkdir sealed && cp evidence/freeze.json evidence/run.json evidence/verdict.json evidence/SHA256SUMS sealed/
gunzip -c evidence/records.jsonl.gz > sealed/records.jsonl
python -I -B check.py evidence sealed
```

| File | SHA-256 |
|---|---|
| `freeze.json` (byte-identical to the published `FREEZE.json`) | `2ff44a12ac5ceed9d8065969f790b00397a50e406a4fa86b05786f215bd45c3b` |
| `records.jsonl` (312 trial records; committed as `records.jsonl.gz`) | `dbee71c9fdf16619e3eb2a9da2c29182b4b7eed850b08992a03f8c065a351c4a` |
| `run.json` | `0833539ff30d1e49731e7ba0f28a27b3d00d44acb976dc6ad2b9d6739f8c1f30` |
| `verdict.json` | `475a9c744393eb375546d6b886687bba637bc63db242acd7ea805762de81d915` |
| `SHA256SUMS` | `b0f785214614e436b435c605514d8f55ec7587c900f8d6ac244189ba6341126b` |

In CI, `tests/research/qualification_k1_recovery_r0_v4/test_evidence.py` re-validates it and requires the
instrument files in the tree to be the ones the freeze bound. A changed instrument is a new instrument and needs new
evidence. `tests/boundary/test_qualification_specs.py` requires this document's status to repeat the recomputed
verdict. The run's stderr was empty; it took 179.7 s.
