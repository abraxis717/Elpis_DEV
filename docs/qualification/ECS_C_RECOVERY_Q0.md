# ECS_C Recovery Q0 under RuntimeCore ABI v6: qualification specification

Status: **RUN PASS** · Subject: **QUALIFIED**

THE ECS_C AUTHORITY DECIDES. CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. ONLY THE OPERATOR RECONCILES AN ATTACK.

Pre-registered before any run, before its instrument exists. It is the qualification the successor of standalone
continuity needs before continuity may be retired (docs/NATIVE_SPLIT_CENSUS.md, sections 9 and 11: "retire standalone
continuity only after qualified successors exist"). K1 Recovery R0 v4 ([K1_RECOVERY_R0_V4.md](K1_RECOVERY_R0_V4.md))
qualified the durability protocol with continuity as the authority register, for commit `18f2f4d` only, and its premise
P1 binds `native/continuity`. It does not extend to the ECS_C authority, nor to RuntimeCore ABI v6, nor to the native
host. This specification asks the same question of the successor and adds the adversary the successor exists to
defend against.

## Qualified scope

Run once, on QUAL inputs, under the freeze record published before the run (`FREEZE.json`, commit `9c7c72b`), with the
verdict recomputed by the independent checker: **RUN PASS**, G1-G6 over all 221 planned trials, with no gate failure
and no mechanical failure. What is qualified, and only that:

* **Subject:** RuntimeCore ABI v6's durability protocol over the ECS_C authority, operated through the native host:
  - the durability lane;
  - the two-slot checkpoint store;
  - the authority's two records and two witnesses.

  It covers the registered process deaths, state-directory attacks, rearrangements and witness loss (Inputs 2-8), and
  recovery work and storage fixed in history length (10, 100 and 1,000 publications).
* **Commit:** `117b41ad8a1a1fd25963489d4b29a66b284c4632` (tree `bcb8e46`), the merge commit of PR #55. It was built
  from a fresh clone (`cmake -DCMAKE_BUILD_TYPE=Release`) and installed with the runtime and testing components.
  - No other commit is qualified by this run, including the later commits that only add this record and its
    evidence.
* **Binaries:** the testing host `elpis_testing` (SHA-256
  `1a2a8d95c341c6433008e4c2bc5c2f3abb38722ba4386d9f21670e41282ece87`) and the testing RuntimeCore build
  `libelpis_runtime_testing.so` (`b8076c0b9f865de017ebaf292c6b0d0168a945d6339ffbcd9ea115f8450cf1e8`), because the
  crash points exist only there.
  - With them: the ECS_C authority library (`75e823db73c63dc8d13ea86063378efbeb8d8ede2f9f9e55e5b278af85e03315`) and
    the other libraries whose digests the freeze record lists, deployed by `init --libraries` from that installation.
  - That the production build (`elpis`, `libelpis_runtime.so`) carries the same protocol rests on the premise bound
    at the freeze:
    - the production views' digests;
    - no testing symbol in the production RuntimeCore;
    - no testing operation in the production host.

    That is a binding to source, not a fault run of the production binaries.
* **Environment:** one machine:
  - Linux 6.18.44 x86_64, glibc 2.39;
  - GCC 13.3.0, rustc 1.97.0, CMake 3.28.3;
  - CPython 3.12.3, isolated, with no Elpis module imported;
  - the work root on an ext2/ext3-family filesystem (statfs).
* **Faults and attacks:** as registered.
  - `SIGKILL` keeps the page cache, so nothing here bears on power loss or media durability.
  - The attacks are the T1 adversary's (docs/ECS_C_AUTHORITY.md); witness loss is outside it.
  - "Would establish / would not establish" holds as written.
* **Deep rollback, as registered:**
  - An accepted rollback of one publication leaves the newer candidate for the operator (`CANDIDATE_UNRESOLVED`).
  - An accepted rollback of two or more publications leaves no checkpoint slot holding the accepted identity
    (`CHECKPOINT_MISSING`), so cognition stays refused.
  - That outcome is what this run qualifies. The re-authorization proposed for it (docs/NATIVE_SPLIT_CENSUS.md, "Stage
    F conflict": K1 Recovery R1 / RuntimeCore ABI v7) is neither implemented nor qualified.

## The accepted source and the instrument

* **The accepted source.** As for K1 Recovery R0 v4, the freeze binds one exact commit: the merge commit integrating the
  native split (Stages B to F of docs/NATIVE_SPLIT_CENSUS.md) into `main`, built and installed from a clean checkout of
  it.
  - Until those stages were merged (PR #55, `117b41a`), no QUAL run was possible.
* **The instrument.** `research/qualification/ecs_c_recovery_q0/` holds the plan, runner, checker, freeze verification
  and seal, with the discipline of K1 Recovery R0 v4's instrument:
  - the freeze verifies the executing object;
  - records carry no expectation;
  - an independent checker derives every expectation from the plan.

  It was rehearsed on DEV inputs (its README) before the freeze. DEV records are not evidence.

## Question

Under RuntimeCore ABI v6 over the ECS_C authority, operated through the native host (`docs/NATIVE_HOST.md`):

1. **Process death.** After a process death at any step of an anchor, a recoverable publication or a checkpoint
   reconciliation, does restart resume exactly the state the ECS_C authority authorizes, without ever needing
   operator reconciliation of the authority? Otherwise it must report the checkpoint disposition that needs the
   operator: an unresolved candidate.
2. **Attacks.** After every registered state-directory attack, does the runtime refuse to open until the operator
   sends exactly the reconciliation the authority names, accepting nothing silently? After that reconciliation, does
   it continue from exactly the record the operator accepted?
3. **Boundedness.** Do recovery work and the deployment's file set stay fixed, independent of history length?

## What is kept apart (each its own observable)

| Concept | Observable |
|---|---|
| Volatile LEARN or observation | the live generation advances; both authority records, both witnesses and both checkpoint slots are byte-identical |
| Durable publication | an `ACKNOWLEDGED` ticket; the authority's generation advances by exactly one PUBLISH whose evidence summary is the turns' committed observations |
| Recovery disposition | the authority report (`CLEAN`, `TRAILING_INCOMPLETE`, `INTERRUPTED_COMPLETED`, ...), its mode and the K1 checkpoint disposition, from a fresh process |
| Operator reconciliation of the authority | `authority-reconcile` naming exactly the kind and digest `authority` reported; any other is refused before an event exists |
| Operator reconciliation of the checkpoint | `reconcile --discard / --adopt` of exactly the named candidate |
| Fixed storage | the home's 17 files, names, sizes and inodes |
| Recovery work | the report's reads, bytes read and replays |

## Subject

- **Sources:**
  - `native/ECS_C/authority` (the authority);
  - `native/runtime/src/authority.rs`, `durable.rs`, `checkpoint.rs` and `core.rs` (RuntimeCore ABI v6's durable
    authority, durability lane and checkpoint store);
  - `native/runtime/host` (the native host).
- **Binaries:** the testing builds (`elpis_testing`, `libelpis_runtime_testing.so`), because the crash points exist
  only there. They are installed and pinned exactly as `docs/OPERATING_ELPIS.md` installs the product, with the
  libraries' digests in the freeze record.
- **The production build.** That it carries the same protocol rests on a premise bound to source, as K1 Recovery R0
  v4's P4: the production view of `durable.rs` and `checkpoint.rs` once testing-only items are removed. That premise
  must be written into the instrument and verified mechanically at the freeze.

## Inputs (the pre-registered list)

The machine-readable list is the instrument's `plan.json` (`research/qualification/ecs_c_recovery_q0/`); where it is
more exact than this text, it governs.

1. **Lineage.** One ECS composite lineage (`create --seed s`, the LineageConfig defaults) and one ACTIVE source
   (per-turn bound 8). Every LEARN turn is followed by one evidence turn of one observation.
2. **Publication faults.**
   - `publish`: `SIGKILL` while the worker is parked at each lane step 1-9 (`testing_durable_fault` action 2).
   - `failure`: the in-process failure plans 3-10 (no kill).
3. **Control-operation faults**, each parked (`testing_control_fault` action 2, reported by the testing host's
   watcher) and `SIGKILL`ed:
   - the anchor at points 1-6, parked inside `create` (`--testing-anchor-park`);
   - the checkpoint discard (points 1-3) and adopt (points 1, 4-6) of a candidate left by a step-5 kill.
4. **Random kills.** 32 randomized `SIGKILL`s while learning, observing and publishing.
5. **Attacks the recovery law refuses**, each made with the runtime closed, against the state directory only (the T1
   adversary of docs/ECS_C_AUTHORITY.md). "The newest record" is the record slot holding the acknowledged authority.
   - **Rollback.** Copying back both records saved 1, 2 and 5 publications earlier.
   - **Byte flips** in the newest record at registered offsets: the header, the generation, the state, the MAC and
     the checksum.
   - **Shortening** the newest record, to zero bytes and to half its size.
   - **Forged generation.** The newest record with its generation raised and its checksum recomputed, MAC unchanged.
   - **Forged evolution revision and head**, the same way, with the state root also recomputed.
   - **A validly checksummed substitute** in the other slot: the newest record with its generation advanced by one
     and its checksum recomputed, MAC unchanged.
6. **Rearrangements the recovery law accepts**, made the same way. Each must resume exactly the acknowledged state with
   no operator action:
   - both record slots swapped;
   - the newest record duplicated into the other slot;
   - the other slot shortened.
7. **Witness loss.** Both witnesses shortened to zero bytes: the `WITNESS_MISSING` path. This is outside T1 (it needs
   the authority directory).
8. **Boundedness.** Histories of 10, 100 and 1,000 publications; the home's file set and the recovery work are recorded
   after each.

Every point runs 4 times.

The operator action is pre-registered:
- **For a refused attack or witness loss:**
  - on even run indices, reconcile exactly what `authority` reports;
  - on odd ones, first a reconciliation with a wrong digest, which must be refused, then the exact one.
- **For a checkpoint candidate:** the candidate's own reconciliation in the discard and adopt families. Elsewhere,
  adopt on even run indices and discard on odd ones.
- **For an interrupted anchor:** `create` again with the same arguments.

## Procedure

For every point and run a fresh deployment is made: `init --libraries`, then `create`. The source is founded
afterwards, except in the anchor family, whose child is `create` itself.

1. **The child.** It learns, observes and publishes (acknowledged: D1), then learns and observes again (D2, live).
2. **The fault.** The child is parked at the point, or the fault or attack is applied, and it is killed where the
   family says.
3. **Observation.** A fresh process records:
   - the authority report and reconciliation;
   - the status, the recovery disposition, the durable and live identities and the resident relation keys;
   - the home's files (names, sizes, inodes);
   - the outcome of the pre-registered operator action.
4. **Continuation.** The process learns, observes and publishes once more. A further fresh process checks that it
   resumes exactly that publication.

## Gates (pre-registered)

* **G1 (process death).** Every recovery disposition and authority report is the one the crash law prescribes for its
  point (docs/RUNTIME_CORE.md, "Durability crash matrix" and "Control-operation crash points"; docs/ECS_C_AUTHORITY.md,
  "Transition and recovery"). `plan.json` states each expectation, and the checker derives every verdict from the plan
  and the records alone. No process-death trial ever needs authority reconciliation.
* **G2 (exactness).**
  - Every resumed state's identity equals the authority's.
  - A publication's observations are present exactly when it became authoritative.
  - The evidence count advances exactly by the committed observations of acknowledged publications.
  - An acknowledged publication is never lost, except by an attack the operator explicitly accepted.
* **G3 (no silent acceptance).**
  - Every refused attack refuses the runtime's open (`RUNTIME_AUTHORITY`) with the disposition the authority's
    recovery law prescribes. Witness loss does the same.
  - A wrong reconciliation is refused and changes nothing.
  - After the exact one:
    - the authority continues from the accepted record, at the abandoned generation + 1 for a rollback acceptance;
    - the K1 checkpoint is reconciled separately.
  - Every accepted rearrangement opens in its prescribed disposition at exactly the acknowledged state.
  - No attack ever resumes a state the operator did not accept.
* **G4 (no implicit action).**
  - Nothing is adopted, discarded or resumed without its explicit call.
  - No recoverable publication succeeds over an unresolved candidate.
  - No ticket violates the acknowledgment law.
* **G5 (fixed storage).** After every trial's continuation, the home holds exactly its 17 files, at their provisioned
  sizes and inodes. Attacks rewrite files in place.
* **G6 (bounded recovery).** The recovery work (reads, bytes read, replays) is the same after 10, 100 and 1,000
  publications.

## Revisions before the freeze

This specification was revised while its instrument was written. Revisions 1-3 came before any run, DEV runs
included, and nothing had been observed yet. Their reasons were found by deriving the plan from the recovery law:

1. **Swapped and duplicated slots.** The law recovers both without loss (`swapped_or_duplicated_slots_recover_without_loss`),
   so G3's "every attack refuses" overclaimed. Both are now rearrangements the law accepts (input 6), each with a
   prescribed disposition.
2. **A shortened record.** Open refused a slot of the wrong size with `CORRUPT` before recovery, which left the operator
   no reconciliation for a truncated newest record. The authority now reads a shortened file as the torn prefix it is
   (docs/ECS_C_AUTHORITY.md, "File sizes"), and input 5 registers it at two lengths. Witness loss is now "shortened to
   zero bytes" for the same reason.
3. **The anchor.** The native host anchors only inside `create`. A `create` that died after writing `lineage.cfg` left a
   home that refused both `create` and every operation. `create` now publishes its files atomically and completes an
   interrupted create of the same lineage (docs/NATIVE_HOST.md). The testing host parks the anchor there, and a
   watcher reports any parked control operation.
4. **The stimulus** (after the first DEV rehearsal, which is not evidence). The plan's first LEARN input varied with
   the turn. After about 930 turns K1 refused a LEARN as non-finite, before any commit, as its law requires. That ended
   the 1,000-publication history before G6 could be judged. The input is now the host suite's fixed fixture, which a
   1,000-publication history is known to keep finite (`test_long_runs_keep_the_home_fixed`). Every LEARN still moves
   W, so every turn still has its own identity. No gate, point or expectation changed.

## Would establish / would not establish

Passing would establish, on the qualifying platform and filesystem, for one commit:
- crash consistency of RuntimeCore ABI v6's durability protocol over the ECS_C authority under the listed faults;
- the authority's refusal of the listed state-directory attacks until explicit operator reconciliation;
- recovery work and storage fixed in history length.

That is the condition docs/NATIVE_SPLIT_CENSUS.md sets for retiring standalone continuity.

It would **not** establish:
- durability beyond `fdatasync`, or power-loss behavior from `SIGKILL`;
- protection against a holder of the deployment key, against code inside the process, or against an adversary who
  can write the authority directory;
- the security of the HMAC construction beyond its stated use;
- the operator's identity (the deployment key authenticates records, not people);
- anything about the meaning of any memory content.

## Evidence

`research/qualification/ecs_c_recovery_q0/evidence/` holds the directory written once by the QUAL run on `117b41a` and
sealed by `runner.py seal`.
- The repository keeps no file over 2 MiB, so `records.jsonl` (5948358 bytes) is committed gzip-compressed as
  `records.jsonl.gz` (SHA-256 `6d40f11b7caf40e90734278db930c1188543460605411fc091407d4f6b8c33e7`).
- Every other file is committed as sealed.

To validate, restore the sealed directory, then run the frozen validator. It checks:
- exactly its four files;
- every SHA-256, so the restored records are the written bytes;
- that the run was a complete QUAL run;
- the verdict, recomputed from the records.

```sh
mkdir sealed && cp evidence/FREEZE.json evidence/verdict.json evidence/SHA256SUMS sealed/
gunzip -c evidence/records.jsonl.gz > sealed/records.jsonl
python -I -B check.py evidence sealed
```

| File | SHA-256 |
|---|---|
| `FREEZE.json` (byte-identical to the published `FREEZE.json`) | `fdfbde9fd184a0ca6255a94d26fe10a4d940e81add1adbc20afcb46b978c7a6f` |
| `records.jsonl` (the run header and 221 trial records; committed as `records.jsonl.gz`) | `61bc0c46b46060cdbe9028b99dcdd072f913afe26be40a940790f3e75cf1902e` |
| `verdict.json` | `5f59f912bc5bbc009d3c58d7ccfd97ae949b54eb9c3dcad31f177a4fb90bac4c` |
| `SHA256SUMS` | `1f07fc8a2d01139d2c488bb3ad509b566218fc27bcf0f9d3954bf1c7e7b83ffb` |

**The run.**
- The QUAL run first compared the executing object with the freeze record and found no difference. The same
  comparison after the run found none either, and the checkout stayed clean.
- The run's stderr held only its timing: 88.3 s.
- Every trial's home was removed once recorded.

**In CI.**
- `tests/research/qualification_ecs_c_recovery_q0/test_evidence.py` re-validates the evidence. It binds it to the
  published freeze record and requires the instrument files in the tree to be the ones the freeze bound. A changed
  instrument is a new instrument and needs new evidence.
- `tests/boundary/test_qualification_specs.py` requires this document's status to repeat the recomputed verdict.
