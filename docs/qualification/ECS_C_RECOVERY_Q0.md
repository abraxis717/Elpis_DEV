# ECS_C Recovery Q0 under RuntimeCore ABI v6: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED**

THE ECS_C AUTHORITY DECIDES. CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. ONLY THE OPERATOR RECONCILES AN ATTACK.

Pre-registered before any run, before its instrument exists. It is the qualification the successor of standalone
continuity needs before continuity may be retired (docs/NATIVE_SPLIT_CENSUS.md, sections 9 and 11: "retire standalone
continuity only after qualified successors exist"). K1 Recovery R0 v4 ([K1_RECOVERY_R0_V4.md](K1_RECOVERY_R0_V4.md))
qualified the durability protocol with continuity as the authority register, for commit `18f2f4d` only, and its premise
P1 binds `native/continuity`. It does not extend to the ECS_C authority, nor to RuntimeCore ABI v6, nor to the native
host. This specification asks the same question of the successor and adds the adversary the successor exists to
defend against.

## Why it cannot run yet (the conflict, stated)

* **The accepted source.** As for K1 Recovery R0 v4, the freeze binds one exact commit: the merge commit integrating the
  native split (Stages B to F of docs/NATIVE_SPLIT_CENSUS.md) into `main`, built and installed from a clean checkout of
  it. Those stages are open pull requests; merging needs the owner's explicit authorization. Until then no QUAL run is
  possible, and standalone continuity stays (it is the authority the offline Python reference interface publishes to).
* **The instrument.** `research/qualification/ecs_c_recovery_q0/` (plan, runner, checker, freeze verification, seal) does
  not exist yet. It must be written, reviewed and rehearsed on DEV inputs before the freeze, with the discipline of
  K1 Recovery R0 v4's instrument: the freeze verifies the executing object, records carry no expectation, and an
  independent checker derives every expectation from the plan.

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

1. **Lineage.** One ECS composite lineage (`create --seed s`, the LineageConfig defaults) and one ACTIVE source
   (per-turn bound 8). Every LEARN turn is followed by one evidence turn of one observation.
2. **Publication faults.**
   - `publish`: `SIGKILL` while the worker is parked at each lane step 1-9 (`testing_durable_fault` action 2).
   - `failure`: the in-process failure plans 3-10 (no kill).
   - Each runs 4 times per point.
3. **Control-operation faults.** The anchor (points 1-6), and the checkpoint discard (points 1-3) and adopt (points 1,
   4-6) of a candidate left by a step-5 kill, each parked and `SIGKILL`ed (`testing_control_fault` action 2), 4 runs
   per point.
4. **Random kills.** 32 randomized `SIGKILL`s while learning, observing and publishing.
5. **Attacks**, each made with the runtime closed, against the state directory only (the T1 adversary of
   docs/ECS_C_AUTHORITY.md), 4 runs each:
   - copying back both records saved 1, 2 and 5 publications earlier (a rollback);
   - a byte flip in the newest record at registered offsets: the header, the generation, the payload, the MAC and
     the checksum;
   - a validly checksummed substitute: a field changed and the checksum recomputed, the MAC left unchanged;
   - a forged generation, and a forged evolution revision and head, each with a valid checksum;
   - the newest record deleted (truncated to zero), and both slots swapped.
6. **Witness loss.** The authority directory's witnesses truncated to zero. This is outside T1 and is the
   `WITNESS_MISSING` path.
7. **Boundedness.** Histories of 10, 100 and 1,000 publications; the home's file set and the recovery work are recorded
   after each.

The operator action is pre-registered:
- **For an attack:** reconcile exactly what `authority` reports on even run indices, and on odd ones first a
  reconciliation with a wrong digest, which must be refused, then the exact one.
- **For a checkpoint candidate:** adopt on even run indices, discard on odd ones.

## Procedure

For every point and run, a fresh deployment is made:
1. `init --libraries`, `create`, and the source.
2. A child learns, observes and publishes (acknowledged), then learns and observes again (the candidate).
3. The child is parked at the point, or the fault or attack is applied, and is killed where the family says.
4. A fresh process records:
   - the authority report and reconciliation;
   - the status, the recovery disposition, the durable and live identities and the resident relation keys;
   - the home's files (names, sizes, inodes);
   - the outcome of the pre-registered operator action.
5. It then learns, observes and publishes once more, and a further fresh process checks that it resumes exactly that
   publication.

## Gates (pre-registered)

* **G1 (process death).** Every recovery disposition and authority report is the one the crash law prescribes for its
  point (docs/RUNTIME_CORE.md, "Durability crash matrix"; the table in `tests/integration/test_native_host.py` states
  the expectation the instrument must derive independently). No process-death trial ever needs authority
  reconciliation.
* **G2 (exactness).** Every resumed state's identity equals the authority's. A publication's observations are present
  exactly when it became authoritative. The evidence count advances exactly by the committed observations of
  acknowledged publications.
* **G3 (no silent acceptance).** Every attack refuses the runtime's open (`RUNTIME_AUTHORITY`) with the disposition the
  authority's recovery law prescribes. A wrong reconciliation is refused and changes nothing. After the exact one,
  the authority continues from the accepted record at the abandoned generation + 1, and the K1 checkpoint is
  reconciled separately. No attack ever resumes a state the operator did not accept.
* **G4 (no implicit action).** Nothing is adopted, discarded or resumed without its explicit call. No recoverable
  publication succeeds over an unresolved candidate. No ticket violates the acknowledgment law.
* **G5 (fixed storage).** After every trial the home holds exactly its 17 files, at their provisioned sizes and
  inodes.
* **G6 (bounded recovery).** The recovery work (reads, bytes read, replays) is the same after 10, 100 and 1,000
  publications.

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

`research/qualification/ecs_c_recovery_q0/evidence/` will hold the directory written once by the QUAL run on the
accepted source and sealed by the instrument, validated by its independent checker. It does not exist: this
specification is NOT RUN.
