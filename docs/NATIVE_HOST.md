# The native operator host

`elpis` (`native/runtime/host`) is the native production entry point of one Elpis deployment: one C program, no
interpreter. It links only the C library and a static SHA-256. Every Elpis library it runs is admitted at run time
from the deployment's pinned `lib/` directory (see "Admission"). RuntimeCore (ABI v7, `docs/RUNTIME_CORE.md`) composes every turn over
the ECS_C authority (`docs/ECS_C_AUTHORITY.md`); the ECS composite holds the lineage's state; structure identifies
every hyperedge. The host computes no ECS mathematics and touches no ECS byte: it validates operator input, marshals
it into the admitted libraries' C ABIs and reports what they answer.

It is Stage F of `docs/NATIVE_SPLIT_CENSUS.md`. The Python operator interface (`python -m elpis.runtime.cli`,
`docs/OPERATING_ELPIS.md`) is the offline reference interface over the same ECS_C authority: differential
tests, the qualification instruments bound to their commits, research. It is not the production path (see
"Holds").

```
elpis --home DIR init --native-build BUILD [--capacity N] [--authority-dir DIR] [--source-commit SHA]
elpis --home DIR create [--dim 6 --width 36 --max-rows 64 --memory-dim 64 --banks 2 --slot-capacity 64
                         --codebook-capacity 128 --seed 0]
elpis --home DIR [--contained] <command> [options]
elpis --home DIR [--contained] session --script FILE|-
```

## Installation

`cmake --install BUILD --prefix P --component runtime` installs the host as `P/bin/elpis` and the seven libraries it
admits, flat, in `P/lib/elpis/`. They are installed without a RUNPATH. `elpis init --libraries P/lib/elpis` pins them
from there; `init --native-build BUILD` pins them from a build tree at their build paths. Either way the source is
never used again.

`--component testing` adds `P/bin/elpis_testing` and `P/lib/elpis/libelpis_runtime_testing.so`, for the crash suites
only. With `ELPIS_NATIVE_HOST_PREFIX=P`, `tests/integration/test_native_host.py` runs against the installed binaries,
and `native/runtime/host/tests/host_smoke.cmake` takes `-DLIBRARIES=P/lib/elpis`. The Installed system acceptance job
of CI does both, offline and unprivileged.

## The home

| Path | Written | Content |
|---|---|---|
| `host.cfg` | once, by `init` (0600) | the configuration and trust root: the host binary's SHA-256, the catalog's SHA-256, the ABI (runtime 7, composite 1, ECS_C 1), the build variant, the authority directory, the entity capacity |
| `catalog` | once, by `init` (0400) | the seven pinned libraries in admission order: id, file, SONAME, size, SHA-256 |
| `lib/` | once, by `init` (files 0500) | the libraries, copied from the operator build; the build tree is never used again |
| `state/` | provisioned by `init` | the ECS_C state directory: `authority.a`, `authority.b`, fixed-size records |
| `authority/` (or `--authority-dir`) | provisioned by `init` | the ECS_C authority directory: `deployment.key`, `witness.a`, `witness.b` |
| `lineage.cfg` | once, by `create` (0400) | the lineage configuration (the Python deployment's `LineageConfig`, same defaults), its initial identity and envelope size, and the provenance of its initial weights |
| `checkpoint/` | provisioned by `create` | the K1 Recovery R0 store: `k1-checkpoint.a`, `k1-checkpoint.b`, sized for the composite envelope |

The home must be an absolute path without `.` or `..` components, a real directory (not a symlink) owned by the
current user and writable by nobody else.

- **`init`** refuses a non-empty home (`DEPLOYMENT_EXISTS`). `host.cfg` is written last: an interrupted `init` leaves a
  home that neither `init` nor any other command accepts, and the operator removes it (it holds no lineage).
- **`create`** publishes `lineage.cfg` and the checkpoint store under temporary names (`lineage.cfg.new`,
  `checkpoint.new`) renamed into place, so each exists whole or not at all. Then it anchors.
- **An interrupted `create`** (process death, or a refusal after `lineage.cfg`) is completed by `create` with the same
  arguments:
  - `lineage.cfg` must be byte-identical to what it would write, or it is refused (`LINEAGE_EXISTS`);
  - its own debris (`*.new`) is removed and what is missing is made;
  - an unanchored lineage (`NOTHING_TO_RECOVER`) is anchored;
  - a lineage the authority already holds at its initial identity is reported (`"resumed": true`) with nothing
    written.
- **Anything else** is refused (`LINEAGE_EXISTS`): another lineage, a checkpoint store without `lineage.cfg`, or a
  lineage that has moved past its anchor. Nothing is ever overwritten.

**Bounded persistence** (census section 5a; `elpis.runtime.persistence`: `native_host_deployment`,
`ecs_c_authority_provisioning`, `ecs_c_authority_slots`, `k1_checkpoint_slots`). After `create` the home is a fixed set
of 17 files of fixed sizes. Every later write rewrites a fixed slot in place: one authority record and one witness per
transition, and one checkpoint slot per recoverable publication. The host writes nothing itself after `create`: no
log, cache, history or temporary file. Recovery reads the two record slots and two witnesses whatever the length of
the history. Entity rows are bounded by the provisioned capacity: founding past it is refused before any write
(`ECS_C_CAPACITY`), and a TERMINATED source's row is freed only by an explicit `source-reclaim`.

## Admission

At every start, after the home checks:

1. **The host binary.** The SHA-256 of `/proc/self/exe` must equal `host.cfg`'s pin (`DEPLOYMENT_PIN`). A rebuilt
   host is a different host: the deployment is re-initialized, never re-pinned implicitly.
2. **The configuration.** `host.cfg` must have this schema and ABI (`DEPLOYMENT_ABI`) and this build variant: the
   production host refuses a testing deployment and the testing host a production one (`DEPLOYMENT_CONFIG`).
3. **The catalog.** Its SHA-256 must equal `host.cfg`'s pin, and it must name exactly this host's library set, in
   order, with these files and SONAMEs (`DEPLOYMENT_PIN`, `DEPLOYMENT_CONFIG`).
4. **Each library, in dependency order:**
   1. it is opened beneath `lib/` with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS)`;
   2. its size must equal its pin;
   3. it is copied into a memfd and checked to be unchanged by the copy (`fstat` before and after);
   4. the memfd is sealed against writes and resizing;
   5. the seal's SHA-256 must equal the pin;
   6. the seal's ELF dynamic section is read: every `DT_NEEDED` must be the SONAME of a library admitted before it,
      or a platform library (the C runtime, `libm`, `libgcc_s`, the loader, the sanitizer runtimes);
   7. only then is it `dlopen`ed from `/proc/self/fd/N` (`RTLD_NOW | RTLD_LOCAL`).

   The sealed descriptor stays open for the life of the process (seven descriptors). The loader knows a loaded object
   by its path, and a reused descriptor number would make it hand back the earlier object. A handle that repeats an
   earlier one is refused.
5. **SONAME binding.** Afterwards every SONAME must bind exactly the object admitted under it (`dlopen(soname,
   RTLD_NOLOAD)`). Every function the host calls is then resolved from those handles. RuntimeCore's ABI must be 7
   and ECS_C's 1.

Admission order: `libelpis_ecsg_math`, `libelpis_ecsg_k1`, `libelpis_ecs_memory`, `libelpis_ecs_composite`,
`libelpis_hacf_observation`, `libelpis_ecs_c_authority`, `libelpis_runtime` (the testing host:
`libelpis_runtime_testing`, whose SONAME is the production one).

**Threat model.** The pins authenticate bytes, not a publisher. The loader, the kernel and the platform libraries are
trusted. The home is the operator's: an adversary who can write `host.cfg`, `catalog` or `lib/` is not defended
against (as code inside the process is not). The ECS_C authority defends its own threat model T1: an adversary who
can write the state directory but neither read the deployment key nor write the authority directory
(`docs/ECS_C_AUTHORITY.md`, "Threat model"). The authority directory is `<home>/authority` unless `init
--authority-dir` names another protected directory.

## Containment

`elpis --home DIR --contained <command>` restricts the host to its deployment's fixed slots. The restriction is the
policy of `native/runtime/containment/elpis_autonomous_run.c`, applied by the host to itself after admission. Admission
copies the libraries into memfds, which a file size limit applied earlier would stop.

1. **Before anything is opened.** A writable regular file or directory on a standard descriptor is refused (use a
   pipe), and every inherited descriptor above them is closed.
2. **After admission, the grant.** Landlock (ABI v5 or later) grants write access to exactly six files: the two
   ECS_C records (`R(capacity)` bytes each), the two witnesses (240 bytes each) and the two checkpoint slots (128 +
   the envelope size). Each is checked to be exactly that: a regular file of that size, one link, owned by the
   user, private.
3. **Everything else is refused.** Creating, removing, renaming, linking or truncating any file, writing any other
   file, and device ioctls are refused.
4. **Further limits.** `RLIMIT_FSIZE` stops any write past the largest slot. A seccomp filter refuses `chmod`,
   `chown`, `utime`, extended attributes and `fallocate`. `PR_SET_NO_NEW_PRIVS` is set.

Refused with `CONTAINMENT_REFUSED`: a privileged process (root keeps capabilities that leave the domain, as the
launcher states), a kernel without Landlock ABI v5, a slot that is not exactly its fixed file, and any restriction that
cannot be applied. `init` and `create` create files and are never contained (`CLI_INPUT`). An authority slot
shortened by an adversary is a torn prefix to the authority (docs/ECS_C_AUTHORITY.md, "File sizes") but not exactly
its fixed file to containment: reconcile it uncontained, which rewrites the slot at its fixed size.

One file size limit covers all six slots, so under containment each slot is bounded by the largest slot's size,
not by its own: a fixed bound of six times the largest slot. The sizes a correct run writes are checked by the
fixed-file-set tests. Linux x86-64 only (the seccomp table is reviewed for that architecture).

## Operations

A one-shot command runs one operation; `learn --publish` and `evidence --publish` run two. A `session` script runs
many in one process. A script is a JSON array of operations, or JSON lines, in which blank lines and lines starting
with `#` are skipped. Every line is parsed before anything runs.

| Operation (command) | What it does |
|---|---|
| `status` | deployment, pins, ABI, budget ceiling, capabilities and non-claims, lineage, the authority summary, the recovery disposition, durable and live identities, footprint, the durability lane, the fail-stop. A session that cannot open is reported (`session_refusal`), not an error of `status` |
| `open` / `close` | open the lineage explicitly; close it (`volatile_work_discarded` says whether unpublished work was dropped). The next operation reopens: a restart inside one process |
| `query --x ROWS` | numeric QUERY of the live state (read-only); values, their SHA-256, the live generation |
| `learn --x --y --schedule --rate --grant` | one managed LEARN turn: VOLATILE until published. Requires an explicit grant and a positive finite rate |
| `evidence --items --grant --sources` | typed HACF evidence in one observe-only turn, all or nothing. Every item's source must be in the grant's sources and be an ACTIVE source of the authority within its per-turn bound (checked natively before the call) |
| `lookup --relation --target-role --cues` | exact LOOKUP by net evidence (a conflict between sources is reported, never resolved) |
| `slot --key` | one memory slot and the hyperedge structure rebuilds from it |
| `live`, `counters [--reset]`, `candidates` | the live identities; RuntimeCore's native crossings; the recovery disposition |
| `publish` | recoverable publication (checkpoint slot verified, then one authority record and witness). Anything but ACKNOWLEDGED is refused with its ticket: `DURABLE_FAILED`, `DURABLE_UNCERTAIN`, `DURABLE_SUPERSEDED`, `DURABLE_ABANDONED` |
| `reconcile (--discard ID \| --adopt ID)` | the K1 checkpoint's unresolved candidate, exactly the one named |
| `authority` | the authority: summary or (no runtime open) its record status, recovery report and the one reconciliation it accepts; every entity row |
| `authority-reconcile --kind --digest` | send exactly the reconciliation `authority` reports (`ROLLBACK`, `UNAUTHENTICATED`, `REPIN` and its digest); anything else is refused before an event exists (`AUTHORITY_RECONCILE_MISMATCH`) |
| `source-found`, `source-lifecycle`, `source-version`, `source-reclaim` | operator events (ENTITY_FOUND then FOUNDED -> ACTIVE unless `--dormant`; ENTITY_LIFECYCLE; SOURCE_VERSION, compare-and-set on `--expected`; ENTITY_RECLAIM of a TERMINATED source) |
| `text` | refused: `ECS_CODEC_UNQUALIFIED`. No qualified ECS<->DSV4 codec map exists |

Operator events reach the authority through RuntimeCore while it holds the authority: the host builds them from
`elpis_runtime_authority_entity_at` and `elpis_runtime_authority_next_entity_id`. When no runtime is open in the
process, they go to the authority directly through its own ABI. Either way an authority refusal is reported by its
exact ECS_C code.
A source's label and a scope given as text (`scope_label`) are hashed:
SHA-256(`elpis.host.label.v1` || 0x00 || text). A scope given as `scope` is 64 hex characters. A source needs a
non-zero scope. A memory source authority identifier names one present source row (a second is
`ECS_C_SCOPE_CONFLICT`).

**Output and exit status.** Each operation prints one JSON object on one line:
`{"op": ..., <its members>, "ok": true}`, or `{"op": ..., "ok": false, "code": ..., "detail": ...}`. An operation
that names `"expect": CODE` is met exactly when it is refused with CODE.

| Exit | Meaning |
|---|---|
| 0 | every operation succeeded or was refused as expected |
| 2 | the command line, an argument or a script is malformed; nothing ran |
| 3 | an operation was refused, or an expectation was unmet |
| 4 | the deployment could not be opened; nothing ran |

Codes are stable names:
- RuntimeCore's (`RUNTIME_*`, and `CONTINUITY_*` for its continuity-space codes);
- the composite's (`ECS_COMPOSITE_*`) and ECS_C's (`ECS_C_*`);
- the host's: `CLI_INPUT`, `DEPLOYMENT_*`, `LINEAGE_*`, `LEARN_UNAUTHORIZED`, `LEARN_AUTHORITY`, `LEARNING_RATE`,
  `STIMULUS`, `OBSERVATION_*`, `DURABLE_*`, `AUTHORITY_RECONCILE_MISMATCH`, `HOST_RESOURCES`, `HOST_OUTPUT`,
  `ECS_CODEC_UNQUALIFIED`.

**Limits.** One operation's JSON, and a script, is at most 8 MiB, and one output line at most 8 MiB. Integers in
operator JSON are exact up to 2^53. QUERY takes at most 4,096 rows; LEARN at most 64 experiences of 1..256 rows
and 1..2^20 steps; one evidence turn at most 64 observations. Structure's assertion validator admits HACF authority
values 0..3 only, so at most four sources can record HACF evidence.

**Volatile and durable.** A session's work is volatile until `publish`. A process that exits without publishing
loses it, and restart resumes exactly the last acknowledged state (`docs/K1_RECOVERY_R0.md`). Process death during a
publication is resolved by the next open: no authority reconciliation is needed. An interrupted checkpoint write
leaves an unresolved candidate, which the operator discards or adopts. Copied-back records, a corrupt newest record,
an unauthenticated substitute and a missing witness refuse the runtime (`RUNTIME_AUTHORITY`) until the operator
reconciles exactly what the authority names.

## The testing host

`elpis_testing` runs only a testing deployment (`libelpis_runtime_testing`). It adds four operations:
- `testing_durable_fault`: RuntimeCore's durability fault plans;
- `testing_control_fault`: its control-operation plans;
- `testing_wait_parked`: wait, bounded, until the durability worker parks at a step;
- `testing_pause`, `testing_exit`: pause at an operation boundary until one byte arrives on standard input, or die
  there.

A harness kills the process at a park or inspects it live.

## Tests

`tests/integration/test_native_host.py` runs the host as separate processes with a minimal environment (no
`PYTHONPATH`, no `LD_*`):

| Area | What is checked |
|---|---|
| Closure | the binaries need only `libc.so.6` and carry no RUNPATH; a live session maps only the C runtime and seven distinct sealed objects; its descriptors are one fixed set across 30 publications and 20 evidence turns |
| Pins | every drift refuses every start: a library, the catalog, the ABI, the catalog pin, the host binary, the build variant; a pinned library naming an unadmitted dependency is refused before it loads |
| Differential | identities, QUERY values, the LEARN transition, relation keys, the LOOKUP answer and the component identities equal the Python operator interface's, over the other durable authority |
| Operation | publication and exact resumption; unpublished work is lost and reported; refusals carry stable codes and change nothing; operator events work directly and through the held runtime; capacity is refused before any write and reclamation frees a row |
| Durability | SIGKILL at each of the nine durable steps, and every failure plan, resolve exactly as the durability law says; copied-back records and a corrupt newest record need the operator's exact reconciliation |
| Boundedness | files, sizes and recovery work stay constant over 60 publications and 4 restarts (FAST), and over 1,000 publications and 20 restarts (STRESS) |
| Containment | a contained session learns, records evidence, publishes and resumes; creating a file, writing the configuration or catalog, and extending a slot past the limit are refused (the same probes succeed uncontained), while the granted slots stay writable; the file set is unchanged. Refused: provisioning under containment, a writable file as stdout, a slot that is not exactly its size, a privileged process |

`runtime.native_host_smoke` (ctest, every native build including the sanitizer and clang jobs) runs init, create, a
session and a resumption with no interpreter. The suite passes on the Release and ASan+UBSan builds. Eight targeted
mutants of the host are each killed by it: each pin, failed-publication reporting, the grant's sources, the
reconciliation digest, refusal translation, the all-or-nothing abort and the learning rate.

## Holds

- **Qualification of ABI v7.** Standalone continuity retired into ECS_C (RuntimeCore ABI v7; census section 9): the
  Python deployment and CLI are the offline reference interface over the same ECS_C authority, off the production
  path. ECS_C Recovery Q0 is RUN PASS for commit `117b41a` (ABI v6) only; its ABI v7 successor is not run.
- **Containment.** The native host contains itself (`--contained`) with the ECS_C and checkpoint slots as its grant.
  `elpis_autonomous_run`, for the Python reference, grants the four files of a provisioned ECS_C authority (its two
  record and two witness slots).
- **Release.** No installed release bundle exists for the native host yet (`ELPIS_SYSTEM.json`
  `identity.release_authority = NONE_DEFERRED`).

## Not claimed

- No text interface and no language understanding.
- No durability across power loss: crash consistency is established for process death only.
- No protection beyond the threat model above.
- No scientific claim of any kind.
