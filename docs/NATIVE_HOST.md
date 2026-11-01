# The native operator host

> **VOLATILE MIGRATION: COMPLETE** (`docs/ELPIS_MISSION.md`, Amendment 3; `docs/VOLATILE_RUNTIME_CENSUS.md`).
> The host is RAM-only:
> - it has no deployment home, configuration file, lineage file, authority, checkpoint or log;
> - it reads immutable libraries and admits them into sealed memory;
> - it contains itself so that it holds no write permission to any file;
> - it refuses to start where it cannot establish that.
>
> Each session lives in memory and ends with `close` or the process. The persistent host of RuntimeCore ABI v7 (`init`,
> `create`, `publish`, `reconcile`, `--home`) is retired; git history keeps its specification.

`elpis` (`native/runtime/host`) is the native production entry point of Elpis: one C program, no interpreter. It links
only the C library and a static SHA-256. Every Elpis library it runs is read, never written, and admitted at run time
into sealed memory against the operator's catalog pin (see "Admission").
- **RuntimeCore** (ABI v8, `docs/RUNTIME_CORE.md`) composes every turn in a volatile ECS_C session
  (`docs/ECS_C_SESSION.md`).
- **The ECS composite** holds the lineage's state, in memory.
- **Structure** identifies every hyperedge.

The host computes no ECS mathematics and touches no ECS byte: it validates operator input, marshals it into the
admitted libraries' C ABIs and reports what they answer.

It is Stage F of `docs/NATIVE_SPLIT_CENSUS.md`. The Python operator interface (`python -m elpis.runtime.cli`,
`docs/OPERATING_ELPIS.md`) is the offline reference interface over the same RuntimeCore library. It serves differential
tests and research, and is not the production path.

```
elpis (--libraries DIR | --native-build BUILD) catalog
elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] <command> [options]
elpis (--libraries DIR | --native-build BUILD) --catalog-sha256 HEX [session options] session --script FILE|-

session options: --dim 6 --width 36 --max-rows 64 --memory-dim 64 --banks 2 --slot-capacity 64
                 --codebook-capacity 128 --seed 0   (the lineage every session starts from)
                 --capacity 16                      (the session's ECS_C entities, 2..4096)
                 --nonce-seed HEX                   (64 hex: reproducible sessions, for differential runs)
```

## Installation

`cmake --install BUILD --prefix P --component runtime` installs:
- the host as `P/bin/elpis`;
- the six libraries it admits, flat, in `P/lib/elpis/`, without a RUNPATH.

The host reads them with `--libraries P/lib/elpis`, or from a build tree at their build paths with `--native-build
BUILD`. Either way the directory is only read. It may be, and in deployment should be, on a read-only mount.

The testing host `elpis_testing` is test support, built in the build tree and never installed. With
`ELPIS_NATIVE_HOST_PREFIX=P`, `tests/integration/test_native_host.py` runs against the installed host, and
`native/runtime/host/tests/host_smoke.cmake` takes `-DLIBRARIES=P/lib/elpis`. The Installed system acceptance job of CI
does both, offline and unprivileged.

## The catalog pin

There is nothing to initialize. `elpis catalog` measures the host binary (`/proc/self/exe`) and every library from
its sealed copy. It prints the catalog, one canonical JSON object naming:
- the host binary's SHA-256;
- for each library in admission order: its id, file, SONAME, size and SHA-256.

It also prints the catalog's SHA-256. Every other command requires that SHA-256 (`--catalog-sha256`), so the
operator's one pin authenticates the host binary and every library byte that runs. Any drift refuses the start with
`INSTALL_PIN`, before any operation: a library, the host binary, a missing library, or a library naming an unadmitted
dependency.

The lineage is not a file. Every session starts from a deterministic initial composite, built in memory from the
session options (the same defaults as the Python `LineageConfig`). The same options and seed always give the same
initial identity.

## Admission

At every start, after the process containment below:

1. **The host binary** is measured: the SHA-256 of `/proc/self/exe`.
2. **Each library, in dependency order:**
   1. it is opened read-only beneath the library directory with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
      RESOLVE_NO_MAGICLINKS)`;
   2. it is copied into a memfd, which is then sealed against writes and resizing;
   3. the seal's size and SHA-256 are measured.

   The library file is read, never written. The bytes that run are the sealed in-memory copy.
3. **The pin.** The measured catalog's SHA-256 must equal `--catalog-sha256` (`INSTALL_PIN`). Only then is anything
   loaded.
4. **Loading, in order.** Each seal's ELF dynamic section is read: every `DT_NEEDED` must be the SONAME of a library
   admitted before it, or a platform library (the C runtime, `libm`, `libgcc_s`, the loader, the sanitizer runtimes).
   Then it is `dlopen`ed from `/proc/self/fd/N` (`RTLD_NOW | RTLD_LOCAL`).

   The sealed descriptors stay open for the life of the process (six descriptors). The loader knows a loaded object by
   its path, and a reused descriptor number would make it hand back the earlier object. Each sealed descriptor is
   reopened read-only, so no descriptor the process holds can write anything.
5. **SONAME binding.** Every SONAME must bind exactly the object admitted under it (`dlopen(soname, RTLD_NOLOAD)`).
   Every function the host calls is then resolved from those handles. RuntimeCore's ABI must be 8 (`INSTALL_ABI`).

Admission order: `libelpis_ecsg_math`, `libelpis_ecsg_k1`, `libelpis_ecs_memory`, `libelpis_ecs_composite`,
`libelpis_hacf_observation`, `libelpis_runtime`. RuntimeCore embeds the volatile ECS_C session, and no other ECS_C
library is loaded.

**Threat model.** The pins authenticate bytes, not a publisher. The loader, the kernel and the platform libraries are
trusted. Code inside the process can read and corrupt its memory; nothing defends against it. There is no persistent
state to protect. The durable authority's threat model T1 is retired with it.

## Containment

Containment is mandatory: there is no uncontained mode, and the v7 `--contained` flag is retired. It runs in two
stages (`host_contain.h`), and any failure refuses the start with exit 4, before any operation runs.

1. **The process, first, before any library is read.**
   - **Privilege.** A privileged process is refused (`CONTAINMENT_REFUSED`). Root keeps capabilities, mounts and
     module loading, which leave the domain.
   - **Standard descriptors.** A standard descriptor that is writable storage is refused: a regular file, a directory,
     a block device, or a character device other than a terminal, `/dev/null`, `/dev/zero` or `/dev/full`. When
     standard output is storage, the refusal goes to standard error, or nowhere if that is storage too: the host
     never writes a byte through such a descriptor. Every inherited descriptor above the standard three is closed.
   - **Core dumps.** `RLIMIT_CORE` is set to 0 and `PR_SET_DUMPABLE` to 0, and both are read back.
   - **Swap.** `/proc/swaps` is read. With any swap area active, every page of the process is locked in RAM
     (`mlockall(MCL_CURRENT | MCL_FUTURE)`), or the host refuses (`SWAP_POLICY`). An unreadable `/proc/swaps` is
     refused too. The host never changes the machine's swap configuration. The host's memory is sized to lock whole
     under a common 8 MiB `RLIMIT_MEMLOCK`.
2. **The files, after admission and before the first operation.**
   - **Landlock** (ABI v5 or later) handles every filesystem mutation right and grants none, so no file anywhere may
     be written, created, removed, renamed, linked or truncated. Device ioctls are refused too.
   - **`RLIMIT_FSIZE`** is set to 0, so no write may give any file a byte. It is set only now, because the memfd seals
     of admission are written before it.
   - **No new privileges, and a seccomp filter.** The filter refuses:
     - the metadata calls Landlock does not mediate: `chmod`, `chown`, `utime` and extended attributes;
     - preallocation and truncation;
     - every namespace operation that can mount or remove names;
     - `sync` and `syncfs`, `swapon` and `swapoff`, and module and `kexec` loading;
     - sockets, so no logging daemon is reachable;
     - `io_uring`, `ptrace` and `process_vm_writev`;
     - process creation, so no hidden writer child exists.

`status` reports what was established: the Landlock ABI, the restriction, the core limit read back, `dumpable`, the
file size limit, the swap areas and whether memory is locked. Linux x86-64 only (the seccomp table is reviewed for
that architecture).

## Sessions

A session is one volatile ECS_C session and one composite, both in memory.
- **Opening.** `open`, or the first operation that needs a session, starts one at a fresh genesis, anchored at the
  lineage's initial identity.
- **Closing.** `close` ends it and discards everything it learned or recorded.
- **Restarting.** A new process starts again at the same initial identity. Nothing is published, checkpointed,
  recovered or resumed.

Without `--nonce-seed`, the kernel draws each session's nonce. With it, sessions are reproducible across processes,
which differential runs need.

The session's footprint is fixed when it opens: the session (`1,424 + 464 × capacity` bytes), the composite envelope,
and its memory's slot capacity. Exhaustion refuses atomically. Founding a source past the capacity is refused before
anything changes (`ECS_C_CAPACITY`), and a TERMINATED source's row is freed only by an explicit `source-reclaim`.

## Operations

A one-shot command runs one operation. A `session` script runs many in one process. A script is a JSON array of
operations, or JSON lines, in which blank lines and lines starting with `#` are skipped. Every line is parsed before
anything runs.

| Operation (command) | What it does |
|---|---|
| `status` | the installation (schema `elpis.native-host.v2`, the host and library pins), `persistence: NONE`, the ABI (runtime 8, composite 1), the containment established, the budget ceiling, capabilities and non-claims; with a session open, the session summary and the footprint |
| `open` / `close` | open a session explicitly; close it, discarding its work |
| `query --x ROWS` | numeric QUERY of the live state (read-only): the values, their SHA-256 and the live generation |
| `learn --x --y --schedule --rate --grant` | one managed LEARN turn, volatile. Requires an explicit grant and a positive finite rate |
| `evidence --items --grant --sources` | typed HACF evidence in one observe-only turn, all or nothing. Every item's source must be in the grant's sources, and must be an ACTIVE source of the session within its per-turn bound (checked natively before the call) |
| `lookup --relation --target-role --cues` | exact LOOKUP by net evidence. A conflict between sources is reported, never resolved |
| `slot --key` | one memory slot, and the hyperedge structure that rebuilds from it |
| `live`, `counters [--reset]` | the live identities; RuntimeCore's native crossings and session transitions |
| `authority` | the session: summary and every entity row |
| `source-found`, `source-lifecycle`, `source-version`, `source-reclaim` | operator events: ENTITY_FOUND, then FOUNDED -> ACTIVE unless `--dormant`; ENTITY_LIFECYCLE; SOURCE_VERSION, compare-and-set on `--expected`; ENTITY_RECLAIM of a TERMINATED source |
| `text` | refused: `ECS_CODEC_UNQUALIFIED`. No qualified ECS<->DSV4 codec map exists |

Operator events reach the session through RuntimeCore (`elpis_runtime_authority_transition`). The host builds them
from `elpis_runtime_authority_entity_at` and `elpis_runtime_authority_next_entity_id`, and reports a session refusal
by its exact ECS_C code.
- **Labels.** A source's label, and a scope given as text (`scope_label`), are hashed:
  SHA-256(`elpis.host.label.v1` || 0x00 || text).
- **Scopes.** A scope given as `scope` is 64 hex characters. A source needs a non-zero scope.
- **Source authorities.** A memory source authority identifier names one present source row. A second is
  `ECS_C_SCOPE_CONFLICT`.

**Retired commands** are refused with `OPERATION_RETIRED` and exit 2, and nothing runs. They are the v7 commands
`init`, `create`, `publish`, `reconcile`, `candidates` and `authority-reconcile`, and the options `--home`,
`--contained` and `--publish`. They belonged to the persistent runtime: there is no home, no publication and nothing
to reconcile.

**Output and exit status.** Each operation prints one JSON object on one line: `{"op": ..., <its members>, "ok":
true}`, or `{"op": ..., "ok": false, "code": ..., "detail": ...}`. An operation that names `"expect": CODE` is met
exactly when it is refused with CODE.

| Exit | Meaning |
|---|---|
| 0 | every operation succeeded or was refused as expected |
| 2 | the command line, an argument or a script is malformed, or names a retired command; nothing ran |
| 3 | an operation was refused, or an expectation was unmet |
| 4 | the host could not start: containment, the swap policy, admission or the catalog pin refused; nothing ran |

Codes are stable names:
- RuntimeCore's: `RUNTIME_*`, and `CONTINUITY_*` for its authority-space codes;
- the composite's (`ECS_COMPOSITE_*`) and the session's (`ECS_C_*`);
- the host's: `CLI_INPUT`, `INSTALL_*`, `CONTAINMENT_REFUSED`, `SWAP_POLICY`, `OPERATION_RETIRED`,
  `LEARN_UNAUTHORIZED`, `LEARN_AUTHORITY`, `LEARNING_RATE`, `STIMULUS`, `OBSERVATION_*`, `HOST_RESOURCES`,
  `HOST_OUTPUT` and `ECS_CODEC_UNQUALIFIED`.

**Limits.**
- One operation's JSON, and a script, is at most 8 MiB. One output line is at most 2 MiB.
- Integers in operator JSON are exact up to 2^53.
- QUERY takes at most 4,096 rows. LEARN takes at most 64 experiences of 1..256 rows and 1..2^20 steps. One evidence
  turn takes at most 64 observations.
- Structure's assertion validator admits HACF authority values 0..3 only, so at most four sources can record HACF
  evidence.

## The testing host

`elpis_testing` is the same RAM-only host, never installed. It adds three operations:
- `testing_pause`: pause at an operation boundary until one byte arrives on standard input;
- `testing_exit`: die there;
- `testing_write_probe`: deliberately attempt a file write (create, write, extend, remove, rename or mkdir) in a
  directory the harness names, so that containment can be shown to refuse it.

It also reads a file in the format of `/proc/swaps` in place of the real one, so that the swap policy is exercised
without changing the machine's swap configuration. The production host has none of these: its binary imports no
write-class C library function (`tests/boundary/test_volatile_runtime.py`).

## Tests

`tests/integration/test_native_host.py` runs the host as separate processes with a minimal environment: no
`PYTHONPATH` and no `LD_*` variables. As root, every host process runs as uid 65534.

| Area | What is checked |
|---|---|
| Closure | the binary links only the C library and carries no RUNPATH; a live process maps only the C runtime and the six sealed objects; it holds no writable file descriptor and no writable shared file mapping; core dumps and file growth are disabled; its descriptors are one fixed set that long sessions never grow |
| Pins | measured read-only; drift of a library or of the host binary, or a library naming an unadmitted dependency, refuses every start |
| Differential | the initial identity, QUERY values, the LEARN transition, relation keys, the LOOKUP answer and the component identities equal the Python operator interface's |
| Sessions | learning and typed evidence admitted by the session's ACTIVE sources, in memory; `close` and a new process both start again at the lineage's initial identity |
| Operator events | lifecycles, versions, capacity and reclamation are typed events of the volatile session |
| Retirement and containment | every retired command is refused; a privileged process and writable storage on a standard descriptor are refused; the contained testing host's deliberate write attempts are refused, while the same uncontained user may write there; the swap policy locks the process memory or refuses |
| Death | a killed process leaves nothing behind |

`runtime.native_host_smoke` (ctest, in every native build, including the sanitizer and clang jobs) runs, with no
interpreter:
- the catalog pin;
- a session that founds a source, learns, records typed evidence and answers LOOKUP;
- the retired operations refused, and `close` discarding the session;
- a new process starting at the initial identity;
- the pin, privilege and storage refusals.

It checks that no file is created or changed.

The composed qualification (`research/volatile_runtime_v1`, `ZERO_WRITE_PASS`) traced the host under `strace`,
unconfined and confined, over short, long (3,000-operation) and restart sessions and an adversarial set. It recorded
zero write-class events on storage.

## Holds

- **Release.** No installed release bundle exists for the native host yet (`ELPIS_SYSTEM.json`
  `identity.release_authority = NONE_DEFERRED`).
- **Host prerequisites.** The census, section 6, lists what the host must provide that a process cannot enforce on
  itself:
  - read-only mounts for the code, the libraries and every static resource;
  - swap off, unless locking the runtime's memory is acceptable;
  - standard output and error to a pipe or a terminal;
  - no host service that persists the runtime's memory or I/O.

## Not claimed

- No text interface and no language understanding.
- No persistence, crash survival or resumption: a crash loses the session, by design.
- Nothing about writes by host services outside Elpis's process-control boundary (`docs/NONCLAIMS.md`).
- No scientific claim of any kind.
