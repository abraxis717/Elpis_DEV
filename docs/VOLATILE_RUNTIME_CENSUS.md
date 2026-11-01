# Volatile runtime census (V0–V5)

Authority: `docs/ELPIS_MISSION.md`, Amendment 3 (the volatile-native runtime). The machine-readable form is
`elpis.runtime.persistence`:
- `WRITERS` and the dispositions;
- `CLOSURE`;
- `RETIRED_RUNTIME_CORE_ABI`;
- `VOLATILE_MIGRATION_COMPLETE`.

Gates:
- `tests/boundary/test_volatile_runtime.py`;
- `tests/boundary/test_autonomous_persistence.py`;
- `tests/integration/test_autonomous_no_growth.py`;
- `tests/integration/test_environment_gate.py`.

The composed evidence is `research/volatile_runtime_v1`.

**VOLATILE MIGRATION: COMPLETE.** Every closure condition of the directive holds (section 8).
- The native host and the Python reference write no persistent state: no file, slot, journal, checkpoint, cache or
  metadata. This holds at initialization, inference, learning, operator events, failure, shutdown and restart.
- They read immutable resources, verified, and nothing else.
- The guarantee ends at Elpis's process-control boundary (section 6): nothing is claimed about how much the
  operating system writes.

## 1. Method

The census is source-grounded and checked on the binaries. Every source file that can write persistent storage is
found by a pattern scan and bound to a registered writer, so a new writer fails a gate:

| Language | Scan | Gate |
|---|---|---|
| Rust | `File::create`, write/create/append/truncate options, `sync_data`/`sync_all`, `fs::` mutation, `DirBuilder`, positional writes, `set_len` | `test_volatile_runtime.py`: **no Rust source writes at all** |
| C/C++ | create/write opens, `pwrite`, rename, link, unlink, mkdir, truncation, preallocation, syncs, `mkstemp`/`mkdtemp`/`tmpfile` | `test_autonomous_persistence.py`: every writing source is a classified offline producer, or the testing host's guarded probe |
| Python | write-mode opens, `write_text`/`write_bytes`, `os.` mutation and syncs, `shutil` copies and removals, `tempfile`, `sqlite3.connect`, file log handlers, write `os.O_*` flags, numpy save and memmap, `shelve`/`dbm` | `test_volatile_runtime.py`: every writing module is a registered offline producer or development code |
| binaries | the dynamic imports of the production host and of every library the runtime loads | `test_volatile_runtime.py`: no write-class C library function is imported (rename, unlink, mkdir, truncation, preallocation, syncs, `pwrite*`, links, `creat`, temporary files, chmod/chown/utimes, xattrs, `copy_file_range`, `sendfile`, `splice`, `mknod`) |

Two further checks bind the census:
- **Retired entry points must be absent from the code.** Every entry point a retired or replaced writer had is
  checked absent: the Python attribute does not resolve, and the native symbol is not in its file's code (comments
  may name it as history). A test merely never calling it is not accepted.
- **Composers reach no writer module.** A fresh interpreter importing the Python runtime and every binding it uses
  loads no module that hosts an offline or development writer (`test_no_composer_imports_an_offline_or_development_writer`).

Two writers were split from the code the runtime loads, so that what the runtime links and imports carries no writer:
- `semantic_snapshot_write` / `semantic_snapshot_publish_cas` moved into `snapshot_publish.c`. The ingress bridge
  had linked them, unused, through `snapshot_manifest.c`.
- `atomic_materialize` moved into `elpis.evolution.materialize`. The runtime's evolution gate had imported it,
  unused, through `elpis.evolution.promotion`.

## 2. The ECS_C authority transition boundary

The ECS_C transition law moved unchanged into the `no_std` session crate (`native/ECS_C/session`,
`docs/ECS_C_SESSION.md`). RuntimeCore ABI v8 holds one session per open (backend 3). The durable authority
(`native/ECS_C/authority`) is retired: its crate, C ABI, tests and fixtures are removed, and git history keeps them.

| Durable C ABI (`elpis_ecs_c_*`, retired) | Volatile session (`elpis_ecs_c_session_*`) |
|---|---|
| `provision(state_dir, authority_dir, …)` | **retired**: `create(config)` reserves memory; no directory, key or file |
| `open(state_dir, authority_dir)` (lock, read, verify, recover) | **retired**: a new session is a new genesis |
| `close` | `destroy` (clears the last state image, frees) |
| `transition(event)` → record and witness slots rewritten, two `fdatasync` | `transition(expected, event)` → staged in RAM, published by a buffer swap |
| `legacy_record`, `IMPORT_LEGACY` (10), the reconciliations (11–13) | **retired**: `PERSISTENCE_RETIRED` (RuntimeCore: `RUNTIME_RETIRED`) |
| `elpis_ecs_c_testing_*` (crash injection at storage boundaries) | **retired**: no storage boundary exists |

## 3. Persistent writers: final dispositions

Dispositions:
- **R** `RETIRED`: no persistent path exists; the entry points listed are absent.
- **V** `REPLACED_WITH_VOLATILE`: the state lives in volatile memory, and the persistent entry points are absent.
- **S** `READ_ONLY_STATIC_RESOURCE`: persistent and immutable; opened read-only, never written by Elpis.
- **A** `OFFLINE_ARTIFACT_PRODUCER`: never imported or linked by a composer; every run is a separately
  human-authorized offline act.
- **D** `DEVELOPMENT_ONLY`.

| Writer | Was reachable from | Disp. | What exists now | Phase |
|---|---|---|---|---|
| `ecs_c_authority_slots` | host, Python (RuntimeCore) | V | the volatile ECS_C session, one per open | V3, V4 |
| `ecs_c_authority_provisioning` | host `init`/`create`, Python `create` | R | — (no directory, key or slot) | V3, V4 |
| `k1_checkpoint_slots` | host, Python | R | — (no store, attach, recover, adopt or discard); a K1 state is never persisted | V3 |
| `k1_checkpoint_provisioning` | host, Python `create` | R | — | V3 |
| `runtime_durability_lane` | host, Python | R | — (no recoverable or identity publication, background lane, wait, status or durable format change) | V3 |
| `native_host_deployment` | host `init`/`create` | V | a RAM-only installation: pinned libraries sealed read-only into memfds; `init`, `create`, `publish`, `reconcile`, `candidates`, `--home` refused `OPERATION_RETIRED` | V3 |
| `python_reference_deployment` | Python CLI | V | `Installation` and `Session` in memory over a read-only catalog | V3 |
| `runtime_canonical_publication` | `Runtime.publish_canonical` | R | — (the publisher is an offline producer, below) | V3 |
| `evolution_reservation` | `Runtime.evolve` | V | session transitions (reserve, finalize, abandon), gone at close | V3 |
| `operator_authority_events` | host and Python operator events | V | typed session events; the persistence-only kinds refused | V3 |
| `fms_posix_cold_store` | Python FMS contexts | R | FMS ABI v3: no COLD residency, no storage domain, no writeback; the writable constructors are not exported; `Context(cold_root=…)` is a `TypeError` | V4 |
| `hgram_fixed_capacity_store` | operator executable | R | — (`elpis_hgram_init`, `elpis_hgram_init_once`, `elpis_hgram_assoc_store_preapproved` removed) | V4 |
| `persistent_hacf_corpus` | — | R | SQLite `:memory:` only (`temp_store`, `journal_mode` MEMORY); persistent ingress refuses | — |
| `persistent_log_sink` | — | R | none: no log handler; the host opens no socket | — |
| `pinned_native_libraries` | host, Python | S | admitted libraries pinned by SHA-256, loaded from sealed read-only copies | — |
| `fms_cold_backing` | FMS file service | S | immutable assets through an `O_RDONLY` descriptor, page digests verified, into WARM; writable descriptors refused | V4 |
| `hgram_retained_store` | — | S | probe and association reads, `O_RDONLY` | V4 |
| `canonical_grid81_publication`, `canonical_candidate_construction`, `publication_ledger_v1`, `application_ledger_v2`, `evolution_child_materialization`, `semantic_*` (7 families) | offline only | A | separately human-authorized offline acts; no composer imports or links them | — |
| `dsv4_synthetic_fixture_generator`, `native_host_testing_write_probe`, `research_laboratories`, `acceptance_harness`, `developer_release_bundle` | — | D | tests, the testing host (never installed), laboratories, qualification, packaging | — |

The V0 census table (baseline `0d96964`), with each writer's resource, trigger and migration plan, is in this file's
git history at `a3b4028`.

## 4. RuntimeCore ABI v8 and the composers

RuntimeCore ABI v8 declares and exports none of the twelve persistent functions of ABI v7 (`RETIRED_RUNTIME_CORE_ABI`;
checked on the header, the ABI inventory and the binary). `elpis_runtime_create` takes a session configuration
(capacity, nonce, labels): no path. The anchor, evolution transitions and operator events are session transitions.
The Python `Runtime` has no persistent operation and no persistent configuration (no `authority_dir`). Each `open` is
a new session; `close` discards it.

## 5. Indirect and hidden channels

| Channel | Standing |
|---|---|
| Python bytecode caches | **Enforced.** `Runtime.open` refuses unless `sys.dont_write_bytecode`. The offline CLI also requires the interpreter's startup flag (`-B` or `PYTHONDONTWRITEBYTECODE=1`), because imports before that point would already have written. |
| Package caches | none at runtime; pip and cargo caches are development operations |
| Native library loading | libraries are admitted through sealed memfds (`host_admit.c`, `elpis.substrate.boundary`): anonymous memory, never a file; the host keeps only read-only descriptors of its seals |
| Temporary directories | no runtime code uses `tempfile`, `mkstemp` or `/tmp`; the only native temporary-file users are offline producers, which no runtime binary links (import gate) |
| Logging | no handler is installed and the host opens no socket (seccomp); the host and the CLI refuse to write anything onto storage standard output or error, not even a refusal |
| Writable mappings | none; the host maps no file writable (live-process test), and its seals are read-only |
| SQLite | the HACF corpus is `:memory:` with `temp_store=MEMORY` and `journal_mode=MEMORY` |
| Compilation and autotuning caches | none: no torch, triton, onnxruntime, tensorflow, jax or transformers import in `src/` |
| libffi trampolines | ctypes callbacks may need executable memory; an old libffi that falls back to a temporary file fails closed under a read-only filesystem or Landlock |
| Swap | **Enforced.** The host reads `/proc/swaps`. With swap active, it locks all memory (`mlockall`, current and future) or refuses (`SWAP_POLICY`). The Python gate does the same where the lock limit can hold the process, and refuses otherwise. The launcher refuses with swap active. Neither changes the host's swap configuration. |
| Crash dumps | **Enforced.** The host and the Python gate set `RLIMIT_CORE` 0 and `PR_SET_DUMPABLE` 0, both read back; a non-dumpable process is never dumped, whatever the core pattern. The launcher sets `RLIMIT_CORE` 0. |
| Background maintenance | none: the durability lane is retired; the contextual lane's worker thread writes nothing |
| Child processes | the host cannot create one (seccomp: fork, vfork, exec, clone without `CLONE_THREAD`); the Python runtime spawns none |
| Inherited descriptors | the host and the launcher refuse writable storage on a standard descriptor and close every descriptor above them |

## 6. What Elpis enforces and what the host must provide

**Enforced by Elpis, fail-closed:**
- **The native host** (`docs/NATIVE_HOST.md`, "Containment"). It refuses root and storage standard descriptors and
  closes inherited descriptors. It sets RLIMIT_CORE 0 and dumpable 0, and applies the swap policy. It admits its
  libraries read-only. Then:
  - Landlock ABI ≥ 5 with every write right handled and none granted;
  - RLIMIT_FSIZE 0 once its seals exist;
  - `no_new_privs`;
  - a seccomp filter (metadata, preallocation, sync, mount, swap control, sockets, io_uring, ptrace, process
    creation).

  If anything cannot be established and read back, it refuses to start.
- **The Python reference.** The process gate covers bytecode, core dumps and the swap policy. The CLI also refuses
  storage stdio.
- **`elpis_autonomous_run`.** For any command it applies:
  - Landlock with no grant;
  - seccomp;
  - `no_new_privs`;
  - RLIMIT_CORE 0;
  - the storage-stdio refusal;
  - `close_range`;
  - refusal with swap active, since memory locks do not survive exec.

  It sets no RLIMIT_FSIZE: the contained runtime must seal its admitted libraries into memfds. The limit would
  protect nothing persistent there, because no path is writable.

**The host must provide (outside Elpis's process-control boundary):**
1. Read-only mounts for the code, the libraries and every static resource. No writable persistent mount should be
   visible to the runtime.
2. Swap off, unless the runtime's memory lock is acceptable (the host checks; the decision is the operator's).
3. stdout and stderr to a pipe or terminal; the runtime refuses storage.
4. No host service that persists the runtime's memory or I/O (crash collectors for other processes, journals of its
   output, backups of its tmpfs).

Process-level controls do not stop unrelated host services from writing to storage: the journal, crash collectors for
other processes, backups, the package manager, filesystem metadata and maintenance. **Nothing here claims they do.**
Elpis claims that its runtime issues no persistent-storage mutation, and that the sandbox denies such writes.

## 7. Phase record

| Phase | What | Commits |
|---|---|---|
| V0, V1, V2 | this census and its gate; the volatile ECS_C session (`ECSC_VOLATILE_SESSION_R0`); the relational view rebound to it; `research/volatile_runtime_v0` (`ZERO_WRITE_PASS` at `e61d5dd`) | `6161ae8`, `e61d5dd`, `0046d0a` (PR #67, merged at `a3b4028`) |
| V3 | RuntimeCore ABI v8; the RAM-only, contained native host and launcher; the Python runtime over the session | `ac57aea`, `96c735e`, `d41be29`, `d4f8361` |
| V4 | FMS ABI v3 (no COLD writeback); H-gram writers retired; the durable ECS_C authority removed; retired subjects' instruments fail closed | `fa28358`, `253cb1f`, `3c9734c`, `df9ac24` |
| V5 | census closed; Python process gate; writer splits; FMS: a held object never moves; composed zero-write qualification (`research/volatile_runtime_v1`, `ZERO_WRITE_PASS` at `66a11a4`) | `5c0de38`, `a35d34c`, `0d6d16f`, `66a11a4` and the closing commit |

## 8. Closure (directive, section 10)

| # | Condition | Evidence |
|---|---|---|
| 1 | RuntimeCore no longer depends on durable ECS_C authority | the Cargo dependency is the session only; `test_ecs_split.py`, `test_ecs_c_authority.py` |
| 2 | The native host starts without a writable deployment home | `test_native_host.py` (sessions in memory, write probes refused); the composed qualification |
| 3 | The Python runtime has no reachable persistent-state writer | the Python scan, the import-closure gate, `test_autonomous_no_growth.py` |
| 4 | The checkpoint and durability lanes are retired | `test_the_runtime_core_abi_has_no_persistent_function` |
| 5 | FMS COLD is immutable, read-only backing | `substrate.test_fms_volatile_cold`, `_rofs` |
| 6 | WARM and HOT are volatile | `substrate.test_fms`, `substrate.test_fms_volatile_cold` |
| 7 | Persistent FMS writeback is retired | case 10 of `test_fms_volatile_cold`; the `cold_root` parameter refused |
| 8 | Runtime H-gram mutation is retired | `structure.test_hgram_assoc` (zero mutations under a write trap) |
| 9 | Runtime canonical publication is retired | `test_the_runtime_exposes_no_canonical_publication` |
| 10 | All remaining runtime-connected storage writers eliminated or read-only | the final dispositions; the binary import gate |
| 11 | The composed runtime passes independent zero-write verification | `research/volatile_runtime_v1` (`ZERO_WRITE_PASS` at `66a11a4`, the native and `src/` code that ships) |
| 12 | Environmental prerequisites are enforced or fail closed | the host's containment tests; `test_environment_gate.py`; the adversarial runs |
| 13 | Bounded-memory and atomic-refusal invariants remain intact | capacity and refusal tests; the qualification's boundedness and allocation-failure runs |
| 14 | Architectural documents and ABI inventories match the implementation | `test_abi_inventory.py`; `test_the_documents_state_the_migration_as_it_is` |

The historical qualifications remain the record of their own commits and qualify nothing under Amendment 3:
- ECS_C Recovery Q0;
- K1 Recovery R0;
- the persistent host's containment and acceptance.

Their instruments fail closed against the volatile runtime.
