# Volatile runtime census (V0)

Authority: `docs/ELPIS_MISSION.md`, Amendment 3 (the volatile-native runtime). Baseline: `0d96964` (PR #66 merged).
Machine-readable form: `elpis.runtime.persistence` (`WRITERS`, `VOLATILE_CENSUS`, `RUNTIME_CORE_PERSISTENT_ABI`,
`NONCONFORMING_OPERATIONS`, `VOLATILE_MIGRATION_COMPLETE`). Gates:
- `tests/boundary/test_volatile_runtime.py`;
- `tests/boundary/test_autonomous_persistence.py`.

**VOLATILE MIGRATION: INCOMPLETE.** Elpis is not diskless:
- Every runtime composer still reaches durable publication functions: RuntimeCore ABI v7, the native host `elpis`
  and the Python reference interface `elpis.runtime`.
- This census lists every one of them.
- Only phases V0–V2 are implemented: this census, the volatile ECS_C session and the relational view's binding to
  it. Sections 7 and 8 give the status.

## 1. Method

The census is source-grounded. Every source file that can write persistent storage is found by a pattern scan and
bound to a registered writer, so a new writer fails a gate:

| Language | Scan | Gate |
|---|---|---|
| Rust | `File::create`, write/create/append/truncate options, `sync_data`/`sync_all`, `fs::` mutation, `DirBuilder`, positional writes, `set_len` | `test_volatile_runtime.py` |
| C/C++ | create/write opens, `pwrite`, rename, unlink, mkdir, truncation, syncs | `test_autonomous_persistence.py` |
| Python | write-mode opens, `write_text`/`write_bytes`, `os.` mutation and syncs, `shutil` copies and removals, `tempfile`, `sqlite3.connect`, file log handlers, write `os.O_*` flags, numpy save and memmap, `shelve`/`dbm` | `test_volatile_runtime.py` |

Indirect channels (section 5) were audited by reading the code and the environment.

Every public durable entry point was enumerated from the declared ABIs and interfaces:
- RuntimeCore's C ABI (`native/runtime/include/elpis/runtime.h`);
- the durable authority's C ABI (`ecs_c_authority.h`);
- the public `Runtime` operations;
- the commands of the native host and of the Python CLI.

The scan found three Python writers that the persistence registry did not list. They are now registered:
- `python_reference_deployment` (`elpis.runtime.deployment`);
- `acceptance_harness` (`elpis.runtime.acceptance`);
- `developer_release_bundle` (`elpis.runtime.release`).

## 2. The ECS_C authority transition boundary

The durable ECS_C authority (`native/ECS_C/authority`, `docs/ECS_C_AUTHORITY.md`) splits into a pure law and its
storage:
- **The law.** `model.rs` holds the transition law, entities, events, receipts and state encoding. It is used with
  `register.rs` (evolution authority), `sha256.rs` and `error.rs`. It performs no I/O.
- **Storage.**
  - `store.rs`: provisioning, the two record slots and two witness slots, `fdatasync`, locks, recovery and
    reconciliation.
  - `record.rs`: the ELPECA01 record and ELPECW01 witness formats.
  - `hmac.rs`: the deployment key's witness MAC.
  - `legacy.rs`: the continuity v2/v3 register reader.
  - `ffi.rs`: a C ABI built on paths and directories.

**The boundary is the law.** In this change:
- The four law modules moved unchanged (git renames) into the new `no_std` crate `native/ECS_C/session`.
- The new crate wraps them in the volatile session (`docs/ECS_C_SESSION.md`).
- The durable crate became a consumer that re-exports them, so one law runs in both. Its tests (51, plus one new
  test binding the law's slot-class values to its record format), RuntimeCore's (97) and the boundary suite are
  unchanged in outcome.
- Every operation that existed only for storage is removed from the session's architecture: refused as retired, or
  simply absent.

The session's state model and transition order are in `docs/ECS_C_SESSION.md`.

| Durable C ABI (`elpis_ecs_c_*`) | Volatile session (`elpis_ecs_c_session_*`) |
|---|---|
| `provision(state_dir, authority_dir, …)` (directories, deployment key, record and witness slots) | **retired**: `create(config)` reserves memory; no directory, key or file |
| `open(state_dir, authority_dir)` (lock, read, verify, recover the newest record) | **retired**: there is nothing to open or recover; a new session is a new genesis |
| `close` | `destroy` (clears the last state image, frees) |
| `transition(event)` → record slot + witness slot rewritten, two `fdatasync` | `transition(expected, event)` → staged in RAM, published by a buffer swap; `SESSION_MISMATCH` / `STALE` on a foreign or old coordinate |
| `legacy_record`, event `IMPORT_LEGACY` (10) | **retired**: `PERSISTENCE_RETIRED` |
| events `RECONCILE_ROLLBACK` / `RECONCILE_UNAUTHENTICATED` / `RECONCILE_REPIN` (11–13), `read_reconciliation`, `read_report` | **retired**: `PERSISTENCE_RETIRED` / absent |
| `record_bytes(capacity)` | `bytes(capacity)` (exact volatile footprint) |
| `read_status`, `entity_at`, `entity_find`, `source_for_authority`, `next_entity_id`, `check_evidence`, `receipt_chain`, `evolution_digest`, `code_name`, `abi_version` | same names under `elpis_ecs_c_session_`, on session memory (`read_status` also reports the session, instance, commit and footprint) |
| record digest (predecessor of a receipt) | commit digest, computed in memory |
| `elpis_ecs_c_testing_arm`, `_testing_counters` (crash injection at storage boundaries) | **retired**: no storage boundary exists |

**The state transition crossing the boundary.** A caller replaces its durable store with:
1. `elpis_ecs_c_session_create` (once per process lifetime of the state);
2. `elpis_ecs_c_session_transition` against the coordinate it holds.

The ABI has no restart path. RuntimeCore ABI v7 still crosses the old boundary, through `elpis_runtime_create_ecs_c`
over directories and `runtime/src/durable.rs`. ABI v8 must cross the new one (phase V3).

## 3. Persistent writers

Disposition key:
- **N** = `NONCONFORMING_RUNTIME`: writes operational state; to be retired.
- **A** = `OFFLINE_ARTIFACT_PRODUCER`: never from a runtime; every run separately authorized (V7).
- **D** = `DEVELOPMENT_ONLY`.
- **X** = `RETIRED` (already refuses or absent).

Other columns:
- Direct means the code issues the write itself. Indirect means it reaches the write through another writer.
- "Host" is the native host `elpis`; "Py" is the Python `Runtime` and CLI; "op" is an operator command.

| Writer | Subsystem | Exact entry points | Resource written | Trigger | Reached from | Disp. | Migration action | Phase |
|---|---|---|---|---|---|---|---|---|
| `ecs_c_authority_slots` | ECS_C | `authority/src/store.rs` `transition` (`write_at`, `sync_data`); `runtime/src/durable.rs` `Stores` | 2 record slots (R = 1344 + 160·capacity B) and 2 × 240 B witness slots, rewritten in place; 2 `fdatasync` per transition | anchor, durable publication, operator events, evolution reserve/finalize, reconciliation | Host, Py (direct, via RuntimeCore) | N | replaced by `native/ECS_C/session` (built, not composed); RuntimeCore ABI v8 holds a session; durable crate archival | V3 |
| `ecs_c_authority_provisioning` | ECS_C | `store.rs` `provision`; `elpis.runtime.authority:provision_authority` | state and authority directories, deployment key, record and witness slots | operator provisioning | Host `init`/`create`, Py `create` (direct) | N | retire with the durable authority | V3 |
| `k1_checkpoint_slots` | RuntimeCore | `runtime/src/checkpoint.rs` `write`, `retract` | 2 fixed slot files (the K1/composite envelope), rewritten in place, `fdatasync` | anchor, recoverable publication, discard/adopt | Host, Py (direct) | N | retire: no checkpoint store, attach, recover, adopt or discard in ABI v8 | V3 |
| `k1_checkpoint_provisioning` | RuntimeCore | `checkpoint.rs` `provision`; `elpis.runtime.recovery:provision_k1_checkpoint` | 1 directory + 2 zeroed slot files | operator | Host/Py `create` (direct) | N | retire with the checkpoint store | V3 |
| `native_host_deployment` | host | `host_deploy.c` `host_init`, `host_create_lineage` | home: `host.cfg`, `catalog`, 7 pinned libraries (write-once); `lineage.cfg`; provisions the two writers above | operator `init`, `create` | Host (direct + indirect) | N | RAM-only initialization: no writable home, authority, checkpoint or lineage; pinned libraries read from a read-only location | V3 |
| `python_reference_deployment` | runtime (Py) | `elpis.runtime.deployment:init_deployment`, `Deployment.create_lineage` | home: `elpis.json`, `native-catalog.json`, `lib/` (write-once, `fsync`), `lineage/lineage.json`; provisions the authority and checkpoint store | operator `init`, `create` | Py CLI (direct + indirect) | N | as the host | V3 |
| `fms_posix_cold_store` | substrate (FMS) | `fms_pal_posix.c` `p_cold_put`; `elpis.substrate.residency:Context` | one file per COLD object under a cold root (bounded by budget and capacity) | COLD demotion | Py (a caller-built FMS context) | N | RAM-only PAL for every composer (`fms_pal_posix_create_ram_only`, as the retrieval epoch already does) | V4 |
| `hgram_fixed_capacity_store` | HACF | `hgram_init.c` `main`; `hgram_store.c` `elpis_hgram_init_once`, `elpis_hgram_assoc_store_preapproved` | 1 preallocated H-gram file; in-place association writes | operator executable `elpis_hgram_init`; preapproved calls | neither composer (no built library carries it) | N | prohibited outright (persistent H-gram learning): retire the association writer; a retained H-gram is a read-only static resource | V4 |
| `canonical_grid81_publication` | pipeline | `elpis.pipeline.canonical.publisher:publish_candidate` | a generation file, a ledger row, an atomic root exchange, a recovery journal | operator capability | Py `Runtime.publish_canonical` (direct) | N | remove `Runtime.publish_canonical`; the publisher may survive only as a separately authorized offline tool | V4 |
| `publication_ledger_v1`, `application_ledger_v2` | pipeline | `elpis.pipeline.application.durable_ledger(_v2)` `append`, `apply_artifact` | ledger rows (append-only journals) | operator | offline pipeline | A | never from a runtime | V4 |
| `canonical_candidate_construction` | pipeline | `elpis.pipeline.canonical.candidate:construct_candidate` | a candidate tree | operator | offline (and via `publish_canonical`) | A | leaves the runtime with `publish_canonical` | V4 |
| `evolution_child_materialization` | evolution | `elpis.evolution.promotion:atomic_materialize`, `EvolutionPolicyAuthority.materialize` | a child workspace | operator grant | offline | A | development artifact; `Runtime.evolve`'s durable reservation retires with the authority | V4 |
| `semantic_*` (7 families) | structure/semantic | the writers listed in `elpis.runtime.persistence` | segments, manifests, CAS head, context/embedding/evidence/view/topology/Grid81 files | explicit call | neither composer (gated: no bridge calls them) | A | offline artifact production only | V4 |
| `dsv4_synthetic_fixture_generator` | inference | `elpis.inference.drivers.dsv4.fixtures:make_fixture` | fixtures | tests | tests | D | — | — |
| `native_host_testing_write_probe` | host (testing) | `elpis_host.c` `op_testing_write_probe` | one probe file / byte | testing host | tests | D | rewritten with V3 | V3 |
| `acceptance_harness` | runtime (Py) | `elpis.runtime.acceptance:Acceptance` | deployments, copies, a JSON report in a work directory | developer | qualification | D | historical once V3 lands; replaced by a session acceptance | V3 |
| `developer_release_bundle` | runtime (Py) | `elpis.runtime.release:build_bundle` | one bundle directory | developer | packaging | D | — | — |
| `research_laboratories` | research | `research/` | write-once evidence | developer | research | D | — | — |
| `persistent_hacf_corpus` | HACF | `ingress_bridge.c` `elpis_ingress_env_open` | none: SQLite `:memory:` with `temp_store=MEMORY` and `journal_mode=MEMORY`; persistent ingress refuses | — | — | X | — | — |
| `persistent_log_sink` | — | — | none: importing the runtime installs no log handler | — | — | X | — | — |

**Impact of retiring the N rows.** These features end:
- crash recovery and restart from the last acknowledged state;
- operator reconciliation of candidates;
- deployment homes holding state;
- the durable evidence summaries;
- the K1 Recovery R0 and ECS_C Recovery Q0 protocols;
- canonical publication from a runtime;
- persistent H-gram associations.

What survives in RAM:
- every law: ECS_C's transition law in the session, K1 and the memory;
- admission, provenance and the relational view;
- in-session evolution coordinates.

The historical qualifications stay the record of their commits and are not promoted.

## 4. Reachable durable paths

**RuntimeCore C ABI v7** (`elpis.runtime.persistence.RUNTIME_CORE_PERSISTENT_ABI`):
- `elpis_runtime_create_ecs_c`: opens the authority's directories read-write and locks them.
- `elpis_runtime_anchor`.
- `elpis_runtime_import_composite`.
- `elpis_runtime_recoverable_request` and `elpis_runtime_identity_request`: the durability lane is a background
  worker thread that writes the checkpoint slot and the authority record.
- `elpis_runtime_durable_wait` and `elpis_runtime_durable_status`.
- `elpis_runtime_checkpoint_provision`, `_attach`, `_recover`, `_discard` and `_adopt`.
- `elpis_runtime_evolution_reserve`, `_finalize`, `_abandon` and `_reconcile`.
- `elpis_runtime_authority_transition`.

QUERY, LEARN, OBSERVE and the turn functions write nothing themselves. They run on a handle that exists only over the
durable authority.

**Python `Runtime`** (`NONCONFORMING_OPERATIONS`). `RuntimeConfig.authority_dir` is required, so every `Runtime` is
built over the durable authority. The operations:
- `open`: attaches a configured checkpoint store;
- `anchor_cognition`, `publish_recoverable`, `publish_identity`, `durability` and `recover_k1`;
- `discard_k1_candidate`, `adopt_k1_candidate` and `import_composite`;
- `evolve`: a durable reservation;
- `publish_canonical`.

**Native host `elpis` and Python CLI commands:**
- `init` and `create` (deployment and provisioning);
- `publish`, and `learn --publish` / `evidence --publish`;
- `reconcile`;
- the session-script authority operations `authority_reconcile`, `source_found`, `source_lifecycle`,
  `source_version` and `source_reclaim`. These are durable authority transitions.

Every restart resumes the last acknowledged state from storage.

## 5. Indirect and hidden channels (V6)

| Channel | Finding | Standing |
|---|---|---|
| Python bytecode caches | the interpreter writes `__pycache__/*.pyc` beside imported modules when it can | host: run the Python interface with `PYTHONDONTWRITEBYTECODE=1` (or `-B`) over a read-only tree. Development runs in this repository do write them (gitignored) |
| Package caches | none at runtime; pip and cargo caches are development operations | — |
| Native library caches | none written; libraries are admitted through sealed memfds (`host_admit.c`, `elpis.substrate.boundary`) | memfd is anonymous shmem: volatile, swappable unless swap is off |
| Temporary directories | No runtime code uses `tempfile` or `/tmp` (Python scan above). Native `mkstemp` appears only inside registered writers, as same-directory temporaries of their own files: the five semantic writer families (A) and the FMS cold store (N). On this host `/tmp` is persistent ext4, not tmpfs | host: never rely on `/tmp` being volatile; a read-only root makes any attempt fail |
| Logging frameworks | none; importing the runtime installs no handler (gated); diagnostics go to stdout/stderr | stdout/stderr are the caller's. If redirected to a file or a journal, the host writes them (section 6) |
| Writable mappings | none outside tests (no `MAP_SHARED` + `PROT_WRITE` and no `mmap.ACCESS_WRITE` in sources) | — |
| SQLite | the HACF corpus is `:memory:` with `temp_store=MEMORY` and `journal_mode=MEMORY` | conforms |
| Framework autotuning / compilation caches | none: no torch, triton, onnxruntime, tensorflow, jax or transformers import in `src/` | — |
| libffi trampolines | ctypes callbacks may need executable memory. Older libffi versions fall back to a temporary file (in `$TMPDIR`, `/tmp` or `$HOME`) where W+X anonymous memory is refused | host: a read-only filesystem makes this fail closed |
| Swap / file-backed paging | session, heap, KV and memfd pages can reach swap; this host has `SwapTotal: 0` | host: swap off (or the process locked in memory); not enforced by any Elpis component |
| Crash dumps | neither `elpis_autonomous_run` nor the host's containment sets `RLIMIT_CORE` or `PR_SET_DUMPABLE`; this host has `core_pattern = core` and `ulimit -c 0` | gap: set `RLIMIT_CORE=0` and `PR_SET_DUMPABLE=0` in the host (V3). A piped `core_pattern` writes from a host service, outside any process control |
| Background maintenance | RuntimeCore's durability lane is a background thread that writes durable state | N (section 4), retires with ABI v8 |
| Child processes | The host and RuntimeCore spawn none. `elpis_autonomous_run` executes exactly its command, by design. The acceptance harness spawns the CLI (development) | — |
| Inherited descriptors | `elpis_autonomous_run` refuses writable stdio files and closes descriptors ≥ 3 (`close_range`) | conforms where the launcher is used |

## 6. What the sandbox enforces and what the host must provide

**Enforced by Elpis components, where they are used:**
- The session library has no path to storage. It is `no_std`; its imports are memory, abort and getrandom (checked
  on the binary); and no function of its ABI takes a path or a descriptor.
- `elpis_autonomous_run` (no `--authority` grant) applies to the process tree:
  - Landlock ABI ≥ 5: write, create, remove, rename, truncate and device ioctl are denied everywhere;
  - seccomp: the metadata syscalls fail with EPERM;
  - `no_new_privs`;
  - descriptors ≥ 3 are closed;
  - writable stdio files are refused;
  - root is refused.
- The zero-write evidence (`research/volatile_runtime_v0`) runs the session's test paths under that launcher. The
  confined runs also use a private mount namespace in which every mount is read-only, as uid 65534. strace records
  every write-class system call.

**The host must configure (not enforceable from inside the process):**
1. Read-only mounts for the code, the libraries and every static resource. A zero-write runtime should have no
   writable persistent mount at all.
2. Swap disabled, or the runtime locked into memory.
3. Core dumps disabled (`RLIMIT_CORE=0`, `PR_SET_DUMPABLE=0`, and a `core_pattern` that is not a host collector).
4. stdout and stderr to a pipe or terminal, not a persistent file or journal.
5. No `TMPDIR` or `HOME` pointing at persistent storage.
6. `PYTHONDONTWRITEBYTECODE=1` for the Python interface.

Process-level controls do not stop unrelated host services from writing to storage: the journal, a crash collector,
backups, the package manager, filesystem maintenance. Nothing here claims they do. Where the host cannot provide
items 1–3, a conforming runtime must refuse to start; that check is part of V3 (the host's RAM-only initialization).

## 7. Retired or replaced in this change (V0–V2)

- **Added:** the volatile session authority `ECSC_VOLATILE_SESSION_R0`:
  - `native/ECS_C/session`, `libelpis_ecs_c_session.so`;
  - C ABI v1, `ecs_c_session.h`, and the shared `ecs_c_types.h`;
  - tests: cargo (12), the C ABI test and the relational session end-to-end test;
  - the gate `test_volatile_runtime.py`.
- **Moved, not copied:** the ECS_C transition law (`model.rs`, `register.rs`, `sha256.rs`, `error.rs`). The durable
  crate consumes it from the session crate. Durable-only definitions remain in `ecs_c_authority.h`: storage codes,
  witness and legacy sizes.
- **Retired in the session's architecture:** each operation of section 2 marked retired. The four persistence-only
  event kinds are refused `PERSISTENCE_RETIRED`.
- **Rebound (V2):** the relational view's end-to-end test (`test_ecs_c_relational_session`) runs over the session.
  The view's header includes `ecs_c_types.h`, not the durable header. A session's identity change (a new session) is
  `LINEAGE` until the view is reset, so no volatile view outlives its session.
- **Registered:** three unlisted Python writers (section 1).
- **Not retired yet:** every N row of section 3 and every path of section 4.

## 8. Next step (V3)

RuntimeCore C ABI v8, over the volatile session:
- `elpis_runtime_create` takes a session configuration, not directories;
- remove `elpis_runtime_anchor`'s publication, the durability lane, `recoverable_request`/`identity_request`, the
  checkpoint functions, `import_composite`'s publication and the evolution reservation's durable writes;
- operator events become session transitions;
- the native host's RAM-only initialization: no writable home, and refusal to start without the section 6
  prerequisites;
- the Python reference interface follows;
- the durable crate and the checkpoint store become archival.

The historical ABI v7 specification is preserved.
