# Native ECS_G / ECS_C split: source and authority census (Stage A)

Status: **census and plan**. It moves no code and changes no law, format, evidence or qualification. It records what
exists at the baseline, who owns it, what binds it, and the dependency-ordered transition to the target architecture
of `proposal/native-ecsc-ecsg-runtime-charter-r0` (`DEV/proposals/NATIVE_ECSC_ECSG_COMPLETION_R0.md`, commit
`204fdf0`). Each later stage is its own pull request and states which rows of this census it closes.

## 1. Baseline

| | |
|---|---|
| `main` head | `9dc2c495bf1a4fbedc6fd84058d329c4f7bf00e3` (2026-10-18, merge of PR #53), tree `f09817f150bf53c03cdbd08c5c9568860ce2740c` |
| `native/ECS` | tree `ebaf2376aa0fb3a357f8ed962e31989f3aab740c` |
| `native/continuity` | tree `e29291d7097057399d6d3524ef00b2820c7d1416` |
| `native/runtime` | tree `88fff084aec76efdcdcb63012b3341f53f81fb41` |
| `native/substrate`, `native/structure`, `native/pipeline` | `a67a164b…`, `569c5b1a…`, `2c87d300…` |
| `src/elpis/ECS`, `src/elpis/continuity`, `src/elpis/runtime` | `0e2a30f1…`, `3153cf47…`, `1c87011e…` |
| Post-merge CI on `main` | green, 15/15 jobs (run 39128976436) |
| Native sources | about 101,000 lines (C, C++, Rust, CMake); Python `src/` about 34,600 lines |

### Qualification status at the baseline (exact, in-repository)

| Specification | Status | Scope |
|---|---|---|
| K1 Recovery R0 under ABI v4 | **RUN PASS · QUALIFIED** | commit `18f2f4d` only; the durability protocol of the testing RuntimeCore build under `SIGKILL`, torn writes and lost-sync faults; never power loss |
| Cognitive Boundary R2 under ABI v5 | NOT RUN · UNQUALIFIED | its instrument is not built |
| Cognitive Boundary R1, R2 (ABI v3, v4) | NOT RUN · UNQUALIFIED | R1 has an out-of-repository closure for `bec6da7` (ABI v3) only |
| K1 Recovery R0 (ABI v3) | NOT RUN · UNQUALIFIED | pre-registration kept as written |
| ECS Codec R0 | NOT RUN · UNQUALIFIED | `ECS_CODEC_UNQUALIFIED` refuses every text path |
| HACF-ECS Bridge R0, Evolution Fitness R0 | NOT RUN · UNQUALIFIED | research infrastructure |
| ECS memory | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE **ADMITTED (mechanical)**; LEARN / QUERY FAST **UNQUALIFIED** (MF-R0 H4 NONPASS); CONSOLIDATE / SLOW **NOT PROVIDED** | `docs/ECS_MEMORY.md` |
| Native K1 law `(W, epoch, H, a)` | K1N-v2 attestation (research, frozen `d=6` regime); Runtime R1 executor evidence | `research/ecs_k1_native`, `research/ecs_runtime_r1` |
| ECS_C HCA R0–R19 | research evidence only; R19 `DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL` | `DEV/ecsc_hca`, `research/ecsc_hca_r0` |

None of these statuses extends to a relocated or re-owned subject. A relocation that preserves bytes keeps the bound
evidence valid for exactly those bytes; it is not a new qualification.

## 2. Target and the constitutional conflict it raises

The charter's target: **ECS_G** owns geometric cognition (native K1 and its complete state, exact forward and S3,
query and deterministic learning, qualified retention and consolidation, transactions, geometric identities,
snapshots, FMS materialization); **ECS_C** owns continuity and memory (lineage, entities, typed evidence and
provenance, durable state and history, semantic replay and recovery, bounded relational memory, gated cellular
memory); **RuntimeCore** is the only native composer; and the production runtime runs **no Python at all**.

The baseline's constitution says otherwise, explicitly:

| Baseline authority | What it says | Conflict |
|---|---|---|
| `docs/ELPIS_MISSION.md` ("There is one ECS"; roles table) | one ECS; continuity "is not an ECS and not a history … no entities, mailboxes, schedulers, projections, topology, event log, compaction or retention" | ECS_C owns entities, events, receipts, history and replay |
| `docs/CONTINUITY.md` (consumer census) | the event-history kernel, entity registry, mailboxes, replay, state roots, compaction were **deleted** for want of a consumer | ECS_C re-establishes entity, provenance and transition authority with semantic validation, as fixed-capacity materialized state, never as a log (section 5a) |
| `tests/boundary/test_one_ecs.py` | `elpis.ECS_G`, `elpis.ECS_C`, `native/ECS_G`, `native/ECS_C` must not exist; the tokens are retired | the target names them |
| `tests/boundary/test_mission.py`, `tests/boundary/test_system_authority.py` | subsystem `ECS`; `python_package == elpis.<id>` for every subsystem | subsystems `ECS_G`, `ECS_C`; native-only subsystems |
| `docs/ARCHITECTURE.md`, `ELPIS_SYSTEM.json` | the Python `Runtime` is the composition; `python -m elpis.runtime.cli` is the operator entry point | the production entry point must be native |

The mission document declares itself authoritative over code and asks that a conflicting change be rejected. The
repository owner's instruction for this work is an explicit amendment of it, so the transition does not weaken the
gates; it replaces them: Stage B carries a **versioned mission amendment** (the split, stated with its reasons and
its new prohibitions) and successor gates at least as strong as those they replace. What the amendment keeps
verbatim: DSV4 only communicates; the prohibited sidecar topology; fail-closed text (`ECS_CODEC_UNQUALIFIED`); no
per-turn durable work on the HOT lane; no autonomously growing persistence. What it changes: the ECS has two native
owners, and the continuity/memory owner keeps entity, provenance and transition authority as a fixed-capacity
materialized state that is semantically validated before any mutation (section 5a). It never keeps a growing log or
history, and it never writes on the HOT lane.

## 3. Ownership and dependency matrix

Current owner → target owner. "Byte move" means the file moves without a byte changing (evidence and pins stay
valid by digest); "successor" means a new, versioned contract replaces it after its own tests.

| Component (baseline path) | Current owner | Target owner | Target path | Transition |
|---|---|---|---|---|
| cubic math, state, Runtime R1 executor (`ecsg_math.c`, `ecsg_state.c`, `ecsg_executor.c`, headers, tests) | ECS | **ECS_G** | `native/ECS_G` | byte move (Runtime R1 evidence records `native/ECS_G/...`: the move restores the recorded paths) |
| mutable FMS residency R0 (`ecsg_fms.c/.h`, tests) | ECS | **ECS_G** | `native/ECS_G` | byte move |
| native K1 and K1 FMS residency (`ecsg_k1*.c/.h`, `ecsg_k1_internal.h`, tests, `test_k1_legacy_differential.c`) | ECS | **ECS_G** | `native/ECS_G` | byte move (K1N-v2 attestation, mission pins, the K1 admitted-file set) |
| ECS memory (`ecs_memory.c/.h`, `ecs_memory_test_util.h`, memory tests) | ECS | **ECS_C** | `native/ECS_C/memory` | byte move; its capability and refusal laws unchanged |
| composite retained state (`ecs_composite.c/.h`, its test) | ECS | **RuntimeCore** (composition of ECS_G and ECS_C) | `native/runtime/composite` | byte move; `ELPISEC1` unchanged |
| HACF -> memory observation adapter (`native/runtime/observation`) | runtime | RuntimeCore | unchanged | none |
| continuity register (`native/continuity`, Rust, C ABI v1, `libelpis_continuity`) | continuity | **ECS_C** authority | `native/ECS_C/authority` | Stage C successor (authenticated fixed-size authority records holding the materialized ECS_C state, bounded semantic replay, rollback witness; section 5a); the standalone library and the Python adapter are retired in Stage F after every consumer passes the successor's tests |
| K1 Recovery R0 checkpoint store (`native/runtime/src/checkpoint.rs`) | runtime | RuntimeCore (bytes) under ECS_C authority (which identity resumes) | unchanged | Stage D: slots stay evidence; the authority that names them is ECS_C's |
| RuntimeCore (`native/runtime`, Rust, ABI v5) | runtime | RuntimeCore | unchanged | Stage D (ABI v6): embeds the ECS_C authority beside the standalone continuity crate, one backend per handle; the continuity backend retires with continuity (Stage F) |
| Python composition, deployment, CLI, acceptance (`src/elpis/runtime/*`) | runtime (Python) | **native** operator host | `native/runtime/host` | Stage D/F successor; Python stays as an offline harness only |
| Python bindings (`src/elpis/ECS/*`, `src/elpis/continuity`, `src/elpis/runtime/core.py`) | ECS / continuity / runtime | offline verification tooling | unchanged | reclassified when Python leaves the production path (Stage F) |
| contextual (WARM) lane (`src/elpis/runtime/context.py`) | runtime (Python) | ECS_C (context and structural-memory interface) | Stage E | native successor only where a production consumer exists; otherwise offline reference |
| typed-hypergraph cellular memory (HCA, `research/ecsc_hca_r0`, Python) | research | ECS_C research substrate, gated | `native/ECS_C/hca` | Stage E: native substrate with B0/B1/B2 comparators; never enabled in the canonical runtime |
| FMS, structure, pipeline, substrate | their subsystems | unchanged | unchanged | none (consumed by narrow native contracts only) |

Dependency direction (target, enforced by Stage B's successor gate):

```
substrate  <-  ECS_G  (math, K1, FMS residency; never ECS_C, RuntimeCore, codec)
substrate  <-  ECS_C  (memory, authority, gated HCA; never ECS_G mutation, RuntimeCore, codec)
continuity crate  <-  ECS_C authority  (Stages C to E only: SHA-256, the evolution digest, the legacy reader;
                                        folds into ECS_C in Stage F; continuity never depends on ECS_C)
structure  <-  RuntimeCore observation adapter
ECS_G, ECS_C, structure, substrate  <-  RuntimeCore (composite, managed turns, durability, recovery)  <-  native host (CLI)
```

## 4. ABI and persisted-format inventory

Exported symbols of every built library at the baseline (Release build of `18f2f4d`, whose native tree equals
`main`'s) are recorded in `native/ABI_INVENTORY.json`; Stage B adds a test that every relocated library still exports
exactly its recorded symbols under its recorded SONAME.

| Library (SONAME) | Exports | Depends on | Target owner |
|---|---|---|---|
| `libelpis_ecsg_math.so` | 48 | libc | ECS_G |
| `libelpis_ecsg_fms.so` | 33 | ecsg_math | ECS_G |
| `libelpis_ecsg_k1.so` | 60 | ecsg_math | ECS_G |
| `libelpis_ecsg_k1_fms.so` | 51 | ecsg_k1 | ECS_G |
| `libelpis_ecs_memory.so` | 28 | libc | ECS_C |
| `libelpis_ecs_composite.so` | 32 | ecsg_k1, ecs_memory | RuntimeCore |
| `libelpis_hacf_observation.so` | 23 | libc (structure validators linked statically) | RuntimeCore |
| `libelpis_continuity.so` (+ `_testing`) | 18 (20) | libgcc_s, libc | ECS_C successor; retired in Stage F |
| `libelpis_runtime.so` (+ `_testing`) | 55 (63) | libgcc_s, libc | RuntimeCore |

Persisted formats and identity domains that must not change meaning (a change is a new version, never a
reinterpretation):

| Format / domain | Owner | Bound by |
|---|---|---|
| `ELPISG01` (W + epoch snapshot), `ELPISGK1` v1 (K1 envelope; trailer = K1 retained-state identity) | ECS_G | Runtime R1 and K1N-v2 evidence, K1 Recovery R0 v4 evidence |
| `ELPISEM1` v1 (memory envelope) | ECS_C | MF-R0 canonical differential, known-answer identity `9b0e06c3…eb8162fc` |
| `ELPISEC1` v1 (composite envelope) | RuntimeCore composite | `test_ecs_composite`, K1 Recovery R0 v4 evidence |
| `ELPCONT\x02` / `\x03` (continuity record v2/v3), domains `elpis.continuity.register.v2/v3\0`, `elpis.continuity.evolution-authority.v2\0` | continuity → ECS_C | frozen parity vectors `native/continuity/tests/fixtures`, K1 Recovery R0 v4 |
| `ELPK1CK\x01` slot header, domain `elpis.k1-checkpoint.slot.v1\0` | RuntimeCore | K1 Recovery R0 v4 |
| `elpis.deployment.v1`, `elpis.lineage.v1`, catalog `elpis.inference-authority.v1` | Python deployment | `tests/integration/test_cli.py`, acceptance |
| evolution assertion v1 / receipt v0 digests | evolution (Python) | `tests/evolution` |

## 5. Consumers of the standalone continuity implementation

`native/continuity` (crate `elpis_continuity`, std only, C ABI v1: 18 functions, `continuity.h`; testing-only
`elpis_continuity_testing_fault` / `_counters`) has exactly one Cargo dependent and no production loader of its own
library:

| Consumer | What it uses (baseline file:line) | Replacement obligation |
|---|---|---|
| RuntimeCore crate (`native/runtime`) | embeds and re-exports the crate (`lib.rs:36-37`, so `libelpis_runtime.so` exports every `elpis_continuity_*` symbol); `Store::new/open/close/snapshot`, `anchor_cognition_as`, `commit_cognition(_as)`, `import_cognition`, `reserve_evolution`, `finalize_evolution` (`core.rs:250-1418`, `durable.rs:529-1146`); `store::resolve`, `RECORD_SIZE`, the testing fault and counter probes; `sha256::digest` for the checkpoint slot and envelope digests (`checkpoint.rs:43-171`) | Stage D: the ECS_C authority serves every one of these behaviours; SHA-256 moves with it |
| `runtime.h` and the five C runtime tests | `elpis_continuity_snapshot` / `_evolution` types in `open`, `snapshot`, `anchor`, `import_composite`, `adopt`, evolution and `elpis_runtime_durability.durable` (`runtime.h:95, 299, 380-504`) | Stage D: a versioned RuntimeCore ABI (v6). The continuity snapshot types stay as the backend-neutral view (generation, cognition lineage, format, evolution); the ECS_C record is reported by `elpis_runtime_authority_status` |
| `native/runtime/containment/elpis_autonomous_run.c` and its test | the two slot names and the 176-byte size (`:35, 69-97, 154-195`) | Stage F (not Stage D): the launcher still grants the continuity slots; under ECS_C the grant becomes the authority's fixed file set and sizes, with the native host |
| Python `elpis.continuity` adapter, `runtime/core.py`, `composition.py`, `cli.py`, `deployment.py`, `acceptance.py`, `persistence.py`, `release.py` | `ContinuityLibrary` over the runtime library (requires the 15 re-exported symbols), snapshots, evolution, testing faults; the continuity directory of a deployment | Stage F: the native host replaces the production uses; the adapter and its tests retire with the standalone library |
| `src/elpis/evolution/path_gate.py:39-50, 111-125, 330-332` | the evolution-authority **v2 digest** is persisted inside `elpis.evolution-path-assertion.v1` | the successor must keep that digest byte-identical for the same `(revision, head, pending, assertion)` |
| `libelpis_continuity.so` itself | loaded only by `tests/continuity/test_adapter.py` and `native/continuity/tests/test_continuity_abi.c` | retired with the crate (Stage F) |
| `research/qualification/k1_recovery_r0_v4` (`check.py:43-118`, `runner.py`, `freeze.py`, `premise.py`) | an independent reader of v2/v3 records; premise **P1**: `native/continuity` byte-identical to `65287ad` | the qualified evidence is bound to `18f2f4d` and stays valid there; `tests/research/qualification_k1_recovery_r0_v4/test_freeze.py` asserts the premise at `HEAD` and must instead assert it at the qualified commit once continuity changes (the qualification does not extend to the successor) |
| tests | `tests/continuity`, about 30 integration and boundary suites reading continuity snapshots, raw slots, I/O counters or the symbol set (`test_one_ecs.py:63-176`, `test_autonomous_persistence.py`, `test_runtime_hot_path.py`, ...) | each is ported to the successor or retired with what it guards, never deleted without its successor |

Duplicated behaviour to remove with it: RuntimeCore's prechecks ahead of the store's own (`core.rs:422-424,
1238-1245, 1325-1341`, `durable.rs:1003-1013`) and the lane's cached snapshot mirror; the stale documentation noted
by the inventory (`docs/ARCHITECTURE.md` `ContinuityLibrary(path)`, `docs/PYTHON_AUTHORITY_CENSUS.md:27`,
`ELPIS_SYSTEM.json` continuity `runtime_participation` and its missing `register.v3` identifier).

## 5a. The persistence invariant (binding on every stage)

**Zero autonomously ever-expanding Elpis-owned persistence.** No ECS_C structure may grow with runtime duration,
turns, observations, memory updates, reclamation or restarts. Therefore ECS_C keeps **no append log, no segments,
no generations and no compaction archive** (the donor's log, session, segments and compaction are not reused as
storage), and every persistent structure is fixed-capacity, preallocated at provisioning and rewritten in place:

| Structure | Files | Size | Rewritten |
|---|---|---|---|
| ECS_C authority record (the complete materialized ECS_C authority state, plus the one transition that produced it and its predecessor's state root) | `authority.a`, `authority.b` | fixed by the provisioned capacity (entities, sources), checked at open | the non-current slot, in place, on each authority transition (WARM / control, never HOT) |
| K1 Recovery R0 checkpoint (the composite envelope, which contains the memory) | `k1-checkpoint.a`, `k1-checkpoint.b` | fixed by the envelope size | as today |
| rollback witness (lineage, generation, state root, authenticated) | one file in a separately protected directory | fixed | after each authority transition |

Consequences, stated rather than hidden:

* **Bounded replay.** Every transition is semantically validated against its predecessor before it is written.
  Recovery reads exactly the two authority slots (constant work), authenticates and validates the materialized
  state, and re-replays the newest transition against the predecessor that the other slot still holds. Earlier
  transitions are folded into the authenticated materialized state; they are not re-replayed and not retained.
* **Receipts.** A transition's receipt is returned to its caller. ECS_C keeps only its digest as the head of a
  hash chain (and its count), so an operator who keeps receipts can verify them against the head; Elpis keeps no
  receipt collection.
* **Capacity.** A transition that would exceed a provisioned capacity (entities, sources) is refused before any
  write. Nothing authoritative is discarded to make room; a larger capacity is an explicit, operator-provisioned
  successor store.
* **Proof obligation.** Each persistence component ships tests that drive arbitrarily long sequences (and crashes
  at every write step) and assert the exact file set, total bytes, recovery work and retained metadata stay
  constant.

If a required authority or recovery guarantee cannot be kept within these bounds, that component stops and the
conflict is reported; expanding storage is never the workaround.

## 6. Historical ECS_C donor

`origin/integration/native-ecsc-turn-continuity-r1` (head `7d00038`) holds exactly the trees that CONT-5
(`c96e64d`, 2026-10-07) deleted from `main` (`native/ECS_C`, `src/elpis/ECS_C`, `tests/ECS_C`; 80 files, -26,548
lines). The later PR #51 merge of that branch added only `DEV/ecsc_hca` and `research/ecsc_hca_r0`, which are HCA
research and import none of it. The CONT series that retired it: CONT-1 `cfec80b` (the consumer census), CONT-2
`4685eb4` (the two-slot register), CONT-3 `a277b3e` (receipts and `runtime/history.py` deleted), CONT-4 `620210e`
(ECS_G -> ECS rename), CONT-5 `c96e64d` (ECS_C deleted), CONT-6 `04aa89b` / CONT-7 `5a715ad` (one-ECS docs and the
retired-name gate).

| Donor piece (`c96e64d^`) | What it is | Disposition |
|---|---|---|
| `native/ECS_C/src/ecsc_history.c` (3904 lines), `ecsc_history.h` ABI v1 | canonical-JSON codec for receipts, envelopes, events (`ecs.event.v1`), genesis / entity / state identities (domain-separated SHA-256 of sorted-key JSON), state-root serialization (`ecs.state_root.v3`), a two-event receipt planner; strict UTF-8 | **concepts only**: the identity derivations (genesis binds the protocol, entity id binds genesis + founding index + label), the lifecycle machine FOUNDED -> ACTIVE <-> DORMANT -> TERMINATED, the per-sender watermark law (`seq == watermark + 1`), the predecessor chain and before/after state roots. Not reused as code: semantic validation of its events existed only in Python, its frames carry no checksum or authentication, and its JSON events need a native parser the canonical runtime does not have |
| `native/ECS_C/src/ecsc_log.c` (767) | 8-byte big-endian length-framed log; structural scan (complete prefix, incomplete tail, a partial header only when a legal length could complete it); truncation of the tail only, never below the scanned prefix; append = write + fsync, rollback by truncate + fsync, otherwise poisoned (`APPEND_UNCERTAIN`) | **not reused as storage** (an append log grows with turns; section 5a). Kept as law: complete-or-absent records, refusal before I/O, an explicit uncertain disposition when durability cannot be established |
| `native/ECS_C/src/ecsc_session.c` (1095) | durable receipt session over a seed state that Python replay had validated; segment base and size policy; `SEGMENT_FULL` before any I/O; `PARTIAL_COMMIT` when the second of two frames fails | **not reused as storage** (segments and compaction accumulate; section 5a). Kept as law: refusal before any I/O when capacity would be exceeded, an explicit partial-commit disposition. Not kept: trust in an externally validated seed (ECS_C validates every transition natively) |
| `src/elpis/ECS_C` (Python kernel, replay, compaction, generations, topology, projections, Structural R0) | the reference semantics (`replay._apply_event`) | reference only (the charter forbids it as the execution path); no code moves into `src/` |
| tests (`tests/ECS_C`, 33 modules; native 3 suites) | differential, fault and determinism suites | patterns reused for the new native suites |

It does not build against `main` as is: `ELPIS_SYSTEM.json` declares no ECS_C subsystem (configure fails),
`test_one_ecs.py` forbids its paths and tokens, and continuity refuses its directory layout as
`CONTINUITY_LEGACY_STORAGE`.

## 7. Python on the production path

`python -m elpis.runtime.cli` imports 95 `elpis` modules (about 22,000 lines); about 7,300 lines in 27 modules run on
some command (runtime: cli, deployment, composition, core, cognition, fuel, observation, context, recovery, errors;
ECS: composite, k1, native; continuity: adapter; substrate: authority, native_admission, boundary, contracts,
digests). The rest (evolution, pipeline, inference, structure: about 14,200 lines) is imported by `composition.py`
and `edges.py` and never called by any CLI command.

Authority that exists only in Python today, and that the native host must carry (or explicitly retire with a
reason) before Python leaves the production path:

| Python authority (file:line at the baseline) | Native successor |
|---|---|
| deployment home safety, write-once files, bounded reads (`deployment.py:79-135`) | native host (Stage F) |
| deployment schema and ABI pin, catalog SHA-256 pin, strict catalog parse (`deployment.py:248-258`, `substrate/authority.py:55-86`) | native deployment authority (Stage C/F), versioned |
| library admission: `openat2(RESOLVE_BENEATH | NO_SYMLINKS)`, sealed memfd, SHA-256, SONAME binding check (`substrate/boundary.py`, `substrate/native_admission.py`) | native host: no runtime code loading at all (dependencies bound by the ELF loader, then every mapped Elpis object verified against its pin before the first operation) |
| `initial_weights` law, domain `elpis.deployment.initial-w.v1` (`deployment.py:142-151`), `LineageConfig` defaults | native host, bit-identical (tested against the Python law offline) |
| `LearnAuthority`: rate finite and **> 0** (native K1 admits 0), non-empty grant (`cognition.py:261-268`) | native host admission |
| `ObservationAuthority`: grant, admitted source set, at most 64 per turn (native ceiling 256), source membership (`cognition.py:287-319`) | native host admission (RuntimeCore never reads observations) |
| composite capabilities forced to 0 (`ECS/composite.py:225`) | native host (and a test that no capability bit is ever set) |
| restore identity equals the authorized one (`deployment.py:329-330`; not repeated after `reconcile`, `:341-348`) | native host, on every restore including after reconciliation |
| recovery double read, ticket acknowledgment-law re-check, observe-only generation check (`core.py:594-598, 682-683`; `composition.py:623-624`) | native host |
| the contextual WARM lane (`context.py`, 655 lines, a thread) | Stage E decision: native only for a production consumer; the native CLI exposes no context operation until then |
| script, output and exit-code contract (`cli.py`) | native host; exceptions that today escape as tracebacks (exit 1) become classified refusals |

RuntimeCore already serves create, open, close, fault, snapshot, counters, anchor, query, both managed turns,
observe, durable requests and waits, live identity, checkpoint provision / attach / recover / discard / adopt and
evolution; it has no deployment, catalog or loader logic and links no ECS code (the caller supplies the substrate
table; `native/runtime/tests/test_runtime_composite.c` already builds one in C).

## 8. Evidence and gate bindings that constrain the transition

| Binding | What it pins | How the transition keeps it |
|---|---|---|
| `research/ecs_runtime_r1/evidence` | SHA-256 of 12 measured native sources under `native/ECS_G/...` (their recorded paths) | the byte move restores the recorded paths; the digests still verify |
| `tests/research/_renamed_sources.py` | the one stated rename `native/ECS_G/` → `native/ECS/` | Stage B states the second, inverse relocation exactly (and the memory/composite moves), per file and byte-checked; nothing else is mapped |
| `tests/boundary/_mission.py` `ECS_KERNEL_PINS` | kernel source digests by path | paths updated, digests unchanged |
| Retention R1/R2/R3 vocabulary guards, `tests/research/_k1_promotion.py` | canonical ECS sources carry no retention vocabulary except the admitted K1 milestone files | the guards scan the ECS_G and ECS_C roots (every file they scanned, at its new path); the admitted set names the same files at their new paths |
| `tests/boundary/test_one_ecs.py` `BYTE_BOUND` | native sources recorded by Runtime R1 evidence | same digests at the restored paths |
| K1 Recovery R0 v4 evidence and premise (`premise.py`, P1–P4 against `65287ad`) | the qualified commit `18f2f4d` | the evidence stays bound to `18f2f4d`; once continuity or the durability law changes, the premise holds for the qualified commit only, and the tests say so instead of asserting it at `HEAD` |
| `research/__init__.py` | frozen labs import `elpis.ECS_G`, mapped to `elpis.ECS` | Python package names are unchanged by the native split |
| CMake `elpis_verify_targets()` | every target belongs to a declared subsystem | ELPIS_SYSTEM.json declares `ECS_G` and `ECS_C` in the same change |

## 9. Retirement ledger

| Surface | Retired when | Successor | Condition |
|---|---|---|---|
| subsystem id `ECS`, `native/ECS` | Stage B | `ECS_G`, `ECS_C`, `native/runtime/composite` | byte moves verified; ABI inventory equal; every native suite green under its new label |
| `tests/boundary/test_one_ecs.py` (one-ECS assertions) | Stage B | `tests/boundary/test_ecs_split.py` | the successor enforces the split's dependency directions, ownership and the HOT-lane prohibitions at least as strongly. The path itself stays until Stage F as a runner of the successor: `native/continuity/qualify.sh` names it and cannot be edited, because the K1 Recovery R0 v4 premise P1 binds `native/continuity` byte for byte to the ABI v4 commit |
| `libelpis_continuity.so`, `elpis.continuity` adapter, `native/continuity` | Stage F | ECS_C authority (`native/ECS_C/authority`) | every consumer listed in section 5 passes the successor's contract tests; legacy v2/v3 directories are imported explicitly, never reinterpreted |
| Python `Runtime` facade, deployment, CLI as the production path | Stage F | native host `elpis` | native end-to-end, crash and acceptance suites pass on the installed native binary; the process closure loads no Python |
| one-ECS wording of `ELPIS_MISSION.md`, `CONTINUITY.md`, `ARCHITECTURE.md` | Stage B (mission amendment), Stage G (final text) | the amended mission | documentation describes only what exists |

## 10. Source-preservation plan

* Every geometric, memory and composite source moves byte for byte (`git mv`); a move never edits a pinned file.
  Comments or headers that need new wording are edited only where no evidence binds them, and each such edit is
  listed in its pull request.
* No numerical code, update order, envelope encoder, identity function, status value or exported symbol changes in
  Stage B. The ABI inventory test proves the exported symbols; the existing suites (K1 legacy differential,
  memory known answer, composite layout) prove behaviour.
* Frozen evidence is never edited. Guards that map recorded paths to current ones state each relocation exactly.
* New contracts (ECS_C authority records, deployment authority and rollback witness, native host) get new format versions,
  new identity domains and their own tests before any consumer switches to them.

## 11. Stage plan and rollback

| Stage | Pull request content | Reverts cleanly because |
|---|---|---|
| A | this census, `native/ABI_INVENTORY.json` | documentation and a data file only |
| B | byte moves into `native/ECS_G`, `native/ECS_C/memory`, `native/runtime/composite`; build graph; ELPIS_SYSTEM subsystems `ECS_G`/`ECS_C`; mission amendment; successor architecture gate; guards re-pointed | no byte of a moved file changes; reverting restores the old paths |
| C | ECS_C authority crate: authenticated fixed-size authority records over the materialized ECS_C state, semantic validation of every transition and bounded replay of the newest, deployment authority and rollback witness, recovery, fault matrices, boundedness proofs (section 5a) | additive: no consumer uses it yet |
| D | RuntimeCore embeds the ECS_C authority; durable publication, recovery and reconciliation record ECS_C events; native host library surface | versioned RuntimeCore ABI; the K1 Recovery R0 v4 subject stays bound to `18f2f4d` |
| E | native memory operations, gated HCA substrate (B0/B1/B2) | gated: the canonical runtime enables none of it |
| F | native operator binary, native end-to-end and crash suites, process-closure proof, retirement of standalone continuity and the Python production path | the retirement is the last step and only after its consumers pass |
| G | documentation and machine-readable authority match the implementation | documentation |

No stage is merged without explicit authorization; a red or incomplete stage is reported as such.

Progress (updated by each stage's pull request): **A** census and ABI inventory; **B** the split (byte moves,
build graph, successor gate; `native/continuity` stays byte-identical to the K1 Recovery R0 v4 premise P1, so its
frozen `qualify.sh` keeps naming `tests/boundary/test_one_ecs.py`, which now runs the successor gate); **C** the ECS_C
authority library (`docs/ECS_C_AUTHORITY.md`): materialized authority in two fixed slots, authenticated records and a
protected rollback witness, one transition law for validation and replay, every recovery disposition, explicit
operator reconciliation, legacy import, crash / adversary / boundedness suites. The authority embeds the continuity
crate until Stage F (an amendment of Amendment 1 stated in `docs/ELPIS_MISSION.md` and enforced by the mission and
split gates), and no runtime consumer uses it before Stage D. **D** RuntimeCore ABI v6 (`docs/RUNTIME_CORE.md`, "The
durable authority (ABI v6)"): one durable-authority interface over two backends, the ECS_C authority composed natively
(`elpis_runtime_create_ecs_c`): typed publication events, native typed-evidence admission against the authority's
sources on the HOT lane (a bounded table, no I/O), evidence summaries recorded by the durability lane, operator events
through the held authority, an open refused while the authority needs reconciliation; the whole RuntimeCore law suite
runs over both backends. The K1 Recovery R0 v4 RUN PASS stays bound to `18f2f4d`: Stage D changes `durable.rs`, so the
premise breaks with exactly P4 at this head (P1 to P3 hold), and its tests assert that, and re-check the premise at the
qualifying commit, instead of asserting it at `HEAD`; the frozen instrument refuses an ABI v6 deployment.

Stage D conflicts, stated rather than worked around:

* **Cognitive Boundary R2 v5** (NOT RUN) names RuntimeCore ABI v5 as its subject and qualifies only its own frozen
  commit. ABI v6 adds mutation paths that specification does not cover (operator events through a held authority, the
  ECS_C publication path, native admission), so it cannot be run against a v6 head and stretched to cover them. It
  stays as written, runnable only at an ABI v5 commit. Proposed successor: **Cognitive Boundary R2 v6**, a separately
  versioned specification naming the v6 subject, adding to R2 v5's catalog an operator event sent while a turn is open,
  RuntimeCore's own event kinds and reconciliations sent through `elpis_runtime_authority_transition`, an observation
  from an unknown, DORMANT or exhausted source, a publication against a stale authority generation, and an open over a
  reconcile-only authority; pre-registered before any run, not written in this stage.
* **Deployments pinned to ABI 5** are refused (`DEPLOYMENT_ABI`) by this facade, as every ABI bump refuses an older
  pin; a deployment is re-initialized against the new libraries. Its continuity lineage directory is the same format.
* **The Python facade** still drives the continuity backend; it gains no ECS_C path (Stage F replaces it).

**E** the HCA native research substrate (`docs/ECS_C_HCA.md`, `native/ECS_C/hca`): B0 (the exact control), B1 and B2
ported from `research/ecsc_hca_r0`, equal to it on every prediction, refusal and B2 activation (offline differential,
SCIENTIFIC lane), with the required mechanics natively: n-ary relations, interference and conflicting scope, source
invalidation, delayed memory, activation damage (and the unsupported-cycle limitation), provenance misbinding and
matched-resource exact-storage controls. Gated: nothing canonical links or loads it (`tests/boundary/test_ecs_c_hca.py`).
The memory's operations are unchanged: exact typed evidence and LOOKUP admitted, LEARN / QUERY FAST behind the
unqualified capability, CONSOLIDATE not provided. No HCA law is admitted; no scientific claim is made.

**F1** the native operator host (`docs/NATIVE_HOST.md`, `native/runtime/host`): `elpis`, one C program linking only the
C library, admits the seven pinned libraries from sealed copies at every start (dependencies checked before loading,
SONAMEs after), pins its own binary, and drives RuntimeCore over the ECS_C authority: provisioning, the lineage,
QUERY, LEARN, typed evidence admitted by the authority's sources, LOOKUP, inspection, recoverable publication,
restart, checkpoint and authority reconciliation, operator source events, the text refusal. Its suites prove the
process closure (no interpreter mapped, fixed descriptors), equality with the Python operator interface, the crash
matrix, the attack refusals and a fixed file set over long runs (`tests/integration/test_native_host.py`,
`runtime.native_host_smoke`). RuntimeCore ABI v6 gained two read-only entry points before any release
(`elpis_runtime_authority_entity_at`, `elpis_runtime_authority_next_entity_id`). The host contains itself after
admission (`--contained`: Landlock grants writes to exactly the six fixed slots; the launcher's file size limit would
stop admission's sealed copies). It is installed as `bin/elpis` and `lib/elpis/` (`cmake --install`, `init
--libraries`), and the Installed system acceptance job runs the full native suite against the installed binaries.
**F3** the Python deployment, CLI, acceptance and release modules are reclassified as the offline reference interface
(differential tests, the qualification instruments bound to their commits, research); `docs/OPERATING_ELPIS.md`
operates the native host.

Stage F conflict, stated rather than worked around:

* **Standalone continuity cannot be retired yet (F2).** Section 9 retires it only after its successor is qualified.
  The successor's qualification is pre-registered: `docs/qualification/ECS_C_RECOVERY_Q0.md` (NOT RUN). Like K1
  Recovery R0 v4, it freezes on the accepted source, the merge commit of these stages into `main`, and merging needs
  the owner's explicit authorization.
  - **The instrument.** `research/qualification/ecs_c_recovery_q0/` is written. Its rehearsal on DEV inputs passes,
    and the same instrument fails the Stage F1 build at exactly the two defects F2 fixed. DEV records are not
    evidence.
  - **Until the QUAL run passes, continuity stays.** It is the authority the offline Python reference interface
    publishes to, the backend RuntimeCore's continuity law suite runs over, and K1 Recovery R0 v4's premise P1. The
    containment launcher keeps granting its slots for that path.
* **A deep rollback, once accepted, leaves the lineage without its state.** The K1 checkpoint store keeps two envelopes
  (the fixed-storage law), and K1 Recovery R0's law never offers a slot as a candidate when no slot holds the
  authorized identity (`CHECKPOINT_MISSING`, `checkpoint.rs` `unresolved`).
  - **The case.** A state-directory adversary copies back records two or more publications old, and the operator
    accepts the rollback (the only reconciliation the authority offers). The authority then names a state neither slot
    holds: cognition is refused, although the newest acknowledged envelope is intact in the store. Nothing is accepted
    silently, but the lineage cannot continue. ECS_C Recovery Q0 registers exactly this outcome (`rollback:2`,
    `rollback:5`). The remedy today is restoring the state directory from outside.
  - **The proposal: K1 Recovery R1, RuntimeCore ABI v7.** An explicit operator re-authorization of a verified slot
    under `CHECKPOINT_MISSING`: the operator names the slot's identity, and the runtime publishes it as the authority's
    next state. It changes K1 Recovery R0's qualified recovery law, so it needs its own versioned law and
    qualification; it is not made here.
