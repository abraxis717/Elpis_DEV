# Elpis mission: the constitutional division of responsibility

This document is repository authority. It takes precedence over any
description of the current code, including `ELPIS_SYSTEM.json`, which must be
reconciled to it. A change that conflicts with it is rejected however well
its mechanics are qualified.

## The mission

    DSV4 COMMUNICATES.
    ECS / EDEN COMPUTES, LEARNS AND PERSISTS.
    FMS MATERIALIZES.
    HACF / STRUCTURE ORGANIZES PERSISTENT STRUCTURAL MEMORY.
    CONTINUITY BINDS MINIMAL DURABLE RUNTIME LINEAGE/AUTHORITY.

Amendment 3 (below) retires persistent runtime state: Elpis computes, learns
and binds its lineage and authority in volatile memory only, and reads
immutable resources. Where this section or Amendments 1 and 2 say "persists"
or "durable", Amendment 3 governs. The migration is complete; see its Status.

DSV4 is not the brain. It is the encode/decode boundary between human/token
space and Elpis's internal computational substrate.

The ECS has two native owners (Amendment 1, below): **ECS_G** (`native/ECS_G`)
owns geometric cognition, and **ECS_C** (`native/ECS_C`) owns continuity and
memory. Only RuntimeCore composes them. ECS_C holds the memory and the
volatile session authority (`docs/ECS_C_SESSION.md`): the entity, provenance,
lineage and evolution authority that binds the runtime's lineage for one
session, in volatile memory, RuntimeCore's only authority since ABI v8
(Amendment 3, below). It keeps no event log, compaction or retention, and
nothing of it outlives its session. The durable ECS_C authority it succeeded
(Amendment 2) is retired (`docs/ECS_C_AUTHORITY.md`, history); its ECS_C
Recovery Q0 pass for commit `117b41a` (ABI v6) remains the record of that
commit. Neither ECS_G nor ECS_C decides when Elpis may think.

## Amendment 1 (2026-10-18): the native ECS_G / ECS_C split

Authority: the repository owner's instruction for the native split, chartered by
`DEV/proposals/NATIVE_ECSC_ECSG_COMPLETION_R0.md`
(`proposal/native-ecsc-ecsg-runtime-charter-r0`), census
`docs/NATIVE_SPLIT_CENSUS.md`. It replaces "there is one ECS" and nothing else
in this document.

* **ECS_G** owns native K1 and its complete state `(W, epoch, H, a)`, the exact
  forward map and `S3`, query and deterministic learning, the qualified
  retention/consolidation law, transactions, geometric identities and snapshots,
  and FMS materialization. It depends on the substrate only: never on ECS_C,
  RuntimeCore, continuity or the codec.
* **ECS_C** owns continuity and memory: the bounded typed relational memory,
  and the lineage, entity, provenance and transition authority that succeeds
  continuity (Stage C, `docs/ECS_C_AUTHORITY.md`). It depends on the substrate
  and, only until standalone continuity retires into it (Stage F), on the
  standard-library-only continuity crate it embeds for SHA-256, the evolution
  digest and the legacy register reader (one implementation each, never a
  copy): never on ECS_G, RuntimeCore or the codec, and continuity never on
  ECS_C. It never performs geometric learning and never mutates ECS_G's
  state.
* **RuntimeCore** is the only composer: the composite retained state (K1 and
  the memory in one object, `native/runtime/composite`), managed turns,
  durability, recovery and reconciliation.
* **Persistence.** There is zero autonomously ever-expanding Elpis-owned
  persistence. ECS_C keeps no append log, segments, generations or compaction
  archive. Its durable state is a fixed set of preallocated files rewritten in
  place, and recovery reads a constant amount (`docs/NATIVE_SPLIT_CENSUS.md`,
  section 5a).
* **Unchanged:**
  - DSV4 only communicates;
  - the prohibited sidecar topology below;
  - the fail-closed text path (`ECS_CODEC_UNQUALIFIED`);
  - no per-turn durable work on the HOT lane;
  - every qualified law, format and identity;
  - every frozen record.
* **Not claimed.** A relocation of qualified sources qualifies nothing new.
  The successor gate is `tests/boundary/test_ecs_split.py`, which replaced
  `tests/boundary/test_one_ecs.py`.

## Amendment 2 (2026-10-19): standalone continuity retires into ECS_C

Authority: the repository owner's instruction to retire standalone continuity
only once a qualified successor exists, and census section 9
(`docs/NATIVE_SPLIT_CENSUS.md`). The successor is qualified: ECS_C Recovery
Q0 is RUN PASS for commit `117b41a`. This amendment replaces the clauses of
Amendment 1 and of this document that described standalone continuity as a
live subsystem (the continuity crate ECS_C embedded, the continuity row of the
roles, and enforcement items 4 to 6), and nothing else.

* **Retired:** `native/continuity` (the Rust register and its C ABI
  `elpis_continuity_*`), the `elpis.continuity` adapter and their tests. Their
  laws were moved, not rewritten: the ECS_C authority carries the SHA-256, the
  evolution law and its digest `elpis.continuity.evolution-authority.v2`, and
  the reader of the v2/v3 register, used only by explicit legacy import.
* **ECS_C** depends on the substrate only. It never depends on ECS_G,
  RuntimeCore or the codec.
* **RuntimeCore ABI v7** has exactly one durable authority, the ECS_C
  authority, on the native host and the offline Python reference alike.
* **Continuity** names a role, not a subsystem: binding the minimal durable
  runtime lineage and authority, which the ECS_C authority performs.
* **Not claimed.** The retirement qualifies nothing new: Q0's pass is for
  `117b41a` (ABI v6) only, and its ABI v7 successor is not run.

## Amendment 3 (2026-10-31): the volatile-native runtime

Authority: the repository owner's directive "ELPIS_DEV — VOLATILE-NATIVE
ARCHITECTURAL MIGRATION" (primary directive: eliminate all persistent runtime
state). Census: `docs/VOLATILE_RUNTIME_CENSUS.md`.

It supersedes every design of this repository that persists, recovers or
resumes runtime state: bounded persistent files, recoverable state, the
durability lane, checkpoint slots and crash-consistent authority. In this
document it replaces:
* "PERSISTS" in the mission's second line and "DURABLE" in its fifth;
* "PERSISTENT" in its fourth, for anything the runtime mutates;
* Amendment 1's Persistence bullet;
* Amendment 2's durable-authority clauses;
* the opening paragraph's durability protocol, as a target.

It is not a configuration option, and no persistent operating mode is offered:
the persistent runtime is retired, not optimized.

**VOLATILE MIGRATION: COMPLETE.** See Status below for what it covers and
what it does not.

* **The invariant.** Elpis may read immutable resources from persistent
  storage. Every mutable operational state transition occurs only in volatile
  memory. No runtime state is written to persistent storage under any
  circumstance:
  - not at initialization, inference or learning;
  - not on an operator command or a failure;
  - not at shutdown or restart;
  - not even when a publication is explicitly requested.
* **Prohibited in the runtime:**
  - persistent record publication, in-place slot overwrites, and checkpoints or
    snapshot files;
  - persistent event journals, and SQLite or other state databases;
  - persistent activation stores;
  - automatic model-weight updates, or H-gram or expert-relationship mutation,
    on disk;
  - writeback caches, and filesystem synchronization for runtime state;
  - recovery metadata or rollback witnesses;
  - runtime log files and persistent temporary files;
  - automatic state export at shutdown.
* **Volatile only:**
  - heap, arenas, in-memory tables, and volatile staging and caches;
  - source-version state in RAM, and ephemeral session identities;
  - read-only mappings of static artifacts;
  - bounded in-memory snapshots for rollback;
  - anonymous memory and memfd, only where non-persistence holds.

  There is no persistent fallback.
* **Static resources are read-only.** They are never modified at runtime.
  Adaptive state lives in separate volatile memory and is never copied back.
* **A session is finite.** It starts clean, and its state is lost on
  termination or crash. A restart is a new session. No crash-persistent
  identity, resurrection or exactly-once-across-restart guarantee is claimed.
* **Bounded.** Every mutable subsystem has a hard ceiling and refuses
  atomically when exhausted. Nothing spills to storage.
* **No hidden persistent channels.** The census audits caches, temporary
  directories, logging, mappings, swap, crash dumps and background components
  (section 5), and lists what the host must provide (section 6). Where the host
  cannot run Elpis over read-only persistent filesystems, Elpis fails closed.
  Process-level controls do not stop unrelated host services from writing.
* **Runtime is not development.** Git and source editing are development
  operations. They are never a channel for operational checkpointing,
  training-state writes or model-state export. An offline process that produces
  persistent model artifacts needs its own explicit authorization.
* **ECS_C** binds the session's lineage and authority in volatile memory,
  through the volatile session authority `ECSC_VOLATILE_SESSION_R0`
  (`native/ECS_C/session`, `docs/ECS_C_SESSION.md`), which holds the one
  transition law. The durable authority is RETIRED
  (`docs/ECS_C_AUTHORITY.md`, history): no source, build target or library
  remains, and git history keeps it. Its durability mechanisms are not kept
  behind a flag.
* **HCA** stays research only. B0, B1 and B2 may remain as in-memory
  comparators; B3 is not authorized.
* **Phases.**
  - V0: census and authority documents.
  - V1: the volatile ECS_C session.
  - V2: the relational view rebound to the session.
  - V3: RuntimeCore (ABI v8) and the host.
  - V4: FMS, HACF and the remaining writers.
  - V5: adversarial zero-write qualification.
* **Status.** V0 to V5 are implemented; the census records each phase's
  commits (`docs/VOLATILE_RUNTIME_CENSUS.md`, section 7).
  - RuntimeCore ABI v8 holds one volatile ECS_C session per open. Its
    persistent v7 interface is retired (`docs/RUNTIME_CORE.md`; the v7
    specification is history, `docs/history/RUNTIME_CORE_ABI_V7.md`).
  - The native host starts with no writable deployment home, and contains
    itself before its first session (`docs/NATIVE_HOST.md`). The Python
    reference interface has no reachable persistent-state writer, and
    refuses to open in a process that could write bytecode, dump core or
    page its state to swap (`elpis.runtime.environment`).
  - Retired: the durable ECS_C authority, the checkpoint store, the
    durability lane, recovery and publication from a runtime; the writable
    FMS COLD store (FMS ABI v3: WARM and HOT in RAM, COLD an immutable,
    read-only backing); the H-gram association writer.
  - `elpis.runtime.persistence.VOLATILE_MIGRATION_COMPLETE` is true: every
    runtime-connected writer is RETIRED, REPLACED_WITH_VOLATILE or a
    READ_ONLY_STATIC_RESOURCE, and all 14 closure conditions hold
    (`docs/VOLATILE_RUNTIME_CENSUS.md`, section 8). The composed runtime
    recorded zero write-class events on storage in every traced path
    (`research/volatile_runtime_v1`, `ZERO_WRITE_PASS`).
  - What it covers is Elpis's own processes. Host services outside Elpis's
    process-control boundary (swap configuration, crash collectors, journald,
    backups, filesystem metadata) are the host's, and the census, section 6,
    lists what the host must provide (`docs/NONCLAIMS.md`).
  - The documents must state the status exactly as
    `VOLATILE_MIGRATION_COMPLETE` computes it
    (`tests/boundary/test_volatile_runtime.py`).
* **Historical qualifications** remain the record of what their commits did:
  ECS_C Recovery Q0, K1 Recovery R0, and the native host's containment and
  acceptance. They qualify nothing under this amendment and are not promoted
  to it.

## Execution lanes (architectural correction, Elpis_DEV)

Active computation is decoupled from durable bookkeeping. Each responsibility
has its own lane and its own coordinate:

* the **geometric (HOT) lane** computes: QUERY and LEARN on the live ECS
  state, under RuntimeCore's managed lease, at the state's live coordinate
  (its K1 generation). A managed commit is volatile: no filesystem write or
  sync, no durable-authority publication, no checkpoint, no digest, and no wait for
  any other lane;
* the **contextual (WARM) lane** maintains the active relational context: a
  bounded, session-local set of provenance-bearing axioms and typed relations
  built from admitted HACF references, published as immutable versioned views
  the HOT lane consumes without waiting (`elpis.runtime.context`). It is not a
  text window, a history or a persistence system, and no contextual dynamics
  law (HCA) is qualified or implemented;
* **HACF addresses** existing resources and their relationships in place;
  **FMS materializes** bounded resources;
* the **durability lane** (ABI v4 to v7), which checkpointed and published a
  captured state on explicit request, is **retired** by Amendment 3: nothing
  is acknowledged durable, and a restart is a new session.

The live state, the published context view and the reserved irreversible
operation (evolution, a transition of the session) are separate coordinates,
all in volatile memory. **Superseded:** through RuntimeCore ABI v3 the runtime
published continuity (and, with a checkpoint store, a complete checkpoint)
after every native K1 commit, before the turn returned. That per-turn
durability requirement was withdrawn by this correction, and Amendment 3
retired durability altogether; evidence recorded under either is not
reinterpreted.

## The topology

```
text / bytes
    |
    v
DSV4 ENCODE (token boundary)
    |
    v
admitted ECS stimulus / perturbation
    |
    v
==================================================
ECS / EDEN
  evolving computational substrate (volatile:
  Amendment 3), recurrent dynamics, distributed state,
  relations, attractors, dynamically materialized
  computation (FMS materializes what is active;
  HACF/structure supplies structural relationships
  from immutable resources and volatile epochs)
==================================================
    |
    v
ECS readout
    |
    v
DSV4 DECODE (token boundary)
    |
    v
token distribution / tokens -> text / bytes
```

The middle computation is ECS/EDEN. The runtime composition orchestrates
codec -> ECS -> codec. Nothing else is the cognitive path.

## The prohibited topology

```
text -> full DSV transformer / MoE model -> text
              ^                  |
              +-- ECS state injected as conditioning, or
                  ECS updated from the model's output statistics
```

ECS as a sidecar around a full DSV model is forbidden in canonical code. This
covers, at least:

* ECS state (for example `S3`) projected into a DSV model and added to token
  embeddings, or otherwise used to condition a DSV model's reasoning;
* a full DSV turn whose final-distribution statistics (entropy, top-k
  probability shape, token IDs, Engram hash rows) are mapped into an ECS
  update;
* any runtime loop of the form `ECS -> DSV conditioning -> DSV cognition ->
  ECS feedback`.

Forensic example: commits `f4e1f05` (turn conditioning into the DSV4.1 tower)
and `75313fb` (`WorldModelLoop`: final-distribution summary -> `DriveMap` ->
one ECS step -> `S3` back into the tower) implemented exactly this
topology. Their mechanics were qualified; their architecture was wrong. They
remain in public history as evidence and are removed from the canonical path.

## Roles

| Subsystem | Role | Must not |
|---|---|---|
| ECS_G (`native/ECS_G`; Python bindings `elpis.ECS`) | owns active microscopic dynamical state (`W`), its qualified recurrence, coarse observables (`S3`) and snapshots; a qualified primitive of the cognitive substrate | import inference, codec, runtime or ECS_C; be an auxiliary conditioning vector for a model |
| ECS_C (`native/ECS_C`) | owns the bounded typed relational memory (exact typed evidence, LOOKUP) and the volatile session authority (entities, lifecycles, sender attribution, source versions and scopes, the cognition lineage, the evolution authority, receipts; `docs/ECS_C_SESSION.md`): the runtime's lineage/authority for one session, in volatile memory, RuntimeCore's only authority since ABI v8 | import inference, codec, runtime or ECS_G; perform geometric learning or mutate ECS_G's state; hold or substitute for active cognitive state; keep history, an event log or any persistence; supervise or gate foreground cognition |
| DSV4 codec (`elpis.inference` codec modules) | communication: text/bytes <-> tokens, vocabulary identity, bounded rendering; the token side of encode/decode | own ECS state; run an autonomous transformer/MoE cognition path |
| substrate / FMS | generic resource authority, residency and materialization: immutable read-only backing (COLD) >= resident materialization (WARM, RAM) >= active materialization (HOT); FMS ABI v3 holds WARM and HOT in volatile memory only | require DSV-specific tensor roles in its generic core; decide semantics; write or spill anything to storage |
| structure / HACF | structural memory, topology, representation, provenance: immutable resources read in place, and volatile retrieval epochs | be defined solely as a context-window supplier for a model; mutate a structural resource at runtime |
| runtime | orchestrates codec -> ECS -> codec on the HOT lane (a managed commit is volatile: it advances the live coordinate only); holds one volatile ECS_C session per open (RuntimeCore ABI v8); its contextual lane keeps the bounded active context | route cognition through a DSV model; record per-turn receipts or event history; make a HOT operation wait for the contextual lane; write any runtime state to persistent storage (Amendment 3) |

## Classification of existing machinery

| Machinery | Classification | Where |
|---|---|---|
| V4.1 tokenizer, chat/text rendering, incremental decode (`text.py`), codec contracts (`contracts.py`), budgeted rendering of admitted structural memory (`admission.py`, `structural.py`) | CODEC / COMMUNICATION (canonical) | `src/elpis/inference` |
| execution port, descriptor capabilities, resource authority, FMS file service (read-only COLD backing) and WARM/HOT residency | GENERIC SUBSTRATE (canonical) | `src/elpis/substrate`, `native/substrate` |
| production-shaped DSV4.1 tower (attention, Engram, mHC, routed/shared MoE, layer recurrence), sealed CPU-native backend, YTS-R0 provider stream, Native Clock R0, DSV-specific Native Materializer R1, donor differentials | FULL DSV COGNITIVE TOWER + DONOR / ORACLE / QUALIFICATION (noncanonical) | `research/dsv41_tower` (Python and native), `tests/research/dsv41_tower`, `docs/research/dsv41_tower` |
| legacy DSV4 compact synthetic target, decode transaction, sequence path, principal engine, steering, speculative drafting, associative rows, experts, prefetch, global context, safetensors preflight | DSV MODEL-EXECUTION MECHANICS, retained for historical replay of persisted identities and qualification (noncanonical; never composed by the runtime) | `src/elpis/inference` outside the codec modules |
| ECS-as-DSV-conditioning (`TurnConditioning`, `ConditioningProjection`, `FEATURE_CONDITIONING`, `WorldModelLoop`, `DriveMap`) | OBSOLETE SIDECAR INTEGRATION (removed) | public history only (`f4e1f05`, `75313fb`) |

The runtime reaches `elpis.inference` only through the codec modules; the
mission gate enforces it. `research/` is never packaged and never imported by
`src/`.

## Mechanics pass is not mission pass

A test proving that a module behaves according to its own specification is a
**mechanics pass**. It says nothing about whether the module occupies the
right place in the system. A **mission pass** requires that the canonical
dataflow is the topology above. Both are required. Mechanical qualification
of a DSV tower, a provider stream or a conditioning projection never
justifies making it the cognitive path.

## Missing interfaces fail closed

Where the mapping between DSV token space and ECS state is not scientifically
defined and qualified, the system says so and refuses:

    ECS codec mapping not yet qualified; text generation unavailable.

An honest missing edge is better than the wrong brain. It is forbidden to
fill the gap with software: no invented production ECS<->DSV map, no
routing through a DSV tower, and no use of final-logit entropy, top-k
probability shape, token IDs as numeric state, Engram hash values or random
projections as a semantic codec. Test fixtures may use deterministic
synthetic adapters to exercise interface mechanics only; they are labelled
`TRAINING=NONE SEMANTICS=NONE` and never presented as cognition.

## Enforcement

Prose is not the only guard. `tests/boundary/test_mission.py` fails when, in
canonical code (`src/`, `native/`; `research/` is noncanonical):

1. ECS-as-conditioning identifiers or structures reappear;
2. the runtime imports DSV model-execution machinery (anything in
   `elpis.inference` beyond the codec modules);
3. a DSV4.1 tower module exists in the canonical tree;
4. ECS_G depends on inference, runtime or ECS_C, or ECS_C on inference,
   runtime or ECS_G (sources; the build, crate and binary link graph is
   checked by `tests/boundary/test_ecs_split.py` and
   `tests/boundary/test_abi_inventory.py`);
5. the codec depends on ECS or the runtime;
6. the retired continuity package or library reappears;
7. the generic substrate names DSV-specific tensor roles;
8. the canonical text turn does not fail closed without a qualified codec map;
9. the gate itself no longer rejects the forensic sidecar topology;
10. the qualified ECS kernel sources change without an explicit,
    reviewed update of their pinned digests.

Amendment 3 has its own gates. `tests/boundary/test_volatile_runtime.py` and
`tests/boundary/test_autonomous_persistence.py` fail when:
- a source that writes storage is not a registered writer of the census;
- a runtime-connected writer has a disposition other than RETIRED,
  REPLACED_WITH_VOLATILE or READ_ONLY_STATIC_RESOURCE;
- a retired entry point reappears;
- a binary the runtime loads imports a write-class C library function;
- a document states the migration's status other than as it is computed.
