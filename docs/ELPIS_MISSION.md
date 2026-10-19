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

DSV4 is not the brain. It is the encode/decode boundary between human/token
space and Elpis's internal computational substrate.

The ECS has two native owners (Amendment 1, below): **ECS_G** (`native/ECS_G`)
owns geometric cognition, and **ECS_C** (`native/ECS_C`) owns continuity and
memory. Only RuntimeCore composes them. ECS_C holds the memory and, since
Stage C of the native split, the ECS_C authority (`docs/ECS_C_AUTHORITY.md`):
the entity, provenance, lineage and evolution authority that succeeds
continuity, built and qualified as a library and, since Stage D, composed by
RuntimeCore (ABI v6) as one of its two durable-authority backends. Until the
Python facade and standalone continuity retire (Stage F), continuity remains
the facade's live register: a fixed-size, crash-safe
register holding only the current durable authority the runtime must verify
(the last explicitly acknowledged K1 state digest and the evolution authority
head). Continuity has no entities, mailboxes, schedulers, projections,
topology, event log, compaction or retention (`docs/CONTINUITY.md`). Neither
decides when Elpis may think.

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

## Execution lanes (architectural correction, Elpis_DEV)

Active computation is decoupled from durable bookkeeping. Each responsibility
has its own lane and its own coordinate:

* the **geometric (HOT) lane** computes: QUERY and LEARN on the live ECS
  state, under RuntimeCore's managed lease, at the state's live coordinate
  (its K1 generation). A managed commit is volatile: no filesystem write or
  sync, no continuity publication, no checkpoint, no digest, and no wait for
  any other lane;
* the **contextual (WARM) lane** maintains the active relational context: a
  bounded, session-local set of provenance-bearing axioms and typed relations
  built from admitted HACF references, published as immutable versioned views
  the HOT lane consumes without waiting (`elpis.runtime.context`). It is not a
  text window, a history or a persistence system, and no contextual dynamics
  law (HCA) is qualified or implemented;
* **HACF addresses** existing resources and their relationships in place;
  **FMS materializes** bounded resources;
* the **durability lane preserves explicitly acknowledged state**: on an
  explicit request it checkpoints and publishes a captured state off the HOT
  lane, and acknowledges only what restart can recover.

The live state, the published context view, the last durably acknowledged
state and the reserved irreversible operations (evolution) are four separate
coordinates. **Superseded:** through RuntimeCore ABI v3 the runtime published
continuity (and, with a checkpoint store, a complete checkpoint) after every
native K1 commit, before the turn returned. That per-turn durability
requirement is withdrawn by this correction; evidence recorded under it is
not reinterpreted.

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
  persistent, evolving computational substrate
  recurrent dynamics, distributed state, relations,
  attractors, dynamically materialized computation
  (FMS materializes what is active; HACF/structure
   supplies persistent structural relationships)
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
| ECS_G (`native/ECS_G`; Python bindings `elpis.ECS`) | owns active microscopic dynamical state (`W`), its qualified recurrence, coarse observables (`S3`) and snapshots; a qualified primitive of the cognitive substrate | import inference, codec, runtime, continuity or ECS_C; be an auxiliary conditioning vector for a model |
| ECS_C (`native/ECS_C`) | owns the bounded typed relational memory (exact typed evidence, LOOKUP) and the ECS_C authority (entities, lifecycles, sender attribution, source versions and scopes, the cognition lineage, the evolution authority, receipts, reconciliation; `docs/ECS_C_AUTHORITY.md`), which RuntimeCore composes from Stage D | import inference, codec, runtime or ECS_G; perform geometric learning or mutate ECS_G's state; keep growing persistence |
| continuity (`native/continuity`, Rust behind a C ABI; `elpis.continuity` adapter) | the current durable runtime lineage/authority: one fixed-size two-slot register binding the last explicitly acknowledged K1 state digest and the evolution authority (revision, head); fail-closed restart verification | hold or substitute for active cognitive state; keep history, events, entities, schedules, projections or topology; supervise or gate foreground cognition; depend on ECS, K1 execution, runtime or inference |
| DSV4 codec (`elpis.inference` codec modules) | communication: text/bytes <-> tokens, vocabulary identity, bounded rendering; the token side of encode/decode | own ECS state; run an autonomous transformer/MoE cognition path |
| substrate / FMS | generic resource authority, residency and materialization: persistent logical capacity >= resident materialization >= active materialization | require DSV-specific tensor roles in its generic core; decide semantics |
| structure / HACF | persistent structural memory, topology, representation, provenance | be defined solely as a context-window supplier for a model |
| runtime | orchestrates codec -> ECS -> codec on the HOT lane (a managed commit is volatile: it advances the live coordinate only); on explicit request, its durability lane captures the live state and publishes it (checkpoint slot when provisioned, then one 176-byte continuity record and one fdatasync); its contextual lane keeps the bounded active context | route cognition through a DSV model; record per-turn receipts or event history; make a HOT operation wait for durable publication or for the contextual lane |

## Classification of existing machinery

| Machinery | Classification | Where |
|---|---|---|
| V4.1 tokenizer, chat/text rendering, incremental decode (`text.py`), codec contracts (`contracts.py`), budgeted rendering of admitted structural memory (`admission.py`, `structural.py`) | CODEC / COMMUNICATION (canonical) | `src/elpis/inference` |
| execution port, descriptor capabilities, resource authority, FMS file service and HOT/WARM/COLD residency | GENERIC SUBSTRATE (canonical) | `src/elpis/substrate`, `native/substrate` |
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
4. ECS_G depends on inference, runtime, continuity or ECS_C, or ECS_C on
   inference, runtime or ECS_G, or on continuity beyond the one embedded crate
   it succeeds (sources; the build, crate and binary link graph is checked by
   `tests/boundary/test_ecs_split.py` and `tests/boundary/test_abi_inventory.py`);
5. the codec depends on ECS, continuity or the runtime;
6. continuity and ECS depend on each other;
7. the generic substrate names DSV-specific tensor roles;
8. the canonical text turn does not fail closed without a qualified codec map;
9. the gate itself no longer rejects the forensic sidecar topology;
10. the qualified ECS kernel sources change without an explicit,
    reviewed update of their pinned digests.
