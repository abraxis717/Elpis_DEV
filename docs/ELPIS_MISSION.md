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

There is one ECS (`elpis.ECS`, `native/ECS`). Continuity is not an ECS and not
a history: it is a fixed-size, crash-safe register holding only the current
durable authority the runtime must verify (the committed K1 state digest and
the evolution authority head). It has no entities, mailboxes, schedulers,
projections, topology, event log, compaction or retention
(`docs/CONTINUITY.md`).

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
| ECS (ECS/EDEN-facing geometric substrate) | owns active microscopic dynamical state (`W`), its qualified recurrence, coarse observables (`S3`) and snapshots; a qualified primitive of the cognitive substrate | import inference, codec, runtime or continuity; be an auxiliary conditioning vector for a model |
| continuity (`elpis.continuity`) | the current durable runtime lineage/authority: one fixed-size two-slot register binding the committed K1 state digest and the evolution authority (revision, head); fail-closed restart verification | hold or substitute for active cognitive state; keep history, events, entities, schedules, projections or topology; import ECS, runtime, inference or native code |
| DSV4 codec (`elpis.inference` codec modules) | communication: text/bytes <-> tokens, vocabulary identity, bounded rendering; the token side of encode/decode | own ECS state; run an autonomous transformer/MoE cognition path |
| substrate / FMS | generic resource authority, residency and materialization: persistent logical capacity >= resident materialization >= active materialization | require DSV-specific tensor roles in its generic core; decide semantics |
| structure / HACF | persistent structural memory, topology, representation, provenance | be defined solely as a context-window supplier for a model |
| runtime | orchestrates codec -> ECS -> codec and, after each native K1 commit, publishes the new K1 state digest to continuity (one 136-byte slot write and one fdatasync) | route cognition through a DSV model; record per-turn receipts or event history |

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
4. ECS depends on inference, runtime or continuity;
5. the codec depends on ECS, continuity or the runtime;
6. continuity and ECS depend on each other;
7. the generic substrate names DSV-specific tensor roles;
8. the canonical text turn does not fail closed without a qualified codec map;
9. the gate itself no longer rejects the forensic sidecar topology;
10. the qualified ECS kernel sources change without an explicit,
    reviewed update of their pinned digests.
