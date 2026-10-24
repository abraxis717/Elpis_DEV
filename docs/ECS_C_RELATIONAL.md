# ECS_C relational view (ECSC_DYNAMIC_RELATIONAL_DEV_R0)

`native/ECS_C/relational` (`libelpis_ecs_c_relational`, C ABI v1 `include/elpis/ecs_c_relational.h`). It maintains
bounded, source-revision-aware typed relational evidence for ECS_C. LOOKUP over it answers only from evidence that the
ECS_C authority's own admission law still accepts.

**Status: DEV, not composed.**
- Nothing canonical links or loads it (`tests/boundary/test_ecs_c_relational.py`): not RuntimeCore, the composite,
  the memory, the deployment's library set or `src/`.
- It changes no law, format or ABI of the memory, the authority or RuntimeCore, and qualifies nothing.
- It is session-local: nothing is persisted, no history is kept and no thread is started.

## The gap it closes (demonstrated, not assumed)

The ECS_C memory (`docs/ECS_MEMORY.md`) is source-blind:
- A slot is keyed by the relation alone, and the observations of every source are merged into its counts.
- Only the last 32-bit `source_authority` is kept.
- The source version and scope an observation was admitted under are recorded nowhere.

RuntimeCore's HOT-lane admission checks only that `source_authority` maps to an ACTIVE source within its per-turn
bound.

`ECS_C.test_ecs_c_relational_authority` runs this against the real authority and the real memory:

| Authority transition | `elpis_ecs_c_check_evidence` on the old evidence | memory LOOKUP | relational view |
|---|---|---|---|
| `SOURCE_VERSION` v1 → v2 | `SOURCE_STALE` | still 1 → 3 (score 2, from v1 evidence) | 1 → 2 (score 1: only admissible evidence) |
| `ACTIVE` → `DORMANT` | `LIFECYCLE` | still 1 → 3 | no live evidence |
| `TERMINATED`, reclaimed, new source on the same `source_authority` | `UNKNOWN_ENTITY` | still 4 → 5, and its `last_authority` now resolves to the new source, which never asserted it | retired; the new source's dependents are empty |

## Objects and law

Every input uses the real ECS_C types:
- authority rows: `elpis_ecs_c_entity`, read through `elpis_ecs_c_read_status` and `elpis_ecs_c_entity_at`;
- observations: `elpis_ecs_mem_observation`, keyed by `elpis_ecs_mem_relation_key`;
- requests and answers: `elpis_ecs_mem_request` and `elpis_ecs_mem_answer`;
- slots: `elpis_ecs_mem_slot`.

The view keeps:

- **Binding.** (source entity id, source version, scope): what an observation was admitted under. Entity ids are never
  reused, so a binding never names two sources, even when a 32-bit `source_authority` is.
- **Contribution.** The evidence of one relation under one binding: polarity counts, ticks, admission order and
  provenance. It is authoritative input, never derived.
- **Relation.** A relation key, its canonical participants and its contributions, plus the derived aggregate of its
  LIVE contributions.

Under the synced table A:

```
LIVE_A(c)     <=>  elpis_ecs_c_check_evidence(A, source(c), version(c), scope(c)) == OK
RETIRED_A(c)  <=>  source absent (reclaimed) or TERMINATED, or its version > version(c)
VIEW(A, E)    =    the memory state an ECS_C memory would hold had it OBSERVEd exactly the LIVE observations of E,
                   in admission order; LOOKUP answers from it with the memory's own LOOKUP law
admit(A, b, o): refused unless b passes the admission law on A (in check_evidence's order), o.source_authority is
                the source's, o is valid and its tick is not older than the clock; compare-and-admit on the
                authority coordinate (generation, state root)
sync(A -> A'):  refused unless A' succeeds A; then RETIRED_A' evidence is released, the changed sources'
                contributions are re-evaluated, and only the relations depending on them are recomputed
```

**A successor table** has:
- the same lineage;
- a generation, next founding index, reclaimed count and every source version that never decrease;
- TERMINATED staying TERMINATED, and FOUNDED only as an initial state;
- scope, bound and `source_authority` unchanged at an equal version;
- no row vanishing beyond the reclamations, and no founder vanishing at all;
- no row founded before the synced coordinate appearing later;
- at an equal generation, the identical table (an idempotent no-op).

Anything else is `REGRESSION`, for example a real `RECONCILE_ROLLBACK` that returns a source to an older version. The
view keeps its coordinate until an explicit reset.

**Invariant.** Every retained contribution names its source's present, non-TERMINATED, current version and scope.
Sync releases RETIRED evidence and admission is at the current binding only. `verify` counts violations and `rebuild`
refuses them.

**Consequences.**
- A revision invalidates the old version's evidence for good, and revalidation under a new version admits nothing old.
- DORMANT suspends evidence, and ACTIVE at the same version restores it exactly, because the binding is unchanged.
- Reclamation releases a source's evidence before any new source can reuse its 32-bit identifier.
- A read pinned to a coordinate the view has not reached is `STALE_VIEW`. It never answers for that coordinate with
  superseded evidence.

**Separation.** Only admitted evidence enters. No function takes a candidate, a salience or an inferred relation.
`elpis_ecs_c_rel_export` is an all-or-nothing, read-only copy, and nothing reads back from its consumers.

## Reads, provenance and dependency

| Call | What it gives |
|---|---|
| `elpis_ecs_c_rel_query` | Memory-equivalent LOOKUP over the live view, optionally restricted to the contributions of one scope. |
| `elpis_ecs_c_rel_relation_read` | The live slot. |
| `elpis_ecs_c_rel_support_of` | Which bindings support a relation (LIVE or SUSPENDED, with their counts and provenance). |
| `elpis_ecs_c_rel_dependents` | Which relations depend on a source: what a revocation would touch. |
| `elpis_ecs_c_rel_check` | The admission law (pure). |

`elpis_ecs_c_rel_identity` is the SHA-256 of the coordinate, the table and the retained evidence in key order.

## Resources, refusals and integrity

**Allocation.** The entity rows (≤ 4096), relations (≤ 65 536) and contributions (≤ 262 144) are fixed at create, in
one allocation and none afterwards. `ECS_C.test_ecs_c_relational_alloc`: 1 heap block per view, 0 per call over
2,000 rounds of every call, including refusals.

**Refusals are atomic.** Identity, status and `verify` are unchanged by every refusal tested.

**Capacity is refused, never evicted.** Only RETIRED evidence is released.

**Bounded work.** A sync reads the table: O(rows), with ordering for its uniqueness checks. It then visits only the
changed sources' contributions and recomputes only their relations. Revoking one source with 8 relations costs the
same logical work over 100, 1 000 and 10 000 unrelated relations; a full `rebuild` visits everything.

**Integrity.** Accidental damage, not an adversary.
- Every relation aggregate (with its key, payload and participants) and every contribution carries a 64-bit check.
- A read that meets a failed check refuses `CORRUPT`.
- A sync that recomputes a relation over a damaged contribution poisons it instead of sealing the damage.
- `verify` recomputes every derived value. `rebuild` is the exact full recomputation, and refuses damaged evidence;
  `reset` is then the remedy.

## Tests

**`ECS_C.test_ecs_c_relational`** (the library source compiled in with the test-only damage hook).
- **Configuration and table validation:** 15 malformed tables and a capacity refusal. **Successor laws:** 16 cases
  (15 `REGRESSION` and 1 `LINEAGE`) and an idempotent re-sync.
- **The admission law:** each of its refusals plus 8 malformed observations, with refusal counts checked.
- **Source revision:** VALID → STALE, pinned reads before and after, revalidation, DORMANT suspension and exact
  restoration, TERMINATED, reclamation and identifier reuse.
- **Scopes and n-ary relations:** scoped reads, cross-scope refusal, a scope moving with its version; arity 8, role
  identity, swapped roles, any participant order canonicalized exactly as the memory does it, ordinals, and
  non-LOOKUP channels refused.
- **Shared dependencies:** a revocation visits 5 of 106 relations. **Contradictory evidence:** net 0 withholds an
  answer, and revoking either side decides it the other way.
- **Ordering:** 40 permutations of one tick. Duplicates count again; determinism.
- **Resources:** capacity, the overflow guard, and all-or-nothing outputs.
- **Damage:** five classes.
- **Locality:** across scales.
- **The differential.** 8 seeds × 1 200 steps of random admissions and authority transitions (versions, scopes,
  dormancy, termination, reclamation with identifier reuse, founding, no-op publications). After every sync, every 40
  steps and around every rebuild, every LOOKUP (3 cue shapes × every target role × 24 templates × 3 scope filters),
  every slot and the export count are compared with a fresh real ECS_C memory replaying exactly the LIVE
  observations. Result: 2,274,048 answers, 189,504 slots and 2,632 states, with 0 mismatches. The comparison is not
  vacuous: 75% of the answers name a filler, 49% carry conflicts and 48% contradictions.

**`ECS_C.test_ecs_c_relational_authority`** (production libraries and the real authority).
- The gap above.
- The view's admission law equals `elpis_ecs_c_check_evidence` on 750 probes over 5 authority states.
- A real rollback (records copied back, then `RECONCILE_ROLLBACK`) is refused `REGRESSION` until reset.

**Sanitizers.** Both suites are clean under GCC ASan+UBSan.

**Mutation sensitivity.** 39 targeted mutants of the law, judged by both suites
(`research/ecsc_dynamic_relational_r0/mutants.py`). Three are predicted equivalent: they remove terms of the liveness
and admission laws that the invariant above makes redundant in every reachable state. The outcome is recorded in
`research/ecsc_dynamic_relational_r0/README.md`.

## Not claimed

No qualification, no composition into RuntimeCore, and no RuntimeCore ABI change. The HOT lane still admits by
`source_authority` alone, and a reviewed successor of RuntimeCore's admission would be the place to use this view.

Also not claimed:
- persistence of the view;
- semantic truth: evidence counts are not truth;
- protection against in-process code.

The per-turn bound remains RuntimeCore's.
