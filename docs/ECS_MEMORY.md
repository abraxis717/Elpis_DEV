# ECS memory (canonical)

Authority: `docs/ELPIS_MISSION.md` (one ECS) and `ELPIS_SYSTEM.json` (subsystem `ECS`). Contract:
`native/ECS/include/elpis/ecs_memory.h` (normative). Library: `libelpis_ecs_memory` (`native/ECS/src/ecs_memory.c`).

The ECS memory is the bounded, model-free typed relational memory of the one ECS. It is the MF-R0 mechanism
(`research/ecs_memory_mf_r0`, `docs/research/ECS_MEMORY_MF_R0.md`) relocated under `native/ECS` with the canonical
prefix (`elpis_ecs_mem_*`, `ELPIS_ECS_MEM_*`), capability gating, and **without the MF-R0 CONSOLIDATE (slow-bank)
law**. It is a **component** of the ECS retained
state, never an authority of its own: durable authority over it exists only through the composite retained state
(`docs/ECS_COMPOSITE_STATE.md`). A managed turn that observes it (a LEARN turn, or an observe-only turn: RuntimeCore ABI v5) commits volatile; the memory
becomes durable only with an explicit durable publication of the composite. It is not the contextual (WARM) lane's
active context (`elpis.runtime.context`), which is session-local, never persisted and holds references, not memory
slots.

## Relation to the research mechanism

| | research MF-R0 (`ecs_mem_*`) | canonical (`elpis_ecs_mem_*`) |
|---|---|---|
| evidence slots, codebook, OBSERVE, LOOKUP, LEARN / FAST laws, encoding, numerical bound | as documented in `docs/research/ECS_MEMORY_MF_R0.md` (M2) | **identical** |
| CONSOLIDATE (law C1), QUERY SLOW | present | **absent**: SLOW is refused UNQUALIFIED; slow banks and cursor always empty |
| envelope and identity | ELPISEM1 v1 | **identical bytes** (no new format) |
| capability gating | none (every law always available) | LEARN / FAST, CONSOLIDATE / SLOW behind explicit capabilities (below) |
| status space | 10 refusals | the same, plus `ELPIS_ECS_MEM_UNQUALIFIED` (-11) |
| config word at offset 44 | `reserved` (must be 0) | `capabilities` (policy, never serialized) |
| slow-bank rate (serialized config word) | `consolidate_rate_q16` | `slow_rate_q16` (same word, same bytes; unused) |
| restore | `ecs_mem_restore(bytes, n, budget, out)` | `elpis_ecs_mem_restore(bytes, n, budget, capabilities, out)` |
| status | RESEARCH_ONLY; frozen as the independent reference | canonical ECS capability |

The research tree stays byte-frozen and is never linked by `native/` or `src/`. The canonical library never
includes a research header. Equivalence is a test, not an assumption: `research.ecs_memory_canonical_differential`
(`research/ecs_memory_mf_r0/integration/`) applies seeded random sequences of OBSERVE, LEARN, QUERY (every channel)
and cross-implementation restores to both, and requires every status, answer, work count, identity and ELPISEM1
byte to agree with RECURRENT admitted; with none admitted it requires agreement on every admitted operation and
UNQUALIFIED refusals with unchanged identity; SLOW is UNQUALIFIED canonically in both arms. The canonical
known-answer identity (`9b0e06c3…eb8162fc`, the MF-R0 mechanics script without its CONSOLIDATE calls) is
reproduced by the research reference running the same script.

### Why CONSOLIDATE is not canonical

The closed Retention R1, R2 and R3 laboratories froze canonical-vocabulary guards
(`tests/research/ecs_retention_r{1,2,3}/test_preregistration.py`): no canonical ECS source may carry a
retention mechanism (consolidation, protection, reconditioning, rehearsal) unless it is admitted by a qualified
promotion record, and the only admitted one is the K1 milestone (`tests/research/_k1_promotion.py`). The MF-R0
slow-bank law is a consolidation mechanism with no such qualification (its H5 passed only for recurring
relations and below exact storage). Admitting it would mean altering frozen authority, so the canonical memory
leaves it out entirely rather than shipping it behind a switch. Its ELPISEM1 fields (slow banks, cursor, rate)
remain in the format so that the bytes and identities stay those of the reference; a canonical handle refuses
any envelope whose slow banks are nonzero or whose cursor moved (UNQUALIFIED). A future slow-bank capability is a
new, separately preregistered and qualified experiment.

## Capability status

Capabilities are host policy (`elpis_ecs_mem_config.capabilities`), not state: they are never serialized and
never part of an identity.

| Operation | Status | Basis |
|---|---|---|
| OBSERVE (exact typed evidence slots, entity codebook) | **ADMITTED** (mechanical) | contract, refusal atomicity, capacity, determinism, allocation, snapshot fault tests |
| QUERY LOOKUP (exact read-only scan of resident evidence) | **ADMITTED** (mechanical) | as above; ordinary exact lookup, no learned content |
| INSPECT, slot read, identity, SNAPSHOT / RESTORE | **ADMITTED** (mechanical) | ELPISEM1 v1 validation suite |
| LEARN, QUERY FAST | **UNQUALIFIED** — `ELPIS_ECS_MEM_CAP_RECURRENT` | MF-R0 QUAL: H1–H3 PASS in one synthetic regime; **H4 NONPASS**: byte-matched exact storage retains more. Not a scientific qualification of a cognitive memory |
| CONSOLIDATE, QUERY SLOW | **NOT PROVIDED** (SLOW refused UNQUALIFIED; no CONSOLIDATE entry point) | see "Why CONSOLIDATE is not canonical" |

Without the capability the operation is refused `ELPIS_ECS_MEM_UNQUALIFIED` before anything is read or written:
state, identity, outputs and authoritative counters are unchanged (only telemetry counts the refusal). A handle
also refuses to **restore** content it does not admit: an envelope with any nonzero fast-bank weight needs
RECURRENT; any nonzero slow-bank weight or a moved slow cursor is never admitted (`restore_into` and `restore`
return UNQUALIFIED after the integrity checks; a corrupt envelope is CORRUPT, never UNQUALIFIED). Unqualified learned content therefore
cannot enter through restore, and it is never silently dropped.

"ADMITTED (mechanical)" means the operation's mechanics are tested and it may be used as exact storage of typed
structural evidence. It does **not** mean that observing a record makes it true, authorized or learned, and it
is not a scientific claim of any kind.

## HACF observations

`libelpis_hacf_observation` (`native/runtime/observation`, runtime subsystem, header
`elpis/ecs_observation_hacf.h`) maps one structure hyperedge (`elpis_semantic_hyperedge_v1`) and one assertion
that names it to one memory observation, and inverts the map from a resident slot. It is the MF-R0 adapter
under the canonical prefix, unchanged in law (`docs/research/ECS_MEMORY_MF_R0.md`, "Typed HACF adapter"). It lives
in the runtime subsystem because ECS depends only on the substrate and structure only on the substrate; the
composition layer may use both. Structure stays the owner of identity and validation: the adapter calls
structure's validators and identity functions and refuses anything whose stored identity is not the recomputed
one. Provenance (first and last), authority, polarity (supplied by the caller; HACF does not define it),
uncertainty (`UNCERTAIN` observations are counted, never learned from) and the exact hyperedge identity are
preserved; `test_runtime_composite` carries an asserted-and-negated hyperedge through a managed turn (committed
volatile, then published durably) and recovers the byte-identical hyperedge from the committed memory, whose LOOKUP
then reports the conflict instead of an answer.
Mapping or observing never makes a record true, authorized or learned.

From Python the same path is `elpis.runtime.observation.HacfObservationLibrary` over the admitted adapter (library
identifier `elpis_hacf_observation`): structure's own native functions canonicalize participants and compute and
validate every hyperedge and assertion identity (Python hashes nothing), the adapter maps the record, and the
observation enters the composite memory only through `Runtime.run_learn(..., observations=...,
observation_authority=...)` (docs/ECS_COMPOSITE_STATE.md, "Python composition"); `hyperedge_of` inverts a resident
slot. `tests/integration/test_hacf_composite_observation.py` carries an asserted, negated, uncertain and re-asserted
hyperedge through managed turns, a recoverable publication and a restart, and recovers the identical slot and
hyperedge; it refuses a tampered identity, a changed provenance, an assertion of another hyperedge, an authority
outside structure's range or outside the `ObservationAuthority`, undefined flags, query-local, too many and repeated
participants, a self-consistent noncanonical record, a failed turn and an unmanaged mutation, and shows that the same
LEARN with and without an observation leaves identical K1 components. This is mechanical transport and preservation
only: no semantic interpretation, no qualified HACF -> ECS observation map and no cognitive use is established.

## Resources

One allocation per handle (the handle and one 64-byte-aligned arena) at create/restore; none afterwards
(`ECS.test_ecs_memory_alloc`, link-time allocator wrapping, uninstrumented builds). Every size is computed with
checked arithmetic before the allocation; the arena must fit the declared `byte_budget` and
`ELPIS_ECS_MEM_MAX_ARENA_BYTES` (1 GiB). `elpis_ecs_mem_required_bytes` reports the arena and the largest
ELPISEM1 size of a configuration exactly. `elpis_ecs_mem_inspect` reports `state_bytes`, `arena_bytes` and
`allocated_bytes` (allocator overhead and bounded stack staging excluded) and per-code refusal counts.

## Concurrency

One writer XOR any number of readers per handle, non-blocking (`BUSY` on conflict), with no fairness: readers can
starve a writer. Distinct handles share nothing. Inside a composite the memory handles are private and only the
composite's own guard is ever contended.

## Tests (`native/ECS`, every native job, ASan+UBSan and TSan)

`ECS.test_ecs_memory` (mechanics; known-answer identity), `ECS.test_ecs_memory_capability` (refusals, restore
admission), `ECS.test_ecs_memory_snapshot` (ELPISEM1 faults), `ECS.test_ecs_memory_concurrency`,
`ECS.test_ecs_memory_alloc`; `research.ecs_memory_canonical_differential` (canonical ≡ research on the shared
laws).

## Non-claims

No semantics, meaning or understanding; no learned similarity or generalization to unseen entities; no claim that
the recurrent memory is better than exact storage at equal bytes (H4 says the opposite, and stays frozen); no
continual-learning solution; no observation map from HACF structure to meaning. Canonical placement is
mechanical integration, not scientific qualification.
