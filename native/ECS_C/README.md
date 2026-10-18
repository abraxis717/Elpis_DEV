# ECS_C — continuity and memory

`ECS_C` owns continuity and memory (`docs/ELPIS_MISSION.md`, Amendment 1; `docs/NATIVE_SPLIT_CENSUS.md`). It
depends on the substrate and, until standalone continuity retires into it (Stage F), on the continuity crate its
authority embeds: it links no geometric (ECS_G) code, no RuntimeCore and no codec, and it never performs geometric
learning or mutates ECS_G's state. Only RuntimeCore composes it with ECS_G (`native/runtime/composite`).

| Directory | Contents | Status |
|---|---|---|
| `memory/` | the bounded, model-free typed relational memory (`libelpis_ecs_memory`, `elpis/ecs_memory.h`, ELPISEM1 v1), moved byte for byte from `native/ECS` | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE admitted (mechanical); LEARN / QUERY FAST UNQUALIFIED behind a capability; CONSOLIDATE not provided (`docs/ECS_MEMORY.md`) |
| `authority/` | the ECS_C authority (`libelpis_ecs_c_authority`, `elpis/ecs_c_authority.h`, records ELPECA01 v1, witness ELPECW01 v1; Rust): entities and lifecycles, sender attribution, source versions and scopes, the cognition lineage, the evolution authority, receipts, reconciliation, legacy import (`docs/ECS_C_AUTHORITY.md`) | mechanics qualified as a library (crash, adversary and boundedness suites); not yet composed by RuntimeCore (Stage D) |

Under the census's persistence invariant (section 5a) ECS_C keeps no growing persistence of any kind: the authority
is five files fixed at provisioning, rewritten in place.
