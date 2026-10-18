# ECS_C — continuity and memory

`ECS_C` owns continuity and memory (`docs/ELPIS_MISSION.md`, Amendment 1; `docs/NATIVE_SPLIT_CENSUS.md`). It
depends on the substrate only: it links no geometric (ECS_G) code, no RuntimeCore and no codec, and it never
performs geometric learning or mutates ECS_G's state. Only RuntimeCore composes it with ECS_G
(`native/runtime/composite`).

| Directory | Contents | Status |
|---|---|---|
| `memory/` | the bounded, model-free typed relational memory (`libelpis_ecs_memory`, `elpis/ecs_memory.h`, ELPISEM1 v1), moved byte for byte from `native/ECS` | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE admitted (mechanical); LEARN / QUERY FAST UNQUALIFIED behind a capability; CONSOLIDATE not provided (`docs/ECS_MEMORY.md`) |

The continuity authority successor arrives in Stage C of the census. Under its persistence invariant (section 5a)
ECS_C keeps no growing persistence of any kind.
