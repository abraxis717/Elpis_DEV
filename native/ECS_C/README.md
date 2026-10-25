# ECS_C — continuity and memory

`ECS_C` owns continuity and memory (`docs/ELPIS_MISSION.md`, Amendments 1 and 2; `docs/NATIVE_SPLIT_CENSUS.md`). It
depends on the substrate only: standalone continuity retired into its authority at RuntimeCore ABI v7, once ECS_C
Recovery Q0 was RUN PASS (for commit `117b41a` only). It links no geometric (ECS_G) code, no RuntimeCore and no codec,
and it never performs geometric learning or mutates ECS_G's state. Only RuntimeCore composes it with ECS_G (`native/runtime/composite`).

| Directory | Contents | Status |
|---|---|---|
| `memory/` | the bounded, model-free typed relational memory (`libelpis_ecs_memory`, `elpis/ecs_memory.h`, ELPISEM1 v1), moved byte for byte from `native/ECS` | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE admitted (mechanical); LEARN / QUERY FAST UNQUALIFIED behind a capability; CONSOLIDATE not provided (`docs/ECS_MEMORY.md`) |
| `relational/` | the ECS_C relational view (`libelpis_ecs_c_relational`, `elpis/ecs_c_relational.h`; ECSC_DYNAMIC_RELATIONAL_DEV_R0): typed evidence per (source, version, scope) binding, synced from the authority's rows, so LOOKUP answers only from evidence the authority's admission law still accepts (`docs/ECS_C_RELATIONAL.md`) | DEV, not composed: nothing canonical links or loads it; session-local, nothing persisted; mechanics, a differential against the memory and a real-authority end-to-end test only |
| `hca/` | the HCA native research substrate (`libelpis_ecs_c_hca`, `elpis/ecs_c_hca.h`): the B0/B1/B2 comparators of `research/ecsc_hca_r0`, ported exactly (`docs/ECS_C_HCA.md`) | RESEARCH ONLY, gated: nothing canonical links or loads it; mechanics and an offline differential only |
| `authority/` | the ECS_C authority (`libelpis_ecs_c_authority`, `elpis/ecs_c_authority.h`, records ELPECA01 v1, witness ELPECW01 v1; Rust): entities and lifecycles, sender attribution, source versions and scopes, the cognition lineage, the evolution authority, receipts, reconciliation, legacy import (`docs/ECS_C_AUTHORITY.md`) | mechanically verified as a library (crash, adversary and boundedness suites); composed by RuntimeCore since ABI v6 (`elpis_runtime_create_ecs_c`), its only durable authority since ABI v7, and run by the native host `elpis`; its durability protocol through the host is RUN PASS under ECS_C Recovery Q0 (`docs/qualification/ECS_C_RECOVERY_Q0.md`) for commit `117b41a` (ABI v6) only |

Under the census's persistence invariant (section 5a) ECS_C keeps no growing persistence of any kind: the authority
is five files fixed at provisioning, rewritten in place.
