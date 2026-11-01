# ECS_C — continuity and memory

`ECS_C` owns continuity and memory (`docs/ELPIS_MISSION.md`, Amendments 1 to 3; `docs/NATIVE_SPLIT_CENSUS.md`). It
depends on the substrate only. It links no geometric (ECS_G) code, no RuntimeCore and no codec, and it never performs
geometric learning or mutates ECS_G's state. Only RuntimeCore composes it with ECS_G (`native/runtime/composite`).

**VOLATILE MIGRATION: COMPLETE** (`docs/ELPIS_MISSION.md`, Amendment 3; `docs/VOLATILE_RUNTIME_CENSUS.md`). ECS_C's
lineage and authority live in volatile memory (`session/`), which RuntimeCore ABI v8 holds. The durable authority
(`authority/`) is retired: its source, build target and libraries are deleted, and git history keeps them
(`docs/ECS_C_AUTHORITY.md`, history).

| Directory | Contents | Status |
|---|---|---|
| `session/` | the volatile session authority (`libelpis_ecs_c_session`, `elpis/ecs_c_session.h` and the shared `elpis/ecs_c_types.h`; Rust, `no_std`; ECSC_VOLATILE_SESSION_R0). It owns the ECS_C transition law (`model.rs`, `register.rs`, `sha256.rs`, `error.rs`) and runs it over bounded RAM-only session state: staged all-or-nothing transitions, session-scoped coordinates, no path, file or descriptor; legacy import and reconciliation are refused `PERSISTENCE_RETIRED` (`docs/ECS_C_SESSION.md`) | composed: RuntimeCore ABI v8 holds one session per open runtime, on the native host and the Python reference; built and tested (a differential against the law, allocation-free transitions, C ABI), with zero filesystem writes on its own test paths (`research/volatile_runtime_v0`) and in the composed runtime (`research/volatile_runtime_v1`) |
| `memory/` | the bounded, model-free typed relational memory (`libelpis_ecs_memory`, `elpis/ecs_memory.h`, ELPISEM1 v1), moved byte for byte from `native/ECS` | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE admitted (mechanical); LEARN / QUERY FAST UNQUALIFIED behind a capability; CONSOLIDATE not provided (`docs/ECS_MEMORY.md`) |
| `relational/` | the ECS_C relational view (`libelpis_ecs_c_relational`, `elpis/ecs_c_relational.h`; ECSC_DYNAMIC_RELATIONAL_DEV_R0): typed evidence per (source, version, scope) binding, synced from the session's rows, so LOOKUP answers only from evidence the session's admission law still accepts (`docs/ECS_C_RELATIONAL.md`) | DEV, not composed: nothing canonical links or loads it; session-local, nothing persisted; mechanics, a differential against the memory, and end-to-end tests over the volatile session and RuntimeCore's session rows only |
| `hca/` | the HCA native research substrate (`libelpis_ecs_c_hca`, `elpis/ecs_c_hca.h`): the B0/B1/B2 comparators of `research/ecsc_hca_r0`, ported exactly (`docs/ECS_C_HCA.md`) | RESEARCH ONLY, gated: nothing canonical links or loads it; mechanics and an offline differential only |

Historically, under the split census's persistence invariant (section 5a, superseded by Amendment 3), ECS_C kept no
growing persistence: the durable authority (`authority/`, retired at RuntimeCore ABI v8) was five files fixed at
provisioning, rewritten in place, and its durability protocol is RUN PASS under ECS_C Recovery Q0 for commit
`117b41a` (ABI v6) only. Under the volatile-native directive ECS_C writes nothing at all.
