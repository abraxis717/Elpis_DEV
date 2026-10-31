# ECS_C — continuity and memory

`ECS_C` owns continuity and memory (`docs/ELPIS_MISSION.md`, Amendments 1 and 2; `docs/NATIVE_SPLIT_CENSUS.md`). It
depends on the substrate only: standalone continuity retired into its authority at RuntimeCore ABI v7, once ECS_C
Recovery Q0 was RUN PASS (for commit `117b41a` only). It links no geometric (ECS_G) code, no RuntimeCore and no codec,
and it never performs geometric learning or mutates ECS_G's state. Only RuntimeCore composes it with ECS_G (`native/runtime/composite`).

**VOLATILE MIGRATION: INCOMPLETE** (`docs/ELPIS_MISSION.md`, Amendment 3; `docs/VOLATILE_RUNTIME_CENSUS.md`).
ECS_C's lineage and authority move into volatile memory (`session/`). The durable authority (`authority/`) is
RETIRING: it remains only while RuntimeCore ABI v7 reaches it.

| Directory | Contents | Status |
|---|---|---|
| `session/` | the volatile session authority (`libelpis_ecs_c_session`, `elpis/ecs_c_session.h` and the shared `elpis/ecs_c_types.h`; Rust, `no_std`; ECSC_VOLATILE_SESSION_R0). It owns the ECS_C transition law (`model.rs`, `register.rs`, `sha256.rs`, `error.rs`) and runs it over bounded RAM-only session state: staged all-or-nothing transitions, session-scoped coordinates, no path, file or descriptor; legacy import and reconciliation are refused `PERSISTENCE_RETIRED` (`docs/ECS_C_SESSION.md`) | built and tested (a differential against the law, allocation-free transitions, C ABI, zero filesystem writes on its test paths); not composed: RuntimeCore ABI v8 (phase V3) is to hold it |
| `memory/` | the bounded, model-free typed relational memory (`libelpis_ecs_memory`, `elpis/ecs_memory.h`, ELPISEM1 v1), moved byte for byte from `native/ECS` | OBSERVE, LOOKUP, INSPECT, SNAPSHOT/RESTORE admitted (mechanical); LEARN / QUERY FAST UNQUALIFIED behind a capability; CONSOLIDATE not provided (`docs/ECS_MEMORY.md`) |
| `relational/` | the ECS_C relational view (`libelpis_ecs_c_relational`, `elpis/ecs_c_relational.h`; ECSC_DYNAMIC_RELATIONAL_DEV_R0): typed evidence per (source, version, scope) binding, synced from an ECS_C authority's rows, so LOOKUP answers only from evidence the authority's admission law still accepts (`docs/ECS_C_RELATIONAL.md`) | DEV, not composed: nothing canonical links or loads it; session-local, nothing persisted; mechanics, a differential against the memory and an end-to-end test over the volatile session only |
| `hca/` | the HCA native research substrate (`libelpis_ecs_c_hca`, `elpis/ecs_c_hca.h`): the B0/B1/B2 comparators of `research/ecsc_hca_r0`, ported exactly (`docs/ECS_C_HCA.md`) | RESEARCH ONLY, gated: nothing canonical links or loads it; mechanics and an offline differential only |
| `authority/` | RETIRING: the durable ECS_C authority (`libelpis_ecs_c_authority`, `elpis/ecs_c_authority.h`, records ELPECA01 v1, witness ELPECW01 v1; Rust): entities and lifecycles, sender attribution, source versions and scopes, the cognition lineage, the evolution authority, receipts, reconciliation, legacy import (`docs/ECS_C_AUTHORITY.md`); it consumes the law from `session/` | mechanically verified as a library (crash, adversary and boundedness suites); composed by RuntimeCore since ABI v6 (`elpis_runtime_create_ecs_c`), its only durable authority since ABI v7, and run by the native host `elpis`; its durability protocol through the host is RUN PASS under ECS_C Recovery Q0 (`docs/qualification/ECS_C_RECOVERY_Q0.md`) for commit `117b41a` (ABI v6) only |

Historically, under the split census's persistence invariant (section 5a, superseded by Amendment 3), ECS_C kept no
growing persistence: the durable authority is five files fixed at provisioning, rewritten in place. Under the
volatile-native directive ECS_C writes nothing at all; the durable authority retires in phase V3.
