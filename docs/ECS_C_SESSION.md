# ECS_C volatile session authority (ECSC_VOLATILE_SESSION_R0)

Authority: `docs/ELPIS_MISSION.md`, Amendment 3 (the volatile-native runtime); census
`docs/VOLATILE_RUNTIME_CENSUS.md`. Contract: `native/ECS_C/session/include/elpis/ecs_c_session.h` (normative C ABI v1)
and `include/elpis/ecs_c_types.h` (the shared entity, evolution, kind, lifecycle and code definitions).
Implementation: `native/ECS_C/session` (Rust crate `elpis_ecs_c_session`, `no_std`; library
`libelpis_ecs_c_session.so`).

**VOLATILE MIGRATION: INCOMPLETE.** The session is built and tested, and the relational view's end-to-end test runs
over it (phase V2). Nothing composes it yet. RuntimeCore ABI v7, the native host and the Python reference interface
still hold the durable authority (`docs/ECS_C_AUTHORITY.md`, RETIRING). Phase V3 replaces it there.

## What it is

The ECS_C authority's logic as bounded state in volatile memory only. It covers:
- entity identity and lifecycle;
- sender attribution and per-sender ordering;
- source identity, version and scope;
- typed-evidence admission and per-source observation bounds;
- the cognition lineage and the evolution authority;
- receipts.

It runs the **same transition law** as the durable authority. `model.rs` (with `register.rs`, `sha256.rs` and
`error.rs`) moved into this crate unchanged. The durable crate now consumes the law from here: it has one
implementation, not a copy. What the session does not have:
- records, witness slots or a deployment key file;
- provisioning or reconciliation;
- legacy import, recovery or resumption;
- a path, directory or file descriptor anywhere in its ABI.

A session:
- starts at a fresh genesis: the founding operator and runtime, both ACTIVE, at generation 1;
- evolves only by compare-and-publish transitions;
- is gone when it is destroyed or its process ends.

A new session is a new genesis.

## State model

| Item | Content |
|---|---|
| Lineage (`session`) | The ECS_C genesis law's lineage digest over the deployment, the nonce and the capacity. The deployment is the session constant `SHA-256("elpis.ecs_c.session.deployment.v1\0")`, where the durable authority uses its provisioned deployment key. A zero nonce draws 32 bytes from getrandom(2); a refused draw is `ENTROPY_UNAVAILABLE` and nothing is created |
| Instance | Process-unique, never 0, from a static counter. It distinguishes two sessions even if a caller reuses a nonce |
| Generation, state root | Generation 1 at genesis; +1 per accepted transition. The state root is computed exactly as the durable authority computes it, over the session's encoded state |
| Commit | `SHA-256("elpis.ecs_c.session.commit.v1\0", lineage, generation, state root, receipt)`. The in-memory digest of a generation, and the predecessor digest of the next receipt (the durable authority used its record checksum here) |
| Live state | Entities (at most `capacity` rows, 2..4096), the cognition lineage, publications, the evolution authority, and the receipt, evidence and reclamation chains |
| Staging state | A second entity buffer of `capacity` rows, which transitions are applied to |
| Encoding buffer | `state_bytes(capacity)` bytes, which the state root is computed from |
| Counters | Accepted and refused transitions (telemetry, not state) |

**Footprint.** `elpis_ecs_c_session_bytes(capacity)` is exact and constant for a session's life: 1,424 bytes fixed
plus 464 bytes per row (two 152-byte entity rows and 160 encoded bytes). It is reserved at creation.

| Capacity | Bytes |
|---:|---:|
| 2 | 2,352 |
| 16 | 8,848 |
| 64 | 31,120 |
| 256 | 120,208 |
| 1,024 | 476,560 |
| 4,096 | 1,901,968 |

## Transitions

`elpis_ecs_c_session_transition(session, expected, event[512], out)`, in this order:

1. **Kind.**
   - `GENESIS` is `INVALID`.
   - `IMPORT_LEGACY` (10) and `RECONCILE_ROLLBACK` / `RECONCILE_UNAUTHENTICATED` / `RECONCILE_REPIN` (11..13) are
     `PERSISTENCE_RETIRED`, even when well formed. They existed only to adopt or repair durable records.
   - A malformed event is `INVALID` without counting a refusal.
2. **Session.** `expected.session` and `expected.instance` must name this session; otherwise `SESSION_MISMATCH`. A
   stale handle, coordinate or receipt from another session is never accepted as this session's.
3. **Coordinate.** `expected.generation` and `expected.state_root` must be current; otherwise `STALE`.
4. **Stage.** The live state is copied into the preallocated staging buffer.
5. **Law.** The transition law is applied to the copy (`model::apply_into`). This checks identity, scope, version,
   permissions, lifecycle, ordering, capacity and invariants, and any of the law's codes refuses.
6. **Root.** The copy is encoded into the preallocated buffer, and the state root and commit are computed.
7. **Publish.** The live and staging buffers are swapped. Nothing after the swap can fail.

A refusal at any step leaves the state, its coordinate and every counter except the refusal count exactly as they
were. No transition allocates. Exhaustion (`CAPACITY`, `EXHAUSTED`) refuses before anything changes. Memory never
spills anywhere.

Reads are pure:
- `read_coordinate` and `read_status`;
- `entity_at`, `entity_find` and `source_for_authority`;
- `next_entity_id` (the receiver an `ENTITY_FOUND` must name);
- `check_evidence`, which is typed-evidence admission: the per-turn bound, or `UNKNOWN_ENTITY`, `UNAUTHORIZED`,
  `LIFECYCLE`, `SOURCE_STALE` or `SCOPE_CONFLICT`.

`receipt_chain` and `evolution_digest` are pure functions, which callers use to verify the receipts they keep.

`destroy` clears the encoding buffer, which holds the last state image, before the memory is freed.

## Codes

The law's codes 0..18 and `CORRUPT` (28) keep their values. The session adds four codes:
- `SESSION_MISMATCH` (34);
- `PERSISTENCE_RETIRED` (35);
- `NOMEM` (36): creation could not reserve memory. Nothing is created, and the process never aborts.
- `ENTROPY_UNAVAILABLE` (37).

The durable authority's storage codes (19..27, 29..33) are never returned. `elpis_ecs_c_session_code_name` returns
NULL for them.

## What is not claimed

- No persistence, crash survival, resumption or exactly-once behaviour across restarts. A crash loses the session.
- No persistent authentication, rollback resistance or cross-restart monotonicity.
- The lineage and the commit identify a session's state within its life. They authenticate nothing to anyone
  outside the process.
- No protection against code inside the process, which can read or corrupt the session's memory.
- The session is not swap-proof by itself. Whether its pages can reach swap is a host property (swap disabled, or
  the process locked into memory). The census, section 6, lists what the host must provide.
- No qualification. The tests and the zero-write evidence (`research/volatile_runtime_v0`) cover the session's own
  test paths only, not a composed runtime.

## Tests

- `ECS_C.cargo_test_session` (`native/ECS_C/session/src/tests.rs`, 12 tests with the C ABI). Among them:
  - a differential against the pure law: 6 seeds × 6,000 events. Every accepted transition yields exactly the pure
    law's state; every refusal yields the same code (or `PERSISTENCE_RETIRED`) and leaves the session unchanged;
  - persistence-only kinds refused even when well formed;
  - foreign and stale coordinates;
  - capacity exhaustion and the reuse of reclaimed rows;
  - evidence admission across versions, scopes and lifecycles;
  - the cognition and evolution coordinates;
  - zero allocations over 20,000 transitions;
  - a 100,000-event run with a constant footprint;
  - sessions that share nothing.
- `ECS_C.test_ecs_c_session_abi` (the C ABI against the built library):
  - names, sizes and malformed creation;
  - founding, sources, admission and revision;
  - anchoring, publication and evolution;
  - every refusal class, all atomic, including `INVALID` (left uncounted) and `STALE`;
  - the four retired kinds, `CAPACITY` and `SESSION_MISMATCH` in both directions;
  - the receipt chain;
  - a 100,000-event run with constant heap;
  - 1,000 create/destroy cycles with zero heap growth.
- `ECS_C.test_ecs_c_relational_session`: the relational view (`docs/ECS_C_RELATIONAL.md`) over the session.
- `tests/boundary/test_volatile_runtime.py`:
  - checks the crate's source: `no_std`, no dependency, no build script, and an extern surface of malloc, free,
    posix_memalign, abort and getrandom;
  - checks the header: no path or descriptor;
  - checks the built library's imports: memory, abort and getrandom only;
  - runs both C test binaries under strace and requires no write-class system call on any file.
