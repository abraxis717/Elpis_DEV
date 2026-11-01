# RuntimeCore: the runtime's systems authority in Rust

> **VOLATILE MIGRATION: COMPLETE** (`docs/ELPIS_MISSION.md`, Amendment 3; `docs/VOLATILE_RUNTIME_CENSUS.md`).
> RuntimeCore ABI v8, specified here, holds one volatile ECS_C session per open runtime and has no persistent-state
> write path: no file, directory, checkpoint, journal, publication, durability lane, recovery or resumption exists in
> its ABI. The persistent v7 interface is retired; its specification is history
> (`docs/history/RUNTIME_CORE_ABI_V7.md`), not a target.

`native/runtime` (crate `elpis_runtime`, C ABI `include/elpis/runtime.h` v8) owns the mutable systems authority of an
open Elpis runtime. It is the one native composer of:
- ECS_G: K1, reached through function tables;
- ECS_C: the memory, through the composite table, and the volatile ECS_C session (`docs/ECS_C_SESSION.md`), embedded
  without its C ABI as the runtime's only authority.

Its production caller is the native host `elpis` (`docs/NATIVE_HOST.md`). `elpis.runtime.Runtime` is the offline
reference interface's facade over the same library (`docs/NATIVE_SPLIT_CENSUS.md`, F3).

Everything RuntimeCore owns lives in memory: its lifecycle and fail-stop disposition, its session, the binding of the
K1 lineage to one native state and its live coordinate, the managed turn's native transaction, and the evolution
attempt's reservation and finalization. It owns no semantics: callers hand it a validated native-ready stimulus or
an assertion digest.

ABI history:
- v2: the turn lifecycle;
- v3: managed ownership;
- v4: the live coordinate, with volatile commits;
- v5: the observe-only managed turn;
- v6 and v7: the durable ECS_C authority;
- **v8: one volatile ECS_C session per open, and nothing persistent.**

Each version is refused by a caller of another (`elpis_runtime_abi_version`), so a v7 caller cannot mistake a v8
library for a durable runtime.

## Sessions (ABI v8)

```text
  elpis_runtime_create(config)  ->  CLOSED  --open-->  OPEN (session n, a fresh genesis)  --close-->  CLOSED
                                       ^                     |  anchor, QUERY, LEARN, OBSERVE,             |
                                       |                     |  operator events, evolution                 |
                                       +---------------------+-------------- (session n is gone) ----------+
  elpis_runtime_destroy: aborts an open turn natively, ends the session, frees the handle
```

- **Create.** `elpis_runtime_create(const elpis_runtime_config *, elpis_runtime **)` takes a configuration, never a
  path:
  - the session's ECS_C entity capacity (2..4096; the founders occupy two rows);
  - the founding operator's and runtime's label digests;
  - an optional 32-byte nonce seed.

  `RUNTIME_INVALID` for a NULL argument, another `abi_version` or a capacity outside the bounds. No session exists
  until `open`.
- **Open.** Each `elpis_runtime_open` starts a new session at a fresh genesis: the founding operator and runtime,
  both ACTIVE, at generation 1, with nothing anchored. Two failures leave the runtime closed:
  - `RUNTIME_NOMEM` if the session's memory cannot be reserved;
  - `RUNTIME_AUTHORITY` if no nonce can be drawn.

  An open runtime refuses another open (`CONTINUITY_OPEN`) and keeps any open turn.
- **Nonce.** A zero seed draws each session's nonce from the kernel (`getrandom(2)`). A nonzero seed derives the n-th
  session's nonce as `SHA-256("elpis.runtime.session-nonce.v1\0" || seed || n, u64 LE)`. Runtimes configured alike
  then open identical sessions, which differential testing uses, while one runtime's successive sessions differ.
- **Close.** `elpis_runtime_close` aborts an open turn natively, drops the residency hold and the binding, and ends
  the session together with everything in it: the anchor, the operator's sources and events, evidence and receipt
  chains, and any evolution reservation. Close then open is the only way to clear a fail-stop, and it clears the
  session that the fail-stop belonged to.
- **Restart.** A restart is a new session. Nothing is resumed. No crash-persistent identity, recovery, or
  exactly-once guarantee across restarts is claimed. A crash loses the session, which is the specified behavior,
  not a failure mode to recover from.
- **Footprint.** The session's memory is reserved once, at open. `elpis_runtime_authority.volatile_bytes` reports
  it, and it is exact and constant for the session's life: `1,424 + 464 × capacity` bytes, for example 8,848 at
  capacity 16 (`docs/ECS_C_SESSION.md`, "Footprint"). No transition allocates. Exhaustion refuses before anything
  changes, and nothing spills anywhere.

`elpis_runtime_authority_status` reports the session, and is readable while the runtime is closed. Closed, it reports
the number of sessions opened and the session code of the last refused open, with everything else zero. Open, it
reports:
- the backend: `ELPIS_RUNTIME_BACKEND_ECS_C_SESSION`, 3;
- the instance, the generation, the entity count and the capacity;
- the lineage, the commit, the state root and the receipt and evidence heads;
- `volatile_bytes`.

## The live coordinate

The bound state's native generation (ecsg_k1.h, "Live coordinate") under RuntimeCore's lease, as RuntimeCore last
established it. It advances by exactly one per managed commit. QUERY, the managed turn and OBSERVE are checked against
it inside the same guarded K1 call. None of them performs I/O or computes a digest.

The session records the lineage's **anchored** identity: the retained-state identity of the state that the explicit
anchor bound. A managed commit installs the candidate and advances the live generation, in memory. The session never
needs to name the new state, and nothing ever publishes it. ABI v4 to v7 kept a second, durable coordinate beside
this one. ABI v8 has no durable coordinate: the live coordinate and the session's anchor are all there is.

## What RuntimeCore owns

| Authority | RuntimeCore (Rust), ABI v8 |
|---|---|
| Lifecycle (create / open / close / destroy) | `Core::open` / `close`. Each open is a new session at a fresh genesis (`authority.rs`, `SessionTemplate`) |
| Fail-stop disposition | `Core.fault`, a session-authority code. Every lineage- or session-dependent call refuses with it until close and open |
| Authority | One volatile ECS_C session per open (`authority.rs`), embedded without its C ABI, with one transition law (`native/ECS_C/session`) |
| K1 lineage binding | `Core.bound` (`Key`: kind, native handle, resident id, owner identity, dimension) and the live generation. Verified once against the session's anchored identity, or against the live identity that an explicit release gave back |
| Managed QUERY (read-only) | `query`: one native `leased_query`, which returns the answer `f_W(x)` and the generation of the state that computed it under one K1 guard. No transaction, commit, publication or digest |
| Managed LEARN (the turn's native transaction) | `turn_begin`: the lineage check at the live generation, begin, one schedule, and cold-path capacity growth. Then `turn_commit` (volatile) or `turn_abort` |
| Typed-evidence admission | The admission table: the session's ACTIVE sources, with their per-turn bounds. A table lookup on the HOT lane |
| Evolution reservation / finalization | `evolution_reserve` / `evolution_finalize` / `evolution_abandon`: session transitions, with one attempt in flight |

What stays at the language boundary, because it is semantics or validation rather than mutable authority:
- the token codec and the (unqualified) ECS codec map: `cognition._validate_turn_request`, `_encode` (text to tokens
  to a validated native-ready `Stimulus`) and `_decode` (readout to tokens to text);
- the evolution gate's validation and execution (`EvolutionPathGate.reject_reason` / `execute`): immutable values and
  digest identities, with no mutable state;
- the reference that keeps the bound K1 state's Python owner alive while RuntimeCore holds the binding. This is
  lifetime only: RuntimeCore decides the binding.

No ECS<->DSV semantic codec is defined, and RuntimeCore adds none. The turn still refuses with `ECS_CODEC_UNQUALIFIED`
before anything else.

## How RuntimeCore reaches K1

RuntimeCore links no ECS code and computes no ECS mathematics. The caller passes a substrate descriptor:

```c
typedef struct {
    uint32_t kind;          /* 1: standalone K1 (ecsg_k1.h); 2: FMS-resident K1 (ecsg_k1_fms.h); 3: composite */
    uint32_t reserved;
    void *handle;           /* elpis_ecsg_k1 *, elpis_ecsg_k1_fms * or elpis_ecs_composite * */
    uint64_t id;            /* resident state id (kind 2) */
    uint64_t owner;         /* caller identity of the handle's owner (unique while bound) */
    uint64_t dim;           /* verified natively */
    const void *api;        /* the library's own entry points */
} elpis_runtime_substrate;
```

`api` is a table of the K1 library's own functions:
- `state_digest`, `shape`;
- `lease_claim`, `lease_release`, `leased_reserve`, `leased_identity`, `leased_query`;
- `leased_txn_begin_at`, `leased_txn_run_schedule`, `leased_txn_commit`, `txn_abort`.

The K1 FMS adapter's table adds `resident_hold` and `resident_release`; the composite's adds `leased_txn_observe`.
ABI v8 removed `leased_capture` and `envelope_seal` from the tables: they produced the envelope that durable
publication wrote. The Python adapter takes the entries' addresses from the loaded library; a C host passes them
directly (`native/runtime/tests/k1_tables.h`).

The declared dimension is verified natively: K1 compares the readout length it implies with the state's own before it
reads any input byte (`schedule_check`).

### Substrate lifetime (ABI v2)

Outside a managed turn, a descriptor need only be live for the call. A successful `turn_begin` opens a native K1
transaction. To end it on every path, RuntimeCore retains an abort-only capability over the state until the turn ends:
a copy of the descriptor's handle, resident id and `txn_abort` entry, and never commit, schedule or any other entry.
The native state must stay live, and its library loaded, for that interval.

- **FMS-resident K1:** enforced natively by the adapter (`ecsg_k1_fms.h`). A resident state with an open transaction
  cannot be closed, and its runtime cannot be destroyed while a state is registered (both `BUSY`).
- **Standalone K1:** `elpis_ecsg_k1_destroy` frees a state whatever transaction it holds, so the owner must not destroy
  it while a turn on it is open. The Python adapter satisfies this: native K1 handles are freed only by an explicit
  `close`, and `Runtime` keeps the bound state's owner alive until RuntimeCore has ended the turn.

### Composite substrates

A substrate of kind `ELPIS_RUNTIME_SUBSTRATE_COMPOSITE` (3) is an ECS composite retained state
(`docs/ECS_COMPOSITE_STATE.md`): K1 and the ECS memory under one `ELPISEC1` identity. Its table,
`elpis_runtime_composite_api`, is the K1 table over composite identities plus `leased_txn_observe`. RuntimeCore binds,
queries and learns it exactly as a K1 state.

The session records the lineage's format. A K1 state never stands for a composite lineage, or the reverse:
`CONTINUITY_STATE_MISMATCH`, which fail-stops on binding as an identity mismatch does.

- `elpis_runtime_turn_observe` records an opaque `elpis_ecs_mem_observation` on the open turn's memory candidate.
  RuntimeCore reads only its source. The memory therefore changes only inside a managed turn and becomes live with
  that turn's native commit, in memory. A refusal is recoverable (`ECS_REFUSED`, with the composite and memory
  statuses reported), and the turn stays open.
- No memory LEARN or CONSOLIDATE is exposed: those operations are UNQUALIFIED (`docs/ECS_MEMORY.md`).
- The v7 legacy import (`elpis_runtime_import_composite`), which moved a durable K1 lineage to a composite, is
  retired. A composite lineage is anchored directly.

`native/runtime/tests/test_runtime_composite.c` covers:
- the composite anchor;
- volatile turns with observations, and their refusals;
- unmanaged mutation refused while bound, and the format mismatch;
- an in-memory rollback: a snapshot buffer restored to the anchored state and rebound, with the volatile work
  abandoned;
- the typed HACF path;
- the observe-only turn below.

### Observe-only turns (ABI v5)

Up to v4, OBSERVE existed only inside a LEARN turn (`turn_begin` refuses an empty K1 schedule), so recording typed
evidence forced a geometric learning transition. v5 separates them.

- **Begin.** `elpis_runtime_turn_begin_observe(runtime, substrate, budget, out)`, on a composite substrate only:
  - It verifies and binds the lineage exactly as `turn_begin` does, and opens the composite transaction at the live
    generation.
  - It runs **no K1 experience schedule**: the K1 candidate is the unchanged copy of the live K1 state.
  - A K1 or K1 FMS substrate is `RUNTIME_INVALID` before any native call. An open turn, an unanchored lineage,
    another substrate, and a budget outside the ceiling are refused as for `turn_begin`.
- **Observe.** Observations enter through `elpis_runtime_turn_observe`. Every managed turn, LEARN or observe-only,
  records at most `budget.max_observations` (ceiling 256). One more is `COGNITION_FUEL_EXCEEDED` before the native call
  and leaves the turn open. `counters.k1_observes` counts the crossings.
- **Commit.** `turn_commit` is volatile, as every managed commit. The live generation advances by one: the composite
  commits its untouched K1 candidate, so the K1 identity is unchanged, and installs the memory candidate.
  - A commit while nothing was observed is `RUNTIME_INVALID` and keeps the turn open.
  - A commit whose K1 transition reports a step or an epoch change fail-stops `STATE_MISMATCH` and is never built
    upon.
  - `turn_abort`, close and destroy end it as any turn.

`Runtime.run_observe` is the Python operation: an explicit `ObservationAuthority`, all or nothing, with the native
bound narrowed to the authority's. Laws are in:
- `native/runtime/src/tests.rs` (`observe_only`);
- `test_runtime_composite.c`;
- `tests/integration/test_observe_only.py`.

## The session authority (ABI v8)

RuntimeCore holds one volatile ECS_C session (`native/runtime/src/authority.rs`; `docs/ECS_C_SESSION.md`). The
transition law is the one the durable authority ran. The session crate owns it, and the crate is `no_std` with no
dependency and can reach no filesystem.

- **RuntimeCore's events are session transitions.** The anchor, an evolution reservation and an evolution
  finalization are each one typed event of the session's founding RUNTIME entity at its next sender sequence.
  - Each is compare-and-publish on the session's coordinate, in memory.
  - Nothing is written, synced or recorded anywhere else.
  - The HOT lane makes no session transition at all. A managed commit is not an event.
- **Typed-evidence admission is native.** At open, and after every operator event, RuntimeCore copies the session's
  ACTIVE sources into a bounded admission table, at most the session's entity capacity. Each source carries:
  - the 32-bit memory `source_authority` its observations record;
  - its per-turn bound.

  `elpis_runtime_turn_observe` reads the observation's `source_authority` and refuses the observation before the
  native call (`RUNTIME_OBSERVATION_REFUSED`) if:
  - its source is not an ACTIVE source of the session; or
  - it would exceed its source's per-turn bound.

  The refusal is recoverable: the turn stays open, the candidate is unchanged, and `k1_observes` is not counted. A
  record of another memory ABI is `RUNTIME_INVALID`. The check is a table lookup on the HOT lane: no I/O, no digest.
- **Operator events reach the session only through `elpis_runtime_authority_transition`.** The accepted kinds are
  ENTITY_FOUND, ENTITY_LIFECYCLE, SOURCE_VERSION and ENTITY_RECLAIM, sent by an operator entity. The refusals:

  | Event or state | Outcome |
  |---|---|
  | The persistence-only kinds: IMPORT_LEGACY and the three reconciliations | `RUNTIME_RETIRED`, even when well formed. Nothing changes |
  | RuntimeCore's own kinds, whoever they claim to be from | `RUNTIME_INVALID` |
  | Any event while a managed turn is open | `RUNTIME_TURN_OPEN`. A turn's admission never changes under it |
  | Any event while closed or fail-stopped | Refused, as every operation |

  The admission table follows the session whatever the outcome.
- **Reads.** The caller builds an event from what the session reads back:
  - `elpis_runtime_authority_entity_at`: one entity row, laid out as `elpis_ecs_c_entity`, with identities,
    lifecycles and sender watermarks;
  - `elpis_runtime_authority_next_entity_id`: the receiver the next ENTITY_FOUND must name.

  Both are pure. They are refused while closed or fail-stopped, and `RUNTIME_INVALID` past the last row or for an
  unknown kind. The ECS_C relational view (`docs/ECS_C_RELATIONAL.md`) syncs from these rows. It refuses the next
  session's table (`LINEAGE`) until it is reset.
- **Error domain.** The call sites speak the authority codes (`ELPIS_CONTINUITY_*`, the names and values continuity
  gave them, kept for the view's layout):
  - a session refusal with an equivalent is that code;
  - `NOMEM` is `RUNTIME_NOMEM`;
  - a persistence-only kind is `RUNTIME_RETIRED`;
  - any other is `RUNTIME_AUTHORITY` for an operator event, or `CONTINUITY_AUTHORITY_MISMATCH` for one of RuntimeCore's
    own.

  `elpis_runtime_authority.ecs_c_code` keeps the exact session code.

Laws:
- `native/runtime/src/tests.rs`, module `session`:
  - each open is a new genesis and close ends it;
  - seeded nonces are reproducible across runtimes and distinct within one;
  - admission, per-turn bounds and lifecycles follow the operator's events;
  - the retired kinds are refused;
  - the reads change nothing;
  - capacity exhaustion refuses atomically.
- `runtime.test_runtime_session`: the production libraries, the real composite and the relational view, composed in
  memory.

## Laws

Each law is a test in `native/runtime/src/tests.rs`, which runs a deterministic in-memory K1 stand-in over the real
volatile session (59 tests). Over real native K1, they are tested in `native/runtime/tests/test_runtime_abi.c`,
`test_runtime_lifecycle.c`, `test_runtime_composite.c`, `test_runtime_session.c` and `tests/integration`.

- **Anchor.** The first managed lineage of a session is anchored explicitly at the state's retained identity.
  RuntimeCore reads the identity only, and claims the state (below). A second anchor in the same session is
  `CONTINUITY_ALREADY_ANCHORED`. A new session anchors afresh.
- **Bind and verify.** The first QUERY or LEARN after open, or after a release, binds the lineage to the supplied state
  after comparing its identity with the session's expectation: one digest, before any claim.
  - Unanchored is `CONTINUITY_UNANCHORED`, without a fail-stop.
  - A mismatch is `CONTINUITY_STATE_MISMATCH` and fail-stops.
  - Both are refused before any K1 mutation.

  The state is then claimed, and its identity and generation are read together under the lease (`leased_identity`).
  The identity must still be the one checked, and the generation becomes the live coordinate. A bound runtime refuses
  any other state, by handle or owner, with `COGNITION_SUBSTRATE_SWITCH` and no native call.
- **Managed ownership.** Binding claims the state under this runtime's K1 lease (the K1 header's "Managed
  ownership"). `lease_claim` refuses `BUSY`, without a fail-stop, while an unmanaged transaction is open. From then on
  the state refuses every unmanaged mutating entry point (`LEASED`, before anything is touched):
  - direct learn, consolidate, reset and reserve;
  - an unmanaged transaction begin;
  - the plain candidate calls on a leased transaction.

  RuntimeCore mutates only through the `leased_*` entries. Every managed transaction begin re-establishes the state's
  live coordinate inside the same guarded K1 call: `leased_txn_begin_at` returns the generation of exactly the source
  it copied. A bound state that is not provably the lineage fail-stops with `CONTINUITY_STATE_MISMATCH` before the
  schedule touches the candidate. That is a state at another generation (the begin is aborted), or one whose lease
  another owner took (refused natively).

  `release` (`elpis_runtime_release`) gives the state back and keeps the live identity it read under the lease just
  before. Rebinding then accepts exactly one of:
  - those bytes: the live lineage, volatile work included;
  - the session's anchored identity: an in-memory rollback that abandons the volatile work.

  The lease is an aliasing defence between cooperating local callers, not a secret or a credential: in-process code
  can replace it, and the owner then fail-stops.
- **QUERY.** Read-only (`docs/COGNITION_R0.md`).
  - One native call (`leased_query`) answers `f_W(x)` from the live state and returns that same state's generation
    under one K1 guard.
  - The generation must be the live one. A lease taken by another owner, or a generation that moved, is
    `CONTINUITY_STATE_MISMATCH` (fail-stop), and the answer is withheld (zeroed).
  - The answer reflects every managed LEARN before it.
  - A query is refused while a managed turn is open (`RUNTIME_TURN_OPEN`).
  - It begins no transaction, commits nothing and hashes nothing: W, epoch, H, a, the generation and the session are
    byte-for-byte unchanged.
- **Total fuel.** Every QUERY, LEARN and observe-only turn is admitted against a budget of totals
  (`elpis_runtime_budget`). The admission comes before the identity check, any reserve, the transaction, or a candidate
  mutation. A refusal is `COGNITION_FUEL_EXCEEDED` and is not a fail-stop.
  - **The ceiling.** The budget can only narrow the compiled ceiling (`elpis_runtime_fuel_ceiling`), and the Python
    boundary refuses a library whose ceiling differs:

    | Budget field | Ceiling |
    |---|---:|
    | Experiences | 64 |
    | Rows | 16,384 |
    | Rows per experience (also the reserve ceiling) | 256 |
    | K1 learning steps | 2^20 |
    | ECS work units | 2^30 |
    | Query rows | 4,096 |
    | Observations per turn | 256 |

  - **Work units.** An ECS work unit is one multiply-accumulate-class operation of the K1 kernels, counted by a fixed
    integer formula (`native/runtime/src/fuel.rs` and `elpis.runtime.fuel`, tested equal):
    - a K1 step on r rows: `2rdw + 2dwF + F^2`;
    - a consolidation: `rF^2 + dwF`;
    - a QUERY: `rdw`.

    It is a deterministic proxy, not wall-clock time.
  - **No deadline is claimed.** The K1 ABI is synchronous and has no cooperative cancellation point, so a running
    schedule cannot be preempted.
- **Turn (LEARN).** Begin at the live generation, run one native schedule on the candidate, let the boundary decode,
  then make the native commit.
  - The commit is volatile: the live generation advances by one. There is no publication, checkpoint, digest or I/O.
  - Capacity beyond the reservation is grown on the cold path (abort, `reserve`, begin again).
  - Every refusal before the commit leaves `(W, epoch, H, a)` byte-for-byte unchanged.
  - A stale source is `ECS_STALE`.
- **Turn lifecycle.** Once a native transaction is open, exactly one terminal native action ends it before RuntimeCore
  forgets the turn: the commit, or an abort. There is no third disposition.
  - **Paths that abort it,** through the retained capability:
    - `turn_abort`;
    - a commit refused natively, or on a fail-stopped runtime;
    - `close`;
    - destruction (`elpis_runtime_destroy`, Rust `Drop`, Python `RuntimeCore.__del__`).
  - **Paths that keep it:** `open` on an open runtime is refused (`CONTINUITY_OPEN`), and `turn_abort` with another
    substrate is refused.
  - **A `BUSY` abort is retried.** An abort refused `BUSY` (a concurrent overlapping call on the same state, which did
    nothing) is retried until the state admits it, never dropped.
  - **An implicit abort installs nothing.** `(W, epoch, generation, H, a)` and the retained-state digest are
    unchanged. The state takes the next transaction, and an FMS-resident state's transaction WRITE pin is released.

  Proven over the stand-in (`tests.rs`, including the `BUSY` retry) and over real standalone and FMS-resident K1
  (`test_runtime_lifecycle.c`, `tests/integration/test_runtime_lifecycle.py`).
- **Evolution.** One attempt in flight: authority, then the caller validates, then reserve, then the caller executes
  once, then finalize.
  - The reservation is a session transition. It is in the session when `reserve` returns, before execution.
  - A refused reservation changes nothing and does not fail-stop.
  - A refused finalization fail-stops: the attempt executed, and it is never re-executed.
  - An attempt the boundary could not complete is abandoned. The reservation stays pending, and the runtime
    fail-stops with `CONTINUITY_EVOLUTION_PENDING` for the rest of the session. Nothing is retried or inferred.

  A new session starts with the evolution authority idle. The operator's `evolution_reconcile` existed to finalize a
  durable reservation after a restart, and is retired with it.
- **Restart law.** A restart, a crash or a close followed by an open begins again at genesis from what the caller
  supplies:
  - nothing learned, observed, founded or reserved in an earlier session is resumed;
  - no state that an earlier session bound can be adopted without being anchored again.

  This replaces v7's crash law and durability crash matrix (`docs/history/RUNTIME_CORE_ABI_V7.md`). There is no
  publication whose outcome could be unknown.
- **Nothing else.** There is no history, receipt log, trajectory or store. RuntimeCore's persistent footprint is zero
  bytes: it opens no file or directory, for reading or for writing. Its memory is the handle, the session (exact and
  constant, above), the admission table (at most the session's capacity) and the bound key.
- **Concurrency.** A handle serializes its calls with one lock. There is no worker thread, so no call waits for
  background work, and no call blocks on another runtime.

## Hot path

A managed `Runtime.run_query` is one Python crossing into RuntimeCore for the query (after the fail-stop probe), and
one K1 crossing (`leased_query`). Nothing is hashed and nothing is written.

A warm managed `Runtime.run_turn` is measured on every run by `tests/integration/test_runtime_hot_path.py`:
- Python makes no K1 call, and three crossings into RuntimeCore: the fail-stop probe, `turn_begin` and `turn_commit`.
- RuntimeCore makes three K1 crossings: `leased_txn_begin_at`, `leased_txn_run_schedule` and `leased_txn_commit`.
- There is no digest, identity read, reservation, abort or session transition.
- Python performs no file I/O.

**Gate A (zero persistent I/O).** `native/runtime/tests/test_runtime_hot_io.c` interposes the C library's
persistent-I/O and read-side entry points, and counts each call by the thread that made it. It counts the test's own
deliberate pipe write first, so the interposer is shown live. Over a complete session lifecycle, RuntimeCore makes
**zero** persistent-I/O calls and **zero** reads on every thread. The lifecycle is:
- create and open;
- the operator's sources and events, including the retired kinds;
- the anchor, 120 managed turns and 12 queries;
- the live identity, an evolution reservation and finalization;
- a release and a rebind;
- close, a second session, and destroy.

The same holds over an FMS-resident state on a RAM-only FMS context under memory pressure. The composed qualification
(`research/volatile_runtime_v1`) shows the same at the system-call boundary, under `strace`, for the whole process.

### HOT residency

An FMS-resident K1 state lives in FMS (`native/substrate/include/elpis/fms.h`, FMS ABI v3: WARM in RAM, HOT, no COLD
tier). A HOT operation must never move it. The guarantee is a **residency hold**:

- **Binding holds.** RuntimeCore takes the adapter's hold (`elpis_ecsg_k1_fms_resident_hold`) before it binds an
  FMS-resident state, and before anything else touches it, the identity precheck included. Binding happens at the
  first QUERY or LEARN after open or release, and at the explicit anchor. The hold succeeds only if the state is
  WARM now: `fms_acquire_resident` pins it where it is, and never promotes it or waits for a move.
- **A held state stays resident.** The hold's pin makes the object non-displaceable, and while any hold exists every
  operation on the state pins resident-only. Memory pressure refuses the newcomer (K1 FMS `-100 + FMS_E_LIMIT`;
  Python `K1Error` `CAPACITY`). Under FMS v3 nothing is demoted below WARM, and a state beyond the WARM budget is
  refused, never spilled.
- **Not resident: a precise refusal.** A state that is not WARM when it would be bound (in motion) is refused
  `RUNTIME_NOT_RESIDENT` (84; K1 FMS `-112`). Nothing is claimed, read, pinned, promoted or bound, and the runtime is
  not fail-stopped. Over a RAM-only context this is unreachable, because nothing demotes; the check stays, as a
  defence.
- **Materialization is a check.** `elpis_ecsg_k1_fms_materialize` (Python `K1FMSState.materialize()`) finds the state
  WARM and moves nothing. Before FMS v3 it was the one path that read a cold replica back.
- **Lifetime.** RuntimeCore retains the hold's release capability exactly while the state is bound. It drops it on
  `release`, `close`, reopen and destruction, and when a bind fails after taking it. The adapter refuses `close` of a
  held state (`BUSY`). A RAM state (standalone K1, composite) needs no hold.

Proofs:
- `runtime.test_runtime_hot_io`: held turns and queries under pressure, with zero read or write calls and zero
  promotions;
- `ECS.test_ecsg_k1_fms` (`test_residency_hold`);
- `test_runtime_lifecycle.c`: the hold across abort, commit, close, destroy, reopen and fail-stop;
- `tests.rs`: the binding and refusal laws over the stand-in;
- `tests/integration/test_hot_residency.py`.

The Rust suite's HOT functions are also checked statically (`tests/boundary/test_ecs_split.py`). They name no store,
lane, sync, publication, digest or identity read.

### Legacy surface

The live-coordinate entry points that the HOT/WARM correction added to K1, the K1 FMS adapter and the ECS composite,
and the residency hold, are additive.
- `native/runtime/composite/tests/test_k1_legacy_differential.c` Part A drives only pre-correction entry points and
  records every status, transition, readout, digest and envelope byte. In this tree it proves the battery
  deterministic: two runs give one transcript.
- Part B proves the additive entry points against the legacy ones in the same build.

The cross-version transcript recorded against the pre-correction tree (`81b738e`) is history
(`docs/history/RUNTIME_CORE_ABI_V7.md`, "Legacy surface"). It predates FMS v3, whose volatile residency changed the
battery's FMS section by design: a second state is now refused at the WARM bound instead of demoting the first. It is
not reproducible over FMS v3 libraries.

## Retired by v8

Each of these v7 entry points is absent from the library (`dlsym` finds none of them: `test_runtime_abi.c`) and
from the ABI inventory (`native/ABI_INVENTORY.json`, amendment V3). They are listed in
`elpis.runtime.persistence.RETIRED_RUNTIME_CORE_ABI`.

| Retired (v7) | What it did | ABI v8 |
|---|---|---|
| `elpis_runtime_create_ecs_c(state_dir, authority_dir)` | A handle over the durable ECS_C authority's directories | `elpis_runtime_create(config)`: no path |
| `elpis_runtime_import_composite` | Made a composite durable and published the import | Anchor a composite lineage directly |
| `elpis_runtime_recoverable_request`, `_identity_request` | Durable publication, through the durability lane's worker | None: a managed commit is volatile, and nothing publishes |
| `elpis_runtime_durable_wait`, `_durable_status` | The lane's tickets | None: there is no lane |
| `elpis_runtime_checkpoint_provision`, `_attach`, `_recover`, `_discard`, `_adopt` | The K1 checkpoint store's two slot files | None: a restart is a new session |
| `elpis_runtime_evolution_reconcile` | Finalized a durable reservation after restart | None: a reservation lives for its session |

Also retired:
- the testing library `libelpis_runtime_testing.so` and its fault hooks;
- the substrate tables' `leased_capture` and `envelope_seal`;
- the authority codes 1, 6, 7, 8, 14, 16, 17 and 255, and the runtime codes 76..83.

Those codes are never returned and never reused: `elpis_runtime_code_name` returns NULL for them.

The Python facade has no persistent operation either: no `publish_recoverable`, `publish_identity`, `durability`,
`recover_k1`, `discard_k1_candidate`, `adopt_k1_candidate`, `import_composite` or `publish_canonical`
(`tests/boundary/test_mission.py`, `tests/boundary/test_autonomous_persistence.py`).

Recorded evidence cites sections of the v7 specification that this document no longer has. The qualification records
of K1 Recovery R0 v4 and ECS_C Recovery Q0, and the performance record of the live/durable separation, are kept as
written, so their citations stand. Those sections are in `docs/history/RUNTIME_CORE_ABI_V7.md`:
- "Live and durable authority (ABI v4)" and "Recoverable and identity-only publication";
- "The durable authority (ABI v6)" and "One durable authority (ABI v7)";
- the laws "Durable publication", "Crash law" (with the durability crash matrix), "Control-operation crash points"
  and "K1 Recovery R0".

## Native admission

RuntimeCore is never loaded from a pathname.
- **The pin.** `RuntimeConfig` names the library by an absolute path and supplies a deployment-pinned
  `native_authority`: a `PinnedAuthority`, whose own SHA-256 comes from trusted configuration.
- **The admission.** `Runtime` admits the library through the substrate's one native admission mechanism
  (`elpis.substrate.native_admission`, the mechanism the HACF bridges use):
  1. opened beneath its directory, as the trusted descriptor root, with `openat2(RESOLVE_BENEATH |
     RESOLVE_NO_SYMLINKS)`;
  2. copied into a sealed memfd;
  3. hashed against the exact pin;
  4. loaded from the seal, under the RuntimeCore library identifier `elpis_runtime`.

  The ABI version and the compiled fuel ceiling are then checked. Without an authority the runtime refuses to start
  (`RUNTIME_UNPINNED`).
- **K1 libraries.** The managed runtime drives only K1 code it admitted. A K1 or K1 FMS state whose library was not
  admitted under `elpis_ecsg_k1` / `elpis_ecsg_k1_fms` is refused (`ECS_NATIVE_UNADMITTED`) before any native call. K1
  sets are admitted in dependency order: `elpis_ecsg_math`, `elpis_ecsg_k1`, `elpis_ecsg_k1_fms`.
- **Dependencies.** Every Elpis dependency the dynamic loader binds by SONAME is verified to be pinned bytes: a sealed
  object, or a mapped file whose inode and SHA-256 equal the pin. Anything else refuses the admission.
- **Trust.** The pins authenticate bytes, not a publisher. The loader, kernel and platform libraries are trusted
  (`tests/integration/test_native_admission.py`).

Admission writes nothing to storage: the sealed copy is an anonymous memory object.

## Build and qualification

- **The library.** `libelpis_runtime.so`, the production library, is the only build. CMake builds it offline through
  `cargo rustc`, and sets the SONAME on the final link only. It embeds the volatile ECS_C session crate and no other
  authority. It exports neither `elpis_ecs_c_*` nor `elpis_continuity_*`.
- **The import gate.** It imports no write-class C library function (`tests/boundary/test_volatile_runtime.py`, the
  import gate over every binary the runtime loads).
- **`ctest -L runtime`:**

  | Test | What it covers |
  |---|---|
  | `runtime.cargo_test` | The Rust law suite over the session |
  | `runtime.test_runtime_abi` | The C ABI over real K1; the v7 entry points absent |
  | `runtime.test_runtime_lifecycle` | The turn lifecycle over real standalone and FMS-resident K1 |
  | `runtime.test_runtime_composite` | Composite substrates |
  | `runtime.test_runtime_session` | The session authority and the relational view, composed |
  | `runtime.test_runtime_hot_io` | Gate A at the C library boundary |
  | `runtime.test_k1_legacy_differential` | The legacy surface |
  | `runtime.native_host_smoke` | The production host |
  | `runtime.test_ecs_observation_hacf` | Typed HACF observations |
  | `runtime.test_autonomous_containment` | The launcher's containment |

- **CI.** CI lints and tests the crate in the Rust job, and runs the native tests in every native job: gcc and clang,
  Debug and Release, ASan+UBSan and TSan.
- **Composed qualification.** `research/volatile_runtime_v1` (`ZERO_WRITE_PASS`) runs RuntimeCore's suites, the
  native host and the Python reference under `strace`, unconfined and confined. Every path records zero write-class
  events on storage.

## Non-claims

- RuntimeCore is systems authority, not cognition. It qualifies no codec and makes no semantic claim. It does not make
  H-ECS or any hierarchy canonical.
- The session's lineage and commit identify a session's state within its life. They authenticate nothing to anyone
  outside the process, and nothing survives the process (`docs/ECS_C_SESSION.md`, "What is not claimed").
- RuntimeCore issues no persistent write. Whether its pages can reach swap, and what host services outside its process
  write, are host properties (`docs/VOLATILE_RUNTIME_CENSUS.md`, section 6; `docs/NONCLAIMS.md`).
- Performance figures for the HOT lane are descriptive measurements on one host, not qualification.
