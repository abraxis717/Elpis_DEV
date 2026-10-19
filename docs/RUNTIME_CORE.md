# RuntimeCore: the runtime's systems authority in Rust

`native/runtime` (crate `elpis_runtime`, C ABI `include/elpis/runtime.h` v6) owns the mutable systems
authority of an open Elpis runtime and is the one native composer of ECS_G (K1, reached through function tables) and
ECS_C (the durable authority, embedded). `elpis.runtime.Runtime` is its compatibility facade until Stage F
(docs/NATIVE_SPLIT_CENSUS.md).

ABI history: v2 the turn lifecycle, v3 managed ownership, v4 the live/durable split (volatile commits, explicit
durability), v5 the observe-only managed turn ("Observe-only turns (ABI v5)" below), v6 the ECS_C durable authority
("The durable authority (ABI v6)" below). Each version is refused by a caller of another (`elpis_runtime_abi_version`).

## Live and durable authority (ABI v4)

RuntimeCore keeps two coordinates and never equates them:

* **Live**: the bound state's K1 generation under RuntimeCore's managed lease (the K1 header's "Live coordinate").
  It advances by exactly one per managed commit. QUERY and LEARN operate on the live state and check this coordinate
  inside the same guarded K1 call; nothing is hashed on the HOT lane.
* **Durable**: continuity's record (the retained-state identity of the last completed durable publication) and,
  with a checkpoint store, a verified slot holding that state's complete envelope. Restart resumes exactly this,
  never anything later.

A managed commit is **volatile**: it installs the native candidate and advances the live coordinate; nothing is
checkpointed, published, hashed or written. Durable publication is a separate, explicit WARM request of one of two
kinds (below): the HOT lane captures the envelope unsealed under the lease (the one synchronous step, a copy) and
hands it to the runtime's **durability lane**, a worker thread that seals and verifies it, makes it durable and
resolves the request's ticket. The caller chooses whether to wait (`durable_wait`). A crash may lose the volatile work
since the last completed publication; the API says so (`durability == "VOLATILE"`, the ticket's status).

### Recoverable and identity-only publication

Making a *state* durable and making its *identity* durable are different promises, and the ABI never lets one stand
for the other:

| | Recoverable | Identity-only |
|---|---|---|
| Request | `elpis_runtime_recoverable_request`, Python `Runtime.publish_recoverable` | `elpis_runtime_identity_request`, Python `Runtime.publish_identity` |
| Requires | an attached checkpoint store (the owner of the bytes); otherwise refused `RUNTIME_RECOVERY_UNAVAILABLE` before anything is bound or captured, never downgraded | no checkpoint store attached; otherwise refused `RUNTIME_CHECKPOINT_ATTACHED` (with a store every publication must be recoverable) |
| Lane | seal + verify, slot write + `fdatasync`, **read back and verify** the slot is exactly the captured envelope (`CHECKPOINT_IO` and the slot withdrawn otherwise), then the continuity record + `fdatasync` | seal + verify, the continuity record + `fdatasync` |
| Success | `ACKNOWLEDGED`, `recoverable` always 1: restart resumes exactly this state from the store | `IDENTITY_PUBLISHED`, `recoverable` always 0: never an acknowledgment; the caller keeps the bytes and restart can rebind only a state the caller supplies |

The ticket carries its `kind`; `last_acknowledged` in the lane's status and the `durable_acknowledged` counter count
recoverable acknowledgments only; the Python adapter re-checks the law on every ticket (an `ACKNOWLEDGED` ticket that
is not recoverable, or a recoverable identity-only one, is refused as a contract violation). Attaching a checkpoint
store is refused while a publication is pending or running (`RUNTIME_DURABLE_BUSY`), so an identity-only request never
runs against a store. The explicit anchor is a control operation, not a publication request: it persists the
envelope when a store is attached and otherwise anchors the identity only (`recover` then reports
`CHECKPOINT_MISSING` until a recoverable publication of a state makes it resumable; a recoverable publication of the
anchored state itself writes and verifies the slot without a new continuity record).

**Superseded requirement.** Through ABI v3, every committed LEARN published its new expected K1 identity to
continuity before returning (one 176-byte record write and one `fdatasync`), and with a checkpoint store also wrote
and synced the candidate's complete envelope before the native commit; QUERY and every begin hashed the state to
compare it with the durable identity. That coupled every HOT transition to filesystem synchronization and made
continuity decide when the runtime could think. It is withdrawn by this correction (Elpis_DEV), not reinterpreted:
recorded evidence produced under v3 stays as recorded.

```text
  HOT (caller thread, handle lock)                       WARM durability lane (one worker per open runtime)
  query        leased_query at the live generation       (idle)
  turn_begin   leased_txn_begin_at the live generation
  turn_commit  leased_txn_commit: volatile, no I/O
  recoverable_request | identity_request                 seal + verify
               leased_capture (unsealed copy) --->       [recoverable] slot write + fdatasync + read-back verify
               returns PENDING at once                   -> record write + fdatasync -> ticket
  durable_wait / durable_status: lane lock only  <---    ACKNOWLEDGED | IDENTITY_PUBLISHED / FAILED / UNCERTAIN /
                                                         ABANDONED / SUPERSEDED / ...
```

## What RuntimeCore owns

| Authority | Before (Python `composition.py`) | Now (RuntimeCore, Rust) |
|---|---|---|
| Lifecycle (open / close / reopen) | `Runtime.open` / `close` around a `ContinuityStore` | `Core::open` / `close`; the continuity store is embedded |
| Fail-stop disposition | `Runtime._continuity_fault` | `Core.fault` (a continuity code); every lineage- or authority-dependent call refuses with it until reopen |
| Continuity store | a Python `ContinuityStore` handle | the `elpis_continuity::Store`, embedded as a crate (one implementation; the library also exports the continuity C ABI) |
| K1 lineage binding | `Runtime._turn_substrate` and `_reconcile_cognition_substrate` | `Core.bound` (`Key`: kind, native handle, resident id, owner identity, dimension; and the live generation), verified once against the durable expected identity (or the live identity an explicit release gave back) |
| Managed QUERY (read-only) | none (the turn was the only cognitive operation) | `query`: one native `leased_query` (the answer `f_W(x)` and the generation of the state that computed it, under one K1 guard); no transaction, no commit, no publication, no digest |
| Managed LEARN (the turn's native transaction) | Python sequenced `txn_begin`, `run_schedule`, `commit_identity`, abort | `turn_begin` (lineage check at the live generation, begin, one schedule, cold-path capacity growth) and `turn_commit` (volatile) / `turn_abort` |
| Durable publication | Python `commit_cognition_transition` + fault flag, once per turn | explicit `recoverable_request` / `identity_request` on the durability lane (`durable.rs`); `durable_wait`, `durable_status` |
| Evolution reservation / finalization | Python reserve, execute, finalize, fault on exception | `evolution_reserve` / `evolution_finalize` / `evolution_abandon` / `evolution_reconcile`: one attempt in flight |

What stays at the language boundary, because it is semantics or validation rather than mutable authority:

* the token codec and the (unqualified) ECS codec map: `cognition._validate_turn_request`, `_encode` (text
  -> tokens -> a validated native-ready `Stimulus`) and `_decode` (readout -> tokens -> text);
* the evolution gate's validation and execution (`EvolutionPathGate.reject_reason` / `execute`): immutable
  values and digest identities, no mutable state (see the census);
* the reference that keeps the bound K1 state's Python owner alive while RuntimeCore holds the binding
  (lifetime only: RuntimeCore decides the binding).

No ECS<->DSV semantic codec is defined; RuntimeCore adds none. The turn still refuses with
`ECS_CODEC_UNQUALIFIED` before anything else.

## How RuntimeCore reaches K1

RuntimeCore links no ECS code and computes no ECS mathematics. The caller passes a substrate descriptor:

```c
typedef struct {
    uint32_t kind;          /* 1: standalone K1 (ecsg_k1.h); 2: FMS-resident K1 (ecsg_k1_fms.h) */
    uint32_t reserved;
    void *handle;           /* elpis_ecsg_k1 * or elpis_ecsg_k1_fms * */
    uint64_t id;            /* resident state id (kind 2) */
    uint64_t owner;         /* caller identity of the handle's owner (unique while bound) */
    uint64_t dim;           /* verified natively */
    const void *api;        /* the K1 library's own entry points */
} elpis_runtime_substrate;
```

`api` is a table of the K1 library's own functions (ABI v4: `state_digest`, `shape`, `lease_claim`,
`lease_release`, `leased_reserve`, `leased_identity`, `leased_query`, `leased_txn_begin_at`,
`leased_txn_run_schedule`, `leased_txn_commit`, `txn_abort`, `leased_capture`, `envelope_seal`); the Python adapter
takes their addresses from the
loaded library, a C or C++ host passes them directly (`native/runtime/tests/test_runtime_abi.c`). The declared
dimension is verified natively: K1 compares the readout length it implies with the state's own before it reads
any input byte (`schedule_check`).

### Substrate lifetime (ABI v2)

Outside a managed turn a descriptor need only be live for the call. A successful `turn_begin` opens a native K1
transaction, and to end it on every path RuntimeCore retains, until the turn ends, an abort-only capability over
the state: a copy of the descriptor's handle, resident id and `txn_abort` entry (never commit, schedule or any
other entry). The native state must stay live, and its library loaded, for that interval. This is the v2 change
from v1, under which a turn could be forgotten and the state then destroyed; the bump makes a v1 caller refuse to
load the library rather than inherit the new obligation silently.

* FMS-resident K1: enforced natively by the adapter (`ecsg_k1_fms.h`): a resident state with an open transaction
  cannot be closed and its runtime cannot be destroyed while a state is registered (both `BUSY`), so the retained
  handle and id are live for as long as the transaction RuntimeCore holds is open.
* Standalone K1: `elpis_ecsg_k1_destroy` frees a state whatever transaction it holds, so the owner must not destroy
  it while a turn on it is open. The Python adapter satisfies this: native K1 handles are freed only by an explicit
  `close` (no finalizer frees them), and `Runtime` keeps the bound state's owner alive until RuntimeCore has ended
  the turn (`close` and `open` release it only afterwards).

### Composite substrates

A substrate of kind `ELPIS_RUNTIME_SUBSTRATE_COMPOSITE` (3) is an ECS composite retained state
(docs/ECS_COMPOSITE_STATE.md). Its table, `elpis_runtime_composite_api`, is the K1 table (same signatures and
status numbers; the digests are `ELPISEC1` identities) plus `leased_txn_observe`. RuntimeCore binds, queries,
learns, publishes durably and recovers it exactly as a K1 state (a composite's envelope size depends on its memory
capacity, so its durable publication needs an attached, provisioned checkpoint store: `CHECKPOINT_INVALID`
otherwise); continuity records the lineage's format (record v3),
and binding or committing a substrate of another format than the lineage's is `CONTINUITY_STATE_MISMATCH`
(fail-stop on binding, as an identity mismatch is).

* `elpis_runtime_turn_observe` records an opaque `elpis_ecs_mem_observation` on the open turn's memory candidate
  (RuntimeCore never reads it). The memory therefore changes only inside a managed turn, becomes live with that
  turn's native commit, and durable only through an explicit durable publication. A refusal is recoverable
  (`ECS_REFUSED`, with the composite and memory statuses reported); the turn stays open.
* No memory LEARN or CONSOLIDATE is exposed: those operations are UNQUALIFIED (docs/ECS_MEMORY.md).
* `elpis_runtime_import_composite` moves a K1 lineage to a composite made by `elpis_ecs_composite_import_k1`
  from exactly the authorized K1 state with an empty memory; it requires an attached checkpoint store sized for
  the composite envelope, makes the envelope durable, then publishes `import_cognition`.

`native/runtime/tests/test_runtime_composite.c` (testing library) covers the composite anchor, volatile turns with
observations, refusals, the format mismatch, process death inside a durable publication (after the slot is synced;
with the continuity record torn), recovery with the memory transition present exactly once or absent, the
legacy import, and the observe-only turn below.

### Observe-only turns (ABI v5)

Up to v4, OBSERVE existed only inside a LEARN turn (`turn_begin` refuses an empty K1 schedule), so recording typed
evidence forced a geometric learning transition. v5 separates them:

* `elpis_runtime_turn_begin_observe(runtime, substrate, budget, out)` (a composite substrate only) verifies and
  binds the lineage exactly as `turn_begin` does and opens the composite transaction at the live generation, but runs
  **no K1 experience schedule**: the K1 candidate is the unchanged copy of the live K1 state. A K1 or K1 FMS
  substrate is `RUNTIME_INVALID` before any native call; an open turn, an unanchored lineage, another substrate and a
  budget outside the ceiling are refused as for `turn_begin`.
* Observations enter through `elpis_runtime_turn_observe`. Every managed turn (LEARN or observe-only) records at most
  `budget.max_observations` (ceiling 256); one more is `COGNITION_FUEL_EXCEEDED` before the native call and leaves
  the turn open. `counters.k1_observes` counts the crossings.
* `turn_commit` is volatile, as every managed commit: the live generation advances by one (the composite commits its
  untouched K1 candidate: W, epoch, H and a reinstalled, the K1 identity unchanged, since the generation is not
  state) and the memory candidate is installed. A commit while nothing was observed is `RUNTIME_INVALID` and keeps
  the turn open; a commit whose K1 transition reports a step or an epoch change fail-stops `STATE_MISMATCH` (never
  built upon). `turn_abort`, close and destroy end it as any turn.
* Durability is unchanged: a recoverable publication captures the composite (its K1 bytes those of the last LEARN)
  and the restart law is v4's.

v5 changed the layout of `elpis_runtime_budget` (`max_observations`, 56 bytes) and `elpis_runtime_counters`
(`k1_observes`, 136 bytes); every v4 entry point keeps its meaning and the ELPISGK1, ELPISEC1 and continuity formats
are unchanged, so durable state written by a v4 library restores under v5. `Runtime.run_observe` is the Python
operation (an explicit `ObservationAuthority`, all or nothing, the native bound narrowed to the authority's);
laws: `native/runtime/src/tests.rs` (`observe_only`), `test_runtime_composite.c`, `tests/integration/test_observe_only.py`
(including a SIGKILLed child). Mutation sensitivity (rebuilt mutants outside the repository, unmutated baseline green): a K1 substrate admitted,
a commit with nothing observed, the native observation bound removed, an observe-only commit reporting K1 learning
built upon, a refused observation not aborting the turn, the native bound not narrowed to the authority's, and
`run_observe` skipping the authority: all seven killed.

## The durable authority (ABI v6)

The durable authority a runtime publishes to is chosen when the handle is created, and every law of this document
holds over either:

* `elpis_runtime_create(path)`: standalone continuity (docs/CONTINUITY.md), the v5 behaviour, unchanged.
* `elpis_runtime_create_ecs_c(state_dir, authority_dir)`: the ECS_C authority (docs/ECS_C_AUTHORITY.md): bounded,
  authenticated, replayed, with a rollback witness in a separately protected directory. The operator provisions and
  reconciles it through its own ABI (`elpis/ecs_c_authority.h`) with no runtime open. RuntimeCore embeds the crate
  without its C ABI: one implementation, one transition law.

`native/runtime/src/authority.rs` puts both behind continuity's interface (`Store`), so the durability lane, the
anchor, import, evolution and recovery code paths are the same code over either backend. Under ECS_C:

* **Publications are typed events.** Each publication (anchor, durable publication, composite import, evolution
  reserve and finalize) is one event of the authority's founding RUNTIME entity at its next sender sequence,
  compare-and-publish on the authority's generation and state root. RuntimeCore is the authority's only writer while
  it holds it open; a durable publication is still the checkpoint slot (verified) followed by the authority's record.
  ECS_C writes two fixed slots per transition (the record, R = 1344 + 160 × capacity bytes, and the 240-byte witness,
  each written in place and synced); nothing is created, renamed or removed, and the HOT lane performs no authority
  I/O (`tests.rs` `ecs_c::the_hot_lane_performs_no_authority_io_...`).
* **Typed-evidence admission is native.** At open, and after every operator event, RuntimeCore copies the authority's
  ACTIVE sources (each with the 32-bit memory `source_authority` its observations record, and its per-turn bound) into
  a bounded admission table (at most the authority's entity capacity). `elpis_runtime_turn_observe` reads the
  observation's `source_authority` (`elpis_ecs_mem_observation`, ABI v1) and refuses, before the native call, an
  observation whose source is not an ACTIVE source of the authority or that exceeds its source's per-turn bound
  (`RUNTIME_OBSERVATION_REFUSED`, recoverable: the turn stays open, the candidate unchanged, `k1_observes` not
  counted); a record of another memory ABI is `RUNTIME_INVALID`. The check is a table lookup on the HOT lane: no I/O,
  no digest. Under continuity there is no table and admission stays the boundary's (`ObservationAuthority`).
* **Each durable publication records an evidence summary.** RuntimeCore counts the typed observations its managed
  commits install; the durability lane (never the HOT lane) gives the PUBLISH event the number committed since the
  last acknowledged publication and the identity of the memory that holds them (the captured ELPISEC1 envelope's
  memory component, offset 80), and the authority chains them into its evidence head. A refused publication accounts
  for nothing, so the next one carries everything since the last acknowledged one; a K1 envelope has no memory and
  carries none (a zero count pairs with a zero digest, the transition law's rule).
* **Operator events reach a held authority only through `elpis_runtime_authority_transition`**: ENTITY_FOUND,
  ENTITY_LIFECYCLE, SOURCE_VERSION, ENTITY_RECLAIM and IMPORT_LEGACY, sent by an operator entity. RuntimeCore's own
  kinds and the reconciliations are `RUNTIME_INVALID` (whoever they claim to be from); an event while a managed turn
  is open is `RUNTIME_TURN_OPEN` (a turn's admission never changes under it); closed or fail-stopped, refused as every
  operation. The admission table follows the authority whatever the outcome.
* **An authority that is not in its normal mode refuses the open** (`RUNTIME_AUTHORITY`): restricted, reconcile-only
  (rollback, corrupt or unauthenticated acknowledged record, missing witness) or refused by recovery. The runtime stays
  closed; `elpis_runtime_authority_status` reports the recovery disposition, the mode and the exact ECS_C code, readable
  closed. The operator reconciles with the authority's ABI (`elpis_ecs_c_read_reconciliation` names the one
  reconciliation it accepts) and reopens.
* **Error domain.** The call sites speak continuity's codes: an ECS_C refusal with a continuity equivalent is that code;
  any other is `RUNTIME_AUTHORITY` at open and `CONTINUITY_AUTHORITY_MISMATCH` in a publication (the realistic case:
  the founding runtime is no longer ACTIVE). `elpis_runtime_authority.ecs_c_code` keeps the exact code.

Not yet: RuntimeCore does not consult source versions or scopes at admission (an observation records neither; Stage E
binds them through the memory's source invalidation), the Python facade still drives continuity (Stage F retires it,
with standalone continuity), and the testing library's durability fault plan maps continuity's publication steps onto
the authority's record slot only (the witness faults are the authority's own suite's).

Laws: `native/runtime/src/tests.rs` (`ecs_c`, and the whole suite a second time over ECS_C:
`ELPIS_RUNTIME_TEST_AUTHORITY=ecs_c`, ctest `runtime.cargo_test_ecs_c`, every test except the continuity-specific
HOT-lane I/O constants, pinned by the K1 Recovery R0 v4 premise), `runtime.test_runtime_ecs_c` (the production
libraries, the real composite: provisioning, admission, operator events, the evidence summary, a copied-back authority
refusing the open, the exact reconciliation, the reopen). The K1 Recovery R0 v4 RUN PASS is bound to its qualifying
commit and does not extend to this RuntimeCore (its premise P4 pins `durable.rs`, which v6 changes).

## Laws

Each law is a test in `native/runtime/src/tests.rs` (deterministic in-memory K1 stand-in, continuity fault
injection) and, over real native K1, in `native/runtime/tests/test_runtime_abi.c`,
`native/runtime/tests/test_runtime_lifecycle.c` and `tests/integration`.

* **Anchor.** The first managed lineage is anchored explicitly at the state's retained identity; RuntimeCore
  reads the identity only (and claims the state, below). A second anchor is `CONTINUITY_ALREADY_ANCHORED`.
* **Bind and verify.** The first operation after open (QUERY, LEARN or a durable request) binds the lineage to the
  supplied state after comparing its identity with the durable expectation (one digest, before any claim):
  unanchored is `CONTINUITY_UNANCHORED` (no fail-stop), a mismatch is `CONTINUITY_STATE_MISMATCH` and fail-stops;
  both before any K1 mutation. Then the state is claimed and its identity and generation are read together under
  the lease (`leased_identity`): the identity must still be the one checked, and the generation becomes the live
  coordinate. A bound runtime refuses any other state, by handle or owner, with `COGNITION_SUBSTRATE_SWITCH` and no
  native call.
* **Managed ownership.** Binding claims the state under this runtime's K1 lease (the K1 header's "Managed
  ownership"; `lease_claim` refuses `BUSY`, without a fail-stop, while an unmanaged transaction is open). From then
  on the state refuses every unmanaged mutating entry point (`LEASED`, before anything is touched): direct learn,
  consolidate, reset and reserve, an unmanaged transaction begin, and the plain candidate calls on a leased
  transaction. RuntimeCore mutates only through the `leased_*` entries, and every managed transaction begin
  re-establishes the state's live coordinate inside the same guarded K1 call (`leased_txn_begin_at` returns the
  generation of exactly the source it copied), even on a warm binding. Under the lease no unmanaged entry can move
  the state, so an equal generation is the same retained state. A bound state that is not provably the lineage
  (another generation: the begin is aborted; or its lease taken by another owner: refused natively) fail-stops with
  `CONTINUITY_STATE_MISMATCH` before the schedule touches the candidate: nothing outside the lineage is built on or
  published. `release` (`elpis_runtime_release`; the Python facade's `close` and `release`) gives the state back,
  keeping the live identity it read under the lease just before: rebinding accepts exactly those bytes (the live
  lineage, volatile work included) or the durable identity (as a restart would); `close` without a descriptor does
  not touch the state. Reads stay open to anyone. The lease is an aliasing defence between cooperating local callers, not a secret or a
  credential: in-process code can replace it (and the owner then fail-stops). Before this law an out-of-band
  mutation was accepted by K1 and the next managed LEARN committed on top of it before publication refused it
  (`tests/integration/test_managed_ownership.py`).
* **QUERY.** Read-only (docs/COGNITION_R0.md). One native call (`leased_query`) answers `f_W(x)` from the LIVE
  state and returns that same state's generation under one K1 guard, so no operation can move the state between
  the answer and its coordinate. The generation must be the live one: unanchored is `CONTINUITY_UNANCHORED`; a
  lease taken by another owner or a generation that moved is `CONTINUITY_STATE_MISMATCH` (fail-stop) and the answer
  is withheld (zeroed). The answer reflects every managed LEARN before it, durable or not. A query is refused while
  a managed turn is open (`RUNTIME_TURN_OPEN`). It begins no transaction, commits nothing, publishes nothing and
  hashes nothing: W, epoch, H, a, the generation and both continuity slots are byte-for-byte unchanged
  (`tests/integration/test_query_learn.py`).
* **Total fuel.** Every QUERY and LEARN is admitted against a budget of totals (`elpis_runtime_budget`: experiences,
  rows, rows of one experience (also the reserve ceiling), K1 learning steps, ECS work units, query rows) before the
  identity check, any reserve, the transaction or a candidate mutation; a refusal is `COGNITION_FUEL_EXCEEDED` and
  is not a fail-stop. The budget can only narrow the compiled ceiling (`elpis_runtime_fuel_ceiling`: 64
  experiences, 16384 rows, 256 rows per experience, 2^20 steps, 2^30 work units, 4096 query rows), and the Python
  boundary refuses a library whose ceiling differs. An ECS work unit is one multiply-accumulate-class operation
  of the K1 kernels as counted by a fixed integer formula (`native/runtime/src/fuel.rs`, `elpis.runtime.fuel`;
  tested equal): a K1 step on r rows is `2rdw + 2dwF + F^2`, a consolidation `rF^2 + dwF`, a QUERY `rdw`. It is
  a deterministic proxy, not wall-clock. **No deadline is claimed**: the K1 ABI is synchronous and has no
  cooperative cancellation point, so a running schedule cannot be preempted; preemptive deadlines are a missing
  interface. RuntimeCore reads the state's immutable shape once per state (`shape`, cold path) to count width.
* **Turn (LEARN).** begin (at the live generation) -> one native schedule on the candidate -> (boundary decodes) ->
  native commit (volatile: the live generation advances by one; no publication, no checkpoint, no digest, no I/O).
  Capacity beyond the reservation is grown on the cold path (abort, `reserve`, begin again). Every refusal before
  the commit leaves `(W, epoch, H, a)` byte-for-byte unchanged; a stale source is `ECS_STALE`.
* **Durable publication.** `recoverable_request` / `identity_request` (the kind checked against the attached store
  first; bind if needed; refused while a turn is open, on an unanchored lineage and while the lane is faulted)
  captures the envelope unsealed under the lease at the live generation and returns a `PENDING` ticket at once. The
  handoff is bounded: one job runs and at most one waits; a newer request replaces a waiting one (`SUPERSEDED`,
  never silently acknowledged); a request never waits for the worker. The worker seals and verifies the capture.
  Recoverable: it refuses while an unresolved candidate exists (`CHECKPOINT_UNRESOLVED`), writes and syncs the slot
  that does not hold the authority (unless a slot already holds the capture), reads the slot back and verifies it,
  and, unless the capture already is the durable authority, continuity publishes `durable -> captured`.
  Identity-only: continuity publishes `durable -> captured` (nothing when it already is the authority). Ticket
  statuses: `ACKNOWLEDGED` (recoverable only, always `recoverable`), `IDENTITY_PUBLISHED` (identity-only only, never
  `recoverable`), `FAILED` (nothing durable changed: a written slot is withdrawn; not a lane fault), `UNCERTAIN`
  (outcome unknown: continuity's record was being written or synced when the publication failed or the process
  died; the lane faults, sticky until reopen, and restart resolves it to exactly one complete record, possibly this
  one), `ABANDONED` (not published: closed, or the process died, before continuity's record was begun), `UNKNOWN`
  (resolved too long ago to be kept).
  Operator discard / adopt and the legacy import refuse while the lane is busy (`RUNTIME_DURABLE_BUSY`). The HOT
  lane keeps thinking through a FAILED or UNCERTAIN publication; only a lost lineage (above) or a simulated process
  death fail-stops it.
* **Turn lifecycle.** Once a native transaction is open, exactly one terminal native action ends it before
  RuntimeCore forgets the turn: the commit, or an abort. There is no third disposition. `turn_abort`, a commit
  refused natively or on a fail-stopped runtime, `close`, and destruction (`elpis_runtime_destroy`, Rust `Drop`,
  Python `RuntimeCore.__del__`) all abort it through the retained capability; `open` on an open runtime is refused
  (`CONTINUITY_OPEN`) and keeps it; `turn_abort` with another substrate is refused and keeps it. An abort refused
  `BUSY` (a concurrent overlapping call on the same state, which did nothing) is retried until the state admits
  it, never dropped. An implicit abort installs nothing and publishes nothing: `(W, epoch, generation, H, a)`, the
  retained-state digest and continuity are unchanged; the state takes the next transaction, and an FMS-resident
  state's transaction WRITE pin is released. Proven over a stand-in (`tests.rs`, including the `BUSY` retry) and
  over real standalone and FMS-resident K1 (`native/runtime/tests/test_runtime_lifecycle.c`,
  `tests/integration/test_runtime_lifecycle.py`): abort, commit, close, double close, destroy, reopen, fail-stop
  then close, and lifecycle calls without a turn (no native call).
* **Crash law.** A K1 commit is never rolled back. Three things are kept apart: the **durable publication** (the
  synced continuity record), the **caller's acknowledgment** (the lane's report of it), and an **unknown outcome**
  (a death or failure between the two, or during the record's write or sync). The invariant: *restart recovers only
  a complete, authorized durable state, which may include a publication completed immediately before the caller
  lost its acknowledgment.* A missing acknowledgment is never evidence that nothing became durable; restart never
  recovers volatile work that was never requested, and never a state continuity does not name (checkpoint bytes
  never authorize themselves; a candidate waits for the operator). Checksums, generation ordering, the predecessor
  authority of every record and operator reconciliation are unchanged. Durability crash matrix (`durable.rs` steps;
  `tests.rs` simulated death at every step; `tests/integration/test_durable_process_kill.py` real `SIGKILL` at every
  step and after the acknowledgment was delivered):

  | Death at | Ticket (in-process view) | On disk | Restart |
  |---|---|---|---|
  | 1 taken, 2 sealed (before the checkpoint write) | `ABANDONED` | nothing | the previous authority |
  | 3 slot torn (a partial checkpoint write) | `ABANDONED` | a partial slot | the previous authority (a torn slot is not a checkpoint) |
  | 4 slot written, 5 slot synced (before continuity) | `ABANDONED` | a complete candidate slot | the previous authority; `CANDIDATE_UNRESOLVED` until an operator discards or adopts it |
  | 6 record torn (a partial continuity write) | `UNCERTAIN` | the slot and a partial record | the previous authority (a torn record is not an authority); candidate unresolved |
  | 7 record written, not synced | `UNCERTAIN` | the complete record in the page cache | process kill: the new state; power loss: possibly the previous authority (the record may be lost) |
  | 8 record synced, before the acknowledgment | `UNCERTAIN` | the synced record | the new state, although the caller was never acknowledged |
  | 9 acknowledged, before or after delivery | `ACKNOWLEDGED` | everything | the new state |

  **Process kill is not power loss.** A process death (`SIGKILL`, abort) keeps the kernel's page cache, so a written
  but unsynced record (step 7) survives it; a power loss may drop it, and restart then finds the previous authority
  (in-process sync-loss faults cover that branch). The `SIGKILL` tests prove the process-death column only; no test
  here proves power-loss durability, which depends on the device honoring `fdatasync`.
* **Control-operation crash points** (testing library only). The anchor, the legacy import and the operator
  reconciliations run outside the lane's step machine, on the caller's thread with the stores held, so the lane's
  steps cannot reach them. `elpis_runtime_testing_control_fault(rt, op, action, point, arg)` (`durable::control`;
  `RuntimeCore.testing_control_fault`) plans one fault of the next operation of kind `op` (1 anchor, 2 import, 3
  discard, 4 adopt) that gets past its lifecycle checks: a simulated death (action 1) or a park of the caller (2) at
  `point`, with exactly that point's on-disk effect. A parked caller holds the handle, so only the process's death
  ends the wait (after 300 s it is a simulated death); the durability status reports `parked` = 100 * op + point.
  Every fault it arms in the checkpoint or continuity store is disarmed right after the step it targets. Restart
  decides from continuity alone (`tests.rs` `control_points`; the real `SIGKILL` matrix is K1 Recovery R0 v4's):

  | Point | Anchor | Import (K1 lineage, composite store) | Discard | Adopt |
  |---|---|---|---|---|
  | 1 begun (checks passed, nothing written) | unanchored: `NOTHING_TO_RECOVER` | the K1 authority | the candidate unresolved | the candidate unresolved |
  | 2 slot written, not synced | unanchored; the slot holds the state but never creates a lineage | the K1 authority; the composite slot is no candidate (no slot of that store holds the authority) | withdrawn (process death keeps the zeroed header) | — |
  | 3 slot synced, continuity not begun | the same | the same | withdrawn | — |
  | 4 record torn after `arg` bytes | unanchored (a torn record is not an authority) | the K1 authority | — | the candidate unresolved |
  | 5 record written, not synced | anchored (process death keeps it) | the composite, `RESUMABLE` from the composite store | — | adopted |
  | 6 record synced, not returned | anchored | the composite | — | adopted |

  The production library exports none of these entry points (`nm -D`: no `elpis_runtime_testing_*` symbol), and
  its control operations carry no fault branch (the hooks compile only under the `testing` feature).
* **K1 Recovery R0.** With an operator-provisioned checkpoint store attached (`elpis_runtime_checkpoint_attach`,
  `RuntimeConfig.k1_checkpoint_dir`), the anchor persists the anchored state's complete envelope before continuity
  anchors it, and every durable publication persists its capture's complete envelope in the slot that does not hold
  the authorized state before continuity names it. A LEARN never touches the store. Restart asks `recover`
  (read-only): `NOTHING_TO_RECOVER`, `RESUMABLE` (the authorized envelope), `CANDIDATE_UNRESOLVED` (durable
  publication refused with `CHECKPOINT_UNRESOLVED` until an operator `discard`s or `adopt`s exactly that candidate;
  LEARN and QUERY proceed on the live state) or `CHECKPOINT_MISSING`. Checkpoint bytes never authorize themselves;
  continuity remains the current-authority register. Two fixed slot files, no history (docs/K1_RECOVERY_R0.md).
* **Evolution.** One attempt in flight. Reserve (durable and synchronous, before execution: a control operation,
  never volatile) -> execute once at the boundary -> finalize. A reservation or finalization failure fail-stops; an attempt the boundary could not complete is
  abandoned: the reservation stays pending, the runtime fail-stops with `CONTINUITY_EVOLUTION_PENDING`. Nothing
  is retried or inferred: volatile cognition makes no executed attempt retryable. `evolution_reconcile` is the
  explicit operator finalization after restart; its `CONTINUITY_AUTHORITY_MISMATCH` refusal is not a fail-stop.
* **Nothing else.** No history, receipts, events, trajectory or second store: the durable footprint is the two
  176-byte continuity slots, plus, only when the operator provisioned one, the two fixed K1 checkpoint slots.
* **Concurrency.** A handle serializes its HOT and control calls with one lock; the durability lane has its own
  lock, and `durable_wait` / `durable_status` take only that one, so they never wait behind a HOT operation and the
  HOT lane never waits for the worker. No call blocks on another runtime.

## Hot path

A managed `Runtime.run_query` is one Python crossing into RuntimeCore for the query (after the fail-stop probe) and
one K1 crossing (`leased_query`); nothing is hashed and nothing is written.

A warm managed `Runtime.run_turn` (`tests/integration/test_runtime_hot_path.py`, every run, with a checkpoint store
attached): Python makes no K1 call and three crossings into RuntimeCore (the fail-stop probe, `turn_begin`,
`turn_commit`); RuntimeCore makes three K1 crossings (`leased_txn_begin_at`, `leased_txn_run_schedule`,
`leased_txn_commit`): no digest, identity read or capture, no publication; zero continuity and zero checkpoint I/O;
Python performs no file I/O. The unmanaged `elpis.runtime.cognition.run_turn` (no continuity) keeps its own three
K1 crossings plus `max_rows`.

Gate A (HOT isolation): with the durability worker held behind a pending request, 120 managed transitions and 12
queries complete with no continuity or checkpoint I/O (`test_runtime_hot_path.py`), and at the system-call boundary
no write, positional write, sync, rename, unlink, truncate or open-for-write, and no read, positional read or
read-only open, on any thread (`native/runtime/tests/test_runtime_hot_io.c`, an interposer counting per thread;
each durable publication is 4 write-side calls, all on the worker).

### HOT residency

An FMS-resident K1 state lives in FMS: when WARM its image is in memory, but FMS may demote it to the cold store,
and the K1 FMS adapter's ordinary pin materializes a demoted state synchronously (`fms_acquire`: a cold read, a
digest check and an allocation) inside whatever operation touched it. A HOT operation must never do that. The
guarantee is a **residency hold** (additive to the adapter's ABI v1 and to FMS ABI v2; no existing entry point or
struct changed):

* **Binding holds.** Before RuntimeCore binds an FMS-resident state (the first QUERY / LEARN / durable request after
  open or release, and the explicit anchor) and before anything else touches it (the identity precheck included),
  it takes the adapter's hold (`elpis_ecsg_k1_fms_resident_hold`), which succeeds only if the state is WARM now:
  `fms_acquire_resident` pins it where it is and never promotes, reads the cold store or waits for a move.
* **A held state stays resident.** The hold's pin makes the object non-demotable (FMS never demotes a pinned
  object; memory pressure picks other victims or refuses the newcomer), and while any hold exists every operation
  on the state pins resident-only, so a promotion on the HOT path is impossible rather than merely unlikely.
* **Not resident: a precise refusal.** A state that is demoted (or in motion) when it would be bound is refused
  `RUNTIME_NOT_RESIDENT` (84; K1 FMS `-112`): nothing is claimed, read, pinned, promoted or bound, and the runtime
  is not fail-stopped. A concurrent explicit materialization of the same state is refused `ECS_REFUSED` (K1 `BUSY`,
  the adapter's single-writer guard) without waiting.
* **Materialization is WARM work.** The owner promotes the state explicitly (`elpis_ecsg_k1_fms_materialize`,
  Python `K1FMSState.materialize()`), then retries; the lineage resumes (a release keeps its live identity).
* **Lifetime.** RuntimeCore retains the hold's release capability exactly while the state is bound and drops it on
  `release`, `close`, reopen and destruction (and when a bind fails after taking it); the adapter refuses `close`
  of a held state (`BUSY`). A RAM state (standalone K1, composite) needs no hold.

Which *unpinned* WARM state FMS demotes is recency- and frequency-weighted on the wall clock (`fms_core.c`,
`evict_cost_ns`; a release counts as an access), so a released state is not necessarily the next victim and no test
relies on a victim order: every residency precondition is made exact by holding every other WARM state, so that exactly
one state is eligible (CI-HOTRES-1, `docs/CI_INCIDENTS.md`).

Proofs: `runtime.test_runtime_hot_io` (real FMS and POSIX cold store: pressure while bound demotes only the other
arrival, and with nothing else eligible the newcomer is refused while the bound state stays WARM; 40 held turns and
queries with zero read or write calls on any thread and zero promotions; a demoted state's QUERY, LEARN and identity-only
durable request refused with zero I/O, zero promotions and continuity and the lane status byte-for-byte unchanged; only
the explicit materialization reads); `ECS.test_ecsg_k1_fms`
(`test_residency_hold`: already resident, demoted, budget exhaustion, a stalled promotion that blocks neither the
held state nor the caller, a lost cold replica the hold never even reads, concurrent readers during demotion
pressure, nested holds and their lifetime, FMS promotion and cold-read counters); `test_runtime_lifecycle.c` (the
hold across abort, commit, close, destroy, reopen and fail-stop); `tests.rs` (the binding and refusal laws over the
stand-in); `tests/integration/test_hot_residency.py` (the three invariants as separate tests: A residency protection,
B nonresident refusal with nothing promoted, read, pinned, held, published or fail-stopped, C explicit restoration of
the released live lineage). This is mechanics: it says nothing about the latency of a
promotion, which now happens only where an owner asks for it. The Rust suite proves the same over the stand-in and forbids, statically, any store,
lane, sync or digest use in the HOT functions (`tests/boundary/test_ecs_split.py`).

### Legacy surface

The live-coordinate entry points the correction added to K1, the K1 FMS adapter and the ECS composite (and the
residency hold) are additive. Reviewed against the pre-correction tree (`81b738e`):

* **Exports.** Every symbol the pre-correction `libelpis_ecsg_k1`, `libelpis_ecsg_k1_fms`, `libelpis_ecs_composite`,
  `libelpis_ecs_memory` and `libelpis_ecsg_math` export is still exported; the change only adds symbols. No public
  struct changed layout; no status value changed meaning.
* **Source.** K1's envelope check and encoder were split by pure extraction (`check_header` + checksum,
  `encode_body` + trailer: the same checks, order, return codes and bytes); `txn_begin_as` gained an optional
  generation output; the commit path is unchanged. In the FMS adapter and the composite, the new branches are
  reachable only from the new entry points (the legacy leased commit-identity still refuses a NULL identity before
  reaching them); the composite's legacy commit keeps its digest poison check, and its cached K1 generation is
  neither state nor envelope nor identity. `ELPISGK1`, `ELPISEM1` and `ELPISEC1` are written and checked by the
  same code; K1 mathematics is untouched.
* **Behavior.** `native/runtime/composite/tests/test_k1_legacy_differential.c` Part A drives only pre-correction entry points
  (K1 create/learn/forward/query-identity, transactions with commit, commit-identity, abort and the
  experience schedule, the refusal contract, the managed lease, restore and reset; FMS-resident states including a
  demotion and the legacy materializing access; the composite's leased turns with observations; K1's
  retention-vocabulary entry point is left to the K1 milestone's own suites, as the Retention R1/R2/R3 guards
  require of every non-milestone file) and records every
  status, transition, readout, digest and envelope byte. Compiled against the pre-correction headers and linked
  over the pre-correction, the `200f73f`-era and the current libraries, the three transcripts are byte-identical
  (882,398 bytes, SHA-256 `1db0250f9328c150806d7d618bafbbfb58ed611c3576746ce0c9d0fd39bd63de` on the review host).
  In CI the same battery must be deterministic and Part B proves the new entry points against the legacy ones in
  the same build (capture + seal equals `snapshot_write` for all three; `begin_at`/`commit` install the bytes
  `begin`/`commit_identity` install; the generation advances by one per installing mutation and never on a
  refusal; another lease is refused before anything is read). The digest is not asserted in CI: the K1 kernels are
  multi-versioned per ISA, so a transcript is compared across builds of one host, not against a constant.
* **Recorded, unchanged.** The adapter's `close` of a state under a managed K1 lease succeeds (the lease guards
  K1's mutating entry points, not the adapter's lifecycle), as before; a runtime-bound state is now additionally
  protected by the residency hold (`close` refused `BUSY`).

Reproduce: `git worktree add <base> 81b738e`, build it, compile the test against `<base>/native/ECS/include` and
`<base>/native/substrate/include` without `ELPIS_K1_LIVE_COORDINATE`, link the one object against each tree's
libraries (and its `libelpis_fms.a`, `libelpis_hash.a`), run each with a scratch directory and a transcript path,
and compare the transcripts byte for byte.

## Native admission

RuntimeCore is never loaded from a pathname. `RuntimeConfig` names the library by an absolute path and supplies a
deployment-pinned `native_authority` (`PinnedAuthority`, whose own SHA-256 comes from trusted configuration);
`Runtime` admits the library through the substrate's one native admission mechanism
(`elpis.substrate.native_admission`, the mechanism the HACF bridges use): opened beneath its directory as the
trusted descriptor root with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`, copied into a sealed memfd, hashed
against the exact pin and loaded from the seal, under a RuntimeCore library identifier (`elpis_runtime`,
`elpis_runtime_testing`); the ABI version and the compiled fuel ceiling are then checked. Without an authority the
runtime refuses to start (`RUNTIME_UNPINNED`). The continuity ABI is bound over the same admitted bytes
(`ContinuityLibrary` loads nothing). The managed runtime drives only K1 code it admitted: a K1 or K1 FMS state
whose library was not admitted under `elpis_ecsg_k1` / `elpis_ecsg_k1_fms` is refused (`ECS_NATIVE_UNADMITTED`)
before any native call. K1 sets are admitted in dependency order (`elpis_ecsg_math`, `elpis_ecsg_k1`,
`elpis_ecsg_k1_fms`), and every Elpis dependency the dynamic loader binds by SONAME is verified to be pinned bytes
(a sealed object, or a mapped file whose inode and SHA-256 equal the pin); anything else refuses the admission.
Research and unmanaged code may still load K1 by pathname; the managed runtime refuses those states. The pins
authenticate bytes, not a publisher; the loader, kernel and platform libraries are trusted
(`tests/integration/test_native_admission.py`).

## Build and qualification

* `libelpis_runtime.so` (production) and `libelpis_runtime_testing.so` (+ continuity fault injection and I/O
  counters), built offline by CMake through `cargo rustc` (the SONAME is set on the final link only). Both embed the
  continuity and ECS_C authority crates; neither exports `elpis_ecs_c_*` nor links `libelpis_ecs_c_authority.so`.
* `ctest -L runtime`: `runtime.cargo_test` (the Rust law suite, including the durability crash matrix),
  `runtime.cargo_test_ecs_c` (the same suite over the ECS_C authority), `runtime.test_runtime_ecs_c` (ABI v6),
  `runtime.test_runtime_abi` (C ABI over real K1), `runtime.test_runtime_lifecycle` (the turn lifecycle over real
  standalone and FMS-resident K1), `runtime.test_runtime_composite` and `runtime.test_runtime_hot_io` (Gate A at the
  system-call boundary). CI lints and tests the crate in the Rust job and runs both tests in every native job (gcc/clang, Debug/
  Release, ASan+UBSan, TSan).

## Non-claims

RuntimeCore is systems authority, not cognition: it qualifies no codec, makes no semantic claim, and does not
make H-ECS or any hierarchy canonical. It does not authenticate the continuity directory (docs/CONTINUITY.md); the
ECS_C authority authenticates its records under its declared threat model only (docs/ECS_C_AUTHORITY.md). The
durability lane's acknowledgment means durable under POSIX `fdatasync` semantics on the store's filesystem; it is not
a claim about power-loss behaviour of a particular device. Performance figures for the corrected HOT lane are
descriptive measurements on one host, not qualification.
