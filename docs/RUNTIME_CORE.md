# RuntimeCore: the runtime's systems authority in Rust

`native/runtime` (crate `elpis_runtime`, C ABI `include/elpis/runtime.h` v3) owns the mutable systems
authority of an open Elpis runtime. `elpis.runtime.Runtime` is its compatibility facade.

## What RuntimeCore owns

| Authority | Before (Python `composition.py`) | Now (RuntimeCore, Rust) |
|---|---|---|
| Lifecycle (open / close / reopen) | `Runtime.open` / `close` around a `ContinuityStore` | `Core::open` / `close`; the continuity store is embedded |
| Fail-stop disposition | `Runtime._continuity_fault` | `Core.fault` (a continuity code); every lineage- or authority-dependent call refuses with it until reopen |
| Continuity store | a Python `ContinuityStore` handle | the `elpis_continuity::Store`, embedded as a crate (one implementation; the library also exports the continuity C ABI) |
| K1 lineage binding | `Runtime._turn_substrate` and `_reconcile_cognition_substrate` | `Core.bound` (`Key`: kind, native handle, resident id, owner identity, dimension), verified once against the durable expected identity |
| Managed QUERY (read-only) | none (the turn was the only cognitive operation) | `query`: lineage check and one native `query_identity` (the answer `f_W(x)` and the identity of the state that computed it); no transaction, no commit, no publication |
| Managed LEARN (the turn's native transaction) | Python sequenced `txn_begin`, `run_schedule`, `commit_identity`, abort | `turn_begin` (lineage check, begin, one schedule, cold-path capacity growth) and `turn_commit` / `turn_abort` |
| Continuity publication of a committed turn | Python `commit_cognition_transition` + fault flag | `turn_commit` publishes or fail-stops; the refusal still reports the standing K1 commit |
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

`api` is a table of the K1 library's own functions (`state_digest`, `reserve`, `txn_begin`,
`txn_run_schedule`, `txn_commit_identity`, `txn_abort`, `query_identity`, `shape`); the Python adapter takes their addresses from the
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

## Laws

Each law is a test in `native/runtime/src/tests.rs` (deterministic in-memory K1 stand-in, continuity fault
injection) and, over real native K1, in `native/runtime/tests/test_runtime_abi.c`,
`native/runtime/tests/test_runtime_lifecycle.c` and `tests/integration`.

* **Anchor.** The first managed lineage is anchored explicitly at the state's retained identity; RuntimeCore
  reads the identity only. A second anchor is `CONTINUITY_ALREADY_ANCHORED`.
* **Bind and verify.** The first turn after open binds the lineage to the supplied state after comparing its
  identity with the durable expectation: unanchored is `CONTINUITY_UNANCHORED` (no fail-stop), a mismatch is
  `CONTINUITY_STATE_MISMATCH` and fail-stops; both before any K1 mutation. A bound runtime refuses any other
  state, by handle or owner, with `COGNITION_SUBSTRATE_SWITCH` and no native call. A warm turn reads no
  identity: the binding holds and continuity verifies `before` at publication.
* **QUERY.** Read-only (docs/COGNITION_R0.md). One native call (`query_identity`) answers `f_W(x)` from the
  authoritative state and returns that same state's retained-state identity under one K1 guard, so no operation
  can move the state between the answer and its identity. The identity must equal the durable expected identity:
  unanchored is `CONTINUITY_UNANCHORED`; any other state, including the bound one moved out of band, is
  `CONTINUITY_STATE_MISMATCH` (fail-stop) and the answer is withheld (zeroed). A query is refused while a managed
  turn is open (`RUNTIME_TURN_OPEN`). It begins no transaction, commits nothing and publishes nothing: W, epoch,
  H, a, the generation and both continuity slots are byte-for-byte unchanged (`tests/integration/test_query_learn.py`).
  ABI v3 added the entry to both K1 tables and the `k1_queries` counter.
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
* **Turn (LEARN).** begin -> one native schedule on the candidate -> (boundary decodes) -> native commit -> one
  publication. Capacity beyond the reservation is grown on the cold path (abort, `reserve`, begin again).
  Every refusal before the commit leaves `(W, epoch, H, a)` byte-for-byte unchanged; a stale source is
  `ECS_STALE`.
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
* **Crash law.** A K1 commit is never rolled back. A refused or uncertain publication fail-stops; restart sees
  the old authority (the moved state is a mismatch) or, when the uncertain write landed, the new one.
* **Evolution.** One attempt in flight. Reserve (durable, before execution) -> execute once at the boundary ->
  finalize. A reservation or finalization failure fail-stops; an attempt the boundary could not complete is
  abandoned: the reservation stays pending, the runtime fail-stops with `CONTINUITY_EVOLUTION_PENDING`. Nothing
  is retried or inferred. `evolution_reconcile` is the explicit operator finalization after restart; its
  `CONTINUITY_AUTHORITY_MISMATCH` refusal is not a fail-stop.
* **Nothing else.** No history, receipts, events, trajectory or second store: the durable footprint is the two
  176-byte continuity slots.
* **Concurrency.** A handle serializes its calls with one lock; no call blocks on another runtime.

## Hot path

A managed `Runtime.run_query` is one Python crossing into RuntimeCore for the query (after the fail-stop probe) and
one K1 crossing (`query_identity`); the store does no I/O.

A warm managed `Runtime.run_turn` (`tests/integration/test_runtime_hot_path.py`, every run): Python makes no K1
call and three crossings into RuntimeCore (the fail-stop probe, `turn_begin`, `turn_commit`); RuntimeCore makes
three K1 crossings (`txn_begin`, `txn_run_schedule`, `txn_commit_identity`) and one publication; the store
writes one 176-byte record and syncs once; Python performs no file I/O. The unmanaged
`elpis.runtime.cognition.run_turn` (no continuity) keeps its own three K1 crossings plus `max_rows`.

## Build and qualification

* `libelpis_runtime.so` (production) and `libelpis_runtime_testing.so` (+ continuity fault injection and I/O
  counters), built offline by CMake through `cargo rustc` (the SONAME is set on the final link only).
* `ctest -L runtime`: `runtime.cargo_test` (the Rust law suite), `runtime.test_runtime_abi` (C ABI over real
  K1) and `runtime.test_runtime_lifecycle` (the turn lifecycle over real standalone and FMS-resident K1). CI lints and tests the crate in the Rust job and runs both tests in every native job (gcc/clang, Debug/
  Release, ASan+UBSan, TSan).

## Non-claims

RuntimeCore is systems authority, not cognition: it qualifies no codec, makes no semantic claim, and does not
make H-ECS or any hierarchy canonical. It does not authenticate the continuity directory (docs/CONTINUITY.md).
