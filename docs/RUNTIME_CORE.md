# RuntimeCore: the runtime's systems authority in Rust

`native/runtime` (crate `elpis_runtime`, C ABI `include/elpis/runtime.h` v1) owns the mutable systems
authority of an open Elpis runtime. `elpis.runtime.Runtime` is its compatibility facade.

## What RuntimeCore owns

| Authority | Before (Python `composition.py`) | Now (RuntimeCore, Rust) |
|---|---|---|
| Lifecycle (open / close / reopen) | `Runtime.open` / `close` around a `ContinuityStore` | `Core::open` / `close`; the continuity store is embedded |
| Fail-stop disposition | `Runtime._continuity_fault` | `Core.fault` (a continuity code); every lineage- or authority-dependent call refuses with it until reopen |
| Continuity store | a Python `ContinuityStore` handle | the `elpis_continuity::Store`, embedded as a crate (one implementation; the library also exports the continuity C ABI) |
| K1 lineage binding | `Runtime._turn_substrate` and `_reconcile_cognition_substrate` | `Core.bound` (`Key`: kind, native handle, resident id, owner identity, dimension), verified once against the durable expected identity |
| Managed turn's native transaction | Python sequenced `txn_begin`, `run_schedule`, `commit_identity`, abort | `turn_begin` (lineage check, begin, one schedule, cold-path capacity growth) and `turn_commit` / `turn_abort` |
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
`txn_run_schedule`, `txn_commit_identity`, `txn_abort`); the Python adapter takes their addresses from the
loaded library, a C or C++ host passes them directly (`native/runtime/tests/test_runtime_abi.c`). The declared
dimension is verified natively: K1 compares the readout length it implies with the state's own before it reads
any input byte (`schedule_check`).

## Laws

Each law is a test in `native/runtime/src/tests.rs` (deterministic in-memory K1 stand-in, continuity fault
injection) and, over real native K1, in `native/runtime/tests/test_runtime_abi.c` and `tests/integration`.

* **Anchor.** The first managed lineage is anchored explicitly at the state's retained identity; RuntimeCore
  reads the identity only. A second anchor is `CONTINUITY_ALREADY_ANCHORED`.
* **Bind and verify.** The first turn after open binds the lineage to the supplied state after comparing its
  identity with the durable expectation: unanchored is `CONTINUITY_UNANCHORED` (no fail-stop), a mismatch is
  `CONTINUITY_STATE_MISMATCH` and fail-stops; both before any K1 mutation. A bound runtime refuses any other
  state, by handle or owner, with `COGNITION_SUBSTRATE_SWITCH` and no native call. A warm turn reads no
  identity: the binding holds and continuity verifies `before` at publication.
* **Turn.** begin -> one native schedule on the candidate -> (boundary decodes) -> native commit -> one
  publication. Capacity beyond the reservation is grown on the cold path (abort, `reserve`, begin again).
  Every refusal before the commit leaves `(W, epoch, H, a)` byte-for-byte unchanged; a stale source is
  `ECS_STALE`.
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

A warm managed `Runtime.run_turn` (`tests/integration/test_runtime_hot_path.py`, every run): Python makes no K1
call and three crossings into RuntimeCore (the fail-stop probe, `turn_begin`, `turn_commit`); RuntimeCore makes
three K1 crossings (`txn_begin`, `txn_run_schedule`, `txn_commit_identity`) and one publication; the store
writes one 176-byte record and syncs once; Python performs no file I/O. The unmanaged
`elpis.runtime.cognition.run_turn` (no continuity) keeps its own three K1 crossings plus `max_rows`.

## Build and qualification

* `libelpis_runtime.so` (production) and `libelpis_runtime_testing.so` (+ continuity fault injection and I/O
  counters), built offline by CMake through `cargo rustc` (the SONAME is set on the final link only).
* `ctest -L runtime`: `runtime.cargo_test` (the Rust law suite) and `runtime.test_runtime_abi` (C ABI over real
  K1). CI lints and tests the crate in the Rust job and runs both tests in every native job (gcc/clang, Debug/
  Release, ASan+UBSan, TSan).

## Non-claims

RuntimeCore is systems authority, not cognition: it qualifies no codec, makes no semantic claim, and does not
make H-ECS or any hierarchy canonical. It does not authenticate the continuity directory (docs/CONTINUITY.md).
