# Production Python authority census (R2, after Rust continuity)

Base: `c5e619a` (main after PRs #36, #37, #38). This census drives the migration on this branch. It traces the
live production path from its entry points, `elpis.runtime.Runtime` (composition) and the canonical turn
(`Runtime.run_turn`), through every subsystem it composes. Each surface is classified as:

| Class | Meaning |
|---|---|
| `PRODUCTION_AUTHORITY` | Python object owning mutable production state, durable state, lifecycle, fail-stop or concurrency authority |
| `PRODUCTION_HOT_PATH` | Python executing per-turn work on the canonical path (beyond marshalling) |
| `PRODUCTION_ADAPTER` | stateless or handle-only Python over a native authority (`ctypes` marshalling, admission checks) |
| `BOUNDARY_ONLY` | validation, codec and immutable value contracts at the language boundary |
| `RESEARCH_ONLY` / `TEST_ONLY` / `OFFLINE_TOOLING` | never on the production path |

The migration question for each authority surface: does it own mutable authority, sit on the per-turn hot path,
manage resources or lifetimes, materially affect latency, and would a native owner simplify rather than
duplicate authority?

## The canonical runtime path

| Surface | Before this branch | After |
|---|---|---|
| `runtime/composition.py` `Runtime` | **PRODUCTION_AUTHORITY**: open/close lifecycle; `_continuity_fault` fail-stop flag; `_turn_substrate` K1 lineage binding and `_reconcile_cognition_substrate`; continuity composition (a `ContinuityStore` it opened and drove); evolution reservation, execution bracketing, finalization and fault law; turn orchestration (K1 transaction sequencing, then publication, then fault) | **PRODUCTION_ADAPTER**: a facade over RuntimeCore (`native/runtime`, Rust). It keeps no fault, binding, transaction or authority value; only a reference keeping the bound K1 owner alive. Deleted: `_continuity_fault`, `_turn_substrate`, `_reconcile_cognition_substrate`, `_snapshot`, `_k1_identity`, every continuity call |
| `runtime/core.py` (new) | - | **PRODUCTION_ADAPTER**: `ctypes` marshalling into RuntimeCore; builds the K1 function table from the loaded K1 library's own entry points |
| `runtime/cognition.py` | **BOUNDARY_ONLY** + unmanaged turn sequencing | unchanged role: request validation, codec encode/decode (`_encode`, `_decode`, shared with the managed turn), `Stimulus` admission. The unmanaged `run_turn` (no continuity, no authority) still sequences the native K1 transaction itself (residual, below) |
| `runtime/edges.py` | BOUNDARY_ONLY (pure adapters) | unchanged |
| `continuity/adapter.py` | PRODUCTION_ADAPTER over Rust continuity (PR #37) | unchanged; the runtime no longer uses it (RuntimeCore embeds the store); kept for direct continuity access and its frozen-vector tests |
| `ECS/k1.py`, `ECS/native.py`, `ECS/residency.py` | PRODUCTION_ADAPTER (handle wrappers; one native call per method; native owns `(W, epoch, H, a)` and transactions) | unchanged except that `K1Library`, `K1State` and `K1FMSRuntime` expose the loaded library for RuntimeCore's function table |
| `ECS/cognition.py` (Cognitive R0 core) | PRODUCTION_ADAPTER over the native executor (qualified R0 surface, not the canonical turn) | unchanged |
| `evolution/path_gate.py` | BOUNDARY_ONLY: immutable assertion/receipt values, digest identities, admission checks; the gate holds only its constructor configuration | unchanged (see Evolution) |
| `inference/text.py`, `admission.py`, `contracts.py`, `structural.py` | BOUNDARY_ONLY (the DSV4 token codec and budgeted rendering) | unchanged |

## Other production subsystems

| Surface | Class | Decision |
|---|---|---|
| `pipeline/canonical/publisher.py`, `authority.py`, `candidate.py`; `pipeline/application/durable_ledger*.py` | **PRODUCTION_AUTHORITY** (durable): the canonical Grid81 publisher owns its serialized, recoverable publication and ledger; the application ledgers are file-backed | Not migrated in this batch. They are rare administrative operations (one-use promotion capability), not on the per-turn path, and their durable formats are frozen by their own tests. They are the highest-value remaining Python durable authority (RESIDUAL_PYTHON_PRODUCTION_SURFACE) |
| `substrate/file_assets.py` (`FMSFileAssets`) | **PRODUCTION_AUTHORITY** (resource): Python `RLock`, asset map, page LRU and pin table over native FMS pages | Not migrated: its only consumers are the retained, noncanonical DSV model-execution mechanics (`inference/rows.py`, `experts.py`) and synthetic fixtures; the canonical turn never constructs it. Residual |
| `substrate/boundary.py` | PRODUCTION_ADAPTER (descriptor capabilities; a process-wide native-load lock) | keep |
| `substrate/execution.py`, `residency.py`, `authority.py` | PRODUCTION_ADAPTER / BOUNDARY_ONLY | keep |
| `pipeline/ingress.py`, `structure/retrieval/hacf.py` | PRODUCTION_ADAPTER (ctypes over native bridges) | keep |
| `pipeline/adjudication`, `capability`, `consumption`, `promotion`; `structure/grid81/*`; `structure/retrieval/*` (except hacf) | BOUNDARY_ONLY: immutable records, deterministic compilers, digests | keep: no mutable authority, not on the per-turn path |
| `evolution/*` other than `path_gate` (genotype, mutation, selection, reproduction, lineage, fitness, organism, promotion) | BOUNDARY_ONLY values plus `promotion.py`, which writes a child workspace once at a caller-supplied new destination | keep: immutable values; reproduction and selection commit against an exact observed revision and return new values; no process-held mutable authority |
| `inference/transaction.py`, `sequence.py`, `steering.py`, `steered.py`, `principal.py`, `speculative.py`, `experts.py`, `rows.py`, `target.py`, `drivers/*` | retained noncanonical model-execution mechanics (mission-gated: the runtime composes none of them) | keep out of the production path; no migration (no production consumer) |
| `research/**`, `tests/**` | RESEARCH_ONLY / TEST_ONLY | - |

## Evolution

`EvolutionPathGate` is immutable validation and identity: assertion v1 and receipt v0 payloads, their digest
domains, the retired-v0 refusal, the trusted-predecessor law, the edit/resource/evaluation bindings and the
stale-authority check, all as pure functions of immutable values. The **mutable** part of evolution, the durable
reservation before execution, the one bounded attempt in flight, the finalization and the fail-stop when either
publication is not certain or the attempt could not complete, moved into RuntimeCore with the rest of the
runtime's authority. Rewriting the gate in Rust would duplicate its digest identities in a second language
without removing any authority, and would put persisted identities at risk for no latency gain (an evolution
attempt is not a hot path). The gate therefore remains a Python boundary module; its identities are unchanged.

## Canonical inference path

| | Path |
|---|---|
| Before | Python `Runtime.run_turn`: validate -> reconcile lineage (Python state) -> encode -> Python sequences `txn_begin` / `run_schedule` -> decode -> Python `commit_identity` -> Python `ContinuityStore.commit_cognition_transition` -> Python fault flag |
| After | Python facade: validate -> encode (boundary) -> **RuntimeCore** `turn_begin` (lineage check, native transaction, one schedule) -> decode (boundary) -> **RuntimeCore** `turn_commit` (native commit, continuity publication, fail-stop) |

No semantic codec exists before or after: the turn still refuses with `ECS_CODEC_UNQUALIFIED` without an
explicitly supplied map.

## Residual Python production surface

1. The canonical Grid81 publisher and the application ledgers (durable authority; not per-turn).
2. `FMSFileAssets` page bookkeeping (resource authority; noncanonical consumers only).
3. The unmanaged `cognition.run_turn` sequences three native K1 calls itself (no durable state or fail-stop;
   the managed turn is RuntimeCore's). Folding it into RuntimeCore would need a runtime-library path for every
   unmanaged caller; recorded as debt.
4. The evolution gate (boundary validation, by decision above).
