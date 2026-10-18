# Cognitive Boundary R2 under RuntimeCore ABI v5: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED**

Successor of [Cognitive Boundary R2](COGNITIVE_BOUNDARY_R2.md), which was pre-registered against RuntimeCore ABI v4
and never run. The product is now ABI v5: the budget and counters changed layout, the observe-only managed turn is a new
mutation path into a composite's memory, and v4 callers are refused. A boundary qualified for ABI v4 would not extend
to ABI v5, and R2 is not stretched to cover it: R2 stays as written (unrun, ABI v4 authority), and this is the
separately versioned specification of the ABI v5 boundary. It names its own subject, catalog, inputs and gates, and
it qualifies only its own frozen commit.

## Question

Under RuntimeCore ABI v5, does every operation that could change ECS state (K1 `(W, epoch, H, a)` or a composite's
memory), the durable authority (continuity, checkpoint slots), a state's residency, or which native code runs, require
the explicit authority its contract names and refuse everything else before any mutation, including the observe-only
turn, which may change a composite's memory and nothing of K1; and do the HOT, contextual and durability lanes stay
isolated?

## Subject (frozen at the qualifying commit)

Every law of [R2](COGNITIVE_BOUNDARY_R2.md) ("Subject"), as ABI v5 implements it, plus the laws ABI v5 introduced:

| Law (ABI v5) | Implementation | Mechanics tests (current head) |
|---|---|---|
| Observe-only transaction authority: `elpis_runtime_turn_begin_observe` admits only a composite substrate under a composite lineage; a K1 or K1 FMS substrate, or a K1 lineage, is refused before any native turn | `native/runtime/src/core.rs` `turn_begin_observe`, `Runtime.run_observe` | `tests.rs` `observe_only`, `test_observe_only.py::test_a_k1_lineage_refuses_the_observe_only_turn_natively` |
| Memory mutation without K1 learning: an observe-only commit keeps the K1 component byte-identical (bytes and identity), runs no schedule (`k1_run_schedules` unchanged), advances the live generation by one; a commit that reports K1 learning fail-stops | `core.rs` `turn_commit` | `test_observe_only.py::test_an_observe_only_turn_records_evidence_and_leaves_k1_byte_identical`, `tests.rs` `observe_only` |
| An observe-only commit with nothing observed is refused (`RUNTIME_INVALID`), the turn stays open | `core.rs` | `tests.rs` `observe_only::a_commit_with_nothing_observed_is_refused_and_the_turn_stays_open` |
| `budget.max_observations`: at most 256 per turn (the ceiling); each observation is fuel-checked before its native call (`COGNITION_FUEL_EXCEEDED`), recoverable | `fuel.rs`, `core.rs` `turn_observe` | `tests.rs` `observe_only::the_observation_bound_is_fuel_checked_before_the_native_call_and_recoverable`, `test_observe_only.py::test_the_native_observation_bound_is_the_authority_bound` |
| Typed structural evidence enters memory only through `record_evidence` / an explicit `ObservationAuthority`, all or nothing | `elpis.runtime.observation` | `test_structural_evidence.py`, `test_hacf_composite_observation.py` |
| ABI compatibility: budget 56 B, counters 136 B, `ELPIS_RUNTIME_ABI_V5`; the envelope (ELPISGK1, ELPISEC1, ELPISEM1) and continuity record v3 formats unchanged | `ffi.rs`, `runtime.h`, `elpis.runtime.core` | `runtime.test_runtime_abi`, `test_runtime_hot_path.py` |
| Obsolete callers are refused: a caller built for ABI v4 (version check), an ABI v4 RuntimeCore library, an ABI 4 deployment | `elpis_runtime_abi_version`, `RuntimeLibrary.admit`, `elpis.runtime.deployment` | acceptance N07 (`elpis.runtime.acceptance`), `test_cli.py` pin drift (abi) |

## Inputs (written before the freeze)

* The qualifying commit (the accepted source, never a branch head the merge would replace), its source digests and the
  digests of the built libraries (production and testing RuntimeCore, the K1 set, the composite set, the HACF
  observation adapter, continuity); GCC and Clang Release builds and ASan+UBSan and TSan builds of the same commit.
* The adversarial catalog: R2's A1-A9, each re-targeted to ABI v5 (A1 and A5 include the composite under an
  observe-only turn), plus:
  * A10 an observe-only turn on a K1 substrate, a K1 FMS substrate and under a K1 lineage;
  * A11 an observe-only commit with nothing observed; observations after a refused one in the same turn;
  * A12 `max_observations` at 255, 256 and 257 (and a budget narrowing it to 0, 1); the 257th refused before the
    native call;
  * A13 an observe-only commit that reports K1 learning (native mutant), K1 bytes and identity before and after;
  * A14 an ABI v4 caller (budget and counters of the v4 layout), an ABI v4 RuntimeCore library (a stub built for the
    run), an ABI 4 deployment record;
  * A15 a context view consumed by an observe-only turn: the composite must equal a twin turn without it, byte for
    byte.
* Every attack names, before the freeze, the test node(s) that execute it and the exact refusal code and the state it
  proves unchanged (`W, epoch, H, a`, memory, generation, residency holds, both continuity slots, both checkpoint
  slots, the durability lane's status). Where no mechanics test executes an attack today (A1 for the composite under
  an observe-only turn; A2 between admission and use for each library set; A12 at 255 and 257 through the public
  API; A15), the instrument adds it before the freeze; adding it is a prerequisite, not something the run may
  improvise.
* The native mutant list (at most 16): the eight CI-HOTRES-1 residency mutants, the durability ticket-window mutant,
  and the seven observe-only mutants recorded in docs/RUNTIME_CORE.md ("Observe-only turns (ABI v5)").

## Procedure

1. Freeze: the instrument binds the executing object as K1 Recovery R0 v4's does (commit, tree, cleanliness,
   instrument, interpreter and imported package, libraries and deployed copies, toolchain, platform), before the run.
2. Run the mechanics suites (FAST lane; `ctest` labels runtime, ECS, continuity, substrate; the RuntimeCore Rust suite
   with `--features testing`) on the production and sanitizer builds and both compilers; record every outcome and
   every skip with its reason.
3. Run the catalog; record each attack's refusal code and the byte-for-byte proof of unchanged state (or the
   fail-stop disposition its contract prescribes).
4. Run the mutation-adequacy suite (STRESS lane) and each registered native mutant once.

## Gates (pre-registered)

* G1: every catalog attack refused with its contract's code; no attack mutates state outside its contract.
* G2: zero sanitizer findings over the executed, instrumented coverage.
* G3: every Python and native mutant killed.
* G4: HOT-path counters equal `test_runtime_hot_path.py`'s (with `k1_observes`); zero persistent-I/O and zero read
  calls on the HOT thread in `runtime.test_runtime_hot_io`.
* G5: no test of the frozen plan skipped.
* G6: an observe-only turn never changes the K1 component (bytes and identity) in any catalog run.

## Would establish / would not establish

Passing would establish that, on the qualifying commit and platform, the listed ABI v5 boundary laws hold against the
listed attacks. It would **not** establish anything about a malicious in-process caller (outside the threat model),
codec semantics (no codec is qualified: `ECS_CODEC_UNQUALIFIED`), the meaning or usefulness of any observation,
memory LEARN (never enabled), cognition quality, power-loss durability, or any other commit. It would not qualify ABI
v4 (R2) or ABI v3 (R1).

## Evidence

`research/qualification/cognitive_boundary_r2_v5/evidence/` (write-once; not present: the run has not happened).
