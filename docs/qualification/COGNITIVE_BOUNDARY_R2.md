# Cognitive Boundary R2: qualification specification (RuntimeCore ABI v4)

Status: **NOT RUN** · Subject: **UNQUALIFIED**

*Superseded before any run:* the product is RuntimeCore ABI v5; its boundary is
[Cognitive Boundary R2 under ABI v5](COGNITIVE_BOUNDARY_R2_V5.md). This specification stays as pre-registered, unrun
authority for ABI v4; nothing of a v4 run would extend to v5.

Successor of [Cognitive Boundary R1](COGNITIVE_BOUNDARY_R1.md), which was pre-registered against RuntimeCore ABI v3
and has a retrospective closure for commit `bec6da7` only ([closure](closures/COGNITIVE_BOUNDARY_R1_CLOSURE_20261011.md);
its evidence is held outside this repository). R1 cannot carry over: ABI v4 changed the subject (volatile commits, the
explicit recoverable / identity-only durability lane, residency holds, composite substrates, typed observations, the
contextual lane). R1's specification and closure stay as written, historical authority for ABI v3; nothing here
reinterprets them.

## Question

Under RuntimeCore ABI v4, does every operation that could change ECS state (K1 `(W, epoch, H, a)` or a composite's
memory), the durable authority (continuity, checkpoint slots), a state's residency, or which native code runs, require
the explicit authority its contract names and refuse everything else before any mutation; and do the HOT, contextual
and durability lanes stay isolated, so that no stalled or failing WARM worker blocks HOT work and no context view
reaches ECS state?

## Subject (frozen at the qualifying commit)

| Law | Implementation | Mechanics tests (current head) |
|---|---|---|
| QUERY is read-only; LEARN only under an explicit `LearnAuthority` | `elpis.runtime.cognition`, RuntimeCore `query` / `turn_*` | `tests/integration/test_query_learn.py` |
| A codec never authorizes itself; QUERY and LEARN capabilities are separate | `elpis.runtime.codec_authority` | `tests/integration/test_codec_authority.py` |
| Total integer fuel before reserve / transaction / mutation | `native/runtime/src/fuel.rs`, `elpis.runtime.fuel` | `tests/integration/test_cognitive_fuel.py` |
| A bound K1, K1 FMS or composite state refuses unmanaged mutation, or the owner fail-stops | K1 and composite leases, RuntimeCore | `test_managed_ownership.py`, `test_composite_runtime.py` |
| Native code runs only from sealed, pinned bytes (RuntimeCore, the K1 set, the composite set, the HACF observation adapter) | `elpis.substrate.native_admission` | `test_native_admission.py`, `test_composite_runtime.py`, `test_hacf_composite_observation.py` |
| Recoverable and identity-only publication are never confused; the acknowledgment law holds at the boundary | `native/runtime/src/durable.rs`, `RuntimeCore._ticket` | `test_k1_recovery.py`, `test_durable_ticket_law.py`, `native/runtime/src/tests.rs` |
| HOT residency: a bound state is held; a demoted one is refused `RUNTIME_NOT_RESIDENT` with nothing promoted, read, held, published or fail-stopped | `core.rs` (`hold`, `reconcile`), the K1 FMS adapter's residency hold | `test_hot_residency.py`, `runtime.test_runtime_hot_io` |
| HOT isolation: no persistent I/O on the HOT thread; HOT never waits for the durability or the contextual worker | `durable.rs`, `elpis.runtime.context` | `test_runtime_hot_path.py`, `runtime.test_runtime_hot_io`, `test_context_lane.py` |
| Memory OBSERVE only under an explicit `ObservationAuthority`, all or nothing inside a LEARN turn; no memory LEARN reachable | `composition.Runtime.run_learn`, `elpis.ECS.composite` | `test_composite_runtime.py`, `test_composite_process_kill.py` |
| No path from a context view to the K1 schedule or the memory | `composition`, `elpis.runtime.context` | `test_context_authority.py` |
| Each finding keeps its regression; Python guards are load-bearing | — | `tests/boundary/test_redteam_regressions.py`, `test_redteam_mutations.py` |

## Inputs

* The qualifying commit, its source digests and the digests of the built libraries (production and testing
  RuntimeCore, the K1 set, the composite set, the HACF observation adapter, continuity), recorded before the run;
  GCC and Clang builds of the same commit.
* The adversarial catalog: every attack of R1's catalog re-targeted to ABI v4, plus this pre-registered extension
  (written before the run):
  * A1 concurrent out-of-band mutation under a managed transaction from a second thread (K1, K1 FMS, composite);
  * A2 a catalog document swapped between admission and use (RuntimeCore, K1 set, composite set, observation adapter);
  * A3 a fuel budget at every field's exact boundary +/- 1;
  * A4 forged durability tickets (every status x recoverable x kind) at the Python boundary;
  * A5 QUERY, LEARN and an identity-only request on a demoted FMS-resident state; and the eight residency mutants of
    CI-HOTRES-1 (`docs/CI_INCIDENTS.md`) as native mutants;
  * A6 an identity-only request with a checkpoint store attached, a recoverable one without, a store attached under a
    pending publication;
  * A7 observations without an authority, from a source it does not admit, beyond its bound, valid and refused mixed in
    one turn, on a K1 substrate; a K1 state for a composite lineage and the reverse; an import of a foreign K1 state or
    of a non-empty memory;
  * A8 a stalled durability worker and a stalled contextual worker under 120 HOT transitions;
  * A9 a populated context view (candidate relations, binding axioms) consumed by a LEARN: the composite must equal a
    twin LEARN without it, byte for byte.
* Sanitizer builds (ASan+UBSan, TSan) of the same commit.

## Procedure

1. Run the mechanics suites (FAST lane; `ctest` with labels runtime, ECS, continuity, substrate; the RuntimeCore Rust
   suite with `--features testing`) on the production and sanitizer builds and both compilers; record every outcome,
   and every skip with its reason.
2. Run the adversarial catalog; for each attack record the refusal code and a byte-for-byte proof that `(W, epoch, H,
   a)`, the memory, the generation, the residency holds, both continuity slots, both checkpoint slots and the
   durability lane's status are unchanged (or the fail-stop disposition where the contract prescribes one).
3. Run the mutation-adequacy suite (STRESS lane) and a pre-registered list of native mutants rebuilt and run once each
   (bounded: at most 16, including the CI-HOTRES-1 residency mutants and the durability ticket-window mutant).

## Gates (pre-registered)

* G1: every catalog attack refused with its contract's code; no attack mutates state outside its contract.
* G2: zero sanitizer findings over the executed, instrumented coverage.
* G3: every Python and native mutant killed.
* G4: HOT-path counters equal `test_runtime_hot_path.py`'s; zero persistent-I/O and zero read calls on the HOT thread
  in `runtime.test_runtime_hot_io`.
* G5: no test of the frozen plan skipped (a test needing an external artifact is either supplied it or excluded by the
  frozen plan before the run, never at run time).

## Would establish / would not establish

Passing would establish that, on the qualifying commit and platform, the listed ABI v4 boundary laws hold against the
listed attacks. It would **not** establish anything about a malicious in-process caller (outside the threat model),
codec semantics (no codec is qualified), the meaning or usefulness of any observation, cognition quality, power-loss
durability, or any other commit.

## Evidence

`research/qualification/cognitive_boundary_r2/evidence/` (write-once; not present: the run has not happened).
