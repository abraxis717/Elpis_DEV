# Cognitive Boundary R1: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED**

## Question

Does the managed cognitive boundary hold under adversarial use: does every operation that could change K1 state,
continuity, or which code runs, require the explicit authority its contract names, and refuse everything else before
any mutation?

## Subject (frozen at the qualifying commit)

| Law | Implementation | Mechanics tests |
|---|---|---|
| QUERY is read-only; LEARN needs explicit authority | `elpis.runtime.cognition`, RuntimeCore `query` / `turn_*` | `tests/integration/test_query_learn.py` |
| A codec never authorizes itself | `elpis.runtime.codec_authority` | `tests/integration/test_codec_authority.py` |
| Total integer fuel before reserve / transaction / mutation | `native/runtime/src/fuel.rs`, `elpis.runtime.fuel` | `tests/integration/test_cognitive_fuel.py` |
| Unmanaged mutation of a bound state is refused or fail-stops | K1 lease (the K1 header's managed ownership), RuntimeCore | `tests/integration/test_managed_ownership.py` |
| Native code runs only from sealed, pinned bytes | `elpis.substrate.native_admission` | `tests/integration/test_native_admission.py` |
| Each finding keeps its regression; Python guards are load-bearing | — | `tests/boundary/test_redteam_regressions.py`, `test_redteam_mutations.py` |

## Inputs

* The qualifying commit, its source digests and the digests of the built libraries (`libelpis_runtime.so`, the K1
  set, continuity), recorded before the run.
* The adversarial catalog: every attack in the mechanics tests above, plus a pre-registered extension written
  before the run (at least: concurrent out-of-band mutation under a managed transaction from a second thread; a
  catalog document swapped between admission and use; a fuel budget at every field's exact boundary +/- 1).
* Sanitizer builds (ASan+UBSan, TSan) of the same commit.

## Procedure

1. Run the mechanics suites (FAST lane and `ctest -L runtime -L ECS -L continuity`) on the production and sanitizer
   builds; record outcomes.
2. Run the adversarial catalog; for each attack record the refusal code and a byte-for-byte proof that `(W, epoch,
   H, a)`, the generation and both continuity slots are unchanged (or the fail-stop disposition where the contract
   prescribes one).
3. Run the mutation-adequacy suite (STRESS lane) and, for native guards, a pre-registered list of native mutants
   rebuilt and run once each (bounded: at most 12 native mutants).

## Gates (pre-registered)

* G1: every catalog attack refused with its contract's code; no attack mutates state outside its contract.
* G2: zero sanitizer findings.
* G3: every Python and native mutant killed.
* G4: hot-path counters equal `tests/integration/test_runtime_hot_path.py`'s on the qualifying build.

## Would establish / would not establish

Passing would establish that, on the qualifying commit and platform, the listed laws hold against the listed
attacks. It would **not** establish anything about a malicious in-process caller (outside the threat model), about
codec semantics (no codec is qualified), about cognition quality, or about any other commit.

## Evidence

`research/qualification/cognitive_boundary_r1/evidence/` (write-once; not present: the run has not happened).
