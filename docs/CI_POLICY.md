# CI and merge policy

Authority: this document governs when work may merge into `main`. It sits under
`docs/ELPIS_MISSION.md` and changes nothing about Elpis's scientific claims.

## Rules

1. **No merge on red.** A milestone is not closure-grade, and must not be
   merged, while any workflow run for its exact head commit is red. A missing,
   pending or cancelled required run is treated as red.
2. **One green run is not a pass.** A green pull-request run with a red push
   run for the same head is a red head. This happened at Runtime R1: head
   `2799f83` was green in pull-request run 37161299328 and red in push run
   37161274480, it was merged anyway, and `main` stayed red (`61f81e6`, run
   37161622841). It happened again at Retention R2. Pull request #24 (head
   `eb252ba`) was red on its exact head in push run #88 (37265079205) and
   pull-request run #89 (37265083925). It was merged anyway as `af5b4c0`, and
   post-merge run #90 (37265111688) was red. That violated rules 1, 4 and 5.
   History was not rewritten. `main` is corrected forward (corrective RR2-CR0,
   `docs/research/ECS_RETENTION_R2_RESULTS.md`).
3. **Same tree, different verdicts means nondeterminism.** If the same source
   tree yields one green and one red run, classify it as
   NONDETERMINISTIC / ENVIRONMENT-SENSITIVE and investigate before merging. Do
   not pick the green run. Re-running until green is not an investigation.
4. **Scientific and evidence branches** (DEV / freeze / QUAL, authority or
   evidence changes) need green pull-request CI before merge, and that
   includes the Scientific authority job.
5. **Main after merge.** Post-merge CI on `main` must also be green. If it
   fails, corrective work takes priority over new science until `main` is
   green again.

## Required jobs

Boundary · Donor parity · Python 3.11 · Python 3.12 · Python stress and historical ·
Python without network · Continuity and RuntimeCore (Rust) · Native (gcc, Debug) ·
Native (gcc, Release) · Native (clang, Debug) · Native (clang, Release) ·
Native ASan+UBSan · Native TSan · Scientific authority.

## Lanes

Every test has exactly one lane (`tests/lanes.py`, first matching rule), and every
lane has an owning job (`tests/boundary/test_ci_policy.py` proves both):

| Lane | Contents | Owner | Local command |
|---|---|---|---|
| FAST | production contracts, runtime, continuity adapter, boundary, small integration | Python 3.11, Python 3.12 (`pytest --lane fast`) | `pytest` (the default) or `tests/qualify.sh fast` |
| NATIVE | ECS, K1, FMS, continuity, RuntimeCore, H-ECS mechanics, native research mechanics (ctest) | Native gcc/clang Debug/Release | `tests/qualify.sh native` |
| SCIENTIFIC | frozen ECS science, retention and H-ECS evidence, replay | Scientific authority (`pytest --lane scientific`) | `tests/qualify.sh scientific` |
| STRESS | ASan+UBSan and TSan builds; multi-process contention, exhaustive fault matrices, scaling | Native ASan+UBSan, Native TSan; Python stress and historical (`pytest --lane stress`) | `tests/qualify.sh stress` |
| HISTORICAL / RESEARCH | DSV4.1 tower, ECS dynamics laboratory, retained noncanonical model-execution mechanics; donor parity | Python stress and historical (`pytest --lane historical`); Donor parity | `tests/qualify.sh historical` |

A bare `pytest` runs FAST only, so ordinary local qualification never executes the
research corpus by accident. Python without network runs every Python lane
(`--lane all`). Explicit test paths run exactly what they name. Every local step runs
under a hard wall-clock limit with live output. Warm `tests/qualify.sh fast` (native
build check, all ctest, the FAST pytest lane) took 46 s on a 4-core VM, of
which the FAST pytest lane is about 38 s.

No test may assert a wall-clock ratio, sleep to order events, or be re-run until
green. A test that does (for example the latency ratio that once lived in
`research/dsv41_tower/native/tests/test_dsv41_stream_provider.c`) is a
TEST-CONTRACT DEFECT: the measurement is printed, and the asserted law stays the
deterministic one.

These should also be required status checks in the repository's branch
protection for `main`. That setting lives in GitHub, outside this repository.
Until it is set, these rules bind by policy.

## Repairing a red line

Classify every failure before changing anything:

| class | meaning | repair |
|---|---|---|
| IMPLEMENTATION DEFECT | the code is wrong | fix the code; if that touches cognitive mathematics or qualified semantics, stop and report it rather than folding it into a CI repair |
| TEST-CONTRACT DEFECT | the test asks more than the contract it guards (for example bitwise replay outside the recorded profile) | correct the contract; never weaken a test merely to turn CI green |
| CI-ENVIRONMENT DEFECT | the job cannot see what the test needs (for example a shallow checkout for a chronology gate) | give the job what it needs |
| EVIDENCE-PROVENANCE DEFECT | a record's binding or bookkeeping is incomplete or wrong | document it next to the evidence; never rewrite write-once evidence |
| SCIENTIFIC FAILURE | a pre-registered gate fails | keep the failure and report it; a repair is a new experiment version |

Rules for any repair:
- Write-once DEV, frozen and QUAL evidence is never edited, re-run or
  regenerated to make CI green.
- Bitwise historical replay is demanded only under the environment the
  evidence recorded. Beyond that, a non-skippable current-runtime regression
  decides whether the science still holds. Executor-versus-reference parity
  stays bitwise, on the host running the tests.
- Experiments record effective runtime properties (BLAS kernel, effective
  thread count, CPU, native binary digest), not only environment strings.
  The v1 Cognitive R0 and Retention R0 records do not, and their limits are
  documented in `docs/research/COGNITION_R0_RESULTS.md` and
  `docs/research/ECS_RETENTION_R0_RESULTS.md` ("Reproduction contract").
- Jobs that run evidence-chronology gates check out full history
  (`fetch-depth: 0`). `tests/boundary/test_ci_policy.py` enforces this.
