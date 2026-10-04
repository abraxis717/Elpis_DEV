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
   37161622841).
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

Boundary · Donor parity · Python 3.11 · Python 3.12 · Python without network ·
Native (gcc, Debug) · Native (gcc, Release) · Native (clang, Debug) ·
Native (clang, Release) · Native ASan+UBSan · Native TSan · Scientific authority.

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
