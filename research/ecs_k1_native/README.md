# Native K1 differential qualification (`research/ecs_k1_native`)

Status: **`PLANNED`**: plan and harness committed; the qualification has not run.

Qualifies the native K1 runtime (docs/ECS_K1_RUNTIME.md) against the frozen Retention R3 record. Plan:
[PLAN.md](PLAN.md), authority `specs/ecsg-k1-native.v1.plan.json`. RESEARCH_ONLY harness; it reads the R3
laboratory and records and never edits them.

* `differential.py`: per-world comparisons (K1-disabled parity, boundaries, resident path, classes, reset challenge,
  W-only control, transplant, state semantics) and the substitution of the native rows into the recorded rows.
* `run.py`: `qualify` (once, writes `evidence/ecsg-k1-native.v1.qualification.json`), `check` (current runtime, no
  record), `transplant` (the clean child process).

Harness development (disclosed): after the plan commit, the harness was exercised on R3 QUAL world `r3qual-0000`
(`check --world r3qual-0000`) and its gate substitution on the recorded R3 rows (no native data), to find harness
defects before the single qualification run. The plan's tolerances were not changed.
