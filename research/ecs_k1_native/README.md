# Native K1 differential qualification (`research/ecs_k1_native`)

Status: **`NOT_QUALIFIED`**: run once (record `0c1c5543…`). 30 of 32 R3 QUAL worlds pass every comparison.
On r3qual-0019 and r3qual-0028 the native float agreement with the R3 laboratory is outside the planned tolerances.
Every exact comparison holds, and the substituted R3 decision record is OUTCOME_A. The native milestone is
**not promoted**. Results: [docs/research/ECS_K1_NATIVE_RESULTS.md](../../docs/research/ECS_K1_NATIVE_RESULTS.md).

Qualifies the native K1 runtime against the frozen Retention R3 record. Plan: [PLAN.md](PLAN.md); authority:
`specs/ecsg-k1-native.v1.plan.json`. This is a RESEARCH_ONLY harness; it reads the R3 laboratory and records and
never edits them.

* `differential.py`: per-world comparisons and the substitution of the native rows into the recorded rows. The
  comparisons are K1-disabled parity, boundaries, resident path, classes, reset challenge, W-only control,
  transplant and state semantics.
* `run.py`: three commands.
  * `qualify` ran once and wrote `evidence/ecsg-k1-native.v1.qualification.json`.
  * `check` runs on the current runtime and writes no record.
  * `transplant` is the clean child process.
* `diagnose.py`: a post-hoc, descriptive check of how sensitive the R3 laboratory's own trajectory is on the two
  failing worlds.

Harness development (disclosed): after the plan commit, the harness was exercised on R3 QUAL world `r3qual-0000`
(`check --world r3qual-0000`). Its gate substitution was also exercised on the recorded R3 rows, with no native data.
Both checks looked for harness defects before the single qualification run. The plan's tolerances were not changed.
