# H-ECS R3 successor C2 — fresh DEV 8/8 PASS (2026-10-09)

`RESEARCH_ONLY` · `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE` · `QUAL_RELEASED_ONCE` · `NO_R4` · `NO_INTEGRATION`

The frozen candidate C2 (`../freeze/`, pre-registered in commit `8a98d9c`) ran once, in order, on the fresh never-executed 8-seed DEV registry, under the C2 supervisor and DEV release `1abb946f…`. There was no fault, retry or replay.

**Per seed:** TASK, S1 and M passed at every binding and bound horizon on all 8 seeds (`seeds/seed_*/C2_OBSERVER_STREAM.log`).

**Cohort (R3-BP law):** TASK_LOCAL, V2, V3, V4, S1 and M all pass.
- V2 means: 0.9948 separated, 0.9948 matched.
- V4 means: 32.26 separated, 1.581 matched.
- Disposition `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE` (`DEV_DISPOSITION.txt`, `R3_BP_FULL_COHORT_C2_DEV.log`).

**Margins:**
- The minimum bound HD/L2 ROLLOUT M capture per seed is 0.9926–1.0089, against a frozen threshold of 0.90.
- The minimum bound capture over all bindings and seeds is 0.9830.
- Escape is 0 everywhere.
- Details are in `HD_L2_SUMMARY.tsv`.

**Next step:** one QUAL execution on the reserved 12-seed held-out registry. Its release is `QUAL_RELEASE.txt` (`d29b0622…`), issued by the supervisor only after this DEV pass and committed before any QUAL seed ran. Candidate, thresholds, observer arithmetic, bindings and horizons are unchanged. There is no retuning and no rescue.

The DEV pass is conditional QUAL eligibility only. C2 is not compute-matched; no superiority claim. Raw telemetry is not in Git; each `COMMIT.txt` binds its sha256.
