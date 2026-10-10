# H-ECS R3 successor C2M — fresh DEV 8/8 PASS (2026-10-10)

`RESEARCH_ONLY` · `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE` · `QUAL_RELEASED_ONCE` · `H2_DEV_NON_BINDING` · `NO_RUNTIME_READINESS_CLAIM` · `NO_INTEGRATION`

The frozen matched family C2M ran once, in order, on the fresh never-executed 8-seed DEV registry. It was pre-registered in commit `c14318e` (`../freeze/`). It ran under the C2M supervisor and DEV release `0a705b6f…`. There was no fault, retry or replay.

**Per seed:** TASK, S1 and M passed at every binding and bound horizon on all 8 seeds, with 0 events and planning complete (`seeds/seed_*/C2M_OBSERVER_STREAM.log`).

**Cohort (R3-BP law):** TASK_LOCAL, V2, V3, V4, S1 and M all pass.
- V2 means: 1.0 separated, 1.0 matched.
- V4 means: 29.28 separated, 1.422 matched.
- Disposition `DEV_WORLD_MODEL_VALID_CONDITIONAL_QUAL_ELIGIBLE` (`DEV_DISPOSITION.txt`, `R3_BP_FULL_COHORT_C2M_DEV.log`).

**Margins** (ROLLOUT, minimum over seeds; `BINDING_SUMMARY.tsv`):

| Binding | Training law | Min bound M | Min S1 |
| --- | --- | --- | --- |
| L1/N72 h16 | w_T = 0.125 | 0.9941 | 0.9988 |
| L1/N36 h4 | w_T = 1.0 | 0.9957 | 0.9976 |
| TS/L2 h4 | w_T = 1.0 | 0.9807 | 0.9872 |
| HD/L2 h4 | w_T = 1.0 | 0.9764 | 0.9986 |

Escape is 0 everywhere.

**H2 on DEV (non-binding by protocol; `PLANNING.tsv`, observer line `R3_C2M_H2`).** Mean SEPARATED success at budget indices 0/1/2:

| Condition | b0 | b1 | b2 |
| --- | --- | --- | --- |
| FLAT_N72 | 0.865 | 0.828 | 0.839 |
| TEMPORAL_SHARED_L2 | 0.656 | 0.719 | 0.703 |
| HIERARCHICAL_DISTINCT_L2 | 0.031 | 0.161 | 0.297 |

H2A, H2B and H2C all fail. An independent Python recomputation from `PLANNING.tsv` agrees. These values decide nothing: the hierarchy is adjudicated only on a `WORLD_MODEL_VALID` QUAL.

**Next step:** one QUAL execution on the fresh 12-seed registry. Its release is `QUAL_RELEASE.txt` (`bd20f847…`), issued by the supervisor only after this DEV pass and committed before any QUAL seed ran. The family, thresholds, observer arithmetic, H2 law, bindings and horizons are unchanged. There is no retuning and no rescue.

Raw telemetry is not in Git; each `COMMIT.txt` binds its sha256.
