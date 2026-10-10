# H-ECS R3 successor C2M — fresh QUAL 12/12 `WORLD_MODEL_VALID`; hierarchy adjudicated `NO_ADVANTAGE_OVER_WIDTH` (2026-10-10)

`RESEARCH_ONLY` · `WORLD_MODEL_VALID` · `HIERARCHY_ADJUDICATED` · `NO_ADVANTAGE_OVER_WIDTH` · `INTEGRATION_AUTHORIZED=NO` · `NO_RUNTIME_READINESS_CLAIM`

The frozen matched family C2M ran once, in order, on the fresh never-executed 12-seed QUAL registry.
- Pre-registration: `../freeze/`, commit `c14318e`.
- DEV: 8/8 pass, `../dev/`, commit `b89c474`.
- Release: QUAL release `bd20f847…`, committed before any QUAL seed ran.

There was no fault, retry or replay. This is the only QUAL execution of C2M.

## Validity (frozen R3-BP law, unchanged)

**Per seed:** TASK, S1 and M passed at every binding and bound horizon on all 12 seeds (48/48 binding × seed gates), with 0 events and planning complete.

**Cohort:** TASK_LOCAL, V2, V3, V4, S1 and M all pass.
- V2 means: 0.9931 separated, 0.9931 matched.
- V4 means: 29.57 separated, 1.376 matched.
- Disposition **`WORLD_MODEL_VALID`** (`QUAL_DISPOSITION.txt`, `R3_BP_FULL_COHORT_C2M_QUAL.log`).

**Margins** (ROLLOUT, minimum over seeds; `BINDING_SUMMARY.tsv`):

| Binding | Min bound M | Min S1 |
| --- | --- | --- |
| L1/N72 h16 | 0.9958 | 0.9971 |
| L1/N36 | 0.9948 | 0.9969 |
| TS/L2 | 0.9760 | 0.9767 |
| HD/L2 | 0.9661 | 0.9797 |

Escape is 0 everywhere.

Every constituent consumed by the hierarchy comparison is therefore a qualified world model trained by the same C2-derived learning law. **The H2 comparison is attributable to hierarchy, not to unequal training.**

## Hierarchy (R2 H2 law, unchanged; adjudicated because QUAL is `WORLD_MODEL_VALID`)

Mean SEPARATED planning success over the 12 seeds, at matched planning compute (256 / 1024 / 4096 forward rows per replan in every condition):

| Condition | b0 | b1 | b2 |
| --- | --- | --- | --- |
| FLAT_N72 | 0.854 | 0.837 | 0.840 |
| TEMPORAL_SHARED_L2 | 0.646 | 0.628 | 0.698 |
| HIERARCHICAL_DISTINCT_L2 | 0.017 | 0.132 | 0.264 |

| Comparison | Requirement | Differences (b0 / b1 / b2) | Result |
| --- | --- | --- | --- |
| H2A (HD − FLAT) | ≥ 0.15 | −0.837 / −0.705 / −0.576 | fails |
| H2B (HD − TS) | ≥ 0.10 | −0.628 / −0.497 / −0.434 | fails |
| H2C (TS − FLAT) | ≥ 0.10 | −0.208 / −0.208 / −0.142 | fails |

**Outcome: `NO_ADVANTAGE_OVER_WIDTH`** (`QUAL_DISPOSITION.txt`: `HIERARCHY=HIERARCHY_ADJUDICATED`, `H2_OUTCOME=NO_ADVANTAGE_OVER_WIDTH`).

HD exceeded FLAT in 0 of 36 seed × budget cells. An independent Python recomputation from `PLANNING.tsv` gives the same outcome and differences.

The result is a negative, not a null: at equal planning compute, the flat width-72 model plans markedly better than both temporal hierarchies. The distinct-latent hierarchy is the worst at every budget. The direction was disclosed in the protocol from DESIGN and from R2's own G1-weight planning. It was never a selection input, and nothing was retried.

## What this does and does not establish

- **Established:**
  - A qualified learning law (C2/C2M) yields valid world models at all four constituent bindings, on fresh DEV and fresh QUAL.
  - Under the frozen R2 planning comparison, temporal hierarchy over these qualified models gives **no** planning advantage over a width-matched flat model.
- **Not established:** any runtime readiness. Under R2, integration is authorized only by `DISTINCT_ADVANTAGE`; it is not authorized here. The persistent-runtime qualification milestone (`../../successor_c2/ARCHITECTURE_CONSTRAINTS.md`) was conditional on the hierarchy passing, and it did not pass. Any next step is an operator decision.

Raw telemetry is not in Git; each `seeds/seed_*/COMMIT.txt` binds its sha256.
