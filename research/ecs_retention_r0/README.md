# ECS_G Retention R0 laboratory

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Question: can one ECS_G state acquire a new experience B while retaining a
previously acquired experience A, and what state does that require?
Cognitive R0 found that plain sequential G1 learning does not retain
(interference in 8 of 8 QUAL worlds, catastrophic in 6 of 8).

* `CANDIDATES.md`: the mechanisms (M0 plain G1, M1 rehearsal control, C1
  coarse S3-space functional consolidation, C2 local per-weight
  consolidation, O representability analysis), their state and
  falsification conditions.
* `specs/ecsg-retention-r0.v1.spec.json`: regime, tasks, splits, metrics,
  DEV rules, gates A-J, dispositions, mechanics checks and binding
  requirements, fixed before DEV.

Chronology: RET0A specification and laboratory (no results) -> RET0B DEV
evidence -> RET0C freeze -> RET0D QUAL evidence (once) -> RET0E
interpretation. Every response is the canonical native forward map of the
evaluated `W`; every G1 step is the canonical native step. No DSV, tokenizer,
transformer or other model participates, and nothing here is imported by
`src/`.

## Result

QUAL (24 worlds, run once): `PARTIAL_REDUCTION` (`OUTCOME_C`). The selected
candidate (C1, S3-space functional consolidation, `lambda` 4) retained A in
18/24 worlds where plain G1 retained 0/24, but learned B in only 10/24;
gates B (acquisition) and C (retention) failed as registered. No candidate is
eligible for promotion. Full report:
`docs/research/ECS_RETENTION_R0_RESULTS.md`.

* `evidence/dev/`, `frozen/`, `evidence/qual/`: write-once records (digests in
  the report).
* `post/perf.py` -> `evidence/perf.json`: PERFORMANCE_ONLY timing taken after
  QUAL, outside the frozen laboratory source digest.

## Commands

From the repository root, with a Release native build
(`cmake -S . -B <build> -DCMAKE_BUILD_TYPE=Release && cmake --build <build> --target elpis_ecsg_math`):

    export PYTHONPATH=src:.
    python -m research.ecs_retention_r0.run status
    ELPIS_NATIVE_BUILD=<build> ELPIS_REQUIRE_NATIVE=1 python -m pytest -q tests/research/ecs_retention_r0

`dev`, `freeze` and `qual` refuse to run again (the records exist); a changed
experiment is a new version. The evidence tests re-run three QUAL worlds and
require bit-identical measurements under the full v1 binding, an AVX-512
OpenBLAS kernel (Cooperlake or SkylakeX) and an effective single BLAS thread;
otherwise they skip with the reason. `test_runtime_regression.py` re-runs all
24 QUAL worlds on any host and requires the recorded gates, disposition and
per-mechanism counts exactly (never skipped). See the results document,
"Reproduction contract and portability".
