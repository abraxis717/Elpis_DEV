# ECS_G Retention R3 laboratory (confirmatory K1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED` · `CONFIRMATORY`

**Status: RET3A, preregistered.** There is no laboratory, no DEV or QUAL world
has been evaluated, and there is no result. Canonical ECS_G learning is
unchanged.

## Question

Does fixed-size K1 consolidation, `(W, epoch, H, a)`, causally preserve
previously acquired function under genuine sequential interference, robustly
and within native-runtime feasibility constraints?

Retention R2 (closed: `PARTIAL_REDUCTION`, `OUTCOME_C`) found K1 held 24/24 QUAL
worlds. K3's extra machinery added nothing and broke the native budget. R2's
reset-causality test was confounded by later re-teaching.

R3 is therefore a confirmatory K1 experiment:

- **One eligible mechanism:** K1. There is no candidate search.
- **The inherited task family** (aliased hidden ridges) at the inherited gain
  2.0, on fresh, disjoint worlds.
- **A repaired causality test:** the reset challenge is read at the first
  post-interference boundary (C), before D can re-teach anything.
- **Gate L covers every decision-bearing quantity, and only those.** This was
  decided before data.
- **A preregistered native budget** for K1.

The design is explained in `PREREGISTRATION.md`. The authority is
`specs/ecsg-retention-r3.v1.spec.json`. Both are write-once.

## Chronology

```
RET3A  preregistration                      <- this step
RET3B  laboratory mechanics (test-* worlds only)
RET3C  DEV (control-only task check; then the K1 DEV rule)
RET3D  freeze
RET3E  QUAL once (with the numerical-robustness children)
RET3F  interpretation
```

`TASK_INVALID_ON_DEV` or `K1_STOPPED_ON_DEV` ends the experiment at RET3C;
RET3F may follow it. Only `OUTCOME_A` permits the native K1 milestone. Anything
else is NO_CANONICAL_PROMOTION.

## Commands (from RET3B)

    export PYTHONPATH=src:.
    python -m research.ecs_retention_r3.run status
    ELPIS_NATIVE_BUILD=<build> ELPIS_REQUIRE_NATIVE=1 python -m pytest -q tests/research/ecs_retention_r3
