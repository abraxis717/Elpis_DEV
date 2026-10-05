# ECS_G Retention R3 laboratory (confirmatory K1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED` · `CONFIRMATORY`

**Status: RET3C, DEV recorded** (`evidence/dev/ecsg-retention-r3.v1.dev.json`:
`K1_PROCEEDS`). No QUAL world has been evaluated, and there is no result.
Canonical ECS_G learning is unchanged.

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
RET3A  preregistration
RET3B  laboratory mechanics (test-* worlds only)
RET3C  DEV (control-only task check; then the K1 DEV rule)   <- this step
RET3D  freeze
RET3E  QUAL once (with the numerical-robustness children)
RET3F  interpretation
```

`TASK_INVALID_ON_DEV` or `K1_STOPPED_ON_DEV` ends the experiment at RET3C;
RET3F may follow it. Only `OUTCOME_A` permits the native K1 milestone. Anything
else is NO_CANONICAL_PROMOTION.

## Laboratory (RET3B)

The engine is the R2 laboratory's K1 and C1R, unchanged in mathematics. K2 and
K3 are not present, and `engine.make` refuses them. Every K1 step is the
canonical native G1 step, then the K1 correction at the pre-step W. With the K1
state absent (`H = 0`) or removed, the engine is bitwise the canonical core.

The complete state `(W, epoch, H, a)` serializes to deterministic bytes:
- `engine.reset` empties H and a and keeps W and the epoch;
- `engine.import_w_only` turns a canonical `ELPISG01` snapshot into an
  UNCONSOLIDATED state and refuses anything else;
- `engine.query` reads W only.

Implementation decisions recorded before DEV:

- **`consolidation_state_shapes_learning`.** The specification words this check
  as "one K1 step on C from the complete state and from the reset state yield
  different W". From the consolidated state at B, that first step is identical
  by construction:
  - consolidation sets `a = S3(W_B)`;
  - so the K1 correction at the anchor is `u = 1/2 H (S3(W_B) - a) = 0`.

  The literal reading would fail in every world for a reason unrelated to the
  question. The laboratory therefore compares the learning of experience C
  from the two states. That is exactly what the uninterrupted run and the reset
  branch already compute: the check reads W after C.

  `test_mechanics.py` proves both facts: the zero correction at the anchor
  with an identical first step, and divergence over later steps. The decision
  was made and committed at RET3B, before any DEV world.
- **Consolidation interface.** Consolidation takes `(W, inputs)` only. R3 has
  no uninformed or mismatched ablation, so no random stream is passed.
- **W-only snapshots.** These are read with the `ELPISG01` R0 layout (magic,
  version, dim, width, epoch, W) and checked against the expected shape.

## Commands (from RET3B)

    export PYTHONPATH=src:.
    python -m research.ecs_retention_r3.run status
    ELPIS_NATIVE_BUILD=<build> ELPIS_REQUIRE_NATIVE=1 python -m pytest -q tests/research/ecs_retention_r3
