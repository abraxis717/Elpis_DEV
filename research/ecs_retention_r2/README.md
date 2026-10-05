# ECS_G Retention R2 laboratory

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

**Status: RET2B, laboratory mechanics.** The laboratory (`protocol.py`,
`numerics.py`, `task.py`, `engine.py`, `experiment.py`, `run.py`) implements the
RET2A specification; its mechanics tests use test-* worlds only. There is no DEV
or QUAL evidence, no frozen record, no selected candidate and no result.
Canonical ECS_G learning is unchanged.

## Question

Retention R1 v1 stopped correctly at `TASK_INVALID_ON_DEV`. Its primary task
was jointly realizable but not conflict-bearing:

- plain sequential G1 already retained the sequence;
- later experiences were not novel enough.

R2 keeps the R1 mechanisms unchanged (experimental isolation) and replaces
only the task:

> On a jointly realizable, aliasing-conflict sequence A -> B -> C -> D in which
> plain sequential G1 materially overwrites earlier function and every later
> experience is novel, which declared ECS_G cognitive state and update law (K1,
> K2, K3) lets one ECS_G state acquire each experience while retaining every
> earlier one, with no stored experience and no task label at query, and is that
> retention caused by the declared state?

## The task: aliased hidden ridges

In a random orthonormal basis `q1..q6`:

- **Teacher.** `T* = [r_A, g q4, g q5, g q6]`, with `r_A` in `S = span(q1, q2, q3)`.
- **Planes.** A observes `S`. B, C and D each replace one axis of `S` by an
  aliased axis at 45 degrees to a hidden direction:
  - `(q1 + q4)/sqrt(2)` for B;
  - `(q2 + q5)/sqrt(2)` for C;
  - `(q3 + q6)/sqrt(2)` for D.

This gives the task its properties by construction:

- **Realizable.** `W* = [T* | 0]` answers everything exactly.
- **Novel.** Each later experience's hidden ridge is invisible to every earlier
  plane.
- **Conflict-bearing.**
  - On B's plane, a ridge along `q4` and one along `q1` are the same function.
  - G1 moves every entity within B's plane, so it grows the new ridge half along
    `q1`. A sees that as a spurious ridge.
- **Resolvable by replay.** The union of experiences identifies everything, and
  rehearsal resolves the ambiguity.

The only task parameter is the hidden gain `g`. Its grid is `(2.0, 2.5, 1.5)`,
chosen on DEV by controls alone. The derivation and the disclosed control-only
design analysis are in `CANDIDATES.md`.

## Chronology

```
RET2A  question + task + candidates + thresholds + pass rule   <- this step
  |
RET2B  laboratory mechanics (test-* worlds only)
  |
RET2C  DEV (task rule on controls V1-V6, then candidate rule)
  |
RET2D  freeze
  |
RET2E  QUAL once (with numerical-robustness children)
  |
RET2F  interpretation
```

`TASK_INVALID_ON_DEV` ends the experiment at RET2C (RET2F may follow it).

## Validity before any candidate runs (every DEV world)

- **V1:** the witness is exact.
- **V2:** plain G1 forgets in absolute error. It holds the sequence in at most
  25% of worlds, its median earlier-experience nmse at D is at least 0.5, and
  its median A ratio is at least 10.
- **V3:** novelty is at least 0.5 at every later stage.
- **V4:** plain G1 learns every stage.
- **V5:** rehearsal holds the whole sequence.
- **V6:** the joint ceiling is exact (nmse at most `1e-6`).

## Pass rule

These are the R1 gates A-M, unchanged:

- every stage is acquired and every earlier experience retained, in every QUAL
  world;
- no catastrophic loss;
- median final nmse at most 0.25;
- one state answers all experiences;
- causality: state removal, consolidation reset and complete-state transplant;
- no answer store, and the state is fixed-size;
- determinism;
- native feasibility;
- numerical robustness under forced Prescott and Haswell kernels.

Validity failures give `TASK_INVALID_UNDER_QUAL`, never a pass.

## Files

* `CANDIDATES.md`: why R1 could not forget (the lemma), the task family and its
  analytic properties, the disclosed design analysis, the mechanisms (as R1),
  the ablations, the nondegenerate PTE construction, validity and predictions.
* `specs/ecsg-retention-r2.v1.spec.json`: the machine-readable authority. It
  reuses R1's mechanism, state, runtime, FMS, binding and evidence sections
  verbatim.

Both are write-once (`tests/research/ecs_retention_r2`).

## Laboratory (RET2B)

The mechanisms are the R1 implementation, unchanged in mathematics. The
engine applies the canonical native G1 step, then the correction at the
pre-step `W`; with the mechanism removed it is bitwise the canonical core.

Implementation decisions recorded before DEV:

- **K2 projection.** `C^T (C C^T)^+ C` is computed from the singular value
  decomposition of `C` (keep `s_i^2 > 1e-12 s_max^2`, the registered Gram
  cutoff). This is the same operator. Forming `(C C^T)^+` from the Gram
  eigendecomposition squared the condition number, and the registered `1e-10`
  constraint check was then met only marginally (`1.005e-10` on one test
  world).
- **PTE pair.** The pair lies along B's aliased axis (`task.pte`).

## Commands

    export PYTHONPATH=src:.
    python -m research.ecs_retention_r2.run status
    python -m research.ecs_retention_r2.run dev --library <build>/native/ECS_G/libelpis_ecsg_math.so
    ELPIS_NATIVE_BUILD=<build> ELPIS_REQUIRE_NATIVE=1 python -m pytest -q tests/research/ecs_retention_r2
