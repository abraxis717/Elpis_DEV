# ECS_G Retention R1 laboratory (preregistered)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

**Status: RET1B, laboratory mechanics.** The laboratory implements the RET1A
specification (`protocol.py`, `numerics.py`, `task.py`, `engine.py`,
`experiment.py`, `run.py`); its mechanics tests use test-* worlds only. There is
no DEV or QUAL evidence, no frozen record, no selected candidate and no result.
Canonical ECS_G learning is unchanged.

## Question

Retention R0 (`docs/research/ECS_RETENTION_R0_RESULTS.md`) found a real
failure. Plain sequential G1 forgets catastrophically. The best consolidation
candidate reduced forgetting, but it protected A at the expense of acquiring B,
and it failed the pre-registered gates. R1 treats this as a stability-plasticity
problem:

> Which declared ECS_G cognitive state and update law lets one ECS_G state
> acquire each new experience of a jointly realizable sequence A -> B -> C -> D
> while retaining every earlier one, with no stored experience and no task
> label at query, and is that retention caused by the declared state?

R1 is not a retuning of R0's C1. `CANDIDATES.md` section 2 derives why. Each
experience's loss is an exact quadratic in `S3`. So C1's state was already a
sufficient statistic for joint training, and R0's outcome was decided by
capacity (the function class's joint least-squares ceiling failed in 16/24
worlds), by the `lambda = 4` weighting, by per-experience anchoring at compromise
states, and by the microscopic dynamics through `K(W) = J J^T`.

## Files

* `CANDIDATES.md`: the analysis of the R0 failure; the arm-S task design; the
  controls (M0 plain G1, M1 rehearsal as `EXTERNAL_MEMORY_CONTROL`, O
  representability ceilings, WSTAR realizability witness); the unchanged R0
  reference C1R; the candidates K1 (exact functional consolidation), K2
  (protected functional subspace) and K3 (fibre reconditioning over K1); the
  ablations; the families excluded before DEV and why; state accounting;
  native-runtime and FMS implications; predictions.
* `specs/ecsg-retention-r1.v1.spec.json`: regime, arms, splits, mechanisms,
  classes, thresholds, DEV rules, validity, gates A-M, disposition, mechanics
  checks, numerical binding and the evidence contract, fixed before DEV.

Both are write-once (`tests/research/ecs_retention_r1`). A changed experiment
is a new version.

## Chronology

```
RET1A specification / pass rule          <- this step
    |
RET1B laboratory mechanics               (test-* worlds only)
    |
RET1C DEV                                (task rule on controls, then candidate rule)
    |
RET1D freeze
    |
RET1E QUAL once                          (with its numerical-robustness children)
    |
RET1F interpretation
```

Each step is its own commit whose subject begins with its tag. A record may
appear only after its predecessor step: DEV evidence after RET1B, the frozen
record after RET1C, QUAL evidence after RET1D, and the results document after
RET1E. The preregistration guard tests enforce this from git history.

## Design in one page

* **Arms.** Arm S (primary, gating) is a jointly realizable sequence. Four
  experiences live on overlapping 3-coordinate subspaces (`{0,1,2}`,
  `{2,3,4}`, `{0,4,5}`, `{1,3,5}`) and are labelled by one world teacher, so a
  single ECS_G state `W* = [T* | 0]` answers all of them exactly. A failure
  cannot be blamed on capacity. Arm R (secondary, descriptive) is the R0 v1
  independent-teacher regime, stratified by the representability ceiling.
* **Primary requirement.** Every stage boundary of A -> B -> C -> D is held: each
  new experience learned, and every earlier one retained (R0's class
  thresholds, 0.5 absolute and 0.5 relative), in every QUAL world. No
  catastrophic loss anywhere, and a median final nmse of at most 0.25 over all
  four.
* **Validity.** Before any candidate is compared, the controls must certify a
  real test. Plain G1 must forget (at most 25% of worlds hold the sequence;
  median A degradation at least 10x). Every later experience must carry
  information the earlier ones do not determine. Cumulative rehearsal must
  succeed.
* **DEV freedom.** DEV chooses only the arm-S input scale from
  `(1.25, 1.0, 0.75)`, by the controls alone, and one candidate among K1, K2, K3
  by a fixed ranking. No candidate has a tuned strength: K1's `lambda = 1` is
  derived, K2's cutoffs are numerical-rank constants, K3's procedure is fixed.
  If no scale qualifies, R1 stops at DEV (`TASK_INVALID_ON_DEV`).
* **Causality.** The gates require snapshot/restore and reset bitwise; loss of
  retention when the declared state is removed and when it is reset after B;
  and bitwise continuation when the complete declared state is transplanted
  after B (nothing undeclared carries retention).
* **Numerical robustness.** The deciding QUAL computation is repeated within
  the one QUAL execution under forced Prescott and Haswell OpenBLAS kernels.
  Verdicts and counts must not change.

### Not easier than Retention R0

| | Retention R0 | Retention R1 |
|---|---|---|
| per-world class thresholds | 0.5 absolute, 0.5 relative | same |
| worlds required for the central gates | every QUAL world (24) | every QUAL world (24) |
| sequence | A -> B gating; A -> D descriptive | A -> B -> C -> D gating at every stage boundary |
| catastrophic loss | not gated | gated (none anywhere) |
| joint quality | median A after A <= 0.25 | also median final nmse over all four <= 0.25 |
| causality | state removal; snapshot/restore | also consolidation reset after B; complete-state transplant bitwise |
| numerical profile | environment strings; audited after QUAL | effective kernel and threads bound; robustness under two forced kernels gated |
| DEV choices | offset; 10 configurations | input scale (controls only); 1 of 3 parameter-free candidates |
| regime | independent teachers; joint ceiling feasible in 8/24 | jointly realizable by construction (gating); R0 regime kept as arm R (descriptive) |

The regime change is deliberate and is the only relaxation. R0 showed that in
16 of 24 worlds its gates tested capacity, not retention. Arm R keeps that
regime visible, and R1 claims nothing about it beyond description.

## Numerical authority

Every R1 record will bind Python, NumPy (version and build configuration), the
BLAS implementation, the effective BLAS kernel and thread count (read
in-process), CPU model and flags, machine, compiler, build type and C flags, the
native library SHA-256, the exact bound source digests, the laboratory source
digest and the spec, pass-rule and `CANDIDATES.md` digests. Four questions are
kept apart: evidence integrity, implementation/reference correctness,
historical bitwise replay (only under a matching binding, never a scientific
gate), and the current-runtime regression (never skipped).

## Laboratory (RET1B)

Every candidate step is the canonical native G1 step of the current `W` (an
executor created from `W` performs one learn step), followed by the
mechanism's correction at the pre-step `W`; with the mechanism removed the
engine is bitwise the canonical core. Learning an experience and consolidating
it are one atomic transition. The complete declared state (`W`, epoch and the
persistent consolidation state) serializes deterministically
(`engine.serialize`), which is what the transplant gate restores.

Implementation decision recorded before DEV: the registered PTE construction
(the R0 pair, differing on coordinate 0, stepped on B) is evaluated on the
arm-R world of each QUAL world, where B lies on axis 0 as in R0. In arm S, B's
inputs vanish on coordinate 0, so a B step never reads or moves that row and
the construction is structurally degenerate; it is recorded there as
descriptive only.

## Commands

From the repository root, with a Release native build:

    export PYTHONPATH=src:.
    python -m research.ecs_retention_r1.run status
    python -m research.ecs_retention_r1.run dev --library <build>/native/ECS_G/libelpis_ecsg_math.so
    ELPIS_NATIVE_BUILD=<build> ELPIS_REQUIRE_NATIVE=1 python -m pytest -q tests/research/ecs_retention_r1

`dev`, `freeze` and `qual` write once and refuse to run again; a changed
experiment is a new version.
