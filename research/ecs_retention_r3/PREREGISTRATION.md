# ECS_G Retention R3: confirmatory K1 preregistration (RET3A)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED` · `CONFIRMATORY`

The machine-readable authority is `specs/ecsg-retention-r3.v1.spec.json`. This
document explains it. Where they differ, the specification decides. Both are
write-once from RET3A.

## 1. Starting point

Retention R2 is closed as `PARTIAL_REDUCTION` (`OUTCOME_C`),
NO_CANONICAL_PROMOTION. R3 does not rescue, re-run or reinterpret it. Its
QUAL record (24 worlds) established:

| mechanism | sequence held |
|---|---|
| M0 plain G1 | 1/24 |
| M1 rehearsal | 24/24 |
| C1R (R0 reference) | 21/24 |
| K1 | 24/24 |
| K2 | 23/24, kernel dependent (gate M, RR2-CR0) |
| K3 | 24/24 |

- **K3 adds nothing over K1.** `MICROSTATE_EFFECT` was false, and K3 without
  its fibre reconditioning (that is, K1) held 24/24. The reconditioning
  exceeded the registered native cold-path budget (gate L).
- **R2's reset-causality clause was confounded.** It reset consolidation after
  B and judged A and B at D. D overlaps A's plane and re-taught what the
  unprotected C step had damaged.

R3 therefore asks one confirmatory question about the simplest mechanism that
held every R2 world. It is not a mechanism search. K2 and K3 are referenced
historically only and are never run.

## 2. Question

> Does fixed-size K1 consolidation, `(W, epoch, H, a)` with the exact K1 law of
> the closed R1/R2 lineage, causally preserve previously acquired function under
> genuine sequential interference, robustly across the registered numerical
> kernels and within a preregistered native-runtime budget?

## 3. Inherited versus new

**Inherited from closed R2 evidence** (no R3 observation informed these):

- **Task family:** aliased hidden ridges, unchanged. The teacher is
  `T* = [r_A, g q4, g q5, g q6]` in a Haar basis. A observes
  `span(q1, q2, q3)`, and B, C and D each replace one axis by a 45° alias of a
  hidden direction.
- **Hidden gain 2.0.** R2's control-only DEV rule chose it, and R2 QUAL validity
  held at it. R3 fixes it as a task parameter: there is no grid and no search,
  and candidate behaviour never touches it.
- **Regime:** `d = 6`, `N = 36`, rate 0.002, 4000 steps per experience, 64
  training and 256 held-out rows.
- **Classes and class thresholds** (0.5 absolute, 0.5 relative).
- **The K1 law** (lambda 1):
  - consolidation after experience t: `H <- H + Sigma_t`, `a <- S3(W_t)`;
  - every step: `W <- G1(W) - eta J(W)^T u`, with
    `u = 1/2 H (S3(W) - a)` at the pre-step W.
- **The control definitions** M0, M1, O, WSTAR and C1R.

**New in R3:** every DEV and QUAL observation, all on fresh worlds.

- The seed is 20261206 and the name `ecsg-retention-r3`. R2's were 20261105
  and `ecsg-retention-r2`.
- World ids are `r3dev-*` and `r3qual-*`, so no R2 world is an R3 world.
- QUAL has 32 worlds; R2 had 24.

## 4. State semantics

The authoritative K1 state is `(W, epoch, H, a)`.

- **QUERY** reads W only, through the canonical native forward map.
- **LEARN** reads W, H, a and the current experience.
- **CONSOLIDATE** reads W and the current experience's inputs; no target ever
  enters it.

Two states with the same W and different `(H, a)` therefore answer every query
identically and may learn differently. Both facts are mechanics checks.

A canonical W-only snapshot (`ELPISG01`: W and its epoch) imports into the
laboratory only as an UNCONSOLIDATED K1 state (`H = 0`, `a = 0`, flagged). It is
never a retained state.

## 5. Controls

| control | role |
|---|---|
| M0 | plain sequential G1 (forgetting baseline; validity) |
| M1 | cumulative rehearsal (external-memory control; validity; never eligible) |
| O | least-squares stage ceilings and novelty (experiment only) |
| WSTAR | exact joint witness (experiment only) |
| C1R | R0's C1, lambda 4 (reference; reported, decides nothing but descriptive qualifiers) |
| K1 | the sole eligible mechanism |

## 6. Causality: the repaired design

Four causal tests run in every DEV and every QUAL world.

1. **State removal.** The K1 engine runs with its correction and consolidation
   removed; it is bitwise the canonical core.
   - It must forget like plain G1: SEQUENCE_HELD in at most 25% of worlds.
   - Its median final earlier-experience nmse must be at least 2× K1's.

2. **Full-state transplant.** After B, the complete state `(W, epoch, H, a)` is
   serialized to deterministic bytes and restored into a fresh engine, which
   then learns C and D.
   - W and state digests at C and D must equal the uninterrupted run's.
   - This is checked in every world, and once in a clean process.

3. **W-only negative control.** After B, take the canonical `ELPISG01` snapshot
   of `W_B` and import it.
   - The result must be flagged UNCONSOLIDATED, with `H = 0` and `a = 0`.
   - Its state digest must differ from the complete state's.
   - Learning C from it must reproduce the reset branch bitwise and differ from
     the uninterrupted run.
   - It never counts as a retained-state transplant.

4. **Immediate reset challenge.** This is the primary causal test and the
   repair of R2's gate G(iv).

   ```
   learn A (canonical core, shared W_A); consolidate
   learn B (K1); consolidate
   snapshot the complete state (W_B, epoch_B, H_B, a_B)
   RESET H <- 0, a <- 0; keep W_B and epoch_B bitwise
   learn the interfering experience C (with H = 0 every K1 correction is zero:
       no protection inherited from A and B)
   IMMEDIATELY evaluate A, B at W_C^reset
   STOP (no D)
   ```

   The comparison is uninterrupted K1 at the same boundary: W after C, before
   D. The protected error is `e = mean(nmse_A, nmse_B)` at that boundary.

   - **RESET_DEGRADED (per world):** `e_reset >= 10 × e_uninterrupted` and
     `e_reset >= 0.05`.
   - **Gate:** RESET_DEGRADED in at least 75% of worlds, and median
     `e_reset / e_uninterrupted >= 10`.
   - **No re-teaching contamination:** the reset branch has no stage after C,
     and the verdict reads only the C boundary. This is checked mechanically:
     every reset record ends at C.

   **Design disclosure.** The closed R2 record contains the K3 reset rows at
   the C boundary; the R2 results read them post hoc. In all 24 R2 worlds they
   show `e_reset / e_uninterrupted >= 21` and `e_reset >= 0.036`. The R3
   criterion was fixed before any R3 data, on these principles:
   - a relative loss of an order of magnitude;
   - an absolute floor of 5% of target variance, so that the test is not a
     ratio of tiny errors (the R1 lesson). 23 of the 24 R2 rows meet that
     floor; the floor was not set at their minimum;
   - a 75% world fraction, mirroring R2's 25% limit.

   R2's rows came from K3, and R3 runs K1, so they inform the design but are
   not R3 evidence.

## 7. DEV (8 fresh worlds)

**Task check, controls only, at the inherited gain 2.0.** All six conditions
must hold:

- **V1:** the witness is exact.
- **V2:** M0 holds the sequence in at most 25% of worlds, with median earlier
  nmse at D of at least 0.5 and median A ratio of at least 10.
- **V3:** novelty is at least 0.5 at every later stage.
- **V4:** M0 learns every stage.
- **V5:** M1 holds every world.
- **V6:** the joint ceiling is exact (at most `1e-6`).

Any failure gives `TASK_INVALID_ON_DEV`, and R3 stops.

**K1 DEV rule (only if the task is valid).** K1 proceeds only if all of these
hold:

- every world LEARNED at every stage and RETAINED for every earlier experience;
- no catastrophe;
- median final mean nmse at most 0.25;
- the state-removal criterion;
- the reset criterion;
- a bitwise full-state transplant in every world;
- the W-only negative control in every world;
- the native budget.

Otherwise the result is `K1_STOPPED_ON_DEV`: R3 stops, with no freeze, no QUAL
and NO_CANONICAL_PROMOTION. There is no candidate selection, and DEV chooses
nothing.

## 8. QUAL gates (32 fresh worlds, run once)

| gate | requirement |
|---|---|
| A current-stage acquisition | `W_A` learned (median nmse at most 0.25); K1 LEARNED at B, C, D in every world |
| B sequential retention | K1 RETAINED every earlier experience at every stage, every world |
| C no catastrophic forgetting | no CATASTROPHIC in any world |
| D joint quality | median final mean nmse at most 0.25 |
| E single joint state | one executor from `W_s` answers all experiences; no task label in query or learning |
| F declared-state causality | state removal (at most 25% held, median at least 2×); reset challenge (at least 75% degraded, median ratio at least 10) at the C boundary; no re-teaching |
| G no external answer store | consolidation receives W and inputs only; fixed bytes; no stored rows; queries read W only |
| H deterministic state | twice in process and once in a clean process give identical digests and metrics |
| I transplant and reset behaviour | full-state transplant bitwise in every world, plus a clean-process probe; the W-only negative control in every world |
| J capacity accounting | bytes and analytic operations per stage; M1 stored bytes; K1 extra bytes 28,552 throughout |
| K native feasibility | the preregistered budget (section 10) |
| L numerical robustness | the decision record (section 9) is identical under forced Prescott and Haswell kernels |

Mechanics checks are listed in the specification. They include:

- feature identity, witness, engine equivalence, shared `W_A`;
- the consolidation interface, gradient identity, the PTE microstate check;
- the query ignores the consolidation state, and the consolidation state shapes
  learning;
- reset mechanics and an effective single thread.

**Disposition.** It is one of these:

- `MECHANICS_FAIL`.
- `TASK_INVALID_UNDER_QUAL` (`OUTCOME_V`).
- `RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME` (`OUTCOME_A`) if gates
  A–L all hold.
- `PARTIAL_REDUCTION` (`OUTCOME_C`).
- `REPLAY_ONLY` (`OUTCOME_B`).
- `NO_MATERIAL_IMPROVEMENT` (`OUTCOME_D`).

Only `OUTCOME_A` permits the native milestone. Anything else is
NO_CANONICAL_PROMOTION, and R3 stops.

## 9. Numerical robustness (gate L): the principle, stated before data

RR2-CR0 established the distinction: historical exact replay is bound to its
recorded environment, while a current-runtime regression runs on the host's
natural kernel and checks the science. R3 starts from that distinction.

Since K1 is the sole eligible mechanism, gate L covers **every decision-bearing
R3 quantity**, and only those. The **decision record** consists of:

- validity, mechanics, gates A–K, disposition and outcome;
- per world, K1's LEARNED, RETAINED, CATASTROPHIC and HELD classes, and
  SEQUENCE_HELD;
- per world, state-removal SEQUENCE_HELD, RESET_DEGRADED, transplant bitwise,
  and the W-only control;
- counts: M0 SEQUENCE_HELD and learned-at-every-stage, and M1 SEQUENCE_HELD.

It must be identical under the forced Prescott and Haswell kernels, inside the
one QUAL execution.

C1R rows, floats, digests, medians and timings decide nothing and are not
required to be kernel invariant. **Any K1 decision-bearing kernel sensitivity
fails R3.**

**Portability:**

- **Forbidden everywhere:** CPU-model branches, kernel-specific behaviour,
  `OPENBLAS_CORETYPE` in ordinary CI, per-host expected outputs and post-hoc
  tolerances.
- **Forced kernels** exist only inside QUAL's gate L.
- **The current-runtime regression** re-runs QUAL on the host's natural kernel
  and compares the decision record. It is never skipped.

## 10. Native feasibility budget (gate K), fixed before data

These are analytic counts of the K1 law at the registered regime: they are
closed form and do not depend on any observation.

| quantity | K1 | budget |
|---|---|---|
| hot-path extra operations per step | 42,506 | at most 1.0 × G1 (82,944 at R = 64) |
| operations per consolidation | `R F + R F (F+1) + F (F+1)/2 + N (d + 2P + 3T)` = 462,782 | at most `1e8` |
| persistent extra state | H packed (3,486 values) + a (83 values) = 28,552 B | exactly 28,552 B, fixed, independent of the number of experiences |
| stored rows, targets, task labels | none | none |
| iterative procedures | none (closed form) | none: no fibre loop, no retraction |

## 11. Evidence contract

1. **Integrity:** SHA-256 pins, record digests, and chronology from full git
   history.
2. **Correctness:** mechanics tests on test-* worlds, and `ctest -R ^ECS_G.`.
3. **Historical replay:** bitwise, only under the full recorded binding;
   otherwise skipped with the reason. It is never a gate.
4. **Current-runtime regression:** never skipped; uses the natural kernel; the
   decision record must be equal; descriptive quantities are not compared.

   If gate L failed, RR2-CR0's principle applies. Only rows the record itself
   shows kernel-sensitive are exempt from host equality, and never a
   decision-bearing K1 row, because that failure is already R3's result.

## 12. Predictions (not gates)

- **P1:** the task is valid on fresh DEV worlds.
- **P2:** K1 holds every QUAL world.
- **P3:** the immediate reset degrades protected function in most worlds.
- **P4:** state removal reproduces M0 bitwise.
- **P5:** K1's decision record is kernel invariant. In R2, K1's counts were
  invariant under Prescott, Haswell and Zen.

## 13. What R3 can and cannot establish

An `OUTCOME_A` supports retention by the declared fixed-size K1 state on
jointly realizable aliasing-conflict sequences of this family. That covers
`d = 6`, `N = 36`, the qualified cubic kernel and G1 as the base step, with
causality shown at the first post-interference boundary. It makes K1 eligible
for the native milestone (specification `promotion`) and changes no canonical
code by itself.

R3 says nothing about:

- capacity-limited regimes, other task families, or larger `d`;
- K2, K3 or any other mechanism;
- language or meaning.
