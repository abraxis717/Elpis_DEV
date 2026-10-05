# ECS_G Retention R1: results (v1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Laboratory: [`research/ecs_retention_r1`](../../research/ecs_retention_r1/README.md).
Preregistration: [`CANDIDATES.md`](../../research/ecs_retention_r1/CANDIDATES.md),
`specs/ecsg-retention-r1.v1.spec.json` (RET1A, write-once, unchanged).
Prior result: [`ECS_RETENTION_R0_RESULTS.md`](ECS_RETENTION_R0_RESULTS.md) (stands unchanged).

**Disposition: `TASK_INVALID_ON_DEV`.** The registered task rule, which runs on the
controls alone, found no arm-S input scale at which the primary arm is a valid
retention test. Under the specification R1 v1 therefore stopped at DEV:

* no candidate (K1, K2, K3) and no reference (C1R) was run on DEV;
* nothing was selected;
* there is no frozen record and no QUAL;
* no gate A-M was evaluated.

**R1 v1 established nothing about any retention mechanism. No candidate is
eligible for promotion. NO_CANONICAL_PROMOTION.** Canonical ECS_G learning is
unchanged.

## Chronology and authority

| step | commit | content |
|---|---|---|
| RET1A | `1ee30d0` | question, candidates, DEV rule, pass rule (merged as `589a167`) |
| RET1B | `dcf73cc` | laboratory mechanics; mechanics tests on test-* worlds only |
| RET1C | `522da82` | DEV evidence (DEV worlds only): `TASK_INVALID_ON_DEV` |
| RET1D | not run | the specification forbids a freeze after `TASK_INVALID_ON_DEV` |
| RET1E | not run | no QUAL without a freeze |
| RET1F | this commit | interpretation and authority pointers |

| item | value |
|---|---|
| base | `main@f6065ff1c39c6c2a3b108eee524dd7b39cb4640f` |
| DEV record digest | `768154701166e4b041c2b5d9440bde8289739f9cb9ca609adb0fb1fe5159543a` (file SHA-256 `0ee685f3...`) |
| produced by | clean RET1B commit `dcf73cc`; laboratory source digest `61b54f27...` |
| library | `libelpis_ecsg_math.so` `2374876e3a837761c87bdbc39389a2c5b2e007fcbaa81dd4b39e05deb1e5dc09` (the Runtime R1 binary), gcc 13.3.0 Release |
| numerical profile | Python 3.11.15, NumPy 1.26.4 (OpenBLAS 0.3.23.dev, ILP64, DYNAMIC_ARCH), effective OpenBLAS core Prescott, effective threads 1, Intel Xeon family 6 model 207 (sse4_2 avx avx2 fma avx512f), x86_64 |

## The registered task rule and what it found

On the 8 DEV worlds, for each input scale in the registered order (1.25, 1.0,
0.75), the rule runs the controls only:

- M0: plain sequential G1;
- M1: cumulative rehearsal;
- O: least-squares ceilings, including `O_prev` for novelty;
- WSTAR: the realizability witness.

It requires all four conditions:

- **V1 witness:** exact.
- **V2 forgetting:** M0 holds the sequence in at most 25% of worlds, with a median A end ratio of at least 10.
- **V3 novelty:** `nmse_s(O_prev) >= 0.25` in every world and stage.
- **V4 learnability:** M0 learns every stage, and M1 holds the sequence in every world.

| input scale | V1 | V2 | V3 | V4 | M0 holds the whole sequence | M0 median earlier-experience nmse at D | M0 median A end ratio | novelty below 0.25 (min) | M1 holds |
|---|---|---|---|---|---|---|---|---|---|
| 1.25 | pass | **fail** | **fail** | **fail** | 5/8 | 0.039 | 129.0 | 7 of 24 (0.121) | 8/8 |
| 1.0 | pass | **fail** | **fail** | pass | 8/8 | 0.030 | 18.1 | 6 of 24 (0.103) | 8/8 |
| 0.75 | pass | **fail** | **fail** | pass | 8/8 | 0.034 | 2.8 | 7 of 24 (0.079) | 8/8 |

At 1.25, plain G1 failed to learn D in 2 of 8 worlds (`dev-0001`, `dev-0005`).
At every scale, the witness answered every experience exactly (nmse 0). The
controls never failed RETAINED in more than 2 of 48 world/experience pairs, and
were never catastrophic.

## Interpretation

**The primary arm, as preregistered, is not a retention test.** Plain
sequential G1 already holds every earlier experience to the class thresholds in
most worlds. So the gates that were meant to distinguish a retention mechanism
from no mechanism would have passed without one.

The large ratios (median A end ratio 18 to 129 at the two larger scales) are
ratios of very small errors. A is learned to an nmse of about 0.002 and drifts
to about 0.03. That is degradation, but far inside the absolute class
threshold of 0.5.

The cause is the arm-S construction chosen at RET1A, in two parts.

- **Shared monomials agree.** Every experience is labelled by one world teacher
  on overlapping coordinate subspaces. A later experience therefore agrees with
  earlier ones on every shared monomial. Plain G1 disturbs only the monomials
  that mix shared and private coordinates, and only mildly.
- **Novelty is too small.** In 6 to 7 of 24 world/stage pairs, the earlier
  experiences already determined at least 75% of a later experience's variance,
  so V3 failed as well.

CANDIDATES.md section 13 predicted that M0 would forget (validity). That
prediction was wrong. The preregistered validity rule is what caught it.

**This is the validity rule working as intended.** Without it, R1 v1 could have
run QUAL on a task that plain G1 passes. It could then have reported
`RETENTION_SUPPORTED` for a candidate whose state caused nothing.
`TASK_INVALID_ON_DEV` instead keeps the R0 question exactly where R0 left it:

- the qualified recurrence forgets in conflicting regimes;
- C1 reduces forgetting at a cost in acquisition;
- whether any declared ECS state resolves the stability-plasticity tradeoff is
  open.

**What R1 v1 does not say.** It says nothing for or against K1, K2, K3 or C1R.
None of them was run on any DEV or QUAL world, so no conclusion about them is
licensed. The mechanics tests on test-* worlds are implementation checks, not
evidence.

## What a repair would need (not executed; a new experiment version)

A repair is a new experiment version with its own RET-A step. This result does
not authorize editing the v1 specification, thresholds, candidates or rules.
Whatever regime a new version chooses, the R1 v1 DEV evidence states what it
must provide, and the controls must certify it on DEV:

- plain G1 must fail the absolute retention classes in most worlds, not only
  degrade in ratio;
- every later experience must carry enough novel variance;
- a single realizable state must still answer every experience;
- rehearsal must still succeed.

These are requirements on the task, recorded here so that a future
preregistration can be checked against them. They are not a design.

## Laboratory notes recorded before DEV

- **PTE check.** The registered construction perturbs coordinate 0 and steps on
  B. RET1B evaluates it on the arm-R world, where B lies on axis 0 as in R0. In
  arm S, B's inputs vanish on coordinate 0, so the construction is structurally
  degenerate there. This decision was made and committed in RET1B, before any
  DEV world was evaluated. It did not affect DEV, which runs no mechanics gate.
- **Evidence format.** Evidence records are written as compact JSON to stay
  within the repository's file-size limit.

## Reproduction contract

| question | test | contract |
|---|---|---|
| evidence integrity | `tests/research/ecs_retention_r1/test_evidence.py` (byte pin, digest, binding, recorded verdicts recomputed from recorded rows, no freeze or QUAL) | always |
| implementation correctness | `test_mechanics.py` (test-* worlds); `ctest -R ^ECS_G.` | always |
| historical replay | `test_evidence.py::test_dev_controls_reproduce_exactly_under_the_recorded_binding` | bitwise, only under the recorded implementation and numerical binding; skipped with the reason otherwise; never a scientific gate |
| current-runtime regression | `test_runtime_regression.py` (the task rule re-run on every DEV world and recorded scale) | never skipped: every verdict and boolean class equal to the record |
| chronology | `test_preregistration.py` | from full git history |

## Consequences for the canonical system

None. R1 v1 qualified no mechanism, so:

- the conditional native promotion (a Retention Runtime R2 with consolidation
  state, an FMS retention-state envelope, a fused native consolidation hot
  path) was **not** started;
- Runtime R1, Mutable FMS R0 and every canonical ECS_G source are unchanged;
- `ELPISG01` (W and epoch) remains the only ECS_G cognitive serialization.

NO_CANONICAL_PROMOTION.
