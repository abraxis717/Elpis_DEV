# ECS_G Retention R2: results (v1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Laboratory: [`research/ecs_retention_r2`](../../research/ecs_retention_r2/README.md).
Preregistration: [`CANDIDATES.md`](../../research/ecs_retention_r2/CANDIDATES.md),
`specs/ecsg-retention-r2.v1.spec.json` (RET2A, write-once, unchanged).
Prior results: [`ECS_RETENTION_R0_RESULTS.md`](ECS_RETENTION_R0_RESULTS.md) and
[`ECS_RETENTION_R1_RESULTS.md`](ECS_RETENTION_R1_RESULTS.md) (both stand unchanged).

**Disposition: `PARTIAL_REDUCTION` (`OUTCOME_C`).**

- **The task was valid.** On DEV the control-only task rule found a valid
  hidden gain (`g = 2.0`), and every QUAL validity condition held. Plain
  sequential G1 forgets on this task; rehearsal and a joint state solve it.
- **The selected candidate retained every experience.** K3 acquired every stage
  and retained every earlier experience in **24/24** QUAL worlds. Plain G1
  retained the sequence in 1/24.
- **Three registered gates failed:**
  - **G** (state causality), on its consolidation-reset clause;
  - **L** (native feasibility), on K3's cold-path cost;
  - **M** (numerical robustness), on per-mechanism counts that changed under
    forced kernels.

The registered rule therefore gives the partial outcome, not the supporting
one. **No candidate is qualified. No candidate is eligible for promotion.
NO_CANONICAL_PROMOTION.** Canonical ECS_G learning is unchanged.

## Chronology and authority

| step | commit | content |
|---|---|---|
| RET2A | `fd598ed` | question, task family, candidates, DEV rule (V1-V6), pass rule |
| RET2B | `5145f28` | laboratory mechanics; mechanics tests on test-* worlds only |
| RET2C | `5837e27` | DEV evidence (DEV worlds only): task valid at `g = 2.0`; K3 selected |
| RET2D | `5217068` | freeze of specification, pass rule, CANDIDATES.md, choices and binding |
| RET2E | `7d0d567` | QUAL, run once on 24 disjoint worlds, with forced-kernel robustness children |
| RET2F | this commit | interpretation and authority pointers |

| item | value |
|---|---|
| base | `main@fb1a96db6eb10f02f6b87cb5dfd2f490ab1ba0c5` |
| DEV record digest | `b9911cd4e7d0113923ef55470bf20dc6c3c3c41ae360dd6e5f3734edf1f1fd20` |
| frozen record digest | `38c8e2e369713bb8f9b7623bcc316274d03aa46c7d93acf074ac529dc82686c0` |
| QUAL record digest | `e14b59bddd92de17dafdbfaddabc2ce68a75e74549c930cc84b34f3f0b1b79f3` |
| produced by | clean trees: RET2B `5145f28` (DEV); RET2D `5217068` (QUAL); laboratory source digest `7723b5d4...` |
| library | `libelpis_ecsg_math.so` `2374876e3a837761c87bdbc39389a2c5b2e007fcbaa81dd4b39e05deb1e5dc09` (the Runtime R1 binary), gcc 13.3.0 Release |
| numerical profile | Python 3.11.15, NumPy 1.26.4 (OpenBLAS 0.3.23.dev, ILP64, DYNAMIC_ARCH), effective OpenBLAS core SkylakeX, effective threads 1, Intel Xeon family 6 model 85 (sse4_2 avx avx2 fma avx512f), x86_64 |
| robustness children | forced OpenBLAS cores Prescott and Haswell, effective threads 1 |

## Validity (DEV task rule, then QUAL)

The DEV rule ran the controls only (M0, M1, O, WSTAR) at the first gain in the
registered order. `g = 2.0` was valid, so no other gain was evaluated.

| DEV, `g = 2.0` (8 worlds) | value |
|---|---|
| V1 witness exact | yes; joint ceiling `6.9e-28` |
| V2 forgetting: M0 holds the sequence | 0/8 |
| V2 forgetting: M0 median earlier-experience nmse at D | 0.895 |
| V2 forgetting: M0 median A end ratio | 760 |
| V3 novelty (min over worlds and stages) | 0.760 |
| V4 M0 learns every stage | 8/8 |
| V5 M1 holds the sequence | 8/8 |
| V6 capacity | exact |

Under QUAL (24 worlds) the registered validity conditions V2, V3, V5 and V6 all
held:

- M0 held the sequence in 1/24, with median earlier-experience nmse 0.539 and
  median A end ratio 488;
- novelty was at least 0.5 at every stage;
- M1 held the sequence in 24/24;
- the joint ceiling was exact.

This is the property R1 v1 lacked: plain G1 forgets in absolute error, not
only in ratio. In 11 of 24 QUAL worlds that forgetting was catastrophic.

## Candidate selection (DEV)

The registered candidate rule ranks by worlds held, then by median final
nmse, then by state bytes, then by hot-path cost:

| candidate | DEV worlds held | median final mean nmse | persistent bytes | hot-path ops per step |
|---|---|---|---|---|
| **K3 (selected)** | 8/8 | 0.00062 | 28,552 | 42,506 |
| K1 | 8/8 | 0.00091 | 28,552 | 42,506 |
| K2 | 8/8 | 0.00225 | 28,552 | 3,913,763 |

## QUAL

### Counts over 24 worlds

| mechanism | sequence held | catastrophic (any) | median earlier-experience nmse at D | median final mean nmse | median A end ratio |
|---|---|---|---|---|---|
| M0 (plain G1) | 1 | 11 | 0.539 | 0.411 | 488 |
| M1 (rehearsal; stores rows) | 24 | 0 | 0.0018 | 0.0041 | 3.0 |
| C1R (R0 reference, `lambda = 4`) | 21 | 0 | 0.030 | 0.031 | 23.8 |
| K1 | 24 | 0 | 0.00057 | 0.0011 | 1.11 |
| K2 | 23 | 1 | 0.00084 | 0.0014 | 1.11 |
| **K3 (selected)** | **24** | **0** | **0.00041** | **0.00054** | **0.69** |

| ablation of K3 | result |
|---|---|
| `state_removed` | identical to M0: held 1/24, 11 catastrophic |
| `extended_state_transplant_at_B` | continuation bitwise equal in 24/24; the clean-process probe was bitwise |
| `consolidation_reset_at_B` | A and B both retained at D in **24/24** |
| `fibre_removed` (K1 law alone) | held 24/24 |
| `mismatched_statistics` | held 0/24, all catastrophic |
| `uninformed_statistics` | held 3/24, 4 catastrophic |

Note that `fibre_removed` drops K3's reconditioning and keeps its K1 law.

### Gates (registered rule, unchanged since RET2A)

| gate | result | evidence |
|---|---|---|
| A first acquisition | pass | |
| B stage acquisition | pass | K3 learned every stage, 24/24 |
| C sequence retention | pass | K3 retained every earlier experience, 24/24 |
| D no catastrophe | pass | |
| E joint quality | pass | median final mean nmse 0.00054, threshold 0.25 |
| F joint state | pass | one W answers every experience; no task label at query |
| **G state causality** | **fail** | clauses (i), (ii) and (iii) pass; clause (iv) fails, see below |
| H no external answer store | pass | the consolidation state receives no targets; 28,552 bytes, fixed |
| I determinism | pass | |
| J control honesty | pass | |
| K capacity accounting | pass | |
| **L native feasibility** | **fail** | K3's maximum cold-path cost was `1.49e10` operations per consolidation, against a limit of `1e10`; 20 of 96 consolidations were over it. The hot path (42,506 ops per step, 0.51 x G1) and the fixed state pass. |
| **M numerical robustness** | **fail** | gates, validity, mechanics, disposition and every count of K3 are identical under Prescott and Haswell. The registered comparison covers every mechanism's counts, and two rows changed: K2 and `uninformed_statistics` (see below). |

Gate G clause by clause:

- **(i)** Snapshot restore and untaught reset are bitwise: pass.
- **(ii)** `state_removed` holds 1/24 (limit 25%), and its median earlier nmse,
  0.539, is at least 2 x K3's 0.00041: pass.
- **(iii)** Transplant: pass.
- **(iv)** The registered clause requires that A and B are both still retained
  at D in **at most 25%** of worlds after a consolidation reset at B. They were
  retained in 24/24: **fail**.

The two count rows that changed under forced kernels:

- **K2:** on `qual-0009` with SkylakeX kernels, K2 was catastrophic (A at D
  nmse 24). Under Prescott and Haswell it held. So K2's sequence-held count is
  23 on the host and 24 forced.
- **`uninformed_statistics`:** held-D 6 on the host and 5 forced.

Qualifiers, as recorded:

| qualifier | value |
|---|---|
| `EXCEEDS_R0_REFERENCE` | true (K3 24/24 against C1R 21/24) |
| `R0_REFERENCE_ALSO_SUFFICIENT` | false |
| `MICROSTATE_EFFECT` | false (K1 and K3 do not differ materially) |
| `NUMERICALLY_FRAGILE` | true (gate M) |
| `CAPACITY_SCOPE` | true (jointly realizable regime only) |

The PTE microstate mechanics check was nondegenerate in R2. Its pair lies
along B's aliased axis, and it passed.

## Interpretation

**What R2 establishes, within its frozen synthetic regime:**

- **The task is a genuine conflict.** On a jointly realizable, aliasing-conflict
  sequence, plain sequential G1 overwrites earlier function in absolute error.
- **The declared state reduces forgetting.** A declared, fixed-size ECS_G
  consolidation state (K3, and K1 with the same hot-path law) reduces that
  forgetting to the level of rehearsal or below. It does so without stored
  rows, targets or task labels, and acquisition is not traded away: every stage
  is learned in every world.
- **The state is causal for the outcome.** Removing it reproduces plain G1
  exactly, and transplanting it continues the run bitwise.

This is the improvement over R0, whose selected candidate retained A but
acquired B in only 10/24 worlds.

**Why it is not qualified.** The registered rule requires every gate, and
three failed.

**G (iv): consolidation reset at B.** The clause asks: once A and B are
learned, is the consolidation state still what keeps them through C and D? The
recorded rows answer at two points.

- **At C, the reset costs retention.** Directly after the unprotected C stage
  (reset ablation, recorded rows):
  - median nmse is A 0.47 and B 0.13 (K3 uninterrupted: 0.0009 and 0.0003);
  - A and B are both still retained in only 13/24 worlds.
- **At D, it does not.** The registered clause measures at D. By then A and B
  are retained again in 24/24, with medians A 0.005 and B 0.061.
  - D observes two of A's three axes fully.
  - Its targets are consistent with A's, so learning D re-teaches most of what
    the unprotected C step broke.
  - Plain G1 shows the same re-acquisition in the recorded counts: M0 retains
    A at C in 5/24 and at D in 15/24.

The clause as registered therefore cannot distinguish protection from
re-acquisition by an overlapping later experience. This reading is post hoc,
from recorded rows only. It does not change the registered result: G fails,
and the reset ablation does not show, at D, that the consolidation state is
needed.

**L: native feasibility.**

- The reconditioning that distinguishes K3 from K1 costs up to `1.49e10`
  operations per consolidation, over the registered `1e10` budget.
- The same record shows that the reconditioning buys nothing measurable:
  - `MICROSTATE_EFFECT` is false;
  - `fibre_removed` holds 24/24;
  - K1 holds 24/24 with no cold-path cost.
- This is not a verdict on K1. K1 was not selected, and the gating ablations
  ran only for the selected candidate.

**M: numerical robustness.** The selected candidate's own outcome is kernel
invariant at the count level. The registered comparison is whole-record, and
it caught one K2 world whose catastrophe depends on the BLAS kernel. That is a
real fragility of K2 (`NUMERICALLY_FRAGILE`). The gate fails as registered.

**Predictions (CANDIDATES.md section 7) against the record:**

| prediction | record |
|---|---|
| M1 holds most worlds | correct: 24/24 |
| K1 matches or exceeds M1 | correct: 24/24, median earlier nmse 0.00057 against 0.0018 |
| K2 risks acquisition and second-order drift | one catastrophic world, kernel dependent |
| K3 against K1: no difference | correct (`MICROSTATE_EFFECT` false); DEV ranking still picked K3 by median nmse |
| C1R risks acquisition | correct: it did not learn D in 3/24 worlds (C in 1/24) |
| `state_removed` reproduces M0 | correct: identical counts |

**What R2 does not say.**

- No retention mechanism is qualified, by the registered rule.
- Nothing about capacity-limited regimes, larger `d`, other kernels, language
  or meaning.
- Nothing about K1 or K2 as selected candidates: their gating ablations were
  not run.

Its failures concern three things:

- a causal clause that the task's overlapping final stage can mask;
- the cold-path cost of the selected candidate;
- one kernel-dependent K2 world.

They are not a failure of the task's validity or of the laboratory's
mechanics.

## What a further version would need (not executed; a new experiment version)

Any follow-up is a new experiment version with its own RET-A step. This result
does not authorize editing the R2 v1 specification, thresholds, candidates,
gates or records. The R2 record states what a follow-up must address before
DEV:

- a consolidation-reset test whose measurement point no later experience can
  repair;
- a selected candidate whose cold path fits the registered native budget;
- an explicit, preregistered statement of which mechanisms' counts the
  numerical-robustness gate covers, justified on its own terms. Narrowing it
  because of this record would be a weakening, not a repair. A kernel-dependent
  catastrophe in any reported mechanism stays a finding.

These are requirements recorded so that a future preregistration can be
checked against them. They are not a design, and none of them is executed
here.

## Reproduction contract

| question | test | contract |
|---|---|---|
| evidence integrity | `tests/research/ecs_retention_r2/test_evidence.py` (byte pins and digests for DEV, freeze and QUAL; binding to RET2B and the freeze; disjoint worlds; recorded verdicts recomputed from recorded rows by the registered rule) | always |
| implementation correctness | `test_mechanics.py` (test-* worlds); `ctest -R ^ECS_G.` | always |
| historical replay | `test_evidence.py` (DEV controls; QUAL world `qual-0007`) | bitwise, only under the recorded implementation and numerical binding; skipped with the reason otherwise; never a scientific gate |
| current-runtime regression | `test_runtime_regression.py` (every QUAL world re-run from the frozen choices) | never skipped: validity, mechanics, gates A-L, disposition, outcome and every count equal to the recorded pre-robustness verdict |
| chronology | `test_preregistration.py` | from full git history |

## Consequences for the canonical system

None. R2 qualified no mechanism, so:

- the conditional native milestone was **not** started;
- Runtime R1, Mutable FMS R0 and every canonical ECS_G source are unchanged;
- `ELPISG01` (W and epoch) remains the only ECS_G cognitive serialization;
- no replay store, target archive or task label was introduced anywhere.

NO_CANONICAL_PROMOTION.
