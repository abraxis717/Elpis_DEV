# ECS_G Retention R3: results (v1, confirmatory K1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Laboratory: [`research/ecs_retention_r3`](../../research/ecs_retention_r3/README.md).
Preregistration: [`PREREGISTRATION.md`](../../research/ecs_retention_r3/PREREGISTRATION.md),
`specs/ecsg-retention-r3.v1.spec.json` (RET3A, write-once, unchanged).
Prior results: [`ECS_RETENTION_R2_RESULTS.md`](ECS_RETENTION_R2_RESULTS.md) (closed:
`PARTIAL_REDUCTION`, `OUTCOME_C`; unchanged, not rescued).

**Disposition: `RETENTION_SUPPORTED_UNDER_FROZEN_SYNTHETIC_REGIME` (`OUTCOME_A`).**

On 32 fresh QUAL worlds of the inherited aliasing-conflict task, the fixed-size
K1 state `(W, epoch, H, a)`:

- acquired every experience and retained every earlier one in every world;
- caused that retention:
  - removing it reproduces plain G1's forgetting;
  - resetting it before an interfering experience loses protected function
    immediately;
  - transplanting it restores the continuation bitwise.

Every decision-bearing quantity was identical under the forced Prescott and
Haswell kernels, and the preregistered native budget held.

**K1 is qualified for the native milestone.** This result changes no canonical
code by itself. The native, FMS-materialized K1 runtime is a separate milestone
with its own differential qualification against this record (specification
`promotion`). Until that milestone merges, no retention mechanism is canonical.

## Chronology and authority

| step | commit | content |
|---|---|---|
| RET3A | `0bb1c7f` | question, inherited choices, the repaired causality design, thresholds, budgets, DEV rules, pass rule |
| RET3B | `3540887` | laboratory mechanics; mechanics tests on test-* worlds only |
| RET3C | `265ad8c` | DEV (8 fresh worlds): task valid, `K1_PROCEEDS` |
| RET3D | `0d33901` | freeze |
| RET3E | `525039b` | QUAL once (32 fresh worlds), with the gate-L robustness children |
| RET3F | this commit | interpretation and authority pointers |

| item | value |
|---|---|
| base | `main@1b5474bdd033aa8a32cb5ca3aa4a06d98bb337bb` |
| DEV record digest | `de81e79eb2e3b6d0316d08290d665db931dd89cc2db6c767efb033e124ecfeff` |
| frozen record digest | `541a2cdd8350c7fe875be9c08e944d6af387a15359723c6b6de58f0163ac3bc5` |
| QUAL record digest | `7e9417b3fbe74329c34bc83daf24bfdf3dcf2ea01ce9d500791ba7e2e08a2627` |
| produced by | clean trees: RET3B `3540887` (DEV), RET3D `0d33901` (QUAL); laboratory commit `3540887` |
| library | `libelpis_ecsg_math.so` `2374876e3a837761c87bdbc39389a2c5b2e007fcbaa81dd4b39e05deb1e5dc09` (the Runtime R1 binary), gcc 13.3.0 Release |
| numerical profile | Python 3.11.15, NumPy 1.26.4 (OpenBLAS 0.3.23.dev, ILP64, DYNAMIC_ARCH), effective OpenBLAS core SkylakeX, effective threads 1, x86_64 |
| robustness children | forced OpenBLAS cores Prescott and Haswell, effective threads 1 |

## DEV (8 fresh worlds)

The task was checked on controls only, at the inherited hidden gain 2.0.
V1–V6 held:

- M0 held the sequence in 0/8, with median earlier-experience nmse 0.615 and
  median A ratio 477;
- the minimum novelty was 0.644;
- M0 learned every stage, and M1 held 8/8;
- the joint ceiling was `1.2e-28`.

The K1 DEV rule held on every criterion:

- K1 held 8/8; state removal held 0/8;
- RESET_DEGRADED in 8/8, with ratios from 25 to 1664;
- transplant and the W-only control passed 8/8.

Disposition: `K1_PROCEEDS`. DEV chose nothing.

## QUAL (32 fresh worlds, run once)

| mechanism | sequence held | catastrophic (any) | median earlier-experience nmse at D | median final mean nmse |
|---|---|---|---|---|
| M0 plain G1 | 0 | 18 | 0.690 | 0.518 |
| M1 rehearsal (stores rows) | 32 | 0 | 0.0030 | 0.0036 |
| C1R (R0 reference) | 24 | 2 | 0.041 | 0.055 |
| **K1** | **32** | **0** | **0.0013** | **0.0012** |
| K1 state removed | 0 | 18 | 0.690 | (identical to M0) |

| causal test | result |
|---|---|
| state removal | 0/32 held; bitwise M0 at every stage (engine equivalence) |
| reset challenge at the C boundary | **RESET_DEGRADED 31/32**; median `e_reset` 0.367 against uninterrupted 0.0015; median ratio **263**; minimum `e_reset` 0.131; every reset record ends at C |
| full-state transplant at B | continuation bitwise 32/32, plus a clean-process probe |
| W-only (`ELPISG01`) negative control | 32/32: flagged UNCONSOLIDATED, different state digest, reproduces the reset branch, differs from the uninterrupted run |
| state semantics | queries identical with and without `(H, a)` in 32/32; learning differs in 32/32 |

One world was not RESET_DEGRADED. Its reset error was 8.5× the uninterrupted
error, under the factor of 10, though its absolute error of at least 0.131 was
above the floor. The registered gate requires 75%, and 31/32 is 97%.

As a descriptive count, A and B both still met the lenient class threshold
(0.5) right after C in 15/32 reset worlds. This is why R3 measures material
degradation rather than class loss, as preregistered.

### Gates (registered rule, unchanged since RET3A)

| gate | result |
|---|---|
| A current-stage acquisition | pass |
| B sequential retention | pass (32/32) |
| C no catastrophic forgetting | pass |
| D joint quality | pass (median 0.0012, threshold 0.25) |
| E single joint state | pass |
| F declared-state causality | pass (state removal; reset challenge at C; no re-teaching) |
| G no external answer store | pass |
| H deterministic state | pass |
| I transplant and reset behaviour | pass |
| J capacity accounting | pass (K1 extra state 28,552 B at every stage) |
| K native feasibility | pass (hot path 42,506 ops/step ≤ 82,944; consolidation 462,782 ops ≤ 1e8; closed form) |
| L numerical robustness | pass (decision record identical under Prescott and Haswell) |

Qualifiers:

- `EXCEEDS_R0_REFERENCE` is true: K1 held 32/32, against C1R's 24/32.
- `R0_REFERENCE_ALSO_SUFFICIENT` is false.
- `NUMERICALLY_FRAGILE` is false.
- `CAPACITY_SCOPE` applies: the claim covers only this task family.

## Interpretation

**What R3 establishes, within its frozen synthetic regime.**

The R2 aliasing-conflict family causes absolute forgetting under plain
sequential G1: 0/32 sequences held, and 18/32 worlds catastrophic. In that
regime, a fixed-size declared ECS_G state, K1's `(H, a)` beside `(W, epoch)`,
lets one W acquire four conflicting experiences and keep all of them. It needs
no stored rows, targets or task labels, and its retention matches or beats
rehearsal.

The repaired causal test answers the question R2 could not. It erases the
consolidation state after B and stops at the first post-interference boundary,
before any later experience can repair the damage. A and B are then lost
materially: a median 263× more error than uninterrupted K1 at the same
boundary.

The state is causal in both directions:

- **Necessary:** removing it, resetting it, or importing W alone forgets.
- **Sufficient:** transplanting the complete state restores the uninterrupted
  continuation bitwise.

**Why this is a repair and not a rescue.**

- R2 stays `PARTIAL_REDUCTION`. Its records and specification are untouched.
- R3 is a new preregistered experiment, with fresh worlds and a single
  eligible mechanism fixed before data.
- The reset criterion was fixed at RET3A, and the R2 rows that informed it were
  disclosed there.
- Gate L's scope was fixed before data and covers every decision-bearing
  quantity, so no K1 row could be exempted after the fact.

**Implementation decision recorded before DEV.** The mechanics check
`consolidation_state_shapes_learning` compares the learning of experience C
from the complete and the reset state. The literal "one K1 step" from the
consolidated state is identical by construction, because the correction at the
anchor is zero. This decision is recorded in the laboratory README at RET3B,
and both facts are proven in the mechanics tests.

**Predictions (PREREGISTRATION.md section 12):**

- **P1:** correct; the task was valid on DEV.
- **P2:** correct; K1 held 32/32.
- **P3:** correct; the reset degraded 31/32 worlds.
- **P4:** correct; state removal is bitwise M0.
- **P5:** correct; the decision record was kernel invariant.

**What R3 does not say:**

- It says nothing about capacity-limited regimes, other task families,
  larger `d`, other kernels, language or meaning.
- It says nothing about K2, K3 or any other mechanism.
- It does not make K1 canonical. It makes K1 eligible for the native
  milestone, which must reproduce this science natively.

## Reproduction contract

| question | test | contract |
|---|---|---|
| evidence integrity | `tests/research/ecs_retention_r3/test_evidence.py` (byte pins and digests; binding to RET3B and the freeze; disjoint worlds; DEV and QUAL verdicts recomputed from recorded rows by the registered rules) | always |
| implementation correctness | `test_mechanics.py` (test-* worlds); `ctest -R ^ECS_G.` | always |
| historical replay | `test_evidence.py` (DEV world `r3dev-0003`, QUAL world `r3qual-0011`) | bitwise, only under the recorded binding; skipped with the reason otherwise; never a scientific gate |
| current-runtime regression | `test_runtime_regression.py` (every QUAL world re-run on the host's natural kernel) | never skipped; the decision record (every decision-bearing quantity) must equal the record; nothing is exempt because gate L held |
| chronology | `test_preregistration.py` | from full git history |

## Consequences for the canonical system

- **Promotion.** K1 is the first retention mechanism to earn `OUTCOME_A`. The
  specification's `promotion` section governs what follows: a native,
  FMS-materialized K1 runtime on a separate integration branch from this
  commit, differentially qualified against every R3 QUAL world. It needs:
  - native QUERY, LEARN and CONSOLIDATE;
  - transactions;
  - a new versioned retained-state envelope;
  - generic FMS residency;
  - Runtime R1 parity with K1 disabled;
  - allocation, sanitizer and performance evidence.
- **Unchanged until that milestone merges:** canonical ECS_G code, Runtime R1,
  Mutable FMS R0 and `ELPISG01`.
