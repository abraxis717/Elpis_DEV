# Native K1 differential qualification — results

## Current authority: K1N-v2 QUALIFIED

The preregistered [v2 plan](../../research/ecs_k1_native/PLAN_V2.md) ran once from clean harness commit
`f720e4a906b0f9c1190536e6123de9c7f4e6160e`, with the host's natural numerical kernel and single-thread constraints.
All nine gates passed on Q (all 32 frozen R3 QUAL worlds) and F (the first evaluation of
`k1n2-0000..0031`). No threshold, world, scientific implementation or qualification rule changed after the harness
commit. Native K1 is canonical through the exact record admitted in `tests/research/_k1_promotion.py`.

Record: `research/ecs_k1_native/evidence/ecsg-k1-native.v2.qualification.json`.

- SHA-256: `c96fa766b61bd05ceb9181ef0566123e6ba6a0afbca7e10367eaf580421994c2`
- Internal qualification digest: `fbea9d43e11706d789c6aac575d3412b0a6ada0c83bd4c0872d2ef2c55cf03aa`
- E1–E3: every registered same-host invariant, native determinism and clean-process transplant passed.
- E4–E5: all six native K1 tests passed; every measured Python operation crossed into native code once,
  including K = 1, 10 and 4000; the warm heap-allocation counter stayed unchanged.
- L1: maximum 16-step W relative difference `6.220227869312611e-16` (bound `1e-10`).
- L2: maximum H relative difference `4.41867780689024e-16` and a difference `7.128190578946582e-16`
  (bound `1e-12` each).
- D1_Q: exact frozen R3 decision-record equivalence, including final `OUTCOME_A`.
- D1_F: exact same-host laboratory decision-record equivalence on every key and world; laboratory determinism
  and transplant passed. The F laboratory's descriptive outcome was `OUTCOME_A`.

Retention R3 remains **OUTCOME_A**. K1N-v1 remains permanently **NOT_QUALIFIED**; its record and the historical
interpretation below are unchanged. Long-horizon coordinate differences are descriptive under v2.

Current-host regression (Q only; never reruns F):

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
python -m research.ecs_k1_native.regression --library build/native/ECS_G/libelpis_ecsg_math.so
```

## Historical K1N-v1 interpretation (unchanged)

`ecsg-k1-native.v1` (`research/ecs_k1_native`). Plan: [PLAN.md](../../research/ecs_k1_native/PLAN.md)
(K1N-B, committed before any differential evidence). Record:
`research/ecs_k1_native/evidence/ecsg-k1-native.v1.qualification.json` (digest `0c1c5543…`, run once from the
clean K1N-C harness commit `44d4d85`).

## Verdict: **NOT_QUALIFIED**

The native K1 runtime is **not promoted**. Under the plan, QUALIFIED required every comparison on every R3 QUAL
world. 30 of 32 worlds pass every comparison. On `r3qual-0019` and `r3qual-0028` the native W, a, query and nmse
are outside the planned float tolerances (1e-9 relative for W/H/a/query, 1e-7 for nmse and e_reset). These
tolerances were fixed before the run and are not changed after it: no post-hoc tolerance, no per-host
expectation, no rescue.

| | result |
|---|---|
| worlds passing every comparison | 30 / 32 (failing: r3qual-0019, r3qual-0028; checks `boundaries`, `reset_challenge`) |
| max W relative difference (128 boundaries) | 8.8e-2 (r3qual-0028 at C); ≤ 4e-14 on the 30 passing worlds |
| max nmse relative difference | 0.56 (r3qual-0028 at B) |
| K1-disabled native = recorded M0 (Runtime R1 G1) digests, bitwise | 32 / 32, every boundary, no corrected step |
| native W_A = recorded K1 W_A, bitwise | 32 / 32 |
| FMS-resident run = standalone run (envelopes, bitwise) | 32 / 32 |
| K1 classes and state-removal classes exact | 32 / 32 |
| reset challenge: RESET_DEGRADED, both_retained_at_C exact | 32 / 32 (e_reset tolerance fails on the 2 worlds) |
| W-only negative control | 32 / 32 |
| transplant continuation, in process / clean process (bitwise) | 32 / 32 / 32 |
| state semantics; determinism | 32 / 32; identical |
| R3 gates over the recorded rows with every native row substituted | decision record identical to the recorded one; gates A-K true; **OUTCOME_A** |

Nativeization did **not** change the R3 outcome or any decision-bearing R3 quantity. It failed the plan's own
float-agreement requirement on two worlds.

## Diagnosis (post hoc, descriptive; decides nothing)

`python -m research.ecs_k1_native.diagnose` (K1N-E). The native and laboratory W agree to rounding (≤ 2e-14) at the
start of the failing stage and for the first ~1000-2000 steps. Then the difference grows by more than ten orders of
magnitude within the stage:

| world, stage | native vs lab after 1000 / 2000 / 3000 / 4000 steps | lab vs lab, one ulp on one W entry at stage start |
|---|---|---|
| r3qual-0028, B | 1.4e-15 / 6.7e-12 / 6.1e-5 / 1.4e-2 | ≈ 1.6e-3 at stage end |
| r3qual-0019, C | 1.2e-14 / 9.7e-15 / 1.5e-2 / 1.1e-4 | ≈ 1.2e-4 at stage end |
| r3qual-0000, B (control) | 2.2e-15 / 3.2e-15 / 5.4e-15 / 6.0e-15 | 4.4e-15 |

The unchanged R3 laboratory amplifies a single-ulp perturbation of its own W to the same order as the native
difference on these two worlds, and not on the control world. The difference is therefore a sensitivity of the frozen
K1 trajectory (the corrected G1 dynamics on these worlds), not a defect in the native arithmetic. In the
second diagnostic run, the perturbed-laboratory figures differed from the first in their third or fourth digit.
That variation is further evidence of the same sensitivity. The plan's rationale, that the differences are rounding
only, held on 30 worlds and failed on these 2. The R3 record compared its BLAS kernels on the decision record only,
not on W. The planned float tolerances were stricter than R3's own reproducibility standard. That is a defect of the
K1N-B plan, and the plan stands as written.

## What this means

* Retention R3 stays **OUTCOME_A** (K1 qualified as research). The native runtime reproduces every R3 classification,
  causal result and the disposition.
* Its float agreement with the laboratory is not uniform. Under its preregistered plan the native milestone is
  **NOT_QUALIFIED**, and no retention mechanism is canonical.
* The canonical-vocabulary guards of Retention R1/R2/R3 now admit the K1 milestone files only when the R3 OUTCOME_A
  record exists **and** a native qualification record states QUALIFIED (`tests/research/_k1_promotion.py`). On this
  branch they fail by design, so the branch cannot merge green.
* Any further qualification must be a new plan version, preregistered before it runs. It would have to declare how
  it treats trajectory sensitivity (for example, by comparing decision-bearing quantities only, as R3 compared its
  BLAS kernels, or on fresh worlds). That is a decision for the repository owner, not a repair of v1.

## Reproduce

```
PYTHONPATH=src:. python -m research.ecs_k1_native.run check --library build/native/ECS_G/libelpis_ecsg_math.so
PYTHONPATH=src:. python -m research.ecs_k1_native.diagnose --library build/native/ECS_G/libelpis_ecsg_math.so
pytest tests/research/ecs_k1_native
```
