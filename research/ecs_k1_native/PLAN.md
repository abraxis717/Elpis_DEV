# Native K1 differential qualification — plan (write-once)

`ecsg-k1-native.v1`. Machine-readable authority: `specs/ecsg-k1-native.v1.plan.json` (sha256 pinned in
`tests/research/ecs_k1_native/test_plan.py`). This file explains it and does not change it.

## Why

Retention R3 (`research/ecs_retention_r3`) ended **OUTCOME_A** with K1 executed as the canonical native G1 step plus
a correction computed in the NumPy laboratory. The native milestone (`libelpis_ecsg_k1`, `libelpis_ecsg_k1_fms`,
`elpis.ECS_G.k1`) re-implements the K1 law natively. It may be promoted only if it reproduces the frozen R3 science.
**Nativeization may not change OUTCOME_A.**

## What is compared, per R3 QUAL world (all 32, recorded order)

| comparison | rule |
|---|---|
| K1-disabled parity | a never-consolidated native state learning A, B, C, D from `w0` has the recorded **M0** W digest at every boundary, bitwise; no corrected step |
| W_A | native W after A equals the recorded K1 `W_A_digest`, bitwise |
| A/B/C/D boundaries | epoch exact; provenance COMPLETE; W, packed H, a within 1e-9 (relative) of the recomputed laboratory; native query within 1e-9 of the laboratory responses; native nmse within 1e-7 (relative) of the recorded nmse; nothing refused |
| resident path | the FMS-resident run equals the standalone run bitwise (envelopes) at every boundary |
| classification | R3 classes from native nmse equal the recorded K1 classes exactly |
| reset challenge | after B: W, epoch bitwise kept, H = a = 0, RESET; learn C; read at C; stop. e_reset within 1e-7 of recorded; RESET_DEGRADED and both_retained_at_C exact |
| W-only negative control | Runtime R1 snapshot of native W_B imports UNCONSOLIDATED; C from it equals the native reset branch bitwise and differs from the uninterrupted state; pass equals recorded |
| transplant continuation | the B envelope, restored (in process and in a clean process) and continued through C and D, equals the uninterrupted native envelopes bitwise |
| state semantics | full and reset B states answer every query bitwise identically; their C learning differs |
| determinism | two fresh native runs of the first world are bitwise identical |
| disposition | the unchanged R3 `experiment.gates` over the recorded rows with native K1, native state-removal and native causal rows substituted reproduce the recorded decision record (validity, mechanics, gates A-K, disposition, outcome, controls, every per-world K1 and causality row) |

The relative difference is `max|native - reference| / max(1, max|reference|)`; the nmse difference is
`|native - recorded| / max(recorded, 1e-12)`. The differences are rounding only: the native correction sums the same
terms in another order. The tolerances are fixed here, before any R3 QUAL world is run through native K1, and bind
every host. No CPU-model branch, forced BLAS kernel, per-host expectation or post-hoc tolerance is allowed.

## Verdict

**QUALIFIED** iff every comparison holds on every world and the substituted decision record is OUTCOME_A.
Anything else is **NOT_QUALIFIED**: the milestone is not promoted and the failure is recorded as found.

The qualification runs once and writes `evidence/ecsg-k1-native.v1.qualification.json` exclusively.
`tests/research/ecs_k1_native` re-runs the differential on the current runtime against the same tolerances and is
never skipped for a host difference.

## Disclosures

* Before this plan the native law was compared with the R3 laboratory on one world outside both R3 splits (id
  `test-0003`, a scratch development smoke, not recorded): W relative difference 7.5e-15 after 4000 corrected steps.
  No R3 DEV or QUAL world had been run through native K1.
* The native unit tests use synthetic data only.
