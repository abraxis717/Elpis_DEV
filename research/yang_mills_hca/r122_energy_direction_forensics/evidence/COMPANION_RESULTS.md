# R122 consumed companion (R121 graphs and seeds) — discovery material, no decision authority

**Mode:** `CONSUMED_COMPANION_NO_DECISION_AUTHORITY`  
**Disposition:** `CONSUMED_COMPANION:MECHANISM_MIXED_BOUNDED`  
**R93:** `ENERGY_NORMALIZED_DENOMINATOR_HOLD` (unchanged; a diagnostic cannot clear it)

Protocol `fe9ec864901d3e027798536576248ede9fe2bb8243fbee146d4edd5ba4bee31d`; source digest `4f3d407f4280641074264eca91cbbc6420b63e10f0d319a1eb651b1fa3db1f4f`; raw tensor `8be796ad7b5fc2e7c2a44fdb1dc4358d575cc4f69a59943720df18ff54066a5b`; source commit `eafdb15293c6e47010c159c9adb8d7977e22e868`.

## Integrity gates

| Gate | Result |
|---|---|
| AUTONOMOUS_GATE | PASS |
| CHECKPOINTS_UNCHANGED | PASS |
| DETECTOR_CONTROLS | PASS |
| ENERGY_DECOMPOSITION_GATE | PASS |
| FROZEN_PARITY | PASS |
| GRAPH_NOVELTY | PASS |
| GRAPH_NOVELTY_OF_COMPANION | NOT_APPLICABLE_CONSUMED |
| LINEAR_OPERATOR_CONTROLS | PASS |
| LOCAL_TRANSITION_GATE | PASS |
| NUMERICAL_PARITY | PASS |
| RESOURCE_BOUND | PASS |
| SEED_DISJOINTNESS | PASS |
| SOURCE_PINS | PASS |

- Fixtures 384; HCA passes 464,640; non-finite 0 = True.
- Composed teacher at k = 0 vs historical isolated teacher: max abs 0.0e+00 (0 bit mismatches in 2560 step checks).
- Independent torch-free teacher reference: max relative deviation 3.81e-16 (192 steps).
- EXACT_GUARDED vs EXACT_R118: 0 mismatching trajectories of 288; fresh-state replay 0/288; autonomous recheck after diagnostics 0/768; DELAYED8 H8 identity 0/192.
- Decomposition identities: max residual 1.58e-15; exact rational audit 3.86e-16 over 2304 samples; telescoping residuals 4.85e-16 (one-step) and 4.43e-16 (restart).
- Gauge: joint covariance 8.9e-16; metric invariance 4.6e-14; state-only negative control ≥ 0.137; prediction-only half / quarter turns detected 2452 / 2452 of 2452.
- Consumed R121 regression fixture: 216 published cell values reproduced (worst relative deviation 1.5e-11); inversion detected, extinguished predictions not reported as reversals.
- Resources: 332.5 s, peak RSS 1033 MiB, raw tensor 260 MiB (volatile).
- READOUT autonomous samples by norm class: {"ORDINARY": 552960}; non-ordinary horizons: none.

## Registered mechanism decision (NATIVE, READOUT, H20, scale 1)

| Scope | DENOMINATOR_COLLAPSE | LOCAL_TRANSITION | ACCUMULATED_DRIFT | MIXED_LOCAL_ACCUMULATED | INVALID | dominant |
|---|---:|---:|---:|---:|---:|---|
| ALL | 0 | 3 | 13 | 8 | 0 | none |
| SIGNED | 0 | 0 | 11 | 1 | 0 | ACCUMULATED_DRIFT |
| UNSIGNED | 0 | 3 | 2 | 7 | 0 | none |

Cohort dispositions: {"SIGNED": "LOCALIZED:ACCUMULATED_DRIFT", "UNSIGNED": "NOT_LOCALIZED"}.

Per graph class (strata = 3 overlaps × 2 cohorts):

| Graph | label counts |
|---|---|
| R121_HOLDOUT_1 | {"ACCUMULATED_DRIFT": 3, "LOCAL_TRANSITION": 1, "MIXED_LOCAL_ACCUMULATED": 2} |
| R121_HOLDOUT_2 | {"ACCUMULATED_DRIFT": 4, "MIXED_LOCAL_ACCUMULATED": 2} |
| R121_HOLDOUT_3 | {"ACCUMULATED_DRIFT": 3, "LOCAL_TRANSITION": 2, "MIXED_LOCAL_ACCUMULATED": 1} |
| R121_HOLDOUT_4 | {"ACCUMULATED_DRIFT": 3, "MIXED_LOCAL_ACCUMULATED": 3} |

### Primary strata

| Stratum | mechanism | E_rel mean | r | c | λ local | local E_rel | skill | A_den | state E_rel | state c | E_perp share | growth H8→H20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R121_HOLDOUT_1/k0/SIGNED | ACCUMULATED_DRIFT | 2.8423 | 2.677 | 0.99995 | 0.024 | 0.0679 | -0.53 | 0.767 | 3.694 | 0.523 | 0.0002 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_1/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.0963 | 1.303 | 0.99998 | 0.457 | 0.0402 | 0.10 | 0.123 | 0.779 | 0.632 | 0.0008 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_1/k2/SIGNED | ACCUMULATED_DRIFT | 7.7215 | 3.778 | 0.99990 | 0.009 | 0.0706 | 0.03 | 0.992 | 7.776 | 0.494 | 0.0004 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_1/k2/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.2760 | 1.502 | 0.99978 | 0.185 | 0.0384 | 0.47 | 0.282 | 0.964 | 0.608 | 0.0047 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_1/k6/SIGNED | ACCUMULATED_DRIFT | 5.5541 | 3.367 | 0.99986 | 0.008 | 0.0420 | 0.57 | 0.864 | 6.476 | 0.602 | 0.0006 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_1/k6/UNSIGNED | LOCAL_TRANSITION | 0.0115 | 1.097 | 0.99991 | 2.873 | 0.0217 | 0.78 | 0.017 | 0.630 | 0.676 | 0.0270 | NO_GROWTH |
| R121_HOLDOUT_2/k0/SIGNED | ACCUMULATED_DRIFT | 2.3810 | 2.535 | 1.00000 | 0.041 | 0.0957 | -0.44 | 0.770 | 3.063 | 0.432 | 0.0000 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_2/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.1382 | 0.630 | 1.00000 | 0.424 | 0.0548 | 0.18 | 0.252 | 0.583 | 0.695 | 0.0000 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_2/k2/SIGNED | ACCUMULATED_DRIFT | 0.8137 | 1.892 | 0.99999 | 0.064 | 0.0481 | 0.39 | 0.433 | 1.837 | 0.572 | 0.0001 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_2/k2/UNSIGNED | ACCUMULATED_DRIFT | 0.3085 | 0.446 | 0.99999 | 0.090 | 0.0265 | 0.66 | 0.540 | 0.586 | 0.731 | 0.0000 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_2/k6/SIGNED | MIXED_LOCAL_ACCUMULATED | 0.3152 | 1.544 | 0.99999 | 0.114 | 0.0284 | 0.60 | 0.248 | 1.196 | 0.671 | 0.0001 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_2/k6/UNSIGNED | ACCUMULATED_DRIFT | 0.3909 | 0.375 | 0.99999 | 0.042 | 0.0158 | 0.78 | 0.696 | 0.568 | 0.764 | 0.0000 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_3/k0/SIGNED | ACCUMULATED_DRIFT | 0.8951 | 1.940 | 0.99947 | 0.054 | 0.0476 | -0.68 | 0.470 | 1.894 | 0.635 | 0.0043 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_3/k0/UNSIGNED | LOCAL_TRANSITION | 0.0147 | 1.120 | 0.99988 | 1.993 | 0.0298 | -0.05 | 0.025 | 0.589 | 0.708 | 0.0204 | NO_GROWTH |
| R121_HOLDOUT_3/k2/SIGNED | ACCUMULATED_DRIFT | 4.9374 | 3.228 | 0.99989 | 0.009 | 0.0444 | 0.44 | 0.865 | 5.703 | 0.582 | 0.0004 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_3/k2/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.1552 | 1.372 | 0.99997 | 0.233 | 0.0245 | 0.69 | 0.172 | 0.884 | 0.652 | 0.0006 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_3/k6/SIGNED | ACCUMULATED_DRIFT | 3.4886 | 2.867 | 0.99997 | 0.009 | 0.0299 | 0.73 | 0.706 | 4.935 | 0.640 | 0.0001 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_3/k6/UNSIGNED | LOCAL_TRANSITION | 0.0128 | 1.011 | 0.99990 | 1.362 | 0.0156 | 0.86 | 0.019 | 0.639 | 0.675 | 0.0168 | NO_GROWTH |
| R121_HOLDOUT_4/k0/SIGNED | ACCUMULATED_DRIFT | 1.8669 | 2.353 | 0.99881 | 0.031 | 0.0568 | -0.61 | 0.623 | 2.957 | 0.571 | 0.0067 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_4/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.1578 | 1.387 | 0.99918 | 0.224 | 0.0344 | 0.02 | 0.171 | 0.907 | 0.643 | 0.0201 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_4/k2/SIGNED | ACCUMULATED_DRIFT | 4.5078 | 3.123 | 0.99897 | 0.013 | 0.0563 | 0.09 | 0.864 | 5.227 | 0.537 | 0.0044 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_4/k2/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.2523 | 1.500 | 0.99925 | 0.125 | 0.0316 | 0.49 | 0.257 | 0.998 | 0.627 | 0.0134 | DENOMINATOR_DRIVEN |
| R121_HOLDOUT_4/k6/SIGNED | ACCUMULATED_DRIFT | 9.7510 | 4.118 | 0.99992 | 0.005 | 0.0489 | 0.60 | 0.956 | 10.160 | 0.529 | 0.0003 | NUMERATOR_DRIVEN |
| R121_HOLDOUT_4/k6/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.3341 | 1.540 | 0.99997 | 0.122 | 0.0246 | 0.80 | 0.247 | 1.260 | 0.575 | 0.0006 | DENOMINATOR_DRIVEN |

Origin of the H20 READOUT error (median projection shares of the step-20 error by the block of steps whose local defects produced it; exact sum 1):

| Stratum | steps 1–4 | 5–8 | 9–12 | 13–16 | 17–20 | Σ|share| |
|---|---:|---:|---:|---:|---:|---:|
| R121_HOLDOUT_1/k0/SIGNED | 0.09 | 0.44 | 0.59 | 0.24 | -0.37 | 1.74 |
| R121_HOLDOUT_1/k0/UNSIGNED | 0.38 | 1.76 | 1.44 | -0.39 | -2.17 | 6.15 |
| R121_HOLDOUT_1/k2/SIGNED | 0.12 | 0.49 | 0.49 | 0.14 | -0.25 | 1.49 |
| R121_HOLDOUT_1/k2/UNSIGNED | 0.16 | 1.40 | 1.06 | -0.25 | -1.36 | 4.22 |
| R121_HOLDOUT_1/k6/SIGNED | 0.13 | 0.51 | 0.46 | 0.13 | -0.23 | 1.47 |
| R121_HOLDOUT_1/k6/UNSIGNED | 0.23 | 4.28 | 3.34 | -1.32 | -5.47 | 14.76 |
| R121_HOLDOUT_2/k0/SIGNED | 0.27 | 0.65 | 0.54 | 0.09 | -0.55 | 2.10 |
| R121_HOLDOUT_2/k0/UNSIGNED | -0.51 | -1.65 | -0.31 | 1.17 | 2.29 | 6.02 |
| R121_HOLDOUT_2/k2/SIGNED | 0.31 | 0.67 | 0.59 | 0.14 | -0.70 | 2.40 |
| R121_HOLDOUT_2/k2/UNSIGNED | -0.13 | -0.42 | -0.09 | 0.56 | 1.08 | 2.28 |
| R121_HOLDOUT_2/k6/SIGNED | 0.39 | 0.79 | 0.62 | 0.13 | -0.94 | 2.88 |
| R121_HOLDOUT_2/k6/UNSIGNED | -0.08 | -0.16 | 0.04 | 0.43 | 0.77 | 1.48 |
| R121_HOLDOUT_3/k0/SIGNED | 0.04 | 0.35 | 0.67 | 0.42 | -0.49 | 1.99 |
| R121_HOLDOUT_3/k0/UNSIGNED | 0.53 | 2.37 | 2.76 | -0.29 | -4.34 | 10.19 |
| R121_HOLDOUT_3/k2/SIGNED | 0.06 | 0.42 | 0.55 | 0.20 | -0.23 | 1.47 |
| R121_HOLDOUT_3/k2/UNSIGNED | -0.22 | 1.41 | 1.37 | -0.09 | -1.47 | 4.56 |
| R121_HOLDOUT_3/k6/SIGNED | 0.13 | 0.46 | 0.49 | 0.16 | -0.25 | 1.49 |
| R121_HOLDOUT_3/k6/UNSIGNED | 0.20 | 0.21 | -0.17 | 0.01 | 0.76 | 10.85 |
| R121_HOLDOUT_4/k0/SIGNED | 0.14 | 0.44 | 0.55 | 0.27 | -0.40 | 1.81 |
| R121_HOLDOUT_4/k0/UNSIGNED | 0.80 | 1.14 | 0.86 | -0.28 | -1.52 | 4.60 |
| R121_HOLDOUT_4/k2/SIGNED | 0.19 | 0.46 | 0.46 | 0.17 | -0.28 | 1.56 |
| R121_HOLDOUT_4/k2/UNSIGNED | 0.77 | 0.98 | 0.69 | -0.26 | -1.18 | 3.89 |
| R121_HOLDOUT_4/k6/SIGNED | 0.32 | 0.43 | 0.37 | 0.08 | -0.20 | 1.40 |
| R121_HOLDOUT_4/k6/UNSIGNED | 0.74 | 1.03 | 0.69 | -0.32 | -1.13 | 3.91 |

### By amplitude scale and by horizon (label counts over all strata)

| Scale | DENOM | LOCAL | ACCUM | MIXED | INVALID | dominant | label agreement with scale 1 |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 0 | 3 | 13 | 8 | 0 | none | — |
| 2^-2 | 7 | 2 | 15 | 0 | 0 | none | 0.25 |
| 2^-20 | 8 | 0 | 16 | 0 | 0 | none | 0.38 |
| 2^2 | 0 | 0 | 1 | 23 | 0 | MIXED_LOCAL_ACCUMULATED | 0.29 |

| Horizon (scale 1) | DENOM | LOCAL | ACCUM | MIXED | INVALID | SIGNED dominant | UNSIGNED dominant |
|---|---:|---:|---:|---:|---:|---|---|
| H1 | 19 | 5 | 0 | 0 | 0 | none | DENOMINATOR_COLLAPSE |
| H12 | 0 | 10 | 4 | 10 | 0 | none | none |
| H16 | 0 | 9 | 10 | 5 | 0 | none | none |
| H20 | 0 | 3 | 13 | 8 | 0 | ACCUMULATED_DRIFT | none |
| H4 | 0 | 16 | 4 | 4 | 0 | none | none |
| H8 | 0 | 15 | 0 | 9 | 0 | LOCAL_TRANSITION | none |

## Normalized-error growth H8 → H20 (R93 pattern = absolute defect falls while normalized defect rises)

| Policy | Scale | Cohort | NO_GROWTH | NUMERATOR_DRIVEN | DENOMINATOR_DRIVEN | dominant |
|---|---|---|---:|---:|---:|---|
| DELAYED8 | 1 | SIGNED | 1 | 9 | 2 | none |
| DELAYED8 | 1 | UNSIGNED | 3 | 0 | 9 | none |
| NATIVE | 1 | SIGNED | 0 | 11 | 1 | NUMERATOR_DRIVEN |
| NATIVE | 1 | UNSIGNED | 3 | 0 | 9 | none |
| NATIVE | 2^-2 | SIGNED | 0 | 11 | 1 | NUMERATOR_DRIVEN |
| NATIVE | 2^-2 | UNSIGNED | 1 | 2 | 9 | DENOMINATOR_DRIVEN |
| NATIVE | 2^-20 | SIGNED | 0 | 4 | 8 | none |
| NATIVE | 2^-20 | UNSIGNED | 0 | 2 | 10 | DENOMINATOR_DRIVEN |
| NATIVE | 2^2 | SIGNED | 4 | 8 | 0 | none |
| NATIVE | 2^2 | UNSIGNED | 0 | 12 | 0 | NUMERATOR_DRIVEN |
| R114 | 1 | SIGNED | 7 | 0 | 5 | none |
| R114 | 1 | UNSIGNED | 0 | 0 | 12 | DENOMINATOR_DRIVEN |
| SIGNED_ONLY | 1 | SIGNED | 7 | 0 | 5 | none |
| SIGNED_ONLY | 1 | UNSIGNED | 3 | 0 | 9 | none |

## Secondary findings (fixed criteria)

- **S1_readout_error_collinear_with_teacher**: SUPPORTED (strata fraction 1.000; median E_perp/E_rel <= 0.05 in >= 0.75 of primary strata).
- **S2_paired_state_direction_not_preserved**: SUPPORTED (strata fraction 1.000; median PAIRED_STATE cosine < 0.9 in >= 0.75 of primary strata).
- **S3_local_step_beats_persistence**: LOCALLY_SKILLFUL (strata fraction 0.792; median teacher-forced one-step skill vs teacher persistence > 0).
- **S4 decayed-regime operator**: PASS — kernel probe is exactly linear (homogeneity 0.0e+00); the eight-gate analytic operator equals it to 1.4e-16; linear-limit trajectories follow operator powers to 5.3e-05.

Limiting gate values (zero invariants): teacher vs checkpoints

| | k0 A←A | k1 A←sal | k2 A←col | k3 A←B | k4 B←B | k5 B←sal | k6 B←col | k7 B←A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher | 0.6570 | 0.2449 | 0.0997 | 0.0300 | 0.6900 | -0.0997 | 0.1781 | 0.0200 |
| 54007/STATE8 | 0.6468 | 0.2228 | 0.1818 | 0.1035 | 0.7693 | -0.0452 | 0.1875 | 0.0184 |
| 54007/STATE8_SIGNED8 | 0.6410 | 0.2152 | 0.1723 | 0.1859 | 0.7621 | -0.0062 | 0.1227 | 0.0276 |
| 54007/STATE8_TF1 | 0.6471 | 0.2236 | 0.1824 | 0.1039 | 0.7694 | -0.0452 | 0.1875 | 0.0185 |
| 54031/STATE8 | 0.6435 | 0.1923 | 0.1700 | 0.1109 | 0.7668 | -0.0381 | 0.1404 | 0.0239 |
| 54031/STATE8_SIGNED8 | 0.6430 | 0.2212 | 0.1666 | 0.2029 | 0.7575 | 0.0090 | 0.1281 | 0.0138 |
| 54031/STATE8_TF1 | 0.6438 | 0.1931 | 0.1708 | 0.1115 | 0.7669 | -0.0382 | 0.1404 | 0.0235 |

Decayed-regime spectra relative to the teacher (over graph × k × checkpoint):

| Cohort / operator | n | spectral-radius ratio median (min–max) | fraction decaying slower than teacher | dominant phase − teacher (°/step, median) | fraction with faster phase |
|---|---:|---|---:|---:|---:|
| SIGNED | 24 | 1.0910 (1.0773–1.0948) | 1.000 | -5.711 | 0.000 |
| UNSIGNED | 48 | 1.0220 (0.9803–1.0326) | 0.667 | -2.686 | 0.000 |
| UNSIGNED_R104 | 48 | 0.9434 (0.8997–0.9857) | 0.000 | -0.693 | 0.271 |

## Directional safety (READOUT, all twenty steps)

| Policy | in-range verdict | induced reversals in-range | above-clamp verdict | induced above-clamp | reversed samples by scale | induced by cohort |
|---|---|---:|---|---:|---|---|
| DELAYED8 | NO_INDUCED_REVERSAL_OBSERVED | 0 | NO_INDUCED_REVERSAL_OBSERVED | 0 | {"1": 1184, "2^-2": 128, "2^-20": 0, "2^2": 8592} | {"SIGNED": 0, "UNSIGNED": 0} |
| NATIVE | REFERENCE | 0 | REFERENCE | 0 | {"1": 1184, "2^-2": 128, "2^-20": 0, "2^2": 8581} | {"SIGNED": 0, "UNSIGNED": 0} |
| R114 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 1843 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 14186 | {"1": 1568, "2^-2": 1491, "2^-20": 0, "2^2": 19764} | {"SIGNED": 4629, "UNSIGNED": 11400} |
| SIGNED_ONLY | HAZARD_CONTROLLER_INDUCED_REVERSAL | 224 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 4405 | {"1": 1376, "2^-2": 128, "2^-20": 0, "2^2": 9672} | {"SIGNED": 4629, "UNSIGNED": 0} |

- DELAYED8: reversed by step [576, 2848, 1760, 1431, 749, 495, 306, 96, 107, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256]; induced by step [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; induced by graph {"R121_HOLDOUT_1": 0, "R121_HOLDOUT_2": 0, "R121_HOLDOUT_3": 0, "R121_HOLDOUT_4": 0}; first-reversal step histogram {"1": 18, "2": 87, "20": 4, "3": 2}.
- NATIVE: reversed by step [576, 2848, 1760, 1431, 749, 495, 306, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256]; induced by step [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; induced by graph {"R121_HOLDOUT_1": 0, "R121_HOLDOUT_2": 0, "R121_HOLDOUT_3": 0, "R121_HOLDOUT_4": 0}; first-reversal step histogram {"1": 18, "2": 87, "20": 4, "3": 2}.
- R114: reversed by step [576, 2336, 1920, 1632, 1472, 1249, 1280, 1139, 1152, 992, 1066, 864, 938, 821, 960, 864, 930, 798, 1039, 795]; induced by step [0, 224, 354, 760, 888, 949, 1087, 1126, 1088, 960, 1002, 832, 875, 789, 910, 832, 888, 766, 1000, 699]; induced by graph {"R121_HOLDOUT_1": 4392, "R121_HOLDOUT_2": 4242, "R121_HOLDOUT_3": 3578, "R121_HOLDOUT_4": 3817}; first-reversal step histogram {"1": 18, "13": 2, "15": 2, "17": 2, "19": 8, "2": 71, "20": 2, "3": 12, "4": 2, "5": 6, "6": 1}.
- SIGNED_ONLY: reversed by step [576, 2272, 1376, 823, 576, 481, 512, 371, 448, 352, 448, 352, 426, 309, 384, 288, 352, 222, 288, 320]; induced by step [0, 224, 0, 0, 0, 181, 319, 358, 384, 320, 384, 320, 363, 277, 334, 256, 310, 190, 249, 160]; induced by graph {"R121_HOLDOUT_1": 1134, "R121_HOLDOUT_2": 754, "R121_HOLDOUT_3": 1494, "R121_HOLDOUT_4": 1247}; first-reversal step histogram {"1": 18, "2": 69, "20": 4, "3": 14, "4": 2, "5": 6, "6": 1}.

## Policy comparison at scale 1 (pooled READOUT NSE, equal-weight graph × checkpoint cells)

| Overlap | Horizon | NATIVE | R114 | SIGNED_ONLY | DELAYED8 |
|---|---|---:|---:|---:|---:|
| k0 | H8 | 0.037547 | 0.177043 | 0.078421 | 0.037547 |
| k0 | H20 | 0.733277 | 0.366036 | 0.102787 | 0.295548 |
| k2 | H8 | 0.041787 | 0.240631 | 0.093276 | 0.041787 |
| k2 | H20 | 1.663713 | 0.469996 | 0.218305 | 0.479591 |
| k6 | H8 | 0.061208 | 0.354259 | 0.142090 | 0.061208 |
| k6 | H20 | 1.717290 | 0.591719 | 0.209831 | 0.226078 |

| Overlap / policy | mean ΔE_rel vs NATIVE (H20) | graph blocks improved | graph-cluster bootstrap 95 % | worst graph block |
|---|---:|---:|---|---|
| k0/DELAYED8 | -0.4377 | 4/4 | [-0.5653, -0.2994] | -0.2170 (R121_HOLDOUT_3) |
| k0/R114 | -0.3672 | 4/4 | [-0.5951, -0.1394] | -0.0557 (R121_HOLDOUT_3) |
| k0/SIGNED_ONLY | -0.6305 | 4/4 | [-0.8119, -0.4126] | -0.2944 (R121_HOLDOUT_3) |
| k2/DELAYED8 | -1.1841 | 4/4 | [-1.6986, -0.5682] | -0.2698 (R121_HOLDOUT_2) |
| k2/R114 | -1.1937 | 3/4 | [-2.0980, -0.1277] | 0.2859 (R121_HOLDOUT_2) |
| k2/SIGNED_ONLY | -1.4454 | 4/4 | [-2.2429, -0.5358] | -0.1686 (R121_HOLDOUT_2) |
| k6/DELAYED8 | -1.4912 | 4/4 | [-2.5133, -0.4908] | -0.0593 (R121_HOLDOUT_2) |
| k6/R114 | -1.1256 | 3/4 | [-2.4670, 0.0321] | 0.4714 (R121_HOLDOUT_2) |
| k6/SIGNED_ONLY | -1.5075 | 3/4 | [-2.7007, -0.3852] | 0.0853 (R121_HOLDOUT_2) |

Worst graph-block harm over k: {"DELAYED8": -0.05928129796, "R114": 0.4713869079, "SIGNED_ONLY": 0.08529640345}.

Controller-specific contributions:
- Teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in 93.5 % of SIGNED samples (by scale {"1": 0.8953776042, "2^-2": 0.925, "2^-20": 0.9604166667, "2^2": 0.9583333333}).
- Teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in 92.4 % of UNSIGNED samples (by scale {"1": 0.8958333333, "2^-2": 0.925, "2^-20": 0.95, "2^2": 0.9270182292}).
- DELAYED8/SIGNED: the controller's displacement raised the step error in 28.7 % of controller-active steps (by scale {"1": 0.3985460069, "2^-2": 0.0, "2^-20": 0.0, "2^2": 0.7514105903}).
- R114/SIGNED: the controller's displacement raised the step error in 50.5 % of controller-active steps (by scale {"1": 0.884765625, "2^-2": 0.49609375, "2^-20": 0.125, "2^2": 0.514453125}).
- R114/UNSIGNED: the controller's displacement raised the step error in 68.8 % of controller-active steps (by scale {"1": 0.9391601563, "2^-2": 0.9416666667, "2^-20": 0.5029622396, "2^2": 0.3662109375}).

