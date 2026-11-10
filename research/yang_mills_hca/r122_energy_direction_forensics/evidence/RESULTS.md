# R122 — energy-normalized fidelity, directional safety and recurrent error localization

**Mode:** `REGISTERED_PROSPECTIVE`  
**Disposition:** `MECHANISM_LOCALIZED_BOUNDED`  
**R93:** `ENERGY_NORMALIZED_DENOMINATOR_HOLD` (unchanged; a diagnostic cannot clear it)

Protocol `fe9ec864901d3e027798536576248ede9fe2bb8243fbee146d4edd5ba4bee31d`; source digest `4f3d407f4280641074264eca91cbbc6420b63e10f0d319a1eb651b1fa3db1f4f`; raw tensor `ac290017c4dd1fec53393be9bf74e0541cee931b7f5e256ec9a532f788184c9d`; source commit `e7e1db29e0f75eb5eda9b3ad6a0615e2e8cb663a`.

## Integrity gates

| Gate | Result |
|---|---|
| AUTONOMOUS_GATE | PASS |
| CHECKPOINTS_UNCHANGED | PASS |
| DETECTOR_CONTROLS | PASS |
| ENERGY_DECOMPOSITION_GATE | PASS |
| FROZEN_PARITY | PASS |
| GRAPH_NOVELTY | PASS |
| LINEAR_OPERATOR_CONTROLS | PASS |
| LOCAL_TRANSITION_GATE | PASS |
| NUMERICAL_PARITY | PASS |
| RESOURCE_BOUND | PASS |
| SEED_DISJOINTNESS | PASS |
| SOURCE_PINS | PASS |

- Fixtures 576; HCA passes 696,960; non-finite 0 = True.
- Composed teacher at k = 0 vs historical isolated teacher: max abs 0.0e+00 (0 bit mismatches in 3840 step checks).
- Independent torch-free teacher reference: max relative deviation 4.38e-16 (288 steps).
- EXACT_GUARDED vs EXACT_R118: 0 mismatching trajectories of 432; fresh-state replay 0/432; autonomous recheck after diagnostics 0/1152; DELAYED8 H8 identity 0/288.
- Decomposition identities: max residual 1.63e-15; exact rational audit 6.78e-16 over 3456 samples; telescoping residuals 5.17e-16 (one-step) and 4.82e-16 (restart).
- Gauge: joint covariance 1.9e-15; metric invariance 2.3e-13; state-only negative control ≥ 0.130; prediction-only half / quarter turns detected 3384 / 3384 of 3384.
- Consumed R121 regression fixture: 216 published cell values reproduced (worst relative deviation 1.5e-11); inversion detected, extinguished predictions not reported as reversals.
- Resources: 487.3 s, peak RSS 1430 MiB, raw tensor 390 MiB (volatile).
- READOUT autonomous samples by norm class: {"BOTH_ZERO": 20736, "ORDINARY": 808704}; non-ordinary horizons: {"1": {"BOTH_ZERO": 20736, "ILL_CONDITIONED": 0, "NEAR_ZERO": 0, "NONFINITE": 0, "ORDINARY": 20736, "ZERO_PREDICTION": 0, "ZERO_TARGET": 0}}.

## Registered mechanism decision (NATIVE, READOUT, H20, scale 1)

| Scope | DENOMINATOR_COLLAPSE | LOCAL_TRANSITION | ACCUMULATED_DRIFT | MIXED_LOCAL_ACCUMULATED | INVALID | dominant |
|---|---:|---:|---:|---:|---:|---|
| ALL | 0 | 4 | 27 | 5 | 0 | ACCUMULATED_DRIFT |
| SIGNED | 0 | 1 | 17 | 0 | 0 | ACCUMULATED_DRIFT |
| UNSIGNED | 0 | 3 | 10 | 5 | 0 | none |

Cohort dispositions: {"SIGNED": "LOCALIZED:ACCUMULATED_DRIFT", "UNSIGNED": "NOT_LOCALIZED"}.

Per graph class (strata = 3 overlaps × 2 cohorts):

| Graph | label counts |
|---|---|
| R122_T1_D2 | {"ACCUMULATED_DRIFT": 4, "LOCAL_TRANSITION": 1, "MIXED_LOCAL_ACCUMULATED": 1} |
| R122_T2_D2 | {"ACCUMULATED_DRIFT": 5, "LOCAL_TRANSITION": 1} |
| R122_T3_D3 | {"ACCUMULATED_DRIFT": 4, "LOCAL_TRANSITION": 2} |
| R122_T4_D3 | {"ACCUMULATED_DRIFT": 4, "MIXED_LOCAL_ACCUMULATED": 2} |
| R122_T5_D3 | {"ACCUMULATED_DRIFT": 5, "MIXED_LOCAL_ACCUMULATED": 1} |
| R122_T6_D3 | {"ACCUMULATED_DRIFT": 5, "MIXED_LOCAL_ACCUMULATED": 1} |

### Primary strata

| Stratum | mechanism | E_rel mean | r | c | λ local | local E_rel | skill | A_den | state E_rel | state c | E_perp share | growth H8→H20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R122_T1_D2/k0/SIGNED | LOCAL_TRANSITION | 0.0455 | 1.208 | 0.99961 | 0.992 | 0.0441 | -2.66 | 0.078 | 0.582 | 0.756 | 0.0258 | DENOMINATOR_DRIVEN |
| R122_T1_D2/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.0774 | 0.722 | 0.99982 | 0.363 | 0.0283 | -1.32 | 0.216 | 0.361 | 0.807 | 0.0024 | NUMERATOR_DRIVEN |
| R122_T1_D2/k2/SIGNED | ACCUMULATED_DRIFT | 19.7924 | 5.363 | 0.99959 | 0.004 | 0.0784 | -0.36 | 1.589 | 11.968 | 0.626 | 0.0013 | NUMERATOR_DRIVEN |
| R122_T1_D2/k2/UNSIGNED | ACCUMULATED_DRIFT | 1.9836 | 2.283 | 0.99977 | 0.041 | 0.0418 | 0.27 | 0.813 | 2.031 | 0.697 | 0.0012 | DENOMINATOR_DRIVEN |
| R122_T1_D2/k6/SIGNED | ACCUMULATED_DRIFT | 40.8243 | 7.242 | 0.99979 | 0.001 | 0.0506 | 0.16 | 1.378 | 28.320 | 0.727 | 0.0006 | DENOMINATOR_DRIVEN |
| R122_T1_D2/k6/UNSIGNED | ACCUMULATED_DRIFT | 4.3305 | 2.882 | 0.99993 | 0.015 | 0.0259 | 0.57 | 0.803 | 4.349 | 0.760 | 0.0003 | DENOMINATOR_DRIVEN |
| R122_T2_D2/k0/SIGNED | ACCUMULATED_DRIFT | 6.2793 | 3.490 | 0.99997 | 0.027 | 0.1705 | -1.22 | 1.465 | 4.260 | 0.400 | 0.0001 | NUMERATOR_DRIVEN |
| R122_T2_D2/k0/UNSIGNED | LOCAL_TRANSITION | 0.0096 | 0.902 | 0.99999 | 9.681 | 0.0916 | -0.19 | 0.017 | 0.604 | 0.636 | 0.0018 | DENOMINATOR_DRIVEN |
| R122_T2_D2/k2/SIGNED | ACCUMULATED_DRIFT | 87.3103 | 9.958 | 0.99934 | 0.002 | 0.1230 | -0.42 | 1.806 | 45.509 | 0.583 | 0.0017 | NUMERATOR_DRIVEN |
| R122_T2_D2/k2/UNSIGNED | ACCUMULATED_DRIFT | 4.3247 | 2.872 | 0.99992 | 0.025 | 0.0578 | 0.33 | 0.854 | 4.155 | 0.667 | 0.0005 | DENOMINATOR_DRIVEN |
| R122_T2_D2/k6/SIGNED | ACCUMULATED_DRIFT | 87.2584 | 10.126 | 0.99997 | 0.001 | 0.0688 | 0.09 | 1.535 | 53.125 | 0.785 | 0.0001 | DENOMINATOR_DRIVEN |
| R122_T2_D2/k6/UNSIGNED | ACCUMULATED_DRIFT | 6.2076 | 3.286 | 0.99999 | 0.012 | 0.0346 | 0.54 | 0.757 | 6.948 | 0.790 | 0.0000 | DENOMINATOR_DRIVEN |
| R122_T3_D3/k0/SIGNED | ACCUMULATED_DRIFT | 1.6168 | 2.262 | 0.99915 | 0.045 | 0.0714 | -0.65 | 0.592 | 2.705 | 0.514 | 0.0053 | NUMERATOR_DRIVEN |
| R122_T3_D3/k0/UNSIGNED | LOCAL_TRANSITION | 0.0011 | 1.000 | 0.99965 | 46.476 | 0.0425 | 0.02 | 0.001 | 0.617 | 0.642 | 0.8579 | NO_GROWTH |
| R122_T3_D3/k2/SIGNED | ACCUMULATED_DRIFT | 14.1828 | 4.775 | 0.99815 | 0.005 | 0.0781 | 0.30 | 1.196 | 11.932 | 0.434 | 0.0059 | NUMERATOR_DRIVEN |
| R122_T3_D3/k2/UNSIGNED | ACCUMULATED_DRIFT | 0.4725 | 1.668 | 0.99856 | 0.091 | 0.0392 | 0.64 | 0.390 | 1.198 | 0.549 | 0.0174 | DENOMINATOR_DRIVEN |
| R122_T3_D3/k6/SIGNED | ACCUMULATED_DRIFT | 6.5115 | 3.549 | 0.99968 | 0.006 | 0.0358 | 0.67 | 0.880 | 7.390 | 0.616 | 0.0013 | NUMERATOR_DRIVEN |
| R122_T3_D3/k6/UNSIGNED | LOCAL_TRANSITION | 0.0169 | 1.118 | 0.99967 | 1.304 | 0.0181 | 0.83 | 0.025 | 0.654 | 0.675 | 0.0517 | NO_GROWTH |
| R122_T4_D3/k0/SIGNED | ACCUMULATED_DRIFT | 3.5603 | 2.888 | 0.99900 | 0.020 | 0.0724 | -0.48 | 0.822 | 4.339 | 0.487 | 0.0047 | NUMERATOR_DRIVEN |
| R122_T4_D3/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.2769 | 1.526 | 0.99941 | 0.154 | 0.0425 | 0.14 | 0.267 | 1.043 | 0.588 | 0.0102 | NUMERATOR_DRIVEN |
| R122_T4_D3/k2/SIGNED | ACCUMULATED_DRIFT | 8.9474 | 3.994 | 0.99954 | 0.007 | 0.0591 | 0.31 | 1.022 | 8.745 | 0.499 | 0.0016 | NUMERATOR_DRIVEN |
| R122_T4_D3/k2/UNSIGNED | ACCUMULATED_DRIFT | 0.5673 | 1.748 | 0.99959 | 0.056 | 0.0318 | 0.63 | 0.443 | 1.308 | 0.595 | 0.0044 | NUMERATOR_DRIVEN |
| R122_T4_D3/k6/SIGNED | ACCUMULATED_DRIFT | 4.7480 | 3.181 | 0.99986 | 0.007 | 0.0343 | 0.59 | 0.832 | 5.633 | 0.627 | 0.0006 | NUMERATOR_DRIVEN |
| R122_T4_D3/k6/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.0564 | 1.232 | 0.99979 | 0.326 | 0.0183 | 0.79 | 0.082 | 0.688 | 0.696 | 0.0124 | DENOMINATOR_DRIVEN |
| R122_T5_D3/k0/SIGNED | ACCUMULATED_DRIFT | 2.3181 | 2.508 | 0.99921 | 0.027 | 0.0617 | -0.59 | 0.679 | 3.364 | 0.545 | 0.0043 | NUMERATOR_DRIVEN |
| R122_T5_D3/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.2081 | 1.448 | 0.99939 | 0.182 | 0.0370 | 0.05 | 0.209 | 0.982 | 0.623 | 0.0121 | NUMERATOR_DRIVEN |
| R122_T5_D3/k2/SIGNED | ACCUMULATED_DRIFT | 21.8354 | 5.668 | 0.99977 | 0.003 | 0.0699 | 0.35 | 1.162 | 18.718 | 0.445 | 0.0007 | NUMERATOR_DRIVEN |
| R122_T5_D3/k2/UNSIGNED | ACCUMULATED_DRIFT | 1.5947 | 2.233 | 0.99987 | 0.027 | 0.0367 | 0.66 | 0.719 | 2.131 | 0.540 | 0.0009 | NUMERATOR_DRIVEN |
| R122_T5_D3/k6/SIGNED | ACCUMULATED_DRIFT | 13.0164 | 4.601 | 0.99969 | 0.003 | 0.0387 | 0.68 | 0.863 | 15.078 | 0.560 | 0.0010 | NUMERATOR_DRIVEN |
| R122_T5_D3/k6/UNSIGNED | ACCUMULATED_DRIFT | 0.3772 | 1.581 | 0.99970 | 0.077 | 0.0199 | 0.84 | 0.283 | 1.256 | 0.628 | 0.0054 | DENOMINATOR_DRIVEN |
| R122_T6_D3/k0/SIGNED | ACCUMULATED_DRIFT | 2.3482 | 2.519 | 0.99882 | 0.032 | 0.0733 | -1.20 | 0.784 | 2.972 | 0.562 | 0.0064 | NUMERATOR_DRIVEN |
| R122_T6_D3/k0/UNSIGNED | MIXED_LOCAL_ACCUMULATED | 0.1573 | 1.391 | 0.99930 | 0.274 | 0.0426 | -0.28 | 0.180 | 0.865 | 0.645 | 0.0168 | NUMERATOR_DRIVEN |
| R122_T6_D3/k2/SIGNED | ACCUMULATED_DRIFT | 34.6749 | 6.819 | 0.99880 | 0.003 | 0.0858 | -0.37 | 1.565 | 21.685 | 0.569 | 0.0033 | DENOMINATOR_DRIVEN |
| R122_T6_D3/k2/UNSIGNED | ACCUMULATED_DRIFT | 3.4189 | 2.750 | 0.99942 | 0.021 | 0.0442 | 0.29 | 1.012 | 3.021 | 0.650 | 0.0030 | DENOMINATOR_DRIVEN |
| R122_T6_D3/k6/SIGNED | ACCUMULATED_DRIFT | 63.7715 | 8.796 | 0.99935 | 0.001 | 0.0572 | 0.13 | 1.408 | 43.392 | 0.716 | 0.0016 | DENOMINATOR_DRIVEN |
| R122_T6_D3/k6/UNSIGNED | ACCUMULATED_DRIFT | 6.0125 | 3.252 | 0.99973 | 0.010 | 0.0282 | 0.57 | 0.874 | 5.822 | 0.740 | 0.0010 | DENOMINATOR_DRIVEN |

Origin of the H20 READOUT error (median projection shares of the step-20 error by the block of steps whose local defects produced it; exact sum 1):

| Stratum | steps 1–4 | 5–8 | 9–12 | 13–16 | 17–20 | Σ|share| |
|---|---:|---:|---:|---:|---:|---:|
| R122_T1_D2/k0/SIGNED | 0.07 | 0.55 | 1.31 | 1.06 | -2.01 | 5.09 |
| R122_T1_D2/k0/UNSIGNED | -0.12 | -0.43 | -0.51 | 0.26 | 1.79 | 3.12 |
| R122_T1_D2/k2/SIGNED | 0.06 | 0.51 | 0.45 | 0.14 | -0.16 | 1.37 |
| R122_T1_D2/k2/UNSIGNED | -0.02 | 0.85 | 0.72 | 0.03 | -0.58 | 2.25 |
| R122_T1_D2/k6/SIGNED | 0.22 | 0.48 | 0.29 | 0.09 | -0.09 | 1.18 |
| R122_T1_D2/k6/UNSIGNED | 0.02 | 0.76 | 0.51 | 0.05 | -0.33 | 1.76 |
| R122_T2_D2/k0/SIGNED | 0.12 | 0.63 | 0.61 | 0.10 | -0.47 | 1.93 |
| R122_T2_D2/k0/UNSIGNED | -0.37 | -9.77 | -4.91 | 4.54 | 11.14 | 31.76 |
| R122_T2_D2/k2/SIGNED | 0.30 | 0.48 | 0.27 | 0.05 | -0.11 | 1.21 |
| R122_T2_D2/k2/UNSIGNED | 0.04 | 0.95 | 0.56 | -0.07 | -0.49 | 2.12 |
| R122_T2_D2/k6/SIGNED | 0.35 | 0.44 | 0.23 | 0.05 | -0.08 | 1.16 |
| R122_T2_D2/k6/UNSIGNED | 0.14 | 0.76 | 0.44 | -0.01 | -0.32 | 1.67 |
| R122_T3_D3/k0/SIGNED | 0.31 | 0.62 | 0.48 | 0.13 | -0.54 | 2.08 |
| R122_T3_D3/k0/UNSIGNED | 2.83 | 0.41 | -0.53 | -0.72 | -0.41 | 23.22 |
| R122_T3_D3/k2/SIGNED | 0.25 | 0.52 | 0.37 | 0.07 | -0.21 | 1.42 |
| R122_T3_D3/k2/UNSIGNED | 0.82 | 1.05 | 0.57 | -0.38 | -1.06 | 3.91 |
| R122_T3_D3/k6/SIGNED | 0.21 | 0.52 | 0.39 | 0.09 | -0.21 | 1.42 |
| R122_T3_D3/k6/UNSIGNED | 1.52 | 2.78 | 1.83 | -1.24 | -3.89 | 11.39 |
| R122_T4_D3/k0/SIGNED | 0.16 | 0.47 | 0.52 | 0.19 | -0.35 | 1.70 |
| R122_T4_D3/k0/UNSIGNED | 0.77 | 1.15 | 0.71 | -0.33 | -1.30 | 4.23 |
| R122_T4_D3/k2/SIGNED | 0.16 | 0.47 | 0.44 | 0.13 | -0.21 | 1.42 |
| R122_T4_D3/k2/UNSIGNED | 0.50 | 0.84 | 0.62 | -0.15 | -0.80 | 2.93 |
| R122_T4_D3/k6/SIGNED | 0.13 | 0.48 | 0.46 | 0.15 | -0.22 | 1.44 |
| R122_T4_D3/k6/UNSIGNED | 0.61 | 1.56 | 1.23 | -0.43 | -1.95 | 5.80 |
| R122_T5_D3/k0/SIGNED | 0.15 | 0.46 | 0.53 | 0.24 | -0.39 | 1.77 |
| R122_T5_D3/k0/UNSIGNED | 0.82 | 1.12 | 0.75 | -0.31 | -1.39 | 4.39 |
| R122_T5_D3/k2/SIGNED | 0.27 | 0.46 | 0.35 | 0.08 | -0.16 | 1.31 |
| R122_T5_D3/k2/UNSIGNED | 0.49 | 0.70 | 0.48 | -0.12 | -0.55 | 2.34 |
| R122_T5_D3/k6/SIGNED | 0.26 | 0.45 | 0.35 | 0.09 | -0.15 | 1.30 |
| R122_T5_D3/k6/UNSIGNED | 0.48 | 0.92 | 0.70 | -0.20 | -0.91 | 3.21 |
| R122_T6_D3/k0/SIGNED | 0.18 | 0.44 | 0.57 | 0.24 | -0.42 | 1.85 |
| R122_T6_D3/k0/UNSIGNED | 0.87 | 1.23 | 0.94 | -0.34 | -1.68 | 5.06 |
| R122_T6_D3/k2/SIGNED | 0.23 | 0.48 | 0.34 | 0.09 | -0.13 | 1.26 |
| R122_T6_D3/k2/UNSIGNED | 0.20 | 0.71 | 0.52 | -0.00 | -0.43 | 1.87 |
| R122_T6_D3/k6/SIGNED | 0.36 | 0.44 | 0.23 | 0.05 | -0.08 | 1.16 |
| R122_T6_D3/k6/UNSIGNED | 0.27 | 0.65 | 0.37 | -0.00 | -0.29 | 1.59 |

### By amplitude scale and by horizon (label counts over all strata)

| Scale | DENOM | LOCAL | ACCUM | MIXED | INVALID | dominant | label agreement with scale 1 |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 0 | 4 | 27 | 5 | 0 | ACCUMULATED_DRIFT | — |
| 2^-2 | 13 | 1 | 22 | 0 | 0 | none | 0.44 |
| 2^-20 | 12 | 0 | 24 | 0 | 0 | none | 0.61 |
| 2^2 | 0 | 2 | 16 | 18 | 0 | none | 0.53 |

| Horizon (scale 1) | DENOM | LOCAL | ACCUM | MIXED | INVALID | SIGNED dominant | UNSIGNED dominant |
|---|---:|---:|---:|---:|---:|---|---|
| H1 | 13 | 5 | 0 | 0 | 18 | none | none |
| H12 | 0 | 11 | 15 | 10 | 0 | none | none |
| H16 | 0 | 7 | 18 | 11 | 0 | none | none |
| H20 | 0 | 4 | 27 | 5 | 0 | ACCUMULATED_DRIFT | none |
| H4 | 18 | 6 | 7 | 5 | 0 | none | none |
| H8 | 11 | 20 | 1 | 4 | 0 | none | none |

## Normalized-error growth H8 → H20 (R93 pattern = absolute defect falls while normalized defect rises)

| Policy | Scale | Cohort | NO_GROWTH | NUMERATOR_DRIVEN | DENOMINATOR_DRIVEN | dominant |
|---|---|---|---:|---:|---:|---|
| DELAYED8 | 1 | SIGNED | 1 | 10 | 7 | none |
| DELAYED8 | 1 | UNSIGNED | 2 | 6 | 10 | none |
| NATIVE | 1 | SIGNED | 0 | 13 | 5 | none |
| NATIVE | 1 | UNSIGNED | 2 | 6 | 10 | none |
| NATIVE | 2^-2 | SIGNED | 0 | 18 | 0 | NUMERATOR_DRIVEN |
| NATIVE | 2^-2 | UNSIGNED | 0 | 4 | 14 | none |
| NATIVE | 2^-20 | SIGNED | 0 | 12 | 6 | none |
| NATIVE | 2^-20 | UNSIGNED | 0 | 9 | 9 | none |
| NATIVE | 2^2 | SIGNED | 17 | 0 | 1 | NO_GROWTH |
| NATIVE | 2^2 | UNSIGNED | 3 | 10 | 5 | none |
| R114 | 1 | SIGNED | 9 | 3 | 6 | none |
| R114 | 1 | UNSIGNED | 6 | 3 | 9 | none |
| SIGNED_ONLY | 1 | SIGNED | 9 | 3 | 6 | none |
| SIGNED_ONLY | 1 | UNSIGNED | 2 | 6 | 10 | none |

## Secondary findings (fixed criteria)

- **S1_readout_error_collinear_with_teacher**: SUPPORTED (strata fraction 0.944; median E_perp/E_rel <= 0.05 in >= 0.75 of primary strata).
- **S2_paired_state_direction_not_preserved**: SUPPORTED (strata fraction 1.000; median PAIRED_STATE cosine < 0.9 in >= 0.75 of primary strata).
- **S3_local_step_beats_persistence**: MIXED_LOCAL_SKILL (strata fraction 0.667; median teacher-forced one-step skill vs teacher persistence > 0).
- **S4 decayed-regime operator**: PASS — kernel probe is exactly linear (homogeneity 0.0e+00); the eight-gate analytic operator equals it to 1.4e-16; linear-limit trajectories follow operator powers to 5.1e-05.

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
| SIGNED | 36 | 1.0911 (1.0773–1.1002) | 1.000 | -5.581 | 0.000 |
| UNSIGNED | 72 | 1.0289 (0.9803–1.0349) | 0.736 | -2.931 | 0.000 |
| UNSIGNED_R104 | 72 | 0.9475 (0.8999–0.9858) | 0.000 | -0.878 | 0.222 |

## Directional safety (READOUT, all twenty steps)

| Policy | in-range verdict | induced reversals in-range | above-clamp verdict | induced above-clamp | reversed samples by scale | induced by cohort |
|---|---|---:|---|---:|---|---|
| DELAYED8 | NO_INDUCED_REVERSAL_OBSERVED | 0 | NO_INDUCED_REVERSAL_OBSERVED | 0 | {"1": 6368, "2^-2": 236, "2^-20": 0, "2^2": 14312} | {"SIGNED": 0, "UNSIGNED": 0} |
| NATIVE | REFERENCE | 0 | REFERENCE | 0 | {"1": 6368, "2^-2": 235, "2^-20": 0, "2^2": 14726} | {"SIGNED": 0, "UNSIGNED": 0} |
| R114 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 1760 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 19978 | {"1": 6642, "2^-2": 1387, "2^-20": 12, "2^2": 30283} | {"SIGNED": 8112, "UNSIGNED": 13626} |
| SIGNED_ONLY | HAZARD_CONTROLLER_INDUCED_REVERSAL | 437 | HAZARD_CONTROLLER_INDUCED_REVERSAL | 7675 | {"1": 6713, "2^-2": 236, "2^-20": 0, "2^2": 17654} | {"SIGNED": 8112, "UNSIGNED": 0} |

- DELAYED8: reversed by step [0, 4557, 3424, 3386, 2264, 1397, 812, 641, 599, 512, 480, 339, 335, 297, 287, 284, 277, 270, 263, 492]; induced by step [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; induced by graph {"R122_T1_D2": 0, "R122_T2_D2": 0, "R122_T3_D3": 0, "R122_T4_D3": 0, "R122_T5_D3": 0, "R122_T6_D3": 0}; first-reversal step histogram {"2": 145, "20": 10, "3": 20, "4": 14, "5": 13, "6": 3, "7": 1}.
- NATIVE: reversed by step [0, 4557, 3424, 3386, 2264, 1397, 812, 641, 640, 545, 512, 395, 363, 325, 317, 320, 320, 298, 291, 522]; induced by step [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; induced by graph {"R122_T1_D2": 0, "R122_T2_D2": 0, "R122_T3_D3": 0, "R122_T4_D3": 0, "R122_T5_D3": 0, "R122_T6_D3": 0}; first-reversal step histogram {"2": 145, "20": 10, "3": 20, "4": 14, "5": 13, "6": 3, "7": 1}.
- R114: reversed by step [0, 4416, 3566, 3487, 2777, 1782, 1783, 1574, 1681, 1556, 1701, 1431, 1579, 1397, 1560, 1418, 1663, 1451, 1703, 1799]; induced by step [0, 403, 344, 418, 794, 742, 1128, 1215, 1286, 1217, 1370, 1278, 1425, 1310, 1438, 1322, 1535, 1355, 1591, 1567]; induced by graph {"R122_T1_D2": 1949, "R122_T2_D2": 2619, "R122_T3_D3": 5196, "R122_T4_D3": 4917, "R122_T5_D3": 4712, "R122_T6_D3": 2345}; first-reversal step histogram {"13": 2, "16": 2, "17": 2, "18": 2, "19": 5, "2": 138, "20": 16, "3": 31, "4": 12, "5": 17, "6": 7, "7": 2}.
- SIGNED_ONLY: reversed by step [0, 4369, 3181, 2870, 2029, 1108, 977, 806, 963, 796, 938, 665, 813, 629, 792, 638, 798, 621, 768, 842]; induced by step [0, 384, 25, 32, 49, 29, 356, 383, 518, 449, 602, 510, 656, 542, 670, 542, 670, 525, 660, 510]; induced by graph {"R122_T1_D2": 931, "R122_T2_D2": 1106, "R122_T3_D3": 1638, "R122_T4_D3": 1575, "R122_T5_D3": 1660, "R122_T6_D3": 1202}; first-reversal step histogram {"2": 138, "20": 10, "3": 31, "4": 13, "5": 17, "6": 5, "7": 2}.

## Policy comparison at scale 1 (pooled READOUT NSE, equal-weight graph × checkpoint cells)

| Overlap | Horizon | NATIVE | R114 | SIGNED_ONLY | DELAYED8 |
|---|---|---:|---:|---:|---:|
| k0 | H8 | 0.027760 | 0.074895 | 0.035000 | 0.027760 |
| k0 | H20 | 0.979381 | 0.390868 | 0.158586 | 0.445128 |
| k2 | H8 | 1.179721 | 0.952694 | 0.975840 | 1.179721 |
| k2 | H20 | 11.748154 | 0.646631 | 1.787236 | 3.724460 |
| k6 | H8 | 2.119300 | 1.282466 | 1.507307 | 2.119300 |
| k6 | H20 | 13.896243 | 0.484283 | 2.082848 | 3.537268 |

| Overlap / policy | mean ΔE_rel vs NATIVE (H20) | graph blocks improved | graph-cluster bootstrap 95 % | worst graph block |
|---|---:|---:|---|---|
| k0/DELAYED8 | -0.5343 | 6/6 | [-0.8262, -0.2656] | -0.0134 (R122_T1_D2) |
| k0/R114 | -0.5885 | 5/6 | [-1.0062, -0.1542] | 0.3181 (R122_T1_D2) |
| k0/SIGNED_ONLY | -0.8208 | 5/6 | [-1.3077, -0.3855] | 0.0211 (R122_T1_D2) |
| k2/DELAYED8 | -8.0237 | 6/6 | [-14.0840, -3.9528] | -2.3454 (R122_T4_D3) |
| k2/R114 | -11.1015 | 6/6 | [-19.2102, -5.3472] | -2.9933 (R122_T4_D3) |
| k2/SIGNED_ONLY | -9.9609 | 6/6 | [-17.3175, -4.9078] | -2.9344 (R122_T4_D3) |
| k6/DELAYED8 | -10.3590 | 6/6 | [-17.8508, -3.9943] | -1.5028 (R122_T4_D3) |
| k6/R114 | -13.4120 | 6/6 | [-23.5248, -4.6910] | -1.1066 (R122_T4_D3) |
| k6/SIGNED_ONLY | -11.8134 | 6/6 | [-20.3885, -4.5140] | -1.5442 (R122_T4_D3) |

Worst graph-block harm over k: {"DELAYED8": -0.01340437833, "R114": 0.3181457792, "SIGNED_ONLY": 0.02109251696}.

Controller-specific contributions:
- Teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in 91.6 % of SIGNED samples (by scale {"1": 0.9180021368, "2^-2": 0.8997061966, "2^-20": 0.9231214387, "2^2": 0.9238782051}).
- Teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in 89.3 % of UNSIGNED samples (by scale {"1": 0.8733751781, "2^-2": 0.8888888889, "2^-20": 0.9097222222, "2^2": 0.8997284544}).
- DELAYED8/SIGNED: the controller's displacement raised the step error in 16.5 % of controller-active steps (by scale {"1": 0.1171875, "2^-2": 0.01164641204, "2^-20": 0.03703703704, "2^2": 0.4955873843}).
- R114/SIGNED: the controller's displacement raised the step error in 40.4 % of controller-active steps (by scale {"1": 0.6152065527, "2^-2": 0.374599359, "2^-20": 0.1956908832, "2^2": 0.4285968661}).
- R114/UNSIGNED: the controller's displacement raised the step error in 68.7 % of controller-active steps (by scale {"1": 0.8315527066, "2^-2": 0.9467815171, "2^-20": 0.5254852208, "2^2": 0.4449563746}).

