# R123 prospective K4 causal-shadow results

Preregistered before execution. Six frozen checkpoints, four new incidence classes, fixed fractions 0.25 and 0.50.
Teacher-derived bias target used only as an offline intervention; neither weights nor production runtime are modified.

**Disposition:** PASS_BOUNDED
**R93:** ENERGY_NORMALIZED_DENOMINATOR_HOLD

| Arm | Cohort | k | H20 relative improvement | Graph wins | Worst graph harm | H8 ratio | Teacher-forced H20 ratio | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0.25 | SIGNED | 0 | 24.6909% | 4/4 | -0.19539 | 0.85464 | 1.03877 | True |
| 0.25 | SIGNED | 2 | 32.0056% | 4/4 | -1.94432 | 0.78572 | 1.04097 | True |
| 0.25 | SIGNED | 6 | 34.2634% | 4/4 | -2.75320 | 0.79707 | 1.03497 | True |
| 0.25 | UNSIGNED | 0 | 23.7263% | 4/4 | -0.03474 | 0.98203 | 1.02302 | True |
| 0.25 | UNSIGNED | 2 | 46.5844% | 4/4 | -0.22009 | 0.85035 | 1.01139 | True |
| 0.25 | UNSIGNED | 6 | 50.4702% | 4/4 | -0.27936 | 0.88630 | 1.00257 | True |
| 0.5 | SIGNED | 0 | 42.6348% | 4/4 | -0.32309 | 0.78749 | 1.08061 | True |
| 0.5 | SIGNED | 2 | 53.4687% | 4/4 | -3.28622 | 0.64166 | 1.08876 | True |
| 0.5 | SIGNED | 6 | 56.1514% | 4/4 | -4.45804 | 0.66378 | 1.07731 | True |
| 0.5 | UNSIGNED | 0 | 26.5974% | 3/4 | +0.00161 | 1.10410 | 1.05361 | False |
| 0.5 | UNSIGNED | 2 | 66.0166% | 4/4 | -0.29584 | 0.85455 | 1.04000 | True |
| 0.5 | UNSIGNED | 6 | 73.5642% | 4/4 | -0.37181 | 0.92938 | 1.02386 | True |

Directional hazards (20 steps, all tested inputs):

- 1/0.0: 0
- 1/0.25: 0
- 1/0.5: 0
- 2^-2/0.0: 0
- 2^-2/0.25: 0
- 2^-2/0.5: 0
- 2^-20/0.0: 0
- 2^-20/0.25: 0
- 2^-20/0.5: 0
- 2^2/0.0: 0
- 2^2/0.25: 40
- 2^2/0.5: 113

Full per-fixture data, provenance and numerical integrity are in `R123_RESULT.json` and `INDEX.json`.
