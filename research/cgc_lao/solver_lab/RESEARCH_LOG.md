# Research log (compact)

Format: `CANDIDATE → MECHANISM → MEASURED RESULT → DIAGNOSIS → NEXT CHANGE`. Full numbers live in the
evidence records named in each entry. Δ = arm − reference (negative favours the arm), terminal regret in
probability units, router-seed paired 95% intervals.

## Phase 0: instrument and TRAIN calibration

- Simulator, counter RNG, numpy CGC port (parity with the checked-in torch reference: identical actions,
  state within 2e-6), baselines and 44 mechanical tests built first.
- TRAIN tuning: SGD η = 0.5, BETA κ = 8 (interior optima, ALL mixture). CGC_REFERENCE trained by ES on
  terminal BCE (400 generations): BCE CLEAN 0.579 vs 0.598 for the zero controller, FLIP40 0.601 vs 0.599.
- TRAIN levels (32 seeds × 2048 episodes; [`evidence/train/levels.json`](evidence/train/levels.json)):

| Arm | CLEAN | FLIP20 | FLIP40 | ALL | online |
|---|---:|---:|---:|---:|---:|
| NOOP | 0.06305 | 0.06347 | 0.06306 | 0.06319 | 0.0934 |
| SGD | 0.04422 | 0.05596 | 0.06849 | 0.05622 | 0.0905 |
| BETA | 0.04470 | 0.05667 | 0.06895 | 0.05677 | 0.0912 |
| CGC_REFERENCE | 0.04705 | 0.05558 | 0.06462 | 0.05575 | 0.0902 |
| STATIC_ATTENUATION | 0.05305 | 0.05786 | 0.06232 | 0.05774 | 0.0911 |
| CLEAN_BAYES | 0.04223 | 0.05853 | 0.07624 | 0.05900 | 0.0922 |
| KNOWN_Q_ORACLE (privileged) | 0.04223 | 0.05506 | 0.06196 | 0.05308 | 0.0891 |

Every admissible learner that learns well on CLEAN is worse than NOOP on FLIP40; the only one below
NOOP on FLIP40 (STATIC_ATTENUATION) keeps 48% of the clean gain. The privileged known-q learner is
below NOOP on FLIP40 (−0.0011) with the full clean gain: corruption knowledge, not capacity, separates
them. Gate scales were frozen from these levels in [PREREGISTRATION.md](PREREGISTRATION.md).
