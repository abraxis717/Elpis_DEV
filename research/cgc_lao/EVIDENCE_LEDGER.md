# CGC–LAO evidence ledger through Q4F3 (2026-11-07)

**Evidence class:** operator-reported results; Q4E5 import source is locally hash-verified and mirrored exactly here; R97W archival evidence and portable reproduction are in [abraxis717/CGC](https://github.com/abraxis717/CGC). The later Q4E/Q4F full raw run archives and model checkpoints are **not** included here. All Q4E14 holdout cohorts remain unopened. Do not treat an experiment's status line alone as an independently verified reproduction.

Conventions: regret `Δ = regret(A) - regret(B)`, so a **negative** value favors A. CIs are reported router-seed-level approximate 95% paired intervals, as supplied in each run summary. `KNOWNQ` knows the synthetic corruption rate and is **privileged**, inadmissible at deployment.

## Grounded historical anchors

| Experiment | Result | Scientific boundary |
|---|---|---|
| R97W | Three-output 120-parameter CGC beat 120-parameter GELU FFN in 48/48 paired held-out adaptation comparisons (4 groups × 12); original archived result and portable reproduction in CGC repo. | First-order Reptile with episode-local fast **weight** adaptation; **not** frozen-weight LAO or universal FFN superiority. |
| Q4E5 | Frozen `PROBE_EARLY` vs GREEDY, 16 router seeds: terminal Δ = **−0.00015245**, 95% CI [−0.00022633, −0.00007857], 14/16 wins; online mean 0.0566153, CI upper 0.0579178 ≤0.075. | Synthetic fixed T=8 acquisition only; no robust FLIP40 claim. The public reference here has no trained checkpoint. |
| Q4D12–Q4D16 | Adaptive HALT/continue policies not qualified against fixed horizon; observable conditional continuation signal weak, privileged latent signal stronger. | No learned dynamic STOP/HALT admission. |
| Q4E6–Q4E8 | Reliability gating/Bayesian posterior could improve FLIP40 but regressed CLEAN and/or ALL; Q4E8 observable FLIP40 AUC at T8 ≈0.819. | More reliable late classification did not produce admissible early decisions. |
| Q4E9–Q4E12 | No validation-eligible augmentation, altered-history, Reptile-for-frozen-operator or zero-parameter temporary-state arm; holdouts unopened. | Q4E12's per-arm validation deltas were not recovered: do **not** invent specific gate failures. |
| Q4E13 | BCE matched pinned CGC bit-for-bit across five seeds. Objective-aligned `RELIABLE` and `HYBRID` improved FLIP40 by approximately −0.00245 and −0.00179 respectively, but heavily regressed CLEAN; `TERM` worsened clean and corrupted performance. | Surrogate improvement did not escape clean/corruption frontier; no holdout. |
| Q4E14 | `TRUST_PREQUENTIAL`: FLIP40 vs CGC −0.00028738, CI [−0.00040076, −0.00017400]; CLEAN +0.00007210, CI [+0.00000127,+0.00014292]. `TRUST_DELAYED` also validation-eligible. | **Validation only**, not holdout/production. Prequential remains worse than NOOP on FLIP40 by +0.00256285 (CI upper +0.00311786). |

Q4E14's update-benefit information test: causal AUC 0.5883; history-shuffled AUC 0.5837; label-shuffled 0.4893; privileged oracle AUC 1.0 and FLIP40 Δ vs CGC −0.00985963. The 0.0046 AUC difference between causal and history-shuffled does **not** support an actionable residual-history signal. The best admissible trust policy reduced average proposal magnitude by ~4.8%, suggesting attenuation rather than better targeting.

## Q4F series: fixed histories and analytical observation limits

Q4F0 tested a 918-float diagnostic posterior. Under FLIP40, unknown-q vs CGC Δ **+0.00003759** CI [−0.00060223,+0.00067741], while known-q vs CGC Δ **−0.00245984** CI [−0.00342724,−0.00149243]. **Q4F0 conflated posterior information and terminal readout**; Q4F1 corrected this.

Q4F1 held CGC-selected observations fixed and crossed unknown/known q with plug-in/predictive readout. Under FLIP40, `UNK_PRED` vs `KNOWN_PRED` Δ **+0.00349946**, CI [+0.00275041,+0.00424851]. Under CLEAN the same comparison was **−0.00107868**, CI [−0.00131310,−0.00084427]. Thus the privileged q information has substantial regime-dependent **decision** value; the readout alone is not the principal FLIP40 explanation. At T8 corruption-classification AUC was 0.82364.

Q4F2 ablated posterior prior support on matched selected feedback. Under FLIP40, the generator-coupled five-hypothesis prior (`COUPLED5`) vs original 12-hypothesis reference (`REF12`) Δ **+0.00035279**, CI [+0.00014588,+0.00055970]; vs `KNOWNQ`, Δ **+0.00373559**, CI [+0.00266972,+0.00480145]. Under out-of-support FLIP20, `COUPLED5` vs `REF12` Δ **+0.00033165**, CI [+0.00014282,+0.00052047]. Correcting hypothesis support and frequency failed to close the known-q advantage. The coupled prior encodes generator structure; it is a research reference, not a generalizable inference admission.

### Q4F3: does additional selected feedback close the gap?

Full pinned diagnostic (4 router seeds, 5 frozen train checkpoints, 3 scopes, 2 reps × 96 episodes; CLEAN/SHIFT/FLIP40/FLIP20); CGC `PROBE_EARLY` first-eight prefix followed by static cyclic sampling. Values below are **FLIP40** and use matched selected-feedback logs for the posterior variants.

| Observations | q-classification AUC | UNKNOWN vs KNOWNQ Δ regret [95% CI] | UNKNOWN vs NOOP Δ regret [95% CI] |
|---:|---:|---|---|
| 8 | 0.80605 | +0.00262799 [+0.00141673,+0.00383925] | +0.00268681 [+0.00107226,+0.00430135] |
| 16 | 0.87306 | +0.00288056 [+0.00188872,+0.00387239] | +0.00316741 [+0.00213583,+0.00419900] |
| 32 | 0.93967 | +0.00271908 [+0.00132675,+0.00411140] | +0.00297450 [+0.00119569,+0.00475332] |
| 64 | 0.98095 | +0.00175210 [+0.00030155,+0.00320265] | +0.00211959 [+0.00019768,+0.00404150] |

**Important:** improved AUC does not eliminate the regret gap at H64. The 64-step result is an exploratory **different observation budget**, not Q4E5 or Q4E14 qualification. BALANCED_PREFIX acquisition did not establish significant FLIP40 regret improvement over CGC_PROBE at any horizon in the operator summary (e.g., H64 Δ +0.00027793 CI [−0.00049255,+0.00104841]).

Q4F3 integrity reported ten checkpoint pins passed, read-only frozen model, sealed Q4E14 holdout, no new weights and no repository mutation. Live historical seed-registry scan was **not attested** (`SEED_REGISTRY_LIVE=NOT_ATTESTED`). The reported OS-write attestation is **NONE**; a tmpfs workspace does not prove zero writes by the whole OS.

## Conclusions and explicit nonclaims

- **Supported:** Fixed-history corruption knowledge matters; simple prior correction does not close the gap; greater observational classification AUC need not improve terminal decisions enough. A tiny trust controller can barely pass validation while remaining worse than NOOP under 40% corruption.
- **Not established:** Fundamental impossibility with eight observations; a universally optimal analytical posterior; an admissible FLIP40 solution; independent external reproduction of Q4E/Q4F; production or on-policy RALPH performance; permanent self-learning or model-weight mutation.
- **Publication boundary:** Do not open Q4E14 reserved holdout cohorts for this handoff; do not retroactively relabel diagnostic studies as qualifying experiments.

## R24 benchmark validity: statistical-provenance caveat (2026-11-12)

**Historical R24 disposition remains recorded** as `BENCHMARK_QUALIFIED_CAPABILITY_PARTIAL`; its original archive and source digests are not rewritten. **Current reliance on its positive benchmark-capability claims is `HOLD_STATISTICAL_AND_TASK_PROVENANCE`**. The historical 20,000-iteration LCG sampler generated only 64 distinct 16-seed sign masks out of 65,536 possible; exact two-sided sign-flip enumeration and fixed Holm24 accounting invalidate the historical reported tail calibration. In the retrospective R25A-R7 analysis, no positive GCC E1/E2 endpoint survives Holm24; Clang favorable survivors use task generators with unqualified narrated semantics. E1 and E2 are deterministically redundant per seed. Native ABI and all generator contracts remain pending. This is a corrective annotation, **not** a promotion and **not** a revision to the historical recorded disposition.

See [dated R24 statistical caveat](stat_provenance/R24_STATISTICAL_PROVENANCE_ADDENDUM.md), [48 typed R7 receipts](stat_provenance/R25A_R7_STATISTICAL_RECEIPTS.json), and [frozen future R8 enumeration method](stat_provenance/R25A_R8_ENUMERATION_REGISTRATION.json). R7 corrected numbers were obtained retrospectively and are **not retroactively preregistered**. No R8 tests or new holdout are run by this amendment.
