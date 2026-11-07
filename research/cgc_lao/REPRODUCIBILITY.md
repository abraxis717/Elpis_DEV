# Reproduction contract and proposed first collaborator experiments

## What is available in public source

1. [CGC's public E0 implementation](https://github.com/abraxis717/CGC/blob/main/src/cgc/ecsg_e0.py), [native C11 reference](https://github.com/abraxis717/CGC/tree/main/native/ECS_G), [R97W reproducible meta-learning benchmark](https://github.com/abraxis717/CGC/tree/main/research/benchmarks/r97w/portable). Historical experiment used **fast weight adaptation**.
2. `reference/src/cgc/lao_q4e5.py` (this directory): verified source with Q4E5 acquisition semantics and **frozen** 120-weight operator interface. `reference/tests/test_lao_q4e5.py` exercises parity/causal boundaries. No pretrained Q4E5 checkpoint distributed.
3. Q4E5 qualification source and later Q4E/Q4F diagnostic runner archives are **not public in this bounded push**. Reported summary metrics are not a substitute for raw RNG logs, exact simulator source, a complete environment lock, original checkpoint tensors and independent reexecution. Request a separately audited, non-sensitive reproducibility release before making cross-machine numerical parity claims.

## Recommended independent first study: adequacy of causal reliability belief

**Question:** Can a compact, causal corruption-uncertainty representation improve *actual discrete terminal regret* against pinned CGC/NOOP without known q, hidden labels or additional feedback? Prior art to examine includes contaminated Bernoulli models, finite-state Bayesian filtering, generalized Bayes/tempered likelihood, confidence sequences for adaptively sampled data, and robust contextual bandit estimation. The selected-feedback model and 8-step budget are central; do not naively translate symmetric-clipping guarantees from real-valued iid noise to corrupted Bernoulli rewards.

**Work sequence:**

- Reconstruct the exact three-expert simulator and frozen CGC baseline, publish source hash, weights hash, stopping/STOP utility, and q/regime data-generation assumptions. If unavailable, mark all new results as *independent illustrative benchmark*, not comparisons to historical Q4E5.
- Start with **no new neural training**: a bounded `Beta(α,β)` selected-reward baseline; a small finite q-mixture with tempered evidence; a no-learning/no-update arm; and a privileged true-q oracle for gap attribution only.
- Keep a common **fixed action log** first to test belief updates and terminal decision rules separately; then run true policy-controlled feedback with correctly indexed counterfactual potential outcomes. A fixed log cannot by itself qualify a newly adaptive selector.
- Predeclare clean, corrupted and out-of-support conditions (including FLIP20/FLIP40), observed and unobserved expert strata, query/STOP costs, unit of pairing and resource caps.
- Assess observability (AUC/Brier/calibration) **separately** from terminal/online regret. Add stronger controls: label-shuffled reward associations, history-shuffled trust stats, static update attenuation, wrong-corruption-prior and privilege-oracle. Verify output invariant to poisoned hidden variables.
- Compare all candidates against original pinned CGC, FFN, SGD, NOOP and matched static/analytical controls. Report effect sizes and paired seed confidence intervals; specify how seed uncertainty, multiplicity and candidate selection are handled.

## Historical Q4E candidate gates (do not retroactively change)

Validation admission relative to pinned CGC used: FLIP40 mean terminal-regret delta strictly `<0`; CLEAN mean delta `<=+0.00025`; ALL mean delta `<=+0.00010`; online regret upper approximate 95% CI `<=0.075`. Historical holdout required FLIP40 paired CI upper `<0`, CLEAN paired CI upper `<=+0.00025`, ALL paired CI upper `<=+0.00010`, online 95% CI upper `<=0.075`, and favorable predefined cohorts. A separate stronger robustness diagnostic checked FLIP40 relative to **NOOP**; failure there must be disclosed even when relative-to-CGC validation passes.

Do not reuse Q4E14's sealed validation/holdout pools, inspect reserved seeds, move the threshold after viewing results, or use test-time latent q. Q4F0–Q4F3 are **diagnostic only**, not validation candidates. For brand-new benchmarks, declare fresh pre-registered thresholds appropriate to their scale **before** testing rather than pretending historic absolute regret tolerances universally transfer.

## Resource and trust contract

- Frozen external model coefficients at inference; no inference gradients, model-weight, H-gram, NAND/NVMe or persistent runtime-state mutation.
- Volatile bounded state, per-episode reset, saturating counters; define worst-case scalars, runtime, CPU/RAM and GPU memory **before** evaluation. Avoid dynamic allocation or unbounded state growth.
- Explicit host-controlled admission; causal selected-only feedback; no latent bias/true corruption/terminal oracle at inference.
- Research outputs and tests must not enter canonical `src/elpis`, native/runtime, system authority, production ABI or CI science status without separate review.
- Use a distinct, fail-closed research branch and PR. Preserve original evidence and source hashes, assert all existing history pins, keep test stdout visible, and avoid any promotion if qualified measures regress.

## Suggested preregistration fields

`HYPOTHESIS`, `SOURCE_SHA`, `CHECKPOINT_SHA`, `DATA_GENERATOR_SHA`, `OBSERVATION_MODEL`, `REGIME_SUPPORT`, `INFERENCE_INPUTS`, `PARAM_COUNT`, `STATE_SCALARS_MAX`, `UPDATE_CAP`, `HORIZON`, `POTENTIAL_OUTCOME_INDEX`, `TRAIN_SEEDS`, `DEV_SEEDS`, `VAL_SEEDS`, `RESERVED_HOLDOUT_SEEDS`, `CONTROLS`, `PRIMARY_REGRET`, `SECONDARY_REGRET`, `PAIRING_UNIT`, `CI_METHOD`, `MULTIPLICITY_POLICY`, `PASS_GATES`, `NOOP_GATE`, `INTEGRITY_TESTS`, `OUTPUT_HASH`, `EXIT_STATUS`.

**Scientifically useful exit:** If even an admissible analytical estimator remains much worse than a known-q reference with adequate feedback, report a bounded information or model-misspecification gap and pivot toward better evidence acquisition, not a larger hidden-state optimizer. A negative result is acceptable; relaxing gates or opening old holdouts is not.
