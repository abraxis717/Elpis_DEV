# R24 statistical provenance — append-only qualification caveat

**Date:** 2026-11-12
**Status:** `STATISTICAL_PROVENANCE_CHALLENGED_CURRENT_RELIANCE_HOLD`
**Historical R24 recorded disposition:** `BENCHMARK_QUALIFIED_CAPABILITY_PARTIAL` — **unchanged as a historical record**.
**Current interpretation:** the historical score-derived positive capability claim is **not qualified for reliance, promotion, or deployment** until the sampling/statistical and task/native contracts are repaired and independently requalified. This annotation supplements rather than edits R24 evidence.

## Authentication boundaries

- Historical R24 ZIP SHA-256: `cd605851e954ce0e07a1ef12854749918dc227ec98d7e7830967d8e473786169` (the R24 ZIP itself is not repackaged or modified).
- R25A-R7 original evidence capsule SHA-256 (identified inside the supplied reanalysis): `3cb892ad833e3d47539d69b3a5929c9da39e02f1dc783e23f7e537da77f671b8`. The capsule was not uploaded for independent trace reexecution in this review.
- R25A-R7 independent reanalysis supplied for this annotation: SHA-256 `9b6c287637405ed9de78900995120be5942117b139595b0423624af6f1588402`; its internal `MANIFEST.json` and ZIP CRC were verified. The report's claim of prior 16/16 capsule-payload verification is **reported evidence**, not a new independent audit of the absent original capsule.
- This is a **retrospective** statistical defect notice, not prospective qualification. Neither native ABI legality nor generator semantic fidelity was independently executed here.

## Statistically decisive defect

The historical sampler uses `rng = (1103515245*rng + 12345) & 0xffffffff` and extracts `(rng >> 9) & 1` for each sign. At n=16 paired seeds, the 20,000 sequential historical resamples cover only **64 unique sign masks**, compared with **2^16=65,536 possible assignments**. The draws therefore cannot be treated as 20,000 iid uniform null sign assignments. The reported Monte Carlo p-value minimum near `1/20001` has no valid tail calibration for the intended permutation distribution. '64' is the observed *distinct-mask support*, **not** an independently estimated effective sample size.

R25A-R7 retrospectively computed all 2^n sign assignments for each nonempty endpoint, using the two-sided paired statistic |sum(seed deltas)| with the declared 1e-12 tie tolerance; then fixed-size Holm step-down across **24 endpoints per compiler** (including unobserved E3 with p=1). This is an **archival score reanalysis**, not a retroactively preregistered result.

| Comparison | Historical p | Corrected exact p | Corrected Holm24 p | Interpretation |
|---|---:|---:|---:|---|
| GCC CONTRADICTORY E1 | 0.00005 | 0.0078125 | 0.1328125 | Not familywise significant |
| Clang CONTRADICTORY E1 | 0.00005 | 0.0078125 | 0.1171875 | Not familywise significant |
| GCC DISTRIBUTION_SHIFT E1 | 0.031298 | 0.0078125 | 0.1328125 | Not familywise significant |
| GCC RETIREMENT_FRESH E1 | 0.00005 | 0.00390625 | 0.078125 | Not familywise significant |
| Clang DISTRIBUTION_SHIFT E1 | 0.031298 | 0.0009765625 | 0.0234375 | Significant arithmetic; invalid narrated task semantics |
| Clang RETIREMENT_FRESH E1 | 0.00005 | 0.001953125 | 0.03515625 | Significant arithmetic; main comparison lacks retirement/reopen |
| GCC REVERSAL E1 | 0.00005 | 0.00048828125 | 0.01171875 | Harm survives; reversal task semantics mismatch |
| Clang REVERSAL E1 | 0.00005 | 0.0009765625 | 0.0234375 | Harm survives; reversal task semantics mismatch |

**Not independent E1/E2 confirmations:** across all eight families and both compilers, each seed's E2 regret difference is exactly 16× its E1 accuracy difference. Both appear in the original 24-endpoint correction to prevent post-hoc family shrinkage, but must not be presented as separate replications.

**Exploratory E3:** the GCC `DISTRIBUTION_SHIFT` E3 endpoint survives Holm24 (p=0.0205078125) but conditions on observed ADAPT publications, is missing in some families, and is not a deployable selective-policy effect.

## Generator and native interpretation block

- `DISTRIBUTION_SHIFT`: fixed-truth observation-polarity reversal, not a validated target-logic shift.
- `RETIREMENT_FRESH`: retirement/reopen is only exercised in `ctl==5`, outside the principal comparator.
- `CONTRADICTORY`: intended authoritative-source reliability distinction is not exposed to the learner.
- `REVERSAL`: pre-change observation labels are complemented across observed slots, not a clean single-slot truth flip.
- `NOISY_INCOMPLETE`: eight observation steps instead of the narrated six.
- `REPEATED`: observations precede both query halves; not a learning curve.
- Native/FG/H11 ABI and absent scratch sources remain unqualified in this archival reanalysis.

## Corrective authority instruction

Historical R24 `BENCHMARK_QUALIFIED_CAPABILITY_PARTIAL` **must not be rewritten, silently erased, or promoted**. Every current ledger, summary, benchmark claim, or decision citing that historical disposition must carry this addendum or equivalent linked warning: `CURRENT_RELIANCE=HOLD_STATISTICAL_AND_TASK_PROVENANCE`. No CGC/LAO capability promotion on R24 positive p-values or R25A-R7 archival statistics. The reserved holdout remains sealed.

Follow-on registration: `R25A_R8_ENUMERATION_REGISTRATION.json`. It fixes **future** test method and receipt validation ahead of any R8 run; it does not make the already-observed R7 results prospectively preregistered. Any corrected task generator or new data requires separately frozen source, data, tests and qualification gates.
