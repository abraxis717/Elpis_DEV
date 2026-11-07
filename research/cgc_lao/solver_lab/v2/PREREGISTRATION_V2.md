# Preregistration: CGC-LAO solver lab, study v2 (SLB2)

Written after the SLB2 TRAIN calibration and **before any v2 DEV, VALIDATION, HOLDOUT or OOD evaluation**.
Git history records this file before every v2 DEV, VALIDATION, HOLDOUT and OOD record. The v1 study
(`../PREREGISTRATION.md`, `../FROZEN.json`, `../evidence/`) is immutable and grants v2 nothing; v1's consumed
partitions are never reused. Benchmark: [SLB2](specs/slb2.spec.json) (new RNG root, fresh partitions).

## What was seen before this document

- Everything in v1, including its holdout (consumed) and its post-holdout DEV diagnostic on **v1** seeds,
  which motivated the q = 0.48 extension (`../evidence/dev/next_grid.json`).
- SLB2 TRAIN only: baseline and control tuning ([`evidence/train/tune.json`](evidence/train/tune.json)), CGC
  retraining ([`coefficients/cgc_reference.v2.json`](coefficients/cgc_reference.v2.json)) and reference levels
  ([`evidence/train/levels.json`](evidence/train/levels.json)).

## Hypotheses

- **H-A2.** Adding a near-noise channel hypothesis (q = 0.48) to the v1 learner materially reduces the FLIP45
  harm, keeps v1's clean-regime advantage, and passes G1–G8 under the independent-readings contract.
- **H-B.** Duplicate-reading adaptation is safe only when reading-level corruption is independent. Its
  advantage degrades as the reading correlation ρ grows; the admissible range of ρ is characterised.
- **H-C.** The learner beats equal-information controls that receive the same readings, or a simpler control
  matches it and is preferred.
- **H-F.** A single-reading conservative configuration (Contract II fallback) is never worse than no-learning
  under heavy or near-noise corruption, and still learns on clean feedback.

## Arms

| Role | Arms | Contract / provider mode |
|---|---|---|
| Primary candidates | `RMF5_E25`, `RMF5_E10`, `RMF5_E50`, `RMF4_E25`, `RMF5B_E25` (band hypothesis q ∈ {0.42, 0.46, 0.49}), `RMF5_E25_GREEDY` | DUP2_4 / DUPLICATE |
| Equal-information controls (also eligible for selection) | `RMF_V1` (frozen v1 learner), `RMF_NOEVENT`, `BETA_ALL` (κ 8), `SGD_ALL` (η 0.5), `DISAGREE_SGD` (η 1.0), `PLUGIN_Q_BAYES`, `STATIC_ATT_ALL`, `NOOP` | DUP2_4 / DUPLICATE |
| Fallback candidates (Contract II) | `SAFE4_P2`, `SAFE4_P4`, `SAFE4_P8` | DUP2_4 / FIRST_ONLY |
| System-level references | `NOOP`, `SGD` (η 0.5), `BETA` (κ 8), `CGC_REFERENCE` (v2 weights `d79b38db…1cd48`), `STATIC_ATTENUATION`, `CLEAN_BAYES` | ORIGINAL / FIRST_ONLY |
| Privileged diagnostic | `KNOWN_Q_ORACLE` (never a candidate or gate reference) | as evaluated |

## Gate scales (SLB2 TRAIN router seeds 16–47, fixed formulas as v1)

| Quantity | Value |
|---|---|
| Clean reference | `CLEAN_BAYES` (TRAIN clean gain 0.02057429) |
| `δ_clean` = 0.20 × clean gain | **0.00411486** |
| ALL reference | `CGC_REFERENCE` (TRAIN ALL gain 0.00725295) |
| `δ_all` = 0.20 × ALL gain | **0.00145059** |
| Online budget `B` = 1.10 × TRAIN NOOP online regret (ALL) | **0.10061226** |
| Safety tolerance `τ_safe` (near-noise channels) | **+0.0005** |

## Gates for the Contract I primary (Δ = candidate − reference, paired router-seed intervals)

| Gate | Requirement |
|---|---|
| G1 | FLIP40 Δ vs `NOOP`: 95% upper < 0 |
| G2 | CLEAN Δ vs `NOOP`: 95% upper < 0 |
| G3 | CLEAN Δ vs `CLEAN_BAYES`: 95% upper ≤ δ_clean |
| G4 | ALL Δ vs `CGC_REFERENCE`: 95% upper ≤ δ_all |
| G5 | online regret (ALL): 95% upper ≤ B |
| G6 | G1–G5, G9, G10 on VALIDATION **and** HOLDOUT |
| G7 | QUNIF or BURST Δ vs `NOOP`: 97.5% upper < 0 |
| G8 | the v1 and v2 test suites pass, including native parity; logical state ≤ 64 scalars |
| G9 (FLIP45 improvement) | FLIP45 Δ vs `RMF_V1`: 95% upper < 0, **and** FLIP45 Δ vs `NOOP`: 95% upper ≤ τ_safe |
| G10 (v1 clean advantage kept) | CLEAN Δ vs `RMF_V1`: 95% upper ≤ +0.0005 |

**Family claims.** For each of CLEAN, FLIP20, FLIP40, FLIP45, FLIP48, QUNIF, BURST and EVENT40, with
Bonferroni-adjusted two-sided 99.375% intervals: `USEFUL` if Δ vs NOOP upper < 0, `SAFE` if upper ≤ τ_safe,
otherwise `NOT_QUALIFIED`. A candidate is qualified only for the families it passes. CLEAN additionally needs
G3.

**Equal-information comparison.** Δ vs every control with 95% intervals is reported per design and
robustness regime (algorithm-level gain). System-level gain is the comparison with the ORIGINAL-contract
references. The two are never pooled.

## Gates for the Contract II fallback (FIRST_ONLY: the duplicate is discarded)

F1 FLIP40 Δ vs NOOP 95% upper < 0 · F2 CLEAN Δ vs NOOP 95% upper < 0 · F3 FLIP45 and FLIP48 Δ vs NOOP
95% upper ≤ τ_safe · F4 EVENT40 Δ vs NOOP 95% upper < 0 · F5 online 95% upper ≤ B · F6 QUNIF or BURST useful
(97.5%). There is no clean-noninferiority gate; clean retention is reported. All must hold on VALIDATION
and HOLDOUT.

## Selection (DEV, design and robustness regimes; the latest round decides)

- **Primary.** The pool is the primary candidates plus the equal-information controls except NOOP. A
  candidate is *eligible* if it passes G1–G5, G9 and G10 by point estimate. Let A* be the lowest DEV ALL
  terminal regret among eligible candidates. Select the **simplest** eligible candidate within ε = 0.0002 of
  A*: fewest logical state scalars, then fewest likelihood evaluations per update, then name. If none is
  eligible, select the candidate passing most gates, ties by ALL regret, and record it as not qualifiable.
- **Fallback.** Eligible if F1–F5 hold by point estimate; select the lowest DEV CLEAN regret.

The selection record is committed before the native implementation is written against it. One primary and
one fallback are frozen, together with the native and host sources, before VALIDATION.

## Native component (Milestone D) tolerances, fixed now

- Native vs the explicit-loop scalar reference, same machine: **bitwise**. Every state scalar after every
  update, every proposal and every decision must be exactly equal.
- Native vs the vectorised learner: actions and decisions equal except documented near-ties (predictive gap
  < 1e-9, or |predictive − STOP payoff| < 1e-9, at the decision); state |Δ| ≤ 1e-9.
- Samples: VALIDATION router seeds 0–7 × 256 episodes per VALIDATION regime for the frozen primary and
  fallback, and the same on HOLDOUT after it is opened.

## Correlation characterisation and admission (Milestone B)

On the OOD partition, opened once after the HOLDOUT, the frozen primary runs in DUPLICATE mode under ρ ∈
{0 (FLIP40), 0.1, 0.25, 0.5, 0.75, 0.9, 1 (EVENT40)}. `ρ_admit` is the largest grid value such that every
ρ' ≤ ρ_admit has Δ vs NOOP 95% upper < 0. The host admits DUPLICATE mode only for a provider whose
declaration of independent reading-level corruption is backed by registered measurement provenance. A
provider certifying only a bound ρ ≤ ρ_cert is admitted iff ρ_cert ≤ ρ_admit. Any other provider runs the
Contract II fallback. The other OOD regimes are reported, pass or fail.

## Statistics, multiplicity and holdout discipline

Router seed is the pairing unit, with Student-t intervals on n_seeds − 1 degrees of freedom. Core gates must
all pass (intersection–union, no adjustment); family claims are Bonferroni-adjusted; G7 and F6 use 97.5%.
Selection uses DEV only. VALIDATION (32 seeds × 4096 episodes) runs once per frozen arm. HOLDOUT (64 × 4096)
is opened once, only if the frozen primary passes its VALIDATION gates. OOD (64 × 2048) is opened once after
the HOLDOUT. Dispositions:

- Contract I primary: `QUALIFIED_SYNTHETIC_CONTRACT_I` if G1–G10 hold.
- Contract II fallback: `QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE` if F1–F6 hold.
- Otherwise `NOT_QUALIFIED`, with the failing gates named.

None of these is production admission: a real host must establish its provider's independence separately.
