# Preregistration: CGC-LAO solver lab on SLB1

Written after TRAIN calibration and **before any DEV, VALIDATION or HOLDOUT evaluation**. Git history
records this file before every DEV, VALIDATION or HOLDOUT evidence record. Nothing below may be
changed after a VALIDATION or HOLDOUT number has been seen; a change would be a new, separately
labelled experiment version.

## What was seen before this document

- Exploratory scratch prototypes on throwaway `numpy.random.default_rng` seeds, outside every lab
  partition: they showed the unknown-corruption mixture learner losing to no-learning under 40% flips
  while a known-q learner wins, and that duplicate readings or longer horizons change that. These
  observations motivated the candidate families. They are not evidence.
- TRAIN partition only: baseline tuning ([`evidence/train/tune.json`](evidence/train/tune.json)), CGC
  coefficient training ([`coefficients/cgc_reference.v1.json`](coefficients/cgc_reference.v1.json)) and
  reference levels ([`evidence/train/levels.json`](evidence/train/levels.json)).

## Hypotheses

- **H-A (original contract).** A bounded analytical reliability-mixture learner with an expected-utility
  terminal decision can beat no-learning under FLIP40 while keeping at least 80% of the strongest
  clean reference's clean-regime gain, with eight single readings.
- **H-E (information contingency).** If H-A fails because corruption is not identified from eight single
  readings, a declared contract that delivers more *independent* evidence (duplicate readings of the
  selected event, or more acquisitions) lets the same bounded learner meet every gate.

## Arms

Admissible baselines: `NOOP`, `SGD` (η = 0.5), `BETA` (κ = 8), `CGC_REFERENCE` (coefficients
`95296ad5…b742`), `STATIC_ATTENUATION` (CGC proposals × 0.5), `CLEAN_BAYES`. Privileged diagnostic:
`KNOWN_Q_ORACLE` (told the true corruption rate and bias scale; never a candidate and never a gate
reference). Candidates are defined in `src/cgc_lao_lab/candidates.py`; each is frozen with its source
digest before VALIDATION.

## Gate scales (computed on TRAIN router seeds 16–47 by fixed formulas)

| Quantity | Formula | Value |
|---|---|---|
| Clean reference | admissible baseline with the lowest TRAIN CLEAN terminal regret | `CLEAN_BAYES` |
| `δ_clean` | 0.20 × (NOOP − clean reference), TRAIN CLEAN | **0.00416316** |
| ALL reference | admissible baseline with the lowest TRAIN ALL terminal regret | `CGC_REFERENCE` |
| `δ_all` | 0.20 × (NOOP − ALL reference), TRAIN ALL | **0.00148820** |
| Online budget `B` | 1.10 × TRAIN NOOP online regret on ALL (PROBE_EARLY) | **0.10273330** |

## Gates (all evaluated on paired router-seed differences; Δ = candidate − reference, negative favours the candidate)

| Gate | Requirement |
|---|---|
| G1 corrupted improvement | FLIP40 terminal Δ vs `NOOP`: 95% CI upper bound **< 0** |
| G2 useful clean learning | CLEAN terminal Δ vs `NOOP`: 95% CI upper bound **< 0** |
| G3 clean noninferiority | CLEAN terminal Δ vs `CLEAN_BAYES`: 95% CI upper bound **≤ δ_clean** |
| G4 ALL noninferiority | ALL terminal Δ vs `CGC_REFERENCE`: 95% CI upper bound **≤ δ_all** |
| G5 online budget | candidate online regret on ALL: 95% CI upper bound **≤ B** |
| G6 generalisation | G1–G5 hold on VALIDATION **and** on HOLDOUT |
| G7 shift robustness | terminal Δ vs `NOOP` with two-sided **97.5%** CI upper bound < 0 on at least one of the primary shift tests `QUNIF`, `BURST` (Bonferroni for "one of two"); every other shift regime is reported, pass or fail |
| G8 boundaries | the lab test suite passes; volatile state ≤ 64 scalars per episode; no privileged input |

For a changed contract, the candidate runs under that contract and every reference arm runs under
`ORIGINAL` (the references do not model extra readings; `NOOP` does not learn). The changed contract's
extra expert queries and readings are reported with the result.

## Selection and holdout discipline

1. DEV (rounds of 16 fresh router seeds, 2048 episodes per seed and regime) is used for development,
   in design regimes only. Selection per contract: the candidate passing the most gates by point
   estimate on the latest DEV round, ties broken by ALL terminal regret.
2. At most one candidate per contract is frozen (configuration + source SHA-256) and run on VALIDATION
   (32 seeds × 4096 episodes, design and shift regimes). VALIDATION may be used at most twice per
   contract; a second use is disclosed.
3. HOLDOUT (64 seeds × 4096 episodes) is opened once, for the frozen candidates (at most one per
   contract: `ORIGINAL` and the single cheapest qualifying changed contract), with all reference arms.
   Each contract's result is a separate claim; the `ORIGINAL` result is never relabelled by a changed
   contract's result.
4. Dispositions: `QUALIFIED_SYNTHETIC` for a contract whose frozen candidate passes G1–G8 on
   VALIDATION and HOLDOUT; otherwise `NOT_QUALIFIED`, with the failing gates named. A qualified result
   is synthetic (SLB1 only): it is not runtime integration, production admission or a claim about the
   historical Q4E simulator.
