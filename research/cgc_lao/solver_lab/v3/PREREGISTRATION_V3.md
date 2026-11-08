# Preregistration: qualification version v3

Written **before any v3 evaluation**. Git history records this file before every v3 record.

## Why v3 exists

The v2 primary (`RMF_NOEVENT`) failed G9 on v2 VALIDATION. Its FLIP45 harm against no-learning had a 95%
upper bound of +0.000655, above the +0.0005 tolerance; FLIP45 still improved significantly against the v1
learner. By the v2 preregistration, the v2 HOLDOUT and OOD partitions stay sealed and are never used. See
[`../v2/evidence/DISPOSITION_V2.json`](../v2/evidence/DISPOSITION_V2.json).

Diagnosis carried into v3:

1. **Selection.** v2 selected on point estimates, although the selected arm's DEV upper bound already exceeded
   the tolerance. v3 selects on the gates' own 95% bounds.
2. **Mechanism.** With four duplicate pairs, a near-noise channel (q ≈ 0.45) leaves weight on the q = 0.2
   hypothesis whenever chance produces few disagreements, and that hypothesis over-adapts. v3 adds
   candidates without that hypothesis (`RMF3`) or with it down-weighted (`RMF4_W`).

## What was seen before this document

All v1 and v2 records: v2 TRAIN, v2 DEV rounds 0–1 and the correlation diagnostic, and v2 VALIDATION for
`RMF_NOEVENT`, `SAFE4_P8` and the controls. No v3 data.

## Reused unchanged (frozen v2, read-only)

- The SLB2 world and mechanisms. `env3.generate` with the v2 root reproduces `env2.episodes` bitwise (tested).
- The gate scales and every coefficient from the v2 TRAIN partition: `δ_clean = 0.00411486`,
  `δ_all = 0.00145059`, online budget `0.10061226`, `τ_safe = +0.0005`, v1 clean margin `+0.0005`.
- The gate definitions G1–G10, F1–F6 and the Bonferroni family claims (`cgc_lao_v2.report2`).
- The agents, the native C11 component, the provider contract, the host, and the native parity procedure and
  tolerances.

v3 does no training.

## New in v3

- **RNG root** `elpis/cgc_lao/solver_lab/v3/`, with fresh partitions disjoint from every v1 and v2 partition
  (tested): DEV 256, VALIDATION 32, HOLDOUT 64, OOD 64 router seeds.
- **Primary candidates** (DUP2_4, DUPLICATE):

  | Arm | Hypotheses | Prior |
  |---|---|---|
  | `RMF3` | {0, 0.4, 0.48} | uniform |
  | `RMF4_E25` | {0, 0.4, 0.48, event 0.4} | (1, 1, 1, 0.25) |
  | `RMF4_W` | {0, 0.2, 0.4, 0.48} | (1, 0.5, 1, 2) |
  | `RMF_NOEVENT` | {0, 0.2, 0.4, 0.48} | uniform |
  | `RMF5_E25` | {0, 0.2, 0.4, 0.48, event 0.4} | (1, 1, 1, 1, 0.25) |

  The equal-information controls `RMF_V1`, `BETA_ALL`, `SGD_ALL`, `DISAGREE_SGD`, `PLUGIN_Q_BAYES` and
  `STATIC_ATT_ALL` are also eligible.
- **Fallback candidates** (FIRST_ONLY): `SAFE4_P8`, `SAFE4_P4`, `SAFE3_P4` ({0, 0.4, 0.48}, prior (1, 4, 4)).

## Selection: one DEV round, final

DEV router seeds 0–31 × 2048 episodes, design and robustness regimes, every candidate, control and reference.

- **Primary.** Eligible if G1, G2, G3, G4, G5, G7, G9 and G10 **all pass by their own 95% (G7: 97.5%) bounds**
  on this round. Among eligible arms, take the simplest within ε = 0.0002 of the best ALL terminal regret
  (state scalars, then likelihood evaluations, then name). If none is eligible, v3 ends `NOT_QUALIFIED` for
  the primary and no primary VALIDATION is run.
- **Fallback.** Eligible if F1–F6 pass by their bounds; take the lowest DEV CLEAN regret.

## Qualification (as v2)

- VALIDATION: 32 × 4096 episodes, design and robustness regimes, plus native parity on seeds 0–7 × 256.
- HOLDOUT: 64 × 4096, plus native parity. It is opened once, and only if the primary passes G1–G5, G7, G9 and
  G10 on VALIDATION.
- OOD: 64 × 2048, the correlation sweep and the OOD regimes. Opened once, after the HOLDOUT.
- Dispositions:
  - primary `QUALIFIED_SYNTHETIC_CONTRACT_I` if it passes every gate on both VALIDATION and HOLDOUT;
  - fallback `QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE` if it passes F1–F6 on both;
  - family claims are listed per regime family.
- `ρ_admit` is taken from the OOD correlation sweep exactly as preregistered in v2.

## Disclosure of the research path

This is the third qualification version of the mechanism family. v1 qualified a different arm under the v1
gates; v2's primary failed VALIDATION. Every version has its own untouched partitions, and v3's claims rest
only on v3's VALIDATION and HOLDOUT. The candidate set was informed by v2 results, which is why v3 uses
fresh seeds for every evaluation.
