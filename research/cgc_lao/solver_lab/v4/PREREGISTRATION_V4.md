# Preregistration: qualification version v4 (final)

Written **before any v4 evaluation**; git history records it before every v4 record.

## Why v4 exists

`RMF_NOEVENT` failed G9 on both the v2 and v3 VALIDATION partitions, with FLIP45 harm +0.000496 and
+0.000473 against the +0.0005 tolerance. Its true near-noise harm sits at the tolerance. The cause is the
selection rule, not the data: choosing the best ALL regret among DEV-eligible arms picks the arm on the FLIP45
boundary (a winner's curse), because the q = 0.2 hypothesis that helps FLIP20 is the one that over-adapts near
noise. The v2 and v3 HOLDOUT and OOD partitions stay sealed. See `../v2/evidence/DISPOSITION_V2.json` and
`../v3/evidence/DISPOSITION_V3.json`.

## What was seen

Every v1, v2 and v3 record, including the v3 DEV round (`../v3/evidence/dev/selection_round.json`), where
`RMF3`, `RMF4_W` and `RMF4_E25` passed every gate with FLIP45 margins. No v4 data.

## Reused unchanged

- SLB2 world and mechanisms (`env3.generate`, bitwise SLB2), v2 TRAIN gate scales and coefficients.
- The 95% gates G1–G10, F1–F6 and the family claims (`cgc_lao_v2.report2`) for VALIDATION and HOLDOUT.
- The agents, native component, provider contract, host and native parity procedure and tolerances.
- The VALIDATION, HOLDOUT and OOD plans and the holdout-opening rule of v3.

## New in v4

- **RNG root** `elpis/cgc_lao/solver_lab/v4/`: DEV 256, VALIDATION 32, HOLDOUT 64 and OOD 64 router seeds,
  disjoint from every v1, v2 and v3 partition (tested).
- **Candidates.**
  - Primary (DUP2_4, DUPLICATE): `RMF3` {0, 0.4, 0.48} (24 scalars), `RMF4_W`, `RMF4_E25`, `RMF_NOEVENT`,
    `RMF5_E25`, plus the equal-information controls.
  - Fallback (FIRST_ONLY): `SAFE3_P4` (24 scalars) and `SAFE4_P8`.
- **Robust Occam selection.** One DEV round, router seeds 0–31 × 2048 episodes, design and robustness regimes,
  final whatever it shows. A candidate is eligible if every primary gate (G1, G2, G3, G4, G5, G7, G9, G10)
  passes with its interval computed at **99.9%** two-sided (fallback: F1–F6 at 99.9%). The selected arm is the
  **smallest eligible mechanism**: fewest logical state scalars, then fewest likelihood evaluations per
  update, then lowest DEV ALL regret (fallback: CLEAN regret), then name. The ALL objective is only a
  tie-break, so the rule cannot reward sitting on a safety boundary. If no primary is eligible, v4 ends
  `NOT_QUALIFIED` for the primary.

## Qualification and disposition

As in v3:

- VALIDATION is 32 × 4096 episodes with native parity.
- HOLDOUT is 64 × 4096 with native parity. It is opened once, only if the primary passes every 95% gate on
  VALIDATION.
- OOD is 64 × 2048 (correlation sweep and OOD regimes), opened once after the HOLDOUT.
- `ρ_admit` follows the v2 rule.
- Dispositions:
  - primary `QUALIFIED_SYNTHETIC_CONTRACT_I` if it passes every gate on VALIDATION and HOLDOUT;
  - fallback `QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE` if it passes F1–F6 on both;
  - family claims are listed per family;
  - otherwise `NOT_QUALIFIED`.

This is the fourth qualification version of the mechanism family. Each version's claims rest only on its own
untouched partitions, and all versions are reported.
