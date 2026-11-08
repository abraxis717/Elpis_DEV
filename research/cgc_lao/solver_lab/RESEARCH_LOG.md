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

## Phase 1: DEV research loop (design regimes only)

**Round 0** (seeds 0–15; [`evidence/dev/round0_original.json`](evidence/dev/round0_original.json)).

- `RMF_U3` (family A) → bounded q-mixture ADF filter {0, 0.2, 0.4}, Bayes terminal decision → CLEAN −0.0180
  (87% of CLEAN_BAYES's gain), FLIP40 **+0.0012** vs NOOP → the unknown-q posterior reproduces the
  historical failure on an independent benchmark; ADF equals exact grid Bayes to 1e-5 (scratch check), so
  it is not an approximation defect → map the frontier.
- `RMF_P50/P67/P80`, `RMF_U6` (family A, prior mass shifted to corruption) → retention 81/73/64/82%, FLIP40
  +0.0002/−0.0007/−0.0010/+0.0003 → a monotone Bayes frontier: FLIP40 below NOOP costs >20% of clean
  gain → the decision rule is not the limit; the regime posterior is (mean P(clean|h) 0.45 CLEAN vs 0.23
  FLIP40).
- `RMF_U3_CONS1` (family B, conservative advantage τ=1) → never adapts (CLEAN −0.0002) → posterior
  advantage SDs are large at T=8; dominated.
- `GATED_CGC_U3` (family B, CGC proposals × E[(1−2q)²|h]) → retention 46%, FLIP40 −0.0010 → behaves like
  static attenuation: gating by an unidentified reliability is attenuation, not targeting.
- `RMF_U3_GREEDY/INFOQ` (family C) → online regret 0.089 → 0.065; terminal frontier unchanged → acquisition
  cannot buy reliability information from single readings.

**Round 1** (seeds 16–31; `round1_*.json`).

- `RMF_Q2_*` (family A, priors on {clean, 0.4} only: by the complete-class argument these trace the best
  CLEAN-vs-FLIP40 trade-off) → Q2_P67: retention 81%, FLIP40 +0.0001 → frontier confirmed; no point passes
  G1 and G3 together.
- `RMF_U3_MINIMAX` (family B, minimax regret over plausible hypotheses) → FLIP40 **+0.0023** → worse: the
  plausible set is still dominated by the clean hypothesis. `RMF_U3_CONS025` → dominated.
- `RMF_U3_INFOQ01` (family C, information-heavy) → AUC 0.83 vs 0.80, FLIP40 +0.0010 → no frontier shift.
- **Family E (changed contracts).** `DUP2` (+8 readings, 0 queries): RMF_U3 retention 99.8%, FLIP40
  −0.0018, every gate passes by CI. `DUP2_4` (+4 readings): retention 99%, FLIP40 −0.0006, every gate
  passes. `T12/T16` (+4/+8 queries): RMF_U3 still fails G1 (+0.0009/+0.0002); P67 passes.
  **Diagnosis:** a second reading of the same event identifies channel corruption independently of the
  unknown bias (disagreement probability 2q(1−q)); extra single readings do not, because each expert's
  bias absorbs them.

**Diagnostics** (seeds 32–47; [`evidence/dev/diagnostics.json`](evidence/dev/diagnostics.json)).

| Contract | Extra reads | Extra queries | Regime AUC (CLEAN vs FLIP40) | RMF_U3 FLIP40 Δ vs NOOP | RMF_U3 clean retention |
|---|---:|---:|---:|---:|---:|
| ORIGINAL | 0 | 0 | 0.800 | +0.00150 | 88% |
| DUP2@2 | 2 | 0 | — | +0.00027 | 96% |
| DUP2@3 | 3 | 0 | — | −0.00023 | 99% |
| DUP2_4 | 4 | 0 | 0.983 | −0.00058 | 99.5% |
| DUP2 | 8 | 0 | 0.999 | −0.00121 | 100% |
| T12 | 0 | 4 | — | +0.00088 | 118%* |
| T16 | 0 | 8 | 0.892 | +0.00015 | 140%* |

\*more data than the eight-step clean reference had. The original-contract AUC (0.80) matches the
historical ledger's T8 value (0.81–0.82).

**Design rule fixed before round 2:** a learner run under a duplicate-reading contract keeps an
event-level corruption hypothesis, because duplicate agreement cannot test the contract's own
independence assumption (event-level corruption makes readings agree).

**Round 2: final selection** (seeds 48–63; `round2_*.json`), preregistered rule (most gates by point
estimate, ties by ALL terminal regret):

- ORIGINAL: every candidate passes 4/5 gates (G1 or G3 fails); best ALL: **`RMF_U3`** (0.05447;
  FLIP40 +0.0012, retention 87%).
- DUP2_4: all event-hedged candidates pass 5/5 (also by CI); best ALL: **`RMF_U3E25`** (0.05273;
  FLIP40 −0.00086, CI upper −0.00056; retention 95%).

## Phase 2: freeze, VALIDATION, HOLDOUT (frozen code; per-file digests in `FROZEN.json`)

| Candidate | Phase | G1 FLIP40 vs NOOP | G3 CLEAN vs CLEAN_BAYES | G2, G4, G5, G7 | Status |
|---|---|---|---|---|---|
| `RMF_U3` (ORIGINAL) | VALIDATION | +0.00118 [+0.00097, +0.00138] **fail** | +0.00265 (upper +0.00286) | pass | |
| `RMF_U3` (ORIGINAL) | HOLDOUT | +0.00113 [+0.00096, +0.00131] **fail** | +0.00265 (upper +0.00282) | pass | NOT_QUALIFIED |
| `RMF_U3E25` (DUP2_4) | VALIDATION | −0.00091 [−0.00107, −0.00074] | +0.00089 (upper +0.00098) | pass | |
| `RMF_U3E25` (DUP2_4) | HOLDOUT | −0.00094 [−0.00109, −0.00079] | +0.00089 (upper +0.00095) | pass | QUALIFIED_SYNTHETIC |

VALIDATION was used once per contract; no candidate was revised after it. Disclosed shift harms (both
phases): FLIP45 for both candidates, EVENT40 for both, ROUTERWIDE40 for `RMF_U3` only.

**Post-holdout fixed-log diagnostic** (DEV seeds 64–79; [`evidence/dev/fixed_log.json`](evidence/dev/fixed_log.json)):
with every learner updated on NOOP's own action log, FLIP40 deltas vs NOOP are RMF_U3 +0.0012, CLEAN_BAYES
+0.0122, KNOWN_Q −0.0012, RMF_P67 −0.0004, and RMF_U3E25 (DUP2_4) −0.0008 with 96% clean retention. These
match the on-policy results, so the original-contract failure is not acquisition bias.

**Post-holdout next-experiment diagnostic** (DEV seeds 80–95; [`evidence/dev/next_grid.json`](evidence/dev/next_grid.json);
not a qualification). Adding a q = 0.48 channel hypothesis to the frozen DUP2_4 learner cuts the FLIP45 harm
from +0.0015 to +0.0003 (n.s.) at about 1% of the clean gain, and improves FLIP40 to −0.0014. Raising
event-level prior mass to 3 makes EVENT40 useful (−0.0005) but drops clean retention to 76%. Event-level
corruption is the original single-reading problem, so it must be handled by auditing the contract.
