# Results: a bounded reliability-mixture learner for selected, possibly corrupted feedback

**Benchmark:** SLB1 ([BENCHMARK_SPEC.md](BENCHMARK_SPEC.md)), an independent illustrative benchmark; numbers
are not comparable to the historical Q4E/Q4F measurements. **Gates:** [PREREGISTRATION.md](PREREGISTRATION.md),
frozen before any DEV, VALIDATION or HOLDOUT run. **Disposition record:**
[`evidence/DISPOSITION.json`](evidence/DISPOSITION.json) (recomputed from per-seed numbers by the test suite).

## Disposition

| Contract | Frozen candidate | Status | Failing gates |
|---|---|---|---|
| `ORIGINAL` (8 steps, 1 reading each) | `RMF_U3` | **NOT_QUALIFIED** | G1 on VALIDATION and HOLDOUT |
| `DUP2_4` (8 steps; 2 independent readings of the selected event at steps 0–3) | `RMF_U3E25` | **QUALIFIED_SYNTHETIC** | none |

- Under the original eight-step, single-reading contract, no admissible candidate in five families passed
  both the corrupted-regime gate (beat NOOP under 40% flips) and clean noninferiority. The limit is
  information, not estimator or decision quality (see the mechanism section below).
- With four extra feedback readings and **zero** extra expert queries, the same bounded learner passes
  every preregistered gate on VALIDATION (32 router seeds) and on the untouched HOLDOUT (64 router seeds).
  This is a result for the changed contract only; it is not a solution of `ORIGINAL`.
- Qualification level reached: **synthetic scientific and generalisation qualification on SLB1**. Not
  runtime integration, not production admission, not a claim about the historical simulator.

## The algorithm (frozen as `RMF_U3E25`; `FROZEN.json`)

Per episode, volatile state of **31 scalars** (28 filter + 3 coverage counts), reset at episode start:

- `M = 4` feedback-reliability hypotheses: clean (`q=0`), channel flips `q=0.2`, channel flips `q=0.4`,
  event-level flips `q=0.4` (all readings of an event share the flip). Prior weights `(1, 1, 1, 0.25)`.
- Per hypothesis `m`: a log-weight `ℓ_m`; per hypothesis and expert, a Gaussian belief `N(μ_ma, v_ma)` over
  the hidden bias, starting at `N(0, 1)`.

**Update** after reading(s) `y_1..y_R` of the selected expert `a` at public logit `z = W_a·x_t` (one pass,
`M × 16` quadrature points):

```
for each hypothesis m:
    A1 = P(readings | clean event = 1, m),  A0 = P(readings | clean event = 0, m)
         channel: A1 = (1-q)^{n1} q^{n0},  A0 = q^{n1} (1-q)^{n0}
         event:   readings must agree; A1 = 1-q or q, A0 = q or 1-q; disagreement -> 0
    for the 16 Gauss-Hermite nodes b_g = μ_ma + sqrt(v_ma) ξ_g:
        λ_g = ω_g [σ(z + b_g) A1 + (1 - σ(z + b_g)) A0]
    L = Σ λ_g;  μ_ma <- Σ λ_g b_g / L;  v_ma <- max(Σ λ_g b_g² / L - μ_ma², 1e-6)   (moment matching)
    ℓ_m <- ℓ_m + log L                                                            (Bayes rule)
normalise ℓ (max = 0), floor at -1e4
```

**Acquisition:** the checked-in PROBE_EARLY schedule (`selector.probe_early_from_pred`) applied to the
posterior predictive success probabilities `E[σ(z + b_a) | history]`.

**Terminal decision:** for each query, expected utility under the posterior mixture,
`Q(a) = Σ_m w_m ∫ σ(z'_a + b) N(b; μ_ma, v_ma) db` (16-node quadrature), against the STOP payoff; choose
the maximum. No threshold, no tuning at inference.

**Contract `DUP2_4`:** at acquisitions 0–3 the host delivers a second, independently corrupted reading of
the same selected event; acquisitions 4–7 deliver one reading. Cost: +4 readings per episode, +0 expert
queries, no change to online exposure.

ADF moment matching equals brute-force grid Bayes to 1e-7 on a first update and tracks a five-observation
history within 0.01 (tests). In the exploratory scratch prototype (outside the lab partitions; not an
evidence record), ADF and an 81-point exact grid posterior differed by 1e-5 in terminal regret.

## HOLDOUT (64 untouched router seeds × 4096 episodes per regime; opened once)

Terminal regret (probability units; lower is better). Δ vs NOOP in parentheses. ‡ privileged reference.

| Arm (contract) | CLEAN | FLIP20 | FLIP40 | ALL | online (ALL) |
|---|---:|---:|---:|---:|---:|
| NOOP (ORIGINAL) | 0.06451 | 0.06442 | 0.06476 | 0.06456 | 0.0937 |
| SGD (ORIGINAL) | 0.04470 (−0.01981) | 0.05644 (−0.00798) | 0.06941 (+0.00465) | 0.05685 (−0.00771) | 0.0905 |
| BETA (ORIGINAL) | 0.04531 (−0.01921) | 0.05719 (−0.00723) | 0.06992 (+0.00516) | 0.05747 (−0.00709) | 0.0911 |
| CGC_REFERENCE (ORIGINAL) | 0.04757 (−0.01694) | 0.05617 (−0.00825) | 0.06584 (+0.00108) | 0.05653 (−0.00804) | 0.0903 |
| STATIC_ATTENUATION (ORIGINAL) | 0.05393 (−0.01058) | 0.05855 (−0.00586) | 0.06373 (−0.00103) | 0.05874 (−0.00583) | 0.0912 |
| CLEAN_BAYES (ORIGINAL) | 0.04262 (−0.02189) | 0.05892 (−0.00550) | 0.07709 (+0.01233) | 0.05954 (−0.00502) | 0.0920 |
| RMF_U3 (ORIGINAL) | 0.04527 (−0.01924) | 0.05592 (−0.00850) | 0.06589 (+0.00113) | 0.05570 (−0.00887) | 0.0900 |
| **RMF_U3E25 (DUP2_4)** | **0.04351 (−0.02101)** | **0.05440 (−0.01002)** | **0.06382 (−0.00094)** | **0.05391 (−0.01066)** | **0.0889** |
| KNOWN_Q_ORACLE (ORIGINAL) ‡ | 0.04262 (−0.02189) | 0.05556 (−0.00886) | 0.06341 (−0.00135) | 0.05386 (−0.01070) | 0.0891 |
| KNOWN_Q_ORACLE (DUP2_4) ‡ | 0.04262 (−0.02189) | 0.05346 (−0.01096) | 0.06294 (−0.00182) | 0.05301 (−0.01156) | 0.0886 |

Gates for `RMF_U3E25` (DUP2_4), 95% paired router-seed intervals:

| Gate | VALIDATION | HOLDOUT | Requirement |
|---|---|---|---|
| G1 FLIP40 vs NOOP | −0.00091 [−0.00107, −0.00074] | **−0.00094 [−0.00109, −0.00079]**, 62/64 wins | upper < 0 |
| G2 CLEAN vs NOOP | −0.01957 [−0.02131, −0.01784] | **−0.02101 [−0.02260, −0.01942]**, 64/64 | upper < 0 |
| G3 CLEAN vs CLEAN_BAYES | +0.00089 (upper +0.00098) | **+0.00089 [+0.00083, +0.00095]** | upper ≤ 0.00416 |
| G4 ALL vs CGC_REFERENCE | −0.00257 (upper −0.00240) | **−0.00262 [−0.00277, −0.00247]** | upper ≤ 0.00149 |
| G5 online regret (ALL) | 0.0879 (upper 0.0898) | **0.0889 [0.0872, 0.0907]** | upper ≤ 0.1027 |
| G7 primary shifts (97.5%) | QUNIF −0.00723, BURST −0.00671 | **QUNIF −0.00803, BURST −0.00708** | one upper < 0 |
| G8 boundaries | 51 lab tests pass; 31 ≤ 64 scalars; no privileged input | same | |

The learner keeps **96%** of the strongest clean reference's clean-regime gain (CLEAN_BAYES, which is +0.0123
worse than NOOP under FLIP40) and beats NOOP under 40% flips in 62 of 64 holdout routers. Against the best
admissible baseline per regime: CLEAN +0.00089 vs CLEAN_BAYES; FLIP40 +0.00009 [+0.00001, +0.00018] vs
STATIC_ATTENUATION (which keeps only 48% of the clean gain); ALL −0.00262 vs CGC_REFERENCE.

### Shift and misspecification regimes (HOLDOUT, Δ vs NOOP, 97.5% intervals)

| Regime | RMF_U3 (ORIGINAL) | RMF_U3E25 (DUP2_4) |
|---|---|---|
| QUNIF (q ~ U(0, 0.5) per episode) | −0.00630 [−0.00692, −0.00567] useful | −0.00803 [−0.00879, −0.00728] useful |
| BURST (Markov bursts of pure noise) | −0.00606 [−0.00664, −0.00548] useful | −0.00708 [−0.00773, −0.00643] useful |
| FLIP30 (off-grid rate) | −0.00357 [−0.00397, −0.00317] useful | −0.00541 [−0.00590, −0.00491] useful |
| FLIP45 (rate outside the hypothesis grid) | +0.00347 [+0.00311, +0.00382] **harm** | +0.00126 [+0.00107, +0.00144] **harm** |
| ASYM40 (one-sided 1→0 flips) | −0.00655 [−0.00715, −0.00595] useful | −0.00776 [−0.00840, −0.00712] useful |
| CTXQ40 (context-dependent rate) | −0.00832 [−0.00920, −0.00744] useful | −0.00946 [−0.01042, −0.00850] useful |
| BIASWIDE (bias scale 1.6, clean) | −0.04548 [−0.04908, −0.04188] useful | −0.05264 [−0.05658, −0.04871] useful |
| BIASWIDE40 | −0.00488 [−0.00538, −0.00437] useful | −0.00654 [−0.00716, −0.00592] useful |
| ROUTERWIDE40 (router scale 1.0) | +0.00066 [+0.00052, +0.00080] **harm** | −0.00045 [−0.00055, −0.00035] useful |
| EVENT40 (readings share their corruption) | +0.00129 [+0.00111, +0.00146] **harm** | +0.00404 [+0.00374, +0.00434] **harm** |

VALIDATION reproduces every sign above.

## Mechanism: why it works, what it exploits, when it refuses to adapt, what limits it

**What the learner exploits.** A Bayes decision weighs every reliability hypothesis by its posterior and
acts on expected clean utility, so an update is effectively attenuated by the probability that the
evidence is reliable. Learning is never switched off by a threshold; it is weighted by evidence about the
channel.

**Why eight single readings are not enough.** With one reading per event, corruption is identifiable only
through the *shape* of the reward–logit relationship: flips compress success probabilities toward 0.5
(`P(y=1) = q + (1−2q)p`), whereas a bias shifts them in logit space. Each expert's unknown bias absorbs
most of that signal, so eight readings shared by three experts give a regime-identification AUC of only
**0.80** (DEV diagnostics; the historical ledger reports 0.81–0.82 at T8). Under 40% flips the mixture
posterior still gives the clean hypothesis about 0.23 of its mass (0.45 under clean data), and every
admissible rule sits on one Bayes frontier between clean learning and corrupted safety:

| ORIGINAL-contract point (DEV) | clean-gain retention | FLIP40 Δ vs NOOP |
|---|---:|---:|
| RMF_U3 (uniform prior) | 87–89% | +0.0012 to +0.0013 |
| RMF_Q2_P67 (prior 1:2 on {clean, 0.4}) | 79–81% | ≈ 0 |
| RMF_P67 | 73–75% | −0.0004 to −0.0007 |
| STATIC_ATTENUATION / reliability-gated CGC | 46–48% | −0.0010 |
| KNOWN_Q_ORACLE (privileged) | 100% | −0.0013 |

Priors restricted to {clean, q=0.4} are Bayes rules for every weighting of CLEAN and FLIP40 risk, so by
the complete-class argument they trace the best attainable trade-off for this information. Conservative
advantage thresholds, minimax regret, reliability-gated CGC proposals and information-directed or greedy
acquisition all landed on or inside it (RESEARCH_LOG.md). A learned estimator cannot beat the Bayes risk
under the design distribution, so family D was not built. Information-directed acquisition lowers online
regret by about 27% but raises the identification AUC only from 0.80 to 0.83. A fixed-log diagnostic
(identical NOOP action logs for every learner) reproduces the same FLIP40 deltas, so the failure is not
acquisition bias.

**Why a duplicate reading fixes it.** Two readings of the *same* event disagree with probability
`2q(1−q)`, whatever the bias, the context or the expert. The bias cancels exactly, so a duplicate measures
the channel directly. Regime-identification AUC: 0.80 (ORIGINAL), 0.89 (T16: 8 more expert queries),
**0.983 (DUP2_4: 4 more readings)**, 0.999 (DUP2). Information curve of the same learner (DEV; FLIP40 Δ vs
NOOP / clean retention): 0 duplicates +0.0015 / 88%, 2 → +0.0003 / 96%, **3 → −0.0002 / 99%**,
4 → −0.0006 / 99.5%, 8 → −0.0012 / 100%; whereas 4, 6 and 8 extra *queries* leave it at +0.0009, +0.0005
and +0.0002. The cheapest causal mechanism that resolves the decision-relevant uncertainty is therefore
3–4 independent re-readings of already selected events: no extra exploration, no extra online regret.

**When it refuses to adapt.** A disagreeing duplicate eliminates the clean hypothesis exactly (its
likelihood is zero), and the surviving corrupted hypotheses update each bias by roughly a third of what
the clean hypothesis would (linearised at p ≈ 0.5). Adaptation is then visible only where it changes an
expected-utility comparison. On HOLDOUT, the mean posterior P(clean | history) is 0.71 under CLEAN and
0.035 under FLIP40. The learner overrides NOOP's terminal decision on 23.7% of queries under CLEAN (the
known-q oracle: 27.3%) and on 13.0% under FLIP40 (oracle: 7.2%; the clean-model learner: 29%). Without
duplicates, `RMF_U3` overrides 19.8% under CLEAN and 16.2% under FLIP40: it cannot tell the two apart.

**What limits it (disclosed failure boundaries).**

- *Corruption outside the hypothesis support* (FLIP45): rates above the largest hypothesis (0.4) are
  read as partially informative, and the learner over-adapts (+0.0013). The hypothesis grid must
  cover the deployment's worst channel.
- *Correlated readings* (EVENT40): if both readings share their corruption, duplicates agree and the
  channel hypotheses mistake corrupted evidence for clean (+0.0040; posterior P(clean) stays at 0.48).
  The event hypothesis (prior 0.25 of 3.25) limits the damage relative to a clean learner (+0.0123) but
  does not remove it; removing it would bring back the original trade-off. The contract's independence
  assumption is essential and cannot be tested by duplicates alone.
- *Original contract:* information-limited as shown above; the frozen `RMF_U3` is +0.0011 vs NOOP under
  FLIP40 on HOLDOUT.

## Resources (measured, [`evidence/resources.json`](evidence/resources.json); not asserted)

| Arm | State scalars | State bytes (float32) | Updates / episode | Approx. flops per update | Update latency b=1 (numpy / scalar Python) |
|---|---:|---:|---:|---:|---:|
| NOOP / SGD / CGC_REFERENCE | 6 | 24 | 8 | — | 11–50 µs (numpy) |
| RMF_U3 (ORIGINAL) | 24 | 96 | 8 | ≈ 576 | 136 µs / 47 µs |
| **RMF_U3E25 (DUP2_4)** | **31** | **124** | 8 | ≈ 768 | 103 µs / 81 µs |

Terminal decision: ≈ 1.5 kflops per query (16 queries ≈ 25 kflops; 1.4 ms in pure Python). The scalar
reference (`scalar_rmf.py`) uses only fixed-size lists, fixed loops, no recursion and no allocation after
construction, and reproduces the vectorised learner exactly (actions, decisions, moments within 1e-10).
A native port is a few hundred lines of C with O(1) memory.

## Integration proposal (not implemented; needs separate authorisation)

The learner is a stateless coefficient set plus a 31-scalar volatile episode record:

```
frozen coefficients (immutable, host-owned): q grid (0, .2, .4, .4e), prior (1,1,1,.25), bias variance 1,
    16 Gauss-Hermite nodes and weights, STOP payoff
episode record (volatile, host-allocated, reset per episode): ℓ[4], μ[4][3], v[4][3], counts[3]

propose_action(t, base_logits[3]) -> expert           # proposal; the host admits or overrides
observe(expert, base_logit, readings[1 or 2])         # selected readings only
state_correction() -> s[3] = clip(Σ_m w_m μ_m, ±1.5)  # drop-in for ProbeEarlyEpisode.state
terminal_proposal(query_base_logits[3]) -> expert | STOP
```

1. **Attachment to the geometric operator.** The CGC keeps computing what it computes; the reliability
   filter replaces the CGC's *update policy*, not the operator. `state_correction()` has the reference
   episode's shape and cap, so the frozen `choose_probe_early` selector and `terminal_expert_probabilities`
   read it unchanged. The DEV evidence does not support letting CGC proposals drive the state: gating
   them by posterior reliability reduced to static attenuation (46% clean retention).
2. **Authority.** Every output is a proposal the host admits; the learner holds no handle to ECS state,
   RuntimeCore, files or the network. Its record is allocated by the host per episode and dropped at
   reset; nothing persists and no weight changes.
3. **Contract.** The host decides whether a second independent reading of a selected event is available
   (a re-query of the feedback channel, a second rater). Without it, the host must not deploy `RMF_U3E25`
   as qualified: the ORIGINAL contract is unqualified, and the DEV frontier says a safety-first prior
   (e.g. `RMF_P67`, DEV only) trades about 25% of the clean gain for FLIP40 ≤ NOOP.
4. **Deployment checks before any admission:** the worst plausible channel corruption must lie inside the
   hypothesis grid, and reading-level independence of corruption must hold or be audited independently.

## Handoff

Reproduce from the repository root (numpy only; seconds for tests, minutes for runs):

```sh
# mechanical and evidence suite (51 tests; the torch parity test skips unless torch is installed)
PYTHONPATH=research/cgc_lao/solver_lab/src:research/cgc_lao/solver_lab \
  python3 -B -m unittest discover -s research/cgc_lao/solver_lab/cgc_lao_lab_tests -t research/cgc_lao/solver_lab -v
```

```sh
# re-run a frozen VALIDATION or HOLDOUT command (from research/cgc_lao/solver_lab; see FROZEN.json "commands")
cd research/cgc_lao/solver_lab && PYTHONPATH=src python3 -m cgc_lao_lab.report HOLDOUT --partition HOLDOUT \
  --seeds 0:64 --episodes 4096 --contract DUP2_4 --arms RMF_U3E25,KNOWN_Q_ORACLE \
  --regimes CLEAN,FLIP20,FLIP40,QUNIF,FLIP30,FLIP45,BURST,ASYM40,CTXQ40,BIASWIDE,BIASWIDE40,ROUTERWIDE40,EVENT40 \
  --out /tmp/holdout_dup2_4.json
```

Other entry points: `python3 -m cgc_lao_lab.train_cgc`, `-m cgc_lao_lab.calibrate tune|levels`,
`-m cgc_lao_lab.diagnostics`, `-m cgc_lao_lab.resources`, `scripts/fixed_log.py`.

| Object | SHA-256 |
|---|---|
| Frozen lab source (`source_sha256`, all `src/cgc_lao_lab/*.py`) | `5fd2e9b28250d1f77e21cd719a7b4e5fb8df33d36cb39a5826976efa04364251` |
| `FROZEN.json` | `2038a36ea6a7b9bb08f1e8e211296410be4527619d4ead3d842b0d8effb66e74` |
| `specs/slb1.spec.json` | `ef965e1dddcf3269a1a32cd3730716569cb40293d5b6322dc5c27ea8da7ee122` |
| CGC coefficients (float32 weight bytes) | `95296ad5475fef3e32df9430abf7fc3816010165ff60210f50f6595075b0b742` |
| `evidence/holdout/dup2_4.json` | `82e8ffd7435a7312fc2766c1844717cdd5b87e6c4a85bf7e13a5db57d9215ad2` |
| `evidence/holdout/original.json` | `08185418c8af2a5c9df3e177d8e96b4c57660dbe762aaf6a5bd28257c64b5ecf` |
| `evidence/DISPOSITION.json` | `38c82722d003108a7dea9c5895acded9476a50b1416bf5caae9c3b30028f64ff` |

**Reproducibility contract.** Uniforms and every discrete potential outcome are bitwise identical across
platforms and numpy versions (integer counter RNG; checked under numpy 1.26.4 and 2.5.3). Gaussian draws
and regrets use libm `log`/`cos`/`exp`, so floating-point results can differ in the last bits across
machines; records are reproduced to about 1e-12 relative, and recomputed decisions are exact.

**Nonclaims.** Synthetic SLB1 only; not the Q4E5/Q4E14 simulator or its holdout (unopened, untouched);
no production, RALPH, runtime or native admission; no claim of Bayes optimality beyond the stated model;
no robustness claim outside the measured regimes; `ORIGINAL` remains unsolved at these gates.

**Next experiments.** (1) A hypothesis grid reaching 0.49 and an audited estimate of reading-level
correlation, to close the FLIP45 and EVENT40 boundaries before any deployment discussion. (2) Adaptive
duplicates: request a second reading only while the regime posterior is uncertain (expected < 4 extra
readings; the information curve suggests 3 suffice in-distribution). (3) A native C port of
`scalar_rmf.py` with the parity test as its oracle.
