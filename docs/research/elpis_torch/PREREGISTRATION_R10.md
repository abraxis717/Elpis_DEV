# Elpis Torch R10 — preregistration: CGC complementarity and in-context adaptation

**Status.** FROZEN before any confirmatory or screening run. The machine-readable twin is
`research/elpis_torch/results/r10/preregistration.json` (SHA-256 `9d0bc46d9ea684f22ee60e7dc0904dc73637176e17944ca6f14c78a8ab389e2c`).
- Every run record carries the SHA-256 of that file, of `run_r10.py` and of `adaptation_tasks.py`.
- `run_r10 combine` refuses any record whose digests differ.
- The only runs before this freeze are the PILOT runs (§9). They are labelled and kept, and they used no CGC arm.

**Base.** `main` at `eac4611f35eaa8a3f0ac9864bddacd4e2a9d98da`, with R9 post-merge CI run 44609229785 at 16/16.
Branch `claude/r10-cgc-adaptation`.

## 1. Question and hypotheses

> Does a CGC-augmented Elpis transformer adapt to new in-context relationships and retain earlier conflicting
> relationships more reliably than otherwise equivalent FFN-only transformers, without inference-time weight
> updates?

- **H1, rapid adaptation.** After a few in-context examples of an unseen mapping or rule, arm A's exact-answer
  accuracy rises faster than that of the matched FFN-only arms C and C_mac.
- **H2, interference resistance.** After conflicting regimes, arm A keeps more of its accuracy on an earlier regime
  (named explicitly) than C and C_mac.
- **H3, resource efficiency.** Any gain is set against parameters, active compute, state and measured latency. A
  gain bought only with materially more compute is not an efficiency advantage.

**Falsifiers.** The hypothesis fails to establish practical value if:
- A does not beat C and C_mac on adaptation;
- A does not reduce forgetting;
- a gain vanishes without attention (the D pair) or with ERSA reset;
- a gain appears only against the unmatched B;
- a gain does not survive the held-out variants;
- a gain is smaller than the minimum useful effect;
- a gain would need inference-time gradient updates (none are performed; tested).

A zero or negative result is valid.

## 2. Assay, arms, training

The assay is `adaptation_tasks.py`. Its spec digest is ``d6fae2b6ac77241764b3d98e539d97d8c8685b45247d1f7060f9260d52dda493``. Families A, B and C, the variants and the
held-out variants are as in `DESIGN_R10.md` §2.

The arms are A, B, C, C_mac, D_A and D_C, as in `DESIGN_R10.md` §3. The arm table is frozen in the JSON and checked
by `run_r10.frozen()`.

The training is R9's schedule, `3,000` steps, with batch 16, 256-token episodes, 32-token chunks, the loss on
answers only, and the fixed mix of 6 A, 5 B and 5 C episodes per batch (`DESIGN_R10.md` §4).

## 3. Seeds and partitions

| Stage | Seeds | Arms | Evaluation split seeds (main, held-out, diagnostic, dev) |
|---|---|---|---|
| Screening | 0, 1, 2 | A, B, C, C_mac, D_A, D_C | 7001, 7002, 7003, 7000 |
| Confirmation (§6) | 100–111 | A, C, C_mac, D_A, D_C | 7101, 7102, 7103, 7100 |

**Partitions.**
- Training episodes come from `train_episode(seed, step, row)`.
- Evaluation episodes come from `eval_set(variant, n, split_seed)`.
- The split seeds are disjoint, and every run records its train/test signature overlap, which must be 0.

## 4. Endpoints

**Primary**, from each run's main sets:

| Endpoint | Definition |
|---|---|
| **ADAPT** | The mean, over families A and B, of the mean exact-answer accuracy at k = 1, 2, 4, 8 and 16: a discrete area under the adaptation curve. k = 0 is unknowable by construction, so it is reported but not averaged. |
| **RETAIN** | The mean, over `C_short`, `C_long` and `C_gap`, of post-interference accuracy on the earliest regime. It is reported with the pre-interference accuracy (RETAIN_pre) and the change (RETAIN − RETAIN_pre). |

**Secondary:**
- each family's curve, and each C variant;
- A: inside versus outside the window, collision errors, recency errors;
- B: switch, re-entry and recovery accuracy;
- held-out endpoints:
  - ADAPT on `B_heldout`;
  - RETAIN, as the mean of post accuracy on `C_gap128` and `C_4reg`;
- bits per answer;
- every diagnostic;
- accounting and measured latency.

## 5. Decision rules

**Constants:**

| Constant | Value |
|---|---|
| δ for ADAPT | 0.03 |
| δ for RETAIN | 0.05 |
| Learned threshold, ADAPT | 0.30 (chance 0.05) |
| Learned threshold, RETAIN | RETAIN ≥ 0.20 (chance 0.05, recency 0) and RETAIN_pre ≥ 0.40 |

**Each endpoint is judged separately**, on paired differences across seeds. A comparison X − Y on a seed is X's
endpoint minus Y's.

1. **ASSAY_NOT_LEARNED.** The best seed mean over the primary arms (A, B, C, C_mac) must reach the learned
   threshold, or the endpoint's verdict is ASSAY_NOT_LEARNED:
   - ADAPT: the best arm's ADAPT ≥ 0.30.
   - RETAIN: the best arm's RETAIN ≥ 0.20, and the best arm's RETAIN_pre ≥ 0.40. The bindings must have been
     learned, and retention itself must be off the floor in some arm.

   A is one of the primary arms, so a CGC gain that lifts retention off the floor is never masked by this rule
   (tested). ASSAY_NOT_LEARNED is a failure of the assay, not a falsification of CGC; it also establishes no CGC
   benefit.
2. **NONQUAL.** A missing or failed run, or a digest mismatch, makes the stage NONQUAL for that endpoint.
3. **Screening.** The result is **SCREEN_FAVORABLE** only if both hold:
   - mean(A − C) ≥ δ, with at least 2 of 3 seeds positive;
   - mean(A − C_mac) ≥ δ, with at least 2 of 3 seeds positive.

   Otherwise it is **SCREEN_NOT_FAVORABLE**. Screening alone never supports a positive claim.
4. **Confirmation** runs only after a SCREEN_FAVORABLE, and only for that endpoint (§6). It is **CONFIRMED** only if
   both hold:
   - mean(A − C) ≥ δ with one-sided paired t ≥ 1.796 (df 11, α 0.05);
   - mean(A − C_mac) ≥ δ with the same test.

   **CONFIRMED_IN_DISTRIBUTION_ONLY** applies if A − C is not positive on the held-out endpoint. Otherwise the
   result is **NOT_CONFIRMED**.
5. **Attribution of a confirmed gain.** It is PERSISTS_WITHOUT_ATTENTION if mean(D_A − D_C) ≥ δ/2, else
   ATTENTION_DEPENDENT. The ERSA-reset diagnostic is reported beside it, for both arms.
6. **Efficiency.** A/C decode latency p50 above 1.10 is MATERIAL: a confirmed gain then carries
   QUALITY_LATENCY_TRADEOFF.

**Overall verdicts** (computed by `run_r10.overall`, tested):

| Stage verdicts | Overall verdict for the endpoint |
|---|---|
| ASSAY_NOT_LEARNED or NONQUAL at screening | that verdict |
| SCREEN_NOT_FAVORABLE | NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED (or `_RETENTION_`) |
| SCREEN_FAVORABLE, confirmation missing or fewer than 12 complete seed blocks | PARTIAL: no positive claim |
| NOT_CONFIRMED | NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED (or `_RETENTION_`) |
| CONFIRMED or CONFIRMED_IN_DISTRIBUTION_ONLY | CGC_ADAPTATION_BENEFIT_CONFIRMED (or `_RETENTION_`), with its held-out status, attribution and latency qualifier |

## 6. Confirmation and budget

Confirmation uses fresh seeds 100–111 (12 paired seeds) for arms A, C, C_mac, D_A and D_C. Its evaluation sets are
generated from fresh split seeds: new mappings, rules and episodes.
- **Order.** Seeds run in order, every arm of a seed before the next seed.
- **Budget.** Screening has 4 wall-clock hours. Confirmation has 10 hours on this 4-core VM, with two workers of two
  threads each.
- **If the budget runs out.** The analysis uses only complete seed blocks, and fewer than 12 blocks means
  **PARTIAL**: no positive claim.
- **No added seeds.** Seeds are never added after results are seen.

## 7. Negative controls (diagnostics) and their expected outcomes

- **Shuffled support targets** (A, B). Expected: accuracy within chance + 0.05 in every arm. Above that suggests
  leakage, and is reported as such.
- **Regime identity removed.** Expected: conflicting bindings collapse toward a regime guess (A, C), and family B
  loses its rule.
- **ERSA reset and attention reset** before the C probes. Each states which carrier held the earlier regime.
- **CGC zeroed** (A, D_A). This is reliance, not benefit.
- **Irrelevant demonstrations and the conflicting-recent trap.** These test robustness.

## 8. Stopping rules and freedoms not taken

The following are not allowed:
- no rerun of a confirmatory or screening run;
- no added seeds;
- no threshold or endpoint changed after results;
- no larger model, no width sweep and no architectural change to rescue a null result;
- no inference-time gradient update.

A failed run is recorded as failed.

The scarcity sweep (FFN widths 8–256) is not part of R10. It is justified for a later phase only if the assay is
learned AND some endpoint is SCREEN_FAVORABLE or CONFIRMED.

## 9. Pilot runs before the freeze (kept, labelled PILOT)

The pilot calibrated only the assay and the step count. It used:
- arms B and D_C (no CGC arm);
- development seed 900;
- development split seeds 9100–9103.

**Records** (`results/r10/pilot_*.json`; the 1,500-step pair predates the reserved split):

| Record | Steps | ADAPT | A k=1 | A k=16 | B k=1 | RETAIN_pre | RETAIN | C post answers that are a conflicting binding | B_heldout adapt | Train/test overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| B, 1,500 | 1,500 | 0.538 | 0.124 | 0.799 | 0.457 | 0.442 | 0.065 | 0.15–0.50 | 0.338 | 5 |
| D_C, 1,500 | 1,500 | 0.532 | 0.162 | 0.598 | 0.538 | 0.522 | 0.089 | 0.35–0.51 | 0.152 | 5 |
| B, 3,000 | 3,000 | 0.745 | 0.302 | 0.959 | 0.935 | 0.920 | 0.043 | 0.85–0.97 | 0.082 | 0 |
| D_C, 3,000 | 3,000 | 0.657 | 0.229 | 0.937 | 0.533 | 0.835 | 0.044 | 0.83–0.91 | 0.072 | 0 |

**What the pilot changed, all before the freeze:**
- **The reserved split.** The 1,500-step pair found 5 evaluation episodes whose signature also occurred in
  training (family B's rule space is small). Evaluation episodes now come from a signature class that training
  never draws (`adaptation_tasks.reserved`). The 3,000-step pair has overlap 0.
- **The step count.** The rule was written down (2026-11-20T14:04:45Z) before the 3,000-step records existed:
  3,000 steps if family-A k=1 accuracy or RETAIN rose by at least 0.05, or ADAPT by at least 0.03, and the measured
  step times fit the budgets. Family-A k=1 rose from 0.124 to 0.302 and ADAPT from 0.538 to 0.745. Step times
  (0.34 s for a window arm, 0.28 s for an attention-free arm, two at a time) project screening at about 2.5 hours and
  confirmation at about 8.3 hours. So R10 trains for **3,000 steps**.

**What the pilot shows, and how it bears on the endpoints:**
- **Retention is at the floor in both CGC-negative arms.** They learn the bindings (RETAIN_pre 0.92 and 0.84).
  After interference, though, 83–97% of their answers are another regime's binding of the same key, and 64–88% are
  the latest one. RETAIN is 0.04: chance is 0.05, and a guesser with no memory of the first regime that avoids the
  conflicting values scores at most 1/18 ≈ 0.056.
- **The RETAIN learned rule was revised after the 1,500-step pair and before the 3,000-step pair.** The first draft
  (RETAIN_pre ≥ 0.50 alone) checked only that the bindings were learned, so a floor in post-interference retention
  would have been read as a CGC comparison. The 3,000-step pair did not change the rule. RETAIN is likely to be
  ASSAY_NOT_LEARNED, unless arm A itself lifts retention off the floor; the rule would detect that.
- **In-distribution family B is rule identification.** At 3,000 steps one example identifies a trained shift (k=1
  0.935 with attention), but unseen shifts stay near chance (B_heldout 0.08 and 0.07). The held-out endpoint is
  the test of learning a new rule, and §5 reports a confirmed ADAPT gain as in-distribution only if it does not
  survive there.

No CGC-positive model was trained or evaluated before this freeze.
