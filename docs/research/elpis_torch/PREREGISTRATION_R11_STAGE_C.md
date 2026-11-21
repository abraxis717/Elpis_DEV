# Elpis Torch R11 — Stage C amendment: frozen-weight transfer into the actual ERSA stack

**Status.** FROZEN before any Stage C data.
- **The plan.** `research/elpis_torch/results/r11/amendment_stage_c.json` (SHA-256
  `ede13d3d982963fa341f4e46ccf9d2d6ad87670a61561b38bad8298e3dd51a20`).
- **What it belongs to.** It names the preregistration it amends (`a7df440a…`) and the Stage B analysis that
  triggered it (`study_r11.json`).
- **Integrity.** `run_r11_stage_c.frozen()` refuses a drifted plan. Every Stage C record carries every R11 source
  digest, the amendment's and the Stage C runner's.

**Why Stage C runs.** `PREREGISTRATION_R11.md` §5: Stage B found `EXPERT_SPECIALIZATION_QUALIFIED` for G2, the
fixed-gate bounded micro CGC. Stage B also found `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES`: without inner
updates, the meta-learned specialist is far worse than the FFN alone. Stage C asks the frozen-weight question in the
actual stack.

## 1. Question

Does the micro CGC specialist, trained END TO END inside the actual ERSA stack and then FROZEN, contribute to
in-context adaptation beyond the FFN alone and beyond a matched GELU specialist?
- The episode's support examples are in the sequence.
- There is no inner loop and no weight update at inference.

## 2. Harness

**The stack.** The repository's `ErsaBlock` modules, built and initialized by `ErsaLM` itself:
- d 64, 2 blocks;
- ERSA with 4 heads, 16 × 16 state, convolution 4;
- expert channel FFN 86 + specialist.

They sit between a linear input map (9 → 64) and a linear readout (64 → 3). That I/O is a disclosed harness
difference from the byte language model.

**The sequence.**
- `k` support tokens `[x, y, 0]`, then 48 query tokens `[x, 0, 1]`.
- The loss is on the queries.
- `k` is drawn per step from {0, 1, 2, 4, 8, 16, 32}.
- Namespace `R11C`; seeds 301–306 × both families: 12 groups, 12 test episodes each.

**The arms.**

| Arm | Expert specialist (one expert per block, fixed assignment) |
|---|---|
| G0 | none (FFN only) |
| G1 | micro CGC, learned scalar gate (init 0.1) |
| G2 | micro CGC, fixed `g = 1` |
| G3 | the matched GELU specialist, `g = 1` |

**The settings.**
- **Profile.** `BOUNDED_SPECIALIST` (Stage B's).
- **Branch initialization.** The A1 rule.
- **Training.** Adam 3e-3, cosine to 10%, 2,000 steps of 16 episodes.
- **Selection.** Per-arm checkpoint selection on DEV episodes at k = 32, every 250 steps.

**Routing test.** G2 with two experts per block and a learned top-1 router, trained on mixed families. The test
measures the share of each family's tokens sent to that family's majority expert.

## 3. Decision rules (frozen)

**"Better"** is as in `PREREGISTRATION_R11.md` §6: pooled ratio ≤ 0.95, one-sided sign test p < 0.05, and at least
2/3 of groups below 1.

| Disposition | Rule |
|---|---|
| `ASSAY_NOT_LEARNED` | G0's median k = 32 query MSE exceeds 0.5 × the zero predictor's in some group |
| **`FROZEN_INFERENCE_TRANSFER_QUALIFIED`** (G1, G2 separately) | Better than G0 AND better than G3 at k = 32, AND ablating its specialist in every block at k = 32 raises the loss ≥ 1.10× |
| `NOT_ESTABLISHED_FROZEN_SPECIALIST_WORSE_THAN_FFN` | G0 better than the arm at k = 32 |
| `NOT_ESTABLISHED` | otherwise |
| `LEARNED_ROUTING_SEPARATES_FAMILIES` | In every block of every seed the families' majority experts differ, and mean purity ≥ 0.8 |
| `INTEGRATION_GAP_LEARNED_ROUTING` | otherwise |

**Attribution.**
- The ERSA reset after the support prefix and the per-block specialist ablations are reported for every arm.
- In-context information is ERSA's. The specialist is credited only with the arm-against-G0 difference.
- Ablating a trained branch measures RELIANCE. It is never read as benefit without the comparison against G0.

## 4. Development before this amendment (labelled DEVELOPMENT)

**What was run.** Namespace `R11C_DEV`, seed 31:
- `results/r11/dev_c_smooth_s31.json` (400 steps);
- an ad hoc 2,000-step check (conditional, not kept as a file).

**What it showed.**
- Every arm learns in context.
- An ERSA reset removes the gain.
- At 2,000 steps, G0 kept improving while G2 plateaued.

**What it fixed.** The training length and selection cadence, the same for every arm.
