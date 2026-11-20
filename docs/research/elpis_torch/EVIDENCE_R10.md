# Elpis Torch R10 — evidence: CGC complementarity and in-context adaptation

**Principal question.** Does a CGC-augmented Elpis transformer adapt to new in-context relationships, and retain
earlier conflicting relationships, more reliably than otherwise equivalent FFN-only transformers, without
inference-time weight updates?

**Answer.** No CGC benefit was established, for either capability.
- **Adaptation (H1): `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`.** The assay was learned, and the CGC
  arms adapted worse than their FFN-only counterparts:
  - A − C: −0.143 ADAPT;
  - A − C_mac: −0.155;
  - the attention-free pair, D_A − D_C: −0.054, on every seed.
- **Retention (H2): `ASSAY_NOT_LEARNED`.** No arm, with or without CGC, kept the earlier bindings through conflicting
  interference above the preregistered floor. What retention exists is carried by attention (inside the window)
  or by the ERSA state (behind a gap), never by a CGC-specific path.
- **Cost (H3).** CGC costs 10% decode latency (A/C 1.101; D_A/D_C 1.104) and about 22% training time. There is no
  gain to set against either.
- **Confirmation.** Under the preregistered rules a confirmation stage runs only after a favorable screening, so
  none ran. No width sweep is justified.

R10 is a falsification phase. It adds a task generator and a runner, and changes no model code and nothing under
`src/` or `native/`.
- Plan: `PREREGISTRATION_R10.md`, frozen at `34cdbf5` before any screening run.
- Design: `DESIGN_R10.md`.
- Ledger: `CLAIM_AUDIT_R9.md`, sections "R10 additions" and "R10 result".
- Performance handoff: `R11_PERFORMANCE_NOTE.md`.

## 0. Verdicts, by kind

| Kind | Verdict | Where |
|---|---|---|
| H1, adaptation (primary ADAPT) | **`NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`** (learned; SCREEN_NOT_FAVORABLE; FFN arms better against C_mac, B and in the attention-free pair) | §4 |
| H2, retention (primary RETAIN) | **`ASSAY_NOT_LEARNED`**: best primary arm 0.173 < 0.20. No CGC retention benefit is established | §5 |
| H3, resource efficiency | No gain to set against CGC's cost: +10% decode latency, about +22% training time, +60 MiB peak training memory | §7 |
| Confirmation (12 fresh paired seeds) | NOT_RUN: preregistered only after a SCREEN_FAVORABLE endpoint | §2 |
| Attribution | In-window retention is attention's (an attention reset removes it). Retention behind a gap is ERSA's (an ERSA reset removes it). CGC's evaluation-time contribution is near zero | §6 |
| Integrity controls | All hold in every run: overlap 0, weights unchanged by evaluation, shuffled targets at chance, frozen sets identical | §3 |
| Scarcity (width) sweep | NOT JUSTIFIED (P6 requires a valid and favorable assay) | §10 |
| R9's `CGC_NOT_ESTABLISHED` | Unchanged; now also measured for in-context adaptation at this scale | `CLAIM_AUDIT_R9.md` |
| Gate A | HOLD. No K1-to-HCA interface was built | §10 |
| GPU | CUDA NOT_RUN (no GPU in this VM) | — |

## 1. Provenance

| Item | Value |
|---|---|
| Base | `main` at `eac4611f35eaa8a3f0ac9864bddacd4e2a9d98da` (R9, PR #79), post-merge CI run 44609229785 at 16/16 |
| Branch | `claude/r10-cgc-adaptation`, created from that commit |
| Assay and runner | `4767c80`, then `999b9c2` (reserved split, overall-verdict mapping), both before the freeze |
| Freeze | `34cdbf5`: `results/r10/preregistration.json`, SHA-256 `9d0bc46d9ea684f22ee60e7dc0904dc73637176e17944ca6f14c78a8ab389e2c` |
| Runner | `experiments/run_r10.py`, SHA-256 `c220bc54dac69563d5e3b446907b9891e9926f994292b2109be94aa0a4487e85` |
| Task generator | `adaptation_tasks.py`, SHA-256 `204498f65198ddf5428fccfb5309c93d6ce89a02084f49b204f9b2314ec5dadb` |
| Task spec digest | `d6fae2b6ac77241764b3d98e539d97d8c8685b45247d1f7060f9260d52dda493` |
| Environment | 4-core x86-64 VM (AVX-512), 16 GiB; Python 3.13.18; PyTorch 2.15.0+cpu; no GPU; no PMU |
| Execution | two runs at a time, two threads each. Decode latency on one thread, with the machine otherwise idle |

Every run record carries the three SHA-256 digests and the task-spec digest. `run_r10 combine` refused no record,
and the record tests recompute the digests. The 18 records name 13 git revisions, because records were committed as
they finished. Every one of those revisions holds the frozen preregistration byte for byte (tested).

## 2. What ran

| Stage | Runs | Wall clock | Budget |
|---|---|---|---|
| Pilot (before the freeze; arms B and D_C only, development seed 900, development episodes) | 4 | — | — |
| Decode latency (all six arms, idle machine) | 1 | 5 s | — |
| Screening: arms A, B, C, C_mac, D_A, D_C × seeds 0, 1, 2 | 18, all COMPLETED | 14:29:34–18:00:27 UTC (3 h 31 min) | 4 h |
| Confirmation: seeds 100–111 | 0 | — | NOT_RUN: no endpoint screened favorable |

- **The study.** Every screening run trained exactly 3,000 steps, on episodes from that seed's own stream. It was
  then evaluated on the frozen screening sets: main 256 episodes per variant, held-out 256, diagnostic 128. The
  set digests are identical in all 18 records.
- **Interruptions.** Four runs were killed before finishing, by a VM restart and by a background-task time limit.
  None had left any output, and each was restarted from scratch (§9). No completed run was executed twice.
- **The analysis.** `run_r10 combine` (`results/r10/study_cgc_adaptation.json`) applied the frozen rules
  mechanically. It refused nothing and found all 18 runs.

## 3. Assay validity and integrity

**Reference points.**
- **Chance** is 1/20 = 0.05.
- **The oracle ceiling** comes from an exact memory reader, executed in the tests. It is 1.0 on every probe whose
  answer the context determines: every k ≥ 1 probe and every post-interference probe. The fraction of
  determined answers is:
  - 0.833 in family A, where first exposures are unknowable by construction;
  - 0.952 in B;
  - 0.685 in `C_short`;
  - 0.764 in `C_long`, `C_gap` and `C_gap128`;
  - 0.749 in `C_4reg`;
  - 0.962 in `B_heldout`.
- **The trivial recency strategy** scores 0 on every post-interference probe, by construction.

**Learned floors** (preregistered, §5 rule 1):

| Endpoint | Best primary-arm seed mean | Floor | Learned |
|---|---|---|---|
| ADAPT | 0.800 (B) | 0.30 | yes |
| RETAIN | 0.173 (B); RETAIN_pre 0.979 (C_mac) | RETAIN ≥ 0.20 and RETAIN_pre ≥ 0.40 | **no**: the bindings are learned (pre ≥ 0.93 in the window arms), retention after interference is not |

**Integrity, in every one of the 18 records:**
- train/test signature overlap 0;
- parameters byte-identical before and after evaluation and every diagnostic, so no inference-time update;
- exactly 3,000 steps;
- the frozen evaluation-set digests identical.

The independent-session, reset and restart equivalence, causality, chunking and exact-CGC-removal controls are
tests (`test_r10.py`).

**Negative controls** (diagnostic sets; mean over seeds):

| Arm | A, shuffled support targets | B, shuffled | A, regime identity removed (intact) | B, regime identity removed (intact) |
|---|---|---|---|---|
| A | 0.048 | 0.058 | 0.461 (0.479) | 0.494 (0.657) |
| B | 0.049 | 0.054 | 0.439 (0.599) | 0.341 (0.816) |
| C | 0.050 | 0.056 | 0.439 (0.535) | 0.546 (0.835) |
| C_mac | 0.047 | 0.056 | 0.467 (0.564) | 0.551 (0.829) |
| D_A | 0.045 | 0.053 | 0.332 (0.424) | 0.341 (0.828) |
| D_C | 0.048 | 0.056 | 0.445 (0.504) | 0.654 (0.836) |

- **Shuffled targets leave chance.** With every revealed value replaced by a random one, accuracy is chance in
  every run (maximum 0.065). Family B's ceiling here is 1/15, the prior over the 15 trained shifts. Nothing leaks.
- **Regime identity matters.** Removing it costs every arm accuracy, most in family B.
- **The robustness diagnostics change nothing.** With irrelevant demonstrations inserted, or the latest
  conflicting binding re-shown before each probe, `C_long` post-interference accuracy stays at its floor
  (0.01–0.11).

## 4. Adaptation (H1)

**Accuracy at k** (earlier exposures of the queried binding or rule; main sets; mean over seeds 0–2):

| Arm | Family A k=0 / 1 / 2 / 4 / 8 / 16 | A AUC | Family B k=0 / 1 / 2 / 4 / 8 / 16 | B AUC | ADAPT (mean ± sd) |
|---|---|---|---|---|---|
| A | 0.020 / 0.310 / 0.380 / 0.531 / 0.746 / 0.884 | 0.570 | 0.031 / 0.639 / 0.688 / 0.699 / 0.693 / 0.703 | 0.684 | 0.627 ± 0.209 |
| B | 0.027 / 0.477 / 0.557 / 0.713 / 0.879 / 0.972 | 0.720 | 0.030 / 0.892 / 0.937 / 0.868 / 0.841 / 0.861 | 0.880 | 0.800 ± 0.080 |
| C | 0.027 / 0.413 / 0.484 / 0.614 / 0.786 / 0.895 | 0.638 | 0.020 / 0.914 / 0.966 / 0.892 / 0.859 / 0.880 | 0.902 | 0.770 ± 0.095 |
| C_mac | 0.024 / 0.414 / 0.506 / 0.668 / 0.838 / 0.941 | 0.673 | 0.028 / 0.895 / 0.955 / 0.886 / 0.849 / 0.875 | 0.892 | 0.783 ± 0.059 |
| D_A | 0.029 / 0.243 / 0.311 / 0.475 / 0.679 / 0.793 | 0.500 | 0.020 / 0.911 / 0.954 / 0.883 / 0.855 / 0.875 | 0.895 | 0.698 ± 0.029 |
| D_C | 0.018 / 0.319 / 0.413 / 0.599 / 0.788 / 0.884 | 0.600 | 0.026 / 0.924 / 0.956 / 0.888 / 0.862 / 0.883 | 0.903 | 0.752 ± 0.034 |

k = 0 is unknowable by construction (chance, or below it when a model guesses from context), and is not averaged.

**Paired differences** (the preregistered comparisons; the class uses δ = 0.03):

| Comparison | Seed 0 | Seed 1 | Seed 2 | Mean | Positive | Class |
|---|---|---|---|---|---|---|
| A − C | −0.142 | −0.305 | +0.018 | −0.143 | 1/3 | NOT_DISTINGUISHED |
| A − C_mac | −0.014 | −0.363 | −0.090 | −0.155 | 0/3 | X_BETTER |
| A − B | −0.017 | −0.370 | −0.131 | −0.173 | 0/3 | X_BETTER |
| D_A − D_C | −0.032 | −0.067 | −0.063 | −0.054 | 0/3 | X_BETTER |

Screening needs mean ≥ +0.03 with 2 of 3 seeds positive against both C and C_mac. The result is
**SCREEN_NOT_FAVORABLE**, which maps to `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`.

**Where the deficit is.**
- **Family A, arbitrary new bindings.** The CGC arms' AUC is below the comparators':

  | Comparison | Per seed | Mean |
  |---|---|---|
  | D_A − D_C | −0.046, −0.143, −0.112 | — |
  | A − C_mac | every seed negative | −0.103 |
  | A − B | every seed negative | −0.150 |
  | A − C | −0.298, +0.068, +0.025 | −0.068 |

- **Family B, trained shift rules.** The arms are equal except one run. Arm A's seed-1 model never learned to
  identify the shift from examples (k=1 0.065, against 0.81–0.98 in every other run). That single run is 71% of
  the summed A − C ADAPT deficit.
- **Evidence beyond the window.** Family-A accuracy falls from 0.57–0.81 inside the 64-token window to 0.03–0.12
  outside it. The attention-free arms hold more beyond it (0.09–0.12) than the window arms (0.03–0.05).
- **Generalization to unseen shifts** (`B_heldout`) stays near chance in most runs (§8).

## 5. Retention (H2)

**Post-interference accuracy on the earliest regime** (main sets; mean over seeds; pre-interference in brackets):

| Arm | `C_short`: evidence inside the window | `C_long`: outside it | `C_gap`: outside, + 64-token gap | RETAIN | Held-out `C_gap128` | Held-out `C_4reg` |
|---|---|---|---|---|---|---|
| A | 0.087 (0.939) | 0.041 (0.917) | 0.241 (0.928) | 0.123 | 0.227 | 0.032 |
| B | 0.444 (0.922) | 0.023 (0.932) | 0.051 (0.941) | 0.173 | 0.053 | 0.020 |
| C | 0.395 (0.886) | 0.031 (0.861) | 0.083 (0.866) | 0.170 | 0.078 | 0.027 |
| C_mac | 0.376 (0.973) | 0.015 (0.979) | 0.076 (0.985) | 0.156 | 0.066 | 0.011 |
| D_A | 0.035 (0.841) | 0.049 (0.751) | 0.093 (0.760) | 0.059 | 0.109 | 0.037 |
| D_C | 0.038 (0.937) | 0.066 (0.915) | 0.179 (0.913) | 0.094 | 0.198 | 0.048 |

- **Every arm learns the bindings, and none keeps them through interference.** Pre-interference accuracy is
  0.75–0.99. After two conflicting regimes, most answers are another regime's binding of the same key, usually the
  latest one:
  - `C_long`: 57–89% of post answers are a conflicting binding;
  - `C_short`, in the arms without in-window retrieval: up to 95%.
- **Inside the window, retention is all-or-nothing per run.** It appears when a run acquires in-window,
  regime-conditioned retrieval; §6 shows that is attention:

  | Run | C_short post |
  |---|---|
  | C seed 0 | 0.963 |
  | B seed 2 | 0.979 |
  | C_mac seed 2 | 0.816 |
  | B seed 1 | 0.348 |
  | C_mac seed 1 | 0.311 |
  | Each of A's three runs | at most 0.133 |
  | Attention-free arms | floor |

- **Outside the window, retention is at the floor except behind a gap.** With a distractor gap, A (0.241) and D_C
  (0.179) keep some bindings. §6 shows the ERSA state carries this, and §8 shows why it cannot be credited to CGC.
- **Three interfering regimes** (`C_4reg`) leave every arm at the floor.

**Paired RETAIN differences.** All four comparisons are NOT_DISTINGUISHED:

| Comparison | Mean |
|---|---|
| A − C | −0.046 |
| A − C_mac | −0.033 |
| A − B | −0.050 |
| D_A − D_C | −0.035 |

The endpoint is ASSAY_NOT_LEARNED, so no screening class applies.

## 6. Attribution: attention, ERSA and CGC

**State resets before the C probes** (post-interference accuracy; diagnostic sets; intact on the main sets):

| Arm | `C_short` intact / ERSA reset / attention reset | `C_gap` intact / ERSA reset / attention reset |
|---|---|---|
| A | 0.087 / 0.111 / 0.115 | 0.241 / 0.037 / 0.158 |
| B | 0.444 / 0.445 / 0.033 | 0.051 / 0.057 / 0.050 |
| C | 0.395 / 0.387 / 0.070 | 0.083 / 0.042 / 0.055 |
| C_mac | 0.376 / 0.353 / 0.021 | 0.076 / 0.047 / 0.035 |
| D_A | 0.035 / 0.034 / — | 0.093 / 0.049 / — |
| D_C | 0.038 / 0.046 / — | 0.179 / 0.033 / — |

- **In-window retention is attention's.**
  - Resetting the attention history before the probes removes it in every FFN window arm, to 0.02–0.07.
  - Resetting the ERSA state leaves it intact.
- **Retention behind a gap is ERSA's.**
  - Resetting the ERSA state removes it in both arms that have it: A, 0.241 → 0.037; D_C, 0.179 → 0.033.
- **CGC itself contributes almost nothing at evaluation.**
  - Zeroing the CGC gates of the trained A and D_A models changes little in any run. That comparison is across
    two independent frozen sets from the same generator:
    - no adaptation AUC by more than 0.025;
    - no family accuracy by more than 0.028;
    - no post-interference accuracy by more than 0.023.
  - The final gate magnitudes average 0.05–0.08; they were initialized at 0.1.
  - So the CGC arms' lower adaptation is not CGC computing something worse at inference. Training with the CGC path
    present produced networks that adapt worse.
- **The attention-free pair isolates ERSA from attention.**
  - D_A − D_C is negative on ADAPT on every seed (−0.054).
  - On RETAIN it is NOT_DISTINGUISHED (−0.035).
  - So without attention, CGC neither helps ERSA bind new relationships nor helps it keep old ones.

## 7. Costs (H3): parameters, compute, state, memory, latency

**Exact accounting** (every record; identical across seeds):

| Arm | Parameters | Active per token | Channel multiply-adds per token | Routed expert | State bytes per stream | Matmul FLOPs per token (measured) |
|---|---|---|---|---|---|---|
| A | 4,660,152 | 180,444 | 98,048 | FFN 86 + CGC (24,256 MACs) | 42,772 | 439,936 |
| B | 4,545,448 | 151,768 | 67,072 | FFN 86 (16,512) | 42,772 | 382,592 |
| C | 4,659,112 | 180,184 | 95,488 | FFN 123 (23,616) | 42,772 | 439,424 |
| C_mac | 4,668,328 | 182,488 | 97,792 | FFN 126 (24,192) | 42,772 | 444,032 |
| D_A | 4,635,576 | 155,868 | 98,048 | FFN 86 + CGC (24,256) | 9,744 | 347,776 |
| D_C | 4,634,536 | 155,608 | 95,488 | FFN 123 (23,616) | 9,744 | 347,264 |

- **The PLE table dominates.** It has 4,195,600 entries (16.8 MB in float32) in every arm, 90% of each model.
- **State.** ERSA cells take 8,192 bytes, the convolution tails 1,536, and the PLE history 16. The window arms add
  32,768 bytes of attention KV and 260 of token history. No state is written anywhere persistent.

**Measured costs.**

| Measure | A against C | D_A against D_C | How it was measured |
|---|---|---|---|
| Decode latency p50, one thread | 4.894 vs 4.446 ms (×1.101) | 3.952 vs 3.581 ms (×1.104) | idle machine, interleaved, 96 tokens after a 128-token prefill, seed-0 weights (`results/r10/latency.json`) |
| Training time per run | ×1.25 / 1.21 / 1.22 by seed | ×1.02 / 1.37 / 1.34 | two runs shared the VM and the pairing varied, so these are resource figures, not a speed comparison |
| Peak training RSS | 773 vs 714 MiB | 738 vs 673 MiB | per run |

- **Where CGC's cost is.** Against C, CGC adds 0.1% matmul FLOPs and 2.7% channel multiply-adds, yet 10% latency.
  The cost is its elementwise cubic work and extra kernels, which a multiply-add count does not see.
- **Latency is material, but there is nothing to trade it against.** The preregistered rule marks an A/C latency
  ratio above 1.10 as MATERIAL, and 1.101 is above it. A QUALITY_LATENCY_TRADEOFF needs a confirmed gain, and
  there is none.

## 8. Exploratory observations (not preregistered; no claim)

The patterns below came from looking at the records after the analysis. They do not change any verdict. They are
listed so that a future preregistered test can target them, and so that none of them is mistaken for evidence.

1. **Retention behind a gap, in the window arms.**
   - **The pattern.** A's post-interference accuracy behind the 64-token gap exceeds that of every FFN window arm
     on every seed. Against C:

     | Variant | Seed 0 | Seed 1 | Seed 2 | Mean |
     |---|---|---|---|---|
     | `C_gap` | +0.056 | +0.155 | +0.262 | +0.158 |
     | Held-out `C_gap128` | +0.061 | +0.202 | +0.183 | +0.148 |

   - **Why it is not a CGC benefit.** It is carried by the ERSA state (§6). It reverses in the attention-free pair
     (D_A − D_C on `C_gap`: −0.306, +0.060, −0.011). It is part of RETAIN, an endpoint that was not learned. At
     most it is a CGC × attention interaction on ERSA-carried memory, which three seeds cannot establish.
2. **Unseen shift rules** (`B_heldout`).
   - **The pattern.** A few runs learned to infer shifts they never trained on, with accuracy rising with k:
     - A seed 2: 0.379 at k=1 → 0.507 at k=16;
     - C seed 1: 0.298 → 0.378;
     - C_mac seed 1: 0.234 → 0.339;
     - D_A seeds 1 and 2: about 0.14 → 0.23.

     The rest stay near chance (0.04–0.10).
   - **The pairs.** D_A exceeds D_C on every seed (+0.053, +0.080, +0.127; levels 0.11–0.18 against 0.05–0.10).
     A against C is mixed (+0.011, −0.136, +0.404).
   - **The status.** The held-out endpoints were preregistered only as a confirmation check, which did not run.
3. **In-window retrieval emergence.**
   - **The pattern.** Arm A never acquired in-window regime-conditioned retrieval: C_short post at most 0.133 in
     its three runs. The FFN window arms acquired it fully in 3 of 9 runs and partly in 2.
   - **The status.** It is not distinguishable from chance. At the full-emergence rate of the FFN arms (3 of 9),
     0 of 3 has probability about 0.30; counting partial emergence (5 of 9), about 0.09.

## 9. Execution incidents and deviations

**1. CI: the runner's parameter digest needs NumPy.**
- **Failure.** Every CI run on the branch before `d167609` failed the PyTorch CPU job, on one test
  (`test_evaluation_and_diagnostics_update_no_parameter`; 1 failed, 566 passed). The cause:
  `run_r10._parameters_digest` hashes through `Tensor.numpy()`, and CI's PyTorch-only environment has no NumPy.
- **Fix.** The test now hashes the raw bytes through `ctypes`, the convention of `model.py` and `run_r7.py`.
  Reproduced locally with NumPy made unimportable: the old test fails with the same error, and the new one
  passes. CI run 44759132960 on `d167609` passed every job.
- **What stays.** The runner itself is unchanged, because it is frozen and every record carries its SHA-256. On
  this VM, where every run executed, NumPy is installed and the digest is exact: a test asserts it equals the
  `ctypes` byte digest wherever NumPy exists. Where NumPy is absent, that test is NOT_RUN with an explicit reason.
  Reproducing R10 therefore needs NumPy beside PyTorch. A later runner should hash through `ctypes`.

**2. Two interruptions killed four screening runs mid-training.**
- **What was lost.**
  - At about 15:17 UTC the VM restarted. `C_mac` seed 0 (started 15:04:45) and `D_C` seed 0 (started 15:07:36)
    were killed before finishing.
  - At 17:18:50 the session's 2-hour limit on a background task stopped the queue. `D_A` seed 2 (started
    17:07:39) and `B` seed 2 (started 17:08:53) were killed before finishing.

  The runner writes its record only at the end, so none of the four left any output, and no result of theirs was
  ever seen.
- **What was done.** Each was restarted from scratch, on the same code, seed and episode stream: at 15:18:50 and
  17:19:05. The queue was made resumable: it skips any run whose record exists, so a completed run is never
  executed twice. The queue log in §11 has every start, skip, kill and exit.
- **How it is treated.** These are restarts of interrupted executions, not reruns of observed results. The
  preregistration forbids the latter, and nothing was rerun.

**3. Training times are not comparable between arms.** Two runs always shared the 4-core VM, and which runs were
paired varied. The `train.seconds` in each record is a resource figure. Speed is compared only by the latency
record.

**4. The R11 note's training-step figure is corrected.** The figure quoted before the study (0.74 s against
0.33 s per step) came from the opening steps of a run. The full runs here give a different picture:

| Arms | Time for 3,000 steps | Per step |
|---|---|---|
| Window FFN arms (B, C, C_mac) | 1,162–1,225 s | 0.39–0.41 s |
| D_C, no attention | 991 s | 0.33 s |

Both figures include periodic evaluation. `R11_PERFORMANCE_NOTE.md` now quotes these.

## 10. Scarcity sweep, Gate A, limitations

**The width sweep is not justified.** P6 of the directive allows preregistering it only if the assay is valid and
favorable.
- ADAPT was learned, but CGC was worse.
- RETAIN was not learned.

Nothing here argues for scaling FFN width 8–256 against CGC. The exploratory patterns of §8 are retention behind a
gap, held-out shift learning, and emergence. They would call for a preregistered test of those patterns with more
seeds, not a width sweep.

**Gate A** stays HOLD. R10 built no K1-to-HCA interface and promoted nothing from Yang–Mills/HCA.

**Limitations.**
- **Scale.** d 64, two blocks, 3,000 steps. Adaptation and retention may behave differently at larger scale or with
  longer training. The emergence of in-window retrieval was all-or-nothing per run, and seed noise of that size
  (ADAPT sd up to 0.21) needs many seeds to resolve small effects.
- **Retention was not learned by any arm.** So R10 cannot say whether CGC would help a model that does retain
  earlier bindings. It can only say that CGC did not make retention appear.
- **Only three screening seeds.** By design, the screening stage cannot support a positive claim, and its
  negative verdict is a preregistered decision, not a proof of absence.
- **No GPU** (CUDA NOT_RUN) and **no PMU**. Latency is CPU wall-clock on this VM.
- **The runner's NumPy dependency** (§9).

## 11. Reproduction

R10 needs PyTorch 2.x CPU and NumPy (§9). Commands, from the repository root (`PYTHONDONTWRITEBYTECODE=1`):

```
python3 -B -m research.elpis_torch.experiments.run_r10 plan
python3 -B -m research.elpis_torch.experiments.run_r10 latency --out research/elpis_torch/results/r10/latency.json
python3 -B -m research.elpis_torch.experiments.run_r10 train --arm A --seed 0 --out research/elpis_torch/results/r10/run_A_s0.json
python3 -B -m research.elpis_torch.experiments.run_r10 combine --out research/elpis_torch/results/r10/study_cgc_adaptation.json
python3 -B -m research.elpis_torch.experiments.run_r10 manifest --out research/elpis_torch/results/r10/MANIFEST.json
```

`train` refuses a seed or arm that is not in the frozen plan. Each run is deterministic given its seed. Two at a
time with two threads each, a run takes 16–26 minutes on this VM.

**The screening queue log** (UTC):

```
14:29:34 screening start
14:47:00 train:C:0 exit=0 seconds=1046
14:51:26 train:A:0 exit=0 seconds=1312
15:04:45 train:D_A:0 exit=0 seconds=1065
15:07:36 train:B:0 exit=0 seconds=970
~15:17   CONTAINER RESTART: train:C_mac:0 and train:D_C:0 killed before completion; no record written
15:18:50 screening (re)start; A:0, C:0, D_A:0, B:0 skipped (records exist); C_mac:0 and D_C:0 restarted
15:36:35 train:D_C:0 exit=0 seconds=1065
15:40:37 train:C_mac:0 exit=0 seconds=1307
16:02:15 train:C:1 exit=0 seconds=1298
16:02:52 train:A:1 exit=0 seconds=1577
16:25:13 train:B:1 exit=0 seconds=1341
16:25:41 train:D_A:1 exit=0 seconds=1406
16:42:50 train:D_C:1 exit=0 seconds=1029
16:46:14 train:C_mac:1 exit=0 seconds=1261
17:07:39 train:C:2 exit=0 seconds=1285
17:08:53 train:A:2 exit=0 seconds=1563
17:18:50 QUEUE KILLED by the 2-hour background-task limit: train:D_A:2 and train:B:2 killed; no record written
17:19:05 screening (re)start; 14 runs skipped (records exist); D_A:2 and B:2 restarted
17:40:05 train:B:2 exit=0 seconds=1260
17:40:38 train:D_A:2 exit=0 seconds=1293
17:56:47 train:D_C:2 exit=0 seconds=969
18:00:27 train:C_mac:2 exit=0 seconds=1222
18:00:27 screening done
```
