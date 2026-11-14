# Elpis Torch R0: evidence report

Status: RESEARCH_ONLY, FEASIBILITY_ONLY, SEMANTICS=NONE, NO_LANGUAGE_CLAIM. This report states what was
executed, what passed, what failed and what was not run. Nothing here is a qualification, and nothing here
shows that the model understands language or reasons. Design: `DESIGN.md`. Raw record:
`research/elpis_torch/results/run_r0.json`.

## 1. Environment and identities

| | |
|---|---|
| Repository, base commit | `abraxis717/Elpis_DEV`, `5b24b39d793311244ef0ea1af9508d8ebde45bf0` |
| CGC source | `abraxis717/CGC` at `0895c2bc732a377b5a5f0d063ae13cc72684ec9a`, `src/cgc/ecsg_e0.py`, SHA-256 `92c987e0fb54d80193fd60ecc9a3e31088f0f9b7cd2cfe8e08c4636a198d7084` |
| Machine | one cloud container: 4 vCPU Intel Xeon 2.30 GHz, x86_64, Linux 6.18, 15 GB RAM. No GPU. |
| Software | Python 3.12.3, PyTorch 2.15.0+cpu. Tests on 1 thread; the experiment on 4 threads. |
| Not available | the owner's workstation, local model files, GPUs, unpublished experiments. None was used or assumed. |

The experiment ran on the working tree before its first commit. The package sources it executed are
byte-identical to the committed ones except for a docstring in `e0.py` and a trailing blank line in
`data.py`, both edited afterwards.

## 2. Tests executed

`ELPIS_REQUIRE_TORCH=1 ELPIS_REQUIRE_NATIVE=1 pytest tests/research/elpis_torch`: **197 passed, 0 failed,
0 skipped** (about 60 s, one thread) at the final head.

| File | Tests | What it establishes |
|---|---:|---|
| `test_e0_reference.py` | 28 | the E0 reference against the pinned upstream (bitwise), exact rational arithmetic, the S3 moment identity, autograd, finite differences, and the native ECS_G kernel at `d = 6`, `N in {36, 48, 72}` |
| `test_cgc_operator.py` | 22 | the batched core is E0 per channel; the bottleneck is a factorized E0 weight; the bound is a bounded deviation that caps every cell; exact gradients; formulas |
| `test_model.py` | 90 | configuration validation, deterministic initialization, causality, cache parity, generation, gradients, deep-stack stability, accounting |
| `test_training.py` | 17 | the verified data split, learning on all three configurations, reproducibility, divergence handling, explicit checkpoints |
| `test_session.py` | 33 | fast weights, isolation, bounds, refusal and rollback, provenance, cache coherence, no serialization, zero writes |
| `test_isolation.py` | 7 | the prototype is research only, PyTorch optional, mission unedited |

Without PyTorch (Python 3.11, checked): the five PyTorch modules are skipped with the reason
"not a PASS", `test_isolation.py` passes, and `ELPIS_REQUIRE_TORCH=1` turns the absence into a collection
error.

### Gates

| Gate | Result | Test |
|---|---|---|
| E0 reference equals the upstream source | PASS, bitwise on 5 shapes | `test_forward_gradient_and_step_equal_the_upstream_bitwise` |
| E0 reference equals exact arithmetic, the moment polynomial, the native kernel | PASS (`1e-12`, `1e-11`, `1e-13`) | `test_e0_reference.py` |
| Batched core equals E0 per channel; bottleneck equals factorized E0 | PASS (`1e-12`, `1e-11`) | `test_the_unbounded_core_is_e0_channel_by_channel`, `test_each_cell_of_the_transform_is_e0_on_the_full_hidden_input` |
| Forward: shapes and finiteness for A, B, C under MHA and GQA | PASS | `test_logits_have_the_declared_shape_for_mha_and_gqa` |
| Backward: every parameter receives a finite, nonzero gradient | PASS | `test_every_parameter_receives_a_finite_gradient` |
| Causality: future tokens do not change earlier logits; the Jacobian on later positions is exactly zero | PASS | `test_future_tokens_cannot_affect_earlier_logits`, `test_the_jacobian_of_a_position_is_zero_on_every_later_position` |
| Prefill versus incremental parity | PASS, within `1e-10` (float64) and `2e-5` (float32), 4 chunkings x 3 configurations | `test_prefill_and_incremental_decoding_agree_with_the_batched_forward` |
| Cached generation equals recomputing the prefix | PASS, token-identical | `test_cached_generation_equals_recomputing_the_whole_prefix` |
| C starts as exactly A | PASS, bitwise logits | `test_the_augmented_block_starts_as_exactly_the_ffn_block` |
| Training reduces the loss; held-out accuracy reported separately | PASS | `test_training_reduces_the_loss_and_generalizes_to_unseen_trajectories` |
| CGC gradients finite and nonzero; CGC weights move | PASS | `test_cgc_gradients_are_finite_and_nonzero_and_cgc_weights_move` |
| Reproducibility from seeds | PASS, bitwise parameter digest | `test_training_reproduces_bit_for_bit_from_its_seeds` |
| Evidence changes fast weights, never the checkpoint | PASS | `test_observed_evidence_changes_fast_weights_and_never_the_checkpoint` |
| Adaptation improves a held-out continuation; the disabled baseline does not | PASS | `test_adaptation_improves_a_held_out_continuation_and_the_disabled_baseline_does_not`, `test_a_stream_learns_from_its_observed_past_and_predicts_its_own_future_better` |
| Reset restores initial behaviour exactly | PASS, bitwise logits | `test_reset_restores_the_initial_behavior_exactly` |
| Restart from the checkpoint | PASS, same digest and logits; file digest, size, mtime, inode unchanged | `test_restarting_from_the_checkpoint_reproduces_parameters_and_predictions` |
| Session isolation | PASS | `test_sessions_over_one_model_share_nothing_mutable` |
| Bounds, refusal, rollback (all nine refusal statuses) | PASS | `test_session.py`, "bounds, refusal and rollback" |
| Generated tokens are never targets | PASS, bitwise equal to the observed-only update | `test_generated_tokens_are_context_and_never_targets` |
| Cache invalidated and recomputed after an update; stale caches refused | PASS, with a negative control | `test_an_update_invalidates_the_cache_and_the_stream_is_recomputed_under_the_new_weights` |
| No filesystem write in a session lifetime | PASS at the Python audit level and at the system-call level (strace), each with a negative control | `test_a_session_lifetime_performs_no_python_level_filesystem_write`, `test_a_session_lifetime_issues_no_write_class_system_call` |

Two defects were found by these tests and fixed:

* In the implementation, during development: a candidate fast weight whose norm overflowed was rescaled to
  zero by the projection and committed as a valid update. It is now refused (`REFUSED_NONFINITE_CANDIDATE`).
* In the test itself, after the first push: the strace trace parser assumed one blank after the process id.
  strace pads that column, so with a process id below 10000 no line was recognized, and the "no write-class
  system call" assertion would have passed vacuously. The negative control caught it (locally after a
  process restart, and in the first CI run of the new job: 195 passed, 1 failed). The parser now splits on
  any run of blanks and has its own test over process-id widths. The earlier local passes used five-digit
  process ids, for which the parser was correct.

## 3. The experiment

`python -m research.elpis_torch.experiments.run_r0`: 7 variants x 3 seeds = 21 training runs, all COMPLETED,
no numerical anomaly recorded.

**Task.** Finite-state transduction over 10 states and 5 operations (`research/elpis_torch/data.py`):
sequences such as `Ka=Fbc=D`, where `a=F` is one transition and `bc=D` composes two without writing the
intermediate state. 4,096 training sequences, 512 validation sequences from another seed (none occurs in
training: 0 overlap), 512 compositional sequences. 50 of the 250 double-step triples never occur in
training or validation (0 leaked sequences) and are the compositional test. Only the token after `=` is
determined; operations are random, which is why the full next-token loss cannot fall below about 0.85.

**Protocol, identical for every variant.** `d_model` 64, 2 layers, 4 query heads, 2 key/value heads (GQA),
context 48, float32. 1,500 steps, batch 64, 4,608,000 training tokens per run, the same batches for a given
seed. AdamW, learning rate 3e-3, weight decay 0.01 on matrices, warmup 50, cosine to 10%, gradient clip 1.0.

### Parameters, theoretical compute, memory

| Variant | Parameters | Channel parameters per block (FFN + CGC + gate) | Multiply-accumulates per token at context 48 | Channel multiply-accumulates per block | Uncounted transcendentals per token | Widest channel activation per token | Training activation estimate (batch 64) |
|---|---:|---|---:|---:|---:|---:|---:|
| `A_ffn` | 107,520 | 33,024 + 0 + 0 | 120,856 | 33,196 | 728 | 172 | 36.1 MB |
| `B_cgc_bottleneck` | 107,520 | 0 + 33,024 + 0 | 124,868 | 35,202 | 1,836 | 726 | 79.8 MB |
| `B_cgc_bottleneck_unbounded` | 107,520 | 0 + 33,024 + 0 | 124,868 | 35,202 | 384 | 726 | 62.3 MB |
| `B_cgc_dense` | 107,520 | 0 + 33,024 + 0 | 123,092 | 34,314 | 1,244 | 430 | 56.8 MB |
| `C_matched_gate0`, `C_matched_gate0.1` | 107,522 | 24,576 + 8,448 + 1 | 122,288 | 33,912 | 1,104 | 232 | 51.5 MB |
| `C_augmented_gate0` | 124,418 | 33,024 + 8,448 + 1 | 139,272 | 42,404 | 1,192 | 232 | 54.7 MB |

The three B variants are parameter-matched to A exactly, the matched C variants to within the two scalar
gates. **Parameter-matched is not compute-matched:** the B variants need 1.9% to 3.3% more theoretical
multiply-accumulates than A, 1.7 to 2.5 times as many uncounted transcendentals when bounded, and an
estimated 1.6 to 2.2 times the activation memory. `C_augmented_gate0` is neither: +15.7% parameters,
+15.2% multiply-accumulates. The memory figures are analytic estimates of activation scalars
(`accounting.memory_estimate`); peak process memory was not measured per variant.

Measured rates on this machine (PERFORMANCE_ONLY, one host, never a pass criterion; mean [min, max] over
3 seeds):

| Variant | Training steps per second | Cached generation, tokens per second |
|---|---|---|
| `A_ffn` | 36.9 [28.7, 41.1] | 1964 [1910, 2018] |
| `B_cgc_bottleneck` | 19.9 [18.9, 21.1] | 1491 [1413, 1596] |
| `B_cgc_bottleneck_unbounded` | 25.7 [25.2, 26.4] | 1890 [1687, 1999] |
| `B_cgc_dense` | 27.8 [26.3, 29.2] | 1823 [1740, 1928] |
| `C_matched_gate0` | 33.9 [32.5, 34.8] | 1661 [1596, 1758] |
| `C_matched_gate0.1` | 34.9 [33.0, 36.6] | 1638 [1407, 1754] |
| `C_augmented_gate0` | 31.9 [31.6, 32.6] | 1402 [1356, 1456] |

At equal parameters and nearly equal multiply-accumulates, CGC-only training ran at 54% to 75% of the FFN
baseline's speed here.

### Learning (mean [min, max] over 3 seeds)

Loss before training is about 5.53 to 5.57 for every variant (`log 259 = 5.56`).

| Variant | Final training loss | Held-out validation loss | Held-out single-step accuracy | Held-out double-step accuracy (seen triples) | Free-running rollouts exact | Steps to double-step accuracy 0.99 |
|---|---|---|---|---|---|---|
| `A_ffn` | 0.850 [0.847, 0.855] | 0.851 [0.849, 0.855] | 1.000 | 1.000 | 1.000 | 900, 900, 900 |
| `B_cgc_bottleneck` | 0.850 [0.848, 0.852] | 0.851 [0.850, 0.852] | 1.000 | 1.000 | 1.000 | 800, 1000, 700 |
| `B_cgc_bottleneck_unbounded` | 0.845 [0.844, 0.847] | 0.848 [0.847, 0.849] | 1.000 | 1.000 | 1.000 | 800, 800, 800 |
| `B_cgc_dense` | 0.849 [0.846, 0.853] | 0.851 [0.848, 0.856] | 1.000 | 1.000 | 0.999 [0.996, 1.000] | 800, 800, 800 |
| `C_matched_gate0.1` | 0.848 [0.845, 0.853] | 0.851 [0.848, 0.856] | 1.000 | 1.000 | 1.000 | 900, 700, 600 |
| `C_matched_gate0` | 0.909 [0.850, 1.019] | 0.908 [0.852, 1.014] | 1.000 | 0.776 [0.329, 1.000] | 0.689 [0.066, 1.000] | 900, not reached, 1100 |
| `C_augmented_gate0` | 0.876 [0.848, 0.927] | 0.877 [0.850, 0.927] | 1.000 | 0.911 [0.733, 1.000] | 0.785 [0.355, 1.000] | 900, not reached, 900 |

What this supports, and no more:

* Every configuration trains: loss falls from 5.5 to about 0.85, and training and held-out losses agree.
* On held-out trajectories, the CGC-only models (B) reach the same loss and accuracy as the FFN baseline
  at equal parameters, in a comparable number of steps. Three seeds on one small task do not separate A
  from B in quality. There is no evidence here that CGC is better than an FFN, and none that it is worse.
* The free-running rollout (states generated through the KV cache and fed back) is exact for the
  converged models: errors do not compound on trajectories of this length.

### CGC numerics after training

| Variant | Largest raw pre-activation `\|u\|` | Fraction beyond the bound `b = 3` (worst block) | CGC core gradient norm, min to max over training |
|---|---|---|---|
| `B_cgc_bottleneck` | 8.7 [7.7, 10.3] | 4.9% [2.7%, 7.0%] | 0.002 to 1.4 |
| `B_cgc_dense` | 5.6 [5.0, 5.9] | 1.6% [1.2%, 2.5%] | 0.002 to 4.3 |
| `B_cgc_bottleneck_unbounded` | 9.7 [8.8, 11.1] | no bound | 0.002 to 1.3 |
| `C_matched_gate0.1` | 7.1 [6.4, 8.4] | 1.4% [1.0%, 2.1%] | 0.001 to 0.44 |

All gradients were finite in all 21 runs. CGC core weights moved by 39% to 81% of their initial norm.
Trained pre-activations grow from the initial standard deviation 0.5 to maxima near 10, where an unbounded
unit outputs about `phi(10) = 555`.

## 4. Volatile adaptation

Per trained model: the world's last operation is rewired (all 10 of its transitions change). The session
receives 16 observed trajectories of the changed world as a `SupportSet` and makes 16 bounded updates.
Evaluation is on 256 other trajectories of the changed world, split into transitions that changed and
transitions that did not. Mean [min, max] over 3 seeds; loss in nats.

| Variant | Update status | Changed: loss, learning disabled | Changed: loss after (`lr` 0.05) | Changed: loss after (`lr` 0.2) | Changed: accuracy after (`lr` 0.2) | Unchanged: loss disabled -> after (`lr` 0.2) | Unchanged: accuracy after (`lr` 0.2) |
|---|---|---|---|---|---|---|---|
| `A_ffn` | REFUSED_NO_FAST_WEIGHTS | 12.45 | n/a | n/a | n/a | 0.000 -> n/a | n/a |
| `B_cgc_bottleneck` | APPLIED | 14.70 [13.44, 16.61] | 3.16 [1.75, 4.43] | 0.94 [0.06, 2.65] | 0.895 [0.704, 1.000] | 0.000 -> 0.059 [0.004, 0.167] | 0.974 [0.923, 1.000] |
| `B_cgc_dense` | APPLIED | 13.90 [13.59, 14.09] | 3.03 [2.27, 3.95] | 0.28 [0.04, 0.61] | 0.917 [0.797, 0.999] | 0.000 -> 0.025 [0.011, 0.044] | 0.999 [0.997, 1.000] |
| `B_cgc_bottleneck_unbounded` | APPLIED | 13.41 [12.74, 14.62] | 2.17 [1.02, 3.45] | 0.08 [0.02, 0.13] | 1.000 | 0.000 -> 0.011 [0.002, 0.017] | 1.000 |
| `C_matched_gate0.1` | APPLIED | 12.91 [12.59, 13.51] | 11.70 [10.81, 12.92] | 8.14 [7.19, 9.39] | 0.079 [0.009, 0.188] | 0.000 -> 0.012 [0.009, 0.016] | 0.998 |
| `C_matched_gate0` | APPLIED | 12.34 [11.12, 13.31] | 11.17 [9.86, 12.68] | 8.32 [6.14, 9.43] | 0.098 [0.000, 0.293] | 0.004 -> 0.022 [0.001, 0.043] | 0.999 |
| `C_augmented_gate0` | APPLIED | 12.44 [10.87, 14.30] | 12.13 [10.49, 14.18] | 11.10 [9.84, 13.52] | 0.010 [0.000, 0.029] | 0.001 -> 0.002 [0.001, 0.004] | 1.000 |

In all 21 models and both settings: the learning-disabled session scored 0.000 accuracy on the changed
transitions and refused to adapt; reset restored the checkpoint's logits bit for bit; the base parameter
digest was unchanged. No update was projected (the delta-norm limit never bound at these settings); most
B updates were gradient-clipped.

What this supports:

* With CGC as the block's channel transformation, 16 bounded updates on observed evidence move a
  confidently wrong model (loss 13 to 15 nats, accuracy 0) to a mean accuracy of 0.90 to 1.00 (worst seed
  0.70) on unseen trajectories of the changed world, while the checkpoint stays untouched.
* The fast weights have leverage only where CGC carries the computation. In C the CGC path is a small
  gated addition and adapting it alone barely helps (accuracy at most 0.10).

## 5. Failures and negative results

1. **No compositional generalization, in any configuration.** Accuracy on double steps whose triple never
   occurred in training is 0.12 to 0.23 by variant mean (chance 0.10), against 1.00 on seen triples:
   `A_ffn` 0.198 [0.171, 0.213], `B_cgc_bottleneck` 0.172 [0.074, 0.281], `B_cgc_dense` 0.170 [0.149, 0.200],
   `B_cgc_bottleneck_unbounded` 0.124 [0.069, 0.203], `C_matched_gate0.1` 0.138 [0.102, 0.206]. The models
   fit seen compositions as lookups and do not compose transitions. The sequential-dependency result
   (unseen trajectories) holds; the compositional one does not.
2. **Starting C at gate 0 is unreliable.** It gives exact equivalence to A at initialization, and on seed
   1 both gate-0 variants stayed on the initial plateau about twice as long as A with the same attention
   and FFN weights (loss 1.53 until step 500 to 700; A left it by step 300 to 400) and did not converge in the
   budget (final double-step accuracy 0.33 and 0.73). Started at gate 0.1, C converged on 3 of 3 seeds. The
   small test models show the same lag. The cause was not investigated.
3. **Adaptation interferes with what did not change.** Unchanged transitions go from loss 0.000 to
   between 0.00 and 0.17 in the experiment, and to about 0.3 in the much smaller test model, where a learning rate
   of 0.2 also made the trajectory of updates visibly noisy. No retention mechanism is applied.
4. **The bound was not needed for training at this scale, and it cost something.** The unbounded
   (faithful E0) variant trained without anomaly on 3 of 3 seeds, was faster, and adapted best. With the
   bound, 3% to 7% of pre-activations of the bottleneck variant sit beyond `b`, where gradients shrink. The
   bound's value is shown only by construction: an absurd fast-weight step overflows the unbounded cubic
   (rolled back) and not the bounded one (`test_a_candidate_that_breaks_the_model_is_rolled_back`). Whether
   to keep it, and at what `b`, is open.
5. **CGC is slower per step here** (section 3), although its multiply-accumulate count is within 3.3% of
   the baseline's.
6. **Two defects**, one in the delta projection and one in the zero-write test's trace parser (section 2),
   both found by tests and fixed.

## 6. Not executed, not verified

* Anything on a GPU, in mixed or half precision, or at a scale beyond about 1.2e5 parameters.
* Any real text, any pretrained tokenizer, any comparison with the DSV4.1 research tower.
* Compute-matched comparison by construction (only accounted), and measured peak memory.
* More than 3 seeds, any hyperparameter search, any statistical test. The comparison is descriptive.
* Bitwise reproducibility across thread counts, PyTorch builds or machines (asserted for one process only).
* Parity of gradients or of the bounded update with the native ECS_G kernel (only the forward map was
  compared, in float64, unbounded).
* Long streams: the session stream is bounded by the context length and was exercised up to 64 tokens.
* Concurrent sessions on several threads.
* Host-level zero-write properties: swap, core dumps, read-only mounts, crash collectors (`DESIGN.md`,
  "Zero-write properties"). The strace gate covers one scenario on one host.
* The new CI job (`Elpis Torch research (PyTorch CPU)`) was validated as YAML and its commands were run
  locally; its first run on GitHub is the pull request's.

## 7. Existing repository

No file under `src/`, `native/`, `docs/ELPIS_MISSION.md`, `ELPIS_SYSTEM.json`, `docs/CI_POLICY.md`,
`pyproject.toml`, `tests/lanes.py` or any existing test or evidence was modified. The only edit to an
existing file is the added job in `.github/workflows/ci.yml`.

Local runs (Python 3.12, PyTorch installed, native libraries built from this tree, `ELPIS_REQUIRE_NATIVE=1`):

| Lane | Base commit, clean worktree | Changed tree |
|---|---|---|
| FAST (includes the boundary suite) | 1902 passed, 25 skipped | 1902 passed, 25 skipped |
| HISTORICAL | not run | 444 passed, 102 skipped |
| STRESS | not run | 104 passed |
| SCIENTIFIC | not run as a lane | 830 passed, 24 skipped, 5 failed on the first run; see below |

The five first-run SCIENTIFIC failures, each classified by running it on the base commit as well:

| Test | Class | Finding |
|---|---|---|
| `ecs_k1_native/test_plan.py::test_the_plan_was_committed_before_any_differential_evidence`, `ecs_memory_mf_r0/test_evidence.py::test_chronology_preregistration_laboratory_dev_freeze_qual`, `ecs_retention_r0/test_evidence.py::test_the_specification_predates_dev_and_the_freeze_predates_qual` | CI-ENVIRONMENT (this container) | chronology gates on a shallow clone. They fail identically on the base commit and pass on both trees once full history is fetched. |
| `qualification_k1_recovery_r0_v4/test_freeze.py::test_an_isolated_interpreter_imports_the_checkouts_own_package_whatever_pythonpath_says` | CI-ENVIRONMENT (this container) | needs the Elpis package installed in the interpreter; fails identically on the base commit and passes after `pip install .`. |
| `elpis_torch/test_session.py::test_a_session_lifetime_issues_no_write_class_system_call` | TEST-CONTRACT DEFECT, in this change | the trace parser defect of section 2. Fixed; passes. |

After those corrections the four affected existing test files and the whole new suite pass. The complete
SCIENTIFIC lane was not re-run locally after the corrections; the Scientific authority job is its owner.
The first GitHub run on this pull request failed only in the new job, on the parser defect; the status of
every job on the final head is the pull request's, not this file's.
