# ERSA R16 — Trainable gated-delta-cell benchmark

**Status:** conditional research-only benchmark. This is a trainable gated-delta **cell**, not the full FLA neural layer, and not a claim of GDN parity.

## Scope

- Preserve the 107 frozen ERSA R15 source files with exact manifest pins.
- Implement the published gated-delta recurrence (decay, prediction error, gated correction) using independent PyTorch and NumPy implementations. No external GDN source is bundled.
- Train all Q, K, V, beta, log-forget, output/readout and gain parameters offline on the R13/R14 synthetic 24-record, 1–4-hop pointer-chain task. Explicit teacher-forced path labels are used ONLY as offline supervision, never by inference.
- Compare against the R15 frozen ERSA learned/strong-deterministic science experiment, previously qualified on Ouroboros. R15 science is *not re-executed by default* (source pins are checked); optional `--replay-r15` explicitly replays and verifies the historical science digest.
- Validate stop thresholds using disjoint validation seeds; evaluate four predeclared R15 held-out seeds, 64 episodes each. No test labels in threshold selection.
- Track correctly stopped, false stops, abstentions, teacher-forced similarity and train loss across three independent training seeds.
- Retain RAM-only inference. No persistent weights, checkpoints, runtime learning, authority expansion or GPU use.

## What this does NOT qualify

This experiment does not run the full FLA GatedDeltaNet layer (convolution/mixer/normalization, multilayer stack, GPU kernels), does not parameter-match ERSA's 14-parameter head, does not match total state-bank/index memory, and is not a model-scale or open-world generalization benchmark. The trained gating cell has 1,075 parameters and 1,024 bytes FP32 recurrent state; ERSA's bare 24×(4+4+4) FP32 data is 1,152 bytes, **excluding** ERSA record identities, index, bank state, and native runtime.

The GDN cell is trained with 256 episodes/30 epochs/AdamW, while the frozen ERSA models use the R12/R13 training protocols. Consequently, source inputs and held-out tasks match, but budgets and optimization regimes do not. Report relative results as **diagnostics only**, especially if the trained cell fails.

## Sources

- [FLA MIT reference recurrence](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)
- [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464)

No FLA/NVIDIA code or model weights redistributed. R15's frozen source remains unchanged.

## Execution

Run the extracted `/dev/shm` archive with `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `TMPDIR=/dev/shm`, `HOME=/dev/shm/...`, `CUDA_VISIBLE_DEVICES=-1`. Source pin verification uses live SSC/R2 paths from the original on-host installation. Execute `python3 -B tools/qualify_ersa_r16.py` from the extracted root. The standalone script also prints an `=== SEND BACK ===` section. Do not use `python -I`, as the project's local imports rely on `PYTHONPATH`.

**Status of R15 evidence:** `2eed88cfc7af55adddd9e7752f927e82e29d74217d4beeb9742c5f9f117d6c94` is a prior reported Ouroboros digest, not an R16-recomputed measurement unless the optional replay completes.

**Speed/scoping:** The original 8,192-slot bank stress is inherited as previously verified on Ouroboros and **not rerun by default**. `--rerun-capacity` forces a fresh execution. `--replay-r15` similarly reruns frozen R15 science. Both use the pinned original files. R16 does not mutate the original bank implementation.
