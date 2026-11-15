# ERSA R17 — CPU GDN-v1 layer structural reference (research-only)

## Scope
R16 qualified a trainable 1,075-parameter gated-delta *cell*, not a full
Gated DeltaNet layer. R17 independently implements the CPU PyTorch
architectural **components** described by the MIT-licensed FLA GDN-v1 layer:
Q/K/V projections, causal short depthwise convolutions, A_log + dt_bias
decay, sigmoid beta update, output SiLU gating, RMSNorm, and output linear.
The recurrence uses the FLA published order (decay → delta error → write → read).

Primary comparison sources (not bundled or imported):
- https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
- https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py

**This is NOT the official FLA implementation.** The FLA layer and Triton kernels
were not imported or run. Departures include: external 12→32 token embedding,
single head (K=16,V=32), zero-beta read-only query suffix, task-specific 32→4
readout, CPU eager unrolled recurrence and synthetic pointer chains.
Those departures and trainable parameter disparity preclude architecture-level
ERSA-vs-GDN parity claims.

## Shared task
R17 consumes the existing R16 24-record keys/goals/values and the identical
held-out family seeds. 3 independent training seeds, 256 episodes and 30 epochs
per seed, validation-only threshold selection and 64 held-out cases per family.
It does not persist learned weights. A frozen ERSA R14 study is available as
historical evidence but is **not** rerun by default.

## R17 read-only query state-invariance repair (R1)
The original R17 failed `torch.equal(single, three)` on Ouroboros because
the length of the appended query suffix changed CPU convolution arithmetic
for the preceding store tokens. R1 computes the complete 24-record store
state in a fixed-shape pass, then computes read-only query projections with
the last three store projections as causal context. Query processing never
executes the state recurrence. This enforces **bitwise invariant state**
across query counts and contents without relaxing the original equality test.
Additional 20 BREAK regressions and an independent 80-case CPU probe cover representative old failing seeds,
masked and unmasked stores, a non-vacuous store mutation control, and a
post-training read-only control. The 4,930 trainable parameters, source
authority boundaries and CPU-only execution remain unchanged. The change
may alter low-order floating-point numerical values; R17 science must be
rerun, not reused from the prior failed qualification. The inherited all-in-one runner is unsuitable for qualification here:
combining PyTorch science and native inherited tests can stall either second
gate. Use `tools/qualify_ersa_r17_r1.py --gate tests`, then `--gate science`
in a separate shell invocation, then `--gate finalize`. Each heavyweight
gate creates its own checksum-bound receipt in /dev/shm; finalize requires
both. No standalone full-suite PASS is inferred from a prior session.

## Qualification
- frozen R16 source manifest 114/114 and frozen SSC 7/7
- original R2 four-file external pins verified at run time
- all previous regression tests + new R17 numerical and BREAK tests
- NumPy independent state-update comparison, denied-record read/write
  isolation prior to causal convolution, no state update from read-only query
- three-seed CPU training, finite gradients and model-state immutability in eval

`DISPOSITION=PASS_CPU_GDN_V1_STRUCTURAL_REFERENCE_TRAINING_AND_NUMERICAL_MECHANICS_ONLY`

**Not qualified:** FLA kernel parity, equal memory/compute training conditions,
learned ERSA superiority, open-world retrieval, SSC official fixture transport,
production rights, OS NVMe write attestation.

## Run
Unpack the ZIP under `/dev/shm`, verify `FILES.sha256`, set `TMPDIR=/dev/shm`,
`HOME=/dev/shm/...`, `PYTHONDONTWRITEBYTECODE=1`, CPU only, then run
`python3 -B tools/qualify_ersa_r17.py` with `PYTHONPATH` pointing to the
unpacked root, `src`, `native`, `ssc_predecessor/src`, and `tests`.
The original four source pins must exist at the frozen R2 predecessor directory.

All source changes are isolated to this ZIP. No standalone mutations to the
persistent Elpis project tree are intended or needed.
