ERSA R36 — CPU-native dual-matrix state / donor recurrence differential
======================================================================
Strictly an additive R35 adapter. Frozen R18–R35 source must already exist in
ERSA_DEV and is verified at SHA-256 file granularity. No donor repository or
historical code is shipped in this patch.

Training: frozen R35 ERSASlotWriteRead uses the MIT FLA-derived naive GSA
reference plus Raven-style sparse write routing and the R18 ERSA Torch read.

Inference: model weights, router, projections and R18 read remain frozen Torch;
C is the sole owner of the bounded dual hk/hv slot tensors per inference call.
The native implementation computes the GSA update independently, provides a
versioned import/export snapshot contract, atomically rejects nonfinite updates,
serializes state accesses using pthreads, and releases/zeros memory on close.
Sequential vs split continuation and C-vs-FLA-naive numerical parity tested.

IMPORTANT HOLDS:
- R11 C entity bank and the new GSA matrix bank are DISTINCT state authorities.
  Their identity/admission/transaction contracts have NOT been unified.
- FLA Triton/CUDA kernels are not executed, nor is actual RavenAttention.
- Native C receives projections and authorized sparse gates from host; the
  full neural layer does not execute in C.
- No GPU, pretrained model, GDN comparison, model-scale production trust,
  swap-free OS attestation, or persistent inference-write attestation.

Single source install location:
/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV/worktrees/
  R36_NATIVE_GSA_SLOT_STATE_PARITY_R0

CPU C compilation, testing, receipts and trained tensor state execute in tmpfs.
Only R36 source/support files are explicitly promoted after gates pass. The
frozen R18–R35 predecessor trees are read-only.
