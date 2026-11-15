ERSA R35: donor-backed trainable volatile slot write/read seam

Scope: one recurrent token-mixing layer only, not a language model.
FLA MIT naive GSA recurrent operation (adapted to remove einops dependency)
is contained in third_party/fla_gsa/naive.py with license. Its recurrent
math and dual matrix cache are from FLA, not a claimed original ERSA invention.
Raven supplies the sparse top-k write-routing design; r35/slot_write.py is a
new ERSA adapter, not the upstream RavenAttention implementation.
The existing frozen R18 ERSATokenMixer is the read mechanism.

Important limits:
- R35 state = two fixed float32 matrices hk/hv, explicit & volatile.
- No writes to R11 C state; this is a PyTorch training-reference write bank.
- Host admission controls writes and visible slots. No persistent writes.
- CPU PyTorch reference only; chunk parity tested through split continuation.
- NOT official FLA chunk-vs-fused kernel parity.
- No end-to-end utility advantage or model-level GDN parity claim.
- This donor seam is a research scaffold; replacing it with a native ERSA
  state-owner implementation remains a HOLD until tested.

Upstream: https://github.com/fla-org/flash-linear-attention
          https://github.com/goombalab/raven
Upstream sources are MIT licensed; adapted vendor file and license retained.

Persistent destination:
/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV/worktrees/R35_DONOR_BACKED_SLOT_WRITE_READ_R0
