# ERSA R15 — Gated-delta recurrence-rule baseline screen

**Target:** `PASS_FROZEN_ERSA_AND_GDN_RULE_SCREEN_MECHANICS_ONLY`.
**Never** interpret this as matched ERSA-versus-trained-GDN architectural parity.

R14 is frozen byte for byte (101 source files). The R15 experiment evaluates
ERSA's single-call native learned and strong deterministic controllers against
an **independently implemented sequential gated delta-rule recurrence** documented
by the MIT-licensed FLA reference (`fla/ops/gated_delta_rule/naive.py`).
Upstream reference: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py
The upstream FLA code is **not** redistributed in this archive.

R15 tests its own NumPy recurrence against an independently coded Torch recurrence
for randomized beta/decay sequences and nonzero initial state. It also checks
prefix/suffix continuation and invalid input rejection. This establishes arithmetic
parity of the rule only, not parity against a pinned FLA distribution or its neural
layer.

## Protocol

- Synthetic 24-record variable 1–4 hop chains, exactly the four frozen R14
  held-out families and seeds (64 cases per family).
- ERSA native C state read and single-call bounded recurrence: learned 14-weight
  controller and the same strong deterministic comparator.
- GDN-rule baseline: 8-dimensional **hand-designed key+goal** features and
  4-dimensional value vectors; 8×4 fp64 associative state; no learned
  projections, conv kernels, output normalization, output head, full GDN layer.
- Offline train split: 64 additional episodes are used to choose beta/decay by
  first-hop cosine, grid size 12. Held-out R14 cases never used for gate selection.
- Two GDN-rule variants: grid-selected beta/alpha and fixed beta=alpha=1.
- GDN query produces continuous vector; termination uses goal/key similarities;
  no integer relation identity or pointer oracle is passed to the comparator.
- Missing-goal, no-terminal cases do not count successful terminal emissions.
- Explicit fairness warning: GDN fixed associative memory uses 256 fp64 bytes;
  ERSA's 24×4 exact values alone cost 768 bytes, **excluding keys, goals, metadata,
  admission and native-state overhead**. Models have radically different
  trainable parameter budgets and different semantics; this is a diagnostic
  probe, not fair capacity/compute comparison.

## Gate and interpretation

Inherited 975 tests + 67 new GDN recurrence/BREAK tests, original 8192-capacity
stress, R14 science and R15 new science, and post-run SHA checks.
Runs from `/dev/shm`; frozen external R2 and SSC source paths must exist and
match pinned SHA256. No GPU, persistent checkpoints or intended NVMe writes.
The gate qualifies only GDN-rule screening mechanics. Full GDN training,
matched latent dimensions/memory budgets, true layer reference parity,
open-world semantic retrieval and neural architectural superiority remain HOLD.

## Next genuine architecture gate

Create a **trainable** CPU/GPU GDN layer using FLA MIT sources with pinned
provenance, train both systems using identical tasks and disclosure of
parameter, activation-state and compute budgets, and evaluate held-out
length generalization plus catastrophic interference. Until then, avoid any
claim that ERSA exceeds or matches GDN.
