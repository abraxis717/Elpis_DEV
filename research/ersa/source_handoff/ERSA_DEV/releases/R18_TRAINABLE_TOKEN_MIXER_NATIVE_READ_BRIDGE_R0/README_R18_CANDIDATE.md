# ERSA R18 — Trainable model-compatible PyTorch token mixer (research-only)

## Purpose

R11–R14 qualified bounded C-native SSC state ownership, native feature scoring,
select/continue/halt, and recurrent read mechanics. R17 R1 independently qualified
a CPU GDN-v1 structural *reference*, not official FLA. This phase builds the
first **trainable PyTorch ERSA token-mixing layer** while keeping the R11
state-bank sources and all R17 R1 predecessors completely frozen.

## Interface and model integration

`integration/ersa_trainable_layer_r18.py` exports:

- `ERSAState(keys,goals,values,metadata,issued,allowed,tick,generation)` —
  explicit, bounded, batch-indexed *host-authorized* relational memory. Fixed
  tensors, no hidden global store; `allowed` is mandatory and strictly boolean.
- `ERSATokenMixer(d_model,max_slots,max_hops,max_work_elements)` —
  `forward(x[B,T,D],state,goal=None,direction=None,mode="soft") -> (y,state)`.
  `return_aux=True` returns differentiable per-hop candidate probabilities,
  stop probabilities, selected reads, confidence, path indices, and frozen
  R7-compatible eight-feature scores.
- A differentiable 8-weight relation selector, 6-weight halt head,  D×D output
  projection, and mix gate. **79 trainable weights in the D=8 mixer**; the
  task-specific four-class decoder adds 36, totaling **115 in the model**.
- The soft controller makes a bounded, per-token multi-hop weighted read with
  soft stop/continue; `hard` is only a diagnostic route, **not a replacement**
  for R14's authoritative C recurrence.
- `integration/ersa_native_snapshot_r18.py` reads frozen R11 C-owned exact
  entities under the pinned owner lock, without calling `read_exact` (which
  would advance the generation). A *caller-provided* admission mask and host
  metadata are required. It fails closed on missing admitted entities or
  changed generation. This does NOT establish a writable C↔PyTorch API.

## Memory/runtime boundary

No runtime NAND/NVMe inference writes, no mutating model weights during
inference, no per-token persistence. All mutable runtime memory is volatile.
`max_slots<=8192`, `max_hops<=16`, and a configurable `B*T*N*D*H` work bound.
No automatic changes to the canonical Elpis worktree. The ZIP is the only
handoff artifact. All compile/test artifacts stay in `/dev/shm`.

## Scientific experiment

Offline training of a tiny `ERSATokenMixer + nn.Linear` classifier, with
256 synthetic training episodes and 128 disjoint synthetic held-out episodes
per run, 30 epochs. One 1-hop run and two 2-hop runs. The supervision includes
correct selection indices and halt targets (offline only); labels are NEVER
provided to forward during evaluation. The same test data are given to a
strong hand-coded key+goal baseline and a deliberately limited key-only
baseline. All training happens in RAM; no checkpoint is written.

The 79-weight ERSA mixer plus 36-weight task decoder reaches 128/128 classifications on the one-hop run and
127/128 on each two-hop run. **Exact two-hop route** remains 115/128 and
117/128. The strong deterministic baseline produces 125/128 and 120/128
exact two-hop routes on the same held-out cases: **no learned superiority**.

No tokens, language modeling, open-world semantics, GPU, full FLA GDN layer,
or fully matched model-level GDN comparison are qualified. This is a real
trained token-mixing *component* with explicit relational inputs, not a
production-ready GDN replacement. The existing C-native snapshot bridge is
CPU-only, and the training path is separate from R14's C hard-inference path.

## Gates

- Frozen R17 R1 manifest (122 entries) / SSC (7) / predecessor R2 (4).
- Existing R17 R1 80/80 bitwise read-only state probe.
- Entire inherited regression + R18 tests, including native R7/R12 score
  differential, native R11 snapshot generation immutability, prohibited-mask
  information and gradient isolation, finite bounds and object reuse.
- 3 independent scientific training/eval runs, including strong and key-only
  controls.  Two separate subprocess receipts are required before PASS.

Run the provided wrapper under `/dev/shm` with `PYTHONDONTWRITEBYTECODE=1`,
`TMPDIR=/dev/shm`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `CUDA_VISIBLE_DEVICES=-1`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Each gate takes only verified
frozen source and pinned local tests. Python+PyTorch+NumPy+pytest+cc required.

Disposition: `PASS_MODEL_COMPATIBLE_BOUNDED_TRAINABLE_READ_ONLY_ERSA_LAYER_ONLY`

Still HOLD: native hard-recurrence parity with the trained Torch controller,
full live C bank integration/admission authority, open-domain semantic
selection, matched GDN comparison, production trust and OS-wide NVMe writes.
