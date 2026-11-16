# Elpis Torch R2 — ERSA: design notes and evidence

Research only. Nothing here is adopted, qualified as a native kernel, or a claim about language-model quality.
All numbers are CPU (PyTorch 2.15 CPU, float32 models), on tiny models (about 85k to 110k parameters) and
synthetic tasks. Records: `research/elpis_torch/results/r2/`. Provenance and licences:
`research/elpis_torch/PROVENANCE.md` (section R2).

## 1. What was built

| Piece | File | Status |
|---|---|---|
| Gated delta-rule kernel `S' = r S - (r e)(S k)k^T + w v k^T`, read `o = S' q` | `ersa.py` `delta_step`, `read_cells` | implemented, tested |
| Trainable multi-head recurrent mixer (q/k/v projections, hard-sigmoid retain/erase/write gates, unit keys, RMS read norm, output gate, optional causal local convolution) | `ersa.py` `ERSAMixer` | implemented, tested |
| Exact addressable bank (write, overwrite, add, clear, halve, read, multi-cell reduce, reset) | `ersa.py` `ERSACells` | implemented, tested bit-exact |
| Compiled mixer: weights computed (not learned) so the mixer's own forward pass is exact RAM | `ersa.py` `compile_cell_mixer` | implemented, tested bit-exact |
| ERSA + CGC language model, optional bounded window attention (R1 ring, MHA or GQA) | `ersa_model.py` `ErsaLM`, `ErsaState` | implemented, tested |
| Experiments | `experiments/run_r2.py`; ERSA variants in `experiments/run_r1.py` | implemented |
| Tests | `tests/research/elpis_torch/test_ersa.py` (35), isolation gates | 286 passed in the full research suite |

Two policies share one kernel. The **learning** policy (dense unit-norm keys, soft gates, tied erase) is what
gradient descent trains. The **exact** policy (one-hot keys, saturated gates, erase independent of write) makes
the same state transition exact cell arithmetic. They are the same recurrence with different signals.

Block: `u = RMSNorm(x); x = x + ERSA(u) [+ WindowAttention(u)]; x = x + Channel(RMSNorm(x))`, channel being
FFN, CGC or FFN + gate * CGC as in R0/R1. The last block's CGC sits after the last mixer, so changing its
weight invalidates no ERSA state and no window cache (tested: `last_block_logits` equals a full recompute).

## 2. Measured

### 2.1 Exact addressable state (`exact.json`)

* A marker cell written once is read back **bit for bit after 1,048,576 further instructions** executed through
  the compiled mixer's forward pass. State is 256 bytes before and after; all values finite; 25.7k
  instructions/s on CPU.
* Tests cover overwrite, accumulate, halve, clear, multi-cell reductions, counters/registers, independent
  sessions and reset, each compared against a Python integer reference position by position.
* Not built: arithmetic beyond add/halve, branching, a program counter, an instruction compiler.

### 2.2 Recall capacity of the learning policy (`exact.json`, random unit keys, one head)

| load = items / key_dim | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| key_dim 16 | 1.00 | 0.88 | 0.62 | 0.44 | 0.16 |
| key_dim 32 | 1.00 | 1.00 | 0.75 | 0.36 | 0.15 |
| key_dim 64 | 1.00 | 0.94 | 0.72 | 0.35 | 0.16 |

With random (not learned, not orthogonal) keys, recall is reliable only up to about a quarter to a half of the
key dimension. This is the capacity limit of the dense policy and it bounds everything in 2.4.

### 2.3 Inference cost after an 8,192-token prefix (`latency.json`)

| Architecture | tokens/s | context bytes | grows with stream |
|---|---|---|---|
| attention baseline (dense KV cache) | 764 | 4,456,448 | yes |
| R1 local window 64 | 1,545 | 4,227,136 | no |
| ERSA | 2,031 | 8,192 | no |
| ERSA + conv | 1,315 | 9,728 | no |
| ERSA + conv + window 16 | 734 | 17,988 | no |
| ERSA + conv + GQA window | 726 | 42,756 | no |

Single-stream, tiny models, unoptimized token-serial kernel. The comparison that carries is the context size:
ERSA state is constant and about 500 times smaller than the dense cache at 8K tokens.

### 2.4 Long-range retrieval (R1 task; `near_*.json`, `ladder_ersa_conv.json`)

**Checkpoints are partial**: training was cut by worker restarts at 750 steps (ersa, ersa_conv) and 500 steps
(ersa_window, ersa_gqa) of a planned 2,500. Accuracy of the `direct` probe (chance 0.125), trained at 384
tokens:

| tokens | 384 | 768 | 1,536 | 3,072 | 6,144 |
|---|---|---|---|---|---|
| ERSA | 0.99 | 1.00 | 1.00 | 0.99 | 0.90 |
| ERSA + conv | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| ERSA + conv + window 16 | 0.99 | 1.00 | 1.00 | 0.98 | 0.99 |
| ERSA + conv + GQA window 512 | 1.00 | 1.00 | 0.97 | 0.70 | 0.20 |
| R1 dense attention (2,500 steps) | 1.00 | 0.84 | 0.65 | 0.45 | 0.20 |
| R1 local window | 0.80 | – | – | – | 0.08 |

ERSA without any global attention extrapolates 16x beyond its training length where the R1 attention
topologies do not. `assoc` and `hop2` follow `direct`; **`hop1` is weak everywhere (0.25 to 0.91)**.

Longer, ERSA + conv, 4 streams, state 38,912 bytes constant, 26k to 30k tokens/s:

| tokens | 8,192 | 32,768 | 131,072 | 262,144 |
|---|---|---|---|---|
| direct | 0.92 | 0.65 | 0.35 | 0.37 |
| assoc | 1.00 | 0.83 | 0.37 | 0.46 |

### 2.5 Projection to 1,048,576 tokens (`projection.json`, PROJECTED, not measured)

| Configuration | GPU-resident MiB | within 1 GiB target |
|---|---|---|
| ERSA only | 140.3 | yes |
| ERSA + 4,096-token window | 332.3 | yes |
| ERSA + window + compressed long-range index | 524.3 | yes |

ERSA cells are 12 MiB; the rest is a per-chunk scratch estimate and the optional window/index. No GPU run
exists. CPU-side state (token history, checkpoint) is accounted separately in the record.

## 3. Failures and limits (explicit)

1. **Retrieval decays with length**: above chance but poor beyond about 32K tokens (0.35 at 131K). The stream
   keeps overwriting a state whose dense capacity is small (2.2). 1M-token *retrieval* was not run for ERSA;
   only 1M-step *exact-policy* retention is measured.
2. **GQA-window variant collapses with length** (0.20 at 6,144; `assoc` 0.02). Unexplained; possibly the model
   leans on the 512-token window that covers the whole training length and never learns to use ERSA.
3. **`hop1` weak** in every ERSA variant.
4. **ERSA without the convolution does not learn the R0 transduction task** (stays at chance); with the
   convolution it reaches 1.00 on single-step items in 600 steps (test). The full transduction comparison
   (`run_r2 transduction`, 5 architectures) did not finish: only the attention baseline's first seed completed
   (val 0.849) and no record was written.
5. Trainings stopped at 20 to 30 percent of the planned steps; no seeds beyond the first; no variance.
6. Not implemented: chunk-parallel training kernel (the recurrence is token-serial, so training is slow), a
   volatile fast-weight session class for `ErsaLM` (the invariant it needs is tested), learned exact addressing
   (the exact policy is compiled, not learned), GPU execution.

## 4. Gates

| Gate | Status |
|---|---|
| Kernel correctness, gates, bounded state, finite gradients | PASS (tests) |
| Chunked training equals incremental inference; continuation parity | PASS (tests) |
| Exact cells and compiled mixer, bit-exact; 1,048,576-step retention | PASS (measured) |
| Independent sessions, reset, zero writes to the checkpoint | PASS (tests) |
| ERSA archive unchanged; no third-party code imported | PASS (tests) |
| Constant context memory | PASS (measured to 262,144 tokens) |
| 1M-token GPU memory within 1 GiB | UNVERIFIED (projection only) |
| Long-range retrieval at 1M tokens | NOT MET (decays by 32K; 1M not run) |
| ERSA+CGC versus attention baseline on a trained task | INCOMPLETE |

## 5. Reproduce

```bash
python -B -m research.elpis_torch.experiments.run_r2 smoke
python -B -m research.elpis_torch.experiments.run_r2 exact --out /tmp/exact.json
python -B -m research.elpis_torch.experiments.run_r2 latency --out /tmp/latency.json
python -B -m research.elpis_torch.experiments.run_r2 projection --out /tmp/projection.json
python -B -m research.elpis_torch.experiments.run_r2 transduction --out /tmp/transduction.json
python -B -m research.elpis_torch.experiments.run_r1 train --topology ersa_conv --out /tmp/ersa_conv.ckpt
python -B -m research.elpis_torch.experiments.run_r1 ladder --ckpt /tmp/ersa_conv.ckpt --out /tmp/ladder.json
```

Topologies for `run_r1 train`: `ersa`, `ersa_conv`, `ersa_window`, `ersa_gqa`.
