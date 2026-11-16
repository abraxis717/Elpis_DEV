# Elpis Torch R3 — long-context addressing: evidence

Research only. CPU (PyTorch 2.15 CPU, 4 cores, float32 models), tiny models (about 114k parameters for the
recurrent baseline, 173k with the history path), synthetic tasks. Nothing here is a language-model claim, and
nothing is a GPU measurement. Design: `DESIGN_R3.md`. Records: `research/elpis_torch/results/r3/`. Gates were
declared before the multi-seed results: `results/r3/GATES.md` (commit `928dacf`).

Status words: PASS / FAIL / PARTIAL (a gate), MEASURED (this CPU), INSTANTIATED (exact shapes on `meta`),
PROJECTED (formula), NOT_RUN.

## 1. Setup

Model (all architectures): d_model 64, 2 blocks, ERSA 4 heads x 16 x 16 (chunked kernel), CGC channel
(bottleneck 32, 96 cells, 4 units), tied embeddings. `ersa_conv` adds a 4-tap causal convolution. `history` =
`ersa_conv` + one history read in block 1: block 32 tokens, overlap 8, top-2, index 2 keys x 32 per block, readout
2 query heads / 1 KV head x 32, encoder at d_model. 114,072 parameters (`ersa_conv`) and 173,343 (`history`); training took 1,481 s and 2,568 s (seed 0, one thread, shared CPU; PERFORMANCE_ONLY).

Task (`longrange_tasks`, chance 1/16 on values): 2-token facts `KEY_k VAL_v` over 32 keys with rebinding
(contradictions), aliases, links, direct and associative questions, 2-hop and 3-hop chains, 112 events per 1,024
tokens. Evaluation adds 6 planted facts (begin/middle/end, asked at the end) and keeps 112 events at every
length, so distance grows with length. Training: 1,200 steps, batch 16, 1,024 tokens, AdamW 3e-3 with warmup and
cosine decay, clip 1.0, identical data stream per seed for every architecture; `history` adds the training-only
addressing loss (weight 1). Disjoint seed domains for train / validation / test.

Reproduce:

```bash
python -B -m research.elpis_torch.experiments.run_r3 train --arch history --task facts --seed 0 --out h0.pt
python -B -m research.elpis_torch.experiments.run_r3 facts --ckpt h0.pt --out h0_facts.json
python -B -m research.elpis_torch.experiments.run_r3 train --arch history --task program --seed 0 --out p0.pt
python -B -m research.elpis_torch.experiments.run_r3 program --ckpt p0.pt --out p0_program.json
python -B -m research.elpis_torch.experiments.run_r3 scale --out scale.json
python -B -m research.elpis_torch.experiments.run_r3 gpu      # workstation only; NOT_RUN without CUDA
```

## 2. P0: recurrent-only, oracle-selected and learned-selected evidence

Direct-question accuracy, three seeds each (seed 0 / 1 / 2), `results/r3/facts_*.json`:

| tokens | `ersa_conv` (recurrent only) | `history` learned | `history` oracle address | `history` history off | recall@2 (learned) | evidence token read (learned) | accuracy given source block selected |
|---:|---|---|---|---|---|---|---|
| 1,024 | 0.07 / 0.04 / 0.07 | 0.97 / 0.96 / 0.96 | 0.98 / 0.98 / 0.98 | 0.56 / 0.35 / 0.34 | 0.97 / 0.95 / 0.96 | 0.97 / 0.92 / 0.96 | 0.99 / 1.00 / 1.00 |
| 4,096 | 0.03 / 0.03 / 0.07 | 0.84 / 0.74 / 0.79 | 0.99 / 0.97 / 0.98 | 0.56 / 0.37 / 0.32 | 0.72 / 0.60 / 0.68 | 0.64 / 0.55 / 0.60 | 0.96 / 0.98 / 0.94 |
| 16,384 | 0.13 / 0.06 / 0.06 | 0.83 / 0.71 / 0.77 | 1.00 / 0.98 / 0.98 | 0.59 / 0.38 / 0.36 | 0.75 / 0.65 / 0.70 | 0.68 / 0.59 / 0.62 | 0.99 / 0.99 / 0.96 |
| 65,536 | 0.09 / 0.09 / 0.00 | 0.95 / 0.79 / 0.82 | 1.00 / 0.95 / 0.91 | 0.51 / 0.39 / 0.28 | 0.78 / 0.67 / 0.76 | 0.75 / 0.62 / 0.67 | 1.00 / 0.98 / 0.88 |
| 262,144 | 0.09 / 0.04 / 0.06 | 0.81 / 0.65 / 0.76 | 0.97 / 0.88 / 0.96 | 0.53 / 0.13 / 0.18 | 0.77 / 0.63 / 0.69 | 0.63 / 0.56 / 0.58 | 0.98 / 0.86 / 0.94 |
| 1,048,576 | 0.03 / 0.07 / 0.07 | 0.76 / 0.31 / 0.41 | NOT_RUN | NOT_RUN | 0.51 / 0.31 / 0.48 | 0.41 / 0.26 / 0.39 | 0.86 / 0.67 / 0.63 |

Probe counts: 483, 257, 128, 57, 68, 29 direct probes per seed at the six lengths (16, 8, 4, 2, 2, 1 streams).
The 1M row is one stream: about 29 probes, wide uncertainty. Associative, 2-hop and 3-hop probes follow the same
pattern (records); planted facts are 0.68-0.86 at 1,024 and 0.67-1.00 at 262,144 in learned mode (12 probes per
length beyond 16K), and 0.83-1.00 at 1M (6 probes).

Attribution (the directive's decision gate):

* **Readout works when the right block is supplied.** With the oracle address the same weights answer 0.88-1.00
  up to 262,144 tokens; when the learned selection contains the source block, accuracy is 0.86-1.00 at every
  length to 262,144. The readout's argmax lands on the exact evidence token in 0.91-1.00 of oracle-addressed
  probes.
* **The loss with length is addressing.** recall@2 falls from 0.95-0.97 at the training length to 0.60-0.78
  between 4K and 262K and to 0.31-0.51 at 1M; accuracy when the block was NOT selected is 0.30-0.50.
* **Superseded facts are the addressing weak spot.** Contradicted direct probes: 0.52-0.62 learned vs
  0.79-0.93 with the oracle address at 262,144 (records, `contradicted`). Content scores of an old and a new
  binding of the same key are equal; the learned log-distance slope does not separate them reliably.
* **At 1M the readout also degrades** (accuracy given selection 0.63-0.86, down from 0.86-1.00); the oracle mode
  was not run at 1M (bounded by `--diagnosis-max 262144`).
* **History off**: the same model without its read answers 0.13-0.59: it relies on the history path for most
  probes; its ERSA covers recent facts (distance < 64: 0.93-1.00 in the records).

Fact density at 1,024 tokens (trained at 32 keys; learned direct / recall@2, seeds 0/1/2):
8 keys 0.86/0.85/0.82 (recall 0.92-0.93), 16 keys 0.92-0.94, 32 keys 0.96-0.99, **48 keys 0.62-0.63
(recall 0.76-0.78), 96 keys 0.31 (recall 0.57-0.58)**. Beyond the trained density the index degrades.
`ersa_conv`: 0.02-0.09 at every density.

Distractor-heavy streams (30 % distractor bindings of 32 never-asked keys): learned direct 0.72-0.82 at 4,096
and 0.72-0.85 at 16,384 (recall 0.57-0.70); `ersa_conv` 0.01-0.09.

### Oracle-trained readout and the role of the addressing loss

| Architecture (seed) | how the history path is trained | direct at 1,024, own mode | evidence token read |
|---|---|---|---|
| `history_oracle` (0) | oracle address, LM loss only | 0.07 (chance) after 1,200 steps | 0.03 |
| `history_oracle` (1) | oracle address, LM loss only | 0.26 after 1,200 steps (rising late) | 0.67 |
| `history_oracle_aux` (0) | oracle address + addressing loss | see section 3 | |
| `history_noaux` (0) | learned selection, LM loss only | see section 3 | |
| `history` (0, 1, 2) | learned selection + addressing loss | 0.96-0.97 | 0.92-0.97 |

## 3. Attribution runs

PENDING_ATTRIBUTION

## 4. P2: attention placement, window size, FFN control

PENDING_P2

## 5. P5: long-form generation (program task)

PENDING_P5

PENDING_P34

## 8. Gates

PENDING_GATES

## 9. Failures and limits (explicit)

PENDING_FAILURES
