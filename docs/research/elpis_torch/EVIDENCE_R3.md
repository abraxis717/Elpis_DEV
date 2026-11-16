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
| `history_oracle_aux` (0) | oracle address + addressing loss | 1.00 | 1.00 |
| `history_noaux` (0) | learned selection, LM loss only | 0.04 (chance), recall@2 0.25 | 0.00 |
| `history` (0, 1, 2) | learned selection + addressing loss | 0.96-0.97 | 0.92-0.97 |

## 3. Attribution runs

Single seed each (`results/r3/facts_history_oracle_aux_s0.json`, `facts_history_noaux_s0.json`), same data and
budget as `history`:

| Run | what differs from `history` | direct, own mode, 1K / 4K / 16K / 65K / 262K | learned mode of the same weights | recall@2 |
|---|---|---|---|---|
| `history_oracle_aux` | trained on oracle reads, addressing loss kept | 1.00 / 0.96 / 1.00 / 1.00 / 1.00 (oracle) | 0.84 / 0.69 / 0.73 / 0.82 / 0.75 | 0.97 at 1K, 0.71-0.79 beyond |
| `history_noaux` | no addressing loss | 0.13 at 16K (learned; constant prediction) | | 0.25 (training), 0.02 at 16K |
| `history_oracle` (2 seeds) | trained on oracle reads, no addressing loss | 0.07 / 0.26 at 1K (oracle) | 0.07 / 0.10 | |

What this attributes:

* **The supervised addressing loss is the ingredient that makes the history path learnable here.** With it, both
  learned and oracle-read training succeed within 200-400 steps; without it, neither learned selection (recall stays
  near the random-pick level) nor the oracle-read readout learn in 1,200 steps (one oracle seed starts late:
  0.26 at the end, evidence read 0.67, output not yet mapped).
* **The end-to-end gradient through the selection prior does not bootstrap addressing** in this budget: the hard
  top-K selection is the bottleneck the directive warned about; R3 does not claim otherwise.
* **The index trained by the addressing loss generalizes to the readout**: `history_oracle_aux` was never trained
  on its own selections, yet in learned mode it reaches 0.69-0.84.

## 4. P2: attention placement, window size, FFN control

Single seed each unless stated; 2 blocks, so the 3:1 ERSA:attention ratio cannot be built at this size (the
attention-assisted block is block 1 of 2, i.e. 1:1); the scaling configurations use 3:1. Final validation direct
accuracy at 1,024 tokens (`training_records.json`) and ladder records where run:

| Architecture | params | 32 keys, 1,024 tokens: direct (val) | ladder |
|---|---:|---|---|
| `ersa` (ERSA + CGC, no convolution) | 113,560 | 0.04 | chance to 262K |
| `ersa_conv` (3 seeds) | 114,072 | 0.04 / 0.07 / 0.06 | chance to 1M |
| `ersa_conv_local128` (window 128, 2 query / 1 KV head) | 126,360 | 0.04 | chance to 262K |
| `ersa_conv_local512` | 126,360 | 0.04 | not evaluated (chance in training) |
| `ersa_conv_local1025` (window covers the training length) | 126,360 | 0.04 | chance to 262K |
| `history` (3 seeds) | 173,343 | 0.975 / 0.965 / 0.975 | section 2 |
| `history_ffn` (SwiGLU 107 instead of CGC, 20,544 vs 20,480 channel parameters) | 173,535 | 0.96 | 0.96 / 0.71 / 0.73 / 0.84 / 0.66 (1K-262K); oracle 0.96-1.00 |
| `history_local128` (history + window 128) | 185,631 | 0.04, recall@2 0.40 | chance to 16K |
| 8 keys, 512 tokens: `ersa_conv` / `local128` / `local512` | | 0.07 / 0.17 / 0.09 | 0.05-0.15 at 1K-64K |

Findings:

* **No ERSA-only or ERSA+window model learned the two-token binding task in 1,200 steps**, at 32 keys or at 8 keys
  and 512 tokens. All of them converged to the same constant (unigram) prediction: their validation numbers are
  identical to three decimals. This is a learnability failure under this budget and task, NOT a demonstration of
  ERSA's capacity limit (R2's ERSA learned the one-token version of the task). The capacity question is therefore
  not answered by these baselines.
* **The R2 window-collapse question is NOT TESTABLE here**: the window variants learned nothing at the training
  length, so there is no accuracy to lose with distance. The pre-registered P2 gate is reported as NOT_RUN
  (inconclusive), not as supported or refuted.
* **CGC versus FFN with history: no measurable difference** on this task at this size (CGC seeds 0.74-0.84 at 4K,
  FFN 0.71; at 262K CGC 0.65-0.81, FFN 0.66). With history switched off the FFN model collapses further (0.00-0.20)
  than the CGC models (0.13-0.59). One FFN seed; not a ranking.
* **History + window attention failed** (`history_local128`). The window forces training in 127-token chunks; index
  keys of blocks from earlier chunks are stored DETACHED, so the addressing loss shapes only the keys of the ~4
  blocks completing in the current chunk. This is the truncated-gradient limitation stated in `DESIGN_R3.md`
  section 4, measured. The bounded next step is to recompute resident block keys with gradient during training
  (they are a function of token ids), not to add attention.

## 5. P5: long-form generation (program task)

Structured program language (`program_tasks`), header of about 1,024 tokens declaring 24 functions and 24
variables with revisions and comments, body of statements; models trained 1,200 steps on 1,792-token programs.
Free-running: the header is fed, then 2,048 tokens are generated through the bounded state and checked by the
exact checker (grammar, defined references, arity, printed value = current value; the model's own LET/DEF
revisions are honoured). 8 streams per row (`results/r3/program_*.json`). Chance: arity about 1/4, printed value
1/16.

| Model | teacher-forced: structure / call ends / call continues / print value | free-running, header 1,024, T=0.7: valid / arity / print / violations per 1K tokens | header 4,096, T=0.7 |
|---|---|---|---|
| `ersa_conv` | 0.71 / 1.00 / 0.00 / 0.07 | 0.30 / 0.24 / 0.06 / 146 | 0.28 / 0.25 / 0.05 / 149 |
| `history` + program addressing labels | 0.71 / 1.00 / 0.00 / 0.07 | 0.27 / 0.22 / 0.05 / 158 | 0.42 / 0.24 / 0.06 / 121 |
| `history` without labels | 0.71 / 1.00 / 0.00 / 0.05 | 0.54 / 0.26 / 0.07 / 94 | 0.85 / 0.22 / 0.06 / 30 |

Greedy decoding (T=0) degenerates into one repeated CALL of one function with one argument (arity 0.00-0.18,
no PRINT statements; `distinct_functions_called` 1-2). The higher "valid" fractions at T=0.7 come from choosing
statements that cannot be wrong (LET, DEF, NOTE): the unlabelled history model at header 4,096 wrote only 48 CALLs
and 27 PRINTs in 2,048 tokens. Arity and printed-value accuracy, the measures that need long-range information,
are at chance for every model.

**P5 long-generation result: FAIL.** No model learned the determined decisions. Diagnosis (measured on
`program_history_aux`): the index does not learn on this task (recall@2 0.20, addressing loss 3.0 against about
3.9 for uniform choice), although 96-99 % of the labelled decisions have an eligible source block. Unlike the
facts task, the deciding symbol (`FN_i`, `VAR_j`) recurs in many non-deciding statements, so the index has to
match the bigram `DEF FN_i` / `LET VAR_j` rather than the symbol; counting arguments additionally needs state the
2-block model did not learn. The benchmark, the checker and the free-running harness work; the models do not.

## 6. P3: execution, device and precision

| Check | Result | Status |
|---|---|---|
| `chunked_scan` = serial recurrence, float64, blocks 1/7/64/200, with exact wipes (`r = 0`), no-write and full-erase steps | max error <= 1e-12 (values and final state) | PASS (test) |
| gradients of `chunked_scan` = serial, float64, all seven inputs | max error <= 1e-11 | PASS (test) |
| float32 parity, 256 positions | <= 1e-4 | PASS (test) |
| chunked-kernel `ErsaLM` = serial-kernel `ErsaLM`, with and without window attention | <= 1e-10 | PASS (test) |
| training speed, forward + backward, B=24, T=384, 4 heads, 16x16, 4 CPU threads | serial 0.79 s, chunked 0.068 s (block 64), 0.077 s (32), 0.146 s (128) | MEASURED, PERFORMANCE_ONLY |
| inference after the R3 change, ERSA+conv, 1 thread | 34K - 53K tokens/s per stream through 1,048,576 tokens (chunk 1,024) | MEASURED, PERFORMANCE_ONLY |
| every state tensor on the model's device (ERSA cells, tails, window rings, window token ring, token history, index keys and ids) | all on `meta` when the model is on `meta` | PASS (test; `meta` stands in for CUDA) |
| bfloat16 autocast (CPU): ERSA state stays float32, output finite, within 5 % of float32 max | output error 1.6 %, state error 0.6 % of max | PASS (test) |
| chunk / incremental parity of `ErsaHistoryLM` past the horizon (chunks 64, 13, 1; window and none) | logits <= 1e-10, selections identical | PASS (test) |
| last-block CGC change: index, token history, ERSA state, window untouched; `last_block_logits` = full recompute | <= 1e-10 | PASS (test) |
| CUDA parity, bf16 on CUDA, GPU memory at 1M | `run_r3 gpu` | NOT_RUN (no CUDA device here) |

The R2 defect "recurrent state may not follow a model moved to CUDA" is corrected in code: `ERSAMixer.new_state`,
`new_tail`, `StreamState` (all rings and its index arithmetic), `ErsaState`, `TokenHistory`, `BlockIndex` and every
index tensor of the history path are created on the parameters' device. It is verified only on `meta`.

## 7. P4: 500M and 1B configurations (INSTANTIATED on `meta`; untrained)

| | `r3_500m` | `r3_1b` |
|---|---:|---:|
| d_model x layers | 1,280 x 24 | 2,048 x 20 |
| ERSA heads x key x value | 10 x 128 x 128 | 16 x 128 x 128 |
| CGC bottleneck / cells / units | 512 / 2,560 / 4 | 768 / 4,096 / 4 |
| window attention layers (every 4th: 3 ERSA-only : 1 attention-assisted) | 3,7,...,23 (6) | 3,7,...,19 (5) |
| query / KV heads, head dim, window | 10 / 2, 128, 4,096 | 16 / 2, 128, 4,096 |
| history reads (layers), block, K, index dim x keys per block, encoder width | 11 and 23, 64, 2, 64 x 2, 256 | 9 and 19, 64, 2, 64 x 2, 256 |
| vocabulary (future subword tokenizer) | 65,536 | 65,536 |
| **instantiated parameters** | **492,765,430** | **980,182,246** |
| embedding (tied) / ERSA / CGC / attention / history | 83.9M / 158.3M / 220.2M / 23.6M / 6.7M | 134.2M / 337.7M / 450.9M / 47.2M / 10.1M |
| weights bf16 / fp32 | 0.92 GiB / 1.84 GiB | 1.83 GiB / 3.65 GiB |

Context memory of one 1,048,576-token stream (state INSTANTIATED, scratch PROJECTED; 16-bit activations,
float32 ERSA state):

| Item | `r3_500m` | `r3_1b` | Placement |
|---|---:|---:|---|
| ERSA cells (fp32) | 15.0 MiB | 20.0 MiB | GPU |
| conv tails | 0.35 MiB | 0.47 MiB | GPU |
| window KV (4,096 tokens) | 24.0 MiB | 20.0 MiB | GPU |
| block index (2 reads x 16,384 blocks) | 8.1 MiB | 8.1 MiB | GPU |
| index scores per 512-token chunk (fp32) | 64 MiB | 64 MiB | GPU scratch |
| reread blocks + readout KV per chunk | 128 MiB | 128 MiB | GPU scratch |
| encoder activations per chunk | 320 MiB | 320 MiB | GPU scratch |
| chunk activations | 20 MiB | 32 MiB | GPU scratch |
| last-block CGC fast weights + momentum (fp32) | 40 MiB | 96 MiB | GPU |
| **GPU context total** | **619.5 MiB** | **688.6 MiB** | target 1 GiB: within; ceiling 2 GiB: within |
| token history (int32) | 4.0 MiB | 4.0 MiB | CPU |
| **CPU + GPU context total** | **623.5 MiB** | **692.6 MiB** | |
| for comparison: dense KV cache at 1M, every layer | 24 GiB | 20 GiB | |

Measured on this CPU: the experiment model's history state after streaming 1,048,576 tokens is 12,854,944
bytes (token history 4,194,464, index 8,650,752, ERSA 9,728), identical before and after and equal to the
formula (`results/r3/scale.json`).

Compute per decoded token at the full horizon (PROJECTED multiply-accumulates): `r3_500m` 491.7M weights + 15.7M
ERSA + 62.9M window + 4.2M index scoring + 0.5M readout + 331.9M rereading (uncached) = 906.9M; `r3_1b` 1,483M
(of which 394.8M rereading). Rereading is the largest added term: with the default d_model-wide encoder it
would exceed the model itself (5.1G at 500M), which is why the scaling configurations use `encoder_dim=256`; a
bounded cache of encoded blocks (not built) would remove most of it at decode. Index scoring is linear in
`horizon / block` (4.2M at 1M) and is not the bottleneck at this geometry. Prefill shares rereads across a chunk.

Training memory (PROJECTED, batch 1, 4,096 tokens, AdamW fp32 states, no activation checkpointing): `r3_500m`
18.2 GiB (fits 24 GiB), `r3_1b` 28.9 GiB (does NOT fit without checkpointing, 8-bit optimizer states or
sharding). No 500M/1B model was trained or run: they are architectures, not language models.

Workstation qualification plan (`run_r3 gpu`, each step with its pass condition):
1. `ELPIS_REQUIRE_TORCH=1 pytest tests/research/elpis_torch`: all pass, the CUDA test included.
2. `run_r3 gpu --ckpt <history checkpoint>`: float32 CPU/CUDA max logit difference <= 1e-4 and selection
   agreement >= 0.999; bf16 autocast finite, ERSA state float32.
3. Same command: `r3_500m` streamed to 2^20 tokens; context peak flat (<= 5 % growth) from 2^18 to 2^20 and
   <= 1 GiB (target) / 2 GiB (ceiling). Report the measured peak beside the projection above.
4. Train `history` seeds 0-2 on the GPU with `run_r3 train` and rerun `run_r3 facts`; compare with the CPU records.
5. No swap: run with swap off and record `free -m` before and after.


## 8. Gates

Pre-registered in `results/r3/GATES.md`; the directive's twelve acceptance criteria follow.

| Gate | Result | Evidence |
|---|---|---|
| G5 oracle readout at short length (>= 0.90 at 1K and 4K, every seed) | PASS | 0.98 / 0.98 / 0.98 and 0.99 / 0.97 / 0.98 |
| G6 learned beats recurrent-only by >= 0.10 at 4K and 16K, every seed pair | PASS, with a caveat | 0.74-0.84 vs <= 0.07; 0.71-0.83 vs <= 0.13. The baseline never learned the task (section 4) |
| G7 oracle readout after long gaps (>= 0.80 at 262K, every seed) | PASS | 0.97 / 0.88 / 0.96 |
| G7b learned retrieval at 1M (>= 0.50, every seed) | FAIL | 0.76 / 0.31 / 0.41 (one stream, about 29 probes each) |
| G8 local regression | reported | history-off 0.13-0.59; recent facts (< 64 tokens) 0.93-1.00 in seed 0; the ERSA baselines were not a usable local reference (they learned nothing) |
| P2 window collapse | NOT_RUN (inconclusive) | no window variant learned at the training length |
| P5 free-running program generation | FAIL (reported, no threshold) | section 5 |

Directive acceptance criteria:

| # | Criterion | Status |
|---|---|---|
| 1 | future-token leakage zero | PASS (tests: changing tokens after p changes no logit or selection at or before p; selected blocks complete at their query) |
| 2 | raw history read / eviction / addressing invariants | PASS (tests, beyond 1M positions) |
| 3 | chunked vs incremental parity | PASS (<= 1e-10 float64, equal selections, chunk 64/13/1) |
| 4 | state and index budgets enforceable and measured | PASS (fixed at allocation; measured constant to 1,048,576 tokens, equal to the formula) |
| 5 | correct-source oracle readout functional at short length | PASS (G5) |
| 6 | learned addressing better than recurrent-only, controlled | PASS with the G6 caveat |
| 7 | evidence access after long gaps measured apart from hit rate | PASS (recall@K, evidence-token read and accuracy given selection reported separately at every length) |
| 8 | no unacknowledged regression in local modeling | PARTIAL (reported; no baseline that learned the task) |
| 9 | device/dtype correctness and mixed-precision boundaries | PARTIAL (meta-device placement and CPU bf16 PASS; CUDA NOT_RUN) |
| 10 | no inference-state persistent writes | PASS (no writer in the new modules; the isolation gate registers `run_r3`'s explicit `--out` and checkpoint writers only) |
| 11 | existing research and repository tests unaffected | PASS locally (full Elpis Torch suite, boundary gates); CI: see the PR |
| 12 | model quality vs synthetic success vs numerical infrastructure kept apart | stated throughout; no language-model claim |

## 9. Failures and limits (explicit)

1. **1M learned retrieval FAILS the pre-registered gate**; addressing recall falls with length (0.95-0.97 at the
   training length, 0.60-0.78 to 262K, 0.31-0.51 at 1M) and readout given selection also degrades at 1M.
2. **Superseded facts**: learned addressing picks an older binding of the same key too often (contradicted 0.52-0.62
   at 262K vs 0.79-0.93 with the oracle address).
3. **Density beyond training**: 48 keys 0.62-0.63, 96 keys 0.31 (trained at 32).
4. **The history path needs the training-only addressing labels.** Without them nothing was learned in budget.
   Real text has no such labels; a label-free addressing objective (self-supervised: e.g. the readout's own
   evidence as the target of the index) is unbuilt and is the main open problem.
5. **ERSA-only and ERSA+window baselines learned nothing** on two-token bindings (8 or 32 keys). The comparison
   of G6 is against an untrained-in-effect baseline; ERSA's capacity limit was not measured by R3.
6. **History + window attention failed** because stored index keys are detached across training chunks.
7. **Program generation FAILS** (section 5).
8. Tiny models, one CPU, 1,200 steps; three seeds for the core comparison, one for every other variant. The 1M
   rows are one stream each. CPU throughput with the history path is 1.2-1.3K tokens/s at 1M (one thread,
   shared CPU), against 34-53K for ERSA alone.
9. No GPU execution; no ErsaLM fast-weight session; no hierarchical index; no encoded-block cache.

## 10. Recommendation (bounded next experiments, in order)

1. Train with resident block keys recomputed WITH gradient (they are a function of token ids) and with long
   sequences that put superseded bindings far apart; measure recall at 4K-1M again. This targets failures 1, 2
   and 6 without new subsystems.
2. Replace the supervised addressing labels with a label-free objective and repeat P0 (failure 4). Until this
   works, the mechanism is not applicable to unlabelled text.
3. Diagnose ERSA's failure on two-token bindings (initialization, gate biases, longer budget) before using it as a
   capacity baseline (failure 5).
4. Only then: the program benchmark at larger depth, and the GPU qualification (`run_r3 gpu`).
