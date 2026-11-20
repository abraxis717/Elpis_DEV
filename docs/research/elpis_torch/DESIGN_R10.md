# Elpis Torch R10 — design: CGC complementarity and in-context adaptation

**Question.** Does a CGC-augmented Elpis transformer adapt to new in-context relationships, and retain earlier
conflicting relationships, more reliably than otherwise equivalent FFN-only transformers, without inference-time
weight updates?

**Scope.** R10 is a falsification study. It adds a task generator (`adaptation_tasks.py`) and a runner
(`experiments/run_r10.py`). It changes no model code, and nothing under `src/` or `native/`.
- Plan: `PREREGISTRATION_R10.md` with `results/r10/preregistration.json`.
- Results: `EVIDENCE_R10.md`.

## 1. What CGC can and cannot do here

The PyTorch `CGCTransform` is stateless: a cubic feed-forward map with learned projections, inside routed experts
beside an FFN (`CLAIM_AUDIT_R9.md`, L3 and L15). It owns no addressable or mutable temporal memory.

Within one sequence, information moves only through:
- the ERSA recurrent state and its convolution tail;
- the window-attention KV;
- the activations of the current call.

The PLE's n-gram addressing sees the last two tokens. R5 history is off.

So CGC could help adaptation only by making those mechanisms easier to use: better key and value representations
for ERSA's delta rule, or better features for attention matching. It cannot be the memory. R10 measures whether it
helps, and attributes any help with matched controls and state resets, never with labels.

R97W's meta-learning (gradient steps on 120 parameters) and K1's retention (a bounded state) are different
mechanisms. Neither is evidence here.

## 2. The assay (`adaptation_tasks.py`)

**Episode format.** Byte streams of examples `R K V ␠`: regime, key, value, separator. The value is the answer; it
is scored at the key token before it is revealed (prequential, teacher-forced).

**Mappings and rules are drawn fresh per episode.** Nothing global predicts an answer:
- The best global static mapping scores 0.055 against 0.05 chance (tested).
- Train and test signatures are disjoint (tested).

| Family | Tests | k means |
|---|---|---|
| A, regime-conditioned mappings | 3 regimes × 4 shared keys, each key bound to a different value in every regime; Zipf-random binding frequencies | earlier exposures of the queried binding |
| B, switching transformations | 3 regimes, each a cyclic shift `value = key + c (mod 20)` with its own `c`; segments switch and re-enter; every query is on an input the regime has not shown | earlier examples of the queried regime |
| C, retention under interference | regime P binds 4 (2) keys, then Q and S rebind the same keys; probes ask P again with P's symbol; P is re-entered | earlier exposures of the binding |

**Context ladder (C).**
- `C_short`: the evidence is inside the 64-token attention window.
- `C_long`: it is outside the window (more than 64 tokens back) but in ERSA's reach.
- `C_gap`: as `C_long`, plus a 64-token distractor gap.

Family A also reports accuracy for evidence inside and outside the window.

**Held-out (never in training).**
- `B_heldout`: shift offsets 15–19.
- `C_gap128`: a 128-token gap, in 384-token episodes.
- `C_4reg`: three interfering regimes.

**Reference points.**
- Chance is 1/20.
- The oracle ceiling is 1.0 wherever an answer is determined. An executable exact-memory reader verifies it.
- The trivial recency strategy scores 0 on every post-interference probe, by construction.

## 3. Arms

Every arm is R9's d 64 backbone, R7's study model:
- two ERSA blocks (4 heads, 16 × 16 state per head, convolution 4);
- routed channels in both blocks (8 experts, top-2);
- R7's PLE.

Every arm is trained from the same seed on the same episode stream. Arms differ only where named:

| Arm | Routed expert | Attention | Role |
|---|---|---|---|
| A | FFN 86 + CGC | window 64, both blocks | the complementary design |
| B | FFN 86 | window 64 | CGC absent (incremental; not matched) |
| C | FFN 123 | window 64 | parameter-matched resource competitor |
| C_mac | FFN 126 | window 64 | multiply-add-matched resource competitor |
| D_A | FFN 86 + CGC | none (ERSA only) | attention-restricted, CGC-positive |
| D_C | FFN 123 | none (ERSA only) | attention-restricted, CGC-negative |

**Matching** (EXACT, `run_r10 plan`):
- C is within 0.022% of A's parameters, with 2.6% fewer routed multiply-adds.
- C_mac is within 0.26% of A's multiply-adds, with 0.18% more parameters.
- The attention parameters (24,576) are identical in A, B, C and C_mac.
- D_A and D_C are exactly R9's arms A and C.

**Why a 64-token window.** R9's backbone had no attention. The assay needs evidence both inside and outside a
window. 64 tokens hold 16 examples. The window must exceed the 32-token chunk.

**Why D is attention-free.** It is the extreme restriction: no demonstration can be attended to directly. Whatever
D_A and D_C retain, ERSA carries. That is never evidence of CGC-specific memory.

## 4. Training

The training is R9's schedule:
- AdamW, learning rate 3e-3, 50 warmup steps, cosine decay to 10%, weight decay 0.01, gradient clip 1.0;
- the PLE table under lazy Adam at 5× the rate;
- batch 16, 256-token episodes, 32-token chunks, full backpropagation through each episode;
- 3,000 steps;
- the cross-entropy on answer positions only (`IGNORE` elsewhere).

**The training mix** per batch:
- 6 family-A episodes;
- 5 family-B episodes;
- 5 family-C episodes (`C_short`, `C_long` ×2, `C_gap` ×2).

Step s, row j of seed n is `train_episode(n, s, j)`, so every arm of a seed sees the same episodes in the same
order. Initialization is a function of seed and parameter name, so shared parameters start identical.

## 5. Evaluation and diagnostics

**Frozen sets.** Every run is evaluated on frozen sets generated from split seeds that no training run uses:
- main: 256 episodes per variant;
- held-out: 256 per variant;
- diagnostic: 128 per variant.

Screening and confirmation use independent split seeds, so confirmation sees fresh mappings.

**Measures:**
- accuracy at k = 0, 1, 2, 4, 8, 16;
- A: inside versus outside the window;
- A: collision errors (another regime's value) and recency errors;
- B: before and after a switch, re-entry, and recovery after 1–3 examples;
- C: pre-interference, post-interference, middle-regime and re-entry accuracy, forgetting, recency errors;
- bits per answer (secondary).

**Diagnostics on every trained model** (causal dependence, not matched controls):
- CGC zeroed;
- ERSA state reset before the C probes;
- attention history reset before the C probes;
- support targets shuffled in A and B (expected: chance);
- regime identity removed in A, B and C;
- irrelevant demonstrations inserted before the probes;
- the latest conflicting binding re-shown before each probe.

**State integrity (tested):**
- independent sessions;
- reset and restart equivalence;
- no future-token influence;
- stable chunking;
- parameters byte-identical before and after evaluation, so no inference-time update.

## 6. Accounting

Every run records:
- parameters by group, total and active per token;
- routed multiply-adds;
- ERSA state, conv-tail, window-KV, PLE-history and PLE-table bytes;
- measured matmul FLOPs per token;
- peak RSS.

`run_r10 latency` measures decode milliseconds per token for every arm: one thread, interleaved.
