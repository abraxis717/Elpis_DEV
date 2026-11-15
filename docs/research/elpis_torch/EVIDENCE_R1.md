# Elpis Torch R1: evidence report and recommendation

Status: RESEARCH_ONLY, FEASIBILITY_ONLY, SEMANTICS=NONE, NO_LANGUAGE_CLAIM. Design: `DESIGN_R1.md`. Records:
`research/elpis_torch/results/r1/`. R0 baseline: `EVIDENCE_R0.md`, unchanged.

**Headline.** The scrolling-context mechanics are correct and bounded, and a one-million-token stream (and
1.31 million, past the horizon) was actually executed on a CPU in constant memory. Last-block CGC adaptation
provably and measurably leaves every cache valid. **The long-range retrieval capability was not
demonstrated:** the sparse memory gave no accuracy benefit over a local window at any length, and the dense
reference itself fell to chance beyond a few thousand tokens. Nothing in this report supports a claim that
any of these models uses a million tokens of context.

## 1. P0: R0 closed and preserved

| | |
|---|---|
| R0 head | `320a6e4028bc56ddce88b84b3a62cd65cad9a9e2` |
| Push run 389 | completed, success, 16 of 16 jobs |
| Pull-request run 390 | completed, success, 16 of 16 jobs |
| Changes to R0 code in R1 | additive only: `SessionConfig.fast_scope` (default `"all"`, R0 behaviour), the bounded update moved into a shared function, `freeze` unchanged. All 197 R0 tests pass unmodified, except one isolation assertion extended to register the new experiment's two explicit offline writers. |
| R0 results | `results/run_r0.json` and `EVIDENCE_R0.md` untouched, including the compositional-generalization failure and the adaptation interference. |

## 2. Environment

One cloud container: 4 vCPU Intel Xeon 2.30 GHz, 15 GB RAM, no GPU. Python 3.12.3, PyTorch 2.15.0+cpu.
Trainings and ladders ran detached, some concurrently; every wall-clock figure is PERFORMANCE_ONLY.
No Ouroboros hardware, mounted file or GPU was available or assumed.

## 3. Tests

`ELPIS_REQUIRE_TORCH=1 ELPIS_REQUIRE_NATIVE=1 pytest tests/research/elpis_torch`: **249 passed, 0 failed,
0 skipped** (about 75 s, one thread). 197 are R0's; 52 are new.

| File | Tests | Establishes |
|---|---:|---|
| `test_scroll.py` | 39 | the scrolling model against an independent per-token oracle; chunk and origin invariance; bounded state; eviction; coordinates; refusal; the task; the capacity model |
| `test_scroll_session.py` | 13 | the cache-validity proof numerically; R0 full-stack against last-block; the scrolling session's bounds, provenance, isolation, zero writes |

| Gate | Result |
|---|---|
| Local topology with everything in the window equals the R0 dense model | PASS, `< 1e-10` (float64), identical weights by name |
| Chunked ring implementation equals the per-token oracle over 2.3 horizons, memory in one or two layers, with and without sinks, shared and separate memory keys, origin 0 and `2^20` | PASS, `< 1e-9` (8 cases) |
| Chunk size does not change the logits (1, 7, 31 against 32) | PASS, `< 1e-10` |
| Logits independent of absolute position: origins `2^20 - 160`, `2^20 - 8`, `2^20`, `2^40 + 8`, `2^62 - 8000` | PASS, **bitwise** |
| State allocated once: sizes and tensor addresses unchanged over 5.5 horizons; size equals the capacity formula | PASS |
| Every selected block lies inside its query's horizon and outside its window (more than 4,000 reads checked) | PASS |
| FIFO eviction; token history addressable by absolute position, `EvictedError` beyond it | PASS |
| An oversized chunk is refused whole and the state is untouched | PASS |
| Training path equals inference path; every memory parameter receives a gradient | PASS, `< 1e-12` |
| CGC finite over a 4,000-token stream wrapped 30 times, bounded and unbounded | PASS |
| R0 KV cache bitwise identical under a last-block CGC change; different in layer 1 under a block-0 change | PASS |
| Scrolling state after adapting is bitwise the state a model holding `W + dW` from token 0 would have written; logits agree to `1e-5` | PASS |
| R0 session: `fast_scope="all"` drops and rebuilds the cache, `"last"` keeps it; both give the recomputed logits | PASS |
| Scrolling session: only observed followers teach; learning window and all state constant in size; every refusal leaves the session unchanged; reset exact; no serialization | PASS |
| No Python-level filesystem write over a scrolling session lifetime of 3.5 horizons, with a negative control | PASS |

One finding from the tests changed the design document: with memory in two layers information persists beyond
one horizon (layer 1 remembers what layer 0 remembered). It is now asserted, not assumed away.

## 4. The long-stream experiment

**Task** (`scroll_data.py`): filler with sparse single-token facts and questions: bindings, contradictions
(the latest binding wins), a second question family with no token in common with the facts, and a two-hop
alias question. 8 keys, 8 values, 4 aliases; chance 0.125.

**Models.** 64 wide, 2 layers, 4 heads, 2 key/value heads, CGC bottleneck channel (R0's configuration B),
75,520 parameters (75,652 with the memory gate and slope). `sparse`: window 64, block 16, chunk 48, memory
in the last layer, 8 blocks read per query of which 6 as raw tokens, 4 sinks, shared memory keys, horizon
`2^20`. `local`: the same without memory. `dense`: the R0 model with full attention.

**Training.** Identical for all: 2,500 steps, 24 streams of 384 tokens with 84 events per step, AdamW 3e-3,
seed 0. One seed. All three `COMPLETED`, no non-finite loss.

### 4.1 At the training length (384 tokens, held out)

| Topology | Facts within the window | Facts beyond the window: direct / assoc / hop1 / hop2 |
|---|---|---|
| dense | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |
| local | 1.00 | 0.42 / 0.37 / 0.45 / 0.66 |
| sparse (run C, shared keys) | 1.00 | 0.39 / 0.38 / 0.59 / 0.68 |

The local model's beyond-window score is what two stacked 64-token windows reach. **The sparse model does
not beat it on value retrieval.**

### 4.2 Just past the training length (accuracy on direct questions; n in parentheses)

| Tokens | dense | local | sparse | sparse: fact block selected / read as raw tokens |
|---:|---|---|---|---|
| 384 | 1.00 (658) | 0.80 (658) | 0.80 (658) | 0.69 / 0.59 |
| 768 | 0.84 (207) | 0.53 (207) | 0.53 (207) | 0.52 / 0.43 |
| 1,536 | 0.65 (213) | 0.41 (213) | 0.42 (213) | 0.32 / 0.23 |
| 3,072 | 0.45 (117) | 0.26 (117) | 0.15 (117) | 0.32 / 0.25 |
| 6,144 | 0.20 (111) | 0.08 (111) | 0.05 (111) | 0.33 / 0.28 |

Dense attention is better than both at every length where it was run, and **all three decay toward chance
within sixteen training lengths**. Evaluation streams hold 48 events whatever their length, so they are far
sparser than the training streams; the decay mixes length and density shift and the two were not separated.

### 4.3 The ladder (accuracy on direct questions; other kinds in the records)

| Tokens x streams | dense | local | sparse | sparse: fact block selected / read raw (beyond-window probes) |
|---:|---|---|---|---|
| 8,192 x 4 | 0.21 (48) | 0.10 (48) | 0.00 (48) | 0.35 / 0.31 (119) |
| 32,768 x 4 | 0.04 (57) | 0.12 (57) | 0.04 (57) | 0.36 / 0.30 (110) |
| 131,072 x 4 | not run | 0.11 (54) | 0.04 (54) | 0.35 / 0.27 (119) |
| 262,144 x 4 | not run | 0.30 (43) | 0.02 (43) | 0.33 / 0.27 (108) |
| 1,048,576 x 4 | not run | 0.00 (59) | 0.02 (59) | 0.31 / 0.24 (117) |
| 1,310,720 x 2 (beyond the horizon) | not run | 0.21 (24) | 0.00 (24) | 0.20 / 0.18 (49) |

Chance is 0.125. Every topology is at chance or below from 8K on; with 24 to 59 probes per cell the local
model's 0.30 and 0.21 are noise around chance. The sparse model is below chance and confidently wrong (mean
probe loss 8 to 12 nats against 2.5 to 4 for local): its memory reads change the prediction, to the wrong
answer.

One thing does hold across two orders of magnitude of length. The block that contains the needed fact is
among the 8 selected in 31% to 36% of beyond-window probes from 8K to 1M tokens, and read as raw tokens in
24% to 31%. Picking 8 of 65,536 blocks at random would hit 0.012%. **Addressing generalizes with length;
reading the answer out does not.** This is a diagnostic, not a capability.

### 4.4 Measured memory and throughput (sparse topology)

| Tokens x streams | Seconds | Tokens per second | Stream state | Constant? | Equals formula? | Peak process RSS | Evicted blocks |
|---:|---:|---:|---:|---|---|---:|---:|
| 8,192 x 4 | 2 | 19,775 | 1,104.2 MiB | yes | yes | 1,362 MiB | 0 |
| 32,768 x 4 | 5 | 28,427 | 1,104.2 MiB | yes | yes | 1,379 MiB | 0 |
| 131,072 x 4 | 41 | 12,720 | 1,104.2 MiB | yes | yes | 1,461 MiB | 0 |
| 262,144 x 4 | 125 | 8,417 | 1,104.2 MiB | yes | yes | 1,572 MiB | 0 |
| 1,048,576 x 4 | 2,380 | 1,762 | 1,104.2 MiB | yes | yes | 2,195 MiB | 0 |
| 1,310,720 x 2 | 1,801 | 1,455 | 552.1 MiB | yes | yes | 2,195 MiB | 16,384 |

* **Measured.** 276.0 MiB of state per stream at a `2^20` horizon (fine tier 256, compressed 16, token history
  4, window and sinks 0.03), identical at every length and equal to `accounting.scroll_state_bytes`. Process
  RSS grows with the length only by the benchmark's own arrays (tokens, labels, predictions for the whole
  stream), not by model state. At 1,310,720 tokens the state reports position 1,310,720, 16,384 evicted
  blocks and oldest reachable position 262,144 = 1,310,720 - `2^20`, as the design requires.
* **Throughput falls with the filled horizon**, 12,720 to 1,762 tokens per second from 131K to 1M: the block
  scoring term is linear in resident blocks (`DESIGN_R1.md`, section 6). This is the cost the design predicts
  and the main thing to fix before scale.
* Local topology: 16.1 MiB for 4 streams (token history only matters), 38,000 to 63,000 tokens per second at
  every length. Dense reference: 8,094 tokens per second at 8K, 2,053 at 32K, KV cache 64 MiB and 2.0 GiB RSS
  at 32K for this 75K-parameter model; not run beyond.
* **Projected, not measured:** the 24 GB placement in `DESIGN_R1.md`, section 6.

## 5. P2: full-stack against last-block fast weights

R0 transduction task, last operation rewired after training; 2 CGC variants x 3 seeds; 16 observed support
trajectories, 16 bounded updates; evaluation on 256 unseen trajectories. Means over 3 seeds, accuracy range
in brackets. `results/r1/adapt.json`.

| Variant | Scope | `lr` | Changed loss before -> after | Changed accuracy after | Unchanged loss after | Unchanged accuracy | ms per update | Fast-weight bytes | Stream cache kept | Cache rebuilds | Logit error against recompute |
|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| bottleneck | all | 0.05 | 14.70 -> 3.16 | 0.614 [0.49, 0.70] | 0.026 | 0.998 | 16.4 | 371,712 | no | 1 | 0 |
| bottleneck | all | 0.2 | 14.70 -> 0.94 | 0.895 [0.70, 1.00] | 0.059 | 0.974 | 16.9 | 371,712 | no | 1 | 0 |
| bottleneck | all | 1.0 | 14.70 -> 1.35 | 0.558 [0.39, 0.67] | 0.684 | 0.789 | 17.7 | 371,712 | no | 1 | 0 |
| bottleneck | last | 0.05 | 14.70 -> 6.10 | 0.300 [0.29, 0.31] | 0.052 | 0.994 | 6.2 | 185,856 | yes | 0 | 7.6e-06 |
| bottleneck | last | 0.2 | 14.70 -> 1.76 | 0.668 [0.49, 0.87] | 0.061 | 0.988 | 5.3 | 185,856 | yes | 0 | 5.7e-06 |
| bottleneck | last | 1.0 | 14.70 -> 1.04 | 0.677 [0.57, 0.75] | 0.226 | 0.931 | 5.8 | 185,856 | yes | 0 | 7.6e-06 |
| dense | all | 0.05 | 13.90 -> 3.03 | 0.526 [0.39, 0.61] | 0.026 | 0.999 | 12.6 | 440,320 | no | 1 | 0 |
| dense | all | 0.2 | 13.90 -> 0.28 | 0.917 [0.80, 1.00] | 0.025 | 0.999 | 13.1 | 440,320 | no | 1 | 0 |
| dense | all | 1.0 | 13.90 -> 1.11 | 0.575 [0.49, 0.66] | 0.607 | 0.828 | 12.3 | 440,320 | no | 1 | 0 |
| dense | last | 0.05 | 13.90 -> 5.10 | 0.250 [0.16, 0.40] | 0.068 | 0.988 | 3.9 | 220,160 | yes | 0 | 5.7e-06 |
| dense | last | 0.2 | 13.90 -> 1.21 | 0.760 [0.58, 0.93] | 0.041 | 0.999 | 4.0 | 220,160 | yes | 0 | 5.7e-06 |
| dense | last | 1.0 | 13.90 -> 1.10 | 0.785 [0.58, 1.00] | 0.174 | 0.934 | 4.7 | 220,160 | yes | 0 | 5.7e-06 |

In all 36 runs every update was applied and the base parameter digest was unchanged.

* **Cache validity: yes.** Last-block updates never invalidated or rebuilt the cache, and the logits read from
  the kept cache differ from a full recomputation by at most 7.6e-06 (float32 round-off). Full-stack updates
  invalidated it every time.
* **Gain: lower at the same rate, more robust at a high one.** At `lr` 0.2 last-block reaches 0.67 to 0.76
  accuracy on changed transitions against 0.90 to 0.92 for full-stack. At `lr` 1.0 full-stack degrades (0.56
  to 0.58, and unchanged transitions fall to 0.79 to 0.83) while last-block holds 0.68 to 0.79 and 0.93.
* **Interference: comparable at `lr` 0.2** (unchanged loss 0.04 to 0.06 against 0.03 to 0.06).
* **Cost: about one third per update** (3.9 to 6.2 ms against 12.3 to 17.7 ms), because nothing before the last
  channel transformation is recomputed, and half the fast-weight memory.

On a scrolling stream (65,536 tokens through the trained sparse model): 8 updates applied at 1.75 ms each
with the stream state bitwise unchanged; streaming those tokens once took 6.6 s, so the full recomputation a
full-stack update would need costs about 3,800 last-block updates, and is impossible once tokens have left the
horizon. Session memory: 17.3 MiB of stream state at a 65,536-token horizon, 96 KiB of fast weights, 66 KiB
of learning window. The support loss did not move on that stream (2.0799 -> 2.0799): its window held almost
only filler, which is noise. That run measures cost, not learning.

## 6. Failures and negative results

1. **The sparse memory did not learn to retrieve, in three configurations.** Beyond-window direct accuracy at
   the training length: run A (separate memory keys, 6 blocks read, 2 as raw tokens) 0.39; run B (separate
   keys, 8 read, 6 raw) 0.41; run C (shared keys, 8 read, 6 raw) 0.39; local-only 0.42; dense 1.00. Alias
   lookups improved slightly (0.56, 0.68, 0.59 against 0.45). Runs A and B are in
   `results/r1/failed_training_runs.json`. Each change was made for a diagnosed reason and none fixed it.
2. **No topology retrieves at long range.** Including the dense reference (section 4.2). The million-token
   ladder therefore measures mechanics, memory and cost, and **no accuracy capability**.
3. **At long range the sparse model is worse than local**: confidently wrong, below chance.
4. **Contradictory evidence** could not be assessed: accuracy is at chance for contradicted and uncontradicted
   facts alike. The learned recency slope stayed near its initial value (0.14 to 0.16).
5. **The first version of the task failed for every topology.** Byte-level records needing a two-layer
   induction circuit: after 4,000 (local) and 1,750 (dense, sparse) steps direct accuracy was 0.24 to 0.29
   for all three. It was replaced by single-token facts; the numbers are in `failed_training_runs.json`.
6. **The whole-stream ("span") probes are invalid.** Their reserved keys never occur in training, so they
   score 0.00 for every topology, the dense model at the training length included. Eviction beyond the
   horizon is established by the mechanical gates (section 3) and the state coordinates (section 4.4), not by
   an accuracy drop. This is a defect of the benchmark, found in its results.
7. **Block scoring is linear in the horizon** and dominates the cost at 1M tokens (7x slower than at 131K).
8. **One seed, tens of probes per cell.** The ladder's accuracy cells cannot support any comparison finer
   than "at chance".
9. A bookkeeping defect: the local topology's recorded `state_bytes_formula` lists two empty tiers, so that
   record's equality flag is false although the totals are equal. The formula is fixed; the record is kept.

## 7. Not executed, not verified

* Any GPU run; the 24 GB placement; transfer cost of a CPU-resident fine tier.
* Training at long length, or with evaluation-like event density; more than one seed; any hyperparameter search.
* The dense reference beyond 32,768 tokens.
* System-call (strace) zero-write tracing of the scrolling session (the Python audit gate ran; R0's strace
  gate covers the R0 session).
* Kimi Delta Attention or any linear attention; hierarchical blocks.
* Adaptation gain of the scrolling session on a task (only its cost and its cache validity were measured; the
  gain was measured on the R0 task through the R0 session).
* The Ouroboros protocol of `DESIGN_R1.md`, section 9.

## 8. Recommendation

1. **Which attention topology?** The hybrid: a ring-buffer local window in every layer and long-range memory
   in few layers. It is the only one of the three whose state fits (a dense KV cache at `2^20` tokens is
   projected at 48 GiB for the deployment example against 3.4 to 15.4 GiB), and its execution semantics are
   now verified. **But the memory design is not validated.** It did not train here, and the evidence does not
   show that it can. Adopt the bounded execution substrate; do not adopt this memory mechanism as settled.
   The next experiment is narrow: train with streams several windows long at evaluation-like density and
   measure, in this order, block selection (already length-stable), then read-out.
2. **Is CGC numerically stable in it?** Yes, within what was run: every training completed without a
   non-finite value, bounded and unbounded cells stayed finite over wrapped streams, and the 8.6 million tokens of
   the sparse ladder produced finite losses. The CGC block is unchanged from R0; the topology changes what it attends to,
   not what it computes.
3. **Can last-block adaptation avoid global KV invalidation?** Yes: proven, and measured bitwise. It costs
   leverage (about 0.7 against 0.9 accuracy at the same rate on the R0 task) and buys a third of the compute,
   half the memory and no recomputation. For a scrolling stream it is the only option that remains correct.
4. **Which 1M capabilities are implemented, measured or hypothetical?**

   | Capability | Status |
   |---|---|
   | Bounded ring window, compressed and fine memory, sinks, deterministic eviction | implemented, verified against an oracle |
   | Position handling across `2^20` and beyond | implemented, verified bitwise |
   | Streaming 1,048,576 and 1,310,720 tokens in constant state | **measured** on CPU (276 MiB per stream) |
   | Throughput at 1M tokens | measured: 1,762 tokens per second for 4 streams on 4 CPU cores; too slow, and the cause is known |
   | Last-block adaptation without cache invalidation | implemented, proven, measured |
   | Retrieval of a fact from beyond the window | **failed** at every length |
   | Multi-hop, contradictory evidence, non-lexical association at long range | **failed** (at chance) |
   | 24 GB GPU placement | projected only |
   | Use of a million tokens of context by a model | hypothetical |

5. **What should stay out of scope?** RALPH in any role; any retrieval controller, re-ranker or index model;
   persistent or cross-session memory; full-stack fast weights on a scrolling stream; linear attention until
   the present memory either works or is abandoned; claims of full-context equivalence; and canonical runtime
   integration, which the mission does not permit and this evidence would not justify.
