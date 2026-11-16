# R3 scientific gates, declared before the multi-seed results

Committed while `history` seeds 1-2, `ersa_conv` seed 2, `history_oracle` seed 1, the P2 variants and the program
models were still training. Seed 0 of `history`, `ersa_conv` and `history_oracle` (and seed 1 of `ersa_conv`) had
been observed when these thresholds were written; that is stated, not hidden. Chance on value probes is 1/16.
Every threshold applies to the `facts` evaluation of `run_r3` (test seeds 700,000,000+), mode named.

| Gate | Criterion | Applies to |
|---|---|---|
| G5 oracle readout functional at short length | `oracle_override` direct accuracy >= 0.90 at 1,024 and 4,096 tokens, every `history` seed | readout |
| G6 learned addressing beats recurrent-only | `learned` direct accuracy of `history` exceeds `ersa_conv` by >= 0.10 at 4,096 and 16,384 tokens, every seed pair | addressing |
| G7 evidence after long gaps, measured apart from hits | report recall@K and evidence-token read rate at every length; PASS if `oracle_override` direct >= 0.80 at 262,144 tokens, every `history` seed (readout after long gaps given the right block) | readout at distance |
| G7b learned retrieval at 1M (reported, not expected) | `learned` direct >= 0.50 at 1,048,576 tokens, every `history` seed | addressing at distance |
| G8 no unacknowledged local regression | history-off accuracy of `history` and in-window probes are reported; any loss against `ersa_conv` is listed as a finding | local modeling |
| P2 attention collapse | `ersa_conv_local1025` (window covers training) loses >= 0.2 direct accuracy from 1,024 to 16,384 tokens while `ersa_conv_local128` does not: the R2 hypothesis is SUPPORTED; otherwise NOT SUPPORTED | P2 |
| P5 free-running program generation | reported only (no threshold): valid-statement fraction, arity and print accuracy over 2,048 generated tokens | long generation |

No threshold is changed after results. A gate that fails is reported as FAIL.
