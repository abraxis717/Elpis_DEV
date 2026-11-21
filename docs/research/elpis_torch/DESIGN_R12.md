# Elpis Torch R12 — design: the micro-CGC volatile fast state

**What R12 adds.** R12 restores one missing operation: the episode-local update of the micro CGC core's 120
parameters from observed support error. It adds that operation inside the existing expert architecture:

    Expert_i(h) = SwiGLU_i(h) + g_i · CGC_i(h; W_active,i)          W_active,i = W_base,i + ΔW_session,i

- CGC stays inside `ChannelExpert`.
- Only `ΔW_session` is mutable: private, volatile and bounded, owned by one session, never written anywhere.
- The checkpoint stays frozen.

The scientific question and its frozen rules are in `PREREGISTRATION_R12.md`. The state contract is in
`CGC_FAST_STATE_CONTRACT_R12.md`. The evidence is in `EVIDENCE_R12.md`.

**What R12 does not change.**
- Nothing under `src/` or `native/`.
- No CI policy and no R0–R11 record.
- No default configuration.
- No file any earlier record pins by digest. `test_r11_records.py` binds R11's records to the current bytes of
  `cgc.py`, `sparse_channel.py`, `model.py`, `config.py`, `specialist_tasks.py`, `cgc_specialist.py`,
  `specialist_stack.py` and the R11 runners, so R12 adds new files only and changes none of those.
- No K1, LAO, RALPH, larger core, additional expert, memory tier, or native or GGUF runtime code.

## 1. P0: the runtime boundary, audited

| # | Question | Answer (source) |
|---|---|---|
| 1 | Where is each expert-local CGC core stored? | `blocks[l].sparse.experts[e].cgc.core.weight`, an `nn.Parameter` `[5, 3, 8]` = `[d, cells, units]` (`cgc.py` `CGCCore`). In R11's Stage C stack, l ∈ {0, 1} and each block has one expert, e = 0. The GELU control's core is `...experts[e].mlp.{hidden,out}.{weight,bias}`, also 120 values |
| 2 | Which parameters form the 120-value fast state? | Exactly the names `cgc_specialist.specialist_core_names` gives: `cgc.core.weight` (5·3·8 = 120), or the GELU's four core tensors (65 + 13 + 39 + 3 = 120). These are the parameters R11 Stage B's inner loop adapted |
| 3 | Which parameters must stay immutable? | All of them, the adapted core's own base value `W_base` included: `in_proj`, both blocks' ERSA mixers and norms, block 0's whole expert, block 1's FFN, `P_in`, `P_out` and gate, both routers, `final_norm` and `readout`. Tests check this through the parameter digest and the `state_dict` bytes before and after |
| 4 | How is an expert identified within the routing structure? | By (block l, expert e), the module path `blocks.{l}.sparse.experts.{e}`. A block's `TokenRouter` picks the top-K indices per token from the hidden state, and the grouped dispatch runs each expert on its own rows (`sparse_channel.py`). With one expert, every token is routed to e = 0 with weight exactly 1, so the channel's output equals the expert's bit for bit (tested) |
| 5 | How do gradients reach a selected expert? | Only through the tokens routed to it, scaled by their routing weights (`SparseChannel._grouped`; R11 test `test_an_expert_receives_gradient_only_from_the_tokens_routed_to_it`) |
| 6 | Can a last-block CGC update preserve the existing ERSA, attention and evidence state? | Yes, for three reasons. **(a)** `ErsaLM.forward` computes every state from each block's input before that block's channel: ERSA cells and convolution tails, the window K/V ring and token history, the PLE n-gram history, and R5's evidence ring, descriptors and fine keys. **(b)** The last channel feeds only `final_norm` and the head. **(c)** R5's evidence read is added before the last channel, and its stored keys come from the embedding and the evidence encoder only. Verified: under a changed last core, the states are bit-identical in the R11 stack and in an ERSA LM with window attention over two chunks, while the outputs do change |
| 7 | What would go stale if an earlier block's CGC changed? | Block l's channel enters every later block's input. So these would change: every later block's ERSA cells and tails, its window K/V entries for every past token, the evidence query, and every later output. Recurrent state cannot be patched; it would need a replay of the whole history. **Refused** (`FastStateError`, tested) |
| 8 | Can `session.bounded_update` be reused? | Yes, unchanged. It is model-agnostic: it takes a dict of deltas and `loss_of(delta)`, then takes the gradient, clips the global norm, steps, projects onto a ball, checks finiteness, and commits or refuses. R12 passes lr 0.08, clip 2.0, 8 steps and momentum 0 (R11 Stage B). The ball is an absolute bound of 1.28 per tensor, passed as `base_norm` with ratio 1. That is the largest displacement 8 clipped steps can reach, so the bound is a hard safety limit that never binds inside the protocol |
| 9 | What prevents using `VolatileSession` directly with the sparse `ErsaLM` or the R11 `RegressionStack`? | Five things. **(a)** It accepts only `ElpisTorchLM`. **(b)** It calls the model with `cache` and `weights_tag`, the dense model's KV-cache contract. **(c)** It selects every parameter named `*cgc.core.weight` (or the last block's), with no single-expert, fixed-assignment check for routed channels. **(d)** Its last-block path (`lastblock.last_block_logits`) re-implements a DENSE block's channel and an LM head. **(e)** Its teaching interface is next-token `SupportSet`s, not (x, y) pairs. R12 therefore reuses the transaction (`bounded_update`) and the last-block hook (`lastblock.capture_pre`), and replaces only the model-specific glue |
| 10 | Did R11 keep trained checkpoints? | **No.** R11's runners write only their `--out` JSON (one write site each; `test_isolation.py`), and `results/r11` holds 51 JSON records and a manifest. R12 rebuilds R11's Stage C models from their recorded seeds with R11's own code and checks each against the committed record. In development (smooth, seed 301), every row, training trace, selected step and gate was bit-identical |

## 2. The dependency diagram (R11 Stage C stack; the same order holds for `ErsaLM`)

```
tokens [x | y | is_query] ─► in_proj ─► x0
block 0   u0 = mix_norm(x0) ─► ERSA_0(u0; S0) ─► S0'             S0' depends on x0
          pre0 = x0 + mixed0 ─► channel_norm ─► SparseChannel_0 ─► Expert_0 = FFN_0 + g0·P_out0 C(P_in0 h; W0)
          x1 = pre0 + channel_0                                  W0 FROZEN (a change would stale S1' for all history)
block 1   u1 = mix_norm(x1) ─► ERSA_1(u1; S1) ─► S1'             S1' depends on x1, NOT on block 1's channel
          pre1 = x1 + mixed1           ◄── cached once per input: everything the fast path needs
          h1 = channel_norm(pre1) ─► SparseChannel_1 (1 expert, weight 1) ─► Expert_1(h1) =
                                     FFN_1(h1) + g1·P_out1 C(P_in1 h1; W_base + ΔW)
          x2 = pre1 + channel_1 ─► final_norm ─► readout ─► ŷ
ΔW (session-owned, volatile) ◄── bounded_update ◄── ∂/∂ΔW ‖ŷ(probes; W_base + ΔW) − y_s‖² ◄── observed (x_s, y_s)
```

**The cached path.** `pre1` of any input is computed once by the frozen model. Every inner step and every adapted
output then recomputes only:

    readout(final_norm(pre1 + SparseChannel_1(channel_norm(pre1); W_base + ΔW)))

It equals a full recomputation of the stack bit for bit, in output and in gradient (`parity.json`).

## 3. The adaptation contract

For an observed support set `S = {(x_s, y_s)}`, the probes are `[x_s, 0, 1]` with no context. The step is:

    q_s = P_in h_s,   r = q_sᵀ w,   r̃ = b·tanh(r/b),   a_c = Σ_i φ(r̃_ci),   Δh = g·P_out a,   ŷ = readout(final_norm(pre + FFN(h) + Δh))
    L(W) = mean_s ‖ŷ_s − y_s‖²
    W_{t+1} = W_t − η·min(1, c/‖∇_W L‖)·∇_W L,    η = 0.08,  c = 2.0,  t = 0 … 7,   ‖W_t − W_base‖ ≤ 8ηc = 1.28

The gradient is the local update the CGC lineage defines:

    ∇_{w_ci} L = Σ_s δ_{s,c} φ′(r̃_sci) κ(r_sci) q_s,    δ_{s,c} = ∂L/∂a_{s,c},    κ = sech²(r/b)

`parity.json` checks it against autograd: at most 6.9e-17 in float64.

**What teaches.** Only observed `(x, y)` pairs, through `ObservedSupport.observed(…, source=…)`. The R11 teacher's
targets are observed synthetic data. Nothing else teaches:
- no model output, generated token or confidence;
- no ERSA state, and no LAO signal.

A general next-token teaching interface would need its own gradient and provenance contract. R12 does not build it.

## 4. Code

| File | Content |
|---|---|
| `research/elpis_torch/fast_state.py` (new) | `FastStateConfig`, `ObservedSupport`, `SpecialistFastState` (identity, `adapt` through `session.bounded_update`, `reset`, `close`, refusals, `channel`/`output` through the last block's own `SparseChannel` and `ChannelExpert` by `functional_call`), `last_channel`, `head`, `frozen_pass` |
| `research/elpis_torch/fast_episodes.py` (new) | The Stage C seam harness: context-free probes (the leakage contract), `evaluate_fast` (curve, controls, parity, integrity), the batched training update (`torch.func.vmap`, the same mathematics), `override_last_core`, `train_fast` (R11's `train_stack` plus the inner update) |
| `research/elpis_torch/experiments/run_r12.py` (new) | `audit`, `parity`, `reproduce` (T1), `episodic` (T2), `latency`, `combine`, `manifest`, `dev` |
| `research/elpis_torch/experiments/run_r12_p5.py` (new) | P5, diagnosis only (`PREREGISTRATION_R12_P5.md`): `stage_b` (R11 Stage B's procedure on an R12 group's own tasks), `stack` (T2's G2 rebuilt, with a fixed `P_in`, misrouted, and the decodability of the task coordinates), `combine`, `dev` |
| `tests/research/elpis_torch/test_r12.py`, `test_r12_p5.py`, `test_r12_records.py` (new) | Contract, parity, refusal, isolation, leakage and trainer-equivalence tests; the P5 mechanics; the JSON-only record tests, R12's and P5's |
| `tests/research/elpis_torch/test_isolation.py` | The single write sites of `run_r12.py` and `run_r12_p5.py` registered |

## 5. The experiment, in one paragraph

**T1 (the PDF's decisive test, literally).** R11's Stage C models, rebuilt exactly, are adapted at test time. Each
episode's 32 support pairs stay in the context, and the last-block core takes 8 steps on the same pairs as
context-free probes. The 48 queries are then scored.

**T2.** The same stack and training plan, except that the specialist arms train with this update in the loop. This
separates a missing mechanism from a missing adaptation-compatible initialization, which the directive asks R12 to
distinguish.

Controls:
- the FFN-only stack;
- frozen CGC;
- fast CGC with a fixed or learned gate;
- fast GELU with the same 120-value, 8-step budget;
- shuffled targets;
- reset before the queries;
- updates disabled.

Everything is on fresh episodes and seeds. The rules are in `PREREGISTRATION_R12.md`.

## 6. Runtime and packaging

See `CGC_FAST_STATE_CONTRACT_R12.md`. The GGUF or sidecar interface is specified only as far as the evidence
justifies (`EVIDENCE_R12.md` §9). No runtime outside PyTorch is implemented. The results are in `EVIDENCE_R12.md`.
