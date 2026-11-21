# Elpis Torch R12 — the CGC fast-state contract

**Scope.** This contract governs `research/elpis_torch/fast_state.py`, `SpecialistFastState`. It is RESEARCH_ONLY.
It authorizes no runtime: the canonical Elpis runtime is unchanged, and nothing here is promoted to it.

## 1. What the state is

| Item | Value |
|---|---|
| Mutable tensor | `ΔW_session`: one tensor per specialist-core parameter of ONE expert. For the micro CGC, `cgc.core.weight` `[5, 3, 8]` = 120 values; for the matched GELU control, its four core tensors = 120 values |
| Precision | The core's dtype: float32 in the experiment (480 bytes for `ΔW`). float64 is the scientific reference (`parity.json`) |
| Allocated with it | A velocity buffer of the same size, required by `bounded_update`; it stays zero because momentum is 0. Total fast-state bytes: 960 in float32 |
| Not counted above | Measured separately (`latency.json`, `EVIDENCE_R12.md` §8): the support probes' last-block stream, the targets, transient gradients and the identity metadata. There is no replay buffer and no cache |
| Active weight | `W_active = W_base + ΔW`, formed per call. It is substituted through `torch.func.functional_call` into the last block's own `SparseChannel` and `ChannelExpert` |

## 2. What stays immutable

**Every model parameter, and the `state_dict` bytes.** This includes:
- `W_base`, the adapted core's own checkpoint value;
- FFN and SwiGLU weights, routers and ERSA parameters;
- the CGC input and output projections, and the gate;
- the readout or head, and any PLE or evidence parameters.

The state never writes a parameter. It reads `W_base` and adds `ΔW` out of place. `close()` re-verifies the model's
SHA-256 parameter digest. The tests check the digest and the `state_dict` bytes before and after adaptation.

## 3. Ownership and identity

- **Opening.** A state opens only over a frozen model: eval mode, and no parameter requires grad.
- **Identity.** Every state carries:
  - an explicit session identity (a non-empty string);
  - the model's parameter digest;
  - the layer;
  - the expert index;
  - the core names;
  - a version, which advances on every commit and reset;
  - the step counter.
- **Isolation.** Two states over one model share no mutable tensor (tested).
- **No persistence.** Pickling, copying and serializing are refused. The module has no save, export or file write.

## 4. Where it may act (cache and state coherence)

- **Only the last block.** Its channel runs after that block's ERSA write, window K/V write and evidence read. It
  feeds only `final_norm` and the readout or head. So no recurrent state, ring or cached key depends on the core.
  This is verified bit for bit: ERSA cells and tails, the window ring and the token history are unchanged under a
  changed core, while the outputs change.
- **The cached path is exact.** It computes `pre` (the last block's stream before its channel) once per input, then
  recomputes only the channel and the head. It equals full recomputation in output and gradient (difference 0.0).
- **Any earlier block is refused** (`FastStateError`). Its channel feeds every later block's recurrent state, and a
  changed core would need an unbounded replay. The contract fails closed rather than mix state versions.
- **The routed channel must hold exactly one expert and no shared expert.** That is a fixed assignment, a labelled
  laboratory condition. Learned routing among adapted experts is out of scope, and refused.

## 5. The update (a bounded transaction)

`adapt(loss_of, steps)` calls `session.bounded_update`, unchanged, once per step:
- the gradient of the support loss with respect to `ΔW`;
- the global gradient-norm clip, `c = 2.0`;
- an SGD step, `η = 0.08`, no momentum;
- projection onto `‖ΔW‖ ≤ 1.28` per tensor;
- a finite check of the loss, the gradient and the candidate;
- a candidate loss evaluated before commit;
- atomic commit, or refusal with the state unchanged.

**The step cap is 8** (`REFUSED_STEP_BUDGET` beyond it). The 8 steps and their parameters are R11 Stage B's. The
bound is the largest displacement 8 clipped steps can reach, so it never binds inside the protocol: the update is
exactly R11's (tested against plain clipped SGD to 1e-14).

| Refusal | Meaning |
|---|---|
| `REFUSED_LEARNING_DISABLED` | Learning is off for this session |
| `REFUSED_STEP_BUDGET` | 8 commits already made |
| `REFUSED_NONFINITE_LOSS` / `_GRADIENT` / `_CANDIDATE` | Nothing committed |
| `ROLLED_BACK_NONFINITE_AFTER` | The candidate's loss is not finite. (Loss increase is not rejected: `reject_loss_increase` is off, as in R11) |

## 6. Teaching signal and provenance

**The only teacher is `ObservedSupport.observed(inputs, targets, source=…)`.**
- The inputs carry no target. The harness asserts that the target slots of every probe token are zero.
- The caller names the targets' source.

**Probes run without any context.** A prediction being trained can therefore never read its own target from the
sequence. The rejected alternative, probes after a context that holds the pairs, would let it do so (shown in
`test_r12.py`).

**Query targets are read only for scoring, after adaptation.** The queries never teach.

**Excluded as teachers:**
- model outputs, generated tokens and confidence;
- ERSA state;
- LAO beliefs.

**Next-token teaching in a language model is a separate, future interface.** It would need its own gradient and
provenance contract.

## 7. Lifetime

| Event | Effect |
|---|---|
| Open | `ΔW = 0`. The output equals the frozen model's bit for bit |
| `reset()` | `ΔW` and the velocity return to zero, the step count to 0, the version advances. The output equals the frozen model's bit for bit (tested on every evaluation episode) |
| `close()` | The state is dropped, the model digest re-verified, and every further call refused |
| Episode boundary | One state per episode. The interference control measures what a stale state costs |

## 8. Numerical policy

- **Precision.** `ΔW` stays in float32, with a float64 reference. The 120 mutable values are not quantized.
- **Parity tolerances**, from `parity.json`:
  - cached against full recomputation: 0.0;
  - analytic against autograd gradient: at most 3.0e-8 in float32 and 6.9e-17 in float64;
  - batched training update against the transactional update: about 3e-8 and 6e-17;
  - adapted output, float32 against float64: recorded per arm.
