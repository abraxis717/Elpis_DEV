# Elpis Torch R0

A trainable, autoregressive, decoder-only transformer in PyTorch. Its channel sublayer carries a neural operator
derived from the Cubic Geometric Core: batched ECS_G E0 cubic cells behind a pre-activation bound (default 3.0),
with learned input and output projections. It is not the native 120-parameter CGC and not K1; it is exactly E0
only with the bound off (`docs/research/elpis_torch/CLAIM_AUDIT_R9.md`, L3). The operator sits in every block's
dense channel from R0 to R5, and inside routed FFN + CGC experts from R6 on. The prototype also has a bounded,
session-local, volatile fast-weight mechanism for the CGC weights.

**Claim status (R9).** `docs/research/elpis_torch/CLAIM_AUDIT_R9.md` records what each result here supports.
- In R7's architecture, a preregistered parameter- and compute-matched comparison found no measurable benefit of
  CGC over FFN width: `CGC_NOT_ESTABLISHED` (`EVIDENCE_R9.md`).
- The trained models do use their CGC path: removing it at evaluation raises the loss.
- At this scale, CGC cost 11% decode latency.
- **Scope** (`ASSAY_CANNOT_TEST_META_LEARNING_CLAIM`, ledger section "R10 additions"):
  - R9 measured static next-byte prediction only.
  - It neither establishes nor refutes a CGC benefit for in-context adaptation or meta-learning.
  - The CGC here is stateless: unlike native K1 it holds no memory of its own, and unlike the historical R97W
    setting it is not adapted by gradient steps at inference.
  - R10 tests adaptation and retention directly (`PREREGISTRATION_R10.md`, `EVIDENCE_R10.md`).

**Claim status (R10).** R10 is a preregistered test of in-context adaptation and retention (`EVIDENCE_R10.md`). It
used R9's d 64 backbone with a 64-token attention window, against parameter- and compute-matched FFN arms and an
attention-free pair. At this scale (3,000 steps, three screening seeds):
- **Adaptation: `NO_INCREMENTAL_CGC_ADAPTATION_BENEFIT_ESTABLISHED`.** The CGC arms adapted worse than the FFN
  arms, including without attention on every seed.
- **Retention: `ASSAY_NOT_LEARNED`.** The models learned regime-conditioned bindings (pre-interference accuracy
  often above 0.90), but none reliably kept earlier bindings through conflicting interference: best 0.173 against the
  0.20 floor. This is a weakness of the tested architectures, not a failure to learn the task. It does not show that
  every attention or ERSA configuration forgets, nor that K1 would solve the task if integrated (`CGC_CAPABILITY_R11.md`
  §6).
- **Attribution.** What retention exists is carried by attention (inside the window) or by the ERSA state (behind
  a gap). Zeroing CGC at evaluation changes almost nothing.
- **Cost.** CGC costs 10% decode latency.
- **What did not run.** Confirmation did not run (only a favorable screening triggers it), and no width sweep is
  justified.

**Claim status (R11).** R11 bounds the expert-local CGC and tests it as an explicit specialist, in three
preregistered stages (`EVIDENCE_R11.md`).
- **Footprint.** The opt-in micro profile is R97W's geometry: 5 inputs, 3 cells, 8 units, so 120 core parameters
  inside an `8H + 121` adapter. It takes CGC from 28.6% of the R6 500M layout to 0.39%, and from 42.5% of the 1B
  layout to 0.38%. These are exact counts; nothing at that scale was built. The defaults are unchanged.
- **Stage A, the microscopic reference.**
  - Faithful E0 meta-learns on fresh episodes and beats the matched 120-parameter GELU: `MICRO_REFERENCE_QUALIFIED`.
  - With learned projections, the bounded adapter keeps that behaviour: `PROJECTED_ADAPTATION_QUALIFIED`.
- **Stage B, inside the real FFN expert, under explicit offline inner updates only.**
  - The fixed-gate bounded CGC is causal, made by its specialist objective, and 15% better than a matched GELU
    specialist: `EXPERT_SPECIALIZATION_QUALIFIED`.
  - The historical learned gate works, but without the cubic-specific advantage.
  - Without inner updates, both are worse than the FFN alone: `ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES`.
- **Stage C, frozen weights in the actual ERSA stack.**
  - Trained end to end and then frozen, with the support examples in context, neither CGC specialist beats the FFN
    alone: `NOT_ESTABLISHED`.
  - The in-context information is ERSA's.
  - A learned router does not separate the task families: `INTEGRATION_GAP_LEARNED_ROUTING`.
  - The specialist's value therefore lives in explicit adaptation, which frozen-weight inference does not provide.
    R11 reports the gap and implements no runtime learning.
- **Not established.**
  - Any language-model benefit.
  - Anything at 500M or 1B.
  - GPU throughput.
  - Runtime adaptation: R11 did not implement it, since inference weights stay frozen.
- **Unchanged.** The verdicts of R9 and R10 stand.

**Claim status (R12).** R12 restores the micro CGC's episode-local update inside the Stage C stack, as a private,
volatile and bounded fast state, and tests it under preregistered rules (`EVIDENCE_R12.md`).
- **The mechanism is exact.** It adapts only the last block's 120 core values, through the block's own
  `SparseChannel` and `ChannelExpert`.
  - The cached path equals full recomputation, and reset and disabled learning restore the frozen model bit for bit.
  - No recurrent state changes, no target leaks, and earlier blocks are refused.
  - It costs 960 bytes, and 15–25 ms for 8 updates on one CPU thread.
- **The decisive test: `NOT_ESTABLISHED`.**
  - T1, R11's Stage C models adapted at test time: the update barely moves them and, for the fixed gate, makes the
    queries worse (1.60×).
  - T2, the same stack trained with the update in the loop: the update is steep, and it beats a matched GELU (0.52).
    But the stack leans on it instead of the context, and only draws level with the FFN-only stack (0.98,
    p = 0.66).
- **P5, the representation seam (diagnosis only).**
  - The task coordinates reach the specialist, a fixed projection suffices, and the specialist's identity matters.
  - But even with exact coordinates, the 8-step CGC is about 2× worse than the FFN-only stack's in-context
    inference. On these tasks there is no advantage to recover.
- **Not established.** Any specialist advantage in the stack, and any language-model, scale, GPU, packaging or GGUF
  claim. No sidecar is proposed.
- **Unchanged.** The verdicts of R9, R10 and R11 stand.

**RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. NO_LANGUAGE_CLAIM.** This is an architectural research prototype at
toy scale. It is not part of the Elpis runtime, nothing in `src/` or `native/` imports it, and passing its
tests says nothing about language or reasoning ability. Under the current mission a transformer is not the
Elpis cognitive path; what it would take to change that is a separate, unadopted proposal
(`DEV/proposals/CGC_NATIVE_GENERATIVE_SUBSTRATE_R0.md`).

| | |
|---|---|
| R0: architecture, tensor shapes, update semantics, scaling | `docs/research/elpis_torch/DESIGN.md` |
| R0: what was run and what it showed | `docs/research/elpis_torch/EVIDENCE_R0.md` |
| R1: scrolling context, last-block fast weights, capacity model | `docs/research/elpis_torch/DESIGN_R1.md` |
| R1: long-horizon results, failures, recommendation | `docs/research/elpis_torch/EVIDENCE_R1.md` |
| R2: ERSA design notes, results, failures, gates | `docs/research/elpis_torch/EVIDENCE_R2.md` |
| R3–R8: design and evidence of each phase | `docs/research/elpis_torch/DESIGN_R*.md`, `EVIDENCE_R*.md` |
| R9: claim audit, CGC attribution (preregistered), whole-model memory and bandwidth, Gate A (HOLD) | `docs/research/elpis_torch/CLAIM_AUDIT_R9.md`, `PREREGISTRATION_R9.md`, `DESIGN_R9.md`, `EVIDENCE_R9.md`, `GATE_A_PROTOCOL_R9.md` |
| R10: CGC and in-context adaptation and retention (preregistered), R11 performance handoff | `docs/research/elpis_torch/PREREGISTRATION_R10.md`, `DESIGN_R10.md`, `EVIDENCE_R10.md`, `R11_PERFORMANCE_NOTE.md` |
| R11: the bounded expert-local CGC specialist: capability audit, bounding, specialization, preregistration (Stages A–C), evidence | `docs/research/elpis_torch/CGC_CAPABILITY_R11.md`, `CGC_BOUNDING_R11.md`, `CGC_SPECIALIZATION_R11.md`, `DESIGN_R11.md`, `PREREGISTRATION_R11.md`, `PREREGISTRATION_R11_STAGE_C.md`, `EVIDENCE_R11.md` |
| R12: the micro-CGC volatile fast state: design (P0 audit, dependency diagram), state contract, preregistration, the P5 amendment (the representation seam), evidence (the Stage C seam experiment) | `docs/research/elpis_torch/DESIGN_R12.md`, `CGC_FAST_STATE_CONTRACT_R12.md`, `PREREGISTRATION_R12.md`, `PREREGISTRATION_R12_P5.md`, `EVIDENCE_R12.md` |
| Where the CGC mathematics comes from | `PROVENANCE.md` |

## Install

PyTorch is optional for this repository and is needed only here. From the repository root:

```sh
python -m pip install -r research/elpis_torch/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install "pytest>=8"
```

Developed with Python 3.12 and PyTorch 2.15 (CPU). Nothing else is required; `pip install .` of the Elpis
package is unaffected and still installs no dependency.

## Reproduce

```sh
# Tests: about 70 s on one CPU thread. Without PyTorch they are skipped; with ELPIS_REQUIRE_TORCH=1 they fail.
ELPIS_REQUIRE_TORCH=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider -q tests/research/elpis_torch

# Example: train, generate from text, adapt in a session, reset. About 20 s. Writes nothing.
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.examples.quickstart

# The R0 experiment (FFN versus CGC versus FFN+CGC, three seeds, then session adaptation). About 20 minutes on 4 cores.
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r0 --out /tmp/elpis_torch_r0.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r0 --quick     # 20 s smoke run

# R1 (scrolling context). Checkpoints and records go only where --out says; nothing is committed.
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r1 smoke        # every path once, about 2 min
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r1 train --topology sparse --out /tmp/sparse.ckpt
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r1 ladder --ckpt /tmp/sparse.ckpt --out /tmp/sparse.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r1 adapt --ckpt /tmp/sparse.ckpt --out /tmp/adapt.json
PYTHONDONTWRITEBYTECODE=1 python -B -m research.elpis_torch.experiments.run_r1 capacity     # the projected 24 GB placement
```

`--topology` is `sparse`, `local` or `dense`. The ladder streams 8K, 32K, 128K, 256K, 1M and 1.31M tokens in
48-token chunks; the dense reference stops at 32K.

The system-call zero-write test needs `strace`; without it that one test is skipped (it fails instead under
`ELPIS_REQUIRE_TORCH=1`).

## Use

```python
import torch
from research.elpis_torch.config import ModelConfig
from research.elpis_torch.model import ElpisTorchLM
from research.elpis_torch.session import SessionConfig, SupportSet, VolatileSession
from research.elpis_torch.tokenizer import ByteTokenizer
from research.elpis_torch.training import TrainConfig, freeze, train

tokenizer = ByteTokenizer()
model = ElpisTorchLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4,
                                 n_kv_heads=2, max_seq_len=48, channel="cgc",      # "ffn" | "cgc" | "ffn+cgc"
                                 cgc_bottleneck=32, cgc_channels=121, cgc_units=6))
# report = train(model, batch, TrainConfig(steps=1500))       # offline; `batch` from research.elpis_torch.data
freeze(model)                                                   # the immutable checkpoint

prompt = torch.tensor([tokenizer.encode("Ka=", bos=True)])
print(tokenizer.decode(model.generate(prompt, 16)[0].tolist()))   # prefill, then KV-cached decoding

with VolatileSession(model, SessionConfig(lr=0.1, max_steps=16)) as session:
    session.observe(tokenizer.encode("Ka=Fb=C", bos=True))      # observed evidence: may be a target
    session.generate(4)                                         # the session's own samples: never a target
    for result in session.adapt_on_stream(steps=2):             # bounded, atomic; refusals are reported
        print(result.status.name, result.loss_before, result.loss_after)
    session.reset()                                             # every fast weight gone
```

## Layout

| File | Content |
|---|---|
| `e0.py` | float64 reference of the E0 cell, validated against the pinned upstream file |
| `upstream/cgc_ecsg_e0.py` | the upstream CGC source, byte-identical, used only as a test oracle |
| `cgc.py` | `CGCCore` (C batched E0 cells), the pre-activation bound, `CGCTransform` (bottleneck and readout) |
| `config.py`, `tokenizer.py` | validated model configuration; reversible byte tokenizer |
| `model.py` | RMSNorm, rotary positions, causal MHA/GQA, SwiGLU, the three blocks, explicit KV cache, generation |
| `training.py` | offline training, evaluation, free-running rollout, explicit checkpoint save and load |
| `session.py` | `VolatileSession`: `W_active = W_base + dW_session` |
| `data.py` | the generated finite-state transduction task and its verified splits |
| `accounting.py` | parameters, theoretical multiply-accumulates, memory estimates |
| `experiments/run_r0.py`, `examples/quickstart.py` | the comparison protocol; a short walk-through |
| `scroll.py` | R1: `ScrollLM`, ring-buffer window, compressed and fine memory tiers, sinks, `StreamState` |
| `lastblock.py`, `scroll_session.py` | R1: last-block CGC fast weights that leave every cache valid; `ScrollSession` |
| `scroll_data.py`, `experiments/run_r1.py` | R1: the long-stream retrieval task; train, ladder, adapt, capacity |
| `ersa.py`, `ersa_model.py`, `experiments/run_r2.py` | R2: ERSA kernel, mixer, exact cells, `ErsaLM`; `run_r2 smoke|exact|latency|projection|transduction`; train with `run_r1 train --topology ersa_conv` |
| `ersa.py` `chunked_scan` | R3: the same ERSA recurrence evaluated block-parallel (exact; `ErsaConfig(ersa_kernel="chunked")`) |
| `history.py`, `history_model.py` | R3: exact token-history ring, bounded block index, `ErsaHistoryLM` (learned addressing, rereading, readout; `off` / `learned` / oracle diagnosis modes) |
| `longrange_tasks.py`, `program_tasks.py` | R3: long-range fact streams with source labels; a structured program language with an exact checker for free-running generation |
| `scaling.py`, `experiments/run_r3.py` | R3: 500M / 1B configurations and context accounting; `run_r3 train|facts|program|scale|gpu|smoke` (`docs/research/elpis_torch/DESIGN_R3.md`, `EVIDENCE_R3.md`) |
| `context_runtime.py` | R4: deterministic exact top-K, tiled exact index scoring, bounded encoded-block cache, `ContextSession` (bounded streaming prefill/decode, reset, budget refusal) |
| `spark_reference.py`, `experiments/run_r4.py` | R4: a Spark-style 3:1 hybrid comparator (published geometry, untrained); `run_r4 bench|parity|spark|scale|gradients|gpu|gpu-horizon|cache-patterns|cache-workloads|smoke` (`docs/research/elpis_torch/EVIDENCE_R4.md`; R4.1 corrections: `EVIDENCE_R4_1.md`) |
| `evidence_ring.py` | R5: exact FIFO token ring of 2^20 tokens and bounded descriptor index with session / epoch / slot / generation identity; stale addresses refused |
| `evidence_model.py` | R5: `EvidenceLM` (unchanged ERSA/CGC core + coarse learned block index, exact candidate reread, fine reranker with a supersession feature, bounded injection in the last block) |
| `evidence_tasks.py`, `experiments/run_r5.py` | R5: records / scoped-symbol / R3 tasks with explicit hard negatives; `run_r5 train|eval|ablate|r3base|r3eval|memory|smoke` (`docs/research/elpis_torch/DESIGN_R5.md`, `EVIDENCE_R5.md`) |
| `sparse_channel.py` | R6: token-routed channel experts (each a SwiGLU + its own CGC + gate), optional shared expert, deterministic top-K routing, grouped dispatch, routing statistics, auxiliary losses, exact upcycling; `ErsaConfig.sparse` places it in selected blocks |
| `lm_training.py`, `experiments/run_r6.py` | R6: one autoregressive training path for every ERSA-family variant (documents, chunked state, full or truncated backpropagation, router losses, generation, versioned checkpoints); `run_r6 scale|dispatch|controls|cache|smoke` (`docs/research/elpis_torch/DESIGN_R6.md`, `EVIDENCE_R6.md`) |
| `ple.py`, `ple_store.py`, `experiments/run_r7.py` | R7 (admission candidate): parametric lookup embeddings: seeded hashed n-gram addresses (bounded history, document boundaries), trainable tables with lazy updates, a gated fusion before one block, an exact removal control; immutable and read-only file-backed stores; `run_r7 study|local|revision|collisions|latency|accounting|combine|smoke` (`docs/research/elpis_torch/DESIGN_R7.md`, `EVIDENCE_R7.md`) |
| `evidence_ring.py`, `evidence_model.py` (`EvidenceConfig.fine_keys`), `experiments/run_r8.py` | R8 (admission candidate, opt-in): the R5 read with stored fine keys: each block's fine keys computed once at completion and kept in the ring's index beside its descriptors (same slots, same validation, reset and invalidation); candidates reranked from them, only injected blocks reread and re-encoded; the R5 reread stays the default; `run_r8 baseline|represent|admission|smoke` (`docs/research/elpis_torch/DESIGN_R8.md`, `EVIDENCE_R8.md`) |
| `experiments/run_r9.py`, `experiments/run_r9_p3.py` | R9: the preregistered CGC attribution study (`run_r9 accounting|integrity|train|latency|combine|manifest|smoke`) and whole-model memory and bandwidth accounting (`run_r9_p3 account|trace|bandwidth|rss|roofline|measured`; `docs/research/elpis_torch/DESIGN_R9.md`, `EVIDENCE_R9.md`) |
| `adaptation_tasks.py`, `experiments/run_r10.py` | R10: the in-context adaptation and retention assay and its preregistered study (`docs/research/elpis_torch/DESIGN_R10.md`, `EVIDENCE_R10.md`) |
| `cgc_specialist.py`, `specialist_tasks.py`, `specialist_stack.py`, `experiments/run_r11.py`, `experiments/run_r11_stage_c.py`, `experiments/bench_r11_operators.py` | R11: the opt-in micro CGC profile (R97W's 5 / 3 / 8 core, adapter `8H + 121`), exact census and budget, branch diagnostics, the matched GELU control; the R97W-form Stage A/B harness around the real `ChannelExpert`; the Stage C regression stack of the repository's `ErsaBlock`s; `run_r11 audit|parity|calibrate|stage_a|stage_b|latency|combine|manifest`, `run_r11_stage_c run|router|combine|dev` (`docs/research/elpis_torch/DESIGN_R11.md`, `EVIDENCE_R11.md`) |
| `fast_state.py`, `fast_episodes.py`, `experiments/run_r12.py`, `experiments/run_r12_p5.py` | R12: `SpecialistFastState`, a private, volatile, bounded `ΔW` for the LAST block's single expert's specialist core (120 values), updated through `session.bounded_update` and executed through the block's own `SparseChannel`/`ChannelExpert`; the Stage C seam harness (context-free probes, controls, batched training update); `run_r12 audit|parity|reproduce|episodic|latency|combine|manifest|dev`; the P5 diagnosis, `run_r12_p5 stage_b|stack|combine|dev` (`docs/research/elpis_torch/DESIGN_R12.md`, `PREREGISTRATION_R12_P5.md`, `EVIDENCE_R12.md`) |

## Not here, on purpose

RALPH in any role, MoE routing beyond R6's token-routed channel experts, Engram, mHC, HCA, AWR, GDN, persistent inference learning, an energy
function, a metacognitive controller, a retrieval controller or host-side retriever (R3's block selection and R5's
coarse scan and reranking are the model's own learned functions; the host only stores, validates, evicts and
gathers), model weights, and any claim that the DSV4.1
research tower was translated: it was read for its mechanics and not ported.
