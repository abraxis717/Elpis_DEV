# Elpis Torch R0

A trainable, autoregressive, decoder-only transformer in PyTorch with the Cubic Geometric Core (CGC, the
ECS_G E0 cubic cell) as a neural operator inside every block, and a bounded, session-local, volatile
fast-weight mechanism for the CGC weights.

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

## Not here, on purpose

RALPH in any role, MoE routing, Engram, mHC, HCA, AWR, GDN, persistent inference learning, an energy
function, a metacognitive controller, a retrieval controller or host-side retriever (R3's block selection and R5's
coarse scan and reranking are the model's own learned functions; the host only stores, validates, evicts and
gathers), model weights, and any claim that the DSV4.1
research tower was translated: it was read for its mechanics and not ported.
