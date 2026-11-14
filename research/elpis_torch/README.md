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
| Architecture, tensor shapes, update semantics, scaling | `docs/research/elpis_torch/DESIGN.md` |
| What was run and what it showed | `docs/research/elpis_torch/EVIDENCE_R0.md` |
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
```

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

## Not here, on purpose

RALPH in any role, MoE routing, Engram, mHC, HCA, AWR, GDN, ERSA, persistent inference learning, an energy
function, a metacognitive controller, a retrieval controller, model weights, and any claim that the DSV4.1
research tower was translated: it was read for its mechanics and not ported.
