"""Elpis Torch R0 tests (research/elpis_torch). RESEARCH_ONLY.

PyTorch is optional for this repository. Without it every module here is SKIPPED and says so (never an
implicit PASS); with ``ELPIS_REQUIRE_TORCH=1`` a missing PyTorch is a collection error instead. The dedicated
CI job sets it.

Everything runs on one CPU thread with fixed seeds, so a fixture trained twice is the same model.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HAVE_TORCH = importlib.util.find_spec("torch") is not None

if not HAVE_TORCH and os.environ.get("ELPIS_REQUIRE_TORCH") == "1":
    raise RuntimeError("ELPIS_REQUIRE_TORCH=1 but PyTorch is not installed (research/elpis_torch/requirements.txt)")

SKIP_REASON = "PyTorch is not installed (optional: research/elpis_torch/requirements.txt); not a PASS"

# Small on purpose: three models train in well under a minute on one thread.
BASE = dict(vocab_size=259, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=64, seed=1)
CHANNELS = {
    "A": dict(channel="ffn", ffn_hidden=64),
    "B": dict(channel="cgc", cgc_bottleneck=16, cgc_channels=32, cgc_units=4),
    "C": dict(channel="ffn+cgc", ffn_hidden=64, cgc_bottleneck=8, cgc_channels=16, cgc_units=4),
}
TRAIN = dict(steps=500, batch_size=32, lr=4e-3, warmup_steps=20, seed=0)


@pytest.fixture(scope="session", autouse=True)
def _single_thread():
    if not HAVE_TORCH:
        yield
        return
    import torch
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="session")
def world():
    from research.elpis_torch.data import make_world
    return make_world(6, 3, seed=1)


@pytest.fixture(scope="session")
def splits(world):
    from research.elpis_torch.data import build_splits
    return build_splits(world, train=1024, val=128, comp=128, length=64, seed=1)


def make_model(name: str, **overrides):
    from research.elpis_torch.config import ModelConfig
    from research.elpis_torch.model import ElpisTorchLM
    return ElpisTorchLM(ModelConfig(**{**BASE, **CHANNELS[name], **overrides}))


@pytest.fixture(scope="session")
def trained(splits):
    """A, B and C trained offline on the same data, token budget and optimizer settings, then frozen.

    Returns ``{name: (frozen model, TrainReport, validation metrics before training)}``.
    """
    from research.elpis_torch.training import TrainConfig, evaluate, freeze, train
    out = {}
    for name in CHANNELS:
        model = make_model(name)
        before = evaluate(model, splits.val)
        report = train(model, splits.train, TrainConfig(**TRAIN), val=splits.val)
        out[name] = (freeze(model), report, before)
    return out
