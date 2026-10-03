"""Cognitive R0 core mechanics: QUERY is read-only, LEARN is one receipt-bearing atomic transition."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from elpis.ECS_G.cognition import CognitiveCore, Transition, experience_digest
from elpis.ECS_G.native import ECSGError, ECSGLibrary, WorldState

from ._math_oracle import forward, gd_step
from .test_math_r0 import _library_path

DIM, WIDTH, LR = 6, 36, 0.002


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


def _setup(seed=3):
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0.0, 0.18, size=(DIM, WIDTH))
    x = rng.normal(0.0, 0.5, size=(24, DIM))
    y = forward(rng.normal(0.0, 0.5, size=(DIM, 4)), x)
    return w0, x, y


def test_query_is_read_only_and_learn_matches_the_qualified_recurrence(api):
    w0, x, y = _setup()
    with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=LR) as core:
        identity, epoch = core.identity, core.epoch
        first = core.query(x.tolist())
        assert core.query(x.tolist()) == first and core.identity == identity and core.epoch == epoch == 0
        receipt = core.learn(x.tolist(), y.tolist(), steps=40)
        w = w0
        for _ in range(40):
            w = gd_step(w, x, y, LR)
        np.testing.assert_allclose(core.query(x.tolist()), forward(w, x), rtol=1e-10, atol=1e-11)
        assert type(receipt) is Transition and (receipt.epoch_before, receipt.epoch_after) == (0, 40)
        assert receipt.before == identity and receipt.after == core.identity != identity
        assert receipt.experience == experience_digest(x.tolist(), y.tolist())
        assert receipt.steps == 40 and receipt.learning_rate == LR and len(receipt.digest) == 64


def test_receipts_and_states_are_deterministic(api):
    w0, x, y = _setup()
    runs = []
    for _ in range(2):
        with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=LR) as core:
            r1 = core.learn(x.tolist(), y.tolist(), steps=10)
            r2 = core.learn(x[:12].tolist(), y[:12].tolist(), steps=5)
            runs.append((r1.digest, r2.digest, core.snapshot(), core.query(x.tolist())))
    assert runs[0] == runs[1]


def test_snapshot_restore_preserves_the_learned_response(api):
    w0, x, y = _setup()
    with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=LR) as core:
        core.learn(x.tolist(), y.tolist(), steps=30)
        blob, answer = core.snapshot(), core.query(x.tolist())
    assert len(blob) > 8 * DIM * WIDTH  # W itself is persisted, not a coarse summary
    with CognitiveCore.restore(api, blob, learning_rate=LR) as again:
        assert again.query(x.tolist()) == answer and again.epoch == 30 and again.snapshot() == blob


def test_a_divergence_mid_learning_installs_nothing(api):
    w0, x, y = _setup()
    with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=5.0) as core:
        before = core.snapshot()
        with WorldState.restore(api, before) as probe:   # the first step succeeds, so the refusal is mid-way
            probe.step(x.tolist(), y.tolist(), 5.0)
        with pytest.raises(ECSGError):
            core.learn(x.tolist(), y.tolist(), steps=50)
        assert core.snapshot() == before and core.epoch == 0


@pytest.mark.parametrize("bad", [0, -1, 100_001, 2.0, True])
def test_invalid_step_counts_and_rates_are_refused(api, bad):
    w0, x, y = _setup()
    with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=LR) as core:
        with pytest.raises(ECSGError):
            core.learn(x.tolist(), y.tolist(), steps=bad)
        assert core.epoch == 0
    for rate in (0.0, -LR, float("inf"), 1):
        with pytest.raises(ECSGError):
            CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=rate)
