"""Cognitive R0 contract, behavioural part (docs/COGNITION_R0.md): ECS response depends on ECS state.

Not "DSV code was not imported" but: a query is the native forward map of the
current authoritative W. It equals f_W computed by an independent native state
holding the same W, follows W through learning, returns when W is restored,
vanishes when W is replaced, and cannot be produced once the native forward
call is unavailable. Learning is atomic, and the core keeps no state beside W.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS_G.native import ECSGError, ECSGLibrary, WorldState

from ._math_oracle import forward, gd_step
from .test_math_r0 import REPO, _library_path

DIM, WIDTH, LR = 6, 36, 0.002

PENDING = {
    "core": "Cognitive R0 core not implemented yet (K2)",
}


def pending(key):
    return pytest.mark.xfail(strict=True, reason=PENDING[key]) if key in PENDING else (lambda f: f)


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


def _w(seed=36, scale=0.18):
    return np.random.default_rng(seed).normal(0.0, scale, size=(DIM, WIDTH))


def _experience(seed=7, rows=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=(rows, DIM))
    teacher = rng.normal(0.0, 0.5, size=(DIM, 4))
    return x, forward(teacher, x)


@pending("forward")
def test_binding_forward_is_the_native_state_forward(api):
    w = _w()
    x, _ = _experience()
    with WorldState.create(api, DIM, WIDTH, w.reshape(-1).tolist()) as state:
        out = state.forward(x.tolist())
        assert type(out) is tuple and len(out) == len(x) and all(type(v) is float for v in out)
        np.testing.assert_allclose(out, forward(w, x), rtol=1e-12, atol=1e-12)
        y = forward(_w(5), x)
        state.step(x.tolist(), y.tolist(), LR)
        np.testing.assert_allclose(state.forward(x.tolist()), forward(gd_step(w, x, y, LR), x), rtol=1e-11, atol=1e-12)
        for bad in ([], [[0.0] * (DIM - 1)], [[float("nan")] * DIM], [[1e300] * DIM]):
            with pytest.raises(ECSGError):
                state.forward(bad)


@pending("forward")
def test_adoption_is_one_atomic_commit_or_nothing(api):
    x, y = _experience()
    with WorldState.create(api, DIM, WIDTH, _w().reshape(-1).tolist()) as state:
        before = state.snapshot()
        candidate = state.fork()
        for _ in range(3):
            candidate.step(x.tolist(), y.tolist(), LR)
        expected = candidate.snapshot()
        assert state.snapshot() == before                    # nothing installed yet
        state.adopt(candidate)
        assert state.snapshot() == expected and state.epoch == 3
        with pytest.raises(ECSGError):
            candidate.s3()                                   # the candidate was consumed
        stale = state.fork()
        state.step(x.tolist(), y.tolist(), LR)               # authoritative state moved after the fork
        moved = state.snapshot()
        with pytest.raises(ECSGError):
            state.adopt(stale)
        assert state.snapshot() == moved
        with WorldState.create(api, DIM, WIDTH, _w(9).reshape(-1).tolist()) as stranger:
            with pytest.raises(ECSGError):
                state.adopt(stranger)                        # not a fork of this state
        assert state.snapshot() == moved
        stale.close()


@pending("core")
def test_ecs_response_depends_on_ecs_state(api, monkeypatch):
    from elpis.ECS_G.cognition import CognitiveCore
    x, y = _experience()
    q = np.random.default_rng(99).normal(0.0, 0.5, size=(32, DIM))
    w0 = _w()
    with CognitiveCore.create(api, DIM, WIDTH, w0.reshape(-1).tolist(), learning_rate=LR) as core:
        initial, snapshot0 = core.query(q.tolist()), core.snapshot()
        with WorldState.create(api, DIM, WIDTH, w0.reshape(-1).tolist()) as twin:
            assert initial == twin.forward(q.tolist())                    # follows W, nothing else
        receipt = core.learn(x.tolist(), y.tolist(), steps=25)
        learned = core.query(q.tolist())
        assert learned != initial and receipt.epoch_after == 25 and receipt.before != receipt.after
        w = w0
        for _ in range(25):
            w = gd_step(w, x, y, LR)
        np.testing.assert_allclose(learned, forward(w, q), rtol=1e-10, atol=1e-11)
        with CognitiveCore.restore(api, snapshot0, learning_rate=LR) as reset:
            assert reset.query(q.tolist()) == initial                     # restored W, restored response
        with CognitiveCore.create(api, DIM, WIDTH, [0.0] * (DIM * WIDTH), learning_rate=LR) as blank:
            assert blank.query(q.tolist()) == (0.0,) * len(q)             # replaced W, response gone

        def unavailable(self, rows):
            raise ECSGError("UNAVAILABLE", "native forward withheld")
        monkeypatch.setattr(WorldState, "forward", unavailable)
        with pytest.raises(ECSGError):
            core.query(q.tolist())                                         # no answer without ECS_G forward


@pending("core")
def test_learning_is_atomic_and_the_core_keeps_nothing_but_w(api):
    from elpis.ECS_G.cognition import CognitiveCore
    x, y = _experience()
    with CognitiveCore.create(api, DIM, WIDTH, _w().reshape(-1).tolist(), learning_rate=LR) as core:
        assert not hasattr(core, "__dict__") and set(CognitiveCore.__slots__) == {"_state", "learning_rate"}
        before, epoch = core.snapshot(), core.epoch
        with pytest.raises(ECSGError):
            core.learn((x * 1e100).tolist(), y.tolist(), steps=3)         # refused mid-way: nothing installed
        assert core.snapshot() == before and core.epoch == epoch


_CLEAN = r"""
import ctypes, json, sys
from elpis.ECS_G.native import ECSGLibrary
from elpis.ECS_G.cognition import CognitiveCore
api = ECSGLibrary(ctypes.CDLL(sys.argv[1]))
x = [[0.1 * ((r * 7 + a) % 11 - 5) for a in range(6)] for r in range(12)]
y = [0.05 * (r % 5 - 2) for r in range(12)]
with CognitiveCore.create(api, 6, 36, [0.01 * (i % 17 - 8) for i in range(216)], learning_rate=0.002) as core:
    before = core.query(x)
    core.learn(x, y, steps=10)
    after = core.query(x)
print(json.dumps({"changed": before != after,
                  "modules": sorted(m for m in sys.modules if m.startswith(("elpis", "research", "numpy")))}))
"""


@pending("core")
def test_a_clean_process_learns_and_answers_with_no_model_machinery():
    result = subprocess.run([sys.executable, "-c", _CLEAN, str(_library_path())], capture_output=True, text=True,
                            cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["changed"]
    assert set(report["modules"]) <= {"elpis", "elpis.ECS_G", "elpis.ECS_G.native", "elpis.ECS_G.cognition"}, \
        report["modules"]
