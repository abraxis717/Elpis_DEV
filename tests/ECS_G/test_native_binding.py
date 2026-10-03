"""The ECS_G Python binding exposes exactly the qualified native state surface."""

from __future__ import annotations

import ctypes
import importlib
import subprocess
import sys

import numpy as np
import pytest

from elpis.ECS_G.native import ECSGError, ECSGLibrary, WorldState

from ._math_oracle import gd_step, project_s3
from .test_math_r0 import REPO, _library_path

DIM, WIDTH, LR = 6, 36, 0.002


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


def _initial(seed=36):
    return np.random.Generator(np.random.PCG64(seed)).normal(0.0, 0.18, size=(DIM, WIDTH))


def _drive(seed, rows=8):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(0.0, 0.5, size=(rows, DIM)), rng.normal(0.0, 0.5, size=rows)


def test_s3_and_step_match_the_oracle(api):
    w0 = _initial()
    with WorldState.create(api, DIM, WIDTH, w0.reshape(-1).tolist()) as state:
        assert (state.dim, state.width, state.epoch) == (DIM, WIDTH, 0)
        mu, m, t3 = project_s3(w0)
        assert len(state.s3()) == api.s3_size(DIM) == 83
        np.testing.assert_allclose(state.s3(), np.concatenate((mu, m, t3)), rtol=1e-12, atol=1e-12)
        x, y = _drive(1)
        state.step(x.tolist(), y.tolist(), LR)
        expected = gd_step(w0, x, y, LR)
        np.testing.assert_allclose(np.asarray(state.w()).reshape(DIM, WIDTH), expected, rtol=1e-12, atol=1e-14)
        assert state.epoch == 1


def test_rejected_step_leaves_w_and_epoch_unchanged(api):
    with WorldState.create(api, DIM, WIDTH, _initial().reshape(-1).tolist()) as state:
        before = state.w()
        x, y = _drive(2)
        with pytest.raises(ECSGError):
            state.step(x.tolist(), [float("nan")] * len(y), LR)
        with pytest.raises(ECSGError):
            state.step((x * 1e200).tolist(), y.tolist(), LR)
        assert state.w() == before and state.epoch == 0


def test_snapshot_restore_continues_identically(api):
    with WorldState.create(api, DIM, WIDTH, _initial(7).reshape(-1).tolist()) as a:
        for k in range(3):
            x, y = _drive(10 + k)
            a.step(x.tolist(), y.tolist(), LR)
        blob = a.snapshot()
        with WorldState.restore(api, blob) as b:
            assert b.epoch == a.epoch == 3 and b.w() == a.w() and b.snapshot() == blob
            for k in range(3):
                x, y = _drive(20 + k)
                a.step(x.tolist(), y.tolist(), LR)
                b.step(x.tolist(), y.tolist(), LR)
                assert a.w() == b.w() and a.s3() == b.s3() and a.epoch == b.epoch
    with pytest.raises(ECSGError):
        WorldState.restore(api, blob[:-1])


def test_closed_state_is_refused(api):
    state = WorldState.create(api, DIM, WIDTH, _initial().reshape(-1).tolist())
    state.close()
    state.close()
    with pytest.raises(ECSGError):
        state.s3()


def test_binding_imports_no_numpy_inference_runtime_or_ecs_c():
    probe = ("import json, sys; import elpis.ECS_G.native; "
             "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('numpy', 'elpis')))))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(REPO / "src")}).stdout
    loaded = set(importlib.import_module("json").loads(out))
    assert not any(m.startswith("numpy") for m in loaded)
    assert loaded <= {"elpis", "elpis.ECS_G", "elpis.ECS_G.native"}, loaded


def test_fork_is_an_independent_identical_copy(api):
    with WorldState.create(api, DIM, WIDTH, _initial(5).reshape(-1).tolist()) as a:
        x, y = _drive(30)
        a.step(x.tolist(), y.tolist(), LR)
        with a.fork() as b:
            assert b.snapshot() == a.snapshot() and b.epoch == a.epoch == 1
            x, y = _drive(31)
            b.step(x.tolist(), y.tolist(), LR)
            assert b.epoch == 2 and a.epoch == 1 and b.w() != a.w()
