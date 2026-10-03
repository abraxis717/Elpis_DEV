"""Runtime R1 randomized differential: the executor against the scalar reference, bit for bit.

Seeded random shapes (dim, width, rows, steps), rates (including 0), data
scales (including divergent ones) and input forms. For every case the
executor's learn, schedule, transaction and forward must equal the reference
WorldState byte for byte, or refuse exactly when the reference refuses, with
the authoritative state untouched.
"""
from __future__ import annotations

from array import array
import ctypes
import random

import pytest

from elpis.ECS_G.native import ECSGError, ECSGLibrary, Executor, WorldState

from .test_math_r0 import _library_path

CASES = 160


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


def _case(seed):
    rng = random.Random(f"ecs-runtime-r1-differential-{seed}")
    dim, width, rows = rng.randint(1, 8), rng.choice([1, 2, 3, 7, 8, 9, 15, 16, 17, 36, 37, 64, 71]), rng.randint(1, 70)
    steps = rng.choice([1, 2, 3, 5, 17])
    lr = rng.choice([0.0, 0.002, 0.01, 0.05])
    scale = rng.choice([0.05, 0.18, 0.5, 1.0, 3.0])  # the larger ones often diverge: refusals are compared too
    w0 = [rng.gauss(0.0, scale / max(1.0, width ** 0.5)) for _ in range(dim * width)]
    x = [[rng.gauss(0.0, scale) for _ in range(dim)] for _ in range(rows)]
    y = [rng.gauss(0.0, scale) for _ in range(rows)]
    return dim, width, rows, steps, lr, w0, x, y, rng


def _reference(api, dim, width, w0, drives, lr):
    """Applies (x, y, steps) drives; returns (snapshot, refused)."""
    with WorldState.create(api, dim, width, w0) as ref:
        before = ref.snapshot()
        for x, y, steps in drives:
            for _ in range(steps):
                try:
                    ref.step(x, y, lr) if lr > 0 else _zero_rate_step(api, ref, x, y)
                except ECSGError:
                    return before, True
        return ref.snapshot(), False


def _zero_rate_step(api, ref, x, y):
    """The Python reference binding refuses lr <= 0; call the C reference step directly for lr = 0."""
    lib, rows = api._lib, len(y)
    xs = (ctypes.c_double * (rows * ref.dim))(*[v for row in x for v in row])
    ys = (ctypes.c_double * rows)(*y)
    count = int(lib.elpis_ecsg_state_gd_step_scratch_f64(ref.dim, ref.width, rows))
    scratch = (ctypes.c_double * count)()
    if lib.elpis_ecsg_state_gd_step_f64(ref._live(), xs, ys, rows, 0.0, scratch, count) != 0:
        raise ECSGError("NONFINITE", "reference step")


@pytest.mark.parametrize("seed", range(CASES))
def test_executor_matches_the_reference_bit_for_bit(api, seed):
    dim, width, rows, steps, lr, w0, x, y, rng = _case(seed)
    expected, refused = _reference(api, dim, width, w0, [(x, y, steps)], lr)
    form = seed % 3
    xs = x if form == 0 else array("d", [v for row in x for v in row])
    ys = y if form == 0 else array("d", y)
    with Executor.create(api, dim, width, w0, max_rows=rows) as e:
        initial = e.snapshot()
        if refused:
            with pytest.raises(ECSGError) as info:
                e.learn(xs, ys, lr, steps)
            assert info.value.code == "NONFINITE"
            assert e.snapshot() == initial and e.epoch == 0 and e.generation == 0
        else:
            assert e.learn(xs, ys, lr, steps).epoch_after == steps
            assert e.snapshot() == expected
            q = [[rng.gauss(0.0, 0.5) for _ in range(dim)] for _ in range(rng.randint(1, 40))]
            with WorldState.restore(api, expected) as ref:
                try:
                    want = ref.forward(q)
                except ECSGError:
                    want = None
            if want is None:
                with pytest.raises(ECSGError):
                    e.forward(q)
            else:
                assert e.forward(q) == want


@pytest.mark.parametrize("seed", range(0, CASES, 4))
def test_schedules_and_transactions_match_the_reference(api, seed):
    dim, width, rows, steps, lr, w0, x, y, rng = _case(seed)
    if rows < 2:
        rows, x, y = 2, x * 2, y * 2
    cut = rng.randint(1, rows - 1)
    drives = [(x[:cut], y[:cut], steps), (x[cut:], y[cut:], steps)]
    expected, refused = _reference(api, dim, width, w0, drives, lr)
    with Executor.create(api, dim, width, w0, max_rows=rows) as e:
        initial = e.snapshot()
        with e.transaction() as txn:
            if refused:
                with pytest.raises(ECSGError):
                    txn.learn_schedule(x, y, (cut, rows - cut), lr, steps)
                assert not txn.open
            else:
                txn.learn_schedule(x, y, (cut, rows - cut), lr, steps)
                assert e.snapshot() == initial          # candidate only
                assert txn.commit().epoch_after == 2 * steps
        assert e.snapshot() == (initial if refused else expected)
