"""Runtime R1 executor binding, behavioural part (docs/ECS_RUNTIME_R1.md).

Python controls the ECS and never executes its hot path: a query is one native
call, CognitiveCore.learn(K) is one native learn call whatever K is (the native
call pattern is identical for K=1 and K=4000), commits are native, buffers
cross without conversion, results are bitwise the scalar reference, and the
native workspace allocates nothing after creation.
"""
from __future__ import annotations

from array import array
from collections import Counter
import ctypes
import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from elpis.ECS.cognition import CognitiveCore, Transition
from elpis.ECS.native import Commit, ECSGError, ECSGLibrary, Executor, WorldState

from .test_math_r0 import REPO, _library_path

DIM, WIDTH, LR = 6, 36, 0.002


@pytest.fixture(scope="module")
def api():
    return ECSGLibrary(ctypes.CDLL(str(_library_path())))


class _CountingLibrary:
    """Wraps a loaded library and counts every native call made through it."""

    def __init__(self, lib):
        self._lib, self.calls = lib, Counter()

    def __getattr__(self, name):
        fn, calls = getattr(self._lib, name), self.calls

        class Counted:
            def __setattr__(self, attr, value):
                setattr(fn, attr, value)

            def __call__(self, *args):
                calls[name] += 1
                return fn(*args)
        counted = Counted()
        object.__setattr__(self, name, counted)
        return counted


@pytest.fixture()
def counted():
    lib = _CountingLibrary(ctypes.CDLL(str(_library_path())))
    return ECSGLibrary(lib), lib.calls


def _data(seed=11, rows=64):
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0.0, 0.18, size=DIM * WIDTH)
    x = rng.normal(0.0, 0.5, size=(rows, DIM))
    y = rng.normal(0.0, 0.5, size=rows)
    return w0.tolist(), x.tolist(), y.tolist()


def _reference(api, w0, x, y, steps):
    with WorldState.create(api, DIM, WIDTH, w0) as ref:
        for _ in range(steps):
            ref.step(x, y, LR)
        return ref.snapshot()


def test_learn_is_one_native_call_whatever_k(counted):
    api, calls = counted
    w0, x, y = _data()
    patterns = {}
    for receipt in (False, True):
        for k in (1, 4000):
            with CognitiveCore.create(api, DIM, WIDTH, w0, learning_rate=LR) as core:
                calls.clear()
                result = core.learn(x, y, steps=k, receipt=receipt)
                patterns[receipt, k] = dict(calls)
                assert type(result) is (Transition if receipt else Commit) and result.epoch_after == k
    assert patterns[False, 1] == patterns[False, 4000] == {"elpis_ecsg_executor_learn": 1}
    # The audited receipt adds two snapshot reads around the same single learn call, for any K.
    assert patterns[True, 1] == patterns[True, 4000] == {"elpis_ecsg_executor_learn": 1,
                                                         "elpis_ecsg_executor_snapshot_write": 2}


def test_query_is_one_native_call_and_read_only(counted):
    api, calls = counted
    w0, x, y = _data()
    with CognitiveCore.create(api, DIM, WIDTH, w0, learning_rate=LR) as core:
        core.learn(x, y, steps=3)
        before, generation = core.snapshot(), core._state.generation
        flat, out = array("d", np.asarray(x).ravel().tolist()), array("d", bytes(8 * len(x)))
        calls.clear()
        answer = core.query(x)
        assert dict(calls) == {"elpis_ecsg_executor_forward": 1}
        calls.clear()
        assert core.query_into(flat, out) == len(x)
        assert dict(calls) == {"elpis_ecsg_executor_forward": 1}
        assert tuple(out) == answer
        assert core.snapshot() == before and core._state.generation == generation and core.epoch == 3


def test_executor_is_bitwise_the_scalar_reference_for_lists_and_buffers(api):
    w0, x, y = _data(rows=37)
    expected = _reference(api, w0, x, y, 9)
    flat_x = array("d", np.asarray(x).ravel().tolist())
    inputs = {
        "lists": (x, y),
        "tuples": (tuple(map(tuple, x)), tuple(y)),
        "array": (flat_x, array("d", y)),
        "memoryview-2d": (memoryview(flat_x).cast("B").cast("d", (37, DIM)), memoryview(array("d", y))),
        "readonly": (memoryview(flat_x.tobytes()).cast("d"), memoryview(array("d", y).tobytes()).cast("d")),
        "numpy-buffer": (memoryview(np.ascontiguousarray(x)), memoryview(np.asarray(y))),
    }
    for name, (xs, ys) in inputs.items():
        with Executor.create(api, DIM, WIDTH, w0, max_rows=37) as e:
            commit = e.learn(xs, ys, LR, steps=9)
            assert commit == Commit(0, 9, 0, 1, 9), name
            assert e.snapshot() == expected, name
            with WorldState.restore(api, expected) as ref:
                assert e.forward(xs) == ref.forward(x), name


def test_epoch_and_generation_and_receipts_are_unchanged_in_meaning(api):
    w0, x, y = _data()
    with CognitiveCore.create(api, DIM, WIDTH, w0, learning_rate=LR) as core:
        identity = core.identity
        receipt = core.learn(x, y, steps=25)
        assert (receipt.epoch_before, receipt.epoch_after, receipt.steps) == (0, 25, 25)
        assert receipt.before == identity and receipt.after == core.identity
        assert core.snapshot() == _reference(api, w0, x, y, 25)
        commit = core.learn(x, y, steps=5, receipt=False)
        assert commit == Commit(25, 30, 1, 2, 5) and core.epoch == 30
    # A buffer experience yields the same receipt as the same values given as lists.
    with CognitiveCore.create(api, DIM, WIDTH, w0, learning_rate=LR) as a, \
            CognitiveCore.create(api, DIM, WIDTH, w0, learning_rate=LR) as b:
        assert a.learn(x, y, steps=4) == b.learn(array("d", np.asarray(x).ravel().tolist()), array("d", y), steps=4)


def test_refused_learn_after_steps_changes_nothing(api):
    w0, x, y = _data()
    with Executor.create(api, DIM, WIDTH, w0) as e:
        e.learn(x, y, LR)
        before, stats = e.snapshot(), e.stats()
        with pytest.raises(ECSGError) as info:
            e.learn(x, (np.asarray(y) * 1e3).tolist(), 50.0, steps=500)
        assert info.value.code == "NONFINITE" and info.value.step >= 2
        assert e.snapshot() == before and e.epoch == 1 and e.generation == 1
        after = e.stats()
        assert after["commits"] == stats["commits"] and after["refusals"] == stats["refusals"] + 1


def test_native_workspace_allocates_nothing_after_creation(api):
    w0, x, y = _data()
    with Executor.create(api, DIM, WIDTH, w0) as e:
        created = e.stats()
        assert created["heap_allocations"] == 2 and created["workspace_bytes"] == api.workspace_bytes(DIM, WIDTH)
        for _ in range(20):
            e.forward(x)
            e.learn(x, y, LR, steps=50)
            with e.transaction() as txn:
                txn.learn(x, y, LR)
                txn.s3()
                txn.commit()
        assert e.stats()["heap_allocations"] == 2
        e.reserve(1024)
        assert e.stats()["heap_allocations"] == 3 and e.max_rows == 1024


def test_transactions_commit_natively_and_detect_staleness(api):
    w0, x, y = _data()
    with Executor.create(api, DIM, WIDTH, w0) as e:
        with e.transaction() as txn:
            assert txn.learn_schedule(x, y, (40, 24), LR, steps=2).epoch_after == 4
            assert txn.epoch == 4 and e.epoch == 0
            with WorldState.create(api, DIM, WIDTH, w0) as ref:
                for _ in range(2):
                    ref.step(x[:40], y[:40], LR)
                for _ in range(2):
                    ref.step(x[40:], y[40:], LR)
                assert txn.s3() == ref.s3() and txn.forward(x) == ref.forward(x)
                assert txn.commit() == Commit(0, 4, 0, 1, 4)
                assert e.snapshot() == ref.snapshot()
        assert not txn.open
        with e.transaction():
            with pytest.raises(ECSGError) as info:
                e.transaction()              # one transaction at a time
            assert info.value.code == "BUSY"
        with e.transaction() as stale:
            stale.learn(x, y, LR)
            e.learn(x, y, LR)
            with pytest.raises(ECSGError) as info:
                stale.s3()
            assert info.value.code == "STALE" and not stale.open
        assert e.stats()["stale_refusals"] == 1
        # Schedules must cover the admitted rows exactly.
        with e.transaction() as bad:
            for drives in ((40, 23), (64, 0), (), (40.0, 24)):
                with pytest.raises(ECSGError):
                    bad.learn_schedule(x, y, drives, LR)
            assert bad.open


def test_single_writer_refuses_overlapping_calls_from_threads(api):
    w0 = np.random.default_rng(1).normal(0.0, 0.05, size=DIM * 1152).tolist()
    x = np.random.default_rng(2).normal(0.0, 0.5, size=(256, DIM))
    y = np.zeros(256)
    with Executor.create(api, DIM, 1152, w0) as e, \
            Executor.create(api, DIM, 1152, w0) as twin:
        xb, yb = array("d", x.ravel().tolist()), array("d", y.tolist())
        probe = array("d", x[:4].ravel().tolist())
        done, refused = threading.Event(), []

        def worker():
            while True:
                try:
                    e.learn(xb, yb, LR, steps=60)
                    break
                except ECSGError as exc:
                    assert exc.code == "BUSY"
            done.set()
        thread = threading.Thread(target=worker)
        thread.start()
        while not done.is_set():
            try:
                e.forward(probe)
            except ECSGError as exc:
                assert exc.code == "BUSY"
                refused.append(exc)
            time.sleep(0.0002)  # yield, so the worker is not starved of entry
        thread.join()
        twin.learn(xb, yb, LR, steps=60)
        assert refused and e.snapshot() == twin.snapshot()
        assert e.stats()["busy_refusals"] >= len(refused)


@pytest.mark.parametrize("bad", [
    lambda e, x, y: e.learn(x, y, LR, steps=0),
    lambda e, x, y: e.learn(x, y, LR, steps=True),
    lambda e, x, y: e.learn(x, y, LR, steps=-1),
    lambda e, x, y: e.learn(x, y, 1, steps=1),
    lambda e, x, y: e.learn(x, y, -LR),
    lambda e, x, y: e.learn(x, y, float("nan")),
    lambda e, x, y: e.learn(x, y[:-1], LR),
    lambda e, x, y: e.learn(x + [[0.0] * DIM] * 300, y + [0.0] * 300, LR),
    lambda e, x, y: e.learn([row[:-1] for row in x], y, LR),
    lambda e, x, y: e.learn([[True] * DIM] * len(y), y, LR),
    lambda e, x, y: e.learn([["a"] * DIM] * len(y), y, LR),
    lambda e, x, y: e.learn(x, [float("inf")] + y[1:], LR),
    lambda e, x, y: e.learn(array("f", [0.0] * DIM * len(y)), y, LR),
    lambda e, x, y: e.forward([]),
    lambda e, x, y: e.forward_into(array("d", [0.0] * DIM), bytes(8)),
    lambda e, x, y: e.forward_into(array("d", [0.0] * DIM * 2), array("d", [0.0])),
    lambda e, x, y: e.forward_into(array("d", [0.0] * (DIM + 1)), array("d", [0.0] * 2)),
])
def test_bounds_are_refused_and_change_nothing(api, bad):
    w0, x, y = _data()
    with Executor.create(api, DIM, WIDTH, w0) as e:
        before = e.snapshot()
        with pytest.raises(ECSGError) as info:
            bad(e, x, y)
        assert info.value.code in ("INVALID", "NONFINITE", "CAPACITY")
        assert e.snapshot() == before and e.epoch == 0 and e.generation == 0


def test_capacity_is_explicit_and_closed_executors_refuse(api):
    w0, x, y = _data(rows=300)
    e = Executor.create(api, DIM, WIDTH, w0)
    with pytest.raises(ECSGError) as info:
        e.learn(x, y, LR)
    assert info.value.code == "CAPACITY" and e.epoch == 0
    e.reserve(300)
    assert e.learn(x, y, LR).epoch_after == 1
    e.close()
    e.close()
    for call in (lambda: e.forward(x), lambda: e.learn(x, y, LR), lambda: e.snapshot(), lambda: e.epoch,
                 lambda: e.transaction()):
        with pytest.raises(ECSGError) as info:
            call()
        assert info.value.code == "CLOSED"
    with pytest.raises(ECSGError):
        Executor.create(api, DIM, WIDTH, w0, max_rows=0)
    with pytest.raises(ECSGError):
        Executor.restore(api, b"ELPISG01" + bytes(31))


_CLEAN = r"""
import ctypes, json, sys
from array import array
from elpis.ECS.native import ECSGLibrary, Executor
from elpis.ECS.cognition import CognitiveCore
api = ECSGLibrary(ctypes.CDLL(sys.argv[1]))
x = array("d", [0.1 * ((i * 7) % 11 - 5) for i in range(12 * 6)])
y = array("d", [0.05 * (r % 5 - 2) for r in range(12)])
out = array("d", bytes(8 * 12))
with CognitiveCore.create(api, 6, 36, [0.01 * (i % 17 - 8) for i in range(216)], learning_rate=0.002) as core:
    core.query_into(x, out)
    before = out.tolist()
    commit = core.learn(x, y, steps=100, receipt=False)
    core.query_into(x, out)
    with core._state.transaction() as txn:
        txn.learn(x, y, 0.002, 5)
        txn.commit()
print(json.dumps({"changed": before != out.tolist(), "epoch": commit.epoch_after,
                  "modules": sorted(m for m in sys.modules if m.startswith(("elpis", "research", "numpy",
                                                                             "torch")))}))
"""


def test_a_clean_process_runs_the_executor_with_no_model_or_numpy():
    result = subprocess.run([sys.executable, "-c", _CLEAN, str(_library_path())], capture_output=True, text=True,
                            cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["changed"] and report["epoch"] == 100
    assert set(report["modules"]) <= {"elpis", "elpis.ECS", "elpis.ECS.native", "elpis.ECS.cognition"}, \
        report["modules"]
