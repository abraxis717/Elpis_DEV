"""Native K1 runtime through its Python control plane (docs/ECS_K1_RUNTIME.md). Mechanics, no scientific claim.

The scientific qualification of the native law against Retention R3 is research/ecs_k1_native (differential over
every R3 QUAL world). Here: Runtime R1 parity with an empty consolidation, one native call per operation whatever K,
query independence from (H, a), refusal atomicity, complete-state transactions, the retained-state envelope, W-only
imports, and the FMS warm path over resident bytes.
"""
from array import array
import ctypes
from pathlib import Path
import random

import pytest

from elpis.ECS_G.k1 import K1Error, K1FMSRuntime, K1Library, K1State
from elpis.ECS_G.native import ECSGLibrary, Executor
from elpis.substrate.residency import Context

from .test_math_r0 import _library_path

D, N, R = 6, 36, 64


def _beside(name):
    path = Path(_library_path()).with_name(name)
    if not path.is_file():
        raise AssertionError(f"{name} not built beside the ECS_G library: {path}")
    return path


@pytest.fixture(scope="module")
def libs():
    math = ctypes.CDLL(str(_library_path()))
    return ECSGLibrary(math), K1Library(ctypes.CDLL(str(_beside("libelpis_ecsg_k1.so"))))


def data(seed, rows=R):
    r = random.Random(seed)
    return (array("d", (r.uniform(-0.18, 0.18) for _ in range(D * N))),
            array("d", (r.uniform(-0.5, 0.5) for _ in range(D * rows))),
            array("d", (r.uniform(-0.4, 0.4) for _ in range(rows))))


@pytest.mark.parametrize("steps", [1, 7, 400])
def test_empty_consolidation_is_bitwise_runtime_r1(libs, steps):
    api, k1 = libs
    w, x, y = data(steps)
    with Executor.create(api, D, N, w, max_rows=R) as e, K1State.create(k1, D, N, w, max_rows=R) as s:
        assert e.learn(x, y, 0.002, steps).epoch_after == s.learn(x, y, 0.002, steps).epoch_after == steps
        assert e.w() == s.w()
        assert e.forward(x) == s.query(x)
        assert s.stats()["corrected_steps"] == 0


def test_one_native_call_per_operation_whatever_k(libs):
    _, k1 = libs
    w, x, y = data(3)
    with K1State.create(k1, D, N, w, max_rows=R) as s:
        s.consolidate(x)
        s.learn(x, y, 0.002, 4000)
        stats = s.stats()
        assert stats["learn_calls"] == 1 and stats["steps_executed"] == 4000 and stats["corrected_steps"] == 4000
        assert stats["consolidations"] == 1 and stats["heap_allocations"] == 3


def test_query_reads_w_only_and_consolidation_shapes_learning(libs):
    _, k1 = libs
    w, x, y = data(5)
    _, x2, y2 = data(6)
    with K1State.create(k1, D, N, w) as a, K1State.create(k1, D, N, w) as b:
        a.consolidate(x)
        assert a.query(x2) == b.query(x2)
        assert a.h_packed() != b.h_packed()
        a.learn(x2, y2, 0.002, 1)
        b.learn(x2, y2, 0.002, 1)
        assert a.w() == b.w()                     # at the anchor u = 0: the first step is identical
        a.learn(x2, y2, 0.002, 30)
        b.learn(x2, y2, 0.002, 30)
        assert a.w() != b.w()


def test_refusals_leave_the_complete_state_unchanged(libs):
    _, k1 = libs
    w, x, y = data(7)
    with K1State.create(k1, D, N, w, max_rows=R) as s:
        s.learn(x, y, 0.002, 20)
        s.consolidate(x)
        before = s.snapshot()
        bad = array("d", x)
        bad[11] = float("nan")
        for call in (lambda: s.learn(bad, y, 0.002, 3), lambda: s.consolidate(bad),
                     lambda: s.learn(x, y, 1e9, 30), lambda: s.learn(x + x, y + y, 0.002, 1),
                     lambda: s.learn(x, y, -1.0, 1)):
            with pytest.raises(K1Error):
                call()
        assert s.snapshot() == before
        with pytest.raises(K1Error) as exc:
            s.learn(x, y, 1e9, 30)
        assert exc.value.code == "NONFINITE" and exc.value.step >= 1


def test_transactions_commit_w_epoch_h_and_a_together(libs):
    _, k1 = libs
    w, x, y = data(9)
    with K1State.create(k1, D, N, w) as s, K1State.create(k1, D, N, w) as direct:
        before = s.snapshot()
        with s.transaction() as t:
            t.learn(x, y, 0.002, 25)
            t.consolidate(x)
            assert s.snapshot() == before and t.epoch() == 25
            commit = t.commit()
        assert commit.epoch_after == 25 and commit.generation_after == commit.generation_before + 1
        direct.learn(x, y, 0.002, 25)
        direct.consolidate(x)
        assert s.snapshot() == direct.snapshot()
        t = s.transaction()
        t.learn(x, y, 0.002, 5)
        s.consolidate(x)                         # replaces the source
        with pytest.raises(K1Error) as exc:
            t.commit()
        assert exc.value.code == "STALE"


def test_envelope_is_deterministic_and_corruption_is_refused(libs):
    _, k1 = libs
    w, x, y = data(11)
    with K1State.create(k1, D, N, w) as s:
        s.learn(x, y, 0.002, 12)
        s.consolidate(x)
        env = s.snapshot()
        assert env[:8] == b"ELPISGK1" and len(env) == k1.envelope_bytes(D, N) == 30376
        with K1State.restore(k1, env) as r:
            assert r.snapshot() == env and r.epoch == 12 and r.provenance == "COMPLETE"
        for i in (0, 9, 20, 47, 100, 3000, len(env) - 1):
            bad = bytearray(env)
            bad[i] ^= 1
            with pytest.raises(K1Error) as exc:
                K1State.restore(k1, bytes(bad))
            assert exc.value.code == "CORRUPT"


def test_w_only_snapshot_imports_only_as_unconsolidated(libs):
    api, k1 = libs
    w, x, y = data(13)
    with Executor.create(api, D, N, w) as e:
        e.learn(x, y, 0.002, 9)
        snap = e.snapshot()
        with pytest.raises(K1Error):
            K1State.restore(k1, snap)
        with K1State.import_w_only(k1, snap) as s:
            assert s.provenance == "UNCONSOLIDATED_IMPORT" and s.epoch == 9
            assert s.w() == e.w() and not any(s.h_packed()) and not any(s.a())


def test_fms_warm_path_runs_over_resident_bytes(libs, tmp_path):
    _, k1 = libs
    adapter = ctypes.CDLL(str(_beside("libelpis_ecsg_k1_fms.so")))
    w, x, y = data(15)
    image = k1.envelope_bytes(D, N) - 32
    with Context(adapter, warm_bytes=8 * image, cold_bytes=10 ** 6, max_objects=4,
                 cold_root=tmp_path / "cold") as ctx, K1FMSRuntime(k1, ctx, adapter, max_states=4) as r, \
            K1State.create(k1, D, N, w) as s:
        assert ctx.handle.value is None
        sid = r.register(b"\x01" * 32, D, N, w)
        for target in (r, s):
            args = (sid,) if target is r else ()
            target.learn(*args, x, y, 0.002, 200)
            target.consolidate(*args, x)
        with r.transaction(sid) as t:
            t.learn(x, y, 0.002, 50)
            t.consolidate(x)
            t.commit()
        with s.transaction() as t:
            t.learn(x, y, 0.002, 50)
            t.consolidate(x)
            t.commit()
        assert r.snapshot(sid) == s.snapshot()
        before = r.k1_stats(sid)
        answers = {r.query(sid, x) for _ in range(100)}
        after = r.k1_stats(sid)
        assert answers == {s.query(x)}
        assert after["heap_allocations"] == before["heap_allocations"] == 2      # the workspace, once
        assert after["forward_calls"] == before["forward_calls"] + 100
        info = r.inspect(sid)
        assert info["tier"] == 1 and info["lease_count"] == 0 and info["epoch"] == 250
        r.close_state(sid)
