"""Mutable ECS_G FMS residency mechanics; no scientific claims."""
from array import array
import ctypes
import hashlib
from pathlib import Path
import random
import struct

import pytest

from elpis.ECS_G.native import ECSGLibrary, ECSGError, Executor
from elpis.ECS_G.cognition import CognitiveCore
from elpis.ECS_G.residency import FMSRuntime
from elpis.substrate.residency import Context
from .test_math_r0 import _library_path


def _adapter_path():
    math = Path(_library_path())
    candidate = math.with_name("libelpis_ecsg_fms.so")
    if candidate.is_file():
        return candidate
    raise AssertionError(f"mutable FMS adapter not built beside ECS_G library: {candidate}")


@pytest.fixture
def runtime(tmp_path):
    math = ctypes.CDLL(str(_library_path()))
    adapter = ctypes.CDLL(str(_adapter_path()))
    api = ECSGLibrary(math)
    snapshot = 40 + 6 * 72 * 8
    with Context(adapter, warm_bytes=2 * snapshot, cold_bytes=100000,
                 max_objects=16, cold_root=tmp_path / "cold") as ctx:
        with FMSRuntime(api, ctx, adapter, max_states=16) as r:
            assert ctx.handle.value is None
            yield api, r


def key(n):
    return n.to_bytes(32, "little")


def data(seed, width, rows):
    r = random.Random(seed)
    return (array("d", (r.uniform(-0.15, 0.15) for _ in range(6 * width))),
            array("d", (r.uniform(-0.4, 0.4) for _ in range(6 * rows))),
            array("d", (r.uniform(-0.5, 0.5) for _ in range(rows))))


@pytest.mark.parametrize("width", [36, 48, 72])
@pytest.mark.parametrize("rows", [1, 7, 17, 64])
@pytest.mark.parametrize("steps", [1, 5, 37])
def test_parity(runtime, width, rows, steps):
    api, r = runtime
    w, x, y = data(width + rows + steps, width, rows)
    with r.create(key(1), 6, width, w, max_rows=rows) as s, \
            Executor.create(api, 6, width, w, max_rows=rows) as e:
        assert type(s) is Executor
        assert s.snapshot() == e.snapshot()
        assert s.forward(x) == e.forward(x)
        assert s.learn(x, y, 0.002, steps) == e.learn(x, y, 0.002, steps)
        assert s.snapshot() == e.snapshot() and s.s3() == e.s3()
        with s.transaction() as a, e.transaction() as b:
            schedule = [rows] if rows == 1 else [1, rows - 1]
            assert a.learn_schedule(x, y, schedule, 0.002, steps) == b.learn_schedule(x, y, schedule, 0.002, steps)
            assert a.forward(x) == b.forward(x) and a.s3() == b.s3() and a.epoch == b.epoch
            assert a.commit() == b.commit()
        assert s.snapshot() == e.snapshot()
        r.pump()
        assert s.forward(x) == e.forward(x)
        blob = s.snapshot()
        with Executor.restore(api, blob, max_rows=rows) as restored:
            assert restored.forward(x) == s.forward(x)
        with r.restore(key(2), e.snapshot(), max_rows=rows) as restored:
            assert type(restored) is Executor
            assert restored.snapshot() == blob and restored.forward(x) == e.forward(x)


def test_identity_scaling_isolation_and_accounting(runtime):
    _, r = runtime
    w, x, y = data(717, 72, 32)
    states = []
    try:
        for i in range(10):
            s = r.create(key(i), 6, 72, w)
            states.append(s)
            s.learn(x, array("d", (v + i * 0.1 for v in y)), 0.002, i + 1)
        infos = [r.inspect(s) for s in states]
        identities = [i["logical_identity"] for i in infos]
        assert len(set(identities)) == 10
        expected = hashlib.sha256(b"elpis.ecsg.logical.v1\0" + key(0) + struct.pack("<QQ", 6, 72)).hexdigest()
        assert identities[0] == expected
        answers = [s.forward(x) for s in states]
        assert len(set(answers)) == 10
        a, b = states[:2]
        assert a.forward(x) == answers[0]
        b.learn(x, y, 0.002, 10)
        r.pump()
        assert a.forward(x) == answers[0] and b.forward(x) != answers[1]
        assert a.epoch == 1 and b.epoch == 12
        assert r.inspect(a)["logical_identity"] == identities[0]
        assert r.inspect(b)["logical_identity"] == identities[1]
        m = r.stats()
        snapshot = 40 + 6 * 72 * 8
        assert m["states"] == 10 and m["logical_bytes"] == 10 * snapshot
        assert m["logical_bytes"] > m["resident_authoritative_bytes"]
        assert m["resident_authoritative_bytes"] <= 2 * snapshot
        assert m["resident_high_water"] <= 2 * snapshot
        assert m["leases"] == 0 and m["residency"]["pinned_bytes"] == 0
        assert m["residency"]["cold_reads"] > 0 and m["residency"]["cold_writes"] > 0
        assert m["residency"]["tier_bytes"][0] == 0
        assert m["active_workspace_bytes"] == 0
        with pytest.raises(ECSGError, match="INVALID"):
            r.create(key(0), 6, 72, w)
        with pytest.raises(ECSGError, match="BUSY"):
            r.close()
    finally:
        for s in states:
            s.close()
            s.close()


def test_refusals_transactions_and_cognitive_core(runtime):
    api, r = runtime
    w, x, y = data(31, 36, 64)
    with r.create(key(1), 6, 36, w) as s:
        assert type(s) is Executor
        core = CognitiveCore(s, learning_rate=0.002)
        with Executor.create(api, 6, 36, w) as e:
            other = CognitiveCore(e, learning_rate=0.002)
            assert core.learn(x, y, steps=7) == other.learn(x, y, steps=7)
            assert core.query(x) == other.query(x)
        before, generation = s.snapshot(), s.generation
        for rate, targets in [(1e200, y), (0.002, array("d", [float("nan")] * 64))]:
            with pytest.raises(ECSGError, match="NONFINITE"):
                s.learn(x, targets, rate, 10)
            assert s.snapshot() == before and s.generation == generation
            assert r.inspect(s)["lease_count"] == 0
        with s.transaction() as t:
            with pytest.raises(ECSGError, match="BUSY"):
                s.close()
            t.learn(x, y, 0.002, 2)
            assert s.snapshot() == before
            r.pump()
            info = r.inspect(s)
            assert info["lease_count"] == 1 and info["tier"] == 1
        assert s.snapshot() == before and r.inspect(s)["lease_count"] == 0
        with s.transaction() as t:
            t.learn(x, y, 0.002, 2)
            s.learn(x, y, 0.002, 1)
            moved = s.snapshot()
            with pytest.raises(ECSGError, match="STALE"):
                t.commit()
            assert s.snapshot() == moved and r.inspect(s)["lease_count"] == 0
        with s.transaction() as t:
            with pytest.raises(ECSGError, match="NONFINITE"):
                t.learn(x, y, 1e200, 10)
            assert not t.open and r.inspect(s)["lease_count"] == 0
        s.reserve(512)
        assert s.max_rows == 512 and s.snapshot() == moved


def test_capacity_and_cold_lifecycle(tmp_path):
    math = ctypes.CDLL(str(_library_path()))
    adapter = ctypes.CDLL(str(_adapter_path()))
    api = ECSGLibrary(math)
    size = 40 + 6 * 36 * 8
    w, x, y = data(31, 36, 64)
    with Context(adapter, warm_bytes=size, cold_bytes=size * 8, max_objects=4,
                 cold_root=tmp_path / "cold") as ctx, FMSRuntime(api, ctx, adapter, max_states=4) as r:
        with r.create(key(1), 6, 36, w) as a, r.create(key(2), 6, 36, w) as b:
            assert r.inspect(a)["tier"] == 2
            with a.transaction():
                with pytest.raises(ECSGError, match="CAPACITY"):
                    b.forward(x)
                r.pump()
                info = r.inspect(a)
                assert info["tier"] == 1 and info["lease_count"] == 1
            r.pump()
            assert r.inspect(a)["tier"] == 2
            a.forward(x)
            assert r.inspect(a)["cold_replica"] == 1
            a.learn(x, y, 0.002, 3)
            assert r.inspect(a)["cold_replica"] == 0
            learned = a.snapshot()
            r.pump()
            assert r.inspect(a)["tier"] == 2
            assert a.snapshot() == learned
            assert r.inspect(a)["cold_replica"] == 1
