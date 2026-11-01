"""Mutable ECS FMS residency mechanics; no scientific claims.

FMS ABI v3: residency is volatile (WARM in RAM; no COLD tier, no storage, no scratch root). A state beyond the WARM
budget is refused (``CAPACITY``), never spilled, and the pump never moves a resident state."""
from array import array
import ctypes
import hashlib
from pathlib import Path
import random
import struct

import pytest

from elpis.ECS.native import ECSGLibrary, ECSGError, Executor
from elpis.ECS.cognition import CognitiveCore
from elpis.ECS.residency import FMSRuntime
from elpis.substrate.residency import Context
from .test_math_r0 import _library_path


def _adapter_path():
    math = Path(_library_path())
    candidate = math.with_name("libelpis_ecsg_fms.so")
    if candidate.is_file():
        return candidate
    raise AssertionError(f"mutable FMS adapter not built beside ECS library: {candidate}")


@pytest.fixture
def runtime():
    math = ctypes.CDLL(str(_library_path()))
    adapter = ctypes.CDLL(str(_adapter_path()))
    api = ECSGLibrary(math)
    snapshot = 40 + 6 * 72 * 8
    with Context(adapter, warm_bytes=2 * snapshot,
                 max_objects=16) as ctx:
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


def test_identity_isolation_bounded_residency_and_accounting(runtime):
    _, r = runtime
    w, x, y = data(717, 72, 32)
    snapshot = 40 + 6 * 72 * 8
    states = []
    try:
        for i in range(2):                    # the WARM budget holds exactly two states of this shape
            s = r.create(key(i), 6, 72, w)
            states.append(s)
            s.learn(x, array("d", (v + i * 0.1 for v in y)), 0.002, i + 1)
        with pytest.raises(ECSGError, match="CAPACITY"):
            r.create(key(2), 6, 72, w)        # refused at the bound, never spilled
        infos = [r.inspect(s) for s in states]
        identities = [i["logical_identity"] for i in infos]
        assert len(set(identities)) == 2
        expected = hashlib.sha256(b"elpis.ecsg.logical.v1\0" + key(0) + struct.pack("<QQ", 6, 72)).hexdigest()
        assert identities[0] == expected
        answers = [s.forward(x) for s in states]
        assert len(set(answers)) == 2
        a, b = states
        assert a.forward(x) == answers[0]
        b.learn(x, y, 0.002, 10)
        r.pump()
        assert a.forward(x) == answers[0] and b.forward(x) != answers[1]
        assert a.epoch == 1 and b.epoch == 12
        assert r.inspect(a)["logical_identity"] == identities[0]
        assert r.inspect(b)["logical_identity"] == identities[1]
        assert all(r.inspect(s)["tier"] == 1 and r.inspect(s)["cold_replica"] == 0 for s in states)
        m = r.stats()
        assert m["states"] == 2 and m["logical_bytes"] == 2 * snapshot
        assert m["resident_authoritative_bytes"] == m["logical_bytes"]       # everything is resident, nothing spilled
        assert m["resident_high_water"] <= 2 * snapshot
        assert m["leases"] == 0 and m["residency"]["pinned_bytes"] == 0
        assert m["residency"]["cold_reads"] == m["residency"]["cold_writes"] == 0
        assert m["residency"]["demotions"] == 0 and m["residency"]["tier_bytes"][2] == 0
        assert m["residency"]["domain_bytes"][2] == 0
        assert m["residency"]["tier_bytes"][0] == 0
        assert m["active_workspace_bytes"] == 0
        with pytest.raises(ECSGError, match="INVALID"):
            r.create(key(0), 6, 72, w)
        with pytest.raises(ECSGError, match="BUSY"):
            r.close()
        states.pop().close()                  # bounded, not stuck: a closed state frees its place
        states.append(r.create(key(3), 6, 72, w))
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


def test_capacity_bound_and_volatile_lifecycle():
    math = ctypes.CDLL(str(_library_path()))
    adapter = ctypes.CDLL(str(_adapter_path()))
    api = ECSGLibrary(math)
    size = 40 + 6 * 36 * 8
    w, x, y = data(31, 36, 64)
    with Context(adapter, warm_bytes=size, max_objects=4) as ctx, FMSRuntime(api, ctx, adapter, max_states=4) as r:
        with r.create(key(1), 6, 36, w) as a:
            assert r.inspect(a)["tier"] == 1
            with pytest.raises(ECSGError, match="CAPACITY"):
                r.create(key(2), 6, 36, w)    # one state fits: the second is refused, the first is not displaced
            with a.transaction():
                r.pump()
                info = r.inspect(a)
                assert info["tier"] == 1 and info["lease_count"] == 1
            r.pump()
            assert r.inspect(a)["tier"] == 1
            a.forward(x)
            a.learn(x, y, 0.002, 3)
            learned = a.snapshot()
            r.pump()
            assert r.inspect(a)["tier"] == 1 and a.snapshot() == learned
            assert r.inspect(a)["cold_replica"] == 0
            residency = r.stats()["residency"]
            assert residency["demotions"] == residency["cold_reads"] == residency["cold_writes"] == 0
            assert residency["tier_bytes"][2] == 0 and residency["domain_bytes"][2] == 0
        with r.create(key(2), 6, 36, w) as b:   # the closed state's place is free again
            assert r.inspect(b)["tier"] == 1


def test_the_retired_cold_root_is_not_a_parameter():
    adapter = ctypes.CDLL(str(_adapter_path()))
    with pytest.raises(TypeError):
        Context(adapter, warm_bytes=4096, cold_bytes=4096, max_objects=4, cold_root="/nonexistent")
    with pytest.raises(TypeError):
        Context(adapter, warm_bytes=4096, max_objects=4, cold_root="/nonexistent")
