"""HOT residency of an FMS-resident K1 state bound to the runtime (docs/RUNTIME_CORE.md, "HOT residency").

A HOT operation (QUERY, LEARN) must never trigger a synchronous disk-backed restoration. Binding a state takes the
K1 FMS adapter's residency hold before anything touches it, only if the state is WARM now; while bound the hold pins
it (FMS never demotes a pinned object, so memory pressure picks other victims) and every operation on it pins
resident-only. Unbound and demoted, a HOT operation is refused ``RUNTIME_NOT_RESIDENT`` (no fail-stop, nothing
bound, nothing promoted); the owner materializes it explicitly (a WARM operation) and the same lineage resumes.
Mechanics only: the fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE. The syscall-level proof (no read or write
on any thread) is native: ``runtime.test_runtime_hot_io``; stalled promotion, budget exhaustion, a lost cold
replica and concurrent readers are the K1 FMS adapter's native suite (its residency-hold case).
"""
from __future__ import annotations

import numpy as np
import pytest

from elpis.ECS.k1 import K1Error, K1FMSRuntime
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest
from elpis.runtime.composition import CompositionError
from elpis.substrate.residency import Context

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _config, adapter, initial_w, k1  # noqa: F401 (module fixtures)


class _Pressured:
    """FMS-resident K1 states over a WARM budget of two and a half images: a third state demotes someone."""

    def __init__(self, k1, adapter, tmp_path):   # noqa: F811
        image = k1.envelope_bytes(DIM, WIDTH) - 32
        self.ctx = Context(adapter, warm_bytes=2 * image + image // 2, cold_bytes=10 ** 7, max_objects=8,
                           cold_root=tmp_path / "cold")
        self.runtime = K1FMSRuntime(k1, self.ctx, adapter, max_states=8)
        self.others = []

    def state(self, key):
        w = memoryview(np.ascontiguousarray(initial_w(key)).reshape(-1))
        return self.runtime.state(self.runtime.register(bytes([key]) * 32, DIM, WIDTH, w, max_rows=8))

    def pressure(self, key):
        """Another state arrives and the pump runs (FMS demotes an unpinned victim)."""
        self.others.append(self.state(key))
        self.runtime.pump()

    def close(self):
        for s in self.others:
            self.runtime.close_state(s.id)
        self.runtime.close()
        self.ctx.close()


def test_a_bound_state_is_held_resident_and_a_demoted_one_is_refused_until_explicitly_materialized(
        k1, adapter, tmp_path):   # noqa: F811
    learn = LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN)
    query = QueryRequest("query", ByteTokens(), admitted(FixtureMap()))
    fms = _Pressured(k1, adapter, tmp_path)
    try:
        state = fms.state(1)
        with Runtime(_config(tmp_path / "c")) as runtime:
            runtime.anchor_cognition(state)
            assert fms.runtime.resident_holds(state.id) == 1                # bound: held resident
            for n in range(3):
                fms.pressure(10 + n)                                       # memory pressure while bound
                runtime.run_learn(state, learn)
                runtime.run_query(state, query)
            info = fms.runtime.inspect(state.id)
            assert (info["tier"], info["lease_count"]) == (1, 1)           # WARM; the hold's one pin
            assert any(fms.runtime.inspect(s.id)["tier"] == 2 for s in fms.others)   # someone else was demoted
            live = runtime.live_identity(state)
            durable = runtime.continuity.snapshot()

            # Unbound, it may be demoted. A HOT operation on it is then refused before anything touches it.
            runtime.release(state)
            assert fms.runtime.resident_holds(state.id) == 0
            for n in range(4):
                if fms.runtime.inspect(state.id)["tier"] == 2:
                    break
                fms.pressure(20 + n)
            before = fms.runtime.inspect(state.id)
            assert before["tier"] == 2                                     # COLD
            for operation in (lambda: runtime.run_query(state, query), lambda: runtime.run_learn(state, learn)):
                with pytest.raises(CompositionError) as info:
                    operation()
                assert info.value.code == "RUNTIME_NOT_RESIDENT"
            after = fms.runtime.inspect(state.id)
            # Nothing pinned, promoted or materialized; no hold left behind; no fail-stop; nothing durable moved.
            assert (after["tier"], after["lease_count"], after["acquisitions"], after["materialization_ns"]) == \
                (before["tier"], 0, before["acquisitions"], before["materialization_ns"])
            assert fms.runtime.resident_holds(state.id) == 0
            assert runtime.fault is None and runtime.continuity.snapshot() == durable

            # The explicit WARM-lane materialization; the same live lineage resumes (the release kept it).
            state.materialize()
            assert fms.runtime.inspect(state.id)["tier"] == 1
            assert runtime.live_identity(state) == live
            result = runtime.run_learn(state, learn)
            assert result.generation_after == live[1] + 1
            assert fms.runtime.resident_holds(state.id) == 1
        assert fms.runtime.resident_holds(state.id) == 0                   # closing the runtime unbinds it
        with pytest.raises(K1Error):
            fms.runtime.inspect(state.id) and fms.runtime.materialize(10 ** 9)   # an unknown state is refused
        fms.runtime.close_state(state.id)
    finally:
        fms.close()
