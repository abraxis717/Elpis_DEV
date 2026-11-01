"""HOT residency of an FMS-resident K1 state bound to the runtime (docs/RUNTIME_CORE.md, "HOT residency").

A HOT operation (QUERY, LEARN) must never trigger a synchronous restoration. Under FMS ABI v3 residency is volatile:
WARM in RAM, no COLD tier, no storage, nothing ever demoted below WARM, and a state beyond the WARM budget refused
(``CAPACITY``), never spilled. Three invariants, each its own test so that each discriminates on its own:

* **A. Residency protection.** Binding a state takes the K1 FMS adapter's residency hold before anything touches it;
  memory pressure refuses the newcomer and never displaces the bound state, and HOT work on it promotes nothing,
  demotes nothing and reads nothing.
* **B. No demotion at all.** A released (unbound) state is not displaced either: pressure refuses the newcomer, the
  released state stays WARM-resident with exactly the live bytes the release gave back, and the runtime binds it
  again with no restoration of any kind.
* **C. Materialization is a check.** The owner's explicit materialization finds the state WARM and moves nothing
  (before FMS v3 it was the one path that read a cold replica back); the live lineage then resumes, held again.

``RUNTIME_NOT_RESIDENT`` (a bound state not WARM) is unreachable over a RAM-only context: nothing demotes and a K1
state is never moved to a device. The runtime's check stays (defensive) and is the native suites' to exercise.

Mechanics only: the fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE. The syscall-level proof (no read or write
on any thread) is native: ``runtime.test_runtime_hot_io``; budget exhaustion and concurrent readers are the K1 FMS
adapter's native suite (its residency-hold case).
"""
from __future__ import annotations

import numpy as np
import pytest

from elpis.ECS.k1 import K1Error, K1FMSRuntime
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest
from elpis.substrate.residency import Context

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _config, adapter, initial_w, k1  # noqa: F401 (module fixtures)

WARM = 1


class _Residency:
    """FMS-resident K1 states over a WARM budget of two and a half images: two fit, a third is refused.

    It tracks every state it registered and every hold it took itself (never the runtime's), so that closing it
    after a failure reports the failure rather than a leftover state, and closing it after a pass proves that no
    hold leaked: a held state cannot be closed (``BUSY``).
    """

    def __init__(self, k1, adapter):   # noqa: F811
        image = k1.envelope_bytes(DIM, WIDTH) - 32
        self.ctx = Context(adapter, warm_bytes=2 * image + image // 2, max_objects=8)
        self.runtime = K1FMSRuntime(k1, self.ctx, adapter, max_states=8)
        self.states, self.held = [], []

    def __enter__(self):
        return self

    def state(self, key):
        w = memoryview(np.ascontiguousarray(initial_w(key)).reshape(-1))
        state = self.runtime.state(self.runtime.register(bytes([key]) * 32, DIM, WIDTH, w, max_rows=8))
        self.states.append(state)
        return state

    def refused(self, key):
        """A newcomer at the bound: refused ``CAPACITY``, nothing registered and nothing displaced."""
        tiers = [self.tier(s) for s in self.states]
        with pytest.raises(K1Error) as refused:
            self.state(key)
        assert refused.value.code == "CAPACITY"
        assert [self.tier(s) for s in self.states] == tiers == [WARM] * len(tiers)

    def tier(self, state):
        return self.runtime.inspect(state.id)["tier"]

    def holds(self, state):
        return self.runtime.resident_holds(state.id)

    def residency(self):
        """The FMS context's promotion, demotion and cold-store counters (all of them stay 0 under FMS v3)."""
        r = self.runtime.metrics()["residency"]
        return {k: r[k] for k in ("promotions", "demotions", "cold_reads", "cold_writes")}

    def hold(self, state):
        self.runtime.resident_hold(state.id)
        self.held.append(state)

    def __exit__(self, exc_type, *_):
        for s in self.held:
            self.runtime.resident_release(s.id)
        for s in self.states:
            try:
                self.runtime.close_state(s.id)
            except K1Error:
                if exc_type is None:
                    raise   # a pass must leave nothing held or open; a failure keeps its own report
        self.runtime.close()
        self.ctx.close()


NOTHING_MOVED = {"promotions": 0, "demotions": 0, "cold_reads": 0, "cold_writes": 0}


def _requests():
    return (LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN),
            QueryRequest("query", ByteTokens(), admitted(FixtureMap())))


def test_a_a_bound_state_is_never_displaced_by_memory_pressure_and_hot_work_on_it_moves_nothing(
        k1, adapter):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter) as fms:
        state = fms.state(1)
        with Runtime(_config()) as runtime:
            runtime.anchor_cognition(state)
            assert fms.holds(state) == 1 and fms.runtime.inspect(state.id)["lease_count"] == 1   # bound: held
            assert fms.residency() == NOTHING_MOVED

            # Pressure: a second state fits, every further arrival is refused; HOT work between arrivals runs on
            # the bound state's WARM image.
            other = fms.state(10)
            for n in range(1, 4):
                fms.refused(10 + n)
                fms.runtime.pump()
                runtime.run_learn(state, learn)
                runtime.run_query(state, query)
                assert fms.tier(state) == fms.tier(other) == WARM

            info = fms.runtime.inspect(state.id)
            assert (info["tier"], info["lease_count"], info["transaction_open"]) == (WARM, 1, 0)
            assert fms.residency() == NOTHING_MOVED
            assert fms.holds(state) == 1
        assert fms.holds(state) == 0   # closing the runtime drops its hold


def test_b_a_released_state_is_not_displaced_either_and_binds_again_without_restoration(
        k1, adapter):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter) as fms:
        state = fms.state(1)
        with Runtime(_config()) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, learn)   # volatile live work the release keeps
            live = runtime.live_identity(state)
            answer = runtime.run_query(state, query).readout
            runtime.release(state)
            assert fms.holds(state) == 0 and fms.runtime.inspect(state.id)["lease_count"] == 0

            fms.state(2)
            fms.refused(3)   # the released state is the one unheld state: still nothing is demoted
            assert fms.tier(state) == WARM
            assert fms.runtime.state_digest(state.id) == live[0]

            # The next cognitive operation binds the released lineage again: no restoration, the same bytes at the
            # same generation, held again.
            assert runtime.live_identity(state) == live and fms.holds(state) == 1
            again = runtime.run_query(state, query).readout
            assert (again.values, again.live_generation) == (answer.values, live[1])
            assert fms.residency() == NOTHING_MOVED
            assert runtime.fault is None


def test_c_explicit_materialization_is_a_residency_check_and_the_live_lineage_resumes(
        k1, adapter):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter) as fms:
        state = fms.state(1)
        with Runtime(_config()) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, learn)
            answer = runtime.run_query(state, query).readout
            live = runtime.live_identity(state)
            runtime.release(state)
            fms.state(2)
            fms.refused(3)

            # The explicit WARM-lane materialization finds the state WARM: nothing is promoted or read.
            state.materialize()
            assert fms.tier(state) == WARM and fms.residency() == NOTHING_MOVED

            # The same live lineage resumes, held again: the exact bytes the release gave back, at their generation.
            assert runtime.live_identity(state) == live
            assert fms.holds(state) == 1
            again = runtime.run_query(state, query).readout
            assert (again.values, again.live_generation) == (answer.values, answer.live_generation) == \
                (answer.values, live[1])
            result = runtime.run_learn(state, learn)
            assert (result.generation_before, result.generation_after) == (live[1], live[1] + 1)
            assert runtime.run_query(state, query).readout.live_generation == live[1] + 1
            assert fms.residency() == NOTHING_MOVED and runtime.fault is None
        assert fms.holds(state) == 0
