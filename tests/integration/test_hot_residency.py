"""HOT residency of an FMS-resident K1 state bound to the runtime (docs/RUNTIME_CORE.md, "HOT residency").

A HOT operation (QUERY, LEARN) must never trigger a synchronous disk-backed restoration. Three invariants, each its
own test so that each discriminates on its own:

* **A. Residency protection.** Binding a state takes the K1 FMS adapter's residency hold before anything touches it;
  while bound the hold pins it, so memory pressure demotes other states or refuses the newcomer, and HOT work on it
  promotes nothing and reads no cold replica.
* **B. Nonresident refusal.** Unbound and demoted, a managed QUERY, LEARN or durable request is refused
  ``RUNTIME_NOT_RESIDENT``: nothing is promoted, read from the cold store, pinned or held, K1 is unchanged, nothing
  durable moves, no ticket is issued and the runtime is not fail-stopped.
* **C. Explicit restoration.** The owner's explicit materialization (WARM work) is the one path that reads the cold
  replica back; the released live lineage then resumes, held resident again.

Which *unpinned* WARM state FMS demotes is recency- and frequency-weighted on the PAL's wall clock (``fms_core.c``,
``evict_cost_ns``; a release counts as an access), so it is no contract that a released state is the next victim.
Every precondition here is therefore made exact through FMS's one hard law, a pinned object is never demoted: the
fixture holds every other WARM state (the owner's residency hold, ``K1FMSRuntime.resident_hold``) so that exactly one
state is eligible, and only then admits a newcomer.

Mechanics only: the fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE. The syscall-level proof (no read or write
on any thread) is native: ``runtime.test_runtime_hot_io``; stalled promotion, budget exhaustion, a lost cold replica
and concurrent readers are the K1 FMS adapter's native suite (its residency-hold case).
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

WARM, COLD = 1, 2


class _Residency:
    """FMS-resident K1 states over a WARM budget of two and a half images: two fit, a third displaces one.

    It tracks every state it registered and every hold it took itself (never the runtime's), so that closing it
    after a failure reports the failure rather than a leftover state, and closing it after a pass proves that no
    hold leaked: a held state cannot be closed (``BUSY``).
    """

    def __init__(self, k1, adapter, tmp_path):   # noqa: F811
        image = k1.envelope_bytes(DIM, WIDTH) - 32
        self.ctx = Context(adapter, warm_bytes=2 * image + image // 2, cold_bytes=10 ** 7, max_objects=8,
                           cold_root=tmp_path / "cold")
        self.runtime = K1FMSRuntime(k1, self.ctx, adapter, max_states=8)
        self.states, self.held = [], []

    def __enter__(self):
        return self

    def state(self, key):
        w = memoryview(np.ascontiguousarray(initial_w(key)).reshape(-1))
        state = self.runtime.state(self.runtime.register(bytes([key]) * 32, DIM, WIDTH, w, max_rows=8))
        self.states.append(state)
        return state

    def tier(self, state):
        return self.runtime.inspect(state.id)["tier"]

    def holds(self, state):
        return self.runtime.resident_holds(state.id)

    def residency(self):
        """The FMS context's promotion, demotion and cold-store counters."""
        r = self.runtime.metrics()["residency"]
        return {k: r[k] for k in ("promotions", "demotions", "cold_reads", "cold_writes")}

    def hold(self, state):
        self.runtime.resident_hold(state.id)
        self.held.append(state)

    def isolate(self, victim):
        """Hold every other WARM state that nobody holds yet: ``victim`` becomes the only state FMS may demote."""
        for s in self.states:
            if s is not victim and self.tier(s) == WARM and self.holds(s) == 0:
                self.hold(s)

    def demote(self, victim, key):
        """Demote exactly ``victim``: every other WARM state is held, then a newcomer arrives (WARM must make room)."""
        self.isolate(victim)
        others = [s for s in self.states if s is not victim and self.tier(s) == WARM]
        newcomer = self.state(key)
        assert self.tier(victim) == COLD, "the only demotable WARM state was not the one demoted"
        assert all(self.tier(s) == WARM for s in others) and self.tier(newcomer) == WARM
        return newcomer

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


def _requests():
    return (LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN),
            QueryRequest("query", ByteTokens(), admitted(FixtureMap())))


def test_a_bound_state_is_never_demoted_by_memory_pressure_and_hot_work_on_it_never_promotes(
        k1, adapter, tmp_path):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter, tmp_path) as fms:
        state = fms.state(1)
        with Runtime(_config(tmp_path / "c")) as runtime:
            runtime.anchor_cognition(state)
            assert fms.holds(state) == 1 and fms.runtime.inspect(state.id)["lease_count"] == 1   # bound: held
            before = fms.residency()

            # Pressure with one eligible competitor: each arrival must displace the previous arrival, never the bound
            # state, and HOT work between arrivals runs on the WARM image.
            arrivals = [fms.state(10)]
            for n in range(1, 4):
                arrivals.append(fms.state(10 + n))
                fms.runtime.pump()
                runtime.run_learn(state, learn)
                runtime.run_query(state, query)
                assert fms.tier(state) == WARM and fms.tier(arrivals[-2]) == COLD
            assert [fms.tier(s) for s in arrivals] == [COLD, COLD, COLD, WARM]

            # No eligible competitor at all: every other WARM state is held. FMS refuses the newcomer rather than
            # demote the bound state (without the runtime's hold, the bound state would be the one victim).
            fms.isolate(state)
            with pytest.raises(K1Error) as refused:
                fms.state(20)
            assert refused.value.code == "CAPACITY"
            runtime.run_learn(state, learn)
            runtime.run_query(state, query)

            info = fms.runtime.inspect(state.id)
            assert (info["tier"], info["lease_count"], info["transaction_open"]) == (WARM, 1, 0)
            after = fms.residency()
            # The bound state was never promoted and no cold replica was read; only the arrivals were demoted.
            assert (after["promotions"], after["cold_reads"]) == (before["promotions"], before["cold_reads"])
            assert after["demotions"] == before["demotions"] + 3
            assert fms.holds(state) == 1
        assert fms.holds(state) == 0   # closing the runtime drops its hold


def test_b_hot_operations_on_a_demoted_state_are_refused_and_touch_nothing(k1, adapter, tmp_path):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter, tmp_path) as fms:
        state = fms.state(1)
        with Runtime(_config(tmp_path / "c")) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, learn)   # volatile live work the release keeps
            live = runtime.live_identity(state)
            runtime.release(state)
            assert fms.holds(state) == 0 and fms.runtime.inspect(state.id)["lease_count"] == 0

            fms.state(2)
            fms.demote(state, 3)   # the deterministic precondition: COLD, its replica the only copy
            durable, status = runtime.continuity.snapshot(), runtime.durability()
            info, counters = fms.runtime.inspect(state.id), fms.residency()

            refusals = (lambda: runtime.run_query(state, query), lambda: runtime.run_learn(state, learn),
                        lambda: runtime.publish_identity(state))
            for operation in refusals:
                with pytest.raises(CompositionError) as refused:
                    operation()
                assert refused.value.code == "RUNTIME_NOT_RESIDENT"

            # Not promoted, no cold read, nothing pinned, held or acquired, no transaction, K1 unchanged.
            assert fms.residency() == counters
            assert fms.runtime.inspect(state.id) == info and info["tier"] == COLD and info["lease_count"] == 0
            assert fms.holds(state) == 0
            # No fail-stop; nothing durable moved; no ticket was issued and the lane is untouched.
            assert runtime.fault is None
            assert runtime.continuity.snapshot() == durable and runtime.durability() == status
        # On disk too, continuity is the record the refusals found; and the demoted bytes are exactly the live state
        # the release gave back (read only after the explicit materialization; test C proves the resumption).
        with Runtime(_config(tmp_path / "c")) as reopened:
            assert reopened.continuity.snapshot() == durable
        state.materialize()
        assert fms.runtime.state_digest(state.id) == live[0]


def test_c_explicit_materialization_restores_residency_and_resumes_the_live_lineage(
        k1, adapter, tmp_path):   # noqa: F811
    learn, query = _requests()
    with _Residency(k1, adapter, tmp_path) as fms:
        state = fms.state(1)
        with Runtime(_config(tmp_path / "c")) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, learn)
            answer = runtime.run_query(state, query).readout
            live = runtime.live_identity(state)
            durable = runtime.continuity.snapshot()
            runtime.release(state)
            fms.state(2)
            newcomer = fms.demote(state, 3)
            with pytest.raises(CompositionError) as refused:
                runtime.run_query(state, query)
            assert refused.value.code == "RUNTIME_NOT_RESIDENT"
            counters = fms.residency()

            # The explicit WARM-lane materialization: exactly one promotion and one cold read. WARM must make room,
            # and the newcomer is the only state it may displace.
            state.materialize()
            assert fms.tier(state) == WARM and fms.tier(newcomer) == COLD
            restored = fms.residency()
            assert restored["promotions"] == counters["promotions"] + 1
            assert restored["cold_reads"] == counters["cold_reads"] + 1

            # The same live lineage resumes, held again: the exact bytes the release gave back, at their generation.
            assert runtime.live_identity(state) == live
            assert fms.holds(state) == 1
            again = runtime.run_query(state, query).readout
            assert (again.values, again.live_generation) == (answer.values, answer.live_generation) == \
                (answer.values, live[1])
            result = runtime.run_learn(state, learn)
            assert (result.generation_before, result.generation_after) == (live[1], live[1] + 1)
            assert runtime.run_query(state, query).readout.live_generation == live[1] + 1
            # HOT work after the restoration reads nothing back and publishes nothing (volatile).
            after = fms.residency()
            assert (after["promotions"], after["cold_reads"]) == (restored["promotions"], restored["cold_reads"])
            assert runtime.continuity.snapshot() == durable and runtime.fault is None
        assert fms.holds(state) == 0
