"""Unmanaged mutation of a RuntimeCore-bound K1 state is refused or fail-stops; it is never silently incorporated.

Before this law, every out-of-band mutation of a bound state was accepted by K1, and the next managed LEARN committed
natively on top of it before continuity publication noticed. Now RuntimeCore claims every state it binds with its own
K1 lease (the K1 header's "Managed ownership"): every unmanaged mutating entry point is refused (``LEASED``) before it
touches anything, and every managed transaction begin re-establishes the state's exact retained-state identity
inside the same guarded K1 call. A bound state that is not provably the lineage (its lease taken by another owner, or
a move no guard could see) fail-stops before anything is built on it or published. Real native K1, standalone and
FMS-resident; the fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE.
"""
from __future__ import annotations

import pytest

from elpis.ECS.k1 import K1Error
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest, run_learn, run_turn
from elpis.runtime.composition import CompositionError
from elpis.runtime.core import describe
from elpis.runtime.fuel import CEILING

from ._turn_fixtures import LEARN, RATE, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import _config, _Resident, adapter, k1, world  # noqa: F401 (module fixtures)

MISMATCH = "CONTINUITY_STATE_MISMATCH"
X = FixtureMap(experiences=1).encode(tuple(b"out of band"))


def _learn(text="managed"):
    return LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN)


@pytest.fixture(params=["k1", "fms-k1"])
def bound(request, k1, adapter, tmp_path):   # noqa: F811 (module fixtures)
    """A K1 state of either kind, bound to an open runtime by an anchor and one managed LEARN."""
    def opened(state):
        runtime = Runtime(_config(tmp_path / "c")).open()
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("warm"))
        return runtime

    if request.param == "k1":
        with world(k1, max_rows=8) as state:
            runtime = opened(state)
            yield state, runtime
            runtime.close()
    else:
        with _Resident(k1, adapter, tmp_path, max_rows=8) as state:
            runtime = opened(state)
            yield state, runtime
            runtime.close()


def _evidence(state, runtime):
    slots = tuple((runtime.continuity.directory / n).read_bytes() for n in ("continuity.a", "continuity.b"))
    return state.snapshot(), state.state_digest(), state.epoch, slots, runtime.continuity.snapshot()


def _direct_mutations(state):
    """Every unmanaged mutating entry point a local caller can reach for this state."""
    if hasattr(state, "learn"):   # standalone K1State
        def txn():
            with state.transaction() as t:
                t.learn(X.x, X.y, RATE, 1)
                t.commit()
        return {"learn": lambda: state.learn(X.x, X.y, RATE, 1), "consolidate": lambda: state.consolidate(X.x),
                "reset": state.reset, "reserve": lambda: state.reserve(64), "transaction": txn}
    r, sid = state._r, state.id

    def fms_txn():
        with r.transaction(sid) as t:
            t.learn(X.x, X.y, RATE, 1)
            t.commit()
    return {"learn": lambda: r.learn(sid, X.x, X.y, RATE, 1), "consolidate": lambda: r.consolidate(sid, X.x),
            "reset": lambda: r.reset(sid), "reserve": lambda: r.reserve(sid, 64), "transaction": fms_txn}


def test_every_unmanaged_mutation_of_a_bound_state_is_refused_and_the_lineage_continues(bound):
    state, runtime = bound
    assert state.managed_lease != 0
    before = _evidence(state, runtime)
    for name, mutate in _direct_mutations(state).items():
        with pytest.raises(K1Error) as info:
            mutate()
        assert info.value.code == "LEASED", name
        assert _evidence(state, runtime) == before, name
    for name, call in (("run_learn", lambda: run_learn(state, _learn("oob"))),
                       ("run_turn", lambda: run_turn(state, "oob", tokenizer=ByteTokens(), codec=admitted(FixtureMap()),
                                                     authority=LEARN))):
        with pytest.raises(CompositionError) as info:
            call()
        assert info.value.code == "ECS_LEASED", name
        assert _evidence(state, runtime) == before, name
    # Reads stay available, and the managed lineage continues as if nothing happened.
    state.query(X.x)
    result = runtime.run_learn(state, _learn("next"))
    assert result.state_before_digest == before[1]
    assert runtime.continuity.snapshot().k1_state_digest == state.state_digest() and runtime.fault is None


def test_an_unmanaged_mutation_during_an_open_managed_transaction_is_refused(bound):
    state, runtime = bound
    descriptor, _ = describe(state)
    stimulus = FixtureMap().encode(ByteTokens().encode("t"))
    runtime._core.turn_begin(descriptor, stimulus, RATE, CEILING)
    for name, mutate in _direct_mutations(state).items():
        with pytest.raises(K1Error) as info:
            mutate()
        assert info.value.code in ("LEASED", "BUSY"), name   # BUSY: one transaction per state, and it is open
    committed, _ = runtime._core.turn_commit(descriptor)      # the managed commit is not STALE: nothing moved it
    assert committed.state_after_digest == state.state_digest() == runtime.continuity.snapshot().k1_state_digest
    assert runtime.fault is None


def test_a_second_runtime_taking_the_state_makes_the_first_fail_stop_before_any_mutation(k1, tmp_path):
    with world(k1) as state:
        first = Runtime(_config(tmp_path / "a")).open()
        first.anchor_cognition(state)
        first.run_learn(state, _learn("first"))
        second = Runtime(_config(tmp_path / "b")).open()
        second.anchor_cognition(state)        # proves its own lineage on the same bytes, then takes the lease
        before = state.snapshot()
        with pytest.raises(CompositionError) as info:
            first.run_learn(state, _learn("lost"))
        assert info.value.code == MISMATCH and first.fault == MISMATCH
        assert state.snapshot() == before      # refused natively at the leased begin: nothing built, nothing published
        second.run_learn(state, _learn("second"))
        assert second.continuity.snapshot().k1_state_digest == state.state_digest()
        first.close()
        second.close()


def test_close_gives_the_state_back_and_restart_verifies_it(k1, tmp_path):
    with world(k1) as state:
        with Runtime(_config(tmp_path / "c")) as runtime:
            runtime.anchor_cognition(state)
            runtime.run_learn(state, _learn())
            assert state.managed_lease != 0
        assert state.managed_lease == 0
        state.learn(X.x, X.y, RATE, 1)        # unmanaged again: outside the lineage
        with Runtime(_config(tmp_path / "c")) as restarted:
            with pytest.raises(CompositionError) as info:
                restarted.run_learn(state, _learn())
            assert info.value.code == MISMATCH
            assert state.managed_lease == 0    # a state that is not the lineage is never claimed


def test_an_explicit_release_and_rebind_verifies_identity_again(bound):
    state, runtime = bound
    runtime.release(state)
    assert state.managed_lease == 0
    runtime.run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap())))   # rebinds: same bytes
    assert state.managed_lease != 0 and runtime.fault is None


def test_close_after_the_owner_closed_the_state_is_clean(k1, tmp_path):
    state = world(k1)
    runtime = Runtime(_config(tmp_path / "c")).open()
    runtime.anchor_cognition(state)
    state.close()                              # the owner freed it while bound: nothing to give back
    runtime.close()
    assert runtime.fault is None or runtime.fault == "RUNTIME_CLOSED"
