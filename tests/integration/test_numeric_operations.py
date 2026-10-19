"""Managed QUERY and LEARN of admitted numeric stimuli (``Runtime.run_numeric_query`` / ``run_numeric_learn``).

Numbers in, numbers out: no codec and no text are involved, so these are not language operations and the text turn
still fails closed (``ECS_CODEC_UNQUALIFIED``). They are the same managed operations as ``run_query`` /
``run_learn`` at their edge: the managed numeric LEARN produces exactly the K1 state the qualified law produces
unmanaged from the same state and stimulus (the native kernel computes; Python computes nothing), the numeric QUERY
answers exactly the native ``f_W(x)`` of the live state, both are refused without their authority or outside fuel,
and the LEARN is volatile.
"""
from __future__ import annotations

from array import array
from dataclasses import replace

import pytest

from elpis.ECS.k1 import K1State
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnAuthority, QueryStimulus, Stimulus
from elpis.runtime.composition import CompositionError
from elpis.runtime.fuel import CEILING

from .test_codec_ecs_turn import DIM, WIDTH, _config, initial_w, k1  # noqa: F401 (module fixture)
from .test_composite_runtime import AUTHORITY, _query, _store, fresh, library, obs  # noqa: F401
from .test_k1_recovery import _cfg

GRANT = LearnAuthority(0.002, "test: numeric fixture experience (TEST_ONLY)")


def _x(rows, offset=0.0):
    return array("d", [((i * 7 + j * 3) % 11 - 5) / 16.0 + offset for i in range(rows) for j in range(DIM)])


def _stimulus():
    return Stimulus(_x(4), array("d", [0.1, -0.2, 0.05, 0.3]), array("Q", [2, 3, 2, 5]), dim=DIM)


def _refused(call, code):
    with pytest.raises(CompositionError) as info:
        call()
    assert info.value.code == code, (info.value.code, str(info.value))


def test_the_managed_numeric_learn_is_the_qualified_k1_law_and_volatile(k1, library, tmp_path):   # noqa: F811
    w = memoryview(array("d", initial_w(36).reshape(-1)))
    stimulus = _stimulus()
    with K1State.create(k1, DIM, WIDTH, w, max_rows=8) as reference:
        with reference.transaction() as tx:
            tx.run_schedule(stimulus.x, stimulus.y, stimulus.schedule, GRANT.learning_rate)
            tx.commit()
        expected_k1 = reference.state_digest()
        expected_answer = reference.query(_x(3, 0.25))
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        anchored = runtime.anchor_cognition(x)
        result = runtime.run_numeric_learn(x, stimulus, GRANT)
        assert (result.durability, result.experiences, result.rows, result.steps) == ("VOLATILE", 2, 4, 8)
        assert (result.epoch_before, result.epoch_after, result.generation_after) == (0, 8, result.generation_before + 1)
        assert library.envelope_identities(x.snapshot())[1] == expected_k1   # the composite's K1 component
        answer = runtime.run_numeric_query(x, QueryStimulus(_x(3, 0.25), dim=DIM))
        assert answer.values == tuple(expected_answer) and answer.live_generation == result.generation_after
        assert runtime.continuity.snapshot() == anchored                       # volatile: nothing published
        # Observations ride the same numeric turn (all or nothing).
        both = runtime.run_numeric_learn(x, stimulus, GRANT, observations=(obs(1, 2, 10),),
                                         observation_authority=AUTHORITY)
        assert len(both.observed) == 1 and x.slot(both.observed[0]).assert_count == 1


def test_numeric_operations_refuse_without_authority_outside_fuel_and_for_the_wrong_shape(library, tmp_path):   # noqa: F811
    store = _store(library, tmp_path / "ck")
    with fresh(library) as x, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(x)
        state = x.state_digest()
        _refused(lambda: runtime.run_numeric_learn(x, _stimulus(), None), "LEARN_UNAUTHORIZED")
        _refused(lambda: runtime.run_numeric_learn(x, _stimulus(), "grant"), "LEARN_AUTHORITY")
        tight = LearnAuthority(0.002, "test: two steps at most", replace(CEILING, max_steps=2))
        _refused(lambda: runtime.run_numeric_learn(x, _stimulus(), tight), "COGNITION_FUEL_EXCEEDED")
        other = Stimulus(array("d", [0.1] * 5), array("d", [0.0]), array("Q", [1, 1]), dim=5)
        _refused(lambda: runtime.run_numeric_learn(x, other, GRANT), "STIMULUS")
        _refused(lambda: runtime.run_numeric_query(x, QueryStimulus(array("d", [0.1] * 5), dim=5)), "STIMULUS")
        _refused(lambda: runtime.run_numeric_query(x, _x(1)), "STIMULUS")
        _refused(lambda: runtime.run_numeric_query(x, QueryStimulus(_x(3), dim=DIM),
                                                   budget=replace(CEILING, max_query_rows=2)),
                 "COGNITION_FUEL_EXCEEDED")
        _refused(lambda: runtime.run_numeric_learn(x, _stimulus(), GRANT, observations=(obs(1, 2, 10),)),
                 "OBSERVATION_UNAUTHORIZED")
        assert x.state_digest() == state and runtime.fault is None
        # The text turn still fails closed without a qualified codec.
        from elpis.runtime.cognition import QueryRequest
        from ._turn_fixtures import ByteTokens
        _refused(lambda: runtime.run_query(x, QueryRequest("text", ByteTokens())), "ECS_CODEC_UNQUALIFIED")


def test_a_k1_lineage_learns_and_queries_numerically_too(k1, tmp_path):   # noqa: F811
    from .test_codec_ecs_turn import world
    with world(k1) as state, Runtime(_config(tmp_path / "authority")) as runtime:
        runtime.anchor_cognition(state)
        before = runtime.run_numeric_query(state, QueryStimulus(_x(2), dim=DIM))
        result = runtime.run_numeric_learn(state, _stimulus(), GRANT)
        after = runtime.run_numeric_query(state, QueryStimulus(_x(2), dim=DIM))
        assert after.live_generation == result.generation_after == before.live_generation + 1
        assert after.values != before.values
