"""QUERY and LEARN are different operations (docs/COGNITION_R0.md), over real native K1.

QUERY reads the authoritative K1 state and runs the native forward map ``f_W(x)``: it does not learn, consolidate,
advance the epoch, write W, H or a, open a transaction or publish continuity. These tests prove that exactly, by
comparing the complete retained state ``(W, epoch, H, a)`` (its portable envelope, its retained-state digest and its
components), the native generation and the continuity slot *bytes* before and after every successful and every
refused query, unmanaged and managed, standalone and FMS-resident. LEARN needs an explicit ``LearnAuthority`` and
commits atomically or not at all. The fixture map is TRAINING=NONE SEMANTICS=NONE: interface mechanics only.
"""
from __future__ import annotations

from array import array

import pytest

from elpis.runtime import Runtime
from elpis.runtime.cognition import (CognitiveOperation, LearnAuthority, LearnRequest, LearnResult, QueryRequest,
                                     QueryResult, QueryStimulus, run_learn, run_query, run_turn)
from elpis.runtime.composition import CompositionError

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap
from .test_codec_ecs_turn import DIM, _config, _Resident, adapter, k1, world  # noqa: F401 (module fixtures)


def _query(codec_map=None, text="what is f_W here?", max_output_tokens=256):
    return QueryRequest(text, ByteTokens(), FixtureMap() if codec_map is None else codec_map, max_output_tokens)


def _learn(text="an experience", authority=LEARN, codec_map=None):
    return LearnRequest(text, ByteTokens(), FixtureMap() if codec_map is None else codec_map, authority)


class _Subject:
    """One native K1 state of either kind and every observable piece of its retained state."""

    def __init__(self, state, resident):
        self.state, self.resident = state, resident

    def retained(self):
        s = self.state
        if self.resident:
            info = s._r.inspect(s.id)
            components = (s._r.k1_stats(s.id)["commits"], info["transaction_open"], info["lease_count"])
            generation = info["generation"]
        else:
            components = (s.w(), s.h_packed(), s.a(), s.provenance)
            generation = s.generation
        return s.snapshot(), s.state_digest(), s.epoch, generation, components


@pytest.fixture(params=["k1", "fms-k1"])
def subject(request, k1, adapter, tmp_path):   # noqa: F811 (module fixtures)
    if request.param == "k1":
        with world(k1, max_rows=8) as state:
            yield _Subject(state, False)
    else:
        with _Resident(k1, adapter, tmp_path, max_rows=8) as state:
            yield _Subject(state, True)


def _slots(runtime):
    directory = runtime.continuity.directory
    return tuple((directory / name).read_bytes() for name in ("continuity.a", "continuity.b"))


def _warm(state):
    """Give the state non-zero H and a, so a query that touched them would show."""
    run_learn(state, _learn("warm the consolidation"))


# --- unmanaged ------------------------------------------------------------------------------------------------

def test_a_query_answers_f_w_of_the_authoritative_state_and_changes_nothing(subject):
    _warm(subject.state)
    before = subject.retained()
    fixture = FixtureMap()
    result = run_query(subject.state, _query(fixture))
    assert type(result) is QueryResult and result.operation is CognitiveOperation.QUERY
    assert subject.retained() == before                       # W, epoch, H, a, generation: byte-for-byte
    assert result.state_digest == before[1] and result.text == "ok"
    # The answer is K1's own forward map of the query rows on the authoritative W, nothing else.
    rows = fixture.query_rows_of(ByteTokens().encode("what is f_W here?"))
    assert result.readout.values == subject.state.query(memoryview(rows))
    assert len(result.readout.values) == 3 and fixture.calls[0][0] == "encode_query"
    # Repeatable: the same state answers the same way, and still nothing moved.
    assert run_query(subject.state, _query()).readout == result.readout
    assert subject.retained() == before


def test_the_answer_follows_w_and_never_h_or_a(k1):
    with world(k1) as a, world(k1) as b:
        _warm(a)                                   # a learns: W, H, a move
        rows = memoryview(FixtureMap().query_rows_of(ByteTokens().encode("q")))
        answer_a = run_query(a, _query(text="q")).readout.values
        assert answer_a == a.query(rows) and answer_a != run_query(b, _query(text="q")).readout.values
        with type(a).restore(k1, a.snapshot()) as same:
            same.reset()                           # same W, H = a = 0: a query never reads H or a
            assert run_query(same, _query(text="q")).readout.values == answer_a


@pytest.mark.parametrize("case", ["no_codec", "unclassified", "no_query_side", "learn_request", "wrong_dim",
                                  "nonfinite_row", "decode_out_of_vocab", "decode_over_limit", "not_a_query"])
def test_a_refused_query_changes_nothing(subject, case):
    _warm(subject.state)
    before = subject.retained()
    fixture = FixtureMap(reply=(300,) if case == "decode_out_of_vocab" else b"x" * 9)
    if case == "unclassified":
        fixture.classification = ""
    if case == "no_query_side":
        fixture.encode_query = None
    if case == "wrong_dim":
        fixture.encode_query = lambda tokens: QueryStimulus(array("d", [0.1] * 5), dim=5)
    if case == "nonfinite_row":
        fixture.encode_query = lambda tokens: QueryStimulus(array("d", [float("inf")] * DIM), dim=DIM)
    if case == "not_a_query":
        fixture.encode_query = lambda tokens: fixture.encode(tokens)   # a LEARN stimulus is not a query
    request = _query(fixture, max_output_tokens=4 if case == "decode_over_limit" else 256)
    if case == "no_codec":
        request = QueryRequest("q", ByteTokens())
    if case == "learn_request":
        request = _learn()
    expected = {"no_codec": "ECS_CODEC_UNQUALIFIED", "unclassified": "CODEC_MAP", "no_query_side": "CODEC_MAP",
                "learn_request": "OPERATION", "wrong_dim": "STIMULUS", "nonfinite_row": "ECS_REFUSED",
                "decode_out_of_vocab": "DECODE", "decode_over_limit": "DECODE", "not_a_query": "STIMULUS"}[case]
    with pytest.raises(CompositionError) as info:
        run_query(subject.state, request)
    assert info.value.code == expected
    assert subject.retained() == before


# --- LEARN needs explicit authority ------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["none", "query_request", "not_an_authority"])
def test_learn_without_explicit_authority_is_refused_before_any_state_is_touched(subject, case):
    before = subject.retained()
    fixture = FixtureMap()
    request = {"none": _learn(authority=None, codec_map=fixture), "query_request": _query(fixture),
               "not_an_authority": _learn(authority=0.002, codec_map=fixture)}[case]
    with pytest.raises(CompositionError) as info:
        run_learn(subject.state, request)
    assert info.value.code == {"none": "LEARN_UNAUTHORIZED", "query_request": "OPERATION",
                               "not_an_authority": "LEARN_AUTHORITY"}[case]
    assert subject.retained() == before and fixture.calls == []   # not even encoded
    with pytest.raises(CompositionError) as info:              # the legacy learned turn is a LEARN too
        run_turn(subject.state, "x", tokenizer=ByteTokens(), codec_map=fixture)
    assert info.value.code == "LEARN_UNAUTHORIZED" and subject.retained() == before


def test_a_learning_authority_is_explicit_and_bounded():
    for rate in (float("nan"), float("inf"), 0.0, -0.1, 1):
        with pytest.raises(CompositionError) as info:
            LearnAuthority(rate, "grant")
        assert info.value.code == "LEARNING_RATE"
    for grant in ("", "   ", None):
        with pytest.raises(CompositionError) as info:
            LearnAuthority(0.002, grant)
        assert info.value.code == "LEARN_AUTHORITY"
    with pytest.raises(TypeError):
        QueryRequest("q", ByteTokens(), FixtureMap(), 256, LEARN)   # a query has no slot for learning authority
    assert QueryRequest.operation is CognitiveOperation.QUERY and LearnRequest.operation is CognitiveOperation.LEARN


def test_learn_commits_the_whole_schedule_atomically_and_equals_the_legacy_learned_turn(k1):
    with world(k1) as a, world(k1) as b:
        learned = run_learn(a, _learn("the same experience"))
        legacy = run_turn(b, "the same experience", tokenizer=ByteTokens(), codec_map=FixtureMap(), authority=LEARN)
        assert type(learned) is LearnResult and learned.operation is CognitiveOperation.LEARN
        assert a.snapshot() == b.snapshot() and learned.state_after_digest == a.state_digest()
        assert (learned.epoch_before, learned.epoch_after, learned.experiences) == (0, 6, 2)
        assert learned.grant == LEARN.grant and legacy.state_after_digest == learned.state_after_digest
        before = a.snapshot()
        with pytest.raises(CompositionError) as info:   # a non-finite middle experience: nothing installed
            run_learn(a, _learn("poison", codec_map=FixtureMap(experiences=3, poison_experience=1)))
        assert info.value.code == "ECS_REFUSED" and a.snapshot() == before


# --- managed (RuntimeCore + continuity) ---------------------------------------------------------------------------

def test_a_managed_query_changes_no_state_and_no_continuity_byte(subject, tmp_path):
    with Runtime(_config(tmp_path / "c", testing=True)) as runtime:
        runtime.anchor_cognition(subject.state)
        runtime.run_learn(subject.state, _learn("warm"))
        before, slots, durable = subject.retained(), _slots(runtime), runtime.continuity.snapshot()
        runtime._core.counters(reset=True)
        runtime.continuity.testing_counters(reset=True)
        results = [runtime.run_query(subject.state, _query(text=t)) for t in ("first", "second", "first")]
        assert subject.retained() == before                    # W, epoch, H, a, generation
        assert _slots(runtime) == slots                        # continuity slot bytes
        assert runtime.continuity.snapshot() == durable        # continuity current authority
        assert all(r.state_digest == before[1] == durable.k1_state_digest for r in results)
        assert results[0].readout == results[2].readout
        native = runtime._core.counters()
        assert native["k1_queries"] == 3 and native["publications"] == 0
        assert native["k1_txn_begins"] == native["k1_commits"] == native["k1_aborts"] == native["k1_reserves"] == 0
        io = runtime.continuity.testing_counters()
        assert not any(io.values()), io                        # the store did no I/O at all
        assert runtime.fault is None


@pytest.mark.parametrize("case", ["no_codec", "learn_request", "nonfinite_row", "decode_out_of_vocab"])
def test_a_refused_managed_query_changes_no_state_and_no_continuity_byte(subject, tmp_path, case):
    with Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(subject.state)
        before, slots, durable = subject.retained(), _slots(runtime), runtime.continuity.snapshot()
        fixture = FixtureMap(reply=(300,) if case == "decode_out_of_vocab" else b"ok")
        if case == "nonfinite_row":
            fixture.encode_query = lambda tokens: QueryStimulus(array("d", [float("nan")] * DIM), dim=DIM)
        request = {"no_codec": QueryRequest("q", ByteTokens()), "learn_request": _learn()}.get(case, _query(fixture))
        with pytest.raises(CompositionError) as info:
            runtime.run_query(subject.state, request)
        assert info.value.code == {"no_codec": "ECS_CODEC_UNQUALIFIED", "learn_request": "OPERATION",
                                   "nonfinite_row": "ECS_REFUSED", "decode_out_of_vocab": "DECODE"}[case]
        assert (subject.retained(), _slots(runtime), runtime.continuity.snapshot()) == (before, slots, durable)
        assert runtime.fault is None


def test_a_managed_query_of_an_unanchored_or_foreign_state_is_refused_and_writes_nothing(k1, tmp_path):
    with world(k1) as state, world(k1, w=None) as other, Runtime(_config(tmp_path / "c")) as runtime:
        before, slots = state.snapshot(), _slots(runtime)
        with pytest.raises(CompositionError) as info:
            runtime.run_query(state, _query())
        assert info.value.code == "CONTINUITY_UNANCHORED" and runtime.fault is None
        assert state.snapshot() == before and _slots(runtime) == slots
        runtime.anchor_cognition(state)
        with pytest.raises(CompositionError) as info:
            runtime.run_query(other, _query())
        assert info.value.code == "COGNITION_SUBSTRATE_SWITCH"


def test_a_managed_query_of_a_state_moved_out_of_band_fail_stops_and_withholds_the_answer(k1, tmp_path):
    with world(k1) as state, Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_query(state, _query())
        run_learn(state, _learn("out of band"))   # an unmanaged LEARN on the bound state
        slots, durable = _slots(runtime), runtime.continuity.snapshot()
        with pytest.raises(CompositionError) as info:
            runtime.run_query(state, _query())
        assert info.value.code == "CONTINUITY_STATE_MISMATCH" and runtime.fault == "CONTINUITY_STATE_MISMATCH"
        assert _slots(runtime) == slots and runtime.continuity.snapshot() == durable


def test_managed_learn_needs_authority_and_publishes_exactly_its_commit(subject, tmp_path):
    with Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(subject.state)
        before, slots = subject.retained(), _slots(runtime)
        for request in (_learn(authority=None), _query()):
            with pytest.raises(CompositionError) as info:
                runtime.run_learn(subject.state, request)
            assert info.value.code in ("LEARN_UNAUTHORIZED", "OPERATION")
            assert (subject.retained(), _slots(runtime)) == (before, slots) and runtime.fault is None
        generation = runtime.continuity.snapshot().generation
        result = runtime.run_learn(subject.state, _learn())
        assert result.state_before_digest == before[1]
        assert result.state_after_digest == subject.state.state_digest() == \
            runtime.continuity.snapshot().k1_state_digest
        assert runtime.continuity.snapshot().generation == generation + 1
        # QUERY answers from the learned state, and does not move it.
        after = subject.retained()
        assert runtime.run_query(subject.state, _query()).state_digest == result.state_after_digest
        assert subject.retained() == after
