"""Total cognitive fuel is admitted before any reserve, transaction, candidate mutation or publication.

The per-field stimulus bounds (64 experiences x 256 rows x 2^20 K1 steps) admit a schedule of 2^26 learning steps;
the totals of a ``CognitiveBudget`` do not. These tests prove, over real native K1 and RuntimeCore, that an
oversized operation is refused with ``COGNITION_FUEL_EXCEEDED`` while the K1 state (envelope, digest, epoch,
generation, transaction count, heap allocations, reserved rows), RuntimeCore's crossing counters and the continuity
slots are unchanged; that RuntimeCore refuses it natively even when the Python admission is bypassed; and that the
work-unit formula is the same integer function in Python and in RuntimeCore, at its exact boundaries. Fixture maps are
TEST_ONLY TRAINING=NONE SEMANTICS=NONE.
"""
from __future__ import annotations

from array import array

import pytest

from elpis.runtime import Runtime
from elpis.runtime.cognition import (LearnAuthority, LearnRequest, QueryRequest, Stimulus, run_learn, run_query,
                                     run_turn)
from elpis.runtime.composition import CompositionError
from elpis.runtime.core import RuntimeLibrary, _Budget, describe
from elpis.runtime.fuel import CEILING, CognitiveBudget, query_units, schedule_units

from ._turn_fixtures import RATE, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _config, k1, world  # noqa: F401 (module fixture)

FUEL = "COGNITION_FUEL_EXCEEDED"
DEFAULT_UNITS = schedule_units(DIM, WIDTH, ((4, 3), (4, 3)))   # FixtureMap(): 2 experiences x 4 rows x 3 steps


def _authority(**budget):
    return LearnAuthority(RATE, "TEST_ONLY fuel test grant", CognitiveBudget(**budget))


def _k1_evidence(state):
    stats = state.stats()
    return (state.snapshot(), state.state_digest(), state.epoch, state.generation, state.max_rows,
            stats["txn_begins"], stats["heap_allocations"], stats["commits"], stats["steps_executed"])


@pytest.fixture(scope="module")
def library():
    return RuntimeLibrary(_config(__import__("pathlib").Path("/unused")).runtime_library)


def test_the_python_formula_and_ceiling_are_runtimecores(library):
    assert DEFAULT_UNITS == 357_806
    cases = [(6, 36, ((4, 3), (4, 3))), (6, 36, ((1, 1),)), (6, 72, ((256, 1 << 20),) * 64), (1, 1, ((1, 1),)),
             (12, 5, ((7, 9), (3, 1), (256, 2)))]
    for dim, width, pairs in cases:
        native = library.work_units(dim, width, array("Q", [v for pair in pairs for v in pair]))
        assert native == schedule_units(dim, width, pairs), (dim, width)
    for dim, width, rows in ((6, 36, 1), (6, 36, 4096), (64, 3, 17)):
        assert library.query_work_units(dim, width, rows) == query_units(dim, width, rows)
    # A total beyond uint64 is refused by both, never wrapped.
    assert schedule_units(6, 36, ((1 << 63, 1),)) is None
    with pytest.raises(CompositionError) as info:
        library.work_units(6, 36, array("Q", [1 << 63, 1]))
    assert info.value.code == "RUNTIME_INVALID"
    # RuntimeLibrary refuses a library whose compiled ceiling is not the canonical one (checked at load).
    assert CEILING.native_fields() == (64, 16_384, 256, 1 << 20, 1 << 30, 4096)


def test_a_budget_only_narrows_the_ceiling():
    for name in ("max_experiences", "max_rows", "max_experience_rows", "max_steps", "max_work_units",
                 "max_query_rows", "max_output_tokens"):
        with pytest.raises(CompositionError) as info:
            CognitiveBudget(**{name: getattr(CEILING, name) + 1})
        assert info.value.code == FUEL
        with pytest.raises(CompositionError):
            CognitiveBudget(**{name: 0})
    with pytest.raises(CompositionError):
        LearnAuthority(RATE, "grant", budget={"max_steps": 1})


def test_the_per_field_maximum_schedule_is_refused_before_anything_native(k1):
    huge = FixtureMap(experiences=64, rows=256, steps=1 << 20)   # every per-field bound at its maximum
    stimulus = huge.encode(ByteTokens().encode("x"))
    assert (stimulus.experiences, stimulus.steps) == (64, 64 << 20)
    assert schedule_units(DIM, WIDTH, ((256, 1 << 20),) * 64) > CEILING.max_work_units
    with world(k1, max_rows=8) as state:
        before = _k1_evidence(state)
        for call in (lambda: run_learn(state, LearnRequest("x", ByteTokens(), admitted(huge), _authority())),
                     lambda: run_turn(state, "x", tokenizer=ByteTokens(), codec=admitted(huge),
                                      authority=_authority())):
            with pytest.raises(CompositionError) as info:
                call()
            assert info.value.code == FUEL
            assert _k1_evidence(state) == before   # not reserved (max_rows 8 < 256), no transaction, no step


@pytest.mark.parametrize("bound", ["max_work_units", "max_steps", "max_rows", "max_experience_rows",
                                   "max_experiences"])
def test_each_total_binds_at_its_exact_integer_boundary(k1, bound):
    need = {"max_work_units": DEFAULT_UNITS, "max_steps": 6, "max_rows": 8, "max_experience_rows": 4,
            "max_experiences": 2}[bound]
    with world(k1) as state:
        before = _k1_evidence(state)
        with pytest.raises(CompositionError) as info:
            run_learn(state, LearnRequest("x", ByteTokens(), admitted(FixtureMap()), _authority(**{bound: need - 1})))
        assert info.value.code == FUEL
        assert _k1_evidence(state) == before
        result = run_learn(state, LearnRequest("x", ByteTokens(), admitted(FixtureMap()), _authority(**{bound: need})))
        assert result.epoch_after == 6


def test_query_rows_and_output_tokens_are_bounded_before_the_native_query(k1):
    with world(k1) as state:
        before = _k1_evidence(state)
        stats = state.stats()["forward_calls"]
        for budget, tokens in ((CognitiveBudget(max_query_rows=2), 256), (CognitiveBudget(max_output_tokens=255), 256),
                               (CognitiveBudget(max_work_units=3 * DIM * WIDTH - 1), 256)):
            with pytest.raises(CompositionError) as info:
                run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap()), tokens, budget))
            assert info.value.code == FUEL
        assert _k1_evidence(state) == before and state.stats()["forward_calls"] == stats
        exact = CognitiveBudget(max_query_rows=3, max_work_units=3 * DIM * WIDTH)
        assert run_query(state, QueryRequest("q", ByteTokens(), admitted(FixtureMap()), 256, exact)).text == "ok"


def _slots(runtime):
    return tuple((runtime.continuity.directory / n).read_bytes() for n in ("continuity.a", "continuity.b"))


def test_managed_oversized_learn_is_refused_before_reserve_begin_schedule_commit_and_publication(k1, tmp_path):
    with world(k1, max_rows=2) as state, Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(state)
        before, slots, durable = _k1_evidence(state), _slots(runtime), runtime.continuity.snapshot()
        runtime._core.counters(reset=True)
        tight = _authority(max_work_units=DEFAULT_UNITS - 1)
        for call in (lambda: runtime.run_learn(state, LearnRequest("x", ByteTokens(), admitted(FixtureMap()), tight)),
                     lambda: runtime.run_turn(state, "x", tokenizer=ByteTokens(), codec=admitted(FixtureMap()),
                                              authority=tight)):
            with pytest.raises(CompositionError) as info:
                call()
            assert info.value.code == FUEL
        # The native authority refuses too when the Python admission is bypassed: RuntimeCore admits natively.
        descriptor, _ = describe(state)
        stimulus = FixtureMap().encode(ByteTokens().encode("x"))
        for budget in (CognitiveBudget(max_work_units=DEFAULT_UNITS - 1), CognitiveBudget(max_experience_rows=3)):
            with pytest.raises(CompositionError) as info:
                runtime._core.turn_begin(descriptor, stimulus, RATE, budget)
            assert info.value.code == FUEL
        over = _Budget(*CEILING.native_fields())
        over.max_steps += 1   # a budget beyond the ceiling, which the Python type would refuse to construct
        rc = runtime._core._f.elpis_runtime_turn_begin(
            runtime._core._handle, __import__("ctypes").byref(descriptor), *_raw(stimulus), RATE,
            __import__("ctypes").byref(over), *_s3(runtime))
        assert runtime._core.library.code(rc) == FUEL
        counters = runtime._core.counters()
        assert counters["k1_reserves"] == counters["k1_txn_begins"] == counters["k1_run_schedules"] == 0
        assert counters["k1_commits"] == counters["publications"] == counters["k1_aborts"] == 0
        assert _k1_evidence(state) == before                       # max_rows still 2: nothing reserved
        assert (_slots(runtime), runtime.continuity.snapshot(), runtime.fault) == (slots, durable, None)
        # Within budget the same LEARN runs and publishes once (reserving on the cold path as before).
        runtime.run_learn(state, LearnRequest("x", ByteTokens(), admitted(FixtureMap()),
                                              _authority(max_work_units=DEFAULT_UNITS)))
        assert runtime.continuity.snapshot().generation == durable.generation + 1 and state.max_rows >= 4


def test_managed_query_beyond_its_budget_is_refused_natively(k1, tmp_path):
    with world(k1) as state, Runtime(_config(tmp_path / "c")) as runtime:
        runtime.anchor_cognition(state)
        descriptor, _ = describe(state)
        query = FixtureMap().encode_query(ByteTokens().encode("q"))
        runtime._core.counters(reset=True)
        with pytest.raises(CompositionError) as info:
            runtime._core.query(descriptor, query, CognitiveBudget(max_query_rows=2))
        assert info.value.code == FUEL and runtime._core.counters()["k1_queries"] == 0 and runtime.fault is None
        values, digest = runtime._core.query(descriptor, query, CognitiveBudget(max_query_rows=3))
        assert len(values) == 3 and digest == state.state_digest()


def _raw(stimulus):
    import ctypes as C
    from elpis.runtime.core import _Experience
    x = (C.c_double * len(stimulus.x)).from_buffer(stimulus.x)
    y = (C.c_double * len(stimulus.y)).from_buffer(stimulus.y)
    schedule = (_Experience * stimulus.experiences).from_buffer(stimulus.schedule)
    return x, len(stimulus.x), y, len(stimulus.y), schedule, stimulus.experiences


def _s3(runtime):
    import ctypes as C
    from elpis.runtime.core import _Begin
    features = runtime._core.library.features(DIM)
    return (C.c_double * features)(), features, C.byref(_Begin())


def test_stimulus_per_field_bounds_are_unchanged():
    with pytest.raises(CompositionError):
        Stimulus(array("d", [0.0] * DIM), array("d", [0.0]), array("Q", [1, (1 << 20) + 1]), dim=DIM)
