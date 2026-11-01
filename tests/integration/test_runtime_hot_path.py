"""The managed turn's HOT lane, measured dynamically (docs/RUNTIME_CORE.md).

``Runtime.run_turn`` is codec -> RuntimeCore (one native K1 transaction begun at the live coordinate: begin, schedule
and readout) -> decode -> RuntimeCore (native commit: volatile). Measured on a warm turn (RuntimeCore ABI v8: there is
no persistent path at all, so nothing could write even if it were on this path):

* Python makes no K1 call at all and three crossings into RuntimeCore: the fail-stop probe, turn begin and turn
  commit;
* RuntimeCore's K1 crossings are those of the transaction itself (begin at the live generation, one schedule,
  commit): no state digest, no identity read, no reservation, no abort, and no session transition;
* Python performs no file I/O at all (the native side is measured by ``runtime.test_runtime_hot_io``, a C library
  interposer: zero persistent-I/O calls over complete sessions);
* no retired history, receipt, event, scheduler or projection module is loaded;
* the session's generation, commit and state root are unchanged by the turn.

Gate A (HOT isolation): well over a hundred managed transitions and queries complete in one session with no session
transition and no Python file I/O; the session's anchor names the anchored state throughout (nothing is published).

Interface mechanics only (TRAINING=NONE SEMANTICS=NONE fixture map); no cognition claim.
"""
from __future__ import annotations

import os
import sys

import pytest

from elpis.ECS.k1 import K1Library
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest, run_turn

from ..conftest import admit_k1_set, require_runtime_library, runtime_config, session_marks
from ._turn_fixtures import LEARN, TEST_CODEC_PIN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import _Counting, k1, world  # noqa: F401 (the k1 fixture)

_OS_CALLS = ("open", "close", "pwrite", "write", "pread", "read", "fsync", "fdatasync", "rename", "replace", "unlink",
             "scandir", "fstat", "stat")


class _CountingOS:
    def __init__(self, monkeypatch):
        self.calls: dict[str, int] = {}
        for name in _OS_CALLS:
            fn = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap(name, fn))

    def _wrap(self, name, fn):
        def call(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return fn(*args, **kwargs)
        return call


class _CountingSymbols:
    """Counts calls of the already-bound C symbols with a prefix (ctypes caches bound functions)."""

    def __init__(self, lib, prefix):
        self.calls: dict[str, int] = {}
        for name, fn in list(vars(lib).items()):
            if name.startswith(prefix):
                setattr(lib, name, self._wrap(name, fn))

    def _wrap(self, name, fn):
        def call(*args):
            self.calls[name] = self.calls.get(name, 0) + 1
            return fn(*args)
        return call


# The retired receipt-history modules (the retired packages themselves are absent: tests/boundary/test_ecs_split.py).
_RETIRED_MODULES = ("elpis.runtime.history", "elpis.runtime.native_history", "elpis.runtime.continuity",
                    "elpis.runtime.recovery")


def _config():
    return runtime_config(require_runtime_library(), TEST_CODEC_PIN)


@pytest.mark.parametrize("experiences,steps", [(1, 1), (8, 60)])
def test_a_managed_turn_is_three_runtimecore_crossings_and_writes_nothing(experiences, steps, monkeypatch):
    k1 = K1Library(admit_k1_set()[1].lib)   # the managed runtime drives only admitted K1 code
    fixture = dict(experiences=experiences, steps=steps)
    counter = _Counting(k1._k)

    # The bare canonical turn: the reference transaction (begin, one schedule, commit).
    with world(k1) as bare:
        run_turn(bare, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        counter.calls.clear()
        run_turn(bare, "turn", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        bare_crossings = dict(counter.calls)

    with world(k1) as state, Runtime(_config()) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_turn(state, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        marks, anchored = session_marks(runtime), runtime.continuity.snapshot()
        generation = runtime.live_identity(state)[1]
        runtime._core.counters(reset=True)
        codec = admitted(FixtureMap(**fixture))   # admission is the cold path (it measures the codec's source)
        core_calls = _CountingSymbols(runtime._core._f, "elpis_runtime_")
        counter.calls.clear()
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            result = runtime.run_turn(state, "turn", tokenizer=ByteTokens(), codec=codec, authority=LEARN)
        python_k1_calls, turn_core_calls = dict(counter.calls), dict(core_calls.calls)
        native = runtime._core.counters()

        # The turn is volatile: the live generation advanced by exactly this commit; the session did not move.
        assert result.durability == "VOLATILE" and result.state_after_digest is None
        assert (result.generation_before, result.generation_after) == (generation, generation + 1)
        # Python calls K1 not at all; it crosses into RuntimeCore three times (the fail-stop probe, begin, commit).
        assert python_k1_calls == {}, python_k1_calls
        assert turn_core_calls == {"elpis_runtime_fault": 1, "elpis_runtime_turn_begin": 1,
                                   "elpis_runtime_turn_commit": 1}, turn_core_calls
        assert {k: bare_crossings.get(k, 0) for k in ("txn_begin", "txn_run_schedule", "txn_commit_identity")} == \
            {"txn_begin": 1, "txn_run_schedule": 1, "txn_commit_identity": 1}, bare_crossings
        # RuntimeCore's K1 crossings are the transaction's own; no digest or identity read, no session transition.
        assert native == {"k1_state_digests": 0, "k1_reserves": 0, "k1_txn_begins": 1, "k1_run_schedules": 1,
                          "k1_commits": 1, "k1_aborts": 0, "session_transitions": 0, "k1_queries": 0,
                          "k1_shapes": 0, "k1_lease_claims": 0, "k1_identities": 0, "k1_observes": 0}, native
        # No file I/O on the turn in Python; the session's marks and anchor are unchanged.
        assert fs.calls == {}, fs.calls
        assert session_marks(runtime) == marks and runtime.continuity.snapshot() == anchored
        assert runtime.live_identity(state)[1] == generation + 1

    assert not [m for m in sys.modules if m.startswith(_RETIRED_MODULES)]


TRANSITIONS = 120


def test_gate_a_hot_transitions_and_queries_make_no_session_transition_and_no_file_io(k1, monkeypatch):   # noqa: F811
    learn = LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN)
    query = QueryRequest("query", ByteTokens(), admitted(FixtureMap()))
    with world(k1, max_rows=8) as state, Runtime(_config()) as runtime:
        anchored = runtime.anchor_cognition(state)
        runtime.run_learn(state, learn)
        marks = session_marks(runtime)
        generation = runtime.live_identity(state)[1]
        runtime._core.counters(reset=True)
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            for n in range(TRANSITIONS):
                result = runtime.run_learn(state, learn)
                assert result.durability == "VOLATILE" and result.generation_after == generation + n + 1, n
                if n % 10 == 0:
                    runtime.run_query(state, query)
        native = runtime._core.counters()
        assert native["k1_commits"] == TRANSITIONS and native["k1_aborts"] == 0
        assert (native["session_transitions"], native["k1_identities"], native["k1_state_digests"]) == (0, 0, 0)
        assert fs.calls == {}, fs.calls
        assert session_marks(runtime) == marks and runtime.continuity.snapshot() == anchored
        assert runtime.live_identity(state)[1] == generation + TRANSITIONS
