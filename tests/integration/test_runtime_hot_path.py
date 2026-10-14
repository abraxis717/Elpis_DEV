"""The managed turn's HOT lane, measured dynamically (docs/RUNTIME_CORE.md, "Live and durable authority").

``Runtime.run_turn`` is codec -> RuntimeCore (one native K1 transaction begun at the live coordinate: begin, schedule
and readout) -> decode -> RuntimeCore (native commit: volatile). Measured on a warm turn with a provisioned K1
checkpoint store attached, so durable publication *could* write if it were on this path:

* Python makes no K1 call at all and three crossings into RuntimeCore: the fail-stop probe, turn begin and turn
  commit;
* RuntimeCore's K1 crossings are those of the transaction itself (begin at the live generation, one schedule,
  commit): no state digest, no identity read, no envelope capture, no reservation, no abort;
* nothing is published: zero continuity I/O and zero checkpoint I/O (testing-library I/O counters), and Python
  performs no file I/O at all;
* no retired history, receipt, event, scheduler or projection module is loaded;
* the durable footprint (two continuity slots, two checkpoint slots) is byte-for-byte unchanged by the turn.

Gate A (HOT isolation): with the durability worker stalled behind a pending durable request, well over a hundred
managed transitions and queries complete with no continuity or checkpoint I/O; they cannot have waited for the lane,
since it never ran. When the worker resumes, the newest request publishes the state captured when it was requested
(never a later volatile one), the older waiting request is superseded, and continuity names exactly the acknowledged
identity.

Interface mechanics only (TRAINING=NONE SEMANTICS=NONE fixture map); no cognition claim.
"""
from __future__ import annotations

from dataclasses import replace
import os
import sys

import pytest

from elpis.ECS.k1 import K1Library
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest, run_turn
from elpis.runtime.core import RuntimeLibrary
from elpis.runtime.recovery import provision_k1_checkpoint

from ..conftest import (admit_k1_set, library_id_of, native_authority, require_runtime_library,
                        runtime_config)
from ._turn_fixtures import LEARN, TEST_CODEC_PIN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _Counting, k1, world  # noqa: F401 (the k1 fixture)

_OS_CALLS = ("open", "close", "pwrite", "write", "pread", "read", "fsync", "fdatasync", "rename", "replace", "unlink",
             "scandir", "fstat", "stat")
_SLOTS = ("continuity.a", "continuity.b")
_CHECKPOINT_SLOTS = ("k1-checkpoint.a", "k1-checkpoint.b")


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


# The retired receipt-history modules (the retired packages themselves are absent: tests/boundary/test_one_ecs.py).
_RETIRED_MODULES = ("elpis.runtime.history", "elpis.runtime.native_history", "elpis.runtime.continuity")

_NO_IO = {"opens": 0, "preads": 0, "pread_bytes": 0, "pwrites": 0, "pwrite_bytes": 0, "data_syncs": 0,
          "dir_syncs": 0, "renames": 0, "unlinks": 0}
_NO_CHECKPOINT_IO = {"preads": 0, "pread_bytes": 0, "pwrites": 0, "pwrite_bytes": 0, "data_syncs": 0}


def _config(tmp_path, k1):
    """A testing-library runtime with a provisioned K1 checkpoint store attached."""
    path = require_runtime_library(testing=True)
    library = RuntimeLibrary.admit(path.parent, path.name, native_authority(), library_id_of(path))
    store = tmp_path / "k1-checkpoint"
    provision_k1_checkpoint(library, store, k1.envelope_bytes(DIM, WIDTH))
    return replace(runtime_config(tmp_path / "continuity", path, TEST_CODEC_PIN), k1_checkpoint_dir=store)


def _durable_bytes(tmp_path):
    return (tuple((tmp_path / "continuity" / n).read_bytes() for n in _SLOTS)
            + tuple((tmp_path / "k1-checkpoint" / n).read_bytes() for n in _CHECKPOINT_SLOTS))


@pytest.mark.parametrize("experiences,steps", [(1, 1), (8, 60)])
def test_a_managed_turn_is_three_runtimecore_crossings_and_publishes_nothing(experiences, steps, tmp_path,
                                                                            monkeypatch):
    k1 = K1Library(admit_k1_set()[1].lib)   # the managed runtime drives only admitted K1 code
    fixture = dict(experiences=experiences, steps=steps)
    counter = _Counting(k1._k)

    # The bare canonical turn: the reference transaction (begin, one schedule, commit).
    with world(k1) as bare:
        run_turn(bare, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        counter.calls.clear()
        run_turn(bare, "turn", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        bare_crossings = dict(counter.calls)

    with world(k1) as state, Runtime(_config(tmp_path, k1)) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_turn(state, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        durable = _durable_bytes(tmp_path)
        footprint = sorted((p.name, p.stat().st_size) for d in ("continuity", "k1-checkpoint")
                           for p in (tmp_path / d).iterdir())
        generation = runtime.durability().live_generation
        runtime.continuity.testing_counters(reset=True)
        runtime._core.testing_checkpoint_io_counters(reset=True)
        runtime._core.counters(reset=True)
        codec = admitted(FixtureMap(**fixture))   # admission is the cold path (it measures the codec's source)
        core_calls = _CountingSymbols(runtime._core._f, "elpis_runtime_")
        counter.calls.clear()
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            result = runtime.run_turn(state, "turn", tokenizer=ByteTokens(), codec=codec, authority=LEARN)
        python_k1_calls, turn_core_calls = dict(counter.calls), dict(core_calls.calls)
        native = runtime._core.counters()
        io, checkpoint_io = runtime.continuity.testing_counters(), runtime._core.testing_checkpoint_io_counters()

        # The turn is volatile: the live generation advanced by exactly this commit, nothing durable moved.
        assert result.durability == "VOLATILE" and result.state_after_digest is None
        assert (result.generation_before, result.generation_after) == (generation, generation + 1)
        assert runtime.durability().live_generation == generation + 1
        # Python calls K1 not at all; it crosses into RuntimeCore three times (the fail-stop probe, begin, commit).
        assert python_k1_calls == {}, python_k1_calls
        assert turn_core_calls == {"elpis_runtime_fault": 1, "elpis_runtime_turn_begin": 1,
                                   "elpis_runtime_turn_commit": 1}, turn_core_calls
        assert {k: bare_crossings.get(k, 0) for k in ("txn_begin", "txn_run_schedule", "txn_commit_identity")} == \
            {"txn_begin": 1, "txn_run_schedule": 1, "txn_commit_identity": 1}, bare_crossings
        # RuntimeCore's K1 crossings are the transaction's own; no digest, identity read or capture, no publication.
        assert native == {"k1_state_digests": 0, "k1_reserves": 0, "k1_txn_begins": 1, "k1_run_schedules": 1,
                          "k1_commits": 1, "k1_aborts": 0, "publications": 0, "k1_queries": 0, "k1_shapes": 0,
                          "k1_lease_claims": 0, "k1_identities": 0, "k1_captures": 0, "durable_requests": 0,
                          "durable_superseded": 0, "durable_acknowledged": 0, "durable_failed": 0}, native
        # No file I/O anywhere on the turn: not in Python, not in continuity, not in the checkpoint store.
        assert fs.calls == {}, fs.calls
        assert io == _NO_IO, io
        assert checkpoint_io == _NO_CHECKPOINT_IO, checkpoint_io
        assert _durable_bytes(tmp_path) == durable
        assert sorted((p.name, p.stat().st_size) for d in ("continuity", "k1-checkpoint")
                      for p in (tmp_path / d).iterdir()) == footprint
        assert [n for n, _ in footprint] == [*_SLOTS, *_CHECKPOINT_SLOTS]

    assert not [m for m in sys.modules if m.startswith(_RETIRED_MODULES)]


TRANSITIONS = 120


def test_gate_a_hot_transitions_never_wait_for_a_stalled_durability_worker(k1, tmp_path):   # noqa: F811
    learn = LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN)
    query = QueryRequest("query", ByteTokens(), admitted(FixtureMap()))
    with world(k1, max_rows=8) as state, Runtime(_config(tmp_path, k1)) as runtime:
        runtime.anchor_cognition(state)
        anchored = runtime.continuity.snapshot()
        runtime._core.testing_durable_hold(True)   # the worker takes no job until released
        runtime.run_learn(state, learn)
        first = runtime.publish_durable(state, wait=False)
        requested = state.state_digest()            # what that request captured
        assert first.status == "PENDING" and runtime.durability().pending

        runtime.continuity.testing_counters(reset=True)
        runtime._core.testing_checkpoint_io_counters(reset=True)
        runtime._core.counters(reset=True)
        generation = runtime.durability().live_generation
        for n in range(TRANSITIONS):
            result = runtime.run_learn(state, learn)
            assert result.durability == "VOLATILE" and result.generation_after == generation + n + 1, n
            if n % 10 == 0:
                runtime.run_query(state, query)
        native = runtime._core.counters()
        # Every transition committed while the only durability work was stalled: none of them waited for it.
        assert native["k1_commits"] == TRANSITIONS and native["k1_aborts"] == 0
        assert (native["publications"], native["durable_requests"], native["k1_captures"],
                native["k1_identities"], native["k1_state_digests"]) == (0, 0, 0, 0, 0), native
        assert runtime.continuity.testing_counters() == _NO_IO
        assert runtime._core.testing_checkpoint_io_counters() == _NO_CHECKPOINT_IO
        assert runtime.continuity.snapshot() == anchored
        status = runtime.durability()
        assert status.pending and status.last_acknowledged.status == "NONE"
        assert status.live_generation == generation + TRANSITIONS

        # A newer request supersedes the waiting one; nothing ran yet, so it is still the only job.
        second = runtime.publish_durable(state, wait=False)
        captured = state.state_digest()
        assert runtime._core.durable_wait(first.sequence, 0).status == "SUPERSEDED"
        runtime.run_learn(state, learn)              # volatile work after the request is not part of it
        runtime._core.testing_durable_hold(False)
        done = runtime._core.durable_wait(second.sequence, 30)
        assert (done.status, done.recoverable, done.identity) == ("ACKNOWLEDGED", True, captured)
        assert captured != requested and captured != state.state_digest()
        assert runtime.continuity.snapshot().k1_state_digest == captured
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.authorized == captured
