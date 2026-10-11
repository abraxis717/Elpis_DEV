"""The managed canonical turn's hot path, measured dynamically (docs/CONTINUITY.md, docs/ARCHITECTURE.md).

``Runtime.run_turn`` is codec -> RuntimeCore (one native K1 transaction: begin, schedule and readout) -> decode ->
RuntimeCore (native commit, one continuity publication). Measured on a warm turn:

* Python makes no K1 call at all and three crossings into RuntimeCore: the fail-stop probe, turn begin and turn
  commit;
* RuntimeCore's K1 crossings are those of the bare canonical turn's transaction (begin, one schedule, commit):
  no identity read (the binding holds), no reservation, no abort; and it publishes once;
* Python performs no file I/O at all; the library's filesystem work is one ``pwrite`` of one 176-byte record
  and one ``fdatasync``: no open, rename, directory sync, read or unlink (testing-library I/O counters);
* no retired history, receipt, event, scheduler or projection module is loaded;
* the durable footprint is unchanged by the turn (two fixed slots).

Interface mechanics only (TRAINING=NONE SEMANTICS=NONE fixture map); no cognition claim.
"""
from __future__ import annotations

import ctypes
import os
import sys

import pytest

from elpis.ECS.k1 import K1Library
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.cognition import run_turn

from ..conftest import admit_k1_set, require_runtime_library, runtime_config
from ._turn_fixtures import LEARN, TEST_CODEC_PIN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import RATE, _beside, _Counting, world

_OS_CALLS = ("open", "close", "pwrite", "write", "pread", "read", "fsync", "fdatasync", "rename", "replace", "unlink",
             "scandir", "fstat", "stat")


class _CountingOS:
    def __init__(self, monkeypatch):
        self.calls: dict[str, int] = {}
        self.bytes_written = 0
        for name in _OS_CALLS:
            fn = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap(name, fn))

    def _wrap(self, name, fn):
        def call(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            if name in ("pwrite", "write"):
                self.bytes_written += len(args[1])
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


@pytest.mark.parametrize("experiences,steps", [(1, 1), (8, 60)])
def test_managed_turn_is_two_runtimecore_crossings_and_one_fixed_publication(experiences, steps, tmp_path, monkeypatch):
    k1 = K1Library(admit_k1_set()[1].lib)   # the managed runtime drives only admitted K1 code
    fixture = dict(experiences=experiences, steps=steps)
    counter = _Counting(k1._k)

    # The bare canonical turn: the reference crossing count.
    with world(k1) as bare:
        run_turn(bare, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        counter.calls.clear()
        run_turn(bare, "turn", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        bare_crossings = dict(counter.calls)

    config = runtime_config(tmp_path / "continuity", require_runtime_library(testing=True), TEST_CODEC_PIN)
    with world(k1) as state, Runtime(config) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_turn(state, "warm", tokenizer=ByteTokens(), codec=admitted(FixtureMap(**fixture)), authority=LEARN)
        footprint = sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir())
        runtime.continuity.testing_counters(reset=True)
        runtime._core.counters(reset=True)
        codec = admitted(FixtureMap(**fixture))   # admission is the cold path (it measures the codec's source)
        core_calls = _CountingSymbols(runtime._core._f, "elpis_runtime_")
        counter.calls.clear()
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            result = runtime.run_turn(state, "turn", tokenizer=ByteTokens(), codec=codec, authority=LEARN)
        python_k1_calls, turn_core_calls = dict(counter.calls), dict(core_calls.calls)
        native = runtime._core.counters()
        io = runtime.continuity.testing_counters()

        assert result.state_after_digest == state.state_digest() == runtime.continuity.snapshot().k1_state_digest
        # Python calls K1 not at all; it crosses into RuntimeCore three times (the fail-stop probe, begin, commit).
        assert python_k1_calls == {}, python_k1_calls
        assert turn_core_calls == {"elpis_runtime_fault": 1, "elpis_runtime_turn_begin": 1,
                                   "elpis_runtime_turn_commit": 1}, turn_core_calls
        # RuntimeCore's K1 crossings are the bare turn's transaction: begin, one schedule, commit.
        assert {k: bare_crossings.get(k, 0) for k in ("txn_begin", "txn_run_schedule", "txn_commit_identity")} == \
            {"txn_begin": 1, "txn_run_schedule": 1, "txn_commit_identity": 1}, bare_crossings
        assert native == {"k1_state_digests": 0, "k1_reserves": 0, "k1_txn_begins": 1, "k1_run_schedules": 1,
                          "k1_commits": 1, "k1_aborts": 0, "publications": 1, "k1_queries": 0,
                          "k1_shapes": 0, "k1_lease_claims": 0}, native
        # Python does no file I/O; the library writes one complete fixed-size record and syncs it once.
        assert fs.calls == {}, fs.calls
        assert io == {"opens": 0, "preads": 0, "pread_bytes": 0, "pwrites": 1, "pwrite_bytes": 176,
                      "data_syncs": 1, "dir_syncs": 0, "renames": 0, "unlinks": 0}, io
        assert sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir()) == footprint
        assert footprint == [("continuity.a", 176), ("continuity.b", 176)]

    assert not [m for m in sys.modules if m.startswith(_RETIRED_MODULES)]
