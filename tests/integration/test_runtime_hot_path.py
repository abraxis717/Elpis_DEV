"""The managed canonical turn's hot path, measured dynamically (docs/CONTINUITY.md, docs/ARCHITECTURE.md).

``Runtime.run_turn`` is codec -> one native K1 transaction (schedule and readout) -> decode -> native commit ->
one continuity publication. Measured on a warm turn:

* native K1 crossings are exactly those of the bare canonical turn;
* continuity adds exactly one crossing into the Rust continuity library (the publication);
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

from ..conftest import require_continuity_library
from ._turn_fixtures import ByteTokens, FixtureMap
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
def test_managed_turn_adds_no_native_crossing_and_one_fixed_publication(experiences, steps, tmp_path, monkeypatch):
    k1 = K1Library(ctypes.CDLL(str(_beside("libelpis_ecsg_k1.so"))))
    fixture = dict(experiences=experiences, steps=steps)
    counter = _Counting(k1._k)

    # The bare canonical turn: the reference crossing count.
    with world(k1) as bare:
        run_turn(bare, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        counter.calls.clear()
        run_turn(bare, "turn", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        bare_crossings = dict(counter.calls)

    config = RuntimeConfig(tmp_path / "continuity", require_continuity_library(testing=True))
    with world(k1) as state, Runtime(config) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_turn(state, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        footprint = sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir())
        runtime.continuity.testing_counters(reset=True)
        continuity_calls = _CountingSymbols(runtime.continuity.library._lib, "elpis_continuity_")
        counter.calls.clear()
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            result = runtime.run_turn(state, "turn", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture),
                                      learning_rate=RATE)
        managed_crossings, turn_continuity_calls = dict(counter.calls), dict(continuity_calls.calls)
        io = runtime.continuity.testing_counters()

        assert result.state_after_digest == state.state_digest() == runtime.continuity.snapshot().k1_state_digest
        # The K1 crossings are the canonical turn's own (the restart check reads the digest the native commit
        # already returned); continuity is one crossing into its library, the publication itself.
        assert managed_crossings == bare_crossings, (managed_crossings, bare_crossings)
        assert turn_continuity_calls == {"elpis_continuity_commit_cognition": 1}, turn_continuity_calls
        # Python does no file I/O; the library writes one complete fixed-size record and syncs it once.
        assert fs.calls == {}, fs.calls
        assert io == {"opens": 0, "preads": 0, "pread_bytes": 0, "pwrites": 1, "pwrite_bytes": 176,
                      "data_syncs": 1, "dir_syncs": 0, "renames": 0, "unlinks": 0}, io
        assert sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir()) == footprint
        assert footprint == [("continuity.a", 176), ("continuity.b", 176)]

    assert not [m for m in sys.modules if m.startswith(_RETIRED_MODULES)]
