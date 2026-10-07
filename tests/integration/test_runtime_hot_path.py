"""The managed canonical turn's hot path, measured dynamically (docs/CONTINUITY.md, docs/ARCHITECTURE.md).

``Runtime.run_turn`` is codec -> one native K1 transaction (schedule and readout) -> decode -> native commit ->
one continuity publication. Measured on a warm turn:

* native K1 crossings are exactly those of the bare canonical turn (continuity adds none);
* filesystem work is one ``pwrite`` of one 136-byte record and one ``fdatasync``: no open, rename, fsync,
  directory sync, read or unlink;
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
from elpis.continuity import RECORD_SIZE
from elpis.runtime import Runtime, RuntimeConfig
from elpis.runtime.cognition import run_turn

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


_RETIRED_MODULES = ("elpis.ECS_C", "elpis.ECS_G", "elpis.runtime.history", "elpis.runtime.native_history",
                    "elpis.runtime.continuity")


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

    with world(k1) as state, Runtime(RuntimeConfig(tmp_path / "continuity")) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_turn(state, "warm", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture), learning_rate=RATE)
        footprint = sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir())
        counter.calls.clear()
        with monkeypatch.context() as patch:
            fs = _CountingOS(patch)
            result = runtime.run_turn(state, "turn", tokenizer=ByteTokens(), codec_map=FixtureMap(**fixture),
                                      learning_rate=RATE)
        managed_crossings = dict(counter.calls)

        assert result.state_after_digest == state.state_digest() == runtime.continuity.snapshot().k1_state_digest
        # Continuity adds no native crossing beyond the canonical turn's own (the restart check reads the digest
        # the native commit already returned).
        assert managed_crossings == bare_crossings, (managed_crossings, bare_crossings)
        # Exactly one in-place fixed-size write and one data sync.
        assert fs.calls == {"pwrite": 1, "fdatasync": 1}, fs.calls
        assert fs.bytes_written == RECORD_SIZE == 136
        assert sorted((p.name, p.stat().st_size) for p in (tmp_path / "continuity").iterdir()) == footprint
        assert footprint == [("continuity.a", 136), ("continuity.b", 136)]

    assert not [m for m in sys.modules if m.startswith(_RETIRED_MODULES)]
