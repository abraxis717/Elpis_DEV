"""RuntimeCore's turn lifecycle through the Python adapter, over real native K1 (standalone and FMS-resident).

Once a managed turn has opened a native K1 transaction, RuntimeCore ends it with exactly one native commit or
abort before it forgets the turn, on every lifecycle path the adapter exposes: explicit abort, close, destruction
(``RuntimeCore.__del__`` -> ``elpis_runtime_destroy``), reopen, and close after a fail-stop. An implicit abort
installs nothing and publishes nothing; the same live state takes the next transaction; an FMS-resident state's
WRITE pin is released. A commit is volatile: it ends the turn and publishes nothing (RuntimeCore ABI v8 has no
publication; a close ends the session). The fixture map is TRAINING=NONE SEMANTICS=NONE: interface mechanics only.
"""
from __future__ import annotations

import ctypes as C

import pytest

from elpis.runtime.composition import CompositionError
from elpis.runtime.core import RuntimeCore, RuntimeLibrary, describe
from elpis.runtime.fuel import CEILING

from ._turn_fixtures import ByteTokens, FixtureMap
from ..conftest import admit_native
from .test_codec_ecs_turn import RATE, _config, _Resident, adapter, k1, world  # noqa: F401 (fixtures)

PATHS = ("abort", "close", "destroy", "reopen", "fail-stop+close")


def _stimulus(text="lifecycle"):
    return FixtureMap().encode(ByteTokens().encode(text))


class _Subject:
    """One native K1 state of either kind and its native evidence: retained bytes, aborts, open transaction."""

    def __init__(self, state, resident):
        self.state, self.resident = state, resident

    def retained(self):
        info = self.info() if self.resident else None
        generation = info["generation"] if info else None
        return self.state.snapshot(), self.state.state_digest(), self.state.epoch, generation

    def info(self):
        return self.state._r.inspect(self.state.id)

    def native_aborts(self):
        return self.info()["aborts"] if self.resident else self.state.stats()["txn_aborts"]

    def probe_begin(self):
        """Begin a transaction under whatever managed lease holds the state (the lease is no secret):
        it observes whether one is open without being refused as an unmanaged caller. Returns the abort."""
        lease, token, source = self.state.managed_lease, C.c_uint64(), (C.c_uint8 * 32)()
        if not lease:
            txn = self.state.transaction()   # BUSY while another transaction is open
            return txn.abort
        if self.resident:
            r = self.state._r
            rc = r._f.leased_txn_begin(r._live(), self.state.id, lease, source, C.byref(token))
            abort = lambda: r._f.txn_abort(r._live(), self.state.id, token.value)
        else:
            k = self.state._k
            rc = k.leased_txn_begin(self.state._live(), lease, source, C.byref(token))
            abort = lambda: k.txn_abort(self.state._live(), token.value)
        assert rc == 0, rc
        return abort

    def released(self):
        """No transaction open (and, resident, no pin or residency hold held); another transaction begins and aborts
        cleanly."""
        if self.resident:
            info = self.info()
            assert info["transaction_open"] == 0 and info["lease_count"] == 0
            assert self.state._r.resident_holds(self.state.id) == 0
        abort = self.probe_begin()
        if self.resident:
            assert self.info()["transaction_open"] == 1 and self.info()["lease_count"] == 1
        assert abort() in (None, 0)


def _core():
    """A RuntimeCore over its volatile session (opened: a fresh genesis, nothing anchored)."""
    config = _config()
    core = RuntimeCore(RuntimeLibrary(admit_native(config.runtime_library)), config.session)
    core.open()
    return core


@pytest.fixture(params=["k1", "fms-k1"])
def subject(request, k1, adapter, tmp_path):   # noqa: F811 (module fixtures)
    if request.param == "k1":
        with world(k1, max_rows=8) as state:
            yield _Subject(state, False)
    else:
        with _Resident(k1, adapter, max_rows=8) as state:
            yield _Subject(state, True)


@pytest.mark.parametrize("path", PATHS)
def test_an_open_turn_is_ended_natively_on_every_lifecycle_path(subject, path, tmp_path):
    core = _core()
    descriptor, _owner = describe(subject.state)
    core.anchor(descriptor)
    core.turn_begin(descriptor, _stimulus("first"), RATE, CEILING)
    core.turn_commit(descriptor)
    before, aborts = subject.retained(), subject.native_aborts()

    core.turn_begin(descriptor, _stimulus(), RATE, CEILING)   # the schedule ran on the candidate; nothing committed
    if subject.resident:
        # Two FMS pins: the transaction's WRITE pin and the runtime's residency hold (docs/RUNTIME_CORE.md, "HOT
        # residency"), which the runtime drops whenever it unbinds the state (``released`` checks: none left).
        assert subject.info()["transaction_open"] == 1 and subject.info()["lease_count"] == 2
        assert subject.state._r.resident_holds(subject.state.id) == 1
        with pytest.raises(Exception, match="BUSY"):   # the adapter's own lifetime guard while the turn is open
            subject.state._r.close_state(subject.state.id)
    core.counters(reset=True)

    if path == "abort":
        core.turn_abort(descriptor)
        assert core.counters()["k1_aborts"] == 1
    elif path == "close":
        core.close()
        assert core.counters()["k1_aborts"] == 1
        core.close()   # double close: nothing left to end
        assert core.counters()["k1_aborts"] == 1
    elif path == "destroy":
        del core   # RuntimeCore.__del__ -> elpis_runtime_destroy -> RuntimeCore's Drop
        core = None
    elif path == "reopen":
        with pytest.raises(CompositionError) as info:
            core.open()
        assert info.value.code == "CONTINUITY_OPEN"
        assert core.counters()["k1_aborts"] == 0 and subject.native_aborts() == aborts   # kept, not forgotten
        core.close()
        assert core.counters()["k1_aborts"] == 1
        core.open()
    else:
        authority = core.evolution_authority().evolution
        core.evolution_reserve(authority, "11" * 32)
        with pytest.raises(CompositionError) as info:   # a stale reservation: refused, no fail-stop (ABI v8)
            core.evolution_reserve(authority, "33" * 32)
        assert info.value.code == "RUNTIME_EVOLUTION_IN_FLIGHT" and core.fault() is None
        core.evolution_abandon()                         # an abandoned attempt fail-stops the session
        assert core.fault() == "CONTINUITY_EVOLUTION_PENDING"
        core.close()
        assert core.counters()["k1_aborts"] == 1
        core.open()                                      # a new session: the fail-stop is gone with the old one
        assert core.fault() is None

    # Exactly one native abort ended the transaction; nothing installed.
    assert subject.native_aborts() == aborts + 1
    assert subject.retained() == before
    if core is not None:
        core.close()
    subject.released()
    # The same live state takes the next managed turn in a new session, after the operator anchors it explicitly
    # (a session resumes nothing).
    core = _core()
    assert not core.snapshot().anchored
    record = core.anchor(descriptor)
    core.turn_begin(descriptor, _stimulus("next"), RATE, CEILING)
    committed = core.turn_commit(descriptor)
    assert committed.generation_after == committed.generation_before + 1 == committed.live_generation
    assert subject.state.epoch > before[2] and core.snapshot() == record   # volatile: the session did not move
    core.close()
    subject.released()


def test_a_commit_ends_the_turn_without_an_abort(subject, tmp_path):
    core = _core()
    descriptor, _owner = describe(subject.state)
    core.anchor(descriptor)
    aborts = subject.native_aborts()
    anchored = core.snapshot()
    core.turn_begin(descriptor, _stimulus(), RATE, CEILING)
    committed = core.turn_commit(descriptor)
    assert core.counters()["k1_aborts"] == 0 and subject.native_aborts() == aborts
    assert core.live_identity(descriptor) == (subject.state.state_digest(), committed.live_generation)
    assert core.snapshot() == anchored   # the commit is volatile
    del core
    assert subject.native_aborts() == aborts
    subject.released()


def test_lifecycle_without_a_turn_makes_no_native_call(subject, tmp_path):
    core = _core()
    descriptor, _owner = describe(subject.state)
    core.anchor(descriptor)
    core.counters(reset=True)
    core.close()
    core.close()
    core.open()
    core.close()
    core.open()
    assert core.counters() == dict.fromkeys(core.counters(), 0)
    del core
    subject.released()


def test_the_facade_releases_the_bound_owner_only_after_the_turn_is_ended(k1, tmp_path):   # noqa: F811
    from elpis.runtime import Runtime

    with world(k1, max_rows=8) as state:
        before = state.snapshot()
        runtime = Runtime(_config()).open()
        runtime.anchor_cognition(state)
        descriptor, _owner = describe(state)
        runtime._core.turn_begin(descriptor, _stimulus(), RATE, CEILING)   # an interrupted turn, left open
        aborts = state.stats()["txn_aborts"]
        assert runtime._bound_owner is state
        runtime.close()
        assert state.stats()["txn_aborts"] == aborts + 1 and runtime._bound_owner is None
        assert state.snapshot() == before
        state.transaction().abort()
