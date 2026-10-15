"""K1 Recovery R0 over real native K1 (docs/K1_RECOVERY_R0.md).

CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

A distinct, bounded owner keeps the complete ``(W, epoch, H, a)`` envelope of the authorized K1 state, and of at most
one newer candidate, in exactly two operator-provisioned slot files. Only the anchor and an explicit durable
publication write them (RuntimeCore ABI v4, docs/RUNTIME_CORE.md "Live and durable authority"); a LEARN is volatile
and never touches them. Restart resumes exactly continuity's identity: the last acknowledged state, never a later
volatile one. A crash inside a durable publication leaves an explicit unresolved candidate that only an operator
discard or adopt settles (no silent roll forward or back); until then durable publication refuses. The fixture map is
TEST_ONLY TRAINING=NONE SEMANTICS=NONE; nothing here is a cognition claim.
"""
from __future__ import annotations

from dataclasses import replace
import os

import pytest

from elpis.ECS.k1 import K1State
from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest, run_learn
from elpis.runtime.composition import CompositionError
from elpis.runtime.core import RuntimeLibrary
from elpis.runtime.recovery import provision_k1_checkpoint

from ..conftest import library_id_of, native_authority, require_runtime_library
from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, _config, _Resident, adapter, k1, world  # noqa: F401 (fixtures)

HEADER = 128
SLOTS = ("k1-checkpoint.a", "k1-checkpoint.b")


def _library(testing=False):
    path = require_runtime_library(testing=testing)
    return RuntimeLibrary.admit(path.parent, path.name, native_authority(), library_id_of(path))


def _provision(k1, path, dim=DIM, width=WIDTH):   # noqa: F811
    provision_k1_checkpoint(_library(), path, k1.envelope_bytes(dim, width))
    return path


@pytest.fixture
def store(k1, tmp_path):   # noqa: F811
    return _provision(k1, tmp_path / "k1-checkpoint")


def _cfg(tmp_path, checkpoint, testing=False, continuity="c"):
    return replace(_config(tmp_path / continuity, testing=testing), k1_checkpoint_dir=checkpoint)


def _learn(text="learn"):
    return LearnRequest(text, ByteTokens(), admitted(FixtureMap()), LEARN)


def _query(text="query"):
    return QueryRequest(text, ByteTokens(), admitted(FixtureMap()))


def _footprint(path):
    return sorted((p.name, p.stat().st_size, p.stat().st_ino) for p in path.iterdir())


def _refused(call, code):
    with pytest.raises(CompositionError) as info:
        call()
    assert info.value.code == code, (info.value.code, str(info.value))


def test_provisioning_is_an_explicit_bounded_operator_act(k1, tmp_path):   # noqa: F811
    envelope = k1.envelope_bytes(DIM, WIDTH)
    path = _provision(k1, tmp_path / "ck")
    assert [(n, s) for n, s, _ in _footprint(path)] == [(n, HEADER + envelope) for n in SLOTS]
    assert all((path / n).read_bytes() == bytes(HEADER + envelope) for n in SLOTS)
    library = _library()
    _refused(lambda: provision_k1_checkpoint(library, path, envelope), "CHECKPOINT_INVALID")   # never overwritten
    _refused(lambda: provision_k1_checkpoint(library, tmp_path / "small", 16), "CHECKPOINT_INVALID")
    assert not (tmp_path / "small").exists()
    with pytest.raises(CompositionError):
        provision_k1_checkpoint(library, "relative", envelope)
    # The runtime never provisions: a missing, malformed or locked store refuses its open.
    (tmp_path / "extra").mkdir()
    for directory in (tmp_path / "missing", tmp_path / "extra"):
        runtime = Runtime(_cfg(tmp_path, directory))
        _refused(runtime.open, "CHECKPOINT_INVALID" if directory.exists() else "CHECKPOINT_IO")
        assert runtime.fault == "RUNTIME_CLOSED" and not directory.joinpath(SLOTS[0]).exists()
    with Runtime(_cfg(tmp_path, path)):
        _refused(Runtime(_cfg(tmp_path, path, continuity="other")).open, "CHECKPOINT_INVALID")   # exclusively owned
    (path / "stray").write_bytes(b"")
    _refused(Runtime(_cfg(tmp_path, path)).open, "CHECKPOINT_INVALID")


def _slot_bytes(store):
    return tuple((store / n).read_bytes() for n in SLOTS)


def test_volatile_learns_never_touch_the_slots_and_each_durable_publication_is_resumable(
        k1, store, tmp_path):   # noqa: F811
    footprint = _footprint(store)
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        assert runtime.recover_k1().disposition == "NOTHING_TO_RECOVER"
        runtime.anchor_cognition(state)
        published = state.snapshot()
        for n in range(6):
            recovery = runtime.recover_k1()
            assert recovery.disposition == "RESUMABLE" and recovery.candidate is None, n
            # The complete (W, epoch, H, a) envelope of the last acknowledged state, byte for byte.
            assert recovery.envelope == published and recovery.authorized == published[-32:]
            assert runtime.continuity.snapshot().k1_state_digest == published[-32:]
            slots = _slot_bytes(store)
            for k in range(3):
                runtime.run_learn(state, _learn(f"turn {n}.{k}"))
            assert _slot_bytes(store) == slots, n                        # volatile: not one slot byte moved
            assert runtime.recover_k1().envelope == published, n         # restart would resume the published state
            ticket = runtime.publish_recoverable(state)
            assert (ticket.status, ticket.recoverable, ticket.identity) == \
                ("ACKNOWLEDGED", True, state.state_digest()), n
            published = state.snapshot()
        with K1State.restore(k1, runtime.recover_k1().envelope) as twin:
            assert (twin.w(), twin.h_packed(), twin.a(), twin.epoch) == (state.w(), state.h_packed(), state.a(), state.epoch)
    # Physically bounded: the same two files, same sizes, same inodes, whatever the number of publications.
    assert _footprint(store) == footprint


def test_restart_resumes_exactly_the_last_acknowledged_state(k1, store, tmp_path):   # noqa: F811
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(state)
        for n in range(3):
            runtime.run_learn(state, _learn(f"before {n}"))
        assert runtime.publish_recoverable(state).acknowledged
        before = state.snapshot()
        answer = runtime.run_query(state, _query()).readout.values
        runtime.run_learn(state, _learn("volatile, then lost"))
        lost = state.snapshot()
    # The process ends; nothing of the K1 state survives but the checkpoint and continuity.
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.envelope == before and recovery.envelope != lost
        with K1State.restore(k1, recovery.envelope, max_rows=8) as resumed, \
                K1State.restore(k1, before, max_rows=8) as reference:
            assert runtime.run_query(resumed, _query()).readout.values == answer
            result = runtime.run_learn(resumed, _learn("after"))
            run_learn(reference, _learn("after"))      # the same LEARN on the same state, outside the lineage
            assert result.durability == "VOLATILE" and resumed.state_digest() == reference.state_digest()
            assert runtime.live_identity(resumed)[0] == reference.state_digest()
            assert runtime.publish_recoverable(resumed).acknowledged
            assert runtime.continuity.snapshot().k1_state_digest == resumed.state_digest()


def _lineage(k1, tmp_path, name, learns):   # noqa: F811
    """Another lineage's (valid, complete) checkpoint store."""
    path = _provision(k1, tmp_path / name)
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, path, continuity=name + "-c")) as runtime:
        runtime.anchor_cognition(state)
        for n in range(learns):
            runtime.run_learn(state, _learn(f"{name} {n}"))
        assert runtime.publish_recoverable(state).acknowledged
        return path, state.snapshot()


def _copy_slots(source, target):
    for n in SLOTS:
        (target / n).write_bytes((source / n).read_bytes())


def test_checkpoint_bytes_never_authorize_themselves(k1, store, tmp_path):   # noqa: F811
    foreign, envelope = _lineage(k1, tmp_path, "foreign", 2)
    # Unanchored continuity: valid slots of another lineage create no lineage.
    _copy_slots(foreign, store)
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.envelope, recovery.authorized) == ("NOTHING_TO_RECOVER", None, None)
        assert not runtime.continuity.snapshot().anchored
    # An anchored lineage whose slots were replaced by another lineage's: not resumable, and the foreign bytes,
    # restored, are refused as the lineage's state (right bytes, wrong authority).
    own = _provision(k1, tmp_path / "own")
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, own, continuity="own-c")) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("own"))
        assert runtime.publish_recoverable(state).acknowledged
        authorized = state.state_digest()
    _copy_slots(foreign, own)
    with Runtime(_cfg(tmp_path, own, continuity="own-c")) as runtime:
        recovery = runtime.recover_k1()
        assert (recovery.disposition, recovery.envelope, recovery.candidate) == ("CHECKPOINT_MISSING", None, None)
        assert recovery.authorized == authorized and not recovery.authorized_present
        _refused(lambda: runtime.adopt_k1_candidate(envelope[-32:]), "CHECKPOINT_INVALID")
        with K1State.restore(k1, envelope, max_rows=8) as impostor:
            _refused(lambda: runtime.run_query(impostor, _query()), "CONTINUITY_STATE_MISMATCH")
        assert runtime.continuity.snapshot().k1_state_digest == authorized


def test_a_torn_slot_is_not_a_checkpoint(k1, store, tmp_path):   # noqa: F811
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn())
        assert runtime.publish_recoverable(state).acknowledged
        authorized = state.state_digest()
    held = next(n for n in SLOTS if (store / n).read_bytes()[32:64] == authorized)
    older = next(n for n in SLOTS if n != held)
    for name, expected in ((older, "RESUMABLE"), (held, "CHECKPOINT_MISSING")):
        raw = bytearray((store / name).read_bytes())
        raw[HEADER + 100] ^= 0x01
        (store / name).write_bytes(bytes(raw))
        with Runtime(_cfg(tmp_path, store)) as runtime:
            assert runtime.recover_k1().disposition == expected, name


def test_a_state_of_another_shape_is_refused_never_resized_for(k1, tmp_path):   # noqa: F811
    store = _provision(k1, tmp_path / "ck", width=WIDTH * 2)
    footprint = _footprint(store)
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        _refused(lambda: runtime.anchor_cognition(state), "CHECKPOINT_SHAPE")
        assert not runtime.continuity.snapshot().anchored and state.managed_lease == 0
    assert _footprint(store) == footprint


# Durability-lane faults of the testing library (runtime.h, "Durability crash matrix"): action 1 dies (a simulated
# process death) at a step: 5 the checkpoint slot synced, continuity not yet published; 6 the continuity record torn.
_DIE, _SLOT_SYNCED, _RECORD_TORN = 1, 5, 6


def _crash(k1, tmp_path, store, point):   # noqa: F811
    """Anchor, LEARN and publish durably once, LEARN again (volatile), then die at ``point`` inside the durable
    publication of that state. Returns (the acknowledged envelope, the dead process's captured envelope)."""
    with world(k1, max_rows=8) as state:
        runtime = Runtime(_cfg(tmp_path, store, testing=True)).open()
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("warm"))
        assert runtime.publish_recoverable(state).acknowledged
        authorized = state.snapshot()
        runtime.run_learn(state, _learn("dies"))
        died = state.snapshot()
        runtime._core.testing_durable_fault(_DIE, point, 40)
        ticket = runtime.publish_recoverable(state)
        # Died after the slot sync: continuity's record never begun, not published (ABANDONED). Died tearing the
        # record: its outcome is unknown to the dying process (UNCERTAIN); restart resolves it (torn: not published).
        status = {_SLOT_SYNCED: "ABANDONED", _RECORD_TORN: "UNCERTAIN"}[point]
        assert (ticket.status, ticket.code) == (status, "CONTINUITY_TESTING_PROCESS_DEATH")
        assert runtime.fault == "CONTINUITY_TESTING_PROCESS_DEATH"   # the process is gone: nothing more runs
        runtime.close()
    return authorized, died


@pytest.mark.parametrize("point", [_SLOT_SYNCED, _RECORD_TORN], ids=["after-slot-sync", "torn-record"])
@pytest.mark.parametrize("resolution", ["discard", "adopt"])
def test_a_crash_inside_a_durable_publication_needs_explicit_reconciliation(k1, store, tmp_path, point,
                                                                           resolution):   # noqa: F811
    authorized, died = _crash(k1, tmp_path, store, point)
    assert died != authorized
    with Runtime(_cfg(tmp_path, store, testing=True)) as runtime:
        assert runtime.continuity.snapshot().k1_state_digest == authorized[-32:]   # never published
        recovery = runtime.recover_k1()
        assert recovery.disposition == "CANDIDATE_UNRESOLVED" and recovery.envelope is None
        assert recovery.authorized == authorized[-32:] and recovery.authorized_present
        candidate = recovery.candidate
        assert candidate == died[-32:]
        with K1State.restore(k1, authorized, max_rows=8) as resumed:
            # Thinking proceeds on the authorized state (volatile), but nothing is made durable over an
            # unresolved candidate: the explicit publication fails until the operator decides.
            runtime.run_query(resumed, _query())
            runtime.run_learn(resumed, _learn("next"))
            ticket = runtime.publish_recoverable(resumed)
            assert (ticket.status, ticket.code) == ("FAILED", "CHECKPOINT_UNRESOLVED")
            assert runtime.recover_k1().candidate == candidate                 # the evidence is untouched
            _refused(lambda: runtime.adopt_k1_candidate(candidate), "RUNTIME_TURN_OPEN")   # a state is bound
            runtime.release(resumed)
        for wrong in (authorized[-32:], bytes(32)):
            _refused(lambda: runtime.discard_k1_candidate(wrong), "CHECKPOINT_INVALID")
            _refused(lambda: runtime.adopt_k1_candidate(wrong), "CHECKPOINT_INVALID")
        if resolution == "discard":
            runtime.discard_k1_candidate(candidate)
            expected = authorized
        else:
            assert runtime.adopt_k1_candidate(candidate).k1_state_digest == candidate
            expected = died
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.candidate is None
        assert recovery.envelope == expected
        with K1State.restore(k1, recovery.envelope, max_rows=8) as resumed:
            runtime.run_learn(resumed, _learn("continues"))
            assert runtime.publish_recoverable(resumed).acknowledged
            assert runtime.continuity.snapshot().k1_state_digest == resumed.state_digest()


# Durability-lane faults: 6 the continuity record write fails, 8 its sync fails with the bytes lost, 9 its sync fails
# although the record is durable.
@pytest.mark.parametrize("failure", ["write", "sync-lost", "sync-durable"])
def test_a_failed_durable_publication_is_explicit(k1, store, tmp_path, failure):   # noqa: F811
    with world(k1, max_rows=8) as state:
        with Runtime(_cfg(tmp_path, store, testing=True)) as runtime:
            runtime.anchor_cognition(state)
            authorized = state.snapshot()
            runtime.run_learn(state, _learn("commit"))
            captured = state.snapshot()
            runtime._core.testing_durable_fault({"write": 6, "sync-lost": 8, "sync-durable": 9}[failure])
            ticket = runtime.publish_recoverable(state)
            assert ticket.status == ("FAILED" if failure == "write" else "UNCERTAIN")
            assert runtime.fault is None                          # the HOT lane keeps thinking
            runtime.run_learn(state, _learn("after"))
            assert captured != authorized
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        durable = runtime.continuity.snapshot().k1_state_digest
        if failure == "write":
            # Definitive: nothing was published and the slot written for it was withdrawn.
            assert durable == authorized[-32:]
            assert recovery.disposition == "RESUMABLE" and recovery.envelope == authorized
        elif failure == "sync-lost":
            assert durable == authorized[-32:]
            assert recovery.disposition == "CANDIDATE_UNRESOLVED" and recovery.candidate == captured[-32:]
        else:
            # The uncertain write did land: continuity authorizes the captured state, which is simply resumable.
            assert durable == captured[-32:]
            assert recovery.disposition == "RESUMABLE" and recovery.envelope == captured


def test_an_fms_resident_state_checkpoints_the_same_complete_envelope(k1, adapter, tmp_path):   # noqa: F811
    store = _provision(k1, tmp_path / "ck")
    with _Resident(k1, adapter, tmp_path, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("resident"))
        assert runtime.publish_recoverable(state).acknowledged
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.envelope == state.snapshot()


def test_without_an_attached_store_recovery_is_refused_and_learn_is_unchanged(k1, tmp_path):   # noqa: F811
    with world(k1, max_rows=8) as state, Runtime(_config(tmp_path / "c")) as runtime:
        _refused(runtime.recover_k1, "CHECKPOINT_INVALID")
        _refused(lambda: runtime.discard_k1_candidate(bytes(32)), "CHECKPOINT_INVALID")
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn())
        _refused(lambda: runtime.publish_recoverable(state), "RUNTIME_RECOVERY_UNAVAILABLE")   # never downgraded
        ticket = runtime.publish_identity(state)
        # Identity only: no store holds the bytes, so it is never a (recoverable) acknowledgment.
        assert ticket.identity_published and not ticket.acknowledged and not ticket.recoverable
        assert sorted(os.listdir(tmp_path / "c")) == ["continuity.a", "continuity.b"]


_READ_BACK_FAIL, _SYNC_LOST, _SYNC_DURABLE = 10, 8, 9
_DISPOSITIONS = ("acknowledged", "read-back-failed", "uncertain-lost", "uncertain-durable",
                 "recoverable-without-store", "identity-only")


@pytest.mark.parametrize("disposition", _DISPOSITIONS)
def test_every_publication_disposition_restarts_at_exactly_what_it_reported(k1, tmp_path, disposition):   # noqa: F811
    """What a ticket reports is what restart finds. ACKNOWLEDGED is never without complete state bytes that restart
    re-reads and re-verifies independently; identity-only publication is never an acknowledgment; a recoverable request
    without a checkpoint owner is refused (never downgraded); a failed request changes nothing durable; an UNCERTAIN one
    is resolved by restart to exactly one complete record, which may be the requested one."""
    with_store = disposition not in ("recoverable-without-store", "identity-only")
    store = _provision(k1, tmp_path / "ck") if with_store else None
    restart = _cfg(tmp_path, store) if with_store else _config(tmp_path / "c")
    with world(k1, max_rows=8) as state:
        with Runtime(_cfg(tmp_path, store, testing=True) if with_store else _config(tmp_path / "c", testing=True)) \
                as runtime:
            anchored = runtime.anchor_cognition(state).k1_state_digest
            runtime.run_learn(state, _learn("requested"))
            captured = state.snapshot()
            ticket = None
            if disposition == "recoverable-without-store":
                _refused(lambda: runtime.publish_recoverable(state), "RUNTIME_RECOVERY_UNAVAILABLE")
                assert runtime.durability().last_requested.sequence == 0          # nothing captured or enqueued
            elif disposition == "identity-only":
                ticket = runtime.publish_identity(state)
                assert (ticket.status, ticket.kind, ticket.recoverable) == ("IDENTITY_PUBLISHED", "IDENTITY", False)
            else:
                _refused(lambda: runtime.publish_identity(state), "RUNTIME_CHECKPOINT_ATTACHED")
                action = {"acknowledged": 0, "read-back-failed": _READ_BACK_FAIL, "uncertain-lost": _SYNC_LOST,
                          "uncertain-durable": _SYNC_DURABLE}[disposition]
                if action:
                    runtime._core.testing_durable_fault(action)
                ticket = runtime.publish_recoverable(state)
                assert ticket.kind == "RECOVERABLE" and ticket.recoverable == ticket.acknowledged
            if ticket is not None:
                assert ticket.acknowledged == (disposition == "acknowledged")
    expected = {"acknowledged": ("ACKNOWLEDGED", None), "read-back-failed": ("FAILED", "CHECKPOINT_IO"),
                "uncertain-lost": ("UNCERTAIN", "CONTINUITY_PUBLICATION_UNCERTAIN"),
                "uncertain-durable": ("UNCERTAIN", "CONTINUITY_PUBLICATION_UNCERTAIN"),
                "identity-only": ("IDENTITY_PUBLISHED", None)}
    if ticket is not None:
        assert (ticket.status, ticket.code) == expected[disposition]
    # Restart in a fresh runtime over the same directories.
    with Runtime(restart) as runtime:
        durable = runtime.continuity.snapshot().k1_state_digest
        names = {"acknowledged": captured[-32:], "uncertain-durable": captured[-32:], "identity-only": captured[-32:]}
        assert durable == names.get(disposition, anchored), disposition
        if not with_store:
            _refused(runtime.recover_k1, "CHECKPOINT_INVALID")                  # no bytes anywhere to resume
            assert sorted(os.listdir(tmp_path / "c")) == ["continuity.a", "continuity.b"]
            return
        recovery = runtime.recover_k1()
        if disposition == "uncertain-lost":
            assert (recovery.disposition, recovery.candidate) == ("CANDIDATE_UNRESOLVED", captured[-32:])
            return
        assert recovery.disposition == "RESUMABLE" and recovery.candidate is None
        assert recovery.envelope[-32:] == durable
        if ticket.acknowledged or disposition == "uncertain-durable":
            assert recovery.envelope == captured
            # Independently verifiable: the restored state hashes to the acknowledged identity.
            with K1State.restore(k1, recovery.envelope, max_rows=8) as restored:
                assert restored.state_digest() == ticket.identity == captured[-32:]
