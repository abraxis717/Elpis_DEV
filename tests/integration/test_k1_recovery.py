"""K1 Recovery R0 over real native K1 (docs/K1_RECOVERY_R0.md).

CHECKPOINT BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

A distinct, bounded owner keeps the complete ``(W, epoch, H, a)`` envelope of the authorized K1 state, and of at most
one newer candidate, in exactly two operator-provisioned slot files. Restart resumes exactly continuity's identity;
a crash inside a checkpointed LEARN leaves an explicit unresolved candidate that only an operator discard or adopt
settles (no silent roll forward or back). The fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE; nothing here
is a cognition claim.
"""
from __future__ import annotations

from dataclasses import replace
import os

import pytest

from elpis.continuity import ContinuityProcessDeath
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


def test_anchor_and_every_learn_keep_the_complete_authorized_state_resumable_in_two_fixed_slots(
        k1, store, tmp_path):   # noqa: F811
    footprint = _footprint(store)
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        assert runtime.recover_k1().disposition == "NOTHING_TO_RECOVER"
        runtime.anchor_cognition(state)
        for n in range(12):
            recovery = runtime.recover_k1()
            assert recovery.disposition == "RESUMABLE" and recovery.candidate is None, n
            assert recovery.authorized == state.state_digest() == runtime.continuity.snapshot().k1_state_digest
            # The complete (W, epoch, H, a) envelope, byte for byte.
            assert recovery.envelope == state.snapshot()
            runtime.run_learn(state, _learn(f"turn {n}"))
        with K1State.restore(k1, runtime.recover_k1().envelope) as twin:
            assert (twin.w(), twin.h_packed(), twin.a(), twin.epoch) == (state.w(), state.h_packed(), state.a(), state.epoch)
    # Physically bounded: the same two files, same sizes, same inodes, whatever the number of LEARNs.
    assert _footprint(store) == footprint


def test_restart_resumes_exactly_the_authorized_state(k1, store, tmp_path):   # noqa: F811
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(state)
        for n in range(3):
            runtime.run_learn(state, _learn(f"before {n}"))
        before = state.snapshot()
        answer = runtime.run_query(state, _query()).readout
    # The process ends; nothing of the K1 state survives but the checkpoint and continuity.
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.envelope == before
        with K1State.restore(k1, recovery.envelope, max_rows=8) as resumed, \
                K1State.restore(k1, before, max_rows=8) as reference:
            assert runtime.run_query(resumed, _query()).readout == answer
            result = runtime.run_learn(resumed, _learn("after"))
            run_learn(reference, _learn("after"))      # the same LEARN on the same state, outside the lineage
            assert result.state_after_digest == resumed.state_digest() == reference.state_digest()
            assert runtime.continuity.snapshot().k1_state_digest == resumed.state_digest()


def _lineage(k1, tmp_path, name, learns):   # noqa: F811
    """Another lineage's (valid, complete) checkpoint store."""
    path = _provision(k1, tmp_path / name)
    with world(k1, max_rows=8) as state, Runtime(_cfg(tmp_path, path, continuity=name + "-c")) as runtime:
        runtime.anchor_cognition(state)
        for n in range(learns):
            runtime.run_learn(state, _learn(f"{name} {n}"))
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


def _crash(k1, tmp_path, store, point):   # noqa: F811
    """Anchor, LEARN once, then die at ``point`` inside the next checkpointed LEARN. Returns (authorized envelope,
    the dead process's K1 snapshot)."""
    with world(k1, max_rows=8) as state:
        runtime = Runtime(_cfg(tmp_path, store, testing=True)).open()
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("warm"))
        authorized = state.snapshot()
        runtime._core.testing_checkpoint_crash(point)
        with pytest.raises(ContinuityProcessDeath):
            runtime.run_learn(state, _learn("dies"))
        died = state.snapshot()
        runtime.close()
    return authorized, died


@pytest.mark.parametrize("point", [1, 2], ids=["after-checkpoint", "after-native-commit"])
@pytest.mark.parametrize("resolution", ["discard", "adopt"])
def test_a_crash_inside_a_checkpointed_learn_needs_explicit_reconciliation(k1, store, tmp_path, point,
                                                                         resolution):   # noqa: F811
    authorized, died = _crash(k1, tmp_path, store, point)
    assert (died == authorized) is (point == 1)   # the native commit happened only at point 2
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        assert recovery.disposition == "CANDIDATE_UNRESOLVED" and recovery.envelope is None
        assert recovery.authorized == authorized[-32:] and recovery.authorized_present
        candidate = recovery.candidate
        assert candidate not in (None, authorized[-32:])
        if point == 2:
            assert candidate == died[-32:]
        with K1State.restore(k1, authorized, max_rows=8) as resumed:
            # No LEARN builds on an unresolved candidate; QUERY (read-only) still answers from the authority.
            _refused(lambda: runtime.run_learn(resumed, _learn("next")), "CHECKPOINT_UNRESOLVED")
            assert resumed.snapshot() == authorized
            runtime.run_query(resumed, _query())
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
            expected = None
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.candidate is None
        if expected is not None:
            assert recovery.envelope == expected
        else:
            assert recovery.envelope[-32:] == candidate and (point == 1 or recovery.envelope == died)
        with K1State.restore(k1, recovery.envelope, max_rows=8) as resumed:
            runtime.run_learn(resumed, _learn("continues"))
            assert runtime.continuity.snapshot().k1_state_digest == resumed.state_digest()


# Testing-library continuity faults: 2 write fails, 5 sync fails with the bytes lost, 6 sync fails though durable.
@pytest.mark.parametrize("failure", ["write", "sync-lost", "sync-durable"])
def test_a_failed_publication_after_a_checkpointed_commit_is_explicit(k1, store, tmp_path, failure):   # noqa: F811
    with world(k1, max_rows=8) as state:
        with Runtime(_cfg(tmp_path, store, testing=True)) as runtime:
            runtime.anchor_cognition(state)
            authorized = state.snapshot()
            runtime.continuity.testing_fault(1, {"write": 2, "sync-lost": 5, "sync-durable": 6}[failure])
            with pytest.raises(CompositionError):
                runtime.run_learn(state, _learn("commit then fail"))
            committed = state.snapshot()
            assert committed != authorized
    with Runtime(_cfg(tmp_path, store)) as runtime:
        recovery = runtime.recover_k1()
        if runtime.continuity.snapshot().k1_state_digest == authorized[-32:]:
            assert failure != "sync-durable"
            assert recovery.disposition == "CANDIDATE_UNRESOLVED" and recovery.candidate == committed[-32:]
        else:
            # The uncertain write did land: continuity authorizes the candidate, which is then simply resumable.
            assert failure == "sync-durable"
            assert recovery.disposition == "RESUMABLE" and recovery.envelope == committed


def test_an_fms_resident_state_checkpoints_the_same_complete_envelope(k1, adapter, tmp_path):   # noqa: F811
    store = _provision(k1, tmp_path / "ck")
    with _Resident(k1, adapter, tmp_path, max_rows=8) as state, Runtime(_cfg(tmp_path, store)) as runtime:
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn("resident"))
        recovery = runtime.recover_k1()
        assert recovery.disposition == "RESUMABLE" and recovery.envelope == state.snapshot()


def test_without_an_attached_store_recovery_is_refused_and_learn_is_unchanged(k1, tmp_path):   # noqa: F811
    with world(k1, max_rows=8) as state, Runtime(_config(tmp_path / "c")) as runtime:
        _refused(runtime.recover_k1, "CHECKPOINT_INVALID")
        _refused(lambda: runtime.discard_k1_candidate(bytes(32)), "CHECKPOINT_INVALID")
        runtime.anchor_cognition(state)
        runtime.run_learn(state, _learn())
        assert sorted(os.listdir(tmp_path / "c")) == ["continuity.a", "continuity.b"]
