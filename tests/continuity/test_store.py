"""Continuity: a fixed-size, crash-safe current-authority register (docs/CONTINUITY.md)."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

from elpis.continuity import (
    RECORD_SIZE,
    SLOT_NAMES,
    ContinuityError,
    ContinuitySnapshot,
    ContinuityStore,
    EvolutionAuthority,
)
from elpis.continuity import store as S
from elpis.continuity.record import decode_record, encode_record


def _d(n: int) -> bytes:
    return hashlib.sha256(b"k1-%d" % n).digest()


def _h(n: int) -> str:
    return hashlib.sha256(b"receipt-%d" % n).hexdigest()


class _Death(BaseException):
    """Simulated process death: no in-process recovery runs."""


def _code(fn):
    with pytest.raises(ContinuityError) as info:
        fn()
    return info.value.code


def _files(path):
    return sorted((p.name, p.stat().st_size) for p in path.iterdir())


# -- basic law -------------------------------------------------------------------------------

def test_new_store_is_unanchored_generation_one(tmp_path):
    with ContinuityStore(tmp_path / "c") as store:
        snap = store.snapshot()
        assert snap.generation == 1 and not snap.anchored and snap.k1_state_digest is None
        assert snap.evolution == EvolutionAuthority(0, "0" * 64)
    assert _files(tmp_path / "c") == [(SLOT_NAMES[0], RECORD_SIZE), (SLOT_NAMES[1], RECORD_SIZE)]


def test_explicit_anchor_then_transitions_and_restart(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        assert _code(lambda: store.commit_cognition_transition(_d(0), _d(1))) == "CONTINUITY_UNANCHORED"
        store.anchor_cognition(_d(0))
        assert _code(lambda: store.anchor_cognition(_d(9))) == "CONTINUITY_ALREADY_ANCHORED"
        store.commit_cognition_transition(_d(0), _d(1))
        before = store.snapshot()
        # A transition that does not start from the expected identity writes nothing.
        assert _code(lambda: store.commit_cognition_transition(_d(0), _d(2))) == "CONTINUITY_STATE_MISMATCH"
        assert store.snapshot() == before
    with ContinuityStore(path) as store:
        assert store.snapshot().k1_state_digest == _d(1) and store.snapshot().generation == 3


def test_evolution_authority_advances_and_refuses_stale_expectations(tmp_path):
    with ContinuityStore(tmp_path / "c") as store:
        a0 = store.snapshot().evolution
        assert _code(lambda: store.commit_evolution_transition(a0, _h(1))) == "CONTINUITY_EVOLUTION_NOT_PENDING"
        pending = store.reserve_evolution_assertion(a0, _h(100)).evolution
        assert pending == EvolutionAuthority(0, a0.head, _h(100))
        assert pending.digest != a0.digest
        impostor = EvolutionAuthority(pending.revision, pending.head, _h(999))
        assert _code(lambda: store.commit_evolution_transition(impostor, _h(1))) == "CONTINUITY_AUTHORITY_MISMATCH"
        assert _code(lambda: store.reserve_evolution_assertion(pending, _h(101))) == "CONTINUITY_EVOLUTION_PENDING"
        assert _code(lambda: store.commit_evolution_transition(pending, "0" * 64)) == "CONTINUITY_INVALID"
        assert _code(lambda: store.commit_evolution_transition(pending, "XYZ")) == "CONTINUITY_INVALID"
        a1 = store.commit_evolution_transition(pending, _h(1)).evolution
        assert a1 == EvolutionAuthority(1, _h(1)) and a1.digest != a0.digest
        assert _code(lambda: store.commit_evolution_transition(pending, _h(1))) == "CONTINUITY_AUTHORITY_MISMATCH"
        assert _code(lambda: store.reserve_evolution_assertion(a0, _h(2))) == "CONTINUITY_AUTHORITY_MISMATCH"
        assert store.snapshot().evolution == a1


def test_snapshot_digest_is_record_content_identity(tmp_path):
    with ContinuityStore(tmp_path / "c") as store:
        first = store.snapshot()
        second = store.anchor_cognition(_d(0))
        assert first.digest != second.digest
        assert decode_record(encode_record(second)) == second


# -- bounds ----------------------------------------------------------------------------------

def test_storage_is_two_fixed_slots_whatever_the_number_of_updates(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))
        for n in range(1, 2001):
            store.commit_cognition_transition(_d(n - 1), _d(n))
            if n % 7 == 0:
                pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(n)).evolution
                store.commit_evolution_transition(pending, _h(n))
            assert _files(path) == [(SLOT_NAMES[0], RECORD_SIZE), (SLOT_NAMES[1], RECORD_SIZE)]
        assert store.snapshot().generation > 2000
    assert sum(size for _, size in _files(path)) == 2 * RECORD_SIZE == 352


def test_resident_state_does_not_grow(tmp_path):
    with ContinuityStore(tmp_path / "c") as store:
        store.anchor_cognition(_d(0))

        def footprint():
            values = vars(store)
            assert not any(isinstance(v, (list, dict, set, bytearray)) for v in values.values())
            return sorted((k, sys.getsizeof(v)) for k, v in values.items() if k != "_current")

        before = footprint()
        for n in range(1, 501):
            store.commit_cognition_transition(_d(n - 1), _d(n))
            pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(n)).evolution
            assert not any(isinstance(v, (list, dict, set, bytearray)) for v in vars(pending).values())
            store.commit_evolution_transition(pending, _h(n))
        assert footprint() == before
        assert sorted(vars(store)) == ["_current", "_fds", "_lock", "_poisoned", "_slot", "directory"]


def _write_slots(path, a: bytes, b: bytes):
    path.mkdir()
    (path / SLOT_NAMES[0]).write_bytes(a)
    (path / SLOT_NAMES[1]).write_bytes(b)


@pytest.mark.parametrize("generation", [10, 1_000_000, (1 << 63) - 1])
def test_restart_work_is_two_fixed_reads_whatever_the_lifetime(tmp_path, monkeypatch, generation):
    # The protocol state of a long-lived store is constructed directly: no
    # million fsyncs are needed to show that restart inspects exactly two slots.
    path = tmp_path / "c"
    current = ContinuitySnapshot(generation, _d(generation), EvolutionAuthority(generation // 3, _h(generation)))
    older = ContinuitySnapshot(generation - 1, _d(generation - 1), current.evolution)
    _write_slots(path, encode_record(older), encode_record(current))
    reads = []
    real = os.pread
    monkeypatch.setattr(S.os, "pread", lambda fd, n, off: reads.append(n) or real(fd, n, off))
    with ContinuityStore(path) as store:
        assert store.snapshot() == current
    assert reads == [RECORD_SIZE, RECORD_SIZE]


# -- crash matrix --------------------------------------------------------------------------

PUBLISH_STEPS = ("publish.begin", "publish.written", "publish.synced")


@pytest.mark.parametrize("step", PUBLISH_STEPS)
def test_death_at_each_publication_step_yields_previous_or_next(tmp_path, monkeypatch, step):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))
        previous = store.snapshot()
    store = ContinuityStore(path).open()

    def die(name):
        if name == step:
            raise _Death(name)

    monkeypatch.setattr(S, "_crash_point", die)
    with pytest.raises(_Death):
        store.commit_cognition_transition(_d(0), _d(1))
    store.close()  # the process is gone; descriptors close
    monkeypatch.undo()
    with ContinuityStore(path) as reopened:
        snap = reopened.snapshot()
    # Process death after the write leaves the complete record in the page
    # cache; before it, the target slot was never touched.
    expected_next = ContinuitySnapshot(previous.generation + 1, _d(1), previous.evolution)
    assert snap == (previous if step == "publish.begin" else expected_next)


@pytest.mark.parametrize("cut", [1, 8, 16, 63, 64, 104, RECORD_SIZE - 1])
def test_torn_write_never_destroys_the_previous_authority(tmp_path, monkeypatch, cut):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))
        store.commit_cognition_transition(_d(0), _d(1))
        previous = store.snapshot()
    store = ContinuityStore(path).open()
    real = os.pwrite

    def torn(fd, data, offset):
        real(fd, bytes(data[:cut]), offset)
        raise _Death("torn")

    monkeypatch.setattr(S.os, "pwrite", torn)
    with pytest.raises(_Death):
        store.commit_cognition_transition(_d(1), _d(2))
    store.close()
    monkeypatch.undo()
    with ContinuityStore(path) as reopened:
        assert reopened.snapshot() == previous
        # And the store keeps publishing into the torn slot afterwards.
        assert reopened.commit_cognition_transition(_d(1), _d(2)).k1_state_digest == _d(2)


def test_write_failure_is_refused_and_the_previous_authority_stays_in_force(tmp_path, monkeypatch):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))
        previous = store.snapshot()

        def fail(fd, data, offset):
            raise OSError(28, "ENOSPC")

        monkeypatch.setattr(S.os, "pwrite", fail)
        assert _code(lambda: store.commit_cognition_transition(_d(0), _d(1))) == "CONTINUITY_PUBLICATION_REFUSED"
        monkeypatch.undo()
        assert store.snapshot() == previous
        assert store.commit_cognition_transition(_d(0), _d(1)).generation == previous.generation + 1


def test_sync_failure_is_uncertain_and_poisons_until_reopen(tmp_path, monkeypatch):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))
        previous = store.snapshot()
    store = ContinuityStore(path).open()

    def fail(fd):
        raise OSError(5, "EIO")

    monkeypatch.setattr(S.os, "fdatasync", fail, raising=False)
    monkeypatch.setattr(S.os, "fsync", fail)
    assert _code(lambda: store.commit_cognition_transition(_d(0), _d(1))) == "CONTINUITY_PUBLICATION_UNCERTAIN"
    assert _code(store.snapshot) == "CONTINUITY_PUBLICATION_UNCERTAIN"
    monkeypatch.undo()
    with ContinuityStore(path) as reopened:
        snap = reopened.snapshot()
    assert snap in (previous, ContinuitySnapshot(previous.generation + 1, _d(1), previous.evolution))


INIT_STEPS = ("init.continuity.b.written", "init.continuity.a.written", "init.continuity.b.renamed",
              "init.continuity.a.renamed", "init.dirsync")


@pytest.mark.parametrize("step", INIT_STEPS)
def test_death_during_initialization_restarts_or_completes_it(tmp_path, monkeypatch, step):
    path = tmp_path / "c"

    def die(name):
        if name == step:
            raise _Death(name)

    monkeypatch.setattr(S, "_crash_point", die)
    with pytest.raises(_Death):
        ContinuityStore(path).open()
    monkeypatch.undo()
    with ContinuityStore(path) as store:
        assert store.snapshot() == ContinuitySnapshot(1, None, EvolutionAuthority(0, "0" * 64))
    assert _files(path) == [(SLOT_NAMES[0], RECORD_SIZE), (SLOT_NAMES[1], RECORD_SIZE)]


# -- fail closed -------------------------------------------------------------------------------

def test_both_slots_invalid_is_corrupt(tmp_path):
    path = tmp_path / "c"
    _write_slots(path, b"\x01" * RECORD_SIZE, bytes(RECORD_SIZE))
    assert _code(ContinuityStore(path).open) == "CONTINUITY_CORRUPT"


def test_equal_generations_are_ambiguous_and_corrupt(tmp_path):
    path = tmp_path / "c"
    one = ContinuitySnapshot(5, _d(1), EvolutionAuthority(0, "0" * 64))
    two = ContinuitySnapshot(5, _d(2), EvolutionAuthority(0, "0" * 64))
    _write_slots(path, encode_record(one), encode_record(two))
    assert _code(ContinuityStore(path).open) == "CONTINUITY_CORRUPT"


@pytest.mark.parametrize("mutate", [
    lambda p: (p / SLOT_NAMES[1]).unlink(),
    lambda p: (p / SLOT_NAMES[0]).write_bytes(b"x" * (RECORD_SIZE + 1)),
    lambda p: (p / "stray").write_bytes(b""),
])
def test_damaged_layouts_fail_closed(tmp_path, mutate):
    path = tmp_path / "c"
    ContinuityStore(path).open().close()
    mutate(path)
    assert _code(ContinuityStore(path).open) == "CONTINUITY_CORRUPT"


def test_deleting_the_first_slot_of_a_published_store_fails_closed(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))          # published into continuity.b
    (path / SLOT_NAMES[0]).unlink()
    assert _code(ContinuityStore(path).open) == "CONTINUITY_CORRUPT"
    assert sorted(p.name for p in path.iterdir()) == [SLOT_NAMES[1]]


def test_a_flipped_bit_in_the_current_slot_falls_back_or_fails_closed(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        store.anchor_cognition(_d(0))      # generation 2 in slot b; slot a holds generation 1
    raw = bytearray((path / SLOT_NAMES[1]).read_bytes())
    raw[40] ^= 0x01
    (path / SLOT_NAMES[1]).write_bytes(bytes(raw))
    # Integrity evidence only: the untouched older record is the authority left.
    with ContinuityStore(path) as store:
        assert store.snapshot().generation == 1 and not store.snapshot().anchored


@pytest.mark.parametrize("name", ["MANIFEST", "events.log", "g0000000000000001.seg", "checkpoint.bin", "LOCK"])
def test_retired_receipt_history_layout_is_refused_not_read(tmp_path, name):
    path = tmp_path / "c"
    path.mkdir()
    (path / name).write_bytes(b"legacy")
    assert _code(ContinuityStore(path).open) == "CONTINUITY_LEGACY_STORAGE"
    assert sorted(p.name for p in path.iterdir()) == [name]


def test_second_owner_is_refused(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path):
        assert _code(ContinuityStore(path).open) == "CONTINUITY_LOCKED"


def test_relative_path_and_bad_digests_are_refused(tmp_path):
    assert _code(lambda: ContinuityStore("relative")) == "CONTINUITY_PATH"
    with ContinuityStore(tmp_path / "c") as store:
        assert _code(lambda: store.anchor_cognition(b"short")) == "CONTINUITY_INVALID"
        assert _code(lambda: store.commit_cognition_transition(_d(0), "x" * 32)) == "CONTINUITY_INVALID"
    assert _code(ContinuityStore(tmp_path / "c").snapshot) == "CONTINUITY_UNINITIALIZED"


def test_the_component_has_no_history_or_ecs_surface():
    import elpis.continuity as C

    public = {n for n in dir(ContinuityStore) if not n.startswith("_")}
    assert public == {"open", "close", "snapshot", "anchor_cognition", "commit_cognition_transition",
                      "reserve_evolution_assertion", "commit_evolution_transition"}
    for name in ("record", "records", "events", "projection", "topology", "entity_port", "propose",
                 "run_until_quiescent", "scheduler", "compact"):
        assert not hasattr(ContinuityStore, name) and name not in C.__all__

def test_many_evolution_reservations_have_constant_state_and_restart_work(tmp_path, monkeypatch):
    path = tmp_path / "c"

    def shape(value):
        # Traverse retained object state, not just the store's immediate attributes.
        assert not isinstance(value, (list, dict, set, bytearray))
        if hasattr(value, "__dict__"):
            return tuple((key, shape(item)) for key, item in vars(value).items())
        if isinstance(value, tuple):
            return tuple(shape(item) for item in value)
        return type(value), sys.getsizeof(value)

    with ContinuityStore(path) as store:
        first_idle = first_pending = None
        for n in range(1000):
            pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(n)).evolution
            if first_pending is None:
                first_pending = shape(store)
            assert shape(store) == first_pending
            store.commit_evolution_transition(pending, _h(n))
            if first_idle is None:
                first_idle = shape(store)
            assert shape(store) == first_idle
            assert _files(path) == [(SLOT_NAMES[0], 176), (SLOT_NAMES[1], 176)]
        pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(1000)).evolution
    reads = []
    real = os.pread
    monkeypatch.setattr(S.os, "pread", lambda fd, n, off: reads.append(n) or real(fd, n, off))
    for _ in range(3):
        with ContinuityStore(path) as store:
            assert store.snapshot().evolution == pending
        assert reads == [RECORD_SIZE, RECORD_SIZE]
        reads.clear()
    assert sum(size for _, size in _files(path)) == 352


@pytest.mark.parametrize("bad", [None, b"x" * 32, "", "F" * 64, "x" * 64])
def test_invalid_reservation_identity_never_changes_authority(tmp_path, bad):
    with ContinuityStore(tmp_path / "c") as store:
        before = store.snapshot()
        assert _code(lambda: store.reserve_evolution_assertion(before.evolution, bad)) == "CONTINUITY_INVALID"
        assert store.snapshot() == before


def test_pending_record_schema_is_explicit_and_fixed_size():
    from elpis.continuity import record as R
    idle = ContinuitySnapshot(1, None, EvolutionAuthority(0, "0" * 64))
    pending = ContinuitySnapshot(2, None, EvolutionAuthority(0, "0" * 64, _h(1)))
    for snap, state in ((idle, 0), (pending, 1)):
        raw = encode_record(snap)
        assert len(raw) == RECORD_SIZE == 176
        assert raw[:8] == b"ELPCONT\x02" and raw[8:10] == b"\x00\x02"
        assert raw[104] == state and raw[105:112] == bytes(7)
        assert raw[112:144] == (bytes.fromhex(_h(1)) if state else bytes(32))
        assert decode_record(raw) == snap
    for offset, byte in ((104, 2), (105, 1), (112, 1)):
        raw = bytearray(encode_record(idle))
        raw[offset] = byte
        raw[-32:] = R._checksum(bytes(raw[:-32]))
        assert _code(lambda: decode_record(bytes(raw))) == "CONTINUITY_CORRUPT"


def test_old_record_format_is_refused_without_reinterpretation(tmp_path):
    import struct
    body = struct.pack(">8sH6sQB7s32sQ32s", b"ELPCONT\x01", 1, bytes(6),
                       1, 0, bytes(7), bytes(32), 0, bytes(32))
    old = body + hashlib.sha256(b"elpis.continuity.register.v1\x00" + body).digest()
    path = tmp_path / "old"
    _write_slots(path, old, bytes(136))
    assert _code(ContinuityStore(path).open) == "CONTINUITY_CORRUPT"
    assert (path / SLOT_NAMES[0]).read_bytes() == old
    assert (path / SLOT_NAMES[1]).read_bytes() == bytes(136)


def test_reservation_refuses_counter_exhaustion_before_execution(tmp_path):
    path = tmp_path / "c"
    maximum = (1 << 63) - 1
    idle = ContinuitySnapshot(maximum - 1, None, EvolutionAuthority(0, "0" * 64))
    _write_slots(path, encode_record(idle), bytes(RECORD_SIZE))
    with ContinuityStore(path) as store:
        assert _code(lambda: store.reserve_evolution_assertion(idle.evolution, _h(1))) == "CONTINUITY_EXHAUSTED"
        assert store.snapshot() == idle


def test_cognition_publications_preserve_pending_identity(tmp_path):
    path = tmp_path / "c"
    with ContinuityStore(path) as store:
        pending = store.reserve_evolution_assertion(store.snapshot().evolution, _h(1)).evolution
        store.anchor_cognition(_d(0))
        for n in range(1, 5):
            assert store.commit_cognition_transition(_d(n - 1), _d(n)).evolution == pending
    with ContinuityStore(path) as store:
        assert store.snapshot().evolution == pending
        assert store.snapshot().k1_state_digest == _d(4)
