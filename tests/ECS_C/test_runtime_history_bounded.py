"""Bounded runtime history: finite disk, resident state and restart work.

Every claim the runtime history makes about bounds is demonstrated here with a
small finite policy and many compactions:

* the directory never exceeds ``policy.max_directory_bytes`` (measured at
  every generation-switch step), the active segment never exceeds its bound,
  and only one generation exists between operations;
* the resident record window never exceeds the segment's receipt capacity;
* restart replays only the active segment (instrumented), never the lifetime;
* the global history continues exactly across compaction (index, clock,
  chain, roots, watermarks, native global count);
* retention is explicit (floor, refusals, retained-window projection);
* a crash at every generation-switch step reopens exactly one consistent
  generation, never losing the last known-good one;
* legacy single-log histories migrate once, deterministically.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from elpis.ECS_C import generations
from elpis.ECS_C.errors import RetentionFloorError
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.projection import ProjectionRequest
from elpis.ECS_C.replay import StreamReplay
from elpis.runtime.history import (
    HISTORY_GENESIS_LABEL,
    ROLES,
    HistoryError,
    ReceiptHistory,
    ReceiptRecord,
    RuntimeHistoryPolicy,
)
from elpis.runtime.native_history import SEGMENT_FULL, NativeHistoryError

from ..conftest import require_native_library

KiB = 1024


def _library() -> Path:
    return require_native_library("elpis_ecsc_history")


def _policy(**overrides) -> RuntimeHistoryPolicy:
    values = dict(max_segment_bytes=64 * KiB, max_segment_events=64, max_checkpoint_bytes=16 * KiB)
    values.update(overrides)
    probe = RuntimeHistoryPolicy(**values, max_directory_bytes=1 << 40)
    return RuntimeHistoryPolicy(**values, max_directory_bytes=probe.required_directory_bytes)


SMALL = _policy()


def _receipt(n: int, subsystem: str = "pipeline") -> ReceiptRecord:
    return ReceiptRecord.of(subsystem, "bounded.fixture", format(n, "064x"), n=str(n), pad="p" * 200)


def _artifacts(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


def _assert_one_generation(path: Path, generation: int) -> None:
    assert _artifacts(path) == sorted([
        "LOCK", "MANIFEST", generations.checkpoint_name(generation), generations.segment_name(generation),
    ])


def _open(path: Path, policy=SMALL, native=True) -> ReceiptHistory:
    return ReceiptHistory(path, native_library=_library() if native else None, policy=policy).open()


def _die(history: ReceiptHistory) -> None:
    """Simulated process death: descriptors and flocks go, no cleanup logic runs."""
    if history._native is not None:
        history._native.close()
        history._native = None
    if history._kernel is not None:
        history._kernel.close()
    history._lock.release()


class _Crash(BaseException):
    pass


# -- policy ----------------------------------------------------------------------------

def test_policy_is_finite_with_finite_defaults():
    default = RuntimeHistoryPolicy()
    assert default.max_segment_bytes == 8 * 1024 * KiB
    assert default.max_directory_bytes >= default.required_directory_bytes
    for bad in (None, 0, -1, True, 1.5, 1 << 60):
        with pytest.raises(HistoryError, match="HISTORY_POLICY"):
            RuntimeHistoryPolicy(max_segment_bytes=bad)
        with pytest.raises(HistoryError, match="HISTORY_POLICY"):
            RuntimeHistoryPolicy(max_directory_bytes=bad)
    with pytest.raises(HistoryError, match="HISTORY_POLICY"):
        RuntimeHistoryPolicy(max_segment_bytes=1 * KiB)
    with pytest.raises(HistoryError, match="HISTORY_POLICY"):
        RuntimeHistoryPolicy(max_segment_bytes=64 * KiB, max_segment_events=64,
                             max_checkpoint_bytes=16 * KiB,
                             max_directory_bytes=SMALL.required_directory_bytes - 1)


def test_runtime_config_carries_a_finite_policy(tmp_path):
    from elpis.runtime import Runtime, RuntimeConfig
    config = RuntimeConfig(tmp_path / "rt", history_native_library=_library())
    assert config.history_policy == RuntimeHistoryPolicy()
    with pytest.raises(HistoryError, match="HISTORY_POLICY"):
        RuntimeConfig(tmp_path / "rt", history_policy=None)
    small = RuntimeConfig(tmp_path / "rt", history_native_library=_library(), history_policy=SMALL)
    with Runtime(small) as runtime:
        assert runtime.history.policy is SMALL


# -- bounded disk / resident / continuation -----------------------------------------------

def test_disk_and_resident_state_stay_bounded_across_repeated_compactions(tmp_path, monkeypatch):
    storage = tmp_path / "h"
    peaks = []
    history = _open(storage)

    def probe(step):
        peaks.append((step, history._store.usage_bytes()))

    monkeypatch.setattr(generations, "_crash_point", probe)
    try:
        receipts = []
        indices = []
        for n in range(1, 241):
            item = history.record(_receipt(n))
            receipts.append(item)
            indices.append(item.event_index)
            # Hard bounds hold after every committed receipt.
            assert history.storage_bytes() <= SMALL.max_directory_bytes
            assert history._native.segment_bytes <= SMALL.max_segment_bytes
            assert history._native.segment_frames <= SMALL.max_segment_events
            assert len(history._records) <= SMALL.max_segment_events // 2
            assert len(history._record_index) == len(history._records)
            # The native owner reports the GLOBAL count, never the local one.
            assert history._native.event_count == 12 + 2 * n
            assert history._native.segment_base == history.retention_floor
        assert history.generation >= 6
        _assert_one_generation(storage, history.generation)
        # Global numbering is never reset or renumbered by compaction.
        assert indices == [12 + 2 * i for i in range(240)]
        assert history.receipt_count == 240
        # Peak usage DURING every switch step stayed within the budget.
        assert peaks and max(b for _, b in peaks) <= SMALL.max_directory_bytes
        assert {s for s, _ in peaks} >= {"checkpoint.write", "manifest.replace", "cleanup.segment"}
        native_root = history.state_root
        floor = history.retention_floor
    finally:
        history.close()

    # Canonical Python replay (replay-only) verifies the native-written history.
    replay = _open(storage, native=False)
    try:
        assert replay.state_root == native_root
        assert replay.retention_floor == floor
        state = replay._kernel.state
        assert state.logical_clock == replay.event_count == 12 + 2 * 240
        assert state.watermarks.as_sorted_dict()[replay._ids["pipeline"]] == 240
        segment = replay._kernel.retained_events()
        assert segment[0]["event_index"] == floor
        assert segment[0]["prev_event_digest"] == replay._base.head_event_digest
        assert segment[0]["before_state_root"] == replay._base.state_root_digest
        assert replay.retained_records() == tuple(receipts[-len(replay.retained_records()):])
    finally:
        replay.close()


def test_restart_replays_only_the_bounded_segment(tmp_path, monkeypatch):
    storage = tmp_path / "h"
    history = _open(storage)
    try:
        for n in range(1, 301):
            history.record(_receipt(n))
        lifetime = history.event_count
        frames = history._native.segment_frames
    finally:
        history.close()
    assert lifetime == 612 and frames < SMALL.max_segment_events

    applied = []
    original = StreamReplay.apply

    def counting(self, event):
        applied.append(event["event_index"])
        return original(self, event)

    def forbidden(*_a, **_k):
        raise AssertionError("LIFETIME_REPLAY_ON_RESTART")

    monkeypatch.setattr(StreamReplay, "apply", counting)
    monkeypatch.setattr("elpis.ECS_C.kernel.replay_with_checkpoint", forbidden)
    monkeypatch.setattr("elpis.ECS_C.kernel.replay_from_events", forbidden)

    for native in (True, False):
        applied.clear()
        reopened = _open(storage, native=native)
        try:
            assert reopened.event_count == lifetime
            assert len(applied) == frames <= SMALL.max_segment_events
            assert applied == list(range(lifetime - frames, lifetime))
        finally:
            reopened.close()


def test_retention_floor_is_explicit_and_refusals_are_stable(tmp_path):
    storage = tmp_path / "h"
    history = _open(storage)
    try:
        first = history.record(_receipt(1))
        assert history.retention_floor == 0
        assert history.records() == (first,)
        n = 2
        while history.retention_floor == 0:
            history.record(_receipt(n))
            n += 1
        floor = history.retention_floor
        assert floor > 0

        with pytest.raises(HistoryError) as caught:
            history.records()
        assert caught.value.code == "HISTORY_BELOW_RETENTION_FLOOR"
        retained = history.retained_records()
        assert all(item.event_index >= floor for item in retained)

        # Duplicate detection is exact WITHIN the retained window only.
        recent = history.record(_receipt(n))
        assert history.record(_receipt(n)) == recent
        again = history.record(_receipt(1))  # retired below the floor
        assert again.event_index > recent.event_index and again != first

        # Default projection = retained window with the floor bound in.
        projected = history.projection()
        assert projected.schema == "ecs.context-projection.retained.v1"
        assert projected.source.retention_floor == floor
        assert projected.request.clock_min == floor + 1
        assert projected.source.event_count == history.event_count
        assert [r.event_index for r in projected.records] == [
            item.event_index for item in history.retained_records()][:len(projected.records)]
        # Explicit requests reaching into the retired prefix fail closed.
        for clock_min in (0, 1, floor):
            with pytest.raises(HistoryError) as caught:
                history.projection(ProjectionRequest(clock_min=clock_min))
            assert caught.value.code == "HISTORY_BELOW_RETENTION_FLOOR"
        assert history.projection(ProjectionRequest(clock_min=floor + 1)).source.retention_floor == floor
        # Native ownership was reacquired after each read window.
        assert history._native is not None
        history.record(_receipt(10_000))
    finally:
        history.close()

    replay = _open(storage, native=False)
    try:
        with pytest.raises(RetentionFloorError):
            replay._kernel.events()
        with pytest.raises(RetentionFloorError, match="TOPOLOGY_REQUIRES_COMPLETE_HISTORY"):
            replay._kernel.topology_projection()
    finally:
        replay.close()


def test_uncompacted_history_projection_is_unchanged(tmp_path):
    history = _open(tmp_path / "h")
    try:
        history.record(_receipt(1))
        projected = history.projection()
        assert projected.schema == "ecs.context-projection.v1"
        assert projected.request.clock_min == 0
        assert len(history.records()) == 1
    finally:
        history.close()


# -- capacity disposition and retry ------------------------------------------------------

def test_segment_full_compacts_once_and_retries_the_exact_receipt_once(tmp_path, monkeypatch):
    history = _open(tmp_path / "h")
    try:
        history.record(_receipt(1))
        calls = []

        def always_full(record):
            calls.append(record)
            raise NativeHistoryError(SEGMENT_FULL, "RECORD:SEGMENT_FULL")

        monkeypatch.setattr(history, "_native_record", always_full)
        generation = history.generation
        with pytest.raises(HistoryError) as caught:
            history.record(_receipt(2))
        assert caught.value.code == "HISTORY_STORAGE_CAPACITY"
        assert calls == [_receipt(2), _receipt(2)]  # original + exactly one retry
        assert history.generation == generation + 1
    finally:
        history.close()


def test_other_native_failures_are_not_retried(tmp_path, monkeypatch):
    history = _open(tmp_path / "h")
    try:
        calls = []

        def uncertain(record):
            calls.append(record)
            raise NativeHistoryError(-8, "RECORD:APPEND_UNCERTAIN")

        monkeypatch.setattr(history, "_native_record", uncertain)
        with pytest.raises(HistoryError) as caught:
            history.record(_receipt(1))
        assert caught.value.code == "NATIVE_HISTORY_RECORD"
        assert caught.value.__cause__.code == -8
        assert len(calls) == 1 and history.generation == 1
    finally:
        history.close()


def test_native_session_refuses_inconsistent_global_base_and_local_counts(tmp_path):
    from elpis.runtime.native_history import SEGMENT_MISMATCH, NativeHistorySession
    history = _open(tmp_path / "h", native=False)
    try:
        kernel = history._kernel
        state, snap = kernel.state, kernel.snapshot()
        segment = Path(kernel.log_path)
        kernel.close()
        for base in (1, snap["event_count"] + 1):
            with pytest.raises(NativeHistoryError) as caught:
                NativeHistorySession(_library(), segment, state, snap["event_digest"], snap["event_count"],
                                     segment_base=base, max_segment_bytes=SMALL.max_segment_bytes,
                                     max_segment_frames=SMALL.max_segment_events)
            assert caught.value.code == SEGMENT_MISMATCH
        with pytest.raises(NativeHistoryError) as caught:
            NativeHistorySession(_library(), segment, state, snap["event_digest"], snap["event_count"],
                                 segment_base=0, max_segment_bytes=0, max_segment_frames=64)
        assert caught.value.code == -1
    finally:
        history.close()


# -- crash matrix --------------------------------------------------------------------------

SWITCH_STEPS = (
    "checkpoint.write", "checkpoint.fsync", "checkpoint.rename",
    "segment.create", "segment.fsync", "artifacts.dirsync",
    "manifest.write.data", "manifest.write", "manifest.replace", "manifest.dirsync",
    "cleanup.checkpoint", "cleanup.segment",
)
PUBLISHED_AT = SWITCH_STEPS.index("manifest.replace")


@pytest.mark.parametrize("step", SWITCH_STEPS)
def test_crash_at_every_switch_step_reopens_one_consistent_generation(tmp_path, monkeypatch, step):
    storage = tmp_path / "h"
    history = _open(storage)
    recorded = []
    usage = []

    def crash(name):
        usage.append(history._store.usage_bytes())
        if name == step:
            raise _Crash(name)

    monkeypatch.setattr(generations, "_crash_point", crash)
    n = 0
    try:
        with pytest.raises(_Crash):
            while True:
                n += 1
                recorded.append(history.record(_receipt(n)))
    finally:
        _die(history)
    monkeypatch.undo()
    assert max(usage) <= SMALL.max_directory_bytes
    assert history.storage_bytes() <= SMALL.max_directory_bytes

    expected_generation = 2 if SWITCH_STEPS.index(step) >= PUBLISHED_AT else 1
    reopened = _open(storage)
    try:
        assert reopened.generation == expected_generation
        _assert_one_generation(storage, expected_generation)
        assert reopened.receipt_count == len(recorded)
        if expected_generation == 1:
            assert reopened.records() == tuple(recorded)
        else:
            assert reopened.retention_floor == 12 + 2 * len(recorded)
        # The interrupted receipt was never appended; it records now, with
        # the next global index.
        item = reopened.record(_receipt(n))
        assert item.event_index == 12 + 2 * len(recorded)
        root = reopened.state_root
    finally:
        reopened.close()
    replay = _open(storage, native=False)
    try:
        assert replay.state_root == root
    finally:
        replay.close()


@pytest.mark.parametrize("step", ["checkpoint.fsync", "segment.create", "artifacts.dirsync", "manifest.write"])
def test_failed_compaction_never_deletes_the_last_known_good_generation(tmp_path, monkeypatch, step):
    storage = tmp_path / "h"
    history = _open(storage)
    recorded = []

    def fail(name):
        if name == step:
            raise OSError(28, "simulated ENOSPC")

    monkeypatch.setattr(generations, "_crash_point", fail)
    try:
        with pytest.raises(HistoryError) as caught:
            n = 0
            while True:
                n += 1
                recorded.append(history.record(_receipt(n)))
        assert caught.value.code == "HISTORY_COMPACTION_FAILED"
        # Old generation intact, new generation debris removed in-process.
        _assert_one_generation(storage, 1)
        # Fail-stop: no further writes through this handle.
        with pytest.raises(HistoryError, match="HISTORY_CLOSED"):
            history.record(_receipt(n))
    finally:
        history.close()
    monkeypatch.undo()
    reopened = _open(storage)
    try:
        assert reopened.generation == 1 and reopened.records() == tuple(recorded)
    finally:
        reopened.close()


def test_failed_cleanup_after_publication_keeps_the_new_generation(tmp_path, monkeypatch):
    storage = tmp_path / "h"
    history = _open(storage)
    recorded = []

    def fail(name):
        if name == "cleanup.checkpoint":
            raise OSError(5, "simulated EIO")

    monkeypatch.setattr(generations, "_crash_point", fail)
    try:
        with pytest.raises(HistoryError) as caught:
            n = 0
            while True:
                n += 1
                recorded.append(history.record(_receipt(n)))
        assert caught.value.code == "HISTORY_COMPACTION_FAILED"
    finally:
        history.close()
    monkeypatch.undo()
    assert history.storage_bytes() <= SMALL.max_directory_bytes
    reopened = _open(storage)
    try:
        assert reopened.generation == 2
        _assert_one_generation(storage, 2)
        assert reopened.receipt_count == len(recorded)
    finally:
        reopened.close()


def test_corrupt_checkpoint_or_manifest_fails_closed(tmp_path):
    storage = tmp_path / "h"
    history = _open(storage)
    try:
        history.record(_receipt(1))
        generation = history.generation
    finally:
        history.close()
    checkpoint = storage / generations.checkpoint_name(generation)
    good = checkpoint.read_bytes()
    bad = bytearray(good)
    bad[len(bad) // 2] ^= 0x01
    checkpoint.write_bytes(bytes(bad))
    with pytest.raises(HistoryError) as caught:
        _open(storage)
    assert caught.value.code == "HISTORY_CHECKPOINT_INVALID"
    checkpoint.write_bytes(good)

    manifest = storage / "MANIFEST"
    good = manifest.read_bytes()
    manifest.write_bytes(good[:-3] + b"}}}")
    with pytest.raises(HistoryError) as caught:
        _open(storage)
    assert caught.value.code == "HISTORY_GENERATION_MISMATCH"
    manifest.write_bytes(good)

    (storage / "stray.bin").write_bytes(b"x")
    with pytest.raises(HistoryError) as caught:
        _open(storage)
    assert caught.value.code == "HISTORY_GENERATION_MISMATCH"
    (storage / "stray.bin").unlink()
    _open(storage).close()


def test_directory_budget_is_admitted_before_writing(tmp_path):
    from elpis.ECS_C.errors import StorageCapacityError
    storage = tmp_path / "h"
    history = _open(storage)
    try:
        store = history._store
        before = _artifacts(storage)
        tight = generations.GenerationStore(storage, genesis_label=HISTORY_GENESIS_LABEL,
                                            mailbox_capacity=16, max_checkpoint_bytes=16 * KiB,
                                            max_directory_bytes=store.usage_bytes() + 10)
        with pytest.raises(StorageCapacityError, match="HISTORY_STORAGE_CAPACITY"):
            tight.publish(history._base, previous=history._manifest)
        assert _artifacts(storage) == before
    finally:
        history.close()


def test_second_owner_is_refused(tmp_path):
    storage = tmp_path / "h"
    first = _open(storage)
    try:
        with pytest.raises(HistoryError) as caught:
            _open(storage)
        assert caught.value.code == "HISTORY_LOCKED"
    finally:
        first.close()


# -- legacy migration --------------------------------------------------------------------

def _legacy(storage: Path, records) -> int:
    kernel = Kernel(str(storage), genesis_label=HISTORY_GENESIS_LABEL)
    kernel.open()
    try:
        ids = {role: kernel.found_entity(role) for role in ROLES}
        kernel.run_until_quiescent()
        for record in records:
            kernel.entity_port(ids[record.subsystem]).propose(ids["history"], record.payload())
            kernel.run_until_quiescent()
        kernel.checkpoint()
        return kernel.snapshot()["event_count"]
    finally:
        kernel.close()


def _lineage(count):
    state = format(1, "064x")
    out = [ReceiptRecord.of("ecs_g", "cognition.anchor", state, mechanism="1", state=state)]
    for i in range(count):
        after = format(i + 2, "064x")
        out.append(ReceiptRecord.of("ecs_g", "cognition.turn", format(0xabc + i, "064x"),
                                    mechanism="1", state_before=state, state_after=after))
        state = after
    return out, state


def test_small_legacy_history_migrates_once_verbatim(tmp_path):
    storage = tmp_path / "legacy"
    records = [_receipt(i) for i in range(3)]
    count = _legacy(storage, records)
    legacy_bytes = (storage / "events.log").read_bytes()
    assert (storage / "checkpoint.bin").exists()

    history = _open(storage)
    try:
        _assert_one_generation(storage, 1)
        assert history.retention_floor == 0
        assert [item.record for item in history.records()] == records
        assert history.event_count == count
        # The legacy bytes ARE the first segment, never reinterpreted.
        assert history.segment_path.read_bytes() == legacy_bytes
        history.record(_receipt(99))
    finally:
        history.close()
    again = _open(storage)
    try:
        assert again.generation == 1 and len(again.records()) == 4
    finally:
        again.close()


def test_large_legacy_history_migrates_to_a_head_checkpoint(tmp_path):
    storage = tmp_path / "legacy"
    lineage, tip = _lineage(3)
    records = lineage + [_receipt(i) for i in range(40)]
    count = _legacy(storage, records)
    assert (storage / "events.log").stat().st_size > SMALL.max_segment_bytes // 2

    history = _open(storage)
    try:
        _assert_one_generation(storage, 1)
        assert history.retention_floor == count == history.event_count
        assert history.retained_records() == ()
        assert history.receipt_count == len(records)
        # The lineage of retired receipts survives as the fixed-size summary.
        assert history.cognition_tip() == tip
        assert history.continuity.records == len(lineage)
        item = history.record(_receipt(1000))
        assert item.event_index == count
    finally:
        history.close()


def test_crash_during_migration_redoes_it_from_the_intact_legacy_log(tmp_path, monkeypatch):
    storage = tmp_path / "legacy"
    records = [_receipt(i) for i in range(3)]
    _legacy(storage, records)
    history = ReceiptHistory(storage, native_library=_library(), policy=SMALL)

    def crash(name):
        if name == "segment.fsync":
            raise _Crash(name)

    monkeypatch.setattr(generations, "_crash_point", crash)
    with pytest.raises(_Crash):
        history.open()
    _die(history)
    monkeypatch.undo()
    assert (storage / "events.log").exists() and not (storage / "MANIFEST").exists()

    reopened = _open(storage)
    try:
        assert [item.record for item in reopened.records()] == records
        _assert_one_generation(storage, 1)
    finally:
        reopened.close()


def test_corrupt_legacy_history_is_an_explicit_migration_failure(tmp_path):
    storage = tmp_path / "legacy"
    _legacy(storage, [_receipt(1)])
    log = storage / "events.log"
    data = bytearray(log.read_bytes())
    data[200] ^= 0x01
    log.write_bytes(bytes(data))
    with pytest.raises(HistoryError) as caught:
        _open(storage)
    assert caught.value.code == "HISTORY_LEGACY_MIGRATION_FAILED"
    assert not (storage / "MANIFEST").exists()
    assert log.read_bytes() == bytes(data)
