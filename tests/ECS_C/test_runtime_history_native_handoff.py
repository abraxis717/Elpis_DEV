from __future__ import annotations

import os
from pathlib import Path

import pytest

from elpis.runtime.history import (
    ReceiptHistory,
    ReceiptRecord,
)


def _native_library() -> Path:
    root = Path(
        os.environ["ELPIS_NATIVE_BUILD"]
    ).resolve()

    matches = sorted(
        p
        for p in root.rglob(
            "libelpis_ecsc_history.so"
        )
        if p.is_file()
    )

    assert len(matches) == 1
    return matches[0]


def _record(n: int) -> ReceiptRecord:
    return ReceiptRecord.of(
        "ecs_g",
        "cognition.turn",
        format(n, "064x"),
        epoch_before=str(n - 1),
        epoch_after=str(n),
        mechanism="native-k1",
    )


def test_native_handoff_record_hot_path_avoids_python_kernel_mutation(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "history"

    history = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    assert history._native is not None
    assert not history._ports

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "PYTHON_KERNEL_HOT_PATH_USED"
        )

    # A committed native record must not rescan history, propose through a
    # Python EntityPort, or drive the Python scheduler.
    monkeypatch.setattr(
        history._kernel,
        "events",
        forbidden,
    )

    monkeypatch.setattr(
        history._kernel,
        "run_until_quiescent",
        forbidden,
    )

    monkeypatch.setattr(
        history._kernel,
        "entity_port",
        forbidden,
    )

    before = history.state_root

    one = history.record(_record(1))

    assert one.record == _record(1)
    assert one.event_index == 12
    assert one.message_id
    assert one.event_digest
    assert history.state_root != before

    # Idempotence is now O(1) over the cold-built record index and appends
    # nothing.
    root_after_one = history.state_root
    duplicate = history.record(_record(1))

    assert duplicate == one
    assert history.state_root == root_after_one

    two = history.record(_record(2))

    assert two.record == _record(2)
    assert two.event_index == 14
    assert len(history.records()) == 2

    history.close()

    # Canonical Python replay remains the independent verifier of the native
    # durable history.
    replay = ReceiptHistory(storage).open()

    try:
        assert replay.records() == (
            one,
            two,
        )
        assert replay.state_root != before
    finally:
        replay.close()


def test_native_handoff_projection_temporarily_returns_log_to_python_replay(
    tmp_path,
):
    storage = tmp_path / "history"

    history = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        one = history.record(_record(1))

        projected = history.projection()

        assert projected is not None
        assert history._native is not None

        # Native ownership was reacquired, so another receipt can commit
        # without reopening ReceiptHistory.
        two = history.record(_record(2))

        assert one.event_index == 12
        assert two.event_index == 14

    finally:
        history.close()

    replay = ReceiptHistory(storage).open()

    try:
        assert replay.records() == (
            one,
            two,
        )
    finally:
        replay.close()


def test_native_handoff_rejects_lock_conflict_without_sidecar_history(
    tmp_path,
):
    storage = tmp_path / "history"

    first = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        with pytest.raises(Exception):
            ReceiptHistory(
                storage,
                native_library=_native_library(),
            ).open()
    finally:
        first.close()


def test_runtime_config_explicitly_owns_native_history_writer(
    tmp_path,
):
    from elpis.runtime import Runtime, RuntimeConfig

    library = _native_library()

    config = RuntimeConfig(
        tmp_path / "runtime-history",
        history_native_library=library,
    )

    runtime = Runtime(config).open()

    try:
        assert runtime.config.history_native_library == library
        assert runtime.history._native is not None
        assert not runtime.history._ports

        recorded = runtime.history.record(_record(1))

        assert recorded.record == _record(1)
        assert recorded.event_index == 12

    finally:
        runtime.close()


def test_receipt_history_without_native_library_is_replay_only(
    tmp_path,
):
    storage = tmp_path / "history"

    writer = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        committed = writer.record(_record(1))
    finally:
        writer.close()

    replay = ReceiptHistory(storage).open()

    try:
        assert replay.records() == (committed,)

        with pytest.raises(Exception, match="HISTORY_CLOSED"):
            replay.record(_record(2))

        assert replay.records() == (committed,)

    finally:
        replay.close()



def _enqueue_only_prefix(storage: Path, record: ReceiptRecord):
    history = ReceiptHistory(storage).open()
    try:
        before = tuple(history._kernel.events())

        message_id = history._ports[record.subsystem].propose(
            history.history_entity,
            record.payload(),
        )

        after = tuple(history._kernel.events())

        assert len(after) == len(before) + 1
        assert after[:-1] == before
        assert after[-1]["event_kind"] == "MESSAGE_ENQUEUED"

        return message_id, len(after)
    finally:
        history.close()


def test_native_writer_reopen_completes_enqueue_only_prefix(tmp_path):
    storage = tmp_path / "partial-prefix"
    record = _record(41)

    message_id, prefix_count = _enqueue_only_prefix(
        storage,
        record,
    )

    writer = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        assert writer._native is not None
        assert not writer._ports

        matching = [
            item
            for item in writer.records()
            if item.record == record
        ]

        assert len(matching) == 1
        assert matching[0].message_id == message_id
    finally:
        writer.close()

    replay = ReceiptHistory(storage).open()
    try:
        events = tuple(replay._kernel.events())

        assert len(events) == prefix_count + 1
        assert events[-2]["event_kind"] == "MESSAGE_ENQUEUED"
        assert events[-1]["event_kind"] == "MESSAGE_PROCESSED"

        matching = [
            item
            for item in replay.records()
            if item.record == record
        ]
        assert len(matching) == 1
        assert matching[0].message_id == message_id
    finally:
        replay.close()


def test_native_writer_reopen_recovery_is_idempotent(tmp_path):
    storage = tmp_path / "partial-prefix-idempotent"
    record = _record(42)

    _, prefix_count = _enqueue_only_prefix(
        storage,
        record,
    )

    first = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()
    first.close()

    replay = ReceiptHistory(storage).open()
    try:
        recovered_count = len(replay._kernel.events())
    finally:
        replay.close()

    assert recovered_count == prefix_count + 1

    second = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()
    second.close()

    replay = ReceiptHistory(storage).open()
    try:
        events = tuple(replay._kernel.events())

        assert len(events) == recovered_count
        assert events[-1]["event_kind"] == "MESSAGE_PROCESSED"
        assert len([
            item
            for item in replay.records()
            if item.record == record
        ]) == 1
    finally:
        replay.close()


def test_replay_only_history_never_runs_nonquiescent_recovery(tmp_path, monkeypatch):
    storage = tmp_path / "partial-prefix-replay-only"
    record = _record(43)

    _, prefix_count = _enqueue_only_prefix(
        storage,
        record,
    )

    from elpis.ECS_C.kernel import Kernel

    calls = []
    original = Kernel.run_until_quiescent

    def observed(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        Kernel,
        "run_until_quiescent",
        observed,
    )

    replay = ReceiptHistory(storage).open()
    try:
        events = tuple(replay._kernel.events())

        assert len(events) == prefix_count
        assert events[-1]["event_kind"] == "MESSAGE_ENQUEUED"
        assert calls == []
    finally:
        replay.close()


def test_native_hot_record_path_still_does_not_use_python_scheduler(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "hot-path-remains-native"

    history = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("PYTHON_SCHEDULER_USED_ON_NATIVE_HOT_PATH")

        monkeypatch.setattr(
            history._kernel,
            "run_until_quiescent",
            forbidden,
        )

        one = history.record(_record(44))

        assert one.record == _record(44)
    finally:
        history.close()
