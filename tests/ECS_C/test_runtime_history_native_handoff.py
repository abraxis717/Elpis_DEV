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
