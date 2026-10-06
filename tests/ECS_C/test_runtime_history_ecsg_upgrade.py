from __future__ import annotations

import os
from pathlib import Path

import pytest

from elpis.ECS_C.kernel import Kernel
from elpis.runtime.history import (
    HISTORY_GENESIS_LABEL,
    RECORDERS,
    ROLES,
    HistoryError,
    ReceiptHistory,
    ReceiptRecord,
)


LEGACY_ROLES = (
    "history",
    "pipeline",
    "structure",
    "evolution",
    "inference",
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


def _legacy_history_with_one_record(storage: Path):
    kernel = Kernel(
        str(storage),
        genesis_label=HISTORY_GENESIS_LABEL,
    )
    kernel.open()

    try:
        ids = {}

        for role in LEGACY_ROLES:
            ids[role] = kernel.found_entity(role)

        kernel.run_until_quiescent()

        record = ReceiptRecord.of(
            "pipeline",
            "legacy.fixture",
            "a" * 64,
            phase="before-ecs-g",
        )

        message_id = kernel.entity_port(
            ids["pipeline"]
        ).propose(
            ids["history"],
            record.payload(),
        )

        kernel.run_until_quiescent()

        events = tuple(kernel.events())

        assert events[-2]["event_kind"] == "MESSAGE_ENQUEUED"
        assert events[-1]["event_kind"] == "MESSAGE_PROCESSED"

        return (
            record,
            message_id,
            len(events),
            kernel.log_path,
        )

    finally:
        kernel.close()


def test_current_runtime_history_authority_has_one_ecs_g_recorder():
    assert ROLES == (
        "history",
        "pipeline",
        "structure",
        "evolution",
        "inference",
        "ecs_g",
    )

    assert RECORDERS == (
        "pipeline",
        "structure",
        "evolution",
        "inference",
        "ecs_g",
    )

    record = ReceiptRecord.of(
        "ecs_g",
        "cognition.turn",
        "1" * 64,
        epoch_after="1",
    )

    assert record.subsystem == "ecs_g"


def test_new_history_materializes_exact_six_role_authority(tmp_path):
    storage = tmp_path / "history"

    with ReceiptHistory(storage) as history:
        assert tuple(history._ids) == ROLES
        assert set(history._ports) == set(RECORDERS)

        ids = history._kernel.entity_ids()

        assert len(ids) == 6
        assert set(ids) == set(history._ids.values())

        events = history._kernel.events()

        assert len(events) == 12

        assert [
            event["event_kind"]
            for event in events[:6]
        ] == ["ENTITY_FOUNDED"] * 6

        assert [
            event["event_kind"]
            for event in events[6:]
        ] == ["ENTITY_ACTIVATED"] * 6


def test_valid_legacy_history_upgrades_once_and_preserves_records(tmp_path):
    storage = tmp_path / "legacy"

    (
        legacy_record,
        legacy_message_id,
        legacy_event_count,
        _,
    ) = _legacy_history_with_one_record(storage)

    history = ReceiptHistory(storage).open()

    try:
        events = history._kernel.events()

        # Exactly one normal ECS_C founding transition and one activation.
        assert len(events) == legacy_event_count + 2

        founded, activated = events[-2:]

        assert founded["event_kind"] == "ENTITY_FOUNDED"
        assert founded["payload"]["label"] == "ecs_g"
        assert founded["payload"]["founding_index"] == 5

        assert activated["event_kind"] == "ENTITY_ACTIVATED"
        assert activated["entity_id"] == founded["entity_id"]
        assert activated["payload"] == {
            "from": "FOUNDED",
            "to": "ACTIVE",
        }

        assert history._ids["ecs_g"] == founded["entity_id"]

        records = history.records()

        assert len(records) == 1
        assert records[0].record == legacy_record
        assert records[0].message_id == legacy_message_id

        migrated_event_count = len(events)

    finally:
        history.close()

    # Reopen is idempotent: no second migration.
    history = ReceiptHistory(storage).open()

    try:
        assert len(history._kernel.events()) == migrated_event_count
        assert history.records()[0].record == legacy_record
    finally:
        history.close()


def test_ecs_g_recording_is_idempotent_after_migration(tmp_path):
    storage = tmp_path / "legacy"

    _legacy_history_with_one_record(storage)

    history = ReceiptHistory(
        storage,
        native_library=_native_library(),
    ).open()

    try:
        assert history._native is not None
        assert not history._ports

        record = ReceiptRecord.of(
            "ecs_g",
            "cognition.turn",
            "b" * 64,
            epoch_before="0",
            epoch_after="1",
            mechanism="native-k1",
        )

        before = history._native.event_count

        first = history.record(record)

        after_first = history._native.event_count

        assert after_first == before + 2
        assert first.record == record

        second = history.record(record)

        assert second == first
        assert history._native.event_count == after_first

    finally:
        history.close()

    # Canonical replay sees the native-written receipt exactly once.
    history = ReceiptHistory(storage).open()

    try:
        matches = [
            item
            for item in history.records()
            if item.record == record
        ]

        assert len(matches) == 1
        assert matches[0] == first
    finally:
        history.close()


def test_foreign_sixth_entity_is_refused_without_mutation(tmp_path):
    storage = tmp_path / "foreign"

    kernel = Kernel(
        str(storage),
        genesis_label=HISTORY_GENESIS_LABEL,
    )
    kernel.open()

    try:
        for role in LEGACY_ROLES:
            kernel.found_entity(role)

        kernel.found_entity("intruder")
        kernel.run_until_quiescent()

        log_path = Path(kernel.log_path)

    finally:
        kernel.close()

    before = log_path.read_bytes()

    with pytest.raises(HistoryError) as caught:
        ReceiptHistory(storage).open()

    assert caught.value.code == "FOREIGN_HISTORY"
    assert log_path.read_bytes() == before


def test_partial_legacy_topology_is_foreign_and_not_completed(tmp_path):
    storage = tmp_path / "partial"

    kernel = Kernel(
        str(storage),
        genesis_label=HISTORY_GENESIS_LABEL,
    )
    kernel.open()

    try:
        for role in LEGACY_ROLES[:-1]:
            kernel.found_entity(role)

        kernel.run_until_quiescent()

        log_path = Path(kernel.log_path)

    finally:
        kernel.close()

    before = log_path.read_bytes()

    with pytest.raises(HistoryError) as caught:
        ReceiptHistory(storage).open()

    assert caught.value.code == "FOREIGN_HISTORY"
    assert log_path.read_bytes() == before
