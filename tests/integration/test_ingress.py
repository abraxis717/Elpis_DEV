"""Ingress end to end: caller bytes -> Regex -> HACF -> published proposal -> recorded history."""
from __future__ import annotations

import json

import pytest

from elpis.ECS_C.errors import EcsError
from elpis.runtime import HistoryError, Runtime, RuntimeConfig

from .conftest import CONTRADICTION, POSITIVE

ZERO_AUTHORITY = ("semantic_authority", "admission_authority", "execution_authority", "runtime_admission")


def test_published_proposal_is_recorded_with_its_bindings(runtime, ingress, corpus):
    result, recorded = runtime.run_ingress(ingress, POSITIVE)
    assert result.batch_published and not result.fail_closed
    proposal = json.loads(result.proposal_json)
    assert all(proposal[flag] is False for flag in ZERO_AUTHORITY)
    record = recorded.record
    assert (record.subsystem, record.kind, record.digest) == ("pipeline", "ingress.proposal",
                                                              result.proposal_digest)
    bindings = dict(record.bindings)
    assert bindings["corpus"] == result.corpus_manifest_digest
    assert bindings["overlay"] == result.overlay_identity
    assert runtime.history.records() == (recorded,)


def test_fail_closed_ingress_is_not_recorded(runtime, ingress):
    root = runtime.history.state_root
    result, recorded = runtime.run_ingress(ingress, CONTRADICTION)
    assert result.fail_closed and not result.batch_published and recorded is None
    assert runtime.history.records() == () and runtime.history.state_root == root


def test_same_input_is_recorded_once(runtime, ingress):
    _, first = runtime.run_ingress(ingress, POSITIVE)
    _, again = runtime.run_ingress(ingress, POSITIVE)
    assert again == first and len(runtime.history.records()) == 1


def test_reopen_replays_and_tampered_history_is_refused(
    tmp_path,
    ingress,
    history_library,
):
    config = RuntimeConfig(
        tmp_path / "history",
        history_native_library=history_library,
    )
    with Runtime(config) as rt:
        _, recorded = rt.run_ingress(ingress, POSITIVE)
        root = rt.history.state_root
    with Runtime(config) as rt:
        assert rt.history.records() == (recorded,) and rt.history.state_root == root
    log = config.history_dir / "events.log"
    data = bytearray(log.read_bytes())
    index = data.index(b"ingress.proposal".hex().encode())
    data[index] ^= 0x01
    log.write_bytes(bytes(data))
    with pytest.raises((EcsError, HistoryError)):
        Runtime(config).open()


def test_history_founded_for_another_purpose_is_refused(
    tmp_path,
    history_library,
):
    from elpis.ECS_C.kernel import Kernel
    from elpis.runtime.history import HISTORY_GENESIS_LABEL
    path = tmp_path / "foreign"
    with Kernel(str(path), genesis_label=HISTORY_GENESIS_LABEL).open() as kernel:
        kernel.found_entity("something-else")
    with pytest.raises(HistoryError, match="FOREIGN_HISTORY"):
        Runtime(
            RuntimeConfig(
                path,
                history_native_library=history_library,
            )
        ).open()
