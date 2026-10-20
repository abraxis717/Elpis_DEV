"""Ingress end to end: caller bytes -> Regex -> HACF -> published zero-authority proposal."""
from __future__ import annotations

import json

import pytest

from elpis.runtime import CompositionError, Runtime, RuntimeConfig

from .conftest import CONTRADICTION, POSITIVE
from ..conftest import runtime_config

ZERO_AUTHORITY = ("semantic_authority", "admission_authority", "execution_authority", "runtime_admission")


def test_published_proposal_carries_its_bindings_and_writes_no_continuity(runtime, ingress, corpus):
    before = runtime.continuity.snapshot()
    result = runtime.run_ingress(ingress, POSITIVE)
    assert result.batch_published and not result.fail_closed
    proposal = json.loads(result.proposal_json)
    assert all(proposal[flag] is False for flag in ZERO_AUTHORITY)
    assert result.proposal_digest and result.corpus_manifest_digest and result.overlay_identity
    # Ingress owns its result: no synchronous continuity or audit write.
    assert runtime.continuity.snapshot() == before


def test_fail_closed_ingress_changes_nothing(runtime, ingress):
    before = runtime.continuity.snapshot()
    result = runtime.run_ingress(ingress, CONTRADICTION)
    assert result.fail_closed and not result.batch_published
    assert runtime.continuity.snapshot() == before


def test_same_input_is_deterministic(runtime, ingress):
    first = runtime.run_ingress(ingress, POSITIVE)
    again = runtime.run_ingress(ingress, POSITIVE)
    assert again.proposal_digest == first.proposal_digest


_ECS_C_NOT_PROVISIONED, _ECS_C_RECOVERY_REFUSED, _ECS_C_UNRECOVERABLE = 25, 29, 23   # elpis/ecs_c_authority.h


def test_reopen_keeps_the_authority_and_a_tampered_authority_is_refused(tmp_path, ingress, runtime_library):
    config = runtime_config(tmp_path / "authority", runtime_library)
    with Runtime(config) as rt:
        rt.run_ingress(ingress, POSITIVE)
        before = rt.continuity.snapshot()
    with Runtime(config) as rt:
        assert rt.continuity.snapshot() == before
    for name in ("authority.a", "authority.b"):
        path = config.authority_dir / "state" / name
        data = bytearray(path.read_bytes())
        data[20] ^= 0x01
        path.write_bytes(bytes(data))
    runtime = Runtime(config)
    with pytest.raises(CompositionError) as info:
        runtime.open()
    # Neither record verifies: recovery refuses (UNRECOVERABLE), and the open with it; nothing is reinterpreted.
    assert info.value.code == "RUNTIME_AUTHORITY"
    status = runtime.continuity.status()
    assert (status["ecs_c_code"], status["disposition"]) == (_ECS_C_RECOVERY_REFUSED, _ECS_C_UNRECOVERABLE)


def test_a_directory_that_is_not_a_provisioned_authority_is_refused(tmp_path, runtime_library):
    path = tmp_path / "legacy"   # here the retired continuity receipt-history layout: never read, never adopted
    path.mkdir()
    (path / "MANIFEST").write_bytes(b"retired receipt-history layout")
    runtime = Runtime(runtime_config(path, runtime_library))
    with pytest.raises(CompositionError) as info:
        runtime.open()
    assert info.value.code == "RUNTIME_AUTHORITY"
    assert runtime.continuity.status()["ecs_c_code"] == _ECS_C_NOT_PROVISIONED
    assert sorted(p.name for p in path.iterdir()) == ["MANIFEST"]
