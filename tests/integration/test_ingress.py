"""Ingress end to end: caller bytes -> Regex -> HACF -> published zero-authority proposal."""
from __future__ import annotations

import json

import pytest

from elpis.runtime import CompositionError, Runtime, RuntimeConfig

from .conftest import CONTRADICTION, POSITIVE

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


def test_reopen_keeps_continuity_and_tampered_continuity_is_refused(tmp_path, ingress, continuity_library):
    config = RuntimeConfig(tmp_path / "continuity", continuity_library)
    with Runtime(config) as rt:
        rt.run_ingress(ingress, POSITIVE)
        before = rt.continuity.snapshot()
    with Runtime(config) as rt:
        assert rt.continuity.snapshot() == before
    for name in ("continuity.a", "continuity.b"):
        path = config.continuity_dir / name
        data = bytearray(path.read_bytes())
        data[20] ^= 0x01
        path.write_bytes(bytes(data))
    with pytest.raises(CompositionError) as info:
        Runtime(config).open()
    assert info.value.code == "CONTINUITY_CORRUPT"


def test_retired_receipt_history_directory_is_refused(tmp_path, continuity_library):
    path = tmp_path / "legacy"
    path.mkdir()
    (path / "MANIFEST").write_bytes(b"retired receipt-history layout")
    with pytest.raises(CompositionError) as info:
        Runtime(RuntimeConfig(path, continuity_library)).open()
    assert info.value.code == "CONTINUITY_LEGACY_STORAGE"
