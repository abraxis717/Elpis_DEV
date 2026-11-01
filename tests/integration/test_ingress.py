"""Ingress end to end: caller bytes -> Regex -> HACF -> published zero-authority proposal."""
from __future__ import annotations

import json

import pytest

from elpis.runtime import CompositionError, Runtime, RuntimeConfig

from .conftest import CONTRADICTION, POSITIVE
from ..conftest import TEST_SESSION_GENERATION, runtime_config

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


def test_each_open_is_a_new_session_and_nothing_of_the_last_one_remains(ingress, runtime_library):
    config = runtime_config(runtime_library)
    with Runtime(config) as rt:
        rt.run_ingress(ingress, POSITIVE)
        first, before = rt.continuity.status(), rt.continuity.snapshot()
    with Runtime(config) as rt:
        second, after = rt.continuity.status(), rt.continuity.snapshot()
    # Ingress made no session transition; each open is a fresh genesis with the configured sources founded again.
    assert before.generation == after.generation == TEST_SESSION_GENERATION
    assert first["session"] != second["session"] and first["instance"] != second["instance"]


def test_the_runtime_names_no_authority_path(runtime_library):
    from elpis.runtime import RuntimeConfig
    assert "authority_dir" not in RuntimeConfig.__dataclass_fields__
    assert "k1_checkpoint_dir" not in RuntimeConfig.__dataclass_fields__
    with Runtime(runtime_config(runtime_library)) as rt:
        assert rt.continuity.status()["backend"] == 3   # ELPIS_RUNTIME_BACKEND_ECS_C_SESSION
