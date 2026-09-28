from __future__ import annotations

import json

import pytest

from elpis.ecs.errors import EcsError
from elpis.ecs.kernel import Kernel
from elpis.ecs.persistence import genesis_descriptor_digest
from elpis.ecs.scheduler import SCHEDULER_V1, SCHEDULER_V2

from elpis.ecs.projection import ProjectionRequest, project_history
from elpis.ecs.projection.kernel_adapter import project_kernel_history


def _direct(kernel: Kernel, request: ProjectionRequest):
    return project_history(
        genesis_descriptor_digest(
            kernel.genesis_label,
            kernel.scheduler_protocol,
        ),
        kernel.events(),
        request,
        mailbox_capacity=kernel.mailbox_capacity,
        scheduler_protocol=kernel.scheduler_protocol,
    )


def test_live_kernel_projection_matches_direct_replay(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        kernel.found_entity("alpha")
        request = ProjectionRequest(max_records=64)
        actual = project_kernel_history(kernel, request)
        expected = _direct(kernel, request)

        assert actual.canonical_bytes() == expected.canonical_bytes()
        assert actual.projection_digest == expected.projection_digest
        assert actual.source == expected.source


def test_projection_does_not_mutate_kernel_state_or_history(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        kernel.found_entity("alpha")
        before_root = kernel.state_root_digest()
        before_events = kernel.events()

        project_kernel_history(kernel, ProjectionRequest(max_records=64))

        assert kernel.state_root_digest() == before_root
        assert kernel.events() == before_events


def test_empty_open_kernel_projects_empty_history(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        result = project_kernel_history(
            kernel,
            ProjectionRequest(max_records=4),
        )

        assert result.source.event_count == 0
        assert result.total_matches == 0
        assert result.records == ()
        assert result.record_bytes_used == 2
        assert result.truncated is False


def test_entity_selector_survives_kernel_adapter(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        alpha = kernel.found_entity("alpha")
        kernel.found_entity("beta")

        result = project_kernel_history(
            kernel,
            ProjectionRequest(
                entity_ids=(alpha,),
                max_records=64,
            ),
        )

        assert result.total_matches == 1
        assert len(result.records) == 1
        assert result.records[0].to_dict()["event"]["entity_id"] == alpha


def test_record_budget_survives_kernel_adapter(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        for label in ("alpha", "beta", "gamma"):
            kernel.found_entity(label)

        result = project_kernel_history(
            kernel,
            ProjectionRequest(max_records=1),
        )

        assert result.total_matches == 3
        assert len(result.records) == 1
        assert result.truncated is True
        assert result.budget_exhausted == ("records",)


def test_repeated_projection_is_byte_and_digest_identical(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        kernel.found_entity("alpha")
        request = ProjectionRequest(max_records=64)

        a = project_kernel_history(kernel, request)
        b = project_kernel_history(kernel, request)

        assert a.canonical_bytes() == b.canonical_bytes()
        assert a.projection_digest == b.projection_digest


def test_returned_decoded_views_cannot_mutate_projection(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        kernel.found_entity("alpha")
        result = project_kernel_history(
            kernel,
            ProjectionRequest(max_records=64),
        )
        before = result.canonical_bytes()
        view = result.to_dict()

        view["records"][0]["event"]["event_kind"] = "CORRUPTED_VIEW"
        view["source"]["event_count"] = 999

        assert result.canonical_bytes() == before
        assert result.to_dict()["records"][0]["event"]["event_kind"] == "ENTITY_FOUNDED"


def test_projection_before_and_after_transition_are_individually_coherent(tmp_path):
    with Kernel(str(tmp_path)) as kernel:
        kernel.found_entity("alpha")
        first = project_kernel_history(
            kernel,
            ProjectionRequest(max_records=64),
        )
        first_bytes = first.canonical_bytes()

        kernel.found_entity("beta")
        second = project_kernel_history(
            kernel,
            ProjectionRequest(max_records=64),
        )

        assert first.source.event_count == 1
        assert second.source.event_count == 2
        assert len(first.records) == 1
        assert len(second.records) == 2
        assert first.canonical_bytes() == first_bytes
        assert first.projection_digest != second.projection_digest


@pytest.mark.parametrize("protocol", [SCHEDULER_V1, SCHEDULER_V2])
def test_explicit_scheduler_profiles_match_direct_replay(tmp_path, protocol):
    with Kernel(str(tmp_path), scheduler_protocol=protocol) as kernel:
        kernel.found_entity("alpha")
        request = ProjectionRequest(max_records=64)

        actual = project_kernel_history(kernel, request)
        expected = _direct(kernel, request)

        assert actual.canonical_bytes() == expected.canonical_bytes()
        assert actual.source.genesis_digest == expected.source.genesis_digest
        assert actual.source.scheduler_protocol == protocol


def test_custom_genesis_label_matches_direct_replay(tmp_path):
    with Kernel(str(tmp_path), genesis_label="ecs-context-projector-test") as kernel:
        kernel.found_entity("alpha")
        request = ProjectionRequest(max_records=64)

        assert (
            project_kernel_history(kernel, request).canonical_bytes()
            == _direct(kernel, request).canonical_bytes()
        )


def test_unopened_kernel_fails_closed(tmp_path):
    kernel = Kernel(str(tmp_path))
    with pytest.raises(EcsError, match="KERNEL_NOT_OPEN"):
        project_kernel_history(kernel, ProjectionRequest())


def test_closed_kernel_fails_closed(tmp_path):
    kernel = Kernel(str(tmp_path)).open()
    kernel.close()
    with pytest.raises(EcsError, match="KERNEL_NOT_OPEN"):
        project_kernel_history(kernel, ProjectionRequest())


def test_non_kernel_is_rejected():
    with pytest.raises(TypeError, match="elpis.ecs.kernel.Kernel"):
        project_kernel_history(object(), ProjectionRequest())
