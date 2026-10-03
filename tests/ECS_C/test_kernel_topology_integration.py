"""Integration qualification for Kernel -> verified topology -> analysis."""

from __future__ import annotations

import pytest

import elpis.ECS_C.kernel as kernel_module
from elpis.ECS_C.errors import EcsError
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import genesis_descriptor_digest
from elpis.ECS_C.topology import project_topology, verify_projection
from elpis.ECS_C.topology_analysis import analyze_projection, verify_analysis


def _build_runtime_scenario(kernel: Kernel) -> dict[str, str]:
    a = kernel.found_entity("alpha")
    b = kernel.found_entity("beta")
    c = kernel.found_entity("gamma")
    kernel.run_until_quiescent()

    kernel.entity_port(a).propose(b, b"a-to-b")
    kernel.entity_port(b).propose(a, b"b-to-a")
    kernel.entity_port(c).propose(c, b"c-self")
    kernel.run_until_quiescent()
    return {"a": a, "b": b, "c": c}


def _edge_keys(projection) -> set[tuple[str, str]]:
    return {
        (edge.sender_entity_id, edge.receiver_entity_id)
        for edge in projection.edges
    }


def test_kernel_runtime_edge_matches_direct_qualified_derivation_and_is_read_only(
    tmp_path,
) -> None:
    kernel = Kernel(str(tmp_path)).open()
    try:
        ids = _build_runtime_scenario(kernel)
        before_snapshot = kernel.snapshot()
        before_events = kernel.events()

        projection = kernel.topology_projection()
        analysis = kernel.topology_analysis()

        verify_projection(projection)
        verify_analysis(analysis)

        expected_projection = project_topology(
            kernel._genesis_digest,
            before_events,
            kernel.mailbox_capacity,
            scheduler_protocol=kernel.scheduler_protocol,
        )
        verify_projection(expected_projection)
        expected_analysis = analyze_projection(expected_projection)
        verify_analysis(expected_analysis)

        assert projection == expected_projection
        assert analysis == expected_analysis
        assert analysis.topology_digest == projection.topology_digest

        assert projection.event_count == len(before_events)
        assert {node.entity_id for node in projection.nodes} == set(ids.values())
        assert _edge_keys(projection) == {
            (ids["a"], ids["b"]),
            (ids["b"], ids["a"]),
            (ids["c"], ids["c"]),
        }
        assert analysis.node_count == 3
        assert analysis.edge_count == 3
        assert analysis.total_message_count == 3

        assert kernel.snapshot() == before_snapshot
        assert kernel.events() == before_events
    finally:
        kernel.close()


def test_kernel_topology_analysis_executes_the_explicit_composition_chain(
    tmp_path,
    monkeypatch,
) -> None:
    kernel = Kernel(str(tmp_path)).open()
    try:
        _build_runtime_scenario(kernel)

        calls: list[str] = []
        original_fast_project = kernel_module._project_topology_from_validated_state
        original_verify_projection = kernel_module.verify_projection
        original_analyze = kernel_module.analyze_projection
        original_verify_analysis = kernel_module.verify_analysis

        def fast_project(*args, **kwargs):
            calls.append("project_validated_state")
            return original_fast_project(*args, **kwargs)

        def verify_projection_call(*args, **kwargs):
            calls.append("verify_projection")
            return original_verify_projection(*args, **kwargs)

        def analyze(*args, **kwargs):
            calls.append("analyze_projection")
            return original_analyze(*args, **kwargs)

        def verify_analysis_call(*args, **kwargs):
            calls.append("verify_analysis")
            return original_verify_analysis(*args, **kwargs)

        monkeypatch.setattr(
            kernel_module,
            "_project_topology_from_validated_state",
            fast_project,
        )
        monkeypatch.setattr(
            kernel_module,
            "verify_projection",
            verify_projection_call,
        )
        monkeypatch.setattr(kernel_module, "analyze_projection", analyze)
        monkeypatch.setattr(
            kernel_module,
            "verify_analysis",
            verify_analysis_call,
        )

        import elpis.ECS_C.topology as topology_module

        def forbidden_replay(*args, **kwargs):
            raise AssertionError("duplicate semantic replay invoked")

        monkeypatch.setattr(
            topology_module,
            "replay_from_events",
            forbidden_replay,
        )

        analysis = kernel.topology_analysis()
        verify_analysis(analysis)

        assert calls == [
            "project_validated_state",
            "verify_projection",
            "analyze_projection",
            "verify_analysis",
        ]
    finally:
        kernel.close()


def test_kernel_topology_edge_is_recovery_deterministic(tmp_path) -> None:
    first = Kernel(str(tmp_path)).open()
    try:
        _build_runtime_scenario(first)
        projection_digest = first.topology_projection().topology_digest
        analysis_digest = first.topology_analysis().analysis_digest
        state_root = first.state_root_digest()
        events = first.events()
    finally:
        first.close()

    reopened = Kernel(str(tmp_path)).open()
    try:
        assert reopened.state_root_digest() == state_root
        assert reopened.events() == events
        assert reopened.topology_projection().topology_digest == projection_digest
        assert reopened.topology_analysis().analysis_digest == analysis_digest
    finally:
        reopened.close()


def test_kernel_topology_edge_requires_open_kernel(tmp_path) -> None:
    kernel = Kernel(str(tmp_path))
    with pytest.raises(EcsError, match="KERNEL_NOT_OPEN"):
        kernel.topology_projection()
    with pytest.raises(EcsError, match="KERNEL_NOT_OPEN"):
        kernel.topology_analysis()
