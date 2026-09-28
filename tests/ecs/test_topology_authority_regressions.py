from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from elpis.ecs import canonical
from elpis.ecs.kernel import Kernel
from elpis.ecs.persistence import genesis_descriptor_digest
from elpis.ecs.topology import (
    DOMAIN_TOPOLOGY,
    TopologyError,
    project_topology,
    topology_digest,
    verify_projection,
)
from elpis.ecs.topology_analysis import (
    TopologyAnalysisError,
    analysis_digest,
    analyze_projection,
    analyze_topology,
    verify_analysis,
)


class OneShotSequence(Sequence):
    def __init__(self, items):
        self.items = tuple(items)
        self.iterations = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("caller sequence iterated more than once")
        return iter(self.items)


def _scenario(tmp_path):
    k = Kernel(str(tmp_path)).open()
    a = k.found_entity("alpha")
    b = k.found_entity("beta")
    k.run_until_quiescent()
    k.entity_port(a).propose(b, b"m1")
    k.run_until_quiescent()
    return k


def _redigest_projection(projection):
    return replace(
        projection,
        topology_digest=canonical.domain_digest(
            DOMAIN_TOPOLOGY,
            projection.to_dict(),
        ),
    )


def test_project_topology_snapshots_caller_sequence_once(tmp_path):
    k = _scenario(tmp_path)
    seq = OneShotSequence(k.events())
    projection = project_topology(k._genesis_digest, seq, k.mailbox_capacity, scheduler_protocol=k.scheduler_protocol)
    assert seq.iterations == 1
    verify_projection(projection)
    k.close()


def test_projection_equality_includes_digest_and_accessor_recomputes(tmp_path):
    k = _scenario(tmp_path)
    projection = project_topology(k._genesis_digest, k.events(), k.mailbox_capacity, scheduler_protocol=k.scheduler_protocol)
    forged = replace(projection, topology_digest="0" * 64)
    assert projection != forged
    with pytest.raises(TopologyError, match="TOPOLOGY_DIGEST_MISMATCH"):
        verify_projection(forged)
    with pytest.raises(TopologyError, match="TOPOLOGY_DIGEST_MISMATCH"):
        topology_digest(forged)
    k.close()


def test_projection_rejects_forged_self_loop_even_with_fresh_digest(tmp_path):
    k = _scenario(tmp_path)
    projection = project_topology(k._genesis_digest, k.events(), k.mailbox_capacity, scheduler_protocol=k.scheduler_protocol)
    edge = projection.edges[0]
    bad_edge = replace(edge, self_loop=not edge.self_loop)
    forged = replace(
        projection,
        edges=(bad_edge,) + projection.edges[1:],
        topology_digest="0" * 64,
    )
    forged = _redigest_projection(forged)
    with pytest.raises(TopologyError, match="TOPOLOGY_EDGE_SELF_LOOP_INVALID"):
        verify_projection(forged)
    with pytest.raises(TopologyAnalysisError, match="TOPOLOGY_EDGE_SELF_LOOP_INVALID"):
        analyze_projection(forged)
    k.close()


def test_analysis_rejects_dangling_projection_as_domain_error(tmp_path):
    k = _scenario(tmp_path)
    projection = project_topology(k._genesis_digest, k.events(), k.mailbox_capacity, scheduler_protocol=k.scheduler_protocol)
    edge = projection.edges[0]
    dangling = replace(edge, receiver_entity_id="f" * 64)
    forged = replace(
        projection,
        edges=(dangling,) + projection.edges[1:],
        topology_digest="0" * 64,
    )
    forged = _redigest_projection(forged)
    with pytest.raises(
        TopologyAnalysisError,
        match="TOPOLOGY_EDGE_DANGLING_RECEIVER",
    ):
        analyze_projection(forged)
    k.close()


def test_analysis_equality_includes_digest_and_accessor_recomputes(tmp_path):
    k = _scenario(tmp_path)
    analysis = analyze_topology(k._genesis_digest, k.events(), k.mailbox_capacity, scheduler_protocol=k.scheduler_protocol)
    forged = replace(analysis, analysis_digest="0" * 64)
    assert analysis != forged
    verify_analysis(analysis)
    with pytest.raises(
        TopologyAnalysisError,
        match="ANALYSIS_DIGEST_MISMATCH",
    ):
        verify_analysis(forged)
    with pytest.raises(
        TopologyAnalysisError,
        match="ANALYSIS_DIGEST_MISMATCH",
    ):
        analysis_digest(forged)
    k.close()
