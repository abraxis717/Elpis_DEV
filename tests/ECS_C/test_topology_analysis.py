"""Adversarial qualification tests for the deterministic structural analysis.

These tests establish that:
  * the committed ECS interaction history admits DETERMINISTIC, REPRODUCIBLE
    graph-structural descriptors derived through the EXISTING qualified
    topology-projection authority (``topology.project_topology``);
  * the analysis is a PURE DERIVED VIEW (no second mutable state authority,
    no mutation authority, no semantic/trust/causal meaning);
  * node metrics, exact directed SCCs, and the condensation graph are
    deterministic and correctly ordered/indexed;
  * the analysis FAILS CLOSED on wrong genesis, wrong capacity, corrupted
    history, and forged attribution (reusing the existing authoritative
    replay/validation path — NOT duplicating those validators);
  * payload text has NO authority over graph structure;
  * the output is deterministic across repeated calls, independent storage
    directories, reopen/fresh replay, fresh Python processes, and multiple
    PYTHONHASHSEED values (0, 1, 42);
  * the analysis makes ZERO persistent changes (committed event bytes,
    state-root digest, entity IDs/lifecycle, and mailbox state are unchanged);
  * the analysis digest is domain-separated from the topology digest, the
    undomained digest, the topology domain, and the state-root domain.

The analysis is a read-only projection (structural analysis of the
already-qualified topology). These tests qualify it WITHOUT widening kernel
authority: the analysis grants no mutation authority, is not semantic truth,
and is not an independent durable authority.
"""

from __future__ import annotations
from elpis.ECS_C.scheduler import SCHEDULER_V1

import copy
import os
import subprocess
import sys
import tempfile

import pytest

from elpis.ECS_C import canonical
from elpis.ECS_C.errors import (
    BrokenChainError,
    CorruptEventError,
    EnvelopeError,
    EcsError,
    ReplayError,
    WrongAuthorityError,
)
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import genesis_descriptor_digest
from elpis.ECS_C.topology import (
    DOMAIN_TOPOLOGY,
    TOPOLOGY_SCHEMA,
    TopologyError,
    TopologyProjection,
    project_topology,
)
from elpis.ECS_C.topology_analysis import (
    ANALYSIS_SCHEMA,
    DOMAIN_ANALYSIS,
    CondensationEdge,
    NodeMetric,
    SccRecord,
    TopologyAnalysis,
    TopologyAnalysisError,
    analysis_digest,
    analyze_projection,
    analyze_topology,
)

GENESIS_LABEL = "ecs-m1a-genesis"
GENESIS = genesis_descriptor_digest(GENESIS_LABEL)
CAP = 16  # DEFAULT_MAILBOX_CAPACITY


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _analyze(kernel):
    """Analyze the topology from the kernel's committed history."""
    return analyze_topology(kernel._genesis_digest, kernel.events(),
                            kernel.mailbox_capacity)


def _project(kernel):
    """Project the topology from the kernel's committed history."""
    return project_topology(kernel._genesis_digest, kernel.events(), kernel.mailbox_capacity, scheduler_protocol=kernel.scheduler_protocol)


def _edge_key(e):
    return (e.sender_entity_id, e.receiver_entity_id)


def _metric_by_id(analysis, entity_id):
    return next(m for m in analysis.node_metrics if m.entity_id == entity_id)


def _scc_members(analysis):
    return [tuple(s.entity_ids) for s in analysis.strongly_connected_components]


def _build_scenario(kernel):
    """A deterministic multi-entity scenario exercising the analysis surface.

    Entities: alpha (a), beta (b), gamma (c), delta (d, isolated).
    Interactions (committed MESSAGE_ENQUEUED facts):
      a -> b  (twice: repeated directed pair)
      b -> a  (opposite direction)
      a -> c  (payload carries a FAKE topology claim about d; no a->d edge)
      c -> c  (self-loop)
    d is isolated (no interactions) but must remain a node.
    All messages are then processed (popped from mailboxes) to prove edges
    derive from the committed EVENT HISTORY, not the live mailbox.
    """
    a = kernel.found_entity("alpha")
    b = kernel.found_entity("beta")
    c = kernel.found_entity("gamma")
    d = kernel.found_entity("delta")  # isolated
    kernel.run_until_quiescent()  # activate a, b, c, d
    pa = kernel.entity_port(a)
    pb = kernel.entity_port(b)
    pc = kernel.entity_port(c)
    pa.propose(b, b"m1")  # a -> b
    pa.propose(b, b"m2")  # a -> b (repeat)
    pb.propose(a, b"m3")  # b -> a (opposite)
    # Fake topology claim: payload text references d's real entity_id, but the
    # committed envelope attribution is a -> c. No a -> d edge may arise.
    pa.propose(c, b"connected to " + d.encode("utf-8"))  # a -> c
    pc.propose(c, b"self")  # c -> c (self-loop)
    kernel.run_until_quiescent()  # process all (pop from mailboxes)
    return {"a": a, "b": b, "c": c, "d": d}


def _runtime_dir():
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runtime")


# ---------------------------------------------------------------------------
# Positive cases 1-14
# ---------------------------------------------------------------------------


class TestPositiveCases:
    def test_1_empty_topology(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = analyze_topology(GENESIS, [], CAP)
        assert a.schema == ANALYSIS_SCHEMA
        assert a.node_count == 0
        assert a.edge_count == 0
        assert a.total_message_count == 0
        assert a.node_metrics == ()
        assert a.strongly_connected_components == ()
        assert a.condensation_edges == ()
        assert len(a.analysis_digest) == 64
        k.close()

    def test_2_one_isolated_founded_entity(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        k.run_until_quiescent()
        an = _analyze(k)
        assert an.node_count == 1
        assert an.edge_count == 0
        assert an.total_message_count == 0
        m = _metric_by_id(an, a)
        assert m.in_degree == 0
        assert m.out_degree == 0
        assert m.inbound_message_count == 0
        assert m.outbound_message_count == 0
        assert m.self_loop_message_count == 0
        # One singleton SCC, not cyclic (no self-loop).
        assert len(an.strongly_connected_components) == 1
        s = an.strongly_connected_components[0]
        assert s.entity_ids == (a,)
        assert s.size == 1
        assert s.cyclic is False
        assert s.scc_index == 0
        assert an.condensation_edges == ()
        k.close()

    def test_3_multiple_isolated_entities(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        k.run_until_quiescent()
        an = _analyze(k)
        assert an.node_count == 3
        assert an.edge_count == 0
        assert an.total_message_count == 0
        # Three singleton SCCs, each non-cyclic, sorted by member tuple.
        assert _scc_members(an) == [(a,), (b,), (c,)]
        for i, s in enumerate(an.strongly_connected_components):
            assert s.scc_index == i
            assert s.size == 1
            assert s.cyclic is False
        assert an.condensation_edges == ()
        k.close()

    def test_4_one_directed_edge(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        an = _analyze(k)
        assert an.node_count == 2
        assert an.edge_count == 1
        assert an.total_message_count == 1
        ma = _metric_by_id(an, a)
        mb = _metric_by_id(an, b)
        assert ma.out_degree == 1 and ma.in_degree == 0
        assert ma.outbound_message_count == 1 and ma.inbound_message_count == 0
        assert mb.in_degree == 1 and mb.out_degree == 0
        assert mb.inbound_message_count == 1 and mb.outbound_message_count == 0
        # Two singleton SCCs; one condensation edge a_scc -> b_scc.
        assert _scc_members(an) == [(a,), (b,)]
        ia = an.strongly_connected_components[0].scc_index
        ib = an.strongly_connected_components[1].scc_index
        assert an.condensation_edges == (CondensationEdge(ia, ib),)
        k.close()

    def test_5_directed_chain(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(c, b"m2")
        an = _analyze(k)
        assert an.node_count == 3
        assert an.edge_count == 2
        assert an.total_message_count == 2
        # Three singleton SCCs; two condensation edges (a->b, b->c).
        assert _scc_members(an) == [(a,), (b,), (c,)]
        ia = an.strongly_connected_components[0].scc_index
        ib = an.strongly_connected_components[1].scc_index
        ic = an.strongly_connected_components[2].scc_index
        assert an.condensation_edges == (
            CondensationEdge(ia, ib), CondensationEdge(ib, ic),
        )
        k.close()

    def test_6_directed_fork(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(a).propose(c, b"m2")
        an = _analyze(k)
        assert an.node_count == 3
        assert an.edge_count == 2
        assert an.total_message_count == 2
        ma = _metric_by_id(an, a)
        assert ma.out_degree == 2 and ma.in_degree == 0
        assert ma.outbound_message_count == 2
        # Three singleton SCCs; two condensation edges (a->b, a->c).
        assert _scc_members(an) == [(a,), (b,), (c,)]
        ia = an.strongly_connected_components[0].scc_index
        ib = an.strongly_connected_components[1].scc_index
        ic = an.strongly_connected_components[2].scc_index
        assert an.condensation_edges == (
            CondensationEdge(ia, ib), CondensationEdge(ia, ic),
        )
        k.close()

    def test_7_directed_cycle_one_scc(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(c, b"m2")
        k.entity_port(c).propose(a, b"m3")
        an = _analyze(k)
        assert an.node_count == 3
        assert an.edge_count == 3
        assert an.total_message_count == 3
        # One 3-node SCC, cyclic.
        assert len(an.strongly_connected_components) == 1
        s = an.strongly_connected_components[0]
        assert s.entity_ids == tuple(sorted((a, b, c)))
        assert s.size == 3
        assert s.cyclic is True
        assert s.scc_index == 0
        # No condensation edges (single SCC).
        assert an.condensation_edges == ()
        k.close()

    def test_8_two_sccs_one_way_bridge(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        d = k.found_entity("delta")
        k.run_until_quiescent()
        # SCC1: a<->b (2-node cycle). SCC2: c<->d (2-node cycle).
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(a, b"m2")
        k.entity_port(c).propose(d, b"m3")
        k.entity_port(d).propose(c, b"m4")
        # One-way bridge: b -> c.
        k.entity_port(b).propose(c, b"m5")
        an = _analyze(k)
        assert an.node_count == 4
        assert an.edge_count == 5
        assert an.total_message_count == 5
        # Two SCCs, each cyclic (size > 1).
        assert len(an.strongly_connected_components) == 2
        s1 = an.strongly_connected_components[0]
        s2 = an.strongly_connected_components[1]
        assert s1.entity_ids == tuple(sorted((a, b)))
        assert s2.entity_ids == tuple(sorted((c, d)))
        assert s1.cyclic is True and s2.cyclic is True
        assert s1.scc_index == 0 and s2.scc_index == 1
        # One condensation edge (SCC1 -> SCC2), from the b->c bridge.
        assert an.condensation_edges == (CondensationEdge(0, 1),)
        k.close()

    def test_9_opposite_pair_one_two_node_scc(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(a, b"m2")
        an = _analyze(k)
        assert an.node_count == 2
        assert an.edge_count == 2
        assert an.total_message_count == 2
        # One 2-node SCC, cyclic.
        assert len(an.strongly_connected_components) == 1
        s = an.strongly_connected_components[0]
        assert s.entity_ids == tuple(sorted((a, b)))
        assert s.size == 2
        assert s.cyclic is True
        assert s.scc_index == 0
        assert an.condensation_edges == ()
        k.close()

    def test_10_singleton_self_loop_cyclic(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        k.run_until_quiescent()
        k.entity_port(a).propose(a, b"self")
        an = _analyze(k)
        assert an.node_count == 1
        assert an.edge_count == 1
        assert an.total_message_count == 1
        m = _metric_by_id(an, a)
        # Self-loop contributes the node once to both distinct sets.
        assert m.in_degree == 1
        assert m.out_degree == 1
        # Self-loop contributes its edge count to both message counts.
        assert m.inbound_message_count == 1
        assert m.outbound_message_count == 1
        assert m.self_loop_message_count == 1
        # One singleton SCC, cyclic (self-loop).
        assert len(an.strongly_connected_components) == 1
        s = an.strongly_connected_components[0]
        assert s.entity_ids == (a,)
        assert s.size == 1
        assert s.cyclic is True
        assert s.scc_index == 0
        # Self-SCC edges are omitted from the condensation graph.
        assert an.condensation_edges == ()
        k.close()

    def test_11_repeated_messages_weighted_metrics(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(a).propose(b, b"m2")
        k.entity_port(a).propose(b, b"m3")
        an = _analyze(k)
        assert an.node_count == 2
        assert an.edge_count == 1  # repeated pair aggregates to ONE edge
        assert an.total_message_count == 3  # weighted message count
        ma = _metric_by_id(an, a)
        mb = _metric_by_id(an, b)
        # Weighted message metrics change with the count.
        assert ma.outbound_message_count == 3
        assert mb.inbound_message_count == 3
        # Distinct in/out degree remains one.
        assert ma.out_degree == 1
        assert mb.in_degree == 1
        # SCC structure does not change (two singleton SCCs).
        assert _scc_members(an) == [(a,), (b,)]
        k.close()

    def test_12_opposite_directions_distinct_edges(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(a, b"m2")
        an = _analyze(k)
        # Opposite directions remain distinct edge facts.
        assert an.edge_count == 2
        # They affect SCC membership correctly (one 2-node SCC).
        assert len(an.strongly_connected_components) == 1
        s = an.strongly_connected_components[0]
        assert s.entity_ids == tuple(sorted((a, b)))
        assert s.size == 2
        assert s.cyclic is True
        k.close()

    def test_13_isolated_entities_represented(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        an = _analyze(k)
        # d is isolated but remains in metrics and SCCs.
        md = _metric_by_id(an, ids["d"])
        assert md.in_degree == 0
        assert md.out_degree == 0
        assert md.inbound_message_count == 0
        assert md.outbound_message_count == 0
        assert md.self_loop_message_count == 0
        # d is in exactly one SCC (a singleton).
        members = _scc_members(an)
        assert (ids["d"],) in members
        k.close()

    def test_14_processed_messages_do_not_erase_structure(self, tmp_path):
        # Edges derive from the committed EVENT HISTORY, not the live mailbox.
        # After run_until_quiescent the mailboxes are empty, but the analysis
        # must still see the full structure (the ENQUEUED events remain).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)  # scenario processes all messages
        assert k.mailbox_size(ids["b"]) == 0
        assert k.mailbox_size(ids["a"]) == 0
        an = _analyze(k)
        # Full structure preserved despite empty mailboxes.
        assert an.edge_count == 4
        assert an.total_message_count == 5
        assert an.node_count == 4
        # a<->b is one 2-node SCC; c is a cyclic singleton (self-loop);
        # d is a non-cyclic singleton.
        members = _scc_members(an)
        assert tuple(sorted((ids["a"], ids["b"]))) in members
        assert (ids["c"],) in members
        assert (ids["d"],) in members
        k.close()


# ---------------------------------------------------------------------------
# SCC / condensation edge cases (explicit property tests)
# ---------------------------------------------------------------------------


class TestSccCondensation:
    def test_condensation_acyclic(self, tmp_path):
        # The condensation graph must be acyclic by construction. Test the
        # property explicitly on a scenario with multiple SCCs and bridges.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        an = _analyze(k)
        # Build the condensation adjacency and check for cycles via
        # deterministic topological sort (Kahn). If every node is emitted,
        # the graph is acyclic.
        scc_count = len(an.strongly_connected_components)
        adj = {i: set() for i in range(scc_count)}
        indeg = {i: 0 for i in range(scc_count)}
        for e in an.condensation_edges:
            adj[e.from_scc].add(e.to_scc)
            indeg[e.to_scc] += 1
        queue = sorted(i for i in range(scc_count) if indeg[i] == 0)
        emitted = 0
        while queue:
            n = queue.pop(0)
            emitted += 1
            for m in sorted(adj[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
            queue.sort()
        assert emitted == scc_count, "condensation graph has a cycle"
        k.close()

    def test_condensation_unique_edges(self, tmp_path):
        # Duplicate entity-level edges collapsing onto the same SCC pair
        # produce only one condensation edge.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        d = k.found_entity("delta")
        k.run_until_quiescent()
        # Two 2-node SCCs: a<->b and c<->d.
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(a, b"m2")
        k.entity_port(c).propose(d, b"m3")
        k.entity_port(d).propose(c, b"m4")
        # Two distinct entity-level bridges from SCC1 to SCC2: a->c and b->d.
        k.entity_port(a).propose(c, b"m5")
        k.entity_port(b).propose(d, b"m6")
        an = _analyze(k)
        # Two SCCs; the two bridges collapse onto ONE condensation edge.
        assert len(an.strongly_connected_components) == 2
        assert an.condensation_edges == (CondensationEdge(0, 1),)
        k.close()

    def test_scc_partition_complete(self, tmp_path):
        # Every founded entity belongs to exactly one SCC (partition).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        an = _analyze(k)
        all_members = [e for s in an.strongly_connected_components
                       for e in s.entity_ids]
        assert sorted(all_members) == sorted(ids.values())
        # No entity appears in more than one SCC.
        assert len(all_members) == len(set(all_members))
        k.close()

    def test_scc_index_assigned_after_sort(self, tmp_path):
        # scc_index is assigned only after the deterministic sort by the
        # tuple of member entity IDs.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        an = _analyze(k)
        members = [tuple(s.entity_ids) for s in an.strongly_connected_components]
        assert members == sorted(members)
        for i, s in enumerate(an.strongly_connected_components):
            assert s.scc_index == i
        k.close()

    def test_node_metrics_sorted_by_entity_id(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        ids = [m.entity_id for m in an.node_metrics]
        assert ids == sorted(ids)
        k.close()

    def test_condensation_edges_sorted(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        keys = [(e.from_scc, e.to_scc) for e in an.condensation_edges]
        assert keys == sorted(keys)
        k.close()


# ---------------------------------------------------------------------------
# Payload non-authority
# ---------------------------------------------------------------------------


class TestPayloadNonAuthority:
    def test_payload_claims_no_authority(self, tmp_path):
        # Payload text falsely claims relationships to other entities. The
        # structural analysis must remain unchanged unless the committed
        # envelope topology itself changes.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        d = k.found_entity("delta")
        k.run_until_quiescent()
        # a proposes to b with a payload claiming connections to c and d.
        k.entity_port(a).propose(
            b,
            b"connected to " + c.encode() +
            b" parent of " + d.encode() +
            b" trusts " + c.encode() +
            b" edge to " + d.encode(),
        )
        an = _analyze(k)
        # Only the committed a->b edge exists. No edges to c or d.
        assert an.edge_count == 1
        assert an.total_message_count == 1
        # c and d are isolated (no edges), so their metrics are all zero.
        mc = _metric_by_id(an, c)
        md = _metric_by_id(an, d)
        assert mc.in_degree == 0 and mc.out_degree == 0
        assert md.in_degree == 0 and md.out_degree == 0
        # The analysis is unchanged by the payload text.
        keys = [(m.entity_id, m.in_degree, m.out_degree)
                for m in an.node_metrics]
        assert (a, 0, 1) in keys
        assert (b, 1, 0) in keys
        k.close()

    def test_analysis_unchanged_when_payload_changes(self, tmp_path):
        # Two kernels with identical committed envelope topology but DIFFERENT
        # payload text produce identical analysis (payload has no authority).
        with tempfile.TemporaryDirectory() as d1, \
                tempfile.TemporaryDirectory() as d2:
            k1 = Kernel(d1, scheduler_protocol=SCHEDULER_V1).open()
            a1 = k1.found_entity("alpha")
            b1 = k1.found_entity("beta")
            k1.run_until_quiescent()
            k1.entity_port(a1).propose(b1, b"payload-text-one")
            an1 = _analyze(k1)
            k1.close()
            k2 = Kernel(d2, scheduler_protocol=SCHEDULER_V1).open()
            a2 = k2.found_entity("alpha")
            b2 = k2.found_entity("beta")
            k2.run_until_quiescent()
            k2.entity_port(a2).propose(b2, b"payload-text-two-different")
            an2 = _analyze(k2)
            k2.close()
            # Identical committed envelope topology -> identical analysis.
            assert an1.analysis_digest == an2.analysis_digest
            assert an1.to_dict() == an2.to_dict()


# ---------------------------------------------------------------------------
# Authority preservation (cannot bypass topology.py)
# ---------------------------------------------------------------------------


class TestAuthorityPreservation:
    def test_wrong_genesis_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        k.close()
        wrong = genesis_descriptor_digest("different-genesis")
        with pytest.raises(WrongAuthorityError):
            analyze_topology(wrong, events, CAP)

    def test_wrong_capacity_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        genesis = k._genesis_digest
        k.close()
        with pytest.raises(WrongAuthorityError):
            analyze_topology(genesis, events, 1)

    def test_corrupted_history_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Corrupt an event's payload_digest -> integrity failure.
        events[1]["payload_digest"] = "f" * 64
        with pytest.raises(CorruptEventError):
            analyze_topology(GENESIS, events, CAP)

    def test_forged_sender_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Forge the sender of a committed enqueue -> envelope integrity fails.
        enq = next(e for e in events if e["event_kind"] == "MESSAGE_ENQUEUED")
        enq["payload"]["envelope"]["sender_entity_id"] = "f" * 64
        with pytest.raises((EnvelopeError, CorruptEventError, BrokenChainError)):
            analyze_topology(GENESIS, events, CAP)

    def test_broken_clock_linkage_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        events[3]["logical_clock"] += 1
        with pytest.raises((BrokenChainError, CorruptEventError)):
            analyze_topology(GENESIS, events, CAP)

    def test_broken_prev_digest_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        events[2]["prev_event_digest"] = "e" * 64
        with pytest.raises((BrokenChainError, CorruptEventError)):
            analyze_topology(GENESIS, events, CAP)

    def test_semantically_impossible_lifecycle_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        k.run_until_quiescent()
        events = copy.deepcopy(k.events())
        k.close()
        act = next(e for e in events if e["event_kind"] == "ENTITY_ACTIVATED")
        act["payload"]["from"] = "ACTIVE"
        act["payload"]["to"] = "ACTIVE"
        with pytest.raises((ReplayError, CorruptEventError, BrokenChainError)):
            analyze_topology(GENESIS, events, CAP)

    def test_analyze_projection_rejects_non_projection(self):
        # The private helper is fail-closed on a non-projection record.
        with pytest.raises(TopologyAnalysisError):
            analyze_projection({"not": "a projection"})

    def test_invalid_preconditions(self, tmp_path):
        # Invalid preconditions are raised by the EXISTING qualified
        # topology-projection path (which analyze_topology delegates to), so
        # they surface as the existing TopologyError (fail-closed propagation,
        # not a new duplicate validator).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        k.close()
        with pytest.raises(TopologyError):
            analyze_topology("not-a-digest", events, CAP)
        with pytest.raises(TopologyError):
            analyze_topology(GENESIS, events, 0)
        with pytest.raises(TopologyError):
            analyze_topology(GENESIS, events, True)
        with pytest.raises(TopologyError):
            analyze_topology(GENESIS, "not-a-sequence", CAP)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_calls_identical(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        a1 = _analyze(k)
        a2 = _analyze(k)
        a3 = _analyze(k)
        assert a1.analysis_digest == a2.analysis_digest == a3.analysis_digest
        assert a1.to_dict() == a2.to_dict() == a3.to_dict()
        k.close()

    def test_independent_storage_dirs_identical(self):
        # Same scenario in two different storage dirs -> identical analysis.
        with tempfile.TemporaryDirectory() as d1, \
                tempfile.TemporaryDirectory() as d2:
            k1 = Kernel(d1, scheduler_protocol=SCHEDULER_V1).open()
            _build_scenario(k1)
            a1 = _analyze(k1)
            k1.close()
            k2 = Kernel(d2, scheduler_protocol=SCHEDULER_V1).open()
            _build_scenario(k2)
            a2 = _analyze(k2)
            k2.close()
            assert a1.analysis_digest == a2.analysis_digest
            assert a1.to_dict() == a2.to_dict()
            assert a1.canonical_bytes() == a2.canonical_bytes()

    def test_reopen_fresh_replay_identical(self, tmp_path):
        # Live analysis vs a FRESH kernel instance over the same durable
        # history: identical analysis.
        d = str(tmp_path)
        k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        a_live = _analyze(k)
        k.close()
        k2 = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        a_reopen = _analyze(k2)
        k2.close()
        assert a_live.analysis_digest == a_reopen.analysis_digest
        assert a_live.to_dict() == a_reopen.to_dict()

    def test_fresh_python_process_identical(self, tmp_path):
        # Live execution vs a FRESH PYTHON PROCESS analyzing the same durable
        # history: identical analysis digest.
        d = str(tmp_path)
        k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        a_live = _analyze(k)
        k.close()
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from elpis.ECS_C.kernel import Kernel\n"
"from elpis.ECS_C.scheduler import SCHEDULER_V1\n"
            "from elpis.ECS_C.topology_analysis import analyze_topology\n"
            "k = Kernel(%r, scheduler_protocol=SCHEDULER_V1).open()\n"
            "a = analyze_topology(k._genesis_digest, k.events(), k.mailbox_capacity)\n"
            "print(a.analysis_digest)\n"
            "k.close()\n"
        ) % (_runtime_dir(), d)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == a_live.analysis_digest

    def test_hash_seed_independence_subprocess(self):
        # Vary PYTHONHASHSEED in a FRESH PROCESS: analysis digest must not
        # change. Each run uses a fresh storage dir (the scenario is built
        # from scratch each time).
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from elpis.ECS_C.kernel import Kernel\n"
"from elpis.ECS_C.scheduler import SCHEDULER_V1\n"
            "from elpis.ECS_C.topology_analysis import analyze_topology\n"
            "d = %r\n"
            "k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()\n"
            "a = k.found_entity('alpha')\n"
            "b = k.found_entity('beta')\n"
            "c = k.found_entity('gamma')\n"
            "dd = k.found_entity('delta')\n"
            "k.run_until_quiescent()\n"
            "pa = k.entity_port(a); pb = k.entity_port(b); pc = k.entity_port(c)\n"
            "pa.propose(b, b'm1')\n"
            "pa.propose(b, b'm2')\n"
            "pb.propose(a, b'm3')\n"
            "pa.propose(c, b'connected to ' + dd.encode())\n"
            "pc.propose(c, b'self')\n"
            "k.run_until_quiescent()\n"
            "a = analyze_topology(k._genesis_digest, k.events(), k.mailbox_capacity)\n"
            "print(a.analysis_digest)\n"
            "k.close()\n"
        ) % (_runtime_dir(), "%s")
        digests = set()
        for seed in ("0", "1", "42"):
            with tempfile.TemporaryDirectory() as d:
                env = dict(os.environ, PYTHONHASHSEED=seed)
                result = subprocess.run(
                    [sys.executable, "-c", script % d],
                    capture_output=True, text=True, timeout=120, env=env,
                )
                assert result.returncode == 0, result.stderr
                digests.add(result.stdout.strip())
        assert len(digests) == 1, f"hash-seed dependence: {digests}"


# ---------------------------------------------------------------------------
# Non-mutation (pure, read-only; no second mutable state authority)
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_committed_event_bytes_unchanged(self, tmp_path):
        d = str(tmp_path)
        k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events_before = copy.deepcopy(k.events())
        root_before = k.state_root_digest()
        ids_before = k.entity_ids()
        mailboxes_before = {e: k.mailbox_size(e) for e in k.entity_ids()}
        # Repeated structural-analysis calls.
        _analyze(k)
        _analyze(k)
        _analyze(k)
        events_after = copy.deepcopy(k.events())
        assert events_before == events_after
        assert k.state_root_digest() == root_before
        assert k.entity_ids() == ids_before
        assert {e: k.mailbox_size(e) for e in k.entity_ids()} == mailboxes_before
        k.close()
        # Committed event bytes on disk remain identical.
        with open(k.log_path, "rb") as f:
            log_after = f.read()
        # Re-open and re-read the committed events to confirm byte identity.
        k2 = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        events_reopen = copy.deepcopy(k2.events())
        k2.close()
        assert events_reopen == events_before
        assert len(log_after) > 0

    def test_kernel_state_unchanged(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        root_before = k.state_root_digest()
        ids_before = k.entity_ids()
        _analyze(k)
        _analyze(k)  # repeated analysis must not accumulate state
        assert k.state_root_digest() == root_before
        assert k.entity_ids() == ids_before
        k.close()

    def test_no_second_mutable_authority(self, tmp_path):
        # The analysis is a pure function: analyzing the same history
        # repeatedly yields the identical frozen record (no accumulation, no
        # mutable state).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        genesis = k._genesis_digest
        cap = k.mailbox_capacity
        k.close()
        results = {analyze_topology(genesis, events, cap).analysis_digest
                   for _ in range(5)}
        assert len(results) == 1

    def test_input_events_not_mutated(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        events_before = copy.deepcopy(events)
        _analyze(k)
        events_after = copy.deepcopy(k.events())
        assert events_before == events_after
        k.close()


# ---------------------------------------------------------------------------
# Domain separation
# ---------------------------------------------------------------------------


class TestDomainSeparation:
    def test_digest_domain_separated(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        k.close()
        record = an.to_dict()
        expected = canonical.domain_digest(DOMAIN_ANALYSIS, record)
        assert an.analysis_digest == expected
        # The domain-separated digest differs from the undomained digest of
        # the same record (domain separation is real, not cosmetic).
        undomained = canonical.digest(record)
        assert an.analysis_digest != undomained

    def test_digest_differs_from_topology_digest(self, tmp_path):
        # The analysis digest is not equal to the topology digest.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        t = _project(k)
        k.close()
        assert an.analysis_digest != t.topology_digest
        # The analysis record binds the topology digest.
        assert an.topology_digest == t.topology_digest

    def test_digest_differs_from_topology_domain(self, tmp_path):
        # The analysis digest is not equal to a digest under the topology
        # domain of the same canonical analysis payload.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        k.close()
        topology_domain = canonical.domain_digest(
            DOMAIN_TOPOLOGY, an.to_dict())
        assert an.analysis_digest != topology_domain

    def test_digest_differs_from_state_root_domain(self, tmp_path):
        # The analysis digest is not equal to a state-root-domain digest of
        # the same canonical analysis payload.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        k.close()
        state_root_domain = canonical.domain_digest(
            canonical.DOMAIN_STATE_ROOT, an.to_dict())
        assert an.analysis_digest != state_root_domain


# ---------------------------------------------------------------------------
# Schema / versioning
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_versioned(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        assert an.schema == ANALYSIS_SCHEMA
        assert an.to_dict()["schema"] == ANALYSIS_SCHEMA
        k.close()

    def test_node_metric_fields(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        an = _analyze(k)
        m = _metric_by_id(an, a)
        assert isinstance(m, NodeMetric)
        assert m.to_dict() == {
            "entity_id": a,
            "in_degree": 0,
            "out_degree": 1,
            "inbound_message_count": 0,
            "outbound_message_count": 1,
            "self_loop_message_count": 0,
        }
        k.close()

    def test_scc_record_fields(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        k.entity_port(b).propose(a, b"m2")
        an = _analyze(k)
        s = an.strongly_connected_components[0]
        assert isinstance(s, SccRecord)
        assert s.to_dict() == {
            "scc_index": 0,
            "entity_ids": sorted((a, b)),
            "size": 2,
            "cyclic": True,
        }
        k.close()

    def test_condensation_edge_fields(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        an = _analyze(k)
        e = an.condensation_edges[0]
        assert isinstance(e, CondensationEdge)
        assert e.to_dict() == {"from_scc": 0, "to_scc": 1}
        k.close()

    def test_analysis_digest_helper(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        k.close()
        assert analysis_digest(an) == an.analysis_digest
        with pytest.raises(TopologyAnalysisError):
            analysis_digest("not-an-analysis")

    def test_analysis_binds_topology_digest(self, tmp_path):
        # The analysis record binds the exact topology digest it derived from.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        an = _analyze(k)
        t = _project(k)
        k.close()
        assert an.topology_digest == t.topology_digest
        assert an.to_dict()["topology_digest"] == t.topology_digest
