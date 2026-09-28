"""Elpis ECS — deterministic structural analysis of the qualified topology.

This module establishes ONE narrow capability:

    The committed ECS interaction history admits DETERMINISTIC, REPRODUCIBLE
    graph-structural descriptors derived through the EXISTING qualified
    topology-projection authority (``topology.project_topology``).

It is a PURE DERIVED VIEW of the already-qualified topology projection:

* it creates NO second mutable state authority;
* it grants NO mutation authority;
* it assigns NO semantic, trust, or causal meaning to graph structure.

Authority rule (exact)
----------------------
The public entrypoint (``analyze_topology``) obtains the topology ONLY by
calling the existing qualified topology-projection entrypoint
(``topology.project_topology``). It does NOT:

* replay event history itself (no duplicate replay authority);
* scan storage as an alternative authority;
* parse ``MESSAGE_ENQUEUED`` events independently;
* accept a caller-supplied arbitrary topology record as authoritative;
* inspect payload text;
* infer edges from entity self-report;
* mutate ECS state;
* persist analysis output.

A private pure helper (``analyze_projection``) analyzes an ALREADY-qualified
projection record. It is exposed for implementation/tests only; the
authority-bearing public entrypoint always derives through ``topology.py``.
Existing replay/topology failures propagate fail-closed rather than being
converted into a partial graph.

Nonclaims (preserved, explicit)
-------------------------------
* topology is NOT semantic truth;
* an edge does NOT authenticate either endpoint;
* an SCC is NOT a community, coalition, organism, agent, or trusted group;
* degree does NOT imply importance or influence;
* cyclicity does NOT imply cooperation;
* structural position does NOT establish causation;
* graph structure does NOT authorize mutation;
* graph structure does NOT authorize Structural R0 or Grid81;
* graph structure does NOT establish federation or cross-process identity;
* graph structure does NOT establish learned efficacy;
* the analysis is NOT an independent durable authority.

It is only a deterministic structural descriptor of the already-qualified
interaction topology.

Determinism: the analysis depends only on the qualified projection record.
It does not depend on dictionary insertion order, set iteration order, the
Python hash seed, object address, filesystem path, wall-clock time, or
process ID. All ordering is explicit (sorted by entity ID / SCC member
tuple / SCC index).
"""

from __future__ import annotations

from .scheduler import SCHEDULER_V1

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import canonical
from .errors import EcsError
from .topology import (
    DOMAIN_TOPOLOGY,
    TOPOLOGY_SCHEMA,
    TopologyError,
    TopologyProjection,
    project_topology,
    verify_projection,
)

# ---------------------------------------------------------------------------
# Schema / domain constants (explicit, versioned)
# ---------------------------------------------------------------------------

# The analysis schema version. Bumped only on canonical-form change.
ANALYSIS_SCHEMA = "ecs.topology-analysis.v1"

# Domain tag for the analysis digest. Distinct from the topology domain
# (``ecs.topology.v1``) and every other domain, so the same payload under two
# domains yields two different digests.
DOMAIN_ANALYSIS = "ecs.topology.analysis.v1"


class TopologyAnalysisError(EcsError):
    """A topology-analysis precondition failed (fail-closed)."""


# ---------------------------------------------------------------------------
# Canonical analysis records (frozen, versioned)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeMetric:
    """Deterministic structural metrics for one topology node.

    ``in_degree`` is the number of DISTINCT predecessor entity IDs with an
    edge into this node; ``out_degree`` is the number of DISTINCT successor
    entity IDs with an edge out of this node. A self-loop contributes the
    node itself ONCE to both distinct predecessor and successor sets.
    ``inbound_message_count`` / ``outbound_message_count`` use the topology
    edge ``count`` weights; a self-loop contributes its edge count to BOTH
    inbound and outbound message counts. ``self_loop_message_count`` is the
    count weight of the node->same-node edge, or zero.

    These are structural counts only. They are NOT importance, influence,
    rank, leader, centrality, trust, or fitness.
    """

    entity_id: str
    in_degree: int
    out_degree: int
    inbound_message_count: int
    outbound_message_count: int
    self_loop_message_count: int

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "inbound_message_count": self.inbound_message_count,
            "outbound_message_count": self.outbound_message_count,
            "self_loop_message_count": self.self_loop_message_count,
        }


@dataclass(frozen=True)
class SccRecord:
    """One exact directed strongly connected component.

    ``entity_ids`` is the lexicographically sorted member tuple. ``size`` is
    the member count. ``cyclic`` is True iff the SCC has size > 1, or the
    singleton node has a self-loop; otherwise False. ``scc_index`` is
    assigned only AFTER the deterministic sort of the SCC list by the tuple
    of member entity IDs.

    An SCC is a structural equivalence class of mutual reachability. It is
    NOT a community, coalition, organism, agent, or trusted group.
    """

    scc_index: int
    entity_ids: tuple
    size: int
    cyclic: bool

    def to_dict(self) -> dict:
        return {
            "scc_index": self.scc_index,
            "entity_ids": list(self.entity_ids),
            "size": self.size,
            "cyclic": self.cyclic,
        }


@dataclass(frozen=True)
class CondensationEdge:
    """One unique directed condensation edge between two SCCs.

    Self-SCC edges are omitted; duplicate entity-level edges collapsing onto
    the same SCC pair produce only one condensation edge. The condensation
    graph is acyclic by construction (SCCs are the maximal strongly
    connected components of the directed graph).
    """

    from_scc: int
    to_scc: int

    def to_dict(self) -> dict:
        return {"from_scc": self.from_scc, "to_scc": self.to_scc}


@dataclass(frozen=True)
class TopologyAnalysis:
    """The frozen, versioned, canonical structural-analysis record.

    ``schema`` is the analysis schema version. ``topology_digest`` binds the
    analysis to the EXACT qualified topology projection it was derived from
    (so two different topologies cannot silently share an analysis).
    ``node_count`` / ``edge_count`` / ``total_message_count`` are structural
    counts of the projection (``total_message_count`` is the sum of the
    topology edge ``count`` weights, i.e. the number of committed
    ``MESSAGE_ENQUEUED`` facts). ``node_metrics`` is sorted by
    ``entity_id``; ``strongly_connected_components`` is sorted by the tuple
    of member entity IDs (``scc_index`` assigned after that sort);
    ``condensation_edges`` is sorted by ``(from_scc, to_scc)``.
    ``analysis_digest`` is the domain-separated canonical digest of the
    complete canonical record.

    The record contains no volatile filesystem paths, object identities,
    Python hash values, timestamps, or runtime-specific metadata.
    """

    schema: str
    topology_digest: str
    node_count: int
    edge_count: int
    total_message_count: int
    node_metrics: tuple
    strongly_connected_components: tuple
    condensation_edges: tuple
    analysis_digest: str

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "topology_digest": self.topology_digest,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "total_message_count": self.total_message_count,
            "node_metrics": [m.to_dict() for m in self.node_metrics],
            "strongly_connected_components": [
                s.to_dict() for s in self.strongly_connected_components
            ],
            "condensation_edges": [
                e.to_dict() for e in self.condensation_edges
            ],
        }

    def canonical_bytes(self) -> bytes:
        return canonical.canonical_bytes(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TopologyAnalysis):
            return NotImplemented
        return (
            self.to_dict() == other.to_dict()
            and self.analysis_digest == other.analysis_digest
        )

    def __hash__(self) -> int:
        return hash(self.analysis_digest)


# ---------------------------------------------------------------------------
# Private pure helper: already-qualified projection -> frozen analysis
# ---------------------------------------------------------------------------



def _is_analysis_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def verify_analysis(analysis: TopologyAnalysis) -> None:
    """Recompute and fail-close the frozen analysis record."""
    if type(analysis) is not TopologyAnalysis:
        raise TopologyAnalysisError(
            "ANALYSIS_RECORD_INVALID: exact TopologyAnalysis required"
        )
    if analysis.schema != ANALYSIS_SCHEMA:
        raise TopologyAnalysisError("ANALYSIS_SCHEMA_INVALID")
    if not _is_analysis_digest(analysis.topology_digest):
        raise TopologyAnalysisError("ANALYSIS_TOPOLOGY_DIGEST_INVALID")
    if not _is_analysis_digest(analysis.analysis_digest):
        raise TopologyAnalysisError("ANALYSIS_DIGEST_INVALID")
    expected = canonical.domain_digest(
        DOMAIN_ANALYSIS,
        analysis.to_dict(),
    )
    if analysis.analysis_digest != expected:
        raise TopologyAnalysisError("ANALYSIS_DIGEST_MISMATCH")


def _strongly_connected_components(nodes: Sequence[str], edges: Sequence) -> tuple:
    """Exact directed SCCs over the projected topology (iterative Tarjan).

    Bounded to O(V + E) aside from deterministic sorting. Every node belongs
    to exactly one SCC, including isolated nodes. Returns the SCC list as
    tuples of lexicographically sorted member entity IDs, sorted by that
    member tuple (``scc_index`` is assigned by the caller after this sort).
    """
    # Adjacency: node -> sorted tuple of successor nodes (deduplicated).
    # ``set`` is used only for membership; every emitted order is explicit.
    adj: dict = {n: set() for n in nodes}
    for e in edges:
        adj[e.sender_entity_id].add(e.receiver_entity_id)
    adj = {n: tuple(sorted(s)) for n, s in adj.items()}

    index: dict = {}
    lowlink: dict = {}
    on_stack: set = set()
    stack: list = []
    counter = [0]
    result: list = []

    for root in sorted(nodes):
        if root in index:
            continue
        # Iterative Tarjan. Each frame is (node, successor iterator).
        work: list = [(root, iter(adj[root]))]
        index[root] = lowlink[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for succ in it:
                if succ not in index:
                    index[succ] = lowlink[succ] = counter[0]
                    counter[0] += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, iter(adj[succ])))
                    advanced = True
                    break
                if succ in on_stack:
                    lowlink[node] = min(lowlink[node], index[succ])
            if advanced:
                continue
            # All successors processed: pop the frame.
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                result.append(tuple(sorted(comp)))
    return tuple(sorted(result))


def analyze_projection(projection: TopologyProjection) -> TopologyAnalysis:
    """Analyze an ALREADY-qualified topology projection (private pure helper).

    This helper does NOT re-derive topology authority. It accepts an existing
    ``TopologyProjection`` and immediately verifies that projection's
    structural/digest contract with ``verify_projection`` before analysis.
    The authority-bearing public entrypoint (``analyze_topology``) remains the
    only path that derives the projection through ``topology.py``.

    Raises:
        TopologyAnalysisError: the supplied record is not a
            ``TopologyProjection`` (fail-closed).
    """
    try:
        verify_projection(projection)
    except TopologyError as exc:
        raise TopologyAnalysisError(
            f"ANALYSIS_PROJECTION_INVALID: {exc}"
        ) from exc

    node_ids = [n.entity_id for n in projection.nodes]
    edges = projection.edges

    # --- Node metrics (sorted by entity_id) --------------------------------
    pred: dict = {n: set() for n in node_ids}
    succ: dict = {n: set() for n in node_ids}
    inbound: dict = {n: 0 for n in node_ids}
    outbound: dict = {n: 0 for n in node_ids}
    self_loop_count: dict = {n: 0 for n in node_ids}
    for e in edges:
        pred[e.receiver_entity_id].add(e.sender_entity_id)
        succ[e.sender_entity_id].add(e.receiver_entity_id)
        inbound[e.receiver_entity_id] += e.count
        outbound[e.sender_entity_id] += e.count
        if e.self_loop:
            self_loop_count[e.sender_entity_id] += e.count
    node_metrics = tuple(
        NodeMetric(
            entity_id=n,
            in_degree=len(pred[n]),
            out_degree=len(succ[n]),
            inbound_message_count=inbound[n],
            outbound_message_count=outbound[n],
            self_loop_message_count=self_loop_count[n],
        )
        for n in sorted(node_ids)
    )

    # --- Strongly connected components (deterministic sort, then index) ----
    scc_members = _strongly_connected_components(node_ids, edges)
    self_loop_nodes = {
        e.sender_entity_id for e in edges if e.self_loop
    }
    scc_records = tuple(
        SccRecord(
            scc_index=i,
            entity_ids=members,
            size=len(members),
            cyclic=(len(members) > 1) or (members[0] in self_loop_nodes),
        )
        for i, members in enumerate(scc_members)
    )

    # --- Condensation graph (unique directed edges, acyclic by construction)
    node_to_scc: dict = {}
    for i, members in enumerate(scc_members):
        for m in members:
            node_to_scc[m] = i
    cond_set: set = set()
    for e in edges:
        fs = node_to_scc[e.sender_entity_id]
        ts = node_to_scc[e.receiver_entity_id]
        if fs != ts:
            cond_set.add((fs, ts))
    condensation_edges = tuple(
        CondensationEdge(from_scc=fs, to_scc=ts) for fs, ts in sorted(cond_set)
    )

    # --- Canonical record + domain-separated digest ------------------------
    total_message_count = sum(e.count for e in edges)
    record = {
        "schema": ANALYSIS_SCHEMA,
        "topology_digest": projection.topology_digest,
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "total_message_count": total_message_count,
        "node_metrics": [m.to_dict() for m in node_metrics],
        "strongly_connected_components": [
            s.to_dict() for s in scc_records
        ],
        "condensation_edges": [
            e.to_dict() for e in condensation_edges
        ],
    }
    analysis_digest = canonical.domain_digest(DOMAIN_ANALYSIS, record)

    return TopologyAnalysis(
        schema=ANALYSIS_SCHEMA,
        topology_digest=projection.topology_digest,
        node_count=len(node_ids),
        edge_count=len(edges),
        total_message_count=total_message_count,
        node_metrics=node_metrics,
        strongly_connected_components=scc_records,
        condensation_edges=condensation_edges,
        analysis_digest=analysis_digest,
    )


# ---------------------------------------------------------------------------
# Public authority-bearing entrypoint (derives through topology.py)
# ---------------------------------------------------------------------------


def analyze_topology(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    mailbox_capacity: int,
    scheduler_protocol: str = SCHEDULER_V1,
) -> TopologyAnalysis:
    """Derive the deterministic structural analysis of committed history.

    This is a PURE, READ-ONLY derived view. It:
      1. obtains the topology ONLY through the EXISTING qualified
         topology-projection entrypoint (``topology.project_topology``), so
         that wrong genesis, wrong capacity, malformed transitions, broken
         sequence/clock/root linkage, invalid lifecycle behavior, or corrupted
         events retain their EXISTING failure semantics (fail-closed);
      2. derives the structural descriptors (node metrics, exact SCCs,
         condensation graph) from that qualified projection;
      3. computes a domain-separated analysis digest over the complete
         canonical record.

    It does NOT replay event history itself, scan storage, parse
    ``MESSAGE_ENQUEUED`` events independently, accept a caller-supplied
    topology record as authoritative, inspect payload text, infer edges from
    entity self-report, mutate ECS state, or persist analysis output.

    Raises:
        WrongAuthorityError / BrokenChainError / CorruptEventError /
        ReplayError / ... : the existing replay failure classes, propagated
            unchanged (fail-closed).
        TopologyError: a topology-specific precondition failed.
        TopologyAnalysisError: an analysis-specific precondition failed.
    """
    # Step 1: the EXISTING qualified topology-projection authority. This is
    # the ONLY derivation path; no weaker duplicate validator is invented.
    projection = project_topology(genesis_digest, events, mailbox_capacity, scheduler_protocol=scheduler_protocol)

    # Step 2: pure structural analysis of the qualified projection.
    return analyze_projection(projection)


def analysis_digest(analysis: TopologyAnalysis) -> str:
    """Return the recomputation-verified analysis digest."""
    verify_analysis(analysis)
    return analysis.analysis_digest
