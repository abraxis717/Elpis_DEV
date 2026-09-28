"""Elpis ECS — interaction-derived topology projection (read-only).

This module establishes ONE narrow capability:

    A canonical directed topology view can be DETERMINISTICALLY DERIVED from
    already-committed, kernel-verifiable interaction history.

It does NOT create a second mutable state authority. The projection is a pure
function of (genesis authority, committed event history, mailbox capacity):
it re-runs the EXISTING authoritative replay/validation path
(``replay_from_events``) and then folds the validated committed facts into a
frozen, versioned, canonical graph record. No new mutable state is created,
no existing state is mutated, and no self-reported topology observation is
accepted as an edge.

Edge semantics (exact)
----------------------
* A NODE is every entity identity ever founded in the validated history
  (every ``ENTITY_FOUNDED`` event). Isolated entities (no interactions) are
  represented as nodes with no edges.
* A DIRECTED EDGE ``sender -> receiver`` arises ONLY from a committed
  ``MESSAGE_ENQUEUED`` fact whose envelope attribution was validated by the
  replay path (kernel-owned sender attribution; sender ACTIVE at commit;
  receiver deliverable; envelope clock == event clock; watermark admitted).
  The edge is derived from the committed envelope's ``sender_entity_id`` and
  ``receiver_entity_id`` fields — NEVER from arbitrary application payload
  text. A payload that says "I am connected to X" has no effect unless an
  actual qualifying committed interaction fact independently establishes the
  edge.
* Repeated sender->receiver interactions AGGREGATE into one edge with a
  deterministic count and first/last commit-clock provenance. Opposite
  directions are DISTINCT directed edges.
* Self-interaction (sender == receiver) is represented as a self-loop edge
  when the kernel legally permits it (it does: an ACTIVE entity may propose
  to itself; the receiver is deliverable and the mailbox is bounded).

Nonclaims (preserved, explicit)
-------------------------------
* topology does NOT grant mutation authority;
* topology is NOT semantic truth;
* edge existence is NOT authentication or trust;
* topology does NOT authorize Structural R0;
* topology does NOT authorize Grid81;
* topology does NOT establish federation;
* topology does NOT establish cross-process identity;
* topology does NOT establish learned efficacy;
* topology is NOT an independent durable authority.

Determinism: the projection depends only on the validated committed facts.
It does not depend on dictionary insertion order, set iteration order, the
Python hash seed, object address, filesystem path, wall-clock time, or
process ID. All ordering is explicit (sorted by entity ID / commit clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import canonical
from .errors import EcsError
from .replay import KernelState, replay_from_events
from .scheduler import SCHEDULER_V1

# ---------------------------------------------------------------------------
# Schema / domain constants (explicit, versioned)
# ---------------------------------------------------------------------------

# The projection schema version. Bumped only on canonical-form change.
TOPOLOGY_SCHEMA = "ecs.topology.v1"

# Domain tag for the topology digest. Distinct from every other domain so the
# same payload under two domains yields two different digests.
DOMAIN_TOPOLOGY = "ecs.topology.v1"

# The event kind that establishes a directed interaction edge.
_EDGE_KIND = "MESSAGE_ENQUEUED"


class TopologyError(EcsError):
    """A topology projection precondition failed (fail-closed)."""


@dataclass(frozen=True)
class TopologyNode:
    """One founded entity identity (a node in the topology).

    ``entity_id`` is the stable domain-separated founding digest (an
    identifier, NOT authentication). ``founding_index`` is the kernel-assigned
    monotonic founding counter. ``lifecycle`` is the entity's lifecycle at the
    END of the validated history (a projection fact, not an authority).
    """

    entity_id: str
    founding_index: int
    lifecycle: str

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "founding_index": self.founding_index,
            "lifecycle": self.lifecycle,
        }


@dataclass(frozen=True)
class TopologyEdge:
    """One directed interaction edge, aggregated over repeated interactions.

    ``sender_entity_id`` -> ``receiver_entity_id``. ``count`` is the number of
    committed ``MESSAGE_ENQUEUED`` facts for this directed pair.
    ``first_commit_clock`` / ``last_commit_clock`` are the logical clocks of
    the first and last committed enqueues for this pair (deterministic
    provenance; the clock is the commit clock of the enqueue event).
    ``self_loop`` is True iff sender == receiver.
    """

    sender_entity_id: str
    receiver_entity_id: str
    count: int
    first_commit_clock: int
    last_commit_clock: int
    self_loop: bool

    def to_dict(self) -> dict:
        return {
            "sender_entity_id": self.sender_entity_id,
            "receiver_entity_id": self.receiver_entity_id,
            "count": self.count,
            "first_commit_clock": self.first_commit_clock,
            "last_commit_clock": self.last_commit_clock,
            "self_loop": self.self_loop,
        }


@dataclass(frozen=True)
class TopologyProjection:
    """The frozen, versioned, canonical topology projection.

    ``schema`` is the projection schema version. ``genesis_digest`` and
    ``mailbox_capacity`` bind the projection to the exact authority it was
    derived from (so two histories with different genesis/capacity cannot
    silently share a topology). ``event_count`` is the number of committed
    events in the validated history. ``nodes`` and ``edges`` are tuples in
    deterministic order (nodes by entity_id; edges by
    (sender_entity_id, receiver_entity_id)). ``topology_digest`` is the
    domain-separated canonical digest of the projection record.
    """

    schema: str
    genesis_digest: str
    mailbox_capacity: int
    event_count: int
    nodes: tuple
    edges: tuple
    topology_digest: str

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "genesis_digest": self.genesis_digest,
            "mailbox_capacity": self.mailbox_capacity,
            "event_count": self.event_count,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def canonical_bytes(self) -> bytes:
        return canonical.canonical_bytes(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TopologyProjection):
            return NotImplemented
        return (
            self.to_dict() == other.to_dict()
            and self.topology_digest == other.topology_digest
        )

    def __hash__(self) -> int:
        return hash(self.topology_digest)


# ---------------------------------------------------------------------------
# Projection construction (pure: validated history -> frozen projection)
# ---------------------------------------------------------------------------



def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def verify_projection(projection: TopologyProjection) -> None:
    """Recompute and fail-close the complete frozen projection record."""
    if type(projection) is not TopologyProjection:
        raise TopologyError(
            "TOPOLOGY_PROJECTION_INVALID: exact TopologyProjection required"
        )
    if projection.schema != TOPOLOGY_SCHEMA:
        raise TopologyError("TOPOLOGY_SCHEMA_INVALID")
    if not _is_digest(projection.genesis_digest):
        raise TopologyError("TOPOLOGY_GENESIS_INVALID")
    if (
        isinstance(projection.mailbox_capacity, bool)
        or not isinstance(projection.mailbox_capacity, int)
        or projection.mailbox_capacity < 1
    ):
        raise TopologyError("TOPOLOGY_CAPACITY_INVALID")
    if (
        isinstance(projection.event_count, bool)
        or not isinstance(projection.event_count, int)
        or projection.event_count < 0
    ):
        raise TopologyError("TOPOLOGY_EVENT_COUNT_INVALID")
    if not isinstance(projection.nodes, tuple):
        raise TopologyError("TOPOLOGY_NODES_INVALID: tuple required")
    if not isinstance(projection.edges, tuple):
        raise TopologyError("TOPOLOGY_EDGES_INVALID: tuple required")

    node_ids = []
    for node in projection.nodes:
        if type(node) is not TopologyNode:
            raise TopologyError("TOPOLOGY_NODE_INVALID")
        if not _is_digest(node.entity_id):
            raise TopologyError("TOPOLOGY_NODE_ID_INVALID")
        if (
            isinstance(node.founding_index, bool)
            or not isinstance(node.founding_index, int)
            or node.founding_index < 0
        ):
            raise TopologyError("TOPOLOGY_FOUNDING_INDEX_INVALID")
        if not isinstance(node.lifecycle, str) or not node.lifecycle:
            raise TopologyError("TOPOLOGY_LIFECYCLE_INVALID")
        node_ids.append(node.entity_id)

    if node_ids != sorted(node_ids) or len(node_ids) != len(set(node_ids)):
        raise TopologyError("TOPOLOGY_NODE_ORDER_OR_DUPLICATE_INVALID")
    node_set = set(node_ids)

    edge_keys = []
    total_messages = 0
    for edge in projection.edges:
        if type(edge) is not TopologyEdge:
            raise TopologyError("TOPOLOGY_EDGE_INVALID")
        if edge.sender_entity_id not in node_set:
            raise TopologyError("TOPOLOGY_EDGE_DANGLING_SENDER")
        if edge.receiver_entity_id not in node_set:
            raise TopologyError("TOPOLOGY_EDGE_DANGLING_RECEIVER")
        if (
            isinstance(edge.count, bool)
            or not isinstance(edge.count, int)
            or edge.count < 1
        ):
            raise TopologyError("TOPOLOGY_EDGE_COUNT_INVALID")
        if (
            isinstance(edge.first_commit_clock, bool)
            or not isinstance(edge.first_commit_clock, int)
            or isinstance(edge.last_commit_clock, bool)
            or not isinstance(edge.last_commit_clock, int)
            or edge.first_commit_clock < 0
            or edge.last_commit_clock < edge.first_commit_clock
        ):
            raise TopologyError("TOPOLOGY_EDGE_CLOCK_INVALID")
        expected_self_loop = edge.sender_entity_id == edge.receiver_entity_id
        if type(edge.self_loop) is not bool or edge.self_loop != expected_self_loop:
            raise TopologyError("TOPOLOGY_EDGE_SELF_LOOP_INVALID")
        edge_keys.append((edge.sender_entity_id, edge.receiver_entity_id))
        total_messages += edge.count

    if edge_keys != sorted(edge_keys) or len(edge_keys) != len(set(edge_keys)):
        raise TopologyError("TOPOLOGY_EDGE_ORDER_OR_DUPLICATE_INVALID")
    if total_messages > projection.event_count:
        raise TopologyError("TOPOLOGY_MESSAGE_COUNT_EXCEEDS_EVENT_COUNT")
    if not _is_digest(projection.topology_digest):
        raise TopologyError("TOPOLOGY_DIGEST_INVALID")

    expected_digest = canonical.domain_digest(
        DOMAIN_TOPOLOGY,
        projection.to_dict(),
    )
    if projection.topology_digest != expected_digest:
        raise TopologyError("TOPOLOGY_DIGEST_MISMATCH")


def _fold_edges(events: Sequence[Mapping[str, Any]]) -> tuple:
    """Fold the validated committed interaction facts into directed edges.

    Reads ONLY the committed ``MESSAGE_ENQUEUED`` events' envelope
    attribution (``sender_entity_id`` / ``receiver_entity_id``) — NEVER the
    application payload text. These events have already passed the
    authoritative replay/validation path (kernel-owned sender attribution,
    sender ACTIVE at commit, receiver deliverable, envelope clock == event
    clock, watermark admitted), so the envelope fields are the committed
    attribution. Edges are aggregated per directed (sender, receiver) pair
    and returned in deterministic order.

    Deriving from the EVENT HISTORY (not the live mailbox) is essential:
    processed messages are popped from the mailbox, so the mailbox would lose
    edges. The committed event history is the authoritative source.
    """
    # Aggregate per directed pair. Use a plain dict keyed by (sender,
    # receiver); the FINAL order is determined by an explicit sort, so dict
    # insertion order is irrelevant.
    agg: dict = {}
    for ev in events:
        if ev["event_kind"] != _EDGE_KIND:
            continue
        env = ev["payload"]["envelope"]
        sender = env["sender_entity_id"]
        receiver = env["receiver_entity_id"]
        # The envelope clock is the commit clock of the enqueue event
        # (verified equal to the event clock by replay).
        clock = env["logical_clock"]
        key = (sender, receiver)
        slot = agg.get(key)
        if slot is None:
            agg[key] = {"count": 1, "first": clock, "last": clock}
        else:
            slot["count"] += 1
            # The commit clock is monotonic per event, so the first observed
            # is the minimum and the last observed is the maximum. (Explicit
            # min/max for clarity and safety.)
            if clock < slot["first"]:
                slot["first"] = clock
            if clock > slot["last"]:
                slot["last"] = clock
    edges = []
    for (sender, receiver) in sorted(agg):
        slot = agg[(sender, receiver)]
        edges.append(TopologyEdge(
            sender_entity_id=sender,
            receiver_entity_id=receiver,
            count=slot["count"],
            first_commit_clock=slot["first"],
            last_commit_clock=slot["last"],
            self_loop=(sender == receiver),
        ))
    return tuple(edges)


def _fold_nodes(state: KernelState) -> tuple:
    """Fold the validated founded entities into nodes (deterministic order)."""
    nodes = []
    for eid in state.registry.ids_sorted():
        rec = state.registry.get(eid)
        nodes.append(TopologyNode(
            entity_id=eid,
            founding_index=rec.founding_index,
            lifecycle=rec.lifecycle,
        ))
    return tuple(nodes)



def _project_topology_from_validated_state(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    mailbox_capacity: int,
    state: KernelState,
) -> TopologyProjection:
    """Project topology from one already-authoritative live Kernel snapshot.

    This internal path is only for the live Kernel, which owns both the
    projection state and EventLog. It does not replace the public
    ``project_topology`` validator for caller-supplied histories.

    The EventLog snapshot has already passed frame/schema/digest/link checks.
    Before folding, bind that exact snapshot back to the live projection:
    state genesis/capacity must match; event count must equal logical clock;
    and for non-empty history the live state root must equal the final event's
    committed ``after_state_root``.
    """
    if type(state) is not KernelState:
        raise TopologyError(
            "TOPOLOGY_LIVE_STATE_INVALID: exact KernelState required"
        )
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TopologyError("TOPOLOGY_EVENTS_INVALID: must be a sequence of events")
    events = tuple(events)

    if state.genesis_digest != genesis_digest:
        raise TopologyError("TOPOLOGY_LIVE_STATE_GENESIS_MISMATCH")
    if state.mailbox_capacity != mailbox_capacity:
        raise TopologyError("TOPOLOGY_LIVE_STATE_CAPACITY_MISMATCH")
    if state.logical_clock != len(events):
        raise TopologyError("TOPOLOGY_LIVE_STATE_EVENT_COUNT_MISMATCH")

    if events:
        live_root = state.state_root_digest()
        if live_root != events[-1]["after_state_root"]:
            raise TopologyError("TOPOLOGY_LIVE_STATE_ROOT_MISMATCH")

    nodes = _fold_nodes(state)
    edges = _fold_edges(events)
    record = {
        "schema": TOPOLOGY_SCHEMA,
        "genesis_digest": genesis_digest,
        "mailbox_capacity": mailbox_capacity,
        "event_count": len(events),
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }
    topology_digest = canonical.domain_digest(DOMAIN_TOPOLOGY, record)
    return TopologyProjection(
        schema=TOPOLOGY_SCHEMA,
        genesis_digest=genesis_digest,
        mailbox_capacity=mailbox_capacity,
        event_count=len(events),
        nodes=nodes,
        edges=edges,
        topology_digest=topology_digest,
    )

def project_topology(
    genesis_digest: str,
    events: Sequence[Mapping[str, Any]],
    mailbox_capacity: int,
    scheduler_protocol: str = SCHEDULER_V1,
) -> TopologyProjection:
    """Derive the canonical topology projection from committed history.

    This is a PURE, READ-ONLY projection. It:
      1. re-runs the EXISTING authoritative replay/validation path
         (``replay_from_events``) so that wrong genesis, wrong capacity,
         malformed transitions, broken sequence/clock/root linkage, invalid
         lifecycle behavior, or corrupted events retain their EXISTING
         failure semantics (fail-closed);
      2. folds the validated committed facts into a frozen, versioned,
         canonical graph record;
      3. computes a domain-separated topology digest.

    It does NOT mutate the supplied events or any kernel state. It does NOT
    create a second mutable state authority. It does NOT accept self-reported
    topology as an edge.

    Raises:
        WrongAuthorityError / BrokenChainError / CorruptEventError /
        ReplayError / ... : the existing replay failure classes, propagated
            unchanged (fail-closed).
        TopologyError: a topology-specific precondition failed.
    """
    if not isinstance(genesis_digest, str) or len(genesis_digest) != 64:
        raise TopologyError("TOPOLOGY_GENESIS_INVALID: must be a 64-hex digest")
    if isinstance(mailbox_capacity, bool) or not isinstance(mailbox_capacity, int) \
            or not 1 <= mailbox_capacity:
        raise TopologyError("TOPOLOGY_CAPACITY_INVALID: must be a positive int")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TopologyError("TOPOLOGY_EVENTS_INVALID: must be a sequence of events")

    # Snapshot the caller-supplied sequence ONCE. Replay, folding, event_count,
    # and digest construction must observe one identical committed-history view.
    events = tuple(events)

    # Step 1: the EXISTING authoritative replay/validation path. This is the
    # ONLY validation used; no weaker duplicate validator is invented.
    state = replay_from_events(genesis_digest, events, mailbox_capacity, scheduler_protocol=scheduler_protocol)

    # Step 2: fold the validated committed facts (read-only).
    nodes = _fold_nodes(state)
    edges = _fold_edges(events)

    # Step 3: canonical record + domain-separated digest.
    record = {
        "schema": TOPOLOGY_SCHEMA,
        "genesis_digest": genesis_digest,
        "mailbox_capacity": mailbox_capacity,
        "event_count": len(events),
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }
    topology_digest = canonical.domain_digest(DOMAIN_TOPOLOGY, record)

    return TopologyProjection(
        schema=TOPOLOGY_SCHEMA,
        genesis_digest=genesis_digest,
        mailbox_capacity=mailbox_capacity,
        event_count=len(events),
        nodes=nodes,
        edges=edges,
        topology_digest=topology_digest,
    )


def topology_digest(projection: TopologyProjection) -> str:
    """Return the recomputation-verified topology digest."""
    verify_projection(projection)
    return projection.topology_digest
