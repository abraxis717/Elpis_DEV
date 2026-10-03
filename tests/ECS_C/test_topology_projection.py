"""Adversarial qualification tests for the interaction-derived topology projection.

These tests establish that:
  * a canonical directed topology view is DETERMINISTICALLY DERIVED from
    committed, kernel-verifiable interaction history;
  * the projection is a PURE, READ-ONLY function (no mutation of events or
    kernel state; no second mutable state authority);
  * edges arise ONLY from committed MESSAGE_ENQUEUED envelope attribution,
    NEVER from application payload text (self-reported topology has no
    authority);
  * the projection FAILS CLOSED on wrong genesis, wrong capacity, broken
    linkage, corrupted events, and semantically impossible histories (reusing
    the existing authoritative replay/validation path);
  * the output is deterministic across dictionary/set order, hash seed,
    object address, filesystem path, wall-clock, and process ID.

The projection is read-only (interaction-derived topology).
These tests qualify it WITHOUT widening kernel authority: the projection grants
no mutation authority, is not semantic truth, and is not an independent
durable authority.
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
from elpis.ECS_C.replay import replay_from_events
from elpis.ECS_C.topology import (
    DOMAIN_TOPOLOGY,
    TOPOLOGY_SCHEMA,
    TopologyEdge,
    TopologyNode,
    TopologyProjection,
    TopologyError,
    project_topology,
    topology_digest,
)

GENESIS_LABEL = "ecs-m1a-genesis"
GENESIS = genesis_descriptor_digest(GENESIS_LABEL)
CAP = 16  # DEFAULT_MAILBOX_CAPACITY


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _build_scenario(kernel):
    """A deterministic multi-entity scenario exercising the topology surface.

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


def _project(kernel):
    """Project the topology from the kernel's committed history."""
    return project_topology(kernel._genesis_digest, kernel.events(), kernel.mailbox_capacity, scheduler_protocol=kernel.scheduler_protocol)


def _edge_key(e):
    return (e.sender_entity_id, e.receiver_entity_id)


# ---------------------------------------------------------------------------
# Basic projection
# ---------------------------------------------------------------------------


class TestBasicProjection:
    def test_empty_history(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        t = project_topology(GENESIS, [], CAP)
        assert t.schema == TOPOLOGY_SCHEMA
        assert t.genesis_digest == GENESIS
        assert t.mailbox_capacity == CAP
        assert t.event_count == 0
        assert t.nodes == ()
        assert t.edges == ()
        # Digest is well-formed (64-hex).
        assert len(t.topology_digest) == 64
        k.close()

    def test_founded_entities_no_edges(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()  # activate both
        t = _project(k)
        assert t.event_count > 0
        # Both founded entities are nodes.
        node_ids = {n.entity_id for n in t.nodes}
        assert node_ids == {a, b}
        # No interactions -> no edges.
        assert t.edges == ()
        k.close()

    def test_one_directed_interaction(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        t = _project(k)
        assert len(t.edges) == 1
        e = t.edges[0]
        assert isinstance(e, TopologyEdge)
        assert e.sender_entity_id == a
        assert e.receiver_entity_id == b
        assert e.count == 1
        assert e.self_loop is False
        k.close()

    def test_multiple_directed_interactions(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        # Edges: a->b, b->a, a->c, c->c. (No d edges; no a->d edge.)
        keys = {_edge_key(e) for e in t.edges}
        assert keys == {
            (ids["a"], ids["b"]),
            (ids["b"], ids["a"]),
            (ids["a"], ids["c"]),
            (ids["c"], ids["c"]),
        }
        k.close()

    def test_repeated_same_directed_pair(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        ab = next(e for e in t.edges if _edge_key(e) == (ids["a"], ids["b"]))
        assert ab.count == 2  # two committed a->b enqueues aggregate
        # first/last commit clock are well-ordered.
        assert ab.first_commit_clock <= ab.last_commit_clock
        k.close()

    def test_opposite_direction_distinct(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        ab = next(e for e in t.edges if _edge_key(e) == (ids["a"], ids["b"]))
        ba = next(e for e in t.edges if _edge_key(e) == (ids["b"], ids["a"]))
        # Opposite directions are DISTINCT directed edges.
        assert ab is not ba
        assert ab.sender_entity_id != ba.sender_entity_id
        assert ab.receiver_entity_id != ba.receiver_entity_id
        k.close()

    def test_self_interaction(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        cc = next(e for e in t.edges if _edge_key(e) == (ids["c"], ids["c"]))
        assert cc.self_loop is True
        assert cc.sender_entity_id == cc.receiver_entity_id == ids["c"]
        assert cc.count == 1
        k.close()

    def test_isolated_entity_represented(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        # d is isolated (no interactions) but remains a node.
        node_ids = {n.entity_id for n in t.nodes}
        assert ids["d"] in node_ids
        # d has no edges (neither as sender nor receiver).
        for e in t.edges:
            assert e.sender_entity_id != ids["d"]
            assert e.receiver_entity_id != ids["d"]
        k.close()

    def test_edges_survive_processing(self, tmp_path):
        # Edges derive from the committed EVENT HISTORY, not the live mailbox.
        # After run_until_quiescent the mailboxes are empty, but the edges
        # must still be present (the ENQUEUED events remain in history).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)  # scenario processes all messages
        # Mailboxes are now empty (all processed).
        assert k.mailbox_size(ids["b"]) == 0
        assert k.mailbox_size(ids["a"]) == 0
        t = _project(k)
        assert len(t.edges) == 4  # edges preserved despite empty mailboxes
        k.close()


# ---------------------------------------------------------------------------
# Payload authority (self-reported topology has no authority)
# ---------------------------------------------------------------------------


class TestPayloadAuthority:
    def test_fake_topology_claim_no_authority(self, tmp_path):
        # The a->c payload text contains d's real entity_id ("connected to
        # <d>"), but the committed envelope attribution is a->c. The fake
        # claim must NOT manufacture an a->d edge.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        ids = _build_scenario(k)
        t = _project(k)
        keys = {_edge_key(e) for e in t.edges}
        # No edge involving d as sender or receiver.
        assert (ids["a"], ids["d"]) not in keys
        assert (ids["d"], ids["a"]) not in keys
        assert (ids["d"], ids["c"]) not in keys
        # The a->c edge exists (from envelope attribution, not payload text).
        assert (ids["a"], ids["c"]) in keys
        k.close()

    def test_payload_text_never_read_for_edges(self, tmp_path):
        # Even a payload that is pure topology-claim text cannot create edges
        # beyond the committed envelope attribution.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        c = k.found_entity("gamma")
        k.run_until_quiescent()
        # a proposes to b with a payload claiming connections to c and a.
        k.entity_port(a).propose(
            b, b"I am connected to " + c.encode() + b" and " + a.encode())
        t = _project(k)
        keys = {_edge_key(e) for e in t.edges}
        # Only the committed a->b edge exists.
        assert keys == {(a, b)}
        k.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic_node_ordering(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t = _project(k)
        node_ids = [n.entity_id for n in t.nodes]
        assert node_ids == sorted(node_ids)
        k.close()

    def test_deterministic_edge_ordering(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t = _project(k)
        keys = [_edge_key(e) for e in t.edges]
        assert keys == sorted(keys)
        k.close()

    def test_deterministic_digest(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t1 = _project(k)
        t2 = _project(k)
        assert t1.topology_digest == t2.topology_digest
        assert t1.to_dict() == t2.to_dict()
        k.close()

    def test_pure_function_of_inputs(self, tmp_path):
        # Two independent projections from the same (genesis, events, cap)
        # give identical results (pure function, no accumulation).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        genesis = k._genesis_digest
        cap = k.mailbox_capacity
        k.close()
        t1 = project_topology(genesis, events, cap)
        t2 = project_topology(genesis, events, cap)
        assert t1.topology_digest == t2.topology_digest
        assert t1.to_dict() == t2.to_dict()

    def test_byte_identical_across_dirs(self):
        # Same scenario in two different storage dirs -> identical topology.
        with tempfile.TemporaryDirectory() as d1, \
                tempfile.TemporaryDirectory() as d2:
            k1 = Kernel(d1, scheduler_protocol=SCHEDULER_V1).open()
            _build_scenario(k1)
            t1 = _project(k1)
            k1.close()
            k2 = Kernel(d2, scheduler_protocol=SCHEDULER_V1).open()
            _build_scenario(k2)
            t2 = _project(k2)
            k2.close()
            assert t1.topology_digest == t2.topology_digest
            assert t1.to_dict() == t2.to_dict()
            # Canonical bytes identical.
            assert t1.canonical_bytes() == t2.canonical_bytes()

    def test_fresh_replay_reopen_equivalence(self, tmp_path):
        # Live projection vs a FRESH kernel instance over the same durable
        # history: identical topology.
        d = str(tmp_path)
        k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t_live = _project(k)
        k.close()
        # Fresh kernel instance over the same durable history.
        k2 = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        t_reopen = _project(k2)
        k2.close()
        assert t_live.topology_digest == t_reopen.topology_digest
        assert t_live.to_dict() == t_reopen.to_dict()

    def test_hash_seed_independence_subprocess(self, tmp_path):
        # Vary PYTHONHASHSEED in a FRESH PROCESS: topology digest must not
        # change. Each run uses a fresh storage dir (the scenario is built
        # from scratch each time).
        runtime_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "runtime")
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from elpis.ECS_C.kernel import Kernel\n"
"from elpis.ECS_C.scheduler import SCHEDULER_V1\n"
            "from elpis.ECS_C.topology import project_topology\n"
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
            "t = project_topology(k._genesis_digest, k.events(), k.mailbox_capacity)\n"
            "print(t.topology_digest)\n"
            "k.close()\n"
        ) % (runtime_dir, "%s")
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

    def test_fresh_process_replay(self, tmp_path):
        # Live execution vs a FRESH PYTHON PROCESS projecting the same durable
        # history: identical topology digest.
        d = str(tmp_path)
        k = Kernel(d, scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t_live = _project(k)
        k.close()
        runtime_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "runtime")
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from elpis.ECS_C.kernel import Kernel\n"
"from elpis.ECS_C.scheduler import SCHEDULER_V1\n"
            "from elpis.ECS_C.topology import project_topology\n"
            "k = Kernel(%r, scheduler_protocol=SCHEDULER_V1).open()\n"
            "t = project_topology(k._genesis_digest, k.events(), k.mailbox_capacity)\n"
            "print(t.topology_digest)\n"
            "k.close()\n"
        ) % (runtime_dir, d)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == t_live.topology_digest


# ---------------------------------------------------------------------------
# Fail-closed (reuse the existing authoritative replay/validation path)
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_wrong_genesis_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        k.close()
        wrong = genesis_descriptor_digest("different-genesis")
        with pytest.raises(WrongAuthorityError):
            project_topology(wrong, events, CAP)

    def test_wrong_capacity_rejection(self, tmp_path):
        # History built with capacity 16; projecting with capacity 1 must fail
        # closed (the first event's before_state_root binds the capacity).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        genesis = k._genesis_digest
        k.close()
        with pytest.raises(WrongAuthorityError):
            project_topology(genesis, events, 1)

    def test_broken_clock_linkage_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Mutate a later event's logical clock -> clock-progression violation.
        events[3]["logical_clock"] += 1
        with pytest.raises((BrokenChainError, CorruptEventError)):
            project_topology(GENESIS, events, CAP)

    def test_corrupt_payload_digest_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Corrupt an event's payload_digest -> integrity failure.
        events[1]["payload_digest"] = "f" * 64
        with pytest.raises(CorruptEventError):
            project_topology(GENESIS, events, CAP)

    def test_forged_envelope_sender_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Forge the sender of a committed enqueue -> envelope integrity fails
        # (message_id recomputed from the forged sender mismatches).
        enq = next(e for e in events if e["event_kind"] == "MESSAGE_ENQUEUED")
        enq["payload"]["envelope"]["sender_entity_id"] = "f" * 64
        with pytest.raises((EnvelopeError, CorruptEventError, BrokenChainError)):
            project_topology(GENESIS, events, CAP)

    def test_broken_prev_digest_rejection(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = copy.deepcopy(k.events())
        k.close()
        # Break the prev_event_digest linkage -> fail-closed. (The mutation
        # also breaks the event's self-digest, which is checked first; either
        # rejection class proves the history is rejected, not accepted.)
        events[2]["prev_event_digest"] = "e" * 64
        with pytest.raises((BrokenChainError, CorruptEventError)):
            project_topology(GENESIS, events, CAP)

    def test_semantically_impossible_lifecycle_rejection(self, tmp_path):
        # A history where an entity is activated twice (illegal lifecycle) is
        # semantically impossible and must fail closed.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        k.run_until_quiescent()  # activate a
        events = copy.deepcopy(k.events())
        k.close()
        # Force an illegal transition: from ACTIVE to ACTIVE is not allowed.
        act = next(e for e in events if e["event_kind"] == "ENTITY_ACTIVATED")
        act["payload"]["from"] = "ACTIVE"
        act["payload"]["to"] = "ACTIVE"
        with pytest.raises((ReplayError, CorruptEventError, BrokenChainError)):
            project_topology(GENESIS, events, CAP)


# ---------------------------------------------------------------------------
# Non-mutation (pure, read-only; no second mutable state authority)
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_input_events_not_mutated(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        events_before = copy.deepcopy(events)
        _project(k)
        events_after = copy.deepcopy(k.events())
        assert events_before == events_after
        k.close()

    def test_kernel_state_unchanged(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        root_before = k.state_root_digest()
        ids_before = k.entity_ids()
        _project(k)
        _project(k)  # repeated projection must not accumulate state
        assert k.state_root_digest() == root_before
        assert k.entity_ids() == ids_before
        k.close()

    def test_no_second_mutable_authority(self, tmp_path):
        # The projection is a pure function: projecting the same history
        # repeatedly yields the identical frozen record (no accumulation, no
        # mutable state).
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        genesis = k._genesis_digest
        cap = k.mailbox_capacity
        k.close()
        results = {project_topology(genesis, events, cap).topology_digest
                   for _ in range(5)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Domain separation
# ---------------------------------------------------------------------------


class TestDomainSeparation:
    def test_digest_domain_separated(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t = _project(k)
        k.close()
        record = t.to_dict()
        expected = canonical.domain_digest(DOMAIN_TOPOLOGY, record)
        assert t.topology_digest == expected
        # The domain-separated digest differs from the undomained digest of
        # the same record (domain separation is real, not cosmetic).
        undomained = canonical.digest(record)
        assert t.topology_digest != undomained

    def test_digest_differs_from_state_root_domain(self, tmp_path):
        # The topology digest is in its own domain; it must not collide with
        # the state-root domain digest of an equivalent payload.
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t = _project(k)
        k.close()
        from elpis.ECS_C import canonical as _c
        state_root_domain = _c.domain_digest(_c.DOMAIN_STATE_ROOT, t.to_dict())
        assert t.topology_digest != state_root_domain


# ---------------------------------------------------------------------------
# Schema / versioning
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_versioned(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        t = _project(k)
        assert t.schema == TOPOLOGY_SCHEMA
        assert t.to_dict()["schema"] == TOPOLOGY_SCHEMA
        k.close()

    def test_node_fields(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        k.run_until_quiescent()
        t = _project(k)
        node = next(n for n in t.nodes if n.entity_id == a)
        assert isinstance(node, TopologyNode)
        assert node.founding_index == 0
        assert node.lifecycle == "ACTIVE"
        assert node.to_dict() == {
            "entity_id": a, "founding_index": 0, "lifecycle": "ACTIVE",
        }
        k.close()

    def test_edge_fields(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        a = k.found_entity("alpha")
        b = k.found_entity("beta")
        k.run_until_quiescent()
        k.entity_port(a).propose(b, b"m1")
        t = _project(k)
        e = t.edges[0]
        assert isinstance(e, TopologyEdge)
        assert e.to_dict() == {
            "sender_entity_id": a,
            "receiver_entity_id": b,
            "count": 1,
            "first_commit_clock": e.first_commit_clock,
            "last_commit_clock": e.last_commit_clock,
            "self_loop": False,
        }
        k.close()

    def test_invalid_preconditions(self, tmp_path):
        k = Kernel(str(tmp_path), scheduler_protocol=SCHEDULER_V1).open()
        _build_scenario(k)
        events = k.events()
        k.close()
        # Invalid genesis digest.
        with pytest.raises(TopologyError):
            project_topology("not-a-digest", events, CAP)
        # Invalid capacity.
        with pytest.raises(TopologyError):
            project_topology(GENESIS, events, 0)
        with pytest.raises(TopologyError):
            project_topology(GENESIS, events, True)
        # Invalid events.
        with pytest.raises(TopologyError):
            project_topology(GENESIS, "not-a-sequence", CAP)
