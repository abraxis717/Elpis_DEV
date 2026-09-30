"""One public-ECS experiment: distinct valid histories with one coarse state and different futures. RESEARCH_ONLY.

Everything here goes through the public kernel API (``Kernel``, ``entity_port``,
``step``, ``events``) and the public topology-analysis entry point, in
temporary storage created and deleted by this module. No production module is
changed. The laboratory never writes to an existing history: futures are
explored on byte copies of a closed history's storage directory.

Coarse state ``C``: the topology analysis (node metrics, SCCs, condensation
edges and counts) with its ``topology_digest`` removed. That digest binds the
exact topology projection, whose edges carry first and last commit clocks, so
keeping it would make ``C`` nearly injective. With it removed, ``C`` depends
only on the multiset of committed enqueues. Two histories that enqueue the
same messages in a different order therefore share ``C`` exactly.

Targets:

* EVENT: the (sender, receiver) edge of the next ``MESSAGE_PROCESSED`` after
  one kernel step;
* K_STEP_TRAJECTORY: the sequence of ``C`` after each of ``k`` rounds of a host
  continuation policy (step once; the receiver of the processed message
  replies to its sender). The policy is the laboratory's host code, not an
  ECS rule. The whole sequence is the target: coarse futures may diverge and
  later re-converge.

Representations: ``C_t``; a weak summary (node count, message count);
delay-augmented ``(C_t, C_{t-1}, .., C_{t-depth})`` from analyses of history
prefixes; and the full microscopic state (the state-root digest). ECS has no
real-valued microscopic state, so no random-projection control exists here.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from elpis.ecs.kernel import Kernel
from elpis.ecs.persistence import genesis_descriptor_digest
from elpis.ecs.topology_analysis import analyze_topology

from .spec import canonical_json

GENESIS_LABEL = "elpis.research.ecs-dynamics.collision.v1"


def coarse_state(analysis) -> dict:
    data = analysis.to_dict()
    data.pop("topology_digest")
    return data


def weak_state(analysis) -> dict:
    return {"node_count": analysis.node_count, "total_message_count": analysis.total_message_count}


class History:
    """A closed ECS history in its own temporary directory."""

    def __init__(self, labels, sends, payload_prefix: bytes = b"m"):
        self.root = tempfile.mkdtemp(prefix="elpis-research-ecs-")
        self.path = os.path.join(self.root, "h")
        os.mkdir(self.path)
        kernel = Kernel(self.path, genesis_label=GENESIS_LABEL)
        kernel.open()
        try:
            self.ids = [kernel.found_entity(label) for label in labels]
            kernel.run_until_quiescent()
            for n, (s, r) in enumerate(sends):
                kernel.entity_port(self.ids[s]).propose(self.ids[r], payload_prefix + str(n).encode("ascii"))
            self.protocol = kernel.scheduler_protocol
            self.capacity = kernel.mailbox_capacity
            self.events = kernel.events()
            self.state_root = kernel.state_root_digest()
            self.analysis = kernel.topology_analysis()
            self.root_after_analysis = kernel.state_root_digest()
        finally:
            kernel.close()
        self.genesis = genesis_descriptor_digest(GENESIS_LABEL, self.protocol)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def prefix_coarse(self, depth: int) -> list:
        """``C`` at event counts ``n, n-1, .., n-depth`` (prefixes of this valid history)."""
        n = len(self.events)
        out = []
        for j in range(depth + 1):
            if n - j <= 0:
                out.append(None)
                continue
            a = analyze_topology(self.genesis, self.events[: n - j], self.capacity, self.protocol)
            out.append(coarse_state(a))
        return out

    def fork(self) -> Kernel:
        """Open a kernel on a byte copy of this history (the original is never reopened for writing)."""
        dst = tempfile.mkdtemp(prefix="elpis-research-ecs-fork-", dir=self.root)
        target = os.path.join(dst, "h")
        shutil.copytree(self.path, target)
        kernel = Kernel(target, genesis_label=GENESIS_LABEL)
        kernel.open()
        return kernel


def _sender_of(events, message_id):
    for e in events:
        if e["event_kind"] == "MESSAGE_ENQUEUED" and e["payload"]["envelope"]["message_id"] == message_id:
            return e["payload"]["envelope"]["sender_entity_id"]
    raise KeyError(message_id)


def next_processed_edge(history: History):
    """(sender, receiver) of the message the next kernel step processes, or None."""
    kernel = history.fork()
    try:
        if kernel.step() == 0:
            return None
        events = kernel.events()
        last = events[-1]
        if last["event_kind"] != "MESSAGE_PROCESSED":
            return None
        return [_sender_of(events, last["payload"]["message_id"]), last["payload"]["receiver_entity_id"]]
    finally:
        kernel.close()


def reply_policy_future(history: History, rounds: int) -> list:
    """``C`` after each of ``rounds`` rounds of: one kernel step, then the receiver replies to the sender."""
    kernel = history.fork()
    states = []
    try:
        for r in range(rounds):
            if kernel.step() == 0:
                break
            events = kernel.events()
            last = events[-1]
            if last["event_kind"] == "MESSAGE_PROCESSED":
                receiver = last["payload"]["receiver_entity_id"]
                sender = _sender_of(events, last["payload"]["message_id"])
                if sender != receiver:
                    kernel.entity_port(receiver).propose(sender, b"reply-" + str(r).encode("ascii"))
            states.append(coarse_state(kernel.topology_analysis()))
        return states
    finally:
        kernel.close()


def _key(value) -> bytes:
    return canonical_json(value)


def compare_pair(labels, sends_a, sends_b, *, depth: int, rounds: int) -> dict:
    """Coarse collision and target divergence for two histories; every representation is scored."""
    ha, hb = History(labels, sends_a), History(labels, sends_b)
    try:
        ca, cb = coarse_state(ha.analysis), coarse_state(hb.analysis)
        da, db = ha.prefix_coarse(depth), hb.prefix_coarse(depth)
        event_a, event_b = next_processed_edge(ha), next_processed_edge(hb)
        # Entity ids depend only on labels, founding order and genesis, so edges compare directly.
        future_a, future_b = reply_policy_future(ha, rounds), reply_policy_future(hb, rounds)
        return {
            "histories_distinct": ha.events[-1]["event_digest"] != hb.events[-1]["event_digest"],
            "coarse_equal": _key(ca) == _key(cb),
            "weak_equal": _key(weak_state(ha.analysis)) == _key(weak_state(hb.analysis)),
            "delay_equal": _key(da) == _key(db),
            "full_equal": ha.state_root == hb.state_root,
            "analysis_read_only": ha.state_root == ha.root_after_analysis and hb.state_root == hb.root_after_analysis,
            "event_differs": event_a != event_b,
            "trajectory_differs": _key(future_a) != _key(future_b),
            "endpoint_differs": _key(future_a[-1:]) != _key(future_b[-1:]),
            "first_divergent_round": next((i + 1 for i, (x, y) in enumerate(zip(future_a, future_b))
                                           if _key(x) != _key(y)), None),
            "event_edges": [event_a, event_b],
        }
    finally:
        ha.cleanup()
        hb.cleanup()


MINIMAL_PAIR = (("a", "b", "c"), ((0, 2), (1, 2)), ((1, 2), (0, 2)))
