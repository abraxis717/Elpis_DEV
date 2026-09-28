
from __future__ import annotations

import pytest

from elpis.ecs.errors import ReplayError, WrongAuthorityError
from elpis.ecs.kernel import Kernel
from elpis.ecs.persistence import genesis_descriptor_digest
from elpis.ecs.replay import replay_from_events
from elpis.ecs.scheduler import SCHEDULER_V1, SCHEDULER_V2
from elpis.ecs.topology import project_topology


def _activate_three(kernel):
    sender = kernel.found_entity("sender")
    left = kernel.found_entity("left")
    right = kernel.found_entity("right")
    kernel.activate(sender)
    kernel.activate(left)
    kernel.activate(right)
    return sender, left, right


def _build_divergent_queue(kernel):
    sender, left, right = _activate_three(kernel)
    low, high = sorted((left, right))
    port = kernel.entity_port(sender)
    old_mid = port.propose(high, b"old")
    new_mid = port.propose(low, b"new")
    return old_mid, new_mid, low, high


def test_default_genesis_helper_remains_historical_v1():
    implicit = genesis_descriptor_digest("ecs-m1a-genesis")
    explicit = genesis_descriptor_digest(
        "ecs-m1a-genesis",
        SCHEDULER_V1,
    )
    assert implicit == explicit


def test_fresh_kernel_defaults_to_v2_and_defeats_entity_id_priority(tmp_path):
    path = tmp_path / "v2"
    with Kernel(str(path)) as kernel:
        assert kernel.scheduler_protocol == SCHEDULER_V2
        old_mid, new_mid, low, high = _build_divergent_queue(kernel)

        ready = kernel.ready_items()
        assert ready[0].ref == old_mid
        assert ready[0].entity_id == high

        kernel.step()
        events = kernel.events()
        assert events[-1]["event_kind"] == "MESSAGE_PROCESSED"
        assert events[-1]["payload"]["message_id"] == old_mid

        genesis = kernel._genesis_digest
        capacity = kernel.mailbox_capacity
        profile = kernel.scheduler_protocol
        live_root = kernel.state_root_digest()

    replayed = replay_from_events(
        genesis,
        events,
        capacity,
        scheduler_protocol=profile,
    )
    assert replayed.state_root_digest() == live_root

    with pytest.raises(ReplayError, match="PROCESS_SCHEDULER_ORDER"):
        replay_from_events(
            genesis,
            events,
            capacity,
            scheduler_protocol=SCHEDULER_V1,
        )

    projection = project_topology(
        genesis,
        events,
        capacity,
        scheduler_protocol=profile,
    )
    assert projection.event_count == len(events)

    with Kernel(str(path)) as reopened:
        assert reopened.scheduler_protocol == SCHEDULER_V2
        assert reopened.state_root_digest() == live_root


def test_explicit_v1_history_keeps_historical_order_and_auto_reopens_v1(tmp_path):
    path = tmp_path / "v1"
    with Kernel(str(path), scheduler_protocol=SCHEDULER_V1) as kernel:
        assert kernel.scheduler_protocol == SCHEDULER_V1
        old_mid, new_mid, low, high = _build_divergent_queue(kernel)

        ready = kernel.ready_items()
        assert ready[0].ref == new_mid
        assert ready[0].entity_id == low

        kernel.step()
        events = kernel.events()
        root = kernel.state_root_digest()

    with Kernel(str(path)) as reopened:
        assert reopened.scheduler_protocol == SCHEDULER_V1
        assert reopened.state_root_digest() == root
        assert reopened.events() == events

    with pytest.raises(WrongAuthorityError, match="SCHEDULER_PROTOCOL_MISMATCH"):
        Kernel(str(path), scheduler_protocol=SCHEDULER_V2).open()

    with Kernel(str(path)) as reopened:
        assert reopened.scheduler_protocol == SCHEDULER_V1
        assert reopened.events() == events
        assert reopened.state_root_digest() == root


def test_explicit_v2_history_rejects_requested_v1_on_reopen(tmp_path):
    path = tmp_path / "v2-mismatch"
    with Kernel(str(path), scheduler_protocol=SCHEDULER_V2) as kernel:
        _build_divergent_queue(kernel)
        before = kernel.events()

    with pytest.raises(WrongAuthorityError, match="SCHEDULER_PROTOCOL_MISMATCH"):
        Kernel(str(path), scheduler_protocol=SCHEDULER_V1).open()

    with Kernel(str(path)) as reopened:
        assert reopened.scheduler_protocol == SCHEDULER_V2
        assert reopened.events() == before
