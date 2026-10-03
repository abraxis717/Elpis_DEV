import pytest

from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.topology import (
    TopologyError,
    _project_topology_from_validated_state,
    project_topology,
)


def _history(tmp_path):
    k = Kernel(str(tmp_path)).open()
    a = k.found_entity("a")
    b = k.found_entity("b")
    k.run_until_quiescent()
    k.entity_port(a).propose(b, b"one")
    k.run_until_quiescent()
    return k


def test_kernel_fast_projection_equals_full_replay_projection(tmp_path):
    k = _history(tmp_path)
    events = tuple(k.events())
    full = project_topology(k._genesis_digest, events, k.mailbox_capacity)
    fast = k.topology_projection()
    assert fast == full
    k.close()


def test_kernel_topology_projection_does_not_replay_history(tmp_path, monkeypatch):
    k = _history(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate semantic replay invoked")

    import elpis.ECS_C.topology as topology
    monkeypatch.setattr(topology, "replay_from_events", forbidden)
    projection = k.topology_projection()
    assert projection.event_count == len(k.events())
    k.close()


def test_fast_path_rejects_event_count_mismatch(tmp_path):
    k = _history(tmp_path)
    state = k.state
    events = tuple(k.events())
    state.logical_clock += 1
    with pytest.raises(TopologyError, match="EVENT_COUNT_MISMATCH"):
        _project_topology_from_validated_state(
            k._genesis_digest, events, k.mailbox_capacity, state
        )
    k.close()


def test_fast_path_rejects_root_mismatch(tmp_path):
    k = _history(tmp_path)
    state = k.state
    events = [dict(event) for event in k.events()]
    events[-1] = dict(events[-1])
    events[-1]["after_state_root"] = "0" * 64
    with pytest.raises(TopologyError, match="ROOT_MISMATCH"):
        _project_topology_from_validated_state(
            k._genesis_digest, tuple(events), k.mailbox_capacity, state
        )
    k.close()


def test_fast_path_rejects_genesis_and_capacity_mismatch(tmp_path):
    k = _history(tmp_path)
    state = k.state
    events = tuple(k.events())
    with pytest.raises(TopologyError, match="GENESIS_MISMATCH"):
        _project_topology_from_validated_state(
            "0" * 64, events, k.mailbox_capacity, state
        )
    with pytest.raises(TopologyError, match="CAPACITY_MISMATCH"):
        _project_topology_from_validated_state(
            k._genesis_digest, events, k.mailbox_capacity + 1, state
        )
    k.close()
