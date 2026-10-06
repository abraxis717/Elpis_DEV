"""ECS_C compaction base: state-bearing checkpoint, base-mode kernel, retention.

These tests qualify the generic ECS_C mechanics the bounded runtime history is
built on: a verified compaction checkpoint reproduces the exact kernel state of
a retired prefix; a kernel opened on (checkpoint, segment) continues ONE global
history (indices, clock, chain, roots, watermarks) without renumbering; and
every complete-history read refuses below the retention floor instead of
presenting a tail as complete.
"""
from __future__ import annotations

import json

import pytest

from elpis.ECS_C import canonical
from elpis.ECS_C.compaction import CompactionCheckpoint
from elpis.ECS_C.errors import (
    BrokenChainError,
    CorruptCompactionCheckpointError,
    PersistenceError,
    RetentionFloorError,
    StorageCapacityError,
    WrongAuthorityError,
)
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.persistence import GENESIS_PREV_DIGEST, LENGTH_PREFIX, EventLog
from elpis.ECS_C.projection import (
    ProjectionError,
    ProjectionRequest,
    RetainedHistoryBinding,
    project_history,
    project_retained_history,
    project_verified_events,
)
from elpis.ECS_C.replay import StreamReplay, replay_from_events

LABEL = "ecs-compaction-test"


def _history(path, *, messages=6, process=True):
    """A small multi-entity history with messages, returning its events."""
    kernel = Kernel(str(path), genesis_label=LABEL)
    kernel.open()
    try:
        a = kernel.found_entity("a")
        b = kernel.found_entity("b")
        c = kernel.found_entity("c")
        kernel.run_until_quiescent()
        port_a, port_b = kernel.entity_port(a), kernel.entity_port(b)
        for i in range(messages):
            (port_a if i % 2 else port_b).propose(c if i % 3 else a, b"m%d" % i)
            if process:
                kernel.run_until_quiescent()
        events = kernel.events()
        state = kernel.state
        return kernel.genesis_label, events, state, kernel.scheduler_protocol
    finally:
        kernel.close()


def _checkpoint_at(events, count, scheduler, extension=None):
    from elpis.ECS_C.persistence import genesis_descriptor_digest
    genesis = genesis_descriptor_digest(LABEL, scheduler)
    state = replay_from_events(genesis, events[:count], scheduler_protocol=scheduler)
    head = events[count - 1]["event_digest"] if count else GENESIS_PREV_DIGEST
    return CompactionCheckpoint.from_state(state, genesis_label=LABEL, event_count=count,
                                           head_event_digest=head, extension=extension)


def _write_segment(path, events):
    with open(path, "wb") as fh:
        for event in events:
            payload = canonical.canonical_bytes(event)
            fh.write(LENGTH_PREFIX.pack(len(payload)) + payload)


def _reframe(checkpoint, mutate):
    """Re-encode a checkpoint after ``mutate(record)`` with a VALID digest."""
    raw = checkpoint.to_bytes()
    record = json.loads(raw[8:])
    mutate(record)
    body = {k: v for k, v in record.items() if k != "checkpoint_digest"}
    record["checkpoint_digest"] = canonical.domain_digest("ecs.compaction-checkpoint.v1", body)
    payload = canonical.canonical_bytes(record)
    return LENGTH_PREFIX.pack(len(payload)) + payload


# -- checkpoint ------------------------------------------------------------------------

def test_checkpoint_reproduces_exact_state_and_is_content_identified(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, len(events), scheduler, extension={"note": "x"})
    rebuilt = cp.kernel_state()
    assert rebuilt.state_root() == live.state_root()
    assert cp.state_root_digest == live.state_root_digest()
    assert cp.event_count == cp.logical_clock == len(events)
    assert cp.head_event_digest == events[-1]["event_digest"]
    # Full registry provenance (causing_event_id) survives, not just the root.
    assert rebuilt.registry.as_sorted_list() == live.registry.as_sorted_list()
    again = CompactionCheckpoint.from_bytes(cp.to_bytes(), genesis_label=LABEL,
                                            mailbox_capacity=live.mailbox_capacity)
    assert again == cp and again.checkpoint_digest == cp.checkpoint_digest
    assert cp.extension == {"note": "x"}
    # kernel_state() is always a fresh object.
    assert cp.kernel_state() is not rebuilt


def test_checkpoint_binds_non_quiescent_mailboxes(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h", process=False)
    assert any(box["contents"] for box in live.state_root()["mailboxes"])
    cp = _checkpoint_at(events, len(events), scheduler)
    assert cp.kernel_state().state_root() == live.state_root()


@pytest.mark.parametrize("mutate, code", [
    (lambda r: r["state_root"]["watermarks"].update(
        {k: v + 1 for k, v in r["state_root"]["watermarks"].items()}), "STATE_ROOT_REBUILD_MISMATCH"),
    (lambda r: r["state_root"]["entities"][0].update(state_version=99), "STATE_VERSION_DIGEST"),
    (lambda r: r["state_root"].update(next_founding_index=7), "ENTITY_SET"),
    (lambda r: r.update(event_count=r["event_count"] + 1), "CLOCK_COUNT_MISMATCH"),
    (lambda r: r.update(head_event_digest=GENESIS_PREV_DIGEST), "GENESIS_COORDINATES"),
    (lambda r: r.update(schema="ecs.checkpoint.v1"), "SCHEMA"),
    (lambda r: r.update(genesis_digest="0" * 63 + "1"), "GENESIS_DIGEST"),
    (lambda r: r.update(extension={"x": "y" * 5000}), "EXTENSION"),
    (lambda r: r["causing_event_ids"].popitem(), "ENTITY_SET"),
])
def test_checkpoint_is_semantically_verified_not_just_digest_checked(tmp_path, mutate, code):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, len(events), scheduler)
    raw = _reframe(cp, mutate)
    with pytest.raises(CorruptCompactionCheckpointError, match=code):
        CompactionCheckpoint.from_bytes(raw, genesis_label=LABEL, mailbox_capacity=live.mailbox_capacity)


def test_checkpoint_rejects_bit_flips_wrong_authority_and_oversize(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, len(events), scheduler)
    raw = bytearray(cp.to_bytes())
    raw[len(raw) // 2] ^= 0x01
    with pytest.raises(CorruptCompactionCheckpointError):
        CompactionCheckpoint.from_bytes(bytes(raw), genesis_label=LABEL, mailbox_capacity=16)
    with pytest.raises(CorruptCompactionCheckpointError, match="GENESIS_LABEL"):
        CompactionCheckpoint.from_bytes(cp.to_bytes(), genesis_label="other", mailbox_capacity=16)
    with pytest.raises(CorruptCompactionCheckpointError, match="MAILBOX_CAPACITY_CONFIG"):
        CompactionCheckpoint.from_bytes(cp.to_bytes(), genesis_label=LABEL, mailbox_capacity=8)
    with pytest.raises(CorruptCompactionCheckpointError, match="TOO_LARGE"):
        CompactionCheckpoint.from_bytes(cp.to_bytes(), genesis_label=LABEL, mailbox_capacity=16,
                                        max_bytes=len(cp.to_bytes()) - 1)
    with pytest.raises(CorruptCompactionCheckpointError, match="SCHEDULER_PROTOCOL_MISMATCH"):
        other = [p for p in ("active-mailbox-fifo/entity-id-before-founded-activation.v1",
                             "active-mailbox-global-arrival/founding-index-activation.v2") if p != scheduler][0]
        CompactionCheckpoint.from_bytes(cp.to_bytes(), genesis_label=LABEL, mailbox_capacity=16,
                                        scheduler_protocol=other)


def test_count_zero_checkpoint_must_be_genesis_empty(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    zero = _checkpoint_at(events, 0, scheduler)
    assert zero.event_count == 0 and zero.head_event_digest == GENESIS_PREV_DIGEST
    with pytest.raises(CorruptCompactionCheckpointError):
        CompactionCheckpoint.from_bytes(
            _reframe(zero, lambda r: r.update(event_count=0, logical_clock=0,
                                              state_root=_checkpoint_at(events, 3, scheduler).kernel_state().state_root())),
            genesis_label=LABEL, mailbox_capacity=16)


# -- streaming replay ---------------------------------------------------------------------

def test_stream_replay_from_any_base_equals_full_replay(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    for cut in (0, 1, 5, len(events) // 2, len(events)):
        cp = _checkpoint_at(events, cut, scheduler)
        replay = StreamReplay(cp.kernel_state(), cp.event_count, cp.head_event_digest)
        for event in events[cut:]:
            replay.apply(event)
        assert replay.state.state_root() == live.state_root()
        assert replay.event_count == len(events)
        assert replay.head_event_digest == events[-1]["event_digest"]


def test_stream_replay_rejects_a_tail_that_does_not_continue_the_base(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, 6, scheduler)
    replay = StreamReplay(cp.kernel_state(), cp.event_count, cp.head_event_digest)
    with pytest.raises((BrokenChainError, PersistenceError)):
        replay.apply(events[7])  # skips global index 6


# -- base-mode kernel ----------------------------------------------------------------------

def _base_kernel(tmp_path, events, cut, scheduler, **kwargs):
    cp = _checkpoint_at(events, cut, scheduler)
    segment = tmp_path / "g.seg"
    _write_segment(segment, events[cut:])
    kernel = Kernel(str(tmp_path / "base"), genesis_label=LABEL, log_path=str(segment), base=cp, **kwargs)
    return kernel, cp, segment


def test_base_kernel_continues_one_global_history(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cut = 9
    kernel, cp, _ = _base_kernel(tmp_path, events, cut, scheduler)
    kernel.open()
    try:
        assert kernel.retention_floor == cut
        assert kernel.state_root_digest() == live.state_root_digest()
        snap = kernel.snapshot()
        assert snap["event_count"] == len(events) and snap["event_digest"] == events[-1]["event_digest"]
        assert kernel.retained_events() == events[cut:]
        assert kernel.retained_events()[0]["prev_event_digest"] == cp.head_event_digest
        # Continuation: the next event gets the next GLOBAL index and clock.
        ids = kernel.entity_ids()
        port = kernel.entity_port(ids[0])
        port.propose(ids[1], b"after-compaction")
        tail = kernel.retained_events()
        assert tail[-1]["event_index"] == len(events)
        assert tail[-1]["logical_clock"] == len(events) + 1
        assert tail[-1]["prev_event_digest"] == events[-1]["event_digest"]
        kernel.run_until_quiescent()
        state = kernel.state
    finally:
        kernel.close()

    # The same transitions on the uncompacted history produce the same state.
    reference = Kernel(str(tmp_path / "h"), genesis_label=LABEL)
    reference.open()
    try:
        ids = reference.entity_ids()
        reference.entity_port(ids[0]).propose(ids[1], b"after-compaction")
        reference.run_until_quiescent()
        assert reference.state.state_root() == state.state_root()
        assert reference.state.watermarks.as_sorted_dict() == state.watermarks.as_sorted_dict()
    finally:
        reference.close()


def test_base_kernel_refuses_complete_history_reads_below_the_floor(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    kernel, cp, _ = _base_kernel(tmp_path, events, 9, scheduler)
    kernel.open()
    try:
        with pytest.raises(RetentionFloorError, match="BELOW_RETENTION_FLOOR"):
            kernel.events()
        with pytest.raises(RetentionFloorError, match="TOPOLOGY_REQUIRES_COMPLETE_HISTORY"):
            kernel.topology_projection()
        with pytest.raises(PersistenceError, match="CHECKPOINT_MARKER_UNSUPPORTED"):
            kernel.checkpoint()
    finally:
        kernel.close()


def test_base_kernel_with_zero_floor_is_the_complete_history(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    kernel, cp, _ = _base_kernel(tmp_path, events, 0, scheduler)
    kernel.open()
    try:
        assert kernel.retention_floor == 0
        assert kernel.events() == events
        assert kernel.topology_projection() is not None
    finally:
        kernel.close()


def test_base_kernel_rejects_a_segment_that_does_not_continue_the_checkpoint(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, 9, scheduler)
    segment = tmp_path / "g.seg"
    _write_segment(segment, events[10:])  # gap at global index 9
    kernel = Kernel(str(tmp_path / "b"), genesis_label=LABEL, log_path=str(segment), base=cp)
    with pytest.raises((BrokenChainError, PersistenceError, WrongAuthorityError)):
        kernel.open()
    assert kernel._state is None


def test_base_kernel_never_creates_a_missing_segment(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, 9, scheduler)
    kernel = Kernel(str(tmp_path / "b"), genesis_label=LABEL, log_path=str(tmp_path / "missing.seg"), base=cp)
    with pytest.raises(PersistenceError, match="LOG_SEGMENT_MISSING"):
        kernel.open()
    assert not (tmp_path / "missing.seg").exists()


def test_base_kernel_segment_bound_is_enforced_before_writing(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cut = len(events)
    cp = _checkpoint_at(events, cut, scheduler)
    segment = tmp_path / "g.seg"
    segment.write_bytes(b"")
    kernel = Kernel(str(tmp_path / "b"), genesis_label=LABEL, log_path=str(segment), base=cp,
                    max_log_bytes=64)
    kernel.open()
    try:
        ids = kernel.entity_ids()
        with pytest.raises(StorageCapacityError, match="SEGMENT_CAPACITY_REACHED"):
            kernel.entity_port(ids[0]).propose(ids[1], b"too big for 64 bytes")
        assert segment.stat().st_size == 0
        assert kernel.snapshot()["event_count"] == cut  # still open, unchanged
    finally:
        kernel.close()


def test_event_log_reports_global_and_local_counts(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cut = 9
    cp = _checkpoint_at(events, cut, scheduler)
    segment = tmp_path / "g.seg"
    _write_segment(segment, events[cut:])
    log = EventLog(str(segment), base_count=cut, base_head=cp.head_event_digest,
                   base_root=cp.state_root_digest, create=False)
    log.open()
    try:
        seen = []
        assert log.scan_events(seen.append, recovery=True) == len(events) - cut
        log.finish_recovery()
        assert seen == events[cut:]
        assert log.event_count() == len(events)
        assert log.frame_count() == len(events) - cut
        assert log.size() == segment.stat().st_size
    finally:
        log.close()
    with pytest.raises(PersistenceError, match="LOG_BASE_INCONSISTENT"):
        EventLog(str(segment), base_count=3)


# -- retained projection ----------------------------------------------------------------

def test_retained_projection_binds_the_floor_and_refuses_below_it(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cut = 9
    cp = _checkpoint_at(events, cut, scheduler)
    tail = events[cut:]

    with pytest.raises(ProjectionError, match="BELOW_RETENTION_FLOOR"):
        project_retained_history(cp, tail, ProjectionRequest())
    with pytest.raises(ProjectionError, match="BELOW_RETENTION_FLOOR"):
        project_retained_history(cp, tail, ProjectionRequest(clock_min=cut))

    request = ProjectionRequest(clock_min=cut + 1, max_records=4096)
    projected = project_retained_history(cp, tail, request)
    assert projected.schema == "ecs.context-projection.retained.v1"
    assert projected.to_dict()["schema"] == "ecs.context-projection.retained.v1"
    assert type(projected.source) is RetainedHistoryBinding
    assert projected.retention_floor == cut
    assert projected.source.floor_event_digest == cp.head_event_digest
    assert projected.source.floor_state_root == cp.state_root_digest
    assert projected.source.final_state_root == live.state_root_digest()
    assert [r.event_index for r in projected.records] == list(range(cut, len(events)))

    # The same window over the complete history is a different, v1 identity.
    complete = project_history(cp.genesis_digest, events, request, scheduler_protocol=scheduler)
    assert complete.schema == "ecs.context-projection.v1"
    assert [r.record_bytes for r in complete.records] == [r.record_bytes for r in projected.records]
    assert complete.projection_digest != projected.projection_digest

    # The streaming hook links the window to the stated floor, not genesis.
    bad = RetainedHistoryBinding(**{**projected.source.__dict__, "floor_event_digest": "1" * 64})
    with pytest.raises(ProjectionError):
        project_verified_events(iter(tail), bad, request)


def test_retained_projection_with_zero_floor_is_project_history(tmp_path):
    _, events, live, scheduler = _history(tmp_path / "h")
    cp = _checkpoint_at(events, 0, scheduler)
    request = ProjectionRequest(max_records=4096)
    assert (project_retained_history(cp, events, request).projection_digest
            == project_history(cp.genesis_digest, events, request,
                               scheduler_protocol=scheduler).projection_digest)
