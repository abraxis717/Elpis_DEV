"""Behavioral qualification using actual ECS kernel histories."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc

import pytest

from elpis.identity import canonical_json_bytes, content_digest
from elpis.ECS_C import canonical
from elpis.ECS_C.kernel import Kernel
from elpis.ECS_C.scheduler import SCHEDULER_V1, SCHEDULER_V2
from elpis.ECS_C.projection import (
    HistoryBinding, ProjectionError, ProjectionRequest,
    project_history, project_verified_events,
)


@pytest.fixture
def history(tmp_path):
    with Kernel(str(tmp_path / "kernel")).open() as kernel:
        a = kernel.found_entity("alpha")
        b = kernel.found_entity("MESSAGE_ENQUEUED")  # prose must not match patterns
        kernel.run_until_quiescent()
        kernel.entity_port(a).propose(b, b"ENTITY_FOUNDED")
        kernel.run_until_quiescent()
        kernel.dormant(a)
        kernel.reactivate(a)
        kernel.entity_port(b).propose(a, b"second")
        kernel.run_until_quiescent()
        events = kernel.events()
        state = kernel.state
        return state.genesis_digest, events, a, b


def project(history, request=ProjectionRequest()):
    return project_history(history[0], history[1], request)


def test_repeat_bytes_digest_and_tie_break(history):
    request = ProjectionRequest(max_records=3)
    expected = project(history, request)
    for _ in range(30):
        result = project(history, request)
        assert result == expected
        assert result.canonical_bytes() == expected.canonical_bytes()
        assert result.projection_digest == expected.projection_digest
    assert [r.event_index for r in expected.records] == [0, 1, 2]
    assert expected.projection_digest == content_digest("ecs.context-projection.v1", expected.to_dict())


def test_permutations_and_duplicate_candidates(history):
    genesis, events, a, b = history
    expected = project(history, ProjectionRequest(entity_ids=(a, b)))
    for permutation in (list(reversed(events)), events[::2] + events[1::2], events * 2):
        reordered = [{key: event[key] for key in reversed(event)} for event in permutation]
        result = project_history(genesis, reordered, ProjectionRequest(entity_ids=(b, a, b)))
        assert result.canonical_bytes() == expected.canonical_bytes()
        assert result.projection_digest == expected.projection_digest


def test_record_boundary_and_explicit_truncation(history):
    n = len(history[1])
    exact = project(history, ProjectionRequest(max_records=n))
    assert len(exact.records) == exact.total_matches == n
    assert not exact.truncated and exact.budget_exhausted == ()
    shorter = project(history, ProjectionRequest(max_records=n - 1))
    assert shorter.truncated and shorter.total_matches == n
    assert shorter.budget_exhausted == ("records",)
    empty = project(history, ProjectionRequest(max_records=0))
    assert empty.truncated and empty.total_matches == n and empty.records == ()


def test_byte_boundary_and_prefix_semantics(history):
    full = project(history)
    first = canonical_json_bytes([full.records[0].to_dict()])
    exact = project(history, ProjectionRequest(max_record_bytes=len(first)))
    assert exact.records == full.records[:1]
    assert exact.record_bytes_used == len(first)
    assert exact.budget_exhausted == ("bytes",)
    below = project(history, ProjectionRequest(max_record_bytes=len(first) - 1))
    assert below.records == () and below.record_bytes_used == 2
    assert below.total_matches == full.total_matches and below.truncated
    # Complete array overhead is included, with no off-by-one at the full boundary.
    size = len(canonical_json_bytes([r.to_dict() for r in full.records]))
    all_exact = project(history, ProjectionRequest(max_record_bytes=size))
    assert all_exact.record_bytes_used == size and not all_exact.truncated
    assert project(history, ProjectionRequest(max_record_bytes=size - 1)).truncated


def test_both_budgets_and_no_match(history):
    result = project(history, ProjectionRequest(max_records=0, max_record_bytes=2))
    assert result.budget_exhausted == ("records", "bytes")
    result = project(history, ProjectionRequest(entity_ids=("f" * 64,), max_records=0))
    assert result.total_matches == 0 and result.records == ()
    assert not result.truncated and result.budget_exhausted == ()


def test_typed_selectors_inclusive_clock_and_provenance(history):
    _, events, sender, receiver = history
    enqueue = next(e for e in events if e["event_kind"] == "MESSAGE_ENQUEUED")
    clock = enqueue["logical_clock"]
    result = project(history, ProjectionRequest(
        entity_ids=(receiver,), event_kinds=("MESSAGE_ENQUEUED",),
        sender_ids=(sender,), receiver_ids=(receiver,), clock_min=clock, clock_max=clock,
        kind_pattern="MESSAGE_*",
    ))
    assert result.total_matches == 1
    item = result.records[0]
    assert item.record_bytes == canonical.canonical_bytes(enqueue)
    assert item.to_dict()["event"] == enqueue
    assert item.to_dict()["provenance"] == {
        "event_index": enqueue["event_index"], "event_digest": enqueue["event_digest"],
    }
    assert result.source.head_event_digest == events[-1]["event_digest"]
    assert result.source.final_state_root == events[-1]["after_state_root"]
    assert result.to_dict()["request"]["clock_min"] == clock
    assert project(history, ProjectionRequest(sender_ids=(receiver,), receiver_ids=(receiver,))).total_matches == 0


@pytest.mark.parametrize("pattern,expected", [
    ("*", None), ("MESSAGE_*", {"MESSAGE_ENQUEUED", "MESSAGE_PROCESSED"}),
    ("ENTITY_FOUNDED", {"ENTITY_FOUNDED"}), ("MOVE*", set()),
])
def test_symbol_patterns_only(history, pattern, expected):
    result = project(history, ProjectionRequest(kind_pattern=pattern))
    if expected is not None:
        assert {r.to_dict()["event"]["event_kind"] for r in result.records} == expected
    else:
        assert result.total_matches == len(history[1])


@pytest.mark.parametrize("pattern", [
    "", "a", "ENTITY.*", "(A+)+$", "A*A*", "*A", "**", "A?", "[A-Z]",
    "é", "A\n", "A" * 65, "*" * 100000, None,
])
def test_invalid_and_adversarial_patterns(pattern):
    with pytest.raises(ProjectionError, match="INVALID_KIND_PATTERN"):
        ProjectionRequest(kind_pattern=pattern)


@pytest.mark.parametrize("fields", [
    {"entity_ids": ["a" * 64]}, {"entity_ids": (418,)},
    {"entity_ids": ("A" * 64,)}, {"receiver_ids": ("",)},
    {"sender_ids": (None,)}, {"event_kinds": ("MOVE",)},
    {"event_kinds": ([],)}, {"event_kinds": ("ENTITY_FOUNDED",) * 257},
    {"clock_min": True}, {"clock_min": -1}, {"clock_min": 3, "clock_max": 2},
    {"clock_max": 1 << 63}, {"max_records": -1}, {"max_records": 4097},
    {"max_records": 1.5}, {"max_record_bytes": 1}, {"max_record_bytes": 16777217},
])
def test_invalid_selectors(fields):
    with pytest.raises(ProjectionError):
        ProjectionRequest(**fields)


def test_unavailable_relations_rejected():
    with pytest.raises(TypeError):
        ProjectionRequest(causal_radius=3)
    with pytest.raises(TypeError):
        ProjectionRequest(lineage_ids=("a",))


def test_no_mutation_or_shared_nested_identity(history):
    before = copy.deepcopy(history)
    result = project(history)
    view = result.to_dict()
    view["records"][0]["event"]["payload"]["label"] = "mutated"
    assert history == before
    assert result.records[0].to_dict()["event"] == history[1][0]
    old_bytes = result.canonical_bytes()
    history[1][0]["payload"]["label"] = "later caller mutation"
    assert result.canonical_bytes() == old_bytes
    with pytest.raises(FrozenInstanceError):
        result.request.max_records = 999


def test_read_only_live_kernel(tmp_path):
    with Kernel(str(tmp_path / "live")).open() as kernel:
        kernel.found_entity("alpha")
        before = kernel.snapshot()
        events = kernel.events()
        project_history(kernel.state.genesis_digest, events, ProjectionRequest())
        assert kernel.snapshot() == before
        assert kernel.events() == events


def test_conflicting_duplicate_and_bad_unselected_tail(history):
    genesis, events, _, _ = history
    conflicting = copy.deepcopy(events[0])
    conflicting["payload"]["label"] = "different"
    conflicting["payload_digest"] = canonical.digest(conflicting["payload"])
    conflicting["event_digest"] = canonical.domain_digest(
        canonical.DOMAIN_EVENT, {k: v for k, v in conflicting.items() if k != "event_digest"})
    with pytest.raises(ProjectionError, match="CONFLICTING_EVENT_INDEX"):
        project_history(genesis, events + [conflicting], ProjectionRequest())
    events[-1]["event_digest"] = "0" * 64
    with pytest.raises(ProjectionError, match="EVENT_DIGEST_MISMATCH"):
        project(history, ProjectionRequest(max_records=0))


def test_wrong_authority_and_missing_history_fail(history):
    with pytest.raises(ProjectionError):
        project_history("f" * 64, history[1], ProjectionRequest())
    with pytest.raises(ProjectionError):
        project_history(history[0], history[1][1:], ProjectionRequest())
    with pytest.raises(ProjectionError):
        project_history(history[0], history[1], ProjectionRequest(), mailbox_capacity=1)


def test_existing_replay_semantic_checks_are_used(history):
    events = copy.deepcopy(history[1][:1])
    # Internally consistent hashes are insufficient to admit a false founding fact.
    events[0]["payload"]["founding_index"] = 8
    events[0]["payload_digest"] = canonical.digest(events[0]["payload"])
    events[0]["event_digest"] = canonical.domain_digest(
        canonical.DOMAIN_EVENT, {k: v for k, v in events[0].items() if k != "event_digest"})
    with pytest.raises(ProjectionError, match="FOUNDING_IDENTITY_MISMATCH"):
        project_history(history[0], events, ProjectionRequest())


@pytest.mark.parametrize("scheduler", [SCHEDULER_V1, SCHEDULER_V2])
def test_scheduler_authority_and_empty_history(tmp_path, scheduler):
    with Kernel(str(tmp_path / "scheduler"), scheduler_protocol=scheduler).open() as kernel:
        genesis = kernel.state.genesis_digest
        empty = project_history(genesis, [], ProjectionRequest(), scheduler_protocol=scheduler)
        assert not empty.truncated and empty.records == ()
        assert empty.source.final_state_root == kernel.state_root_digest()
        kernel.found_entity("one")
        result = project_history(genesis, kernel.events(), ProjectionRequest(), scheduler_protocol=scheduler)
        assert result.source.scheduler_protocol == scheduler


def test_stream_matches_materialized_bytes_and_digest(history):
    full = project(history)
    streamed = project_verified_events(iter(history[1]), full.source, full.request)
    assert streamed == full
    assert streamed.canonical_bytes() == full.canonical_bytes()
    assert streamed.projection_digest == full.projection_digest


@pytest.mark.parametrize("divergence,code", [
    ("adjacent", "EVENT_INDEX_MISMATCH"),
    ("final_duplicate", "SOURCE_EVENT_COUNT_MISMATCH"),
    ("nonadjacent", "EVENT_INDEX_MISMATCH"),
    ("extra", "SOURCE_EVENT_COUNT_MISMATCH"),
    ("conflicting", "EVENT_INDEX_MISMATCH"),
    ("gap", "EVENT_INDEX_MISMATCH"),
    ("reordered", "EVENT_INDEX_MISMATCH"),
])
def test_stream_rejects_physical_divergence(history, divergence, code):
    full = project(history)
    events = copy.deepcopy(history[1])
    if divergence == "adjacent":
        events.insert(1, copy.deepcopy(events[0]))
    elif divergence == "final_duplicate":
        events.append(copy.deepcopy(events[-1]))
    elif divergence == "nonadjacent":
        events.insert(3, copy.deepcopy(events[0]))
    elif divergence == "extra":
        events.append(copy.deepcopy(events[0]))
    elif divergence == "conflicting":
        duplicate = copy.deepcopy(events[0])
        duplicate["payload"]["label"] = "conflicting founding label"
        duplicate["payload_digest"] = canonical.digest(duplicate["payload"])
        duplicate["event_digest"] = canonical.domain_digest(
            canonical.DOMAIN_EVENT, {k: v for k, v in duplicate.items() if k != "event_digest"})
        events.insert(1, duplicate)
    elif divergence == "gap":
        del events[1]
    else:
        events[0], events[1] = events[1], events[0]
    # A zero emission budget must not mask physical divergence.
    for request in (full.request, ProjectionRequest(max_records=0)):
        with pytest.raises(ProjectionError, match=code):
            project_verified_events(iter(events), full.source, request)


def test_stream_count_boundary_precedes_parsing_extra_record(history):
    full = project(history)
    with pytest.raises(ProjectionError, match="SOURCE_EVENT_COUNT_MISMATCH"):
        project_verified_events(iter(history[1] + [None]), full.source, full.request)
    empty = project_history(history[0], [], full.request)
    with pytest.raises(ProjectionError, match="SOURCE_EVENT_COUNT_MISMATCH"):
        project_verified_events(iter([None]), empty.source, empty.request)


def test_digest_binds_valid_request_even_when_selected_records_unchanged(history):
    original = project(history, ProjectionRequest(max_records=2))
    changed = project(history, replace(original.request, clock_max=len(history[1])))
    assert original.request != changed.request
    assert original.source == changed.source
    assert original.records == changed.records
    assert original.total_matches == changed.total_matches
    assert original.projection_digest != changed.projection_digest


def test_digest_binds_legitimate_history_extension(history):
    # Both bindings are independently derived by authoritative replay, not forged
    # by editing a receipt. The added last event lies beyond the query interval.
    genesis, events, _, _ = history
    request = ProjectionRequest(clock_max=2)
    prefix = project_history(genesis, events[:-1], request)
    extended = project_history(genesis, events, request)
    assert prefix.request == extended.request
    assert prefix.records == extended.records
    assert prefix.total_matches == extended.total_matches
    assert extended.source.event_count == prefix.source.event_count + 1
    assert extended.source.head_event_digest != prefix.source.head_event_digest
    assert prefix.projection_digest != extended.projection_digest


def test_digest_binds_each_history_field(history):
    # Unit-level serialization binding checks; altered coordinates are NOT
    # submitted as qualified history. Valid history integration is covered above.
    result = project(history)
    changes = {
        "genesis_digest": "f" * 64,
        "mailbox_capacity": result.source.mailbox_capacity + 1,
        "scheduler_protocol": SCHEDULER_V2,
        "event_count": result.source.event_count + 1,
        "head_event_digest": "e" * 64,
        "final_state_root": "d" * 64,
    }
    for field, value in changes.items():
        changed = replace(result, source=replace(result.source, **{field: value}))
        assert changed.records == result.records
        assert changed.projection_digest != result.projection_digest


@pytest.mark.parametrize("field,value,code", [
    ("event_count", 0, "COUNT"), ("event_count", 100, "COUNT"),
    ("head_event_digest", "0" * 64, "HEAD"),
    ("final_state_root", "0" * 64, "ROOT"),
    ("genesis_digest", "0" * 64, "ROOT"),
])
def test_stream_anchor_validation(history, field, value, code):
    full = project(history)
    source = replace(full.source, **{field: value})
    with pytest.raises(ProjectionError, match=code):
        project_verified_events(iter(history[1]), source, full.request)


def test_stream_validates_after_budget_exhaustion(history):
    full = project(history)
    events = copy.deepcopy(history[1])
    events[-1]["payload_digest"] = "0" * 64
    with pytest.raises(ProjectionError, match="PAYLOAD_DIGEST_MISMATCH"):
        project_verified_events(iter(events), full.source, ProjectionRequest(max_records=0))


def test_one_shot_materialized_input(history):
    class Once(list):
        def __iter__(self):
            assert not getattr(self, "used", False)
            self.used = True
            return super().__iter__()
    expected = project(history)
    assert project_history(history[0], Once(history[1]), expected.request) == expected


def test_cross_process_hash_seed_determinism(history, tmp_path):
    file = tmp_path / "history.json"
    file.write_bytes(canonical.canonical_bytes({"genesis": history[0], "events": history[1]}))
    script = '''
import json, sys
from elpis.ECS_C.projection import ProjectionRequest, project_history
d = json.load(open(sys.argv[1]))
by_digest = {e["event_digest"]: e for e in d["events"]}
events = [{k: by_digest[h][k] for k in set(by_digest[h])} for h in set(by_digest)]
ids = tuple(set(e["entity_id"] for e in events))
r = project_history(d["genesis"], events, ProjectionRequest(entity_ids=ids, max_records=3))
sys.stdout.buffer.write(r.canonical_bytes() + b"\\n" + r.projection_digest.encode())
'''
    outputs = []
    import elpis
    import_root = str(Path(elpis.__file__).resolve().parents[1])
    for seed in ("1", "23", "999"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", PYTHONPATH=import_root)
        outputs.append(subprocess.run([sys.executable, "-B", "-c", script, str(file)],
                       env=env, check=True, capture_output=True, timeout=30).stdout)
    assert len(set(outputs)) == 1


def test_large_qualified_stream_bounded_memory(tmp_path):
    path = tmp_path / "committed.jsonl"
    with Kernel(str(tmp_path / "large")).open() as kernel:
        entity = kernel.found_entity("single")
        kernel.activate(entity)
        for _ in range(1500):
            kernel.dormant(entity)
            kernel.reactivate(entity)
        # Fixture generation uses real commit authority, outside measurement.
        events = kernel.events()
        state = kernel.state
        source = HistoryBinding(state.genesis_digest, kernel.mailbox_capacity,
                                kernel.scheduler_protocol, len(events),
                                events[-1]["event_digest"], kernel.state_root_digest())
        with path.open("wb") as output:
            for event in events:
                output.write(canonical.canonical_bytes(event) + b"\n")
        del events
    gc.collect()
    traversed = 0

    def stream():
        nonlocal traversed
        with path.open("rb") as source_file:
            for line in source_file:
                traversed += 1
                yield json.loads(line)

    tracemalloc.start()
    try:
        result = project_verified_events(stream(), source, ProjectionRequest(max_records=2))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert traversed == result.total_matches == 3002
    assert len(result.records) == 2 and result.truncated
    # Whole-history materialization would retain >2 MB of encoded events alone.
    assert path.stat().st_size > 2_000_000
    assert peak < 1_000_000


@pytest.mark.parametrize("source", [None, "", b"", 418])
def test_invalid_history_container(history, source):
    with pytest.raises(ProjectionError):
        project_history(history[0], source, ProjectionRequest())
    full = project(history)
    with pytest.raises(ProjectionError):
        project_verified_events(source, full.source, full.request)


@pytest.mark.parametrize("field,value", [
    ("event_count", True), ("event_count", -1), ("mailbox_capacity", 0),
    ("scheduler_protocol", "future"), ("genesis_digest", "not-a-digest"),
])
def test_invalid_binding(history, field, value):
    with pytest.raises(ProjectionError):
        replace(project(history).source, **{field: value})


def test_stream_does_not_replay_and_keeps_canonical_prefix(history, monkeypatch):
    full = project(history)

    def forbidden(*args, **kwargs):
        raise AssertionError("streaming hook must not materialize/replay")

    monkeypatch.setattr("elpis.ECS_C.projection.projector.replay_from_events", forbidden)
    first_size = len(canonical_json_bytes([full.records[0].to_dict()]))
    assert any(len(canonical_json_bytes([r.to_dict()])) < first_size - 1 for r in full.records[1:])
    result = project_verified_events(iter(history[1]), full.source,
                                     ProjectionRequest(max_record_bytes=first_size - 1))
    assert result.records == () and result.truncated


def test_stream_truncation_and_empty_anchor(history):
    full = project(history)
    with pytest.raises(ProjectionError, match="COUNT"):
        project_verified_events(iter(history[1][:-1]), full.source, full.request)
    empty = project_history(history[0], [], full.request)
    with pytest.raises(ProjectionError, match="HEAD"):
        project_verified_events(iter(()), replace(empty.source, head_event_digest="f" * 64), full.request)


def test_selector_cardinality_boundary_256_accepted():
    # The ceiling is 256 values per selector: 256 must be accepted (257 is
    # rejected by test_invalid_selectors). After dedup/sort the 256 identical
    # entries collapse to one, proving the count check is on the raw tuple.
    request = ProjectionRequest(event_kinds=("ENTITY_FOUNDED",) * 256)
    assert request.event_kinds == ("ENTITY_FOUNDED",)
    # A 256-distinct digest selector is also accepted at the boundary.
    distinct = tuple(f"{i:064x}" for i in range(256))
    assert len(ProjectionRequest(entity_ids=distinct).entity_ids) == 256


def test_kind_pattern_length_boundary_64_accepted():
    # Maximum pattern length is 64 characters: 64 must be accepted (65 is
    # rejected by test_invalid_and_adversarial_patterns).
    request = ProjectionRequest(kind_pattern="A" * 64)
    assert request.kind_pattern == "A" * 64
    # A 63-char stem plus one trailing wildcard is also within the limit.
    assert ProjectionRequest(kind_pattern="A" * 63 + "*").kind_pattern == "A" * 63 + "*"


def test_ecs_error_normalization_preserves_cause(history):
    # A genuine ECS rejection (wrong genesis) must surface as a ProjectionError
    # whose __cause__ is the original EcsError, not a ProjectionError. This is
    # the documented "wrapped with the original ECS error as its cause" contract.
    from elpis.ECS_C.errors import EcsError
    with pytest.raises(ProjectionError, match="HISTORY_REJECTED") as excinfo:
        project_history("f" * 64, history[1], ProjectionRequest())
    cause = excinfo.value.__cause__
    assert isinstance(cause, EcsError)
    assert not isinstance(cause, ProjectionError)
