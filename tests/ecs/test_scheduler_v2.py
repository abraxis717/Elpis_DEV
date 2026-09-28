from itertools import permutations

import pytest

from elpis.ecs import scheduler


def item(
    rank,
    eid,
    idx,
    mid,
    *,
    clock=None,
    founding_index=None,
    kind="PROCESS_MESSAGE",
):
    return scheduler.ReadyItem(
        rank=rank,
        entity_id=eid,
        mailbox_index=idx,
        message_id=mid,
        kind=kind,
        ref=mid,
        ready_clock=clock,
        founding_index=founding_index,
    )


def test_v1_exact_compatibility_key_is_unchanged():
    x = item(0, "b", 3, "m", clock=99, founding_index=77)
    assert x.ordering_key() == (0, "b", 3, "m")
    assert x.ordering_key(scheduler.SCHEDULER_V1) == (0, "b", 3, "m")


def test_v2_older_committed_message_beats_grinded_low_entity_id():
    grinded_low = item(0, "00000001", 0, "new", clock=20)
    honest_high = item(0, "ffffffff", 0, "old", clock=10)

    assert scheduler.order_ready(
        [honest_high, grinded_low],
        protocol=scheduler.SCHEDULER_V1,
    )[0] is grinded_low

    assert scheduler.order_ready(
        [grinded_low, honest_high],
        protocol=scheduler.SCHEDULER_V2,
    )[0] is honest_high


def test_v2_refills_cannot_overtake_already_queued_message():
    honest = item(0, "ffffffff", 0, "honest", clock=10)
    refills = [
        item(0, "00000001", i, f"refill-{i}", clock=11+i)
        for i in range(12)
    ]
    ordered = scheduler.order_ready(
        [*reversed(refills), honest],
        protocol=scheduler.SCHEDULER_V2,
    )
    assert ordered[0] is honest
    assert [x.ready_clock for x in ordered] == list(range(10, 23))


def test_v2_message_order_is_input_order_independent():
    items = [
        item(0, "c", 0, "m3", clock=3),
        item(0, "a", 0, "m1", clock=1),
        item(0, "b", 0, "m2", clock=2),
    ]
    expected = ["m1", "m2", "m3"]
    for perm in permutations(items):
        assert [
            x.message_id for x in scheduler.order_ready(
                perm, protocol=scheduler.SCHEDULER_V2
            )
        ] == expected


def test_v2_lifecycle_uses_kernel_assigned_founding_index():
    grinded_low = item(
        scheduler.RANK_ACTIVATE,
        "00000001",
        0,
        "",
        founding_index=9,
        kind="LIFECYCLE",
    )
    honest_high = item(
        scheduler.RANK_ACTIVATE,
        "ffffffff",
        0,
        "",
        founding_index=2,
        kind="LIFECYCLE",
    )
    ordered = scheduler.order_ready(
        [grinded_low, honest_high],
        protocol=scheduler.SCHEDULER_V2,
    )
    assert ordered == [honest_high, grinded_low]


@pytest.mark.parametrize(
    "candidate",
    [
        item(0, "a", 0, "m", clock=None),
        item(0, "a", 0, "m", clock=0),
        item(0, "a", 0, "m", clock=True),
    ],
)
def test_v2_message_missing_or_invalid_clock_fails_closed(candidate):
    with pytest.raises(
        scheduler.SchedulerError,
        match="READY_CLOCK_REQUIRED",
    ):
        scheduler.order_ready(
            [candidate],
            protocol=scheduler.SCHEDULER_V2,
        )


@pytest.mark.parametrize(
    "candidate",
    [
        item(1, "a", 0, "", founding_index=None, kind="LIFECYCLE"),
        item(1, "a", 0, "", founding_index=-1, kind="LIFECYCLE"),
        item(1, "a", 0, "", founding_index=True, kind="LIFECYCLE"),
    ],
)
def test_v2_lifecycle_missing_or_invalid_founding_index_fails_closed(candidate):
    with pytest.raises(
        scheduler.SchedulerError,
        match="FOUNDING_INDEX_REQUIRED",
    ):
        scheduler.order_ready(
            [candidate],
            protocol=scheduler.SCHEDULER_V2,
        )


def test_unknown_scheduler_profile_fails_closed():
    with pytest.raises(
        scheduler.SchedulerError,
        match="UNKNOWN_SCHEDULER_PROTOCOL",
    ):
        scheduler.order_ready([], protocol="scheduler.invalid.v999")
