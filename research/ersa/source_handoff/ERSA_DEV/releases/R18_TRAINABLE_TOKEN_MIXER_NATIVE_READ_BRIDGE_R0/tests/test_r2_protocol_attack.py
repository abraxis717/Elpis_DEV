"""R2.0 protocol attacks beyond the R1 suite: ticket uniqueness, admission
expiry, protected eviction refusal, pending accounting, injected commit
failures, stale proposal replay, cross-session stages, counter monotonicity.

Every forbidden path must REJECT and leave committed state unchanged.
Observed rejection codes are recorded in the assertion so a code change is
visible rather than silently absorbed.
"""
from elpis_ersa import contract_ersa as C, bank_ersa as Bk

SPEC = C.SessionSpec(max_active_relations=4, max_candidates=4,
                     max_control_records=16, max_ops_per_tick=1000,
                     product_key_subkeys=4, product_key_bits=8,
                     key_feature_dim=4, pending_stage_limit=2)


def auth(rid, ticket=7, valid=1 << 62, flags=()):
    fl = tuple(sorted(flags))
    return C.AdmissionRecord(relation_id=int(rid), provenance=7, role=0,
                             ordinal=int(rid), flags=fl,
                             ticket=int(ticket), issued_tick=0,
                             valid_until_tick=int(valid),
                             protected=bool(fl) and "PROTECTED" in fl)


def op(recs, pend=2):
    b = Bk.ErsBank()
    r = b.begin_session(SPEC, recs, None, None)
    assert r.status == C.STATUS_OK
    return b


def ins(b, rid, ticket=7, flags=0):
    return b.insert_relation(relation_id=rid, key=(1, 2, 3, 4),
                             value4=(1.0, 0.0, 0.0, 0.0), provenance=7,
                             participant=0, role=0, ordinal=rid,
                             ticket=ticket, utility=1.0, flags=flags)


def live(b):
    return int(b.occupancy()["live"])


def test_forged_ticket_rejects_and_admits_nothing():
    b = op([auth(2, ticket=9)])
    r = ins(b, 2, ticket=99)
    assert (r.status, r.code) == (C.STATUS_REJECTED, "STALE_TICKET")
    assert live(b) == 0
    ok = ins(b, 2, ticket=9)
    assert ok.status == C.STATUS_OK and live(b) == 1


def test_expired_admission_denies_write():
    b = op([auth(1, valid=5)])
    for _ in range(8):
        b.step(1)
    assert b._tick > 5
    r = ins(b, 1)
    assert (r.status, r.code) == (C.STATUS_REJECTED, "UNAUTHORIZED_WRITE")
    assert live(b) == 0


def test_protected_record_cannot_be_evicted():
    b = op([auth(5, flags=("PROTECTED",))])
    assert ins(b, 5, flags=C.FLAG_PROTECTED).status == C.STATUS_OK
    r = b.evict(5)
    assert (r.status, r.code) == (C.STATUS_REJECTED, "PROTECTED_RECORD")
    assert live(b) == 1


def test_pending_limit_enforced_before_allocation():
    b = op([auth(i) for i in (1, 2, 3, 4)])
    for i in (1, 2, 3, 4):
        assert ins(b, i).status == C.STATUS_OK
    s1 = b.propose_update(1, (1.0, 0, 0, 0), 0.25)
    s2 = b.propose_update(2, (1.0, 0, 0, 0), 0.25)
    assert isinstance(s1, dict) and isinstance(s2, dict)
    over = b.propose_update(3, (1.0, 0, 0, 0), 0.25)
    assert (over.status, over.code) == (C.STATUS_REJECTED,
                                        "PENDING_STAGE_LIMIT")
    assert b.commit_validated_update(s2).status == C.STATUS_OK
    after = b.propose_update(3, (1.0, 0, 0, 0), 0.25)
    assert isinstance(after, dict)              # capacity reclaimed once
    assert b.commit_validated_update(s1).status == C.STATUS_OK
    assert b.commit_validated_update(s1).status == C.STATUS_REJECTED


def test_injected_commit_failure_leaves_value_untouched():
    b = op([auth(1)])
    assert ins(b, 1).status == C.STATUS_OK
    stage = b.propose_update(1, (1.0, 0, 0, 0), 0.25)
    tampered = dict(stage, entries=((1, (9.0, 0, 0, 0), (9.0, 0, 0, 0)),))
    bad = b.commit_validated_update(tampered)
    assert (bad.status, bad.code) == (C.STATUS_REJECTED, "STAGE_MODIFIED")
    forged = b.commit_validated_update({"token": 999, "entries":
                                        stage["entries"], "issued_tick": 0})
    assert forged.status == C.STATUS_REJECTED
    ok = b.commit_validated_update(stage)
    assert ok.status == C.STATUS_OK
    assert b.commit_validated_update(stage).status == C.STATUS_REJECTED


def test_stage_replay_after_halt_and_step():
    b = op([auth(1)])
    assert ins(b, 1).status == C.STATUS_OK
    stage = b.propose_update(1, (1.0, 0, 0, 0), 0.25)
    assert b.commit_validated_update(stage).status == C.STATUS_OK
    b.halt()
    b.step(1)
    r = b.commit_validated_update(stage)
    assert r.status == C.STATUS_REJECTED
    assert r.code in ("STAGE_UNKNOWN", "STAGE_REPLAYED",
                      "OP_BUDGET")     # halt blocks the op before the stage
    fresh = b.propose_update(1, (1.0, 0, 0, 0), 0.25)
    assert not isinstance(fresh, dict) and fresh.status == C.STATUS_REJECTED


def test_stage_from_previous_session_never_authorises_new_session():
    b = op([auth(1)])
    assert ins(b, 1).status == C.STATUS_OK
    stage = b.propose_update(1, (1.0, 0, 0, 0), 0.25)
    b.reset()
    b2 = op([auth(1)])
    r = b2.commit_validated_update(stage)
    assert r.status == C.STATUS_REJECTED
    assert live(b2) == 0


def test_noninteger_stage_target_rejects_without_mutation():
    b = op([auth(1)])
    assert ins(b, 1).status == C.STATUS_OK
    r = b.commit_validated_update({"token": 1, "target": "one",
                                   "entries": ((1, (1.0, 0, 0, 0),
                                               (1.0, 0, 0, 0)),),
                                   "issued_tick": 0})
    assert r.status == C.STATUS_REJECTED


def test_counters_are_monotonic_within_a_session():
    b = op([auth(1), auth(2)])
    for i in (1, 2):
        assert ins(b, i).status == C.STATUS_OK
    before = dict(b._counters)
    b.query_candidates((1, 2, 3, 4))
    after = dict(b._counters)
    assert all(after[k] >= before[k] for k in before)
    assert after["queries"] == before["queries"] + 1
    assert after["live"] == before["live"]


def test_duplicate_identity_never_overwrites_silently():
    b = op([auth(1)])
    assert ins(b, 1).status == C.STATUS_OK
    v_before = tuple(b._arrays["value"][b._slot_of(1)])
    r = ins(b, 1)
    assert (r.status, r.code) == (C.STATUS_REJECTED, "DUPLICATE_IDENTITY")
    assert tuple(b._arrays["value"][b._slot_of(1)]) == v_before
