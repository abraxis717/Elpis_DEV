"""ERSA R4 read-path BREAK — independent of frozen fixture expectations.

The source fixtures and oracle are not modified. The tests distinguish:
(1) verified identity-scoped reads, (2) conditional score parity WHEN the
index covers the admissible records, and (3) unqualified open-world retrieval.
"""
from __future__ import annotations
import copy
import random
from dataclasses import replace

import pytest

from elpis_ersa import bank_ersa as B
from elpis_ersa import contract_ersa as C
from elpis_ersa import fixtures_r2 as F2
from elpis_ersa import fixtures_ersa as F0
from elpis_ersa import harness_ersa as H
from elpis_ersa import oracle_ersa as O


class ScoreIndexed(B.ErsBank):
    def begin_session(self, spec=None, registry=None, transport=None, authority=None):
        a = dict(authority or {})
        a['modes'] = {'discovery':'score_indexed', 'selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)


class PhysicalIndexed(B.ErsBank):
    def begin_session(self, spec=None, registry=None, transport=None, authority=None):
        a = dict(authority or {})
        a['modes'] = {'discovery':'indexed', 'selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)


def _fixture_query(b, q, *, exact_relation_id=None):
    return b.aggregate(q['qkey'], query_goal=q.get('qgoal'),
                       qdir=q.get('qdir'), k=q['k'],
                       exact_relation_id=exact_relation_id)


def _actual(b):
    return [int(rid) for rid, _ in getattr(b, '_last_selection', ())]


@pytest.mark.parametrize('k', [4, 8])
def test_frozen_prospective_score_indexed_parity_12_of_12(k):
    for fam in F2.FAMILIES:
        sc=F2.build(fam,cand=k)
        r=H.evaluate_backend(ScoreIndexed,sc,fam)
        assert r['exact_match'] is True, fam
    # This is parity with a dense scoring heuristic, not the separate
    # forbidden-neighbour policy oracle.


@pytest.mark.parametrize('k', [4,8])
def test_old_r0_not_claimed_exact(k):
    results={name: H.evaluate_backend(ScoreIndexed, F0.build(name), name)
             for name in F0.FAMILIES}
    # Some old fixtures build fixed max_candidates; changing k here does not
    # change the frozen source or certify universal sparse recall.
    assert sum(r['exact_match'] for r in results.values()) == 16


@pytest.mark.parametrize('rid', [0,2,4])
def test_identity_scope_beats_duplicate_key_and_score(rid):
    sc=F2.build('near_identical_keys_distinct_identity',cand=4)
    b=H.replay(ScoreIndexed,sc)
    q=sc['queries'][0]
    assert b.query_candidates(q['qkey'], exact_relation_id=rid).status == C.STATUS_OK
    assert b._last_candidates == [rid]
    r=_fixture_query(b,q,exact_relation_id=rid)
    assert r.status == C.STATUS_OK
    assert _actual(b) == [rid]
    assert [x['relation_id'] for x in b._last_evidence] == [rid]
    assert b._last_scored_count == 0


def test_frozen_forbidden_neighbour_unscoped_still_exposes_gap():
    sc=F2.build('near_identical_keys_distinct_identity',cand=8)
    assert H.evaluate_backend(ScoreIndexed,sc,'negative')['forbidden_leak'] is True
    # The frozen dense oracle also leaks; a dense parity PASS is not an
    # authorization or semantic safety PASS.
    class Dense(B.ErsBank):
        def begin_session(self,spec=None,registry=None,transport=None,authority=None):
            a=dict(authority or {});a['modes']={'discovery':'dense','selection':'ersa'}
            return super().begin_session(spec,registry,transport,a)
    assert H.evaluate_backend(Dense,sc,'negative')['forbidden_leak'] is True


def test_identity_scope_resolves_sparse_evidence_without_index_overlap():
    sc=F2.build('sparse_evidence',cand=4)
    b=H.replay(ScoreIndexed,sc);q=sc['queries'][0]
    assert b.select(q['qkey'], k=q['k']).status == C.STATUS_OK
    assert 9 not in _actual(b)
    assert _fixture_query(b,q,exact_relation_id=9).status == C.STATUS_OK
    assert _actual(b) == [9]


@pytest.mark.parametrize('bad', [True, 1.0, '0', (0,), -42, 100000])
def test_scoped_identity_rejects_bad_or_missing_without_stale_evidence(bad):
    sc=F2.build('near_identical_keys_distinct_identity',cand=8)
    b=H.replay(ScoreIndexed,sc);q=sc['queries'][0]
    assert _fixture_query(b,q,exact_relation_id=0).status == C.STATUS_OK
    assert b._last_evidence
    r=_fixture_query(b,q,exact_relation_id=bad)
    assert r.status == C.STATUS_REJECTED
    assert r.code in ('INVALID_ADDRESS','UNKNOWN_RELATION')
    assert not b._last_selection
    assert not b._last_slots_sel
    assert not b._last_candidates
    assert not b._last_evidence
    assert b._last_aggregate is None


def test_scoped_identity_survives_eviction_of_other_record():
    sc=F2.build('near_identical_keys_distinct_identity',cand=8)
    b=H.replay(ScoreIndexed,sc);q=sc['queries'][0]
    assert b.evict(3).status == C.STATUS_OK
    assert _fixture_query(b,q,exact_relation_id=0).status == C.STATUS_OK
    assert _actual(b)==[0]
    assert _fixture_query(b,q,exact_relation_id=3).status == C.STATUS_REJECTED


def test_auth_expires_after_tick_and_is_rejected_by_identity():
    sp=F2.spec(cap=4,cand=4)
    r=F2.rec(42, 1, 442, valid_until=0)
    b=ScoreIndexed()
    assert b.begin_session(sp,[r],None,{}).status == C.STATUS_OK
    assert b.insert_relation(relation_id=42,key=(1,2,3,4),value4=(1,2,3,4),
                             provenance=1,participant=0,role=0,ordinal=42,
                             ticket=442,goal=(1,2,3,4)).status==C.STATUS_OK
    assert b.query_candidates((1,2,3,4),exact_relation_id=42).status==C.STATUS_OK
    assert b.step(1).status==C.STATUS_OK
    t=b.aggregate((1,2,3,4),exact_relation_id=42)
    assert t.status==C.STATUS_REJECTED
    assert t.code=='UNAUTHORIZED_WRITE'
    assert b._last_evidence==[]


def _scenario(seed, cap=16, cand=4):
    rng=random.Random(seed)
    ins=[];records=[]
    for i in range(cap):
        rid=200+i;prov=i%3;ticket=1000+rid
        records.append(F2.rec(rid,prov,ticket))
        key=(rng.randint(-100,100),rng.randint(-100,100),rng.randint(-100,100),7)
        ins.append(dict(relation_id=rid,key=key,value4=tuple(rng.uniform(-2,2) for _ in range(4)),
                        provenance=prov,participant=0,role=0,ordinal=rid,ticket=ticket,
                        utility=rng.uniform(.0,2.),credibility=rng.uniform(.1,1.),
                        goal=tuple(rng.uniform(-1,1) for _ in range(4))))
    q=dict(qkey=(rng.randint(-50,50),rng.randint(-50,50),rng.randint(-50,50),7),
           qgoal=tuple(rng.uniform(-1,1) for _ in range(4)),
           qdir=tuple(rng.uniform(-2,2) for _ in range(4)), k=cand,
           required=[], forbidden=[])
    return dict(spec=replace(F2.spec(cap=cap,cand=cand),max_ops_per_tick=max(256,cap+32)),records=records,authority={},
                inserts=ins,queries=[q])


@pytest.mark.parametrize('seed', list(range(20)))
def test_metamorphic_full_index_coverage_matches_independent_dense_oracle(seed):
    sc=_scenario(seed,cand=4)
    b=H.replay(ScoreIndexed,sc);orc=H._replay_oracle(sc)
    q=sc['queries'][0]
    r=b.select(q['qkey'],query_goal=q['qgoal'],qdir=q['qdir'],k=q['k'])
    assert r.status==C.STATUS_OK
    assert b._last_scored_count==len(sc['inserts'])  # complete overlap in PK4
    expected=[rid for rid,_ in orc.expected_selection(q['qkey'],q['qgoal'],q['k'],q['qdir'])]
    assert _actual(b)==expected
    # Permute physical admission order; identity must not be slot-based.
    perm=copy.copy(sc)
    perm['inserts']=list(reversed(sc['inserts']))
    b2=H.replay(ScoreIndexed,perm)
    b2.select(q['qkey'],query_goal=q['qgoal'],qdir=q['qdir'],k=q['k'])
    assert _actual(b2)==expected


def test_mutant_physical_slot_order_fails_conditional_parity():
    mismatches=[]
    for fam in F2.FAMILIES:
        sc=F2.build(fam,cand=8)
        r=H.evaluate_backend(PhysicalIndexed,sc,fam)
        if not r['exact_match']: mismatches.append(fam)
    assert len(mismatches)>=5  # predecessor mutation is detectable


def test_hard_ceiling_scored_union_and_fixed_arrays():
    sc=_scenario(998,cap=128,cand=4)
    b=H.replay(ScoreIndexed,sc);q=sc['queries'][0]
    bytes_before=sum(x.nbytes for x in b._arrays.values())+sum(
        x.nbytes for x in b._index.values() if hasattr(x,'nbytes'))
    for _ in range(8):
        b.select(q['qkey'],q['qgoal'],qdir=q['qdir'],k=4)
        assert 0<=b._last_scored_count<=b._spec.max_active_relations
        assert len(b._last_candidates)<=b._spec.max_candidates
    bytes_after=sum(x.nbytes for x in b._arrays.values())+sum(
        x.nbytes for x in b._index.values() if hasattr(x,'nbytes'))
    assert bytes_after == bytes_before


def test_uncovered_query_remains_unqualified_not_mislabeled_zero_evidence():
    sc=F2.build('sparse_evidence',cand=8)
    b=H.replay(ScoreIndexed,sc)
    q=sc['queries'][0]
    b.query_candidates(q['qkey'],q['qgoal'])
    assert 9 not in b._last_candidates
    assert H.evaluate_backend(ScoreIndexed,sc,'sparse')['candidate_recall']==0.0
