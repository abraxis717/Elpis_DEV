"""R5 external BREAK: complete discovery vs evidence selection; explicit scoped reads.

All task-target IDs come from frozen prospective fixtures, never injected into
unscoped ranking. Scope is caller-provided, NOT an authorization credential.
"""
import copy, random
from dataclasses import replace
import pytest
from elpis_ersa import bank_ersa as B,fixtures_r2 as F,contract_ersa as C,harness_ersa as H,oracle_ersa as O,geometry_ersa as G


class Complete(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        a=dict(authority or {});a['modes']={'discovery':'score_complete','selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)


def populated(family='contradictory_attested',k=8):
    sc=F.build(family,cand=k)
    return H.replay(Complete,sc),sc


def call(b,q,ids,**kw):
    return b.aggregate_scoped(ids,q['qkey'],q.get('qgoal'),q.get('qdir'),**kw)


def state(b):
    return (list(getattr(b,'_last_candidates',[])),list(getattr(b,'_last_evidence',[])),getattr(b,'_last_aggregate',None))


def test_full_scan_is_complete_not_grounded_selection():
    b,sc=populated('sparse_evidence');q=sc['queries'][0]
    assert b.query_candidates(q['qkey'],q['qgoal'],qdir=q['qdir']).status=='OK'
    assert 9 in b._last_candidates and b._last_scored_count==12
    assert b.aggregate(q['qkey'],q['qgoal'],qdir=q['qdir'],k=q['k']).status=='OK'
    assert 9 not in [rid for rid,_ in b._last_selection]
    assert call(b,q,(9,)).status=='OK'
    assert [e['relation_id'] for e in b._last_evidence]==[9]


def test_explicit_attested_pair_matches_independent_geometry_oracle():
    b,sc=populated();q=sc['queries'][0]
    o=H._replay_oracle(sc)
    selected=[o.features(r,q['qkey'],q['qgoal'],q['qdir']) for r in (0,1)]
    tv=[o.transported(r) for r in (0,1)]
    w=[o.score(f) for f in selected]
    exp,_=G.aggregate_signed(tv,w)
    assert b.aggregate_reciprocal_contradiction(0,q['qkey'],q['qgoal'],q['qdir']).status=='OK'
    assert [x['relation_id'] for x in b._last_evidence]==[0,1]
    assert max(abs(x-y) for x,y in zip(exp,b._last_aggregate))<1e-12


def test_forbidden_neighbor_excluded_when_explicit_exact_scope():
    b,sc=populated('near_identical_keys_distinct_identity');q=sc['queries'][0]
    assert call(b,q,(0,)).status=='OK'
    assert [x['relation_id'] for x in b._last_evidence]==[0]
    assert 1 not in b._last_candidates


@pytest.mark.parametrize('bad', [(),(0,0),(True,),('0',),(0.0,),(-1,),(99999,),tuple(range(9)),None])
def test_scope_invalid_rejects_atomically(bad):
    b,sc=populated();q=sc['queries'][0]
    assert call(b,q,(0,)).status=='OK'
    rc=call(b,q,bad)
    assert rc.status=='REJECTED'
    assert state(b)==([],[],None)


def test_rejects_missing_member_without_partial_read():
    b,sc=populated();q=sc['queries'][0]
    rc=call(b,q,(0,9000))
    assert rc.status=='REJECTED' and state(b)==([],[],None)


def test_expiry_revokes_entire_scope():
    sp=F.spec(cap=4,cand=4)
    records=[F.rec(i,0,100+i,valid_until=(0 if i==1 else 10)) for i in (0,1)]
    bank=Complete();assert bank.begin_session(sp,records,None,{}).status=='OK'
    for i in (0,1):
        assert bank.insert_relation(relation_id=i,key=(i,1,1,1),value4=(1.,0.,0.,0.),
                                    provenance=0,participant=0,role=0,ordinal=i,ticket=100+i).status=='OK'
    assert bank.step(1).status=='OK'
    rc=bank.aggregate_scoped((0,1),(0,1,1,1))
    assert rc.status=='REJECTED' and state(bank)==([],[],None)


def test_snapshot_revision_and_epoch_pin_blocks_reinsert_aba():
    b,sc=populated();q=sc['queries'][0]
    s=b._slot_of(0);original=int(b._arrays['revision'][s]); epoch=b._epoch
    assert call(b,q,(0,),revision_pins=(original,),expected_epoch=epoch).status=='OK'
    assert b.evict(0).status=='OK'
    rec=sc['inserts'][0]
    assert b.insert_relation(**rec).status=='OK'
    assert int(b._arrays['revision'][b._slot_of(0)])!=original
    assert call(b,q,(0,),revision_pins=(original,),expected_epoch=epoch).status=='REJECTED'
    assert state(b)==([],[],None)
    assert call(b,q,(0,),expected_epoch=epoch+1).status=='REJECTED'


def test_reset_reopen_invalidate_epoch_pin():
    b,sc=populated();q=sc['queries'][0];old=b._epoch
    assert b.reset().status=='OK'
    assert b.begin_session(sc['spec'],sc['records'],None,{}).status=='OK'
    assert b.insert_relation(**sc['inserts'][0]).status=='OK'
    assert call(b,q,(0,),expected_epoch=old).status=='REJECTED'
    assert state(b)==([],[],None)


def test_tampered_revision_format_and_input_validation():
    b,sc=populated();q=sc['queries'][0]
    for bad in ((0,), (0,1), ('x',), (True,)):
        rc=call(b,q,(0,),revision_pins=bad)
        assert rc.status=='REJECTED' and state(b)==([],[],None)
    assert call(b,q,(0,),expected_epoch=True).status=='REJECTED'
    for qkey in ((1,2,3), (True,1,2,3), (1,2,3,'4')):
        assert b.aggregate_scoped((0,),qkey).status=='REJECTED'
        assert state(b)==([],[],None)


def test_nonreciprocal_and_dangling_contradiction_fail_closed():
    for target in (2,90,-1,0):
        b,sc=populated('dense_evidence')
        b._arrays['contra'][b._slot_of(0)]=target
        q=sc['queries'][0]
        rc=b.aggregate_reciprocal_contradiction(0,q['qkey'],q['qgoal'],q['qdir'])
        assert rc.status=='REJECTED' and state(b)==([],[],None)


def test_pair_revoked_during_tick_fails_closed():
    b,sc=populated()
    q=sc['queries'][0]
    b._authority['denied']=frozenset((1,))
    rc=b.aggregate_reciprocal_contradiction(0,q['qkey'],q['qgoal'],q['qdir'])
    assert rc.status=='REJECTED' and state(b)==([],[],None)


@pytest.mark.parametrize('seed', list(range(16)))
def test_complete_dense_selection_parity_in_randomized_corpus(seed):
    rng=random.Random(seed+100)
    cap=12
    spec=replace(F.spec(cap=cap,cand=4),max_ops_per_tick=64)
    records=[F.rec(i,0,100+i) for i in range(cap)]
    b=Complete();o=O.NaiveOracle(spec,records,None,{})
    assert b.begin_session(spec,records,None,{}).status=='OK'
    inputs=[]
    for i in range(cap):
        inp=dict(relation_id=i,key=tuple(rng.randrange(-100,100) for _ in range(4)),
                 value4=tuple(rng.uniform(-1,1) for _ in range(4)),
                 provenance=0,participant=0,role=0,ordinal=i,ticket=100+i,
                 goal=tuple(rng.uniform(-1,1) for _ in range(4)))
        inputs.append(inp)
        assert b.insert_relation(**inp).status=='OK'
        o.insert(**inp)
    for _ in range(3):
        qk=tuple(rng.randrange(-20,20) for _ in range(4))
        qd=tuple(rng.uniform(-1,1) for _ in range(4))
        assert b.select(qk,k=4,qdir=qd).status=='OK'
        assert [r for r,_ in b._last_selection]==[r for r,_ in o.expected_selection(qk,None,4,qd)]
        assert b._last_scored_count==cap


def test_complete_is_explicitly_not_sparse_free_lunch():
    b,sc=populated('sparse_evidence')
    q=sc['queries'][0]
    assert b.select(q['qkey'],q['qgoal'],qdir=q['qdir']).status=='OK'
    assert b._last_scored_count==len(sc['inserts'])
    assert b._last_scored_count>b._spec.max_candidates
    assert b.occupancy()['live']==12


def test_mutant_product_key_sparse_discovery_does_not_get_target():
    from elpis_ersa import bank_ersa
    class Index(B.ErsBank):
        def begin_session(self,spec=None,registry=None,transport=None,authority=None):
            a=dict(authority or {});a['modes']={'discovery':'score_indexed','selection':'ersa'}
            return super().begin_session(spec,registry,transport,a)
    sc=F.build('sparse_evidence',cand=8);b=H.replay(Index,sc);q=sc['queries'][0]
    b.query_candidates(q['qkey'],q['qgoal'],qdir=q['qdir'])
    assert 9 not in b._last_candidates


def test_reciprocal_read_is_exactly_one_budgeted_operation():
    sp=replace(F.spec(cap=2,cand=2),max_ops_per_tick=1)
    rec=[F.rec(i,0,100+i) for i in (0,1)]
    b=Complete();assert b.begin_session(sp,rec).status=='OK'
    for i,peer in ((0,1),(1,0)):
        assert b.insert_relation(relation_id=i,key=(i,0,0,0),value4=(1,0,0,0),
            provenance=0,participant=0,role=0,ordinal=i,ticket=100+i,
            contradicts=peer).status=='OK'
        assert b.step(1).status=='OK'
    r=b.aggregate_reciprocal_contradiction(0,(0,0,0,0))
    assert r.status=='OK' and b._ops_this_tick==1
    assert b.aggregate_reciprocal_contradiction(0,(0,0,0,0)).status=='REJECTED'
    assert state(b)==([],[],None)


def test_exclusion_of_entire_set_on_one_revoked_provenance():
    b,sc=populated('contradictory_attested');q=sc['queries'][0]
    b._authority['denied']=frozenset((1,))
    assert call(b,q,(0,1)).status=='REJECTED'
    assert state(b)==([],[],None)


def test_scoped_operator_equivariant_when_participant_input_is_transformed():
    sc=F.build('gauge_equivalent_queries');q=sc['queries'][0]
    gam,gp=sc['gauge']
    b=H.replay(Complete,sc)
    bt=H.replay(Complete,sc,gauge=(gam,gp))
    # The full gauge transformation requires x'_participant=Gp*x.
    # The inherited prospective harness transforms U but does not rotate x.
    for j in range(bt._spec.max_active_relations):
        if bt._arrays['live'][j]:
            part=int(bt._arrays['part'][j])
            old=tuple(float(x) for x in bt._arrays['value'][j])
            bt._arrays['value'][j]=G.rho_apply(gp[part],old)
    assert b.aggregate_scoped((6,),q['qkey'],q['qgoal'],q['qdir']).status=='OK'
    qd=G.rho_apply(gam,q['qdir'])
    assert bt.aggregate_scoped((6,),q['qkey'],q['qgoal'],qd).status=='OK'
    assert G.max_abs_diff(G.rho_apply(gam,b._last_aggregate),
                          bt._last_aggregate)<1e-12


def test_frozen_r2_prospective_gauge_mismatch_not_hidden():
    sc=F.build('gauge_equivalent_queries')
    mismatch=H.evaluate_backend(Complete,sc,'gauge')['gauge_covariance']
    assert mismatch>0.1   # Negative control retained; not a production pass.
