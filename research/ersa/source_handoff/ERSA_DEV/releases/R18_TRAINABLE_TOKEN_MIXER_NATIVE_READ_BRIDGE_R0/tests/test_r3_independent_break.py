"""Independent R3 BREAK: reproduce known R2 review defects and negative controls.

These tests are not in the frozen R2 suite.  No catch-all test assertion
swallowing.  Do not call them a neural/semantic qualification.
"""
import numpy as np
import pytest
from elpis_ersa import bank_ersa as B, contract_ersa as C


def make(cap=3, stages=3, protected=(), flags_protected=(), auth_protected=(),
         count=3, ops=400):
    b = B.ErsBank()
    recs = []
    for rid in range(1, count + 1):
        recs.append(C.AdmissionRecord(
            rid, 7, 0, rid,
            ('PROTECTED',) if rid in flags_protected else (),
            7, 0, 1<<62, rid in protected))
    spec=C.SessionSpec(max_active_relations=cap,max_candidates=cap,
                       max_ops_per_tick=ops,pending_stage_limit=stages,
                       max_control_records=max(count,cap),
                       relational_namespace=(1, max(count, cap, 4096)))
    assert b.begin_session(spec,recs,authority={"protected_relations":tuple(auth_protected)}).status==C.STATUS_OK
    return b


def ins(b,rid,flags=0,v=(1.,0.,0.,0.)):
    return b.insert_relation(relation_id=rid,key=(rid,0,0,0),value4=v,
                             provenance=7,role=0,ordinal=rid,flags=flags,
                             ticket=7)


def stage(b, rid=1):
    s=b.propose_update(rid,(1.,0.,0.,0.),0.25)
    assert isinstance(s,dict)
    return s


def rejected(r, code=None):
    assert isinstance(r,C.Receipt) and r.status==C.STATUS_REJECTED, r
    if code: assert r.code==code, r


def test_f01_aba_identical_reinsert_fails_closed():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    s=stage(b)
    assert b.evict(1).status==C.STATUS_OK
    assert ins(b,1).status==C.STATUS_OK
    before=b.snapshot()
    rejected(b.commit_validated_update(s),'STALE_PROPOSAL')
    assert b.snapshot()==before


def test_f02_aba_repeat_cycles_no_revival():
    b=make(count=1,stages=8)
    assert ins(b,1).status==C.STATUS_OK
    stale=[]
    for i in range(5):
        s=stage(b)
        stale.append(s)
        assert b.evict(1).status==C.STATUS_OK
        assert ins(b,1).status==C.STATUS_OK
        snap=b.snapshot()
        for x in stale:
            rejected(b.commit_validated_update(x),'STALE_PROPOSAL')
            assert b.snapshot()==snap


def test_f03_slot_resolved_once_not_reresolved_after_validation():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    s=stage(b)
    original=b._slot_of
    count=[]
    def once(rid):
        count.append(rid)
        if len(count)==1:return original(rid)
        return None
    b._slot_of=once
    assert b.commit_validated_update(s).status==C.STATUS_OK
    assert count==[1]  # historical write path made a second lookup
    assert b.snapshot()['relations'][1]['value'][0]==1.25


def test_f04_zero_pending_exactly_zero_not_upgraded():
    b=make(count=1,stages=0)
    assert ins(b,1).status==C.STATUS_OK
    rejected(b.propose_update(1,(1.,0.,0.,0.)), 'PENDING_STAGE_LIMIT')
    assert len(b._pending)==0
    assert b._pending_used==0


def test_f05_stage_list_container_rejected():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    s=stage(b)
    wrong=dict(s, entries=list(s['entries']))
    before=b.snapshot()
    rejected(b.commit_validated_update(wrong),'STAGE_MODIFIED')
    assert b.snapshot()==before
    assert b.commit_validated_update(s).status==C.STATUS_OK


@pytest.mark.parametrize('origin', ['rec','flag','authority'])
def test_f06_all_three_protection_sources_block_explicit_evict(origin):
    b=make(count=2,protected=(1,) if origin=='rec' else (),
           flags_protected=(1,) if origin=='flag' else (),
           auth_protected=(1,) if origin=='authority' else ())
    assert ins(b,1, C.FLAG_PROTECTED if origin=='flag' else 0).status==C.STATUS_OK
    before=b.snapshot()
    rejected(b.evict(1),'PROTECTED_RECORD')
    assert b.snapshot()==before


def test_f06_protected_admission_survives_capacity_eviction():
    b=make(cap=2,count=3,protected=(1,))
    assert ins(b,1).status==C.STATUS_OK
    assert ins(b,2).status==C.STATUS_OK
    assert ins(b,3).status==C.STATUS_OK
    assert 1 in b.snapshot()['relations']
    assert 2 not in b.snapshot()['relations']


def test_f06_all_protected_refuses_capacity_eviction():
    b=make(cap=2,count=3,protected=(1,2))
    assert ins(b,1).status==C.STATUS_OK
    assert ins(b,2).status==C.STATUS_OK
    before=b.snapshot()
    rejected(ins(b,3), 'PROTECTED_CAPACITY_FULL')
    assert b.snapshot()==before


def test_f06_protected_admission_limits_gain():
    b=make(count=1,protected=(1,))
    assert ins(b,1).status==C.STATUS_OK
    rejected(b.propose_update(1,(1.,0.,0.,0.),0.75),'PROTECTED_RECORD')


def test_f07_bookkeeping_fault_rolls_back_all_state():
    class FaultDict(dict):
        fail_once=True
        def __setitem__(self, key, value):
            if key=='commits' and self.fail_once:
                self.fail_once=False
                raise RuntimeError('injected postwrite accounting fault')
            super().__setitem__(key,value)
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    s=stage(b)
    before=b.snapshot()
    b._counters=FaultDict(b._counters)
    rejected(b.commit_validated_update(s),'INVALID_VALUE')
    assert b.snapshot()==before
    assert b._pending_used==1
    assert s['token'] in b._stage_index
    assert b.commit_validated_update(s).status==C.STATUS_OK
    assert b._pending_used==0


def test_f08_partial_numpy_write_fault_rolled_back():
    class OneFault(np.ndarray):
        injected=True
        def __setitem__(self,key,val):
            super().__setitem__(key,val)
            if self.injected:
                self.injected=False
                raise RuntimeError('injected after partial write')
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    s=stage(b)
    before=b.snapshot()
    b._arrays['value']=b._arrays['value'].view(OneFault)
    rejected(b.commit_validated_update(s),'INVALID_VALUE')
    assert b.snapshot()==before
    assert b._pending_used==1
    assert b.commit_validated_update(s).status==C.STATUS_OK


def test_f09_missing_target_rejected_and_all_values_untouched():
    b=make(count=2)
    assert ins(b,1).status==C.STATUS_OK
    assert ins(b,2).status==C.STATUS_OK
    s=stage(b)
    before=b.snapshot()
    b._slot_of=lambda rid: None
    rejected(b.commit_validated_update(s),'UNKNOWN_RELATION')
    assert b.snapshot()==before


def test_f10_no_none_index_broadcast():
    b=make(count=3)
    for i in (1,2,3):assert ins(b,i).status==C.STATUS_OK
    s=stage(b)
    before=b.snapshot()
    old=b._slot_of
    times=[]
    def fake(r):
        times.append(1)
        return None if len(times)>1 else old(r)
    b._slot_of=fake
    assert b.commit_validated_update(s).status==C.STATUS_OK
    after=b.snapshot()
    assert after['relations'][1]['value'][0]==1.25
    assert after['relations'][2]==before['relations'][2]
    assert after['relations'][3]==before['relations'][3]
    assert len(times)==1


def test_stage_conflict_detected_by_generation_even_if_value_is_same():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    a=stage(b)
    x=stage(b)
    assert b.commit_validated_update(a).status==C.STATUS_OK
    # After a +delta commit, a -delta commit returns exact same value.
    y=b.propose_update(1,(-1.,0.,0.,0.),0.25)
    assert b.commit_validated_update(y).status==C.STATUS_OK
    assert b.snapshot()['relations'][1]['value'][0]==1
    rejected(b.commit_validated_update(x),'STALE_PROPOSAL')


def test_revision_exhaustion_rejects_insert_before_victim_eviction():
    b=make(cap=1,count=2)
    assert ins(b,1).status==C.STATUS_OK
    b._revision_clock=(1<<63)-1
    before=b.snapshot()
    rejected(ins(b,2),'STALE_PROPOSAL')
    assert b.snapshot()==before


def test_commit_revision_exhaustion_does_not_consume_stage():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    st=stage(b)
    b._revision_clock=(1<<63)-1
    before=b.snapshot()
    rejected(b.commit_validated_update(st),'STALE_PROPOSAL')
    assert b.snapshot()==before
    assert st['token'] in b._stage_index


def test_stage_token_exhaustion_rejects_without_pending_allocation():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    b._token=(1<<63)-1
    rejected(b.propose_update(1,(1.,0.,0.,0.)),'STAGE_UNKNOWN')
    assert b._pending_used==0


def test_protection_flag_and_attestation_do_not_cancel_each_other():
    b=make(cap=2,count=3,protected=(1,), flags_protected=(2,))
    assert ins(b,1).status==C.STATUS_OK
    assert ins(b,2,C.FLAG_PROTECTED).status==C.STATUS_OK
    rejected(ins(b,3),'PROTECTED_CAPACITY_FULL')


def test_zero_stage_does_not_allocate_implicit_token():
    b=make(count=1, stages=0)
    assert ins(b,1).status==C.STATUS_OK
    for _ in range(5):rejected(b.propose_update(1,(1.,0.,0.,0.)),'PENDING_STAGE_LIMIT')
    assert getattr(b,'_token',0)==0


def test_reset_isolation_from_prior_generation():
    b=make(count=1)
    assert ins(b,1).status==C.STATUS_OK
    stale=stage(b)
    assert b.reset().status==C.STATUS_OK
    reg=[C.AdmissionRecord(1,7,0,1,(),7,0,1<<62,False)]
    assert b.begin_session(C.SessionSpec(max_active_relations=1,max_ops_per_tick=100),reg).status==C.STATUS_OK
    assert ins(b,1).status==C.STATUS_OK
    before=b.snapshot()
    rejected(b.commit_validated_update(stale),'STAGE_UNKNOWN')
    assert b.snapshot()==before


@pytest.mark.parametrize('method,args', [
    ('read_exact', ('R1',)),
    ('read_assoc', ((1,2,3,4),)),
    ('observe', ((1,2),)),
    ('upsert_exact', ('R1',(1,2,3,4),7)),
    ('propose_relation', (object(),)),
    ('apply_delta', (object(),)),
    ('transport_and_aggregate', ((),)),
    ('reinforce', (1,1.,1.)),
    ('decay', (1,)),
    ('evict', (1,)),
    ('step', (1,)),
])
def test_predecessor_adapter_mapping_fails_closed(method,args):
    from elpis_ersa.adapter_ssc import SscAdapter
    adapter=SscAdapter()
    b=make(count=1)
    adapter._bank=b
    assert ins(b,1).status==C.STATUS_OK
    before=b.snapshot()
    r=getattr(adapter,method)(*args)
    rejected(r)
    assert b.snapshot()==before
    assert adapter.predecessor_compatible is False



def test_admission_duplicate_identity_rejected_before_open():
    b=B.ErsBank()
    rec=C.AdmissionRecord(1,7,0,1,(),7,0,1<<62,False)
    bad=b.begin_session(C.SessionSpec(),[rec,rec])
    rejected(bad,'DUPLICATE_IDENTITY')
    assert b.occupancy()['live']==0


def test_admission_generator_bounded_no_unbounded_consumption():
    b=B.ErsBank()
    count=[0]
    def records():
        for i in range(100000000):
            count[0]+=1
            yield C.AdmissionRecord(i,7,0,i,(),7,0,1<<62,False)
    spec=C.SessionSpec(max_active_relations=2,max_control_records=4)
    rejected(b.begin_session(spec,records()),'CAPACITY_EXCEEDED')
    assert count[0]==5


def test_unknown_flag_bits_rejected_before_mutation():
    b=make(count=1)
    before=b.snapshot()
    rejected(ins(b,1,flags=16),'FIELD_VALIDATION')
    assert b.snapshot()==before


def test_invalid_registry_records_rejected_typed():
    b=B.ErsBank()
    rejected(b.begin_session(C.SessionSpec(),[object()]),'FIELD_VALIDATION')
    assert b.occupancy()['live']==0


@pytest.mark.parametrize('bad',[
    {'max_active_relations': 10**9},
    {'max_candidates': 10**9},
    {'max_control_records': 10**9},
    {'pending_stage_limit': 10**9},
    {'product_key_bits': 52},
    {'product_key_subkeys': 10**9},
    {'key_feature_dim': 10**9},
    {'adjacency_degree': 10**9},
])
def test_unbounded_session_spec_rejected_before_allocations(bad):
    b=B.ErsBank()
    spec=C.SessionSpec(**bad)
    rejected(b.begin_session(spec,[]),'CAPACITY_EXCEEDED')
    assert b._arrays is None and b._index is None

@pytest.mark.parametrize('seed',(17,31,71,127))
def test_seeded_mixed_workload_index_and_authority_invariants(seed):
    import random
    rng=random.Random(seed)
    b=make(cap=16,count=64,ops=3000,
           protected=(1,2,3),stages=8)
    for _ in range(800):
        rid=rng.randrange(1,65)
        if rng.random()<0.48:
            ins(b,rid,v=(float(rid%7),0.,1.,0.))
        elif rng.random()<0.75:
            b.evict(rid)
        else:
            s=b.propose_update(rid,(1.,0.,0.,0.),0.25)
            if isinstance(s,dict):
                if rng.random()<0.3:
                    b.evict(rid)
                b.commit_validated_update(s)
        occ=b.occupancy()
        assert occ['live']<=16
        live={int(b._arrays['rid'][s]) for s in range(16)
              if b._arrays['live'][s]}
        assert len(live)==occ['live']
        for s in range(16):
            if b._arrays['live'][s]:
                r=int(b._arrays['rid'][s]);assert b._slot_of(r)==s
        for r in (1,2,3):
            if r in live:
                assert b._is_protected(b._slot_of(r))
        assert 0 <= b._pending_used <= 8
        assert b._pending_used==len(b._stage_index)
