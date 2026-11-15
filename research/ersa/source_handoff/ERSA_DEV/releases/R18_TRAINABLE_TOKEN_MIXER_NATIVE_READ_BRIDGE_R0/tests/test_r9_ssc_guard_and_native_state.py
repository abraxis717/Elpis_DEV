"""R9 direct SSC state-operation and bounded native-shadow differential/BREAK."""
from __future__ import annotations
import random
import shutil
import math
import pytest
from dataclasses import replace
from native.ersa_state_r9 import NativeState,StateHold,compile_state
from integration.ssc_r9 import GuardedSscReference,GuardHold
from elpis_ssc import contract_r0 as SC
from elpis_ssc import oracle_r0 as SO
from test_r8_ssc_real_contract import context,proposal

@pytest.fixture(scope='module')
def shared_lib():
    lib=compile_state()
    yield lib
    shutil.rmtree(lib.parent)

@pytest.mark.parametrize('capacity', [1,2,13,128,8192])
def test_native_fixed_capacity_creation(shared_lib,capacity):
    with NativeState(shared_lib,capacity,0,999999,1000.) as s:
        assert s.count==0 and s.cap==capacity and s.generation==0
        assert s.peek(0) is None

@pytest.mark.parametrize('bad', [0,-1,8193,True,1.0])
def test_native_capacity_fails_closed(shared_lib,bad):
    with pytest.raises(StateHold):NativeState(shared_lib,bad,0,10,1.)

@pytest.mark.parametrize('value',[(math.inf,0,0,0),(math.nan,0,0,0),(2,0,0,0),
                                   (True,0,0,0),[0,0,0],('0',0,0,0)])
def test_native_rejects_invalid_values_atomically(shared_lib,value):
    with NativeState(shared_lib,2,0,10,1.) as s:
        s.upsert(1,(.2,0,0,0),0)
        pre=(s.generation,s.count,s.peek(1))
        with pytest.raises(StateHold):s.upsert(1,value,s.generation)
        assert (s.generation,s.count,s.peek(1))==pre

@pytest.mark.parametrize('bad', [True,1.0,1<<70,None])
def test_native_rejects_alias_addresses(shared_lib,bad):
    with NativeState(shared_lib,2,0,10,1.) as s:
        with pytest.raises(StateHold):s.upsert(bad,(0,0,0,0),0)
        assert s.count==s.generation==0

@pytest.mark.parametrize('bad', [-1,3,99999999])
def test_native_rejects_outside_namespace(shared_lib,bad):
    with NativeState(shared_lib,2,0,2,1.) as s:
        with pytest.raises(StateHold):s.upsert(bad,(0,0,0,0),0)
        assert s.count==0 and s.generation==0

@pytest.mark.parametrize('bad', [-1,True,1.0,2**64,None])
def test_native_invalid_expected_generation(shared_lib,bad):
    with NativeState(shared_lib,2,0,10,1.) as s:
        with pytest.raises(StateHold):s.upsert(1,(0,0,0,0),bad)

@pytest.mark.parametrize('capacity', [1,2,8])
def test_native_full_capacity_no_mutation(shared_lib,capacity):
    with NativeState(shared_lib,capacity,1,100,1.) as s:
        for a in range(1,capacity+1):s.upsert(a,(a/100,0,0,0),s.generation)
        version=s.generation
        with pytest.raises(StateHold,match='NATIVE_UPSERT_RC:2'):
            s.upsert(capacity+1,(0,0,0,0),version)
        assert s.generation==version and s.count==capacity
        for a in range(1,capacity+1):assert s.peek(a)[0]==a/100

@pytest.mark.parametrize('cap', [2,13,64])
def test_native_random_reference_dictionary(shared_lib,cap):
    rng=random.Random(cap)
    with NativeState(shared_lib,cap,1,200,5.) as s:
        oracle={}
        for i in range(250):
            a=rng.randint(1,200)
            if rng.random()<.4:
                if a not in oracle:
                    with pytest.raises(StateHold,match='NATIVE_DELETE_RC:6'):
                        s.delete(a,s.generation)
                else:
                    s.delete(a,s.generation);del oracle[a]
            else:
                val=(float(i%7)/7,0.,-float(i%11)/11,float(i%3))
                if a not in oracle and len(oracle)==cap:
                    with pytest.raises(StateHold,match='NATIVE_UPSERT_RC:2'):
                        s.upsert(a,val,s.generation)
                else:
                    s.upsert(a,val,s.generation);oracle[a]=val
            assert s.count==len(oracle)
            assert s.peek(a)==oracle.get(a)


def test_native_8192_admissions_and_bounded_replacement(shared_lib):
    with NativeState(shared_lib,8192,1,100000,2.) as s:
        for a in range(1,8193):s.upsert(a,(a/8192,0,0,0),s.generation)
        assert s.count==8192
        with pytest.raises(StateHold,match='NATIVE_UPSERT_RC:2'):
            s.upsert(8193,(0,0,0,0),s.generation)
        for a in range(1,65):
            s.delete(a,s.generation)
            s.upsert(a+8192,(1.,-.5,0.,0.),s.generation)
        assert s.count==8192
        assert s.peek(1) is None and s.peek(8256)==(1.,-.5,0.,0.)


def test_native_aba_stale_generation(shared_lib):
    with NativeState(shared_lib,2,1,10,1.) as s:
        s.upsert(2,(.5,0,0,0),0)
        old=s.generation
        s.delete(2,old)
        s.upsert(2,(.5,0,0,0),s.generation)
        with pytest.raises(StateHold,match='NATIVE_UPSERT_RC:1'):
            s.upsert(2,(.9,0,0,0),old)
        assert s.peek(2)==(.5,0,0,0)


def test_native_rebuild_failure_keeps_original(shared_lib):
    with NativeState(shared_lib,2,1,10,1.) as s:
        s.upsert(1,(.1,0,0,0),0)
        pre=(s.count,s.generation,s.peek(1))
        with pytest.raises(StateHold):
            NativeState.replace_from_rows(s,[(2,(0,0,0,0),0,0,False),
                                               (2,(0,0,0,0),0,0,False)])
        assert (s.count,s.generation,s.peek(1))==pre

@pytest.mark.parametrize('bad', [False,-1,1.0,1025,None,'1'])
def test_guard_step_rejects_overcap_and_invalid(shared_lib,bad):
    spec,reg,tr,b,o=context()
    with_guard=GuardedSscReference(b,shared_lib)
    try:
        snapshot=with_guard.snapshot();generation=with_guard.generation
        r=with_guard.step(bad)
        assert r.status=='REJECTED' and r.code=='STEP_BUDGET'
        assert with_guard.snapshot()==snapshot
        assert with_guard.generation==generation+1
    finally:with_guard.close()

@pytest.mark.parametrize('bad', [-3,True,2.0,None,'2',1025])
def test_guard_decay_rejects_invalid_and_clock_reversal(shared_lib,bad):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        before=g.snapshot()
        r=g.decay(bad)
        assert r.status=='REJECTED' and r.code=='STEP_BUDGET'
        assert g.snapshot()==before
    finally:g.close()


def test_guard_accepts_max_legal_step_and_decay(shared_lib):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        assert g.step(256).status=='OK'
        assert g.decay(256).status=='OK'
        assert g.snapshot()['tick']==512
    finally:g.close()

@pytest.mark.parametrize('seed',range(12))
def test_guard_full_14op_valid_trace_vs_real_oracle(shared_lib,seed):
    spec,reg,tr,b,oracle=context()
    g=GuardedSscReference(b,shared_lib)
    p1=proposal('R1',101,vec=(.2+seed*.001,-.1,.2,.5))
    p2=proposal('R2',102,vec=(.1,.3,0,-.4))
    events=[('begin_session',()),('observe',((1+seed,2),)),
       ('upsert_exact',(101,(.1,0,0,0),'fixture:write:r0')),
       ('upsert_exact',(102,(.2,0,0,0),'fixture:write:r0')),
       ('read_exact',(101,)), ('propose_relation',(p1,)),('apply_delta',(p1,)),
       ('propose_relation',(p2,)),('apply_delta',(p2,)),
       ('read_assoc',(101,)),('transport_and_aggregate',((p1,p2),)),
       ('reinforce',('R1',.2,-.1)),('decay',(2,)),('step',(2,)),
       ('evict',('R2',)),('reset',()),('halt',())]
    try:
        for method,args in events:
            left=getattr(g,method)(*args)
            right=getattr(oracle,method)(*args)
            assert (left.status,left.code)==(right.status,right.code),(seed,method,left,right)
            assert SC.digest_state(g.snapshot())==SC.digest_state(oracle.snapshot()),(seed,method)
            assert g.occupancy()==oracle.occupancy()
            # An operation can influence state, but never grants native write rights.
            for address,val,*_ in g.snapshot()['entities']:
                assert g.peek_exact_native(address)==tuple(val)
    finally:g.close()


def test_guard_projection_is_invalidated_after_rejected_step(shared_lib):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        proj=g.relation_projection()
        assert g.step(1025).status=='REJECTED'
        with pytest.raises(Exception,match='SSC_VIEW_STALE'):
            proj._check_current()
    finally:g.close()


def test_guard_rejects_rebind_without_changing_contract(shared_lib):
    spec,reg,tr,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        old=g.snapshot()
        with pytest.raises(GuardHold,match='SSC_SESSION_REBIND'):
            g.begin_session(spec=spec)
        assert g.snapshot()==old
    finally:g.close()


def test_guard_external_alias_detected(shared_lib):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        assert b.upsert_exact(101,(.1,0,0,0),'fixture:write:r0').status=='OK'
        with pytest.raises(GuardHold,match='UNMONITORED_MUTATION'):
            g.peek_exact_native(101)
        with pytest.raises(GuardHold,match='GUARD_HELD'):
            g.reset()
    finally:g.close()


def test_guard_negative_time_refuses_predecessor_bug(shared_lib):
    _,_,_,b,o=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        old=g.snapshot()['tick']
        assert g.decay(-2).code=='STEP_BUDGET'
        assert g.snapshot()['tick']==old
        # Original SSC oracle is known to accept this invalid advance.
        assert o.decay(-2).status=='OK' and o.snapshot()['tick']==old-2
    finally:g.close()


def test_guard_overcap_refuses_predecessor_bug(shared_lib):
    _,_,_,b,o=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        assert g.step(1025).code=='STEP_BUDGET'
        assert g.snapshot()['tick']==0
        assert o.step(1025).code=='STEP_BUDGET'
        assert b.step(1025).status=='OK'  # frozen SSC reference defective
        with pytest.raises(GuardHold,match='UNMONITORED_MUTATION'):
            g.snapshot()
    finally:g.close()


def test_guard_native_shadow_is_not_authority(shared_lib):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        assert g.upsert_exact(101,(.5,.3,0,0),'fixture:write:r0').status=='OK'
        before=g.snapshot()
        assert g.peek_exact_native(101)==(.5,.3,0,0)
        assert g.snapshot()==before
        assert g.read_exact(101).status=='OK'
        assert g.snapshot()['entities'][0][3]==g.snapshot()['tick']
    finally:g.close()


def test_guard_explicit_identity_not_implicit_semantic_selection(shared_lib):
    _,_,_,b,_=context()
    g=GuardedSscReference(b,shared_lib)
    try:
        assert g.upsert_exact(101,(0.1,0,0,0),'fixture:write:r0').status=='OK'
        assert g.peek_exact_native(102) is None
        assert g.read_assoc(101).status=='NOT_FOUND'
    finally:g.close()
