"""Independent oracle and adversarial native-C-authority tests (not R9 shadow)."""
import json, math, random, shutil
import pytest
from native.ersa_authority_r10 import NativeAuthority, NativeAuthorityHold, compile_authority
from integration.ssc_native_r10 import NativeSscAuthorityR10,R10Hold
from elpis_ssc import contract_r0 as SC
from test_r8_ssc_real_contract import context,proposal

@pytest.fixture(scope='module')
def lib():
    p=compile_authority();yield p;shutil.rmtree(p.parent)

def new(lib,cap=4,protected=(),policy='UTILITY_HEBBIAN'):
    spec,reg,tr,_,oracle=context(cap=cap,protected=protected)
    return NativeSscAuthorityR10(spec,reg,tr,lib,policy=policy),oracle

@pytest.mark.parametrize('seed',range(18))
def test_all_operations_native_state_vs_independent_oracle(lib,seed):
    b,o=new(lib)
    p1=proposal('R1',101,vec=(.2+.001*seed,-.1,.2,.5))
    p2=proposal('R2',102,vec=(.1,.3,0,-.4))
    trace=[('begin_session',()),('observe',((1+seed,2),)),
        ('upsert_exact',(101,(.1,0,0,0),'fixture:write:r0')),
        ('upsert_exact',(102,(.2,0,0,0),'fixture:write:r0')),
        ('read_exact',(101,)),('propose_relation',(p1,)),('apply_delta',(p1,)),
        ('propose_relation',(p2,)),('apply_delta',(p2,)),
        ('read_assoc',(101,)),('transport_and_aggregate',((p1,p2),)),
        ('reinforce',('R1',.2,-.1)),('decay',(2,)),('step',(2,)),
        ('evict',('R2',)),('reset',()),('halt',())]
    try:
        assert not SC.check_adapter(b)
        for name,args in trace:
            a=getattr(b,name)(*args)
            c=getattr(o,name)(*args)
            assert (a.status,a.code)==(c.status,c.code),(name,a,c)
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot()),(seed,name,b.snapshot(),o.snapshot())
            assert b.occupancy()==o.occupancy()
    finally:b.close()

@pytest.mark.parametrize('size',[128,1024,16384])
def test_native_image_bounds_and_cas(lib,size):
    with NativeAuthority(lib,size) as a:
        assert a.generation==0
        a.replace(b'{"a":0}',0)
        first,v=a.read();assert first==b'{"a":0}' and v==1
        with pytest.raises(NativeAuthorityHold,match='CAS_REJECT_1'):
            a.replace(b'{"a":1}',0)
        assert a.read()==(first,1)
        with pytest.raises(NativeAuthorityHold,match='CAS_REJECT_2'):
            a.replace(b'Z'*(size+1),1)
        assert a.read()==(first,1)
        with pytest.raises(NativeAuthorityHold,match='STALE_READ'):a.read(expected=0)

@pytest.mark.parametrize('bad',[-1,False,True,0,64,32*1024*1024+1,1.5])
def test_native_invalid_image_capacity(lib,bad):
    with pytest.raises(NativeAuthorityHold):NativeAuthority(lib,bad)

@pytest.mark.parametrize('bad',[-1,True,1.4,'1',None])
def test_native_invalid_generation(lib,bad):
    a=NativeAuthority(lib,1024)
    try:
        with pytest.raises(NativeAuthorityHold):a.replace(b'{}',bad)
    finally:a.close()

@pytest.mark.parametrize('method,arg',[('step',1025),('step',-1),('step',True),('decay',-3),('decay',256.5)])
def test_native_rejects_predecessor_clock_defects(lib,method,arg):
    b,o=new(lib)
    try:
        before=b.snapshot();v=b.generation
        assert getattr(b,method)(arg).code=='STEP_BUDGET'
        assert b.snapshot()==before and b.generation==v+1
    finally:b.close()

@pytest.mark.parametrize('method,args',[('upsert_exact',(101,(.1,0,0,0),'INVALID')),
        ('upsert_exact',(True,(0,0,0,0),'fixture:write:r0')),
        ('upsert_exact',(101,(math.nan,0,0,0),'fixture:write:r0')),
        ('read_exact',(9999999999,)),('evict',('NOT_A_RELATION',)),('read_assoc',(100,))])
def test_rejected_requests_commit_only_allowed_telemetry(lib,method,args):
    b,o=new(lib)
    try:
        before=b.snapshot()
        left=getattr(b,method)(*args);right=getattr(o,method)(*args)
        assert (left.status,left.code)==(right.status,right.code)
        assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot())
        assert b.generation>=2
    finally:b.close()

def test_proposal_identity_not_content_equivalent(lib):
    b,o=new(lib)
    p=proposal(); q=proposal()
    try:
        b.upsert_exact(101,(0,0,0,0),'fixture:write:r0')
        assert b.propose_relation(p).status=='PENDING'
        assert b.apply_delta(q).code=='STALE_PROPOSAL'
        assert b.apply_delta(p).status=='OK'
        assert b.apply_delta(p).code=='DUPLICATE_IDENTITY'
    finally:b.close()

@pytest.mark.parametrize('policy', ['FIFO','LRU','HEBBIAN_ONLY','UTILITY_HEBBIAN'])
def test_native_policy_and_bounded_eviction(lib,policy):
    b,o=new(lib,cap=2,policy=policy)
    o.policy_name=policy
    try:
        for i in (101,102,103):
            for obj in (b,o):assert obj.upsert_exact(i,(i/1000,0,0,0),'fixture:write:r0').status=='OK'
        for rid,part in [('R1',101),('R2',102),('R3',103)]:
            p=proposal(rid,part)
            for obj in (b,o):
                assert obj.propose_relation(p).status=='PENDING'
                assert obj.apply_delta(p).status=='OK'
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot())
        assert len(b.snapshot()['relations'])==2
    finally:b.close()

@pytest.mark.parametrize('seed',range(12))
def test_random_native_reference_differential(lib,seed):
    rng=random.Random(seed+1000)
    b,o=new(lib)
    try:
        for t in range(45):
            typ=rng.choice(['upsert','read','step','observe','assoc','reinforce','evict'])
            i=rng.choice([101,102,103,104])
            if typ=='upsert':method,args='upsert_exact',(i,(rng.choice([0.,.2,-.3]),0,0,0),'fixture:write:r0')
            elif typ=='read':method,args='read_exact',(i,)
            elif typ=='step':method,args='step',(rng.randint(0,3),)
            elif typ=='observe':method,args='observe',((rng.randint(0,200),),)
            elif typ=='assoc':method,args='read_assoc',(i,)
            elif typ=='reinforce':method,args='reinforce',(rng.choice(['R1','R2']),.2,-.1)
            else:method,args='evict',(rng.choice(['R1','R2']),)
            left=getattr(b,method)(*args);right=getattr(o,method)(*args)
            assert (left.status,left.code)==(right.status,right.code)
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot()),(seed,t,method)
    finally:b.close()

# Deliberate mutation-adequacy control: a wrong native state must be caught by oracle.
def test_native_oracle_detects_divergent_values(lib):
    b,o=new(lib)
    try:
        b.upsert_exact(101,(.5,0,0,0),'fixture:write:r0')
        assert SC.digest_state(b.snapshot())!=SC.digest_state(o.snapshot())
    finally:b.close()

@pytest.mark.parametrize('threads', [2,4,8])
def test_c_native_concurrent_same_generation_exactly_one_winner(lib,threads):
    from concurrent.futures import ThreadPoolExecutor
    with NativeAuthority(lib,4096) as a:
        a.replace(b'A',0)
        v=a.generation
        def attempt(i):
            try:a.replace(('new-%d'%i).encode(),v);return 'OK'
            except NativeAuthorityHold as e:return str(e)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            results=list(ex.map(attempt,range(threads)))
        assert results.count('OK')==1
        assert results.count('CAS_REJECT_1')==threads-1
        assert a.generation==v+1

@pytest.mark.parametrize('cap',[768,1024,1400])
def test_native_image_overflow_atomic_and_no_unbounded_growth(lib,cap):
    b,o=None,None
    try:
        spec,reg,tr,_,o=context()
        b=NativeSscAuthorityR10(spec,reg,tr,lib,max_bytes=cap)
        previous=b.snapshot();old_generation=b.generation
        failed=False
        for i in range(101,250):
            previous=b.snapshot();old_generation=b.generation
            try:b.upsert_exact(i,(.1,0,0,0),'fixture:write:r0')
            except NativeAuthorityHold as exc:
                assert 'CAS_REJECT_2' in str(exc)
                assert b.snapshot()==previous
                assert b.generation==old_generation
                failed=True;break
        assert failed
    finally:
        if b:b.close()

def test_transient_transport_exception_keeps_c_authority_intact(lib):
    b,o=new(lib)
    p=proposal()
    try:
        b.upsert_exact(101,(0,0,0,0),'fixture:write:r0')
        b.propose_relation(p)
        pre=b.snapshot();old=b.generation
        def broken(*args):raise RuntimeError('BROKEN_TRANSPORT')
        b.transport.to_canonical=broken
        with pytest.raises(RuntimeError,match='BROKEN_TRANSPORT'):
            b.apply_delta(p)
        assert b.snapshot()==pre and b.generation==old
    finally:b.close()

def test_proposal_liveness_growth_capped_and_reset_clears(lib):
    import integration.ssc_native_r10 as mod
    b,o=new(lib)
    old=mod.MAX_PROPOSAL_IDENTITIES
    mod.MAX_PROPOSAL_IDENTITIES=3
    try:
        b.upsert_exact(101,(0,0,0,0),'fixture:write:r0')
        pp=[]
        for i in range(3):
            p=proposal(vec=(i*.1,0,0,0));pp.append(p)
            assert b.propose_relation(p).status=='PENDING'
        assert len(b._references)==3
        q=proposal(vec=(.9,0,0,0))
        assert b.propose_relation(q).code=='CAPACITY_EXCEEDED'
        assert len(b._references)==3
        assert b.reset().status=='OK'
        assert not b._references
    finally:
        mod.MAX_PROPOSAL_IDENTITIES=old
        b.close()
