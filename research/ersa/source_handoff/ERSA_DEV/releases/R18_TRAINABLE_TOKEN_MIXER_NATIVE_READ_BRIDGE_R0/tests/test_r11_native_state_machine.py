"""ERSA R11: C-executed SSC operations vs the frozen independent oracle."""
import math,random,shutil,pytest
from native import ersa_state_r9
from integration.ssc_native_r11 import NativeSscAuthorityR11,compile_r11,R11Hold
from elpis_ssc import contract_r0 as SC
from test_r8_ssc_real_contract import context,proposal

@pytest.fixture(scope='module')
def lib():
    p=compile_r11()
    yield p
    shutil.rmtree(p.parent)

def new(lib,cap=4,protected=(),policy='UTILITY_HEBBIAN',proposal_cap=8192):
    spec,reg,tr,_,o=context(cap=cap,protected=protected)
    o.policy_name=policy
    return NativeSscAuthorityR11(spec,reg,tr,lib,policy=policy,proposal_cap=proposal_cap),o

@pytest.mark.parametrize('seed',range(18))
def test_full_14_ops_oracle(lib,seed):
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
            a=getattr(b,name)(*args);c=getattr(o,name)(*args)
            assert (a.status,a.code)==(c.status,c.code),(name,a,c)
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot()),(seed,name,b.snapshot(),o.snapshot())
            assert b.occupancy()==o.occupancy(),(name,b.occupancy(),o.occupancy())
    finally:b.close()

@pytest.mark.parametrize('seed',range(12))
def test_random_45step_oracle(lib,seed):
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
            a=getattr(b,method)(*args);c=getattr(o,method)(*args)
            assert (a.status,a.code)==(c.status,c.code),(seed,t,method,a,c)
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot()),(seed,t,method,b.snapshot(),o.snapshot())
    finally:b.close()

@pytest.mark.parametrize('policy',('FIFO','LRU','HEBBIAN_ONLY','UTILITY_HEBBIAN'))
def test_eviction_policy(lib,policy):
    b,o=new(lib,cap=2,policy=policy)
    try:
        for i in (101,102,103):
            for obj in (b,o):assert obj.upsert_exact(i,(i/1000,0,0,0),'fixture:write:r0').status=='OK'
        for rid,part in [('R1',101),('R2',102),('R3',103)]:
            p=proposal(rid,part)
            for obj in (b,o):
                assert obj.propose_relation(p).status=='PENDING'
                assert obj.apply_delta(p).status=='OK'
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot())
    finally:b.close()

@pytest.mark.parametrize('method,arg',[('step',1025),('step',-1),('step',True),('decay',-3),('decay',256.5)])
def test_clock_defects_rejected_no_state_change(lib,method,arg):
    b,o=new(lib)
    try:
        before=b.snapshot();g=b.generation
        assert getattr(b,method)(arg).code=='STEP_BUDGET'
        assert before==b.snapshot();assert b.generation==g+1
    finally:b.close()

def test_proposal_identity_and_reset(lib):
    b,o=new(lib)
    p=proposal();q=proposal()
    try:
        b.upsert_exact(101,(0,0,0,0),'fixture:write:r0')
        assert b.propose_relation(p).status=='PENDING'
        assert b.apply_delta(q).code=='STALE_PROPOSAL'
        assert b.apply_delta(p).status=='OK'
        assert b.apply_delta(p).code=='DUPLICATE_IDENTITY'
        assert b.reset().status=='OK'
        assert b.apply_delta(p).code=='STALE_PROPOSAL'
    finally:b.close()

def test_budget_and_stale_view_native(lib):
    b,o=new(lib)
    try:
        for j in range(70):b.read_exact(101)
        assert b.occupancy()['rejections']>0
        assert b.step(1).status=='OK' if b.occupancy()['rejections']==0 else b.step(1).code=='OP_BUDGET'
    finally:b.close()

@pytest.mark.parametrize('bad',[-1,True,0,8193,1.5])
def test_proposal_cap_failclosed(lib,bad):
    spec,reg,tr,_,oracle=context()
    with pytest.raises(R11Hold):NativeSscAuthorityR11(spec,reg,tr,lib,proposal_cap=bad)

@pytest.mark.parametrize('seed',range(12))
def test_random_relation_lifecycle_oracle(lib,seed):
    rng=random.Random(seed+700)
    b,o=new(lib,cap=3,protected=('R1',))
    pending=[]
    try:
        for j in range(90):
            kind=rng.choice(['upsert','propose','apply','aggregate','reinforce','evict','assoc','step','reset'])
            rid=rng.choice(('R1','R2','R3'))
            if kind=='upsert':name,args='upsert_exact',(rng.choice((101,102,103)),(rng.uniform(-1,1),0,0,0),'fixture:write:r0')
            elif kind=='propose':
                ix=int(rid[1])-1;p=proposal(rid,101+ix,vec=(rng.uniform(-.5,.5),.2,.3,.4))
                pending.append(p);name,args='propose_relation',(p,)
            elif kind=='apply':
                p=rng.choice(pending) if pending else proposal(rid,101+int(rid[1])-1)
                name,args='apply_delta',(p,)
            elif kind=='aggregate':
                if pending:
                    p=rng.choice(pending);name,args='transport_and_aggregate',((p,),)
                else:name,args='read_assoc',(101,)
            elif kind=='reinforce':name,args='reinforce',(rid,rng.uniform(-1,1),rng.uniform(-1,1))
            elif kind=='evict':name,args='evict',(rid,)
            elif kind=='assoc':name,args='read_assoc',(rng.choice((101,102,103)),)
            elif kind=='step':name,args='step',(rng.randint(0,8),)
            else:name,args='reset',();pending.clear()
            a=getattr(b,name)(*args);c=getattr(o,name)(*args)
            assert (a.status,a.code)==(c.status,c.code),(seed,j,name,a,c)
            assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot()),(seed,j,name,b.snapshot(),o.snapshot())
            assert b.occupancy()==o.occupancy()
    finally:b.close()

@pytest.mark.parametrize('method,args',[('upsert_exact',(101,(.1,0,0,0),'BAD')),
    ('upsert_exact',(101,(math.nan,0,0,0),'fixture:write:r0')),
    ('upsert_exact',(True,(0,0,0,0),'fixture:write:r0')),
    ('read_exact',(9999999999,)),('read_assoc',(9999,)),('evict',('BOGUS',))])
def test_invalid_input_receipt_and_state_parity(lib,method,args):
    b,o=new(lib)
    try:
        a=getattr(b,method)(*args);c=getattr(o,method)(*args)
        assert (a.status,a.code)==(c.status,c.code),(method,a,c)
        assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot())
    finally:b.close()

def test_aggregation_invalid_empty_no_false_stale(lib):
    b,o=new(lib)
    try:
        a=b.transport_and_aggregate(())
        c=o.transport_and_aggregate(())
        assert (a.status,a.code)==(c.status,c.code)
        assert SC.digest_state(b.snapshot())==SC.digest_state(o.snapshot())
    finally:b.close()

def test_native_large_bounded_capacity_and_evictions(lib):
    from dataclasses import replace
    spec,reg,tr,_,o=context()
    spec=replace(spec,max_active_entities=8192,max_active_relations=8192,
        max_ops_per_tick=10000,max_control_records=32)
    with NativeSscAuthorityR11(spec,reg,tr,lib) as b:
        for address in range(1,8193):
            assert b.upsert_exact(address,(0.125,0.0,0.0,0.0),'fixture:write:r0').status=='OK'
        occ=b.occupancy();assert occ['entities']==8192
        for address in range(8193,8257):
            assert b.upsert_exact(address,(0.,0.,0.,0.),'fixture:write:r0').status=='OK'
        occ=b.occupancy();assert occ['entities']==8192
        assert occ['evictions']==64
        assert b.read_exact(8256).status=='OK'

def test_concurrent_owner_lock_safety(lib):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    spec,reg,tr,_,o=context()
    spec=replace(spec,max_ops_per_tick=1000)
    with NativeSscAuthorityR11(spec,reg,tr,lib) as b:
        def run(i):return b.upsert_exact(101+(i%8),(float(i)/1000,0,0,0),'fixture:write:r0').status
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(run,range(300)))==['OK']*300
        assert b.occupancy()['entities']==8
        assert b.generation==300

def test_nonfinite_aggregate_output_rejected(lib):
    from dataclasses import replace
    spec,reg,tr,_,o=context()
    spec=replace(spec,finite_bound=1e308)
    with NativeSscAuthorityR11(spec,reg,tr,lib) as b:
        assert b.upsert_exact(101,(0,0,0,0),'fixture:write:r0').status=='OK'
        p=proposal(vec=(1e308,1e308,1e308,1e308))
        assert b.propose_relation(p).status=='PENDING'
        before=b.snapshot()
        out=b.transport_and_aggregate((p,))
        assert (out.status,out.code)==('REJECTED','NONFINITE_VALUE')
        assert b.snapshot()['relations']==before['relations']
        assert b.occupancy()['relations']==0
