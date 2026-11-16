"""R7 true host-integrated read-path differential and fault tests.

The R5 host remains sole source of admitted identities; native code never
accepts an attacker-supplied selection as a permission to read a relation.
"""
import math
import shutil
import numpy as np
import pytest
from native.ersa_native import NativeOps,NativeError,build_native
from native.ersa_read_r7 import NativeReadPath
from elpis_ersa import bank_ersa as B, fixtures_r2 as F, harness_ersa as H, contract_ersa as C

@pytest.fixture(scope='module')
def native():
    lib=build_native()
    try:yield NativeOps(lib)
    finally:shutil.rmtree(lib.parent)

class Complete(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        auth=dict(authority or {})
        auth['modes']={'discovery':'score_complete','selection':'ersa'}
        return super().begin_session(spec,registry,transport,auth)

class Indexed(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        auth=dict(authority or {})
        auth['modes']={'discovery':'score_indexed','selection':'ersa'}
        return super().begin_session(spec,registry,transport,auth)

@pytest.mark.parametrize('family', F.FAMILIES)
@pytest.mark.parametrize('cls',[Complete,Indexed])
def test_native_integrated_selection_reduction_vs_reference(native,family,cls):
    fixture=F.build(family,cand=8)
    b=H.replay(cls,fixture)
    reference=H.replay(cls,fixture)
    path=NativeReadPath(native)
    for q in fixture['queries']:
        if q.get('note')=='':pass
        r=path.read(b,q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),k=q['k'])
        got=reference.aggregate(q['qkey'],q.get('qgoal'),k=q['k'],qdir=q.get('qdir'))
        assert r.status == got.status == C.STATUS_OK,(family,r,got)
        assert r.relation_ids == tuple(int(z[0]) for z in reference._last_selection),(family,r.relation_ids,reference._last_selection)
        assert np.allclose(r.salient,reference._last_aggregate,atol=3e-12,rtol=0),(family,r.salient,reference._last_aggregate)
        assert np.allclose(r.collective,reference._last_collective,atol=3e-12,rtol=0)

@pytest.mark.parametrize('family', F.FAMILIES)
def test_native_features_against_reference(native,family):
    f=F.build(family,cand=8)
    b=H.replay(Complete,f)
    for q in f['queries']:
        out=b.query_candidates(q['qkey'],q.get('qgoal'),qdir=q.get('qdir'))
        assert out.status==C.STATUS_OK
        slots=list(b._last_slots)
        if not slots:continue
        a=b._arrays
        vecs=[b._transported_value(s) for s in slots]
        assert all(v is not None for v in vecs)
        meta=[[float(a['util'][s]),float(a['cred'][s]),
            -1 if (b._slot_of(int(a['contra'][s])) is not None and
                   b._admitted_rid(int(a['contra'][s]))) else 0,
             int(b._is_protected(s))] for s in slots]
        qdir=q.get('qdir') or q['qkey']
        features=native.features(a['key'][slots].astype(float),a['goal'][slots],
                vecs,meta,a['issued'][slots],b._tick,b._spec.halflife_ticks,
                q['qkey'],q.get('qgoal'),qdir)
        for s,row in zip(slots,features):
            expected=b._features(s,q['qkey'],q.get('qgoal'),qdir)
            assert np.allclose(row,[expected[k] for k in B.RELEVANCE_WEIGHTS],atol=3e-14,rtol=0), (family,s,row,expected)

@pytest.mark.parametrize('n',[0,1,8,64,512,8192])
def test_feature_envelope_and_liveness(native,n):
    f=np.zeros((n,4));v=np.zeros((n,4));meta=np.zeros((n,4));i=np.zeros(n,np.int64)
    x=native.features(f,f,v,meta,i,0,64.,(0,0,0,0),None,(1,0,0,0))
    assert x.shape==(n,8)
    if n:assert np.array_equal(x[:,2],np.ones(n))

@pytest.mark.parametrize('malformed',[
    dict(query_key=(1,2,3)),dict(query_direction=(1,2,3)),
    dict(query_direction=(1,2,float('nan'),4)),
    dict(goals=[[1,2,3]]),dict(metadata=[[0,0,1,0]]),
    dict(metadata=[[0,0,0,2]]),dict(tick=-1),dict(issued=[-1]),
    dict(issued=[3]),dict(halflife=0),dict(halflife=float('inf')),
])
def test_native_features_fail_closed(native,malformed):
    d=dict(keys=[[1,2,3,4]],goals=[[1,2,3,4]],vectors=[[1,0,0,0]],
           metadata=[[0,0,0,0]],issued=[0],tick=0,halflife=64.,
           query_key=(1,2,3,4),query_goal=(1,2,3,4),query_direction=(1,0,0,0))
    d.update(malformed)
    with pytest.raises((NativeError,ValueError,TypeError)):
        native.features(**d)

def test_mid_read_revision_invalidation(native,monkeypatch):
    f=F.build('rare_decisive',cand=8)
    b=H.replay(Complete,f)
    old=native.rank
    def invalidating(*args,**kwargs):
        s=b._last_slots[0]
        b._arrays['revision'][s] += 1
        return old(*args,**kwargs)
    monkeypatch.setattr(native,'rank',invalidating)
    q=f['queries'][0]
    got=NativeReadPath(native).read(b,q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),k=q['k'])
    assert got.status=='REJECTED' and got.code=='MID_READ_REVISION_CHANGE'
    assert not got.evidence and not got.relation_ids

def test_mid_read_authority_invalidation(native,monkeypatch):
    f=F.build('rare_decisive',cand=8)
    b=H.replay(Complete,f)
    old=native.rank
    def invalidating(*args,**kwargs):
        b._authority=dict(b._authority,denied=frozenset({99999}))
        return old(*args,**kwargs)
    monkeypatch.setattr(native,'rank',invalidating)
    q=f['queries'][0]
    got=NativeReadPath(native).read(b,q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),k=q['k'])
    assert got.status=='REJECTED' and got.code=='MID_READ_AUTHORITY_CHANGE'

def test_bad_query_and_selection_mode_rejected(native):
    f=F.build('rare_decisive')
    b=H.replay(Complete,f)
    r=NativeReadPath(native)
    q=f['queries'][0]
    assert r.read(b,q['qkey'],k=0).code=='INVALID_K'
    assert r.read(b,q['qkey'][:-1]).code=='INVALID_QUERY'
    b._modes['selection']='fifo'
    assert r.read(b,q['qkey']).code=='UNSUPPORTED_SELECTION_MODE'

@pytest.mark.parametrize('case',['missing','bool','float'])
def test_exact_identity_fail_closed(native,case):
    f=F.build('near_identical_keys_distinct_identity')
    b=H.replay(Complete,f)
    q=f['queries'][0]
    rid={'missing':10**9,'bool':True,'float':2.0}[case]
    out=NativeReadPath(native).read(b,q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),
                                  exact_relation_id=rid,k=2)
    assert out.status=='REJECTED'


def test_signed_reduction_fails_magnitude_only_mutation(native):
    v=[[1.,0,0,0],[0.,1.,0,0]]
    good,_=native.aggregate(v,[1.,-1.])
    bad,_=native.aggregate(v,[1.,1.])
    assert np.max(np.abs(np.asarray(good)-bad))>.9

@pytest.mark.parametrize('rids,eligible',[
    ([1],[257]),([1],[-255]),([1],[-256]),([1],[True]),
    ([2**64+1],[1]),([-2**64+1],[1]),([True],[1]),
    ([1.0],[1]),(['1'],[1]),
])
def test_no_integer_wrap_or_bool_alias(native,rids,eligible):
    with pytest.raises(NativeError):
        native.rank([1.],rids,eligible,1)

@pytest.mark.parametrize('family',F.FAMILIES)
def test_native_full_discovery_does_not_alias_other_identity(native,family):
    f=F.build(family,cand=8)
    class Dense(B.ErsBank):
        def begin_session(self,spec=None,registry=None,transport=None,authority=None):
            auth=dict(authority or {})
            auth['modes']={'discovery':'dense','selection':'ersa'}
            return super().begin_session(spec,registry,transport,auth)
    b=H.replay(Dense,f)
    r=H.replay(Complete,f)
    for q in f['queries']:
        got=NativeReadPath(native).read(b,q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),k=q['k'])
        reference=r.aggregate(q['qkey'],q.get('qgoal'),qdir=q.get('qdir'),k=q['k'])
        assert got.status==reference.status==C.STATUS_OK
        assert got.relation_ids==tuple(int(z[0]) for z in r._last_selection)
        assert np.allclose(got.salient,r._last_aggregate,atol=3e-12,rtol=0)
