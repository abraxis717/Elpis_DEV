"""R6 native stateless kernel adversarial and differential tests.

Intentional boundary: this kernel is a downstream deterministic operator, never
an authority source, autonomous semantic selector, or SSC predecessor adapter.
"""
import math
import random
import shutil
import numpy as np
import pytest
from ersa_native import NativeOps,NativeError,build_native
from elpis_ersa import bank_ersa as B, fixtures_r2 as F, harness_ersa as H, geometry_ersa as G

@pytest.fixture(scope='module')
def n():
    lib=build_native()
    try: yield NativeOps(lib)
    finally: shutil.rmtree(lib.parent)

class Complete(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        auth=dict(authority or {})
        auth['modes']={'discovery':'score_complete','selection':'ersa'}
        return super().begin_session(spec,registry,transport,auth)

ORDER=tuple(B.RELEVANCE_WEIGHTS)

def all_valid_scores(b,q):
    a=b._arrays;slots=[];features=[];values=[];weights=[]
    for s in range(b._spec.max_active_relations):
        if not a['live'][s]:continue
        rid=int(a['rid'][s])
        if not b._admitted_rid(rid) or rid in b._authority.get('denied',()) or int(a['prov'][s]) in b._authority['denied']:
            continue
        tv=b._transported_value(s)
        if tv is None: continue
        fs=b._features(s,q['qkey'],q['qgoal'],q['qdir'],tv)
        slots.append(s)
        features.append([fs[x] for x in ORDER])
        values.append(tv)
        weights.append(b._score(fs))
    return slots,np.asarray(features,dtype=np.float64).reshape((-1,8)),values,weights

@pytest.mark.parametrize('family',F.FAMILIES)
def test_fixture_native_score_rank_identity_parity(n,family):
    sc=F.build(family,cand=8)
    b=H.replay(Complete,sc)
    for q in sc['queries']:
        slots,x,vs,ws=all_valid_scores(b,q)
        cs=n.score(x)
        ref=[b._score(dict(zip(ORDER,row))) for row in x]
        assert len(cs)==len(ref)
        assert all(abs(float(a)-float(z))<1e-12 for a,z in zip(cs,ref))
        # Rank using R5 canonical rounding; float64 fsum and C long-double
        # scoring are not claimed to have bitwise-identical rounding.
        canonical=[round(float(v),12) for v in ref]
        ids=[int(b._arrays['rid'][s]) for s in slots]
        rr=n.rank(canonical,ids,[1]*len(ids),8)
        native_ids=[ids[i] for i in rr]
        out=b.select(q['qkey'],q['qgoal'],k=8,qdir=q['qdir'])
        assert out.status=='OK', (family,out)
        r5_ids=[rid for rid,_fs in b._last_selection]
        assert native_ids==r5_ids, (family,native_ids,r5_ids)

@pytest.mark.parametrize('seed',[1,2,4,8,16,32,64,128,256,512,1024,2048])
def test_random_signed_reduction_and_scores(n,seed):
    rng=np.random.default_rng(seed)
    num=min((seed%19)*23,8192)
    x=rng.normal(size=(num,8))
    score=n.score(x)
    expected=[math.fsum(float(row[j])*B.RELEVANCE_WEIGHTS[k] for j,k in enumerate(ORDER)) for row in x]
    assert np.allclose(score,expected,rtol=0,atol=5e-14)
    vs=rng.normal(size=(num,4)); ws=rng.normal(size=num)
    s,c=n.aggregate(vs,ws)
    sr,cr=G.aggregate_signed([tuple(x) for x in vs],list(ws))
    assert np.allclose(s,sr,rtol=0,atol=1e-12)
    assert np.allclose(c,cr,rtol=0,atol=1e-12)

@pytest.mark.parametrize('size',[0,1,2,8,64,1024,8192])
def test_capacity_and_identity_ties(n,size):
    ids=np.arange(size,dtype=np.int64)[::-1]
    scores=np.zeros(size)
    keep=np.ones(size,dtype=np.uint8)
    top=n.rank(scores,ids,keep,8)
    assert [int(ids[z]) for z in top]==list(range(min(size,8)))
    x=np.zeros((size,8))
    assert len(n.score(x))==size
    assert n.aggregate(np.zeros((size,4)),np.zeros(size)) == ((0.,)*4,(0.,)*4)

@pytest.mark.parametrize('cut',[1,2,3,5,8,16])
def test_topk_and_exclusion(n,cut):
    scores=[1.0,1.0,3.0,5.0,4.0,0.0,7.0,1.0]
    ids=[10,9,4,3,8,1,12,2]
    eligible=[1,1,0,1,1,1,0,1]
    got=n.rank(scores,ids,eligible,cut)
    expected=sorted((i for i in range(8) if eligible[i]),key=lambda i:(-scores[i],ids[i]))[:cut]
    assert got==expected

@pytest.mark.parametrize('invalid',[
    ('rank',[0.,float('nan')],[1,2],[1,1],1),
    ('rank',[1.,2.],[1,1],[1,1],1),
    ('rank',[1.,2.],[1,2],[1,2],1),
    ('rank',[1.,2.],[1,2],[1,1],0),
    ('rank',[1.,2.],[1,2],[1,1],8193),
    ('score',[[float('nan')]*8],None,None,None),
    ('aggregate',[[1.,2.,3.,4.]],[float('inf')],None,None),
    ('aggregate',[[1.,2.,3.,float('nan')]],[1.0],None,None),
])
def test_fail_closed_inputs(n,invalid):
    op,a,b,c,d=invalid
    with pytest.raises(NativeError):
        if op=='rank': n.rank(a,b,c,d)
        elif op=='score': n.score(a)
        else: n.aggregate(a,b)


def test_inactive_nan_is_never_returned(n):
    assert n.rank([1.,float('nan'),2.],[5,6,7],[1,0,1],2)==[2,0]


def test_signed_reduction_not_absolute_weighted(n):
    vec=[[1,0,0,0],[1,0,0,0]]
    got,_=n.aggregate(vec,[1.,-3.])
    assert got[0]==pytest.approx(-.5,abs=1e-14)
    assert got[0]!=pytest.approx(1.)


def test_zero_weight_fallback_mean(n):
    v=[[1.,3.,5.,7.],[3.,5.,7.,9.]]
    a,c=n.aggregate(v,[0.,0.])
    assert a == c == (2.,4.,6.,8.)


def test_gauge_covariance_native_reducer(n):
    g=(.37,-.81)
    v=[[1.2,-.4,.6,.8],[-.1,1.4,.7,-.2],[.8,.1,-.3,.2]]
    w=[-1.,.7,2.]
    before,_=n.aggregate(v,w)
    rot=[G.rho_apply(g,x) for x in v]
    after,_=n.aggregate(rot,w)
    assert np.allclose(after,G.rho_apply(g,before),rtol=0,atol=5e-15)


def test_wrong_mutant_is_detected_by_oracle(n):
    v=[[1.,0.,0.,0.],[0.,1.,0.,0.]]
    w=[1.,-1.]
    good=n.aggregate(v,w)[0]
    bad=G.aggregate_magnitude_only(v,w)
    assert max(abs(a-b) for a,b in zip(good,bad))>.5
