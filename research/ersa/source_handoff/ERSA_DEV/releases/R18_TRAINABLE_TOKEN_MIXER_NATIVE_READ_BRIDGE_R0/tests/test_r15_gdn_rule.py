import numpy as np
import pytest
from integration import gdn_rule_r15 as G
from integration.ersa_controller_r13 import make_chain

@pytest.mark.parametrize('seed',[3,7,13,29,101,202,1033,5017,9001,22004])
@pytest.mark.parametrize('n',[1,2,8,32])
def test_reference_torch_recurrence(seed,n):
    r=np.random.default_rng(seed+n)
    k=r.standard_normal((n,8));k/=np.linalg.norm(k,axis=1,keepdims=True)
    v=r.standard_normal((n,4));b=r.uniform(0.05,1.,n);g=r.uniform(-1.,0.,n)
    s0=r.standard_normal((8,4))
    assert np.allclose(G.recurrence_numpy(k,v,b,g,s0),G.recurrence_torch(k,v,b,g,s0),rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('seed',[5,11,31,52,89,144])
def test_episode_recurrence(seed):
    case=make_chain(seed)
    for beta in (0.5,0.95,1.0):
        for decay in (0.9,1.0):
            a=G.state_for_episode(case.episode,beta=beta,decay=decay)
            b=G.state_for_episode(case.episode,beta=beta,decay=decay,method='torch')
            assert np.allclose(a,b,rtol=1e-12,atol=1e-12)
            assert G.read(a,case.episode.query_key,case.episode.query_goal).shape==(4,)

def test_closed_form_single_update():
    k=np.zeros((1,8));k[0,0]=1
    v=np.array([[1.,2.,3.,4.]])
    out=G.recurrence_numpy(k,v,np.ones(1),np.zeros(1))
    assert np.allclose(out[0],v[0]);assert np.allclose(out[1:],0)

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_rejection(bad):
    k=np.zeros((2,8));v=np.ones((2,4));k[0,0]=bad
    with pytest.raises(ValueError):G.recurrence_numpy(k,v,np.ones(2),np.zeros(2))
    with pytest.raises(ValueError):G.recurrence_torch(k,v,np.ones(2),np.zeros(2))

@pytest.mark.parametrize('bad',[(-.1,1.),(1.1,1.),(.5,0.),(.5,1.1)])
def test_bad_hyperparams(bad):
    with pytest.raises(ValueError):G.state_for_episode(make_chain(101).episode,beta=bad[0],decay=bad[1])

def test_admission_mask_excludes_values():
    ep=make_chain(111).episode
    m=np.ones(len(ep.keys),bool);m[0]=False
    a=G.state_for_episode(ep,mask=m)
    ep.values[0]=np.array([42.,1.,3.,4.])
    b=G.state_for_episode(ep,mask=m)
    assert np.array_equal(a,b)

def test_duplicate_or_malformed_mask_reject():
    ep=make_chain(111).episode
    m=np.ones(len(ep.keys),dtype=np.int32);m[0]=2
    with pytest.raises(ValueError):G.state_for_episode(ep,mask=m)

def test_zero_capacity_reject():
    with pytest.raises(ValueError):G.recurrence_numpy(np.zeros((0,8)),np.zeros((0,4)),np.zeros(0),np.zeros(0))

@pytest.mark.parametrize('i',list(range(8)))
def test_prefix_reset_separable(i):
    r=np.random.default_rng(900+i)
    k=r.standard_normal((16,8));k/=np.linalg.norm(k,axis=1,keepdims=True)
    v=r.standard_normal((16,4));b=np.full(16,.7);g=np.full(16,-.03)
    mid=G.recurrence_numpy(k[:7],v[:7],b[:7],g[:7])
    end=G.recurrence_numpy(k[7:],v[7:],b[7:],g[7:],initial=mid)
    whole=G.recurrence_numpy(k,v,b,g)
    assert np.allclose(end,whole,rtol=1e-12,atol=1e-12)

def test_missing_goal_must_not_claim_success():
    from tools.r15_science import gdn_episode
    case=make_chain(717, family='missing_goal')
    pred=gdn_episode(case,.5,1.)
    assert pred['status']!='CONTINUOUS_OK'

def test_no_terminal_must_not_claim_success():
    from tools.r15_science import gdn_episode
    case=make_chain(719, family='no_terminal')
    pred=gdn_episode(case,.5,1.)
    assert pred['status']!='CONTINUOUS_OK'
