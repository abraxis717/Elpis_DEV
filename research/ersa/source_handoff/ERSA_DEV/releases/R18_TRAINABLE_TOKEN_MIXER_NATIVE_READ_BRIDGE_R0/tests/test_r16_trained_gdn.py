"""Independent numerical, negative, frozen-state, and training controls for R16.
No GDN parity claim; no oracle paths at inference.
"""
import numpy as np
import pytest
import torch
from integration import gdn_trainable_r16 as M

@pytest.mark.parametrize('seed',range(12))
def test_published_rule_differential(seed):
    rng=np.random.default_rng(seed+4701)
    m=M.new_model(seed+8)
    data=M.make_data(seed*91+718,1)
    xx,vv,qq,yy,mask=M.batchify(data)
    with torch.no_grad():out,state=m(xx,vv,qq)
    k,v,b,g=M.projected_inputs(m,(xx,vv))
    ref=M.independent_numpy_rule(k[0],v[0],b[0],g[0])
    assert np.allclose(state[0].numpy(),ref,rtol=1e-5,atol=2e-6)

@pytest.mark.parametrize('seed',range(8))
def test_gradients_enter_q_k_v_beta_forget(seed):
    m=M.new_model(seed+99)
    xx,vv,qq,yy,mask=M.batchify(M.make_data(seed*818+411,2))
    pred,_=m(xx,vv,qq)
    loss=M.loss_fn(pred,yy,mask)
    loss.backward()
    for f in (m.q,m.k,m.v,m.beta,m.forget,m.out):
        for p in f.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.k.weight.grad.abs().sum()>0
    assert m.q.weight.grad.abs().sum()>0
    assert m.v.weight.grad.abs().sum()>0
    assert m.beta.weight.grad.abs().sum()>0
    assert m.forget.weight.grad.abs().sum()>0

@pytest.mark.parametrize('seed',range(6))
def test_denied_admission_is_state_invariant(seed):
    m=M.new_model(seed+400)
    xx,vv,qq,yy,mask=M.batchify(M.make_data(seed*138+902,1))
    allow=torch.ones((1,M.SLOTS),dtype=torch.bool)
    ix=(seed*13)%M.SLOTS;allow[0,ix]=False
    v2=vv.clone();x2=xx.clone()
    v2[0,ix]=500; x2[0,ix]=200
    with torch.no_grad():
        a,s1=m(xx,vv,qq,allowed=allow)
        b,s2=m(x2,v2,qq,allowed=allow)
    assert torch.equal(a,b) and torch.equal(s1,s2)

@pytest.mark.parametrize('kind',['x_nan','v_inf','q_nan','bad_keys','bad_values','bad_queries','wrong_dtype','wrong_mask','bad_initial'])
def test_malformed_inputs_fail_closed(kind):
    m=M.new_model(1101)
    xx,vv,qq,yy,mask=M.batchify(M.make_data(20122,1))
    kwargs={};xx=xx.clone();vv=vv.clone();qq=qq.clone()
    if kind=='x_nan':xx[0,0,0]=float('nan')
    if kind=='v_inf':vv[0,0,0]=float('inf')
    if kind=='q_nan':qq[0,0,0]=float('nan')
    if kind=='bad_keys':xx=xx[:,:-1]
    if kind=='bad_values':vv=vv[:,:-1]
    if kind=='bad_queries':qq=qq[:,:,:-1]
    if kind=='wrong_dtype':xx=xx.double()
    if kind=='wrong_mask':kwargs['allowed']=torch.ones((1,1),dtype=torch.bool)
    if kind=='bad_initial':kwargs['initial']=torch.zeros((1,3,4))
    with pytest.raises(ValueError):m(xx,vv,qq,**kwargs)

def test_finite_capacity_and_parameter_census():
    m=M.new_model(22)
    assert M.count_params(m)==1075
    assert M.DKEY*M.DVAL*4==1024
    xx,vv,qq,yy,mask=M.batchify(M.make_data(4490,1))
    with torch.no_grad():
        _,s=m(xx,vv,qq)
    assert s.shape==(1,64,4) and s.element_size()==4
    with pytest.raises(ValueError):m(torch.cat((xx,xx),dim=1),vv,qq)

def test_training_reduces_loss_and_no_persistent_checkpoint():
    model,stats=M.train(seed=991,episodes=64,epochs=10)
    assert stats['final'] < stats['initial']
    assert stats['state_bytes']==1024 and stats['parameters']==1075
    assert all(torch.isfinite(p).all() for p in model.parameters())

def test_no_oracle_path_consumption_in_autonomous():
    import copy
    m=M.new_model(991)
    case=M.make_data(7131,1)[0][-1]
    left=M.autonomous(m,case)
    spoof=copy.copy(case)
    spoof.path=(999,998,997)
    spoof.family='missing_goal'
    right=M.autonomous(m,spoof)
    assert left==right

@pytest.mark.parametrize('seed',range(8))
def test_no_cross_episode_state_leak(seed):
    m=M.new_model(seed+23)
    xx,vv,qq,yy,mask=M.batchify(M.make_data(seed*741+771,1))
    with torch.no_grad():
        a,st=m(xx,vv,qq)
        other=M.batchify(M.make_data(seed*741+773,1))
        m(*other[:3])
        b,st2=m(xx,vv,qq)
    assert torch.equal(a,b) and torch.equal(st,st2)

def test_rejected_gate_range_in_numpy_reference():
    k=np.zeros((1,M.DKEY));v=np.ones((1,M.DVAL))
    for beta,g in ((-1,0),(2,0),(.5,.1)):
        with pytest.raises(ValueError):M.independent_numpy_rule(k,v,np.array([beta]),np.array([g]))

def test_calibration_not_from_heldout():
    import inspect
    from tools import r16_science
    src=inspect.getsource(r16_science.choose_threshold)
    assert 'VAL_SEEDS' in src and 'TEST_SEEDS' not in src

def test_shuffled_supervision_negative_control():
    _,honest=M.train(seed=914,episodes=96,epochs=12,shuffle_labels=False)
    _,shuffled=M.train(seed=914,episodes=96,epochs=12,shuffle_labels=True)
    assert honest['final'] + .10 < shuffled['final']
