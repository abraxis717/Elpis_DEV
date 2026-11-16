import dataclasses
import pathlib
import shutil
import numpy as np
import pytest
import torch
from integration import ersa_trainable_layer_r18 as r
from integration import ersa_native_snapshot_r18 as bridge
from tools import r18_science
from integration.ersa_selector_r12 import make_episode,open_state,score_native_source,CSelector
from integration.ssc_native_r11 import compile_r11
from native.ersa_native import build_native,NativeOps

@pytest.fixture(autouse=True)
def torch_single_thread():
    torch.set_num_threads(1)

@pytest.fixture
def case():
    return r18_science.batch(38271,3,two_hop=True)


def test_batched_shape_identity_and_gradients(case):
    x,g,s,_,path=case
    module=r.ERSATokenMixer(8,max_slots=12,max_hops=2)
    y,returned,aux=module(x,s,goal=g,return_aux=True)
    assert returned is s and y.shape==x.shape
    assert aux['slot_probs'].shape==(3,1,2,12)
    assert aux['stop_probs'].shape==(3,1,2)
    assert aux['path_indices'].shape==(3,1,2)
    assert aux['confidence'].min()>=0 and aux['confidence'].max()<=1
    loss=(y-x).square().mean()+aux['slot_probs'][:,0,0,:].gather(1,path[:,0,None]).mean()
    loss.backward()
    for p in [module.selection,module.halt,module.projection.weight,module.mix_logit]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


@pytest.mark.parametrize('hops',[1,2,3,4])
def test_no_runtime_state_or_parameter_mutation(case,hops):
    x,g,s,_,_=case
    module=r.ERSATokenMixer(8,max_slots=12,max_hops=hops)
    before=[a.detach().clone() for a in (s.keys,s.goals,s.values,s.metadata,s.issued,s.allowed)]
    params=[p.detach().clone() for p in module.parameters()]
    with torch.no_grad():
        a,ret,aux=module(x,s,goal=g,return_aux=True)
        b,ret2,aux2=module(x,s,goal=g,return_aux=True)
    assert ret is s and ret2 is s and torch.equal(a,b)
    assert all(torch.equal(p,q) for p,q in zip(params,module.parameters()))
    assert all(torch.equal(p,q) for p,q in zip(before,(s.keys,s.goals,s.values,s.metadata,s.issued,s.allowed)))
    assert aux['generation']==s.generation


@pytest.mark.parametrize('hops',[1,2,3])
def test_masked_metadata_and_values_have_no_effect(case,hops):
    x,g,s,_,_=case
    mask=s.allowed.clone();mask[:,3:]=False
    s=dataclasses.replace(s,allowed=mask)
    other=dataclasses.replace(s,keys=s.keys.clone(),goals=s.goals.clone(),
       values=s.values.clone(),metadata=s.metadata.clone())
    for a in ('keys','goals','values'):
        getattr(other,a)[:,3:]=999.25
    other.metadata[:,3:,0:2]=999.25
    other.metadata[:,3:,2]=-1
    other.metadata[:,3:,3]=1
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=hops)
    with torch.no_grad():
        y1,_,aux1=m(x,s,goal=g,return_aux=True)
        y2,_,aux2=m(x,other,goal=g,return_aux=True)
    assert torch.equal(y1,y2)
    assert torch.equal(aux1['read'],aux2['read'])


def test_all_denied_is_exact_noop(case):
    x,g,s,_,_=case
    s=dataclasses.replace(s,allowed=torch.zeros_like(s.allowed))
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=4)
    y,_,aux=m(x,s,goal=g,return_aux=True)
    assert torch.equal(y,x)
    assert torch.count_nonzero(aux['read'])==0
    assert torch.count_nonzero(aux['confidence'])==0
    assert (aux['path_indices']==-1).all()


@pytest.mark.parametrize('invalid',[ 'shape','slots','work','dtype','nan','mask','issued','time','metadata','query'])
def test_invalid_state_fails_closed(case,invalid):
    x,g,s,_,_=case
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=4)
    if invalid=='shape':s=dataclasses.replace(s,goals=s.goals[:,:10])
    elif invalid=='slots':m=r.ERSATokenMixer(8,max_slots=4)
    elif invalid=='work':m=r.ERSATokenMixer(8,max_slots=12,max_work_elements=1)
    elif invalid=='dtype':s=dataclasses.replace(s,keys=s.keys.double())
    elif invalid=='nan':
        k=s.keys.clone();k[0,0,0]=float('nan');s=dataclasses.replace(s,keys=k)
    elif invalid=='mask':s=dataclasses.replace(s,allowed=s.allowed.float())
    elif invalid=='issued':s=dataclasses.replace(s,issued=torch.full_like(s.issued,70))
    elif invalid=='time':s=dataclasses.replace(s,tick=-1)
    elif invalid=='metadata':
        meta=s.metadata.clone();meta[0,0,2]=1;s=dataclasses.replace(s,metadata=meta)
    elif invalid=='query':x=x.double()
    with pytest.raises(ValueError):m(x,s,goal=g)


def test_sentinel_stale_generation_is_explicit_state(case):
    x,g,s,_,_=case
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=1)
    _,ret,aux=m(x,dataclasses.replace(s,generation=412),goal=g,return_aux=True)
    assert ret.generation==412 and aux['generation']==412
    # This is a copied version field; caller must still check the live C owner.


@pytest.mark.parametrize('hops',[1,2,4])
def test_hard_mode_bounded_and_pointer_veto(case,hops):
    x,g,s,_,_=case
    module=r.ERSATokenMixer(8,max_slots=12,max_hops=hops)
    module.halt.data[:]=torch.tensor([0.,0.,0.,0.,0.,100.])
    _,_,aux=module(x,s,goal=g,mode='hard',return_aux=True)
    assert aux['steps']==hops
    assert torch.isfinite(aux['read']).all()
    if hops>=2:
        # A selected pointer equal to an admitted key cannot be stopped on.
        pointer=aux['selected_reads'][:,0,0,:]
        cos=r.ERSATokenMixer._cos(pointer[:,None,:],s.keys).amax(dim=-1)
        assert ((cos>=.999999) <= (aux['stop_probs'][:,0,0]==0)).all()


@pytest.mark.parametrize('case_seed',[171,172,173,174,175,176])
def test_native_r7_r12_first_hop_feature_and_score_parity(case_seed):
    ep=make_episode(case_seed,n=24)
    c_lib=build_native();selector_lib=score_native_source()
    try:
        ops=NativeOps(c_lib);sel=CSelector(selector_lib)
        native=ops.features(ep.keys,ep.goals,ep.values,ep.meta,ep.issued,64,64.,
                            ep.query_key,ep.query_goal,ep.query_dir)
        weight=np.array([.4,0,0,0,0,.8,0,.1],dtype=np.float64)
        state=r.ERSAState(torch.from_numpy(ep.keys.copy()).unsqueeze(0),
             torch.from_numpy(ep.goals.copy()).unsqueeze(0),torch.from_numpy(ep.values.copy()).unsqueeze(0),
             torch.from_numpy(ep.meta.copy()).unsqueeze(0),torch.from_numpy(ep.issued.copy()).unsqueeze(0),
             torch.ones((1,24),dtype=torch.bool))
        m=r.ERSATokenMixer(4,max_slots=24,max_hops=1).double()
        with torch.no_grad():m.selection.copy_(torch.from_numpy(weight))
        q=torch.from_numpy(ep.query_key.copy()).reshape(1,1,4)
        goal=torch.from_numpy(ep.query_goal.copy()).reshape(1,1,4)
        direction=torch.from_numpy(ep.query_dir.copy()).reshape(1,1,4)
        _,_,aux=m(q,state,goal=goal,direction=direction,return_aux=True)
        feat=aux['features'][0,0,0].detach().numpy()
        scores=aux['scores'][0,0,0].detach().numpy()
        assert np.allclose(feat,native,rtol=0,atol=2e-12)
        assert np.allclose(scores,sel.score(native,weight),rtol=0,atol=2e-12)
        rank=ops.rank(np.round(scores,12),np.arange(24,dtype=np.int64),np.ones(24,dtype=np.uint8),2)
        assert rank[0]==int(np.argmax(scores))
    finally:
        shutil.rmtree(pathlib.Path(c_lib).parent,ignore_errors=True)
        shutil.rmtree(pathlib.Path(selector_lib).parent,ignore_errors=True)


@pytest.mark.parametrize('denied',[False,True])
def test_r11_native_state_snapshot_bridge_no_generation_mutation(denied):
    ep=make_episode(1864,n=12)
    path=compile_r11()
    try:
        with open_state(path,ep) as owner:
            initial=owner.generation
            admitted=np.ones(12,dtype=np.bool_)
            if denied:admitted[6:]=False
            state=bridge.from_native_owner(owner,ep,admitted)
            assert owner.generation==initial==state.generation
            if denied:assert torch.count_nonzero(state.values[:,6:])==0
            assert np.allclose(state.values[0,:6 if denied else 12].numpy(),
                              ep.values[:6 if denied else 12],atol=0,rtol=0)
            m=r.ERSATokenMixer(4,max_slots=12,max_hops=2).double()
            q=torch.tensor(ep.query_key).reshape(1,1,4)
            goal=torch.tensor(ep.query_goal).reshape(1,1,4)
            with torch.no_grad():_,same,_=m(q,state,goal=goal,return_aux=True)
            assert same is state and owner.generation==initial
    finally:
        shutil.rmtree(pathlib.Path(path).parent,ignore_errors=True)


def test_tiny_model_trains_selection_and_halt_without_runtime_writes():
    # One small optimizer step is the contract test; multi-seed science runs separately.
    torch.manual_seed(12)
    q,g,s,labels,path=r18_science.batch(19001,12,two_hop=True)
    model=r18_science.TinyNeuralERSAModel(2)
    opt=torch.optim.Adam(model.parameters(),lr=.02)
    logits,returned,aux=model(q,g,s)
    assert returned is s
    loss=torch.nn.functional.cross_entropy(logits,labels)
    loss=loss-torch.log(aux['slot_probs'][:,0,0,:].gather(1,path[:,0,None]).clamp_min(1e-12)).mean()
    loss.backward()
    assert model.ersa.selection.grad.norm()>0
    assert model.ersa.halt.grad is not None and torch.isfinite(model.ersa.halt.grad).all()
    before=model.ersa.selection.detach().clone();opt.step()
    assert not torch.equal(before,model.ersa.selection)

@pytest.mark.parametrize('seed',[211,212,213,214,215,216,217,218])
def test_repeated_heterogeneous_batch_and_token_shapes(seed):
    torch.manual_seed(seed)
    b=2+(seed%3);t=1+(seed%4);n=8+(seed%5);d=8
    q=torch.randn(b,t,d)
    keys=torch.randn(b,n,d);goals=torch.randn(b,n,d);values=torch.randn(b,n,d)
    mask=torch.rand(b,n)>.2
    s=r.ERSAState(keys,goals,values,torch.zeros(b,n,4),
                  torch.zeros(b,n,dtype=torch.long),mask)
    m=r.ERSATokenMixer(d,max_slots=12,max_hops=3)
    y,ident,aux=m(q,s,return_aux=True)
    assert ident is s and y.shape==q.shape
    assert aux['slot_probs'].shape==(b,t,3,n)
    assert torch.isfinite(y).all()
    assert torch.allclose(aux['slot_probs'].sum(-1),mask.any(-1)[:,None,None].float().expand(b,t,3),atol=1e-6)


@pytest.mark.parametrize('scale',[1e1,1e3,1e5,1e6])
def test_large_finite_values_do_not_leak_denied_entries(case,scale):
    x,g,s,_,_=case
    allow=s.allowed.clone();allow[:,1:]=False
    s=dataclasses.replace(s,allowed=allow)
    tampered=dataclasses.replace(s,values=s.values.clone())
    tampered.values[:,1:]=scale
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=2)
    with torch.no_grad():
        a,_,_=m(x,s,goal=g,return_aux=True)
        b,_,_=m(x,tampered,goal=g,return_aux=True)
    assert torch.equal(a,b)


@pytest.mark.parametrize('badmode',[None,1,'native','sample'])
def test_unsupported_routing_mode_fails_closed(case,badmode):
    x,g,s,_,_=case
    m=r.ERSATokenMixer(8,max_slots=12)
    with pytest.raises(ValueError):m(x,s,goal=g,mode=badmode)


@pytest.mark.parametrize('attempt',[0,1,2,3,4,5])
def test_denied_record_gradient_is_zero(attempt):
    torch.manual_seed(820+attempt)
    x,g,s,_,_=r18_science.batch(91881+attempt,2,two_hop=True)
    allowed=s.allowed.clone();allowed[:,0]=False
    vals=s.values.clone().requires_grad_()
    s=dataclasses.replace(s,allowed=allowed,values=vals)
    m=r.ERSATokenMixer(8,max_slots=12,max_hops=2)
    y,_,_=m(x,s,goal=g,return_aux=True)
    y.sum().backward()
    assert torch.equal(vals.grad[:,0],torch.zeros_like(vals.grad[:,0]))


def test_explicit_read_only_snapshot_never_mutates_native_source():
    ep=make_episode(19566,n=8)
    lib=compile_r11()
    try:
        with open_state(lib,ep) as owner:
            snap_before=owner.snapshot()
            initial=owner.generation
            permitted=np.array([True,True,True,False,False,False,False,False])
            for _ in range(5):
                st=bridge.from_native_owner(owner,ep,permitted)
                assert st.generation==initial
                assert owner.snapshot()==snap_before
                assert owner.generation==initial
    finally:shutil.rmtree(pathlib.Path(lib).parent,ignore_errors=True)


def test_parameters_are_accounted_separately_from_task_decoder():
    layer=r.ERSATokenMixer(8,max_slots=12,max_hops=2)
    model=r18_science.TinyNeuralERSAModel(2)
    assert sum(p.numel() for p in layer.parameters())==79
    assert sum(p.numel() for p in model.output.parameters())==36
    assert sum(p.numel() for p in model.parameters())==115
