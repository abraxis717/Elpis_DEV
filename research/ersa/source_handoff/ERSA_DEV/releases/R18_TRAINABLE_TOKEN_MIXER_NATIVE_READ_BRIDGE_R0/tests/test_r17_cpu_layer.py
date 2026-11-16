import pytest, torch
from integration import gdn_layer_reference_r17 as g
from integration import gdn_trainable_r16 as r16

def data():
    x=torch.rand(2,24,8);y=torch.rand(2,24,4);q=torch.rand(2,3,8)
    return x,y,q

def test_structural_components():
    m=g.new_model();names=set(dict(m.named_parameters()))
    assert {'A_log','dt_bias','q_proj.weight','k_proj.weight','v_proj.weight','g_proj.weight','a_proj.weight','b_proj.weight','o_proj.weight','readout.weight'} <= names
    assert all(f'{x}_conv.weight' in names for x in ('q','k','v'))
    assert g.count_parameters(m)>r16.count_params(r16.new_model())

def test_shapes():
    m=g.new_model();x,y,q=data();out,state=m(x,y,q,return_state=True)
    assert out.shape==(2,3,4) and state.shape==(2,16,32)
    assert torch.isfinite(out).all() and torch.isfinite(state).all()

def test_determinism():
    m=g.new_model();x,y,q=data()
    with torch.no_grad(): assert torch.equal(m(x,y,q),m(x,y,q))

@pytest.mark.parametrize('bad', ['nonfinite','dtype','keys','values','query','mask'])
def test_bad_input(bad):
    m=g.new_model();x,y,q=data();kwargs={}
    if bad=='nonfinite':x[0,0,0]=float('nan')
    elif bad=='dtype':y=y.double()
    elif bad=='keys':x=x[:,:23]
    elif bad=='values':y=y[:,:23]
    elif bad=='query':q=q[:,:,:7]
    elif bad=='mask':kwargs['allow']=torch.ones(2,24)
    with pytest.raises(ValueError):m(x,y,q,**kwargs)

def test_mask_rejected_write():
    m=g.new_model();x,y,q=data();allow=torch.ones(2,24,dtype=torch.bool);allow[:,12:]=False
    y2=y.clone();y2[:,12:]=1e6
    with torch.no_grad():
        # Denied record data is zeroed before convolution; no leakage permitted.
        a=m(x,y,q,allow=allow);b=m(x,y2,q,allow=allow)
    assert a.shape==b.shape and torch.allclose(a,b,rtol=0,atol=0)

def test_mask_all_records_no_state():
    m=g.new_model();x,y,q=data();allow=torch.zeros(2,24,dtype=torch.bool)
    with torch.no_grad():_,state=m(x,y,q,allow=allow,return_state=True)
    assert torch.count_nonzero(state)==0

def test_rejected_inference_preserves_weights():
    m=g.new_model();x,y,q=data();before={k:v.detach().clone() for k,v in m.state_dict().items()}
    with torch.no_grad():m(x,y,q)
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())

def test_training_gradients():
    m=g.new_model();x,y,q=data();out=m(x,y,q);loss=(out**2).mean();loss.backward()
    for key in ('q_proj.weight','k_proj.weight','v_proj.weight','a_proj.weight','b_proj.weight','g_proj.weight','o_proj.weight'):
        grad=dict(m.named_parameters())[key].grad
        assert grad is not None and torch.isfinite(grad).all()

def test_cpu_reference_24_records():
    case=r16.make_data(92000,1,family='positive')[0][-1]
    result=g.autonomous(g.new_model(),case)
    assert result['status'] in ('STOP','HOLD_LOW_CONFIDENCE','HOLD_HOP_BUDGET','HOLD_CYCLE','HOLD_INVALID_QUERY')
    assert 0<=result['reads']<=4

@pytest.mark.parametrize('seed', [11,12,13,14,15])
def test_independent_numpy_state_differential(seed):
    import numpy as np
    torch.manual_seed(seed)
    model=g.new_model(seed)
    x,y,q=data()
    allow=torch.rand(2,24)>.20
    with torch.no_grad():
        _,state=model(x,y,q,allow=allow,return_state=True)
        tokens=torch.cat([torch.cat([x,y],-1),torch.cat([q,torch.zeros(2,3,4)],-1)],1)
        tokens=torch.cat([tokens[:,:24]*allow[:,:,None],tokens[:,24:]],1)
        h=model.embed(tokens)
        kk=torch.nn.functional.normalize(model.k_conv(model.k_proj(h)),dim=-1).numpy()
        vv=model.v_conv(model.v_proj(h)).numpy()
        aa=model.a_proj(h).squeeze(-1).numpy()
        bb=torch.sigmoid(model.b_proj(h).squeeze(-1)).numpy()
        a=float(model.A_log.exp().item());dt=float(model.dt_bias.item())
    for bi in range(2):
        s=np.zeros((16,32),dtype=np.float64)
        for ti in range(24):
            if allow[bi,ti]:
                rate=np.exp(-a*np.logaddexp(0.,float(aa[bi,ti]+dt)))
                s*=rate
                k=kk[bi,ti,:].astype(np.float64)
                v=vv[bi,ti,:].astype(np.float64)
                err=v-k@s
                s+=np.outer(k,bb[bi,ti]*err)
        assert np.max(np.abs(s-state[bi].numpy()))<3e-5

def test_read_only_queries_do_not_change_state():
    model=g.new_model();x,y,q=data()
    with torch.no_grad():
        _,single=model(x,y,q[:,:1],return_state=True)
        _,three=model(x,y,q,return_state=True)
    assert torch.equal(single,three)


def test_deny_masks_are_no_read_no_write():
    m=g.new_model();x,y,q=data()
    allow=torch.ones((2,24),dtype=torch.bool);allow[:,4:9]=False
    x2=x.clone();y2=y.clone();x2[:,4:9]=1e7;y2[:,4:9]=-1e8
    with torch.no_grad():
        a,sa=m(x,y,q,allow=allow,return_state=True)
        b,sb=m(x2,y2,q,allow=allow,return_state=True)
    assert torch.equal(a,b) and torch.equal(sa,sb)


def test_train_seed_reproducible():
    a,info_a=g.train(17005,episodes=24,epochs=2)
    b,info_b=g.train(17005,episodes=24,epochs=2)
    assert info_a==info_b
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("seed", [1,2,14,16,24,29,32,37,38])
@pytest.mark.parametrize("masked", [False,True])
def test_read_only_suffix_bitwise_invariant_regression(seed,masked):
    # Regresses the original CPU convolution-length-dependent state mismatch.
    # Some of these seeds failed bitwise equality in the unmodified R17.
    model=g.new_model(seed)
    torch.manual_seed(seed+324)
    x=torch.randn(2,24,8)
    y=torch.randn(2,24,4)
    q=torch.randn(2,4,8)
    mask=(torch.rand(2,24)>.2) if masked else None
    with torch.no_grad():
        first,one=model(x,y,q[:,:1],allow=mask,return_state=True)
        many,four=model(x,y,q,allow=mask,return_state=True)
        modified=q.clone()
        modified[:,:,:]+=23.0
        _,changed=model(x,y,modified,allow=mask,return_state=True)
    assert torch.equal(one,four)
    assert torch.equal(one,changed)
    assert torch.allclose(first[:,0],many[:,0],atol=2e-6,rtol=2e-6)

def test_read_only_query_is_not_vacuous_update():
    # Control: a change in an authorized store value MUST change the state.
    model=g.new_model(17017)
    torch.manual_seed(127)
    x=torch.randn(2,24,8);y=torch.randn(2,24,4);q=torch.randn(2,4,8)
    with torch.no_grad():
        _,before=model(x,y,q,return_state=True)
        altered=y.clone();altered[:,12,:]+=9.0
        _,after=model(x,altered,q,return_state=True)
    assert not torch.equal(before,after)

def test_read_only_query_is_invariant_after_offline_training():
    model,_=g.train(17055,episodes=24,epochs=2)
    x,y,q=data()
    model.eval()
    with torch.no_grad():
        _,first=model(x,y,q[:,:1],return_state=True)
        _,more=model(x,y,q,return_state=True)
    assert torch.equal(first,more)
