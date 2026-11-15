import pytest
import torch
from r35.slot_write import ERSASlotWriteRead

@pytest.fixture(autouse=True)
def cpu_single_thread():
    torch.set_num_threads(1)


def model():
    torch.manual_seed(35)
    return ERSASlotWriteRead(d_model=8,num_slots=8,topk=2,max_steps=16)


def test_output_state_shapes_and_explicit_memory():
    m=model();x=torch.randn(2,6,8)
    y,s,a=m(x,return_aux=True)
    assert y.shape==(2,6,8) and s.hk.shape==(2,1,8,8) and s.hv.shape==(2,1,8,8)
    assert a['write_slots'].sum(-1).max()<=2
    assert s.step==6 and a['state_bytes']==1024


@pytest.mark.parametrize('cut',[1,2,3,4,5])
def test_continuation_chunk_equivalence(cut):
    m=model();x=torch.randn(2,6,8)
    w=torch.tensor([[True,True,False,True,True,False],[True,False,True,True,True,True]])
    full,fs=m(x,write_allowed=w)
    p,ps=m(x[:,:cut],write_allowed=w[:,:cut])
    q,qs=m(x[:,cut:],ps,write_allowed=w[:,cut:])
    assert torch.allclose(torch.cat([p,q],1),full,atol=2e-5,rtol=2e-5)
    assert torch.allclose(qs.hv,fs.hv,atol=2e-5,rtol=2e-5)
    assert torch.allclose(qs.hk,fs.hk,atol=2e-5,rtol=2e-5)


def test_denied_write_is_invisible_and_has_zero_gradient():
    m=model();x=torch.randn(1,4,8,requires_grad=True)
    mask=torch.tensor([[True,False,True,True]])
    y,s=m(x,write_allowed=mask)
    alt=x.detach().clone();alt[:,1,:]+=10000
    yy,ss=m(alt,write_allowed=mask)
    # Same-token query still sees its own value; subsequent tokens must not see denied write.
    assert torch.allclose(y[:,2:],yy[:,2:],atol=1e-6)
    assert torch.allclose(ss.hv,s.hv,atol=1e-6)
    s.hv.square().sum().backward()
    assert x.grad[:,1,:].abs().max()==0


def test_denied_slots_unmodified_and_no_gradient():
    m=model();x=torch.randn(1,4,8)
    s0=m.initial_state(1)
    mask=torch.tensor([[True,False,True,False,True,False,True,False]])
    _,s=m(x,s0,slot_allowed=mask)
    assert torch.equal(s.hv[:,0,~mask[0]],s0.hv[:,0,~mask[0]])
    assert torch.equal(s.hk[:,:,:,~mask[0]],s0.hk[:,:,:,~mask[0]])


def test_trainable_write_and_read_gradients():
    m=model();x=torch.randn(2,4,8)
    y,state=m(x)
    loss=y.square().mean()+state.hv.square().mean()
    loss.backward()
    for p in [m.router.weight,m.forget.weight,m.v_proj.weight,m.ersa.projection.weight]:
        assert p.grad is not None and p.grad.abs().sum()>0


@pytest.mark.parametrize('slots,topk',[(4,1),(8,1),(8,2),(16,4)])
def test_bounded_sparse_routing(slots,topk):
    m=ERSASlotWriteRead(num_slots=slots,topk=topk)
    _,_,a=m(torch.randn(2,3,8),return_aux=True)
    assert (a['write_slots'].sum(-1)<=topk).all()


def test_query_only_does_not_mutate_memory():
    m=model();x=torch.randn(2,3,8)
    initial=m.initial_state(2)
    _,final=m(x,initial,write_allowed=torch.zeros(2,3,dtype=torch.bool))
    assert torch.equal(final.hk,initial.hk) and torch.equal(final.hv,initial.hv)
    assert final.step==3


def test_rejects_work_overflow():
    m=model()
    with pytest.raises(ValueError):m(torch.randn(1,17,8))


def test_state_input_never_modified():
    m=model();s0=m.initial_state(2)
    before=(s0.hk.clone(),s0.hv.clone())
    m(torch.randn(2,5,8),s0)
    assert torch.equal(s0.hk,before[0]) and torch.equal(s0.hv,before[1])
