"""Independent algebraic checks of the FLA GSA donor-derived CPU recurrence."""
import pytest
import torch
from third_party.fla_gsa.naive import naive_recurrent_gsa
from r35.slot_write import ERSASlotWriteRead


def donor_inputs():
    torch.manual_seed(351)
    q=torch.randn(2,3,1,4)
    k=torch.randn_like(q)
    v=torch.randn(2,3,1,5)
    g=-torch.rand(2,3,1,6)
    s=-torch.expm1(g)
    return q,k,v,s,g


def test_first_token_independent_algebra():
    q,k,v,s,g=donor_inputs()
    out,(hk,hv)=naive_recurrent_gsa(q[:,:1],k[:,:1],v[:,:1],s[:,:1],g[:,:1])
    kh=k[:,0,0,:,None]*s[:,0,0,None,:]
    wh=torch.softmax(torch.einsum('bd,bdm->bm',q[:,0,0]/(4**.5),kh),dim=-1)
    vh=s[:,0,0,:,None]*v[:,0,0,None,:]
    expected=torch.einsum('bm,bmd->bd',wh,vh)
    assert torch.allclose(out[:,0,0],expected,atol=2e-6)
    assert torch.allclose(hk[:,0],kh,atol=1e-6)
    assert torch.allclose(hv[:,0],vh,atol=1e-6)


@pytest.mark.parametrize('cut',[1,2])
def test_donor_split_state_equals_whole(cut):
    q,k,v,s,g=donor_inputs()
    whole,full=naive_recurrent_gsa(q,k,v,s,g)
    left,st=naive_recurrent_gsa(q[:,:cut],k[:,:cut],v[:,:cut],s[:,:cut],g[:,:cut])
    right,tail=naive_recurrent_gsa(q[:,cut:],k[:,cut:],v[:,cut:],s[:,cut:],g[:,cut:],initial_state=st)
    assert torch.allclose(torch.cat([left,right],dim=1),whole,atol=2e-6)
    for a,b in zip(tail,full):assert torch.allclose(a,b,atol=2e-6)


def test_donor_gradients_to_all_projections():
    q,k,v,s,g=[x.detach().requires_grad_(True) for x in donor_inputs()]
    output,final=naive_recurrent_gsa(q,k,v,s,g)
    (output.square().mean()+sum(x.square().mean() for x in final)).backward()
    for x in (q,k,v,s,g):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_denied_all_slots_and_writes_is_identity_state():
    m=ERSASlotWriteRead()
    torch.manual_seed(352)
    x=torch.randn(1,4,8)
    first=m.initial_state(1)
    y,after=m(x,first,write_allowed=torch.zeros(1,4,dtype=torch.bool),slot_allowed=torch.zeros(1,8,dtype=torch.bool))
    assert torch.equal(y,x)
    assert torch.equal(first.hk,after.hk) and torch.equal(first.hv,after.hv)


def test_nonfinite_input_rejected_before_recurrent_write():
    m=ERSASlotWriteRead();x=torch.zeros(1,2,8)
    x[:,1,0]=float('nan')
    with pytest.raises(ValueError,match='INVALID_INPUT'):m(x)
