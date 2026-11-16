"""R36: training/inference slot-state contract and C recurrence parity."""
import numpy as np
import pytest
import torch
from r35.slot_write import ERSASlotWriteRead
from r36.native_bridge import build_native, CSlotState, infer_native

@pytest.fixture(scope='module')
def libdir():
    import shutil
    lib,where=build_native()
    yield lib
    shutil.rmtree(where)

@pytest.fixture(autouse=True)
def single_thread():torch.set_num_threads(1)

def model(d=8,n=8,k=2):
    torch.manual_seed(3501)
    return ERSASlotWriteRead(d_model=d,num_slots=n,topk=k,max_steps=16)

@pytest.mark.parametrize('seed',list(range(36,48)))
def test_native_full_forward_vs_torch(seed,libdir):
    torch.manual_seed(seed)
    m=model();x=torch.randn(2,6,8)
    w=torch.rand(2,6)>.35
    s=torch.rand(2,8)>.4
    y,st=m(x,write_allowed=w,slot_allowed=s)
    yy,ss=infer_native(m,x,write_allowed=w,slot_allowed=s,lib=libdir)
    assert torch.allclose(y,yy,atol=3e-6,rtol=3e-6)
    assert torch.allclose(st.hk,ss.hk,atol=3e-6,rtol=3e-6)
    assert torch.allclose(st.hv,ss.hv,atol=3e-6,rtol=3e-6)
    assert ss.step==6

@pytest.mark.parametrize('cut',[1,2,3,4,5])
def test_native_split_continuation(cut,libdir):
    torch.manual_seed(300+cut)
    m=model();x=torch.randn(2,6,8);w=torch.rand(2,6)>.3
    full,fulls=infer_native(m,x,write_allowed=w,lib=libdir)
    a,state=infer_native(m,x[:,:cut],write_allowed=w[:,:cut],lib=libdir)
    b,done=infer_native(m,x[:,cut:],state,write_allowed=w[:,cut:],lib=libdir)
    assert torch.allclose(torch.cat([a,b],1),full,atol=4e-6,rtol=4e-6)
    assert torch.allclose(done.hk,fulls.hk,atol=4e-6)
    assert torch.allclose(done.hv,fulls.hv,atol=4e-6)
    assert done.step==6

@pytest.mark.parametrize('slots,topk',[(4,1),(8,1),(8,2),(16,4),(32,2)])
def test_slot_shapes_and_sparse_routing(slots,topk,libdir):
    m=model(n=slots,k=topk);x=torch.randn(2,3,8)
    y,s=m(x);yy,ss,aux=infer_native(m,x,lib=libdir,return_aux=True)
    assert torch.allclose(y,yy,atol=4e-6,rtol=4e-6)
    assert (aux['write_slots'].sum(-1)<=topk).all()
    assert aux['native_state_bytes']==2*2*8*slots*4
    assert s.hk.shape==ss.hk.shape

@pytest.mark.parametrize('seed',range(6))
def test_denied_payload_no_state_leak(seed,libdir):
    torch.manual_seed(seed)
    m=model();x=torch.randn(2,5,8)
    w=torch.tensor([[True,False,True,False,True],[False,True,True,False,True]])
    s=torch.tensor([[True,False,True,False,True,False,True,False],
                    [False,True,False,True,False,True,False,True]])
    _,final=infer_native(m,x,write_allowed=w,slot_allowed=s,lib=libdir)
    poison=x.clone();poison[~w]=1e5
    _,other=infer_native(m,poison,write_allowed=w,slot_allowed=s,lib=libdir)
    assert torch.equal(final.hk,other.hk)
    assert torch.equal(final.hv,other.hv)
    assert torch.equal(final.hk[:,0,:,~s[0]][0], torch.zeros_like(final.hk[:,0,:,~s[0]][0]))

@pytest.mark.parametrize('seed',range(4))
def test_native_read_only_slot_state(seed,libdir):
    m=model();torch.manual_seed(seed)
    st=m.initial_state(2)
    w=torch.zeros(2,5,dtype=torch.bool)
    _,fin=infer_native(m,torch.randn(2,5,8),st,write_allowed=w,lib=libdir)
    assert torch.equal(st.hk,fin.hk) and torch.equal(st.hv,fin.hv)
    assert fin.step==5


def test_stale_generation_rejected_without_mutation(libdir):
    with CSlotState(libdir,1,8,8) as s:
        a=torch.randn(1,8);g=torch.zeros(1,8);old=s.export_torch()
        with pytest.raises(ValueError,match='REJECTED'):
            s.step_once(a,a,a,g,g,expected=1)
        assert s.step==0
        new=s.export_torch()
        assert torch.equal(old.hk,new.hk) and torch.equal(old.hv,new.hv)


def test_nonfinite_step_atomic_reject(libdir):
    with CSlotState(libdir,1,8,8) as s:
        x=torch.ones(1,8);z=torch.zeros(1,8)
        s.step_once(x,x,x,z,z)
        old=s.export_torch()
        bad=x.clone();bad[0,2]=float('nan')
        with pytest.raises(ValueError,match='REJECTED'):
            s.step_once(x,bad,x,z,z)
        cur=s.export_torch()
        assert s.step==1
        assert torch.equal(old.hk,cur.hk) and torch.equal(old.hv,cur.hv)


def test_nonfinite_result_atomic_reject(libdir):
    with CSlotState(libdir,1,8,8) as s:
        x=torch.ones(1,8);z=torch.zeros(1,8)
        before=s.export_torch()
        bad=torch.full((1,8),1000.)
        with pytest.raises(ValueError,match='REJECTED'):
            s.step_once(x,x,x,z,bad)
        after=s.export_torch()
        assert torch.equal(before.hk,after.hk) and torch.equal(before.hv,after.hv) and s.step==0


def test_import_nonfinite_atomic_reject(libdir):
    with CSlotState(libdir,1,8,8) as s:
        a=s.export_torch()
        bad=a.hk.clone();bad[0,0,0,0]=float('nan')
        with pytest.raises(ValueError,match='INVALID_IMPORT'):
            from r35.slot_write import SlotState
            s.import_torch(SlotState(bad,a.hv,0))
        cur=s.export_torch()
        assert torch.equal(cur.hk,a.hk)


def test_native_bank_bounds(libdir):
    with pytest.raises(ValueError):CSlotState(libdir,129,8,8)
    with pytest.raises(ValueError):CSlotState(libdir,1,8,65)
    with pytest.raises(ValueError):CSlotState(libdir,1,3,8)
    with pytest.raises(ValueError):CSlotState(libdir,1,8,8,2**31)


def test_no_inference_weight_or_input_mutation(libdir):
    m=model();x=torch.randn(2,4,8)
    before=[p.detach().clone() for p in m.parameters()]
    src=x.clone()
    infer_native(m,x,lib=libdir)
    assert torch.equal(src,x)
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))


def test_admission_shape_fail_closed(libdir):
    m=model();x=torch.randn(2,4,8)
    for kw in ({'write_allowed':torch.ones(2,4)},
               {'slot_allowed':torch.ones(2,8)},
               {'write_allowed':torch.ones(2,5,dtype=torch.bool)}):
        with pytest.raises(ValueError):infer_native(m,x,lib=libdir,**kw)


def test_state_import_rejects_stale(libdir):
    m=model();st=m.initial_state(1)
    with CSlotState(libdir,1,8,8,2) as native:
        with pytest.raises(ValueError):native.import_torch(st)


def test_native_requires_explicit_library():
    m=model()
    with pytest.raises(ValueError,match='EXPLICIT_NATIVE_LIB_REQUIRED'):
        infer_native(m,torch.randn(1,2,8))

@pytest.mark.parametrize('seed',range(8))
def test_native_gsa_recurrence_independent_step(seed,libdir):
    """The independent C recurrence matches the pinned MIT FLA donor reference."""
    from third_party.fla_gsa.naive import naive_recurrent_gsa
    torch.manual_seed(seed+10100)
    b,d,n=2,8,8
    q=torch.randn(b,d);k=torch.randn(b,d);v=torch.randn(b,d)
    g=-torch.rand(b,n)*.8;s=-torch.expm1(g)
    with CSlotState(libdir,b,d,n) as slot:
        native=slot.step_once(q,k,v,s,g)
        actual=slot.export_torch()
    yy,(hk,hv)=naive_recurrent_gsa(q[:,None,None,:],k[:,None,None,:],
        v[:,None,None,:],s[:,None,None,:],g[:,None,None,:])
    assert torch.allclose(yy[:,0,0,:],native,atol=2e-6,rtol=2e-6)
    assert torch.allclose(hk,actual.hk,atol=2e-6,rtol=2e-6)
    assert torch.allclose(hv,actual.hv,atol=2e-6,rtol=2e-6)

@pytest.mark.parametrize('workers',[2,4,8])
def test_concurrent_same_generation_exactly_one_commit(workers,libdir):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    with CSlotState(libdir,1,8,8) as slot:
        q=torch.ones(1,8);s=torch.full((1,8),.1);g=torch.full((1,8),-.2)
        barrier=threading.Barrier(workers)
        def attempt(_):
            barrier.wait()
            try:slot.step_once(q,q,q,s,g,expected=0);return 0
            except ValueError:return 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results=list(ex.map(attempt,range(workers)))
        assert results.count(0)==1 and results.count(1)==workers-1
        assert slot.step==1
