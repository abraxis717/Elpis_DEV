"""R35: experimental learned volatile writes + frozen R18 ERSA read interface.

Donor: MIT FLA GSA naive reference, Raven top-k sparse-router policy.
Not a full upstream RavenAttention layer. Not C-native write authority.
Not GDN and not a production inference implementation.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from integration.ersa_trainable_layer_r18 import ERSATokenMixer, ERSAState
from third_party.fla_gsa.naive import naive_recurrent_gsa


@dataclass(frozen=True)
class SlotState:
    hk: torch.Tensor  # [B,1,D,N]
    hv: torch.Tensor  # [B,1,N,D]
    step: int


class ERSASlotWriteRead(nn.Module):
    def __init__(self, d_model=8, num_slots=8, topk=2, max_steps=32):
        super().__init__()
        if not (4 <= d_model <= 32 and 1 <= topk <= num_slots <= 64 and 1 <= max_steps <= 128):
            raise ValueError('INVALID_R35_BOUNDS')
        self.d_model=d_model
        self.num_slots=num_slots
        self.topk=topk
        self.max_steps=max_steps
        self.q_proj=nn.Linear(d_model,d_model,bias=False)
        self.k_proj=nn.Linear(d_model,d_model,bias=False)
        self.v_proj=nn.Linear(d_model,d_model,bias=False)
        self.router=nn.Linear(d_model,num_slots,bias=False)  # Raven-style top-k router
        self.forget=nn.Linear(d_model,num_slots,bias=False)
        self.slot_keys=nn.Parameter(torch.randn(num_slots,d_model)*.1)
        self.ersa=ERSATokenMixer(d_model,max_slots=num_slots,max_hops=2)

    def initial_state(self, batch, *, device=None):
        if not 1 <= batch <= 128:
            raise ValueError('INVALID_BATCH')
        device = device or self.slot_keys.device
        d,n=self.d_model,self.num_slots
        return SlotState(torch.zeros(batch,1,d,n,device=device),
                         torch.zeros(batch,1,n,d,device=device),0)

    def forward(self,x,state=None,*,write_allowed=None,slot_allowed=None,return_aux=False):
        if x.ndim!=3 or x.shape[-1]!=self.d_model or x.dtype!=torch.float32 or not torch.isfinite(x).all():
            raise ValueError('INVALID_INPUT')
        b,t,d=x.shape;n=self.num_slots
        if not (1<=b<=128 and 1<=t<=self.max_steps and b*t*n*d*2<=2_000_000):
            raise ValueError('BOUNDED_WORK_EXCEEDED')
        if state is None:
            state=self.initial_state(b,device=x.device)
        if not isinstance(state,SlotState) or state.hk.shape!=(b,1,d,n) or state.hv.shape!=(b,1,n,d) or not 0<=state.step<=2**31-t:
            raise ValueError('BAD_INITIAL_STATE')
        if state.hk.device!=x.device or state.hv.device!=x.device or state.hk.dtype!=torch.float32 or state.hv.dtype!=torch.float32:
            raise ValueError('BAD_STATE_DEVICE_DTYPE')
        if not torch.isfinite(state.hk).all() or not torch.isfinite(state.hv).all():
            raise ValueError('NONFINITE_STATE')
        if write_allowed is None:
            write_allowed=torch.ones(b,t,device=x.device,dtype=torch.bool)
        if slot_allowed is None:
            slot_allowed=torch.ones(b,n,device=x.device,dtype=torch.bool)
        if write_allowed.shape!=(b,t) or slot_allowed.shape!=(b,n) or write_allowed.dtype!=torch.bool or slot_allowed.dtype!=torch.bool or write_allowed.device!=x.device or slot_allowed.device!=x.device:
            raise ValueError('BAD_ADMISSION')
        # Denied writes never reach the projection/router; no value or gradient leakage.
        source=torch.where(write_allowed[...,None],x,torch.zeros_like(x))
        hk,hv=state.hk,state.hv
        ys=[]; writes=[]; histories=[]
        key_bank=self.slot_keys[None,:,:].expand(b,n,d)
        zeros_meta=torch.zeros(b,n,4,device=x.device)
        issued=torch.zeros(b,n,device=x.device,dtype=torch.int64)
        for i in range(t):
            xi=source[:,i,:]
            scores=self.router(xi).masked_fill(~slot_allowed,-1.e9)
            active=slot_allowed & write_allowed[:,i,None]
            chosen=torch.zeros_like(active)
            top=scores.topk(self.topk,dim=-1).indices
            chosen.scatter_(1,top,True)
            chosen=chosen & active
            # ST preserves exact sparse forward writes while providing gradients.
            softened=torch.sigmoid(scores)
            routing=chosen.float()+softened-softened.detach()
            routing=torch.where(active,routing,torch.zeros_like(routing))
            # Unselected slots are exactly untouched; admitted writes gated by router+forget.
            forget=-F.softplus(self.forget(xi))/8.
            g=routing*forget
            s=-torch.expm1(g)
            q=self.q_proj(xi)[:,None,None,:]
            k=self.k_proj(xi)[:,None,None,:]
            v=self.v_proj(xi)[:,None,None,:]
            _,(hk,hv)=naive_recurrent_gsa(q,k,v,s[:,None,None,:],g[:,None,None,:],
                                          initial_state=(hk,hv),output_final_state=True)
            val=hv[:,0,:,:]
            # Every read is inside frozen R18's ERSA recurrent read operator.
            er_state=ERSAState(keys=key_bank,goals=key_bank,values=val,
                               metadata=zeros_meta,issued=issued,allowed=slot_allowed,
                               tick=64,generation=state.step+i)
            y,_,aux=self.ersa(x[:,i:i+1,:],er_state,return_aux=True)
            ys.append(y)
            writes.append(chosen)
            histories.append(aux['confidence'])
        final=SlotState(hk,hv,state.step+t)
        result=torch.cat(ys,dim=1)
        if not return_aux:return result,final
        return result,final,{'write_slots':torch.stack(writes,dim=1),
                            'read_confidence':torch.cat(histories,dim=1),
                            'state_bytes':2*b*d*n*4}
