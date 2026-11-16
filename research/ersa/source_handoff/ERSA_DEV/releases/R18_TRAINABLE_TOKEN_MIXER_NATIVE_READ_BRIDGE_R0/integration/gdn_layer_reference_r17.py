"""R17 CPU PyTorch GDN-v1-layer *structural reference*, not FLA module or kernel.

Reference architectural elements: https://github.com/fla-org/flash-linear-attention/
  fla/layers/gated_deltanet.py and fla/ops/gated_delta_rule/fused_recurrent.py

Departures, always reported: external 12->32 token embed; read-only query suffix
(beta=0 for queries); a 32->4 task readout; single head 16x32; CPU eager
recurrence rather than Triton chunk or fused kernel. These make the model a
research reference, NOT an official fully matched FLA layer comparison.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from integration import gdn_trainable_r16 as r16

MAX_RECORDS=24
MAX_QUERIES=4

class CausalDepthwise(nn.Module):
    def __init__(self, channels:int, size:int=4):
        super().__init__()
        self.size=size
        self.weight=nn.Parameter(torch.randn(channels,1,size)*0.04)
    def forward(self, x):
        return F.silu(F.conv1d(F.pad(x.transpose(1,2),(self.size-1,0)),self.weight,groups=x.shape[2]).transpose(1,2))

class GDNLayerReference(nn.Module):
    """Trainable full projection/conv/gating/norm architecture with explicit query read boundary."""
    def __init__(self, hidden=32, kdim=16, vdim=32):
        super().__init__()
        if hidden!=32 or kdim!=16 or vdim!=32:raise ValueError('PINNED_R17_ARCH')
        self.embed=nn.Linear(12,hidden,bias=False)
        self.q_proj=nn.Linear(hidden,kdim,bias=False)
        self.k_proj=nn.Linear(hidden,kdim,bias=False)
        self.v_proj=nn.Linear(hidden,vdim,bias=False)
        self.q_conv=CausalDepthwise(kdim)
        self.k_conv=CausalDepthwise(kdim)
        self.v_conv=CausalDepthwise(vdim)
        self.a_proj=nn.Linear(hidden,1,bias=False)
        self.b_proj=nn.Linear(hidden,1,bias=False)
        self.g_proj=nn.Linear(hidden,vdim,bias=False)
        self.A_log=nn.Parameter(torch.zeros(1))
        self.dt_bias=nn.Parameter(torch.tensor([-3.89],dtype=torch.float32))
        self.o_proj=nn.Linear(vdim,hidden,bias=False)
        self.readout=nn.Linear(hidden,4,bias=False)
        self.kdim=kdim;self.vdim=vdim
    def forward(self, x, value, query, *, allow=None, return_state=False):
        if x.ndim!=3 or x.shape[1:]!=(MAX_RECORDS,8):raise ValueError('INVALID_STORE')
        B=len(x)
        if value.shape!=(B,MAX_RECORDS,4):raise ValueError('INVALID_VALUES')
        if query.ndim!=3 or query.shape[0]!=B or query.shape[-1]!=8 or not (1<=query.shape[1]<=MAX_QUERIES):
            raise ValueError('INVALID_QUERY')
        if any(y.dtype!=torch.float32 for y in (x,value,query)):raise ValueError('DTYPE')
        if any(not bool(torch.isfinite(y).all()) for y in (x,value,query)):raise ValueError('NONFINITE')
        if allow is not None and (allow.shape!=(B,MAX_RECORDS) or allow.dtype!=torch.bool):raise ValueError('ADMISSION_MASK')
        Q=query.shape[1]
        # Compute the persistent state using STORE TOKENS ONLY.  Previously the
        # store and the variable-length read-only query suffix were convolved
        # in one batch.  CPU convolution dispatch can change its accumulation
        # order with the suffix length, so identical stores could yield states
        # with different low bits.  A read-only query must not influence the
        # state even through a change of numerical execution path.
        stores=torch.cat([x,value],dim=-1)
        if allow is not None:
            # Authorization precedes every convolution and embedding.
            stores=stores*allow[:,:,None].to(stores.dtype)
        h_store=self.embed(stores)
        k=F.normalize(self.k_conv(self.k_proj(h_store)),dim=-1,eps=1e-6)
        v=self.v_conv(self.v_proj(h_store))
        a=self.a_proj(h_store).squeeze(-1)
        beta=torch.sigmoid(self.b_proj(h_store).squeeze(-1))
        decay=torch.exp(-self.A_log.exp()*F.softplus(a+self.dt_bias))
        good=(allow if allow is not None else
              torch.ones((B,MAX_RECORDS),dtype=torch.bool,device=x.device))
        state=torch.zeros((B,self.kdim,self.vdim),dtype=x.dtype,device=x.device)
        for t in range(MAX_RECORDS):
            mask=good[:,t].to(x.dtype)
            factor=torch.where(good[:,t],decay[:,t],torch.ones_like(decay[:,t]))
            state=state*factor[:,None,None]
            pred=torch.bmm(k[:,t:t+1,:],state).squeeze(1)
            correction=beta[:,t,None]*(v[:,t,:]-pred)*mask[:,None]
            state=state+k[:,t,:,None]*correction[:,None,:]

        # Read-only query path: the causal Q convolution sees the three
        # preceding STORE/QUERY projections but never feeds a state update.
        # This is mathematically the suffix of a causal depthwise convolution
        # over stores+queries, while keeping the store recurrence invariant to
        # the number and content of queries.  Convolution width is pinned to 4.
        query_tokens=torch.cat([query,torch.zeros(B,Q,4,dtype=x.dtype,device=x.device)],dim=-1)
        h_query=self.embed(query_tokens)
        context=self.q_conv.size-1
        q_input=torch.cat([self.q_proj(h_store[:,-context:,:]),self.q_proj(h_query)],dim=1)
        q=F.normalize(self.q_conv(q_input)[:,context:,:],dim=-1,eps=1e-6)
        o=torch.bmm(q,state)*(self.kdim**-.5)
        o=F.rms_norm(o,(self.vdim,),eps=1e-5)*F.silu(self.g_proj(h_query))
        z=self.readout(self.o_proj(o))
        return (z,state) if return_state else z

def count_parameters(model):return sum(p.numel() for p in model.parameters())

def new_model(seed=17001):
    torch.manual_seed(seed)
    return GDNLayerReference()

def train(seed=17001,episodes=128,epochs=10,batch_size=16,shuffle_labels=False):
    torch.set_num_threads(1)
    data=r16.make_data(seed*23+199,episodes,family='mixed')
    model=new_model(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=0.004)
    rng=np.random.default_rng(seed+33)
    def batch(chunk):
        x,y,q,target,m=r16.batchify(chunk)
        return x,y,q,target,m
    def evaluate():
        vals=[]
        model.eval()
        with torch.no_grad():
            for off in range(0,len(data),batch_size):
                x,y,q,target,m=batch(data[off:off+batch_size]); p=model(x,y,q)
                vals.append((float(r16.loss_fn(p,target,m)),float(m.sum())))
        return sum(l*n for l,n in vals)/sum(n for _,n in vals)
    initial=evaluate()
    model.train()
    for epoch in range(epochs):
        for start in range(0,len(data),batch_size):
            inds=rng.permutation(len(data))[start:start+batch_size]
            x,y,q,target,m=batch([data[int(i)] for i in inds])
            if shuffle_labels:
                rr=torch.Generator().manual_seed(seed+epoch*1000+start)
                target=target[torch.randperm(len(target),generator=rr)]
            opt.zero_grad(set_to_none=True)
            p=model(x,y,q)
            loss=r16.loss_fn(p,target,m)
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
    final=evaluate()
    return model,{'seed':seed,'episodes':episodes,'epochs':epochs,'start':round(initial,7),'end':round(final,7),
                 'parameters':count_parameters(model),'state_bytes':16*32*4,
                 'conv_cache_bytes':(16+16+32)*3*4,'query_read_only':True}

def autonomous(model,case,*,max_hops=4,goal_threshold=.78,key_threshold=.93):
    ep=case.episode
    if len(ep.keys)!=24 or max_hops>4:raise ValueError('CAP')
    if min(np.linalg.norm(ep.query_goal),np.linalg.norm(ep.query_key))<1e-12:
        return {'status':'HOLD_INVALID_QUERY','reads':0}
    x=np.concatenate([ep.keys,ep.goals],axis=1).astype(np.float32)
    values=np.asarray(ep.values,np.float32)
    seen=set();qkey=ep.query_key.copy()
    model.eval()
    with torch.no_grad():
        for hop in range(1,max_hops+1):
            q=np.concatenate([qkey,ep.query_goal]).reshape(1,1,8).astype(np.float32)
            out=model(torch.from_numpy(x[None]),torch.from_numpy(values[None]),torch.from_numpy(q))
            v=out[0,0].numpy().astype(np.float64)
            if not np.isfinite(v).all():return {'status':'HOLD_NONFINITE','reads':hop}
            sims=np.array([r16._cos(v,k) for k in ep.keys]);ix=int(np.argmax(sims))
            if sims[ix]>=key_threshold:
                if ix in seen:return {'status':'HOLD_CYCLE','reads':hop}
                seen.add(ix);qkey=v;continue
            if r16._cos(v,ep.query_goal)>=goal_threshold:
                return {'status':'STOP','reads':hop,'vector':tuple(v)}
            return {'status':'HOLD_LOW_CONFIDENCE','reads':hop}
    return {'status':'HOLD_HOP_BUDGET','reads':max_hops}
