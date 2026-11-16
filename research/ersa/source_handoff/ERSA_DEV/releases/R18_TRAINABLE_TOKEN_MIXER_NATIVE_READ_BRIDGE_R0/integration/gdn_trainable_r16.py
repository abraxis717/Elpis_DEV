"""R16 trainable gated-delta cell benchmark; independent, no external source copied.

Uses FLA's published decay -> error -> delta update order. NOT the full FLA
GatedDeltaNet neural layer: no causal convolution, gate normalization or mixer.
Oracle paths are used only for offline teacher-forced supervised training/scoring.
Inference uses the observable query and recurrent output only.
"""
from __future__ import annotations
import math
import numpy as np

DKEY=64
DVAL=4
SLOTS=24
MAX_HOPS=4


def make_data(start, count, *, family='positive'):
    from integration.ersa_controller_r13 import make_chain
    out=[]
    for i in range(count):
        tag=family if family!='mixed' else ('hard_pointer' if i%4==0 else 'positive')
        case=make_chain(start+i*104729,family=tag)
        ep=case.episode
        x=np.concatenate([ep.keys,ep.goals],axis=1).astype(np.float32)
        q=[];ys=[]
        for step,idx in enumerate(case.path):
            key=ep.query_key if step==0 else ep.values[case.path[step-1]]
            q.append(np.concatenate([key,ep.query_goal]).astype(np.float32))
            ys.append(ep.values[idx].astype(np.float32))
        out.append((x,ep.values.astype(np.float32),q,ys,case))
    return out


def batchify(items):
    import torch
    B=len(items)
    x=np.stack([it[0] for it in items]);v=np.stack([it[1] for it in items]);
    q=np.zeros((B,MAX_HOPS,8),np.float32);y=np.zeros((B,MAX_HOPS,4),np.float32)
    mask=np.zeros((B,MAX_HOPS),np.float32)
    for b,(_,_,qq,yy,_) in enumerate(items):
        n=len(qq);q[b,:n]=qq;y[b,:n]=yy;mask[b,:n]=1
    return (torch.from_numpy(x),torch.from_numpy(v),torch.from_numpy(q),
            torch.from_numpy(y),torch.from_numpy(mask))


def count_params(model):return sum(p.numel() for p in model.parameters())


def _torch_model():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class TrainedDelta(nn.Module):
        def __init__(self):
            super().__init__()
            self.k=nn.Linear(8,DKEY,bias=False)
            self.q=nn.Linear(8,DKEY,bias=False)
            self.v=nn.Linear(4,DVAL,bias=False)
            self.out=nn.Linear(DVAL,DVAL,bias=False)
            self.beta=nn.Linear(8,1)
            self.forget=nn.Linear(8,1)
            self.loggain=nn.Parameter(torch.tensor(0.0))
            with torch.no_grad():
                self.k.weight.normal_(0,.025)
                self.k.weight[:8,:]=torch.eye(8)*.5
                self.q.weight.copy_(self.k.weight)
                self.v.weight.copy_(torch.eye(DVAL))
                self.out.weight.copy_(torch.eye(DVAL))
                self.beta.weight.zero_();self.beta.bias.fill_(3.)
                self.forget.weight.zero_();self.forget.bias.fill_(-7.)
        def forward(self,x,values,queries,*,initial=None,allowed=None):
            if x.ndim!=3 or x.shape[1:]!=(SLOTS,8):raise ValueError('INVALID_KEYS')
            if values.shape!=(len(x),SLOTS,DVAL):raise ValueError('INVALID_VALUES')
            if queries.ndim!=3 or queries.shape[0]!=len(x) or queries.shape[2]!=8:
                raise ValueError('INVALID_QUERIES')
            if x.dtype!=torch.float32 or values.dtype!=torch.float32 or queries.dtype!=torch.float32:
                raise ValueError('INVALID_DTYPE')
            if not (torch.isfinite(x).all() and torch.isfinite(values).all() and torch.isfinite(queries).all()):
                raise ValueError('NONFINITE_INPUT')
            if initial is not None and (initial.shape!=(len(x),DKEY,DVAL) or not torch.isfinite(initial).all()):
                raise ValueError('BAD_INITIAL_STATE')
            if allowed is not None and (allowed.shape!=(len(x),SLOTS) or allowed.dtype!=torch.bool):
                raise ValueError('BAD_ADMISSION_MASK')
            k=F.normalize(torch.tanh(2.0*self.k(x)),dim=-1,eps=1e-6)
            v=self.v(values)
            beta=torch.sigmoid(self.beta(x)).squeeze(-1)
            # g is nonpositive; using the FLA decay -> delta update order.
            g=-F.softplus(self.forget(x)).squeeze(-1)
            if allowed is not None:
                g=torch.where(allowed,g,torch.zeros_like(g))
                beta=beta*allowed.to(beta.dtype)
            s=torch.zeros((len(x),DKEY,DVAL),device=x.device,dtype=x.dtype) if initial is None else initial
            for t in range(SLOTS):
                s=s*torch.exp(g[:,t,None,None])
                err=v[:,t,:]-torch.bmm(k[:,t:t+1,:],s).squeeze(1)
                s=s+torch.bmm(k[:,t,:,None], (beta[:,t,None]*err)[:,None,:])
            q=F.normalize(torch.tanh(2.0*self.q(queries)),dim=-1,eps=1e-6)
            result=torch.einsum('bhd,bdv->bhv',q,s)
            return self.out(result)*torch.exp(self.loggain),s
    return TrainedDelta()


def new_model(seed=16):
    import torch
    torch.manual_seed(seed)
    return _torch_model()


def loss_fn(pred,target,mask):
    import torch
    from torch.nn import functional as F
    # Cosine objective independent of output magnitude, plus direct MSE.
    cos=1-F.cosine_similarity(pred,target,dim=-1,eps=1e-7)
    mag=((pred-target)**2).mean(-1)
    return ((cos+.2*mag)*mask).sum()/mask.sum()


def train(*,seed=16001,episodes=160,epochs=12,batch_size=16,shuffle_labels=False):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed&0xffffffff)
    data=make_data(seed*23+199,episodes,family='mixed')
    model=new_model(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.007,weight_decay=1e-4)
    rng=np.random.default_rng(seed+123)
    before=[]
    for j in range(0,len(data),batch_size):
        batch=batchify(data[j:j+batch_size])
        with torch.no_grad():before.append((float(loss_fn(model(*batch[:3])[0],batch[3],batch[4])),float(batch[4].sum())))
    initial=sum(a*b for a,b in before)/sum(b for _,b in before)
    for e in range(epochs):
        ordering=rng.permutation(len(data))
        for j in range(0,len(ordering),batch_size):
            batch=batchify([data[k] for k in ordering[j:j+batch_size]])
            if shuffle_labels:
                gen=torch.Generator().manual_seed(seed+e*777+j)
                batch=(batch[0],batch[1],batch[2],batch[3][torch.randperm(len(batch[3]),generator=gen)],batch[4])
            opt.zero_grad(set_to_none=True)
            pred,_=model(*batch[:3])
            loss=loss_fn(pred,batch[3],batch[4])
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE_TRAINING_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
    after=[]
    for j in range(0,len(data),batch_size):
        batch=batchify(data[j:j+batch_size])
        with torch.no_grad():after.append((float(loss_fn(model(*batch[:3])[0],batch[3],batch[4])),float(batch[4].sum())))
    final=sum(a*b for a,b in after)/sum(b for _,b in after)
    return model,dict(seed=seed,episodes=episodes,epochs=epochs,initial=initial,final=final,
                      parameters=count_params(model),state_bytes=DKEY*DVAL*4,
                      all_projections_trainable=True)


def teacher_forced(model,data):
    import torch
    stats=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(data),32):
            batch=batchify(data[i:i+32])
            pred,_=model(*batch[:3]);true=batch[3];mask=batch[4]
            p=pred.detach().cpu().numpy();t=true.numpy();m=mask.numpy()
            for b in range(len(p)):
                cs=[_cos(p[b,j],t[b,j]) for j in range(MAX_HOPS) if m[b,j]]
                stats.extend(cs)
    return float(np.mean(stats)),len(stats)


def _cos(x,y):
    n=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/n) if n>1e-12 else 0.


def autonomous(model,case,*,max_hops=4,threshold=0.78,key_threshold=.93):
    """No true path or family labels accessed. Uses only observed keys/values/goal."""
    import torch
    ep=case.episode
    if len(ep.keys)>8192 or max_hops>4:raise ValueError('CAP')
    # Identical input information as ERSA, pointer matching from admitted keys.
    if min(np.linalg.norm(ep.query_goal),np.linalg.norm(ep.query_key))<1e-12:
        return dict(status='HOLD_INVALID_QUERY',reads=0)
    x=np.concatenate([ep.keys,ep.goals],axis=1).astype(np.float32)
    values=np.asarray(ep.values,dtype=np.float32)
    seen=set();qkey=ep.query_key.copy()
    with torch.no_grad():
        for hop in range(1,max_hops+1):
            query=np.concatenate([qkey,ep.query_goal]).reshape(1,1,8).astype(np.float32)
            pred,_=model(torch.from_numpy(x[None]),torch.from_numpy(values[None]),torch.from_numpy(query))
            v=pred[0,0].numpy().astype(np.float64)
            if not np.isfinite(v).all():return dict(status='HOLD_NONFINITE',reads=hop)
            sims=np.array([_cos(v,k) for k in ep.keys]);ix=int(np.argmax(sims))
            if sims[ix]>=key_threshold:
                if ix in seen:return dict(status='HOLD_CYCLE',reads=hop)
                seen.add(ix);qkey=v;continue
            if _cos(v,ep.query_goal)>=threshold:
                return dict(status='STOP',reads=hop,vector=tuple(v))
            return dict(status='HOLD_LOW_CONFIDENCE',reads=hop)
    return dict(status='HOLD_HOP_BUDGET',reads=max_hops)


def independent_numpy_rule(k,v,b,g):
    """Separate NumPy matrix recurrence. Arrays are projected, pre-normalized."""
    k=np.asarray(k,np.float64);v=np.asarray(v,np.float64)
    b=np.asarray(b,np.float64);g=np.asarray(g,np.float64)
    if k.ndim!=2 or k.shape[1]!=DKEY or v.shape!=(len(k),DVAL):raise ValueError('BAD_SHAPE')
    if b.shape!=(len(k),) or g.shape!=(len(k),):raise ValueError('BAD_GATES')
    if not (np.isfinite(k).all() and np.isfinite(v).all() and np.isfinite(b).all() and np.isfinite(g).all()):
        raise ValueError('BAD_FINITE')
    if np.any(b<0) or np.any(b>1) or np.any(g>0):raise ValueError('BAD_GATE_RANGE')
    s=np.zeros((DKEY,DVAL),np.float64)
    for t in range(len(k)):
        decay=math.exp(float(g[t]))
        for j in range(DKEY):
            for z in range(DVAL):s[j,z]*=decay
        error=np.array([v[t,z]-sum(k[t,j]*s[j,z] for j in range(DKEY)) for z in range(DVAL)])
        for j in range(DKEY):
            for z in range(DVAL):s[j,z]+=b[t]*k[t,j]*error[z]
    return s

def projected_inputs(model,batch):
    import torch
    from torch.nn import functional as F
    x,v=batch[:2]
    with torch.no_grad():
        k=F.normalize(torch.tanh(2*model.k(x)),dim=-1).numpy()
        values=model.v(v).numpy()
        b=torch.sigmoid(model.beta(x)).squeeze(-1).numpy()
        g=(-F.softplus(model.forget(x))).squeeze(-1).numpy()
    return k,values,b,g
