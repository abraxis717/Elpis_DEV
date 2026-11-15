"""R15 CPU baseline: mathematical gated-delta rule, not a GDN neural layer.

Implements the recurrence documented by MIT-licensed FLA project's
fla/ops/gated_delta_rule/naive.py (external source not redistributed).
State S[k,v] <- exp(g) * S[k,v];
S <- S + beta * outer(key, value - key@S).
Uses precomputed q/k/v, not trainable projections, convolutions or norms.
"""
from __future__ import annotations
import math
import numpy as np

KDIM=8; VDIM=4

def _valid_arr(x,shape):
    a=np.asarray(x)
    if a.shape!=shape or a.dtype.kind not in 'f' or not np.isfinite(a).all():
        raise ValueError('INVALID_GDN_INPUT')
    return np.asarray(a,dtype=np.float64)

def recurrence_numpy(keys,values,betas,logdecays,initial=None):
    """Independent scalar-index NumPy reference, output exact64 state."""
    keys=np.asarray(keys,dtype=np.float64); values=np.asarray(values,dtype=np.float64)
    if keys.ndim!=2 or keys.shape[1]!=KDIM or values.shape!=(len(keys),VDIM):
        raise ValueError('INVALID_SHAPE')
    n=len(keys)
    if n<1 or n>8192:raise ValueError('CAPACITY')
    keys=_valid_arr(keys,(n,KDIM));values=_valid_arr(values,(n,VDIM))
    betas=_valid_arr(betas,(n,));logdecays=_valid_arr(logdecays,(n,))
    if np.any(betas<0) or np.any(betas>1) or np.any(logdecays>0) or np.any(logdecays < -60):
        raise ValueError('INVALID_GATES')
    s=np.zeros((KDIM,VDIM),dtype=np.float64) if initial is None else _valid_arr(initial,(KDIM,VDIM)).copy()
    for t in range(n):
        alpha=math.exp(float(logdecays[t]))
        for i in range(KDIM):
            for j in range(VDIM):s[i,j]*=alpha
        predicted=np.array([sum(keys[t,i]*s[i,j] for i in range(KDIM)) for j in range(VDIM)])
        for i in range(KDIM):
            for j in range(VDIM):s[i,j]+=betas[t]*keys[t,i]*(values[t,j]-predicted[j])
    if not np.isfinite(s).all():raise ValueError('NONFINITE_STATE')
    return s

def recurrence_torch(keys,values,betas,logdecays,initial=None):
    """Torch sequential gated-delta rule using FLA ordering of decay+delta."""
    import torch
    keys=np.asarray(keys,dtype=np.float64); values=np.asarray(values,dtype=np.float64)
    n=len(keys)
    if keys.shape!=(n,KDIM) or values.shape!=(n,VDIM):raise ValueError('INVALID_SHAPE')
    _valid_arr(keys,(n,KDIM));_valid_arr(values,(n,VDIM))
    _valid_arr(betas,(n,));_valid_arr(logdecays,(n,))
    if np.any(betas<0) or np.any(betas>1) or np.any(logdecays>0) or np.any(logdecays<-60):
        raise ValueError('INVALID_GATES')
    k=torch.as_tensor(keys,dtype=torch.float64)
    v=torch.as_tensor(values,dtype=torch.float64)
    b=torch.as_tensor(betas,dtype=torch.float64)
    g=torch.as_tensor(logdecays,dtype=torch.float64)
    s=torch.zeros((KDIM,VDIM),dtype=torch.float64) if initial is None else torch.as_tensor(_valid_arr(initial,(KDIM,VDIM)).copy())
    with torch.no_grad():
        for t in range(n):
            s=s*torch.exp(g[t])
            err=v[t]-k[t]@s
            s=s+torch.outer(k[t],b[t]*err)
    out=s.numpy().copy()
    if not np.isfinite(out).all():raise ValueError('NONFINITE_STATE')
    return out

def packed_keys(ep):
    """Untrained transparent content+goal feature map. Not an ERSA scorer."""
    k=np.asarray(ep.keys,np.float64);g=np.asarray(ep.goals,np.float64)
    if k.ndim!=2 or k.shape[1]!=4 or g.shape!=k.shape:raise ValueError('INVALID_EPISODE')
    kn=k/np.maximum(1e-12,np.linalg.norm(k,axis=1,keepdims=True))
    gn=g/np.maximum(1e-12,np.linalg.norm(g,axis=1,keepdims=True))
    return np.concatenate((kn,gn),axis=1)/math.sqrt(2.)

def query(qkey,qgoal):
    k=np.asarray(qkey,np.float64);g=np.asarray(qgoal,np.float64)
    if k.shape!=(4,) or g.shape!=(4,) or not np.isfinite(k).all() or not np.isfinite(g).all():
        raise ValueError('INVALID_QUERY')
    kn=k/max(1e-12,np.linalg.norm(k));gn=g/max(1e-12,np.linalg.norm(g))
    return np.concatenate((kn,gn))/math.sqrt(2.)

def state_for_episode(ep,*,beta=1.0,decay=1.0,mask=None,method='numpy'):
    n=len(ep.keys)
    if mask is None:mask=np.ones(n,dtype=np.bool_)
    m=np.asarray(mask)
    if m.shape!=(n,) or m.dtype.kind not in 'biu' or not np.isin(m,[0,1]).all():
        raise ValueError('INVALID_ADMISSION')
    if not 0<=beta<=1 or not 0<decay<=1:raise ValueError('INVALID_HYPERPARAMS')
    k=packed_keys(ep)[m.astype(bool)];v=np.asarray(ep.values,np.float64)[m.astype(bool)]
    if len(k)<1:raise ValueError('NO_ADMITTED_MEMORY')
    b=np.full(len(k),beta,np.float64);g=np.full(len(k),math.log(decay),np.float64)
    if method=='numpy':return recurrence_numpy(k,v,b,g)
    if method=='torch':return recurrence_torch(k,v,b,g)
    raise ValueError('INVALID_METHOD')

def read(s,qkey,qgoal):
    s=_valid_arr(s,(KDIM,VDIM));q=query(qkey,qgoal)
    return np.asarray(q@s,dtype=np.float64)

def cosine(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1.e-12 else 0.
