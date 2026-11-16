# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
# Based on fla-org/flash-linear-attention/fla/ops/gsa/naive.py,
# MIT license. Reference implementation adapted solely to remove einops dependency.
# Original: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gsa/naive.py
# This adaptation retains the two-stage GSA recurrence and return-state contract.
import torch


def naive_recurrent_gsa(q, k, v, s, g, *, initial_state=None,
                        output_final_state=True, scale=None):
    """FLA GSA naive recurrent reference, explicit g, one or more heads.

    q,k: [B,T,H,K], v: [B,T,H,V], s,g: [B,T,H,M].
    state: (hk[B,H,K,M], hv[B,H,M,V]).
    """
    if g is None:
        raise ValueError('explicit sparse gate required')
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError('bad GSA input shapes')
    if s.shape != g.shape or s.shape[:3] != q.shape[:3]:
        raise ValueError('bad slot gate shapes')
    b, t, h, d = q.shape
    m, vd = s.shape[-1], v.shape[-1]
    orig_dtype = q.dtype
    q, k, v, s, g = [a.transpose(1,2).contiguous().float() for a in (q,k,v,s,g)]
    if scale is None:
        scale = d ** -0.5
    if initial_state is None:
        hk = torch.zeros(b,h,d,m,device=q.device,dtype=torch.float32)
        hv = torch.zeros(b,h,m,vd,device=q.device,dtype=torch.float32)
    else:
        hk,hv = initial_state
        if hk.shape != (b,h,d,m) or hv.shape != (b,h,m,vd):
            raise ValueError('bad GSA cache shape')
        hk,hv = hk.float(), hv.float()
    scores=[]
    for i in range(t):
        gi=g[:,:,i].exp()
        hk=hk*gi[...,None,:]+k[:,:,i,:][...,None]*s[:,:,i,:][...,None,:]
        scores.append((q[:,:,i,:][...,None]*hk).sum(-2)*scale)
    weights=torch.stack(scores,dim=2).softmax(-1)
    outs=[]
    for i in range(t):
        gi=g[:,:,i].exp()
        hv=hv*gi[..., :,None]+s[:,:,i,:][...,None]*v[:,:,i,:][...,None,:]
        outs.append((weights[:,:,i,:][...,None]*hv).sum(-2))
    output=torch.stack(outs,dim=2).transpose(1,2).contiguous().to(orig_dtype)
    return output, (hk,hv) if output_final_state else None
