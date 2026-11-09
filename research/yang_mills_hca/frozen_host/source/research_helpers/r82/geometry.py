"""Local SO(2)xSO(2) gauge action on node states AND directed transport links.

Any local node rotation requires U_dst<-src -> G_dst U_dst<-src G_src^-1.
State-only rotations with links frozen are *not* local gauge transformations.
"""
from __future__ import annotations
import math


def rotation(node:int, plane:int, angle:float, torch):
    if not (0 <= node < 4 and plane in (0,1)):
        raise ValueError('LOCAL_GENERATOR_RANGE')
    out=[]
    for idx in range(4):
        g=torch.eye(4,dtype=torch.float64)
        if idx==node:
            j=2*plane;c=math.cos(angle);s=math.sin(angle)
            g[j,j]=c;g[j,j+1]=-s;g[j+1,j]=s;g[j+1,j+1]=c
        out.append(g)
    return out


def transform_state(state, gs, torch):
    return tuple(torch.stack([torch.einsum('ij,bj->bi',gs[i],ch[:,i,:]) for i in range(4)],dim=1) for ch in state)


def transform_links(adj,gs,torch):
    result={}
    for dst,(src,ops) in adj.items():
        result[dst]=(src,torch.stack([gs[dst]@op@gs[int(si)].T for si,op in zip(src,ops)]))
    return result


def tangent(state,node,plane,torch):
    pieces=[]
    for ch in state:
        d=torch.zeros_like(ch)
        i=2*plane
        d[:,node,i]=-ch[:,node,i+1]
        d[:,node,i+1]=ch[:,node,i]
        pieces.append(d)
    return tuple(pieces)


def flatten(state,torch):
    return torch.cat([state[0].reshape(state[0].shape[0],16),state[1].reshape(state[1].shape[0],16)],dim=-1)


def deriv(plus,minus,eps,torch):
    return (flatten(plus,torch)-flatten(minus,torch))/(2*eps)


def maxnorm(z,torch):
    return float(z.norm(dim=-1).max())
