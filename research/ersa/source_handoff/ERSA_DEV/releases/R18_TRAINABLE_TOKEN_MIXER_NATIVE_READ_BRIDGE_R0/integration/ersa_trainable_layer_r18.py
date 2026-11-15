"""R18: differentiable, bounded read-only ERSA token mixer (PyTorch CPU reference).

Source authority remains the frozen R11 native SSC bank; this layer accepts an
explicit, host-authorized COPY of relation metadata and values. It is not an
authority to admit new records or mutate the C bank. Soft routing is trainable;
`hard` mode is a diagnostic and not a substitute for R14's native loop.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ERSAState:
    keys: torch.Tensor        # B,N,D
    goals: torch.Tensor       # B,N,D
    values: torch.Tensor      # B,N,D
    metadata: torch.Tensor    # B,N,4 [utility,credibility,polarity,protected]
    issued: torch.Tensor      # B,N integer timestamps; 0 <= issued <= tick
    allowed: torch.Tensor     # B,N BOOL admission mask, from host authority
    tick: int = 64
    generation: int = 0


class ERSATokenMixer(nn.Module):
    """(hidden[B,T,D], state, optional diagnostics) from (query[B,T,D], state).

    External `goal` and `direction` default to query for convenience. The state
    is returned untouched and cannot be modified in forward. A `read_only`
    external snapshot is required; no implicit heap or persistent KV exists.
    """
    def __init__(self, d_model=8, *, max_slots=64, max_hops=4,
                 temperature=12.0, max_work_elements=2_000_000):
        super().__init__()
        if not (4<=d_model<=32 and 1<=max_slots<=8192 and 1<=max_hops<=16
                and 0<temperature<=64 and max_work_elements>=1):
            raise ValueError('INVALID_LAYER_BOUNDS')
        self.d_model=d_model
        self.max_slots=max_slots
        self.max_hops=max_hops
        self.temperature=float(temperature)
        self.max_work_elements=int(max_work_elements)
        # Same eight feature coordinates as frozen R7/R12 native scorer.
        self.selection=nn.Parameter(torch.tensor([.35,0,0,0,0,.35,0,0],dtype=torch.float32))
        # R13-compatible five halt features and one bias (not frozen R13 weights).
        self.halt=nn.Parameter(torch.tensor([.1,-3.,1.,0.,0.,1.],dtype=torch.float32))
        self.projection=nn.Linear(d_model,d_model,bias=False)
        nn.init.eye_(self.projection.weight)
        self.mix_logit=nn.Parameter(torch.tensor(-.2,dtype=torch.float32))

    def _validate(self,x,state,goal,direction):
        if not isinstance(state,ERSAState) or x.ndim!=3 or x.dtype not in (torch.float32,torch.float64):
            raise ValueError('INVALID_LAYER_INPUT')
        b,t,d=x.shape
        if d!=self.d_model or b<1 or t<1:raise ValueError('INVALID_QUERY_SHAPE')
        n=state.keys.shape[1] if state.keys.ndim==3 else -1
        if not (1<=n<=self.max_slots) or b*t*n*d*self.max_hops>self.max_work_elements:
            raise ValueError('BOUNDED_WORK_EXCEEDED')
        if not isinstance(state.tick,int) or not 0<=state.tick<(1<<53) or not isinstance(state.generation,int):
            raise ValueError('INVALID_STATE_VERSION')
        for a,shape in ((state.keys,(b,n,d)),(state.goals,(b,n,d)),
                        (state.values,(b,n,d)),(state.metadata,(b,n,4)),
                        (state.issued,(b,n)),(state.allowed,(b,n)),
                        (goal,(b,t,d)),(direction,(b,t,d))):
            if a.shape!=shape or a.device!=x.device:
                raise ValueError('INVALID_STATE_SHAPE_OR_DEVICE')
        for a in (state.keys,state.goals,state.values,state.metadata,goal,direction):
            if a.dtype!=x.dtype or not bool(torch.isfinite(a).all()):
                raise ValueError('INVALID_STATE_FLOAT_VALUES')
        if not bool(torch.isfinite(x).all()) or state.allowed.dtype!=torch.bool or state.issued.dtype not in (torch.int32,torch.int64):
            raise ValueError('INVALID_STATE_TYPES')
        if not bool(((state.issued>=0)&(state.issued<=state.tick)).all()):
            raise ValueError('INVALID_ISSUED_TIME')
        if not bool(((state.metadata[...,2]==-1)|(state.metadata[...,2]==0)).all()) or not bool(((state.metadata[...,3]==0)|(state.metadata[...,3]==1)).all()):
            raise ValueError('INVALID_METADATA_FLAGS')
        if not bool(torch.isfinite(self.selection).all()) or not bool(torch.isfinite(self.halt).all()):
            raise ValueError('INVALID_PARAMETERS')

    @staticmethod
    def _cos(a,b):
        # R7 cosine uses zero if either norm is negligible.
        numerator=(a*b).sum(dim=-1)
        denominator=torch.linalg.vector_norm(a,dim=-1)*torch.linalg.vector_norm(b,dim=-1)
        return torch.where(denominator>1.e-12,numerator/denominator.clamp_min(1.e-12),torch.zeros_like(numerator))

    def forward(self,x:torch.Tensor,state:ERSAState,*,goal=None,direction=None,
                mode='soft',return_aux=False):
        if mode not in ('soft','hard'):raise ValueError('INVALID_ROUTING_MODE')
        if goal is None:goal=x
        if direction is None:direction=goal
        self._validate(x,state,goal,direction)
        b,t,d=x.shape;n=state.keys.shape[1]
        mask=state.allowed[:,None,:].expand(b,t,n)
        valid=mask.any(dim=-1)
        # Mask BEFORE similarities, so disallowed rows never enter scoring,
        # max-pointer detection, feature normalization or read mixtures.
        keys=torch.where(state.allowed[...,None],state.keys,torch.zeros_like(state.keys))[:,None,:,:]
        goals=torch.where(state.allowed[...,None],state.goals,torch.zeros_like(state.goals))[:,None,:,:]
        values=torch.where(state.allowed[...,None],state.values,torch.zeros_like(state.values))[:,None,:,:]
        meta=torch.where(state.allowed[...,None],state.metadata,torch.zeros_like(state.metadata))[:,None,:,:]
        age=torch.exp(-(state.tick-state.issued).to(dtype=x.dtype)/64.0)[:,None,:]
        q=x
        aggregate=torch.zeros_like(x)
        survival=torch.ones((b,t),device=x.device,dtype=x.dtype)
        slot_history=[];stop_history=[];paths=[];selected_values=[];feature_history=[];score_history=[]
        for step in range(1,self.max_hops+1):
            ks=self._cos(keys,q[:,:,None,:]);gs=self._cos(goals,goal[:,:,None,:])
            ds=self._cos(values,direction[:,:,None,:])
            features=torch.stack((ks,meta[...,0].expand(b,t,n),age.expand(b,t,n),
                 meta[...,1].expand(b,t,n),meta[...,2].expand(b,t,n),gs,
                 meta[...,3].expand(b,t,n),ds),dim=-1)
            score=features@self.selection.to(dtype=x.dtype)
            feature_history.append(features)
            score_history.append(score)
            masked=score.masked_fill(~mask,-1.e9)
            if mode=='hard':
                index=masked.argmax(dim=-1)
                prob=F.one_hot(index,n).to(dtype=x.dtype)*valid[...,None]
            else:
                prob=torch.softmax(self.temperature*masked,dim=-1)
                prob=prob*valid[...,None]
            read=torch.sum(prob[...,None]*values,dim=-2)
            masked_top=torch.topk(masked,k=min(n,2),dim=-1).values
            margin=(masked_top[...,0]-masked_top[...,-1]).clamp(min=0,max=2) / self.selection.norm().clamp_min(1.e-6)
            # Feature 1: max similarity to any authorized exact key.
            max_key=self._cos(read[:,:,None,:],keys).masked_fill(~mask,-1).amax(dim=-1)
            sim_goal=self._cos(read,goal)
            stopfeat=torch.stack((sim_goal,max_key,sim_goal-max_key,
                torch.full_like(sim_goal,step/self.max_hops),margin),dim=-1)
            logit=stopfeat@self.halt[:5].to(dtype=x.dtype)+self.halt[5].to(dtype=x.dtype)
            p_stop=torch.sigmoid(logit)
            if mode=='hard':
                # Exact observed pointer can never be a terminal answer.
                pointer=max_key>=.999999
                p_stop=((p_stop>=.5)&(~pointer)&valid).to(x.dtype)
            else:
                # Smooth differentiable approximation to the pointer veto.
                pointer_prob=torch.sigmoid(80.*(max_key-.98))
                p_stop=p_stop*(1.-pointer_prob)*valid
            this=survival*p_stop
            aggregate=aggregate+this[...,None]*read
            survival=survival*(1.-p_stop)
            q=read
            slot_history.append(prob)
            stop_history.append(p_stop)
            paths.append(torch.where(valid,masked.argmax(dim=-1),-1))
            selected_values.append(read)
        confidence=(1.-survival)*valid
        mixed=torch.sigmoid(self.mix_logit)*self.projection(aggregate)
        y=x+torch.where(valid[...,None],mixed,torch.zeros_like(mixed))
        if not return_aux:return y,state
        aux={'read':aggregate,'confidence':confidence,'slot_probs':torch.stack(slot_history,dim=2),
             'stop_probs':torch.stack(stop_history,dim=2),
             'path_indices':torch.stack(paths,dim=2),'steps':self.max_hops,
             'survival':survival,'generation':state.generation,
             'selected_reads':torch.stack(selected_values,dim=2),
             'features':torch.stack(feature_history,dim=2),
             'scores':torch.stack(score_history,dim=2)}
        return y,state,aux
