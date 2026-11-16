"""ERSA R12 — eight-weight offline-trained selector and bounded two-hop probe.

This is intentionally a *conditional neural utility probe* rather than a
complete recurrent network. C-owned SSC exact state supplies the vectors;
the host still supplies validated immutable metadata and admission authority.
All mutable inference state remains in volatile RAM. No checkpoint write.
"""
from __future__ import annotations
import ctypes as C
import hashlib, math, os, pathlib, shutil, subprocess, tempfile
from dataclasses import dataclass
from dataclasses import replace
import numpy as np
from native.ersa_native import NativeOps, build_native
from integration.ssc_native_r11 import NativeSscAuthorityR11, compile_r11
from elpis_ssc import contract_r0 as SC

N=24; K=1
FROZEN_WEIGHTS=np.array([1.,.5,.35,.4,-.6,.45,.3,.55],np.float64)


def unit(rng):
    a=rng.standard_normal(4)
    return a/np.linalg.norm(a)

def cosine(a,b):
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>1e-14 and nb>1e-14 else 0.

@dataclass
class Episode:
    keys:np.ndarray
    goals:np.ndarray
    values:np.ndarray
    meta:np.ndarray
    issued:np.ndarray
    source:int
    target:int
    query_key:np.ndarray
    query_goal:np.ndarray
    query_dir:np.ndarray


def make_episode(seed,*,regime='iid',n=N):
    if not 8<=n<=8192:raise ValueError('BAD_N')
    rng=np.random.default_rng(seed)
    keys=np.array([unit(rng) for _ in range(n)])
    goals=np.array([unit(rng) for _ in range(n)])
    values=np.array([unit(rng) for _ in range(n)])
    ids=rng.choice(n,4,replace=False);src,tgt,alias_src,alias_tgt=map(int,ids)
    goal=unit(rng)
    # Two aliases have nearly identical keys but opposite task goals. The
    # ground-truth selection is not available to the head at evaluation.
    keys[alias_src]=keys[src]+rng.normal(0,.006,4)
    keys[alias_tgt]=keys[tgt]+rng.normal(0,.006,4)
    goals[src]=goal;goals[tgt]=goal
    goals[alias_src]=-goal;goals[alias_tgt]=-goal
    values[src]=keys[tgt].copy() # source value is the second-hop pointer
    # The target contains the final answer; distractors contain arbitrary data.
    meta=np.zeros((n,4),np.float64)
    meta[:,0]=rng.uniform(.05,.95,n)
    meta[:,1]=rng.uniform(.1,.99,n)
    meta[:,2]=rng.choice([0.,-1.],n,p=[.75,.25])
    meta[:,3]=rng.choice([0.,1.],n,p=[.8,.2])
    # Metadata is IID of the identity labels. Otherwise the learned selector
    # can cheat by using utility or credibility as an oracle-label proxy.
    issued=rng.integers(0,32,n,dtype=np.int64)
    qnoise=.06 if regime!='hard_goal' else .17
    qkey=keys[src]+rng.normal(0,.05,4)
    qgoal=goal+rng.normal(0,qnoise,4)
    qdir=unit(rng)
    if regime=='hard_goal':
        goals[src]+=rng.normal(0,.15,4);goals[tgt]+=rng.normal(0,.15,4)
    if regime=='no_goal':
        goals[alias_src]=goals[src].copy();goals[alias_tgt]=goals[tgt].copy()
    if regime=='misleading_meta':
        meta[[alias_src,alias_tgt],0]=.99;meta[[alias_src,alias_tgt],1]=.99
        meta[[src,tgt],0]=.02;meta[[src,tgt],1]=.02
        meta[[alias_src,alias_tgt],2]=0;meta[[alias_src,alias_tgt],3]=1.
        meta[[src,tgt],2]=-1;meta[[src,tgt],3]=0.
    return Episode(keys,goals,values,meta,issued,src,tgt,qkey,qgoal,qdir)


def features_py(ep,qkey,qgoal,vals=None):
    v=ep.values if vals is None else vals
    out=np.empty((len(ep.keys),8),np.float64)
    for i in range(len(out)):
        out[i]=[cosine(ep.keys[i],qkey),ep.meta[i,0],
                math.exp(-(64-int(ep.issued[i]))/64),ep.meta[i,1],ep.meta[i,2],
                cosine(ep.goals[i],qgoal),ep.meta[i,3],cosine(v[i],ep.query_dir)]
    return out


def training_data(seed,n=480):
    x=[];y=[]
    for i in range(n):
        ep=make_episode(seed+i*7919)
        x.append(features_py(ep,ep.query_key,ep.query_goal));y.append(ep.source)
        x.append(features_py(ep,ep.values[ep.source],ep.query_goal));y.append(ep.target)
    return np.array(x),np.array(y,dtype=np.int64)


def train(seed=20261106,episodes=480,epochs=95,*,shuffle_labels=False):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed & 0xffffffff)
    x,y=training_data(seed,episodes)
    xx=torch.from_numpy(x);yy=torch.from_numpy(y)
    if shuffle_labels:
        # Negative control: labels permuted across queries, inputs unchanged.
        gen=torch.Generator().manual_seed(seed+19)
        yy=yy[torch.randperm(len(yy),generator=gen)]
    w=torch.nn.Parameter(torch.zeros(8,dtype=torch.float64))
    opt=torch.optim.Adam([w],lr=.09,weight_decay=.00015)
    initial=float(torch.nn.functional.cross_entropy(xx@w,yy).detach())
    for i in range(epochs):
        # Fixed, seeded permuted batches; no validation or test labels are used.
        perm=torch.randperm(len(xx))
        for chunk in perm.split(128):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(xx[chunk]@w,yy[chunk]);loss.backward();opt.step()
    final=float(torch.nn.functional.cross_entropy(xx@w,yy).detach())
    weights=w.detach().numpy().copy()
    if not np.isfinite(weights).all() or not final<initial:return weights,initial,final
    return weights,initial,final


def score_native_source():
    src=pathlib.Path(__file__).resolve().parents[1]/'native/ersa_selector_r12.c'
    directory=pathlib.Path(tempfile.mkdtemp(prefix='ersa12-scorer-',dir='/dev/shm'))
    target=directory/'libersa12.so'
    proc=subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',
          '-fno-fast-math','-fPIC','-shared',str(src),'-o',str(target)],
          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
    if proc.returncode:shutil.rmtree(directory);raise RuntimeError(proc.stderr)
    return target

class CSelector:
    def __init__(self,path):
        self.lib=C.CDLL(str(path));self.f=self.lib.ersa12_score
        self.f.restype=C.c_int;self.f.argtypes=[C.c_int64,C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_double)]
    def score(self,features,weights):
        x=np.ascontiguousarray(features,dtype=np.float64)
        w=np.ascontiguousarray(weights,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=8 or len(x)>8192 or w.shape!=(8,):raise ValueError('INVALID_SHAPE')
        if not np.isfinite(x).all() or not np.isfinite(w).all() or np.max(np.abs(w))>1e6:raise ValueError('INVALID_VALUES')
        out=np.full(len(x),np.nan,dtype=np.float64)
        rc=self.f(len(x),x.ctypes.data_as(C.POINTER(C.c_double)),w.ctypes.data_as(C.POINTER(C.c_double)),out.ctypes.data_as(C.POINTER(C.c_double)))
        if rc:raise RuntimeError('NATIVE_SCORE_REJECT:'+str(rc))
        return out

class NoTransport:
    def list_frames(self):return ()


def open_state(lib,ep):
    spec=replace(SC.SessionSpec(),max_active_entities=len(ep.keys),max_active_relations=len(ep.keys),
      max_ops_per_tick=1024,max_control_records=8)
    owner=NativeSscAuthorityR11(spec,SC.AdmissionRegistry([]),NoTransport(),lib)
    if owner.begin_session().status!='OK':owner.close();raise RuntimeError('BEGIN_FAILED')
    for i,v in enumerate(ep.values):
        o=owner.upsert_exact(1000+i,tuple(float(z) for z in v),'fixture:write:r0')
        if o.status!='OK':owner.close();raise RuntimeError('WRITE_FAILED:'+str(o))
    return owner


def read_values(owner,n):
    values=[]
    for i in range(n):
        o=owner.read_exact(1000+i)
        if o.status!='OK':raise RuntimeError('NATIVE_EXACT_READ_FAILED')
        values.append(o.detail)
    return np.asarray(values,dtype=np.float64)


def native_features(ops,ep,qk,qg,values):
    return ops.features(ep.keys,ep.goals,values,ep.meta,ep.issued,64,64.,qk,qg,ep.query_dir)


def two_hop(owner,ops,selector,ep,weights,*,allowed=None,step_limit=2,
            min_margin=0.,inject=None):
    """Host-coordinated two-step recurrent read, no oracle labels in path.

    The caller supplies a pre-authorized mask. Generation check rejects any
    mutation injected between the exact-value materialization and ranking.
    """
    n=len(ep.keys)
    if step_limit!=2:return {'status':'HOLD_STEP_BUDGET','ids':()}
    if allowed is None:allowed=np.ones(n,dtype=np.uint8)
    if len(allowed)!=n or any(type(int(z)) is not int or z not in (0,1) for z in allowed):
        return {'status':'HOLD_AUTHORITY_MASK','ids':()}
    q=ep.query_key.copy();ids=[]
    try:
        for hop in range(2):
            vals=read_values(owner,n)
            gen=owner.generation
            feat=native_features(ops,ep,q,ep.query_goal,vals)
            scores=selector.score(feat,weights)
            order=ops.rank(np.round(scores,12),np.arange(n,dtype=np.int64),allowed,2)
            if inject is not None:inject(owner,hop)
            if gen!=owner.generation:return {'status':'HOLD_STALE_GENERATION','ids':tuple(ids)}
            if not order:return {'status':'HOLD_NO_ADMITTED_CANDIDATE','ids':tuple(ids)}
            if (len(order)>1 and scores[order[0]]-scores[order[1]]<min_margin):
                return {'status':'HOLD_AMBIGUOUS','ids':tuple(ids)}
            idx=order[0]
            if not allowed[idx]:return {'status':'HOLD_UNAUTHORIZED_ID','ids':tuple(ids)}
            ids.append(idx)
            if hop==0:q=vals[idx].copy()
            else:return {'status':'OK','ids':tuple(ids),'answer':tuple(vals[idx]),'steps':2}
    except (ValueError,RuntimeError,TypeError):
        return {'status':'HOLD_INVALID_NATIVE_READ','ids':tuple(ids)}
    return {'status':'HOLD_UNREACHABLE','ids':tuple(ids)}


def evaluate(weights,*,selector,ops,owner_lib,seed,count,regime='iid',baselines=None,min_margin=0.,normalized_margin=0.):
    allweights={'learned':weights}
    allweights.update(baselines or {})
    stats={name:dict(success=0,source=0,target=0,abstain=0,fail=0) for name in allweights}
    for j in range(count):
        ep=make_episode(seed+j*104729,regime=regime)
        with open_state(owner_lib,ep) as owner:
            # All competing controllers see precisely the same C state episode.
            for name,w in allweights.items():
                margin=(normalized_margin*float(np.linalg.norm(w)) if normalized_margin else
                        (min_margin if name=='learned' else 0.))
                got=two_hop(owner,ops,selector,ep,w,min_margin=margin)
                s=stats[name]
                if got['status']=='OK':
                    a,b=got['ids'];s['source']+=int(a==ep.source)
                    s['target']+=int(b==ep.target)
                    s['success']+=int(a==ep.source and b==ep.target and
                        np.allclose(got['answer'],ep.values[ep.target],atol=0,rtol=0))
                else:s['abstain']+=1
    return stats


def cleanup_compilations(*libraries):
    for lib in libraries:
        p=pathlib.Path(lib)
        if p.parent.is_relative_to('/dev/shm'):shutil.rmtree(p.parent,ignore_errors=True)
