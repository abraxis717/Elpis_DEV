"""R13: learned bounded stop/continue/abstain over actual frozen ERSA/SSC C-owned RAM.

Synthetic variable-hop pointer chain; all oracle path indices used exclusively
for offline training and evaluation scoring, never as controller inputs. Entire
trained head is frozen for inference; no runtime training or persistence.
"""
from __future__ import annotations
import ctypes, dataclasses, math, pathlib, shutil, subprocess, tempfile
import numpy as np
from integration.ersa_selector_r12 import (
    make_episode, cosine, score_native_source, CSelector, open_state,
    native_features, read_values, train as train_selector, cleanup_compilations,
)
from native.ersa_native import NativeOps, build_native
from integration.ssc_native_r11 import compile_r11

MAX_HOPS=4
KEY_GOAL=np.array([1.,0.,0.,0.,0.,1.,0.,0.],dtype=np.float64)

@dataclasses.dataclass
class ChainCase:
    episode:object
    path:tuple[int,...]
    terminal:bool
    family:str


def make_chain(seed:int, *, family='positive', hops:int|None=None):
    if family not in ('positive','no_terminal','hard_pointer','missing_goal'):
        raise ValueError('INVALID_FAMILY')
    rng=np.random.default_rng(seed)
    ep=make_episode(seed+41719,regime='iid')
    if hops is None:hops=int(rng.integers(1,MAX_HOPS+1))
    if not 1<=hops<=MAX_HOPS:raise ValueError('INVALID_HOPS')
    src,tgt=ep.source,ep.target
    # Reserve the R12 near-key aliases; avoid accidental aliasing in added hops.
    pool=[i for i in range(len(ep.keys)) if i not in (src,tgt) and
        cosine(ep.keys[i],ep.keys[src])<.97 and
        cosine(ep.keys[i],ep.keys[tgt])<.97 and
        cosine(ep.keys[i],ep.query_goal)<.70]
    if len(pool)<MAX_HOPS:raise RuntimeError('INSUFFICIENT_CHAINS')
    extras=list(map(int,rng.choice(pool,MAX_HOPS-2,replace=False)))
    path=(src,) if hops==1 else (src,*extras[:hops-2],tgt)
    core=ep.goals[src].copy()
    for idx in path:ep.goals[idx]=core
    if family=='missing_goal':
        # This query lacks a trustworthy discriminating goal; a hard case.
        ep.goals[:]=core
    for a,b in zip(path[:-1],path[1:]):ep.values[a]=ep.keys[b].copy()
    if family=='hard_pointer' and len(path)>1:
        # Deliberately make one valid pointer look like the answer direction.
        b=path[-1]
        ep.keys[b]=ep.query_goal/np.linalg.norm(ep.query_goal)
        ep.values[path[-2]]=ep.keys[b].copy()
    if family=='no_terminal':
        ep.values[path[-1]]=ep.keys[path[0]].copy()  # observable cycle
    else:
        ep.values[path[-1]]=ep.query_goal/np.linalg.norm(ep.query_goal)+rng.normal(0,.04,4)
    return ChainCase(ep,tuple(path),family!='no_terminal',family)


def halt_features(value,ep,step, margin):
    v=np.asarray(value,dtype=np.float64)
    simgoal=cosine(v,ep.query_goal)
    maxkey=max(cosine(v,k) for k in ep.keys)
    return np.array([simgoal,maxkey,simgoal-maxkey,step/MAX_HOPS,
                     min(max(float(margin),0.),2.)],dtype=np.float64)


def train_halt(seed=60121, episodes=640, epochs=100, *,shuffle_labels=False):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x=[];y=[]
    for i in range(episodes):
        family='no_terminal' if i%5==0 else 'positive'
        case=make_chain(seed+i*7919,family=family)
        for step,idx in enumerate(case.path,1):
            # Offline oracle supplies path ONLY when constructing supervised data.
            v=case.episode.values[idx]
            # Stop input contains only read value and admitted key metadata.
            x.append(halt_features(v,case.episode,step,margin=.5))
            y.append(int(case.terminal and step==len(case.path)))
    xx=torch.tensor(np.asarray(x),dtype=torch.float64)
    yy=torch.tensor(y,dtype=torch.float64)
    if shuffle_labels:
        gen=torch.Generator().manual_seed(seed+73)
        yy=yy[torch.randperm(len(yy),generator=gen)]
    w=torch.nn.Parameter(torch.zeros(6,dtype=torch.float64))
    opt=torch.optim.Adam([w],lr=.055,weight_decay=1.e-5)
    loss0=float(torch.nn.functional.binary_cross_entropy_with_logits(xx[:,:5]@w[:5]+w[5],yy).detach())
    for _ in range(epochs):
        ix=torch.randperm(len(yy))
        for sub in ix.split(256):
            opt.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(xx[sub,:5]@w[:5]+w[5],yy[sub])
            loss.backward();opt.step()
    loss1=float(torch.nn.functional.binary_cross_entropy_with_logits(xx[:,:5]@w[:5]+w[5],yy).detach())
    return w.detach().numpy().copy(),loss0,loss1


def compile_halt():
    src=pathlib.Path(__file__).resolve().parents[1]/'native/ersa_halt_r13.c'
    d=pathlib.Path(tempfile.mkdtemp(dir='/dev/shm',prefix='ersa13-halt-'))
    so=d/'libersa13.so'
    p=subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',
      '-fno-fast-math','-fPIC','-shared',str(src),'-o',str(so),'-lm'],
      capture_output=True,text=True,timeout=30)
    if p.returncode:
        shutil.rmtree(d);raise RuntimeError('C_BUILD_FAILED:'+p.stderr)
    return so

class CHalt:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path));self.f=self.lib.ersa13_halt
        P=ctypes.POINTER(ctypes.c_double)
        self.f.argtypes=[P,P,ctypes.c_int,ctypes.c_int,P,ctypes.POINTER(ctypes.c_int)]
        self.f.restype=ctypes.c_int
    def action(self,features,weights,step,max_steps=MAX_HOPS):
        feat=np.ascontiguousarray(features,dtype=np.float64)
        ww=np.ascontiguousarray(weights,dtype=np.float64)
        if feat.shape!=(5,) or ww.shape!=(6,):raise ValueError('INVALID_HEAD_SHAPE')
        prob=ctypes.c_double(-1.);act=ctypes.c_int(-1)
        rc=self.f(feat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  ww.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  step,max_steps,ctypes.byref(prob),ctypes.byref(act))
        if rc:raise ValueError('INVALID_NATIVE_HALT:'+str(rc))
        return act.value,prob.value


def deterministic_action(feat,step,kind):
    if kind=='strong':
        return 1 if feat[0]>=.80 else (2 if step==MAX_HOPS else 0)
    if kind=='fixed2':
        return 1 if step==2 else (2 if step==MAX_HOPS else 0)
    raise ValueError('UNKNOWN_RULE')


def execute(owner,ops,selector,halter,case,selector_w,halt_w=None,*,policy='learned',
            allowed=None,inject=None,margin_norm=.015):
    ep=case.episode;n=len(ep.keys)
    if allowed is None:allowed=np.ones(n,dtype=np.uint8)
    if len(allowed)!=n or any(v not in (0,1) for v in allowed):return {'status':'HOLD_AUTHORITY'}
    qkey=ep.query_key.copy(); path=[]
    try:
        for step in range(1,MAX_HOPS+1):
            vals=read_values(owner,n)
            generation=owner.generation
            features=native_features(ops,ep,qkey,ep.query_goal,vals)
            scores=selector.score(features,selector_w)
            ordered=ops.rank(np.round(scores,12),np.arange(n,dtype=np.int64),
                             np.asarray(allowed,dtype=np.uint8),2)
            if inject is not None:inject(owner,step)
            if generation!=owner.generation:return {'status':'HOLD_STALE_GENERATION','path':tuple(path)}
            if not ordered:return {'status':'HOLD_NO_CANDIDATE','path':tuple(path)}
            gap=float(scores[ordered[0]]-scores[ordered[1]]) if len(ordered)>1 else float('inf')
            if gap<margin_norm*np.linalg.norm(selector_w):
                return {'status':'HOLD_AMBIGUOUS','path':tuple(path)}
            index=int(ordered[0])
            if not allowed[index]:return {'status':'HOLD_UNAUTHORIZED','path':tuple(path)}
            path.append(index)
            hf=halt_features(vals[index],ep,step,gap/np.linalg.norm(selector_w))
            if policy=='learned':action,_=halter.action(hf,halt_w,step)
            else:action=deterministic_action(hf,step,policy)
            # An exact admitted-key match is an explicit pointer, NOT evidence of
            # a terminal answer. This veto is an observable safety constraint,
            # not an oracle label. It also protects against near-goal pointers.
            if action==1 and hf[1]>=.999999:
                action=0 if step<MAX_HOPS else 2
            if action==1:
                return {'status':'OK','path':tuple(path),'answer':tuple(vals[index]),'steps':step}
            if action==2:return {'status':'HOLD_UNCERTAIN_STOP','path':tuple(path)}
            qkey=vals[index].copy()
    except (ValueError,RuntimeError,TypeError,OverflowError):
        return {'status':'HOLD_INVALID_NATIVE','path':tuple(path)}
    return {'status':'HOLD_BUDGET','path':tuple(path)}


def evaluate(selector_w,halt_w,*,ops,selector,halter,state_lib,seed,count,family):
    policies=('learned','keygoal_learned_halt','strong','fixed2')
    out={p:dict(success=0,false_stop=0,abstain=0,reads=0) for p in policies}
    for i in range(count):
        case=make_chain(seed+i*104729,family=family)
        with open_state(state_lib,case.episode) as owner:
            for p in policies:
                use_w=selector_w if p=='learned' else KEY_GOAL
                use_policy='learned' if p=='keygoal_learned_halt' else p
                got=execute(owner,ops,selector,halter,case,use_w,halt_w,policy=use_policy)
                stat=out[p];stat['reads']+=len(got.get('path',()))
                if got['status']=='OK':
                    success=(case.terminal and tuple(got['path'])==case.path and
                             got.get('answer')==tuple(case.episode.values[case.path[-1]]))
                    stat['success']+=int(success)
                    stat['false_stop']+=int(not success)
                else:stat['abstain']+=1
    return out
