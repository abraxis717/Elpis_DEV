"""R18: train and evaluate a small neural model containing ERSA token mixing.
No test labels enter the controller. No checkpoint writes. Isolated CPU training.
"""
from __future__ import annotations
import hashlib, json, math
import torch
from torch import nn
from torch.nn import functional as F
from integration.ersa_trainable_layer_r18 import ERSAState, ERSATokenMixer

D=8
N=12
CLASSES=4
CODE=F.normalize(torch.tensor([[1.,0,0,0,0,0,0,0], [0,1.,0,0,0,0,0,0],
 [0,0,1.,0,0,0,0,0], [0,0,0,1.,0,0,0,0]],dtype=torch.float32),dim=-1)


def batch(seed,count,*,two_hop=True):
    gen=torch.Generator().manual_seed(seed)
    keys=F.normalize(torch.randn(count,N,D,generator=gen),dim=-1)
    goals=F.normalize(torch.randn(count,N,D,generator=gen),dim=-1)
    labels=torch.randint(0,CLASSES,(count,N),generator=gen)
    vals=CODE[labels].clone()
    source=torch.randint(0,N,(count,),generator=gen)
    target=torch.randint(0,N-1,(count,),generator=gen)
    target=target+(target>=source).long()
    row=torch.arange(count)
    # Identical-key decoy with an explicitly contradictory goal.
    decoy=(source+1)%N
    keys[row,decoy]=keys[row,source]+.002*torch.randn(count,D,generator=gen)
    keys[row,decoy]=F.normalize(keys[row,decoy],dim=-1)
    goals[row,decoy]=-goals[row,source]
    if two_hop:
        goals[row,target]=goals[row,source]
        vals[row,source]=keys[row,target]
    query=keys[row,source]+.02*torch.randn(count,D,generator=gen)
    query=F.normalize(query,dim=-1).unsqueeze(1)
    goal=F.normalize(goals[row,source]+.02*torch.randn(count,D,generator=gen),dim=-1).unsqueeze(1)
    target_labels=(labels[row,target] if two_hop else labels[row,source])
    path=torch.stack((source,target),dim=1) if two_hop else source[:,None]
    meta=torch.zeros(count,N,4)
    issued=torch.zeros(count,N,dtype=torch.int64)
    allowed=torch.ones(count,N,dtype=torch.bool)
    state=ERSAState(keys,goals,vals,meta,issued,allowed)
    return query,goal,state,target_labels,path


class TinyNeuralERSAModel(nn.Module):
    def __init__(self,hops):
        super().__init__()
        self.ersa=ERSATokenMixer(D,max_slots=N,max_hops=hops,temperature=12.)
        self.output=nn.Linear(D,CLASSES)
    def forward(self,query,goal,state):
        y,unchanged,aux=self.ersa(query,state,goal=goal,mode='soft',return_aux=True)
        # Explicitly select the ERSA residual: predictions require memory reads.
        logits=self.output(y-query)
        return logits.squeeze(1),unchanged,aux


def deterministic_baseline(query,goal,state,labels,paths,*,use_goal=True,two_hop=True):
    # Identical complete record information, no oracle labels at inference.
    b,n,d=state.keys.shape
    qq=query[:,0].clone()
    correct=0; route=0; false_stop=0
    allpath=[]
    for row in range(b):
        q=qq[row];g=goal[row,0];observed=[];pred=None
        for step in range(2 if two_hop else 1):
            keys=F.normalize(state.keys[row],dim=-1)
            goals=F.normalize(state.goals[row],dim=-1)
            scores=keys@F.normalize(q,dim=-1)
            if use_goal:scores=scores+goals@F.normalize(g,dim=-1)
            scores=scores.masked_fill(~state.allowed[row],-1.e9)
            if not bool(state.allowed[row].any()):break
            index=int(scores.argmax());observed.append(index)
            value=state.values[row,index]
            max_key=float((keys@F.normalize(value,dim=-1)).max())
            if max_key>=.999999:
                q=value
                continue
            pred=int((CODE@F.normalize(value,dim=-1)).argmax())
            break
        route+=int(observed==paths[row].tolist())
        correct+=int(pred is not None and pred==int(labels[row]) and observed==paths[row].tolist())
        false_stop+=int(pred is not None and observed!=paths[row].tolist())
    return {'correct':correct,'route_exact':route,'false_stop':false_stop,'cases':b}


def train_eval(seed,*,two_hop,episodes=256,epochs=30):
    torch.manual_seed(seed)
    model=TinyNeuralERSAModel(2 if two_hop else 1)
    train=batch(seed+177,episodes,two_hop=two_hop)
    test=batch(seed+271828,128,two_hop=two_hop)
    optim=torch.optim.Adam(model.parameters(),lr=.035)
    x,g,s,label,path=train
    x2,g2,s2,l2,p2=test
    def objective(q,go,st,y,pa):
        logits,_,a=model(q,go,st)
        classloss=F.cross_entropy(logits,y)
        # Oracle indices only used offline as a routing supervision signal.
        route=torch.zeros((),dtype=q.dtype)
        for i in range(pa.shape[1]):
            prob=a['slot_probs'][:,0,i,:].clamp_min(1e-12)
            route=route+F.nll_loss(prob.log(),pa[:,i])
        stop=a['stop_probs'][:,0,:]
        # Two-hop cases must continue at first hop and stop at second.
        targets=torch.ones_like(stop)
        if two_hop:targets[:,0]=0.
        stoploss=F.binary_cross_entropy(stop.clamp(1e-6,1-1e-6),targets)
        return classloss+.45*route+.15*stoploss,classloss,route,stoploss,a
    with torch.no_grad():
        li,*_=objective(x,g,s,label,path)
        te0,_,_,_,a0=objective(x2,g2,s2,l2,p2)
        initial_acc=float((model(x2,g2,s2)[0].argmax(-1)==l2).float().mean())
    for epoch in range(epochs):
        order=torch.randperm(episodes)
        for indices in order.split(64):
            part=ERSAState(*(getattr(s,k)[indices] for k in ('keys','goals','values','metadata','issued','allowed')),
                           tick=s.tick,generation=s.generation)
            optim.zero_grad(set_to_none=True)
            loss,*_=objective(x[indices],g[indices],part,label[indices],path[indices])
            loss.backward()
            if not (model.ersa.selection.grad is not None and
                    torch.isfinite(model.ersa.selection.grad).all() and
                    model.ersa.selection.grad.norm()>0):
                raise RuntimeError('NO_DIFFERENTIABLE_SELECTION_GRADIENT')
            optim.step()
    model.eval()
    with torch.no_grad():
        fin,*_=objective(x,g,s,label,path)
        te1,_,route,sl,a=objective(x2,g2,s2,l2,p2)
        logits,returned,aux=model(x2,g2,s2)
        assert returned is s2
        acc=float((logits.argmax(-1)==l2).float().mean())
        truepaths=0
        predicted=aux['path_indices'][:,0,:path.shape[1]]
        truepaths=int((predicted==p2).all(-1).sum())
        coverage=float(aux['confidence'].mean())
        wbefore=[p.detach().clone() for p in model.parameters()]
        model(x2,g2,s2)
        weights_unchanged=all(torch.equal(a,b) for a,b in zip(wbefore,model.parameters()))
    return {'seed':seed,'two_hop':two_hop,'parameters':sum(p.numel() for p in model.parameters()),
      'train_initial':round(float(li),6),'train_final':round(float(fin),6),
      'test_initial':round(float(te0),6),'test_final':round(float(te1),6),
      'classification_accuracy_initial':round(initial_acc,6),
      'classification_accuracy_final':round(acc,6),
      'true_route_complete':truepaths,'test_cases':128,
      'mean_stop_coverage':round(coverage,6),'eval_weights_unchanged':weights_unchanged,
      'deterministic_key_goal':deterministic_baseline(x2,g2,s2,l2,p2,two_hop=two_hop),
      'deterministic_key_only':deterministic_baseline(x2,g2,s2,l2,p2,use_goal=False,two_hop=two_hop)}


def run():
    torch.set_num_threads(1)
    results=[train_eval(18001,two_hop=False),train_eval(18002,two_hop=True),train_eval(18003,two_hop=True)]
    data={'runs':results,'official_fla_executed':False,'native_bank_mutated':False,
          'learning_path':'PYTORCH_SOFT_RELATIONAL_TOKEN_MIXER',
          'native_parity_scope':'FROZEN_R7_R12_SCORE_FEATURES_ONLY',
          'gdn_parity':'HOLD','open_world_semantics':'HOLD'}
    raw=json.dumps(data,sort_keys=True,separators=(',',':'))
    print('R18_SCIENCE='+raw,flush=True)
    print('R18_SCIENCE_SHA256='+hashlib.sha256(raw.encode()).hexdigest(),flush=True)
    return data
if __name__=='__main__':run()
