#!/usr/bin/env python3
"""R60: B-channel transport windows H1-H20, frozen R52 model and original baseline teacher.
Counterfactual synthetic seam diagnostic; not a training or utility claim.
"""
from __future__ import annotations
import hashlib,json,math,os,statistics,sys,tempfile,traceback,zipfile
from pathlib import Path
ROOT=Path('/mnt/primesauce/Downloads')
class Hold(Exception):pass
def need(value,why):
 if not value:raise Hold(why)
def sha(x):return hashlib.sha256(x).hexdigest()
def filesha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def zin(p,n):
 with zipfile.ZipFile(p) as z:
  need(z.testzip() is None,'ZIP_CRC:'+p.name)
  need(n in z.namelist(),'ZIP_MEMBER:'+p.name+':'+n)
  return z.read(n)
def mean(x):return statistics.mean(x)
def fmt(x):return format(float(x),'.9g')


# Fixed paired gate groups, as in R43 source order:
# RETAIN: (0,4), SALIENT: (1,5), COLLECTIVE: (2,6), CROSS: (3,7).
# Source labels are privileged OFFLINE oracle labels, not deployable.
GROUPS={'RET':(0,4),'SAL':(1,5),'COL':(2,6),'CROSS':(3,7)}
class _HybridGroupLaw:
 """Selected pair groups are exact teacher law on CURRENT model state; others native."""
 def __init__(self,model,teacher,r30,torch,mask):
  self.model=model;self.teacher=teacher;self.r30=r30;self.torch=torch;self.mask=mask
  bits=[i for k,v in GROUPS.items() if mask&(1<<list(GROUPS).index(k)) for i in v]
  self.idx=tuple(sorted(bits))
  self.choose=torch.tensor([i in self.idx for i in range(8)],dtype=torch.bool)
 def __call__(self,state,sal,col):
  t=self.torch;a,b=state
  z=self.model.gates(self.r30.invariants(state,sal,col,t))
  km=t.cat((t.sigmoid(z[...,0:1]),t.tanh(z[...,1:4]),t.sigmoid(z[...,4:5]),t.tanh(z[...,5:8])),dim=-1)
  kt=self.teacher.coeffs(state,sal,col)
  k=t.where(self.choose,kt,km)
  pa=k[...,0,None]*a+k[...,1,None]*sal+k[...,2,None]*col+k[...,3,None]*b
  pb=k[...,4,None]*b+k[...,5,None]*sal+k[...,6,None]*col+k[...,7,None]*a
  rad=lambda x:x*t.clamp(4.0/(t.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-30),max=1.)
  return rad(pa),rad(pb)

def paired_delta(state,B):
 return state[0][:B,3,:]-state[0][B:,3,:]

def _legacy_intervention_run(model,teacher,adj,init,flip,arms,r4,r30,torch,disconnected=False):
 """Fixed mask/time windows; never inject oracle states; no mutation."""
 B=init[0].shape[0];initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
 ref=(initial[0].clone(),initial[1].clone())
 states={x['name']:(initial[0].clone(),initial[1].clone()) for x in arms}
 impls={m:_HybridGroupLaw(model,teacher,r30,torch,m) for m in (1,2,4,8,14,13,11,7,5,10)}
 outputs={name:{} for name in states};parity=0.
 for h in range(1,21):
  ref=r4.graph_step(ref,adj,teacher,torch)
  for x in arms:
   name=x['name'];active=x['start']<=h<=x['end'];mask=x['mask'] if active else 0
   impl=model if mask==0 else teacher if mask==15 else impls[mask]
   states[name]=r4.graph_step(states[name],adj,impl,torch)
   if name=='ORACLE_ALL20':
    parity=max(parity,max(float((states[name][k]-ref[k]).abs().max()) for k in (0,1)))
  if h in (8,12,16,20):
   rr=paired_delta(ref,B);den=rr.square().sum(-1)
   if disconnected:continue
   need(bool((den>1e-25).all()),'REFERENCE_ZERO_H'+str(h))
   for name,state in states.items():
    d=paired_delta(state,B);err=d-rr
    outputs[name][h]=(
     float((err.square().sum(-1)/den).mean()),
     float(((d*rr).sum(-1)/den).mean()),
     float((state[0]-ref[0]).square().mean()),
     float((state[1]-ref[1]).square().mean()),
     float(((d*rr).sum(-1)/den>0).double().mean()))
 return outputs,parity,states

class _BatchedGroupLaw:
 """One native graph step for independent arms, using exact arm/step coefficient masks."""
 def __init__(self,model,teacher,r30,torch):
  self.model=model;self.teacher=teacher;self.r30=r30;self.torch=torch
  self.coef_mask=None
 def set_arm_masks(self,masks,batch):
  t=self.torch
  # group bit 0=RET, 1=SAL, 2=COL, 3=CROSS; replicated on output A and B
  bits=t.tensor([bool(m&(1<<g)) for m in masks for _ in range(batch) for g in (0,1,2,3,0,1,2,3)],dtype=t.bool)
  self.coef_mask=bits.reshape(len(masks)*batch,1,8)
 def __call__(self,state,sal,col):
  t=self.torch;a,b=state
  z=self.model.gates(self.r30.invariants(state,sal,col,t))
  km=t.cat((t.sigmoid(z[...,0:1]),t.tanh(z[...,1:4]),t.sigmoid(z[...,4:5]),t.tanh(z[...,5:8])),dim=-1)
  kt=self.teacher.coeffs(state,sal,col)
  k=t.where(self.coef_mask,kt,km)
  pa=k[...,0,None]*a+k[...,1,None]*sal+k[...,2,None]*col+k[...,3,None]*b
  pb=k[...,4,None]*b+k[...,5,None]*sal+k[...,6,None]*col+k[...,7,None]*a
  rad=lambda x:x*t.clamp(4.0/(t.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-30),max=1.)
  return rad(pa),rad(pb)

def intervention_run(model,teacher,adj,init,flip,arms,r4,r30,torch,disconnected=False):
 """Vectorized INDEPENDENT arm dimension; no cross-arm graph edges or norm reductions."""
 B=init[0].shape[0]; n=len(arms); per_arm=2*B
 initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
 ref=(initial[0].clone(),initial[1].clone())
 packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
 law=_BatchedGroupLaw(model,teacher,r30,torch)
 outputs={x['name']:{} for x in arms};parity=0.
 for h in range(1,21):
  ref=r4.graph_step(ref,adj,teacher,torch)
  masks=[x['mask'] if x['start']<=h<=x['end'] else 0 for x in arms]
  law.set_arm_masks(masks,per_arm)
  packed=r4.graph_step(packed,adj,law,torch)
  positive=next(i for i,x in enumerate(arms) if x['name']=='ORACLE_ALL20')
  ss=slice(positive*per_arm,(positive+1)*per_arm)
  parity=max(parity, max(float((packed[k][ss]-ref[k]).abs().max()) for k in (0,1)))
  if h in (8,12,16,20):
   rr=paired_delta(ref,B);den=rr.square().sum(-1)
   if disconnected:continue
   need(bool((den>1e-25).all()),'REFERENCE_ZERO_H'+str(h))
   for i,x in enumerate(arms):
    ss=slice(i*per_arm,(i+1)*per_arm)
    state=(packed[0][ss],packed[1][ss]);d=paired_delta(state,B);err=d-rr
    outputs[x['name']][h]=(
     float((err.square().sum(-1)/den).mean()),
     float(((d*rr).sum(-1)/den).mean()),
     float((state[0]-ref[0]).square().mean()),
     float((state[1]-ref[1]).square().mean()),
     float(((d*rr).sum(-1)/den>0).double().mean()))
 states={x['name']:(packed[0][i*per_arm:(i+1)*per_arm],packed[1][i*per_arm:(i+1)*per_arm]) for i,x in enumerate(arms)}
 return outputs,parity,states


# R46 paired counterfactual training records; privileged teacher OFFLINE targets only.
import copy

def adaptation_records(train_seeds,cfg,r3,r4,r4cfg,torch,basegraphs,basecases,teacher):
 rows=[];traincases={**basecases,**{k:tuple(v) for k,v in cfg['train_cases'].items()}}
 graphnames=list(r4cfg['train_graphs']);casenames=list(cfg['train_cases'])
 with torch.no_grad():
  for seed in train_seeds:
   for i in range(cfg['adapt_fixtures_per_seed']):
    graph=graphnames[(i+seed)%len(graphnames)];case=casenames[(i//len(graphnames)+seed)%len(casenames)]
    mapped,mats,init,flip=r3.generate(graph,case,cfg['train_init_offset']+seed*100003+i*197,cfg['adapt_batch'],basegraphs,traincases,torch)
    adj=r4.build_links(mapped,mats,torch)
    original=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    st=original;targets=[];refs={};B=cfg['adapt_batch']
    for h in range(1,cfg['adapt_horizon']+1):
     st=r4.graph_step(st,adj,teacher,torch)
     targets.append((st[0].detach().clone(),st[1].detach().clone()))
     if h in cfg['response_horizons']:refs[h]=(st[0][:B,3,:]-st[0][B:,3,:]).detach().clone()
    rows.append((adj,original,targets,refs))
 return rows

def remote_stats(ref,pred,torch):
 den=ref.square().sum(-1)
 need(bool((den>1e-25).all()),'REFERENCE_ZERO')
 gain=(ref*pred).sum(-1)/den
 err=(pred-ref).square().sum(-1)/den
 return float(err.mean()),float(gain.mean()),float((gain>0).double().mean()),float(ref.norm(dim=-1).min())

def test_seed(seed,models,cfg,r3,r4,torch,teacher,graphs,cases):
 ev=cfg['evaluations'];result={a:{c:[] for c in ev['cases']} for a in models}
 hashes={a:r4.model_hash(m) for a,m in models.items()}
 covariance=0.;disconnected=0.;minref=1e30
 with torch.no_grad():
  for gi,graph in enumerate(ev['graphs']):
   for ci,case in enumerate(ev['cases']):
    mapped,mats,init,flip=r3.generate(graph,case,ev['eval_offset']+seed*100109+gi*293+ci*59,ev['batch'],graphs,cases,torch)
    adj=r4.build_links(mapped,mats,torch)
    tr=r4.rollout(init,adj,teacher,ev['horizon'],torch)[-1]
    tf=r4.rollout(flip,adj,teacher,ev['horizon'],torch)[-1]
    if graph==ev['disconnected_graph']:
     disconnected=max(disconnected,float(tr[0][:,3,:].abs().max()),float((tr[0][:,3,:]-tf[0][:,3,:]).abs().max()))
    ref=tr[0][:,3,:]-tf[0][:,3,:]
    for name,model in models.items():
     pr=r4.rollout(init,adj,model,ev['horizon'],torch)[-1]
     pf=r4.rollout(flip,adj,model,ev['horizon'],torch)[-1]
     if graph==ev['disconnected_graph']:
      disconnected=max(disconnected,float(pr[0][:,3,:].abs().max()),float((pr[0][:,3,:]-pf[0][:,3,:]).abs().max()))
      continue
     e,g,s,rr=remote_stats(ref,pr[0][:,3,:]-pf[0][:,3,:],torch)
     minref=min(minref,rr)
     result[name][case].append((e,g,s,seed,graph))
     if (seed==cfg['evaluation_seeds'][0] or check_direct) and graph==ev['gauge_graph'] and case==ev['gauge_case']:
      gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
      gadj=r4.links_gauge(adj,gauges,torch)
      for state,pred in ((init,pr),(flip,pf)):
       rot=r4.rollout(r4.gauge_transform(state,gauges,torch),gadj,model,ev['horizon'],torch)[-1]
       expected=r4.gauge_transform(pred,gauges,torch)
       covariance=max(covariance,*(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
    if (seed==cfg['evaluation_seeds'][0] or check_direct) and graph==ev['gauge_graph'] and case==ev['gauge_case']:
     gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
     gadj=r4.links_gauge(adj,gauges,torch)
     rot=r4.rollout(r4.gauge_transform(init,gauges,torch),gadj,teacher,ev['horizon'],torch)[-1]
     expected=r4.gauge_transform(tr,gauges,torch)
     covariance=max(covariance,*(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
 for name,model in models.items():need(r4.model_hash(model)==hashes[name],'INFERENCE_WEIGHTS_MUTATED:'+name)
 return result,covariance,disconnected,minref

def summarize(rows,models,cases):
 report={}
 for case in cases:
  report[case]={}
  for arm in models:
   data=rows[arm][case]
   report[case][arm]={'err':mean([t[0] for t in data]),'median':statistics.median(t[0] for t in data),
                      'worst':max(t[0] for t in data),'gain':mean(t[1] for t in data),'sign':mean(t[2] for t in data),
                      'by_seed':{seed:mean(x[0] for x in data if x[3]==seed) for seed in sorted(set(x[3] for x in data))}}
 return report

def science_gates(report,rows,cfg,fitstats):
 acc=cfg['scientific_acceptance'];s='SAL_SCALED_SIGNED8';b='SAL_STATE8';orig='SAL_ORIG_SIGNED8';floor='SAL_FLOOR_SIGNED8';r='RET_SCALED_SIGNED8'
 weak='r48_weak192';reverse='r48_reverse192';anti='r48_anti192'
 def wins(case,opp):
  p=report[case][s]['by_seed'];q=report[case][opp]['by_seed']
  return sum(p[i]<q[i] for i in p)
 def cells(case):return sum(p[0]<q[0] for p,q in zip(rows[s][case],rows[b][case]))
 w=report[weak][s];v=report[reverse][s];a=report[anti][s]
 metrics={'WEAK_SEEDS':wins(weak,b),'REVERSE_SEEDS':wins(reverse,b),'ANTI_SEEDS':wins(anti,b),
          'WEAK_CELLS':cells(weak),'REVERSE_CELLS':cells(reverse),'ANTI_CELLS':cells(anti),
          'BEAT_ORIGINAL_SEEDS':wins(weak,orig),'BEAT_FLOOR_SEEDS':wins(weak,floor),
          'BEAT_RETENTION_SEEDS':wins(weak,r),'SCALED_CLIPPED':fitstats[s]['clipped']}
 gates={
  'weak_seed_improvement':metrics['WEAK_SEEDS']>=acc['weak_seed_min'],
  'reverse_seed_improvement':metrics['REVERSE_SEEDS']>=acc['reverse_seed_min'],
  'anti_seed_improvement':metrics['ANTI_SEEDS']>=acc['anti_seed_min'],
  'weak_cell_improvement':metrics['WEAK_CELLS']>=acc['weak_cells_min'],
  'reverse_cell_improvement':metrics['REVERSE_CELLS']>=acc['reverse_cells_min'],
  'anti_cell_improvement':metrics['ANTI_CELLS']>=acc['anti_cells_min'],
  'beat_original_signed':metrics['BEAT_ORIGINAL_SEEDS']>=acc['beat_original_signed_seeds_min'],
  'beat_floor_only':metrics['BEAT_FLOOR_SEEDS']>=acc['beat_floor_only_seeds_min'],
  'beat_retention':metrics['BEAT_RETENTION_SEEDS']>=acc['beat_retention_seeds_min'],
  'weak_median':w['median']<=acc['weak_median_max'],
  'weak_worst':w['worst']<=acc['weak_worst_max'],
  'reverse_median':v['median']<=acc['reverse_median_max'],
  'reverse_worst':v['worst']<=acc['reverse_worst_max'],
  'weak_gain':acc['weak_gain_min']<=w['gain']<=acc['weak_gain_max'],
  'reverse_gain':acc['reverse_gain_min']<=v['gain']<=acc['reverse_gain_max'],
  'anti_gain':acc['anti_gain_min']<=a['gain']<=acc['anti_gain_max'],
  'signed_fraction':min(w['sign'],v['sign'],a['sign'])>=acc['signed_fraction_min'],
  'anti_worst_nondegradation':a['worst']<=acc['anti_worst_nondegrade_max_factor']*report[anti][b]['worst'],
  'scaled_clip_gate':metrics['SCALED_CLIPPED']<=acc['scaled_max_clipped_updates']}
 return gates,metrics

class _StackedGates:
 """Exact native 8-gate HCA, independent arm dimension, no cross-arm state mixing."""
 def __init__(self,weights,bias,r30,torch,per_arm):
  self.weights=weights;self.bias=bias;self.r30=r30;self.torch=torch;self.per_arm=per_arm
 def __call__(self,state,sal,col):
  t=self.torch;a,b=state;n=self.weights.shape[0]
  feat=self.r30.invariants(state,sal,col,t).reshape(n,self.per_arm*4,10)
  logits=t.bmm(feat,self.weights.transpose(1,2))+self.bias[:,None,:]
  z=logits.reshape(n*self.per_arm,4,8)
  k=t.cat((t.sigmoid(z[...,0:1]),t.tanh(z[...,1:4]),t.sigmoid(z[...,4:5]),t.tanh(z[...,5:8])),dim=-1)
  pa=k[...,0,None]*a+k[...,1,None]*sal+k[...,2,None]*col+k[...,3,None]*b
  pb=k[...,4,None]*b+k[...,5,None]*sal+k[...,6,None]*col+k[...,7,None]*a
  proj=lambda x:x*t.clamp(4.0/(t.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-30),max=1.)
  return proj(pa),proj(pb)

def fit_all_arms(base,records,cfg,r30,r4,torch):
 arms=list(cfg['arms']);n=len(arms);B=cfg['adapt_batch'];per=2*B
 W0=base.gates.weight.detach().clone();b0=base.gates.bias.detach().clone()
 W=torch.nn.Parameter(W0.unsqueeze(0).expand(n,-1,-1).clone())
 bias=torch.nn.Parameter(b0.unsqueeze(0).expand(n,-1).clone())
 mask=torch.zeros((n,8,1),dtype=torch.float64)
 for j,name in enumerate(arms):
  if cfg['arms'][name]:mask[j,cfg['arms'][name],0]=1.
 optimizer=torch.optim.Adam((W,bias),lr=cfg['learning_rate'],weight_decay=cfg['adam_weight_decay'])
 maxgrad=torch.zeros(n,dtype=torch.float64);clipped=torch.zeros(n,dtype=torch.long)
 final_loss=torch.zeros(n,dtype=torch.float64)
 for it in range(cfg['updates']):
  adj,initial,targets,refs=records[it%len(records)]
  inp=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
  law=_StackedGates(W,bias,r30,torch,per)
  with torch.enable_grad():
   state=inp;losses=torch.zeros(n,dtype=torch.float64)
   for h in range(1,cfg['adapt_horizon']+1):
    state=r4.graph_step(state,adj,law,torch)
    pa=state[0].reshape(n,per,4,4);pb=state[1].reshape(n,per,4,4)
    ta,tb=targets[h-1]
    mse=.5*((pa-ta).square().mean(dim=(1,2,3))+(pb-tb).square().mean(dim=(1,2,3)))
    # Matched eight-step trajectory objective; source term is the only objective difference.
    stepweights=torch.tensor([0.0 if a=='BASE_LS' else 1./8 for a in arms],dtype=torch.float64)
    losses=losses+stepweights*mse
    if h in cfg['response_horizons']:
     delta=pa[:,:B,3,:]-pa[:,B:,3,:]
     target=refs[h]
     floors=torch.tensor([cfg['objective_variants'][a]['response_floor'] for a in arms],dtype=torch.float64)
     normalized=((delta-target).square().sum(-1)/(target.square().sum(-1)[None,:]+floors[:,None])).mean(dim=1)
     responseweights=torch.tensor([cfg['objective_variants'][a]['response_lambda']/len(cfg['response_horizons']) for a in arms],dtype=torch.float64)
     losses=losses+responseweights*normalized
   need(bool(torch.isfinite(losses).all()),'TRAIN_NONFINITE_LOSS')
   optimizer.zero_grad(set_to_none=True);losses.sum().backward()
   need(bool(torch.isfinite(W.grad).all() and torch.isfinite(bias.grad).all()),'TRAIN_NONFINITE_GRAD')
   W.grad.mul_(mask);bias.grad.mul_(mask.squeeze(-1))
   norms=(W.grad.square().sum(dim=(1,2))+bias.grad.square().sum(dim=1)).sqrt()
   maxgrad=torch.maximum(maxgrad,norms.detach());clipped+=(norms.detach()>cfg['grad_clip_norm']).long()
   factors=(cfg['grad_clip_norm']/(norms.detach()+1e-30)).clamp(max=1.)
   W.grad.mul_(factors[:,None,None]);bias.grad.mul_(factors[:,None])
   optimizer.step();final_loss=losses.detach().clone()
   if (it+1)%16==0:print('[R49] training_updates='+str(it+1)+'/'+str(cfg['updates'])+' arm_losses='+','.join(a+':'+fmt(final_loss[i]) for i,a in enumerate(arms)),flush=True)
 models={};stats={}
 with torch.no_grad():
  for i,a in enumerate(arms):
   model=copy.deepcopy(base);model.gates.weight.copy_(W[i]);model.gates.bias.copy_(bias[i]);model.eval()
   for j in range(8):
    if j not in cfg['arms'][a]:need(bool(torch.equal(model.gates.weight[j],W0[j]) and torch.equal(model.gates.bias[j],b0[j])),'FROZEN_ROW_MODIFIED:'+a+':'+str(j))
   need(bool(torch.equal(model.gates.weight[:,9],W0[:,9])),'UNUSED_NATIVE_EXTRA_SLOT_MODIFIED')
   delta=float(((model.gates.weight-W0).square().sum()+(model.gates.bias-b0).square().sum()).item())**.5
   if a!='BASE_LS':need(delta>0,'TRAIN_NO_CHANGE:'+a)
   models[a]=model;stats[a]={'last':float(final_loss[i]),'delta':delta,'max_grad':float(maxgrad[i]),'clipped':int(clipped[i])}
 return models,stats

def test_seed_stacked(seed,models,cfg,r3,r4,r30,torch,teacher,graphs,cases,check_direct=False):
 """R48 identical six-arm H20 evaluator, plus exact checkpoints on natural trajectories.
    Some early graph/case cells have zero remote target: skip those only at that horizon.
 """
 ev=cfg['evaluations'];arms=list(models);n=len(arms);B=ev['batch'];per=2*B
 hashes={a:r4.model_hash(m) for a,m in models.items()}
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 result={a:{c:[] for c in ev['cases']} for a in models}
 trace={a:{c:{h:[] for h in cfg['audit_horizons']} for c in ev['cases']} for a in models}
 covariance=0.;disconnected=0.;minref=1e30;direct_parity=0.
 with torch.no_grad():
  for gi,graph in enumerate(ev['graphs']):
   for ci,case in enumerate(ev['cases']):
    mapped,mats,init,flip=r3.generate(graph,case,ev['eval_offset']+seed*100109+gi*293+ci*59,B,graphs,cases,torch)
    adj=r4.build_links(mapped,mats,torch)
    tinit=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    tref=(tinit[0].clone(),tinit[1].clone())
    packed=(tinit[0].repeat(n,1,1),tinit[1].repeat(n,1,1))
    for h in range(1,ev['horizon']+1):
     tref=r4.graph_step(tref,adj,teacher,torch)
     packed=r4.graph_step(packed,adj,law,torch)
     if h not in cfg['audit_horizons']:continue
     ref=tref[0][:B,3,:]-tref[0][B:,3,:]
     den=ref.square().sum(-1)
     safe=den>1e-25
     valid=int(safe.sum().item())
     if graph==ev['disconnected_graph']:
      disconnected=max(disconnected,float(tref[0][:,3,:].abs().max()),float((tref[0][:B,3,:]-tref[0][B:,3,:]).abs().max()))
     for j,a in enumerate(arms):
      ss=slice(j*per,(j+1)*per)
      st=(packed[0][ss],packed[1][ss])
      if graph==ev['disconnected_graph']:
       disconnected=max(disconnected,float(st[0][:,3,:].abs().max()),float((st[0][:B,3,:]-st[0][B:,3,:]).abs().max()))
       continue
      pred=st[0][:B,3,:]-st[0][B:,3,:]
      if valid:
       err=(pred-ref).square().sum(-1)[safe]/den[safe]
       gain=(pred*ref).sum(-1)[safe]/den[safe]
       sign=(gain>0).double()
       trace[a][case][h].append((float(err.mean()),float(gain.mean()),float(sign.mean()),
                                 float(den[safe].mean().sqrt()),float(pred.square().sum(-1)[safe].mean().sqrt()),valid,B))
      if h==20:
       need(valid==B,'REFERENCE_ZERO_H20')
       e,g,s,rr=remote_stats(ref,pred,torch);minref=min(minref,rr)
       result[a][case].append((e,g,s,seed,graph))
       if check_direct and gi==0 and ci==0:
        dr=r4.rollout(tinit,adj,models[a],ev['horizon'],torch)[-1]
        direct_parity=max(direct_parity,max(float((dr[k]-st[k]).abs().max()) for k in (0,1)))
       if (seed==cfg['evaluation_seeds'][0] or check_direct) and graph==ev['gauge_graph'] and case==ev['gauge_case']:
        gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
        gadj=r4.links_gauge(adj,gauges,torch)
        rotated=r4.rollout(r4.gauge_transform(tinit,gauges,torch),gadj,models[a],ev['horizon'],torch)[-1]
        expect=r4.gauge_transform(st,gauges,torch)
        covariance=max(covariance,*(float((rotated[k]-expect[k]).abs().max()) for k in (0,1)))
     if h==20 and (seed==cfg['evaluation_seeds'][0] or check_direct) and graph==ev['gauge_graph'] and case==ev['gauge_case']:
      gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
      gadj=r4.links_gauge(adj,gauges,torch)
      rotated=r4.rollout(r4.gauge_transform(tinit,gauges,torch),gadj,teacher,ev['horizon'],torch)[-1]
      expect=r4.gauge_transform(tref,gauges,torch)
      covariance=max(covariance,*(float((rotated[k]-expect[k]).abs().max()) for k in (0,1)))
 for a,model in models.items():need(hashes[a]==r4.model_hash(model),'INFERENCE_WEIGHT_MUTATION:'+a)
 return result,covariance,disconnected,minref,direct_parity,trace


def audit_train_signed_response(models,adapt,cfg,r30,r4,torch):
 """Evaluate the final frozen R48 arms on ALL original adaptation fixtures H4/H6/H8.
    This is in-sample diagnostics, not a new optimization or reweighting stage.
 """
 arms=list(models);n=len(arms);B=cfg['adapt_batch'];per=2*B
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 out={a:{h:[] for h in cfg['response_horizons']} for a in arms}
 with torch.no_grad():
  for adj,initial,targets,refs in adapt:
   packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
   for h in range(1,cfg['adapt_horizon']+1):
    packed=r4.graph_step(packed,adj,law,torch)
    if h not in cfg['response_horizons']:continue
    target=refs[h];den=target.square().sum(-1);safe=den>1e-25
    for j,arm in enumerate(arms):
     ss=slice(j*per,(j+1)*per)
     pa=packed[0][ss].reshape(per,4,4)
     pb=packed[1][ss].reshape(per,4,4)
     ta,tb=targets[h-1]
     state_mse=.5*((pa-ta).square().mean()+(pb-tb).square().mean())
     d=pa[:B,3,:]-pa[B:,3,:]
     raw=(d-target).square().sum(-1)
     floor=cfg['objective_variants'][arm]['response_floor']
     floored=raw/(den+floor)
     if bool(safe.any()):
      e=float((raw[safe]/den[safe]).mean());gain=float(((d[safe]*target[safe]).sum(-1)/den[safe]).mean())
     else:e=float('nan');gain=float('nan')
     out[arm][h].append((float(state_mse),float(floored.mean()),e,gain,int(safe.sum()),B))
 return out


def audit_replay(reference,rows,cfg):
 """Exact R48 published H20 aggregate parity. Prints no new scientific qualification."""
 checked=0;maxres=0.
 for case,arms in reference.items():
  for arm,(e,g) in arms.items():
   record=rows[arm][case]
   need(bool(record),'R48_NO_ROWS:'+arm+':'+case)
   actual_e=mean(x[0] for x in record);actual_g=mean(x[1] for x in record)
   for expected,actual in ((e,actual_e),(g,actual_g)):
    limit=cfg['technical_tolerances']['h20_replay_abs']+cfg['technical_tolerances']['h20_replay_rel']*abs(expected)
    need(abs(actual-expected)<limit,'R48_H20_REPLAY:'+case+':'+arm+':'+fmt(actual)+':'+fmt(expected))
    maxres=max(maxres,abs(actual-expected));checked+=1
 return checked,maxres


def summarize_trajectory(trajectories,cfg):
 report={};no_ref={}
 for case in cfg['evaluations']['cases']:
  report[case]={}
  for arm in cfg['arms']:
   report[case][arm]={}
   for h in cfg['audit_horizons']:
    row=trajectories[arm][case][h]
    valid=sum(r[5] for r in row);possible=sum(r[6] for r in row)
    if not row:
     report[case][arm][h]=(float('nan'),)*5+(0,0)
    else:
     report[case][arm][h]=tuple(mean(r[i] for r in row) for i in range(5))+(valid,possible)
    no_ref[(case,arm,h)]=(valid,possible)
 return report,no_ref


# TRAINING-ONLY horizon-gradient probes. Teacher labels are privileged OFFLINE.
def active_vector(grads, torch):
 w,b=grads
 return torch.cat((w[[1,5],:9].reshape(-1),b[[1,5]].reshape(-1)))

def cosine(a,b,torch):
 return float((a*b).sum()/(a.norm()*b.norm()+1e-30))

def probe_horizon(model,record,cfg,r4,r30,torch,teacher,preflight=False):
 adj,initial,old_targets,old_refs=record
 B=cfg['adapt_batch'];per=2*B
 with torch.no_grad():
  teacher_states={};st=(initial[0].clone(),initial[1].clone())
  for h in range(1,21):
   st=r4.graph_step(st,adj,teacher,torch)
   if h<=8:
    for k in (0,1):need(torch.equal(st[k],old_targets[h-1][k]),'R48_TRAIN_TARGET_PARITY_H'+str(h))
   if h in cfg['diagnostic']['horizons']:
    teacher_states[h]=(st[0].clone(),st[1].clone())
    if h in old_refs:
     need(torch.equal(st[0][:B,3,:]-st[0][B:,3,:],old_refs[h]),'R48_TRAIN_RESPONSE_PARITY_H'+str(h))
 W=model.gates.weight.detach().clone().requires_grad_(True)
 bias=model.gates.bias.detach().clone().requires_grad_(True)
 law=_StackedGates(W.unsqueeze(0),bias.unsqueeze(0),r30,torch,per)
 st=(initial[0].clone(),initial[1].clone());values={}; gradients={}
 for h in range(1,21):
  st=r4.graph_step(st,adj,law,torch)
  if h in cfg['diagnostic']['horizons']:
   ta,tb=teacher_states[h]
   state_loss=.5*((st[0]-ta).square().mean()+(st[1]-tb).square().mean())
   predicted=st[0][:B,3,:]-st[0][B:,3,:]
   reference=ta[:B,3,:]-ta[B:,3,:]
   den=reference.square().sum(-1);num=(predicted-reference).square().sum(-1)
   floorlosses={str(f):(num/(den+f)).mean() for f in cfg['diagnostic']['floors']}
   values[h]={'state_loss':state_loss,'floorlosses':floorlosses,'den':den,'numerator':num,
              'gain':((predicted*reference).sum(-1)/(den+1e-30)).mean(),
              'model_pre_norm':float(st[0].norm(dim=-1).max().detach())}
 def g(loss):
  z=torch.autograd.grad(loss,(W,bias),retain_graph=True,allow_unused=False)
  v=active_vector(z,torch).detach();need(bool(torch.isfinite(v).all()),'NONFINITE_GRAD');return v
 for h in cfg['diagnostic']['horizons']:
  o=values[h]
  gradients[h]={'STATE':g(o['state_loss'])}
  for key,factor,floor in [('ORIGINAL',.5,.0001),('SCALED',.05,.1)]:
   gradients[h][key]=g(factor*o['floorlosses'][str(floor)])
 summaries={};horizons=cfg['diagnostic']['horizons']
 for h in horizons:
  o=values[h];den=o['den'].detach().tolist();gr=gradients[h]
  terms={k:float(v.norm()) for k,v in gr.items()}
  summaries[h]={'reference_min':min(den),'reference_median':statistics.median(den),
    'den_below_orig':sum(float(x<.0001) for x in den)/len(den),
    'den_below_scaled':sum(float(x<.1) for x in den)/len(den),
    'state_loss':float(o['state_loss'].detach()),
    'signed_original_loss':float(o['floorlosses'][str(.0001)].detach()),
    'signed_scaled_loss':float(o['floorlosses'][str(.1)].detach()),
    'signed_gain':float(o['gain'].detach()),
    'norms':terms,
    'state_original_cos':cosine(gr['STATE'],gr['ORIGINAL'],torch),
    'state_scaled_cos':cosine(gr['STATE'],gr['SCALED'],torch),
    'state_h8_cos':cosine(gr['STATE'],gradients[8]['STATE'],torch),
    'orig_h8_cos':cosine(gr['ORIGINAL'],gradients[8]['ORIGINAL'],torch),
    'scaled_h8_cos':cosine(gr['SCALED'],gradients[8]['SCALED'],torch),
    'orig_total':float((gr['STATE']+gr['ORIGINAL']).norm()),
    'scaled_total':float((gr['STATE']+gr['SCALED']).norm()),
    'orig_clip':int((gr['STATE']+gr['ORIGINAL']).norm()>1.),
    'scaled_clip':int((gr['STATE']+gr['SCALED']).norm()>1.),
    'state_to_response_orig_ratio':terms['ORIGINAL']/(terms['STATE']+1e-30),
    'state_to_response_scaled_ratio':terms['SCALED']/(terms['STATE']+1e-30)}
 if preflight:
  # Independent loss gradient cross-check using a clean second 20-step replay.
  st=(initial[0].clone(),initial[1].clone())
  for hh in range(1,21):
   st=r4.graph_step(st,adj,law,torch)
   if hh==8:
    ta,tb=teacher_states[hh]
    loss=.5*((st[0]-ta).square().mean()+(st[1]-tb).square().mean())
    check=active_vector(torch.autograd.grad(loss,(W,bias),retain_graph=True),torch)
    need(bool(torch.allclose(check,gradients[8]['STATE'],atol=1e-10,rtol=1e-10)),'H8_GRAD_RECONSTRUCTION')
 return (summaries, gradients)

def aggregate_probes(probes,cfg):
 out={}
 for arm, rows in probes.items():
  out[arm]={}
  for h in cfg['diagnostic']['horizons']:
   ss=[r[h] for r in rows]
   o={k:mean(x[k] for x in ss) for k in ('state_loss','signed_original_loss','signed_scaled_loss','signed_gain','state_original_cos','state_scaled_cos','state_h8_cos','orig_h8_cos','scaled_h8_cos','orig_total','scaled_total','reference_min','reference_median','den_below_orig','den_below_scaled')}
   o['norms']={k:mean(x['norms'][k] for x in ss) for k in ('STATE','ORIGINAL','SCALED')}
   o['clipped_orig']=sum(x['orig_clip'] for x in ss)
   o['clipped_scaled']=sum(x['scaled_clip'] for x in ss)
   o['ratio_orig_median']=statistics.median(x['state_to_response_orig_ratio'] for x in ss)
   o['ratio_scaled_median']=statistics.median(x['state_to_response_scaled_ratio'] for x in ss)
   out[arm][h]=o
 return out


# R51: deterministic non-training combinations of four exact R50 horizon gradients.
def combined_gradients(g,torch,method,hs=(8,12,16,20)):
 v=[g[h]['STATE']+g[h]['SCALED'] for h in hs]
 if method=='H8_ONLY':d=v[0]
 elif method=='H20_ONLY':d=v[-1]
 elif method=='MEAN4':d=sum(v)/4
 elif method=='NORM4':
  # Each horizon has identical norm contribution; retain mean input norm as scale.
  scale=sum(float(z.norm()) for z in v)/4
  d=sum(z/(z.norm()+1e-30) for z in v)*(scale/4)
 elif method=='PCGRAD4':
  projected=[]
  for i,x in enumerate(v):
   u=x.clone()
   for j,y in enumerate(v):
    if i==j:continue
    dot=torch.dot(u,y)
    if float(dot)<0:u=u-dot/(torch.dot(y,y)+1e-30)*y
   projected.append(u)
  d=sum(projected)/4
 elif method=='MINNORM8_20':
  x,y=v[0],v[-1]
  delta=x-y
  alpha=float(torch.clamp((torch.dot(y,y)-torch.dot(x,y))/(torch.dot(delta,delta)+1e-30),min=0,max=1))
  d=alpha*x+(1-alpha)*y
 else:raise Hold('UNKNOWN_COMBINATION:'+method)
 need(bool(torch.isfinite(d).all()),'NONFINITE_COMBINATION:'+method)
 return d

def combine_probe(g,cfg,torch,preflight=False):
 methods=cfg['combination']['methods'];hs=cfg['diagnostic']['horizons']
 total={h:g[h]['STATE']+g[h]['SCALED'] for h in hs}
 if preflight:
  # Exact algebraic parity and deterministic method reconstruction on direct vectors.
  assert torch.allclose(combined_gradients(g,torch,'MEAN4'),torch.stack(list(total.values())).mean(dim=0),atol=1e-14,rtol=1e-14)
  for method in methods:
   a=combined_gradients(g,torch,method);b=combined_gradients(g,torch,method)
   need(torch.equal(a,b),'COMBINE_NONDETERMINISTIC:'+method)
  # All vectors use ONLY the frozen twenty active salient weights/biases.
  need(all(x.numel()==20 for x in total.values()),'PARAM_DIMENSION_DRIFT')
 out={}
 for name in methods:
  d=combined_gradients(g,torch,name);l=float(d.norm())
  cosines={h:cosine(d,total[h],torch) for h in hs}
  cos_signed={h:cosine(d,g[h]['SCALED'],torch) for h in hs}
  cos_state={h:cosine(d,g[h]['STATE'],torch) for h in hs}
  out[name]={'norm':l,'clip':int(l>cfg['combination']['hypothetical_clip']),
   'negative_total':sum(int(torch.dot(d,total[h])<0) for h in hs),
   'negative_signed':sum(int(torch.dot(d,g[h]['SCALED'])<0) for h in hs),
   'cos_h8':cosines[8],'cos_h20':cosines[20],
   'h20_grad_retention':float(torch.dot(d,total[20])/(torch.dot(total[20],total[20])+1e-30)),
   'cos':cosines,'cos_signed':cos_signed,'cos_state':cos_state}
 return out

def aggregate_combinations(rows, cfg):
 out={};hs=cfg['diagnostic']['horizons'];ms=cfg['combination']['methods']
 for arm,records in rows.items():
  out[arm]={}
  for method in ms:
   items=[r[method] for r in records]
   q={'norm':mean(x['norm'] for x in items),'clipped':sum(x['clip'] for x in items),
      'neg_total':sum(x['negative_total'] for x in items),
      'neg_signed':sum(x['negative_signed'] for x in items),
      'cos_h8':mean(x['cos_h8'] for x in items),'cos_h20':mean(x['cos_h20'] for x in items),
      'h20_retention':mean(x['h20_grad_retention'] for x in items),
      'cos':{h:mean(x['cos'][h] for x in items) for h in hs},
      'cos_signed':{h:mean(x['cos_signed'][h] for x in items) for h in hs},
      'cos_state':{h:mean(x['cos_state'][h] for x in items) for h in hs}}
   out[arm][method]=q
 return out



def prospective_config(cfg):
 import copy
 p=cfg['prospective']; x=copy.deepcopy(cfg)
 for k in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds','base_fixtures_per_seed','adapt_fixtures_per_seed','base_train_batch','adapt_batch','base_snapshots','train_cases','train_init_offset','updates','learning_rate','grad_clip_norm','adam_weight_decay','adapt_horizon'):
  x[k]=p[k]
 x['response_horizons']=p['loss_horizons']
 x['evaluations']=dict(p['evaluations']);x['evaluations']['graphs']=p['evaluations']['graphs_order']
 return x

def train_prospective(base,records,cfg,r30,r4,torch,preflight=False):
 import copy
 p=cfg['prospective']; names=p['arms'][1:]; updates=2 if preflight else p['updates']; B=p['adapt_batch'];per=2*B
 W0=base.gates.weight.detach().clone();b0=base.gates.bias.detach().clone();models={'BASE_LS':copy.deepcopy(base)}
 stats={'BASE_LS':{'last':0.,'delta':0.,'max_grad':0.,'clipped':0}}
 allfinal={}; rowmask=torch.zeros_like(W0);rowmask[[1,5],:9]=1
 bmask=torch.zeros_like(b0);bmask[[1,5]]=1
 for name in names:
  W=torch.nn.Parameter(W0.clone());bias=torch.nn.Parameter(b0.clone())
  opt=torch.optim.Adam([W,bias],lr=p['learning_rate'],weight_decay=p['adam_weight_decay'])
  maximum=0.;clipcount=0;last=0.
  for step in range(updates):
   adj,initial,targets,refs=records[step%len(records)]
   law=_StackedGates(W.unsqueeze(0),bias.unsqueeze(0),r30,torch,per)
   st=initial; firsteight=[]; per_h={}
   for h in range(1,21):
    st=r4.graph_step(st,adj,law,torch)
    t=targets[h-1]
    stloss=.5*((st[0]-t[0]).square().mean()+(st[1]-t[1]).square().mean())
    if h<=8:firsteight.append(stloss)
    if h in p['loss_horizons']:
     target=refs[h];out=st[0][:B,3,:]-st[0][B:,3,:]
     response=((out-target).square().sum(-1)/(target.square().sum(-1)+p['response_floor'])).mean()
     per_h[h]=stloss+p['response_lambda']*response
   if name=='SAL_STATE8':
    loss=sum(firsteight)/len(firsteight);last=float(loss.detach())
    gg=torch.autograd.grad(loss,(W,bias),allow_unused=False)
    vector=active_vector(gg,torch)
   else:
    grads={}; last=float((per_h[20] if name=='SAL_H20_ONLY' else torch.stack(list(per_h.values())).mean()).detach())
    for i,h in enumerate(p['loss_horizons']):
     gg=torch.autograd.grad(per_h[h],(W,bias),retain_graph=(i<3 or (name=='SAL_MEAN4' and step==0)),allow_unused=False)
     grads[h]={'STATE':torch.zeros(20,dtype=torch.float64),'SCALED':active_vector(gg,torch)}
    if name=='SAL_H20_ONLY':vector=combined_gradients(grads,torch,'H20_ONLY')
    elif name=='SAL_MEAN4':vector=combined_gradients(grads,torch,'MEAN4')
    elif name=='SAL_PCGRAD4':vector=combined_gradients(grads,torch,'PCGRAD4')
    else:raise Hold('UNKNOWN_TRAIN_ARM')
    if name=='SAL_MEAN4' and step==0:
     chk=torch.autograd.grad(sum(per_h.values())/4,(W,bias),allow_unused=False)
     need(bool(torch.allclose(vector,active_vector(chk,torch),rtol=1e-10,atol=1e-11)),'MEAN_GRADIENT_PARITY')
    if name=='SAL_PCGRAD4' and step==0:
     need(bool(torch.equal(vector,combined_gradients(grads,torch,'PCGRAD4'))),'PCGRAD_DETERMINISM')
   need(bool(torch.isfinite(vector).all()),'NONFINITE_TRAIN_GRAD:'+name)
   norm=float(vector.norm()); maximum=max(maximum,norm);clipcount+=int(norm>p['grad_clip_norm'])
   factor=min(1.,p['grad_clip_norm']/(norm+1e-30))
   v=(vector.detach()*factor)
   opt.zero_grad(set_to_none=True)
   W.grad=torch.zeros_like(W);bias.grad=torch.zeros_like(bias)
   W.grad[[1,5],:9]=v[:18].reshape(2,9);bias.grad[[1,5]]=v[18:]
   opt.step()
   with torch.no_grad():
    need(bool(torch.equal(W[~rowmask.bool()],W0[~rowmask.bool()]) and torch.equal(bias[~bmask.bool()],b0[~bmask.bool()])),'NONACTIVE_ROW_MUTATION:'+name)
   if (step+1)%16==0 or step+1==updates:
    print('[R52] TRAIN '+name+' updates='+str(step+1)+'/'+str(updates)+' loss='+fmt(last)+' grad='+fmt(norm)+' clip='+str(clipcount),flush=True)
  model=copy.deepcopy(base)
  with torch.no_grad():model.gates.weight.copy_(W);model.gates.bias.copy_(bias)
  model.eval();delta=float(torch.sqrt((W-W0).square().sum()+(bias-b0).square().sum()).detach())
  need(delta>0,'ZERO_PARAMETER_UPDATE:'+name)
  models[name]=model;stats[name]={'last':last,'delta':delta,'max_grad':maximum,'clipped':clipcount}
  allfinal[name]=float(last)
 return models,stats

def train_r53_audited(base,records,cfg,r30,r4,torch,preflight=False):
 import copy
 p=cfg['prospective']; names=p['arms'][1:]; updates=2 if preflight else p['updates']; B=p['adapt_batch'];per=2*B
 W0=base.gates.weight.detach().clone();b0=base.gates.bias.detach().clone();models={'BASE_LS':copy.deepcopy(base)}
 stats={'BASE_LS':{'last':0.,'delta':0.,'max_grad':0.,'clipped':0}}
 snapshots={'BASE_LS':{k:(W0.clone(),b0.clone()) for k in cfg['r53_diagnostic']['checkpoints']}};history={}
 allfinal={}; rowmask=torch.zeros_like(W0);rowmask[[1,5],:9]=1
 bmask=torch.zeros_like(b0);bmask[[1,5]]=1
 for name in names:
  W=torch.nn.Parameter(W0.clone());bias=torch.nn.Parameter(b0.clone())
  opt=torch.optim.Adam([W,bias],lr=p['learning_rate'],weight_decay=p['adam_weight_decay'])
  maximum=0.;clipcount=0;last=0.
  snapshots[name]={0:(W.detach().clone(),bias.detach().clone())}
  history[name]=[]
  for step in range(updates):
   adj,initial,targets,refs=records[step%len(records)]
   law=_StackedGates(W.unsqueeze(0),bias.unsqueeze(0),r30,torch,per)
   st=initial; firsteight=[]; per_h={}; per_gain={}; per_e2={}
   for h in range(1,21):
    st=r4.graph_step(st,adj,law,torch)
    t=targets[h-1]
    stloss=.5*((st[0]-t[0]).square().mean()+(st[1]-t[1]).square().mean())
    if h<=8:firsteight.append(stloss)
    if h in p['loss_horizons']:
     target=refs[h];out=st[0][:B,3,:]-st[0][B:,3,:]
     response=((out-target).square().sum(-1)/(target.square().sum(-1)+p['response_floor'])).mean()
     per_h[h]=stloss+p['response_lambda']*response
     den=target.square().sum(-1); valid=den>1e-25
     if bool(valid.any()):
      per_gain[h]=float(((out*target).sum(-1)[valid]/den[valid]).mean().detach())
      per_e2[h]=float(((out-target).square().sum(-1)[valid]/den[valid]).mean().detach())
   neg_pairs=0;neg_directions=0; raw_mean_norm=0.
   if name=='SAL_STATE8':
    loss=sum(firsteight)/len(firsteight);last=float(loss.detach())
    gg=torch.autograd.grad(loss,(W,bias),allow_unused=False)
    vector=active_vector(gg,torch)
   else:
    grads={}; last=float((per_h[20] if name=='SAL_H20_ONLY' else torch.stack(list(per_h.values())).mean()).detach())
    for i,h in enumerate(p['loss_horizons']):
     gg=torch.autograd.grad(per_h[h],(W,bias),retain_graph=(i<3 or (name=='SAL_MEAN4' and step==0)),allow_unused=False)
     grads[h]={'STATE':torch.zeros(20,dtype=torch.float64),'SCALED':active_vector(gg,torch)}
    raw=[grads[h]['SCALED'] for h in p['loss_horizons']]
    raw_mean_norm=float(torch.stack(raw).mean(0).norm())
    neg_pairs=sum(int(float(torch.dot(raw[i],raw[j]))<0) for i in range(4) for j in range(i+1,4))
    if name=='SAL_H20_ONLY':vector=combined_gradients(grads,torch,'H20_ONLY')
    elif name=='SAL_MEAN4':vector=combined_gradients(grads,torch,'MEAN4')
    elif name=='SAL_PCGRAD4':vector=combined_gradients(grads,torch,'PCGRAD4')
    else:raise Hold('UNKNOWN_TRAIN_ARM')
    if name=='SAL_MEAN4' and step==0:
     chk=torch.autograd.grad(sum(per_h.values())/4,(W,bias),allow_unused=False)
     need(bool(torch.allclose(vector,active_vector(chk,torch),rtol=1e-10,atol=1e-11)),'MEAN_GRADIENT_PARITY')
    if name=='SAL_PCGRAD4' and step==0:
     need(bool(torch.equal(vector,combined_gradients(grads,torch,'PCGRAD4'))),'PCGRAD_DETERMINISM')
   need(bool(torch.isfinite(vector).all()),'NONFINITE_TRAIN_GRAD:'+name)
   if name!='SAL_STATE8':
    neg_directions=sum(int(float(torch.dot(vector,grads[h]['SCALED']))<0) for h in p['loss_horizons'])
   norm=float(vector.norm()); maximum=max(maximum,norm);clipcount+=int(norm>p['grad_clip_norm'])
   history[name].append({'step':step+1,'record_index':step%len(records),'norm':norm,'clip':int(norm>p['grad_clip_norm']),
      'raw_mean_norm':raw_mean_norm,'negative_pairs':neg_pairs,'negative_directions':neg_directions,
      'gain8':per_gain.get(8,float('nan')),'gain20':per_gain.get(20,float('nan')),
      'err8':per_e2.get(8,float('nan')),'err20':per_e2.get(20,float('nan'))})
   factor=min(1.,p['grad_clip_norm']/(norm+1e-30))
   v=(vector.detach()*factor)
   opt.zero_grad(set_to_none=True)
   W.grad=torch.zeros_like(W);bias.grad=torch.zeros_like(bias)
   W.grad[[1,5],:9]=v[:18].reshape(2,9);bias.grad[[1,5]]=v[18:]
   opt.step()
   if (step+1) in cfg['r53_diagnostic']['checkpoints']:
    snapshots[name][step+1]=(W.detach().clone(),bias.detach().clone())
   with torch.no_grad():
    need(bool(torch.equal(W[~rowmask.bool()],W0[~rowmask.bool()]) and torch.equal(bias[~bmask.bool()],b0[~bmask.bool()])),'NONACTIVE_ROW_MUTATION:'+name)
   if (step+1)%16==0 or step+1==updates:
    print('[R52] TRAIN '+name+' updates='+str(step+1)+'/'+str(updates)+' loss='+fmt(last)+' grad='+fmt(norm)+' clip='+str(clipcount),flush=True)
  model=copy.deepcopy(base)
  with torch.no_grad():model.gates.weight.copy_(W);model.gates.bias.copy_(bias)
  model.eval();delta=float(torch.sqrt((W-W0).square().sum()+(bias-b0).square().sum()).detach())
  need(delta>0,'ZERO_PARAMETER_UPDATE:'+name)
  models[name]=model;stats[name]={'last':last,'delta':delta,'max_grad':maximum,'clipped':clipcount}
  allfinal[name]=float(last)
 return models,stats,snapshots,history

def prospective_gates(report,rows,cfg,stats):
 p=cfg['prospective'];a=p['scientific_acceptance'];arms=p['arms'];pc='SAL_PCGRAD4';mean4='SAL_MEAN4';base='BASE_LS';h8='SAL_STATE8';h20='SAL_H20_ONLY'
 cases=['r52_weak160','r52_reverse160','r52_anti160'];out={};metrics={}
 for case in cases:
  for opp in (mean4,h8,h20,base):
   q=report[case][pc]['by_seed'];r=report[case][opp]['by_seed'];nm=case.split('_')[1]+'_VS_'+opp
   metrics[nm+'_SEEDS']=sum(q[s]<r[s] for s in q)
   metrics[nm+'_CELLS']=sum(x[0]<y[0] for x,y in zip(rows[pc][case],rows[opp][case]))
 for case in cases:
  alias=case.split('_')[1];row=report[case][pc]
  out[alias+'_seeds_mean']=metrics[alias+'_VS_'+mean4+'_SEEDS']>=a['paired_seeds_min']
  out[alias+'_cells_mean']=metrics[alias+'_VS_'+mean4+'_CELLS']>=a['paired_cells_min']
  out[alias+'_seeds_state']=metrics[alias+'_VS_'+h8+'_SEEDS']>=a['paired_seeds_min']
  out[alias+'_median']=row['median']<=a['weak_median_max' if alias=='weak160' else 'reverse_median_max' if alias=='reverse160' else 'weak_median_max']
  out[alias+'_worst']=row['worst']<=a['weak_worst_max' if alias=='weak160' else 'reverse_worst_max' if alias=='reverse160' else 'anti_worst_max']
  out[alias+'_gain']=a['weak_gain_min' if alias=='weak160' else 'reverse_gain_min' if alias=='reverse160' else 'anti_gain_min']<=row['gain']<=a['weak_gain_max' if alias=='weak160' else 'reverse_gain_max' if alias=='reverse160' else 'anti_gain_max']
 out['signed_fraction']=min(report[case][pc]['sign'] for case in cases)>=a['signed_fraction_min']
 out['clip_budget']=stats[pc]['clipped']<=a['max_clip_updates']
 out['no_regress_anti_base']=report[cases[2]][pc]['worst']<=report[cases[2]][base]['worst']
 out['beat_H20_only_weak_seeds']=metrics['weak160_VS_'+h20+'_SEEDS']>=a['paired_seeds_min']
 return out,metrics


class _ClipCountingStackedGates(_StackedGates):
 """Strictly observational native forward with pre-radius instrumentation."""
 def __init__(self,weights,bias,r30,torch,per_arm):
  super().__init__(weights,bias,r30,torch,per_arm)
  self.hits=torch.zeros(weights.shape[0],dtype=torch.long)
  self.total=0
  self.max_pre=torch.zeros(weights.shape[0],dtype=torch.float64)
 def __call__(self,state,sal,col):
  t=self.torch;a,b=state;n=self.weights.shape[0]
  feat=self.r30.invariants(state,sal,col,t).reshape(n,self.per_arm*4,10)
  logits=t.bmm(feat,self.weights.transpose(1,2))+self.bias[:,None,:]
  z=logits.reshape(n*self.per_arm,4,8)
  k=t.cat((t.sigmoid(z[...,0:1]),t.tanh(z[...,1:4]),t.sigmoid(z[...,4:5]),t.tanh(z[...,5:8])),dim=-1)
  pa=k[...,0,None]*a+k[...,1,None]*sal+k[...,2,None]*col+k[...,3,None]*b
  pb=k[...,4,None]*b+k[...,5,None]*sal+k[...,6,None]*col+k[...,7,None]*a
  na=t.linalg.vector_norm(pa,dim=-1).reshape(n,self.per_arm,4)
  nb=t.linalg.vector_norm(pb,dim=-1).reshape(n,self.per_arm,4)
  self.hits+=(na>4.).sum(dim=(1,2))+(nb>4.).sum(dim=(1,2))
  self.total+=2*self.per_arm*4
  self.max_pre=t.maximum(self.max_pre,t.maximum(na.amax(dim=(1,2)),nb.amax(dim=(1,2))))
  proj=lambda x:x*t.clamp(4.0/(t.linalg.vector_norm(x,dim=-1,keepdim=True)+1e-30),max=1.)
  return proj(pa),proj(pb)

def trace_training_checkpoints(snapshots, records, cfg, r30, r4, torch, preflight=False):
 """No learning, no eval seeds, six frozen training records balanced across 3 cases.
 Independent arms are packed across batch only, never across graph adjacency.
 """
 p=cfg['prospective'];d=cfg['r53_diagnostic'];B=p['adapt_batch'];per=2*B
 chosen=d['preflight_indices'] if preflight else d['probe_indices']
 checkpoints=[0,2] if preflight else d['checkpoints']
 names=d['arms'];case_keys=list(p['train_cases']); graph_count=3
 agg={a:{c:{k:{'gain':[],'e2':[],'floored':[],'state':[],'radius':[],'valid':0}
                  for k in checkpoints for h in []} for c in case_keys} for a in names}
 # Explicit 4-level nesting; no reference to heldout evaluation cases/graphs.
 agg={a:{c:{k:{h:{'gain':[],'e2':[],'floored':[],'state':[],'pre_radius':[],'clip_frac':[],'valid':0}
                     for h in d['supervision_horizons']} for k in checkpoints} for c in case_keys} for a in names}
 parity=0.;tested=0;max_pre=0.
 for index in chosen:
  seed=p['adapt_train_seeds'][index//p['adapt_fixtures_per_seed']]
  i=index%p['adapt_fixtures_per_seed']
  case=case_keys[(i//graph_count+seed)%len(case_keys)]
  adj,initial,targets,refs=records[index]
  tested+=1
  for k in checkpoints:
   W=torch.stack([snapshots[a][k][0] for a in names]);bias=torch.stack([snapshots[a][k][1] for a in names]);n=len(names)
   law=_ClipCountingStackedGates(W,bias,r30,torch,per)
   packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
   with torch.no_grad():
    for h in range(1,max(d['supervision_horizons'])+1):
     packed=r4.graph_step(packed,adj,law,torch)
     if h not in d['supervision_horizons']:continue
     targetA,targetB=targets[h-1];target=refs[h];den=target.square().sum(-1)
     valid=den>1e-25
     for armidx,name in enumerate(names):
      ss=slice(armidx*per,(armidx+1)*per)
      sA=packed[0][ss];sB=packed[1][ss]
      pred=sA[:B,3,:]-sA[B:,3,:]
      if bool(valid.any()):
       gain=float(((pred*target).sum(-1)[valid]/den[valid]).mean())
       e2=float(((pred-target).square().sum(-1)[valid]/den[valid]).mean())
       agg[name][case][k][h]['gain'].append(gain)
       agg[name][case][k][h]['e2'].append(e2)
       agg[name][case][k][h]['valid']+=int(valid.sum())
      agg[name][case][k][h]['floored'].append(float(((pred-target).square().sum(-1)/(den+.1)).mean()))
      agg[name][case][k][h]['state'].append(float(.5*((sA-targetA).square().mean()+(sB-targetB).square().mean())))
      agg[name][case][k][h]['pre_radius'].append(float(law.max_pre[armidx]))
      agg[name][case][k][h]['clip_frac'].append(float(law.hits[armidx])/max(1,law.total))
   if preflight and k==2 and index==chosen[0]:
    # Strict forward-parity test of the observational clipping instrument.
    origlaw=_StackedGates(W,bias,r30,torch,per)
    control=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
    with torch.no_grad():
     for _ in range(20):control=r4.graph_step(control,adj,origlaw,torch)
    need(all(torch.equal(control[kk],packed[kk]) for kk in (0,1)),'CLIP_INSTRUMENT_FORWARD_PARITY')
 return agg,tested

def summarized_training_checkpoints(agg,cfg):
 out={}
 for arm,cases in agg.items():
  out[arm]={}
  for case,steps in cases.items():
   out[arm][case]={}
   for ck,hs in steps.items():
    out[arm][case][ck]={}
    for h,vals in hs.items():
     row={key:mean(v) if v else float('nan') for key,v in vals.items() if key!='valid'}
     row['valid']=vals['valid'];out[arm][case][ck][h]=row
 return out

def summarize_training_updates(history):
 out={}
 for arm,rows in history.items():
  out[arm]={}
  for lo,hi in ((1,16),(17,32),(33,48),(49,64)):
   block=[x for x in rows if lo<=x['step']<=hi]
   if not block:continue
   out[arm][(lo,hi)]={'grad':mean(x['norm'] for x in block),
    'clip':sum(x['clip'] for x in block),
    'negative_pairs':sum(x['negative_pairs'] for x in block),
    'negative_directions':sum(x['negative_directions'] for x in block),
    'gain8':mean(x['gain8'] for x in block),
    'gain20':mean(x['gain20'] for x in block)}
 return out



def r54_factorial_test(seed, models, cfg, r3, r4, r30, torch, teacher, basegraphs, basecases, check_direct=False):
 """Fresh paired-seed factorial; frozen weights, batch over all five original R52 arms."""
 f=cfg['r54_factorial'];ev=cfg['prospective']['evaluations']
 arms=f['arms']; B=f['eval_batch']; per=2*B; n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 outputs={(topo,amp,case,arm):[] for topo in f['topology'] for amp in f['amplitudes'] for case in f['cases'] for arm in arms}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 extra={k:tuple(tuple(v) for v in ev['graphs'][k]) for k in f['topology']['NOVEL']}
 graphs={**basegraphs,**extra}; maxcov=0.;disc=0.;minref=1e30;maxdirect=0.;pair_seed_parity=0.
 with torch.no_grad():
  for topo,graphnames in f['topology'].items():
   for gi,graph in enumerate(graphnames):
    for ci,case in enumerate(f['cases']):
     for amp,s in f['amplitudes'].items():
      spec= (s,4.,'antialigned') if case=='anti' else (4.,s,'independent') if case=='reverse' else (s,4.,'independent')
      # Same RNG seed across amplitude variants, preserving graph/seed/case alignment.
      rng=f['eval_offset']+seed*100109+gi*293+ci*59+(0 if topo=='FAMILIAR' else 701)
      mapped,mats,init,flip=r3.generate(graph,'r54_case',rng,B,graphs,{**basecases,'r54_case':spec},torch)
      adj=r4.build_links(mapped,mats,torch)
      initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
      tstate=(initial[0].clone(),initial[1].clone())
      packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
      for h in range(20):
       tstate=r4.graph_step(tstate,adj,teacher,torch)
       packed=r4.graph_step(packed,adj,law,torch)
      ref=tstate[0][:B,3,:]-tstate[0][B:,3,:]
      den=ref.square().sum(-1)
      need(bool((den>1e-25).all()),'REFERENCE_ZERO:'+topo+':'+amp+':'+case+':'+graph)
      minref=min(minref,float(ref.norm(dim=-1).min()))
      for j,a in enumerate(arms):
       x=slice(per*j,per*(j+1));st=(packed[0][x],packed[1][x]);pred=st[0][:B,3,:]-st[0][B:,3,:]
       err=float(((pred-ref).square().sum(-1)/den).mean());gain=float(((pred*ref).sum(-1)/den).mean());sign=float((((pred*ref).sum(-1)/den)>0).double().mean())
       need(all(math.isfinite(z) for z in (err,gain,sign)),'NONFINITE:'+a)
       outputs[(topo,amp,case,a)].append((err,gain,sign,seed,graph))
       if check_direct and topo=='FAMILIAR' and gi==0 and ci==0 and amp=='FAMILIAR64':
        dr=r4.rollout(initial,adj,models[a],20,torch)[-1]
        maxdirect=max(maxdirect,*[float((dr[k]-st[k]).abs().max()) for k in (0,1)])
       if gi==0 and ci==0 and amp=='FAMILIAR64' and topo=='FAMILIAR' and (seed==f['evaluation_seeds'][0] or check_direct):
        gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
        gadj=r4.links_gauge(adj,gauges,torch)
        rot=r4.rollout(r4.gauge_transform(initial,gauges,torch),gadj,models[a],20,torch)[-1]
        expected=r4.gauge_transform(st,gauges,torch)
        maxcov=max(maxcov,*[float((rot[k]-expected[k]).abs().max()) for k in (0,1)])
  # Retain disconnected isolation on the exact R52 disconnected fixture, including all frozen models.
  g=ev['disconnected_graph'];graphs[g]=tuple(tuple(row) for row in ev['graphs'][g]);mapped,mats,init,flip=r3.generate(g,'r54_case',f['eval_offset']+seed*100109+8891,B,graphs,{**basecases,'r54_case':(0.0625,4.,'independent')},torch)
  adj=r4.build_links(mapped,mats,torch);initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  tr=r4.rollout(initial,adj,teacher,20,torch)[-1]
  disc=max(disc,float(tr[0][:,3,:].abs().max()))
  for a in arms:
   st=r4.rollout(initial,adj,models[a],20,torch)[-1]
   disc=max(disc,float(st[0][:,3,:].abs().max()))
 for a in arms:need(r4.model_hash(models[a])==hashes[a],'WEIGHT_MUTATION:'+a)
 return outputs,maxcov,disc,minref,maxdirect

def r54_summarize(rows,cfg):
 f=cfg['r54_factorial'];out={}
 for topo in f['topology']:
  for amp in f['amplitudes']:
   for case in f['cases']:
    for arm in f['arms']:
     data=rows[(topo,amp,case,arm)];need(len(data)>0,'EMPTY_STRATUM')
     err=[x[0] for x in data]
     out[(topo,amp,case,arm)]={'e2':mean(err),'median':statistics.median(err),'worst':max(err),'gain':mean(x[1] for x in data),'sign':mean(x[2] for x in data),
      'by_seed':{s:mean(x[0] for x in data if x[3]==s) for s in sorted({x[3] for x in data})}}
 return out

def r54_contrast(report,cfg):
 f=cfg['r54_factorial'];out={}
 for arm in f['arms']:
  for case in f['cases']:
   E=lambda t,a:report[(t,a,case,arm)]['e2']
   fam='FAMILIAR';nov='NOVEL';lo='FAMILIAR64';hi='NOVEL160'
   # log1p(error) interaction avoids unstable zero/small-error ratios; descriptive only.
   interaction=(math.log1p(E(nov,hi))-math.log1p(E(nov,lo)))-(math.log1p(E(fam,hi))-math.log1p(E(fam,lo)))
   out[(case,arm)]=interaction
 return out



def graph_structure(g):
 """The generator admits four relation rows; multiplicities count shared participant incidences."""
 edges=[set(x) for x in g]
 need(len(edges)==4 and all(len(x)==4 for x in edges),'R55_RELATION_ARITY')
 w={(i,j):len(edges[i].intersection(edges[j])) for i in range(4) for j in range(i+1,4)}
 degrees={n:sum(n in x for x in edges) for n in set.union(*edges)}
 direct=w[(0,3)]
 twohop=w[(0,1)]*w[(1,3)]+w[(0,2)]*w[(2,3)]
 threehop=w[(0,1)]*w[(1,2)]*w[(2,3)]+w[(0,2)]*w[(1,2)]*w[(1,3)]
 return {'W':w,'DIRECT':direct,'TWOHOP':twohop,'THREEHOP':threehop,'REMOTE_IN':sum(w[(i,3)] for i in range(3)),
         'PARTICIPANT_DEGREES':sorted(degrees.values()),'DEGREE_HIST':{str(k):sum(v==k for v in degrees.values()) for k in (1,2,3,4)}}

def r55_structural_test(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,check_direct=False):
 f=cfg['r55_structural'];arms=f['arms'];B=f['eval_batch'];per=2*B;n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in graph) for k,graph in f['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in f['cases'].items()}}
 outputs={(graph,case,arm):[] for graph in f['graphs'] for case in f['cases'] for arm in arms}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 maxcov=0.;disc=0.;minref=1e30;maxdirect=0.
 with torch.no_grad():
  for gi,graph in enumerate(f['graphs']):
   for ci,case in enumerate(f['cases']):
    rng=f['eval_offset']+seed*100109+ci*59  # paired across all four rewires
    mapped,mats,init,flip=r3.generate(graph,case,rng,B,graphs,cases,torch)
    adj=r4.build_links(mapped,mats,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    ref=(initial[0].clone(),initial[1].clone())
    packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
    for _ in range(f['horizon']):
     ref=r4.graph_step(ref,adj,teacher,torch)
     packed=r4.graph_step(packed,adj,law,torch)
    rr=ref[0][:B,3,:]-ref[0][B:,3,:]; den=rr.square().sum(-1)
    need(bool((den>1e-25).all()),'R55_REF_ZERO:'+graph+':'+case)
    minref=min(minref,float(rr.norm(dim=-1).min()))
    for j,a in enumerate(arms):
     ss=slice(per*j,per*(j+1));st=(packed[0][ss],packed[1][ss]);pr=st[0][:B,3,:]-st[0][B:,3,:]
     e=float(((pr-rr).square().sum(-1)/den).mean());gain=float(((pr*rr).sum(-1)/den).mean());sign=float((((pr*rr).sum(-1)/den)>0).double().mean())
     need(all(math.isfinite(z) for z in (e,gain,sign)),'R55_NONFINITE:'+a+':'+graph+':'+case)
     outputs[(graph,case,a)].append((e,gain,sign,seed,graph))
     if check_direct and graph=='D1_P2' and case=='weak':
      direct=r4.rollout(initial,adj,models[a],20,torch)[-1]
      maxdirect=max(maxdirect,*[float((direct[k]-st[k]).abs().max()) for k in (0,1)])
     if graph=='D1_P2' and case=='weak' and (seed==f['evaluation_seeds'][0] or check_direct):
      gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
      gadj=r4.links_gauge(adj,gauges,torch)
      rot=r4.rollout(r4.gauge_transform(initial,gauges,torch),gadj,models[a],20,torch)[-1]
      exp=r4.gauge_transform(st,gauges,torch)
      maxcov=max(maxcov,*[float((rot[k]-exp[k]).abs().max()) for k in (0,1)])
  # Original R52 disconnected fixture remains a negative control on all arms.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g]);
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  adj=r4.build_links(mapped,mats,torch);initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  tr=r4.rollout(initial,adj,teacher,20,torch)[-1];disc=max(disc,float(tr[0][:,3,:].abs().max()))
  for a in arms:
   st=r4.rollout(initial,adj,models[a],20,torch)[-1];disc=max(disc,float(st[0][:,3,:].abs().max()))
 for a in arms:need(r4.model_hash(models[a])==hashes[a],'WEIGHT_MUTATION:'+a)
 return outputs,maxcov,disc,minref,maxdirect

def r55_summarize(rows,cfg):
 f=cfg['r55_structural'];out={}
 for g in f['graphs']:
  for c in f['cases']:
   for a in f['arms']:
    v=rows[(g,c,a)];need(v,'R55_EMPTY_STRATUM')
    out[(g,c,a)]={'e2':mean(x[0] for x in v),'median':statistics.median(x[0] for x in v),'worst':max(x[0] for x in v),
      'gain':mean(x[1] for x in v),'sign':mean(x[2] for x in v),'by_seed':{s:mean(x[0] for x in v if x[3]==s) for s in sorted(set(x[3] for x in v))}}
 return out

def r55_contrast(report,cfg):
 f=cfg['r55_structural'];out={}
 for c in f['cases']:
  for a in f['arms']:
   E=lambda d,p:report[('D'+str(d)+'_P'+str(p),c,a)]['e2']
   out[(c,a)]={'direct_at_P1':math.log1p(E(2,1))-math.log1p(E(1,1)),
     'direct_at_P2':math.log1p(E(2,2))-math.log1p(E(1,2)),
     'twohop_at_D1':math.log1p(E(1,2))-math.log1p(E(1,1)),
     'twohop_at_D2':math.log1p(E(2,2))-math.log1p(E(2,1)),
     'interaction':(math.log1p(E(2,2))-math.log1p(E(1,2)))-(math.log1p(E(2,1))-math.log1p(E(1,1)))}
 return out



def transport_matrices(mapped,base_mats,mode,cfg,r3,torch):
 """Source relation (r=0) is unchanged by both interventions."""
 f=cfg['r56_transport'];s=f['nonflat'];out={}
 for (r,p),matrix in base_mats.items():
  if mode=='BASE_SEPARABLE':rot=r3.so2x2_torch(0.,0.,torch)
  elif mode=='GAUGE_ONLY':rot=r3.so2x2_torch(f['gauge_shift'][0]*r, f['gauge_shift'][1]*r,torch)
  elif mode=='PARTICIPANT_NONFLAT':
   rot=r3.so2x2_torch(s['a_theta1']*(r/s['r_scale'])*math.sin(s['p_factor_1']*(p+1)),
                        s['a_theta2']*(r/s['r_scale'])*math.cos(s['p_factor_2']*(p+1)),torch)
  else:raise Hold('UNKNOWN_TRANSPORT_MODE:'+mode)
  out[(r,p)]=rot@matrix
  if r==0:need(torch.equal(out[(r,p)],matrix),'SOURCE_FRAME_MUTATED:'+mode)
 return out

def pair_transport_spread(mapped,mats,r4,torch):
 """Maximum Frobenius-distance of links sharing the same ordered (src,dst) relation pair."""
 links=r4.build_links(mapped,mats,torch);spread=0.;double=0
 for d,(idx,ops) in links.items():
  for s in set(idx.tolist()):
   group=ops[idx==s]
   if len(group)>1:
    double+=1
    spread=max(spread,float(torch.linalg.vector_norm(group[1:]-group[0],dim=(-2,-1)).max()))
 return spread,double

def r56_counterfactual_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,check_direct=False):
 f=cfg['r56_transport'];s=cfg['r55_structural'];B=f['eval_batch'];per=2*B;arms=f['arms'];n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in g) for k,g in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 output={};maxcov=0.;disc=0.;minref=1e30;maxdirect=0.;maxgauge=0.;maxsource=0.;maxbase=0.;maxnonflat=0.;doublepairs=0
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    rng=f['eval_offset']+seed*100109+ci*59
    mapped,mats,init,flip=r3.generate(g,c,rng,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:maxsource=max(maxsource,max(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    baseline={}
    for mode in f['transport_modes']:
     changed=transport_matrices(mapped,mats,mode,cfg,r3,torch)
     adj=r4.build_links(mapped,changed,torch)
     if mode=='BASE_SEPARABLE' or mode=='PARTICIPANT_NONFLAT':
      spread,cnt=pair_transport_spread(mapped,changed,r4,torch)
      if mode=='BASE_SEPARABLE':maxbase=max(maxbase,spread)
      else:maxnonflat=max(maxnonflat,spread);doublepairs=max(doublepairs,cnt)
     reference=(initial[0].clone(),initial[1].clone())
     packed=(initial[0].repeat(n,1,1),initial[1].repeat(n,1,1))
     for _ in range(f['horizon']):
      reference=r4.graph_step(reference,adj,teacher,torch)
      packed=r4.graph_step(packed,adj,law,torch)
     rr=reference[0][:B,3,:]-reference[0][B:,3,:];den=rr.square().sum(-1)
     need(bool((den>1e-25).all()),'R56_REFERENCE_ZERO:'+g+':'+c+':'+mode)
     minref=min(minref,float(rr.norm(dim=-1).min()))
     if mode=='GAUGE_ONLY':
      gauges={r:r3.so2x2_torch(f['gauge_shift'][0]*r,f['gauge_shift'][1]*r,torch) for r in range(4)}
      tr=r4.gauge_transform(baseline['teacher'],gauges,torch)
      maxgauge=max(maxgauge,max(float((reference[k]-tr[k]).abs().max()) for k in (0,1)))
     else:
      gauges=None
     for j,a in enumerate(arms):
      ss=slice(per*j,per*(j+1));st=(packed[0][ss],packed[1][ss]);pr=st[0][:B,3,:]-st[0][B:,3,:]
      d=(pr-rr).square().sum(-1)/den;gs=(pr*rr).sum(-1)/den
      error=float(d.mean());gain=float(gs.mean());sign=float((gs>0).double().mean());worst=float(d.max())
      need(all(math.isfinite(z) for z in (error,gain,sign,worst)),'R56_METRICS_NONFINITE:'+g+':'+c+':'+mode+':'+a)
      output[(g,c,mode,a)]=(error,gain,sign,worst,seed)
      if mode=='BASE_SEPARABLE':baseline[a]=st
      if mode=='GAUGE_ONLY':
       expected=r4.gauge_transform(baseline[a],gauges,torch)
       maxgauge=max(maxgauge,max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
       old=output[(g,c,'BASE_SEPARABLE',a)]
       maxgauge=max(maxgauge,max(abs(x-y)/(1.+abs(x)+abs(y)) for x,y in zip(old[:4],(error,gain,sign,worst))))
      if check_direct and g=='D1_P2' and c=='weak' and mode in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT'):
       direct=r4.rollout(initial,adj,models[a],20,torch)[-1]
       maxdirect=max(maxdirect,max(float((direct[k]-st[k]).abs().max()) for k in (0,1)))
      if g=='D1_P2' and c=='weak' and mode=='PARTICIPANT_NONFLAT' and (check_direct or seed==f['evaluation_seeds'][0]):
       gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
       gadj=r4.links_gauge(adj,gauges,torch)
       rot=r4.rollout(r4.gauge_transform(initial,gauges,torch),gadj,models[a],20,torch)[-1]
       expected=r4.gauge_transform(st,gauges,torch)
       maxcov=max(maxcov,max(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
     if mode=='BASE_SEPARABLE':baseline['teacher']=reference
  # Disconnected fixture from R52, testing altered transport too.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph']
  graphs[g]=tuple(tuple(v) for v in ev['graphs'][g]);mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  for mode in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT'):
   adj=r4.build_links(mapped,transport_matrices(mapped,mats,mode,cfg,r3,torch),torch)
   tr=r4.rollout(initial,adj,teacher,20,torch)[-1]
   disc=max(disc,float(tr[0][:,3,:].abs().max()))
   for a in arms:
    st=r4.rollout(initial,adj,models[a],20,torch)[-1]
    disc=max(disc,float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R56_WEIGHT_MUTATION:'+a)
 return output,{'cov':maxcov,'disconnected':disc,'min_ref':minref,'parity':maxdirect,'gauge':maxgauge,'source':maxsource,'base_spread':maxbase,'nonflat_spread':maxnonflat,'double_pair':doublepairs}

def r56_summarize(rows,cfg):
 f=cfg['r56_transport'];out={}
 for g in f['graphs']:
  for c in f['cases']:
   for m in f['transport_modes']:
    for a in f['arms']:
     r=rows[(g,c,m,a)];need(len(r)>0,'R56_EMPTY_CELL')
     out[(g,c,m,a)]={'e2':mean(x[0] for x in r),'gain':mean(x[1] for x in r),
        'sign':mean(x[2] for x in r),'worst':max(x[3] for x in r),
        'seed_e2':{x[4]:x[0] for x in r}}
 return out

def r56_aggregate(report,cfg):
 f=cfg['r56_transport'];out={}
 for c in f['cases']:
  for a in f['arms']:
   for mode in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT'):
    cells=[report[(g,c,mode,a)] for g in f['graphs']]
    out[(c,a,mode)]={'e2':mean(x['e2'] for x in cells),'gain':mean(x['gain'] for x in cells),
      'better_seed_graph_cells':None}
   b=out[(c,a,'BASE_SEPARABLE')];z=out[(c,a,'PARTICIPANT_NONFLAT')]
   b['delta_log1p']=math.log1p(z['e2'])-math.log1p(b['e2'])
   b['nonflat_better_paired']=sum(report[(g,c,'PARTICIPANT_NONFLAT',a)]['seed_e2'][seed] < report[(g,c,'BASE_SEPARABLE',a)]['seed_e2'][seed] for g in f['graphs'] for seed in f['evaluation_seeds'])
 return out



def graph_step_split(state,adj_a,adj_b,model,torch):
 """Exact native graph_step algebra with independent salience and collective edge operators."""
 a,b=state;sal_all=[];col_all=[]
 for d in range(4):
  if d not in adj_a:sal_all.append(torch.zeros_like(a[:,d,:]))
  else:
   idx,ops=adj_a[d];w=torch.einsum('eij,bej->bei',ops,a[:,idx,:])
   sal_all.append((torch.softmax(8.*torch.linalg.vector_norm(w,dim=-1),dim=1)[...,None]*w).sum(dim=1))
  if d not in adj_b:col_all.append(torch.zeros_like(b[:,d,:]))
  else:
   idx,ops=adj_b[d];wc=torch.einsum('eij,bej->bei',ops,b[:,idx,:])
   col_all.append(wc.mean(dim=1))
 return model((a,b),torch.stack(sal_all,dim=1),torch.stack(col_all,dim=1))

def prepare_mode_adj(mode_adjs,torch):
 """All transport modes share membership; only per-edge transport differs."""
 out={}
 for d in range(4):
  if d not in mode_adjs[0]:
   need(all(d not in adj for adj in mode_adjs),'R57_MIXED_EDGE_MEMBERSHIP')
   continue
  idx=mode_adjs[0][d][0]
  need(all(d in adj and torch.equal(idx,adj[d][0]) for adj in mode_adjs),'R57_EDGE_MEMBERSHIP_DRIFT')
  out[d]=(idx,torch.stack([adj[d][1] for adj in mode_adjs]))
 return out

def graph_step_modes(state,adja,adjb,model,torch,per_mode):
 """Exact split-forward for independent transport modes, vectorized along mode axis."""
 a,b=state;M=a.shape[0]//per_mode
 aa=a.reshape(M,per_mode,4,4);bb=b.reshape(M,per_mode,4,4)
 sal=[];col=[]
 for d in range(4):
  if d not in adja:sal.append(torch.zeros_like(aa[:,:,d,:]))
  else:
   idx,ops=adja[d];w=torch.einsum('meij,mbej->mbei',ops,aa[:,:,idx,:])
   sal.append((torch.softmax(8.*torch.linalg.vector_norm(w,dim=-1),dim=2)[...,None]*w).sum(dim=2))
  if d not in adjb:col.append(torch.zeros_like(bb[:,:,d,:]))
  else:
   idx,ops=adjb[d];wc=torch.einsum('meij,mbej->mbei',ops,bb[:,:,idx,:])
   col.append(wc.mean(dim=2))
 state=(a,b);sa=torch.stack(sal,dim=2).reshape(M*per_mode,4,4);cb=torch.stack(col,dim=2).reshape(M*per_mode,4,4)
 return model(state,sa,cb)

def split_rollout(initial,adj_a,adj_b,model,steps,torch):
 state=initial
 for _ in range(steps):state=graph_step_split(state,adj_a,adj_b,model,torch)
 return state

def r57_split_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
 f=cfg['r57_channels'];tf=cfg['r56_transport'];s=cfg['r55_structural'];B=f['eval_batch'];per=2*B;arms=f['arms'];n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in graph) for k,graph in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 rows={};diag={'cov':0.,'disconnected':0.,'min_ref':1e30,'parity':0.,'gauge':0.,'source':0.,'native':0.,'spread':0.,'min_teacher_base':1e30,'max_teacher_response_ratio':0.}
 modes=f['modes'];mp={'BASE_AB':('BASE_SEPARABLE','BASE_SEPARABLE'), 'NONFLAT_A':('PARTICIPANT_NONFLAT','BASE_SEPARABLE'), 'NONFLAT_B':('BASE_SEPARABLE','PARTICIPANT_NONFLAT'),'NONFLAT_AB':('PARTICIPANT_NONFLAT','PARTICIPANT_NONFLAT'),'GAUGE_AB':('GAUGE_ONLY','GAUGE_ONLY')}
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    mapped,mats,init,flip=r3.generate(g,c,f['eval_offset']+seed*100109+ci*59,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:diag['source']=max(diag['source'],max(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    a_mats={name:transport_matrices(mapped,mats,name,cfg,r3,torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT','GAUGE_ONLY')}
    edges={name:r4.build_links(mapped,ms,torch) for name,ms in a_mats.items()}
    spread,cnt=pair_transport_spread(mapped,a_mats['PARTICIPANT_NONFLAT'],r4,torch)
    diag['spread']=max(diag['spread'],spread)
    # Some degree-preserving graphs lack repeated links; require the witness across the complete family.
    baseline={}
    channel_pairs=[(edges[mp[mode][0]],edges[mp[mode][1]]) for mode in modes]
    prep_a=prepare_mode_adj([x[0] for x in channel_pairs],torch)
    prep_b=prepare_mode_adj([x[1] for x in channel_pairs],torch)
    M=len(modes)
    reference_all=(initial[0].repeat(M,1,1),initial[1].repeat(M,1,1))
    packed_all=(initial[0].repeat(M*n,1,1),initial[1].repeat(M*n,1,1))
    mode_law=_StackedGates(W.repeat(M,1,1),bias.repeat(M,1),r30,torch,per)
    for _ in range(f['horizon']):
     reference_all=graph_step_modes(reference_all,prep_a,prep_b,teacher,torch,per)
     packed_all=graph_step_modes(packed_all,prep_a,prep_b,mode_law,torch,n*per)
    for mi,mode in enumerate(modes):
     a_mode,b_mode=mp[mode];adj_a,adj_b=channel_pairs[mi]
     reference=(reference_all[0][mi*per:(mi+1)*per],reference_all[1][mi*per:(mi+1)*per])
     packed=(packed_all[0][mi*n*per:(mi+1)*n*per],packed_all[1][mi*n*per:(mi+1)*n*per])
     rr=paired_delta(reference,B);den=rr.square().sum(-1)
     need(bool((den>1e-25).all()),'R57_ZERO_REFERENCE:'+g+':'+c+':'+mode)
     rrnorm=float(rr.norm(dim=-1).mean())
     diag['min_ref']=min(diag['min_ref'],float(rr.norm(dim=-1).min()))
     if mode=='BASE_AB':diag['min_teacher_base']=min(diag['min_teacher_base'],rrnorm)
     else:diag['max_teacher_response_ratio']=max(diag['max_teacher_response_ratio'],rrnorm/max(baseline['ref_norm'],1e-25))
     if mode=='GAUGE_AB':
      gauges={r:r3.so2x2_torch(tf['gauge_shift'][0]*r,tf['gauge_shift'][1]*r,torch) for r in range(4)}
      expected=r4.gauge_transform(baseline['teacher'],gauges,torch)
      diag['gauge']=max(diag['gauge'],max(float((reference[k]-expected[k]).abs().max()) for k in (0,1)))
     for j,a in enumerate(arms):
      sl=slice(j*per,(j+1)*per);st=(packed[0][sl],packed[1][sl]);pr=paired_delta(st,B)
      e=(pr-rr).square().sum(-1)/den;pg=(pr*rr).sum(-1)/den
      stats=(float(e.mean()),float(pg.mean()),float((pg>0).double().mean()),float(e.max()),rrnorm,seed)
      need(all(math.isfinite(z) for z in stats[:5]),'R57_METRICS_NONFINITE:'+g+':'+c+':'+mode+':'+a)
      rows[(g,c,mode,a)]=stats
      if mode=='BASE_AB':baseline[a]=st
      if mode=='GAUGE_AB':
       expected=r4.gauge_transform(baseline[a],gauges,torch)
       diag['gauge']=max(diag['gauge'],max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
       old=rows[(g,c,'BASE_AB',a)]
       diag['gauge']=max(diag['gauge'],max(abs(v-w)/(1.+abs(v)+abs(w)) for v,w in zip(stats[:5],old[:5])))
      if g=='D1_P2' and c=='weak' and (preflight or seed==f['eval_seeds'][0]):
       if mode in ('BASE_AB','NONFLAT_AB'):
        native=r4.rollout(initial,adj_a,models[a],f['horizon'],torch)[-1]
        diag['native']=max(diag['native'],max(float((st[k]-native[k]).abs().max()) for k in (0,1)))
       independent=split_rollout(initial,adj_a,adj_b,models[a],f['horizon'],torch)
       diag['parity']=max(diag['parity'],max(float((st[k]-independent[k]).abs().max()) for k in (0,1)))
       if mode in ('NONFLAT_A','NONFLAT_B') and a in ('SAL_STATE8','SAL_PCGRAD4'):
        gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
        gadj_a=r4.links_gauge(adj_a,gauges,torch);gadj_b=r4.links_gauge(adj_b,gauges,torch)
        rot=split_rollout(r4.gauge_transform(initial,gauges,torch),gadj_a,gadj_b,models[a],f['horizon'],torch)
        expected=r4.gauge_transform(st,gauges,torch)
        diag['cov']=max(diag['cov'],max(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
     if mode=='BASE_AB':baseline['teacher']=reference;baseline['ref_norm']=rrnorm
  # Both-channel and split-channel disconnected isolation: no path from source to remote.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g])
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  edge_modes={name:r4.build_links(mapped,transport_matrices(mapped,mats,name,cfg,r3,torch),torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT')}
  for mode in modes[:-1]:
   a_mode,b_mode=mp[mode];adj_a=edge_modes[a_mode];adj_b=edge_modes[b_mode]
   tr=split_rollout(initial,adj_a,adj_b,teacher,20,torch)
   diag['disconnected']=max(diag['disconnected'],float(tr[0][:,3,:].abs().max()))
   for a in ('SAL_STATE8','SAL_PCGRAD4'):
    st=split_rollout(initial,adj_a,adj_b,models[a],20,torch)
    diag['disconnected']=max(diag['disconnected'],float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R57_WEIGHT_MUTATION:'+a)
 return rows,diag
def r58_decompose_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
 f=cfg['r58_reference'];tf=cfg['r56_transport'];s=cfg['r55_structural'];B=f['eval_batch'];per=2*B;arms=f['arms'];n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in graph) for k,graph in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 rows={};diag={'cov':0.,'disconnected':0.,'min_ref':1e30,'parity':0.,'gauge':0.,'source':0.,'native':0.,'spread':0.,'min_teacher_base':1e30,'max_teacher_response_ratio':0.,'decomp':0.,'metric':0.,'min_base_ref':1e30,'min_model_norm':1e30}
 modes=f['modes'];mp={'BASE_AB':('BASE_SEPARABLE','BASE_SEPARABLE'), 'NONFLAT_A':('PARTICIPANT_NONFLAT','BASE_SEPARABLE'), 'NONFLAT_B':('BASE_SEPARABLE','PARTICIPANT_NONFLAT'),'NONFLAT_AB':('PARTICIPANT_NONFLAT','PARTICIPANT_NONFLAT'),'GAUGE_AB':('GAUGE_ONLY','GAUGE_ONLY')}
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    mapped,mats,init,flip=r3.generate(g,c,f['eval_offset']+seed*100109+ci*59,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:diag['source']=max(diag['source'],max(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    a_mats={name:transport_matrices(mapped,mats,name,cfg,r3,torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT','GAUGE_ONLY')}
    edges={name:r4.build_links(mapped,ms,torch) for name,ms in a_mats.items()}
    spread,cnt=pair_transport_spread(mapped,a_mats['PARTICIPANT_NONFLAT'],r4,torch)
    diag['spread']=max(diag['spread'],spread)
    # Some degree-preserving graphs lack repeated links; require the witness across the complete family.
    baseline={}
    channel_pairs=[(edges[mp[mode][0]],edges[mp[mode][1]]) for mode in modes]
    prep_a=prepare_mode_adj([x[0] for x in channel_pairs],torch)
    prep_b=prepare_mode_adj([x[1] for x in channel_pairs],torch)
    M=len(modes)
    reference_all=(initial[0].repeat(M,1,1),initial[1].repeat(M,1,1))
    packed_all=(initial[0].repeat(M*n,1,1),initial[1].repeat(M*n,1,1))
    mode_law=_StackedGates(W.repeat(M,1,1),bias.repeat(M,1),r30,torch,per)
    for _ in range(f['horizon']):
     reference_all=graph_step_modes(reference_all,prep_a,prep_b,teacher,torch,per)
     packed_all=graph_step_modes(packed_all,prep_a,prep_b,mode_law,torch,n*per)
    for mi,mode in enumerate(modes):
     a_mode,b_mode=mp[mode];adj_a,adj_b=channel_pairs[mi]
     reference=(reference_all[0][mi*per:(mi+1)*per],reference_all[1][mi*per:(mi+1)*per])
     packed=(packed_all[0][mi*n*per:(mi+1)*n*per],packed_all[1][mi*n*per:(mi+1)*n*per])
     rr=paired_delta(reference,B);den=rr.square().sum(-1)
     need(bool((den>1e-25).all()),'R58_ZERO_REFERENCE:'+g+':'+c+':'+mode)
     rrnorm=float(rr.norm(dim=-1).mean())
     diag['min_ref']=min(diag['min_ref'],float(rr.norm(dim=-1).min()))
     if mode=='BASE_AB':diag['min_teacher_base']=min(diag['min_teacher_base'],rrnorm)
     else:diag['max_teacher_response_ratio']=max(diag['max_teacher_response_ratio'],rrnorm/max(baseline['ref_norm'],1e-25))
     if mode=='GAUGE_AB':
      gauges={r:r3.so2x2_torch(tf['gauge_shift'][0]*r,tf['gauge_shift'][1]*r,torch) for r in range(4)}
      expected=r4.gauge_transform(baseline['teacher'],gauges,torch)
      diag['gauge']=max(diag['gauge'],max(float((reference[k]-expected[k]).abs().max()) for k in (0,1)))
     for j,a in enumerate(arms):
      sl=slice(j*per,(j+1)*per);st=(packed[0][sl],packed[1][sl]);pr=paired_delta(st,B)
      e=(pr-rr).square().sum(-1)/den;pg=(pr*rr).sum(-1)/den
      stats=(float(e.mean()),float(pg.mean()),float((pg>0).double().mean()),float(e.max()),rrnorm,seed)
      need(all(math.isfinite(z) for z in stats[:5]),'R58_METRICS_NONFINITE:'+g+':'+c+':'+mode+':'+a)
      # Identical source-flip realization and relation-3 frame for all four physical/synthetic modes.
      # Gauge-only is independently verified against gauge-transformed baseline, never compared unrotated.
      if mode!='GAUGE_AB':
       r0=rr if mode=='BASE_AB' else baseline['r0']
       p0=pr if mode=='BASE_AB' else baseline['p0_'+a]
       dT=rr-r0;dP=pr-p0;den0=r0.square().sum(-1)
       need(bool((den0>1e-25).all()),'R58_ZERO_FIXED_REFERENCE:'+g+':'+c)
       dot=lambda x,y:(x*y).sum(-1)
       sq=lambda x:dot(x,x)
       base_err=sq(p0-r0)/den0;mo=sq(pr-r0)/den0;to=sq(p0-rr)/den0
       joint=sq(pr-rr)/den0;matched=sq(pr-rr)/den
       teacher_shift=sq(dT)/den0;model_shift=sq(dP)/den0
       inter=joint-mo-to+base_err;cross=-2.*dot(dP,dT)/den0
       target_cross=joint-mo-teacher_shift;cross2=-2.*dot(pr-r0,dT)/den0
       nratio=den/den0
       # A near-zero model vector can have undefined direction; zero as explicit convention.
       cosine=lambda x,y: dot(x,y)/(torch.sqrt(sq(x)*sq(y))+1e-30)
       cs=cosine(pr,rr);cf=cosine(pr,r0);cd=cosine(dP,dT)
       check=max(float((inter-cross).abs().max()),float((target_cross-cross2).abs().max()),float((joint-matched*nratio).abs().max()))
       diag['decomp']=max(diag['decomp'],check)
       diag['metric']=max(diag['metric'],abs(float(matched.mean())-stats[0]))
       diag['min_base_ref']=min(diag['min_base_ref'],float(torch.sqrt(den0).min()))
       diag['min_model_norm']=min(diag['min_model_norm'],float(torch.sqrt(sq(pr)).min()))
       # One tuple per cell: each metric is a per-example mean (not mean-of-norms squared).
       rows[(g,c,mode,a)]=tuple(float(x.mean()) for x in (base_err,mo,to,joint,matched,teacher_shift,model_shift,inter,cs,cf,cd,nratio))+(float(den0.mean()),float(den.mean()),float(sq(pr-rr).mean()),float(sq(pr-r0).mean()),float(sq(p0-rr).mean()),float(sq(dT).mean()),float(sq(dP).mean()),seed)
      else:
       rows[(g,c,mode,a)]=stats
      if mode=='BASE_AB':baseline[a]=st;baseline['p0_'+a]=pr.detach().clone()
      if mode=='GAUGE_AB':
       expected=r4.gauge_transform(baseline[a],gauges,torch)
       diag['gauge']=max(diag['gauge'],max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
       old_ref=paired_delta(baseline['teacher'],B);old_pred=baseline['p0_'+a];old_den=old_ref.square().sum(-1)
       prior=(float(((old_pred-old_ref).square().sum(-1)/old_den).mean()),float(((old_pred*old_ref).sum(-1)/old_den).mean()),float((((old_pred*old_ref).sum(-1)/old_den)>0).double().mean()),float(((old_pred-old_ref).square().sum(-1)/old_den).max()),float(old_ref.norm(dim=-1).mean()))
       diag['gauge']=max(diag['gauge'],max(abs(v-w)/(1.+abs(v)+abs(w)) for v,w in zip(stats[:5],prior)))
      if g=='D1_P2' and c=='weak' and (preflight or seed==f['eval_seeds'][0]):
       if mode in ('BASE_AB','NONFLAT_AB'):
        native=r4.rollout(initial,adj_a,models[a],f['horizon'],torch)[-1]
        diag['native']=max(diag['native'],max(float((st[k]-native[k]).abs().max()) for k in (0,1)))
       independent=split_rollout(initial,adj_a,adj_b,models[a],f['horizon'],torch)
       diag['parity']=max(diag['parity'],max(float((st[k]-independent[k]).abs().max()) for k in (0,1)))
       if mode in ('NONFLAT_A','NONFLAT_B') and a in ('SAL_STATE8','SAL_PCGRAD4'):
        gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
        gadj_a=r4.links_gauge(adj_a,gauges,torch);gadj_b=r4.links_gauge(adj_b,gauges,torch)
        rot=split_rollout(r4.gauge_transform(initial,gauges,torch),gadj_a,gadj_b,models[a],f['horizon'],torch)
        expected=r4.gauge_transform(st,gauges,torch)
        diag['cov']=max(diag['cov'],max(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
     if mode=='BASE_AB':baseline['teacher']=reference;baseline['ref_norm']=rrnorm;baseline['r0']=rr.detach().clone()
  # Both-channel and split-channel disconnected isolation: no path from source to remote.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g])
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  edge_modes={name:r4.build_links(mapped,transport_matrices(mapped,mats,name,cfg,r3,torch),torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT')}
  for mode in modes[:-1]:
   a_mode,b_mode=mp[mode];adj_a=edge_modes[a_mode];adj_b=edge_modes[b_mode]
   tr=split_rollout(initial,adj_a,adj_b,teacher,20,torch)
   diag['disconnected']=max(diag['disconnected'],float(tr[0][:,3,:].abs().max()))
   for a in ('SAL_STATE8','SAL_PCGRAD4'):
    st=split_rollout(initial,adj_a,adj_b,models[a],20,torch)
    diag['disconnected']=max(diag['disconnected'],float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R58_WEIGHT_MUTATION:'+a)
 return rows,diag
def r59_trace_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
 f=cfg['r59_trace'];tf=cfg['r56_transport'];s=cfg['r55_structural'];B=f['eval_batch'];per=2*B;arms=f['arms'];n=len(arms)
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms]);law=_StackedGates(W,bias,r30,torch,per)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in graph) for k,graph in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 hashes={a:r4.model_hash(models[a]) for a in arms}
 rows={};traces={};diag={'cov':0.,'disconnected':0.,'min_ref':1e30,'parity':0.,'gauge':0.,'source':0.,'native':0.,'spread':0.,'min_teacher_base':1e30,'max_teacher_response_ratio':0.,'decomp':0.,'metric':0.,'min_base_ref':1e30,'min_model_norm':1e30,'trace_metric':0.,'trace_decomposition':0.,'valid_h20':0}
 modes=f['modes'];mp={'BASE_AB':('BASE_SEPARABLE','BASE_SEPARABLE'), 'NONFLAT_A':('PARTICIPANT_NONFLAT','BASE_SEPARABLE'), 'NONFLAT_B':('BASE_SEPARABLE','PARTICIPANT_NONFLAT'),'NONFLAT_AB':('PARTICIPANT_NONFLAT','PARTICIPANT_NONFLAT'),'GAUGE_AB':('GAUGE_ONLY','GAUGE_ONLY')}
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    mapped,mats,init,flip=r3.generate(g,c,f['eval_offset']+seed*100109+ci*59,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:diag['source']=max(diag['source'],max(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    a_mats={name:transport_matrices(mapped,mats,name,cfg,r3,torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT','GAUGE_ONLY')}
    edges={name:r4.build_links(mapped,ms,torch) for name,ms in a_mats.items()}
    spread,cnt=pair_transport_spread(mapped,a_mats['PARTICIPANT_NONFLAT'],r4,torch)
    diag['spread']=max(diag['spread'],spread)
    # Some degree-preserving graphs lack repeated links; require the witness across the complete family.
    baseline={}
    channel_pairs=[(edges[mp[mode][0]],edges[mp[mode][1]]) for mode in modes]
    prep_a=prepare_mode_adj([x[0] for x in channel_pairs],torch)
    prep_b=prepare_mode_adj([x[1] for x in channel_pairs],torch)
    M=len(modes)
    reference_all=(initial[0].repeat(M,1,1),initial[1].repeat(M,1,1))
    packed_all=(initial[0].repeat(M*n,1,1),initial[1].repeat(M*n,1,1))
    mode_law=_StackedGates(W.repeat(M,1,1),bias.repeat(M,1),r30,torch,per)
    snapshots={}
    for hh in f['horizons']:
     reference_all=graph_step_modes(reference_all,prep_a,prep_b,teacher,torch,per)
     packed_all=graph_step_modes(packed_all,prep_a,prep_b,mode_law,torch,n*per)
     # Hold only remote source-flip responses in RAM: all teacher/model modes.
     ta=reference_all[0].reshape(M,per,4,4)
     pp=packed_all[0].reshape(M,n,per,4,4)
     snapshots[hh]=(ta[:,:B,3,:]-ta[:,B:,3,:],pp[:,:,:B,3,:]-pp[:,:,B:,3,:])
    for mi,mode in enumerate(modes):
     a_mode,b_mode=mp[mode];adj_a,adj_b=channel_pairs[mi]
     reference=(reference_all[0][mi*per:(mi+1)*per],reference_all[1][mi*per:(mi+1)*per])
     packed=(packed_all[0][mi*n*per:(mi+1)*n*per],packed_all[1][mi*n*per:(mi+1)*n*per])
     rr=paired_delta(reference,B);den=rr.square().sum(-1)
     need(bool((den>1e-25).all()),'R59_ZERO_REFERENCE:'+g+':'+c+':'+mode)
     rrnorm=float(rr.norm(dim=-1).mean())
     diag['min_ref']=min(diag['min_ref'],float(rr.norm(dim=-1).min()))
     if mode=='BASE_AB':diag['min_teacher_base']=min(diag['min_teacher_base'],rrnorm)
     else:diag['max_teacher_response_ratio']=max(diag['max_teacher_response_ratio'],rrnorm/max(baseline['ref_norm'],1e-25))
     if mode=='GAUGE_AB':
      gauges={r:r3.so2x2_torch(tf['gauge_shift'][0]*r,tf['gauge_shift'][1]*r,torch) for r in range(4)}
      expected=r4.gauge_transform(baseline['teacher'],gauges,torch)
      diag['gauge']=max(diag['gauge'],max(float((reference[k]-expected[k]).abs().max()) for k in (0,1)))
     for j,a in enumerate(arms):
      sl=slice(j*per,(j+1)*per);st=(packed[0][sl],packed[1][sl]);pr=paired_delta(st,B)
      e=(pr-rr).square().sum(-1)/den;pg=(pr*rr).sum(-1)/den
      stats=(float(e.mean()),float(pg.mean()),float((pg>0).double().mean()),float(e.max()),rrnorm,seed)
      need(all(math.isfinite(z) for z in stats[:5]),'R59_METRICS_NONFINITE:'+g+':'+c+':'+mode+':'+a)
      # Identical source-flip realization and relation-3 frame for all four physical/synthetic modes.
      # Gauge-only is independently verified against gauge-transformed baseline, never compared unrotated.
      if mode!='GAUGE_AB':
       r0=rr if mode=='BASE_AB' else baseline['r0']
       p0=pr if mode=='BASE_AB' else baseline['p0_'+a]
       dT=rr-r0;dP=pr-p0;den0=r0.square().sum(-1)
       need(bool((den0>1e-25).all()),'R59_ZERO_FIXED_REFERENCE:'+g+':'+c)
       dot=lambda x,y:(x*y).sum(-1)
       sq=lambda x:dot(x,x)
       base_err=sq(p0-r0)/den0;mo=sq(pr-r0)/den0;to=sq(p0-rr)/den0
       joint=sq(pr-rr)/den0;matched=sq(pr-rr)/den
       teacher_shift=sq(dT)/den0;model_shift=sq(dP)/den0
       inter=joint-mo-to+base_err;cross=-2.*dot(dP,dT)/den0
       target_cross=joint-mo-teacher_shift;cross2=-2.*dot(pr-r0,dT)/den0
       nratio=den/den0
       # A near-zero model vector can have undefined direction; zero as explicit convention.
       cosine=lambda x,y: dot(x,y)/(torch.sqrt(sq(x)*sq(y))+1e-30)
       cs=cosine(pr,rr);cf=cosine(pr,r0);cd=cosine(dP,dT)
       check=max(float((inter-cross).abs().max()),float((target_cross-cross2).abs().max()),float((joint-matched*nratio).abs().max()))
       diag['decomp']=max(diag['decomp'],check)
       diag['metric']=max(diag['metric'],abs(float(matched.mean())-stats[0]))
       diag['min_base_ref']=min(diag['min_base_ref'],float(torch.sqrt(den0).min()))
       diag['min_model_norm']=min(diag['min_model_norm'],float(torch.sqrt(sq(pr)).min()))
       # One tuple per cell: each metric is a per-example mean (not mean-of-norms squared).
       rows[(g,c,mode,a)]=tuple(float(x.mean()) for x in (base_err,mo,to,joint,matched,teacher_shift,model_shift,inter,cs,cf,cd,nratio))+(float(den0.mean()),float(den.mean()),float(sq(pr-rr).mean()),float(sq(pr-r0).mean()),float(sq(p0-rr).mean()),float(sq(dT).mean()),float(sq(dP).mean()),seed)
      else:
       rows[(g,c,mode,a)]=stats
      if mode=='BASE_AB':baseline[a]=st;baseline['p0_'+a]=pr.detach().clone()
      if mode=='GAUGE_AB':
       expected=r4.gauge_transform(baseline[a],gauges,torch)
       diag['gauge']=max(diag['gauge'],max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
       old_ref=paired_delta(baseline['teacher'],B);old_pred=baseline['p0_'+a];old_den=old_ref.square().sum(-1)
       prior=(float(((old_pred-old_ref).square().sum(-1)/old_den).mean()),float(((old_pred*old_ref).sum(-1)/old_den).mean()),float((((old_pred*old_ref).sum(-1)/old_den)>0).double().mean()),float(((old_pred-old_ref).square().sum(-1)/old_den).max()),float(old_ref.norm(dim=-1).mean()))
       diag['gauge']=max(diag['gauge'],max(abs(v-w)/(1.+abs(v)+abs(w)) for v,w in zip(stats[:5],prior)))
      if g=='D1_P2' and c=='weak' and (preflight or seed==f['eval_seeds'][0]):
       if mode in ('BASE_AB','NONFLAT_AB'):
        native=r4.rollout(initial,adj_a,models[a],f['horizons'][-1],torch)[-1]
        diag['native']=max(diag['native'],max(float((st[k]-native[k]).abs().max()) for k in (0,1)))
       independent=split_rollout(initial,adj_a,adj_b,models[a],f['horizons'][-1],torch)
       diag['parity']=max(diag['parity'],max(float((st[k]-independent[k]).abs().max()) for k in (0,1)))
       if mode in ('NONFLAT_A','NONFLAT_B') and a in ('SAL_STATE8','SAL_PCGRAD4'):
        gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
        gadj_a=r4.links_gauge(adj_a,gauges,torch);gadj_b=r4.links_gauge(adj_b,gauges,torch)
        rot=split_rollout(r4.gauge_transform(initial,gauges,torch),gadj_a,gadj_b,models[a],f['horizons'][-1],torch)
        expected=r4.gauge_transform(st,gauges,torch)
        diag['cov']=max(diag['cov'],max(float((rot[k]-expected[k]).abs().max()) for k in (0,1)))
     if mode=='BASE_AB':baseline['teacher']=reference;baseline['ref_norm']=rrnorm;baseline['r0']=rr.detach().clone()
    # The R58 fixed-reference decomposition at H20 is kept unchanged above.
    # At every h compute B-only/BASE signed response vectors in a COMMON frame.
    for hh,(teach,preds) in snapshots.items():
     t0=teach[0];tm=teach[2]
     den0=t0.square().sum(-1);denm=tm.square().sum(-1)
     valid0=den0>1e-25;validm=denm>1e-25
     diag['valid_h20']+=int(valid0.sum()) if hh==20 else 0
     dot=lambda x,y:(x*y).sum(-1)
     sq=lambda x:dot(x,x)
     for arm in f['analyzed_arms']:
      j=arms.index(arm);p0=preds[0,j];pm=preds[2,j]
      dt=tm-t0;dp=pm-p0
      base_abs=sq(p0-t0);fixed_abs=sq(pm-t0);matched_abs=sq(pm-tm)
      teacher_only_abs=sq(p0-tm);teacher_shift_abs=sq(dt);model_shift_abs=sq(dp)
      # Always include absolute response metrics, including before remote signal arrives.
      entry={'n_valid':int(valid0.sum()),'n_mode':int(validm.sum()),'n_total':B,
        'T0_EN':float(den0.mean()),'TM_EN':float(denm.mean()),
        'P0_EN':float(sq(p0).mean()),'PM_EN':float(sq(pm).mean()),
        'TEACH_SHIFT_ABS':float(teacher_shift_abs.mean()),'MODEL_SHIFT_ABS':float(model_shift_abs.mean()),
        'BASE_ERR_ABS':float(base_abs.mean()),'FIXED_ERR_ABS':float(fixed_abs.mean()),
        'MATCHED_ERR_ABS':float(matched_abs.mean())}
      if bool(valid0.any()):
       sel=valid0
       den=den0[sel]
       def avgdiv(x):return float((x[sel]/den).mean())
       e0=avgdiv(base_abs);ef=avgdiv(fixed_abs)
       em=avgdiv(matched_abs);eto=avgdiv(teacher_only_abs)
       es=avgdiv(teacher_shift_abs);eps=avgdiv(model_shift_abs)
       cross=(-2*dot(dp,dt))[sel]/den
       inter=(matched_abs-fixed_abs-teacher_only_abs+base_abs)[sel]/den
       residual=float((cross-inter).abs().max())/(1+max(abs(float(cross.mean())),abs(float(inter.mean()))))
       diag['trace_decomposition']=max(diag['trace_decomposition'],residual)
       entry.update(E0=e0,FIXED=ef,JOINT_FIXED=em,TEACHER_ONLY=eto,
         TEACH_SHIFT=es,MODEL_SHIFT=eps,INTERACTION=float(inter.mean()),
         ADVANTAGE=ef-e0,
         COS_FIXED=float((dot(pm,t0)/(torch.sqrt(sq(pm)*den0)+1e-30))[sel].mean()),
         COS_MATCHED=float((dot(pm,tm)/(torch.sqrt(sq(pm)*denm)+1e-30))[sel].mean()),
         GAIN_FIXED=float((dot(pm,t0)[sel]/den).mean()))
       if hh==20:
        refrow=rows[(g,c,'NONFLAT_B',arm)]
        for x,y in ((ef,refrow[1]),(em,refrow[3]),(es,refrow[5]),(eps,refrow[6])):
         diag['trace_metric']=max(diag['trace_metric'],abs(x-y)/(1+abs(y)))
      if bool(validm.any()):entry['MATCHED']=float((matched_abs[validm]/denm[validm]).mean())
      traces[(g,c,arm,hh)]=entry
  # Both-channel and split-channel disconnected isolation: no path from source to remote.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g])
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  edge_modes={name:r4.build_links(mapped,transport_matrices(mapped,mats,name,cfg,r3,torch),torch) for name in ('BASE_SEPARABLE','PARTICIPANT_NONFLAT')}
  for mode in modes[:-1]:
   a_mode,b_mode=mp[mode];adj_a=edge_modes[a_mode];adj_b=edge_modes[b_mode]
   tr=split_rollout(initial,adj_a,adj_b,teacher,20,torch)
   diag['disconnected']=max(diag['disconnected'],float(tr[0][:,3,:].abs().max()))
   for a in ('SAL_STATE8','SAL_PCGRAD4'):
    st=split_rollout(initial,adj_a,adj_b,models[a],20,torch)
    diag['disconnected']=max(diag['disconnected'],float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R59_WEIGHT_MUTATION:'+a)
 return rows,diag,traces

def r59_aggregate(traces,cfg):
 f=cfg['r59_trace'];out={}
 for c in f['cases']:
  for arm in f['analyzed_arms']:
   for h in f['horizons']:
    rr=[v for (g,case,a,hh),rows in traces.items() if case==c and a==arm and hh==h for v in rows]
    need(len(rr) in (len(f['graphs']),len(f['graphs'])*len(f['eval_seeds'])),'R59_SAMPLE_COUNT')
    s={'VALID':sum(r['n_valid'] for r in rr),'MODE_VALID':sum(r['n_mode'] for r in rr),
       'POSSIBLE':sum(r['n_total'] for r in rr),'CELLS':len(rr)}
    # Per-record equal batch; these are means of per-example energies, no division by a tiny reference.
    for name in ('T0_EN','TM_EN','P0_EN','PM_EN','TEACH_SHIFT_ABS','MODEL_SHIFT_ABS','BASE_ERR_ABS','FIXED_ERR_ABS','MATCHED_ERR_ABS'):
     s[name]=mean(r[name] for r in rr)
    for name in ('E0','FIXED','JOINT_FIXED','TEACHER_ONLY','TEACH_SHIFT','MODEL_SHIFT','INTERACTION','ADVANTAGE','COS_FIXED','COS_MATCHED','GAIN_FIXED','MATCHED'):
     good=[r for r in rr if name in r]
     s[name]=sum(r[name]*r['n_mode' if name=='MATCHED' else 'n_valid'] for r in good)/sum(r['n_mode' if name=='MATCHED' else 'n_valid'] for r in good) if good else float('nan')
    s['TEACH_RATIO_ENERGY']=s['TM_EN']/s['T0_EN'] if s['T0_EN']>0 else float('nan')
    s['FIXED_RATIO_ENERGY']=s['FIXED_ERR_ABS']/s['T0_EN'] if s['T0_EN']>0 else float('nan')
    out[(c,arm,h)]=s
 return out

# R58 fixed-reference tuple positions, immutable and explicitly manifest-pinned.
R58_KEYS=('E0','MODEL_ONLY','TEACHER_ONLY','JOINT_FIXED','JOINT_MATCHED','TEACHER_SHIFT','MODEL_SHIFT','INTERACTION','COS_MATCHED','COS_FIXED','COS_SHIFT','REF_ENERGY_RATIO','DEN0','DENM','JOINT_ABS','MODEL_ONLY_ABS','TEACHER_ONLY_ABS','TEACHER_SHIFT_ABS','MODEL_SHIFT_ABS','SEED')
def r58_summarize(rows,cfg):
 f=cfg['r58_reference'];out={}
 for g in f['graphs']:
  for c in f['cases']:
   for mode in f['modes'][:-1]:
    for a in f['arms']:
     values=rows[(g,c,mode,a)];need(values and len(values)==len(f['eval_seeds']) if len(values)>1 else bool(values),'R58_EMPTY')
     out[(g,c,mode,a)]={k:mean(x[i] for x in values) for i,k in enumerate(R58_KEYS[:-1])}
 return out

def r58_aggregate(rows,cfg):
 f=cfg['r58_reference'];out={}
 for c in f['cases']:
  for a in f['arms']:
   for m in f['modes'][:-1]:
    sample=[x for g in f['graphs'] for x in rows[(g,c,m,a)]]
    val={k:mean(x[i] for x in sample) for i,k in enumerate(R58_KEYS[:-1])}
    # Ratio of total squared errors to total baseline-teacher squared response energy.
    val['JOINT_FIXED_ENERGY']=sum(x[14] for x in sample)/sum(x[12] for x in sample)
    val['JOINT_MATCHED_ENERGY']=sum(x[14] for x in sample)/sum(x[13] for x in sample)
    val['MODEL_ONLY_ENERGY']=sum(x[15] for x in sample)/sum(x[12] for x in sample)
    val['TEACHER_SHIFT_ENERGY']=sum(x[17] for x in sample)/sum(x[12] for x in sample)
    val['MODEL_SHIFT_ENERGY']=sum(x[18] for x in sample)/sum(x[12] for x in sample)
    val['REF_RATIO_ENERGY']=sum(x[13] for x in sample)/sum(x[12] for x in sample)
    out[(c,a,m)]=val
 return out

def r57_summarize(rows,cfg):
 f=cfg['r57_channels'];out={}
 for g in f['graphs']:
  for c in f['cases']:
   for mode in f['modes']:
    for a in f['arms']:
     values=rows[(g,c,mode,a)];need(len(values)>0,'R57_EMPTY_CELL')
     out[(g,c,mode,a)]={'e2':mean(x[0] for x in values),'gain':mean(x[1] for x in values),'sign':mean(x[2] for x in values),'worst':max(x[3] for x in values),'ref':mean(x[4] for x in values)}
 return out

def r57_effects(report,cfg):
 f=cfg['r57_channels'];out={};modes=f['modes'][:-1]
 for c in f['cases']:
  for a in f['arms']:
   cells={m:[report[(g,c,m,a)] for g in f['graphs']] for m in modes}
   e={m:mean(x['e2'] for x in cells[m]) for m in modes}
   gain={m:mean(x['gain'] for x in cells[m]) for m in modes}
   ref={m:mean(x['ref'] for x in cells[m]) for m in modes}
   log=lambda x:math.log1p(x)
   out[(c,a)]={'e':e,'gain':gain,'ref':ref,'Alog':log(e['NONFLAT_A'])-log(e['BASE_AB']),'Blog':log(e['NONFLAT_B'])-log(e['BASE_AB']), 'interaction':log(e['NONFLAT_AB'])-log(e['NONFLAT_A'])-log(e['NONFLAT_B'])+log(e['BASE_AB']), 'gain_interaction':gain['NONFLAT_AB']-gain['NONFLAT_A']-gain['NONFLAT_B']+gain['BASE_AB']}
 return out

# R60 frozen time-local B-only transport seam: no model training or inference mutation.
def r60_window_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
 f=cfg['r60_windows']; s=cfg['r55_structural'];B=f['eval_batch'];per=2*B
 arms=f['arms'];n=len(arms); modes=list(f['modes']);M=len(modes)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in g) for k,g in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms])
 law=_StackedGates(W.repeat(M,1,1),bias.repeat(M,1),r30,torch,per)
 hashes={a:r4.model_hash(models[a]) for a in arms}
 rows={};diagn={'cov':0.,'disconnected':0.,'parity':0.,'prefix':0.,'gauge':0.,'spread':0.,'min_ref':1e30,'decomposition':0.,'source':0.,'matched_min':1e30}
 def is_nonflat(mode,h):return any(lo<=h<=hi for lo,hi in f['modes'][mode])
 def dot(x,y):return (x*y).sum(-1)
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    mapped,mats,init,flip=r3.generate(g,c,f['eval_offset']+seed*100109+ci*59,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:diagn['source']=max(diagn['source'],*(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    base_mat=transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,torch)
    modified_mat=transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,torch)
    spread,_=pair_transport_spread(mapped,modified_mat,r4,torch)
    diagn['spread']=max(diagn['spread'],spread)
    edge_base=r4.build_links(mapped,base_mat,torch)
    edge_nonflat=r4.build_links(mapped,modified_mat,torch)
    pa=prepare_mode_adj([edge_base]*M,torch)
    pb={}
    # Cache the 20 possible mode-wise adjacency tensors, reused by both teacher/model.
    for h in range(1,21):
     key=tuple(is_nonflat(m,h) for m in modes)
     if key not in pb:pb[key]=prepare_mode_adj([edge_nonflat if v else edge_base for v in key],torch)
    teacher_states=(initial[0].repeat(M,1,1),initial[1].repeat(M,1,1))
    model_states=(initial[0].repeat(M*n,1,1),initial[1].repeat(M*n,1,1))
    for h in range(1,21):
     key=tuple(is_nonflat(m,h) for m in modes)
     teacher_states=graph_step_modes(teacher_states,pa,pb[key],teacher,torch,per)
     model_states=graph_step_modes(model_states,pa,pb[key],law,torch,n*per)
     # Until the activation window, both teacher and model must match baseline at every step.
     if preflight:
      tt=teacher_states[0].reshape(M,per,4,4);tm=teacher_states[1].reshape(M,per,4,4)
      pp=model_states[0].reshape(M,n,per,4,4);pm=model_states[1].reshape(M,n,per,4,4)
      for mi,m in enumerate(modes[1:],1):
       if not any(is_nonflat(m,z) for z in range(1,h+1)):
        diagn['prefix']=max(diagn['prefix'],float((tt[mi]-tt[0]).abs().max()),float((tm[mi]-tm[0]).abs().max()),float((pp[mi]-pp[0]).abs().max()),float((pm[mi]-pm[0]).abs().max()))
    ta=teacher_states[0].reshape(M,per,4,4)
    ppa=model_states[0].reshape(M,n,per,4,4)
    t0=ta[0,:B,3,:]-ta[0,B:,3,:]
    den=dot(t0,t0);valid=den>1e-25
    diagn['min_ref']=min(diagn['min_ref'],float(t0.norm(dim=-1).min()))
    # Explicit valid count; no silent truncation of denominator-zero samples.
    for mi,mode in enumerate(modes):
     tr=ta[mi,:B,3,:]-ta[mi,B:,3,:];denm=dot(tr,tr);validm=denm>1e-25
     if bool(validm.any()):diagn['matched_min']=min(diagn['matched_min'],float(tr.norm(dim=-1)[validm].min()))
     for j,a in enumerate(arms):
      p0=ppa[0,j,:B,3,:]-ppa[0,j,B:,3,:]
      pr=ppa[mi,j,:B,3,:]-ppa[mi,j,B:,3,:]
      dP=pr-p0;dT=tr-t0
      e0=dot(p0-t0,p0-t0);fixed=dot(pr-t0,pr-t0);matched=dot(pr-tr,pr-tr)
      fixed_den=den[valid];mode_den=denm[validm]
      row={'N':B,'VALID':int(valid.sum()),'MODE_VALID':int(validm.sum()),
       'REF_ENERGY':float(den.mean()),'MODE_REF_ENERGY':float(denm.mean()),
       'BASE_ABS_ERR':float(e0.mean()),'FIXED_ABS_ERR':float(fixed.mean()),'MATCHED_ABS_ERR':float(matched.mean()),
       'MODEL_SHIFT':float(dot(dP,dP).mean()),'TEACHER_SHIFT':float(dot(dT,dT).mean()),
       'PRED_ENERGY':float(dot(pr,pr).mean())}
      if bool(valid.any()):
       row.update(BASE_E2=float((e0[valid]/fixed_den).mean()),FIXED_E2=float((fixed[valid]/fixed_den).mean()),
        GAIN_FIXED=float((dot(pr,t0)[valid]/fixed_den).mean()),
        COS_FIXED=float((dot(pr,t0)[valid]/(torch.sqrt(dot(pr,pr)[valid]*fixed_den)+1e-30)).mean()),
        WORST_FIXED=float((fixed[valid]/fixed_den).max()),
        TEACHER_SHIFT_REL=float((dot(dT,dT)[valid]/fixed_den).mean()),
        MODEL_SHIFT_REL=float((dot(dP,dP)[valid]/fixed_den).mean()))
       # Verify matched vs fixed decomposition per-example at native float64 precision.
       teacher_only=dot(p0-tr,p0-tr)/fixed_den
       interaction=(matched-fixed-dot(p0-tr,p0-tr)+e0)[valid]/fixed_den
       exact=(-2*dot(dP,dT)[valid])/fixed_den
       res=float((interaction-exact).abs().max())/(1+abs(float(exact.mean())))
       diagn['decomposition']=max(diagn['decomposition'],res)
      if bool(validm.any()):row['MATCHED_E2']=float((matched[validm]/mode_den).mean())
      rows[(g,c,mode,a)]=row
    # Independent native and dynamic-step forward parity, plus gauge covariance on excluded seed.
    if g=='D1_P2' and c=='weak' and preflight:
     for mi,mode in enumerate(modes):
      st=initial
      for h in range(1,21):st=graph_step_split(st,edge_base,edge_nonflat if is_nonflat(mode,h) else edge_base,teacher,torch)
      batchstate=(teacher_states[0][mi*per:(mi+1)*per],teacher_states[1][mi*per:(mi+1)*per])
      diagn['parity']=max(diagn['parity'],max(float((st[k]-batchstate[k]).abs().max()) for k in (0,1)))
      for j,a in enumerate(arms):
       st=initial
       for h in range(1,21):st=graph_step_split(st,edge_base,edge_nonflat if is_nonflat(mode,h) else edge_base,models[a],torch)
       sl=slice((mi*n+j)*per,(mi*n+j+1)*per)
       batchstate=(model_states[0][sl],model_states[1][sl])
       diagn['parity']=max(diagn['parity'],max(float((st[k]-batchstate[k]).abs().max()) for k in (0,1)))
       if mode=='BASE_AB':
        native=r4.rollout(initial,edge_base,models[a],20,torch)[-1]
        diagn['parity']=max(diagn['parity'],max(float((native[k]-batchstate[k]).abs().max()) for k in (0,1)))
      # True gauge covariance of the time-switching B seam with both links gauged together.
      if mode=='B_H09_12':
       gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
       gb=r4.links_gauge(edge_base,gauges,torch);gn=r4.links_gauge(edge_nonflat,gauges,torch)
       st=r4.gauge_transform(initial,gauges,torch)
       for h in range(1,21):st=graph_step_split(st,gb,gn if is_nonflat(mode,h) else gb,models['SAL_PCGRAD4'],torch)
       sl=slice((mi*n+arms.index('SAL_PCGRAD4'))*per,(mi*n+arms.index('SAL_PCGRAD4')+1)*per)
       expected=r4.gauge_transform((model_states[0][sl],model_states[1][sl]),gauges,torch)
       diagn['cov']=max(diagn['cov'],max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
  # Disconnected structure must remain source-isolated under every B time window.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g])
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  eb=r4.build_links(mapped,transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,torch),torch)
  en=r4.build_links(mapped,transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,torch),torch)
  for mode in modes:
   for law1 in (teacher,models['SAL_STATE8'],models['SAL_PCGRAD4']):
    st=initial
    for h in range(1,21):st=graph_step_split(st,eb,en if is_nonflat(mode,h) else eb,law1,torch)
    diagn['disconnected']=max(diagn['disconnected'],float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R60_INFERENCE_WEIGHT_MUTATION:'+a)
 return rows,diagn

def r60_aggregate(rows,cfg):
 f=cfg['r60_windows'];out={}
 for c in f['cases']:
  for arm in f['arms']:
   for mode in f['modes']:
    rec=[v for (g,cc,m,a),vs in rows.items() if cc==c and m==mode and a==arm for v in vs]
    need(len(rec)==len(f['graphs'])*len(f['eval_seeds']),'R60_AGGREGATION_COUNT')
    def ew(name,den='VALID'):
     good=[x for x in rec if name in x]
     count=sum(x[den] for x in good)
     return sum(x[name]*x[den] for x in good)/count if count else float('nan')
    a={'N':sum(v['N'] for v in rec),'VALID':sum(v['VALID'] for v in rec),
       'MODE_VALID':sum(v['MODE_VALID'] for v in rec),'CELLS':len(rec)}
    for key in ('BASE_E2','FIXED_E2','GAIN_FIXED','COS_FIXED','TEACHER_SHIFT_REL','MODEL_SHIFT_REL'):
     a[key]=ew(key)
    a['MATCHED_E2']=ew('MATCHED_E2','MODE_VALID')
    for key in ('REF_ENERGY','MODE_REF_ENERGY','BASE_ABS_ERR','FIXED_ABS_ERR','MATCHED_ABS_ERR','MODEL_SHIFT','TEACHER_SHIFT','PRED_ENERGY'):
     a[key]=mean(v[key] for v in rec)
    a['ENERGY_NORM_FIXED']=a['FIXED_ABS_ERR']/(a['REF_ENERGY']+1e-30)
    a['ENERGY_NORM_MATCHED']=a['MATCHED_ABS_ERR']/(a['MODE_REF_ENERGY']+1e-30)
    a['WORST_FIXED']=max(v.get('WORST_FIXED',float('nan')) for v in rec)
    a['SEED_WINS_BASE']=sum(ew1<ew2 for ee in f['eval_seeds'] for ew1,ew2 in [
     (mean(x['FIXED_E2'] for (g,cc,m,aa),vs in rows.items() if cc==c and m==mode and aa==arm for x in [vs[f['eval_seeds'].index(ee)]]),
      mean(x['FIXED_E2'] for (g,cc,m,aa),vs in rows.items() if cc==c and m=='BASE_AB' and aa==arm for x in [vs[f['eval_seeds'].index(ee)]]))])
    out[(c,arm,mode)]=a
 return out

def r61_step_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
 f=cfg['r61_early_steps']; s=cfg['r55_structural'];B=f['eval_batch'];per=2*B
 arms=f['arms'];n=len(arms); modes=list(f['modes']);M=len(modes)
 graphs={**basegraphs,**{k:tuple(tuple(v) for v in g) for k,g in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 W=torch.stack([models[a].gates.weight.detach() for a in arms]);bias=torch.stack([models[a].gates.bias.detach() for a in arms])
 law=_StackedGates(W.repeat(M,1,1),bias.repeat(M,1),r30,torch,per)
 hashes={a:r4.model_hash(models[a]) for a in arms}
 rows={};diagn={'cov':0.,'disconnected':0.,'parity':0.,'prefix':0.,'gauge':0.,'spread':0.,'min_ref':1e30,'decomposition':0.,'source':0.,'matched_min':1e30}
 def is_nonflat(mode,h):return any(lo<=h<=hi for lo,hi in f['modes'][mode])
 def dot(x,y):return (x*y).sum(-1)
 with torch.no_grad():
  for ci,c in enumerate(f['cases']):
   first_source=None
   for g in f['graphs']:
    mapped,mats,init,flip=r3.generate(g,c,f['eval_offset']+seed*100109+ci*59,B,graphs,cases,torch)
    initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
    if first_source is None:first_source=initial
    else:diagn['source']=max(diagn['source'],*(float((initial[k]-first_source[k]).abs().max()) for k in (0,1)))
    base_mat=transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,torch)
    modified_mat=transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,torch)
    spread,_=pair_transport_spread(mapped,modified_mat,r4,torch)
    diagn['spread']=max(diagn['spread'],spread)
    edge_base=r4.build_links(mapped,base_mat,torch)
    edge_nonflat=r4.build_links(mapped,modified_mat,torch)
    pa=prepare_mode_adj([edge_base]*M,torch)
    pb={}
    # Cache the 20 possible mode-wise adjacency tensors, reused by both teacher/model.
    for h in range(1,21):
     key=tuple(is_nonflat(m,h) for m in modes)
     if key not in pb:pb[key]=prepare_mode_adj([edge_nonflat if v else edge_base for v in key],torch)
    teacher_states=(initial[0].repeat(M,1,1),initial[1].repeat(M,1,1))
    model_states=(initial[0].repeat(M*n,1,1),initial[1].repeat(M*n,1,1))
    for h in range(1,21):
     key=tuple(is_nonflat(m,h) for m in modes)
     teacher_states=graph_step_modes(teacher_states,pa,pb[key],teacher,torch,per)
     model_states=graph_step_modes(model_states,pa,pb[key],law,torch,n*per)
     # Until the activation window, both teacher and model must match baseline at every step.
     if preflight:
      tt=teacher_states[0].reshape(M,per,4,4);tm=teacher_states[1].reshape(M,per,4,4)
      pp=model_states[0].reshape(M,n,per,4,4);pm=model_states[1].reshape(M,n,per,4,4)
      for mi,m in enumerate(modes[1:],1):
       if not any(is_nonflat(m,z) for z in range(1,h+1)):
        diagn['prefix']=max(diagn['prefix'],float((tt[mi]-tt[0]).abs().max()),float((tm[mi]-tm[0]).abs().max()),float((pp[mi]-pp[0]).abs().max()),float((pm[mi]-pm[0]).abs().max()))
    ta=teacher_states[0].reshape(M,per,4,4)
    ppa=model_states[0].reshape(M,n,per,4,4)
    t0=ta[0,:B,3,:]-ta[0,B:,3,:]
    den=dot(t0,t0);valid=den>1e-25
    diagn['min_ref']=min(diagn['min_ref'],float(t0.norm(dim=-1).min()))
    # Explicit valid count; no silent truncation of denominator-zero samples.
    for mi,mode in enumerate(modes):
     tr=ta[mi,:B,3,:]-ta[mi,B:,3,:];denm=dot(tr,tr);validm=denm>1e-25
     if bool(validm.any()):diagn['matched_min']=min(diagn['matched_min'],float(tr.norm(dim=-1)[validm].min()))
     for j,a in enumerate(arms):
      p0=ppa[0,j,:B,3,:]-ppa[0,j,B:,3,:]
      pr=ppa[mi,j,:B,3,:]-ppa[mi,j,B:,3,:]
      dP=pr-p0;dT=tr-t0
      e0=dot(p0-t0,p0-t0);fixed=dot(pr-t0,pr-t0);matched=dot(pr-tr,pr-tr)
      fixed_den=den[valid];mode_den=denm[validm]
      row={'N':B,'VALID':int(valid.sum()),'MODE_VALID':int(validm.sum()),
       'REF_ENERGY':float(den.mean()),'MODE_REF_ENERGY':float(denm.mean()),
       'BASE_ABS_ERR':float(e0.mean()),'FIXED_ABS_ERR':float(fixed.mean()),'MATCHED_ABS_ERR':float(matched.mean()),
       'MODEL_SHIFT':float(dot(dP,dP).mean()),'TEACHER_SHIFT':float(dot(dT,dT).mean()),
       'PRED_ENERGY':float(dot(pr,pr).mean())}
      if bool(valid.any()):
       row.update(BASE_E2=float((e0[valid]/fixed_den).mean()),FIXED_E2=float((fixed[valid]/fixed_den).mean()),
        GAIN_FIXED=float((dot(pr,t0)[valid]/fixed_den).mean()),
        COS_FIXED=float((dot(pr,t0)[valid]/(torch.sqrt(dot(pr,pr)[valid]*fixed_den)+1e-30)).mean()),
        WORST_FIXED=float((fixed[valid]/fixed_den).max()),
        TEACHER_SHIFT_REL=float((dot(dT,dT)[valid]/fixed_den).mean()),
        MODEL_SHIFT_REL=float((dot(dP,dP)[valid]/fixed_den).mean()))
       # Verify matched vs fixed decomposition per-example at native float64 precision.
       teacher_only=dot(p0-tr,p0-tr)/fixed_den
       interaction=(matched-fixed-dot(p0-tr,p0-tr)+e0)[valid]/fixed_den
       exact=(-2*dot(dP,dT)[valid])/fixed_den
       res=float((interaction-exact).abs().max())/(1+abs(float(exact.mean())))
       diagn['decomposition']=max(diagn['decomposition'],res)
      if bool(validm.any()):row['MATCHED_E2']=float((matched[validm]/mode_den).mean())
      # R62 evidence: source-response vectors and exact per-example metrics (B=8).
      # Invalid denominators are retained explicitly as null; no silent exclusions.
      row['PER_SAMPLE']={
       'REF4':t0.tolist(), 'MODE_REF4':tr.tolist(), 'PRED4':pr.tolist(),
       'FIXED_E2':[float(fixed[z]/den[z]) if bool(valid[z]) else None for z in range(B)],
       'MATCHED_E2':[float(matched[z]/denm[z]) if bool(validm[z]) else None for z in range(B)],
       'GAIN':[float(dot(pr,t0)[z]/den[z]) if bool(valid[z]) else None for z in range(B)],
       'COS':[float(dot(pr,t0)[z]/(torch.sqrt(dot(pr,pr)[z]*den[z])+1e-30)) if bool(valid[z]) else None for z in range(B)]}
      need(all(math.isfinite(v) for key in ('REF4','MODE_REF4','PRED4') for vec in row['PER_SAMPLE'][key] for v in vec),'R62_NONFINITE_RESPONSE')
      rows[(g,c,mode,a)]=row
    # Independent native and dynamic-step forward parity, plus gauge covariance on excluded seed.
    if g=='D1_P2' and c=='weak' and preflight:
     for mi,mode in enumerate(modes):
      st=initial
      for h in range(1,21):st=graph_step_split(st,edge_base,edge_nonflat if is_nonflat(mode,h) else edge_base,teacher,torch)
      batchstate=(teacher_states[0][mi*per:(mi+1)*per],teacher_states[1][mi*per:(mi+1)*per])
      diagn['parity']=max(diagn['parity'],max(float((st[k]-batchstate[k]).abs().max()) for k in (0,1)))
      for j,a in enumerate(arms):
       st=initial
       for h in range(1,21):st=graph_step_split(st,edge_base,edge_nonflat if is_nonflat(mode,h) else edge_base,models[a],torch)
       sl=slice((mi*n+j)*per,(mi*n+j+1)*per)
       batchstate=(model_states[0][sl],model_states[1][sl])
       diagn['parity']=max(diagn['parity'],max(float((st[k]-batchstate[k]).abs().max()) for k in (0,1)))
       if mode=='BASE_AB':
        native=r4.rollout(initial,edge_base,models[a],20,torch)[-1]
        diagn['parity']=max(diagn['parity'],max(float((native[k]-batchstate[k]).abs().max()) for k in (0,1)))
      # True gauge covariance of the time-switching B seam with both links gauged together.
      if mode in modes:
       gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
       gb=r4.links_gauge(edge_base,gauges,torch);gn=r4.links_gauge(edge_nonflat,gauges,torch)
       st=r4.gauge_transform(initial,gauges,torch)
       for h in range(1,21):st=graph_step_split(st,gb,gn if is_nonflat(mode,h) else gb,models['SAL_PCGRAD4'],torch)
       sl=slice((mi*n+arms.index('SAL_PCGRAD4'))*per,(mi*n+arms.index('SAL_PCGRAD4')+1)*per)
       expected=r4.gauge_transform((model_states[0][sl],model_states[1][sl]),gauges,torch)
       diagn['cov']=max(diagn['cov'],max(float((st[k]-expected[k]).abs().max()) for k in (0,1)))
  # Disconnected structure must remain source-isolated under every B time window.
  ev=cfg['prospective']['evaluations'];g=ev['disconnected_graph'];graphs[g]=tuple(tuple(v) for v in ev['graphs'][g])
  mapped,mats,init,flip=r3.generate(g,'weak',f['eval_offset']+seed*100109+8891,B,graphs,cases,torch)
  initial=(torch.cat((init[0],flip[0]),0),torch.cat((init[1],flip[1]),0))
  eb=r4.build_links(mapped,transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,torch),torch)
  en=r4.build_links(mapped,transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,torch),torch)
  for mode in modes:
   for law1 in (teacher,models['SAL_STATE8'],models['SAL_PCGRAD4']):
    st=initial
    for h in range(1,21):st=graph_step_split(st,eb,en if is_nonflat(mode,h) else eb,law1,torch)
    diagn['disconnected']=max(diagn['disconnected'],float(st[0][:,3,:].abs().max()))
 for a in arms:need(hashes[a]==r4.model_hash(models[a]),'R61_INFERENCE_WEIGHT_MUTATION:'+a)
 return rows,diagn

def r61_aggregate(rows,cfg):
 f=cfg['r61_early_steps'];out={}
 for c in f['cases']:
  for arm in f['arms']:
   for mode in f['modes']:
    rec=[v for (g,cc,m,a),vs in rows.items() if cc==c and m==mode and a==arm for v in vs]
    need(len(rec)==len(f['graphs'])*len(f['eval_seeds']),'R61_AGGREGATION_COUNT')
    def ew(name,den='VALID'):
     good=[x for x in rec if name in x]
     count=sum(x[den] for x in good)
     return sum(x[name]*x[den] for x in good)/count if count else float('nan')
    a={'N':sum(v['N'] for v in rec),'VALID':sum(v['VALID'] for v in rec),
       'MODE_VALID':sum(v['MODE_VALID'] for v in rec),'CELLS':len(rec)}
    for key in ('BASE_E2','FIXED_E2','GAIN_FIXED','COS_FIXED','TEACHER_SHIFT_REL','MODEL_SHIFT_REL'):
     a[key]=ew(key)
    a['MATCHED_E2']=ew('MATCHED_E2','MODE_VALID')
    for key in ('REF_ENERGY','MODE_REF_ENERGY','BASE_ABS_ERR','FIXED_ABS_ERR','MATCHED_ABS_ERR','MODEL_SHIFT','TEACHER_SHIFT','PRED_ENERGY'):
     a[key]=mean(v[key] for v in rec)
    a['ENERGY_NORM_FIXED']=a['FIXED_ABS_ERR']/(a['REF_ENERGY']+1e-30)
    a['ENERGY_NORM_MATCHED']=a['MATCHED_ABS_ERR']/(a['MODE_REF_ENERGY']+1e-30)
    a['WORST_FIXED']=max(v.get('WORST_FIXED',float('nan')) for v in rec)
    a['SEED_WINS_BASE']=sum(ew1<ew2 for ee in f['eval_seeds'] for ew1,ew2 in [
     (mean(x['FIXED_E2'] for (g,cc,m,aa),vs in rows.items() if cc==c and m==mode and aa==arm for x in [vs[f['eval_seeds'].index(ee)]]),
      mean(x['FIXED_E2'] for (g,cc,m,aa),vs in rows.items() if cc==c and m=='BASE_AB' and aa==arm for x in [vs[f['eval_seeds'].index(ee)]]))])
    out[(c,arm,mode)]=a
 return out

def r61_step_contrasts(summary,rows,cfg):
 f=cfg['r61_early_steps'];out={}
 modes=list(f['modes']);need('B_H01_04' in modes and 'B_H02_04' in modes,'R61_MISSING_CONTROLS')
 for c in f['cases']:
  for a in f['arms']:
   E=lambda m:summary[(c,a,m)]['FIXED_E2']
   e0=E('BASE_AB')
   single={i:E('B_H0'+str(i))-e0 for i in range(1,5)}
   prefix={1:E('B_H01'),2:E('B_H01_02'),3:E('B_H01_03'),4:E('B_H01_04')}
   marg={1:single[1],2:prefix[2]-prefix[1],3:prefix[3]-prefix[2],4:prefix[4]-prefix[3]}
   synergy=prefix[4]-e0-sum(single.values())
   step1_necessary=prefix[4]-E('B_H02_04') # negative => including step1 helps over H2-H4
   # Unfavorable signed errors are not silently transformed into "benefits".
   seed_effects=[];seed_synergy=[]
   for ix,seed in enumerate(f['eval_seeds']):
    by_mode={m:mean(rows[(g,c,m,a)][ix]['FIXED_E2'] for g in f['graphs']) for m in modes}
    seed_effects.append((seed,by_mode['B_H01_04']-by_mode['BASE_AB']))
    seed_synergy.append((seed,by_mode['B_H01_04']-sum(by_mode['B_H0'+str(i)] for i in range(1,5))+3*by_mode['BASE_AB']))
   out[(c,a)]={'BASE':e0,'SINGLES':single,'PREFIX':prefix,'MARGINAL':marg,'SYNERGY':synergy,
    'STEP1_ADD_TO_H2_H4':step1_necessary,'PREFIX_WINS':sum(v<0 for _,v in seed_effects),
    'SYNERGY_NEG':sum(v<0 for _,v in seed_synergy),'SYNERGY_POS':sum(v>0 for _,v in seed_synergy),
    'H5_8_EFFECT':E('B_H05_08')-e0,'ALL_EFFECT':E('B_ALL')-e0,
    'SYNERGY_IDENTITY_RESID':abs(synergy-(sum(marg.values())-sum(single.values())))}
 return out

# R62 additions: protocol validation, exact subset algebra, robust experiment controls.
# The original R61 mathematics and transitive source checks above remain inherited.

def r62_modes():
 out={'BASE_AB':[]}
 for mask in range(16):
  out[f'MASK_{mask:04b}']=[[h,h] for h in range(1,5) if mask&(1<<(h-1))]
 out['B_H05_08']=[[5,8]]
 out['B_ALL']=[[1,20]]
 return out

def r62_validate_protocol(delta,parent):
 f=delta['r62_full_factorial'];p=parent['r61_early_steps']
 need(delta['parent_r61_archive']=='YM_AWR_R60_EARLY_B_SINGLE_STEP_PREFIX_R61.zip','R62_PARENT_NAME')
 need(delta['parent_r61_sha256']=='0d09605632e30850902d9f558f753aa6051e991a23d401ac016e5b9fd215bda3','R62_PARENT_SHA')
 need(delta['parent_r61_protocol_sha256']=='632222d5ecfdb8fd0acb51a6de732a819251bfe2cdc3351e900be78445a9dda4','R62_PARENT_PROTOCOL_SHA')
 need(delta['parent_r61_script_sha256']=='5c439ca36f1dfed9f6e3286c6e2785ad13134ec7e1d97ca04efb538faeb71cf1','R62_PARENT_SCRIPT_SHA')
 need(f['status']=='FROZEN_BEFORE_NEW_SEEDS','R62_FROZEN_STATUS')
 need(f['bit_convention']=='MASK_b4b3b2b1;LSB=H1;BIT1=H2;BIT2=H3;MSB=H4','R62_BIT_ORDER')
 need(list(f['modes'])==list(r62_modes()) and f['modes']==r62_modes(),'R62_MODE_COVERAGE')
 need(len(f['modes'])==19 and len({int(m.split('_')[1],2) for m in f['modes'] if m.startswith('MASK_')})==16,'R62_DUPLICATE_MASK')
 need(f['graphs']==p['graphs'] and f['cases']==p['cases'] and f['arms']==p['arms'],'R62_GRAPH_CASE_ARM_DRIFT')
 need(f['technical']==p['technical'] and f['primary_horizon']==20 and f['steps']==20,'R62_TECH_DRIFT')
 need(f['eval_offset']==82040000000 and f['eval_batch']==8 and f['source_ratio']==64,'R62_EVAL_DRIFT')
 need(f['eval_seeds']==[6301,6311,6323,6337,6343,6353,6361,6373] and f['excluded_preflight_seed']==6397,'R62_SEED_DRIFT')
 need(len(set(f['eval_seeds']+[f['excluded_preflight_seed']]))==9,'R62_SEED_DUPLICATE')
 need(f['teacher_reference']=='FIXED_BASE_AB_TEACHER' and f['mode_matched_teacher']=='SAME_MODE_INTERVENED_TEACHER','R62_REF_DRIFT')
 need(f['no_retraining'] is True and f['no_weight_mutation'] is True,'R62_LEARNING_DRIFT')
 need(f['preflight_gauge_mode']=='MASK_0111','R62_GAUGE_DRIFT')
 need(f['uncertainty']=={'method':'SEED_CLUSTER_PERCENTILE_BOOTSTRAP','draws':2048,'random_seed':16262,'alpha':0.05},'R62_UNCERTAINTY_DRIFT')
 need(f['report']['maximum_bytes']==20000000 and f['report']['format']=='JSON_IN_ZIP','R62_REPORT_DRIFT')
 return True

def r62_check_seed(rows,f,preflight=False):
 required={(g,c,m,a) for g in f['graphs'] for c in f['cases'] for m in f['modes'] for a in f['arms']}
 need(set(rows)==required,'R62_CELL_COVERAGE')
 for (g,c,m,a),v in rows.items():
  need(v['N']==f['eval_batch'] and 0<=v['VALID']<=v['N'] and 0<=v['MODE_VALID']<=v['N'],'R62_VALID_COUNTS')
  ps=v.get('PER_SAMPLE');need(isinstance(ps,dict),'R62_MISSING_PER_SAMPLE')
  for key in ('REF4','MODE_REF4','PRED4','FIXED_E2','MATCHED_E2','GAIN','COS'):
   need(len(ps[key])==v['N'],'R62_SAMPLE_LENGTH:'+key)
  need(sum(x is not None for x in ps['FIXED_E2'])==v['VALID'],'R62_FIXED_DENOMINATORS')
  need(sum(x is not None for x in ps['MATCHED_E2'])==v['MODE_VALID'],'R62_MATCHED_DENOMINATORS')
  for key in ('FIXED_E2','MATCHED_E2','GAIN','COS'):
   need(all(x is None or math.isfinite(x) for x in ps[key]),'R62_NONFINITE_SAMPLE:'+key)
  if m.startswith('MASK_'):
   base=rows[(g,c,'BASE_AB',a)]
   need(v['VALID']==base['VALID'] and v['N']==base['N'],'R62_FIXED_REFERENCE_CARDINALITY')
  if m=='MASK_0000':
   base=rows[(g,c,'BASE_AB',a)]
   for key in ('FIXED_E2','MATCHED_E2','GAIN_FIXED','COS_FIXED','PRED_ENERGY','REF_ENERGY','MODE_REF_ENERGY','TEACHER_SHIFT','MODEL_SHIFT'):
    need(v.get(key)==base.get(key),'R62_BASE_DUPLICATE_MISMATCH:'+key)
   need(v['PER_SAMPLE']==base['PER_SAMPLE'],'R62_BASE_VECTOR_MISMATCH')
 need(len(rows)==len(required),'R62_MASK_MULTIPLICITY')
 return True

def r62_mobius(F):
 need(set(F)==set(range(16)),'R62_MOBIUS_COVERAGE')
 i={}
 for mask in range(16):
  i[mask]=math.fsum((-1 if ((mask.bit_count()-sub.bit_count())%2) else 1)*F[sub] for sub in range(16) if (sub&mask)==sub)
 residual=max(abs(math.fsum(i[sub] for sub in range(16) if (sub&mask)==sub)-F[mask]) for mask in range(16))
 return i,residual

def r62_shapley(F):
 need(set(F)==set(range(16)),'R62_SHAPLEY_COVERAGE')
 phi={}
 for h in range(1,5):
  bit=1<<(h-1)
  phi[h]=math.fsum(
   math.factorial(mask.bit_count())*math.factorial(3-mask.bit_count())/24.0 * (F[mask|bit]-F[mask])
   for mask in range(16) if not mask&bit)
 residual=abs(math.fsum(phi.values())-(F[15]-F[0]))
 return phi,residual

def r62_percentile(values,q):
 need(len(values)>0 and 0<=q<=1,'R62_CI_INPUT')
 s=sorted(values);pos=q*(len(s)-1);lo=int(pos);hi=min(lo+1,len(s)-1)
 return s[lo]+(pos-lo)*(s[hi]-s[lo])

def r62_interactions(rows,f):
 import random
 out={};seeds=f['eval_seeds'];graphs=f['graphs']
 for c in f['cases']:
  for a in f['arms']:
   Fs=[];seeds_detail=[];graph_detail={}
   for si,seed in enumerate(seeds):
    Fs_graph={}
    for g in graphs:
     cell={mask:rows[(g,c,f'MASK_{mask:04b}',a)][si] for mask in range(16)}
     n={x['VALID'] for x in cell.values()}
     need(len(n)==1 and next(iter(n))>0,'R62_ZERO_REFERENCE_OR_MODE_VARIANCE')
     Fg={mask:cell[mask]['FIXED_E2'] for mask in range(16)}
     need(all(math.isfinite(z) for z in Fg.values()),'R62_NONFINITE_F')
     I,ir=r62_mobius(Fg);S,sr=r62_shapley(Fg)
     scale=max(1.,*(abs(z) for z in Fg.values()))
     need(ir/scale<=1e-10 and sr/scale<=1e-10,'R62_GRAPH_ATTRIBUTION_IDENTITY')
     Fs_graph[g]=Fg
     graph_detail.setdefault(g,[]).append({'seed':seed,'F':Fg,'mobius':I,'shapley':S,'i4':I[15],'h1_empty':Fg[1]-Fg[0]})
    Fseed={m:statistics.mean(Fs_graph[g][m] for g in graphs) for m in range(16)}
    I,ir=r62_mobius(Fseed);S,sr=r62_shapley(Fseed)
    scale=max(1.,*(abs(z) for z in Fseed.values()))
    need(ir/scale<=1e-10 and sr/scale<=1e-10,'R62_SEED_ATTRIBUTION_IDENTITY')
    seeds_detail.append({'seed':seed,'F':Fseed,'mobius':I,'shapley':S,'i4':I[15],'h1_empty':Fseed[1]-Fseed[0]})
    Fs.append(Fseed)
   F={m:statistics.mean(v[m] for v in Fs) for m in range(16)}
   I,ir=r62_mobius(F);S,sr=r62_shapley(F)
   scale=max(1.,*(abs(z) for z in F.values()))
   # Seed-cluster resampling preserves graph/case/mask pairing, no IID graph fiction.
   rng=random.Random(f['uncertainty']['random_seed']+100*f['cases'].index(c)+f['arms'].index(a))
   b_h1=[];b_i4=[];b_mobius={m:[] for m in range(1,16)};b_shapley={i:[] for i in range(1,5)}
   for _ in range(f['uncertainty']['draws']):
    inds=[rng.randrange(len(Fs)) for __ in Fs]
    b={m:statistics.mean(Fs[k][m] for k in inds) for m in range(16)}
    b_s,_=r62_shapley(b);b_i,_=r62_mobius(b)
    b_h1.append(b_s[1]);b_i4.append(b_i[15])
    for m in range(1,16):b_mobius[m].append(b_i[m])
    for h in range(1,5):b_shapley[h].append(b_s[h])
   phi_impact=[F[mask|1]-F[mask] for mask in range(16) if not mask&1]
   out[(c,a)]={'F':F,'interactions':I,'first_order':{h:I[1<<(h-1)] for h in range(1,5)},
    'pairwise':{f'H{i}_H{j}':I[(1<<(i-1))|(1<<(j-1))] for i in range(1,5) for j in range(i+1,5)},
    'third_order':{str(m):I[m] for m in range(16) if m.bit_count()==3},
    'shapley':S,'h1_conditional_range':[min(phi_impact),max(phi_impact)],
    'mobius_resid_abs':ir,'shapley_resid_abs':sr,'mobius_resid_rel':ir/scale,'shapley_resid_rel':sr/scale,
    'shapley_ci95_h1':[r62_percentile(b_h1,.025),r62_percentile(b_h1,.975)],
    'i4_ci95':[r62_percentile(b_i4,.025),r62_percentile(b_i4,.975)],
    'mobius_ci95':{m:[r62_percentile(b_mobius[m],.025),r62_percentile(b_mobius[m],.975)] for m in range(1,16)},
    'shapley_ci95':{h:[r62_percentile(b_shapley[h],.025),r62_percentile(b_shapley[h],.975)] for h in range(1,5)},
    'mobius_seed_median':{m:statistics.median(x['mobius'][m] for x in seeds_detail) for m in range(1,16)},
    'shapley_seed_median':{h:statistics.median(x['shapley'][h] for x in seeds_detail) for h in range(1,5)},
    'by_graph_mean':{g:{'F':{m:statistics.mean(x['F'][m] for x in graph_detail[g]) for m in range(16)},
                        'shapley':{h:statistics.mean(x['shapley'][h] for x in graph_detail[g]) for h in range(1,5)},
                        'mobius':{m:statistics.mean(x['mobius'][m] for x in graph_detail[g]) for m in range(16)}} for g in graphs},
    'by_seed':seeds_detail,'by_graph':graph_detail}
 return out

def r62_report(rows,summary,contrasts,delta,psha,root):
 """Bounded diagnostic handoff only after inference and immutable-source postchecks."""
 from io import BytesIO
 data={'phase':'YM_AWR_R61_FULL_EARLY_B_FACTORIAL_R62','scientific_disposition':'RETROSPECTIVE_DIAGNOSTIC_ONLY',
       'protocol_sha256':psha,'parent_r61_sha256':delta['parent_r61_sha256'],
       'no_native_gdn_or_learned_utility_claim':True,
       'rows':[{'graph':g,'case':c,'mode':m,'arm':a,'seed':delta['r62_full_factorial']['eval_seeds'][i],**v}
              for (g,c,m,a),vs in sorted(rows.items()) for i,v in enumerate(vs)],
       'aggregates':[{'case':c,'arm':a,'mode':m,**v} for (c,a,m),v in sorted(summary.items())],
       'interactions':[{'case':c,'arm':a,**v} for (c,a),v in sorted(contrasts.items())]}
 blob=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 need(len(blob)<=delta['r62_full_factorial']['report']['maximum_bytes'],'R62_REPORT_SIZE_LIMIT')
 zbuf=BytesIO()
 with zipfile.ZipFile(zbuf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  info=zipfile.ZipInfo('R62_RESULT.json',(2026,11,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
  info.external_attr=(0o600<<16)
  z.writestr(info,blob,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 report=zbuf.getvalue();h=sha(report)
 name='YM_AWR_R62_RESULT_'+h[:16]+'.zip'
 path=root/name
 # An existing immutable handoff is not overwritten by accident.
 if path.exists():
  need(filesha(path)==h,'R62_REPORT_NAME_COLLISION')
 else:
  with path.open('xb') as fout:fout.write(report)
 need(filesha(path)==h,'R62_REPORT_WRITTEN_MISMATCH')
 return path,h

def r62_selftest():
 try:
  delta=json.loads(Path(__file__).with_name('FROZEN_PROTOCOL_R62.json').read_bytes())
  p={'r61_early_steps':{'graphs':['D1_P1','D1_P2','D2_P1','D2_P2'],
     'cases':['weak','reverse','anti'],'arms':['SAL_STATE8','SAL_PCGRAD4'],
     'technical':delta['r62_full_factorial']['technical']}}
  assert r62_validate_protocol(delta,p)
  assert list(r62_modes())[1]=='MASK_0000' and r62_modes()['MASK_1111']==[[1,1],[2,2],[3,3],[4,4]]
  assert len(r62_modes())==19 and len({k for k in r62_modes() if k.startswith('MASK_')})==16
  F={m:7+2*(bool(m&1))-3*(bool(m&2))+5*((m&3)==3)+4*(bool(m&8)) for m in range(16)}
  I,ir=r62_mobius(F);S,sr=r62_shapley(F)
  assert ir<1e-12 and sr<1e-12
  assert I[0]==7 and I[1]==2 and I[2]==-3 and I[3]==5 and I[8]==4 and I[15]==0
  assert abs(S[1]-4.5)<1e-12 and abs(S[2]+.5)<1e-12
  # Non-additive fourth-order coefficient, with exact reconstruction.
  G={m:F[m]+11*(m==15) for m in range(16)}
  J,jr=r62_mobius(G);T,tr=r62_shapley(G)
  assert abs(J[15]-11)<1e-12 and jr<1e-12 and tr<1e-12
  assert abs(sum(T.values())-(G[15]-G[0]))<1e-12
  # BREAK tests must catch the forbidden paths, not swallow self-generated asserts.
  def reject(fn):
   try:fn()
   except (Hold,KeyError):return
   raise AssertionError('FORBIDDEN_INPUT_ACCEPTED')
  reject(lambda:r62_mobius({0:1}))
  reject(lambda:r62_shapley({0:1}))
  bad=json.loads(json.dumps(delta));bad['r62_full_factorial']['modes']['MASK_0001']=[[2,2]]
  reject(lambda:r62_validate_protocol(bad,p))
  bad=json.loads(json.dumps(delta));bad['r62_full_factorial']['eval_seeds'][2]=bad['r62_full_factorial']['eval_seeds'][0]
  reject(lambda:r62_validate_protocol(bad,p))
  bad=json.loads(json.dumps(delta));bad['r62_full_factorial']['bit_convention']='MSB=H1'
  reject(lambda:r62_validate_protocol(bad,p))
  print('=== SEND BACK ===')
  print('PHASE=YM_AWR_R62_SELFTEST RESULT=PASS')
  print('TESTS=PROTOCOL_MASK_COVERAGE_MOBIUS_SHAPLEY_I4_NEGATIVE_CASES_PASS')
  print('DOMAIN_R62=NOT_EXECUTED PARENT_CHAIN=NOT_EXECUTED')
  print('CHILD_EXIT=0')
  print('=== END SEND BACK ===')
  return 0
 except Exception as e:
  print('=== SEND BACK ===')
  print('PHASE=YM_AWR_R62_SELFTEST RESULT=HOLD BLOCKER='+str(e))
  print('CHILD_EXIT=2')
  print('=== END SEND BACK ===')
  return 2

def run():
 mode=sys.argv[1] if len(sys.argv)>1 else 'run'
 local=mode.startswith('local_');full=mode in ('run','local_run')
 status='HOLD';pins='NOT_RUN';proto='NOT_RUN';postcheck='NOT_RUN';blocker='NONE';rc=1;science='NOT_RUN';psha='NOT_RUN';cov=0.;disc=0.;minref=0.;max_parity=0.;gates={};metrics={}
 done=[];fitstats={};summary={};contrasts={};aggregates={};diag={};max_parity=0.;max_ref=0.;r49ok=False;contrasts={}
 try:
  need(mode in ('run','local_run','preflight','local_preflight'),'MODE')
  import torch
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_default_dtype(torch.float64)
  need(Path('/dev/shm').is_dir(),'DEV_SHM_MISSING')
  blob=Path(__file__).with_name('FROZEN_PROTOCOL_R62.json').read_bytes();r62cfg=json.loads(blob);psha=sha(blob)
  need(r62cfg['schema']=='elpis.ym.awr.g5.r62_full_early_b_factorial.v1' and r62cfg['status']=='FROZEN_BEFORE_NEW_SEEDS' and r62cfg['parent_r61_sha256']=='0d09605632e30850902d9f558f753aa6051e991a23d401ac016e5b9fd215bda3','R62_PROTOCOL_SCHEMA')
  root=Path(os.getenv('YM_R61_LOCAL_ROOT','/mnt/data')) if local else ROOT
  r61zip=root/r62cfg['parent_r61_archive']
  need(r61zip.is_file() and filesha(r61zip)==r62cfg['parent_r61_sha256'],'R61_ARCHIVE')
  pblob=zin(r61zip,'FROZEN_PROTOCOL_R61.json')
  need(sha(pblob)==r62cfg['parent_r61_protocol_sha256'],'R61_PROTOCOL_PIN')
  need(sha(zin(r61zip,'r61.py'))==r62cfg['parent_r61_script_sha256'],'R61_SCRIPT_PIN')
  cfg=json.loads(pblob)
  need(cfg['r61_early_steps']['eval_seeds']==[6203,6211,6221,6233,6247,6257,6263,6277],'R61_PARENT_SEEDS')
  need(r62_validate_protocol(r62cfg,cfg),'R62_PROTOCOL_DRIFT')
  need(cfg['schema']=='elpis.ym.awr.g5.r61_early_b_individual_prefix_factorial.v1' and cfg['status']=='FROZEN_RETROSPECTIVE_EARLY_B_STEP_ATTRIBUTION_NO_UTILITY','PROTOCOL_SCHEMA')
  need(cfg['diagnostic']['horizons']==[8,12,16,20] and cfg['diagnostic']['record_indices']==[0,5,12,17,23,35] and cfg['diagnostic']['preflight_indices']==[0] and cfg['diagnostic']['arms']==['BASE_LS','SAL_STATE8','SAL_ORIG_SIGNED8','SAL_SCALED_SIGNED8'],'AUDIT_DRIFT')
  need(cfg['combination']['methods']==['H8_ONLY','H20_ONLY','MEAN4','NORM4','PCGRAD4','MINNORM8_20'] and cfg['combination']['hypothetical_clip']==1. and cfg['combination']['no_training'] and cfg['combination']['no_holdout_access'],'COMBINATION_PROTOCOL_DRIFT')
  need(cfg['base_fit_seeds']==[3607,3617] and cfg['adapt_train_seeds']==[3701,3719] and cfg['evaluation_seeds']==[3907,3911,3919,3923,3931,3943,3947,3967],'R48_SEED_DRIFT')
  need(cfg['updates']==64 and cfg['adapt_horizon']==8 and cfg['response_horizons']==[4,6,8] and cfg['learning_rate']==.003,'TRAIN_DRIFT')
  root=Path(os.getenv('YM_R61_LOCAL_ROOT','/mnt/data')) if local else ROOT
  # Direct source chain R60 -> R59 -> ... -> R0 must remain immutable.
  r60zip=root/cfg['r60_archive']
  need(r60zip.is_file() and filesha(r60zip)==cfg['r60_sha256'],'R60_ARCHIVE')
  need(sha(zin(r60zip,'r60.py'))==cfg['r60_script_sha256'] and sha(zin(r60zip,'FROZEN_PROTOCOL_R60.json'))==cfg['r60_protocol_sha256'],'R60_SOURCE_PROTOCOL')
  parent60=json.loads(zin(r60zip,'FROZEN_PROTOCOL_R60.json'))
  for key,value in parent60.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R60_PARENT_DRIFT:'+key)
  t61=cfg['r61_early_steps']
  need(t61['status']=='FROZEN_BEFORE_NEW_SEEDS' and t61['eval_seeds']==[6203,6211,6221,6233,6247,6257,6263,6277] and t61['excluded_preflight_seed']==6299,'R61_SEED_DRIFT')
  need(t61['graphs']==cfg['r60_windows']['graphs'] and t61['cases']==cfg['r60_windows']['cases'] and t61['arms']==['SAL_STATE8','SAL_PCGRAD4'],'R61_GRAPH_CASE_ARM_DRIFT')
  need(t61['modes']=={'BASE_AB':[],'B_H01':[[1,1]],'B_H02':[[2,2]],'B_H03':[[3,3]],'B_H04':[[4,4]],'B_H01_02':[[1,2]],'B_H01_03':[[1,3]],'B_H01_04':[[1,4]],'B_H02_04':[[2,4]],'B_H05_08':[[5,8]],'B_ALL':[[1,20]]},'R61_MODE_DRIFT')
  need(t61['eval_offset']==81420000000 and t61['eval_batch']==8 and t61['source_ratio']==64 and t61['steps']==20,'R61_EVAL_DRIFT')
  history=[]
  for oldkey in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds'):history+=cfg[oldkey]
  for key in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds'):history+=cfg['prospective'][key]
  for key,name in [('r54_factorial','evaluation_seeds'),('r55_structural','evaluation_seeds'),('r56_transport','evaluation_seeds'),('r57_channels','eval_seeds'),('r58_reference','eval_seeds'),('r59_trace','eval_seeds'),('r60_windows','eval_seeds')]:history+=cfg[key][name]
  need(len(set(t61['eval_seeds']+[t61['excluded_preflight_seed']]))==9 and not set(history).intersection(t61['eval_seeds']+[t61['excluded_preflight_seed']]),'R61_HISTORY_SEED_OVERLAP')
  t62=r62cfg['r62_full_factorial']
  # Distinct source/fixture seeds and the excluded preflight are verified before inference.
  prior=set(history+t61['eval_seeds']+[t61['excluded_preflight_seed']])
  need(not prior.intersection(t62['eval_seeds']+[t62['excluded_preflight_seed']]),'R62_HISTORY_SEED_OVERLAP')
  need(t62['eval_offset'] not in (t61['eval_offset'],cfg['r60_windows']['eval_offset']),'R62_EVAL_OFFSET_DRIFT')
  # R59 is the byte-identical parent authority; keep every older source/protocol check.
  r59zip=root/cfg['r59_archive']
  need(r59zip.is_file() and filesha(r59zip)==cfg['r59_sha256'],'R59_ARCHIVE')
  need(sha(zin(r59zip,'r59.py'))==cfg['r59_script_sha256'] and sha(zin(r59zip,'FROZEN_PROTOCOL_R59.json'))==cfg['r59_protocol_sha256'],'R59_SOURCE_PROTOCOL')
  parent59=json.loads(zin(r59zip,'FROZEN_PROTOCOL_R59.json'))
  for key,value in parent59.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R59_PARENT_DRIFT:'+key)
  t60=cfg['r60_windows']
  need(t60['status']=='FROZEN_BEFORE_NEW_SEEDS' and t60['eval_seeds']==[6101,6113,6121,6131,6143,6151,6163,6173] and t60['excluded_preflight_seed']==6193,'R60_SEED_DRIFT')
  need(t60['graphs']==cfg['r59_trace']['graphs'] and t60['cases']==cfg['r59_trace']['cases'] and t60['arms']==['SAL_STATE8','SAL_PCGRAD4'],'R60_GRAPH_CASE_ARM_DRIFT')
  need(t60['modes']=={'BASE_AB':[],'B_H01_04':[[1,4]],'B_H05_08':[[5,8]],'B_H09_12':[[9,12]],'B_H13_16':[[13,16]],'B_H17_20':[[17,20]],'B_ALL':[[1,20]]},'R60_WINDOW_DRIFT')
  need(t60['eval_offset']==80930000000 and t60['eval_batch']==8 and t60['steps']==20 and t60['source_ratio']==64,'R60_EVAL_DRIFT')
  for oldkey in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds'):
   need(not set(cfg[oldkey]).intersection(t60['eval_seeds']+[t60['excluded_preflight_seed']]),'R60_R48_SEED_OVERLAP')
  for group in (cfg['prospective']['base_fit_seeds'],cfg['prospective']['adapt_train_seeds'],cfg['prospective']['evaluation_seeds'],cfg['r54_factorial']['evaluation_seeds'],cfg['r55_structural']['evaluation_seeds'],cfg['r56_transport']['evaluation_seeds'],cfg['r57_channels']['eval_seeds'],cfg['r58_reference']['eval_seeds'],cfg['r59_trace']['eval_seeds']):
   need(not set(group).intersection(t60['eval_seeds']+[t60['excluded_preflight_seed']]),'R60_SEED_OVERLAP')
  # R58 is the immutable direct parent, with exact byte-pinned source/protocol.
  r58zip=root/cfg['r58_archive']
  need(r58zip.is_file() and filesha(r58zip)==cfg['r58_sha256'],'R58_ARCHIVE')
  need(sha(zin(r58zip,'r58.py'))==cfg['r58_script_sha256'] and sha(zin(r58zip,'FROZEN_PROTOCOL_R58.json'))==cfg['r58_protocol_sha256'],'R58_SOURCE_PROTOCOL')
  parent58=json.loads(zin(r58zip,'FROZEN_PROTOCOL_R58.json'))
  for key,value in parent58.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R58_PARENT_DRIFT:'+key)
  t59=cfg['r59_trace']
  need(t59['status']=='FROZEN_BEFORE_NEW_SEEDS' and t59['eval_seeds']==[6007,6011,6029,6037,6043,6053,6067,6073] and t59['excluded_preflight_seed']==6091,'R59_SEED_DRIFT')
  need(t59['horizons']==list(range(1,21)) and t59['analyzed_arms']==['SAL_STATE8','SAL_PCGRAD4'] and t59['analyzed_modes']==['BASE_AB','NONFLAT_B'],'R59_HORIZON_DRIFT')
  need(t59['graphs']==cfg['r58_reference']['graphs'] and t59['cases']==cfg['r58_reference']['cases'] and t59['arms']==cfg['r58_reference']['arms'] and t59['modes']==cfg['r58_reference']['modes'],'R59_PARENT_DESIGN_DRIFT')
  need(t59['eval_offset']==80610000000 and t59['eval_batch']==8 and t59['source_ratio']==64 and t59['no_retraining'],'R59_EVAL_DRIFT')
  prior_seeds=(cfg['base_fit_seeds']+cfg['adapt_train_seeds']+cfg['evaluation_seeds']+cfg['prospective']['base_fit_seeds']+cfg['prospective']['adapt_train_seeds']+cfg['prospective']['evaluation_seeds']+
    cfg['r54_factorial']['evaluation_seeds']+cfg['r55_structural']['evaluation_seeds']+cfg['r56_transport']['evaluation_seeds']+cfg['r57_channels']['eval_seeds']+cfg['r58_reference']['eval_seeds'])
  need(not set(prior_seeds).intersection(t59['eval_seeds']+[t59['excluded_preflight_seed']]),'R59_SEED_OVERLAP')
  # Exact R57 parent chain and full historical protocol inherited without mutation.
  r57zip=root/cfg['r57_archive']
  need(r57zip.is_file() and filesha(r57zip)==cfg['r57_sha256'],'R57_ARCHIVE')
  need(sha(zin(r57zip,'r57.py'))==cfg['r57_script_sha256'] and sha(zin(r57zip,'FROZEN_PROTOCOL_R57.json'))==cfg['r57_protocol_sha256'],'R57_SOURCE_PROTOCOL')
  r57cfg=json.loads(zin(r57zip,'FROZEN_PROTOCOL_R57.json'))
  for key,value in r57cfg.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R57_PARENT_DRIFT:'+key)
  # Direct R56 frozen source authority, including complete unchanged parent configuration.
  r56zip=root/cfg['r56_archive']
  need(r56zip.is_file() and filesha(r56zip)==cfg['r56_sha256'],'R56_ARCHIVE')
  need(sha(zin(r56zip,'r56.py'))==cfg['r56_script_sha256'] and sha(zin(r56zip,'FROZEN_PROTOCOL_R56.json'))==cfg['r56_protocol_sha256'],'R56_SOURCE_PROTOCOL')
  r56cfg=json.loads(zin(r56zip,'FROZEN_PROTOCOL_R56.json'))
  for key,value in r56cfg.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R56_PARENT_DRIFT:'+key)
  # R55 is the exact frozen parent: reject protocol or source drift before any fit.
  r55zip=root/cfg['r55_archive']
  need(r55zip.is_file() and filesha(r55zip)==cfg['r55_sha256'],'R55_ARCHIVE')
  need(sha(zin(r55zip,'r55.py'))==cfg['r55_script_sha256'] and sha(zin(r55zip,'FROZEN_PROTOCOL_R55.json'))==cfg['r55_protocol_sha256'],'R55_SOURCE_PROTOCOL')
  parent55=json.loads(zin(r55zip,'FROZEN_PROTOCOL_R55.json'))
  for key,value in parent55.items():
   if key not in ('schema','title','status','science'):
    need(cfg[key]==value,'R55_PARENT_DRIFT:'+key)
  # R54 is the direct source authority. Its historical protocol fields are frozen unchanged.
  r54zip=root/cfg['r54_archive']
  need(r54zip.is_file() and filesha(r54zip)==cfg['r54_sha256'],'R54_ARCHIVE')
  need(sha(zin(r54zip,'r54.py'))==cfg['r54_script_sha256'] and sha(zin(r54zip,'FROZEN_PROTOCOL_R54.json'))==cfg['r54_protocol_sha256'],'R54_SOURCE_PROTOCOL')
  parent54=json.loads(zin(r54zip,'FROZEN_PROTOCOL_R54.json'))
  for key,value in parent54.items():
   if key not in ('schema','title','status','science'):need(cfg[key]==value,'R54_PARENT_DRIFT:'+key)
  # Direct R53 archive and contract source pins; R52 and all predecessors verified below.
  r53zip=root/cfg['r53_archive']
  need(r53zip.is_file() and filesha(r53zip)==cfg['r53_sha256'],'R53_ARCHIVE')
  need(sha(zin(r53zip,'r53.py'))==cfg['r53_script_sha256'] and sha(zin(r53zip,'FROZEN_PROTOCOL_R53.json'))==cfg['r53_protocol_sha256'],'R53_SOURCE_PROTOCOL')
  parent53=json.loads(zin(r53zip,'FROZEN_PROTOCOL_R53.json'))
  for key,value in parent53.items():
   if key not in ('schema','title','status','science'):
    need(cfg[key]==value,'R53_PARENT_DRIFT:'+key)
  f=cfg['r54_factorial']
  need(f['status']=='FROZEN_BEFORE_NEW_SEEDS' and f['evaluation_seeds']==[5501,5503,5507,5519,5521,5527,5531,5557] and f['excluded_preflight_seed']==5591,'R54_PARENT_SEED_DRIFT')
  need(f['topology']=={'FAMILIAR':['deep_3a','split_2a','shortcut_1a'],'NOVEL':['r52_fan','r52_weave','r52_cross']} and f['amplitudes']=={'FAMILIAR64':.0625,'NOVEL160':.025} and f['cases']==['weak','reverse','anti'],'R54_FACTORIAL_DRIFT')
  need(f['arms']==['BASE_LS','SAL_STATE8','SAL_H20_ONLY','SAL_MEAN4','SAL_PCGRAD4'] and f['horizon']==20 and f['eval_batch']==8,'R54_ARM_DRIFT')
  # Newly frozen R55 topology rewires; do not claim perfectly orthogonal topology axes.
  s=cfg['r55_structural']
  need(s['status']=='FROZEN_BEFORE_NEW_SEEDS' and s['evaluation_seeds']==[5603,5609,5617,5623,5639,5641,5651,5669] and s['excluded_preflight_seed']==5693,'R55_SEED_DRIFT')
  need(list(s['graphs'])==['D1_P1','D1_P2','D2_P1','D2_P2'] and s['source_relation']==[0,1,2,7] and s['source_nodes']==[0,1],'R55_GRAPH_DRIFT')
  need(s['source_ratio']==64 and s['cases']=={'weak':[.0625,4.,'independent'],'reverse':[4.,.0625,'independent'],'anti':[.0625,4.,'antialigned']},'R55_CASE_DRIFT')
  need(s['arms']==['BASE_LS','SAL_STATE8','SAL_H20_ONLY','SAL_MEAN4','SAL_PCGRAD4'] and s['horizon']==20 and s['eval_batch']==8,'R55_ARM_DRIFT')
  need(s['degree_preserving_moves_only'] and not s['graphs_new_to_evaluation'] and s['no_retraining'] and s['no_eval_train_overlap'],'R55_NO_RETRAIN_DRIFT')
  structural={k:graph_structure(v) for k,v in s['graphs'].items()}
  for k,m in structural.items():
   need(m['PARTICIPANT_DEGREES']==[1,1,1,1,2,2,2,2,2,2] and m['DEGREE_HIST']==s['degree_histogram'],'R55_DEGREE:'+k)
   need(s['graphs'][k][0]==s['source_relation'],'R55_SOURCE_RELATION:'+k)
   need([m['DIRECT'],m['TWOHOP']]==s['direct_2hop_targets'][k],'R55_PATH_COUNTS:'+k)
  need(s['graphs']['D1_P2']==[[0,1,2,7],[1,3,4,8],[2,4,5,9],[3,5,6,7]],'R55_ORIGINAL_CROSS_GRAPH_MISMATCH')
  # All R55 seed groups are fresh; the R54 family remains held separate.
  for xx in (cfg['evaluation_seeds'],cfg['adapt_train_seeds'],cfg['base_fit_seeds'],cfg['prospective']['base_fit_seeds'],cfg['prospective']['adapt_train_seeds'],cfg['prospective']['evaluation_seeds'],f['evaluation_seeds']):
   need(not set(xx).intersection(s['evaluation_seeds']+[s['excluded_preflight_seed']]),'R55_SEED_OVERLAP')
  # R56 transport design gates: new fresh seeds, original R55 structural family.
  t56=cfg['r56_transport']
  need(t56['status']=='FROZEN_BEFORE_NEW_SEEDS' and t56['evaluation_seeds']==[5701,5707,5711,5717,5723,5737,5741,5749] and t56['excluded_preflight_seed']==5779,'R56_SEED_DRIFT')
  need(t56['graphs']==list(s['graphs']) and t56['arms']==s['arms'] and set(t56['cases'])==set(s['cases']),'R56_GRAPH_ARM_CASE_DRIFT')
  need(t56['transport_modes']==['BASE_SEPARABLE','GAUGE_ONLY','PARTICIPANT_NONFLAT'] and t56['source_anchor_relation']==0 and t56['source_states_identical'] and t56['train_frozen'],'R56_TRANSPORT_DRIFT')
  need(t56['nonflat']=={'a_theta1':0.45,'a_theta2':-0.35,'r_scale':3,'p_factor_1':0.73,'p_factor_2':1.11} and t56['gauge_shift']==[0.21,-0.13],'R56_TRANSPORT_HYPERPARAM_DRIFT')
  need(t56['horizon']==20 and t56['eval_batch']==8 and t56['source_ratio']==64 and t56['eval_offset']==80170000000,'R56_EVAL_DRIFT')
  for xx in (cfg['base_fit_seeds'],cfg['adapt_train_seeds'],cfg['evaluation_seeds'],cfg['prospective']['base_fit_seeds'],cfg['prospective']['adapt_train_seeds'],cfg['prospective']['evaluation_seeds'],f['evaluation_seeds'],s['evaluation_seeds']):
   need(not set(xx).intersection(t56['evaluation_seeds']+[t56['excluded_preflight_seed']]),'R56_SEED_OVERLAP')
  # Newly frozen R57 split-channel attribution; unchanged physical transport parameters.
  ch=cfg['r57_channels']
  need(ch['status']=='FROZEN_BEFORE_NEW_SEEDS' and ch['eval_seeds']==[5801,5807,5813,5821,5827,5839,5843,5851] and ch['excluded_preflight_seed']==5897,'R57_SEED_DRIFT')
  need(ch['modes']==['BASE_AB','NONFLAT_A','NONFLAT_B','NONFLAT_AB','GAUGE_AB'] and ch['transport_source']=='EXACT_R56_PARTICIPANT_NONFLAT_AND_GAUGE_PHASES','R57_MODES_DRIFT')
  need(ch['arms']==t56['arms'] and ch['graphs']==t56['graphs'] and ch['cases']==t56['cases'] and ch['eval_batch']==8 and ch['source_ratio']==64 and ch['horizon']==20 and ch['eval_offset']==80390000000,'R57_DESIGN_DRIFT')
  need(ch['teacher_and_models_same_mode'] and ch['source_init_identical'] and ch['frozen_models'] and ch['no_new_learning'],'R57_MUTATION_DRIFT')
  for prior in (cfg['base_fit_seeds'],cfg['adapt_train_seeds'],cfg['evaluation_seeds'],cfg['prospective']['base_fit_seeds'],cfg['prospective']['adapt_train_seeds'],cfg['prospective']['evaluation_seeds'],f['evaluation_seeds'],s['evaluation_seeds'],t56['evaluation_seeds']):
   need(not set(prior).intersection(ch['eval_seeds']+[ch['excluded_preflight_seed']]),'R57_SEED_OVERLAP')
  # R58 novel seeds, unmodified R57 transport, fixed common teacher comparator.
  d=cfg['r58_reference']
  need(d['status']=='FROZEN_BEFORE_NEW_SEEDS' and d['eval_seeds']==[5903,5917,5923,5939,5953,5959,5963,5981] and d['excluded_preflight_seed']==5999,'R58_SEED_DRIFT')
  need(d['modes']==ch['modes'] and d['arms']==ch['arms'] and d['graphs']==ch['graphs'] and d['cases']==ch['cases'] and d['eval_batch']==8 and d['horizon']==20 and d['source_ratio']==64 and d['eval_offset']==80450000000,'R58_DESIGN_DRIFT')
  need(d['transport_source']=='EXACT_R57_SPLIT_CHANNEL_TRANSPORT_AND_GAUGE' and d['fixed_teacher']=='BASE_AB_TEACHER_ON_SAME_SEED_GRAPH_CASE' and d['denominator']=='PER_SAMPLE_NORM_SQUARED_BASE_AB_TEACHER_RESPONSE_UNFLOORED','R58_METRIC_DRIFT')
  need(d['no_new_training'] and d['frozen_models'] and d['gauge_excluded_from_common_frame_comparisons'] and d['teacher_and_models_same_transport_mode'],'R58_NO_NEW_TRAINING_DRIFT')
  for oldgroup in (cfg['base_fit_seeds'],cfg['adapt_train_seeds'],cfg['evaluation_seeds'],cfg['prospective']['base_fit_seeds'],cfg['prospective']['adapt_train_seeds'],cfg['prospective']['evaluation_seeds'],f['evaluation_seeds'],s['evaluation_seeds'],t56['evaluation_seeds'],ch['eval_seeds']):
   need(not set(oldgroup).intersection(d['eval_seeds']+[d['excluded_preflight_seed']]),'R58_SEED_OVERLAP')
  # Exact R52 source authority plus all unchanged historical configuration fields.
  r52zip=root/cfg['r52_archive']
  need(r52zip.is_file() and filesha(r52zip)==cfg['r52_sha256'],'R52_ARCHIVE')
  need(sha(zin(r52zip,'r52.py'))==cfg['r52_script_sha256'] and sha(zin(r52zip,'FROZEN_PROTOCOL_R52.json'))==cfg['r52_protocol_sha256'],'R52_SOURCE_PROTOCOL')
  r52cfg=json.loads(zin(r52zip,'FROZEN_PROTOCOL_R52.json'))
  for key,value in r52cfg.items():
   if key not in ('schema','title','status','science'):
    need(cfg[key]==value,'R52_FROZEN_PARENT_DRIFT:'+key)
  need(cfg['r53_diagnostic']['checkpoints']==[0,1,2,4,8,16,32,48,64] and cfg['r53_diagnostic']['probe_indices']==[0,3,6,18,21,24] and cfg['r53_diagnostic']['preflight_indices']==[0],'R53_DIAGNOSTIC_DRIFT')
  need(cfg['r53_diagnostic']['no_new_eval'] and cfg['r53_diagnostic']['no_new_objective'] and cfg['r53_diagnostic']['no_evaluation_seed_or_graph_access'],'R53_NO_EVAL_GATE')
  # R51 is retained as pinned transitive source authority; R50-R0 unchanged.
  r51zip=root/cfg['r51_archive']
  need(r51zip.is_file() and filesha(r51zip)==cfg['r51_sha256'],'R51_ARCHIVE')
  need(sha(zin(r51zip,'r51.py'))==cfg['r51_script_sha256'] and sha(zin(r51zip,'FROZEN_PROTOCOL_R51.json'))==cfg['r51_protocol_sha256'],'R51_SOURCE_PROTOCOL')
  original51=json.loads(zin(r51zip,'FROZEN_PROTOCOL_R51.json'))
  need(all(cfg[k]==original51[k] for k in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds','evaluations','arms','objective_variants','updates','learning_rate','adapt_horizon','response_horizons','adapt_fixtures_per_seed','train_cases','base_snapshots','base_fixtures_per_seed','diagnostic','combination')),'R51_PARENT_CONTRACT_DRIFT')
  # R50 is retained as pinned transitive authority.
  r50zip=root/cfg['r50_archive']
  need(r50zip.is_file() and filesha(r50zip)==cfg['r50_sha256'],'R50_ARCHIVE')
  need(sha(zin(r50zip,'r50.py'))==cfg['r50_script_sha256'] and sha(zin(r50zip,'FROZEN_PROTOCOL_R50.json'))==cfg['r50_protocol_sha256'],'R50_SOURCE_PROTOCOL')
  r50pcfg=json.loads(zin(r50zip,'FROZEN_PROTOCOL_R50.json'))
  need(all(cfg[k]==r50pcfg[k] for k in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds','evaluations','arms','objective_variants','updates','learning_rate','adapt_horizon','response_horizons','adapt_fixtures_per_seed','train_cases','base_snapshots','base_fixtures_per_seed')),'R50_PARENT_CONTRACT_DRIFT')
  need(cfg['diagnostic']['arms']==r50pcfg['diagnostic']['arms'] and cfg['diagnostic']['record_indices']==r50pcfg['diagnostic']['record_indices'] and cfg['diagnostic']['horizons']==r50pcfg['diagnostic']['horizons'],'R50_DIAGNOSTIC_DRIFT')
  r49zip=root/cfg['r49_archive']
  need(r49zip.is_file() and filesha(r49zip)==cfg['r49_sha256'],'R49_ARCHIVE')
  need(sha(zin(r49zip,'r49.py'))==cfg['r49_script_sha256'] and sha(zin(r49zip,'FROZEN_PROTOCOL_R49.json'))==cfg['r49_protocol_sha256'],'R49_SOURCE_PROTOCOL')
  r49pcfg=json.loads(zin(r49zip,'FROZEN_PROTOCOL_R49.json'))
  need(all(cfg[k]==r49pcfg[k] for k in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds','evaluations','arms','objective_variants','updates','learning_rate','adapt_horizon','response_horizons','adapt_fixtures_per_seed','train_cases','base_snapshots','base_fixtures_per_seed')),'R49_PARENT_CONTRACT_DRIFT')
  r48zip=root/cfg['r48_archive']
  need(r48zip.is_file() and filesha(r48zip)==cfg['r48_sha256'],'R48_ARCHIVE')
  need(sha(zin(r48zip,'r48.py'))==cfg['r48_script_sha256'] and sha(zin(r48zip,'FROZEN_PROTOCOL_R48.json'))==cfg['r48_protocol_sha256'],'R48_SOURCE_PROTOCOL')
  r48pcfg=json.loads(zin(r48zip,'FROZEN_PROTOCOL_R48.json'))
  need(all(cfg[k]==r48pcfg[k] for k in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds','evaluations','arms','objective_variants','updates','learning_rate','adapt_horizon','response_horizons','adapt_fixtures_per_seed','train_cases','base_snapshots','base_fixtures_per_seed')),'R48_FROZEN_CONTRACT_CHANGED')
  r47zip=root/cfg['r47_archive']
  need(r47zip.is_file() and filesha(r47zip)==cfg['r47_sha256'],'R47_ARCHIVE')
  need(sha(zin(r47zip,'r47.py'))==cfg['r47_script_sha256'] and sha(zin(r47zip,'FROZEN_PROTOCOL_R47.json'))==cfg['r47_protocol_sha256'],'R47_SOURCE_PROTOCOL')
  r46zip=root/cfg['r46_archive']
  need(r46zip.is_file() and filesha(r46zip)==cfg['r46_sha256'],'R46_ARCHIVE')
  need(sha(zin(r46zip,'r46.py'))==cfg['r46_script_sha256'] and sha(zin(r46zip,'FROZEN_PROTOCOL_R46.json'))==cfg['r46_protocol_sha256'],'R46_SOURCE_PROTOCOL')
  r47parent=json.loads(zin(r47zip,'FROZEN_PROTOCOL_R47.json'))
  need(r47parent['r46_sha256']==cfg['r46_sha256'] and r47parent['r46_protocol_sha256']==cfg['r46_protocol_sha256'],'R47_R46_AUTHORITY_MISMATCH')
  r45zip=root/cfg['r45_archive']
  need(r45zip.is_file() and filesha(r45zip)==cfg['r45_sha256'],'R45_ARCHIVE')
  need(sha(zin(r45zip,'r45.py'))==cfg['r45_script_sha256'] and sha(zin(r45zip,'FROZEN_PROTOCOL_R45.json'))==cfg['r45_protocol_sha256'],'R45_SCRIPT_PROTOCOL')
  r44zip=root/cfg['r44_archive']
  need(r44zip.is_file() and filesha(r44zip)==cfg['r44_sha256'],'R44_ARCHIVE')
  need(sha(zin(r44zip,'r44.py'))==cfg['r44_script_sha256'] and sha(zin(r44zip,'FROZEN_PROTOCOL_R44.json'))==cfg['r44_protocol_sha256'],'R44_SOURCE_PROTOCOL')
  par=json.loads(zin(r44zip,'FROZEN_PROTOCOL_R44.json'))
  r43zip=root/par['r43_archive']
  need(r43zip.is_file() and filesha(r43zip)==par['r43_sha256'],'R43_ARCHIVE')
  need(sha(zin(r43zip,'r43.py'))==par['r43_script_sha256'] and sha(zin(r43zip,'FROZEN_PROTOCOL_R43.json'))==par['r43_protocol_sha256'],'R43_SOURCE_PROTOCOL')
  r42zip=root/par['r42_archive']
  need(r42zip.is_file() and filesha(r42zip)==par['r42_sha256'],'R42_ARCHIVE')
  need(sha(zin(r42zip,'r42.py'))==par['r42_script_sha256'] and sha(zin(r42zip,'FROZEN_PROTOCOL_R42.json'))==par['r42_protocol_sha256'],'R42_SOURCE_PROTOCOL')
  r41zip=root/par['r41_archive']
  if not r41zip.is_file():r41zip=root/par['r41_reupload_archive']
  need(r41zip.is_file() and filesha(r41zip)==par['r41_sha256'],'R41_ARCHIVE')
  need(sha(zin(r41zip,'r41.py'))==par['r41_script_sha256'] and sha(zin(r41zip,'FROZEN_PROTOCOL_R41.json'))==par['r41_protocol_sha256'],'R41_SOURCE_PROTOCOL')
  r40zip=root/par['r40_archive']
  need(r40zip.is_file() and filesha(r40zip)==par['r40_sha256'],'R40_ARCHIVE')
  need(sha(zin(r40zip,'r40.py'))==par['r40_script_sha256'] and sha(zin(r40zip,'FROZEN_PROTOCOL_R40.json'))==par['r40_protocol_sha256'],'R40_SOURCE_PROTOCOL')
  r39zip=root/par['r39_archive']
  need(r39zip.is_file() and filesha(r39zip)==par['r39_sha256'],'R39_ARCHIVE')
  need(sha(zin(r39zip,'r39.py'))==par['r39_script_sha256'] and sha(zin(r39zip,'FROZEN_PROTOCOL_R39.json'))==par['r39_protocol_sha256'],'R39_SOURCE_PROTOCOL')
  r38zip=root/par['r38_archive']
  need(r38zip.is_file() and filesha(r38zip)==par['r38_sha256'],'R38_ARCHIVE')
  need(sha(zin(r38zip,'r38.py'))==par['r38_script_sha256'] and sha(zin(r38zip,'FROZEN_PROTOCOL_R38.json'))==par['r38_protocol_sha256'],'R38_SCRIPT_PROTOCOL')
  r38cfg=json.loads(zin(r38zip,'FROZEN_PROTOCOL_R38.json'))
  r37zip=root/r38cfg['r37_archive']
  need(r37zip.is_file() and filesha(r37zip)==r38cfg['r37_sha256'],'R37_ARCHIVE')
  need(sha(zin(r37zip,'r37.py'))==r38cfg['r37_script_sha256'] and sha(zin(r37zip,'FROZEN_PROTOCOL_R37.json'))==r38cfg['r37_protocol_sha256'],'R37_SOURCE_PROTOCOL')
  r37cfg=json.loads(zin(r37zip,'FROZEN_PROTOCOL_R37.json'))
  parent=root/r37cfg['r36_archive']
  need(parent.is_file() and filesha(parent)==r37cfg['r36_sha256'],'R36_ARCHIVE')
  need(sha(zin(parent,'r36.py'))==r37cfg['r36_script_sha256'] and sha(zin(parent,'FROZEN_PROTOCOL_R36.json'))==r37cfg['r36_protocol_sha256'],'R36_SOURCE_PROTOCOL')
  r36cfg=json.loads(zin(parent,'FROZEN_PROTOCOL_R36.json'))
  r31zip=root/r36cfg['r31_archive'];need(r31zip.is_file() and filesha(r31zip)==r36cfg['r31_sha256'],'R31_ARCHIVE')
  need(sha(zin(r31zip,'FROZEN_PROTOCOL_R31.json'))==r36cfg['r31_protocol_sha256'],'R31_PROTOCOL')
  r31cfg=json.loads(zin(r31zip,'FROZEN_PROTOCOL_R31.json'))
  r30zip=root/r31cfg['r30_archive'];fixzip=root/r31cfg['fix_archive']
  for k,p in (('r30',r30zip),('fix',fixzip)):
   need(p.is_file() and filesha(p)==r31cfg[k+'_sha256'],'PARENT:'+k)
  r30cfg=json.loads(zin(r30zip,'FROZEN_PROTOCOL_R30.json'))
  src_r30=zin(r30zip,'r30.py').decode();src_fixed=zin(fixzip,'r30.py').decode()
  base1="    r8=perseed(ARMS[0],reverse,'err');r88=perseed(ARMS[2],reverse,'err')"
  fix1="    r8_errors=perseed(ARMS[0],reverse,'err');r88=perseed(ARMS[2],reverse,'err')"
  base2="'reverse_88_vs_adam8':sum(a<b for a,b in zip(r88,r8)),"
  fix2="'reverse_88_vs_adam8':sum(a<b for a,b in zip(r88,r8_errors)),"
  need(src_r30.count(base1)==src_r30.count(base2)==1 and src_fixed==src_r30.replace(base1,fix1).replace(base2,fix2),'R30_FIX_SCOPE')
  need(sha(zin(fixzip,'FROZEN_PROTOCOL_R30.json'))==sha(zin(r30zip,'FROZEN_PROTOCOL_R30.json'))==r31cfg['r30_protocol_sha256'],'R30_PROTOCOL')
  parents={k:root/r30cfg[k+'_archive'] for k in ('r29','r28','r27','r26')}
  for k,p in parents.items():
   need(p.is_file() and filesha(p)==r30cfg[k+'_sha256'],'PARENT:'+k)
   need(sha(zin(p,'FROZEN_PROTOCOL_'+k.upper()+'.json'))==r30cfg[k+'_protocol_sha256'],'PARENT_PROTOCOL:'+k)
  r26cfg=json.loads(zin(parents['r26'],'FROZEN_PROTOCOL_R26.json'))
  upstream={k:root/n for k,n in r26cfg['parents'].items()}
  for k,p in upstream.items():
   need(p.is_file() and filesha(p)==r26cfg['parent_archive_sha256'][k],'UPSTREAM:'+k)
   need(sha(zin(p,'FROZEN_PROTOCOL_R'+k[1:]+'.json'))==r26cfg['parent_protocol_sha256'][k],'UPSTREAM_PROTOCOL:'+k)
  for k in ('r35','r34','r33','r32'):
   p=root/r36cfg[k+'_archive'];need(p.is_file() and filesha(p)==r36cfg[k+'_sha256'],'PARENT:'+k)
   need(sha(zin(p,'FROZEN_PROTOCOL_'+k.upper()+'.json'))==r36cfg[k+'_protocol_sha256'],'PARENT_PROTOCOL:'+k)
  with tempfile.TemporaryDirectory(prefix='ym_awr_r58_src_',dir='/dev/shm') as t:
   td=Path(t)
   files=[('r31_parent.py',zin(r31zip,'r31.py')),('r30_parent.py',src_fixed.encode()),
          ('r25_parent.py',zin(upstream['r25'],'r25.py')),
          ('r8_parent.py',zin(upstream['r8'],'r8.py')),('r10_parent.py',zin(upstream['r10'],'r10.py'))]
   for n,contents in files:(td/n).write_bytes(contents)
   sys.path.insert(0,str(td))
   import r31_parent as r31,r30_parent as r30,r25_parent as r25,r8_parent as r8,r10_parent as r10
   r8cfg=json.loads(zin(upstream['r8'],'FROZEN_PROTOCOL_R8.json'))
   if local:
    for name,key in r8.LOCAL_KEYS.items():os.environ[key]=str(root/('YM_AWR_R1_G5_SOURCE_HANDOFF.zip' if name==r8.G5_NAME else name))
   sources,g5,_=r8.parent_sources(r8cfg,local)
   r24cfg=json.loads(zin(upstream['r24'],'FROZEN_PROTOCOL_R24.json'))
   hca=zin(sources['YM_AWR_GEOMETRY_R0.zip'],'ym_awr_hca.py')
   need(sha(hca)==r24cfg['hca_source_sha256'],'HCA_SOURCE')
   (td/'ym_awr_hca.py').write_bytes(hca)
   (td/'r3_parent.py').write_bytes(zin(sources['YM_AWR_G5_EQUIVARIANT_GRAPH_BRIDGE_R3.zip'],'probe.py'))
   (td/'r4_parent.py').write_bytes(zin(sources['YM_AWR_G5_SIGNED_CF_OFFLINE_R4.zip'],'train_r4.py'))
   from ym_awr_hca import EquivariantHCA
   import r3_parent as r3,r4_parent as r4
   r4cfg=json.loads(zin(sources['YM_AWR_G5_SIGNED_CF_OFFLINE_R4.zip'],'FROZEN_PROTOCOL_R4.json'))
   basegraphs,basecases=r3.frozen_constants(g5)
   graphs={**basegraphs,**{k:tuple(tuple(x) for x in v) for k,v in r31cfg['new_graphs'].items()}}
   cases={**basecases,**{k:tuple(v) for k,v in r31cfg['new_cases'].items()}}
   teacher=r30.NonlinearTeacher(torch)
   # New R55 degree-preserving topology rewires. Historical R52 training untouched.
   f=cfg['r54_factorial'];s=cfg['r55_structural'];pc=cfg['prospective'];ncfg=prospective_config(cfg)
   fitrecords=[]
   for seed in pc['base_fit_seeds']:
    fitcfg={'training_fixtures':pc['base_fixtures_per_seed'],'teacher_state_snapshots':pc['base_snapshots'],'train_batch':pc['base_train_batch']}
    fitrecords.extend(r30.make_records(seed,fitcfg,r3,r4,r4cfg,torch,basegraphs,basecases,teacher))
   X,Y,C=r31.build_gate_dataset(fitrecords,teacher,r25,r30,torch)
   base,fitinfo=r31.project_gate_oracle(EquivariantHCA,X,Y,torch)
   need(len(X)>0 and float(fitinfo['pre_mse'])>=0,'FIT_DEGENERATE')
   base_hash=r4.model_hash(base)
   adapt=adaptation_records(pc['adapt_train_seeds'],ncfg,r3,r4,r4cfg,torch,basegraphs,basecases,teacher)
   need(len(adapt)==36 and all(len(z[2])==20 and len(z[3])==4 for z in adapt),'ADAPT_TARGET_LENGTH')
   pins='PASS';proto='PASS'
   models,fitstats=train_prospective(base,adapt,cfg,r30,r4,torch,preflight=not full)
   if full:
    for a,ref in f['r52_fitstats_reference'].items():
     for key,wanted in ref.items():
      actual=fitstats[a][key]
      if key=='clipped':need(actual==wanted,'R52_FULL_CLIP_PARITY:'+a)
      else:need(abs(actual-wanted)<=max(1e-7,abs(wanted)*3e-6),'R52_FULL_TRAIN_PARITY:'+a+':'+key)
   rows={};done=[];diag={'cov':0.,'disconnected':0.,'min_ref':1e30,'parity':0.,'prefix':0.,'gauge':0.,'source':0.,'spread':0.,'decomposition':0.,'matched_min':1e30}
   seeds=t62['eval_seeds'] if full else [t62['excluded_preflight_seed']]
   evalcfg={**cfg,'r61_early_steps':t62}  # only R61's domain loop is parameterized; parent cfg remains frozen
   for seed in seeds:
    print('[R62] FACTORIAL seed '+str(seed)+' ('+str(len(done)+1)+'/'+str(len(seeds))+')',flush=True)
    part,checks=r61_step_seed(seed,models,evalcfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=(not full))
    r62_check_seed(part,t62,preflight=(not full))
    for k,v in checks.items():diag[k]=min(diag[k],v) if k in ('min_ref','matched_min') else max(diag[k],v)
    for k,v in part.items():rows.setdefault(k,[]).append(v)
    done.append(seed)
   tol=t62['technical']
   need(diag['cov']<=tol['max_cov'] and diag['disconnected']<=tol['max_disconnected'],'R61_COV_DISCONNECTED')
   need(diag['source']<=1e-12 and diag['gauge']<=tol['max_gauge'] and diag['prefix']<=tol['max_prefix'],'R61_SOURCE_GAUGE_PREFIX')
   need(diag['parity']<=tol['max_parity'] and diag['spread']>=tol['min_spread'] and diag['decomposition']<=tol['max_window_decomp'],'R61_NATIVE_PARITY_OR_WINDOW_DECOMPOSITION')
   summary=r61_aggregate(rows,evalcfg) if full else {}
   contrasts=r62_interactions(rows,t62) if full else {}
   if full:
    need(max(z['mobius_resid_rel'] for z in contrasts.values())<=1e-10,'R62_MOBIUS_IDENTITY')
    need(max(z['shapley_resid_rel'] for z in contrasts.values())<=1e-10,'R62_SHAPLEY_IDENTITY')
   science='R62_RETROSPECTIVE_FULL_16_SUBSET_B_TRANSPORT_DIAGNOSTIC_NO_NEW_UTILITY'
   need(r4.model_hash(base)==base_hash,'BASE_MODEL_MUTATION')
   need(filesha(r61zip)==r62cfg['parent_r61_sha256'],'POSTCHECK:R61')
   need(filesha(r60zip)==cfg['r60_sha256'],'POSTCHECK:R60')
   need(filesha(r59zip)==cfg['r59_sha256'],'POSTCHECK:R59')
   need(filesha(r58zip)==cfg['r58_sha256'],'POSTCHECK:R58')
   need(filesha(r57zip)==cfg['r57_sha256'],'POSTCHECK:R57')
   need(filesha(r56zip)==cfg['r56_sha256'],'POSTCHECK:R56')
   need(filesha(r55zip)==cfg['r55_sha256'],'POSTCHECK:R55')
   need(filesha(r54zip)==cfg['r54_sha256'],'POSTCHECK:R54')
   need(filesha(r53zip)==cfg['r53_sha256'],'POSTCHECK:R53')
   need(filesha(r52zip)==cfg['r52_sha256'],'POSTCHECK:R52')
   need(filesha(r51zip)==cfg['r51_sha256'],'POSTCHECK:R51')
   need(filesha(r50zip)==cfg['r50_sha256'],'POSTCHECK:R50')
   need(filesha(r49zip)==cfg['r49_sha256'],'POSTCHECK:R49')
   need(filesha(r48zip)==cfg['r48_sha256'] and filesha(r47zip)==cfg['r47_sha256'] and filesha(r46zip)==cfg['r46_sha256'],'POSTCHECK:R48_R47_R46')
   for k,p in [('r45',r45zip),('r44',r44zip),('r43',r43zip),('r42',r42zip),('r41',r41zip),('r40',r40zip),('r39',r39zip),('r38',r38zip),('r37',r37zip),('r36',parent),('r31',r31zip),('r30',r30zip),('fix',fixzip)]:
    expected=cfg['r45_sha256'] if k=='r45' else cfg['r44_sha256'] if k=='r44' else par['r43_sha256'] if k=='r43' else par['r42_sha256'] if k=='r42' else par['r41_sha256'] if k=='r41' else par['r40_sha256'] if k=='r40' else par['r39_sha256'] if k=='r39' else par['r38_sha256'] if k=='r38' else r38cfg['r37_sha256'] if k=='r37' else r37cfg['r36_sha256'] if k=='r36' else r36cfg['r31_sha256'] if k=='r31' else r31cfg[k+'_sha256']
    need(filesha(p)==expected,'POSTCHECK:'+k)
   for k in ('r35','r34','r33','r32'):
    need(filesha(root/r36cfg[k+'_archive'])==r36cfg[k+'_sha256'],'POSTCHECK:'+k)
   for k,p in parents.items():need(filesha(p)==r30cfg[k+'_sha256'],'POSTCHECK:'+k)
   for k,p in upstream.items():need(filesha(p)==r26cfg['parent_archive_sha256'][k],'POSTCHECK:'+k)
   r8.parent_sources(r8cfg,local)
   postcheck='PASS';status='PASS' if full else 'PREFLIGHT_PASS';rc=0
   if full:
    reportpath,reporthash=r62_report(rows,summary,contrasts,r62cfg,psha,root)
   else:
    reportpath='NONE';reporthash='NONE'
 except (Hold,KeyError,FileNotFoundError,OSError,ValueError,RuntimeError,ImportError,IndexError,zipfile.BadZipFile) as e:
  blocker=type(e).__name__+':'+str(e)[:200]
 except KeyboardInterrupt:
  blocker='INTERRUPTED';rc=130
 except Exception as e:
  blocker='UNEXPECTED_'+type(e).__name__+':'+str(e)[:200];traceback.print_exc(limit=8)
 finally:
  print('\n=== SEND BACK ===',flush=True)
  print('PHASE=YM_AWR_R61_FULL_EARLY_B_FACTORIAL_R62',flush=True)
  print('MODE='+mode,flush=True)
  print('RESULT='+status+' PARENT_PINS='+pins+' PROTOCOL='+proto+' PARENT_POSTCHECK='+postcheck,flush=True)
  print('R62_PROTOCOL_SHA256='+psha,flush=True)
  print('R61_PARENT_SHA256='+r62cfg.get('parent_r61_sha256','NOT_READ') if 'r62cfg' in locals() else 'R61_PARENT_SHA256=NOT_READ',flush=True)
  print('EVAL_SEEDS_COMPLETED='+str(len(done))+'/'+('8' if full else '1'),flush=True)
  print('R52_TRAIN_REPLAY='+('PASS' if fitstats and status=='PASS' else 'TWO_UPDATE_PREFLIGHT' if fitstats and status=='PREFLIGHT_PASS' else 'NOT_COMPLETE'),flush=True)
  if diag:
   print('MAX_COV='+fmt(diag['cov'])+' DISCONNECTED='+fmt(diag['disconnected'])+' DIRECT_PARITY='+fmt(diag['parity'])+' PREFIX_PARITY='+fmt(diag['prefix'])+' SOURCE_PARITY='+fmt(diag['source'])+' GAUGE_PARITY='+fmt(diag['gauge'])+' DECOMP_PARITY='+fmt(diag['decomposition'])+' NONFLAT_SPREAD='+fmt(diag['spread'])+' MIN_BASE_REF='+fmt(diag['min_ref'])+' MIN_MODE_REF='+fmt(diag['matched_min']),flush=True)
  print('FULL_SUBSET_COVERAGE='+('16/16' if status in ('PASS','PREFLIGHT_PASS') else 'NOT_COMPLETE'),flush=True)
  if contrasts:
   print('MOBIUS_IDENTITY_REL_RESID='+fmt(max(x['mobius_resid_rel'] for x in contrasts.values()))+' SHAPLEY_EFFICIENCY_REL_RESID='+fmt(max(x['shapley_resid_rel'] for x in contrasts.values())),flush=True)
   for c in t62['cases']:
    for a in t62['arms']:
     q=contrasts[(c,a)]
     print('ATTR_'+c+'_'+a+'=H1_EMPTY:'+fmt(q['first_order'][1])+' H1_SHAPLEY:'+fmt(q['shapley'][1])+' H2_SHAPLEY:'+fmt(q['shapley'][2])+' H3_SHAPLEY:'+fmt(q['shapley'][3])+' H4_SHAPLEY:'+fmt(q['shapley'][4])+' I4:'+fmt(q['interactions'][15])+' H1_CONDITIONAL_RANGE:'+fmt(q['h1_conditional_range'][0])+','+fmt(q['h1_conditional_range'][1])+' H1_SHAPLEY_CI95:'+fmt(q['shapley_ci95_h1'][0])+','+fmt(q['shapley_ci95_h1'][1])+' I4_CI95:'+fmt(q['i4_ci95'][0])+','+fmt(q['i4_ci95'][1]),flush=True)
  print('SCIENTIFIC_DISPOSITION='+science,flush=True)
  print('RESULT_REPORT='+str(locals().get('reportpath','NONE'))+' REPORT_SHA256='+str(locals().get('reporthash','NONE')),flush=True)
  print('LIMITS=FROZEN_R52_WEIGHTS;SOURCE_PINNED_R61;H20_SYNTHETIC_COUNTERFACTUAL_ONLY;NO_LEARNED_UTILITY_OR_NATIVE_GDN;NO_INFERENCE_NAND_WRITES',flush=True)
  print('BLOCKER='+blocker,flush=True)
  print('R62_CHILD_EXIT='+str(rc),flush=True)
  print('=== END SEND BACK ===',flush=True)
 return rc

if __name__=='__main__':sys.exit(r62_selftest() if len(sys.argv)>1 and sys.argv[1]=='selftest' else run())
