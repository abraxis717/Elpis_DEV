#!/usr/bin/env python3
"""YM R78: retrospective SO(2)xSO(2)-equivariant finite-horizon tangent growth.
Teacher/R77 metric labels are NOT consulted to create perturbations or trajectories.
No training, no runtime weight mutation, and no inference-time teacher state.
"""
from __future__ import annotations
import hashlib,io,json,math,pathlib,statistics,sys,zipfile
import torch as T
import portable_mechanics as pm,r30_source as r30,r62_source as r62,r63_source as r63,r64,r77
ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'FROZEN_PROTOCOL_R78.json').read_bytes());T.set_num_threads(1)
class Hold(Exception):pass
def need(c,m):
 if not c:raise Hold(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load_r77():
 with zipfile.ZipFile(ROOT/'YM_AWR_R77_RESULT.zip') as z:
  need(z.namelist()==['result.json'] and z.testzip() is None,'R77_ZIP_FAIL')
  raw=z.read('result.json');o=json.loads(raw)
  need(canonical(o)==raw,'R77_NOT_CANONICAL')
  return o
def source_gate():
 for n,d in CFG['source_sha256'].items():need((ROOT/n).is_file() and sha((ROOT/n).read_bytes())==d,'SOURCE_PIN:'+n)
 need(CFG['parent_r77_report_sha256']==sha((ROOT/'YM_AWR_R77_RESULT.zip').read_bytes()),'PARENT_REPORT_PIN')
 need(set(CFG['graphs'])==set(r77.CFG['graphs']) and CFG['arms']==r77.CFG['arms'],'GRAPH_ARM_DRIFT')
 need(CFG['evaluation_seeds']==r77.CFG['evaluation_seeds'][:4] and CFG['cases']==r77.CFG['evaluation_cases'],'EVAL_SEED_DRIFT')
 need(CFG['epsilons']==[1e-5,5e-6] and CFG['time_windows']==[[8,12],[8,20],[12,20]],'TANGENT_PROTOCOL_DRIFT')
 need(CFG['records']==4*6*4*3 and CFG['ratio']==160,'TARGET_COUNT')
 report=load_r77();need(report['phase']==r77.CFG['phase'],'PARENT_PHASE')
 return report

def models_from_r77(o):
 out={}
 for k,pin in o['training'].items():
  m=r64.HCA()
  with T.no_grad():
   m.gates.weight.copy_(T.tensor(pin['weights'],dtype=T.float64))
   m.gates.bias.copy_(T.tensor(pin['bias'],dtype=T.float64))
  m.requires_grad_(False)
  need(sum(q.numel() for q in m.parameters())==88 and r64.parameter_pin(m)==pin['checkpoint_sha256'],'MODEL_PIN:'+k)
  out[k]=m
 need(len(out)==6,'MODEL_COUNT')
 return out

def J(x):
 y=T.empty_like(x)
 y[...,0]=-x[...,1];y[...,1]=x[...,0]
 y[...,2]=-x[...,3];y[...,3]=x[...,2]
 return y

def inputs():
 ss=[r64.pack_seeds(tuple(CFG['evaluation_seeds']),case,160,4)[0] for case in CFG['cases']]
 K=len(CFG['evaluation_seeds'])*CFG['batch_per_seed'];need(K==16,'SEED_PER_CASE')
 return tuple(T.cat((T.cat([s[ch][:K] for s in ss]),T.cat([s[ch][K:] for s in ss]))) for ch in (0,1))

def rebuild_variants(s,weights,bias,per):
 # Baseline packed as [model, sample, relation, R4]. Eight independent "+/-"
 # tangents per model: A/B x two epsilon levels x positive/negative.
 n=weights.shape[0];need(n==6,'GROUP_SIZE');sh=[x.reshape(n,per,4,4) for x in s]
 variants=[(channel,eps,sign) for channel in (0,1) for eps in CFG['epsilons'] for sign in (1,-1)]
 states=[]
 for ch in (0,1):
  values=[]
  for i in range(n):
   for source,eps,sign in variants:
    v=sh[ch][i].clone()
    if source==ch:v.add_(J(sh[ch][i]),alpha=eps*sign)
    values.append(v)
  states.append(T.stack(values,dim=0).reshape(n*len(variants)*per,4,4))
 w=weights.repeat_interleave(len(variants),dim=0);b=bias.repeat_interleave(len(variants),dim=0)
 return tuple(states),r62._StackedGates(w,b,r30,T,per),variants

def scalar_for_samples(t, indices):
 # t: [per,4,4]; scalar squared norm over selected paired samples and nodes.
 return float(t[indices].square().sum())

def tangent_rows(native,perturbed,models,graph,start,ends):
 # Native at H8 or H12; perturbed list contains outputs at endpoint horizons.
 per=native[0].shape[0]//6;keys=sorted(models);V=8;N=len(CFG['evaluation_seeds']);b=CFG['batch_per_seed'];half=N*b*len(CFG['cases'])
 seen=[]
 for mi,k in enumerate(keys):
  for ci,case in enumerate(CFG['cases']):
   for si,seed in enumerate(CFG['evaluation_seeds']):
    lo=ci*N*b+si*b
    indices=list(range(lo,lo+b))+list(range(half+lo,half+lo+b))
    vals={}
    for channel,ch in [('A',0),('B',1)]:
     base=J(native[ch][mi*per:(mi+1)*per]);init_sq=scalar_for_samples(base,indices)
     need(init_sq>1e-25 and math.isfinite(init_sq),'TANGENT_START_ZERO')
     init=math.sqrt(init_sq)
     for ei,eps in enumerate(CFG['epsilons']):
      variant_idx=ch*4+ei*2
      for end,state in perturbed.items():
       dirs=[]
       for cc in (0,1):
        x=state[cc].reshape(6,8,per,4,4)
        directions=(x[mi,variant_idx]-x[mi,variant_idx+1])/(2.*eps)
        dirs.append(directions)
       endnorm=math.sqrt(sum(scalar_for_samples(d,indices) for d in dirs))
       need(endnorm>0 and math.isfinite(endnorm),'END_NORM_INVALID')
       vals[f'{channel}/{eps:.0e}/{start}_{end}']={'input_norm':init,'derivative_norm':endnorm,'log_gain':math.log(endnorm/init)}
    seen.append((k,graph,case,seed,vals))
 return seen

def get_data_for_graph(name,models,refrows):
 keys=sorted(models);per=2*len(CFG['evaluation_seeds'])*CFG['batch_per_seed']*len(CFG['cases'])
 W=T.stack([models[k].gates.weight.detach() for k in keys]);B=T.stack([models[k].gates.bias.detach() for k in keys]);law=r62._StackedGates(W,B,r30,T,per)
 s0=inputs();s=tuple(x.repeat(len(keys),1,1) for x in s0)
 r61=pm.read_json('FROZEN_PROTOCOL_R61.json');adj=pm.graph_adjs(CFG['graphs_definition'][name],r61)[1] if 'graphs_definition' in CFG else pm.graph_adjs(r77.CFG['graphs'][name],r61)[1]
 with T.no_grad():
  saved={}
  for h in range(1,21):
   s=r62.graph_step_split(s,adj,adj,law,T)
   if h in (8,12,20):saved[h]=tuple(v.clone() for v in s)
  # Pin the actual unperturbed native response to sealed R77 tensor outputs.
  parent_max=0.
  for mi,k in enumerate(keys):
   init,arm=k.split('/');ix=slice(mi*per,(mi+1)*per)
   for end in (8,12,20):
    actual=r63.paired(saved[end][0][ix],per//2)
    for ci,case in enumerate(CFG['cases']):
     for si,seed in enumerate(CFG['evaluation_seeds']):
      sl=slice(ci*len(CFG['evaluation_seeds'])*4+si*4,ci*len(CFG['evaluation_seeds'])*4+(si+1)*4)
      p=refrows[(int(init),arm,name,seed,case)]['h'][str(end)]['response4']
      err=float((actual[sl]-T.tensor(p,dtype=T.float64)).abs().max());parent_max=max(parent_max,err)
  need(parent_max<CFG['native_r77_parity_atol'],'R77_NATIVE_RESPONSE_PARITY')
  # Two-sided finite differences, per arm and independently for H8/H12.
  allparts={};tangent_max=0.
  for start,ends in [(8,(12,20)),(12,(20,))]:
   states,variantlaw,variants=rebuild_variants(saved[start],W,B,per)
   for h in range(start+1,21):
    states=r62.graph_step_split(states,adj,adj,variantlaw,T)
    if h in ends:allparts[(start,h)]=tuple(v.clone() for v in states)
  # SO(2)xSO(2) gauge-orbit negative control: sum of independent A/B
  # directional derivatives must match J applied to the native output.
  # A global rotation acts on both state channels, not one alone.
  gauge_max=0.
  eps=CFG['epsilons'][0]
  for (start,end),packed in allparts.items():
   for cc in (0,1):
    x=packed[cc].reshape(6,8,per,4,4)
    summed=(x[:,0]-x[:,1]+x[:,4]-x[:,5])/(2*eps)
    expected=J(saved[end][cc].reshape(6,per,4,4))
    gauge_max=max(gauge_max,float((summed-expected).abs().max()))
  need(gauge_max<=CFG['simultaneous_A_B_gauge_tangent_identity_max_abs'],'GAUGE_GENERATOR_IDENTITY')
  groups={}
  for start,ends in [(8,(12,20)),(12,(20,))]:
   subsets=tangent_rows(saved[start],{end:allparts[(start,end)] for end in ends},models,name,start,ends)
   for k,g,case,seed,vals in subsets:groups.setdefault((k,g,case,seed),{}).update(vals)
  output=[]
  for k,g,case,seed in sorted(groups):
   v=groups[(k,g,case,seed)];deviations=[]
   for ch in CFG['channels']:
    for start,end in CFG['time_windows']:
     x=v[f'{ch}/{CFG["epsilons"][0]:.0e}/{start}_{end}']['log_gain'];y=v[f'{ch}/{CFG["epsilons"][1]:.0e}/{start}_{end}']['log_gain']
     deviations.append(abs(x-y))
   maxdelta=max(deviations);tangent_max=max(tangent_max,maxdelta)
   need(maxdelta<=CFG['numerical_agreement_log_absolute_tolerance'],'FINITE_DIFFERENCE_MISMATCH')
   init,arm=k.split('/');p=refrows[(int(init),arm,g,seed,case)]
   labels={str(h):p['h'][str(h)]['mean_nse'] for h in (8,12,20)}
   output.append({'initialization':int(init),'arm':arm,'graph':g,'case':case,'eval_seed':seed,'nse':labels,'tf_local_nse':p['tf_local_h20']['mean_nse'],'tangents':v,'max_eps_log_delta':maxdelta})
 return output,parent_max,tangent_max,gauge_max

def rank(x):
 # Average rank for ties.
 vals=sorted((v,i) for i,v in enumerate(x));out=[0.]*len(x);j=0
 while j<len(x):
  k=j+1
  while k<len(x) and vals[k][0]==vals[j][0]:k+=1
  for a in range(j,k):out[vals[a][1]]=(j+k-1)/2
  j=k
 return out

def corr(a,b):
 x=rank(a);y=rank(b);xm=statistics.mean(x);ym=statistics.mean(y)
 s=sum((v-xm)*(w-ym) for v,w in zip(x,y));norm=math.sqrt(sum((v-xm)**2 for v in x)*sum((w-ym)**2 for w in y))
 return s/norm if norm else 0.

def summarize(rows):
 metrics={}
 for g in CFG['graphs']:
  subset=[r for r in rows if r['graph']==g]
  for ch in CFG['channels']:
   for period in ['8_12','8_20','12_20']:
    gain=[r['tangents'][f'{ch}/1e-05/{period}']['log_gain'] for r in subset]
    increase=[math.log((r['nse']['20']+1e-12)/(r['nse']['12']+1e-12)) for r in subset]
    metrics[g+'/'+ch+'/'+period]={'spearman_vs_log_h20_h12_error':corr(gain,increase),'mean_log_gain':statistics.mean(gain),'median_log_gain':statistics.median(gain),'n':len(subset)}
 key='8_20';selected={g:max(metrics[g+'/'+ch+'/'+key]['spearman_vs_log_h20_h12_error'] for ch in CFG['channels']) for g in CFG['graphs']}
 passes=sum(v>CFG['rank_correlation_threshold'] for v in selected.values())
 verdict='RETROSPECTIVE_TANGENT_ASSOCIATION_BOUNDED' if passes>=CFG['positive_graph_requirement'] else 'RETROSPECTIVE_TANGENT_ASSOCIATION_NONQUAL'
 return {'metrics':metrics,'best_channel_per_graph_spearman':selected,'graphs_above_predeclared_correlation':passes,'scientific_gate':verdict,'disposition':'RETROSPECTIVE_FROZEN_HCA_FINITE_HORIZON_TANGENT_DIAGNOSTIC_NO_AUTONOMOUS_CORRECTION'}

def save(obj,dest):
 raw=canonical(obj);info=zipfile.ZipInfo('result.json',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.writestr(info,raw)
 return sha(dest.read_bytes())

def run():
 parent=source_gate();models=models_from_r77(parent);refs={(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']):r for r in parent['rows']}
 output=[];mx=0.;dx=0.;gx=0.
 for g in CFG['graphs']:
  part,a,b,c=get_data_for_graph(g,models,refs);output.extend(part);mx=max(mx,a);dx=max(dx,b);gx=max(gx,c)
  print(f'[R78] graph={g} records={len(output)}/{CFG["records"]} native_r77_delta={a:.3g} eps_log_delta={b:.3g} gauge_identity={c:.3g}',flush=True)
 need(len(output)==CFG['records'],'RECORD_COUNT')
 out={'phase':CFG['phase'],'protocol_sha256':sha((ROOT/'FROZEN_PROTOCOL_R78.json').read_bytes()),'parent_r77_report_sha256':CFG['parent_r77_report_sha256'],'rows':output,'numerics':{'max_r77_native_parity':mx,'max_epsilon_log_disagreement':dx,'max_gauge_generator_identity_error':gx},'summary':summarize(output),'limitations':CFG['limitations']}
 h=save(out,ROOT/'YM_AWR_R78_RESULT.zip')
 print('SCIENTIFIC_GATE='+out['summary']['scientific_gate']);print('GRAPHS_CORR='+json.dumps(out['summary']['best_channel_per_graph_spearman'],sort_keys=True))
 print('R78_RESULT_SHA256='+h,flush=True)
 return out
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='run':run()
 elif len(sys.argv)==2 and sys.argv[1]=='preflight':source_gate();print('R78_SOURCE_GATE=PASS')
 else:raise SystemExit('BAD_MODE')
