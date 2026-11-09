#!/usr/bin/env python3
"""R63 source-pinned H1 gate-versus-collective-injection transport mediation probe.
This is a RETROSPECTIVE diagnostic on frozen R52 weights and an artificial transport seam.
Source authority comes exclusively from byte-pinned R62 and its transitive R61/R0 checks.
"""
from __future__ import annotations
import hashlib, io, importlib, json, math, os, statistics, sys, traceback, zipfile
from pathlib import Path

ROOT = Path('/mnt/primesauce/Downloads')
PARENT_NAME = 'YM_AWR_R61_FULL_EARLY_B_FACTORIAL_R62.zip'
PARENT_HASH = '472c7076b67d0d7161273966b9f44a6102a894e7c3e308be94d7a14738309dea'
PARENT_CODE_HASH = 'fe272da88e3833ad5deaad305dc5ab4d07b666af1fe357a07da7df10e1a799ee'
PARENT_PROTOCOL_HASH = '39ae807dd4a40039912240933c2d5004b002acd3f8458c96636cedb42ef1a886'
MODES=('BASE','INJECT_ONLY','GATE_ONLY','FULL')
class Hold(Exception): pass
def need(ok,why):
 if not ok:raise Hold(why)
def digest(buf):return hashlib.sha256(buf).hexdigest()
def filesha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def fmt(x):return format(float(x),'.9g')
def vnorm(x,t):return t.linalg.vector_norm(x,dim=-1)
def dot(a,b):return (a*b).sum(-1)
def mean(x):return statistics.mean(x)

def verify_inputs(f,root):
 need(f['schema']=='elpis.ym.awr.r63_h1_mediation.v1','R63_SCHEMA')
 need(f['status']=='FROZEN_BEFORE_EVALUATION','R63_FROZEN_STATUS')
 need(f['r62_archive']==PARENT_NAME and f['r62_sha256']==PARENT_HASH,'R63_PARENT_HASH')
 need(f['r62_script_sha256']==PARENT_CODE_HASH and f['r62_protocol_sha256']==PARENT_PROTOCOL_HASH,'R63_PARENT_SOURCE')
 need(f['modes']==list(MODES) and f['gate_injection_bits']=={'BASE':[0,0],'INJECT_ONLY':[0,1],'GATE_ONLY':[1,0],'FULL':[1,1]},'R63_MODE_MATRIX')
 need(f['cases']==['weak','reverse','anti'] and f['arms']==['SAL_STATE8','SAL_PCGRAD4'],'R63_CASE_ARMS')
 need(f['graphs']==['D1_P1','D1_P2','D2_P1','D2_P2'],'R63_GRAPHS')
 need(f['seeds']==[6503,6511,6521,6529,6547,6553,6563,6571] and f['preflight_seed']==6599,'R63_SEEDS')
 need(len(set(f['seeds']+[f['preflight_seed']]))==9,'R63_DUPLICATE_SEEDS')
 need(f['offset']==82710000000 and f['eval_batch']==8 and f['horizon']==20 and f['source_ratio']==64,'R63_EVAL_PARAMETERS')
 need(f['teacher']=='FIXED_BASE_AB_TEACHER' and f['no_retraining'] and f['no_weight_mutation'],'R63_TEACHER_TRAINING_DRIFT')
 need(f['report_max_bytes']==8000000,'R63_REPORT_BOUND')
 p=root/PARENT_NAME
 need(p.is_file() and filesha(p)==PARENT_HASH,'R62_PARENT_ZIP_HASH')
 with zipfile.ZipFile(p) as z:
  need(z.testzip() is None,'R62_PARENT_ZIP_CRC')
  need(digest(z.read('r62.py'))==PARENT_CODE_HASH,'R62_PARENT_SCRIPT_HASH')
  need(digest(z.read('FROZEN_PROTOCOL_R62.json'))==PARENT_PROTOCOL_HASH,'R62_PARENT_PROTOCOL_HASH')
  pc=json.loads(z.read('FROZEN_PROTOCOL_R62.json'))
  prior=pc['r62_full_factorial']
  need(f['cases']==prior['cases'] and f['graphs']==prior['graphs'] and f['arms']==prior['arms'],'R63_DESIGN_DRIFT')
  need(not(set(f['seeds']+[f['preflight_seed']])&set(prior['eval_seeds']+[prior['excluded_preflight_seed']])),'R63_R62_SEED_OVERLAP')
  need(f['offset']!=prior['eval_offset'],'R63_OFFSET_REUSED')
 return pc

def inputs(state,adj_a,adj_b,t):
 """Exact R57/R62 graph_step_split aggregation; no detached/hidden state."""
 a,b=state;sa=[];cb=[]
 for d in range(4):
  if d not in adj_a:sa.append(t.zeros_like(a[:,d,:]))
  else:
   idx,ops=adj_a[d]
   w=t.einsum('eij,bej->bei',ops,a[:,idx,:])
   sa.append((t.softmax(8.*vnorm(w,t),dim=1)[...,None]*w).sum(1))
  if d not in adj_b:cb.append(t.zeros_like(b[:,d,:]))
  else:
   idx,ops=adj_b[d]
   w=t.einsum('eij,bej->bei',ops,b[:,idx,:])
   cb.append(w.mean(1))
 return t.stack(sa,dim=1),t.stack(cb,dim=1)

def coeff(model,state,sal,col,r30,t):
 z=model.gates(r30.invariants(state,sal,col,t))
 return t.cat((t.sigmoid(z[...,0:1]),t.tanh(z[...,1:4]),t.sigmoid(z[...,4:5]),t.tanh(z[...,5:8])),dim=-1)

def hybrid_step(model,state,sal,col_gate,col_inject,r30,t):
 a,b=state;k=coeff(model,state,sal,col_gate,r30,t)
 ua=k[...,0,None]*a+k[...,1,None]*sal+k[...,2,None]*col_inject+k[...,3,None]*b
 ub=k[...,4,None]*b+k[...,5,None]*sal+k[...,6,None]*col_inject+k[...,7,None]*a
 radial=lambda v:v*t.clamp(4.0/(vnorm(v,t)[...,None]+1e-30),max=1.)
 return (radial(ua),radial(ub)),k,(float((vnorm(ua,t)>4).double().mean()),float((vnorm(ub,t)>4).double().mean()))

def paired(v,b):return v[:b,3,:]-v[b:,3,:]

def run_seed(seed,models,cfg,r3,r4,r30,t,teacher,basegraphs,basecases,f,offset,preflight=False):
 s=cfg['r55_structural'];graphs={**basegraphs,**{k:tuple(tuple(x) for x in g) for k,g in s['graphs'].items()}}
 cases={**basecases,**{k:tuple(v) for k,v in s['cases'].items()}}
 b=f['eval_batch'];rows=[];parity=0.;cov=0.;disc=0.;minref=1e30
 hashes={a:r4.model_hash(models[a]) for a in f['arms']}
 with t.no_grad():
  for ci,case in enumerate(f['cases']):
   for graph in f['graphs']:
    mapped,mats,init,flip=r3.generate(graph,case,offset+seed*100109+ci*59,b,graphs,cases,t)
    st=(t.cat((init[0],flip[0]),0),t.cat((init[1],flip[1]),0))
    eb=r4.build_links(mapped,original.transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,t),t)
    en=r4.build_links(mapped,original.transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,t),t)
    sal,cb0=inputs(st,eb,eb,t); sal1,cb1=inputs(st,eb,en,t)
    parity=max(parity,float((sal-sal1).abs().max()))
    ref=r4.rollout(st,eb,teacher,20,t)[-1]
    target=paired(ref[0],b);den=dot(target,target);valid=den>1e-25
    need(bool(valid.all()),'R63_ZERO_FIXED_REFERENCE:'+graph+':'+case)
    minref=min(minref,float(vnorm(target,t).min()))
    agg_shift=float((cb1-cb0).square().mean().sqrt())
    agg_target=float((cb1[:,3,:]-cb0[:,3,:]).square().mean().sqrt())
    source_b_base=float(vnorm(paired(cb0,b),t).mean())
    source_b_change=float(vnorm(paired(cb1-cb0,b),t).mean())
    for arm in f['arms']:
     m=models[arm]
     k0=coeff(m,st,sal,cb0,r30,t);k1=coeff(m,st,sal,cb1,r30,t)
     h1_states={};clip={};results={}
     for mode,(gate,inject) in f['gate_injection_bits'].items():
      h1_states[mode],_,clip[mode]=hybrid_step(m,st,sal,cb1 if gate else cb0,cb1 if inject else cb0,r30,t)
      cur=h1_states[mode]
      for _ in range(19):cur=original.graph_step_split(cur,eb,eb,m,t)
      pr=paired(cur[0],b);er=dot(pr-target,pr-target)
      ratio=er/den; gain=dot(pr,target)/den
      cos=dot(pr,target)/(vnorm(pr,t)*vnorm(target,t)+1e-30)
      need(bool(t.isfinite(ratio).all() and t.isfinite(gain).all() and t.isfinite(cos).all()),'R63_NONFINITE_RESPONSE')
      results[mode]={'FIXED_E2':float(ratio.mean()),'ABS_ERR':float(er.mean()),
        'ENERGY_E2':float(er.sum()/den.sum()),'GAIN':float(gain.mean()),'COS':float(cos.mean()),
        'PRED_ENERGY':float(dot(pr,pr).mean()),'WORST_FIXED':float(ratio.max()),
        'H20_RESPONSE':pr.tolist(),'SAMPLE_FIXED_E2':ratio.tolist(),
        'CLIP_A_H1':clip[mode][0],'CLIP_B_H1':clip[mode][1]}
     # H1 surgery equivalence against exact R57 native transport, both base and full.
     natives={'BASE':original.graph_step_split(st,eb,eb,m,t),'FULL':original.graph_step_split(st,eb,en,m,t)}
     for mode,refstep in natives.items():
      parity=max(parity,*(float((h1_states[mode][z]-refstep[z]).abs().max()) for z in (0,1)))
     if preflight and graph=='D1_P2' and case=='weak':
      gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,t) for d in range(4)}
      gst=r4.gauge_transform(st,gauges,t)
      geb=r4.links_gauge(eb,gauges,t);gen=r4.links_gauge(en,gauges,t)
      gs,gcb0=inputs(gst,geb,geb,t);_,gcb1=inputs(gst,geb,gen,t)
      for mode,(gate,inject) in f['gate_injection_bits'].items():
       gx,_,_=hybrid_step(m,gst,gs,gcb1 if gate else gcb0,gcb1 if inject else gcb0,r30,t)
       expected=r4.gauge_transform(h1_states[mode],gauges,t)
       cov=max(cov,*(float((gx[z]-expected[z]).abs().max()) for z in (0,1)))
     two_d={mode:results[mode]['FIXED_E2'] for mode in MODES}
     # Exact two-variable inclusion-exclusion and symmetric attribution.
     interaction=two_d['FULL']-two_d['GATE_ONLY']-two_d['INJECT_ONLY']+two_d['BASE']
     phi_gate=.5*(two_d['GATE_ONLY']-two_d['BASE']+two_d['FULL']-two_d['INJECT_ONLY'])
     phi_inject=.5*(two_d['INJECT_ONLY']-two_d['BASE']+two_d['FULL']-two_d['GATE_ONLY'])
     closure=abs(phi_gate+phi_inject-(two_d['FULL']-two_d['BASE']))
     need(closure<=1e-9*max(1.,*(abs(v) for v in two_d.values())),'R63_TWO_FACTOR_IDENTITY')
     h1int=tuple(h1_states['FULL'][z]-h1_states['GATE_ONLY'][z]-h1_states['INJECT_ONLY'][z]+h1_states['BASE'][z] for z in (0,1))
     metrics={'AGG_SHIFT':agg_shift,'AGG_TARGET_SHIFT':agg_target,
       'SOURCE_COL_BASE':source_b_base,'SOURCE_COL_SHIFT':source_b_change,
       'GATE_SHIFT_RMS':float((k1-k0).square().mean().sqrt()),
       'GATE_SHIFT_TARGET':float((k1[:,3,:]-k0[:,3,:]).square().mean().sqrt()),
       'H1_TARGET_STATE_A_DELTA':float((h1_states['FULL'][0][:,3,:]-h1_states['BASE'][0][:,3,:]).square().mean().sqrt()),
       'H1_TARGET_STATE_B_DELTA':float((h1_states['FULL'][1][:,3,:]-h1_states['BASE'][1][:,3,:]).square().mean().sqrt()),
       'H1_STATE_INTERACTION':float((h1int[0].square().mean()+h1int[1].square().mean()).sqrt()),
       'H1_GATE_EFFECT':phi_gate,'H1_INJECT_EFFECT':phi_inject,'H1_2WAY_INTERACTION':interaction,
       'H1_IDENTITY_RESID':closure,'TEACHER_REF_ENERGY':float(den.mean()),'MIN_REF':float(vnorm(target,t).min())}
     for name,ix in [('RET',(0,4)),('SAL',(1,5)),('COL',(2,6)),('CROSS',(3,7))]:
      metrics['GATE_SHIFT_'+name]=float((k1[...,list(ix)]-k0[...,list(ix)]).square().mean().sqrt())
     rows.append({'seed':seed,'graph':graph,'case':case,'arm':arm,'metrics':metrics,
       'REFERENCE4':target.tolist(),
       'H1_COLLECTIVE_BASE_TARGET':cb0[:,3,:].tolist(),
       'H1_COLLECTIVE_NONFLAT_TARGET':cb1[:,3,:].tolist(),
       'H1_GATE_BASE_TARGET':k0[:,3,:].tolist(),
       'H1_GATE_NONFLAT_TARGET':k1[:,3,:].tolist(),
       'modes':results})
  if preflight:
   ev=cfg['prospective']['evaluations'];dg=ev['disconnected_graph'];graphs[dg]=tuple(tuple(v) for v in ev['graphs'][dg])
   mapped,mats,init,flip=r3.generate(dg,'weak',offset+seed*100109+8891,b,graphs,cases,t)
   st=(t.cat((init[0],flip[0]),0),t.cat((init[1],flip[1]),0))
   eb=r4.build_links(mapped,original.transport_matrices(mapped,mats,'BASE_SEPARABLE',cfg,r3,t),t)
   en=r4.build_links(mapped,original.transport_matrices(mapped,mats,'PARTICIPANT_NONFLAT',cfg,r3,t),t)
   sal,c0=inputs(st,eb,eb,t);_,c1=inputs(st,eb,en,t)
   for arm in f['arms']:
    for mode,(gate,inject) in f['gate_injection_bits'].items():
     cur,_,_=hybrid_step(models[arm],st,sal,c1 if gate else c0,c1 if inject else c0,r30,t)
     for _ in range(19):cur=original.graph_step_split(cur,eb,eb,models[arm],t)
     disc=max(disc,float(cur[0][:,3,:].abs().max()),float(vnorm(paired(cur[0],b),t).max()))
 for a in f['arms']:need(hashes[a]==r4.model_hash(models[a]),'R63_MODEL_MUTATION:'+a)
 return rows,{'parity':parity,'gauge':cov,'disconnected':disc,'min_ref':minref}

def aggregate(rows,f):
 group={};allgroup={}
 for x in rows:
  k=(x['case'],x['arm'],x['graph']);group.setdefault(k,[]).append(x)
 need(len(group)==len(f['cases'])*len(f['arms'])*len(f['graphs']),'R63_GRAPH_COVERAGE')
 result=[]
 for (case,arm,graph),cell in sorted(group.items()):
  need(len(cell)==len(f['seeds']),'R63_GRAPH_SEED_COUNT')
  fmean={m:mean(r['modes'][m]['FIXED_E2'] for r in cell) for m in MODES}
  enmean={m:mean(r['modes'][m]['ENERGY_E2'] for r in cell) for m in MODES}
  absmean={m:mean(r['modes'][m]['ABS_ERR'] for r in cell) for m in MODES}
  win=sum(r['modes']['FULL']['FIXED_E2'] < r['modes']['BASE']['FIXED_E2'] for r in cell)
  result.append({'case':case,'arm':arm,'graph':graph,'F':fmean,'ENERGY':enmean,'ABS':absmean,
    'H1_FULL_DELTA':fmean['FULL']-fmean['BASE'],'H1_GATE_SHAPLEY':mean(r['metrics']['H1_GATE_EFFECT'] for r in cell),
    'H1_INJECT_SHAPLEY':mean(r['metrics']['H1_INJECT_EFFECT'] for r in cell),
    'H1_TWO_WAY':mean(r['metrics']['H1_2WAY_INTERACTION'] for r in cell),
    'GATE_SHIFT':mean(r['metrics']['GATE_SHIFT_RMS'] for r in cell),
    'AGG_SHIFT':mean(r['metrics']['AGG_SHIFT'] for r in cell),'H1_FAVORABLE_SEEDS':win,
    'H1_GAINS':{m:mean(r['modes'][m]['GAIN'] for r in cell) for m in MODES},
    'H1_COS':{m:mean(r['modes'][m]['COS'] for r in cell) for m in MODES}})
 return result

def write_report(rows,agg,f,proto_hash):
 data={'phase':'YM_AWR_R62_H1_GATE_INJECTION_LOCALIZATION_R63',
   'scientific_disposition':'RETROSPECTIVE_MECHANISM_DIAGNOSTIC_NO_NEW_UTILITY',
   'protocol_sha256':proto_hash,'source_r62_sha256':PARENT_HASH,
   'no_native_gdn_or_learned_utility_claim':True,'seeds':f['seeds'],'rows':rows,'graph_summaries':agg}
 blob=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 need(len(blob)<=f['report_max_bytes'],'R63_REPORT_TOO_LARGE')
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  info=zipfile.ZipInfo('R63_RESULT.json',(2026,11,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16
  z.writestr(info,blob,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 payload=buf.getvalue();h=digest(payload)
 path=ROOT/('YM_AWR_R63_RESULT_'+h[:16]+'.zip')
 if path.exists():need(filesha(path)==h,'R63_REPORT_EXISTING_COLLISION')
 else:
  with path.open('xb') as out:out.write(payload)
 need(filesha(path)==h,'R63_REPORT_POSTWRITE_HASH')
 return path,h

def test_math(t):
 class Tiny:
  def __init__(self):
   self.gates=t.nn.Linear(10,8).to(dtype=t.float64)
   with t.no_grad():self.gates.weight.zero_();self.gates.bias.copy_(t.tensor([.1,.2,.3,.4,.5,.6,.7,.8],dtype=t.float64))
 class Inv:
  @staticmethod
  def invariants(state,sal,col,torch):
   a,b=state;return torch.stack([torch.zeros_like(a[...,0])]*10,dim=-1)
 model=Tiny();a=t.arange(32,dtype=t.float64).reshape(2,4,4)/40.;b=a/2
 state=(a,b);sal=a*.2;c0=b*.3;c1=b*.4
 x0,_,_=hybrid_step(model,state,sal,c0,c0,Inv(),t)
 x1,_,_=hybrid_step(model,state,sal,c1,c1,Inv(),t)
 xg,_,_=hybrid_step(model,state,sal,c1,c0,Inv(),t)
 xi,_,_=hybrid_step(model,state,sal,c0,c1,Inv(),t)
 need(all(bool(t.isfinite(s).all()) for ss in (x0,x1,xg,xi) for s in ss),'R63_SELFTEST_NONFINITE')
 need(all(bool(t.allclose(u,v,rtol=0,atol=1e-12)) for u,v in zip(x0,xg)),'R63_SELFTEST_ZERO_GATE_SHIFT')
 need(any(not bool(t.allclose(u,v,rtol=0,atol=1e-12)) for u,v in zip(x0,xi)),'R63_SELFTEST_NO_INJECT_EFFECT')
 f0,f1,fg,fi=1.,4.,2.,3.
 sg=.5*(fg-f0+f1-fi);si=.5*(fi-f0+f1-fg)
 need(abs(sg+si-(f1-f0))<1e-14,'R63_SELFTEST_SHAPLEY')
 return True

def main():
 global original
 mode=sys.argv[1] if len(sys.argv)>1 else 'run';status='HOLD';blocker='NONE';child=2
 full=mode=='run';diagn={'parity':0.,'gauge':0.,'disconnected':0.,'min_ref':1e30}
 rows=[];agg=[];done=[];parent_rc='NOT_RUN';reportpath='NONE';reporthash='NONE';proto='NOT_RUN'
 try:
  need(mode in ('run','preflight','selftest'),'MODE')
  import torch as t
  t.set_num_threads(1);t.use_deterministic_algorithms(True);t.set_default_dtype(t.float64)
  test_math(t)
  blob=Path(__file__).with_name('FROZEN_PROTOCOL_R63.json').read_bytes();f=json.loads(blob);proto=digest(blob)
  if mode=='selftest':
   need(f['modes']==list(MODES),'SELFTEST_PROTOCOL');status='SELFTEST_PASS';child=0
  else:
   verify_inputs(f,ROOT)
   # run.sh extracts byte-pinned original R62.py into parent directory; do NOT modify it.
   parent=Path(__file__).with_name('parent')
   need((parent/'r62.py').is_file() and filesha(parent/'r62.py')==PARENT_CODE_HASH,'PARENT_SCRIPT_EXTRACT')
   need(filesha(parent/'FROZEN_PROTOCOL_R62.json')==PARENT_PROTOCOL_HASH,'PARENT_PROTOCOL_EXTRACT')
   sys.path.insert(0,str(parent))
   original=importlib.import_module('r62')
   orig_step=original.r61_step_seed
   def hooked(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight=False):
    nonlocal rows,done,diagn
    if not done:
     history=[]
     for key in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds'):history.extend(cfg[key])
     for key in ('base_fit_seeds','adapt_train_seeds','evaluation_seeds'):history.extend(cfg['prospective'][key])
     for grp,key in [('r54_factorial','evaluation_seeds'),('r55_structural','evaluation_seeds'),('r56_transport','evaluation_seeds'),('r57_channels','eval_seeds'),('r58_reference','eval_seeds'),('r59_trace','eval_seeds'),('r60_windows','eval_seeds'),('r61_early_steps','eval_seeds')]:history.extend(cfg[grp][key])
     history.extend(original.r62cfg['r62_full_factorial']['eval_seeds'])
     need(not set(history).intersection(f['seeds']+[f['preflight_seed']]),'R63_TRANSITIVE_SEED_COLLISION')
    prior_rows,checks=orig_step(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,preflight)
    if not done:
     # Independent parity on the exact R62 fixture and learned weights.
     p_rows,p_checks=run_seed(seed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,f,
                             original.r62cfg['r62_full_factorial']['eval_offset'],preflight=True)
     for record in p_rows:
      g,c,a=record['graph'],record['case'],record['arm']
      for pm,rm in [('BASE','BASE_AB'),('FULL','MASK_0001')]:
       lhs=record['modes'][pm]['FIXED_E2'];rhs=prior_rows[(g,c,rm,a)]['FIXED_E2']
       need(abs(lhs-rhs)<=1e-8*max(1.,abs(lhs),abs(rhs)),'R63_R62_DOMAIN_PARITY:'+g+':'+c+':'+a+':'+pm)
     for k in ('parity','gauge','disconnected'):diagn[k]=max(diagn[k],p_checks[k])
     seeds=f['seeds'] if full else [f['preflight_seed']]
     for ix,newseed in enumerate(seeds,1):
      print('[R63] H1_MEDIATION seed '+str(newseed)+' ('+str(ix)+'/'+str(len(seeds))+')',flush=True)
      part,check=run_seed(newseed,models,cfg,r3,r4,r30,torch,teacher,basegraphs,basecases,f,f['offset'],preflight=not full)
      rows.extend(part);done.append(newseed)
      for k in ('parity','gauge','disconnected'):diagn[k]=max(diagn[k],check[k])
      diagn['min_ref']=min(diagn['min_ref'],check['min_ref'])
    return prior_rows,checks
   original.r61_step_seed=hooked
   # Original R62 protocol remains loaded solely from the pinned archive.
   original.r62cfg=json.loads((parent/'FROZEN_PROTOCOL_R62.json').read_bytes())
   original.sys.argv=['r62.py','run' if full else 'preflight']
   print('[R63] PARENT=R62 SHA256_PASS; invoking inherited R62 parent-chain, training replay and domain checks',flush=True)
   parent_rc=original.run()
   need(parent_rc==0,'INHERITED_R62_EXECUTION_FAILED')
   if full:
    r62_expected=ROOT/'YM_AWR_R62_RESULT_3a5803f757ef8911.zip'
    need(r62_expected.is_file() and filesha(r62_expected)=='3a5803f757ef891161b33933f40f4efb9a12efa18b824d419d2b73d3c8381d06','R62_FULL_REPLAY_REPORT_MISMATCH')
   need(diagn['parity']<=1e-10 and diagn['gauge']<=1e-9 and diagn['disconnected']<=1e-10,'R63_MECHANISM_NUMERIC_CHECK')
   need(len(done)==(8 if full else 1),'R63_SEED_COUNT')
   need(len(rows)==len(done)*4*3*2,'R63_CELL_COUNT')
   if full:
    agg=aggregate(rows,f)
    reportpath,reporthash=write_report(rows,agg,f,proto)
   status='PASS' if full else 'PREFLIGHT_PASS';child=0
 except Exception as e:
  blocker=type(e).__name__+':'+str(e)[:170]
  if not isinstance(e,Hold):traceback.print_exc(limit=7)
 finally:
  print('\n=== SEND BACK ===',flush=True)
  print('PHASE=YM_AWR_R62_H1_GATE_INJECTION_LOCALIZATION_R63',flush=True)
  print('MODE='+mode+' RESULT='+status+' PARENT_R62_EXIT='+str(parent_rc),flush=True)
  print('PROTOCOL_SHA256='+str(proto)+' PARENT_R62_SHA256='+PARENT_HASH,flush=True)
  print('EVAL_SEEDS_COMPLETED='+str(len(done))+'/'+str(8 if full else 1) if mode!='selftest' else 'EVAL_SEEDS_COMPLETED=NOT_EXECUTED',flush=True)
  print('R63_CELL_COVERAGE='+str(len(rows))+'/'+str((8 if full else 1)*4*3*2) if mode!='selftest' else 'R63_CELL_COVERAGE=NOT_EXECUTED',flush=True)
  if done: print('H1_NATIVE_PARITY='+fmt(diagn['parity'])+' GAUGE='+fmt(diagn['gauge'])+' DISCONNECTED='+fmt(diagn['disconnected']),flush=True)
  else:print('H1_NATIVE_PARITY=NOT_MEASURED GAUGE=NOT_MEASURED DISCONNECTED=NOT_MEASURED',flush=True)
  print('MIN_REFERENCE_NORM='+fmt(diagn['min_ref']) if done else 'MIN_REFERENCE_NORM=NOT_MEASURED',flush=True)
  for r in agg:
   print('GRAPH_CASE='+r['graph']+'_'+r['case']+'_'+r['arm']+
      ' H1_FULL_DELTA='+fmt(r['H1_FULL_DELTA'])+' GATE_ATTR='+fmt(r['H1_GATE_SHAPLEY'])+
      ' INJECT_ATTR='+fmt(r['H1_INJECT_SHAPLEY'])+' INTERACTION='+fmt(r['H1_TWO_WAY'])+
      ' FAVORABLE_SEEDS='+str(r['H1_FAVORABLE_SEEDS'])+'/8'+
      ' ENERGY_BASE='+fmt(r['ENERGY']['BASE'])+' ENERGY_FULL='+fmt(r['ENERGY']['FULL'])+
      ' COS_BASE='+fmt(r['H1_COS']['BASE'])+' COS_FULL='+fmt(r['H1_COS']['FULL']),flush=True)
  print('SCIENTIFIC_DISPOSITION=R63_RETROSPECTIVE_H1_MECHANISM_LOCALIZATION_NO_NEW_UTILITY',flush=True)
  print('REPORT='+str(reportpath)+' REPORT_SHA256='+str(reporthash),flush=True)
  print('LIMITS=FROZEN_R52_WEIGHTS;NONFLAT_B_SYNTHETIC;SINGLE_H1_SURGERY;NO_NATIVE_GDN_OR_LEARNED_UTILITY;NO_RUNTIME_PERSISTENCE',flush=True)
  print('BLOCKER='+blocker+' R63_CHILD_EXIT='+str(child),flush=True)
  print('=== END SEND BACK ===',flush=True)
 return child
if __name__=='__main__':sys.exit(main())
