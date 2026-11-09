#!/usr/bin/env python3
"""YM/AWR R30: new independent invariant-dependent nonlinear teacher benchmark.
CPU-only; nonlinearity witness, constant-gate LS and Adam, full 88-param HCA.
Synthetic graphs only, no native Spark GDN, no runtime persistent mutation.
"""
from __future__ import annotations
import hashlib,json,math,os,statistics,sys,tempfile,traceback,zipfile
from pathlib import Path
ROOT=Path('/mnt/primesauce/Downloads')
ARMS=('CONST8_ADAM','CONST8_LS','FULL88_ADAM')
class Hold(Exception):pass
def need(ok,msg):
 if not ok:raise Hold(msg)
def sha(buf):return hashlib.sha256(buf).hexdigest()
def filesha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def zin(p,name):
 with zipfile.ZipFile(p) as z:
  need(z.testzip() is None,'ZIP_CRC:'+p.name)
  need(name in z.namelist(),'ZIP_MEMBER:'+p.name+':'+name)
  return z.read(name)
def f(x):return format(x,'.9g')
def avg(xs):return statistics.mean(xs)
def radial(v,torch):return v*torch.clamp(4./(torch.linalg.vector_norm(v,dim=-1,keepdim=True)+1e-30),max=1.)

def invariants(states,s,c,torch):
 a,b=states
 ts=(a,b,s,c)
 feat=[(t*t).sum(-1) for t in ts]
 feat.extend([(a*s).sum(-1),(a*c).sum(-1),(b*s).sum(-1),(b*c).sum(-1),(s*c).sum(-1)])
 feat.append(torch.zeros_like(feat[0]))
 return torch.stack([x/(1+x.abs()) for x in feat],dim=-1)

class NonlinearTeacher:
 """Independent analytic coefficient law. Inputs are only HCA-available gauge invariants.
 Contains products and sine terms, not a constant-gate teacher or an HCA weight transplant.
 """
 def __init__(self,torch):self.torch=torch
 def coeffs(self,states,s,c):
  t=self.torch;v=invariants(states,s,c,t);q=lambda k:v[...,k]
  return t.stack((
   t.sigmoid(.65+.55*t.sin(2.3*q(4))-.25*q(0)*q(8)),
   t.tanh(.25+.70*q(4)*q(4)-.32*t.sin(2.1*q(8))),
   t.tanh(.10+.40*t.sin(2.3*q(5))),
   t.tanh(.03+.32*q(6)*q(7)),
   t.sigmoid(.80-.52*t.sin(2.0*q(7))+.40*q(1)*q(8)),
   t.tanh(-.10-.38*q(6)*q(6)+.27*q(0)),
   t.tanh(.18+.58*t.sin(2.0*q(8))-.24*q(5)*q(6)),
   t.tanh(.02+.40*q(4)*q(7))),dim=-1)
 def __call__(self,states,s,c):
  t=self.torch;a,b=states;k=self.coeffs(states,s,c)
  A=k[...,0,None]*a+k[...,1,None]*s+k[...,2,None]*c+k[...,3,None]*b
  B=k[...,4,None]*b+k[...,5,None]*s+k[...,6,None]*c+k[...,7,None]*a
  return radial(A,t),radial(B,t)

def make_records(seed,cfg,r3,r4,r4cfg,torch,graphs,cases,teacher):
 rows=[];snapshots=cfg['teacher_state_snapshots']
 with torch.no_grad():
  for i in range(cfg['training_fixtures']):
   graph=r4cfg['train_graphs'][(i+seed)%3];case=r4cfg['train_cases'][(i//3+seed)%4]
   mapped,mats,init,flip=r3.generate(graph,case,seed*100000+i*103+1234,cfg['train_batch'],graphs,cases,torch)
   adj=r4.build_links(mapped,mats,torch)
   states=[init,flip]
   for h in range(max(snapshots)+1):
    if h in snapshots:
     target=[r4.graph_step(x,adj,teacher,torch) for x in states]
     rows.append((adj,states[0],states[1],target[0],target[1]))
    if h<max(snapshots):states=[r4.graph_step(x,adj,teacher,torch) for x in states]
 return rows

def identify_ls(records,Class,r25,r4,torch):
 xa=[];xb=[];ya=[];yb=[]
 with torch.no_grad():
  for adj,init,flip,ref,reff in records:
   for state,target in ((init,ref),(flip,reff)):
    ss,cc=r25.graph_signals(state,adj,torch);a,b=state
    for d in adj:
     aa=a[:,d,:];bb=b[:,d,:];s=ss[:,d,:];c=cc[:,d,:]
     xa.append(torch.stack((aa,s,c,bb),dim=-1).reshape(-1,4))
     xb.append(torch.stack((bb,s,c,aa),dim=-1).reshape(-1,4))
     ya.append(target[0][:,d,:].reshape(-1));yb.append(target[1][:,d,:].reshape(-1))
  A=torch.cat(xa);B=torch.cat(xb);Y=torch.cat(ya);Z=torch.cat(yb)
  coeffA=torch.linalg.lstsq(A,Y).solution;coeffB=torch.linalg.lstsq(B,Z).solution
  rank=min(int(torch.linalg.matrix_rank(A)),int(torch.linalg.matrix_rank(B)))
  cond=max(float(torch.linalg.cond(A)),float(torch.linalg.cond(B)))
  pred=torch.cat((A@coeffA-Y,B@coeffB-Z))
  fit=float(pred.square().mean().item())
  vals=[float(x) for x in torch.cat((coeffA,coeffB)).tolist()]
  need(rank==4 and cond<cfg_constant('ls_condition_max'),'LS_IDENTIFIABILITY')
  need(all((0<x<1 if k in (0,4) else abs(x)<1) for k,x in enumerate(vals)),'LS_COEFF_OUT_OF_HCA_RANGE')
  model=Class()
  with torch.no_grad():
   model.gates.weight.zero_()
   for k,x in enumerate(vals):
    model.gates.bias[k]=math.log(x/(1-x)) if k in (0,4) else math.atanh(x)
  model.eval()
  return model,{'rank_min':rank,'condition_max':cond,'fit_mse':fit,'coefficients':vals}

CONSTANTS={'ls_condition_max':1e3}
def cfg_constant(k):return CONSTANTS[k]

def fit(seed,records,cfg,Class,r4,torch,full,updates):
 torch.manual_seed(seed);model=Class();need(model.parameter_count==88,'PARAMETER_COUNT')
 if not full:
  with torch.no_grad():model.gates.weight.zero_()
  model.gates.weight.requires_grad_(False)
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=cfg['learning_rate'])
 init_bias=model.gates.bias.detach().clone()
 maxg=0.;clips=0;last=math.nan
 for it in range(updates):
  adj,init,flip,targ,targf=records[it%len(records)]
  out=r4.graph_step(init,adj,model,torch);outf=r4.graph_step(flip,adj,model,torch)
  loss=.5*(r4.state_loss(out,targ)+r4.state_loss(outf,targf))
  need(bool(torch.isfinite(loss).item()),'LOSS_NONFINITE')
  opt.zero_grad(set_to_none=True);loss.backward()
  need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'GRAD_NONFINITE')
  norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['clip_norm']))
  maxg=max(maxg,norm);clips+=int(norm>cfg['clip_norm'])
  opt.step();last=float(loss.detach())
  if it+1 in cfg['progress_updates']:
   print('[R30] seed='+str(seed)+' arm='+('FULL88' if full else 'CONST8')+' update='+str(it+1)+'/'+str(updates)+' state_mse='+f(last)+' grad='+f(norm),flush=True)
 if not full:need(bool((model.gates.weight==0).all()),'CONST8_WEIGHTS_CHANGED')
 model.eval()
 return model,{'last_mse':last,'max_grad':maxg,'clip_updates':clips,'init_bias':init_bias.tolist()}

def coefficient_spread(records,r25,teacher,torch):
 allcoeff=[]
 with torch.no_grad():
  for adj,init,flip,*_ in records:
   for state in (init,flip):
    s,c=r25.graph_signals(state,adj,torch)
    for d in adj:allcoeff.append(teacher.coeffs((state[0][:,d,:],state[1][:,d,:]),s[:,d,:],c[:,d,:]))
 coeff=torch.cat(allcoeff,dim=0)
 std=coeff.std(0,unbiased=False)
 return {'max_std':float(std.max()),'mean_std':float(std.mean()),'std':std.tolist()}

def evaluate(model,teacher,seed,cfg,r3,r4,torch,graphs,cases):
 before=r4.model_hash(model);out=[];cov=0.;leak=0.;minref=math.inf;state_err=[];ref_clip=0.
 with torch.no_grad():
  for gi,g in enumerate(cfg['new_graphs']):
   for ci,c in enumerate(cfg['new_cases']):
    mapped,mats,init,flip=r3.generate(g,c,cfg['eval_offset']+seed*100109+gi*293+ci*59,cfg['eval_batch'],graphs,cases,torch)
    adj=r4.build_links(mapped,mats,torch)
    tr=r4.rollout(init,adj,teacher,cfg['eval_horizon'],torch)[-1]
    tf=r4.rollout(flip,adj,teacher,cfg['eval_horizon'],torch)[-1]
    pr=r4.rollout(init,adj,model,cfg['eval_horizon'],torch)[-1]
    pf=r4.rollout(flip,adj,model,cfg['eval_horizon'],torch)[-1]
    if g==cfg['disconnected_graph']:
     leak=max(leak,float(pr[0][:,3,:].abs().max()),float((pr[0][:,3,:]-pf[0][:,3,:]).abs().max()),
              float(tr[0][:,3,:].abs().max()))
     continue
    state_err.append(float(r4.state_loss(pr,tr)))
    ref=tr[0][:,3,:]-tf[0][:,3,:];pred=pr[0][:,3,:]-pf[0][:,3,:]
    den=ref.square().sum(-1)
    need(bool((den>1e-25).all()),'DEGENERATE_NL_REFERENCE:'+c+':'+g)
    minref=min(minref,float(ref.norm(dim=-1).min()))
    gain=((ref*pred).sum(-1)/den)
    err=(pred-ref).square().sum(-1)/den
    row={'graph':g,'case':c,'gain':float(gain.mean()),'err':float(err.mean()),'sign':float((gain>0).double().mean())}
    need(all(math.isfinite(v) for v in row.values() if isinstance(v,float)),'NONFINITE_NL_EVAL')
    out.append(row)
    if g==cfg['gauge_graph'] and c==cfg['gauge_case']:
     gauges={d:r3.so2x2_torch(.19+.13*d+.0005*seed,-.29-.11*d-.0007*seed,torch) for d in range(4)}
     gadj=r4.links_gauge(adj,gauges,torch)
     p_rotate=r4.rollout(r4.gauge_transform(init,gauges,torch),gadj,model,cfg['eval_horizon'],torch)[-1]
     t_rotate=r4.rollout(r4.gauge_transform(init,gauges,torch),gadj,teacher,cfg['eval_horizon'],torch)[-1]
     for orig,rot in ((pr,p_rotate),(tr,t_rotate)):
      expect=r4.gauge_transform(orig,gauges,torch)
      cov=max(cov,*(float((rot[k]-expect[k]).abs().max()) for k in (0,1)))
 need(len(out)==3*(len(cfg['new_cases'])),'EVAL_COVERAGE')
 need(before==r4.model_hash(model),'WEIGHT_MUTATION_DURING_INFERENCE')
 return {'rows':out,'cov':cov,'leak':leak,'minref':minref,'state_mse':avg(state_err)}

def run():
 mode=sys.argv[1] if len(sys.argv)>1 else 'run'
 rc=2;status='HOLD';pins='HOLD';proto='HOLD';post='NOT_RUN';blocker='NONE';science='NOT_RUN';cfgsha='NA'
 done=[];results={a:[] for a in ARMS};witness=[];stats={};gates={}
 maxcov=0.;leak=0.;refmin=math.inf;gradmax=0.;clips=0
 try:
  need(mode in ('run','local_run','preflight','local_preflight','local_focus'),'MODE')
  local=mode.startswith('local_');full=mode in ('run','local_run');focus=mode=='local_focus'
  need(Path('/dev/shm').is_dir(),'DEV_SHM_MISSING')
  import torch
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  blob=Path(__file__).with_name('FROZEN_PROTOCOL_R30.json').read_bytes();cfgsha=sha(blob);cfg=json.loads(blob)
  need(cfg['schema']=='elpis.ym.awr.g5.nonlinear_invariant_teacher_r30.v1' and cfg['status']=='FROZEN_BEFORE_EIGHT_SEED_EVALUATION','PROTOCOL_SCHEMA')
  need(cfg['seeds']==[1103,1109,1117,1123,1129,1151,1153,1163] and cfg['arms']==list(ARMS),'SEED_ARM_DRIFT')
  need(cfg['train_updates']==240 and cfg['training_fixtures']==24 and cfg['teacher_state_snapshots']==[0,2,6] and cfg['eval_horizon']==20,'HORIZON_DRIFT')
  root=Path(os.getenv('YM_R30_LOCAL_ROOT','/mnt/data')) if local else ROOT
  paths={k:root/cfg[k+'_archive'] for k in ('r29','r28','r27','r26')}
  for k,p in paths.items():need(p.is_file() and filesha(p)==cfg[k+'_sha256'],'PARENT_ZIP:'+k)
  for k,p in paths.items():need(sha(zin(p,'FROZEN_PROTOCOL_'+k.upper()+'.json'))==cfg[k+'_protocol_sha256'],'PARENT_PROTOCOL:'+k)
  r29cfg=json.loads(zin(paths['r29'],'FROZEN_PROTOCOL_R29.json'))
  r26cfg=json.loads(zin(paths['r26'],'FROZEN_PROTOCOL_R26.json'))
  need(all(r29cfg[k+'_sha256']==cfg[k+'_sha256'] for k in ('r28','r27','r26')),'PARENT_CHAIN')
  upstream={k:root/n for k,n in r26cfg['parents'].items()}
  for k,p in upstream.items():need(p.is_file() and filesha(p)==r26cfg['parent_archive_sha256'][k],'UPSTREAM_ZIP:'+k)
  for k,p in upstream.items():need(sha(zin(p,'FROZEN_PROTOCOL_R'+k[1:]+'.json'))==r26cfg['parent_protocol_sha256'][k],'UPSTREAM_PROTOCOL:'+k)
  with tempfile.TemporaryDirectory(prefix='ym_awr_r30_src_',dir='/dev/shm') as td0:
   td=Path(td0)
   for name,p,member in [('r25_parent.py',upstream['r25'],'r25.py'),('r8_parent.py',upstream['r8'],'r8.py'),('r10_parent.py',upstream['r10'],'r10.py')]:
    (td/name).write_bytes(zin(p,member))
   sys.path.insert(0,str(td));import r25_parent as r25,r8_parent as r8,r10_parent as r10
   r8cfg=json.loads(zin(upstream['r8'],'FROZEN_PROTOCOL_R8.json'))
   src,g5,_=r8.parent_sources(r8cfg,local)
   r24cfg=json.loads(zin(upstream['r24'],'FROZEN_PROTOCOL_R24.json'))
   hca=zin(src['YM_AWR_GEOMETRY_R0.zip'],'ym_awr_hca.py')
   need(sha(hca)==r24cfg['hca_source_sha256'],'HCA_SOURCE_PIN')
   (td/'ym_awr_hca.py').write_bytes(hca)
   (td/'r3_parent.py').write_bytes(zin(src['YM_AWR_G5_EQUIVARIANT_GRAPH_BRIDGE_R3.zip'],'probe.py'))
   (td/'r4_parent.py').write_bytes(zin(src['YM_AWR_G5_SIGNED_CF_OFFLINE_R4.zip'],'train_r4.py'))
   from ym_awr_hca import EquivariantHCA
   import r3_parent as r3,r4_parent as r4
   r4cfg=json.loads(zin(src['YM_AWR_G5_SIGNED_CF_OFFLINE_R4.zip'],'FROZEN_PROTOCOL_R4.json'))
   basegraphs,basecases=r3.frozen_constants(g5)
   oldgraphs=set(basegraphs)|set(r29cfg['new_graphs'])
   oldcases=set(basecases)|set(r29cfg['new_cases'])
   need(not (set(cfg['new_graphs'])&oldgraphs) and not(set(cfg['new_cases'])&oldcases),'HOLDOUT_LABEL_REUSE')
   for n,g in cfg['new_graphs'].items():need(len(g)==4 and r10.reach(g)==(n!=cfg['disconnected_graph']),'GRAPH_REACH:'+n)
   graphs={**basegraphs,**{k:tuple(tuple(x) for x in v) for k,v in cfg['new_graphs'].items()}}
   cases={**basecases,**{k:tuple(v) for k,v in cfg['new_cases'].items()}}
   teacher=NonlinearTeacher(torch)
   pins='PASS';proto='PASS'
   seeds=cfg['seeds'] if full else [cfg['pilot_seed']]
   updates=2 if mode in ('preflight','local_preflight') else cfg['train_updates']
   for seed in seeds:
    records=make_records(seed,cfg,r3,r4,r4cfg,torch,basegraphs,basecases,teacher)
    spread=coefficient_spread(records,r25,teacher,torch)
    ls,ident=identify_ls(records,EquivariantHCA,r25,r4,torch)
    c8,m8=fit(seed,records,cfg,EquivariantHCA,r4,torch,False,updates)
    c88,m88=fit(seed,records,cfg,EquivariantHCA,r4,torch,True,updates)
    # Pairing: same initial gate biases, 88 extra weights initialized by original HCA.
    need(m8['init_bias']==m88['init_bias']==[.8,0.,0.,0.,.8,0.,0.,0.],'PAIR_INITIALIZATION')
    for name,model,meta in ((ARMS[0],c8,m8),(ARMS[1],ls,ident),(ARMS[2],c88,m88)):
     result=evaluate(model,teacher,seed,cfg,r3,r4,torch,graphs,cases)
     results[name].append({'seed':seed,'eval':result,'meta':meta})
     maxcov=max(maxcov,result['cov']);leak=max(leak,result['leak']);refmin=min(refmin,result['minref'])
    witness.append({'spread':spread,'LS_train_fit_mse':ident['fit_mse'],'LS_condition':ident['condition_max'],'rank':ident['rank_min']})
    gradmax=max(gradmax,m8['max_grad'],m88['max_grad']);clips+=m8['clip_updates']+m88['clip_updates']
    done.append(seed)
    def e(name,case):return avg(r['err'] for r in results[name][-1]['eval']['rows'] if r['case']==case)
    print('[R30] seed='+str(seed)+' gate_std='+f(spread['max_std'])+' LS_fit='+f(ident['fit_mse'])+
          ' weak8='+f(e(ARMS[0],cfg['weak_case']))+' weak88='+f(e(ARMS[2],cfg['weak_case']))+
          ' reverse8='+f(e(ARMS[0],cfg['reverse_case']))+' reverse88='+f(e(ARMS[2],cfg['reverse_case'])),flush=True)
   need(maxcov<1e-9 and leak<1e-10 and refmin>1e-11,'INVARIANTS')
   need(min(x['rank'] for x in witness)==4 and max(x['LS_condition'] for x in witness)<1e3,'LS_DESIGN_VALID')
   if full:
    def vals(name,case,k):return [r[k] for rec in results[name] for r in rec['eval']['rows'] if r['case']==case]
    def perseed(name,case,k):return [avg(r[k] for r in rec['eval']['rows'] if r['case']==case) for rec in results[name]]
    l=cfg['acceptance'];w=cfg['weak_case'];reverse=cfg['reverse_case']
    w8=perseed(ARMS[0],w,'err');wls=perseed(ARMS[1],w,'err');w88=perseed(ARMS[2],w,'err')
    r8=perseed(ARMS[0],reverse,'err');r88=perseed(ARMS[2],reverse,'err')
    stats={'gate_std_min':min(x['spread']['max_std'] for x in witness),
     'const_ls_train_fit_min':min(x['LS_train_fit_mse'] for x in witness),
     'paired_88_vs_adam8':sum(a<b for a,b in zip(w88,w8)),
     'paired_88_vs_ls8':sum(a<b for a,b in zip(w88,wls)),
     'reverse_88_vs_adam8':sum(a<b for a,b in zip(r88,r8)),
     'weak88_null_seeds':sum(a<1 for a in w88),
     'weak88_null_cells':sum(a<1 for a in vals(ARMS[2],w,'err')),
     'reverse88_null_cells':sum(a<1 for a in vals(ARMS[2],reverse,'err')),
     'weak88_median':statistics.median(w88),
     'weak88_worst':max(vals(ARMS[2],w,'err')),
     'reverse88_worst':max(vals(ARMS[2],reverse,'err')),
     'weak88_sign':avg(vals(ARMS[2],w,'sign')),
     'anti88_sign':avg(vals(ARMS[2],cfg['anti_case'],'sign')),
     'reverse88_sign':avg(vals(ARMS[2],reverse,'sign')),
     'balanced88_sign':avg(vals(ARMS[2],cfg['balanced_case'],'sign'))}
    gates={'nonconstant_teacher':stats['gate_std_min']>=l['min_teacher_gate_std'],
     'constant_ls_not_exact':stats['const_ls_train_fit_min']>=l['min_constant_ls_mse'],
     'full88_vs_constant_adam':stats['paired_88_vs_adam8']>=l['paired_wins_min'],
     'full88_vs_constant_ls':stats['paired_88_vs_ls8']>=l['paired_wins_min'],
     'reverse_paired':stats['reverse_88_vs_adam8']>=l['paired_wins_min'],
     'weak_null_seeds':stats['weak88_null_seeds']>=l['null_seeds_min'],
     'weak_null_cells':stats['weak88_null_cells']>=l['null_cells_min'],
     'reverse_null_cells':stats['reverse88_null_cells']>=l['null_cells_min'],
     'weak_median':stats['weak88_median']<=l['median_err_max'],
     'weak_worst':stats['weak88_worst']<=l['worst_err_max'],
     'reverse_worst':stats['reverse88_worst']<=l['worst_reverse_err_max'],
     'weak_sign':stats['weak88_sign']>=l['min_sign'],
     'anti_sign':stats['anti88_sign']>=l['min_sign'],
     'reverse_sign':stats['reverse88_sign']>=l['min_sign'],
     'balanced_sign':stats['balanced88_sign']>=l['min_sign']}
    science='R30_NONLINEAR_PILOT_'+('CRITERIA_MET' if all(gates.values()) else 'CRITERIA_NOT_MET')
   else:science='EXCLUDED_PILOT_ONLY_NO_EIGHT_SEED_SCIENCE'
   for k,p in paths.items():need(filesha(p)==cfg[k+'_sha256'],'PARENT_POSTCHECK:'+k)
   for k,p in upstream.items():need(filesha(p)==r26cfg['parent_archive_sha256'][k],'UPSTREAM_POSTCHECK:'+k)
   r8.parent_sources(r8cfg,local)
   post='PASS';status='PASS' if full else 'PREFLIGHT_PASS';rc=0
 except (Hold,KeyError,FileNotFoundError,OSError,ValueError,RuntimeError,ImportError,zipfile.BadZipFile) as e:
  blocker=type(e).__name__+':'+str(e)[:170]
 except KeyboardInterrupt:rc=130;blocker='INTERRUPTED'
 except Exception as e:
  rc=1;blocker='UNEXPECTED_'+type(e).__name__+':'+str(e)[:170];traceback.print_exc(limit=6)
 finally:
  print('\n=== SEND BACK ===',flush=True)
  print('PHASE=YM_AWR_R29_NONLINEAR_INVARIANT_TEACHER_R30',flush=True)
  print('MODE='+mode+' RESULT='+status+' PARENT_PINS='+pins+' PROTOCOL='+proto+' PARENT_POSTCHECK='+post,flush=True)
  print('PROTOCOL_SHA256='+cfgsha,flush=True)
  print('SEEDS_COMPLETED='+str(len(done))+'/'+('8' if full else '1')+' MAX_COV='+f(maxcov)+' DISCONNECTED='+f(leak)+' TEACHER_MIN='+f(refmin),flush=True)
  if witness:
   print('NONLINEAR_TEACHER_WITNESS=GATE_STD_MIN:'+f(min(x['spread']['max_std'] for x in witness))+
    ' GATE_STD_MEAN:'+f(avg(x['spread']['mean_std'] for x in witness))+
    ' CONSTANT_LS_TRAIN_MSE_MIN:'+f(min(x['LS_train_fit_mse'] for x in witness))+
    ' IDENT_RANK_MIN:'+str(min(x['rank'] for x in witness))+
    ' LS_CONDITION_MAX:'+f(max(x['LS_condition'] for x in witness)),flush=True)
  for arm in ARMS:
   if results[arm]:
    def v(case,k):return avg(r[k] for rec in results[arm] for r in rec['eval']['rows'] if r['case']==case)
    print(arm+'=WEAK64_ERR:'+f(v('r30_weak_64','err'))+' GAIN:'+f(v('r30_weak_64','gain'))+
      ' SIGN:'+f(v('r30_weak_64','sign'))+
      ' ANTI64_GAIN:'+f(v('r30_anti_64','gain'))+' ANTI_SIGN:'+f(v('r30_anti_64','sign'))+
      ' REVERSE64_ERR:'+f(v('r30_reverse_64','err'))+' GAIN:'+f(v('r30_reverse_64','gain'))+
      ' SIGN:'+f(v('r30_reverse_64','sign'))+' BALANCED_SIGN:'+f(v('r30_balanced','sign'))+
      ' WEAK512_ERR:'+f(v('r30_weak_512','err'))+
      ' STATE_MSE:'+f(avg(rec['eval']['state_mse'] for rec in results[arm])),flush=True)
  if stats:
   print('PAIRED_88_VS_CONST_ADAM='+str(stats['paired_88_vs_adam8'])+'/8 PAIRED_88_VS_CONST_LS='+str(stats['paired_88_vs_ls8'])+
    '/8 REVERSE_PAIRED='+str(stats['reverse_88_vs_adam8'])+'/8 WEAK_NULL_SEEDS='+str(stats['weak88_null_seeds'])+
    '/8 WEAK_NULL_CELLS='+str(stats['weak88_null_cells'])+'/24 REVERSE_NULL_CELLS='+str(stats['reverse88_null_cells'])+
    '/24 WORST_WEAK='+f(stats['weak88_worst'])+' WORST_REVERSE='+f(stats['reverse88_worst']),flush=True)
   print('FROZEN_SCIENTIFIC_GATES='+','.join(k+':'+('PASS' if v else 'NONPASS') for k,v in gates.items()),flush=True)
  print('OPTIMIZER=MAX_GRAD:'+f(gradmax)+' CLIP_UPDATES:'+str(clips)+' UPDATES_PER_ARM:'+str(2 if mode in ('preflight','local_preflight') else 240),flush=True)
  print('SCIENCE='+science,flush=True)
  print('LIMITS=NEW_HANDSPECIFIED_NONLINEAR_INVARIANT_TEACHER;TEACHER_FORCED_PRIVILEGED_OFFLINE_STATES;SHARED_HCA_FEATURES_NOT_EXTERNAL_DYNAMICS;SAME_UPDATES_NOT_FLOP_MATCHED;SYNTHETIC_GRAPH_PILOT_NOT_NATIVE_GDN_OR_SEMANTIC_CONTRADICTION;NO_RUNTIME_PERSISTENT_MUTATION',flush=True)
  print('BLOCKER='+blocker,flush=True)
  print('R30_CHILD_EXIT='+str(rc),flush=True)
  print('=== END SEND BACK ===',flush=True)
 return rc
if __name__=='__main__':sys.exit(run())
