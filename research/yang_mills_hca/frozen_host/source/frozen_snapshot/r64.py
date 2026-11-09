#!/usr/bin/env python3
"""YM/AWR R64 new prospective 88-param HCA experiment, portable and source-pinned.

NOT historical R52 learned weights or qualification. The exact published R30 nonlinear
teacher and R62/R63 geometric operations are reused byte-for-byte through a source-
pinned portable R63 fixture generator. This runner trains two new 88-param HCAs
only OFFLINE, checks frozen inference, then reports paired H20 signed fidelity.
"""
from __future__ import annotations
import copy, hashlib, io, json, math, os, pathlib, statistics, sys, traceback, zipfile
import torch
import portable_mechanics as pm
import r30_source as r30
import r62_source as r62
import r63_source as r63

T=torch
T.set_num_threads(1)
ROOT=pathlib.Path(__file__).resolve().parent
PROTO=json.loads((ROOT/'FROZEN_PROTOCOL_R64.json').read_text())
ARMS=tuple(PROTO['arms'])
TRAIN_GRAPHS=tuple(PROTO['train_graphs'])
EVAL_GRAPHS=tuple(PROTO['eval_graphs'])
CASES=tuple(PROTO['cases'])
EPS=tuple(PROTO['train_episode_seeds'])
EVAL_SEEDS=tuple(PROTO['eval_seeds'])
EXPECTED={
 'r30_source.py':'51e05a242b9ec6411d8a2bb093f779b4d7ac8bab4e1ac9def28d4d89f6b3d8b9',
 'r62_source.py':'fe272da88e3833ad5deaad305dc5ab4d07b666af1fe357a07da7df10e1a799ee',
 'r63_source.py':'6f2fb05aabbc06e8b1892f35949bdcc168941831a368ede9b1f895a6b4877eea',
 'FROZEN_PROTOCOL_R61.json':'632222d5ecfdb8fd0acb51a6de732a819251bfe2cdc3351e900be78445a9dda4',
 'FROZEN_PROTOCOL_R62.json':'39ae807dd4a40039912240933c2d5004b002acd3f8458c96636cedb42ef1a886',
 'FROZEN_PROTOCOL_R63.json':'c900364137f00309c34a7ca49c89f7cb8bfffb63b10257b6e2d28cb5a4ccd81b',
 'portable_mechanics.py': 'PLACEHOLDER',
 'FROZEN_PROTOCOL_R64.json':'PLACEHOLDER',
}
class Hold(RuntimeError):pass
def need(c,e):
 if not c:raise Hold(e)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):return sha_bytes(pathlib.Path(p).read_bytes())
def fmt(x):return format(float(x),'.9g')
def source_gate():
 for name, pin in EXPECTED.items():
  if pin=='PLACEHOLDER':continue
  need(file_sha(ROOT/name)==pin,'SOURCE_PIN_'+name)
 pm.source_admission()
 need(len(set(EPS+EVAL_SEEDS+(PROTO['excluded_preflight_seed'],)))==len(EPS)+len(EVAL_SEEDS)+1,'SEED_OVERLAP')
 need(set(TRAIN_GRAPHS).isdisjoint(PROTO['heldout_graphs']),'GRAPH_OVERLAP')
 need(PROTO['train_updates_per_arm']==len(EPS)*len(TRAIN_GRAPHS)*len(CASES),'TRAIN_EPISODE_COVERAGE')
 need(tuple(PROTO['eval_horizons'])==(1,8,20),'HORIZON_PROTOCOL')
 need(len(set(ARMS))==3 and ARMS[0]=='FROZEN_INIT','ARM_PROTOCOL')
 return True

class HCA(T.nn.Module):
 def __init__(self):
  super().__init__();self.gates=T.nn.Linear(10,8,dtype=T.float64)
  initial=[.65,.25,.10,.03,.80,-.10,.18,.02]
  with T.no_grad():
   self.gates.weight.zero_()
   self.gates.bias.copy_(T.tensor([math.log(v/(1-v)) if i in (0,4) else math.atanh(v) for i,v in enumerate(initial)],dtype=T.float64))
 def forward(self,state,sal,col):return r63.hybrid_step(self,state,sal,col,col,r30,T)[0]

def parameter_pin(m):return sha_bytes(b''.join(p.detach().cpu().numpy().tobytes() for p in m.parameters()))

def pair_state(seed,case,ratio,batch=8):
 """At ratio 64, EXACT portable R63 fixture; at 160, scale weak source alone."""
 state=pm.state_pair(seed,case,batch)
 if ratio==64:return state
 need(ratio==160,'BAD_SOURCE_RATIO')
 # Preserve the sampled orientation and strong-source amplitude; scale weak a/b source.
 a,b=(x.clone() for x in state)
 if case in ('weak','anti'):
  a[:batch,0,:]*=64/ratio;b[:batch,0,:]*=64/ratio
 else:
  a[batch:,0,:]*=64/ratio;b[batch:,0,:]*=64/ratio
 return a,b

def pack_seeds(seeds,case,ratio,batch=8):
 xs=[pair_state(s,case,ratio,batch) for s in seeds];B=len(seeds)*batch
 def merge(k):return T.cat([T.cat([x[k][:batch] for x in xs],dim=0),T.cat([x[k][batch:] for x in xs],dim=0)])
 return (merge(0),merge(1)),B

def rollout(state,adj,model,steps,keep=(1,4,8,20)):
 s=state;outs={}
 for h in range(1,steps+1):
  s=r62.graph_step_split(s,adj,adj,model,T)
  if h in keep:outs[h]=s
 return outs

def state_mse(s,t):return sum((x-y).square().mean() for x,y in zip(s,t))
def signed_loss(s,t,B,floor):
 a=r63.paired(s[0],B);b=r63.paired(t[0],B)
 return ((a-b).square().sum(-1)/(b.square().sum(-1)+floor*floor)).mean()

def train_models(cfg):
 initial=HCA();need(sum(x.numel() for x in initial.parameters())==88,'PARAMS_NOT_88')
 models={'FROZEN_INIT':initial};hist={}
 ref= r30.NonlinearTeacher(T)
 episodes=[]
 for seed in EPS:
  for graph in TRAIN_GRAPHS:
   for case in CASES:
    state=pair_state(seed,case,cfg['train_ratio'])
    base,_=pm.graph_adjs(pm.read_json('FROZEN_PROTOCOL_R61.json')['r55_structural']['graphs'][graph],pm.read_json('FROZEN_PROTOCOL_R61.json'))
    with T.no_grad():gold=rollout(state,base,ref,8,keep=(4,8))
    episodes.append((seed,graph,case,state,base,gold))
 need(len(episodes)==cfg['train_updates_per_arm'],'EPISODE_COUNT')
 print(f'[R64] frozen train population {len(EPS)} seeds x {len(TRAIN_GRAPHS)} graphs x {len(CASES)} cases = {len(episodes)} updates',flush=True)
 for arm in ARMS[1:]:
  model=copy.deepcopy(initial)
  opt=T.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
  stats={'first_loss':None,'last_loss':None,'clip_count':0,'max_gradient_norm':0.0,'updates':0}
  for n,(seed,graph,case,state,adj,gold) in enumerate(episodes):
   pred=rollout(state,adj,model,8,keep=(4,8))
   loss=cfg['state_h4_weight']*state_mse(pred[4],gold[4])+cfg['state_h8_weight']*state_mse(pred[8],gold[8])
   if arm=='STATE8_SIGNED8':
    B=state[0].shape[0]//2
    loss=loss+cfg['signed_h4_weight']*signed_loss(pred[4],gold[4],B,cfg['signed_norm_floor'])+cfg['signed_h8_weight']*signed_loss(pred[8],gold[8],B,cfg['signed_norm_floor'])
   need(bool(T.isfinite(loss)),'NONFINITE_TRAIN_LOSS')
   opt.zero_grad(set_to_none=True);loss.backward()
   gn=float(T.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']))
   need(math.isfinite(gn),'NONFINITE_GRADIENT')
   stats['clip_count']+=int(gn>cfg['gradient_clip_norm']);stats['max_gradient_norm']=max(gn,stats['max_gradient_norm'])
   opt.step()
   if stats['first_loss'] is None:stats['first_loss']=float(loss.detach())
   stats['last_loss']=float(loss.detach());stats['updates']=n+1
   if (n+1)%16==0:print(f'[R64] {arm} update={n+1}/{len(episodes)} loss={fmt(loss.detach())} clipped={stats["clip_count"]}',flush=True)
  need(stats['updates']==cfg['train_updates_per_arm'],'TRAIN_INCOMPLETE')
  models[arm]=model;stats['param_sha256']=parameter_pin(model);hist[arm]=stats
 return models,hist

def scalar_stats(vec,target):
 dp=(vec*target).sum(-1);den=target.square().sum(-1);s=vec.square().sum(-1)
 err=(vec-target).square().sum(-1)
 need(bool((den>1e-20).all()),'ZERO_TARGET_NORM')
 return {'mean_nse':float((err/den).mean()),'energy_nse':float(err.sum()/den.sum()),
         'mean_abs_sq':float(err.mean()),'gain':float((dp/den).mean()),
         'cosine':float((dp/(T.sqrt(s*den)+1e-30)).mean()),
         'worst_nse':float((err/den).max()),'min_ref':float(T.sqrt(den).min())}

def evaluate_unbatched(models,cfg,seed_subset=EVAL_SEEDS,preflight=False):
 need(len(set(seed_subset))==len(seed_subset),'DUP_EVAL_SEED')
 pins={a:parameter_pin(models[a]) for a in ARMS}
 records=[];max_cov=0.0;checks=0
 teacher=r30.NonlinearTeacher(T)
 for graph in EVAL_GRAPHS:
  gg=pm.read_json('FROZEN_PROTOCOL_R61.json')['r55_structural']['graphs'][graph]
  flat,nonflat=pm.graph_adjs(gg,pm.read_json('FROZEN_PROTOCOL_R61.json'))
  for ratio in cfg['eval_ratios']:
   for case in CASES:
    state,B=pack_seeds(seed_subset,case,ratio,cfg['batch_per_seed']);batch=cfg['batch_per_seed']
    with T.no_grad():
     teacher_flat=rollout(state,flat,teacher,20,keep=(1,8,20))
     for mode in cfg['eval_transport']:
      adj=flat if mode=='BASE_SEPARABLE' else nonflat
      gold=teacher_flat if mode=='BASE_SEPARABLE' else rollout(state,adj,teacher,20,keep=(1,8,20))
      pred={name:rollout(state,adj,models[name],20,keep=(1,8,20)) for name in ARMS}
      for si,seed in enumerate(seed_subset):
       sl=slice(si*batch,(si+1)*batch)
       for arm in ARMS:
        hmetrics={}
        for h in cfg['eval_horizons']:
         tar=r63.paired(gold[h][0],B)[sl]
         base=r63.paired(teacher_flat[h][0],B)[sl]
         out=r63.paired(pred[arm][h][0],B)[sl]
         stats=scalar_stats(out,tar)
         fixed=scalar_stats(out,base)
         stats['fixed_reference_nse']=fixed['mean_nse']
         stats['teacher_displacement_abs_sq']=float((tar-base).square().sum(-1).mean())
         hmetrics[str(h)]=stats
        records.append({'seed':seed,'graph':graph,'ratio':ratio,'case':case,'transport':mode,'arm':arm,'heldout_graph':graph in cfg['heldout_graphs'],'h':hmetrics})
       checks+=1
     print(f'[R64] evaluation graph={graph} case={case} ratio={ratio} completed modes={len(cfg["eval_transport"])}',flush=True)
 for arm in ARMS:need(parameter_pin(models[arm])==pins[arm],'INFERENCE_WEIGHT_MUTATION:'+arm)
 target_rows=len(seed_subset)*len(EVAL_GRAPHS)*len(CASES)*len(cfg['eval_ratios'])*len(cfg['eval_transport'])*len(ARMS)
 need(len(records)==target_rows,'ROW_COUNT')
 return records,pins

def evaluate(models,cfg,seed_subset=EVAL_SEEDS,preflight=False):
 """Batch independent (case,ratio,seed) fixtures per graph without mixing samples.
 Preserves exact per-cell arithmetic; compared to unbatched R64 reference.
 """
 need(len(set(seed_subset))==len(seed_subset),'DUP_EVAL_SEED')
 pins={a:parameter_pin(models[a]) for a in ARMS}
 records=[];teacher=r30.NonlinearTeacher(T)
 b=cfg['batch_per_seed'];B=len(seed_subset)*b
 combos=[(ratio,case) for ratio in cfg['eval_ratios'] for case in CASES]
 raw_states=[pack_seeds(seed_subset,case,ratio,b)[0] for ratio,case in combos]
 def join(ch):
  return T.cat([T.cat([s[ch][:B] for s in raw_states],dim=0),
                T.cat([s[ch][B:] for s in raw_states],dim=0)],dim=0)
 state=(join(0),join(1));batch_total=B*len(combos)
 for graph in EVAL_GRAPHS:
  gg=pm.read_json('FROZEN_PROTOCOL_R61.json')['r55_structural']['graphs'][graph]
  flat,nonflat=pm.graph_adjs(gg,pm.read_json('FROZEN_PROTOCOL_R61.json'))
  with T.no_grad():
   teacher_flat=rollout(state,flat,teacher,20,keep=(1,8,20))
   for mode in cfg['eval_transport']:
    adj=flat if mode=='BASE_SEPARABLE' else nonflat
    gold=teacher_flat if mode=='BASE_SEPARABLE' else rollout(state,adj,teacher,20,keep=(1,8,20))
    pred={name:rollout(state,adj,models[name],20,keep=(1,8,20)) for name in ARMS}
    for gi,(ratio,case) in enumerate(combos):
     for si,seed in enumerate(seed_subset):
      sl=slice(gi*B+si*b,gi*B+(si+1)*b)
      for arm in ARMS:
       hmetrics={}
       for h in cfg['eval_horizons']:
        tar=r63.paired(gold[h][0],batch_total)[sl]
        base=r63.paired(teacher_flat[h][0],batch_total)[sl]
        out=r63.paired(pred[arm][h][0],batch_total)[sl]
        stats=scalar_stats(out,tar)
        fixed=scalar_stats(out,base)
        stats['fixed_reference_nse']=fixed['mean_nse']
        stats['teacher_displacement_abs_sq']=float((tar-base).square().sum(-1).mean())
        hmetrics[str(h)]=stats
       records.append({'seed':seed,'graph':graph,'ratio':ratio,'case':case,'transport':mode,
          'arm':arm,'heldout_graph':graph in cfg['heldout_graphs'],'h':hmetrics})
   print(f'[R64] evaluation graph={graph} complete ratios={len(cfg["eval_ratios"])} cases={len(CASES)} modes={len(cfg["eval_transport"])}',flush=True)
 for arm in ARMS:need(parameter_pin(models[arm])==pins[arm],'INFERENCE_WEIGHT_MUTATION:'+arm)
 target_rows=len(seed_subset)*len(EVAL_GRAPHS)*len(CASES)*len(cfg['eval_ratios'])*len(cfg['eval_transport'])*len(ARMS)
 need(len(records)==target_rows,'ROW_COUNT')
 return records,pins

def summarize(rows):
 table={}
 for mode in ('all','heldout_graphs','heldout_graphs_nonflat_160','familiar_graphs'):
  subset=[r for r in rows if (mode=='all' or (mode=='heldout_graphs' and r['heldout_graph']) or (mode=='heldout_graphs_nonflat_160' and r['heldout_graph'] and r['transport']=='PARTICIPANT_NONFLAT' and r['ratio']==160) or (mode=='familiar_graphs' and not r['heldout_graph']))]
  for case in CASES:
   for arm in ARMS:
    vals=[r['h']['20'] for r in subset if r['case']==case and r['arm']==arm]
    table[f'{mode}/{case}/{arm}']={'n':len(vals),'mean_h20_nse':statistics.mean(v['mean_nse'] for v in vals),
            'median_h20_nse':statistics.median(v['mean_nse'] for v in vals),
            'energy_nse_mean':statistics.mean(v['energy_nse'] for v in vals),
            'gain_mean':statistics.mean(v['gain'] for v in vals),
            'cosine_mean':statistics.mean(v['cosine'] for v in vals),
            'worst_cell_nse':max(v['worst_nse'] for v in vals)}
 return table

def selftest():
 source_gate();m=HCA();need(sum(p.numel() for p in m.parameters())==88,'SELF_88_PARAM')
 cfg=pm.read_json('FROZEN_PROTOCOL_R61.json');gr=cfg['r55_structural']['graphs']['D1_P1']
 flat,_=pm.graph_adjs(gr,cfg);state=pair_state(PROTO['excluded_preflight_seed'],'anti',64)
 state160=pair_state(PROTO['excluded_preflight_seed'],'anti',160)
 need(float((state[0][8:]-state160[0][8:]).abs().max())==0.,'STRONG_SOURCE_CHANGED')
 need(float((state[0][:8,0,:]*.4-state160[0][:8,0,:]).abs().max())<1e-14,'WEAK_SOURCE_SCALING')
 sal,col=r63.inputs(state,flat,flat,T)
 with T.no_grad():
  direct=r62.graph_step_split(state,flat,flat,m,T)
  hybrid=r63.hybrid_step(m,state,sal,col,col,r30,T)[0]
  need(max(float((direct[k]-hybrid[k]).abs().max()) for k in (0,1))<1e-13,'SELF_SOURCE_STEP_PARITY')
 cov=pm.run_one(PROTO['excluded_preflight_seed'],'D1_P1','anti','CONTROL_A',cfg,preflight=True)['h1']['gauge_covariance']
 need(cov<1e-9,'SELF_GAUGE_COVARIANCE')
 print(f'[R64] source gate PASS; HCA=88; exact native step parity PASS; gauge={fmt(cov)}',flush=True)
 return cov

def run(mode):
 selftest()
 if mode=='selftest':return
 if mode=='preflight':
  models={a:HCA() for a in ARMS};rows,pins=evaluate(models,PROTO,seed_subset=(PROTO['excluded_preflight_seed'],),preflight=True)
  print(f'[R64] preflight PASS rows={len(rows)} no training or holdout consumption',flush=True);return
 need(mode=='run','BAD_MODE')
 print('[R64] OFFLINE training fresh 88-param models; not a historical R52 reproduction',flush=True)
 models,train=train_models(PROTO)
 print('[R64] fresh trained models frozen; evaluating heldout graphs, transport and 160:1 ratios',flush=True)
 rows,pins=evaluate(models,PROTO)
 summary=summarize(rows)
 report={'protocol_sha256':file_sha(ROOT/'FROZEN_PROTOCOL_R64.json'),'source_sha256':{k:file_sha(ROOT/k) for k in EXPECTED},
   'scope':'NEW_PROSPECTIVE_PORTABLE_FIXTURES_NOT_HISTORICAL_R52','training':train,
   'evaluation_seed_count':len(EVAL_SEEDS),'models':{a:{'weight':models[a].gates.weight.detach().tolist(),
   'bias':models[a].gates.bias.detach().tolist(),'sha256':pins[a]} for a in ARMS},
   'summary':summary,'records':rows}
 raw=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 need(len(raw)<5_000_000,'REPORT_TOO_LARGE')
 h=sha_bytes(raw);out=pathlib.Path(os.environ.get('YM_R64_HANDOFF_DIR','/dev/shm'))/f'YM_AWR_R64_RESULT_{h[:16]}.zip'
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  z.writestr('result.json',raw)
  z.writestr('REPORT_SHA256.txt',h+'  result.json\n')
 with zipfile.ZipFile(out) as z:need(z.testzip() is None and sha_bytes(z.read('result.json'))==h,'REPORT_RECHECK')
 reportzip=file_sha(out)
 import verify_report
 verified_rows, rejected_mutations = verify_report.verify(str(out),full=False)
 need(verified_rows==24 and rejected_mutations==5,'INDEPENDENT_VERIFIER')
 print('\n=== SEND BACK ===')
 print('PHASE='+PROTO['phase']);print('MODE=run');print('RESULT=PROSPECTIVE_FIXTURE_RUN_PASS')
 print('SOURCE_PINS=PASS PROTOCOL_SHA256='+report['protocol_sha256'])
 print(f'TRAIN_UPDATES={PROTO["train_updates_per_arm"]}/ARM EVAL_SEEDS={len(EVAL_SEEDS)}/{len(EVAL_SEEDS)} ROWS={len(rows)}')
 print('FROZEN_R52=NOT_AVAILABLE_NOT_REPRODUCED;LEARNED_UTILITY_HISTORICAL_R52=NONQUAL')
 for case in CASES:
  for arm in ARMS:
   key=f'heldout_graphs_nonflat_160/{case}/{arm}'
   x=summary[key];print(f'OOD_{case}_{arm}=NSE:{fmt(x["mean_h20_nse"])} GAIN:{fmt(x["gain_mean"])} COS:{fmt(x["cosine_mean"])} WORST:{fmt(x["worst_cell_nse"])}')
 print('INDEPENDENT_REPORT_STRUCTURE=PASS_1152_ROWS SPOT_REPLAY=PASS_24_ROWS ADVERSARIAL_REJECTIONS=5/5')
 print('INFERENCE_WEIGHT_MUTATION=NONE TRAINING_ONLY_IN_VOLATILE_MEMORY=YES')
 print('SCIENTIFIC_DISPOSITION=PROSPECTIVE_PORTABLE_SYNTHETIC_LEARNING_ONLY_NO_NATIVE_GDN_OR_PRODUCTION_AUTHORITY')
 print(f'REPORT={out} REPORT_SHA256={reportzip} RESULT_JSON_SHA256={h}')
 print('BLOCKER=NONE CHILD_EXIT=0');print('=== END SEND BACK ===',flush=True)

if __name__=='__main__':
 try:run(sys.argv[1] if len(sys.argv)>1 else 'selftest')
 except BaseException as e:
  traceback.print_exc()
  print('\n=== SEND BACK ===\nPHASE='+PROTO['phase']+'\nRESULT=HOLD\nBLOCKER='+type(e).__name__+':'+str(e)+'\nCHILD_EXIT=2\n=== END SEND BACK ===',flush=True)
  sys.exit(2)
