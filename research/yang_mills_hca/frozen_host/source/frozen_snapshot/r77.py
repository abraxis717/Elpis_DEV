#!/usr/bin/env python3
"""R77: preregistered offline one-step teacher-state consistency test of 88p HCA.
Source-pinned R30 nonlinear teacher/R62 transport/R63 88p gates. Training only
uses offline oracle states; inference unroll never reads teacher state. Synthetic.
"""
from __future__ import annotations
import copy,hashlib,io,json,math,pathlib,random,statistics,sys,zipfile
import torch as T
import portable_mechanics as pm,r30_source as r30,r62_source as r62,r63_source as r63,r64
from make_proto import generate
ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'FROZEN_PROTOCOL_R77.json').read_bytes())
T.set_num_threads(1)
class Reject(RuntimeError):pass
def need(cond, why):
 if not cond:raise Reject(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canon(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def source_gate():
 for name,digest in CFG['source_sha256'].items():
  need((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==digest,'SOURCE:'+name)
 need(generate()==CFG['graphs'],'GRAPH_GENERATION_DRIFT')
 need(CFG['target_records']==len(CFG['graphs'])*len(CFG['arms'])*len(CFG['train_initializations'])*len(CFG['evaluation_seeds'])*len(CFG['evaluation_cases'])==576,'ROW_COUNT_PROTOCOL')
 old=set(r64.EPS)|set(r64.EVAL_SEEDS)|{r64.PROTO['excluded_preflight_seed']}
 for k in [70,71,72,73,74,75,76]:
  p=json.loads((ROOT/f'FROZEN_PROTOCOL_R{k}.json').read_bytes());old.update(p['evaluation_seeds']);old.update(p.get('training_initializations',[]));old.update(p.get('training_seeds',[]))
 seeds=CFG['training_episode_seeds']+CFG['train_initializations']+CFG['evaluation_seeds']+[CFG['episode_order_seed']]
 need(len(seeds)==len(set(seeds)) and not old.intersection(seeds),'SEED_LEAK')
 need(CFG['arms']==['STATE8','STATE8_SIGNED8','STATE8_TF1'],'ARM_DRIFT')
 need(CFG['train_graphs']==['D1_P1','D2_P1'] and CFG['train_updates_per_model']==48,'TRAIN_CONTRACT')
 need(CFG['predeclared_success']=={'relative_pooled_H20_nse_vs_state_and_signed_at_most':.95,'min_graphs_better_than_state':3,'max_graph_absolute_harm_vs_state':.05,'same_init_paired_comparison':True},'SUCCESS_CONTRACT')
 r64.source_gate();return True
def init_model(seed):
 model=r64.HCA()
 generator=T.Generator(device='cpu').manual_seed(seed)
 with T.no_grad():
  model.gates.weight.add_(T.randn(model.gates.weight.shape,dtype=T.float64,generator=generator)*.01)
  model.gates.bias.add_(T.randn(model.gates.bias.shape,dtype=T.float64,generator=generator)*.002)
 need(sum(p.numel() for p in model.parameters())==88,'NOT_88P')
 return model
def episodes():
 teacher=r30.NonlinearTeacher(T);r61=pm.read_json('FROZEN_PROTOCOL_R61.json');eps=[]
 with T.no_grad():
  for seed in CFG['training_episode_seeds']:
   for g in CFG['train_graphs']:
    adj=pm.graph_adjs(r61['r55_structural']['graphs'][g],r61)[0]
    for case in CFG['train_cases']:
     st=r64.pair_state(seed,case,64,batch=8)
     gold=r64.rollout(st,adj,teacher,8,keep=(4,5,7,8))
     eps.append((st,adj,gold))
 need(len(eps)==48,'EPISODES');return eps
def train():
 source_gate();ep=episodes();order=list(range(48));random.Random(CFG['episode_order_seed']).shuffle(order)
 weights=CFG['loss_weights'];optcfg=CFG['training_optimizer'];models={};hist={}
 for init in CFG['train_initializations']:
  base=init_model(init);initial_pin=r64.parameter_pin(base)
  for arm in CFG['arms']:
   m=copy.deepcopy(base);opt=T.optim.Adam(m.parameters(),lr=optcfg['lr'],weight_decay=optcfg['weight_decay']);losses=[];clips=0;tf_last=None
   for upd,idx in enumerate(order,1):
    st,adj,gold=ep[idx];pred=r64.rollout(st,adj,m,8,keep=(4,8))
    loss=weights['state_h4']*r64.state_mse(pred[4],gold[4])+weights['state_h8']*r64.state_mse(pred[8],gold[8])
    if arm=='STATE8_SIGNED8':
     loss=loss+weights['signed_h4']*r64.signed_loss(pred[4],gold[4],8,weights['signed_norm_floor'])+weights['signed_h8']*r64.signed_loss(pred[8],gold[8],8,weights['signed_norm_floor'])
    if arm=='STATE8_TF1':
     p5=r62.graph_step_split(gold[4],adj,adj,m,T)
     p8=r62.graph_step_split(gold[7],adj,adj,m,T)
     loss=loss+weights['tf_h4_h5']*r64.state_mse(p5,gold[5])+weights['tf_h7_h8']*r64.state_mse(p8,gold[8])
     tf_last=float((r64.state_mse(p5,gold[5])+r64.state_mse(p8,gold[8])).detach())
    need(bool(T.isfinite(loss)),'TRAIN_LOSS_NONFINITE')
    opt.zero_grad(set_to_none=True);loss.backward()
    gn=float(T.nn.utils.clip_grad_norm_(m.parameters(),optcfg['clip_norm']))
    need(math.isfinite(gn),'TRAIN_GRAD_NONFINITE');clips+=int(gn>1.);opt.step();losses.append(float(loss.detach()))
    if upd%16==0:print(f'[R77] init={init} arm={arm} update={upd}/48 loss={losses[-1]:.6g}',flush=True)
   pin=r64.parameter_pin(m);need(pin!=initial_pin and len(losses)==48,'TRAIN_NOT_COMPLETE')
   key=f'{init}/{arm}';models[key]=m
   hist[key]={'checkpoint_sha256':pin,'initial_sha256':initial_pin,'weights':m.gates.weight.detach().tolist(),'bias':m.gates.bias.detach().tolist(),'updates':48,'clip_count':clips,'first_loss':losses[0],'last_loss':losses[-1],'tf_last_unweighted':tf_last}
   m.requires_grad_(False)
 need(len(models)==6,'MODEL_COUNT')
 return models,hist
def train_fast():
 """Source-pinned R62 stacked gates; six independent Adam trajectories in one tensor.
 Check exact per-arm single-update parity against serial source before qualification.
 """
 source_gate();ep=episodes();order=list(range(48));random.Random(CFG['episode_order_seed']).shuffle(order)
 ids=[(init,arm) for init in CFG['train_initializations'] for arm in CFG['arms']]
 n=len(ids);per=ep[0][0][0].shape[0];lw=CFG['loss_weights'];oc=CFG['training_optimizer']
 starts={init:init_model(init) for init in CFG['train_initializations']}
 W=T.nn.Parameter(T.stack([starts[tid].gates.weight.detach().clone() for tid,arm in ids]))
 bias=T.nn.Parameter(T.stack([starts[tid].gates.bias.detach().clone() for tid,arm in ids]))
 opt=T.optim.Adam((W,bias),lr=oc['lr'],weight_decay=oc['weight_decay'])
 masks={a:T.tensor([int(arm==a) for tid,arm in ids],dtype=T.float64) for a in CFG['arms']}
 def mse_batch(u,v):
  return sum((u[ch].reshape(n,per,4,4)-v[ch].unsqueeze(0)).square().mean(dim=(1,2,3)) for ch in (0,1))
 def signed_batch(u,v):
  a=u[0].reshape(n,per,4,4);b=v[0];half=per//2
  x=a[:,:half,3,:]-a[:,half:,3,:];y=b[:half,3,:]-b[half:,3,:]
  return ((x-y).square().sum(-1)/(y.square().sum(-1)+lw['signed_norm_floor']**2)).mean(-1)
 losses={f'{tid}/{arm}':[] for tid,arm in ids};clips={k:0 for k in losses}
 for upd,j in enumerate(order,1):
  st,adj,gold=ep[j];expand=lambda ss:tuple(x.repeat(n,1,1) for x in ss)
  law=r62._StackedGates(W,bias,r30,T,per)
  pred=r64.rollout(expand(st),adj,law,8,keep=(4,8))
  l=lw['state_h4']*mse_batch(pred[4],gold[4])+lw['state_h8']*mse_batch(pred[8],gold[8])
  l=l+masks['STATE8_SIGNED8']*(lw['signed_h4']*signed_batch(pred[4],gold[4])+lw['signed_h8']*signed_batch(pred[8],gold[8]))
  p5=r62.graph_step_split(expand(gold[4]),adj,adj,law,T)
  p8=r62.graph_step_split(expand(gold[7]),adj,adj,law,T)
  l=l+masks['STATE8_TF1']*(lw['tf_h4_h5']*mse_batch(p5,gold[5])+lw['tf_h7_h8']*mse_batch(p8,gold[8]))
  need(bool(T.isfinite(l).all()),'FAST_LOSS_NONFINITE')
  opt.zero_grad(set_to_none=True);l.sum().backward()
  norms=(W.grad.square().sum((1,2))+bias.grad.square().sum(1)).sqrt()
  need(bool(T.isfinite(norms).all()),'FAST_GRAD_NONFINITE')
  for i,(tid,arm) in enumerate(ids):
   key=f'{tid}/{arm}';norm=float(norms[i]);clips[key]+=int(norm>oc['clip_norm'])
   scale=min(1.,oc['clip_norm']/(norm+1e-6));W.grad[i].mul_(scale);bias.grad[i].mul_(scale)
   losses[key].append(float(l[i].detach()))
  opt.step()
  if upd==1:
   # Exact unbatched gate-law and optimizer-step parity, all 6 arms.
   for i,(tid,arm) in enumerate(ids):
    m=copy.deepcopy(starts[tid]);op=T.optim.Adam(m.parameters(),lr=oc['lr'],weight_decay=oc['weight_decay'])
    x=r64.rollout(st,adj,m,8,keep=(4,8))
    sl=lw['state_h4']*r64.state_mse(x[4],gold[4])+lw['state_h8']*r64.state_mse(x[8],gold[8])
    if arm=='STATE8_SIGNED8':
     sl=sl+lw['signed_h4']*r64.signed_loss(x[4],gold[4],per//2,lw['signed_norm_floor'])+lw['signed_h8']*r64.signed_loss(x[8],gold[8],per//2,lw['signed_norm_floor'])
    if arm=='STATE8_TF1':
     q5=r62.graph_step_split(gold[4],adj,adj,m,T);q8=r62.graph_step_split(gold[7],adj,adj,m,T)
     sl=sl+lw['tf_h4_h5']*r64.state_mse(q5,gold[5])+lw['tf_h7_h8']*r64.state_mse(q8,gold[8])
    need(abs(float(sl.detach())-losses[f'{tid}/{arm}'][-1])<1e-11,'STACKED_LOSS_PARITY')
    op.zero_grad(set_to_none=True);sl.backward();T.nn.utils.clip_grad_norm_(m.parameters(),oc['clip_norm']);op.step()
    delta=max(float((m.gates.weight-W[i]).abs().max()),float((m.gates.bias-bias[i]).abs().max()))
    need(delta<2e-12,'STACKED_OPTIMIZER_PARITY')
   print('[R77] SOURCE_STACKED_TRAINING_6_ARM_FIRST_UPDATE_PARITY=PASS',flush=True)
  if upd%16==0:
   print(f'[R77] stacked training update={upd}/48 losses='+json.dumps({k:round(v[-1],7) for k,v in losses.items()},sort_keys=True),flush=True)
 models={};history={}
 for i,(tid,arm) in enumerate(ids):
  key=f'{tid}/{arm}';m=copy.deepcopy(starts[tid])
  with T.no_grad():m.gates.weight.copy_(W[i]);m.gates.bias.copy_(bias[i])
  m.requires_grad_(False);pin=r64.parameter_pin(m)
  need(pin!=r64.parameter_pin(starts[tid]),'UNTRAINED_MODEL');models[key]=m
  history[key]={'checkpoint_sha256':pin,'initial_sha256':r64.parameter_pin(starts[tid]),
   'weights':m.gates.weight.detach().tolist(),'bias':m.gates.bias.detach().tolist(),'updates':48,
   'clip_count':clips[key],'first_loss':losses[key][0],'last_loss':losses[key][-1]}
 return models,history

def join_states(seeds=CFG['evaluation_seeds'],cases=CFG['evaluation_cases'],batch=CFG['batch_per_seed']):
 percase=len(seeds)*batch
 ss=[r64.pack_seeds(tuple(seeds),case,CFG['evaluation_ratio'],batch)[0] for case in cases]
 state=tuple(T.cat((T.cat([s[ch][:percase] for s in ss],0),T.cat([s[ch][percase:] for s in ss],0)),0) for ch in (0,1))
 need(state[0].shape==(2*percase*len(cases),4,4),'STATE_SHAPE')
 return state

def metrics(pred,tar):
 need(pred.shape==tar.shape and pred.shape[-1]==4,'METRICS_SHAPE')
 den=tar.square().sum(-1);need(bool((den>1e-20).all()),'TARGET_NORM')
 return {'mean_nse':float((((pred-tar).square().sum(-1))/den).mean()),'gain':float(((pred*tar).sum(-1)/den).mean()),'cosine':float(((pred*tar).sum(-1)/(pred.square().sum(-1).sqrt()*den.sqrt()+1e-30)).mean())}

def assessment(rows):
 pooled={arm:statistics.mean(row['h']['20']['mean_nse'] for row in rows if row['arm']==arm) for arm in CFG['arms']}
 bygraph={g:{arm:statistics.mean(row['h']['20']['mean_nse'] for row in rows if row['arm']==arm and row['graph']==g) for arm in CFG['arms']} for g in CFG['graphs']}
 c=CFG['predeclared_success'];test='STATE8_TF1';base='STATE8';signed='STATE8_SIGNED8'
 graphwins=sum(bygraph[g][test]<bygraph[g][base] for g in bygraph)
 pooled_ok=pooled[test]<=c['relative_pooled_H20_nse_vs_state_and_signed_at_most']*min(pooled[base],pooled[signed])
 harm_ok=all(bygraph[g][test]-bygraph[g][base]<=c['max_graph_absolute_harm_vs_state'] for g in bygraph)
 seed_effect={}
 for init in CFG['train_initializations']:
  group=[r for r in rows if r['initialization']==init]
  seed_effect[str(init)]={a:statistics.mean(r['h']['20']['mean_nse'] for r in group if r['arm']==a) for a in CFG['arms']}
 byh={str(h):{a:statistics.mean(r['h'][str(h)]['mean_nse'] for r in rows if r['arm']==a) for a in CFG['arms']} for h in CFG['evaluation_horizons']}
 return {'pooled_h20_nse':pooled,'by_graph_h20_nse':bygraph,'by_horizon_nse':byh,'by_initialization_h20_nse':seed_effect,
  'tf_graph_wins_vs_state':graphwins,'tf_no_large_graph_harm':bool(harm_ok),'tf_pooled_better_than_both_baselines':bool(pooled_ok),
  'scientific_gate':'SYNTHETIC_OFFLINE_LOCAL_CONSISTENCY_PASS_BOUNDED' if pooled_ok and harm_ok and graphwins>=c['min_graphs_better_than_state'] else 'SYNTHETIC_OFFLINE_LOCAL_CONSISTENCY_NONQUAL',
  'qualifier':'Offline oracle-state SUPERVISION only; teacher forced local metric is diagnostic; no teacher feedback, resets, or new controller at inference.'}

def evaluate(models):
 keys=sorted(models);state=join_states();per=state[0].shape[0];n=len(keys)
 w=T.stack([models[k].gates.weight.detach() for k in keys]);b=T.stack([models[k].gates.bias.detach() for k in keys]);law=r62._StackedGates(w,b,r30,T,per)
 init=tuple(x.repeat(n,1,1) for x in state);teacher=r30.NonlinearTeacher(T);r61=pm.read_json('FROZEN_PROTOCOL_R61.json');out=[]
 pin={k:r64.parameter_pin(m) for k,m in models.items()};block=len(CFG['evaluation_seeds'])*CFG['batch_per_seed']
 with T.no_grad():
  for graph,g in CFG['graphs'].items():
   adj=pm.graph_adjs(g,r61)[1];ref=r64.rollout(state,adj,teacher,20,keep=(8,12,19,20))
   gold={h:r63.paired(ref[h][0],per//2) for h in (8,12,20)}
   s=init;pred={}
   for h in range(1,21):
    s=r62.graph_step_split(s,adj,adj,law,T)
    if h in (8,12,20):pred[h]=s
   for i,key in enumerate(keys):
    ix=slice(i*per,(i+1)*per);arm=key.split('/')[1];tid=int(key.split('/')[0])
    # Independent teacher-forced local diagnostic, NOT inference-time correction.
    tf=r62.graph_step_split(ref[19],adj,adj,models[key],T)
    teacher_forced=r63.paired(tf[0],per//2)
    response={h:r63.paired(pred[h][0][ix],per//2) for h in (8,12,20)}
    for ci,case in enumerate(CFG['evaluation_cases']):
     for si,seed in enumerate(CFG['evaluation_seeds']):
      z=slice(ci*block+si*CFG['batch_per_seed'],ci*block+(si+1)*CFG['batch_per_seed'])
      states={str(h):{'response4':response[h][z].tolist(),'target4':gold[h][z].tolist(),**metrics(response[h][z],gold[h][z])} for h in CFG['evaluation_horizons']}
      loc={'response4':teacher_forced[z].tolist(),'target4':gold[20][z].tolist(),**metrics(teacher_forced[z],gold[20][z])}
      out.append({'initialization':tid,'arm':arm,'graph':graph,'eval_seed':seed,'case':case,'h':states,'tf_local_h20':loc})
   print(f'[R77] evaluated {graph}: {len(out)}/{CFG["target_records"]} records',flush=True)
 need(len(out)==CFG['target_records'],'RECORD_COUNT')
 need(all(r64.parameter_pin(models[k])==pin[k] for k in keys),'EVAL_MUTATED_WEIGHTS')
 return out

def save_result(obj,path):
 raw=canon(obj);info=zipfile.ZipInfo('result.json',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.writestr(info,raw)
 with zipfile.ZipFile(path) as z:need(z.testzip() is None and z.read('result.json')==raw,'RESULT_ZIP')
 return sha(path.read_bytes()),sha(raw)
def run():
 print('[R77] pinned preflight; offline 88p matched training; no heldout graph selection by outcome',flush=True)
 models,hist=train_fast();rows=evaluate(models)
 obj={'phase':CFG['phase'],'protocol_sha256':sha((ROOT/'FROZEN_PROTOCOL_R77.json').read_bytes()),'source_sha256':CFG['source_sha256'],
  'parent_r76_archive_sha256':CFG['parent_r76_archive_sha256'],'training':hist,'rows':rows,'assessment':assessment(rows),'limitations':CFG['limitations']}
 report=ROOT/'YM_AWR_R77_RESULT.zip';h,raw=save_result(obj,report)
 print('[R77] SCIENTIFIC_GATE='+obj['assessment']['scientific_gate'],flush=True)
 print('[R77] POOLED='+json.dumps(obj['assessment']['pooled_h20_nse'],sort_keys=True),flush=True)
 print('[R77] GRAPH_NSE='+json.dumps(obj['assessment']['by_graph_h20_nse'],sort_keys=True),flush=True)
 print('[R77] REPORT_SHA256='+h+' JSON_SHA256='+raw,flush=True)
 return obj
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'preflight'
 if mode=='preflight':source_gate();print('R77_PREDECLARED_SOURCE_GRAPH_SEED_PREFLIGHT=PASS',flush=True)
 elif mode=='run':run()
 else:raise SystemExit('BAD_MODE')
