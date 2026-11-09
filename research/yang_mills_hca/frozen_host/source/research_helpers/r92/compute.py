"""R92: offline local-defect sensitivity to initial-state scaling and graph family.

No new weights, no hybrid-order attribution and no inference-time teacher access.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,statistics
from r89 import compute as r89c
from r91 import compute as r91c
HERE=pathlib.Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,why):
 if not v:raise RuntimeError(why)
def rowkey(r):return (r['graph'],r['model_initialization'],r['arm'],r['eval_seed'],r['case'],r['initial_scale'])
def case_seed(ix):return P['cases'][ix//8],P['fresh_evaluation_seeds'][(ix%8)//4]

def graphs(pm,r77):
 p76=pm.read_json('FROZEN_PROTOCOL_R76.json')['graphs']; p77=r77.CFG['graphs']
 selected={g:(p77[g] if g in p77 else p76[g]) for g in P['graphs']}
 need(all(g in p77 for g in P['anchor_graphs']),'ANCHOR_PROVENANCE')
 need(all(g in p76 and g not in p77 for g in P['transfer_graphs']),'TRANSFER_PROVENANCE')
 sigs={g:sha(canonical(r89c.iso_signature(v))) for g,v in selected.items()}
 need(len(set(sigs.values()))==len(sigs),'GRAPH_ISOMORPHISM_COLLISION')
 previous,_=r89c.declared_graphs(pm,r77)
 need(not set(sigs[g] for g in P['transfer_graphs'])&{sha(canonical(r89c.iso_signature(v))) for v in previous.values()},'TRANSFER_ISO_REUSED')
 need(sigs==P['graph_signatures_sha256'],'GRAPH_SIGNATURE_PIN')
 return selected,sigs

def start_states(r64,t):
 seeds=P['fresh_evaluation_seeds'];cases=P['cases'];batch=P['batch_per_seed'];K=len(seeds)*batch
 base=[r64.pack_seeds(seeds,case,P['ratio'],batch=batch)[0] for case in cases]
 inputs=tuple(t.cat((t.cat([x[ch][:K] for x in base]),t.cat([x[ch][K:] for x in base]))) for ch in (0,1))
 need(tuple(v.shape for v in inputs)==((32,4,4),(32,4,4)),'BASE_INPUT_SHAPE')
 return inputs

def initial_references(state):
 result={}
 for ix in P['selected_indices']:
  case,seed=case_seed(ix)
  for scale in P['amplitude_scales']:
   a=[float(v*scale) for v in state[0][ix].reshape(-1)]
   b=[float(v*scale) for v in state[1][ix].reshape(-1)]
   result[f'{seed}:{case}:{scale:g}']=a+b
 return result

def verify_initial_references(values):
 expected=P['initial_state_reference']
 need(set(values)==set(expected)==set(P['initial_state_pins_sha256']),'INITIAL_REFERENCE_KEYS')
 for key,observed in values.items():
  ref=expected[key]
  need(len(observed)==len(ref)==32,'INITIAL_REFERENCE_SHAPE')
  need(all(abs(a-b)<=2e-8*(1+abs(b)) for a,b in zip(observed,ref)),'INITIAL_REFERENCE_MISMATCH:'+key)
  need(sha(canonical(ref))==P['initial_state_pins_sha256'][key],'INITIAL_REFERENCE_SHA:'+key)
 return True

def collect(r78,r77,r64,r62,pm,r30,source):
 import torch as t
 from r83 import geometry as g83
 seeds=P['fresh_evaluation_seeds'];
 need(len(seeds)==2 and len(set(seeds))==2,'SEED_CARDINALITY')
 old=set(r77.CFG['evaluation_seeds'])|set(r77.CFG['train_initializations'])|set(r77.CFG['training_episode_seeds'])
 old.update(r64.EPS);old.update(r64.EVAL_SEEDS)
 for k in range(70,78):
  pr=pm.read_json(f'FROZEN_PROTOCOL_R{k}.json')
  for fld in ('evaluation_seeds','training_seeds','training_initializations','training_episode_seeds','train_initializations'):
   old.update(pr.get(fld,[]))
 need(not set(seeds)&old,'SEED_NOT_FRESH')
 need(P['selected_indices']==[0,4,8,12] and P['amplitude_scales']==[.5,1.,2.],'FROZEN_INTERVENTION')
 selected,sigs=graphs(pm,r77)
 models=r78.models_from_r77(source);names=sorted(models)
 need(len(names)==P['models'],'FROZEN_HCA_COUNT')
 W=t.stack([models[k].gates.weight.detach() for k in names]);B=t.stack([models[k].gates.bias.detach() for k in names]);n=len(names)
 start=start_states(r64,t)
 need(verify_initial_references(initial_references(start)),'INITIAL_STATE_SOURCE_PIN')
 r61=pm.read_json('FROZEN_PROTOCOL_R61.json')
 teacher=r30.NonlinearTeacher(t)
 scales=P['amplitude_scales']; rows=[];links={}
 with t.no_grad():
  for graph,incidence in selected.items():
   adj=pm.graph_adjs(incidence,r61)[1];edges=g83.edge_table(adj)
   links[graph]=[[int(e[0]),int(e[1])] for e in edges]
   need(links[graph]==P['reciprocal_link_pairs'][graph],'EDGE_LAYOUT_PIN')
   # Teacher path combines all 3 amplitude scales on the independent sample axis.
   current=tuple(t.cat([v*sc for sc in scales]) for v in start)
   teacher_states=[tuple(v.clone() for v in current)]
   for h in range(20):teacher_states.append(r62.graph_step_split(teacher_states[-1],adj,adj,teacher,t))
   law=r62._StackedGates(W.repeat(len(scales),1,1),B.repeat(len(scales),1),r30,t,start[0].shape[0])
   pred=[]
   for h in range(20):
    s=teacher_states[h]
    inp=tuple(t.cat([v[j*32:(j+1)*32].repeat(n,1,1) for j in range(len(scales))]) for v in s)
    step=r62.graph_step_split(inp,adj,adj,law,t)
    pred.append(tuple(v.reshape(len(scales),n,32,4,4) for v in step))
   for j,scale in enumerate(scales):
    for mi,name in enumerate(names):
     init,arm=name.split('/')
     for ix in P['selected_indices']:
      case,seed=case_seed(ix);before=[];after=[];qn=[];rn=[]
      for h in range(20):
       s=t.cat((teacher_states[h][0][j*32+ix].reshape(16),teacher_states[h][1][j*32+ix].reshape(16)))
       sn=t.cat((teacher_states[h+1][0][j*32+ix].reshape(16),teacher_states[h+1][1][j*32+ix].reshape(16)))
       po=t.cat((pred[h][0][j,mi,ix].reshape(16),pred[h][1][j,mi,ix].reshape(16)))
       q,r=r91c.defect_norm(s,po,sn,edges,t)
       if h==0:before.append([float(v) for v in s])
       before.append([float(v) for v in sn]);after.append([float(v) for v in po]);qn.append(q);rn.append(r)
      sq=sum(x*x for x in qn)
      rows.append({'graph':graph,'graph_signature_sha256':sigs[graph],'model_initialization':int(init),'arm':arm,'eval_seed':seed,'case':case,'initial_scale':scale,'metric_link_scale':1.,
       'initial_state_sha256':P['initial_state_pins_sha256'][f'{seed}:{case}:{scale:g}'],
       'teacher_states':before,'model_teacher_one_steps':after,'quotient_defect_norms':qn,'raw_defect_norms':rn,
       'local_peak_horizon':qn.index(max(qn))+1,'early_energy_fraction':sum(x*x for x in qn[:8])/max(sq,1e-30)})
   print(f'[R92] graph={graph} records={len(rows)}/{P["records"]}',flush=True)
 need(len(rows)==P['records'] and links==P['reciprocal_link_pairs'],'RECORD_OR_LINK_COVERAGE')
 return rows,sigs

def summarize(rows):
 by={};total=0
 for graph in P['graphs']:
  scales={}
  for scale in P['amplitude_scales']:
   records=[r for r in rows if r['graph']==graph and r['initial_scale']==scale]
   need(len(records)==24,'CELL_SIZE')
   early=sum(r['local_peak_horizon']<=P['early_window'] for r in records)
   total+=early
   scales[f'{scale:g}']={'records':len(records),'early_peak':early,'mean_early_energy_fraction':statistics.mean(r['early_energy_fraction'] for r in records),
      'mean_h1_defect':statistics.mean(r['quotient_defect_norms'][0] for r in records),
      'mean_h20_defect':statistics.mean(r['quotient_defect_norms'][19] for r in records)}
  by[graph]=scales
 qualified=all(by[g][f'{s:g}']['early_peak']/24>=P['min_early_fraction_per_graph_scale'] for g in P['graphs'] for s in P['amplitude_scales'])
 return {'by_graph_and_scale':by,'early_peak_total':total,'scientific_gate':'BOUNDED_INITIAL_STATE_AND_GRAPH_TRANSFER_PASS' if qualified else 'BOUNDED_INITIAL_STATE_AND_GRAPH_TRANSFER_NONQUAL',
 'disposition':'OFFLINE_TEACHER_STATE_LOCAL_DEFECT_INITIAL_STATE_SENSITIVITY_NO_AUTONOMOUS_UTILITY'}
