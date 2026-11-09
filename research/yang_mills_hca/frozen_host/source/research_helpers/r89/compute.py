"""Frozen, fresh-seed offline forcing transfer; no new weights or oracle runtime access."""
from __future__ import annotations
import hashlib,itertools,json,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_bytes())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def need(p,msg):
 if not p:raise RuntimeError(msg)

def iso_signature(graph):
 """4-relation incidence-isomorphism signature, ignoring participant labels."""
 need(len(graph)==4 and all(len(set(x))==4 for x in graph),'INCIDENCE_SHAPE')
 masks=[]
 for perm in itertools.permutations(range(4)):
  census=[sum((1<<i) for i,r in enumerate(perm) if p in graph[r]) for p in range(10)]
  masks.append(tuple(census.count(i) for i in range(16)))
 return min(masks)

def declared_graphs(pm,r77):
 p76=pm.read_json('FROZEN_PROTOCOL_R76.json')['graphs'];p77=r77.CFG['graphs']
 selection={n:(p76[n] if n in p76 else p77[n]) for n in P['graphs']}
 signatures=[iso_signature(g) for g in selection.values()]
 need(len(set(signatures))==len(signatures),'GRAPH_ISOMORPHISM_COLLISION')
 need(all(n in p77 for n in P['parent_family_graphs']) and all(n in p76 and n not in p77 for n in P['transfer_family_graphs']),'GRAPH_FAMILY')
 return selection,{k:sha(canonical(iso_signature(v))) for k,v in selection.items()}

def input_state(r64,t):
 seeds=P['fresh_evaluation_seeds'];cases=P['cases'];batch=P['batch_per_seed'];ratio=P['ratio'];K=len(seeds)*batch
 inputs=[r64.pack_seeds(seeds,case,ratio,batch=batch)[0] for case in cases]
 init=tuple(t.cat((t.cat([x[ch][:K] for x in inputs]),t.cat([x[ch][K:] for x in inputs]))) for ch in (0,1))
 need(init[0].shape==(32,4,4) and init[1].shape==(32,4,4),'INPUT_SHAPE')
 return init

def validate_inputs(r77,r64,pm):
 old=set(r77.CFG['evaluation_seeds'])|set(r77.CFG['train_initializations'])|set(r77.CFG['training_episode_seeds'])
 old.update(r64.EPS);old.update(r64.EVAL_SEEDS)
 for k in range(70,78):
  pr=pm.read_json(f'FROZEN_PROTOCOL_R{k}.json')
  for s in ('evaluation_seeds','training_initializations','training_seeds','training_episode_seeds'):
   old.update(pr.get(s,[]))
 need(len(P['fresh_evaluation_seeds'])==2 and not set(P['fresh_evaluation_seeds'])&old,'FRESH_SEED_COLLISION')
 need(P['selected_indices']==[0,4,8,12] and P['cases']==['weak','anti'] and P['ratio']==160,'FIXTURE_PROTOCOL')
 need(P['records']==4*6*4 and P['terms_per_record']==20,'BOUNDS_PROTOCOL')
 return True

def collect(r78,r77,r64,r62,pm,r30,source):
 import torch as t
 from r83 import geometry as r83g
 from r88 import geometry as r88g
 validate_inputs(r77,r64,pm)
 groups,signatures=declared_graphs(pm,r77)
 models=r78.models_from_r77(source);keys=sorted(models);need(len(keys)==6,'FROZEN_MODEL_COUNT')
 W=t.stack([models[k].gates.weight.detach() for k in keys]);B=t.stack([models[k].gates.bias.detach() for k in keys]);n=len(keys)
 start=input_state(r64,t);nodes=start[0].shape[0];r61=pm.read_json('FROZEN_PROTOCOL_R61.json');out=[]
 teacher=r30.NonlinearTeacher(t)
 with t.no_grad():
  for graph,incidence in groups.items():
   adj=pm.graph_adjs(incidence,r61)[1]
   edges=r83g.edge_table(adj);theta=r83g.link_angles(adj,t)
   need([[int(e[0]),int(e[1])] for e in edges]==P['reciprocal_link_pairs'][graph],'PINNED_LINK_LAYOUT')
   teacher_seq=[tuple(v.clone() for v in start)]
   for k in range(1,21):teacher_seq.append(r62.graph_step_split(teacher_seq[-1],adj,adj,teacher,t))
   # Hybrid slot k uses teacher state after step k; independently continues with frozen HCA.
   # 21 slots × 6 models × 32 independent samples. No cross-slot state mixing.
   law=r62._StackedGates(W.repeat(21,1,1),B.repeat(21,1),r30,t,nodes)
   batch=21*n*nodes
   work=tuple(v.repeat(21*n,1,1) for v in start)
   for k in range(1,21):
    work=r62.graph_step_split(work,adj,adj,law,t)
    changed=[]
    for ch in (0,1):
     mat=work[ch].reshape(21,n,nodes,4,4).clone()
     mat[k]=teacher_seq[k][ch].unsqueeze(0).expand(n,-1,-1,-1)
     changed.append(mat.reshape(batch,4,4))
    work=tuple(changed)
   finished=[v.reshape(21,n,nodes,4,4) for v in work]
   for mi,model_key in enumerate(keys):
    initseed,arm=model_key.split('/')
    for index in P['selected_indices']:
     case=P['cases'][index//8];seed=P['fresh_evaluation_seeds'][(index%8)//4]
     hybrids=t.stack([t.cat((finished[0][k,mi,index].reshape(16),finished[1][k,mi,index].reshape(16))) for k in range(21)])
     metrics=r88g.extract(hybrids,theta,edges,1.,t)
     need(metrics['telescoping_max_abs']<P['numeric_thresholds']['identity'],'TELESCOPING')
     need(metrics['early_late_max_abs']<P['numeric_thresholds']['projection'],'EARLY_LATE')
     out.append({'graph':graph,'graph_signature_sha256':signatures[graph],'model_initialization':int(initseed),'arm':arm,'eval_seed':seed,'case':case,'metric_link_scale':1.,**metrics})
   print(f'[R89] graph={graph} records={len(out)}/{P["records"]}',flush=True)
 need(len(out)==P['records'],'ROW_COUNT')
 return out,signatures

def summarize(rows):
 by={}
 for graph in P['graphs']:
  rr=[r for r in rows if r['graph']==graph];need(len(rr)==24,'GRAPH_COUNT')
  early=sum(r['max_forcing_step']<=8 for r in rr)
  by[graph]={'records':len(rr),'early_peak_count':early,'early_peak_fraction':early/len(rr),
             'mean_net_to_gross_ratio':statistics.mean(r['net_to_gross_ratio'] for r in rr),
             'mean_total_norm':statistics.mean(r['total_norm'] for r in rr),
             'mean_early_norm':statistics.mean(r['early_norm'] for r in rr),
             'mean_late_norm':statistics.mean(r['late_norm'] for r in rr)}
 pooled=statistics.mean(r['net_to_gross_ratio'] for r in rows)
 passed=(all(by[g]['early_peak_fraction']>=P['preregistered_hypothesis']['minimum_early_peak_fraction_per_graph'] for g in P['graphs']) and pooled<=P['preregistered_hypothesis']['maximum_pooled_net_to_gross_ratio'])
 return {'by_graph':by,'pooled_net_to_gross_ratio':pooled,'early_peak_all':sum(r['max_forcing_step']<=8 for r in rows),
    'hypothesis_result':'BOUNDED_SYNTHETIC_TRANSFER_PASS' if passed else 'TRANSFER_HYPOTHESIS_NONQUAL',
    'disposition':'OFFLINE_FROZEN_TEACHER_FORCING_TRANSFER_DIAGNOSTIC_NO_AUTONOMOUS_UTILITY'}
