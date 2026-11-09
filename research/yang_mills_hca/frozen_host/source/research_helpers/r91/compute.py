"""R91: path-independent, OFFLINE teacher-state local transition defects.

At each teacher state T[h-1], d[h] = M(T[h-1]) - T(T[h-1]).
This is an intrinsic teacher-path comparison, not a hybrid-order attribution,
not a propagated H20 forcing contribution and not an inference observable.
"""
from __future__ import annotations
import hashlib,json,pathlib,statistics,math
HERE=pathlib.Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(x,m):
 if not x:raise RuntimeError(m)
def key(r):return (r['graph'],r['model_initialization'],r['arm'],r['eval_seed'],r['case'])

def defect_norm(before, model_after, teacher_after, edges, t):
 """Squared metric quotient norm; local generators act on joint state+links.
 The link increment is zero, but the full generator (with link components)
 is used when removing local gauge-orbit directions.
 """
 from r83 import geometry as g
 a=before[:16].reshape(1,4,4);b=before[16:].reshape(1,4,4)
 gen=g.generators((a,b),edges,1.,t)[0]
 delta=model_after-teacher_after
 v=t.cat((delta,t.zeros(gen.shape[0]-32,dtype=t.float64)))
 gram=gen.T@gen;gv=gen.T@v
 qsq=(v@v-gv@t.linalg.solve(gram,gv)).clamp_min(0)
 return float(t.sqrt(qsq)),float(t.linalg.vector_norm(delta))

def replay(r78,r77,r64,r62,pm,r30,source,parents89,parents90):
 import torch as t
 from r83 import geometry as r83g
 from r89 import compute as r89c
 groups,signatures=r89c.declared_graphs(pm,r77)
 r89c.validate_inputs(r77,r64,pm)
 models=r78.models_from_r77(source);keys=sorted(models)
 need(len(keys)==6,'MODEL_COUNT')
 W=t.stack([models[k].gates.weight.detach() for k in keys]);B=t.stack([models[k].gates.bias.detach() for k in keys]);n=len(keys)
 start=r89c.input_state(r64,t);nodes=start[0].shape[0];r61=pm.read_json('FROZEN_PROTOCOL_R61.json')
 output=[]
 with t.no_grad():
  for graph,incidence in groups.items():
   adj=pm.graph_adjs(incidence,r61)[1];edges=r83g.edge_table(adj)
   need([[int(e[0]),int(e[1])] for e in edges]==r89c.P['reciprocal_link_pairs'][graph],'LINK_SIGNATURE')
   teacher=r30.NonlinearTeacher(t);teacher_states=[tuple(v.clone() for v in start)]
   for h in range(20):teacher_states.append(r62.graph_step_split(teacher_states[-1],adj,adj,teacher,t))
   law=r62._StackedGates(W,B,r30,t,nodes)
   # compute from same teacher-state input, not its own recurrent prefix
   one_steps=[]
   for h in range(20):
    inp=tuple(v.repeat(n,1,1) for v in teacher_states[h])
    nxt=r62.graph_step_split(inp,adj,adj,law,t)
    one_steps.append(tuple(v.reshape(n,nodes,4,4).clone() for v in nxt))
   for mi,name in enumerate(keys):
    init,arm=name.split('/')
    for ix in r89c.P['selected_indices']:
     case=r89c.P['cases'][ix//8];seed=r89c.P['fresh_evaluation_seeds'][(ix%8)//4]
     k=(graph,int(init),arm,seed,case)
     need(k in parents89 and k in parents90,'PARENT_CONTEXT')
     teacher_chain=[];model_steps=[];qnorms=[];rawnorms=[]
     for h in range(20):
      before=t.cat((teacher_states[h][0][ix].reshape(16),teacher_states[h][1][ix].reshape(16)))
      after=t.cat((teacher_states[h+1][0][ix].reshape(16),teacher_states[h+1][1][ix].reshape(16)))
      pred=t.cat((one_steps[h][0][mi,ix].reshape(16),one_steps[h][1][mi,ix].reshape(16)))
      q,r=defect_norm(before,pred,after,edges,t)
      if h==0:teacher_chain.append([float(x) for x in before])
      teacher_chain.append([float(x) for x in after]);model_steps.append([float(x) for x in pred]);qnorms.append(q);rawnorms.append(r)
     # Same terminal teacher as R89 and R90; unlike hybrids no per-step ordering.
     p89,p90=parents89[k],parents90[k]
     delta=lambda a,b:max(abs(x-y) for x,y in zip(a,b))
     need(delta(teacher_chain[-1],p89['hybrid_terminal_states'][-1])<P['thresholds']['endpoint'],'R89_TEACHER_PARITY')
     need(delta(teacher_chain[-1],p90['hybrid_terminal_states'][-1])<P['thresholds']['endpoint'],'R90_TEACHER_PARITY')
     peakh=qnorms.index(max(qnorms))+1
     output.append({'graph':graph,'graph_signature_sha256':signatures[graph], 'model_initialization':int(init),'arm':arm,
       'eval_seed':seed,'case':case,'metric_link_scale':1.,'teacher_states':teacher_chain,
       'model_teacher_one_steps':model_steps,'quotient_defect_norms':qnorms,'raw_defect_norms':rawnorms,
       'local_peak_horizon':peakh,'early_energy_fraction':sum(v*v for v in qnorms[:8])/max(sum(v*v for v in qnorms),1e-30),
       'terminal_quotient_error':p89['total_norm'], 'r89_peak_horizon':p89['max_forcing_step'],
       'r90_peak_horizon':p90['reverse_physical_peak_horizon']})
   print(f'[R91] graph={graph} contexts={len(output)}/{P["records"]}',flush=True)
 need(len(output)==P['records'],'RECORDS')
 return output,signatures

def summarize(rows):
 by={};early=0
 for g in P['graphs']:
  rs=[r for r in rows if r['graph']==g];need(len(rs)==24,'GRAPH_RECORDS')
  e=sum(r['local_peak_horizon']<=8 for r in rs);early+=e
  by[g]={'records':24,'local_early_peak':e,'r89_early_peak':sum(r['r89_peak_horizon']<=8 for r in rs),
    'r90_early_peak':sum(r['r90_peak_horizon']<=8 for r in rs),
    'mean_local_early_energy_fraction':statistics.mean(r['early_energy_fraction'] for r in rs),
    'mean_terminal_quotient_error':statistics.mean(r['terminal_quotient_error'] for r in rs)}
 good=all(by[g]['local_early_peak']>=P['hypothesis']['min_early_peak_each_graph'] for g in by)
 return {'by_graph':by,'local_early_peak_total':early,'hypothesis_result':'LOCAL_DEFECT_EARLY_CONCENTRATION_PASS' if good else 'LOCAL_DEFECT_EARLY_CONCENTRATION_NONQUAL',
    'disposition':'RETROSPECTIVE_TEACHER_STATE_LOCAL_TRANSITION_DEFECT_NO_PATH_ATTRIBUTION_OR_ONLINE_UTILITY'}
