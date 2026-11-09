from pathlib import Path
import json, random, itertools, collections, hashlib
D=Path(__file__).resolve().parent
FILES=['portable_mechanics.py','r30_source.py','r62_source.py','r63_source.py','r64.py','FROZEN_PROTOCOL_R61.json','FROZEN_PROTOCOL_R62.json','FROZEN_PROTOCOL_R63.json','FROZEN_PROTOCOL_R64.json']+[f'FROZEN_PROTOCOL_R{i}.json' for i in range(70,77)]
def iso(g):
 m=collections.Counter(sum(1<<i for i,row in enumerate(g) if v in row) for v in set.union(*(set(x) for x in g)))
 return min(tuple(m[sum(1<<i for i,j in enumerate(p) if (k>>j)&1)] for k in range(1,16)) for p in itertools.permutations(range(4)))
def connected(g):
 reached={0};p=set(g[0])
 for _ in range(4):
  for i,r in enumerate(g):
   if p.intersection(r):p.update(r);reached.add(i)
 return len(reached)==4
def generate():
 previous=[json.loads((D/f'FROZEN_PROTOCOL_R{i}.json').read_bytes()) for i in [61,*range(70,77)]]
 prev_graphs=[*previous[0]['r55_structural']['graphs'].values()]
 for d,key in zip(previous[1:],['graphs','graphs','evaluation_graphs','evaluation_graphs','eval_graphs','graphs','graphs']):prev_graphs.extend(d[key].values())
 old={iso(g) for g in prev_graphs};rng=random.Random(770077);out={}
 for name,deg in [('R77_OOD_D2',2),('R77_OOD_A3',3),('R77_OOD_B3',3),('R77_OOD_D4',4)]:
  for _ in range(300000):
   g=[sorted(rng.sample(range(10),4)) for _ in range(4)]
   if len({tuple(row) for row in g})!=4 or len({v for row in g for v in row})!=10 or not connected(g):continue
   if max(collections.Counter(itertools.chain.from_iterable(g)).values())!=deg:continue
   s=iso(g)
   if s in old:continue
   out[name]=g;old.add(s);break
  else:raise RuntimeError('EXHAUSTED:'+name)
 return out
if __name__=='__main__':
 out=generate();print('GRAPH_PROTOCOL_FIXED',out,flush=True)
 source={f:hashlib.sha256((D/f).read_bytes()).hexdigest() for f in FILES}
 protocol={
 'phase':'YM_AWR_R77_88P_OFFLINE_ONE_STEP_CONSISTENCY_R0','version':1,
 'purpose':'Prospective 88-param HCA offline teacher-state one-step consistency at H4->H5 and H7->H8, compared to matched state-only and signed-state-only objectives. No teacher online, no inference-time resets or gain controllers.',
 'parent_r76_archive_sha256':'11fadd1213e8f52425f0f92d80481ad49682bde0ab7b7c14a5e3c452732ba00d',
 'parent_r76_ouroboros':'INDEPENDENT_R76_EVIDENCE_REVIEW_PASS / SYNTHETIC_STATE_GAIN_NONQUAL',
 'source_sha256':source,'graphs':out,'graph_generation_seed':770077,'graph_schedule':[['R77_OOD_D2',2],['R77_OOD_A3',3],['R77_OOD_B3',3],['R77_OOD_D4',4]],
 'arms':['STATE8','STATE8_SIGNED8','STATE8_TF1'],
 'train_initializations':[54007,54031],'episode_order_seed':55001,
 'training_episode_seeds':[52003,52009,52021,52033,52039,52051,52057,52069],
 'train_graphs':['D1_P1','D2_P1'],'train_cases':['weak','reverse','anti'],'train_ratio':64,
 'train_updates_per_model':48,'models':6,'train_horizon':8,
 'loss_weights':{'state_h4':.25,'state_h8':1.,'signed_h4':.0125,'signed_h8':.05,'signed_norm_floor':.1,'tf_h4_h5':.1,'tf_h7_h8':.1},
 'training_optimizer':{'type':'Adam','lr':.012,'weight_decay':0.,'clip_norm':1.},
 'initialization':'Seeded Gaussian weights sigma=.01, biases sigma=.002 added to frozen R64 initialization; matched initializations/order across three arms.',
 'evaluation_seeds':[53003,53011,53017,53027,53039,53047,53053,53069],
 'evaluation_cases':['weak','reverse','anti'],'evaluation_ratio':160,'batch_per_seed':4,'evaluation_transport':'PARTICIPANT_NONFLAT','evaluation_horizons':[8,12,20],
 'target_records':576,
 'predeclared_success':{'relative_pooled_H20_nse_vs_state_and_signed_at_most':.95,'min_graphs_better_than_state':3,'max_graph_absolute_harm_vs_state':.05,'same_init_paired_comparison':True},
 'limitations':{'R52':'NOT_REPRODUCED','Spark_GDN':'NOT_EXECUTED','CGC':'SHADOW_ONLY','HACF':'NOT_PRODUCTION_ATTESTED','autonomous_correction':'NOT_QUALIFIED','teacher_state_online':False,'production_utility':False,'data_class':'SYNTHETIC'},
 'runtime':'CPU torch.float64, offline training in volatile RAM only, inference frozen, no host project files modified, no persistent memory state.'}
 (D/'FROZEN_PROTOCOL_R77.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
 print('PROTOCOL_SHA256',hashlib.sha256((D/'FROZEN_PROTOCOL_R77.json').read_bytes()).hexdigest(),flush=True)
