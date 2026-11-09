#!/usr/bin/env python3
"""Independent stdlib-only R78 evidence review. No torch, no inference or training.
Parent R77 metrics are used only to verify offline labels and exact fixture lineage.
"""
from __future__ import annotations
import copy,hashlib,itertools,json,math,pathlib,statistics,struct,sys,zipfile
ROOT=pathlib.Path(__file__).resolve().parent
P=json.loads((ROOT/'FROZEN_PROTOCOL_R78.json').read_bytes())
class Hold(Exception):pass
def need(b,e):
 if not b:raise Hold(e)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def approx(x,y,atol=1e-9,rtol=2e-9):
 return type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*max(abs(x),abs(y))
def match(a,b,where=''):
 if isinstance(a,dict):
  need(type(b) is dict and set(a)==set(b),'MAP:'+where)
  for k in a:match(a[k],b[k],where+'/'+k)
 elif isinstance(a,list):
  need(type(b) is list and len(a)==len(b),'LIST:'+where)
  for i,(x,y) in enumerate(zip(a,b)):match(x,y,where+'/'+str(i))
 elif type(a) in (int,float):need(approx(a,b),'NUMBER:'+where)
 else:need(a==b,'VALUE:'+where)
def report_of(name):
 with zipfile.ZipFile(ROOT/name) as z:
  need(z.testzip() is None and z.namelist()==['result.json'],'REPORT_ZIP:'+name)
  raw=z.read('result.json');o=json.loads(raw);need(canon(o)==raw,'REPORT_CANON:'+name)
 return o
def source_gate():
 for name,digest in P['source_sha256'].items():
  need((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==digest,'SOURCE_PIN:'+name)
 need(sha((ROOT/'YM_AWR_R77_RESULT.zip').read_bytes())==P['parent_r77_report_sha256'],'PARENT_R77_PIN')
 pins={}
 for line in (ROOT/'SOURCE_SHA256_ALL.txt').read_text().splitlines():
  digest,name=line.split('  ',1)
  need(name not in pins and len(digest)==64,'MANIFEST_SCHEMA');pins[name]=digest
 for name,h in pins.items():need((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==h,'SOURCE_MANIFEST:'+name)
 need('FROZEN_PROTOCOL_R78.json' in pins and 'r78.py' in pins and 'YM_AWR_R78_RESULT.zip' in pins,'PHASE_PIN_MISSING')
 need(set(P['source_sha256']).issubset(pins),'PARENT_PIN_NOT_IN_MANIFEST')
 return pins

def rank(values):
 vs=sorted((v,i) for i,v in enumerate(values));out=[0.]*len(values);i=0
 while i<len(vs):
  j=i+1
  while j<len(vs) and vs[i][0]==vs[j][0]:j+=1
  for k in range(i,j):out[vs[k][1]]=(i+j-1)/2
  i=j
 return out
def spearman(x,y):
 a=rank(x);b=rank(y);ma=statistics.mean(a);mb=statistics.mean(b)
 num=sum((u-ma)*(v-mb) for u,v in zip(a,b))
 den=math.sqrt(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))
 return num/den if den else 0.
def summary(rows):
 out={}
 for graph in P['graphs']:
  subset=[r for r in rows if r['graph']==graph];need(len(subset)==72,'GROUP_COUNT')
  for ch in P['channels']:
   for window in ['8_12','8_20','12_20']:
    tangent=[r['tangents'][f'{ch}/1e-05/{window}']['log_gain'] for r in subset]
    actual=[math.log((r['nse']['20']+1e-12)/(r['nse']['12']+1e-12)) for r in subset]
    out[f'{graph}/{ch}/{window}']={'spearman_vs_log_h20_h12_error':spearman(tangent,actual),'mean_log_gain':statistics.mean(tangent),'median_log_gain':statistics.median(tangent),'n':len(subset)}
 selected={g:max(out[f'{g}/{ch}/8_20']['spearman_vs_log_h20_h12_error'] for ch in P['channels']) for g in P['graphs']}
 good=sum(x>P['rank_correlation_threshold'] for x in selected.values())
 return {'metrics':out,'best_channel_per_graph_spearman':selected,'graphs_above_predeclared_correlation':good,'scientific_gate':'RETROSPECTIVE_TANGENT_ASSOCIATION_BOUNDED' if good>=P['positive_graph_requirement'] else 'RETROSPECTIVE_TANGENT_ASSOCIATION_NONQUAL','disposition':'RETROSPECTIVE_FROZEN_HCA_FINITE_HORIZON_TANGENT_DIAGNOSTIC_NO_AUTONOMOUS_CORRECTION'}
def validate(o):
 need(set(o)=={'phase','protocol_sha256','parent_r77_report_sha256','rows','numerics','summary','limitations'},'REPORT_SCHEMA')
 need(o['phase']==P['phase'] and o['protocol_sha256']==sha((ROOT/'FROZEN_PROTOCOL_R78.json').read_bytes()),'PROTOCOL_PIN')
 need(o['parent_r77_report_sha256']==P['parent_r77_report_sha256'] and o['limitations']==P['limitations'],'BOUNDARY_PIN')
 parent=report_of('YM_AWR_R77_RESULT.zip')
 ref={(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']):r for r in parent['rows']}
 need(len(ref)==576 and len(parent['training'])==6 and parent['phase']=='YM_AWR_R77_88P_OFFLINE_ONE_STEP_CONSISTENCY_R0','R77_INHERITED')
 for k,m in parent['training'].items():
  vals=[v for row in m['weights'] for v in row]+m['bias']
  need(len(vals)==88 and sha(struct.pack('<88d',*vals))==m['checkpoint_sha256'],'R77_CHECKPOINT_HASH:'+k)
 rows=o['rows'];need(type(rows) is list and len(rows)==P['records'],'ROW_COUNT')
 expected=set(itertools.product(P['initialization_seeds'],P['arms'],P['graphs'],P['evaluation_seeds'],P['cases']))
 seen=set();max_eps=0.
 known_tangents={f'{ch}/{ep:.0e}/{start}_{end}' for ch in P['channels'] for ep in P['epsilons'] for start,end in P['time_windows']}
 for r in rows:
  need(set(r)=={'initialization','arm','graph','case','eval_seed','nse','tf_local_nse','tangents','max_eps_log_delta'},'ROW_SCHEMA')
  k=(r['initialization'],r['arm'],r['graph'],r['eval_seed'],r['case']);need(k in expected and k not in seen,'ROW_ID');seen.add(k)
  match(r['nse'],{str(h):ref[k]['h'][str(h)]['mean_nse'] for h in P['horizons']},'R77_HORIZONS')
  need(approx(r['tf_local_nse'],ref[k]['tf_local_h20']['mean_nse']),'R77_TEACHER_LOCAL_PIN')
  tans=r['tangents'];need(set(tans)==known_tangents,'TANGENT_SET')
  for key,val in tans.items():
   need(set(val)=={'input_norm','derivative_norm','log_gain'},'TANGENT_SCHEMA')
   a=val['input_norm'];b=val['derivative_norm'];need(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and a>0 and b>0,'POSITIVE_NORMS')
   need(approx(math.log(b/a),val['log_gain'],1e-10,1e-10),'LOG_GAIN_RECONSTRUCT')
  mismatch=0.
  for ch in P['channels']:
   for start,end in P['time_windows']:
    a=tans[f'{ch}/{P["epsilons"][0]:.0e}/{start}_{end}'];b=tans[f'{ch}/{P["epsilons"][1]:.0e}/{start}_{end}']
    need(approx(a['input_norm'],b['input_norm'],1e-14,1e-14),'EPS_START_NORM')
    mismatch=max(mismatch,abs(a['log_gain']-b['log_gain']))
   for ep in P['epsilons']:
    need(approx(tans[f'{ch}/{ep:.0e}/8_12']['input_norm'],tans[f'{ch}/{ep:.0e}/8_20']['input_norm'],1e-14,1e-14),'H8_NORM_DRIFT')
  need(mismatch<=P['numerical_agreement_log_absolute_tolerance'] and approx(mismatch,r['max_eps_log_delta'],1e-13,1e-12),'EPS_CONVERGENCE')
  max_eps=max(max_eps,mismatch)
 need(seen==expected,'EXPECTED_ROW_COVERAGE')
 need(set(o['numerics'])=={'max_epsilon_log_disagreement','max_r77_native_parity','max_gauge_generator_identity_error'},'NUMERICS_SCHEMA')
 need(approx(max_eps,o['numerics']['max_epsilon_log_disagreement'],1e-12,1e-11),'EPS_MAX_REVIEW')
 need(0<=o['numerics']['max_r77_native_parity']<P['native_r77_parity_atol'],'R77_NATIVE_PARITY_REPORT')
 need(0<=o['numerics']['max_gauge_generator_identity_error']<P['simultaneous_A_B_gauge_tangent_identity_max_abs'],'GAUGE_IDENTITY_REPORT')
 match(summary(rows),o['summary'],'SUMMARY_REBUILD')
 return True

def mutations(o):
 mods=[
  lambda x:x.update(protocol_sha256='0'*64),
  lambda x:x.update(parent_r77_report_sha256='0'*64),
  lambda x:x['rows'].pop(),
  lambda x:x['rows'][0].update(eval_seed=999999),
  lambda x:x['rows'][0]['nse'].update({'20':x['rows'][0]['nse']['20']+.2}),
  lambda x:x['rows'][1]['tangents']['A/1e-05/8_20'].update(derivative_norm=999.),
  lambda x:x['rows'][2]['tangents']['B/5e-06/12_20'].update(log_gain=1e3),
  lambda x:x['rows'][3]['tangents']['A/1e-05/8_12'].update(input_norm=0.),
  lambda x:x['rows'][4].update(max_eps_log_delta=4.),
  lambda x:x['numerics'].update(max_epsilon_log_disagreement=999.),
  lambda x:x['summary'].update(graphs_above_predeclared_correlation=4),
  lambda x:x['limitations'].update(autonomous_correction='QUALIFIED'),
 ]
 rejects=0
 for mutate in mods:
  altered=copy.deepcopy(o);mutate(altered)
  try:validate(altered)
  except (Hold,TypeError,ValueError,KeyError):rejects+=1
  else:raise Hold('MUTATION_NOT_REJECTED')
 need(rejects==12,'ADVERSARIAL_TEST_COUNT')
 return rejects
if __name__=='__main__':
 try:
  source_gate();o=report_of('YM_AWR_R78_RESULT.zip');validate(o)
  print('R78_REPORT_REVIEW=PASS PARENT_R77_CHECKPOINTS=6/6 SOURCE_PINS=PASS',flush=True)
  print(f'R78_RECORDS={len(o["rows"])}/288 TANGENT_INTERVAL_METRICS={len(o["rows"])*12}/3456',flush=True)
  print(f'R78_MAX_GAUGE_GENERATOR_ERROR={o["numerics"]["max_gauge_generator_identity_error"]:.3e}',flush=True)
  print('R78_BY_GRAPH='+json.dumps(o['summary']['best_channel_per_graph_spearman'],sort_keys=True),flush=True)
  print('ADVERSARIAL_REJECTIONS='+str(mutations(o))+'/12 SCIENTIFIC_GATE='+o['summary']['scientific_gate'],flush=True)
 except BaseException as exc:
  print('R78_REVIEW_HOLD='+type(exc).__name__+':'+str(exc),flush=True);sys.exit(2)
