"""R123 sealed prospective K4 causal-shadow evaluation, research-only.

No calibration search, no training, no model or NAND mutation at inference.
Preregistration must be *remotely published* before --registered may execute.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import engine as E118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
from research.yang_mills_hca.r119_numerical_hotpath import guards
from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine
from research.yang_mills_hca.r122_energy_direction_forensics import controls as C
from research.yang_mills_hca.r122_energy_direction_forensics import harness as H
from research.yang_mills_hca.r122_energy_direction_forensics import linear as L
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M
from research.yang_mills_hca.r122_energy_direction_forensics import topology as TOP

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTO = HERE / 'PROTOCOL_R123.json'
EXPECTED_BASE = '501be02d33201ce9c95e211d6ba9a1b531656f43'
BRANCH = 'research/ym-r123-k4-quarter-prospective-r1'
PREREG_R0 = 'd52ccc20c56bd67ef4770ca020145900f834a9cf'
R93 = 'ENERGY_NORMALIZED_DENOMINATOR_HOLD'


def need(condition, error):
    if not condition:
        raise RuntimeError(error)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args], text=True, stderr=subprocess.PIPE).strip()


def original_pins():
    need(P.verify_manifest()['files_verified']==39,'FROZEN_MANIFEST')
    need(P.component_digest()=='770411b1f6788f9593f5f7adf1d4f6efb57a564f8c207d13dd7f55352af0df71','R118_SOURCE')
    oracle = get_oracle()
    need(oracle.parameter_pins()==P.CHECKPOINT_SHA256,'CHECKPOINT_PIN_DRIFT')
    need(C.synthetic_classifier_controls()['pass'], 'DIRECTION_DETECTOR_BROKEN')
    return oracle


def engine(S, ck, frac, tgt, mode='EXACT_GUARDED'):
    """Only a private float64 engine bias copy changes. W and archived weights never change."""
    eng = R119Engine(S.st, ck, controller='NATIVE', mode=mode, budgets=S.budgets)
    need(eng.weights_sha256==P.CHECKPOINT_SHA256[ck], 'CHECKPOINT_SOURCE_DRIFT:'+ck)
    orig = eng.b
    if frac:
        gate0 = 1/(1+math.exp(-float(orig[4])))
        gate = gate0 + frac*(tgt-gate0)
        need(math.isfinite(gate) and 0<gate<1,'INVALID_CALIBRATED_GATE')
        b = np.array(orig, copy=True, dtype=np.float64, order='C')
        b[4] = math.log(gate/(1-gate))
        b.flags.writeable=False
        need(not np.shares_memory(orig,b),'SHADOW_BIAS_ALIAS')
        eng.b = b
        eng._w_ptr = (eng.W.ctypes.data, eng.b.ctypes.data)
        if mode=='EXACT_GUARDED':
            eng.theta = guards.gate_theta(eng.W.tolist(),eng.b.tolist())
        need(abs(1/(1+math.exp(-float(b[4])))-gate)<1e-14,'SHADOW_GATE_RECOVERY')
        need(np.array_equal(orig,E118.load_weights(ck)[1]),'ARCHIVED_BIAS_MUTATION')
    return eng


def simulate(S, X0, eng):
    S.load(X0)
    out = np.empty((21,)+X0.shape, dtype=np.float64)
    out[0] = X0
    for h in range(1,21):
        r = eng.step(S.h)
        need(r.exact and r.disposition=='EXECUTED','ENGINE_REFUSED:'+str(r.disposition))
        out[h] = S.dump()
    return out


def state_nse(S, prediction, target):
    """True full fixture state across all 8 paired/unpaired copies, not a scalar readout."""
    p=np.asarray(prediction[S.fidx], dtype=np.float64).ravel()
    t=np.asarray(target[S.fidx], dtype=np.float64).ravel()
    den=float(np.dot(t,t))
    need(den>0 and math.isfinite(den), 'FULL_STATE_ZERO_TARGET')
    value=float(np.dot(p-t,p-t)/den)
    need(math.isfinite(value), 'NONFINITE_FULL_STATE_ERROR')
    return value


def paired_nse(S, prediction, target):
    p=S.view(prediction,'PAIRED_STATE'); t=S.view(target,'PAIRED_STATE')
    d=M.decompose(p,t)['erel']
    valid=d[np.isfinite(d)]
    return float(np.mean(valid)) if valid.size else None


def readout_nse(S, prediction, target):
    p=S.view(prediction,'READOUT'); t=S.view(target,'READOUT')
    d=M.decompose(p,t)
    valid=d['erel'][np.isfinite(d['erel'])]
    return (float(np.mean(valid)) if valid.size else None,
            int((d['norm_class']!=M.N_ORD).sum()))


def preflight():
    need(git('rev-parse','HEAD')==EXPECTED_BASE,'PREFLIGHT_BASE_DRIFT')
    o=original_pins()
    r121=json.loads((HERE.parent/'r121_temporal_topology/PROTOCOL_R121.json').read_text())
    graph=r121['graphs']['R121_HOLDOUT_2']
    gi=sorted(r121['graphs']).index('R121_HOLDOUT_2')
    systems,_=H.build_systems(o,graph,r121['structure_seed']+gi,ks=(0,))
    S=systems[0]
    a,b=o.r64.pair_state(121101,'weak',160,batch=4)
    X0=S.x0(a.numpy(),b.numpy(),scale=1.0)
    t=L.teacher_limits()[4]
    assert 0<t<1
    for ck in ('54007/STATE8','54007/STATE8_SIGNED8'):
        for frac in (0.0,0.25,0.5):
            a=simulate(S,X0,engine(S,ck,frac,t))
            b=simulate(S,X0,engine(S,ck,frac,t,mode='EXACT_R118'))
            need(np.array_equal(a.view(np.uint64),b.view(np.uint64)), 'PREFLIGHT_EXACT_PARITY')
    need(o.parameter_pins()==P.CHECKPOINT_SHA256,'POST_PREFLIGHT_FROZEN_DRIFT')
    print('PREFLIGHT=PASS CONSUMED_GRAPH=R121_HOLDOUT_2 PARITY=6/6 BITWISE',flush=True)


def prereg_guard(proto):
    need(proto['status']=='FROZEN_BEFORE_EVALUATION','PROTOCOL_STATUS')
    need(proto['base_commit']==EXPECTED_BASE,'BASE_MISMATCH')
    need(proto['arms']==[0.0,0.25,0.5] and proto['primary_fraction']==0.25, 'FRACTION_DRIFT')
    need(proto['r93']==R93,'R93_DRIFT')
    need(proto['source_pins']['qualify.py']==sha(__file__),'MEASUREMENT_SOURCE_DRIFT')
    need(proto['source_pins']['make_protocol.py']==sha(HERE/'make_protocol.py'),'GENERATOR_DRIFT')
    need(proto['source_pins']['r118_component_digest']==P.component_digest(),'R118_COMPONENT_DRIFT')
    for field,rel in {
        'r119_kernel':'r119_numerical_hotpath/csrc/awr_kernel_r119.c',
        'r119_engine':'r119_numerical_hotpath/engine_r119.py',
        'r119_guard':'r119_numerical_hotpath/guards.py',
        'r122_harness':'r122_energy_direction_forensics/harness.py',
        'r122_linear':'r122_energy_direction_forensics/linear.py',
        'r122_metrics':'r122_energy_direction_forensics/metrics.py',
        'r122_topology':'r122_energy_direction_forensics/topology.py',
        'r122_result':'r122_energy_direction_forensics/evidence/R122_RESULT.json',
        'r121_result':'r121_temporal_topology/evidence/R121_RESULT.json',
    }.items():
        need(proto['source_pins'][field]==sha(HERE.parent/rel),'SOURCE_DRIFT:'+field)
    need(proto['frozen_manifest']==P.MANIFEST_SHA256 and P.verify_manifest()['files_verified']==39,'MANIFEST_DRIFT')
    need(proto['checkpoint_pins']==P.CHECKPOINT_SHA256,'CHECKPOINT_PINS_DRIFT')
    need(proto['teacher_k4_limit']==float(L.teacher_limits()[4]),'ORACLE_LIMIT_DRIFT')
    from .make_protocol import candidate_graphs
    need(proto['graphs']==candidate_graphs(),'REGISTERED_GRAPH_REGEN_DRIFT')
    historical=set(TOP.historical_classes())
    r122=json.loads((HERE.parent/'r122_energy_direction_forensics/PROTOCOL_R122.json').read_text())
    historical|={TOP.signature(g) for g in r122['graphs'].values()}
    got=[TOP.signature(g) for g in proto['graphs'].values()]
    need(len(set(got))==4 and not historical.intersection(got),'GRAPH_LEAK')
    need(set(proto['graphs']) == {'R123_H1_A','R123_H2_A','R123_H1_B','R123_H2_B'}
         and proto['required_hops'] == [1,2,1,2]
         and [TOP.hop_distance(proto['graphs'][name]) for name in
              ('R123_H1_A','R123_H2_A','R123_H1_B','R123_H2_B')] == proto['required_hops'],
         'HOP_DRIFT')
    from research.yang_mills_hca.r122_energy_direction_forensics.qualify import consumed_seeds, pair_state_generator_seeds
    old=consumed_seeds() | {122101,122103,122107,122109,122300,122301,122302,122303,122304,122305,122400,122600,122900}
    ev=proto['evaluation_seeds']; st=proto['structure_seed']
    new=set(ev)|{st+i for i in range(4)}|{proto['graph_seed'],123900}
    need(len(new)==10 and not(old&new) and not(pair_state_generator_seeds(ev)&pair_state_generator_seeds(old)), 'SEED_COLLISION')
    need(git('rev-parse','--abbrev-ref','HEAD')==BRANCH,'WRONG_BRANCH')
    need(git('status','--porcelain')=='','REPO_NOT_CLEAN')
    local=git('rev-parse','HEAD')
    remote_url=os.environ.get('ELPIS_GITHUB_REMOTE','')
    need('github.com' in remote_url and 'abraxis717/Elpis_DEV' in remote_url,'GITHUB_REMOTE_NOT_VERIFIED')
    remote=git('ls-remote','--heads',remote_url,'refs/heads/'+BRANCH)
    need(bool(remote) and remote.split()[0]==local,'PREREG_NOT_PUBLISHED_OR_REMOTE_CHANGED')
    need(git('rev-parse','HEAD^')==PREREG_R0 and git('rev-parse','HEAD^^')==EXPECTED_BASE,
         'PREREG_ANCESTRY_DRIFT')
    need(not (HERE/'evidence/R123_RESULT.json').exists(),'PROSPECTIVE_ALREADY_CONSUMED')
    return local


def evaluate(out):
    start=time.monotonic()
    proto=json.loads(PROTO.read_bytes())
    prereg_commit=prereg_guard(proto)  # must pass BEFORE generating any fresh state
    need(C.synthetic_classifier_controls()['pass'],'CLASSIFIER_CONTROLS')
    o=original_pins()
    budget=proto['budget']
    graph_names=list(proto['graphs'])
    ck_list=list(proto['checkpoints'])
    arms=tuple(proto['arms'])
    scales=tuple(proto['scales'].items())
    # Each experiment must execute all three arms from independent identical initial states.
    max_steps=len(graph_names)*len(proto['evaluation_seeds'])*len(proto['cases'])*len(proto['k_values'])*len(scales)*len(ck_list)*len(arms)*22+100
    need(max_steps<=budget['max_engine_steps'],'STEP_BUDGET_PRECHECK')
    cells=defaultdict(lambda: defaultdict(list))
    raw=[]
    hazard=defaultdict(int)
    extinguished=defaultdict(int)
    parity=0
    teacher_parity=0.0
    steps=0
    for gi,(graph_name,g) in enumerate(proto['graphs'].items()):
        print(f'R123_GRAPH_START={graph_name} ({gi+1}/{len(graph_names)})',flush=True)
        systems,_=H.build_systems(o,g,proto['structure_seed']+gi,ks=tuple(proto['k_values']))
        for si,seed in enumerate(proto['evaluation_seeds']):
            for case in proto['cases']:
                a,b=o.r64.pair_state(seed,case,proto['source_ratio'],batch=proto['batch'])
                for scale_name,scale in scales:
                    for k,S in sorted(systems.items()):
                        X0=S.x0(a.numpy(),b.numpy(),scale=scale)
                        T=S.teacher(o,X0)
                        if k==0 and scale_name=='1' and si==0 and case==proto['cases'][0]:
                            _,adj=o.graph_adjs(g)
                            with o.torch.no_grad():
                                independent=o.r64.rollout((a,b),adj,o.r30.NonlinearTeacher(o.torch),20,keep=(8,20))
                            for h in (8,20):
                                expected=o.r63.paired(independent[h][0],4).numpy()
                                teacher_parity=max(teacher_parity,float(np.abs(expected-S.view(T[h],'READOUT')).max()))
                        for ck in ck_list:
                            cohort='SIGNED' if H.is_signed(ck) else 'UNSIGNED'
                            tr={}; d={}; engines={}
                            for frac in arms:
                                eng=engine(S,ck,frac,proto['teacher_k4_limit'])
                                engines[frac]=eng
                                tr[frac]=simulate(S,X0,eng)
                                steps+=20
                                need(np.isfinite(tr[frac]).all(),'NONFINITE_AUTONOMOUS_STATE')
                                d[frac]=np.stack([M.decompose(S.view(tr[frac][h],'READOUT'),
                                                               S.view(T[h],'READOUT'))['dir_class']
                                                  for h in range(1,21)])
                                if gi==0 and si==0 and case==proto['cases'][0] and k in (0,6) and scale_name in ('1','2^-20') and ck in ('54007/STATE8','54007/STATE8_SIGNED8'):
                                    other=simulate(S,X0,engine(S,ck,frac,proto['teacher_k4_limit'],mode='EXACT_R118'))
                                    steps+=20
                                    need(np.array_equal(other.view(np.uint64),tr[frac].view(np.uint64)),
                                         'NUMERICAL_PARITY_FAIL:'+ck+':'+str(frac))
                                    parity+=1
                            for frac in arms:
                                induced=int(np.sum((d[frac]==M.D_REVERSED)&(d[0.0]==M.D_ALIGNED)))
                                extinction=int(np.sum((d[frac]==M.D_EXTINGUISHED)&(d[0.0]==M.D_ALIGNED)))
                                hazard[(scale_name,frac)]+=induced
                                extinguished[(scale_name,frac)]+=extinction
                                key=(graph_name,k,cohort,scale_name,frac)
                                r={'g':graph_name,'k':k,'ck':ck,'cohort':cohort,'scale':scale_name,
                                   'seed':seed,'case':case,'frac':frac,
                                   'h8':state_nse(S,tr[frac][8],T[8]),
                                   'h20':state_nse(S,tr[frac][20],T[20]),
                                   'paired_h20':paired_nse(S,tr[frac][20],T[20]),
                                   'induced':induced,'extinguished':extinction}
                                ro,invalid=readout_nse(S,tr[frac][20],T[20]);r['readout_h20']=ro;r['invalid_readout_h20']=invalid
                                for h in (8,20):
                                    S.load(T[h-1]); result=engines[frac].step(S.h)
                                    need(result.exact and result.disposition=='EXECUTED','TF_STEP_REFUSED')
                                    steps+=1
                                    r['tf_h'+str(h)]=state_nse(S,S.dump(),T[h])
                                raw.append(r)
                                for metric in ('h8','h20','tf_h8','tf_h20'):
                                    cells[key][metric].append(r[metric])
                                if ro is not None:cells[key]['readout_h20'].append(ro)
            print(f'R123_PROGRESS={graph_name} seed={seed} ({si+1}/{len(proto["evaluation_seeds"])})',flush=True)
            need(time.monotonic()-start<=budget['max_seconds'],'TIME_BUDGET_EXCEEDED')
            need(steps<=budget['max_engine_steps'],'STEP_BUDGET_EXCEEDED')
        print('R123_GRAPH_COMPLETE='+graph_name,flush=True)
    need(parity==24,'PARITY_SAMPLE_COUNT:'+str(parity))
    need(teacher_parity<=1e-12,'TEACHER_PARITY')
    need(len(raw)==len(graph_names)*len(proto['evaluation_seeds'])*len(proto['cases'])*len(proto['k_values'])*len(scales)*len(ck_list)*len(arms),'INCOMPLETE_CELLS')
    need(o.parameter_pins()==P.CHECKPOINT_SHA256,'CHECKPOINT_MUTATION')
    need(P.verify_manifest()['files_verified']==39,'MANIFEST_MUTATION')
    summaries=[];passed=True
    for frac in (0.25,0.5):
        arm_ok=True
        for cohort in ('SIGNED','UNSIGNED'):
            for k in proto['k_values']:
                block=[]
                for g in graph_names:
                    key_base=(g,k,cohort,'1',0.0)
                    key_shadow=(g,k,cohort,'1',frac)
                    x=cells[key_base]; y=cells[key_shadow]
                    need(len(x['h20']) in (16,32) and len(y['h20'])==len(x['h20']),'COHORT_COUNT_MISMATCH')
                    def mean(z,n):return float(np.mean(z[n]))
                    block.append({'graph':g,'base_h20':mean(x,'h20'),'shadow_h20':mean(y,'h20'),
                                  'base_h8':mean(x,'h8'),'shadow_h8':mean(y,'h8'),
                                  'base_tf20':mean(x,'tf_h20'),'shadow_tf20':mean(y,'tf_h20')})
                bn=float(np.mean([z['base_h20'] for z in block]))
                sn=float(np.mean([z['shadow_h20'] for z in block]))
                b8=float(np.mean([z['base_h8'] for z in block]))
                s8=float(np.mean([z['shadow_h8'] for z in block]))
                bt=float(np.mean([z['base_tf20'] for z in block]))
                st=float(np.mean([z['shadow_tf20'] for z in block]))
                gwin=sum(z['shadow_h20']<z['base_h20'] for z in block)
                worst=max(z['shadow_h20']-z['base_h20'] for z in block)
                im=(bn-sn)/bn if bn>0 else float('-inf')
                rule=proto['decision_rules']
                ok=(im>=rule['primary_cohort_k_min_relative_improvement'] and
                    gwin>=rule['primary_graphs_improving_min_per_cohort_k'] and
                    worst<=rule['primary_max_graph_cohort_k_abs_h20_harm'] and
                    s8<=b8*(1+rule['primary_h8_pooled_noninferiority_relative']) and
                    st<=bt*(1+rule['primary_teacher_forced_h20_pooled_noninferiority_relative']))
                arm_ok=arm_ok and ok
                summaries.append({'arm':frac,'cohort':cohort,'k':k,'improvement':im,'graph_wins':gwin,
                                  'worst_graph_abs_harm':worst,'h8_ratio':s8/b8,'tf_h20_ratio':st/bt,
                                  'pass':bool(ok),'graphs':block})
        induced=sum(hazard[(s,frac)] for s in proto['decision_rules']['in_range_scales'])
        arm_ok=arm_ok and induced==0
        print(f'R123_ARM={frac} IN_RANGE_INDUCED={induced} ABOVE_CLAMP_INDUCED={hazard[("2^2",frac)]} GATE={"PASS_BOUNDED" if arm_ok else "SCIENTIFIC_FAIL_BOUNDED"}',flush=True)
        if frac==0.25:passed=arm_ok
    result={'schema':'elpis.ym_hca.r123.result.v1','phase':'R123','prereg_commit':prereg_commit,
            'base_commit':EXPECTED_BASE,'protocol_sha256':sha(PROTO),'source_sha256':sha(__file__),
            'r93':R93,'disposition':'PASS_BOUNDED' if passed else 'SCIENTIFIC_FAIL_BOUNDED',
            'primary_fraction':0.25,'secondary_fraction':0.5,
            'summaries':summaries,'raw_records':raw,
            'directional_hazards':{f'{scale}/{frac}':hazard[(scale,frac)] for scale,_ in scales for frac in arms},
            'induced_extinguishments':{f'{scale}/{frac}':extinguished[(scale,frac)] for scale,_ in scales for frac in arms},
            'integrity':{'guarded_reference_parity':parity,'teacher_k0_parity_max_abs':teacher_parity,
                         'checkpoint_pins_unchanged':True,'frozen_manifest_unchanged':True,'seed_disjointness':True,
                         'topology_novelty':True,'complete':True},
            'resources':{'seconds':round(time.monotonic()-start,2),'engine_steps':steps,
                         'max_rss_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)},
            'limitations':proto['limitations']}
    need(steps<=budget['max_engine_steps'],'FINAL_STEP_BUDGET')
    need(result['resources']['seconds']<=budget['max_seconds'],'FINAL_TIME_BUDGET')
    need(result['resources']['max_rss_mib']<=budget['max_rss_mib'],'FINAL_MEMORY_BUDGET')
    blob=(json.dumps(result,indent=1,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(blob)<=budget['max_result_bytes'],'RESULT_SIZE_BUDGET')
    out.parent.mkdir(parents=True,exist_ok=True)
    need(not out.exists(),'RESULT_OUTPUT_ALREADY_EXISTS')
    out.write_bytes(blob)
    print('=== SEND BACK ===',flush=True)
    print('PHASE=R123_K4_PROSPECTIVE',flush=True)
    print('DISPOSITION='+result['disposition'],flush=True)
    print('BASE_COMMIT='+EXPECTED_BASE,flush=True)
    print('PREREG_COMMIT='+prereg_commit,flush=True)
    print('SOURCE_SHA256='+sha(__file__),flush=True)
    print('PROTOCOL_SHA256='+sha(PROTO),flush=True)
    print('RESULT_SHA256='+hashlib.sha256(blob).hexdigest(),flush=True)
    for frac in (0.25,0.5):
        print('K4_'+str(frac)+'_IN_RANGE_INDUCED='+str(sum(hazard[(s,frac)] for s in proto['decision_rules']['in_range_scales'])),flush=True)
        print('K4_'+str(frac)+'_ABOVE_CLAMP_INDUCED='+str(hazard[('2^2',frac)]),flush=True)
    print('PARITY='+str(parity)+'/24 BITWISE',flush=True)
    print('FROZEN=UNCHANGED',flush=True)
    print('R93='+R93,flush=True)
    print('RESULT_PATH='+str(out),flush=True)
    print('BLOCKER=NONE CHILD_EXIT=0',flush=True)


def main():
    pa=argparse.ArgumentParser();gp=pa.add_mutually_exclusive_group(required=True)
    gp.add_argument('--preflight',action='store_true');gp.add_argument('--registered',action='store_true')
    pa.add_argument('--out')
    ns=pa.parse_args()
    try:
        if ns.preflight:preflight()
        else:
            need(ns.out is not None,'OUTPUT_REQUIRED')
            output=Path(ns.out).resolve()
            need(output.is_relative_to('/dev/shm/'),'VOLATILE_OUTPUT_ONLY')
            evaluate(output)
    except BaseException as exc:
        traceback.print_exc()
        print('=== SEND BACK ===\nPHASE=R123_K4_PROSPECTIVE\nDISPOSITION=NONQUAL_NOT_COMPLETED\nBLOCKER='+type(exc).__name__+':'+str(exc)[:250]+'\nCHILD_EXIT=2',flush=True)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
