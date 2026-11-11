"""R125: retrospective H1-H2 causal-prefix enumeration, validated raw preclamp census.

Only consumed R123/R124 fixtures. No new seeds or parameter search. Every causal
sequence executes through unmodified R119 EXACT_GUARDED with private RAM biases.
A separate diagnostic clone of the pinned C kernel measures raw preclamp norms and
must match the original numerical step bitwise.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
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
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
from research.yang_mills_hca.r122_energy_direction_forensics import harness as H
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M
from research.yang_mills_hca.r123_k4_prospective import qualify as R123
from research.yang_mills_hca.r124_stress_reversal_forensics import qualify as R124
from . import instrument as I

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
YM=HERE.parent
PROTO=HERE/'PROTOCOL_R125.json'
R124_RESULT=YM/'r124_stress_reversal_forensics/evidence/R124_RESULT.json'
R123_PROTO=YM/'r123_k4_prospective/PROTOCOL_R123.json'
R123_RESULT=YM/'r123_k4_prospective/evidence/R123_RESULT.json'
BASE='7e8536ab2deafae6aabb423f3639ba4d218ce357'
BRANCH='research/ym-r125-early-prefix-clamp-forensics-r1'
R0='1bac95bebf96843f98adb1200aa70fa0584df700'
R93='ENERGY_NORMALIZED_DENOMINATOR_HOLD'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(condition,why):
    if not condition:raise RuntimeError(why)
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,stderr=subprocess.PIPE).strip()


def source_guard(p):
    need(p['phase']=='R125' and p['mode']=='CONSUMED_R123_R124_RETROSPECTIVE','PROTOCOL_STATUS')
    need(p['base_commit']==BASE and p['r93']==R93,'BASE_R93_DRIFT')
    need(p['arms']==[0.0,0.25] and p['expected_events']==40,'POLICY_DRIFT')
    need(p['target_scale']=='2^2' and p['prefix_steps']==[1,2],'BOUNDARY_DRIFT')
    for path,digest in p['source_pins'].items():
        need(sha(HERE/path)==digest, 'SOURCE_DRIFT:'+path)
    for rel,digest in p['upstream_pins'].items():need(sha(YM/rel)==digest,'UPSTREAM_SOURCE_DRIFT:'+rel)
    need(sha(R124_RESULT)==p['r124_result_sha256'] and sha(R123_RESULT)==p['r123_result_sha256'],'EVIDENCE_DRIFT')
    need(sha(R123_PROTO)==p['r123_protocol_sha256'],'R123_PROTOCOL_DRIFT')
    need(P.component_digest()==p['r118_component_digest'],'R118_DIGEST_DRIFT')
    need(P.verify_manifest()['files_verified']==39 and P.CHECKPOINT_SHA256==p['checkpoint_pins'],'FROZEN_DRIFT')
    need((M.D_ALIGNED,M.D_REVERSED)==(0,3),'DIRECTION_CLASSIFIER_DRIFT')
    # Instrument source verification is a no-model static check.
    need(bool(I.instrument_source()),'INSTRUMENT_SOURCE')
    need(git('rev-parse','--abbrev-ref','HEAD')==BRANCH,'WRONG_BRANCH')
    need(git('status','--porcelain')=='','DIRTY_BEFORE_REPLAY')
    head=git('rev-parse','HEAD')
    need(git('rev-parse','HEAD^')==R0 and git('rev-parse','HEAD^^')==BASE,'PREREG_ANCESTRY_DRIFT')
    need(not (HERE/'evidence/R125_RESULT.json').exists(),'RESULT_EXISTS')
    remote=os.environ.get('ELPIS_GITHUB_REMOTE','')
    need('github.com' in remote and 'abraxis717/Elpis_DEV' in remote,'REMOTE_NOT_GITHUB')
    rr=git('ls-remote',remote,'refs/heads/'+BRANCH).split()
    need(rr and rr[0]==head,'PREREG_NOT_REMOTELY_PUBLISHED')
    return head


def execute(S,X0,ck,tgt,word,engines):
    # Fixed per-step N/K sequence; every step uses real R119 EXACT_GUARDED.
    S.load(X0)
    out=[X0.copy()]
    for ch in word:
        ret=engines[ch].step(S.h)
        need(ret.exact and ret.disposition=='EXECUTED','EXACT_KERNEL_REFUSAL')
        out.append(S.dump())
    return out


def viewpoint(S,vec,T,h,pair):
    d=M.decompose(S.view(vec,'READOUT'),S.view(T[h],'READOUT'))
    cl=int(d['dir_class'][pair]); cosine=d['c'][pair]
    return {'direction':M.DIR_CLASSES[cl], 'cosine':float(cosine) if np.isfinite(cosine) else None,
            'norm_ratio':float(d['r'][pair]) if np.isfinite(d['r'][pair]) else None,
            'readout_relerr':float(d['erel'][pair]) if np.isfinite(d['erel'][pair]) else None}


def candidate_words(h):
    # All H1/H2 assignments (N,K), fixed middle and current-step context N or K.
    # h=3 => exactly eight unique words; longer horizons => 16 max.
    words=set()
    for p1,p2,mid,last in itertools.product('NK','NK','NK','NK'):
        words.add(p1+p2+(mid*(h-3))+last)
    need(len(words)==(8 if h==3 else 16),'WORD_CENSUS_DRIFT')
    return sorted(words)


def evaluate(output_path):
    start=time.monotonic()
    p=json.loads(PROTO.read_bytes())
    prereg=source_guard(p)  # check remote authority before any model execution
    r124=json.loads(R124_RESULT.read_bytes())
    r123=json.loads(R123_RESULT.read_bytes())
    registered=json.loads(R123_PROTO.read_bytes())
    need(r124['disposition']=='FORENSIC_REPLAY_PASS_BOUNDED' and r124['replayed_events']==40,'R124_EVIDENCE_UNQUAL')
    need(r123['directional_hazards']['2^2/0.25']==40,'R123_COUNT_DRIFT')
    need(r124['by_pathway_class']=={'INHERITED_PATH_REVERSED':40},'R124_PATHWAY_DRIFT')
    events=r124['events']
    need(len(events)==40 and all(e['step']>=3 for e in events),'EVENT_INVENTORY_DRIFT')
    need(Counter(e['step'] for e in events)==Counter({3:38,9:1,11:1}),'EVENT_STEPS_DRIFT')
    group=defaultdict(list)
    for ev in events:
        key=(ev['graph'],ev['k'],ev['checkpoint'],ev['seed'],ev['case'])
        group[key].append(ev)
    need(len(group)<=40,'RUN_CAP')
    oracle=get_oracle()
    need(oracle.parameter_pins()==P.CHECKPOINT_SHA256,'CHECKPOINT_INTEGRITY')
    lib,patch_sha=I.build_probe()
    tgt=float(registered['teacher_k4_limit'])
    graph_names=list(registered['graphs'])
    systems_cache={}
    result_events=[]
    totals={'model_steps':0,'probe_steps':0,'bitwise_probe_checks':0,'exact_r118_checks':0}
    global_clamp=Counter()
    by_event_step=Counter()
    first_events=0
    print('R125_SOURCE_AUTH=PASS BASE='+BASE+' PREREG='+prereg,flush=True)
    for index,(key,evs) in enumerate(sorted(group.items())):
        graph,k,ck,seed,case=key
        gi=graph_names.index(graph)
        if graph not in systems_cache:
            systems_cache[graph],_=H.build_systems(oracle,registered['graphs'][graph],registered['structure_seed']+gi,ks=tuple(registered['k_values']))
        S=systems_cache[graph][k]
        a,b=oracle.r64.pair_state(seed,case,registered['source_ratio'],batch=registered['batch'])
        X0=S.x0(a.numpy(),b.numpy(),scale=registered['scales']['2^2'])
        T=S.teacher(oracle,X0)
        en=R123.engine(S,ck,0.0,tgt)
        ek=R123.engine(S,ck,0.25,tgt)
        eng={'N':en,'K':ek}
        N=execute(S,X0,ck,tgt,'N'*20,eng)
        K=execute(S,X0,ck,tgt,'K'*20,eng)
        totals['model_steps']+=40
        dirs_n=np.stack([M.decompose(S.view(N[h],'READOUT'),S.view(T[h],'READOUT'))['dir_class'] for h in range(1,21)])
        dirs_k=np.stack([M.decompose(S.view(K[h],'READOUT'),S.view(T[h],'READOUT'))['dir_class'] for h in range(1,21)])
        loc={(int(ih)+1,int(pair)) for ih,pair in np.argwhere((dirs_n==M.D_ALIGNED)&(dirs_k==M.D_REVERSED))}
        expected={(int(e['step']),int(e['pair'])) for e in evs}
        need(loc==expected,'R124_EVENT_KEY_DRIFT')
        # All published-selected runs are checked at H8 and H20 against R123 source evidence.
        prior=[x for x in r123['raw_records'] if x['g']==graph and x['k']==k and x['ck']==ck and x['seed']==seed and x['case']==case and x['scale']=='2^2' and x['frac'] in (0,0.25)]
        need(len(prior)==2,'R123_REFERENCE_RUN_MISSING')
        for frac,trajectory in ((0.0,N),(0.25,K)):
            item=next(x for x in prior if x['frac']==frac)
            for horizon in (8,20):
                measure=R123.state_nse(S,trajectory[horizon],T[horizon]); target=item['h'+str(horizon)]
                need(abs(measure-target)<=1e-10*max(1.0,abs(target)),'R123_SCORE_DRIFT')
        # Record actual preclamp raw proposals in the first three steps for both autonomous paths.
        counters={}
        for label,traj,frac in (('N',N,0.0),('K',K,0.25)):
            for h in (1,2,3):
                observed,census=I.measured_step(S,traj[h-1],ck,frac,tgt,lib)
                totals['probe_steps']+=1; totals['bitwise_probe_checks']+=1
                need(np.array_equal(observed.view(np.uint64),traj[h].view(np.uint64)), 'PRECLAMP_INSTRUMENT_BITWISE_DRIFT')
                counters[f'{label}{h}']=census
                global_clamp[f'{label}{h}_hits']+=census['clamp_hits']
        # Exact R118 parity on bounded distinct runs, not a replacement for instrument parity.
        if totals['exact_r118_checks']<p['budgets']['max_reference_parity_runs']:
            q1=R123.simulate(S,X0,R123.engine(S,ck,0.0,tgt,mode='EXACT_R118'))
            q2=R123.simulate(S,X0,R123.engine(S,ck,0.25,tgt,mode='EXACT_R118'))
            need(np.array_equal(q1.view(np.uint64),np.asarray(N).view(np.uint64)) and
                 np.array_equal(q2.view(np.uint64),np.asarray(K).view(np.uint64)), 'GUARDED_VS_R118_DRIFT')
            totals['exact_r118_checks']+=1;totals['model_steps']+=40
        for ev in sorted(evs,key=lambda x:(x['step'],x['pair'])):
            h=int(ev['step']); pair=int(ev['pair'])
            ws=candidate_words(h)
            outcomes={}
            for word in ws:
                if word=='N'*h:out=N[h]
                elif word=='K'*h:out=K[h]
                else:
                    out=execute(S,X0,ck,tgt,word,eng)[-1]
                    totals['model_steps']+=len(word)
                outcomes[word]=viewpoint(S,out,T,h,pair)
            need(outcomes['N'*h]['direction']=='ALIGNED' and outcomes['K'*h]['direction']=='REVERSED','PREFIX_REPLAY_FAIL')
            # Match literal full paths. KN vs NK at h=3 are verified through the same kernel.
            # When h>3, the intervention on H1/H2 is conditional on the fixed remainder.
            candidates={
                'K_ONLY_H1_THEN_NATIVE': 'K'+'N'*(h-1),
                'K_ONLY_H2_THEN_NATIVE': 'N'+'K'+'N'*(h-2),
                'K_BOTH_H1_H2_THEN_NATIVE':'KK'+'N'*(h-2),
                'N_H1_THEN_K_REST':'N'+'K'*(h-1),
                'N_H2_THEN_K_OTHER':'K'+'N'+'K'*(h-2),
                'N_BOTH_H1_H2_THEN_K_REST':'NN'+'K'*(h-2),
            }
            for v in candidates.values():need(v in outcomes,'PREFIX_CANDIDATE_MISSING:'+v)
            signs={name:outcomes[word]['direction']=='REVERSED' for name,word in candidates.items()}
            # First differing operator in a sequence of prefix interventions. This
            # is a context-dependent finite intervention, not unique causal attribution.
            rec={'fixture':{'graph':graph,'k':k,'ck':ck,'seed':seed,'case':case,'pair':pair,'step':h},
                 'first_induced_for_pair':bool(ev['first_induced_for_pair']),
                 'outcomes':outcomes,'prefix_counterfactuals':signs,
                 'preclamp':counters,
                 'full_path_N_cos':outcomes['N'*h]['cosine'],'full_path_K_cos':outcomes['K'*h]['cosine']}
            result_events.append(rec)
            by_event_step[str(h)]+=1
            first_events+=bool(ev['first_induced_for_pair'])
        print(f'R125_PROGRESS={index+1}/{len(group)} GRAPH={graph} k={k} CK={ck} EVENTS={len(evs)}',flush=True)
        need(time.monotonic()-start<=p['budgets']['max_seconds'],'TIME_BUDGET')
        need(totals['model_steps']+totals['probe_steps']<=p['budgets']['max_model_steps'],'MODEL_STEP_BUDGET')
    need(len(result_events)==40 and by_event_step==Counter({'3':38,'9':1,'11':1}),'EVENT_REPLAY_COUNT')
    need(totals['bitwise_probe_checks']==len(group)*6,'PROBE_PARITY_COUNT')
    need(totals['exact_r118_checks']>0,'NO_EXACT_REFERENCE')
    need(oracle.parameter_pins()==P.CHECKPOINT_SHA256 and P.verify_manifest()['files_verified']==39,'FROZEN_MUTATION')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    need(rss<=p['budgets']['max_rss_mib'],'RSS_BUDGET')
    elapsed=time.monotonic()-start
    need(elapsed<=p['budgets']['max_seconds'],'FINAL_TIME_BUDGET')
    summary=Counter()
    for e in result_events:
        for name,yes in e['prefix_counterfactuals'].items():summary[name+'_REVERSED']+=int(yes)
        for h in (1,2,3):
            summary['N'+str(h)+'_PRECLAMP_HIT_EVENTS']+=int(e['preclamp']['N'+str(h)]['clamp_hits']>0)
            summary['K'+str(h)+'_PRECLAMP_HIT_EVENTS']+=int(e['preclamp']['K'+str(h)]['clamp_hits']>0)
    result={'schema':'elpis.ym_hca.r125.forensics.v1','phase':'R125','mode':p['mode'],
            'disposition':'FORENSIC_PREFIX_PASS_BOUNDED','base_commit':BASE,'prereg_commit':prereg,
            'r93':R93,'protocol_sha256':sha(PROTO),'source_sha256':sha(__file__),
            'instrument_sha256':sha(HERE/'instrument.py'),'instrumented_c_sha256':patch_sha,
            'r124_result_sha256':sha(R124_RESULT),'events_replayed':len(result_events),
            'by_event_step':dict(by_event_step),'first_events':first_events,
            'prefix_reversal_counts':dict(summary), 'global_preclamp_counts':dict(global_clamp),
            'events':result_events,'integrity':{'source_pins':True,'published_r124_event_replay':True,
                'original_kernel_for_counterfactuals':True,'probe_bitwise_checks':totals['bitwise_probe_checks'],
                'guarded_r118_exact_reference_checks':totals['exact_r118_checks'],'frozen_unchanged':True},
            'resources':{'seconds':round(elapsed,2),'peak_rss_mib':round(rss,1),**totals},
            'limitations':['Consumed R123 high-amplitude failures only; selection-biased, not fresh holdout.',
              'Binary prefix interventions identify conditional sufficiency/necessity, not unique causal history.',
              'Preclamp counts measure original C proposal saturation using a bitwise-validated shadow build; correlation does not prove clamp causation.',
              'No new policy, no coefficient tuning, no performance qualification; R93 HOLD.']}
    blob=(json.dumps(result,sort_keys=True,indent=1,allow_nan=False)+'\n').encode()
    need(len(blob)<=p['budgets']['max_result_bytes'],'RESULT_SIZE_BUDGET')
    need(not output_path.exists(),'RESULT_EXISTS')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_bytes(blob)
    print('=== SEND BACK ===',flush=True)
    for line in [
        'PHASE=R125_EARLY_PREFIX_CLAMP_FORENSICS',
        'DISPOSITION='+result['disposition'],
        'BASE_COMMIT='+BASE,
        'PREREG_COMMIT='+prereg,
        'PROTOCOL_SHA256='+result['protocol_sha256'],
        'SOURCE_SHA256='+result['source_sha256'],
        'RESULT_SHA256='+hashlib.sha256(blob).hexdigest(),
        'EVENTS_REPLAYED='+str(len(result_events))+'/40',
        'PREFIX_REVERSAL_COUNTS='+json.dumps(result['prefix_reversal_counts'],sort_keys=True),
        'PRECLAMP_GLOBAL_COUNTS='+json.dumps(result['global_preclamp_counts'],sort_keys=True),
        'INSTRUMENT_BITWISE='+str(totals['bitwise_probe_checks'])+'/'+str(totals['bitwise_probe_checks']),
        'REFERENCE_PARITY='+str(totals['exact_r118_checks']),
        'PROSPECTIVE_NEW_DATA=NONE',
        'FROZEN=UNCHANGED','R93='+R93,
        'BLOCKER=NONE CHILD_EXIT=0']:
        print(line,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out).resolve()
    if not out.is_relative_to('/dev/shm'):raise SystemExit('VOLATILE_OUTPUT_ONLY')
    try:evaluate(out)
    except Exception as ex:
        traceback.print_exc()
        print('=== SEND BACK ===\nPHASE=R125_EARLY_PREFIX_CLAMP_FORENSICS\nDISPOSITION=NONQUAL_NOT_COMPLETED\nBLOCKER='+type(ex).__name__+':'+str(ex)[:220]+'\nR93='+R93+'\nCHILD_EXIT=2',flush=True)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
