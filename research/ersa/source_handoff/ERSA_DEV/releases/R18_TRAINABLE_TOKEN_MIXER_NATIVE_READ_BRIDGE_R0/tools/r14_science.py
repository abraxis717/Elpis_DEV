"""ERSA R14: paired heldout host vs single-call native recurrence, same state and weights."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
from integration import ersa_controller_r13 as r13
from integration.ersa_recur_r14 import compile_recur,CRecur,run_on_owner
from integration.ersa_selector_r12 import (
    open_state, score_native_source, CSelector, train as train_selector,cleanup_compilations,
)
from native.ersa_native import NativeOps,build_native
from integration.ssc_native_r11 import compile_r11
SEEDS={'positive':760111,'hard_pointer':760222,'no_terminal':760333,'missing_goal':760444}
COUNT=64
POLICIES=('learned','keygoal_learned_halt','strong','fixed2')

def run():
    started=time.monotonic()
    ws,sl0,sl1=train_selector(seed=20261106,episodes=300,epochs=65)
    wh,hl0,hl1=r13.train_halt(seed=60121,episodes=440,epochs=55)
    if not sl1<sl0 or not hl1<hl0:raise RuntimeError('INVALID_OFFLINE_TRAINING')
    paths=[build_native(),score_native_source(),compile_r11(),r13.compile_halt(),compile_recur()]
    try:
        ops=NativeOps(paths[0]);sel=CSelector(paths[1]);halt=r13.CHalt(paths[3]);engine=CRecur(paths[4])
        results={};host_cyclic_ok=0;parity=0
        for family,seed in SEEDS.items():
            stats={p:dict(correct=0,false_stop=0,abstain=0,reads=0,host_us=[],native_us=[])
                for p in POLICIES}
            for j in range(COUNT):
                case=r13.make_chain(seed+j*104729,family=family)
                with open_state(paths[2],case.episode) as owner:
                    for policy in POLICIES:
                        sw=ws if policy=='learned' else r13.KEY_GOAL
                        host_policy='learned' if policy=='keygoal_learned_halt' else policy
                        tic=time.perf_counter_ns()
                        old=r13.execute(owner,ops,sel,halt,case,sw,wh,policy=host_policy)
                        toc=time.perf_counter_ns()
                        new=run_on_owner(owner,engine,case,sw,wh,policy=policy)
                        tend=time.perf_counter_ns()
                        s=stats[policy];s['host_us'].append((toc-tic)/1000)
                        s['native_us'].append((tend-toc)/1000)
                        if old['status']=='OK' and len(set(old['path']))!=len(old['path']):
                            if not (new['status']=='HOLD_CYCLE' and new['path']==old['path']):
                                raise RuntimeError('FAIL_CLOSED_CYCLE_DRIFT')
                            host_cyclic_ok+=1
                        else:
                            if new['status']!=old['status'] or new['path']!=old['path']:
                                raise RuntimeError('NATIVE_HOST_DECISION_DIFF:'+str((family,policy,j,old,new)))
                            if old['status']=='OK' and (new.get('answer')!=old.get('answer') or
                                                          new.get('steps')!=old.get('steps')):
                                raise RuntimeError('NATIVE_HOST_RECEIPT_DIFF')
                            parity+=1
                        s['reads']+=len(new['path'])
                        if new['status']=='OK':
                            correct=case.terminal and tuple(new['path'])==case.path and (
                                tuple(new['answer'])==tuple(case.episode.values[case.path[-1]]))
                            s['correct']+=int(correct);s['false_stop']+=int(not correct)
                        else:s['abstain']+=1
            results[family]={p:{
                'correct':v['correct'],'false_stop':v['false_stop'],
                'abstain':v['abstain'],'reads':v['reads'],
                'host_median_us':round(float(np.median(v['host_us'])),3),
                'native_median_us':round(float(np.median(v['native_us'])),3)}
                for p,v in stats.items()}
        pos=results['positive'];hard=results['hard_pointer'];neg=results['no_terminal'];missing=results['missing_goal']
        if not (pos['learned']['correct']>=52 and hard['learned']['correct']>=52 and
            pos['learned']['false_stop']==0 and hard['learned']['false_stop']==0 and
            neg['learned']['false_stop']==0 and neg['learned']['abstain']==COUNT and
            missing['learned']['abstain']==COUNT and
            pos['strong']['correct']>=52 and parity+host_cyclic_ok==COUNT*len(SEEDS)*len(POLICIES)):
            raise RuntimeError('R14_SCIENCE_MECHANICS_GATE_FAILED')
        report={
            'phase':'ERSA_R14_SINGLE_CALL_NATIVE_BOUNDED_RECURRENCE_R0',
            'max_hops':4,'capacity':8192,'recurrent_state':'C_NATIVE_LOOP_ON_PINNED_C_STATE_SNAPSHOT',
            'state_authority':'R11_TYPED_C_WITH_HOST_INPUT_VALIDATION',
            'generation_policy':'R11_OWNER_LOCK_WITH_AFTER_READ_GENERATION_PIN',
            'train_selector':[300,65,sl0,sl1],'train_halt':[440,55,hl0,hl1],
            'trainable_params':14,'runtime_learning':False,'test_cases_per_family':COUNT,
            'exact_host_parity':parity,'host_duplicate_stop_prevented':host_cyclic_ok,
            'heldout':results,
            'baseline':'R13_STRONG_KEY_GOAL_RULE_SAME_C_MEMORY_AND_BUDGET',
            'gdn':'NOT_TESTED_NO_GENUINE_BASELINE',
            'claims':'SINGLE_CALL_NATIVE_RECURRENCE_DIFFERENTIAL_ONLY;NO_LEARNED_SUPERIORITY;NO_OPEN_WORLD_SEMANTICS',
            'writes':'NO_RUNTIME_DISK_IO;NO_MODEL_MUTATION',
            'scope':'SYNTHETIC_FOUR_VECTOR_POINTER_CHAINS;HOST_ADMISSION_AND_SNAPSHOT'}
        # Latencies are environment-dependent and deliberately EXCLUDED from
        # the source-independent scientific decision digest.
        science=dict(report)
        science['heldout']={family:{policy:{k:v for k,v in stats.items()
                           if k not in ('host_median_us','native_median_us')}
                           for policy,stats in block.items()}
                           for family,block in results.items()}
        wire=json.dumps(science,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(b'ERSA/R14/SCIENCE/R0\0'+wire).hexdigest()
        print('R14_SCIENCE='+json.dumps(report,sort_keys=True,separators=(',',':')))
        print('R14_SCIENCE_SHA256='+digest)
        print('R14_SCIENCE_DIGEST_SCOPE=COUNTS_AND_WEIGHTS_EXCLUDES_TIMINGS')
        print('R14_SCIENCE_SECONDS='+str(round(time.monotonic()-started,3)))
        return digest
    finally:cleanup_compilations(*paths)
if __name__=='__main__':run()
