"""ERSA R13 independently held-out variable-hop recurrent select/stop/abstain probe."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
import integration.ersa_controller_r13 as r
from integration.ersa_selector_r12 import train as train_selector, score_native_source,CSelector,cleanup_compilations
from native.ersa_native import NativeOps,build_native
from integration.ssc_native_r11 import compile_r11

SEEDS={'positive':380112,'no_terminal':1480112,'hard_pointer':2480112,'missing_goal':3480112}
PER_FAMILY=128

def run():
    started=time.monotonic()
    # Independent offline datasets, no evaluation seed reused in training.
    ws,initial_sel,final_sel=train_selector(seed=20261106,episodes=300,epochs=65)
    wh,initial_h,final_h=r.train_halt(seed=60121,episodes=440,epochs=55)
    if not (np.isfinite(ws).all() and np.isfinite(wh).all() and final_sel<initial_sel and final_h<initial_h):
        raise RuntimeError('OFFLINE_TRAINING_FAILED')
    paths=(build_native(),score_native_source(),compile_r11(),r.compile_halt())
    try:
        ops=r.NativeOps(paths[0]);sel=r.CSelector(paths[1]);halt=r.CHalt(paths[3])
        out={fam:r.evaluate(ws,wh,ops=ops,selector=sel,halter=halt,state_lib=paths[2],
                 seed=seed,count=PER_FAMILY,family=fam) for fam,seed in SEEDS.items()}
        replicates=[]
        for seed in (811,812,813):
            ww,ii,ff=r.train_halt(seed=seed,episodes=240,epochs=40)
            probe=r.evaluate(r.KEY_GOAL,ww,ops=ops,selector=sel,halter=halt,
                 state_lib=paths[2],seed=881000+seed,count=64,family='positive')
            replicates.append(dict(seed=seed,learned_success=probe['learned']['success'],
                                   strong_success=probe['strong']['success'],train_loss_gain=ii-ff))
        # Explicitly insist on a strong baseline -- never define success as lift over a weak frozen R7 scorer.
        pos=out['positive'];neg=out['no_terminal'];hard=out['hard_pointer'];missing=out['missing_goal']
        if not (pos['learned']['success']>=110 and hard['learned']['success']>=110 and
                pos['learned']['false_stop']==0 and hard['learned']['false_stop']==0 and
                neg['learned']['false_stop']==0 and neg['learned']['abstain']==PER_FAMILY and
                missing['learned']['abstain']==PER_FAMILY and
                pos['strong']['success']>=110 and
                pos['fixed2']['success']<=pos['learned']['success']-35 and
                all(x['learned_success']>=54 and x['train_loss_gain']>0 for x in replicates)):
            raise RuntimeError('R13_HALT_MECHANICS_GATE_FAILED')
        result=dict(phase='ERSA_R13_TRAINED_BOUNDED_RECURRENT_HALT_R0',
           train_selector_episodes=300,train_selector_epochs=65,
           train_halt_episodes=440,train_halt_epochs=55,
           selector_loss=[initial_sel,final_sel],halt_loss=[initial_h,final_h],
           selector_weights=[float(x) for x in ws],halt_weights=[float(x) for x in wh],
           train_params=8+6,max_hops=r.MAX_HOPS,max_native_slots=8192,
           eval_per_family=PER_FAMILY,heldout=out,replicate_seeds=replicates,
           runtime='C_NATIVE_SSC_EXACT_READ_C_SCORER_C_HALT_HOST_ORCHESTRATED',
           baseline='STRONG_KEY_GOAL_DETERMINISTIC_STOP_WITH_SAME_POINTER_VETO',
           model_scope='SYNTHETIC_VARIABLE_HOP_1_TO_4_POINTER_CHAINS_ONLY',
           inference_weights='FROZEN_RAM_ONLY',
           negative='NO_TERMINAL_AND_MISSING_GOAL_ABSTAIN',
           trust='NONE',rights='HOLD',gdn='NOT_TESTED',
           claims='LEARNED_MECHANICS_ONLY_NOT_SUPERIOR_TO_DETERMINISTIC;NOT_FULL_NEURAL_RUNTIME')
        wire=json.dumps(result,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(b'ERSA/R13/SCIENCE/R0\0'+wire).hexdigest()
        print('R13_SCIENCE='+json.dumps(result,sort_keys=True,separators=(',',':')))
        print('R13_SCIENCE_SHA256='+digest)
        print('R13_HALT_WEIGHTS_SHA256='+hashlib.sha256(np.asarray(wh,dtype='<f8').tobytes()).hexdigest())
        print('R13_SCIENCE_SECONDS='+str(round(time.monotonic()-started,3)))
        return digest
    finally:cleanup_compilations(*paths)

if __name__=='__main__':run()
