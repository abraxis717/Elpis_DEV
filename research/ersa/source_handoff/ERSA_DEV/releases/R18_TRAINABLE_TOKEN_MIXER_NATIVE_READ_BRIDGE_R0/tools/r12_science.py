"""Independent held-out synthetic 2-hop selection evaluation (CPU only)."""
from __future__ import annotations
import hashlib,json,math,os,time
import numpy as np
from integration.ersa_selector_r12 import (
  train,NativeOps,build_native,CSelector,score_native_source,compile_r11,
  evaluate,FROZEN_WEIGHTS,cleanup_compilations)

TRAIN_SEED=20261106
EVAL_SEEDS={'iid':508000,'misleading_meta':1508000,
            'hard_goal':2508000,'no_goal':3508000}
EVAL_EPISODES=256
# Same selection threshold in L2-normalized score units for each controller.
NORMALIZED_MARGIN=.015

def run():
    s=time.monotonic()
    w,initial,final=train(TRAIN_SEED,episodes=480,epochs=95)
    a=build_native();b=score_native_source();c=compile_r11()
    try:
        ops=NativeOps(a);sel=CSelector(b)
        controllers={
          'frozen_r7':FROZEN_WEIGHTS,
          'key_only':np.array([1.,0.,0.,0.,0.,0.,0.,0.]),
          'goal_only':np.array([0.,0.,0.,0.,0.,1.,0.,0.]),
          'key_goal':np.array([1.,0.,0.,0.,0.,1.,0.,0.]),
        }
        by_family={}
        for regime,seed in EVAL_SEEDS.items():
            by_family[regime]=evaluate(w,selector=sel,ops=ops,owner_lib=c,
                seed=seed,count=EVAL_EPISODES,regime=regime,baselines=controllers,
                normalized_margin=NORMALIZED_MARGIN)
        # A second independent random train seed tests whether the result was
        # an accidental favorable initialization or a single training draw.
        replicates=[]
        for seed in (8801,8802,8803):
            alternate,_,_=train(seed,episodes=256,epochs=64)
            outcome=evaluate(alternate,selector=sel,ops=ops,owner_lib=c,
                    seed=8110000,count=64,regime='hard_goal',
                    baselines={'key_goal':controllers['key_goal']},normalized_margin=NORMALIZED_MARGIN)
            replicates.append(dict(seed=seed,learned=outcome['learned']['success'],
                                   key_goal=outcome['key_goal']['success']))
        # 8192 only tests bounded C scoring, not 8192-address native neural read.
        rng=np.random.default_rng(991)
        features=rng.standard_normal((8192,8))
        t0=time.perf_counter_ns()
        score=sel.score(features,w)
        score_us=(time.perf_counter_ns()-t0)/1e3
        if len(score)!=8192 or not np.isfinite(score).all():raise RuntimeError('MAX_SCORER_FAILURE')
    finally:cleanup_compilations(a,b,c)
    learned=by_family['iid']['learned']['success']
    frozen=by_family['iid']['frozen_r7']['success']
    key=by_family['iid']['key_only']['success']
    keygoal=by_family['iid']['key_goal']['success']
    hard=by_family['hard_goal']['learned']['success']
    negative=by_family['no_goal']['learned']['abstain']
    if not (initial>final and learned>=230 and hard>=225 and
            learned>frozen+30 and learned>key+30 and negative>=230 and
            all(r['learned']>=56 for r in replicates)):
        raise RuntimeError('CONDITIONAL_LEARNED_SELECTOR_MECHANICS_GATES_FAILED')
    result=dict(train_seed=TRAIN_SEED,train_episodes=480,train_epochs=95,
      train_loss_initial=initial,train_loss_final=final,
      weights=[float(x) for x in w],param_count=len(w),
      threshold_normalized=NORMALIZED_MARGIN,
      eval_samples_per_family=EVAL_EPISODES,
      heldout=by_family,train_seed_replicates=replicates,
      source_law='SYNTHETIC_DUPLICATE_KEY_DISTINCT_GOAL_TWO_HOP_CPU',
      reference_score_scope='FROZEN_R7_WEIGHTS',
      max_score_candidates=8192,
      max_score_native_micro_us=score_us,
      qualification='HOST_ORCHESTRATED_TWO_HOP_TRAINED_8W_C_SCORER_C_OWNED_EXACT_STATE',
      no_claims='NO_LEARNED_SUPERIORITY_TO_KEY_GOAL;NO_GDN_BASELINE;NO_MODEL_SCALE;NO_OPEN_WORLD_SEMANTICS')
    checksum=hashlib.sha256(json.dumps({k:v for k,v in result.items() if k!='max_score_native_micro_us'},
       sort_keys=True,separators=(',',':')).encode()).hexdigest()
    print('R12_SCIENCE_JSON='+json.dumps(result,sort_keys=True,separators=(',',':')))
    print('R12_SCIENCE_SHA256='+checksum)
    print('R12_TRAIN_WEIGHTS_SHA256='+hashlib.sha256(np.asarray(w,dtype='<f8').tobytes()).hexdigest())
    print('R12_LOCAL_SCIENCE_SECONDS='+str(round(time.monotonic()-s,3)))
    return checksum

if __name__=='__main__':run()
