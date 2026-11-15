"""R16: supervised trainable gated-delta-cell comparator, not full FLA layer.
No runtime mutation; no weights/checkpoints to disk. Evaluation uses disjoint seeds.
"""
from __future__ import annotations
import hashlib,json,math,time
import numpy as np
import torch
from integration import gdn_trainable_r16 as g
from tools.r15_science import TEST_SEEDS,TEST_COUNT

TRAIN_SEEDS=(16001,16002,16003)
VAL_SEEDS={'positive':640111,'hard_pointer':640222,'no_terminal':640333,'missing_goal':640444}
TRAIN_EPISODES=256;EPOCHS=30
GOAL_THRESH=(.75,.85,.93)
KEY_THRESH=(.85,.93,.98)

def validate_case(pred,case):
    if pred['status']=='STOP':
        v=pred['vector'];truth=case.episode.values[case.path[-1]]
        # No discrete ID information exists in the GDN output.
        good=(case.family in ('positive','hard_pointer') and pred['reads']==len(case.path)
              and g._cos(np.asarray(v),truth)>=.90)
        return 'correct' if good else 'false_stop'
    return 'abstain'

def stats(model, cases,gt,kt):
    out=dict(correct=0,false_stop=0,abstain=0,reads=0)
    for case in cases:
        pred=g.autonomous(model,case,threshold=gt,key_threshold=kt)
        stat=validate_case(pred,case)
        out[stat]+=1
        out['reads']+=pred['reads']
    return out

def choose_threshold(model):
    groups={f:[it[-1] for it in g.make_data(seed,16,family=f)] for f,seed in VAL_SEEDS.items()}
    options=[]
    for gt in GOAL_THRESH:
        for kt in KEY_THRESH:
            combined=dict(correct=0,false_stop=0,abstain=0)
            for cs in groups.values():
                s=stats(model,cs,gt,kt)
                for field in combined:combined[field]+=s[field]
            # High penalty for false stop; positive utility for correct answers.
            u=combined['correct']-3*combined['false_stop']
            options.append((u,-combined['false_stop'],combined['correct'],gt,kt,combined))
    win=max(options,key=lambda o:(o[0],o[1],o[2],o[3],o[4]))
    return {'goal_threshold':win[3],'key_threshold':win[4],
            'validation':win[5],'validation_utility':win[0]}

def digest_model(model):
    import struct
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        h.update(name.encode()+b'\0'); h.update(p.detach().cpu().numpy().astype('<f4').tobytes())
    return h.hexdigest()

def run():
    started=time.monotonic();torch.set_num_threads(1)
    allresults=[]
    for seed in TRAIN_SEEDS:
        model,training=g.train(seed=seed,episodes=TRAIN_EPISODES,epochs=EPOCHS)
        if not training['final'] < training['initial']-0.05:
            raise RuntimeError('TRAINING_NOT_EFFECTIVE')
        model.eval();model_digest=digest_model(model)
        calibration=choose_threshold(model)
        groups={fam:[it[-1] for it in g.make_data(start,TEST_COUNT,family=fam)]
                for fam,start in TEST_SEEDS.items()}
        heldout={f:stats(model,cases,calibration['goal_threshold'],calibration['key_threshold'])
                 for f,cases in groups.items()}
        teacher={}
        for family,start in TEST_SEEDS.items():
            tc,n=g.teacher_forced(model,g.make_data(start,TEST_COUNT,family=family))
            teacher[family]={'mean_cosine':round(tc,6),'samples':n}
        if digest_model(model)!=model_digest:raise RuntimeError('EVAL_MUTATED_WEIGHTS')
        for family,counts in heldout.items():
            if sum(counts[k] for k in ('correct','false_stop','abstain'))!=TEST_COUNT:
                raise RuntimeError('NONEXHAUSTIVE_OUTCOME')
        allresults.append(dict(training={**training,'initial':round(training['initial'],7),
                                          'final':round(training['final'],7)},
                               model_sha256=model_digest,calibration=calibration,
                               heldout=heldout,teacher_forced=teacher))
    report={'phase':'ERSA_R16_TRAINABLE_GATED_DELTA_CELL_SCREEN_R0',
            'baseline':'FLA_PUBLISHED_RECURRENCE_MATH_WITH_TRAINED_QKV_BETA_FORGET_AND_OUTPUT',
            'full_gdn_layer_executed':False,
            'train_config':{'seeds':TRAIN_SEEDS,'episodes':TRAIN_EPISODES,'epochs':EPOCHS,
                            'precision':'FP32','training_device':'CPU',
                            'optimizer':'AdamW','loss':'COSINE_PLUS_MSE_TEACHER_FORCED_1_TO_4_HOP'},
            'heldout_seeds':TEST_SEEDS,'validation_seeds':VAL_SEEDS,
            'selection':'VALIDATION_ONLY_GAIN_1_CORRECT_MINUS_3_FALSE_STOP',
            'input':'SAME_24_RECORD_KEYS_GOALS_VALUES;ERSA_ADDITIONAL_METADATA_UNUSED_BY_GDN',
            'feature_state_bytes':g.DKEY*g.DVAL*4,
            'ersa_24_record_keys_goals_values_fp32_bytes':24*(4+4+4)*4,
            'ersa_actual_state_bytes':'NOT_MEASURED_R15_FULL_OWNER_AND_INDEX',
            'trainable_weights_per_seed':g.count_params(g.new_model()),
            'gdn_vs_ersa_fairness':'UNMATCHED_PARAMETER_COUNT_AND_FULL_NATIVE_OWNER;BASELINE_IS_TRAINABLE_CELL_NOT_FULL_LAYER',
            'comparison_status':'SCIENTIFIC_DIAGNOSTIC_NOT_ARCHITECTURE_PARITY',
            'weights_saved':False,'runtime_state_mutation':False,'gpu':False,
            'runs':allresults}
    wire=json.dumps(report,sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(b'ERSA/R16/SCIENCE/R0\0'+wire).hexdigest()
    print('R16_SCIENCE='+wire.decode())
    print('R16_SCIENCE_SHA256='+digest)
    print('R16_SCIENCE_SECONDS='+str(round(time.monotonic()-started,3)))
    return report,digest
if __name__=='__main__':run()
