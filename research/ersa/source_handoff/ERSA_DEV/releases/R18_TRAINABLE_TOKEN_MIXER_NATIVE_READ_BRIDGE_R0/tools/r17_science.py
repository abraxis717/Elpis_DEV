"""R17: CPU PyTorch FLA-GDN-v1 architectural-reference experiment; no FLA kernel claim."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
import torch
from integration import gdn_layer_reference_r17 as layer
from integration import gdn_trainable_r16 as base
from tools import r16_science as previous

TRAIN_SEEDS=(17001,17002,17003)
EPISODES=256;EPOCHS=30
VAL_COUNT=16
TEST_COUNT=64

def sha_model(model):
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        h.update(name.encode()+b'\0'+p.detach().cpu().numpy().astype('<f4').tobytes())
    return h.hexdigest()

def stats(model,cases,gt,kt):
    result=dict(correct=0,false_stop=0,abstain=0,reads=0)
    for case in cases:
        pred=layer.autonomous(model,case,goal_threshold=gt,key_threshold=kt)
        outcome=previous.validate_case(pred,case)
        result[outcome]+=1
        result['reads']+=pred['reads']
    return result

def tune(model):
    cases={f:[it[-1] for it in base.make_data(seed,VAL_COUNT,family=f)] for f,seed in previous.VAL_SEEDS.items()}
    choices=[]
    for gt in previous.GOAL_THRESH:
        for kt in previous.KEY_THRESH:
            tot=dict(correct=0,false_stop=0,abstain=0)
            for group in cases.values():
                r=stats(model,group,gt,kt)
                for key in tot:tot[key]+=r[key]
            objective=tot['correct']-3*tot['false_stop']
            choices.append((objective,-tot['false_stop'],tot['correct'],gt,kt,tot))
    best=max(choices,key=lambda x:x[:5])
    return {'goal_threshold':best[3],'key_threshold':best[4], 'validation':best[5], 'utility':best[0]}

def run():
    started=time.monotonic()
    torch.set_num_threads(1)
    results=[]
    for seed in TRAIN_SEEDS:
        model,training=layer.train(seed,episodes=EPISODES,epochs=EPOCHS)
        if not training['end']<training['start']-.05:raise RuntimeError('TRAINING_LOSS_NOT_IMPROVED')
        model.eval();before=sha_model(model)
        threshold=tune(model)
        heldout={family:stats(model,[it[-1] for it in base.make_data(s,TEST_COUNT,family=family)],
                              threshold['goal_threshold'],threshold['key_threshold'])
                 for family,s in previous.TEST_SEEDS.items()}
        for counts in heldout.values():
            if sum(counts[v] for v in ('correct','false_stop','abstain'))!=TEST_COUNT:
                raise RuntimeError('BAD_OUTCOME_TOTAL')
        if sha_model(model)!=before:raise RuntimeError('EVAL_MUTATED_MODEL')
        results.append({'training':training,'weights_sha256':before,'validation':threshold,'heldout':heldout})
        print('R17_SEED='+str(seed)+';LOSS='+str(training['start'])+'->'+str(training['end'])+
              ';POSITIVE='+str(heldout['positive'])+';HARD='+str(heldout['hard_pointer']),flush=True)
    report={
        'phase':'ERSA_R17_CPU_GDN_LAYER_STRUCTURAL_REFERENCE_R0',
        'basis':'FLA_GATED_DELTANET_V1_LAYER_AND_RECURRENT_RULE_MATH_REIMPLEMENTED_PYTORCH_CPU',
        'official_fla_module_executed':False,'official_fla_gpu_kernel_executed':False,
        'components':'QKV_PROJECTIONS;CAUSAL_DEPTHWISE_SHORT_CONVS;A_LOG_DT_BIAS_DECAY;BETA;OUTPUT_GATE;RMS_NORM;OUTPUT_PROJ',
        'explicit_deviations':'EXTERNAL_12_TO_32_EMBED;READ_ONLY_QUERY_BETA_ZERO;TASK_32_TO_4_READOUT;CPU_EAGER_RECURRENCE;SINGLE_HEAD',
        'train_config':{'seeds':TRAIN_SEEDS,'episodes':EPISODES,'epochs':EPOCHS,'precision':'FP32','device':'CPU'},
        'trainable_weights':layer.count_parameters(layer.new_model()),
        'gdn_state_bytes':16*32*4,'gdn_short_conv_cache_bytes':(16+16+32)*3*4,
        'ersa_authority_bytes':'NOT_MEASURED_NATIVE_16MIB_MAX_CAP;NOT_MATCHED',
        'comparison':'SAME_SYNTHETIC_TASK_LAW_AND_FAMILY_SEEDS;NOT_MATCHED_PARAM_COUNT_MEMORY_BUDGET_OR_LAYER_DEPTH',
        'learned_ersa_superiority':'NOT_ESTABLISHED','full_fla_parity':'NOT_EXECUTED',
        'runtime_state':'RAM_ONLY;NO_INFERENCE_LEARNING;NO_CHECKPOINTS',
        'frozen_r16':'SOURCE_HASHES_CHECKED;R16_SCIENCE_PRIOR_ONLY_NOT_REEXECUTED',
        'runs':results}
    payload=json.dumps(report,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(b'ERSA/R17/SCIENCE/R0\0'+payload.encode()).hexdigest()
    print('R17_SCIENCE='+payload,flush=True)
    print('R17_SCIENCE_SHA256='+digest,flush=True)
    print('R17_SCIENCE_SECONDS='+str(round(time.monotonic()-started,3)),flush=True)
    return report,digest
if __name__=='__main__':run()
