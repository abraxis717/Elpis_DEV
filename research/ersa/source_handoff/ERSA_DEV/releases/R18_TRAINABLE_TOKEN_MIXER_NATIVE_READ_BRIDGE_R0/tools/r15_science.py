"""ERSA R15: paired native ERSA and gated-delta-rule baseline diagnostics.

This is NOT a trained full Gated DeltaNet layer or matched compute/weight benchmark.
Compare synthetic protocols to expose the architecture's memory tradeoff; no GDN-parity
claim is awarded. All data and training are fresh in RAM, no checkpoints.
"""
from __future__ import annotations
import hashlib,json,math,time
import numpy as np
from integration.gdn_rule_r15 import state_for_episode,read,cosine
from integration import ersa_controller_r13 as r13
from integration.ersa_recur_r14 import CRecur,compile_recur,run_on_owner
from integration.ersa_selector_r12 import open_state,train as train_selector,score_native_source,cleanup_compilations
from integration.ssc_native_r11 import compile_r11
from native.ersa_native import NativeOps,build_native

TRAIN_SEED=916200; TRAIN_COUNT=64
TEST_SEEDS={'positive':760111,'hard_pointer':760222,'no_terminal':760333,'missing_goal':760444}
TEST_COUNT=64
BETA_GRID=(.25,.5,.75,1.)
DECAY_GRID=(.95,.99,1.)


def gdn_episode(case,beta,decay):
    ep=case.episode
    s=state_for_episode(ep,beta=beta,decay=decay)
    q=np.asarray(ep.query_key,np.float64).copy()
    actual=[];max_sim=0.;first_cos=-1.
    # GDN's state is fixed for all reads; no oracle pointer indices enter the decoder.
    for hop in range(1,5):
        x=read(s,q,ep.query_goal)
        if not np.isfinite(x).all():return dict(status='HOLD_NUMERICAL',first_cos=first_cos,reads=hop)
        if hop==1:first_cos=cosine(x,ep.values[case.path[0]])
        max_key=max(cosine(x,k) for k in ep.keys)
        goal=cosine(x,ep.query_goal)
        if goal>=.80 and max_key<.999999:
            if not case.terminal or case.family=='missing_goal':return dict(status='FALSE_STOP',first_cos=first_cos,reads=hop)
            # Evaluate final output as a continuous vector, not an exact relation ID.
            truth=ep.values[case.path[-1]]
            score=cosine(x,truth)
            if hop==len(case.path) and score>=.90:
                return dict(status='CONTINUOUS_OK',first_cos=first_cos,reads=hop)
            return dict(status='FALSE_STOP',first_cos=first_cos,reads=hop)
        if np.linalg.norm(x)<1e-10:return dict(status='HOLD_ZERO_READ',first_cos=first_cos,reads=hop)
        q=x
    return dict(status='HOLD_BUDGET',first_cos=first_cos,reads=4)


def select_gates():
    # Strictly separate training episodes from four heldout R14 science seeds.
    train_cases=[r13.make_chain(TRAIN_SEED+j*4831,family='positive') for j in range(TRAIN_COUNT)]
    choices=[]
    for beta in BETA_GRID:
        for decay in DECAY_GRID:
            score=sum(cosine(read(state_for_episode(c.episode,beta=beta,decay=decay),
                                  c.episode.query_key,c.episode.query_goal),
                             c.episode.values[c.path[0]]) for c in train_cases)/len(train_cases)
            choices.append((score,beta,decay))
    # Fix tie-breaks in advance; primary metric is one-hop cosine.
    best=max(choices,key=lambda row:(row[0],row[1],row[2]))
    return dict(train_episodes=TRAIN_COUNT,objective='TRAIN_FIRST_HOP_MEAN_COSINE',
                mean_cosine=round(best[0],8),beta=best[1],decay=best[2],
                grid_size=len(choices))


def run():
    started=time.monotonic()
    np.seterr(all='raise')
    gate=select_gates()
    weights,sl0,sl1=train_selector(seed=20261106,episodes=300,epochs=65)
    halt,hl0,hl1=r13.train_halt(seed=60121,episodes=440,epochs=55)
    if not sl1<sl0 or not hl1<hl0:raise RuntimeError('INVALID_FROZEN_TRAINING_PROTOCOL')
    libraries=[build_native(),score_native_source(),compile_r11(),r13.compile_halt(),compile_recur()]
    try:
        engine=CRecur(libraries[-1])
        output={}
        for family,seed in TEST_SEEDS.items():
            result={p:dict(correct=0,false_stop=0,abstain=0,reads=0) for p in
                    ('ersa_learned','ersa_deterministic','gdn_grid_tuned_rule','gdn_beta1_alpha1_rule')}
            cosines={'gdn_grid_tuned_rule':[], 'gdn_beta1_alpha1_rule':[]}
            for j in range(TEST_COUNT):
                case=r13.make_chain(seed+j*104729,family=family)
                with open_state(libraries[2],case.episode) as owner:
                    for policy,key,sw in [('learned','ersa_learned',weights),
                                          ('strong','ersa_deterministic',r13.KEY_GOAL)]:
                        pred=run_on_owner(owner,engine,case,sw,halt,policy=policy)
                        stat=result[key]; stat['reads']+=len(pred.get('path',()))
                        if pred['status']=='OK':
                            correct=(case.terminal and tuple(pred['path'])==case.path and
                                tuple(pred['answer'])==tuple(case.episode.values[case.path[-1]]))
                            stat['correct']+=int(correct);stat['false_stop']+=int(not correct)
                        else:stat['abstain']+=1
                for key,beta,decay in (
                    ('gdn_grid_tuned_rule',gate['beta'],gate['decay']),
                    ('gdn_beta1_alpha1_rule',1.,1.)):
                    pred=gdn_episode(case,beta,decay);stat=result[key]
                    stat['reads']+=pred['reads'];stat['correct']+=int(pred['status']=='CONTINUOUS_OK')
                    stat['false_stop']+=int(pred['status']=='FALSE_STOP')
                    stat['abstain']+=int(pred['status'] not in ('CONTINUOUS_OK','FALSE_STOP'))
                    cosines[key].append(pred['first_cos'])
            for key in cosines:result[key]['mean_first_hop_cos']=round(float(np.mean(cosines[key])),6)
            for p,stat in result.items():
                if stat['correct']+stat['false_stop']+stat['abstain']!=TEST_COUNT:
                    raise RuntimeError('NONEXHAUSTIVE_PREDICTIONS:'+p)
            output[family]=result
        # Do not claim GDN-equivalent training or representation sizes.
        report=dict(phase='ERSA_R15_GATED_DELTA_RULE_BASELINE_SCREEN_R0',
            gdn_rule_reference='FLA MIT naive_recurrent_gated_delta_rule equation, independent NumPy+Torch verification',
            architecture_scope='RULE_ONLY_PRECOMPUTED_QKV;NOT_FULL_GDN_MODEL;NOT_MATCHED_TRAINING',
            gate_selection=gate,
            gdn_associative_state_bytes=8*4*8,  # fp64, key width 8, value width 4
            ersa_exact_values_only_bytes=24*4*8,
            ersa_additional_keys_goals_meta_not_included=True,
            task='HELDOUT_24_RECORD_SYNTHETIC_POINTER_CHAINS',
            max_hops=4,heldout=output,
            source_authority='FROZEN_R14',rights='HOLD',
            comparator_fairness='UNMATCHED_FEATURE_ENCODING_STATE_BYTES_TRAINING_AND_OPERATIONAL_CAPABILITIES',
            qualification='NONQUAL_GDN_RULE_ONLY_UNMATCHED_REPRESENTATION_AND_BUDGET')
        wire=json.dumps(report,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(b'ERSA/R15/SCIENCE/R0\0'+wire).hexdigest()
        print('R15_SCIENCE='+wire.decode())
        print('R15_SCIENCE_SHA256='+digest)
        print('R15_SCIENCE_SECONDS='+str(round(time.monotonic()-started,3)))
        return report,digest
    finally:cleanup_compilations(*libraries)
if __name__=='__main__':run()
