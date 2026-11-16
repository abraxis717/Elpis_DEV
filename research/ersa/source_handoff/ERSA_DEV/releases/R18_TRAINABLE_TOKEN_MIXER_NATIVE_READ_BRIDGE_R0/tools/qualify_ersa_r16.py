"""Qualify frozen R15 plus trainable gated-delta comparator. Fail closed."""
from __future__ import annotations
import json,os,re,sys,time,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import qualify_ersa_r14 as core

EXPECTED_TESTS=1091
R15_SCIENCE_PIN='2eed88cfc7af55adddd9e7752f927e82e29d74217d4beeb9742c5f9f117d6c94'

def main():
    t0=time.monotonic();tests=0;n15=nssc=nr2=nown=0
    r15='PRIOR_ONLY_NOT_REEXECUTED';r16='NOT_EXECUTED';science=None;error='';status='HOLD'
    try:
        if not ROOT.resolve().is_relative_to('/dev/shm'):
            raise ValueError('TMPFS_ONLY_QUALIFICATION')
        own=core.manifest('FILES.sha256')
        frozen=core.manifest('FROZEN_R15.sha256')
        if len(frozen)!=107 or any(own.get(k)!=v for k,v in frozen.items()):
            raise ValueError('FROZEN_R15_DRIFT')
        n15=len(frozen);nown=len(own)
        ssc={k.removeprefix('ssc_predecessor/'):v for k,v in core.manifest('SSC_PREDECESSOR.sha256').items()}
        sr=core.source_root('ERSA_SSC_ROOT',core.SSC_DEFAULT)
        rr=core.source_root('ERSA_R2_ROOT',core.R2_DEFAULT)
        if len(ssc)!=7:raise ValueError('SSC_PIN_COUNT')
        core.verify_external(sr,ssc);core.verify_external(rr,core.R2_PINS)
        nssc=len(ssc);nr2=len(core.R2_PINS)
        if '--rerun-capacity' in sys.argv:
            stress=core.child(['tools/r3_capacity_stress.py'],timeout=900)
            if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
                raise ValueError('CAPACITY_REGRESSION')
        if '--replay-r15' in sys.argv:
            previous=core.child(['tools/r15_science.py'],timeout=600)
            m=re.search(r'^R15_SCIENCE_SHA256=([a-f0-9]{64})$',previous,re.M)
            if not m or m.group(1)!=R15_SCIENCE_PIN:
                raise ValueError('FROZEN_R15_SCIENCE_DRIFT:'+str(m.group(1) if m else None))
            r15=m.group(1)
        print('GATE=R16_SCIENCE_START',flush=True)
        latest=core.child(['tools/r16_science.py'],timeout=240)
        print('GATE=R16_SCIENCE_END',flush=True)
        m=re.search(r'^R16_SCIENCE_SHA256=([a-f0-9]{64})$',latest,re.M)
        s=re.search(r'^R16_SCIENCE=(.+)$',latest,re.M)
        if not m or not s:raise ValueError('MISSING_R16_SCIENCE')
        science=json.loads(s.group(1));r16=m.group(1)
        if len(science['runs'])!=3 or science['full_gdn_layer_executed']:
            raise ValueError('SCIENCE_SCOPE_MISMATCH')
        for run in science['runs']:
            tr=run['training']
            if not (tr['final'] < tr['initial']-.05 and tr['parameters']==1075 and tr['state_bytes']==1024):
                raise ValueError('LEARNING_UTILITY_NOT_ESTABLISHED')
            for f,stats in run['heldout'].items():
                if stats['correct']+stats['false_stop']+stats['abstain']!=64:
                    raise ValueError('NONEXHAUSTIVE_HELDOUT')
            print('TRAIN_SEED='+str(tr['seed'])+';LOSS='+str(tr['initial'])+'->'+str(tr['final'])+
                  ';POSITIVE='+str(run['heldout']['positive'])+';HARD_POINTER='+
                  str(run['heldout']['hard_pointer']),flush=True)
        suite=core.child(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=240)
        m=re.search(r'(\d+) passed',suite)
        tests=int(m.group(1)) if m else 0
        if tests!=EXPECTED_TESTS:raise ValueError('TEST_COUNT_MISMATCH:'+str(tests))
        print('PYTEST='+suite.strip().splitlines()[-1],flush=True)
        if core.manifest('FILES.sha256')!=own:raise ValueError('POST_RUN_PACKAGE_SOURCE_MUTATION')
        core.verify_external(sr,ssc);core.verify_external(rr,core.R2_PINS)
        status='PASS_TRAINABLE_GATED_DELTA_CELL_AND_FROZEN_ERSA_SCREEN_MECHANICS_ONLY'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R16_TRAINABLE_GATED_DELTA_CELL_COMPARISON_R0')
    print('DISPOSITION='+status)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R15_PINS='+str(n15)+'/107;SSC_PINS='+str(nssc)+'/7;R2_PINS='+str(nr2)+'/4')
    print('R16_OWN_FILES='+str(nown))
    print('R15_CAPACITY_REPLAY='+('EXECUTED' if '--rerun-capacity' in sys.argv else 'PRIOR_OUROBOROS_ONLY'))
    print('R15_SCIENCE_REPLAY='+('EXECUTED' if r15==R15_SCIENCE_PIN else 'PRIOR_OUROBOROS_ONLY'))
    print('R15_PRIOR_EXPECTED_SCIENCE_SHA256='+R15_SCIENCE_PIN)
    print('R15_SCIENCE_SHA256='+r15)
    print('R16_SCIENCE_SHA256='+r16)
    if science:
        for run in science['runs']:
            tr=run['training'];print('SEED='+str(tr['seed'])+';LOSS_START='+str(tr['initial'])+
                  ';LOSS_END='+str(tr['final'])+';PARAMS='+str(tr['parameters'])+
                  ';STATE_BYTES='+str(tr['state_bytes']))
            for family,stat in run['heldout'].items():
                print('FAMILY='+family+';SEED='+str(tr['seed'])+';CORRECT='+str(stat['correct'])+
                      ';FALSE_STOP='+str(stat['false_stop'])+';ABSTAIN='+str(stat['abstain']))
    print('FULL_FLA_LAYER=NOT_EXECUTED;TRAINED_GDN_CELL=QKV_BETA_FORGET_READOUT')
    print('MATCHED_ARCHITECTURAL_GDN_PARITY=HOLD;LEARNED_SUPERIORITY=HOLD;SEMANTIC_GENERALIZATION=HOLD')
    print('TRUST=RESEARCH_ONLY;RIGHTS=HOLD;PERSISTENT_INFERENCE_WRITES=NONE_INTENDED;OS_WRITE_ATTESTATION=NONE')
    print('MODEL_CHECKPOINT_WRITES=0;GPU=NONE;PROJECT_MUTATIONS=NONE_INTENDED')
    print('ELAPSED_S='+str(round(time.monotonic()-t0,3)))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(bool(error)*1))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
