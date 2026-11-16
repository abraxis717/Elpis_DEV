"""ERSA R15 bounded recurrent rule baseline screening qualification.

Inherits R14 frozen validation, deterministic capacity/regression, and native science.
Never calls this a full GDN layer comparison or claims learned superiority.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import qualify_ersa_r14 as frozen
EXPECTED_TESTS=1042
EXPECTED_R14_OWN=101


def main():
    started=time.monotonic();tests=0;frozen_n=ssc_n=r2_n=0;own_n=0
    r14_sha='NONE';r15_sha='NONE';science=None;error='';disposition='HOLD'
    try:
        if not ROOT.resolve().is_relative_to('/dev/shm'):
            raise ValueError('QUALIFICATION_MUST_RUN_IN_TMPFS')
        own=frozen.manifest('FILES.sha256')
        fr=frozen.manifest('FROZEN_R14.sha256')
        if len(fr)!=EXPECTED_R14_OWN or any(own.get(k)!=v for k,v in fr.items()):
            raise ValueError('FROZEN_R14_SOURCE_DRIFT')
        frozen_n=len(fr);own_n=len(own)
        ss={k.removeprefix('ssc_predecessor/'):v for k,v in frozen.manifest('SSC_PREDECESSOR.sha256').items()}
        if len(ss)!=7:raise ValueError('SSC_PIN_COUNT')
        sr=frozen.source_root('ERSA_SSC_ROOT',frozen.SSC_DEFAULT)
        rr=frozen.source_root('ERSA_R2_ROOT',frozen.R2_DEFAULT)
        frozen.verify_external(sr,ss);frozen.verify_external(rr,frozen.R2_PINS)
        ssc_n=len(ss);r2_n=len(frozen.R2_PINS)
        suite=frozen.child(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=180)
        m=re.search(r'(\d+) passed',suite)
        tests=int(m.group(1)) if m else -1
        if tests!=EXPECTED_TESTS:raise ValueError('EXPECTED_TESTS_'+str(EXPECTED_TESTS)+'_GOT_'+str(tests))
        print('PYTEST='+suite.strip().splitlines()[-1],flush=True)
        stress=frozen.child(['tools/r3_capacity_stress.py'],timeout=60)
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('CAPACITY_REGRESSION')
        r14=frozen.child(['tools/r14_science.py'],timeout=180)
        match=re.search(r'^R14_SCIENCE_SHA256=([a-f0-9]{64})$',r14,re.M)
        rep=re.search(r'^R14_SCIENCE=(.+)$',r14,re.M)
        if not match or not rep:raise ValueError('MISSING_R14_SCIENCE')
        r14_sha=match.group(1);rd=json.loads(rep.group(1))
        if rd['exact_host_parity']+rd['host_duplicate_stop_prevented']!=1024:
            raise ValueError('FROZEN_R14_DECISION_REGRESSION')
        r15=frozen.child(['tools/r15_science.py'],timeout=180)
        match=re.search(r'^R15_SCIENCE_SHA256=([a-f0-9]{64})$',r15,re.M)
        rep=re.search(r'^R15_SCIENCE=(.+)$',r15,re.M)
        if not match or not rep:raise ValueError('MISSING_R15_SCIENCE')
        r15_sha=match.group(1);science=json.loads(rep.group(1))
        for family in ('positive','hard_pointer','no_terminal','missing_goal'):
            d=science['heldout'][family]
            for label in ('ersa_learned','ersa_deterministic','gdn_grid_tuned_rule','gdn_beta1_alpha1_rule'):
                q=d[label]
                if q['correct']+q['false_stop']+q['abstain']!=64:
                    raise ValueError('NONEXHAUSTIVE_SCIENCE:'+family+':'+label)
            print('SCIENCE='+family+';ERSA_LEARNED='+str(d['ersa_learned']['correct'])+
                  ';ERSA_STRONG='+str(d['ersa_deterministic']['correct'])+
                  ';GDN_RULE_GRID='+str(d['gdn_grid_tuned_rule']['correct'])+
                  ';GDN_RULE_FALSE_STOP='+str(d['gdn_grid_tuned_rule']['false_stop']),flush=True)
        if science['heldout']['positive']['ersa_learned']['correct']<52 or science['heldout']['positive']['ersa_learned']['false_stop']:
            raise ValueError('FROZEN_ERSA_UTILITY_REGRESSION')
        if science['heldout']['positive']['ersa_deterministic']['correct']<52:
            raise ValueError('STRONG_DETERMINISTIC_REGRESSION')
        for family in ('no_terminal','missing_goal'):
            if science['heldout'][family]['gdn_grid_tuned_rule']['correct']:
                raise ValueError('FORBIDDEN_UNGROUNDED_GDN_SUCCESS')
        if frozen.manifest('FILES.sha256')!=own:raise ValueError('SOURCE_MUTATION')
        frozen.verify_external(sr,ss);frozen.verify_external(rr,frozen.R2_PINS)
        disposition='PASS_FROZEN_ERSA_AND_GDN_RULE_SCREEN_MECHANICS_ONLY'
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R15_GDN_RULE_BASELINE_SCREEN_R0')
    print('DISPOSITION='+disposition)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R14_PINS='+str(frozen_n)+'/101;SSC_PINS='+str(ssc_n)+'/7;R2_PINS='+str(r2_n)+'/4')
    print('R15_OWN_FILES='+str(own_n))
    print('R14_SCIENCE_SHA256='+r14_sha)
    print('R15_SCIENCE_SHA256='+r15_sha)
    if science:
        print('GDN_GATE_GRID='+str(science['gate_selection']))
        for family,block in science['heldout'].items():
            print('FAMILY='+family+';ERSA_LEARNED='+str(block['ersa_learned'])+
                  ';ERSA_STRONG='+str(block['ersa_deterministic'])+
                  ';GDN_GRID='+str(block['gdn_grid_tuned_rule'])+
                  ';GDN_FIXED='+str(block['gdn_beta1_alpha1_rule']))
    print('GDN_RULE=FLA_MIT_MATH_RECURSION_SELF_IMPLEMENTED;FULL_FLA_LAYER=NOT_EXECUTED')
    print('BENCHMARK_FAIRNESS=NOT_MATCHED_QKV_TRAINING_OR_MEMORY;ARCHITECTURAL_GDN_PARITY=HOLD')
    print('ERSA_NEURAL_SUPERIORITY=HOLD;OPEN_WORLD_SEMANTICS=HOLD;PRODUCTION_RIGHTS=HOLD')
    print('NVME_WRITE_ATTESTATION=NONE;PROJECT_MUTATIONS=NONE_INTENDED;RUNTIME_STATE=TMPFS_ONLY;GPU=NONE')
    print('ELAPSED_S='+str(round(time.monotonic()-started,3)))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(int(bool(error))))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
