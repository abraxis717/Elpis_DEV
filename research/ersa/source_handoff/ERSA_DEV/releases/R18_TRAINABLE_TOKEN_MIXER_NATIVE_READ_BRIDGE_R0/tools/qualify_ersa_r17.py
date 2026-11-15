"""R17 qualification. Verify frozen R16 and predecessor, exercise new trainable CPU GDN-layer reference."""
from __future__ import annotations
import json,pathlib,re,sys,time,hashlib,os,subprocess,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import qualify_ersa_r14 as core

FROZEN_N=114
NEW_TOTAL_TESTS=1133
R16_PRIOR='166c93f55c607e03155d5e559c4df3c5900e8b585fd2ef9ccf28e12e9fe70d79'

def child_to_tmpfs(args,timeout=120):
    """Run each gate in its own process with non-pipe tmpfs stdout.

    The inherited R14 subprocess PIPE collector can wait for descriptors
    held by native test descendants.  A regular tmpfs file does not require
    pipe EOF and lets both independent receipts be checked in one runner.
    """
    env=dict(os.environ,
        PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=':'.join(str(x) for x in (
            ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests')))
    with tempfile.TemporaryFile(mode='w+t',dir='/dev/shm') as log:
        run=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
            text=True,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        log.seek(0)
        output=log.read()
    if run.returncode:
        print(output[-10000:],file=sys.stderr,flush=True)
        raise RuntimeError('CHILD_FAILURE:'+str(args)+':'+str(run.returncode))
    return output

def main():
    t0=time.monotonic();frozen_n=own_n=ssc_n=r2_n=tests=0
    science=None;science_sha='NOT_EXECUTED';error='';disposition='HOLD'
    try:
        if not ROOT.resolve().is_relative_to('/dev/shm/'):
            raise ValueError('QUALIFICATION_TMPFS_ONLY')
        own=core.manifest('FILES.sha256')
        frozen=core.manifest('FROZEN_R16.sha256')
        if len(frozen)!=FROZEN_N or any(own.get(name)!=h for name,h in frozen.items()):
            raise ValueError('FROZEN_R16_SOURCE_DRIFT')
        frozen_n=len(frozen);own_n=len(own)
        ssc=core.manifest('SSC_PREDECESSOR.sha256')
        if len(ssc)!=7 or any(own.get(n)!=h for n,h in ssc.items()):
            raise ValueError('FROZEN_SSC_PACKAGE_DRIFT')
        ssc_n=len(ssc)
        rr=core.source_root('ERSA_R2_ROOT',core.R2_DEFAULT)
        core.verify_external(rr,core.R2_PINS);r2_n=len(core.R2_PINS)
        print('GATE=R17_READONLY_STATE_START',flush=True)
        isolation=child_to_tmpfs(['tools/r17_readonly_state_probe.py'],timeout=40)
        if 'READ_ONLY_STATE_BITWISE_PARITY=80/80' not in isolation:
            raise ValueError('READ_ONLY_STATE_INVARIANCE_NOT_ESTABLISHED')
        print('READ_ONLY_STATE_BITWISE_PARITY=80/80',flush=True)
        print('GATE=R17_READONLY_STATE_END',flush=True)
        print('GATE=R17_SCIENCE_START',flush=True)
        output=child_to_tmpfs(['tools/r17_science.py'],timeout=90)
        print('GATE=R17_SCIENCE_END',flush=True)
        m=re.search(r'^R17_SCIENCE_SHA256=([a-f0-9]{64})$',output,re.M)
        s=re.search(r'^R17_SCIENCE=(.+)$',output,re.M)
        if not m or not s:raise ValueError('MISSING_SCIENCE_REPORT')
        science=json.loads(s.group(1));science_sha=m.group(1)
        if (len(science['runs'])!=3 or science['official_fla_module_executed'] or
            science['official_fla_gpu_kernel_executed'] or science['trainable_weights']!=4930):
            raise ValueError('WRONG_LAYER_CLAIM_OR_SHAPE')
        for item in science['runs']:
            tr=item['training']
            if not(tr['end']<tr['start']-.05 and tr['parameters']==4930 and tr['state_bytes']==2048):
                raise ValueError('TRAINING_OR_MEMORY_CHECK_FAILED')
            for family,counts in item['heldout'].items():
                if sum(counts.get(k,-1) for k in ('correct','false_stop','abstain'))!=64:
                    raise ValueError('INVALID_HELDOUT_METRICS:'+family)
            print('SEED='+str(tr['seed'])+';LOSS='+str(tr['start'])+'->'+str(tr['end'])+
                  ';POSITIVE='+str(item['heldout']['positive'])+';HARD_POINTER='+
                  str(item['heldout']['hard_pointer']),flush=True)
        print('GATE=R17_TESTS_START',flush=True)
        suite=child_to_tmpfs(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=90)
        result=re.search(r'(\d+) passed',suite)
        tests=int(result.group(1)) if result else 0
        if tests!=NEW_TOTAL_TESTS:
            raise ValueError('TEST_COUNT_DRIFT:'+str(tests))
        print('PYTEST='+suite.strip().splitlines()[-1],flush=True)
        print('GATE=R17_TESTS_END',flush=True)
        if core.manifest('FILES.sha256')!=own:
            raise ValueError('PACKAGE_SOURCE_MUTATED')
        core.verify_external(rr,core.R2_PINS)
        disposition='PASS_CPU_GDN_V1_STRUCTURAL_REFERENCE_TRAINING_AND_NUMERICAL_MECHANICS_ONLY'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R17_CPU_GDN_LAYER_STRUCTURAL_REFERENCE_R0')
    print('DISPOSITION='+disposition)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R16_PINS='+str(frozen_n)+'/114;SSC_PINS='+str(ssc_n)+'/7;R2_PINS='+str(r2_n)+'/4')
    print('R17_OWN_FILES='+str(own_n))
    print('R16_PRIOR_SCIENCE_SHA256='+R16_PRIOR)
    print('R16_SCIENCE_REPLAY=PRIOR_ONLY_NOT_EXECUTED')
    print('R17_SCIENCE_SHA256='+science_sha)
    print('MODEL=GDN_V1_COMPONENTS_INDEPENDENT_CPU_PYTORCH_REFERENCE;OFFICIAL_FLA_LAYER=NOT_EXECUTED')
    print('TRAINING=3_SEEDS_256_EPISODES_30_EPOCHS;WEIGHTS=4930;STATE_FP32_BYTES=2048;CONV_CACHE_BYTES=768')
    if science:
        for item in science['runs']:
            tr=item['training']
            print('SEED='+str(tr['seed'])+';LOSS_START='+str(tr['start'])+';LOSS_END='+str(tr['end']))
            for family,metric in item['heldout'].items():
                print('FAMILY='+family+';SEED='+str(tr['seed'])+';CORRECT='+str(metric['correct'])+
                      ';FALSE_STOP='+str(metric['false_stop'])+';ABSTAIN='+str(metric['abstain']))
    print('INPUT_MATCHING=SYNTHETIC_24_RECORD_TASK;MEMORY_PARAMETER_COMPUTE_MATCHING=NO')
    print('GDN_ARCHITECTURAL_PARITY=HOLD;ERSA_NEURAL_SUPERIORITY=HOLD;OPEN_WORLD_SEMANTICS=HOLD')
    print('CHECKPOINTS=NONE;PROJECT_MUTATIONS=NONE_INTENDED;INFERENCE_STATE=RAM_ONLY;GPU=NONE')
    print('NVME_WRITE_ATTESTATION=NONE;TRUST=RESEARCH_ONLY;RIGHTS=HOLD')
    print('ELAPSED_S='+str(round(time.monotonic()-t0,3)))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(int(bool(error))))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
