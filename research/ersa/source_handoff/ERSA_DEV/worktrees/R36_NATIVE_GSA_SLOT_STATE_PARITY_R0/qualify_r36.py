"""R36 pin-gated source-only CPU native GSA slot recurrence qualification."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('ERSA_R36_ROOT','/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV'))
RECEIPTS=Path(os.environ.get('ERSA_R36_RECEIPTS','/dev/shm/ersa-r36-receipts'))
R18=ROOT/'releases/R18_TRAINABLE_TOKEN_MIXER_NATIVE_READ_BRIDGE_R0'
R35=ROOT/'worktrees/R35_DONOR_BACKED_SLOT_WRITE_READ_R0'
DEST=ROOT/'worktrees/R36_NATIVE_GSA_SLOT_STATE_PARITY_R0'
R35_MANIFEST='d323cb1f7c316af477da2102e960b0830b5c0f6f772b5ddd90cd5742cbebd322'
SEEDS=(3601,3602,3603)

def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def prior_qualifier():
    path=R35/'qualify_r35.py'
    if not path.is_file() or path.is_symlink():raise RuntimeError('MISSING_PINNED_R35_QUALIFIER')
    spec=importlib.util.spec_from_file_location('frozen_r35_qualifier',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify(tree,manifest,digest,count):
    mf=tree/manifest
    if not mf.is_file() or mf.is_symlink() or sha(mf.read_bytes())!=digest:
        raise RuntimeError('MANIFEST_MISMATCH:'+str(mf))
    rows=mf.read_text().splitlines()
    if len(rows)!=count:raise RuntimeError('MANIFEST_COUNT_MISMATCH:'+str(mf))
    names=set()
    for row in rows:
        m=re.fullmatch(r'([a-f0-9]{64})  (.+)',row)
        if not m:raise RuntimeError('BAD_MANIFEST_RECORD')
        dig,rel=m.groups();p=Path(rel)
        if p.is_absolute() or '..' in p.parts or rel in names:raise RuntimeError('UNSAFE_MANIFEST_PATH')
        names.add(rel);fp=tree/p
        if not fp.is_file() or fp.is_symlink() or sha(fp.read_bytes())!=dig:
            raise RuntimeError('PIN_MISMATCH:'+str(fp))
    return digest,names

def preflight():
    if not HERE.is_relative_to('/dev/shm') or not RECEIPTS.resolve().is_relative_to('/dev/shm'):
        raise RuntimeError('EXECUTION_AND_RECEIPTS_MUST_BE_TMPFS')
    if RECEIPTS.is_symlink() or not RECEIPTS.is_dir():raise RuntimeError('MISSING_TMPFS_RECEIPTS')
    if not (str(ROOT).startswith('/mnt/primesauce/') or str(ROOT).startswith('/dev/shm/')):
        raise RuntimeError('ROOT_OUTSIDE_ALLOWED_STORAGE')
    prior=prior_qualifier()
    pins={'R18':verify(R18,'FILES.sha256',prior.PINS[18],129)[0]}
    for phase in range(19,35):
        dirs=list((ROOT/'worktrees').glob(f'R{phase}_*'))
        if len(dirs)!=1:raise RuntimeError('MISSING_FROZEN_SOURCE:'+str(phase))
        pins[f'R{phase}']=verify(dirs[0],f'R{phase}_FILES.sha256',prior.PINS[phase],7)[0]
    pins['R35']=verify(R35,'R35_FILES.sha256',R35_MANIFEST,10)[0]
    file=HERE/'R36_FILES.sha256';md=sha(file.read_bytes())
    _,listed=verify(HERE,'R36_FILES.sha256',md,8)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != listed|{'R36_FILES.sha256'}:
        raise RuntimeError('UNEXPECTED_R36_FILES:'+repr(actual.symmetric_difference(listed|{'R36_FILES.sha256'})))
    pins['R36']=md
    return pins

def receipt_path(name):return RECEIPTS/(name+'.json')
def receipt_write(name,pins,body):
    path=receipt_path(name)
    if path.exists() or path.is_symlink():raise RuntimeError('RECEIPT_ALREADY_PRESENT:'+name)
    record={'phase':'R36','name':name,'pins':pins,'body':body}
    record['sha256']=sha(canonical(record))
    path.write_bytes(canonical(record)+b'\n')

def receipt_read(name,pins):
    path=receipt_path(name)
    if not path.is_file() or path.is_symlink():raise RuntimeError('MISSING_RECEIPT:'+name)
    obj=json.loads(path.read_bytes());digest=obj.pop('sha256','')
    if obj['phase']!='R36' or obj['name']!=name or obj['pins']!=pins or digest!=sha(canonical(obj)):
        raise RuntimeError('BAD_RECEIPT:'+name)
    return obj['body']

def imports():return [HERE,R35,R18,R18/'src',R18/'native',R18/'tests',R18/'ssc_predecessor/src']
def environment():
    env=dict(os.environ)
    env.update(PYTHONPATH=':'.join(map(str,imports())),PYTHONDONTWRITEBYTECODE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TMPDIR='/dev/shm',HOME=str(RECEIPTS),
        XDG_CACHE_HOME=str(RECEIPTS),CUDA_VISIBLE_DEVICES='-1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env

def execute(gate,seed=None):
    pins=preflight()
    if gate=='tests':
        # 52 new R36 tests + 22 frozen R35 tests + 2 frozen R18 C-owner tests.
        files=[str(HERE/'tests/test_r36_native_slot.py'),
               str(R35/'tests/test_r35_donor_parity.py'),
               str(R35/'tests/test_r35_slot_write.py'),
               str(R18/'tests/test_r18_trainable_layer.py')+'::test_r11_native_state_snapshot_bridge_no_generation_mutation']
        proc=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*files],
            cwd=HERE,env=environment(),capture_output=True,text=True,timeout=120)
        if proc.returncode or '76 passed' not in proc.stdout:
            raise RuntimeError('R36_TEST_GATE_FAILED:'+proc.stdout[-1600:]+proc.stderr[-600:])
        receipt_write('tests',pins,{'passed':76,'new':52,'frozen_r35':22,'frozen_r18_native':2})
        print('R36_TESTS=76/76;NEW=52;R35=22;R18_NATIVE=2',flush=True)
    elif gate=='science-seed':
        if seed not in SEEDS:raise RuntimeError('UNKNOWN_SEED')
        sys.path[:0]=[str(p) for p in imports()]
        from r36.native_bridge import build_native
        from science_r36 import one_seed
        lib,tmp=build_native()
        try:out=one_seed(seed,lib)
        finally:
            del lib
            shutil.rmtree(tmp,ignore_errors=True)
        if out['max_output_abs']>3e-6 or out['max_state_abs']>3e-6:
            raise RuntimeError('NATIVE_PARITY_FAILURE:'+repr(out))
        receipt_write('science-'+str(seed),pins,out)
        print('SEED='+str(seed)+';EPISODES=32;MAX_OUTPUT_ABS='+str(out['max_output_abs'])+
              ';MAX_STATE_ABS='+str(out['max_state_abs']),flush=True)
    elif gate=='science':
        runs=[receipt_read('science-'+str(seed),pins) for seed in SEEDS]
        obj={'phase':'R36','runs':runs,'train_to_inference_contract':'C_GSA_MATRIX_ONLY',
             'fla_optimized':'NOT_TESTED','r11_authority':'NOT_UNIFIED'}
        science=sha(canonical(obj))
        receipt_write('science',pins,{'sha256':science,'runs':runs})
        print('R36_SCIENCE_SHA256='+science,flush=True)
    elif gate=='ready':
        tests=receipt_read('tests',pins)
        runs=receipt_read('science',pins)
        if tests['passed']!=76 or len(runs['runs'])!=3:raise RuntimeError('MISSING_QUALIFIED_GATES')
        print('R36_SOURCE_PROMOTION_READY=PASS',flush=True)
    elif gate=='finalize':
        tests=receipt_read('tests',pins);science=receipt_read('science',pins)
        if tests['passed']!=76 or len(science['runs'])!=3:raise RuntimeError('INVALID_RECEIPTS')
        verify(DEST,'R36_FILES.sha256',pins['R36'],8)
        print('=== SEND BACK ===')
        print('PHASE=ERSA_R36_CPU_NATIVE_GSA_SLOT_STATE_R0')
        print('DISPOSITION=PASS_CPU_C_SLOT_MATRIX_RECURRENCE_AND_R35_TORCH_DIFFERENTIAL_ONLY')
        print('R36_TESTS=76/76;R35_TESTS=22/22;R18_NATIVE=2/2')
        print('R18_PINS=129/129;R19_R34_PINS=7/7_EACH;R35_PINS=10/10')
        print('R36_PATCH_SHA256='+pins['R36']+';R36_SCIENCE_SHA256='+science['sha256'])
        for r in science['runs']:
            print('SEED='+str(r['seed'])+';EPISODES='+str(r['episodes'])+
                  ';MAX_OUTPUT_ABS='+str(r['max_output_abs'])+';MAX_STATE_ABS='+str(r['max_state_abs']))
        print('STATE_AUTHORITY=C_CPU_DUAL_MATRIX_PER_INFERENCE_CALL;GENERATION_LOCKED=YES;ATOMIC_REJECT=YES')
        print('TRAINING=TORCH_DONOR;INFERENCE_C_UPDATE_ONLY;HOST_PROJECTS_ROUTES_AND_READS=YES')
        print('R11_ENTITY_AND_C_MATRIX_UNIFIED=HOLD;FLA_TRITON_KERNEL_PARITY=NOT_EXECUTED;GDN_PARITY=HOLD')
        print('NVME_INFERENCE_WRITES=NONE_INTENDED;CHECKPOINTS=0;GPU=NONE;OS_WRITE_ATTESTATION=NONE')
        print('PERSISTENT_R36_SOURCE=VERIFIED;WORKTREE='+str(DEST))
        print('EXIT_STATUS=0',flush=True)
    else:raise RuntimeError('UNKNOWN_GATE')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--gate',required=True,
        choices=('tests','science-seed','science','ready','finalize'))
    a.add_argument('--seed',type=int)
    args=a.parse_args()
    try:execute(args.gate,args.seed)
    except Exception as e:
        print('=== SEND BACK ===\nPHASE=ERSA_R36_CPU_NATIVE_GSA_SLOT_STATE_R0\nDISPOSITION=HOLD')
        print('BLOCKERS='+type(e).__name__+':'+str(e)[-1500:]);print('EXIT_STATUS=1',flush=True)
        sys.exit(1)
