"""R35 reproducible two-gate qualification with frozen lineage + source promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('ERSA_R35_ROOT','/mnt/primesauce/RALPH/DEV/Hermes/ERSA_DEV'))
RECEIPTS=Path(os.environ.get('ERSA_R35_RECEIPTS','/dev/shm/ersa-r35-receipts'))
DEST=ROOT/'worktrees/R35_DONOR_BACKED_SLOT_WRITE_READ_R0'
R18=ROOT/'releases/R18_TRAINABLE_TOKEN_MIXER_NATIVE_READ_BRIDGE_R0'
SEEDS=(3501,3502)
PINS={
  18:'1ca4d9268fb74287c4395eb79949eea7bf41e3103a144a252e2993505690c46f',
  19:'8132461a403e467e8c3479a82fc7782600c4c55f023519e6ba63e51a4f69bed9',
  20:'d0136472c60677c4bd169c0628fb2b5275bcd2273c1e150050181bbdee09f3be',
  21:'7b62a2b527c77204d1c89483c868a43f861b338677791b79acbceade68905798',
  22:'084367739e1b5b0ef8d4e7eb646b2fe06f8d2b8b81f0195404de77308cf0d42d',
  23:'6fd9082071d0e78f99b19d81661ac7b0262726ab9fbe1bf4f51468b5f104569e',
  24:'f21d1e91df99440745c3e7c8f3ecda706cf535e9648a1c5f482a98bdaf9a87b0',
  25:'362e8f6a6664e00005def4cf64fe3bad9683b3caefde96fd5bc3190fffb8e2f3',
  26:'6be27ee89ae33509174275712a44a50c8a838bfa140372964c0cca01d2441c8e',
  27:'e02b7fca7bae5370e218253e7bc1a1c3fece97bdd47413d89ebcd7453e0b4473',
  28:'e80c7a0d752a94155058179bee429ec2496e6df2069f4d8cd45cd8c44c310e89',
  29:'4a304801396dc9b81ed49915ddb7b0f6a9827b15425ab85123e4d01265c74fac',
  30:'cadce265b237a3f1fb278f0806339ca477ab293fea24f03df1fda4ac977703a0',
  31:'6a8c1f11153bdafacc84fd678d3018597ac720f91fbb8cb634f9a1fc6cc6af83',
  32:'95443d0ab5840cb0f64d4703154d0af67060095e06a6860b439fe573842847f6',
  33:'fe9cd30eedcc62592ec5075c67592515d720993a122ac1dcde3b4c32c17555f6',
  34:'b77ecb121105c21cfa92a8c42a05878daf00c87a3f7d4ab14f07b5a3e8631f32',
}


def digest(b):return hashlib.sha256(b).hexdigest()
def canonical(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def verify(tree,manifest,expected,count):
    p=tree/manifest
    if p.is_symlink() or not p.is_file() or digest(p.read_bytes())!=expected:
        raise RuntimeError(f'FROZEN_MANIFEST_MISMATCH:{p}')
    lines=p.read_text().splitlines()
    if len(lines)!=count:raise RuntimeError('PIN_COUNT_MISMATCH:'+str(p))
    entries=set()
    for line in lines:
        match=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        if not match:raise RuntimeError('BAD_MANIFEST_LINE')
        sha,rel=match.groups()
        rp=Path(rel)
        if rp.is_absolute() or '..' in rp.parts or rel in entries:
            raise RuntimeError('BAD_MANIFEST_PATH')
        entries.add(rel)
        f=tree/rp
        if not f.is_file() or f.is_symlink() or digest(f.read_bytes())!=sha:
            raise RuntimeError('SOURCE_PIN_MISMATCH:'+str(f))
    return expected


def preflight():
    if not HERE.is_relative_to('/dev/shm'):
        raise RuntimeError('EXECUTION_SOURCE_NOT_TMPFS')
    if RECEIPTS.is_symlink() or not RECEIPTS.resolve().is_relative_to('/dev/shm') or not RECEIPTS.is_dir():
        raise RuntimeError('RECEIPTS_NOT_TMPFS')
    if not (str(ROOT).startswith('/mnt/primesauce/') or str(ROOT).startswith('/dev/shm/')):
        raise RuntimeError('INVALID_ROOT')
    pins={'R18':verify(R18,'FILES.sha256',PINS[18],129)}
    for phase in range(19,35):
        candidates=list((ROOT/'worktrees').glob(f'R{phase}_*'))
        if len(candidates)!=1:
            raise RuntimeError('MISSING_OR_DUPLICATED_FROZEN_PHASE:'+str(phase))
        pins[f'R{phase}']=verify(candidates[0],f'R{phase}_FILES.sha256',PINS[phase],7)
    own=HERE/'R35_FILES.sha256'
    pins['R35']=verify(HERE,'R35_FILES.sha256',digest(own.read_bytes()),10)
    listed={line.split('  ',1)[1] for line in own.read_text().splitlines()}
    files={str(f.relative_to(HERE)) for f in HERE.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
    if files!=listed|{'R35_FILES.sha256'}:
        raise RuntimeError('R35_UNEXPECTED_FILE_SET:'+repr(files.symmetric_difference(listed|{'R35_FILES.sha256'})))
    return pins


def path(name):return RECEIPTS/(name+'.json')

def write(name,pins,body):
    p=path(name)
    if p.exists() or p.is_symlink():raise RuntimeError('DUPLICATE_RECEIPT:'+name)
    obj={'phase':'R35','name':name,'pins':pins,'body':body}
    obj['sha256']=digest(canonical(obj))
    p.write_bytes(canonical(obj)+b'\n')

def read(name,pins):
    p=path(name)
    if not p.is_file() or p.is_symlink():raise RuntimeError('MISSING_RECEIPT:'+name)
    obj=json.loads(p.read_bytes())
    sha=obj.pop('sha256',None)
    if sha!=digest(canonical(obj)) or obj['phase']!='R35' or obj['name']!=name or obj['pins']!=pins:
        raise RuntimeError('BAD_RECEIPT:'+name)
    return obj['body']

def paths():return [HERE,R18,R18/'src',R18/'native',R18/'ssc_predecessor/src',R18/'tests']

def environment():
    env=dict(os.environ)
    env.update(PYTHONPATH=':'.join(map(str,paths())),PYTHONDONTWRITEBYTECODE='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TMPDIR='/dev/shm',HOME=str(RECEIPTS),
               XDG_CACHE_HOME=str(RECEIPTS),CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env

def gate(name,seed=None):
    pins=preflight()
    if name=='tests':
        proc=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
            'tests/test_r35_slot_write.py','tests/test_r35_donor_parity.py'],cwd=HERE,env=environment(),
            capture_output=True,text=True,timeout=120)
        if proc.returncode!=0 or '22 passed' not in proc.stdout:
            raise RuntimeError('R35_TEST_FAIL:'+proc.stdout[-1200:]+proc.stderr[-600:])
        write('tests',pins,{'passed':22})
        print('R35_TESTS=22/22;DONOR_GSA_REFERENCE_AND_SPARSE_WRITE=PASS',flush=True)
    elif name=='science-seed':
        if seed not in SEEDS:raise RuntimeError('UNEXPECTED_SEED')
        sys.path[:0]=[str(p) for p in paths()]
        from science_r35 import one_seed
        v=one_seed(seed)
        if not (v['train_last']<v['train_first']*.65 and v['heldout']<v['no_write']*.85):
            raise RuntimeError('TRAINING_OR_HELDOUT_WRITE_UTILITY_NOT_QUALIFIED')
        write('science-'+str(seed),pins,v)
        print('SEED='+str(seed)+';LOSS='+str(v['train_first'])+'->'+str(v['train_last'])+
              ';HELDOUT='+str(v['heldout'])+';NO_WRITE='+str(v['no_write']),flush=True)
    elif name=='science':
        runs=[read('science-'+str(s),pins) for s in SEEDS]
        obj={'phase':'R35','runs':runs,'claims':'CPU_REFERENCE_ONLY'}
        sha=digest(canonical(obj))
        write('science',pins,{'sha256':sha,'runs':runs})
        print('R35_SCIENCE_SHA256='+sha,flush=True)
    elif name=='ready':
        if read('tests',pins)['passed']!=22 or len(read('science',pins)['runs'])!=2:
            raise RuntimeError('GATES_INCOMPLETE')
        print('R35_PROMOTION_READY=PASS',flush=True)
    elif name=='finalize':
        if read('tests',pins)['passed']!=22:raise RuntimeError('TEST_RECEIPT_INVALID')
        science=read('science',pins)
        if verify(DEST,'R35_FILES.sha256',pins['R35'],10)!=pins['R35']:
            raise RuntimeError('PERSISTENT_PROMOTION_INVALID')
        print('=== SEND BACK ===')
        print('PHASE=ERSA_R35_DONOR_BACKED_SLOT_WRITE_READ_R0')
        print('DISPOSITION=PASS_FLA_GSA_REFERENCE_SPARSE_WRITES_WITH_R18_READ_MECHANICS_ONLY')
        print('R35_TESTS=22/22;R18_PINS=129/129;R19_R34_PINS=7/7_EACH')
        print('R35_PATCH_SHA256='+pins['R35']+';R35_SCIENCE_SHA256='+science['sha256'])
        for r in science['runs']:
            print('SEED='+str(r['seed'])+';TRAIN='+str(r['train_first'])+'->'+str(r['train_last'])+
                  ';HELDOUT='+str(r['heldout'])+';NO_WRITE='+str(r['no_write']))
        print('SOURCE_DONOR=FLA_GSA_NAIVE_MIT_ADAPTED;ROUTER=RAVEN_STYLE_TOPK')
        print('STATE=EXPLICIT_VOLATILE_DUAL_MATRIX;NO_C_WRITE_AUTHORITY;NATIVE_KERNEL_PARITY=HOLD')
        print('CHUNK_PARITY=SPLIT_CONTINUATION_CPU_ONLY;GDN_PARITY=HOLD;PRODUCTION_TRUST=HOLD')
        print('PERSISTENT_R35_SOURCE=VERIFIED;WORKTREE='+str(DEST))
        print('CHECKPOINTS=0;PROJECT_MUTATIONS=EXPLICIT_SOURCE_PROMOTION_ONLY;OS_WRITE_ATTESTATION=NONE')
        print('EXIT_STATUS=0',flush=True)
    else:raise RuntimeError('INVALID_GATE')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gate',required=True,choices=['tests','science-seed','science','ready','finalize'])
    p.add_argument('--seed',type=int)
    a=p.parse_args()
    try:gate(a.gate,a.seed)
    except Exception as e:
        print('=== SEND BACK ===\nPHASE=ERSA_R35_DONOR_BACKED_SLOT_WRITE_READ_R0\nDISPOSITION=HOLD')
        print('BLOCKERS='+type(e).__name__+':'+str(e)[-1300:])
        print('EXIT_STATUS=1',flush=True)
        sys.exit(1)
