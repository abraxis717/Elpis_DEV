"""Fail-closed ERSA R12 source/pin/experimental qualification. tmpfs only."""
from __future__ import annotations
import hashlib,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
R2_DEFAULT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2')
SSC_DEFAULT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/UnifiedStateConformance_R0')
R2_PINS={
 'src/elpis_ersa/bank_ersa.py':'1212f16b20848a3cba48b7688236cf0b33266d0a82c5e3eea0c5eb7d0e8de6d3',
 'src/elpis_ersa/adapter_ssc.py':'2dd825efec3c6fe38a71a1c1bd26b2fbe6cfb063cc9d4712486c5db7af894b53',
 'src/elpis_ersa/contract_ersa.py':'5afa1abba0a92e38ea36d62c720be448b45b3de67802e8ee4075f255fd81b502',
 'src/elpis_ersa/fixtures_r2.py':'d0fcb5e0c46f8dbdfde4b3b5dce908a10e03bc76bcb2b10d6f1a9d11a2fbf329'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(name):
    file=ROOT/name
    if not file.is_file() or file.is_symlink():raise ValueError('MISSING_MANIFEST:'+name)
    mapping={}
    for line in file.read_text().splitlines():
        chunks=line.split('  ',1)
        if len(chunks)!=2:raise ValueError('INVALID_LINE:'+name)
        h,r=chunks;rel=pathlib.PurePosixPath(r)
        if (not re.fullmatch('[a-f0-9]{64}',h) or r in mapping or
            r.startswith('.') or '..' in rel.parts or rel.is_absolute() or
            str(rel)!=r or '\\' in r):raise ValueError('UNSAFE_MANIFEST_ENTRY:'+r)
        f=ROOT/rel
        if (not f.is_file() or f.is_symlink() or
            not f.resolve().is_relative_to(ROOT.resolve()) or sha(f)!=h):
            raise ValueError('FILE_DRIFT:'+r)
        mapping[r]=h
    return mapping

def verify_external(root,mapping):
    if not root.is_dir() or not root.resolve().is_relative_to('/mnt/primesauce') and not root.resolve().is_relative_to('/dev/shm'):
        raise ValueError('EXTERNAL_ROOT_UNSAFE:'+str(root))
    for p,h in mapping.items():
        f=root/p
        if not f.is_file() or f.is_symlink() or sha(f)!=h:raise ValueError('EXTERNAL_PIN_DRIFT:'+str(f))

def child(args,timeout=120):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
       TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
       CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
       PYTHONPATH=':'.join(map(str,(ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests'))))
    p=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:
        print(p.stdout[-4500:],file=sys.stderr)
        raise RuntimeError('CHILD_FAILED:'+str(args)+':'+str(p.returncode))
    return p.stdout

def external_root(key,default):
    if key not in os.environ:return default
    p=pathlib.Path(os.environ[key]).resolve()
    if not p.is_relative_to('/dev/shm'):raise ValueError('OVERRIDE_OUTSIDE_TMPFS')
    return p

def main():
    t=time.monotonic();state='HOLD';checks=0;pins=(0,0,0);science='NONE';error=''
    try:
        if not ROOT.is_relative_to('/dev/shm'):raise ValueError('RUN_NOT_TMPFS')
        own=manifest('FILES.sha256')
        previous=manifest('FROZEN_R11.sha256')
        if len(previous)!=80:raise ValueError('FROZEN_R11_COUNT')
        if not set(previous).issubset(own):raise ValueError('R11_FILES_OMITTED')
        if not all(previous[p]==own[p] for p in previous):raise ValueError('R11_SOURCE_MUTATED')
        ssc={k.removeprefix('ssc_predecessor/'):v for k,v in manifest('SSC_PREDECESSOR.sha256').items()}
        if len(ssc)!=7:raise ValueError('SSC_SEVEN_PIN_DRIFT')
        rp=external_root('ERSA_R2_ROOT',R2_DEFAULT)
        sp=external_root('ERSA_SSC_ROOT',SSC_DEFAULT)
        verify_external(rp,R2_PINS);verify_external(sp,ssc)
        pins=len(previous),len(ssc),len(R2_PINS)
        print('PINS=R11_80_OF_80;SSC_7_OF_7;R2_4_OF_4;R12_OWN_'+str(len(own)),flush=True)
        output=child(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=120)
        m=re.search(r'(\d+) passed',output)
        checks=int(m.group(1)) if m else 0
        if checks!=732:raise ValueError('EXPECTED_732_GOT_'+str(checks))
        print('PYTEST='+output.strip().splitlines()[-1],flush=True)
        stress=child(['tools/r3_capacity_stress.py'])
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('R3_CAPACITY_REGRESSION')
        scored=child(['tools/r12_science.py'],timeout=120)
        sci=re.search(r'R12_SCIENCE_SHA256=([0-9a-f]{64})',scored)
        if not sci:raise ValueError('R12_SCIENCE_MISSING')
        science=sci.group(1)
        print(scored,flush=True)
        # Mutant and source-drift checks after all code executed.
        if len(manifest('FILES.sha256'))!=len(own) or len(manifest('FROZEN_R11.sha256'))!=80:
            raise ValueError('AFTER_RUN_SOURCE_DRIFT')
        verify_external(rp,R2_PINS);verify_external(sp,ssc)
        state='PASS_FROZEN_NATIVE_STATE_AND_TRAINED_TWO_HOP_PROBE_ONLY'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R12_OFFLINE_LEARNED_SELECTOR_AND_BOUNDED_TWO_HOP_R0')
    print('DISPOSITION='+state)
    print('PYTEST_PASSED='+str(checks))
    print('FROZEN_R11_PINS='+str(pins[0])+'/80;SSC_PREDECESSOR_PINS='+str(pins[1])+'/7;R2_SOURCE_PINS='+str(pins[2])+'/4')
    print('R12_SCIENCE_SHA256='+science)
    print('RUNTIME=C_NATIVE_SSC_EXACT_STATE_PLUS_C_FEATURES_RANK_AND_TRAINED_SCORE;TWO_HOPS_HOST_ORCHESTRATED')
    print('NEURAL=OFFLINE_TRAINED_EIGHT_WEIGHT_HEAD;NO_RUNTIME_LEARNING;GDN_PARITY=NOT_TESTED')
    print('ADMISSION=HOST_CONTROLLED;SEMANTIC_GENERALIZATION=HOLD;TRAINED_RECURRENT_HALT=HOLD')
    print('SOURCE_AUTHORITY=FROZEN;PROJECT_MUTATIONS=NONE_INTENDED;GPU=NONE;OS_NVME_WRITE_ATTESTATION=NONE')
    print('RIGHTS=HOLD;PRODUCTION_TRUST=HOLD;FULL_GDN_REPLACEMENT=HOLD')
    print('ELAPSED_S='+format(time.monotonic()-t,'.3f'))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(bool(error) and 1 or 0))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
