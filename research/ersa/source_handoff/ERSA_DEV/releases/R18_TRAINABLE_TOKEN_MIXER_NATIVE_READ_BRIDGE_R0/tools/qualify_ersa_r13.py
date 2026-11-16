"""Fail-closed ERSA R13: frozen R12 verification, test collection, fresh training."""
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
        h,r=line.split('  ',1);rel=pathlib.PurePosixPath(r)
        if (not re.fullmatch('[a-f0-9]{64}',h) or r in mapping or r.startswith('.') or
            '..' in rel.parts or rel.is_absolute() or str(rel)!=r or '\\' in r):
            raise ValueError('UNSAFE_MANIFEST:'+r)
        p=ROOT/rel
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()) or sha(p)!=h:
            raise ValueError('SOURCE_HASH_DRIFT:'+r)
        mapping[r]=h
    return mapping
def verify_external(root,mapping):
    if not root.is_dir() or not (root.resolve().is_relative_to('/mnt/primesauce') or root.resolve().is_relative_to('/dev/shm')):
        raise ValueError('UNSAFE_EXTERNAL_ROOT')
    for rel,h in mapping.items():
        p=root/rel
        if not p.is_file() or p.is_symlink() or sha(p)!=h:raise ValueError('EXTERNAL_PIN_FAILED:'+rel)
def override(key,default):
    if key not in os.environ:return default
    p=pathlib.Path(os.environ[key]).resolve()
    if not p.is_relative_to('/dev/shm'):raise ValueError('OVERRIDE_NOT_TMPFS')
    return p
def child(args,timeout=150):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),CUDA_VISIBLE_DEVICES='-1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=':'.join(map(str,(ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests'))))
    p=subprocess.run([sys.executable,'-B',*args],env=env,cwd=ROOT,text=True,
           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:
        print(p.stdout[-6000:],file=sys.stderr)
        raise RuntimeError('FAILED_CHILD:'+str(args)+':'+str(p.returncode))
    return p.stdout

def main():
    started=time.monotonic(); disposition='HOLD';checks=0;pin=(0,0,0);science='NONE';error=''
    try:
        if not ROOT.is_relative_to('/dev/shm'):raise ValueError('RUNTIME_NOT_TMPFS')
        own=manifest('FILES.sha256');frozen=manifest('FROZEN_R12.sha256')
        if len(frozen)!=87 or not set(frozen).issubset(own) or any(own[k]!=v for k,v in frozen.items()):
            raise ValueError('FROZEN_R12_NOT_PRESERVED')
        sc={k.removeprefix('ssc_predecessor/'):v for k,v in manifest('SSC_PREDECESSOR.sha256').items()}
        if len(sc)!=7:raise ValueError('SSC_SEVEN_PIN_MISMATCH')
        r2=override('ERSA_R2_ROOT',R2_DEFAULT)
        ssc=override('ERSA_SSC_ROOT',SSC_DEFAULT)
        verify_external(r2,R2_PINS);verify_external(ssc,sc)
        pin=(len(frozen),len(sc),len(R2_PINS))
        print('FROZEN_R12_PINS=87/87;SSC=7/7;R2=4/4;OWN='+str(len(own)),flush=True)
        output=child(['-m','pytest','-q','-p','no:cacheprovider','tests'])
        m=re.search(r'(\d+) passed',output);checks=int(m.group(1)) if m else 0
        if checks!=777:raise ValueError('PYTEST_EXPECTED_777_GOT_'+str(checks))
        print('PYTEST='+output.strip().splitlines()[-1],flush=True)
        stress=child(['tools/r3_capacity_stress.py'])
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('8192_SLOT_REGRESSION')
        science_out=child(['tools/r13_science.py'])
        match=re.search(r'R13_SCIENCE_SHA256=([a-f0-9]{64})',science_out)
        if not match:raise ValueError('MISSING_SCIENCE_DIGEST')
        science=match.group(1)
        print(science_out,flush=True)
        if manifest('FILES.sha256')!=own:raise ValueError('POST_EXECUTION_SOURCE_DRIFT')
        verify_external(r2,R2_PINS);verify_external(ssc,sc)
        disposition='PASS_FROZEN_NATIVE_VARIABLE_HOP_LEARNED_HALTING_MECHANICS_ONLY'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R13_TRAINED_BOUNDED_RECURRENT_HALT_R0')
    print('DISPOSITION='+disposition)
    print('PYTEST_PASSED='+str(checks))
    print('FROZEN_R12_PINS='+str(pin[0])+'/87;SSC_PINS='+str(pin[1])+'/7;R2_PINS='+str(pin[2])+'/4')
    print('R13_SCIENCE_SHA256='+science)
    print('C_NATIVE=STATE_EXACT_READ_SCORE_RANK_HALT;HOST=CONTINUE_ORCHESTRATION')
    print('RECURSION=VARIABLE_1_TO_4;MAX_HOPS=4;INFERENCE_STATE=RAM_ONLY;NO_RUNTIME_LEARNING')
    print('LEARNED_SUPERIORITY_TO_STRONG_DETERMINISTIC=HOLD;GDN_PARITY=NOT_TESTED')
    print('SEMANTIC_GENERALIZATION=HOLD;FULL_GDN_REPLACEMENT=HOLD;SSC_OFFICIAL_TRANSPORT=UNPINNED')
    print('PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE;RIGHTS=HOLD')
    print('ELAPSED_S='+format(time.monotonic()-started,'.3f'))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(int(bool(error))))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
