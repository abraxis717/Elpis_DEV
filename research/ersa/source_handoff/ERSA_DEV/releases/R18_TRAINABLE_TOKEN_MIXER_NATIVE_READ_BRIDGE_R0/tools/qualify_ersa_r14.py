"""ERSA R14: verify frozen sources, collect all tests, stress and blinded science."""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,sys,time
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
    src=ROOT/name
    if not src.is_file() or src.is_symlink():raise ValueError('MISSING_MANIFEST:'+name)
    seen={}
    for line in src.read_text().splitlines():
        h,r=line.split('  ',1);rel=pathlib.PurePosixPath(r)
        if (not re.fullmatch('[a-f0-9]{64}',h) or r in seen or r.startswith('.') or
            '..' in rel.parts or rel.is_absolute() or str(rel)!=r or '\\' in r):
            raise ValueError('UNSAFE_MANIFEST:'+r)
        target=ROOT/rel
        if (not target.is_file() or target.is_symlink() or
            not target.resolve().is_relative_to(ROOT.resolve()) or sha(target)!=h):
            raise ValueError('FILE_DRIFT:'+r)
        seen[r]=h
    return seen

def source_root(key,default):
    if key not in os.environ:return default
    p=pathlib.Path(os.environ[key]).resolve()
    if not p.is_relative_to('/dev/shm'):raise ValueError('OVERRIDE_OUTSIDE_TMPFS')
    return p

def verify_external(path,pins):
    if not path.is_dir() or not (path.resolve().is_relative_to('/mnt/primesauce') or
                                  path.resolve().is_relative_to('/dev/shm')):
        raise ValueError('INVALID_EXTERNAL_ROOT')
    for rel,h in pins.items():
        target=path/rel
        if not target.is_file() or target.is_symlink() or sha(target)!=h:
            raise ValueError('EXTERNAL_SOURCE_DRIFT:'+rel)

def child(args,timeout=200):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),CUDA_VISIBLE_DEVICES='-1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=':'.join(str(x) for x in (
            ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests')))
    p=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:
        print(p.stdout[-10000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAILURE:'+str(args)+':'+str(p.returncode))
    return p.stdout

def main():
    started=time.monotonic();count=0;frozen_n=0;scn=0;r2n=0;science='NONE';
    duplicate_stop=-1;parity=-1;error='';disposition='HOLD';own_count=0
    try:
        if not ROOT.resolve().is_relative_to('/dev/shm'):
            raise ValueError('QUALIFICATION_MUST_RUN_IN_TMPFS')
        own=manifest('FILES.sha256');frozen=manifest('FROZEN_R13.sha256')
        own_count=len(own)
        if len(frozen)!=94 or any(own.get(k)!=v for k,v in frozen.items()):
            raise ValueError('FROZEN_R13_DRIFT')
        frozen_n=len(frozen)
        ssc={k.removeprefix('ssc_predecessor/'):v for k,v in manifest('SSC_PREDECESSOR.sha256').items()}
        if len(ssc)!=7:raise ValueError('SSC_PIN_COUNT')
        sr=source_root('ERSA_SSC_ROOT',SSC_DEFAULT)
        rr=source_root('ERSA_R2_ROOT',R2_DEFAULT)
        verify_external(sr,ssc);verify_external(rr,R2_PINS)
        scn=len(ssc);r2n=len(R2_PINS)
        output=child(['-m','pytest','-q','-p','no:cacheprovider','tests'])
        found=re.search(r'(\d+) passed',output)
        count=int(found.group(1)) if found else -1
        if count!=975:raise ValueError('EXPECTED_975_TESTS_GOT:'+str(count))
        print('PYTEST='+output.strip().splitlines()[-1],flush=True)
        stress=child(['tools/r3_capacity_stress.py'])
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('R3_CAPACITY_REGRESSION')
        result=child(['tools/r14_science.py'],timeout=180)
        match=re.search(r'^R14_SCIENCE_SHA256=([a-f0-9]{64})$',result,re.M)
        report=re.search(r'^R14_SCIENCE=(.+)$',result,re.M)
        if not match or not report:raise ValueError('MISSING_SCIENCE_REPORT')
        science=match.group(1);data=json.loads(report.group(1))
        parity=data['exact_host_parity'];duplicate_stop=data['host_duplicate_stop_prevented']
        if parity+duplicate_stop!=1024:raise ValueError('SCIENCE_DIFFERENTIAL_INCOMPLETE')
        for family in ('positive','hard_pointer','missing_goal','no_terminal'):
            d=data['heldout'][family]
            print('SCIENCE_FAMILY='+family+';LEARNED_CORRECT='+str(d['learned']['correct'])+
                  ';LEARNED_FALSE_STOP='+str(d['learned']['false_stop'])+
                  ';STRONG_CORRECT='+str(d['strong']['correct'])+
                  ';NATIVE_MEDIAN_US='+str(d['learned']['native_median_us'])+
                  ';HOST_MEDIAN_US='+str(d['learned']['host_median_us']),flush=True)
        if manifest('FILES.sha256')!=own:
            raise ValueError('POST_EXECUTION_SOURCE_DRIFT')
        verify_external(sr,ssc);verify_external(rr,R2_PINS)
        disposition='PASS_FROZEN_R13_SINGLE_CALL_C_RECURRENCE_DIFFERENTIAL_ONLY'
    except BaseException as e:
        error=type(e).__name__+':'+str(e)
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R14_SINGLE_CALL_C_NATIVE_RECURRENCE_R0')
    print('DISPOSITION='+disposition)
    print('PYTEST_PASSED='+str(count))
    print('FROZEN_R13_PINS='+str(frozen_n)+'/94;SSC_PINS='+str(scn)+'/7;R2_PINS='+str(r2n)+'/4')
    print('R14_OWN_FILES='+str(own_count))
    print('R14_SCIENCE_SHA256='+science)
    print('R14_HOST_NATIVE_EXACT_PARITY='+str(parity)+'/1024;R13_UNSAFE_DUPLICATE_STOPS_REJECTED='+str(duplicate_stop))
    print('NATIVE_EXECUTION=ONE_C_CALL_SELECT_CONTINUE_HALT_ON_GENERATION_PINNED_C_STATE_SNAPSHOT')
    print('STATE_AUTHORITY=R11_C_RAM;HOST_SNAPSHOT_AND_ADMISSION=RETAINED;HOPS_MAX=4;SLOTS_MAX=8192')
    print('NEURAL=FROZEN_14_TRAINABLE_WEIGHTS_OFFLINE;LEARNED_SUPERIORITY=HOLD')
    print('GDN_BASELINE=NOT_PRESENT;GDN_PARITY=NOT_TESTED;SEMANTIC_GENERALIZATION=HOLD')
    print('SSC_OFFICIAL_TRANSPORT=UNPINNED;GPU=NONE;MODEL_SCALE=NOT_QUALIFIED')
    print('PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE;RIGHTS=HOLD;PRODUCTION_TRUST=HOLD')
    print('ELAPSED_S='+format(time.monotonic()-started,'.3f'))
    if error:print('BLOCKERS='+error)
    print('EXIT_STATUS='+str(bool(error) and 1 or 0))
    return int(bool(error))
if __name__=='__main__':sys.exit(main())
