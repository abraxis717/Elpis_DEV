"""ERSA R10 frozen-source+actual-oracle diff gate; only execution in tmpfs."""
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
    f=ROOT/name
    if not f.is_file() or f.is_symlink():raise ValueError('MANIFEST_MISSING:'+name)
    found={}
    for line in f.read_text().splitlines():
        parts=line.split('  ',1)
        if len(parts)!=2:raise ValueError('MANIFEST_FORMAT:'+name)
        h,rel=parts;r=pathlib.PurePosixPath(rel)
        if (not re.fullmatch('[a-f0-9]{64}',h) or rel in found or r.is_absolute() or
            '..' in r.parts or '\\' in rel or rel.startswith('./') or str(r)!=rel):
            raise ValueError('MANIFEST_UNSAFE:'+rel)
        p=ROOT/rel
        if (not p.is_file() or p.is_symlink() or
            not p.resolve().is_relative_to(ROOT.resolve()) or sha(p)!=h):
            raise ValueError('MANIFEST_DRIFT:'+rel)
        found[rel]=h
    return found

def external_root(variable,default):
    if variable not in os.environ:return default
    p=pathlib.Path(os.environ[variable]).resolve()
    if not p.is_relative_to('/dev/shm'):
        raise ValueError('ROOT_OVERRIDE_MUST_BE_TMPFS:'+variable)
    return p

def external_pins(root,pins,label):
    for name,digest in pins.items():
        f=root/name
        if (not f.is_file() or f.is_symlink() or
            not f.resolve().is_relative_to(root.resolve()) or sha(f)!=digest):
            raise ValueError(label+'_SOURCE_DRIFT:'+name)

def child(args,timeout=120):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),CUDA_VISIBLE_DEVICES='-1',
        PYTHONPATH=':'.join((str(ROOT),str(ROOT/'src'),str(ROOT/'native'),
                           str(ROOT/'ssc_predecessor/src'),str(ROOT/'tests'))))
    p=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,text=True,timeout=timeout)
    if p.returncode:
        print(p.stdout[-5000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAILED:'+str(p.returncode)+':'+str(args))
    return p.stdout

def main():
    start=time.monotonic();state='HOLD';r10=r9=r8=r7=ssc=r2=tests=0
    result='NONE';issues=[]
    try:
        if not ROOT.is_relative_to('/dev/shm'):raise ValueError('EXECUTION_NOT_TMPFS')
        own=manifest('FILES.sha256');r10=len(own)
        old=manifest('FROZEN_R9.sha256');r9=len(old)
        r8=len(manifest('FROZEN_R8.sha256'))
        r7=len(manifest('FROZEN_R7.sha256'))
        ssc_manifest=manifest('SSC_PREDECESSOR.sha256');ssc=len(ssc_manifest)
        if (r9,r8,r7,ssc)!=(66,59,43,7):
            raise ValueError('FROZEN_PIN_COUNT_DRIFT:'+repr((r9,r8,r7,ssc)))
        for name in ('native/ersa_authority_r10.c','native/ersa_authority_r10.py',
                     'integration/ssc_native_r10.py','tests/test_r10_native_authority.py',
                     'tools/qualify_ersa_r10.py','README_R10_CANDIDATE.md'):
            if name not in own:raise ValueError('UNPINNED_R10_SOURCE:'+name)
        r2root=external_root('ERSA_R2_ROOT',R2_DEFAULT)
        scroot=external_root('ERSA_SSC_ROOT',SSC_DEFAULT)
        sscpins={k.replace('ssc_predecessor/',''):v for k,v in ssc_manifest.items()}
        external_pins(r2root,R2_PINS,'R2');r2=len(R2_PINS)
        external_pins(scroot,sscpins,'SSC')
        print('GATE=PINS_PASS;R9=66;R8=59;R7=43;SSC=7;R2=4;R10='+str(r10),flush=True)
        output=child(['-m','pytest','-q','-p','no:cacheprovider','tests'])
        m=re.search(r'(\d+) passed',output)
        tests=int(m.group(1)) if m else 0
        if tests!=604:raise ValueError('EXPECTED_604_TESTS_GOT_'+str(tests))
        print('PYTEST='+output.splitlines()[-1],flush=True)
        cap=child([str(ROOT/'tools/r3_capacity_stress.py')])
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in cap:
            raise ValueError('FROZEN_CAPACITY_STRESS_FAIL')
        for x in ('FILES.sha256','FROZEN_R9.sha256','FROZEN_R8.sha256','FROZEN_R7.sha256',
                  'SSC_PREDECESSOR.sha256','FROZEN_R6.sha256','FROZEN_R5.sha256'):
            manifest(x)
        external_pins(r2root,R2_PINS,'R2_AFTER');external_pins(scroot,sscpins,'SSC_AFTER')
        evidence=dict(tests=tests,manifests={x:sha(ROOT/x) for x in
            ('FILES.sha256','FROZEN_R9.sha256','FROZEN_R8.sha256','SSC_PREDECESSOR.sha256')},
            source_pins={'R2':R2_PINS,'SSC':sscpins},
            scope='NATIVE_BOUNDED_CAS_SESSION_STATE_HOST_EXECUTED_14OP_ORACLE_DIFFERENTIAL',
            blockers='NOT_ALL_C_NATIVE_OPERATIONS;SSC_OFFICIAL_TRANSPORT_UNPINNED;SEMANTIC_SELECTION;NEURAL_RUNTIME')
        result=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        state='PASS_NATIVE_CAS_STATE_AUTHORITY_AND_14OP_HOST_TRANSITION_DIFFERENTIAL_ONLY'
    except BaseException as e:
        issues.append(type(e).__name__+':'+str(e))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R10_NATIVE_CAS_STATE_AUTHORITY_R0')
    print('DISPOSITION='+state)
    print('PYTEST_PASSED='+str(tests))
    print(f'FROZEN_R9_PINS={r9}/66;FROZEN_R8_PINS={r8}/59;FROZEN_R7_PINS={r7}/43')
    print(f'SSC_PREDECESSOR_PINS={ssc}/7;R2_SOURCE_PINS={r2}/4;R10_OWN_FILES={r10}')
    print('SSC_14_OPS=HOST_TRANSITIONS_WITH_C_OWNED_STATE;ORACLE_VALID_TRACE_SEEDS=18;RANDOM_TRACE_SEEDS=12')
    print('NATIVE_C_IMAGE=FIXED_MAX_16MIB_DEFAULT;CAS_GENERATION;PTHREAD_SERIALIZED_COPY;NO_DISK_IO')
    print('PROPOSAL_IDENTITY_LIVENESS_CAP=8192;ALL_C_NATIVE_OPS=HOLD;FULL_LONG_HORIZON_PARITY=HOLD')
    print('SSC_OFFICIAL_FIXTURE_TRANSPORT=UNPINNED;UNSCOPED_SEMANTIC_SELECTION=HOLD;NEURAL_RUNTIME=HOLD')
    print('SOURCE_AUTHORITY=FROZEN;PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE')
    print('RIGHTS=HOLD;PRODUCTION_TRUST=HOLD')
    print('RESULT_SHA256='+result)
    print('ELAPSED_S='+format(time.monotonic()-start,'.3f'))
    if issues:print('BLOCKERS='+';'.join(issues))
    print('EXIT_STATUS='+('1' if issues else '0'))
    return bool(issues)
if __name__=='__main__':sys.exit(main())
