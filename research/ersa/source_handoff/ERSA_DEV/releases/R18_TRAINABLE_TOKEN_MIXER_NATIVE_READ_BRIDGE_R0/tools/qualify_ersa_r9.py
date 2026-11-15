"""Isolated ERSA R9 qualification; no canonical source mutation."""
import hashlib,json,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
R2_DEFAULT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2')
SSC_DEFAULT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/UnifiedStateConformance_R0')
R2_PINS={
 'src/elpis_ersa/bank_ersa.py':'1212f16b20848a3cba48b7688236cf0b33266d0a82c5e3eea0c5eb7d0e8de6d3',
 'src/elpis_ersa/adapter_ssc.py':'2dd825efec3c6fe38a71a1c1bd26b2fbe6cfb063cc9d4712486c5db7af894b53',
 'src/elpis_ersa/contract_ersa.py':'5afa1abba0a92e38ea36d62c720be448b45b3de67802e8ee4075f255fd81b502',
 'src/elpis_ersa/fixtures_r2.py':'d0fcb5e0c46f8dbdfde4b3b5dce908a10e03bc76bcb2b10d6f1a9d11a2fbf329',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(name):
    f=ROOT/name
    if not f.is_file() or f.is_symlink():raise ValueError('MISSING_MANIFEST:'+name)
    found={}
    for line in f.read_text().splitlines():
        parts=line.split('  ',1)
        if len(parts)!=2:raise ValueError('MANIFEST_FORMAT')
        h,rel=parts
        r=pathlib.PurePosixPath(rel)
        if (not re.fullmatch('[a-f0-9]{64}',h) or rel in found or
            r.is_absolute() or '..' in r.parts or '\\' in rel or
            rel.startswith('./')):
            raise ValueError('MANIFEST_UNSAFE_OR_DUPLICATE:'+rel)
        source=ROOT/rel
        if (not source.is_file() or source.is_symlink() or
            not source.resolve().is_relative_to(ROOT.resolve()) or
            sha(source)!=h):
            raise ValueError('MANIFEST_DRIFT:'+rel)
        found[rel]=h
    return found

def envroot(name,default):
    if name not in os.environ:return default
    root=pathlib.Path(os.environ[name]).resolve()
    if not root.is_relative_to('/dev/shm/'):
        raise ValueError('UNSAFE_ROOT_OVERRIDE:'+name)
    return root

def pins(root,expected,label):
    for rel,h in expected.items():
        p=root/rel
        if (not p.is_file() or p.is_symlink() or
            not p.resolve().is_relative_to(root.resolve()) or sha(p)!=h):
            raise ValueError('SOURCE_PIN_'+label+':'+rel)

def child(cmd,timeout=120):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
      TMPDIR='/dev/shm',CUDA_VISIBLE_DEVICES='-1',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
      PYTHONPATH=':'.join((str(ROOT),str(ROOT/'src'),str(ROOT/'native'),
                           str(ROOT/'ssc_predecessor/src'))))
    proc=subprocess.run([sys.executable,'-B',*cmd],cwd=ROOT,env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,timeout=timeout)
    if proc.returncode:
        print(proc.stdout[-7000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAIL:'+str(proc.returncode)+':'+cmd[0])
    return proc.stdout

def main():
    started=time.monotonic();tests=0;own=r8=r7=ssc=r2=0;state='HOLD';issues=[];digest='NONE'
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise ValueError('EXECUTION_MUST_USE_TMPFS')
        ownfiles=manifest('FILES.sha256');own=len(ownfiles)
        old=manifest('FROZEN_R8.sha256');r8=len(old)
        back=manifest('FROZEN_R7.sha256');r7=len(back)
        third=manifest('SSC_PREDECESSOR.sha256');ssc=len(third)
        if (r8,r7,ssc)!=(59,43,7):
            raise ValueError('FROZEN_FILE_COUNT_MISMATCH:'+repr((r8,r7,ssc)))
        if not all(x in ownfiles for x in ('native/ersa_state_r9.c',
                'native/ersa_state_r9.py','integration/ssc_r9.py',
                'tests/test_r9_ssc_guard_and_native_state.py')):
            raise ValueError('NEW_CODE_MISSING_FROM_MANIFEST')
        r2root=envroot('ERSA_R2_ROOT',R2_DEFAULT)
        scroot=envroot('ERSA_SSC_ROOT',SSC_DEFAULT)
        pins(r2root,R2_PINS,'R2');r2=len(R2_PINS)
        ssc_ext={rel.replace('ssc_predecessor/',''):h for rel,h in third.items()}
        pins(scroot,ssc_ext,'SSC_EXTERNAL')
        print('QUAL_STAGE=FROZEN_PINS_PASS;R8=59;R7=43;SSC=7;R2=4;OWN='+str(own),flush=True)
        result=child(['-m','pytest','-q','-p','no:cacheprovider','tests'])
        match=re.search(r'(\d+) passed',result)
        tests=int(match.group(1)) if match else 0
        if tests!=534:raise ValueError('PYTEST_COUNT_DRIFT:'+str(tests))
        print('PYTEST='+result.splitlines()[-1],flush=True)
        stress=child([str(ROOT/'tools/r3_capacity_stress.py')])
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('CAPACITY_STRESS_FAIL')
        # Recheck every source and manifest *after* execution.
        for name in ('FILES.sha256','FROZEN_R8.sha256','FROZEN_R7.sha256',
                     'SSC_PREDECESSOR.sha256','FROZEN_R5.sha256','FROZEN_R6.sha256'):
            manifest(name)
        pins(r2root,R2_PINS,'R2_AFTER')
        pins(scroot,ssc_ext,'SSC_AFTER')
        evidence={'tests':tests,'own_manifest':sha(ROOT/'FILES.sha256'),
            'r8_manifest':sha(ROOT/'FROZEN_R8.sha256'),
            'ssc_manifest':sha(ROOT/'SSC_PREDECESSOR.sha256'),
            'r2_pins':R2_PINS,'scope':'R9_VERSIONED_SSC_REFERENCE_GUARD_NATIVE_EXACT_SHADOW',
            'predecessor_defects':'OVERCAP_STEP_AND_NEGATIVE_DECAY_BLOCKED_AT_R9_BOUNDARY',
            'full_native_ssc':'HOLD_REFERENCE_STILL_AUTHORITY'}
        digest=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        state='PASS_SSC_VERSIONED_REFERENCE_GUARD_AND_NATIVE_EXACT_SHADOW_ONLY'
    except BaseException as exc:
        issues.append(type(exc).__name__+':'+str(exc))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R9_SSC_STATE_GUARD_AND_NATIVE_EXACT_SHADOW_R0')
    print('DISPOSITION='+state)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R8_PINS='+str(r8)+'/59')
    print('FROZEN_R7_PINS='+str(r7)+'/43')
    print('SSC_PREDECESSOR_PINS='+str(ssc)+'/7')
    print('R2_SOURCE_PINS='+str(r2)+'/4')
    print('R9_OWN_FILES='+str(own))
    print('NATIVE_SHADOW=FIXED_CAP_8192;CAS_GENERATION;REBUILD_ATOMIC_NATIVE_ONLY')
    print('SSC_GUARDED_OPS=14;VALID_REFERENCE_ORACLE_TRACE_SEEDS=12')
    print('SSC_R9_OVERCAP_STEP=REJECTED;SSC_R9_NEGATIVE_DECAY=REJECTED')
    print('SSC_FULL_NATIVE_STATE_BANK_PARITY=HOLD_REFERENCE_AUTHORITY')
    print('SSC_OFFICIAL_FIXTURES=HOLD_ORIGINAL_TRANSPORT_UNPINNED')
    print('UNSCOPED_SEMANTIC_SELECTION=HOLD;NEURAL_RUNTIME=HOLD')
    print('PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE')
    print('RIGHTS=HOLD;PRODUCTION_TRUST=HOLD')
    print('RESULT_SHA256='+digest)
    print('ELAPSED_S='+format(time.monotonic()-started,'.3f'))
    if issues:print('BLOCKERS='+';'.join(issues))
    print('EXIT_STATUS='+('1' if issues else '0'))
    return 1 if issues else 0
if __name__=='__main__':sys.exit(main())
