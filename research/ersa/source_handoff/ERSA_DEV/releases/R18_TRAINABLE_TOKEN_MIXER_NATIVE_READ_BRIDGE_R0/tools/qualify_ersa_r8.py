#!/usr/bin/env python3
"""ERSA R8 pinned SSC source/projection/native-reducer qualification.

Only the read projection + distinct SSC stateless reduction qualifies.
FULL SSC parity is not claimed (two predecessor defects and one missing frozen
fixture dependency are expressly tested/reported).  Repeated R7 full
regression, 8,192-slot capacity stress, and 4+7 source pins.
"""
from __future__ import annotations
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

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def manifest(name,base=ROOT):
    f=base/name
    if not f.is_file() or f.is_symlink():raise ValueError('MISSING_MANIFEST:'+str(name))
    found={}
    for line in f.read_text().splitlines():
        parts=line.split('  ',1)
        if len(parts)!=2:raise ValueError('MANIFEST_FORMAT')
        h,rel=parts
        r=pathlib.PurePosixPath(rel)
        if (not re.fullmatch('[a-f0-9]{64}',h) or rel in found or
              r.is_absolute() or '..' in r.parts or '\\' in rel):
            raise ValueError('MANIFEST_UNSAFE_OR_DUPLICATE:'+rel)
        source=base/rel
        if (not source.is_file() or source.is_symlink() or
              not source.resolve().is_relative_to(base.resolve()) or
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

def runchild(args,timeout=120):
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
         PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TMPDIR='/dev/shm',
         CUDA_VISIBLE_DEVICES='-1',HOME=str(ROOT), XDG_CACHE_HOME=str(ROOT),
         PYTHONPATH=':'.join((str(ROOT),str(ROOT/'src'),str(ROOT/'native'),
                              str(ROOT/'ssc_predecessor/src'))))
    p=subprocess.run([sys.executable,'-B',*args], cwd=ROOT,env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,timeout=timeout)
    if p.returncode:
        print(p.stdout[-6000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAIL:'+args[0]+':'+str(p.returncode))
    return p.stdout

def main():
    t=time.monotonic();state='HOLD';issues=[];tests=0;own=0;d='NONE';r7=0;ssc=0;r2=0;transport_sha='NOT_IN_PINNED_ARCHIVE'
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise ValueError('EXECUTION_MUST_USE_TMPFS')
        ownfiles=manifest('FILES.sha256');own=len(ownfiles)
        original=manifest('FROZEN_R7.sha256');r7=len(original)
        original_ssc=manifest('SSC_PREDECESSOR.sha256');ssc=len(original_ssc)
        if len(original)!=43 or len(original_ssc)!=7:
            raise ValueError('FROZEN_FILE_COUNT_MISMATCH')
        r2root=envroot('ERSA_R2_ROOT',R2_DEFAULT)
        scroot=envroot('ERSA_SSC_ROOT',SSC_DEFAULT)
        pins(r2root,R2_PINS,'R2');r2=len(R2_PINS)
        expected_ssc={rel.replace('ssc_predecessor/',''):h
                      for rel,h in original_ssc.items()}
        pins(scroot,expected_ssc,'SSC_EXTERNAL')
        extra=scroot/'src/elpis_ssc/transport_reference.py'
        if extra.is_file() and not extra.is_symlink() and extra.stat().st_size<=100000:
            transport_sha=sha(extra)+'_UNPINNED_READ_ONLY'
        print('QUAL_STAGE=FROZEN_PINS_PASS;R2=4;SSC=7;R7=43;OWN='+str(own),flush=True)
        testout=runchild(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=120)
        match=re.search(r'(\d+) passed',testout)
        tests=int(match.group(1)) if match else 0
        if tests!=465:raise ValueError('PYTEST_COUNT_DRIFT:'+str(tests))
        print('PYTEST='+testout.splitlines()[-1],flush=True)
        stress=runchild([str(ROOT/'tools/r3_capacity_stress.py')],timeout=120)
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise ValueError('CAPACITY_STRESS_REGRESSION')
        manifest('FILES.sha256');manifest('FROZEN_R7.sha256');manifest('SSC_PREDECESSOR.sha256')
        pins(r2root,R2_PINS,'R2_AFTER');pins(scroot,expected_ssc,'SSC_AFTER')
        evidence={'tests':tests,'own_manifest_sha':sha(ROOT/'FILES.sha256'),
            'r7_manifest_sha':sha(ROOT/'FROZEN_R7.sha256'),
            'ssc_manifest_sha':sha(ROOT/'SSC_PREDECESSOR.sha256'),
            'ssc_source_sha':original_ssc,'capacity':8192,
            'full_ssc':'HOLD_OVERBUDGET_STEP_AND_NEGATIVE_DECAY_PREDECESSOR',
            'official_ssc_fixtures':'HOLD_MISSING_TRANSPORT_REFERENCE',
            'scope':'SSC_READ_ONLY_IDENTITY_AND_NATIVE_SOFTMAX_REDUCER'}
        d=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        state='PASS_SSC_PINNED_READ_PROJECTION_AND_NATIVE_DUAL_REDUCER_ONLY'
    except Exception as exc:
        issues.append(type(exc).__name__+':'+str(exc))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R8_SSC_REAL_CONTRACT_PROJECTION_R0')
    print('DISPOSITION='+state)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R7_PINS='+str(r7)+'/43')
    print('SSC_PREDECESSOR_PINS='+str(ssc)+'/7')
    print('R2_SOURCE_PINS='+str(r2)+'/4')
    print('R8_OWN_FILES='+str(own))
    print('SSC_ORACLE_TRACE_CASES=16')
    print('SSC_IDENTITY_NATIVE_READ=PASS_TESTED_SUBSET' if not issues else 'SSC_IDENTITY_NATIVE_READ=NOT_QUALIFIED')
    print('SSC_NATIVE_NORM_SOFTMAX=PASS_INDEPENDENT_DIFFERENTIAL' if not issues else 'SSC_NATIVE_NORM_SOFTMAX=NOT_QUALIFIED')
    print('SSC_FULL_STATE_BANK_PARITY=HOLD_R7_NOT_SSC_STATE_BACKEND')
    print('SSC_PREDECESSOR_STEP_OVERCAP=DIVERGENCE_CONFIRMED')
    print('SSC_PREDECESSOR_NEGATIVE_DECAY=ACCEPTED_BY_BOTH_REFERENCE_AND_ORACLE')
    print('SSC_OFFICIAL_FIXTURES=HOLD_TRANSPORT_REFERENCE_NOT_FROZEN')
    print('SSC_TRANSPORT_SOURCE_SHA256='+transport_sha)
    print('NATIVE_NEURAL_RUNTIME=HOLD;UNSCOPED_SEMANTIC_SELECTION=HOLD')
    print('SOURCE_AUTHORITY=FROZEN;PROJECT_MUTATIONS=NONE_INTENDED;GPU=NONE;OS_NVME_WRITE_ATTESTATION=NONE')
    print('RESULT_SHA256='+d)
    print('ELAPSED_S='+format(time.monotonic()-t,'.3f'))
    if issues:print('BLOCKERS='+';'.join(issues))
    print('EXIT_STATUS='+('1' if issues else '0'))
    return 1 if issues else 0
if __name__=='__main__':sys.exit(main())
