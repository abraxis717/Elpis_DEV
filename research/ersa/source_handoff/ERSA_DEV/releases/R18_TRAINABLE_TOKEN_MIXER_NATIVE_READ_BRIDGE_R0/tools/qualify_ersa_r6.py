#!/usr/bin/env python3
"""ERSA R6 native kernel / frozen-R5 differential qualification; no auto-promotion."""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
R2_DEFAULT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2')
R2_PINS={
 'src/elpis_ersa/bank_ersa.py':'1212f16b20848a3cba48b7688236cf0b33266d0a82c5e3eea0c5eb7d0e8de6d3',
 'src/elpis_ersa/adapter_ssc.py':'2dd825efec3c6fe38a71a1c1bd26b2fbe6cfb063cc9d4712486c5db7af894b53',
 'src/elpis_ersa/contract_ersa.py':'5afa1abba0a92e38ea36d62c720be448b45b3de67802e8ee4075f255fd81b502',
 'src/elpis_ersa/fixtures_r2.py':'d0fcb5e0c46f8dbdfde4b3b5dce908a10e03bc76bcb2b10d6f1a9d11a2fbf329',
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(path):
    mf=ROOT/path
    if not mf.is_file() or mf.is_symlink():raise RuntimeError('MISSING_MANIFEST:'+path)
    out={}
    for line in mf.read_text().splitlines():
        fields=line.split('  ',1)
        if len(fields)!=2:raise RuntimeError('MALFORMED_MANIFEST:'+path)
        digest,rel=fields
        if not re.fullmatch('[a-f0-9]{64}',digest) or rel in out or not rel or rel.startswith('/') or '\\' in rel or '..' in pathlib.PurePosixPath(rel).parts:
            raise RuntimeError('UNSAFE_MANIFEST_ENTRY:'+rel)
        p=ROOT/rel
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise RuntimeError('UNSAFE_FILE:'+rel)
        if sha(p)!=digest:raise RuntimeError('HASH_MISMATCH:'+rel)
        out[rel]=digest
    return out

def child(*args,timeout=180):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
             TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
             CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'native'))
    p=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
                      stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                      text=True,timeout=timeout)
    if p.returncode:
        print(p.stdout[-7000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAIL:'+str(args[0])+':rc='+str(p.returncode))
    return p.stdout

def main():
    start=time.monotonic();tests=0;status='HOLD';error=[];digest='NONE';cost={};r5={};nfiles=0
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise RuntimeError('ROOT_NOT_SHM')
        own=manifest('FILES.sha256');frozen=manifest('FROZEN_R5.sha256')
        nfiles=len(own)
        if len(frozen)!=26:raise RuntimeError('R5_FILE_COUNT_DRIFT:'+str(len(frozen)))
        r2=R2_DEFAULT
        if 'ERSA_R2_ROOT' in os.environ:
            r2=pathlib.Path(os.environ['ERSA_R2_ROOT']).resolve()
            if not r2.is_relative_to('/dev/shm'):
                raise RuntimeError('UNSAFE_R2_OVERRIDE')
        for rel,h in R2_PINS.items():
            p=r2/rel
            if not p.is_file() or p.is_symlink() or sha(p)!=h:
                raise RuntimeError('FROZEN_R2_PIN:'+rel)
        # The entire original R5 suite plus 50 adversarial native tests.
        out=child('-m','pytest','-q','-p','no:cacheprovider','tests',timeout=120)
        m=re.search(r'(\d+) passed',out)
        tests=int(m.group(1)) if m else 0
        if tests!=296:raise RuntimeError('TEST_COUNT_DRIFT:'+str(tests)+':'+out[-500:])
        print('PYTEST='+out.strip().splitlines()[-1],flush=True)
        retr=child(str(ROOT/'tools/r5_retrieval_check.py'))
        r5=json.loads(next(line.partition('=')[2] for line in retr.splitlines() if line.startswith('R5_RETRIEVAL_METRICS=')))
        complete=r5['complete']
        if len(complete)!=12 or not all(z['exact_match'] and z['candidate_recall']==1 for z in complete.values()):
            raise RuntimeError('FROZEN_R5_PARITY_FAIL')
        for fam in ('sparse_evidence','contradictory_attested','near_identical_keys_distinct_identity'):
            z=r5['scoped'][fam]
            if z['status']!='OK' or z['required_recall']!=1 or z['forbidden_leak']:
                raise RuntimeError('FROZEN_SCOPE_FAIL:'+fam)
        if not (complete['sparse_evidence']['required_recall']==0
                and complete['near_identical_keys_distinct_identity']['forbidden_leak']):
            raise RuntimeError('UNSCOPED_NEGATIVE_CONTROL_MASKED')
        stress=child(str(ROOT/'tools/r3_capacity_stress.py'),timeout=150)
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise RuntimeError('CAPACITY_REGRESSION')
        native=child(str(ROOT/'tools/r6_native_bench.py'),timeout=60)
        cost=json.loads(next(line.partition('=')[2] for line in native.splitlines()
                   if line.startswith('R6_NATIVE_KERNEL_BENCH=')))
        print('R6_NATIVE_KERNEL_BENCH='+json.dumps(cost,sort_keys=True,separators=(',',':')),flush=True)
        for rel,h in R2_PINS.items():
            if sha(r2/rel)!=h:raise RuntimeError('R2_MUTATED:'+rel)
        manifest('FROZEN_R5.sha256')
        manifest('FILES.sha256')
        payload={'r2_pins':R2_PINS,'r5_manifest':sha(ROOT/'FROZEN_R5.sha256'),
                 'own_manifest':sha(ROOT/'FILES.sha256'),'tests':tests,
                 'r5':r5,'bench':cost}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        status='PASS_NATIVE_STATELESS_RANK_AND_SIGNED_REDUCE_DIFFERENTIAL_ONLY'
    except Exception as exc:
        error.append(type(exc).__name__+':'+str(exc))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R6_NATIVE_OPERATOR_DIFF_R0')
    print('DISPOSITION='+status)
    print('PYTEST_PASSED='+str(tests))
    print('R5_FROZEN_FILES='+('26/26' if not error else 'NOT_QUALIFIED'))
    print('R2_SOURCE_PINS='+('4/4' if not error else 'NOT_QUALIFIED'))
    print('R5_COMPLETE_PARITY='+('12/12' if not error else 'NOT_QUALIFIED'))
    print('NATIVE_REFERENCE_DIFF='+('12_FAMILIES;SIGNED_REDUCTION;GAUGE_CONTROL;FAIL_CLOSED' if not error else 'NOT_QUALIFIED'))
    print('R6_KERNEL_ONLY_US='+json.dumps(cost,sort_keys=True,separators=(',',':')) if cost else 'NONE')
    print('SSC_PARITY=HOLD_NO_REAL_SSC_PREDECESSOR_SOURCE_IN_ARCHIVE')
    print('UNSCOPED_SEMANTIC_SELECTION=HOLD;NATIVE_NEURAL_RUNTIME=HOLD;NATIVE_SCORE_ROUNDING=HOST_CANONICAL')
    print('PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE;RIGHTS=HOLD')
    print('RESULT_SHA256='+digest)
    print('ELAPSED_S='+format(time.monotonic()-start,'.3f'))
    if error:print('BLOCKERS='+';'.join(error))
    print('EXIT_STATUS='+('1' if error else '0'))
    return 1 if error else 0
if __name__=='__main__':sys.exit(main())
