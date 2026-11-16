#!/usr/bin/env python3
"""ERSA R7 — native host-authorized read path qualification; fail closed."""
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
def manifest(name):
    file=ROOT/name
    if not file.is_file() or file.is_symlink():raise RuntimeError('MISSING_MANIFEST:'+name)
    seen={}
    for line in file.read_text().splitlines():
        parts=line.split('  ',1)
        if len(parts)!=2:raise RuntimeError('MALFORMED_MANIFEST:'+name)
        h,rel=parts
        pp=pathlib.PurePosixPath(rel)
        if not re.fullmatch('[0-9a-f]{64}',h) or rel in seen or not rel or pp.is_absolute() or '..' in pp.parts or '\\' in rel:
            raise RuntimeError('BAD_MANIFEST_ENTRY:'+rel)
        p=ROOT/rel
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise RuntimeError('BAD_FILE:'+rel)
        if sha(p)!=h:raise RuntimeError('FILE_DRIFT:'+rel)
        seen[rel]=h
    return seen

def child(*args,timeout=180):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
             TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
             CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')+':'+str(ROOT/'native'))
    proc=subprocess.Popen([sys.executable,'-B',*args],cwd=ROOT,env=env,
              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    out=[]
    try:
        for line in proc.stdout:
            out.append(line)
            # Unbuffered milestone lines are forwarded as they happen.
            if ('R7_BENCH_PROGRESS=' in line or 'PYTEST' in line or 'CAPACITY_' in line):
                print(line.rstrip(),flush=True)
        rc=proc.wait(timeout=timeout)
    except BaseException:
        proc.kill();proc.wait();raise
    if rc!=0:
        print(''.join(out)[-5000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAIL:'+str(args[0])+':rc='+str(rc))
    return ''.join(out)

def main():
    t=time.monotonic();state='HOLD';error=[];tests=0;bench=[];metrics={};own=0
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise RuntimeError('ROOT_NOT_VOLATILE')
        own_manifest=manifest('FILES.sha256');own=len(own_manifest)
        pred=manifest('FROZEN_R6.sha256');ref=manifest('FROZEN_R5.sha256')
        if len(pred)!=33 or len(ref)!=26:
            raise RuntimeError('PREDECESSOR_MANIFEST_COUNT')
        r2=R2_DEFAULT
        if 'ERSA_R2_ROOT' in os.environ:
            r2=pathlib.Path(os.environ['ERSA_R2_ROOT']).resolve()
            if not r2.is_relative_to('/dev/shm/'):
                raise RuntimeError('UNSAFE_R2_OVERRIDE')
        for rel,h in R2_PINS.items():
            p=r2/rel
            if not p.is_file() or p.is_symlink() or sha(p)!=h:
                raise RuntimeError('R2_SOURCE_PIN:'+rel)
        print('QUAL_STAGE=FROZEN_PINS_PASS;R2=4;R5=26;R6=33;OWN='+str(own),flush=True)
        output=child('-m','pytest','-q','-p','no:cacheprovider','tests',timeout=120)
        m=re.search(r'(\d+) passed',output)
        tests=int(m.group(1)) if m else 0
        if tests!=377:
            raise RuntimeError('TEST_COUNT_DRIFT:'+str(tests)+':'+output[-500:])
        print('PYTEST='+output.splitlines()[-1],flush=True)
        old=child(str(ROOT/'tools/r5_retrieval_check.py'),timeout=60)
        metrics=json.loads(next(line.split('=',1)[1] for line in old.splitlines()
                              if line.startswith('R5_RETRIEVAL_METRICS=')))
        if (len(metrics['complete'])!=12 or
            not all(x['exact_match'] and x['candidate_recall']==1
                    for x in metrics['complete'].values())):
            raise RuntimeError('R5_COMPLETE_COVERAGE_REGRESSION')
        stress=child(str(ROOT/'tools/r3_capacity_stress.py'),timeout=120)
        if 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in stress:
            raise RuntimeError('CAPACITY_STRESS_REGRESSION')
        out=child(str(ROOT/'tools/r7_end_to_end_bench.py'),timeout=120)
        bench=json.loads(next(line.split('=',1)[1] for line in out.splitlines()
                          if line.startswith('R7_END_TO_END_BENCH=')))
        if [x['n'] for x in bench]!=[256,1024,8192] or not all(x['parity'] for x in bench):
            raise RuntimeError('END_TO_END_PARITY')
        for rel,h in R2_PINS.items():
            if sha(r2/rel)!=h:raise RuntimeError('R2_MUTATED:'+rel)
        manifest('FROZEN_R5.sha256');manifest('FROZEN_R6.sha256');manifest('FILES.sha256')
        result=dict(tests=tests,r2=R2_PINS,r6_hash=sha(ROOT/'FROZEN_R6.sha256'),
                    r5_hash=sha(ROOT/'FROZEN_R5.sha256'),own_hash=sha(ROOT/'FILES.sha256'),
                    old=metrics,bench=bench)
        digest=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        state='PASS_NATIVE_FEATURE_TO_REDUCE_AUTHORIZED_READ_PATH_DIFFERENTIAL_ONLY'
    except Exception as exc:
        error.append(type(exc).__name__+':'+str(exc))
        digest='NONE'
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R7_NATIVE_HOST_AUTHORIZED_READ_PATH_R0')
    print('DISPOSITION='+state)
    print('PYTEST_PASSED='+str(tests))
    print('FROZEN_R6_PINS='+('33/33' if not error else 'NOT_QUALIFIED'))
    print('FROZEN_R5_PINS='+('26/26' if not error else 'NOT_QUALIFIED'))
    print('R2_SOURCE_PINS='+('4/4' if not error else 'NOT_QUALIFIED'))
    print('R7_OWN_FILE_COUNT='+str(own))
    print('R7_END_TO_END_US='+json.dumps(bench,sort_keys=True,separators=(',',':')) if bench else 'NONE')
    print('HOST_AUTHORITY=R5_BANK;NATIVE=FEATURES_SCORE_RANK_REDUCE;SLOTS_LIMIT=8192')
    print('SSC_PARITY=HOLD_REAL_PREDECESSOR_MAPPING_NOT_QUALIFIED')
    print('UNSCOPED_SEMANTIC_SELECTION=HOLD;TRAINED_NEURAL_RUNTIME=HOLD')
    print('OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE;PROJECT_MUTATIONS=NONE_INTENDED')
    print('RESULT_SHA256='+digest)
    print('ELAPSED_S='+format(time.monotonic()-t,'.3f'))
    if error:print('BLOCKERS='+';'.join(error))
    print('EXIT_STATUS='+('1' if error else '0'))
    return 1 if error else 0
if __name__=='__main__':sys.exit(main())
