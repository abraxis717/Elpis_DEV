#!/usr/bin/env python3
"""ERSA R5 isolated read qualification; immutable R2 + R4 regression gates."""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
FROZEN=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2')
PINS={
 'src/elpis_ersa/bank_ersa.py':'1212f16b20848a3cba48b7688236cf0b33266d0a82c5e3eea0c5eb7d0e8de6d3',
 'src/elpis_ersa/adapter_ssc.py':'2dd825efec3c6fe38a71a1c1bd26b2fbe6cfb063cc9d4712486c5db7af894b53',
 'src/elpis_ersa/contract_ersa.py':'5afa1abba0a92e38ea36d62c720be448b45b3de67802e8ee4075f255fd81b502',
 'src/elpis_ersa/fixtures_r2.py':'d0fcb5e0c46f8dbdfde4b3b5dce908a10e03bc76bcb2b10d6f1a9d11a2fbf329',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def child(*args,timeout=120):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
             TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
             CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(ROOT/'src'))
    r=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     text=True,timeout=timeout)
    if r.returncode:
        print(r.stdout[-5000:],file=sys.stderr)
        raise RuntimeError('CHILD_FAILED:'+str(args[0]))
    return r.stdout

def main():
    start=time.monotonic(); disposition='HOLD';checks=0;issues=[];digest='NONE';cost='NONE'
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise RuntimeError('EXECUTION_MUST_BE_TMPFS')
        frozen=FROZEN
        if 'ERSA_R2_ROOT' in os.environ:
            frozen=pathlib.Path(os.environ['ERSA_R2_ROOT']).resolve()
            if not frozen.is_relative_to('/dev/shm'):
                raise RuntimeError('R2_REHEARSAL_OVERRIDE_MUST_BE_TMPFS')
        for rel,expected in PINS.items():
            p=frozen/rel
            if not p.is_file() or p.is_symlink() or sha(p)!=expected:
                raise RuntimeError('R2_SOURCE_MISMATCH:'+rel)
        mf=ROOT/'FILES.sha256'
        if not mf.is_file() or mf.is_symlink():
            raise RuntimeError('PACKAGE_MANIFEST_MISSING')
        entries={}
        for line in mf.read_text().splitlines():
            parts=line.split('  ',1)
            if len(parts)!=2:raise RuntimeError('MALFORMED_MANIFEST')
            h,rel=parts
            if (not re.fullmatch('[0-9a-f]{64}',h) or not rel or rel in entries or
                    rel=='FILES.sha256' or rel.startswith('/') or
                    '\\' in rel or '..' in pathlib.PurePosixPath(rel).parts):
                raise RuntimeError('UNSAFE_MANIFEST_ENTRY')
            p=ROOT/rel
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT) or sha(p)!=h:
                raise RuntimeError('PACKAGE_FILE_MISMATCH:'+rel)
            entries[rel]=h
        tests=child('-m','pytest','-q','-p','no:cacheprovider','tests',timeout=90)
        m=re.search(r'(\d+) passed',tests)
        checks=int(m.group(1)) if m else 0
        if checks!=246:raise RuntimeError('PYTEST_COUNT_DRIFT:'+tests[-600:])
        print('PYTEST='+tests.strip().splitlines()[-1],flush=True)
        earlier=child(str(ROOT/'tools/r4_retrieval_check.py'))
        earlier_json=json.loads(next(x.partition('=')[2] for x in earlier.splitlines()
            if x.startswith('R4_RETRIEVAL_METRICS=')))
        expected={'R0:indexed:K8':15,'R0:score_indexed:K8':16,'R0:dense:K8':18,
                  'R2:indexed:K8':7,'R2:score_indexed:K8':12,'R2:dense:K8':12}
        for x,num in expected.items():
            if earlier_json[x]['exact']!=num:raise RuntimeError('FROZEN_R4_REGRESSION:'+x)
        r5=child(str(ROOT/'tools/r5_retrieval_check.py'))
        results=json.loads(next(x.partition('=')[2] for x in r5.splitlines()
            if x.startswith('R5_RETRIEVAL_METRICS=')))
        matches=results['complete']
        if (len(matches)!=12 or not all(z['exact_match'] and z['candidate_recall']==1
                                         for z in matches.values())):
            raise RuntimeError('COMPLETE_CANDIDATE_OR_DENSE_PARITY')
        if not (matches['near_identical_keys_distinct_identity']['forbidden_leak']
                and matches['sparse_evidence']['required_recall']==0
                and matches['contradictory_attested']['required_recall']==0.5
                and matches['gauge_equivalent_queries']['gauge_covariance']>0.1):
            raise RuntimeError('NEGATIVE_CONTROL_MASKED')
        for fam in ('sparse_evidence','contradictory_attested','near_identical_keys_distinct_identity'):
            case=results['scoped'][fam]
            if (case['status']!='OK' or case['required_recall']!=1
                    or case['forbidden_leak']):
                raise RuntimeError('EXPLICIT_SCOPE_REQUIRED:'+fam)
        stress=child(str(ROOT/'tools/r3_capacity_stress.py'),timeout=180)
        last=next((x for x in stress.splitlines() if x.startswith('CAPACITY_PASS=')),None)
        if last is None or 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in last:
            raise RuntimeError('CAPACITY_STRESS_FAILED')
        print(last,flush=True)
        bench=child(str(ROOT/'tools/r5_performance.py'),timeout=180)
        cost=next(x.partition('=')[2] for x in bench.splitlines() if x.startswith('R5_DIAGNOSTIC_COST='))
        print('R5_DIAGNOSTIC_COST='+cost,flush=True)
        for rel,expected in PINS.items():
            if sha(frozen/rel)!=expected:raise RuntimeError('FROZEN_SOURCE_MUTATED:'+rel)
        for rel,h in entries.items():
            if sha(ROOT/rel)!=h:raise RuntimeError('OWN_SOURCE_MUTATED:'+rel)
        summary={'pins':PINS,'manifest':sha(mf),'tests':checks,
                 'r4_baseline':earlier_json,'r5_results':results,'stress':last,'perf':json.loads(cost)}
        digest=hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        print('R5_RETRIEVAL_SHA256='+hashlib.sha256(json.dumps(results,sort_keys=True,separators=(',',':')).encode()).hexdigest(),flush=True)
        disposition='PASS_COMPLETE_DISCOVERY_AND_EXPLICIT_RELATION_SET_MECHANICS_ONLY'
    except Exception as exc:
        issues.append(type(exc).__name__+':'+str(exc))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R5_COMPLETE_DISCOVERY_AND_SCOPED_RELATIONS_R0')
    print('DISPOSITION='+disposition)
    print('PYTEST_PASSED='+str(checks))
    print('FROZEN_R2_PINS='+('4/4' if not issues else 'NOT_QUALIFIED'))
    print('R4_REGRESSION=PRIOR_METRICS_REPLAYED' if not issues else 'R4_REGRESSION=NOT_QUALIFIED')
    print('COMPLETE_CANDIDATE_COVERAGE=12/12;DENSE_RANKING_PARITY=12/12' if not issues else 'COMPLETE_CANDIDATE_COVERAGE=NOT_QUALIFIED')
    print('SCOPED_TASK_RECALL=3/3;SCOPED_FORBIDDEN_LEAK=0' if not issues else 'SCOPED_TASK_RECALL=NOT_QUALIFIED')
    print('UNSCOPED_TASK_GROUNDED_RECALL=FAIL;GAUGE_R2_INHERITED_FIXTURE=INCONSISTENT_PARTICIPANT_TRANSFORM')
    print('SSC_PARITY=HOLD;NATIVE_NEURAL_RUNTIME=HOLD;UNTRUSTED_SCOPE_AUTHORITY=HOLD')
    print('MODEL=NONE;GPU=NONE;RIGHTS=HOLD;PRODUCTION_TRUST=HOLD')
    print('PROJECT_MUTATIONS=NONE_INTENDED;OS_NVME_WRITE_ATTESTATION=NONE')
    print('RESULT_SHA256='+digest)
    print('ELAPSED_S='+format(time.monotonic()-start,'.3f'))
    if issues: print('ERRORS='+';'.join(issues))
    print('EXIT_STATUS='+('0' if not issues else '1'))
    return 0 if not issues else 1
if __name__=='__main__':sys.exit(main())
