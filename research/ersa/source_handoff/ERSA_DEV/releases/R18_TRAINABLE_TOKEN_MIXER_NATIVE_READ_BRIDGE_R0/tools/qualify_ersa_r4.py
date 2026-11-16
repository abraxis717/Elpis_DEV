#!/usr/bin/env python3
"""ERSA R4: qualified identity read + conditional index ranking, no native claims.

Read-only predecessor verification, tmpfs-only candidate execution.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[1]
FROZEN=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2')
PINS={
    'src/elpis_ersa/bank_ersa.py':'1212f16b20848a3cba48b7688236cf0b33266d0a82c5e3eea0c5eb7d0e8de6d3',
    'src/elpis_ersa/adapter_ssc.py':'2dd825efec3c6fe38a71a1c1bd26b2fbe6cfb063cc9d4712486c5db7af894b53',
    'src/elpis_ersa/contract_ersa.py':'5afa1abba0a92e38ea36d62c720be448b45b3de67802e8ee4075f255fd81b502',
    'src/elpis_ersa/fixtures_r2.py':'d0fcb5e0c46f8dbdfde4b3b5dce908a10e03bc76bcb2b10d6f1a9d11a2fbf329'
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*args,timeout=120):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR='/dev/shm',
             HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
             PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CUDA_VISIBLE_DEVICES='-1',
             PYTHONPATH=str(ROOT/'src'))
    proc=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,timeout=timeout)
    if proc.returncode:
        print(proc.stdout[-4000:],file=sys.stderr)
        raise RuntimeError('SUBPROCESS_FAIL:'+str(args[0]))
    return proc.stdout


def main():
    start=time.monotonic();status='HOLD';checks=0;errs=[];evidence='NONE'
    try:
        if not ROOT.is_relative_to('/dev/shm'):
            raise RuntimeError('EXECUTION_MUST_BE_TMPFS')
        frozen=FROZEN
        if 'ERSA_R2_ROOT' in os.environ:
            frozen=pathlib.Path(os.environ['ERSA_R2_ROOT']).resolve()
            if not frozen.is_relative_to('/dev/shm'):
                raise RuntimeError('REHEARSAL_OVERRIDE_OUTSIDE_TMPFS')
        for rel,expected in PINS.items():
            p=frozen/rel
            if not p.is_file() or p.is_symlink() or sha(p)!=expected:
                raise RuntimeError('FROZEN_R2_PIN:'+rel)
        mf=ROOT/'FILES.sha256'
        if not mf.is_file() or mf.is_symlink():raise RuntimeError('NO_MANIFEST')
        entries={}
        for line in mf.read_text().splitlines():
            parts=line.split('  ',1)
            if len(parts)!=2:raise RuntimeError('MALFORMED_MANIFEST')
            h,rel=parts
            if (not re.fullmatch('[0-9a-f]{64}',h) or not rel or
                rel in entries or rel.startswith('/') or
                '..' in pathlib.PurePosixPath(rel).parts or '\\' in rel or
                rel=='FILES.sha256'):
                raise RuntimeError('UNSAFE_MANIFEST')
            p=ROOT/rel
            if (p.is_symlink() or not p.is_file() or
                not p.resolve().is_relative_to(ROOT) or sha(p)!=h):
                raise RuntimeError('OWN_SOURCE_DRIFT:'+rel)
            entries[rel]=h
        if not entries:raise RuntimeError('EMPTY_MANIFEST')
        tests=run('-m','pytest','-q','-p','no:cacheprovider','tests',timeout=90)
        m=re.search(r'(\d+) passed',tests)
        checks=int(m.group(1)) if m else 0
        if checks!=205:raise RuntimeError('TEST_COUNT_OR_RESULT_DRIFT:'+tests[-500:])
        print('PYTEST='+tests.strip().splitlines()[-1],flush=True)
        raw=run(str(ROOT/'tools/r4_retrieval_check.py'))
        lead='R4_RETRIEVAL_METRICS='
        dataline=next(x for x in raw.splitlines() if x.startswith(lead))
        scores=json.loads(dataline[len(lead):])
        expected_exact={'R0:indexed:K8':15,'R0:score_indexed:K8':16,'R0:dense:K8':18,
                        'R2:indexed:K8':7,'R2:score_indexed:K8':12,'R2:dense:K8':12}
        for name,num in expected_exact.items():
            if scores[name]['exact']!=num:raise RuntimeError('RETRIEVAL_DRIFT:'+name)
        for mode in ('dense','score_indexed'):
            m=scores[f'R2:{mode}:K8']
            if (m['forbidden']!=1 or 'sparse_evidence' not in m['required_zero']
                or 'near_identical_keys_distinct_identity' not in m['required_zero']
                or 'contradictory_attested' not in m['partial_required']):
                raise RuntimeError('RETRIEVAL_NEGATIVE_CONTROL_MASKED:'+mode)
        s=scores['exact_identity_safety']
        if (s['receipt']!='OK' or s['selected']!=[0] or s['evidence']!=[0]
            or not s['forbidden_absent']):
            raise RuntimeError('IDENTITY_SCOPE_NOT_SAFE')
        print('SCORE_INDEXED_R0_K8=16/18;PREDECESSOR_INDEXED_R0_K8=15/18')
        print('SCORE_INDEXED_R2_K8=12/12;PREDECESSOR_INDEXED_R2_K8=7/12')
        print('DENSE_PARITY_NOT_SEMANTIC_CERTIFICATION=TRUE')
        print('R2_UNSCOPED_FORBIDDEN_LEAK=1;SPARSE_CANDIDATE_RECALL_ZERO=TRUE')
        print('EXACT_IDENTITY_SCOPE_FORBIDDEN_NEIGHBOR=EXCLUDED')
        stress=run(str(ROOT/'tools/r3_capacity_stress.py'),timeout=120)
        last=next((x for x in stress.splitlines() if x.startswith('CAPACITY_PASS=')),None)
        if not last or 'CAPACITY_PASS=TRUE;CAP=8192;EXTRA_INSERTS=64;PROTECTED=16;LIVE=8192;' not in last:
            raise RuntimeError('CAPACITY_STRESS_FAIL')
        print(last)
        for rel,expected in PINS.items():
            if sha(frozen/rel)!=expected:raise RuntimeError('FROZEN_MUTATED:'+rel)
        for rel,h in entries.items():
            if sha(ROOT/rel)!=h:raise RuntimeError('PACKAGE_MUTATED:'+rel)
        payload={'pins':PINS,'own_manifest':sha(mf),'tests':checks,
                 'scores':scores,'stress':last,'parent_ersa_r3_zip_sha256':
                 '25e7f31a9b1d97a373ed203e70d5a8981dfec68c538b2ec42e92c271968fa17c'}
        evidence=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        status='PASS_EXACT_IDENTITY_AND_CONDITIONAL_INDEX_RANKING_MECHANICS_ONLY'
    except Exception as exc:
        errs.append(type(exc).__name__+':'+str(exc))
    print('=== SEND BACK ===')
    print('PHASE=ERSA_R4_DIRECT_RETRIEVAL_SEMANTICS_R0')
    print('DISPOSITION='+status)
    print('PYTEST_PASSED='+str(checks))
    print('FROZEN_R2_SOURCE_PINS=4/4' if not errs else 'FROZEN_R2_SOURCE_PINS=NOT_QUALIFIED')
    print('RESULT_SHA256='+evidence)
    print('SEMANTIC_REQUIRED_RECALL=NOT_CERTIFIED;SSC_PARITY=HOLD;NEURAL_RUNTIME=HOLD')
    print('FROZEN_R3_INTEGRITY=PRIOR_OUROBOROS_PASS_NOT_REEXECUTED_IN_R4')
    print('PROJECT_MUTATIONS=NONE_INTENDED;NVME_WRITE_ATTESTATION=NONE;GPU=NONE')
    print('ELAPSED_S='+format(time.monotonic()-start,'.3f'))
    if errs:print('ERRORS='+';'.join(errs))
    print('EXIT_STATUS='+('0' if not errs else '1'))
    return 0 if not errs else 1

if __name__=='__main__':sys.exit(main())
