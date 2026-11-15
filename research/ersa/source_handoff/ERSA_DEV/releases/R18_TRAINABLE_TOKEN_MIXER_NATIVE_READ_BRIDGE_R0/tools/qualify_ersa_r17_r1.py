"""R17 R1 independent, fail-closed three-receipt qualification.

Invoke --gate tests and --gate science in separate shell invocations to avoid
old combined-runner native/PyTorch resource stalls; then --gate finalize.
All transient receipts are bound to the ZIP manifest and remain on /dev/shm.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, subprocess, sys, tempfile, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import qualify_ersa_r14 as core
EXPECTED_TESTS=1133
FROZEN_FILES=114
RECEIPT_NAMES={'tests':'.r17_r1_tests_receipt.json','science':'.r17_r1_science_receipt.json'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def preflight():
    if not ROOT.resolve().is_relative_to('/dev/shm/'):
        raise RuntimeError('TMPFS_ONLY')
    own=core.manifest('FILES.sha256')
    frozen=core.manifest('FROZEN_R16.sha256')
    if len(frozen)!=FROZEN_FILES or any(own.get(k)!=v for k,v in frozen.items()):
        raise RuntimeError('FROZEN_R16_DRIFT')
    ssc=core.manifest('SSC_PREDECESSOR.sha256')
    if len(ssc)!=7 or any(own.get(k)!=v for k,v in ssc.items()):
        raise RuntimeError('FROZEN_SSC_DRIFT')
    r2=core.source_root('ERSA_R2_ROOT',core.R2_DEFAULT)
    core.verify_external(r2,core.R2_PINS)
    return sha_bytes((ROOT/'FILES.sha256').read_bytes()),len(own)


def child(args,timeout):
    env=dict(os.environ,
        PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=':'.join(map(str,(ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests'))))
    with tempfile.TemporaryFile(mode='w+t',dir='/dev/shm') as log:
        run=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        log.seek(0);out=log.read()
    if run.returncode:
        print(out[-7000:],file=sys.stderr,flush=True)
        raise RuntimeError('CHILD_FAILURE:'+str(args)+':'+str(run.returncode))
    return out


def write_receipt(kind,fields,manifest):
    record={'kind':kind,'manifest_sha256':manifest,'evidence':fields}
    record['self_sha256']=sha_bytes(canonical(record))
    (ROOT/RECEIPT_NAMES[kind]).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return record


def read_receipt(kind,manifest):
    path=ROOT/RECEIPT_NAMES[kind]
    if not path.is_file() or path.is_symlink():raise RuntimeError('MISSING_RECEIPT:'+kind)
    record=json.loads(path.read_text())
    digest=record.pop('self_sha256',None)
    if (digest!=sha_bytes(canonical(record)) or record['manifest_sha256']!=manifest or
        record['kind']!=kind):raise RuntimeError('RECEIPT_DRIFT:'+kind)
    return record['evidence']


def run(gate):
    manifest,own=preflight()
    if gate=='tests':
        probe=child(['tools/r17_readonly_state_probe.py'],30)
        if 'READ_ONLY_STATE_BITWISE_PARITY=80/80' not in probe:
            raise RuntimeError('READONLY_STATE_NOT_EXACT')
        print('READ_ONLY_STATE_BITWISE_PARITY=80/80',flush=True)
        out=child(['-m','pytest','-q','-p','no:cacheprovider','tests'],90)
        match=re.search(r'(\d+) passed',out)
        passed=int(match.group(1)) if match else 0
        if passed!=EXPECTED_TESTS:raise RuntimeError('TEST_COUNT_DRIFT:'+str(passed))
        write_receipt('tests',{'pytest_passed':passed,'bitwise_probe':80},manifest)
        print('GATE_TESTS=PASS;PYTEST_PASSED='+str(passed),flush=True)
    elif gate=='science':
        out=child(['tools/r17_science.py'],90)
        match=re.search(r'^R17_SCIENCE_SHA256=([a-f0-9]{64})$',out,re.M)
        data=re.search(r'^R17_SCIENCE=(.+)$',out,re.M)
        if not match or not data:raise RuntimeError('MISSING_SCIENCE_REPORT')
        s=json.loads(data.group(1))
        if (len(s['runs'])!=3 or s['official_fla_module_executed'] or
            s['official_fla_gpu_kernel_executed'] or s['trainable_weights']!=4930):
            raise RuntimeError('INVALID_SCIENCE_SCOPE')
        for item in s['runs']:
            tr=item['training']
            if not(tr['end']<tr['start']-.05 and tr['parameters']==4930 and tr['state_bytes']==2048):
                raise RuntimeError('TRAINING_NOT_VALID')
            for family,metric in item['heldout'].items():
                if sum(metric.get(k,-1) for k in ('correct','false_stop','abstain'))!=64:
                    raise RuntimeError('INVALID_HELDOUT:'+family)
            print('SEED='+str(tr['seed'])+';LOSS_START='+str(tr['start'])+';LOSS_END='+str(tr['end']),flush=True)
        write_receipt('science',{'science_sha256':match.group(1),'seeds':len(s['runs']),
                                 'parameters':4930},manifest)
        print('GATE_SCIENCE=PASS;R17_SCIENCE_SHA256='+match.group(1),flush=True)
    elif gate=='finalize':
        t=read_receipt('tests',manifest)
        s=read_receipt('science',manifest)
        if t['pytest_passed']!=EXPECTED_TESTS or t['bitwise_probe']!=80 or s['seeds']!=3:
            raise RuntimeError('INCOMPLETE_GATES')
        print('=== SEND BACK ===')
        print('PHASE=ERSA_R17_READ_ONLY_QUERY_STATE_REPAIR_R1')
        print('DISPOSITION=PASS_CPU_GDN_STRUCTURAL_REFERENCE_AND_BITWISE_READONLY_STATE_MECHANICS_ONLY')
        print('PYTEST_PASSED='+str(t['pytest_passed']))
        print('READ_ONLY_STATE_BITWISE_PARITY='+str(t['bitwise_probe'])+'/80')
        print('FROZEN_R16_PINS=114/114;SSC_PINS=7/7;R2_SOURCE_PINS=4/4')
        print('OWN_FILES='+str(own))
        print('MANIFEST_SHA256='+manifest)
        print('R17_SCIENCE_SHA256='+s['science_sha256'])
        print('GDN_FULL_FLA_LAYER=NOT_EXECUTED;ERSA_VS_GDN_MATCHED_PARITY=HOLD')
        print('SOURCE_AUTHORITY=FROZEN;PROJECT_MUTATIONS=NONE_INTENDED')
        print('RUNTIME_STATE=TMPFS_ONLY;OS_NVME_WRITE_ATTESTATION=NONE;GPU=NONE')
        print('RIGHTS=HOLD;PRODUCTION_TRUST=HOLD')
        print('EXIT_STATUS=0')
    else:raise RuntimeError('INVALID_GATE')
    core.manifest('FILES.sha256')
    core.verify_external(core.source_root('ERSA_R2_ROOT',core.R2_DEFAULT),core.R2_PINS)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',choices=['tests','science','finalize'],required=True)
    args=p.parse_args()
    start=time.monotonic()
    try:run(args.gate)
    except BaseException as exc:
        print('=== SEND BACK ===')
        print('PHASE=ERSA_R17_READ_ONLY_QUERY_STATE_REPAIR_R1;GATE='+args.gate)
        print('DISPOSITION=HOLD')
        print('BLOCKER='+type(exc).__name__+':'+str(exc))
        print('EXIT_STATUS=1')
        return 1
    print('GATE='+args.gate+';ELAPSED_S='+str(round(time.monotonic()-start,3)),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
