"""R18 isolated two-gate qualification; no changes to frozen project roots.

The manifest, external R2 pins and prior R17 R1 source authority are checked
on every invocation. Full pytest and science execute in separate child Python
processes with /dev/shm receipts, avoiding combined runner deadlocks.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import qualify_ersa_r14 as frozen
TEST_COUNT=1189
R17_PINS=122
R17_SCIENCE_PRIOR='7aafa48c7acb3fb68f6250f3c4b68513761d9515f20da1bc666696a907a8e318'
RECEIPTS={'tests':'.r18_tests_receipt.json','science':'.r18_science_receipt.json'}

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def canonical(data):return json.dumps(data,sort_keys=True,separators=(',',':')).encode()

def preflight():
    if not ROOT.resolve().is_relative_to('/dev/shm/'):
        raise RuntimeError('QUALIFICATION_REQUIRES_TMPFS')
    own=frozen.manifest('FILES.sha256')
    prev=frozen.manifest('FROZEN_R17_R1.sha256')
    if len(prev)!=R17_PINS or any(own.get(path)!=digest for path,digest in prev.items()):
        raise RuntimeError('FROZEN_R17_R1_DRIFT')
    ssc=frozen.manifest('SSC_PREDECESSOR.sha256')
    if len(ssc)!=7 or any(own.get(path)!=digest for path,digest in ssc.items()):
        raise RuntimeError('FROZEN_SSC_DRIFT')
    r2=frozen.source_root('ERSA_R2_ROOT',frozen.R2_DEFAULT)
    frozen.verify_external(r2,frozen.R2_PINS)
    allow=set(own)|{'FILES.sha256'}|set(RECEIPTS.values())
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    if actual!=allow and actual!=allow-set(RECEIPTS.values()) and not actual.issubset(allow):
        raise RuntimeError('EXTRA_UNPINNED_SOURCE:'+str(sorted(actual-allow)[:4]))
    if not set(own).issubset(actual):raise RuntimeError('MISSING_PINNED_SOURCE')
    return sha_bytes((ROOT/'FILES.sha256').read_bytes()),len(own)

def child(args,timeout):
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
             TMPDIR='/dev/shm',HOME=str(ROOT),XDG_CACHE_HOME=str(ROOT),
             CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONPATH=':'.join(str(x) for x in (
               ROOT,ROOT/'src',ROOT/'native',ROOT/'ssc_predecessor/src',ROOT/'tests')))
    with tempfile.TemporaryFile(mode='w+t',dir='/dev/shm') as f:
        p=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
             stdout=f,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        f.seek(0);output=f.read()
    if p.returncode:
        print(output[-8000:],flush=True)
        raise RuntimeError('CHILD_EXIT:'+str(p.returncode)+':'+str(args))
    return output

def receipt(kind,data,manifest):
    obj={'kind':kind,'manifest_sha256':manifest,'evidence':data}
    obj['self_sha256']=sha_bytes(canonical(obj))
    (ROOT/RECEIPTS[kind]).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')

def verify_receipt(kind,manifest):
    p=ROOT/RECEIPTS[kind]
    if not p.is_file() or p.is_symlink():raise RuntimeError('RECEIPT_MISSING:'+kind)
    data=json.loads(p.read_text())
    sha=data.pop('self_sha256',None)
    if sha!=sha_bytes(canonical(data)) or data.get('kind')!=kind or data.get('manifest_sha256')!=manifest:
        raise RuntimeError('RECEIPT_DRIFT:'+kind)
    return data['evidence']

def run(gate):
    manifest,n=preflight()
    if gate=='tests':
        # Inherited R17 R1 numerical failure must remain repaired.
        probe=child(['tools/r17_readonly_state_probe.py'],timeout=60)
        if 'READ_ONLY_STATE_BITWISE_PARITY=80/80' not in probe:
            raise RuntimeError('R17_BITWISE_REGRESSION')
        output=child(['-m','pytest','-q','-p','no:cacheprovider','tests'],timeout=180)
        match=re.search(r'^(\d+) passed',output,re.M)
        count=int(match.group(1)) if match else -1
        if count!=TEST_COUNT:raise RuntimeError('PYTEST_COUNT_DRIFT:'+str(count)+' '+output[-600:])
        receipt('tests',{'pytest':count,'r17_bitwise':80},manifest)
        print('GATE_TESTS=PASS;PYTEST_PASSED='+str(count)+';R17_READONLY=80/80',flush=True)
    elif gate=='science':
        output=child(['tools/r18_science.py'],timeout=180)
        sha=re.search(r'^R18_SCIENCE_SHA256=([a-f0-9]{64})$',output,re.M)
        body=re.search(r'^R18_SCIENCE=(.+)$',output,re.M)
        if not sha or not body:raise RuntimeError('MISSING_SCIENCE_REPORT')
        if sha_bytes(body.group(1).encode())!=sha.group(1):raise RuntimeError('SCIENCE_HASH_MISMATCH')
        science=json.loads(body.group(1))
        if len(science['runs'])!=3 or science['official_fla_executed'] or science['native_bank_mutated']:
            raise RuntimeError('INVALID_SCOPE')
        totals=[]
        for run in science['runs']:
            if (run['parameters']!=115 or run['test_cases']!=128 or
                not run['eval_weights_unchanged'] or
                not run['train_final']<run['train_initial']-.30 or
                not run['test_final']<run['test_initial']-.30 or
                not run['classification_accuracy_final']>=.95):
                raise RuntimeError('FAILED_LEARNED_LAYER_CHECK:'+str(run['seed']))
            if run['true_route_complete']<(100 if run['two_hop'] else 120):
                raise RuntimeError('FAILED_HELDOUT_ROUTE:'+str(run['seed']))
            strong=run['deterministic_key_goal'];simple=run['deterministic_key_only']
            if strong['cases']!=128 or simple['cases']!=128 or strong['correct']<115:
                raise RuntimeError('DETERMINISTIC_BASELINE_MISSING')
            totals.append(run['true_route_complete'])
            print('SEED='+str(run['seed'])+';TWO_HOP='+str(run['two_hop'])+
                  ';CLASSIFICATION='+str(run['classification_accuracy_final'])+
                  ';TRUE_ROUTES='+str(run['true_route_complete'])+
                  ';STRONG_ROUTES='+str(strong['route_exact']),flush=True)
        receipt('science',{'sha256':sha.group(1),'runs':3,'true_routes':totals},manifest)
        print('GATE_SCIENCE=PASS;R18_SCIENCE_SHA256='+sha.group(1),flush=True)
    elif gate=='finalize':
        t=verify_receipt('tests',manifest)
        s=verify_receipt('science',manifest)
        if t['pytest']!=TEST_COUNT or t['r17_bitwise']!=80 or s['runs']!=3:
            raise RuntimeError('GATE_RECEIPTS_INCOMPLETE')
        # Check all pinned sources again after both gates.
        preflight()
        print('=== SEND BACK ===',flush=True)
        print('PHASE=ERSA_R18_TRAINABLE_TORCH_TOKEN_MIXER_AND_NATIVE_READ_BRIDGE_R0')
        print('DISPOSITION=PASS_MODEL_COMPATIBLE_BOUNDED_TRAINABLE_READ_ONLY_ERSA_LAYER_ONLY')
        print('PYTEST_PASSED='+str(t['pytest']))
        print('FROZEN_R17_R1_PINS=122/122;SSC_PINS=7/7;R2_SOURCE_PINS=4/4')
        print('R18_OWN_FILES='+str(n))
        print('R17_R1_BITWISE_REPLAY=80/80;R17_R1_SCIENCE_REPLAY=PRIOR_ONLY')
        print('R17_R1_PRIOR_SCIENCE_SHA256='+R17_SCIENCE_PRIOR)
        print('R18_SCIENCE_SHA256='+s['sha256'])
        print('ERSA_LAYER_PARAMETERS=79;TASK_DECODER_PARAMETERS=36;TOTAL_TRAINABLE_PARAMETERS=115;HELDOUT_SEEDS=3;ROUTES='+str(s['true_routes']))
        print('TORCH_LAYER=SOFT_DIFFERENTIABLE_SELECT_CONTINUE_HALT;NATIVE_C=READONLY_SNAPSHOT_AND_FROZEN_SCORE_DIFFERENTIAL')
        print('INPUT=BATCH_TOKEN_STATE;MAX_SLOTS=8192_HARD_CEILING;MAX_HOPS=16_HARD_CEILING;BOUNDED_WORK=YES')
        print('STATE_AUTHORITY=R11_C_RAM_READ_ONLY_BRIDGE;HOST_ADMISSION_REQUIRED;INFERENCE_LEARNING=NONE')
        print('ERSA_STRONG_DETERMINISTIC_SUPERIORITY=HOLD;GDN_PARITY=HOLD;SEMANTIC_GENERALIZATION=HOLD')
        print('PRODUCTION_RIGHTS=HOLD;GPU=NONE;NO_CHECKPOINTS=TRUE;PROJECT_MUTATIONS=NONE_INTENDED')
        print('OS_NVME_WRITE_ATTESTATION=NONE;EXIT_STATUS=0')
    else:raise RuntimeError('INVALID_GATE')
    preflight()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate',choices=('science','tests','finalize'),required=True)
    a=parser.parse_args();started=time.monotonic()
    try:run(a.gate)
    except BaseException as e:
        print('=== SEND BACK ===')
        print('PHASE=ERSA_R18_TRAINABLE_TORCH_TOKEN_MIXER_AND_NATIVE_READ_BRIDGE_R0;GATE='+a.gate)
        print('DISPOSITION=HOLD;BLOCKER='+type(e).__name__+':'+str(e))
        print('EXIT_STATUS=1')
        return 1
    print('GATE='+a.gate+';ELAPSED_S='+str(round(time.monotonic()-started,3)),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
