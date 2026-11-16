#!/usr/bin/env python3
"""Read-only SSC predecessor handoff, for R8 identity parity. Not runtime code.
Only the original 2026-11-04 pinned versions are accepted.
"""
import hashlib,io,json,os,pathlib,zipfile
ROOT=pathlib.Path('/mnt/primesauce/RALPH/DEV/Hermes/UnifiedStateConformance_R0')
OUT=pathlib.Path('/mnt/primesauce/Downloads/ERSA_SSC_PREDECESSOR_SOURCES_R0.zip')
PINS={
 'contract_r0.py':'8a905676e73226e0c2151130200f1819ced8d9fa6f837a53982c2a2014b406f0',
 'fixtures_r0.py':'65ad060410398abcf6df397617aa2e0bfaabe9bfdd9c92c549e69038e41ac56b',
 'geometry_r0.py':'56d76bd9c5eeaefd5340eece4b235c25cd7fac944250d6e15f4b855a13cabbed',
 'harness_r0.py':'70b58385e2f25926178233f5695cfafa815bd8e6ca96769b52afe59a1ed731a3',
 'mutants_r0.py':'71da2eff159b9098879dcb903c7e7df2a10be5576d4f6471a6eb1096c999dc5e',
 'oracle_r0.py':'bc58ef2dd8c85d802b3d836560e4625db18921db8dfea469d9d1e54501ced8d5',
 'reference_backend.py':'35ef5cf91bfff467224db9ccb1b849171cd0deb792e414ea8cdf5c8fc07ac879',
}
def run():
    data={};fail=[]
    for name,expected in PINS.items():
        file=ROOT/'src/elpis_ssc'/name
        try:
            if not file.is_file() or file.is_symlink() or file.stat().st_size>100000:
                raise ValueError('MISSING_SYMLINK_OR_SIZE')
            body=file.read_bytes()
            if hashlib.sha256(body).hexdigest()!=expected:
                raise ValueError('PIN_DRIFT')
            data['src/elpis_ssc/'+name]=body
        except (OSError,ValueError) as exc:
            fail.append(name+':'+str(exc))
    if fail:
        print('SSC_HANDOFF=HOLD_SOURCE_PIN_DRIFT_OR_MISSING')
        print('SSC_FILES_VERIFIED='+str(len(data))+'/'+str(len(PINS)))
        print('SSC_BLOCKERS='+';'.join(fail))
        return False
    manifest=json.dumps({'source_root':str(ROOT),'sha256':{
        path:hashlib.sha256(body).hexdigest() for path,body in sorted(data.items())},
        'role':'READ_ONLY_SSC_PREDECESSOR_SOURCE','trust':'PINNED_LOCAL_SOURCE_NOT_EXTERNAL_SEAL'},
        sort_keys=True,indent=2).encode()
    data['MANIFEST.json']=manifest
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as zip:
        for name,body in sorted(data.items()):
            zi=zipfile.ZipInfo(name,(1980,1,1,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
            zi.external_attr=(0o100644<<16)
            zip.writestr(zi,body)
    blob=b.getvalue();h=hashlib.sha256(blob).hexdigest()
    if OUT.exists():
        if OUT.is_symlink() or not OUT.is_file() or hashlib.sha256(OUT.read_bytes()).hexdigest()!=h:
            print('SSC_HANDOFF=HOLD_EXISTING_OUTPUT_CONFLICT')
            return False
    else:
        fd=os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(blob)
    print('SSC_HANDOFF=PINNED_SOURCE_ZIP_READY')
    print('SSC_FILES_VERIFIED=7/7')
    print('SSC_HANDOFF_PATH='+str(OUT))
    print('SSC_HANDOFF_SHA256='+h)
    print('SSC_PARITY=NOT_RUN_SOURCE_ONLY')
    return True
if __name__=='__main__':run()
