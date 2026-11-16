"""End-to-end host-authorized read cost; includes bank discovery, validation,
 feature extraction, scoring, native ranking, reduction and ctypes overhead.
 Does NOT include insertion cost or SSC/neural compute. No memory writes outside
 the Python process except the transient compiler .so in /dev/shm.
"""
import json,shutil,statistics,time,random,sys
from pathlib import Path
from native.ersa_native import NativeOps,build_native
from native.ersa_read_r7 import NativeReadPath
from elpis_ersa import bank_ersa as B,contract_ersa as C

class Reference(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        a=dict(authority or {});a['modes']={'discovery':'score_complete','selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)
class Native(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        a=dict(authority or {});a['modes']={'discovery':'dense','selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)

def build(cls,n):
    spec=C.SessionSpec(max_active_relations=n,max_candidates=8,
        max_control_records=n+1,max_ops_per_tick=n*4+128,
        relational_namespace=(0,n+1),pending_stage_limit=0)
    rec=[C.AdmissionRecord(i,7,0,i,(),7,0,1<<30,False) for i in range(1,n+1)]
    b=cls(); assert b.begin_session(spec,rec).status==C.STATUS_OK
    for i in range(1,n+1):
        r=b.insert_relation(i,(i%101,i%17,i%29,i%7),
            (float(i%9)/9.,float(i%11)/11.,float(i%13)/13.,float(i%5)/5.),
            provenance=7,ordinal=i,ticket=7,utility=(i%10)/10.,credibility=.9)
        if r.status!=C.STATUS_OK:raise RuntimeError('INSERT:'+repr(r))
    return b

def run():
    sizes=(256,1024,8192)
    lib=build_native();ops=NativeOps(lib)
    results=[]
    try:
        for n in sizes:
            t=time.monotonic();r=build(Reference,n);b=build(Native,n)
            path=NativeReadPath(ops)
            query=(2,5,8,3);goal=(1.,0.,2.,3.);direction=(.4,.2,-.3,.7)
            # Compare true end-to-end authorized reads and verify parity.
            reference_us=[];native_us=[]
            for rep in range(4):
                t0=time.perf_counter_ns()
                rr=r.aggregate(query,goal,k=8,qdir=direction)
                reference_us.append((time.perf_counter_ns()-t0)/1e3)
                t0=time.perf_counter_ns()
                nn=path.read(b,query,goal,qdir=direction,k=8)
                native_us.append((time.perf_counter_ns()-t0)/1e3)
                if rr.status != C.STATUS_OK or nn.status!='OK':
                    raise RuntimeError('READ_STATUS:'+str((rr,nn)))
                ids=tuple(z[0] for z in r._last_selection)
                if nn.relation_ids!=ids:raise RuntimeError('NATIVE_ID_PARITY_FAIL')
                if max(abs(x-y) for x,y in zip(nn.salient,r._last_aggregate))>1e-10:
                    raise RuntimeError('NATIVE_AGG_PARITY_FAIL')
            results.append(dict(n=n,k=8,reference_median_us=round(statistics.median(reference_us),2),
                native_median_us=round(statistics.median(native_us),2),
                parity=True,build_s=round(time.monotonic()-t,3)))
            print('R7_BENCH_PROGRESS='+json.dumps(results[-1],sort_keys=True),flush=True)
    finally:shutil.rmtree(lib.parent)
    print('R7_END_TO_END_BENCH='+json.dumps(results,sort_keys=True,separators=(',',':')))
    return results

if __name__=='__main__':run()
