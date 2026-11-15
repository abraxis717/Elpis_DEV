"""Bounded CPU diagnostic; latency is NOT a qualified native runtime benchmark."""
import json,statistics,time
from elpis_ersa import bank_ersa as B,contract_ersa as C

def main():
    n=8192
    records=[C.AdmissionRecord(i,7,0,i,(),7,0) for i in range(n)]
    sp=C.SessionSpec(max_active_relations=n,max_candidates=8,
        max_control_records=n,max_ops_per_tick=20000,
        relational_namespace=(0,n),pending_stage_limit=4)
    bank=B.ErsBank();assert bank.begin_session(sp,records).status=='OK'
    before=sum(v.nbytes for part in (bank._arrays,bank._index)
               for v in part.values() if hasattr(v,'nbytes'))
    for i in range(n):
        r=bank.insert_relation(i,(i,i%13,i%41,i%7),(1.,2.,3.,4.),
                               provenance=7,ordinal=i,ticket=7)
        assert r.status=='OK',(i,r)
    obs={}
    qkey=(77,6,17,2)
    for mode in ('score_indexed','score_complete'):
        bank._modes['discovery']=mode
        times=[]
        for j in range(5):
            t=time.perf_counter_ns()
            rc=bank.query_candidates(qkey,qdir=(1.,2.,3.,4.))
            assert rc.status=='OK'
            times.append((time.perf_counter_ns()-t)/1e6)
        obs[mode]={"median_ms":round(statistics.median(times),3),
                   "scored":bank._last_scored_count,"candidates":len(bank._last_candidates)}
    after=sum(v.nbytes for part in (bank._arrays,bank._index)
              for v in part.values() if hasattr(v,'nbytes'))
    assert before==after
    assert obs['score_complete']['scored']==n
    assert obs['score_indexed']['scored']<=n
    obs['bank_array_bytes']=after
    print('R5_DIAGNOSTIC_COST='+json.dumps(obs,sort_keys=True,separators=(',',':')))
    return obs
if __name__=='__main__':main()
