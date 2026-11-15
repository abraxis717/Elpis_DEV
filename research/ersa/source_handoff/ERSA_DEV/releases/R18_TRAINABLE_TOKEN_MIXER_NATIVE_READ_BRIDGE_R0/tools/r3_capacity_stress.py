"""8192 fixed-array stress, zero runtime persistent writes; independent index checks."""
from __future__ import annotations
import random, time
from elpis_ersa import contract_ersa as C, bank_ersa as B

def run(cap=8192,extra=64):
  rng=random.Random(310527)
  recs=[C.AdmissionRecord(i,7,0,i,(),7,0,1<<62,i%512==1) for i in range(1,cap+extra+1)]
  spec=C.SessionSpec(max_active_relations=cap,max_candidates=8,
                     max_control_records=cap+extra+1,max_ops_per_tick=cap*4,
                     relational_namespace=(1,cap+extra+1),pending_stage_limit=4)
  b=B.ErsBank(); assert b.begin_session(spec,recs).status==C.STATUS_OK
  state_bytes=sum(int(v.nbytes) for c in (b._arrays,b._index) for v in c.values() if hasattr(v,'nbytes'))
  protected={i for i in range(1,cap+1) if i%512==1}
  start=time.perf_counter()
  for i in range(1,cap+1):
    x=b.insert_relation(i,(i,i%13,i%41,i%7),(float(i%11),1.,0.,0.),provenance=7,ordinal=i,ticket=7)
    assert x.status==C.STATUS_OK,(i,x)
    if i%1024==0:print('CAPACITY_PROGRESS',i,flush=True)
  assert b.occupancy()['live']==cap
  before={i:b._slot_of(i) for i in protected}
  for i in range(cap+1,cap+extra+1):
    x=b.insert_relation(i,(i,i%13,i%41,i%7),(1.,1.,1.,1.),provenance=7,ordinal=i,ticket=7)
    assert x.status==C.STATUS_OK,(i,x)
  assert b.occupancy()['live']==cap
  for i in protected:
    assert b._slot_of(i) is not None
    assert b._arrays['revision'][b._slot_of(i)]==i
  for s in range(cap):
    if b._arrays['live'][s]:
      rid=int(b._arrays['rid'][s]); assert b._slot_of(rid)==s
  live={int(b._arrays['rid'][s]) for s in range(cap) if b._arrays['live'][s]}
  assert len(live)==cap
  assert b._pending_used==0 and len(b._stage_index)==0
  after_bytes=sum(int(v.nbytes) for c in (b._arrays,b._index) for v in c.values() if hasattr(v,'nbytes'))
  assert state_bytes==after_bytes
  print(f'CAPACITY_PASS=TRUE;CAP={cap};EXTRA_INSERTS={extra};PROTECTED={len(protected)};LIVE={len(live)};BANK_BYTES={state_bytes};ELAPSED_S={time.perf_counter()-start:.3f}')
  return dict(cap=cap,extra=extra,protected=len(protected),live=len(live),bank_bytes=state_bytes)

if __name__=='__main__':run()
