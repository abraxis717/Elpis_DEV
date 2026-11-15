"""Independent frozen-fixture diagnostics for ERSA R4 (no source mutation)."""
from __future__ import annotations
import hashlib,json
from elpis_ersa import bank_ersa as B,fixtures_ersa as F0,fixtures_r2 as F2,harness_ersa as H


def backend(mode):
    class Configured(B.ErsBank):
        def begin_session(self,spec=None,registry=None,transport=None,authority=None):
            a=dict(authority or {});a['modes']={'discovery':mode,'selection':'ersa'}
            return super().begin_session(spec,registry,transport,a)
    return Configured


def families(builder,names,k,mode):
    results={}
    for name in names:
        try:sc=builder(name,cand=k)
        except TypeError:sc=builder(name)
        results[name]=H.evaluate_backend(backend(mode),sc,name)
    return {
        'exact':sum(v['exact_match'] is True for v in results.values()),
        'total':len(results),
        'forbidden':sum(v['forbidden_leak'] for v in results.values()),
        'required_zero':sorted(n for n,v in results.items()
                               if v['required_recall']==0.),
        'candidate_zero':sorted(n for n,v in results.items()
                               if v['candidate_recall']==0.),
        'partial_required':sorted(n for n,v in results.items()
                                  if 0.<v['required_recall']<1.),
        'unmatched':sorted(n for n,v in results.items() if not v['exact_match']),
    }


def main():
    out={}
    for nm,build,names in (('R0',F0.build,F0.FAMILIES),('R2',F2.build,F2.FAMILIES)):
        for mode in ('indexed','score_indexed','dense'):
            out[f'{nm}:{mode}:K8']=families(build,names,8,mode)
    sc=F2.build('near_identical_keys_distinct_identity',cand=8)
    b=H.replay(backend('score_indexed'),sc)
    q=sc['queries'][0]
    sel=b.aggregate(q['qkey'],q['qgoal'],k=q['k'],qdir=q['qdir'],exact_relation_id=0)
    out['exact_identity_safety']={
        'receipt':sel.status,
        'selected':[int(x[0]) for x in b._last_selection],
        'evidence':[e['relation_id'] for e in b._last_evidence],
        'forbidden_absent':all(x['relation_id']!=1 for x in b._last_evidence)}
    canon=json.dumps(out,sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(canon).hexdigest()
    print('R4_RETRIEVAL_METRICS='+json.dumps(out,sort_keys=True,separators=(',',':')))
    print('R4_RETRIEVAL_SHA256='+digest)
    return out,digest

if __name__=='__main__': main()
