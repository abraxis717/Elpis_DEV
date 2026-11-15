"""R5 additional independent diagnostics; frozen fixtures and oracle unchanged."""
import hashlib,json
from elpis_ersa import bank_ersa as B,fixtures_r2 as F,harness_ersa as H,geometry_ersa as G


class Complete(B.ErsBank):
    def begin_session(self,spec=None,registry=None,transport=None,authority=None):
        a=dict(authority or {});a['modes']={'discovery':'score_complete','selection':'ersa'}
        return super().begin_session(spec,registry,transport,a)


def main():
    results={}
    for fam in F.FAMILIES:
        sc=F.build(fam,cand=8)
        result=H.evaluate_backend(Complete,sc,fam)
        results[fam]={x:result[x] for x in ('exact_match','required_recall',
            'candidate_recall','forbidden_leak','gauge_covariance')}
    task={}
    for fam,ids in (('sparse_evidence',(9,)),('contradictory_attested',(0,1)),
                    ('near_identical_keys_distinct_identity',(0,))):
        sc=F.build(fam,cand=8); b=H.replay(Complete,sc); q=sc['queries'][0]
        if fam=='contradictory_attested':
            rc=b.aggregate_reciprocal_contradiction(0,q['qkey'],q['qgoal'],q['qdir'])
        else:
            rc=b.aggregate_scoped(ids,q['qkey'],q['qgoal'],q['qdir'])
        evidence=[x['relation_id'] for x in b._last_evidence]
        task[fam]={'status':rc.status,'evidence':evidence,
                   'required_recall':sum(r in evidence for r in q['required'])/len(q['required']),
                   'forbidden_leak':bool(set(evidence)&set(q.get('forbidden') or ()))}
    out={'complete':results,'scoped':task}
    text=json.dumps(out,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(text.encode()).hexdigest()
    print('R5_RETRIEVAL_METRICS='+text)
    print('R5_RETRIEVAL_SHA256='+digest)
    return out,digest

if __name__=='__main__':main()
