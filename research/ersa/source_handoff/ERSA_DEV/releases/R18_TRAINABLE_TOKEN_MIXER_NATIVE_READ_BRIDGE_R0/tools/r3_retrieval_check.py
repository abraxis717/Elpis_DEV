"""Read-only frozen R0/R2 accuracy diagnostic; no filesystem writes."""
import json,sys
from elpis_ersa import fixtures_ersa as F, fixtures_r2 as F2, harness_ersa as H, bank_ersa as B


def backend(discovery):
    class E(B.ErsBank):
        def begin_session(self,spec=None,registry=None,transport=None,authority=None):
            a=dict(authority or {}); a['modes']={'discovery':discovery, 'selection':'ersa'}
            return super().begin_session(spec,registry,transport,a)
    return E


def main():
    out={}
    for gen,names,prefix in ((F.build,F.FAMILIES,'R0'),(F2.build,F2.FAMILIES,'R2')):
        for mode in ('indexed','dense'):
            for k in (4,8):
                results={}
                for name in names:
                    try: sc=gen(name,cand=k)
                    except TypeError: sc=gen(name)
                    try: results[name]=H.evaluate_backend(backend(mode),sc,name)
                    except Exception as ex: results[name]={'error':f'{type(ex).__name__}:{ex}'}
                out[f'{prefix}:{mode}:K{k}']={
                    'total':len(results),'exact':sum(v.get('exact_match') is True for v in results.values()),
                    'forbidden':sum(bool(v.get('forbidden_leak')) for v in results.values()),
                    'errors':sum('error' in v for v in results.values()),
                    'required_min':min((v.get('required_recall',1.) for v in results.values()),default=1.),
                    'candidate_min':min((v.get('candidate_recall',1.) for v in results.values()),default=1.),
                    'per_family':H.summary(results)}
    print(json.dumps(out,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
