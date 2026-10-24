"""Observer adequacy: forbidden mutations MUST raise AuditFailure.
No broad exception handler can swallow a failed assertion.
"""
from copy import deepcopy
from independent import inspect_split,AuditFailure,independent_labels

def run(ds,gold,pred,work,result,which='TUNE'):
    tests={}
    def attack(name,action):
        x=deepcopy(ds);g=deepcopy(gold);p=deepcopy(pred);w=deepcopy(work);r=deepcopy(result)
        action(x,g,p,w,r)
        try:inspect_split(x,g,p,w,r,which)
        except AuditFailure:tests[name]='REJECTED'
        else:raise AssertionError('MUTANT_ACCEPTED='+name)
    first=lambda d:d['worlds'][0]['cases'][0]
    def gold_flip(d,g,p,w,r):
        s=first(d);g[s['case_id']][0][s['queries'][0]['id']] ^= 1
    attack('gold_flip',gold_flip)
    def prediction_flip(d,g,p,w,r):
        s=first(d);p[s['case_id']][0][s['queries'][0]['id']] ^= 1
    attack('prediction_flip',prediction_flip)
    attack('schema_label_leak',lambda d,g,p,w,r:first(d).update({'gold':1}))
    attack('duplicate_case_identity',lambda d,g,p,w,r:d['worlds'][0]['cases'][1].update(case_id=first(d)['case_id']))
    attack('missing_case',lambda d,g,p,w,r:d['worlds'][0]['cases'].pop())
    attack('duplicate_factor',lambda d,g,p,w,r:first(d)['factors'][1].update(id=first(d)['factors'][0]['id']))
    attack('query_collision',lambda d,g,p,w,r:first(d)['queries'][1].update(id=first(d)['queries'][0]['id']))
    attack('out_of_order_event',lambda d,g,p,w,r:first(d)['events'][1].update(step=-1))
    attack('invalid_status',lambda d,g,p,w,r:first(d)['events'][0].update(status='FABRICATED'))
    attack('invalid_epoch',lambda d,g,p,w,r:first(d)['factors'][0].update(version=22))
    attack('unknown_event_source',lambda d,g,p,w,r:first(d)['events'][0].update(source='unknown'))
    attack('pair_query_drift',lambda d,g,p,w,r:first(d).update(target_query_id='nonsense'))
    attack('extra_prediction_case',lambda d,g,p,w,r:p.update(ghost=[]))
    attack('case_work_duplicate',lambda d,g,p,w,r:w.__setitem__(1,deepcopy(w[0])))
    attack('result_total_tamper',lambda d,g,p,w,r:r['stats'].__setitem__('rows',r['stats']['rows']+1))
    attack('logical_work_tamper',lambda d,g,p,w,r:r['logical_work']['incremental'].__setitem__('setup_factor',-3))
    attack('payload_tamper',lambda d,g,p,w,r:r['serialized_payload_peak_bytes'].__setitem__('incremental',0))
    def pair_nuisance(d,g,p,w,r):
        s=next(s for s in d['worlds'][0]['cases'] if s['kind']=='event_order')
        s['events'][6]['status']='UNKNOWN'
        recal=independent_labels(s)
        g[s['case_id']]=recal;p[s['case_id']]=deepcopy(recal)
        # Target equality and nuisance checks must catch this independently of label parity.
    attack('pair_event_nuisance_with_rescoring',pair_nuisance)
    return tests
