"""Self-tests including adversarial mutations. No catch-all assertion swallowing."""
import copy
from collections import Counter
import checker
from contract import build_split, RULES
import oracle


def gold(ds):
    labels={}
    for w in ds['worlds']:
        rule=oracle.identify_rule(w['support'],RULES)
        for p in w['probes']:
            labels[p['case_id']]=oracle.solve(p,rule)
    return labels

def shall_reject(ds, labels, name, expect):
    try:
        checker.check(ds,labels)
    except checker.ContractError as ex:
        if expect not in str(ex):
            raise AssertionError('wrong rejection for '+name+': '+str(ex)) from ex
    else:
        raise AssertionError('MUTANT ACCEPTED '+name)
    print('MUTANT_REJECTED='+name,flush=True)

def test_all():
    ds=build_split('TRAIN',8271060,2)
    labels=gold(ds)
    stats=checker.check(ds,labels,expected_split='TRAIN')
    if stats['rows']!=64 or stats['pairs']!={x:8 for x in checker.TYPES}:
        raise AssertionError('wrong row count')
    print('PASS_H1B1_VALID_FIXTURE rows=64 pairs=32',flush=True)
    actions=[]
    def mutate(name, fn, expected):
        d,l=copy.deepcopy(ds),dict(labels)
        fn(d,l)
        shall_reject(d,l,name,expected)
        actions.append(name)
    mutate('gold_flip',lambda d,l:l.__setitem__(next(iter(l)),1-l[next(iter(l))]),'golden label')
    mutate('label_leak',lambda d,l:d['worlds'][0]['probes'][0].__setitem__('label',0),'probe public schema')
    mutate('case_duplicate',lambda d,l:d['worlds'][0]['probes'][1].__setitem__('case_id',d['worlds'][0]['probes'][0]['case_id']),'duplicate case id')
    mutate('pair_drop',lambda d,l:(l.pop(d['worlds'][0]['probes'][0]['case_id']),d['worlds'][0]['probes'].pop(0)),'pair cardinality')
    mutate('query_drift',lambda d,l:d['worlds'][0]['probes'][0]['query'].__setitem__('to','FORGED'),'paired query changed')
    # Guaranteed invalid event: duplicate time index.
    mutate('event_collision',lambda d,l:d['worlds'][0]['probes'][0]['events'][1].__setitem__('ordinal',0),'noncanonical time')
    mutate('duplicate_factor',lambda d,l:d['worlds'][0]['probes'][0]['factors'][1].__setitem__('fid',d['worlds'][0]['probes'][0]['factors'][0]['fid']),'duplicate factor id')
    mutate('missing_gold',lambda d,l:l.pop(next(iter(l))),'golden label')
    mutate('world_duplicate',lambda d,l:d['worlds'][1].__setitem__('world_id',d['worlds'][0]['world_id']),'duplicate world')
    mutate('support_truth_flip',lambda d,l:d['worlds'][0]['support'][0].__setitem__('label',1-d['worlds'][0]['support'][0]['label']),'golden label')
    mutate('split_forgery',lambda d,l:d.__setitem__('split','SEALED'),'split')
    mutate('event_enum',lambda d,l:d['worlds'][0]['probes'][0]['events'][0].__setitem__('status','GHOST'),'event enum')
    print('PASS_H1B1_MUTANT_ADEQUACY=%d/%d' %(len(actions),12),flush=True)
    return len(actions)

if __name__=='__main__': test_all()
