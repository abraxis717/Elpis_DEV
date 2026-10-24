"""Independent reference scorer. No import of contract / generator functions."""
from collections import defaultdict

def snapshot(events):
    observed={}
    previous=None
    for x in sorted(events,key=lambda e:(e['tick'],e['ordinal'])):
        assert x['kind']=='SOURCE'
        assert type(x['epoch']) is int and x['epoch']>=0
        key=(x['tick'],x['ordinal'])
        if previous is not None and previous>=key:
            raise ValueError('duplicate/nonmonotone event tick+ordinal')
        previous=key
        # A VALID event at a new epoch invalidates every factor bound to the older epoch.
        observed[x['source']]={'epoch':x['epoch'],'valid':x['status']=='VALID'}
    return observed

def solve(scene, relation_pair):
    state=snapshot(scene['events'])
    q=scene['query']
    active=[]
    for f in scene['factors']:
        s=state.get(f['source'])
        if not s or not s['valid'] or s['epoch']!=f['source_version']:
            continue
        if f['roles']['scope']!=q['scope']:
            continue
        active.append(f)
    right=defaultdict(set)
    for f in active:
        if f['rel']==relation_pair[1]:
            right[f['roles']['from']].add(f['roles']['to'])
    for f in active:
        if (f['rel']==relation_pair[0] and f['roles']['from']==q['from']
                and q['to'] in right.get(f['roles']['to'],set())):
            return 1
    return 0

def identify_rule(support, candidates):
    agreeing=[]
    for candidate in candidates:
        if all(solve(case,candidate)==case['label'] for case in support):
            agreeing.append(candidate)
    if len(agreeing)!=1:
        raise ValueError('support insufficient to identify unique rule: %d' % len(agreeing))
    return agreeing[0]
