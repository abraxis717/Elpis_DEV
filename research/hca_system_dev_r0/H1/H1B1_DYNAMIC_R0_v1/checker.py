"""Independent schema, oracle, privacy and paired-intervention checks; reject mutations."""
from __future__ import annotations
from collections import Counter,defaultdict
import json
import oracle

RULES=(('A','C'),('A','D'),('B','C'),('B','D'))
TYPES=('source_swap','event_order','epoch_rebind','structural')

class ContractError(ValueError):
    pass

def reject(message):
    raise ContractError(message)

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'))

def statuses(scene):
    return Counter(s['valid'] for s in oracle.snapshot(scene['events']).values())

def event_bag(scene):
    return Counter((e['source'],e['status'],e['epoch'],e['kind']) for e in scene['events'])

def factor_shapes(scene):
    return Counter((f['rel'],f['roles']['from'],f['roles']['to'],f['roles']['scope'], f['source'],f['source_version']) for f in scene['factors'])

def role_marginals(scene):
    return Counter((f['rel'],role,node) for f in scene['factors'] for role,node in f['roles'].items())

def check(scene_data, labels, other_world_ids=(), expected_split=None):
    if scene_data.get('schema')!='elpis.hca.h1b1.dynamic.v1': reject('schema')
    if scene_data.get('split') not in ('TRAIN','TUNE'): reject('split')
    if expected_split is not None and scene_data['split']!=expected_split: reject('wrong split')
    worlds=scene_data['worlds']
    if len(worlds)==0: reject('no worlds')
    ids=set(other_world_ids); all_cases=set(); pairs_n=Counter(); score_n=Counter()
    expected_labels={}
    for world in worlds:
        wid=world['world_id']
        if wid in ids: reject('duplicate world or cross-split identity')
        ids.add(wid)
        if len(world['support'])!=2: reject('invalid support count')
        for s in world['support']:
            if type(s.get('label')) is not int or s['label'] not in (0,1): reject('invalid support label')
            validate_scene(s,expect_id=False)
        rule=oracle.identify_rule(world['support'],RULES)
        pairs=defaultdict(list)
        for probe in world['probes']:
            validate_scene(probe,expect_id=True)
            cid=probe['case_id']
            if cid in all_cases: reject('duplicate case id')
            all_cases.add(cid)
            pid=probe['pair_id']
            if probe['pair_type'] not in TYPES: reject('unknown pair type')
            pairs[pid].append(probe)
            calculated=oracle.solve(probe,rule)
            expected_labels[cid]=calculated
            if cid not in labels or type(labels[cid]) is not int or labels[cid]!=calculated:
                reject('missing/mismatched independent golden label')
            score_n['rows']+=1
        for pid,group in pairs.items():
            if len(group)!=2: reject('pair cardinality')
            a,b=group
            typ=a['pair_type']
            if typ!=b['pair_type']: reject('pair type drift')
            if a['query']!=b['query']: reject('paired query changed')
            la,lb=expected_labels[a['case_id']],expected_labels[b['case_id']]
            if sorted((la,lb))!=[0,1]: reject('pair not opposite truth')
            if len(a['factors'])!=len(b['factors']): reject('paired factor count changed')
            if len(a['events'])!=len(b['events']): reject('paired event count changed')
            if typ=='source_swap':
                if factor_shapes(a)!=factor_shapes(b) or statuses(a)!=statuses(b):
                    reject('source swap nuisance mismatch')
            if typ=='event_order':
                if factor_shapes(a)!=factor_shapes(b) or event_bag(a)!=event_bag(b):
                    reject('event order nuisance mismatch')
            if typ=='epoch_rebind':
                if statuses(a)!=statuses(b) or canonical(a['events'])!=canonical(b['events']):
                    reject('epoch pair event mismatch')
            if typ=='structural':
                if canonical(a['events'])!=canonical(b['events']) or role_marginals(a)!=role_marginals(b):
                    reject('structural role-marginal mismatch')
            pairs_n[typ]+=1
        if set(pairs_n)!=set(TYPES) and len(worlds)==1:
            reject('missing intervention type')
    if set(expected_labels)!=set(labels): reject('gold extra/missing ids')
    for typ in TYPES:
        if pairs_n[typ] != len(worlds)*4: reject('pair type count %s %d' % (typ,pairs_n[typ]))
    return {'rows':score_n['rows'],'worlds':len(worlds),'pairs':dict(pairs_n),
            'world_ids':sorted(ids),'gold_labels':dict(Counter(labels.values()))}

def validate_scene(scene, expect_id):
    if expect_id:
        if set(scene)!={'case_id','pair_id','pair_type','query','factors','events'}: reject('probe public schema')
    else:
        if set(scene)!={'query','factors','events','label'}: reject('support schema')
    q=scene['query']
    if set(q)!={'from','to','scope'}: reject('query schema')
    if any(not isinstance(v,str) or not v for v in q.values()): reject('query identity')
    seen=set(); at=set()
    for f in scene['factors']:
        if set(f)!={'fid','rel','roles','source','source_version'}: reject('factor schema')
        if f['fid'] in seen: reject('duplicate factor id')
        seen.add(f['fid'])
        if f['rel'] not in ('A','B','C','D'): reject('relation')
        if set(f['roles'])!={'from','to','scope'}: reject('n-ary roles')
        if any(not isinstance(v,str) or not v for v in f['roles'].values()): reject('factor node identity')
        if not isinstance(f['source'],str) or not f['source']: reject('source identity')
        if type(f['source_version']) is not int or f['source_version']<0: reject('source version')
    previous=None
    for e in scene['events']:
        if set(e)!={'tick','ordinal','source','status','epoch','kind'}: reject('event schema')
        if e['kind']!='SOURCE' or e['status'] not in ('VALID','STALE','UNKNOWN'): reject('event enum')
        if any(type(e[k]) is not int or e[k]<0 for k in ('tick','ordinal','epoch')): reject('event index')
        key=(e['tick'],e['ordinal'])
        if previous is not None and key<=previous: reject('noncanonical time/ordinal')
        previous=key
        if key in at: reject('duplicate time')
        at.add(key)
