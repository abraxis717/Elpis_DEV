"""H1B1 v1 PUBLIC grammar and deterministic TRAIN/TUNE generator; NO sealed examples."""
from __future__ import annotations
import hashlib
import json
import random

SCHEMA = 'elpis.hca.h1b1.dynamic.v1'
RULES = [('A','C'), ('A','D'), ('B','C'), ('B','D')]
SOURCE_STATUS = ('VALID','STALE','UNKNOWN')
EVENT_KINDS = ('SOURCE',)

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def digest(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def opaque(rng):
    return '%032x' % rng.getrandbits(128)

def factor(rng, rel, start, end, scope, source, version=0):
    return {'fid':opaque(rng), 'rel':rel, 'roles':{'from': start, 'to': end, 'scope':scope},
            'source':source, 'source_version':version}

def event(tick, ordinal, source, status, epoch):
    return {'tick':tick,'ordinal':ordinal,'source':source,'status':status,'epoch':epoch,'kind':'SOURCE'}

def base_events(src):
    return [event(0,i,s,'VALID',0) for i,s in enumerate(src)]

def make_case(rng, pair_type, is_positive, rule, identifiers, pair_id, case_id):
    # The positive/negative distinction is solely a generator-side construction aid.
    # No polarity, generator rule or golden label is published in prediction inputs.
    a,b,c,d,scope,other_scope, s0,s1,s2,s3 = identifiers
    q = {'from':a,'to':c,'scope':scope}
    factors = [factor(rng,rule[0],a,b,scope,s0),factor(rng,rule[1],b,c,scope,s1)]
    # Distractors: a viable path of a nonmatching relation composition, and wrong scope.
    wrong = RULES[(RULES.index(rule)+1)%len(RULES)]
    factors += [factor(rng,wrong[0],a,d,scope,s3), factor(rng,wrong[1],d,c,scope,s3),
                factor(rng,rule[0],a,d,other_scope,s3),factor(rng,rule[1],d,c,other_scope,s3)]
    sources = [s0,s1,s2,s3]
    ev = base_events(sources)
    if pair_type == 'source_swap':
        # Same graph and statuses histogram, but VALID assigned to causal versus decoy source.
        ev += [event(1,0,s0,'VALID' if is_positive else 'STALE',0),
               event(1,1,s2,'STALE' if is_positive else 'VALID',0)]
    elif pair_type == 'event_order':
        # Same multiset of source/status events; only their time order changes.
        if is_positive:
            ev += [event(1,0,s0,'STALE',0),event(1,1,s2,'VALID',0),
                   event(2,0,s0,'VALID',0),event(2,1,s2,'STALE',0)]
        else:
            ev += [event(1,0,s0,'VALID',0),event(1,1,s2,'STALE',0),
                   event(2,0,s0,'STALE',0),event(2,1,s2,'VALID',0)]
    elif pair_type == 'epoch_rebind':
        # Both versions advance; exactly one replacement factor is created at epoch 1.
        ev += [event(1,0,s0,'VALID',1),event(1,1,s2,'VALID',1)]
        if is_positive:
            factors.append(factor(rng,rule[0],a,b,scope,s0,1))
        else:
            factors.append(factor(rng,wrong[0],a,d,scope,s2,1))
    elif pair_type == 'structural':
        # Preserve n-ary role-occurrence multiset while rewiring the decisive edge.
        # Query a -> c is connected only in the positive version.
        factors = [factor(rng,rule[0],a,b,scope,s0),factor(rng,rule[0],d,identifiers[3],scope,s0),
                   factor(rng,rule[1],b,c if is_positive else d,scope,s1),
                   factor(rng,rule[1],d,d if is_positive else c,scope,s1),
                   factor(rng,wrong[0],a,d,other_scope,s3)]
        # For negative, b->d and d->c. For positive, b->c and d->d.
        # The 'from'/'to' marginals are identical across the two variants.
    else:
        raise ValueError('unknown pair type')
    # Case and factor ID values are randomized independently of label, then serialized canonically.
    rng.shuffle(factors)
    # Event serialization is canonical: preserve tick/ordinal semantics, not insertion order.
    ev.sort(key=lambda e:(e['tick'],e['ordinal'],e['source']))
    return {'case_id':case_id,'pair_id':pair_id,'pair_type':pair_type,
            'query':q,'factors':factors,'events':ev}

def support_case(rng, subset, rule):
    nodes=[opaque(rng) for _ in range(7)]
    a,b,c,d,e,scope,src=nodes
    factors=[]
    for i in subset:
        x= b if i%2 == 0 else d
        r=RULES[i]
        factors += [factor(rng,r[0],a,x,scope,src), factor(rng,r[1],x,c,scope,src)]
    rng.shuffle(factors)
    return {'query':{'from':a,'to':c,'scope':scope},'factors':factors,
            'events':[event(0,0,src,'VALID',0)],'label':int(RULES.index(rule) in subset)}

def generate_world(master_rng, world_index, split, pairs_per_type=4):
    seed=master_rng.getrandbits(128)
    rng=random.Random(seed)
    world_id=opaque(rng)
    rule=RULES[rng.randrange(len(RULES))]
    supports=[support_case(rng,subset,rule) for subset in ([0,1],[0,2])]
    probes=[]
    for kind in ('source_swap','event_order','epoch_rebind','structural'):
        for _ in range(pairs_per_type):
            pair_id=opaque(rng)
            ids=[opaque(rng) for i in range(10)]
            # The construction has invariant labels, but case IDs are generated before
            # deciding which presentation slot receives each polarity.
            id1,id2=opaque(rng),opaque(rng)
            first_positive=bool(rng.getrandbits(1))
            probes.append(make_case(rng,kind,first_positive,rule,ids,pair_id,id1))
            probes.append(make_case(rng,kind,not first_positive,rule,ids,pair_id,id2))
    rng.shuffle(probes)
    return {'world_id':world_id,'support':supports,'probes':probes}

def build_split(name, seed, worlds, pairs_per_type=4):
    if name not in ('TRAIN','TUNE'):
        raise ValueError('Only TRAIN and TUNE are legal in H1B1; no SEALED generator')
    rng=random.Random(seed)
    output={'schema':SCHEMA,'split':name,'worlds':[]}
    for i in range(worlds):
        output['worlds'].append(generate_world(rng,i,name,pairs_per_type))
    return output
