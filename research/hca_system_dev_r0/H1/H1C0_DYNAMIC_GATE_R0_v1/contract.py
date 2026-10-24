"""Prospective H1C0 TRAIN/TUNE-only multi-query event-sourced hypergraph generator.
This is generator-side source; NEVER provided to predictors as a hidden label feature.
"""
import random
from copy import deepcopy

SCHEMA='elpis.hca.h1c0.dynamic_multiquery.public.v1'
KINDS=('source_swap','event_order','epoch_rebind','structural')
RELS=('R0','R1','R2','RX')

def token(rng):
    return 'z'+format(rng.getrandbits(112),'028x')

def event(t,src,status,epoch=0):
    return {'step':t,'source':src,'status':status,'epoch':epoch}

def factor(fid,rel,a,b,scope,context,source,version=0):
    return {'id':fid,'rel':rel,'roles':{'from':a,'to':b,'scope':scope,'context':context},
            'source':source,'version':version}

def make_pair(rng,kind):
    nodes=[token(rng) for _ in range(18)]
    a,b,c,d,e,f=nodes[:6]
    ss=[token(rng) for _ in range(8)]
    scopes=[token(rng) for _ in range(2)]
    contexts=[token(rng) for _ in range(2)]
    ids=[token(rng) for _ in range(40)]
    primary=factor(ids[0],'R0',a,b,scopes[0],contexts[0],ss[0],int(kind=='epoch_rebind'))
    mid=factor(ids[1],'R1',b,c,scopes[0],contexts[0],ss[1])
    tail=factor(ids[2],'R2',c,d,scopes[0],contexts[0],ss[2])
    decoy_mid=factor(ids[3],'R1',f,e,scopes[0],contexts[0],ss[1])
    # Independently queryable second chain, protected by different scope/context.
    second=[factor(ids[4],'R0',nodes[6],nodes[7],scopes[1],contexts[1],ss[4]),
            factor(ids[5],'R1',nodes[7],nodes[8],scopes[1],contexts[1],ss[5]),
            factor(ids[6],'R2',nodes[8],nodes[9],scopes[1],contexts[1],ss[6])]
    # Three further legitimate 3-hop witness chains give multiple nontrivial
    # simultaneous queries, rather than padding almost every row with zeros.
    more_chains=[]
    for k in range(3):
        x,y,z,w=(nodes[10+k],nodes[11+k],nodes[12+k],nodes[13+k])
        base=7+3*k
        more_chains.extend([factor(ids[base],'R0',x,y,scopes[1],contexts[1],ss[4]),
                            factor(ids[base+1],'R1',y,z,scopes[1],contexts[1],ss[5]),
                            factor(ids[base+2],'R2',z,w,scopes[1],contexts[1],ss[6])])
    fillers=[]
    for i in range(16,36):
        rel=RELS[rng.randrange(4)]
        x=nodes[10+rng.randrange(8)];y=nodes[10+rng.randrange(8)]
        scope=scopes[rng.randrange(2)];context=contexts[rng.randrange(2)]
        fillers.append(factor(ids[i],rel,x,y,scope,context,ss[3+rng.randrange(5)]))
    queries=[{'id':token(rng),'from':a,'to':d,'scope':scopes[0],'context':contexts[0]},
             {'id':token(rng),'from':nodes[6],'to':nodes[9],'scope':scopes[1],'context':contexts[1]}]
    for k in range(3):
        queries.append({'id':token(rng),'from':nodes[10+k],
                        'to':nodes[13+k],'scope':scopes[1],'context':contexts[1]})
    for _ in range(3):
        queries.append({'id':token(rng),'from':nodes[rng.randrange(18)],
                        'to':nodes[rng.randrange(18)],'scope':scopes[rng.randrange(2)],
                        'context':contexts[rng.randrange(2)]})
    target_id=queries[0]['id']
    rng.shuffle(queries)
    common=[event(0,ss[1],'VALID'),event(1,ss[2],'VALID'),
            event(2,ss[4],'VALID'),event(3,ss[5],'VALID'),event(4,ss[6],'VALID'),
            event(5,ss[7],'STALE')]
    post=[event(10,ss[2],'STALE'),event(11,ss[2],'VALID'),
          event(12,ss[4],'UNKNOWN'),event(13,ss[4],'VALID'),
          event(14,ss[1],'STALE'),event(15,ss[1],'VALID')]
    pair_id=token(rng)
    cases=[]
    for positive in (False,True):
        m=factor(mid['id'],mid['rel'],b,c if positive or kind!='structural' else e,
                 scopes[0],contexts[0],ss[1])
        dm=factor(decoy_mid['id'],decoy_mid['rel'],f,e if positive or kind!='structural' else c,
                  scopes[0],contexts[0],ss[1])
        factors=[primary,m,tail,dm]+second+more_chains+fillers
        events=list(common)
        if kind=='source_swap':
            events += [event(6,ss[0],'VALID' if positive else 'STALE'),
                       event(7,ss[3],'STALE' if positive else 'VALID')]
        elif kind=='event_order':
            events += [event(6,ss[0],'STALE' if positive else 'VALID'),
                       event(7,ss[3],'VALID'),
                       event(8,ss[0],'VALID' if positive else 'STALE')]
        elif kind=='epoch_rebind':
            events += [event(6,ss[0],'VALID',0),event(7,ss[3],'VALID',0),
                       event(8,ss[0],'VALID',1 if positive else 0),
                       event(9,ss[3],'VALID',0 if positive else 1)]
        elif kind=='structural':
            events += [event(6,ss[0],'VALID'),event(7,ss[3],'STALE')]
        else:raise ValueError('unknown kind')
        events += post
        # No case-order or identity signal: both case IDs created before polarity shuffle.
        cases.append({'case_id':token(rng),'pair_id':pair_id,'kind':kind,
                      'target_query_id':target_id,'factors':deepcopy(factors),
                      'queries':deepcopy(queries),'events':deepcopy(events)})
    rng.shuffle(cases)
    return cases

def build_split(split,seed,worlds=2,pairs_per_kind=2):
    if split not in ('TRAIN','TUNE'):raise ValueError('no SEALED split generator')
    rng=random.Random(seed)
    out={'schema':SCHEMA,'split':split,'worlds':[]}
    for _ in range(worlds):
        wr=random.Random(rng.getrandbits(128))
        cases=[]
        for kind in KINDS:
            for _ in range(pairs_per_kind):cases.extend(make_pair(wr,kind))
        wr.shuffle(cases)
        out['worlds'].append({'world_id':token(wr),'cases':cases})
    return out
