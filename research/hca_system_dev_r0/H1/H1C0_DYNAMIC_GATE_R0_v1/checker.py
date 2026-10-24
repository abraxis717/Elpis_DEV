"""Separate public-contract checker: independent gold, paired interventions, split custody."""
import copy
from collections import Counter,defaultdict
import oracle

class ContractError(ValueError):pass

def require(condition,why):
    if not condition:raise ContractError(why)

def bag_factors(scene):
    return Counter((f['id'],f['rel'],f['source'],f['version'],tuple(sorted(f['roles'].items()))) for f in scene['factors'])

def shape_marginals(scene):
    return Counter((f['rel'],role,node) for f in scene['factors'] for role,node in f['roles'].items())

def event_bag(scene):
    return Counter((e['source'],e['status'],e['epoch']) for e in scene['events'])

def class_bag(scene):
    return Counter((e['status'],e['epoch']) for e in scene['events'])

def verify_case(s):
    require(set(s)=={'case_id','pair_id','kind','target_query_id','factors','queries','events'},'case schema / possible label leak')
    require(s['kind'] in ('source_swap','event_order','epoch_rebind','structural'),'unknown intervention')
    require(all(isinstance(s[k],str) and len(s[k])>=8 for k in ('case_id','pair_id','target_query_id')),'identity')
    require(len(s['factors'])==36,'factor count')
    require(len(s['queries'])==8,'query count')
    fids=set();qids=set();known_sources=set()
    for f in s['factors']:
        require(set(f)=={'id','rel','roles','source','version'},'factor schema')
        require(f['id'] not in fids,'duplicate factor identity');fids.add(f['id'])
        require(f['rel'] in ('R0','R1','R2','RX'),'factor relation')
        require(set(f['roles'])=={'from','to','scope','context'},'role binding schema')
        require(all(isinstance(v,str) and v for v in f['roles'].values()),'role identity')
        require(isinstance(f['source'],str) and f['source'],'source identity')
        require(type(f['version']) is int and f['version'] in (0,1),'source epoch')
        known_sources.add(f['source'])
    for q in s['queries']:
        require(set(q)=={'id','from','to','scope','context'},'query schema')
        require(q['id'] not in qids,'duplicate query identity');qids.add(q['id'])
        require(all(isinstance(v,str) and v for v in q.values()),'query identity')
    require(s['target_query_id'] in qids,'target query not in query set')
    prev=-1
    for e in s['events']:
        require(set(e)=={'step','source','status','epoch'},'event schema')
        require(type(e['step']) is int and e['step']>prev,'non-monotonic / colliding event step');prev=e['step']
        require(e['source'] in known_sources,'out of graph source')
        require(e['status'] in ('VALID','STALE','UNKNOWN'),'unknown source status')
        require(type(e['epoch']) is int and e['epoch'] in (0,1),'invalid epoch')
    require(len(s['events']) in (14,15,16),'event count')

def audit(ds,gold,split,other_world_ids=(),other_case_ids=()):
    require(set(ds)=={'schema','split','worlds'},'dataset schema')
    require(ds['schema']=='elpis.hca.h1c0.dynamic_multiquery.public.v1','schema version')
    require(ds['split']==split and split in ('TRAIN','TUNE'),'invalid split')
    require(len(ds['worlds'])==(2 if split=='TRAIN' else 1),'wrong world count')
    wid=set(other_world_ids);cids=set(other_case_ids)
    stats=Counter();pairs=defaultdict(list);calc={}
    for w in ds['worlds']:
        require(set(w)=={'world_id','cases'},'world schema')
        require(w['world_id'] not in wid,'duplicate or cross-split world');wid.add(w['world_id'])
        require(len(w['cases'])==16,'world case denominator')
        for s in w['cases']:
            verify_case(s)
            cid=s['case_id'];require(cid not in cids,'duplicate or cross-split case');cids.add(cid)
            require(s['pair_id'] not in (w['world_id'],cid),'identity collision')
            pairs[(w['world_id'],s['pair_id'])].append(s)
            expected=oracle.all_prefix_labels(s)
            require(cid in gold and gold[cid]==expected,'missing/mismatched gold')
            calc[cid]=expected
            stats['cases']+=1;stats['events']+=len(s['events'])
            stats['rows']+=(len(s['events'])+1)*len(s['queries'])
            stats['positives']+=sum(sum(row.values()) for row in expected)
    require(set(calc)==set(gold),'extra gold or missing case')
    for (world,pid),group in pairs.items():
        require(len(group)==2,'pair dropped/duplicated')
        a,b=group
        kind=a['kind'];require(kind==b['kind'],'pair kind mismatch')
        require(a['queries']==b['queries'] and a['target_query_id']==b['target_query_id'],'pair query drift')
        fa,fb=calc[a['case_id']][-1][a['target_query_id']],calc[b['case_id']][-1][b['target_query_id']]
        require(sorted([fa,fb])==[0,1],'no causal target contrast')
        require(len(a['events'])==len(b['events']),'event length mismatch')
        if kind=='structural':
            require(a['events']==b['events'],'structural pair event drift')
            require(shape_marginals(a)==shape_marginals(b),'structural role marginals changed')
        elif kind=='event_order':
            require(bag_factors(a)==bag_factors(b),'event order factor drift')
            require(event_bag(a)==event_bag(b),'event order multiset drift')
        elif kind=='source_swap':
            require(bag_factors(a)==bag_factors(b),'source swap factor drift')
            require(class_bag(a)==class_bag(b),'source status histogram drift')
        elif kind=='epoch_rebind':
            require(bag_factors(a)==bag_factors(b),'epoch factor drift')
            require(class_bag(a)==class_bag(b),'epoch/status histogram drift')
        stats['pairs_'+kind]+=1
    for kind in ('source_swap','event_order','epoch_rebind','structural'):
        require(stats['pairs_'+kind]==len(ds['worlds'])*2,'pair type denominator mismatch')
    return {'worlds':len(ds['worlds']),'cases':stats['cases'],'rows':stats['rows'],
            'positive_rows':stats['positives'],'events':stats['events'],
            'pairs':{k:stats['pairs_'+k] for k in ('source_swap','event_order','epoch_rebind','structural')},
            'world_ids':sorted(wid),'case_ids':sorted(cids)}
