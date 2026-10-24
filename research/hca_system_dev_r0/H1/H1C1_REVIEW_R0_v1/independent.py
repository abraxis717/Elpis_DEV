"""Observer-only, independent exhaustive triplet verifier.
No imports from H1C0 implementation/oracle/checker/engine.
"""
from collections import Counter, defaultdict

class AuditFailure(ValueError):
    pass

def need(cond, why):
    if not cond:
        raise AuditFailure(why)

def independent_labels(scene):
    """Enumerate visible live triples, not iterative frontier nor retained support indices."""
    state={}
    labels=[]
    for prefix in range(len(scene['events'])+1):
        if prefix:
            ev=scene['events'][prefix-1]
            state[ev['source']]=(ev['status'],ev['epoch'])
        live={'R0':[], 'R1':[], 'R2':[]}
        for f in scene['factors']:
            if f['rel'] in live and state.get(f['source'])==('VALID',f['version']):
                live[f['rel']].append(f['roles'])
        answer={}
        for q in scene['queries']:
            truth=False
            for a in live['R0']:
                if (a['from'],a['scope'],a['context'])!=(q['from'],q['scope'],q['context']):continue
                for b in live['R1']:
                    if (b['from'],b['scope'],b['context'])!=(a['to'],q['scope'],q['context']):continue
                    for c in live['R2']:
                        if (c['from'],c['to'],c['scope'],c['context'])==(b['to'],q['to'],q['scope'],q['context']):
                            truth=True;break
                    if truth:break
                if truth:break
            answer[q['id']]=int(truth)
        labels.append(answer)
    return labels

def factor_bag(s):
    return Counter((f['id'],f['rel'],f['source'],f['version'],tuple(sorted(f['roles'].items()))) for f in s['factors'])

def event_bag(s):
    return Counter((e['source'],e['status'],e['epoch']) for e in s['events'])

def status_bag(s):
    return Counter((e['status'],e['epoch']) for e in s['events'])

def role_bag(s):
    return Counter((f['rel'],key,value) for f in s['factors'] for key,value in f['roles'].items())

def inspect_split(ds,gold,pred,case_work,result,which):
    need(set(ds)=={'schema','split','worlds'},'dataset schema')
    need(ds['schema']=='elpis.hca.h1c0.dynamic_multiquery.public.v1','schema mismatch')
    need(ds['split']==which,'split mismatch')
    need(len(ds['worlds'])==(2 if which=='TRAIN' else 1),'world count')
    ids=set();worlds=set();pairs=defaultdict(list)
    stats=Counter(); positives=0; events=0; work={'replay':Counter(),'incremental':Counter()}
    peaks={'replay':0,'incremental':0}
    by_case={}
    for w in ds['worlds']:
        need(set(w)=={'world_id','cases'},'world schema')
        need(w['world_id'] not in worlds,'duplicate world');worlds.add(w['world_id'])
        need(len(w['cases'])==16,'case denominator')
        for s in w['cases']:
            need(set(s)=={'case_id','pair_id','kind','target_query_id','factors','queries','events'},'case schema / label leak')
            cid=s['case_id']
            need(cid not in ids,'duplicate case');ids.add(cid)
            need(s['kind'] in ('source_swap','event_order','epoch_rebind','structural'),'bad kind')
            need(len(s['factors'])==36 and len(s['queries'])==8,'factor/query denominator')
            need(len(s['events']) in (14,15,16),'event denominator')
            fids=set();sources=set()
            for f in s['factors']:
                need(set(f)=={'id','rel','roles','source','version'},'factor schema')
                need(f['id'] not in fids,'duplicate factor');fids.add(f['id'])
                need(f['rel'] in ('R0','R1','R2','RX'),'relation enum')
                need(set(f['roles'])=={'from','to','scope','context'},'role schema')
                need(all(isinstance(x,str) and x for x in f['roles'].values()),'empty role')
                need(type(f['version']) is int and f['version'] in (0,1),'invalid factor version')
                sources.add(f['source'])
            qids=[q['id'] for q in s['queries']]
            need(len(qids)==len(set(qids)),'duplicate query')
            need(s['target_query_id'] in qids,'target identity absent')
            for q in s['queries']:
                need(set(q)=={'id','from','to','scope','context'},'query schema')
            steps=[]
            for e in s['events']:
                need(set(e)=={'step','source','status','epoch'},'event schema')
                need(e['source'] in sources,'unknown event source')
                need(e['status'] in ('VALID','STALE','UNKNOWN'),'event status enum')
                need(type(e['epoch']) is int and e['epoch'] in (0,1),'event epoch enum')
                need(type(e['step']) is int,'step type')
                steps.append(e['step'])
            need(all(x<y for x,y in zip(steps,steps[1:])),'event step order')
            expected=independent_labels(s)
            need(cid in gold and cid in pred,'missing labels or predictions')
            need(expected==gold[cid],'independent oracle mismatch '+cid)
            need(expected==pred[cid],'independent prediction mismatch '+cid)
            pairs[(w['world_id'],s['pair_id'])].append((s,expected[-1][s['target_query_id']]))
            stats['cases']+=1;events+=len(s['events']);stats['rows']+=len(expected)*len(qids)
            positives+=sum(sum(row.values()) for row in expected)
            stats['exact_correct']+=len(expected)*len(qids)
            stats['zero_correct']+=sum(len(row)-sum(row.values()) for row in expected)
            stats['final_rows']+=len(qids)
    need(set(gold)==ids and set(pred)==ids,'gold/pred case universe')
    need(len(case_work)==len(ids),'case work denominator')
    for r in case_work:
        cid=r['case_id']
        need(cid in ids and cid not in by_case,'case work duplicate or extra')
        by_case[cid]=r
        need(r['oracle_replay_incremental_equal'] is True,'case parity statement false')
        for engine,field in [('replay','replay_work'),('incremental','incremental_work')]:
            need(isinstance(r[field],dict) and all(type(v) is int and v>=0 for v in r[field].values()),'invalid work values')
            work[engine].update(r[field])
        peaks['replay']=max(peaks['replay'],r['replay_payload_peak_bytes'])
        peaks['incremental']=max(peaks['incremental'],r['retained_payload_peak_bytes'])
    need(set(by_case)==ids,'case-work universe')
    pc=Counter()
    for pair,items in pairs.items():
        need(len(items)==2,'pair multiplicity')
        (a,ay),(b,by)=items
        need(a['kind']==b['kind'],'pair kind drift')
        kind=a['kind']
        need(a['queries']==b['queries'] and a['target_query_id']==b['target_query_id'],'query pair drift')
        need(sorted([ay,by])==[0,1],'counterfactual lacks opposite gold')
        need(len(a['events'])==len(b['events']),'event pair length')
        if kind=='structural':
            need(a['events']==b['events'] and role_bag(a)==role_bag(b),'structural nuisance drift')
        elif kind=='event_order':
            need(factor_bag(a)==factor_bag(b) and event_bag(a)==event_bag(b),'event order nuisance drift')
        elif kind=='source_swap':
            need(factor_bag(a)==factor_bag(b) and status_bag(a)==status_bag(b),'source nuisance drift')
        else:
            need(factor_bag(a)==factor_bag(b) and status_bag(a)==status_bag(b),'epoch nuisance drift')
        pc['total_'+kind]+=1;pc['both_'+kind]+=1
    for k in ('source_swap','event_order','epoch_rebind','structural'):
        need(pc['total_'+k]==len(ds['worlds'])*2,'paired kind coverage')
    need(set(result['stats'])==set(stats) and result['stats']==dict(stats),'result stats mismatch')
    need(result['pair_metrics']==dict(pc),'result pair metrics mismatch')
    need(result['logical_work']=={k:dict(v) for k,v in work.items()},'logical work census mismatch')
    need(result['serialized_payload_peak_bytes']==peaks,'payload census mismatch')
    return {'cases':stats['cases'],'rows':stats['rows'],'positive_rows':positives,'events':events,
            'pair_both':{k:v for k,v in pc.items() if k.startswith('both_')},
            'aggregate_reported_work':{k:sum(v.values()) for k,v in work.items()},
            'reported_payload_peaks':peaks,'world_ids':sorted(worlds),'case_ids':sorted(ids)}
