"""Two independently organized exact engines. No oracle or generator imports."""
from collections import defaultdict,Counter
import json

R=('R0','R1','R2')
def encoded_len(o):return len(json.dumps(o,sort_keys=True,separators=(',',':')).encode())

def replay(scene,measure_memory=True):
    """Naive exact full source replay + factor rescan at every event prefix and every query."""
    work=Counter();out=[];peak=0
    for n in range(len(scene['events'])+1):
        state={}
        for e in scene['events'][:n]:
            state[e['source']]=(e['status'],e['epoch']);work['replayed_events']+=1
        active=[]
        for f in scene['factors']:
            work['factor_scan']+=1
            if state.get(f['source'])==('VALID',f['version']):active.append(f)
        row={}
        for q in scene['queries']:
            frontier={q['from']}
            for step in R:
                nxt=set()
                for f in active:
                    work['factor_query_visit']+=1
                    role=f['roles']
                    if f['rel']==step and role['scope']==q['scope'] and role['context']==q['context'] and role['from'] in frontier:
                        nxt.add(role['to'])
                frontier=nxt
            row[q['id']]=int(q['to'] in frontier)
            work['query_snapshot']+=1
        if measure_memory:
            peak=max(peak,encoded_len({'status':state,'active':[f['id'] for f in active]}))
        out.append(row)
    return out,work,peak

class Incremental:
    """Eager source-dependent 3-hop path-support indexing, including invalidation counts.
    Setup enumerates all possible materialized witnesses; updates touch only paths
    depending on factors whose active status changed. No generator internals.
    """
    def __init__(self,scene):
        self.factors={f['id']:f for f in scene['factors']}
        if len(self.factors)!=len(scene['factors']):raise ValueError('factor id collision')
        self.queries={q['id']:q for q in scene['queries']}
        self.source_map=defaultdict(list)
        for f in scene['factors']:self.source_map[f['source']].append(f['id'])
        self.live=set();self.status={};self.work=Counter()
        by_relation=defaultdict(list)
        for f in scene['factors']:by_relation[f['rel']].append(f)
        self.paths=[];self.dependents=defaultdict(list)
        self.missing=[];self.supports={k:0 for k in self.queries}
        for q in scene['queries']:
            for a in by_relation['R0']:
                self.work['setup_candidate_a']+=1
                ar=a['roles']
                if (ar['from'],ar['scope'],ar['context'])!=(q['from'],q['scope'],q['context']):continue
                for b in by_relation['R1']:
                    self.work['setup_candidate_b']+=1
                    br=b['roles']
                    if (br['from'],br['scope'],br['context'])!=(ar['to'],q['scope'],q['context']):continue
                    for c in by_relation['R2']:
                        self.work['setup_candidate_c']+=1
                        cr=c['roles']
                        if (cr['from'],cr['to'],cr['scope'],cr['context'])!=(br['to'],q['to'],q['scope'],q['context']):continue
                        path=(q['id'],(a['id'],b['id'],c['id']))
                        pid=len(self.paths);self.paths.append(path);self.missing.append(3)
                        for fid in path[1]:self.dependents[fid].append(pid)
                        self.work['setup_path']+=1
        self.work['setup_factor']+=len(scene['factors'])
        self.work['setup_query']+=len(scene['queries'])

    def apply(self,event):
        s=event['source'];self.status[s]=(event['status'],event['epoch']);self.work['source_update']+=1
        for fid in self.source_map.get(s,[]):
            self.work['affected_factor_visit']+=1
            f=self.factors[fid]
            now=self.status[s]==('VALID',f['version'])
            old=fid in self.live
            if now==old:continue
            if now:self.live.add(fid)
            else:self.live.remove(fid)
            self.work['factor_live_flip']+=1
            for pid in self.dependents.get(fid,[]):
                self.work['path_dependency_update']+=1
                before=self.missing[pid]
                after=before-1 if now else before+1
                if not 0<=after<=3:raise AssertionError('invalid missing factor count')
                self.missing[pid]=after
                qid=self.paths[pid][0]
                if before==0:self.supports[qid]-=1
                if after==0:self.supports[qid]+=1
                if self.supports[qid]<0:raise AssertionError('invalid query support count')
    def snapshot(self):
        self.work['query_snapshot']+=len(self.queries)
        return {k:int(v>0) for k,v in self.supports.items()}
    def footprint(self):
        return encoded_len({'factors':self.factors,'source_map':dict(self.source_map),
                            'queries':self.queries,'paths':self.paths,'missing':self.missing,
                            'dependents':dict(self.dependents),'live':sorted(self.live),
                            'status':self.status,'supports':self.supports})

def incremental(scene,measure_memory=True):
    eng=Incremental(scene)
    out=[eng.snapshot()];peak=eng.footprint() if measure_memory else 0
    for event in scene['events']:
        eng.apply(event);out.append(eng.snapshot())
        if measure_memory:peak=max(peak,eng.footprint())
    return out,eng.work,peak
