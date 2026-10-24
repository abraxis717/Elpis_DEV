"""Two genuinely distinct exact engines for SOURCE temporal prefixes (stdlib only).
NO oracle imports. Exact closure for the H1B1 length-two target only; not general HCA B0/B0I.
"""
from collections import defaultdict, Counter
import json

RULES=(('A','C'),('A','D'),('B','C'),('B','D'))


def canonical(o):
    return json.dumps(o,sort_keys=True,separators=(',',':'))


def _valid(status, f):
    s=status.get(f['source'])
    return s is not None and s[0]=='VALID' and s[1]==f['source_version']


def fresh_score(scene, rule, prefix, footprint=True):
    """Independent full event replay + full eligible-factor rescan on EVERY query."""
    work=Counter()
    state={}
    for event in prefix:
        work['event_replay']+=1
        state[event['source']]=(event['status'],event['epoch'])
    active=[]
    for f in scene['factors']:
        work['factor_scan']+=1
        if _valid(state,f) and f['roles']['scope']==scene['query']['scope']:
            active.append(f)
    # Broad two-pass exact join with no index reused across temporal prefixes.
    r=set()
    for f in active:
        work['active_visit']+=1
        if f['rel']==rule[1] and f['roles']['to']==scene['query']['to']:
            r.add(f['roles']['from'])
    answer=0
    for f in active:
        work['active_visit']+=1
        if (f['rel']==rule[0] and f['roles']['from']==scene['query']['from']
                and f['roles']['to'] in r):
            answer=1
    return answer,work,(len(canonical(state).encode())+len(canonical(active).encode())) if footprint else 0


class SourceIndexedExact:
    """Incremental, source-indexed exact materialization of active n-ary factors."""
    def __init__(self,scene):
        self.q=scene['query']
        self.factors={}
        self.by_source=defaultdict(list)
        self.index=defaultdict(set)
        self.live=set()
        self.source_state={}
        self.work=Counter()
        for f in scene['factors']:
            fid=f['fid']
            if fid in self.factors: raise ValueError('duplicate factor identifier')
            self.factors[fid]=f
            self.by_source[f['source']].append(fid)
            self.work['init_factor_scan']+=1
        self.last=(-1,-1)

    def _key(self,f):
        r=f['roles']
        return (f['rel'],r['from'],r['scope'])

    def apply(self,event):
        tick=(event['tick'],event['ordinal'])
        if tick<=self.last: raise ValueError('nonmonotone source event')
        self.last=tick
        if event['kind']!='SOURCE' or event['status'] not in ('VALID','STALE','UNKNOWN'):
            raise ValueError('invalid source event')
        self.work['event_update']+=1
        s=event['source']
        self.source_state[s]=(event['status'],event['epoch'])
        for fid in self.by_source.get(s,[]):
            self.work['affected_source_factor_scan']+=1
            f=self.factors[fid]
            active=_valid(self.source_state,f) and f['roles']['scope']==self.q['scope']
            if active and fid not in self.live:
                self.live.add(fid);self.index[self._key(f)].add(fid)
                self.work['index_insert']+=1
            elif not active and fid in self.live:
                self.live.remove(fid);self.index[self._key(f)].remove(fid)
                self.work['index_remove']+=1

    def score(self,rule):
        answer=0
        left=sorted(self.index.get((rule[0],self.q['from'],self.q['scope']),()))
        self.work['index_lookup']+=1
        for lid in left:
            self.work['left_candidate_visit']+=1
            middle=self.factors[lid]['roles']['to']
            self.work['index_lookup']+=1
            for rid in sorted(self.index.get((rule[1],middle,self.q['scope']),())):
                self.work['right_candidate_visit']+=1
                if self.factors[rid]['roles']['to']==self.q['to']:
                    answer=1
        return answer

    def footprint_lower_bound_bytes(self):
        """Canonical serialized retained payload size, not process RSS or allocated heap."""
        retained={'source_state':self.source_state,'live':sorted(self.live),
                  'by_source':dict(self.by_source),'index':{
                      '|'.join(k):sorted(v) for k,v in self.index.items()},
                  'factors':self.factors}
        return len(canonical(retained).encode())


def infer_support_rule(support):
    """Same four public candidate laws; use only TWO VISIBLE support labels."""
    agrees=[]
    for candidate in RULES:
        if all(fresh_score(case,candidate,case['events'])[0]==case['label'] for case in support):
            agrees.append(candidate)
    if len(agrees)!=1: raise ValueError('non-identifiable support; '+str(agrees))
    return agrees[0]


def stream_fresh(scene,rule,measure_memory=True):
    """Includes initial empty-state query; cost captures full event replay on each prefix."""
    work=Counter();labels=[];peak=0
    for i in range(len(scene['events'])+1):
        y,w,b=fresh_score(scene,rule,scene['events'][:i],footprint=measure_memory)
        labels.append(y);work.update(w);peak=max(peak,b)
    return labels,work,peak


def stream_indexed(scene,rule,measure_memory=True):
    eng=SourceIndexedExact(scene)
    labels=[eng.score(rule)]
    peak=eng.footprint_lower_bound_bytes() if measure_memory else 0
    for e in scene['events']:
        eng.apply(e)
        labels.append(eng.score(rule))
        if measure_memory: peak=max(peak,eng.footprint_lower_bound_bytes())
    return labels,eng.work,peak


def assert_parity(scene, rule, expected_final=None):
    fs,fw,fm=stream_fresh(scene,rule)
    ii,iw,im=stream_indexed(scene,rule)
    if fs!=ii: raise AssertionError('incremental parity diverged at '+str(next(i for i in range(len(fs)) if fs[i]!=ii[i])))
    if expected_final is not None and fs[-1]!=expected_final: raise AssertionError('final gold divergence')
    return fs,fw,iw,fm,im
