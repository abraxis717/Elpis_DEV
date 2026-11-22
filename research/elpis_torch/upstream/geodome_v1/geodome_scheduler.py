"""C5 single-thread synchronous sparse scheduler; trusted pure callbacks only.
No implicit node allocation, no CGC/HCA dependency, no persistent writes.
A physical arc is not authority until explicitly admitted to this instance.
"""
from collections import defaultdict
from dataclasses import dataclass
from geodome_state import MAX_BATCH

MAX_ENROLLED=4096
MAX_ARCS=24576
MAX_ROUNDS=256
MAX_VISITS=MAX_ARCS*MAX_ROUNDS

@dataclass(frozen=True,slots=True)
class Round:
    status: str
    sources: int
    arcs: int
    destinations: int
    changed: int
    touched: int

class SparseScheduler:
    def __init__(self,state,handles,arcs,*,max_active=1024,max_rounds=64,max_visits=65536):
        for val,cap in ((max_active,MAX_BATCH),(max_rounds,MAX_ROUNDS),(max_visits,MAX_VISITS)):
            if type(val) is not int or not 1<=val<=cap: raise ValueError('BUDGET_RANGE')
        if type(handles) not in (tuple,list) or not 1<=len(handles)<=MAX_ENROLLED: raise ValueError('ENROLLMENT_CAP')
        if type(arcs) not in (tuple,list) or len(arcs)>MAX_ARCS: raise ValueError('ARC_CAP')
        hs={}
        for h in handles:
            state.read(h)
            if h.vertex in hs: raise ValueError('DUPLICATE_VERTEX')
            hs[h.vertex]=h
        outgoing=defaultdict(list); seen=set()
        for arc in arcs:
            u,v=state.index.arc_endpoints(arc)
            if arc in seen or u not in hs or v not in hs: raise ValueError('ARC_AUTHORITY')
            seen.add(arc); outgoing[u].append((arc,v))
        self.state=state; self._handles=hs; self._out={u:tuple(sorted(es)) for u,es in outgoing.items()}
        self._epoch=state.epoch; self.max_active=max_active; self.max_rounds=max_rounds; self.max_visits=max_visits
        self.frontier=(); self.rounds=0; self.visits=0; self.status='NEW'; self._busy=False
    def start(self,vertices):
        if self._busy or self.status!='NEW' or self.state.epoch!=self._epoch: raise ValueError('SESSION_STATE')
        if type(vertices) not in (tuple,list) or not 1<=len(vertices)<=self.max_active: raise ValueError('FRONTIER_CAP')
        if any(type(v) is not int or v not in self._handles for v in vertices) or len(set(vertices))!=len(vertices): raise ValueError('FRONTIER_AUTHORITY')
        for v in vertices: self.state.read(self._handles[v])
        self.frontier=tuple(sorted(vertices)); self.status='RUNNING'
    def step(self,transport,update):
        """transport(arc,source,dest,source_row) -> width-sized message;
        update(vertex,old_row,((arc,message),...)) -> width-sized new row.
        Callbacks must be pure, trusted and nonmutating. No whole-run rollback.
        Per-round rejection is atomic for callback/validation errors.
        """
        if self._busy or self.status!='RUNNING' or self.state.epoch!=self._epoch: raise ValueError('SESSION_STATE')
        sources=self.frontier
        if self.rounds>=self.max_rounds:
            self.status='ROUND_BUDGET'; return Round(self.status,len(sources),0,0,0,0)
        plan=tuple((a,u,v) for u in sources for a,v in self._out.get(u,()))
        dest=tuple(sorted({v for _,_,v in plan}))
        # Union, not just frontier, is the number of touched nodes.
        if len(set(sources)|set(dest))>self.max_active:
            self.status='ACTIVE_BUDGET'; return Round(self.status,len(sources),0,0,0,0)
        if self.visits+len(plan)>self.max_visits:
            self.status='ARC_BUDGET'; return Round(self.status,len(sources),0,0,0,0)
        self._busy=True
        try:
            rows={v:self.state.read(self._handles[v]) for v in sorted(set(sources)|set(dest))}
            inbox=defaultdict(list)
            for a,u,v in plan:
                msg=tuple(self.state._row(transport(a,u,v,rows[u])))
                inbox[v].append((a,msg))
            staged=[]; changed=[]
            for v in dest:
                row=tuple(self.state._row(update(v,rows[v],tuple(sorted(inbox[v])))))
                if row!=rows[v]: staged.append((self._handles[v],row)); changed.append(v)
            if staged: self.state.commit(staged)
        finally:
            self._busy=False
        self.rounds+=1; self.visits+=len(plan); self.frontier=tuple(changed)
        self.status='RUNNING' if changed else 'QUIESCENT'
        return Round(self.status,len(sources),len(plan),len(dest),len(changed),len(rows))
