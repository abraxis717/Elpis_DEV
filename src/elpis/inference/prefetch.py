"""Physical predictions only. A plan cannot select learned rows or experts."""
from dataclasses import dataclass
from time import perf_counter_ns
from .contracts import Code, ContractError, ProposalOnly, digest_value, identity, integer, require


@dataclass(frozen=True)
class RangeHint(ProposalOnly):
    asset: str
    offset: int
    length: int
    priority: int
    expires_at: int
    source: str

    def __post_init__(self):
        digest_value(self.asset); digest_value(self.source)
        integer(self.offset); integer(self.length,1); integer(self.priority); integer(self.expires_at)

    @property
    def key(self): return (self.asset,self.offset,self.length)


@dataclass(frozen=True)
class PrefetchPlan(ProposalOnly):
    committed_state: str
    context_snapshot: str
    step: int
    hints: tuple[RangeHint,...]
    proposals: tuple[str,...]

    @property
    def digest(self): return identity('prefetch-plan',self)


def plan_prefetch(*,committed_state,context_snapshot,step,proposals,catalog,max_hints=16,max_bytes=65536):
    digest_value(committed_state); digest_value(context_snapshot); integer(step)
    integer(max_hints,1); integer(max_bytes,1)
    hints={}
    for proposal in proposals:
        require(proposal.context_snapshot==context_snapshot,Code.STALE,'prefetch context')
        for candidate in proposal.banks+proposal.expert_families:
            for hint in catalog.get(candidate,()):
                require(type(hint) is RangeHint)
                if hint.expires_at>=step:
                    previous=hints.get(hint.key)
                    if previous is None or hint.priority>previous.priority: hints[hint.key]=hint
    selected=[]; total=0
    for h in sorted(hints.values(),key=lambda h:(-h.priority,h.key)):
        if len(selected)<max_hints and total+h.length<=max_bytes:
            selected.append(h); total+=h.length
    return PrefetchPlan(committed_state,context_snapshot,step,tuple(selected),tuple(p.digest for p in proposals))


def execute_prefetch(provider,plan,*,state,context_snapshot,step):
    require((plan.committed_state,plan.context_snapshot)==(state,context_snapshot),Code.STALE,'prefetch state')
    result=[]
    for h in plan.hints:
        if h.expires_at<step: continue
        before=provider.stats(); start=perf_counter_ns(); failure=None
        try:
            with provider.acquire(h.asset,h.offset,h.length): pass
        except ContractError as exc:
            # A prediction failure is telemetry only. Demand must still acquire
            # and fail independently; it can never substitute another object.
            failure=exc.code.value
        after=provider.stats()
        result.append(dict(key=h.key,failure=failure,pread_bytes=after['pread_bytes']-before['pread_bytes'],
                           elapsed_ns=perf_counter_ns()-start))
    return tuple(result)


def compare_prefetch(executions,actual_ranges):
    useful=wasted=hits=0
    for item in executions:
        matched=item['failure'] is None and any(item['key'][0]==a and
                   item['key'][1]<o+n and o<item['key'][1]+item['key'][2] for a,o,n in actual_ranges)
        if matched: hits+=1; useful+=item['pread_bytes']
        else: wasted+=item['pread_bytes']
    return dict(predictions=len(executions),useful_hits=hits,wrong_predictions=len(executions)-hits,
                useful_pread_bytes=useful,wasted_pread_bytes=wasted,
                latency_hidden_ns=None,latency_hidden_disposition='NOT_MEASURED_SERIAL_PREFETCH')
