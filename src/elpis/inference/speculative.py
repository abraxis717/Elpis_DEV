"""DeepSpec-derived Markov proposals and target-authoritative greedy verification.

Reference: DeepSeek4.1/01_DeepSpec/deepspec/modeling/dspark/markov_head.py,
eval/dspark/draft_ops.py and evaluator.py; MIT, revision
005e03b81cec38b7da6399833d609ee89a2587f2. Independent NumPy implementation.
ACCELERATION_ONLY means no correctness authority for the drafter. This CPU
serial verifier qualifies greedy equivalence, NOT sampling-distribution
equivalence or measured acceleration. No target-layer geometry is copied.
"""
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
from elpis.inference.contracts import Code,ContractError,ProposalOnly,identity,integer,require
from elpis.inference.context import Lifetime
from elpis.inference.target import Tensor
from .transaction import DecodeState,DecodeResult,typed_failure


@dataclass(frozen=True)
class Draft(ProposalOnly):
    committed_base: str
    target_model: str
    drafter: str
    tokens: tuple[int,...]
    confidence: tuple[float,...]

    def __post_init__(self):
        require(type(self.tokens) is tuple and len(self.tokens)==len(self.confidence))
        require(all(np.isfinite(x) and 0<=x<=1 for x in self.confidence),detail='draft confidence')

    @property
    def digest(self): return identity('deepspec.draft',self)


@dataclass(frozen=True)
class SpeculativeOverlay:
    committed_base: str
    states: tuple[DecodeState,...]

    @property
    def lifetime(self): return Lifetime.SPECULATIVE


@dataclass(frozen=True)
class Verification:
    accepted_state: DecodeState
    accepted_count: int
    correction: int | None
    draft: str
    verified_prefix: str


class MarkovDrafter:
    def __init__(self,*,target_model,hidden_projection,token_embedding,markov_projection,confidence_head):
        for t in (hidden_projection,token_embedding,markov_projection,confidence_head): require(type(t) is Tensor)
        d,v=hidden_projection.shape
        require(token_embedding.shape[0]==v and markov_projection.shape==(token_embedding.shape[1],v)
                and confidence_head.shape==(d,1),detail='drafter geometry')
        self.target_model=target_model
        self.hidden_projection=hidden_projection; self.token_embedding=token_embedding
        self.markov_projection=markov_projection; self.confidence_head=confidence_head
        self.digest=identity('deepspec.markov-drafter',dict(target=target_model,
             tensors=tuple(t.digest for t in (hidden_projection,token_embedding,markov_projection,confidence_head)),
             mode='ACCELERATION_ONLY',sampling='GREEDY'))

    def propose(self,state,count,*,threshold=0.0):
        integer(count,0,128)
        require(0<=threshold<=1 and state.neural.model==self.target_model,Code.IDENTITY,'drafter target/threshold')
        require(bool(state.neural.hidden) and bool(state.neural.tokens),detail='drafter needs target prefill')
        hidden=np.asarray(state.neural.hidden,dtype='<f4')
        base=hidden@self.hidden_projection.array()
        logit=float((hidden@self.confidence_head.array())[0])
        confidence=float(1/(1+np.exp(-np.clip(logit,-80,80))))
        previous=state.neural.tokens[-1]; tokens=[]
        for _ in range(count):
            if confidence<threshold: break
            logits=base+self.token_embedding.array()[previous]@self.markov_projection.array()
            previous=int(np.argmax(logits)); tokens.append(previous)
        return Draft(state.digest,self.target_model,self.digest,tuple(tokens),(confidence,)*len(tokens))


def _verify_draft(runtime,state,request,draft,*,prefetch_enabled=False,validate_base=True):
    try:
        require(type(draft) is Draft and draft.committed_base==state.digest and
                draft.target_model==runtime.target.model_identity,Code.STALE,'draft committed base/model')
        require(len(draft.tokens)<=128,Code.LIMIT,'draft block')
        if validate_base:
            runtime._validate(state,request)
        require(bool(state.neural.logits),detail='verification requires target logits')
        states=[state]; target_ns=0
        for token in draft.tokens:
            integer(token,0,runtime.target.config.vocab-1)
            begin=perf_counter_ns()
            next_state,_=runtime._advance_validated(
                states[-1],token,request,prefetch_enabled=prefetch_enabled
            )
            target_ns+=perf_counter_ns()-begin
            states.append(next_state)
        overlay=SpeculativeOverlay(state.digest,tuple(states))
        accepted=0
        for i,token in enumerate(draft.tokens):
            if token!=int(np.argmax(overlay.states[i].neural.logits)): break
            accepted+=1
        committed=overlay.states[accepted]
        correction=int(np.argmax(committed.neural.logits)) if accepted<len(draft.tokens) or not draft.tokens else None
        result=Verification(
            committed,accepted,correction,draft.digest,
            identity('deepspec.accepted-prefix',
                     dict(base=state.digest,tokens=draft.tokens[:accepted],output=committed.digest))
        )
        return result,dict(
            candidate_tokens=len(draft.tokens),accepted_prefix=accepted,
            acceptance_by_position=tuple(i<accepted for i in range(len(draft.tokens))),
            target_verifier_steps=len(draft.tokens),target_ns=target_ns
        )
    except ContractError:
        raise
    except Exception as exc:
        raise typed_failure(exc) from exc


def verify_draft(runtime,state,request,draft,*,prefetch_enabled=False):
    return _verify_draft(
        runtime,state,request,draft,
        prefetch_enabled=prefetch_enabled,validate_base=True
    )


def run_speculative(runtime,state,request,drafter,*,expected_state,block_size=4,prefetch_enabled=False):
    require(request.mode=='GREEDY',detail='speculation applies to generation')
    integer(block_size,1,128)
    before=state; telemetry=[]; rounds=[]; remaining=request.count
    try:
        require(state.digest==expected_state,Code.STALE,'speculative committed base')
        runtime._validate(state,request)
        while remaining:
            begin=perf_counter_ns()
            draft=drafter.propose(state,min(block_size,remaining))
            draft_ns=perf_counter_ns()-begin
            require(len(draft.tokens)<=remaining,Code.LIMIT,'drafter overproduction')
            begin=perf_counter_ns()
            verification,metrics=_verify_draft(
                runtime,state,request,draft,
                prefetch_enabled=prefetch_enabled,validate_base=False
            )
            metrics.update(draft_ns=draft_ns,verification_ns=perf_counter_ns()-begin)
            rounds.append((draft.digest,verification.accepted_count,verification.verified_prefix))
            begin=perf_counter_ns()
            state=verification.accepted_state; remaining-=verification.accepted_count
            if remaining and verification.correction is not None:
                state,correction_metrics=runtime._advance_validated(
                    state,verification.correction,request,prefetch_enabled=prefetch_enabled
                )
                remaining-=1; metrics['correction']=correction_metrics
            metrics['commit_ns']=perf_counter_ns()-begin
            telemetry.append(metrics)
    except ContractError as exc:
        receipt=runtime.receipt(
            request,before,before,
            draft=identity('deepspec.rounds',tuple(rounds)),
            failure=exc.code.value,
            request_identity=runtime._failure_request_identity(request),
        )
        return DecodeResult(before,receipt,tuple(telemetry))
    except Exception as exc:
        raise typed_failure(exc) from exc
    receipt=runtime.receipt(
        request,before,state,
        draft=identity('deepspec.rounds',tuple(rounds)),
        accepted=state.neural.tokens[len(before.neural.tokens):]
    )
    runtime._remember_validated(state.digest)
    return DecodeResult(state,receipt,tuple(telemetry))
