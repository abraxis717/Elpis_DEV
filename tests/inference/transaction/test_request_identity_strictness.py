from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from elpis.inference.context import initial_snapshot
from elpis.inference.contracts import ContractError
from elpis.inference.transaction import InferenceRequest, InferenceEngine
from elpis.inference.speculative import Draft, run_speculative
from elpis.inference.transaction import DecodeState


class OracleDrafter:
    def __init__(self, runtime):
        self.runtime=runtime

    def propose(self,state,count,*,threshold=0.0):
        token=int(np.argmax(state.neural.logits))
        return Draft(
            state.digest,
            self.runtime.target.model_identity,
            "7"*64,
            (token,),
            (0.9,),
        )


def prefilled(target):
    runtime=InferenceEngine(target)
    initial=runtime.initial(initial_snapshot())
    result=runtime.execute(
        initial,
        InferenceRequest("prefill",initial.context.digest,"PREFILL",(1,2,3)),
        expected_state=initial.digest,
    )
    assert result.receipt.terminal=="COMMITTED"
    return runtime,result.state


@pytest.mark.parametrize("request_id",[float("nan"),frozenset({1}),7])
def test_request_id_is_exact_nonempty_string(request_id):
    with pytest.raises(ContractError,match="INVALID:request id"):
        InferenceRequest(
            request_id,
            initial_snapshot().digest,
            "GREEDY",
            count=1,
        )


@pytest.mark.parametrize("request_id",[float("nan"),frozenset({1}),frozenset({2})])
def test_speculative_noncanonical_request_never_commits_or_caches(target,request_id):
    runtime,state=prefilled(target[0])
    request=InferenceRequest("greedy",state.context.digest,"GREEDY",count=1)
    object.__setattr__(request,"request_id",request_id)
    before_cache=set(runtime._validated_states)
    before_digest=state.digest

    result=run_speculative(
        runtime,
        state,
        request,
        OracleDrafter(runtime),
        expected_state=state.digest,
        block_size=1,
    )

    assert result.state is state
    assert result.receipt.terminal=="FAILED"
    assert result.receipt.failure=="INVALID"
    assert result.receipt.request==runtime._failure_request_identity(request)
    assert state.digest==before_digest
    assert runtime._validated_states==before_cache


def test_runtime_rejects_decodestate_subclass_before_cache_trust(target):
    runtime,state=prefilled(target[0])
    trusted_digest=state.digest

    class Liar(DecodeState):
        @property
        def digest(self):
            return trusted_digest

    forged_neural=replace(
        state.neural,
        hidden=tuple(float(x)+1.0 for x in state.neural.hidden),
    )
    forged=Liar(
        forged_neural,
        state.context,
        state.structural,
        state.prefetch,
        state.receipts,
        state.step_latents,
        state.step_proposals,
    )
    before_cache=set(runtime._validated_states)
    result=runtime.execute(
        forged,
        InferenceRequest("greedy",state.context.digest,"GREEDY",count=1),
        expected_state=trusted_digest,
    )

    assert result.state is forged
    assert result.receipt.terminal=="FAILED"
    assert result.receipt.failure=="INVALID"
    assert runtime._validated_states==before_cache
