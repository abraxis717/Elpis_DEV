from dataclasses import replace
import numpy as np
import pytest
from elpis.inference.transaction import InferenceEngine,InferenceRequest
from elpis.inference.speculative import Draft,MarkovDrafter,verify_draft,run_speculative
from elpis.inference.context import initial_snapshot
from elpis.inference.drivers.dsv4.fixtures import make_fixture
from elpis.inference.target import Tensor
from elpis.inference.contracts import ContractError


def ready(provider,tmp_path):
    f,*_=provider; target,_,_=make_fixture(f,tmp_path/'target')
    rt=InferenceEngine(target); state=rt.initial(initial_snapshot())
    pre=InferenceRequest('prefill',state.context.digest,'PREFILL',(1,2,3))
    state=rt.execute(state,pre,expected_state=state.digest).state
    req=InferenceRequest('greedy',state.context.digest,'GREEDY',count=5)
    return rt,state,req


@pytest.mark.parametrize('accepted',[0,1,2,4,5])
def test_rejected_suffix_all_state_planes_rollback(provider,tmp_path,accepted):
    rt,state,req=ready(provider,tmp_path)
    plain=rt.execute(state,req,expected_state=state.digest)
    tokens=list(plain.state.neural.tokens[len(state.neural.tokens):])
    if accepted<5: tokens[accepted]=(tokens[accepted]+1)%16
    draft=Draft(state.digest,rt.target.model_identity,'1'*64,tuple(tokens),(.99,)*5)
    checked,metrics=verify_draft(rt,state,req,draft,prefetch_enabled=True)
    prefix=rt.execute(state,replace(req,count=accepted),expected_state=state.digest)
    assert checked.accepted_count==accepted
    # Dataclass equality includes token and n-gram history, local/global KV,
    # pending compression, index, context, structural state, prefetch and receipts.
    assert checked.accepted_state==prefix.state
    assert len(checked.accepted_state.receipts)==len(state.receipts)+accepted
    if accepted==0: assert checked.accepted_state==state
    # Planted rejected-suffix merge is detected by the same state equality gate.
    if accepted<5: assert checked.accepted_state!=plain.state


def test_markov_runtime_exact_greedy_and_stale_draft(provider,tmp_path):
    rt,state,req=ready(provider,tmp_path)
    rng=np.random.default_rng(61)
    def t(shape): return Tensor(shape,(rng.normal(size=shape)*.1).astype('<f4').tobytes())
    drafter=MarkovDrafter(target_model=rt.target.model_identity,hidden_projection=t((4,16)),
                          token_embedding=t((16,2)),markov_projection=t((2,16)),confidence_head=t((4,1)))
    spec=run_speculative(rt,state,req,drafter,expected_state=state.digest,block_size=3)
    plain=rt.execute(state,req,expected_state=state.digest)
    assert spec.state==plain.state and spec.receipt.target_steps==plain.receipt.target_steps
    assert spec.receipt.draft and spec.receipt.terminal=='COMMITTED'
    draft=drafter.propose(state,2)
    with pytest.raises(ContractError,match='STALE'):
        verify_draft(rt,state,req,replace(draft,committed_base='0'*64))
    assert not draft.execution_authority
