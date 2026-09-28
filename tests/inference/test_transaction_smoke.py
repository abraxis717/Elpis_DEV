from dataclasses import replace
from elpis.inference.transaction import InferenceEngine,InferenceRequest
from elpis.inference.context import initial_snapshot
from elpis.inference.drivers.dsv4.fixtures import make_fixture


def test_atomic_runtime_replay_and_prefetch_equivalence(provider,tmp_path):
    f,*_=provider; target,_,_=make_fixture(f,tmp_path/'target')
    runtime=InferenceEngine(target); state=runtime.initial(initial_snapshot())
    request=InferenceRequest('prefill',state.context.digest,'PREFILL',(1,2,3,4))
    off=runtime.execute(state,request,expected_state=state.digest)
    assert off.receipt.terminal=='COMMITTED'
    on=runtime.execute(state,request,expected_state=state.digest,prefetch_enabled=True)
    assert off.state==on.state and off.receipt==on.receipt
    assert runtime.replay(state,request,off.receipt).state==off.state
    generation=InferenceRequest('generate',state.context.digest,'GREEDY',count=5)
    out=runtime.execute(off.state,generation,expected_state=off.state.digest)
    assert len(out.state.neural.tokens)==9
    assert len(out.receipt.accepted_prefix)==5


def test_transaction_error_discards_all_partial_effects(provider,tmp_path):
    f,*_=provider; target,_,_=make_fixture(f,tmp_path/'target')
    runtime=InferenceEngine(target); state=runtime.initial(initial_snapshot())
    request=InferenceRequest('invalid-suffix',state.context.digest,'PREFILL',(1,2,999))
    result=runtime.execute(state,request,expected_state=state.digest)
    assert result.state==state and result.receipt.terminal=='FAILED'
    assert result.receipt.target_steps==() and result.receipt.failure=='INVALID'
    assert f.stats()['pinned']==0 and target.experts.staged_bytes==0


def test_runtime_rejects_tampered_committed_receipt_lineage(provider,tmp_path):
    f,*_=provider; target,_,_=make_fixture(f,tmp_path/'target')
    runtime=InferenceEngine(target); initial=runtime.initial(initial_snapshot())
    prefill=InferenceRequest('prefill-lineage',initial.context.digest,'PREFILL',(1,2,3,4))
    committed=runtime.execute(initial,prefill,expected_state=initial.digest)
    assert committed.receipt.terminal=='COMMITTED'
    assert committed.state.receipts
    tampered_first=replace(committed.state.receipts[0],output_state='0'*64)
    tampered=replace(
        committed.state,
        receipts=(tampered_first,)+committed.state.receipts[1:],
    )
    request=InferenceRequest('continue-lineage',tampered.context.digest,'GREEDY',count=1)
    result=runtime.execute(tampered,request,expected_state=tampered.digest)
    assert result.state==tampered
    assert result.receipt.terminal=='FAILED'
    assert result.receipt.failure=='STALE'
