"""Surface 1: complete receipt-lineage validation.

Attacks missing, reordered, duplicated, substituted, wrong-linked, wrong-identity,
truncated, stale-predecessor and tampered-terminal receipt chains. Every attack
must fail closed: typed ContractError, original state returned, FAILED receipt.
"""
from __future__ import annotations

from dataclasses import replace


from elpis.inference.transaction import InferenceRequest, InferenceEngine
from elpis.inference.context import initial_snapshot


def commit_prefill(rt, state, tokens=(1, 2, 3, 4)):
    req = InferenceRequest("prefill", state.context.digest, "PREFILL", tokens)
    out = rt.execute(state, req, expected_state=state.digest)
    assert out.receipt.terminal == "COMMITTED"
    return out.state


def greedy(rt, state, count=1, tag="g"):
    req = InferenceRequest(tag, state.context.digest, "GREEDY", count=count)
    return rt.execute(state, req, expected_state=state.digest)


def test_missing_receipt_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=state.receipts[:-1])
    result = greedy(rt, bad)
    assert result.state == bad
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"
    assert result.receipt.target_steps == ()


def test_reordered_receipts_fail_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    assert len(state.receipts) >= 2
    r0, r1 = state.receipts[0], state.receipts[1]
    bad = replace(state, receipts=(r1, r0) + state.receipts[2:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure in ("IDENTITY", "STALE")


def test_duplicated_receipt_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=(state.receipts[0],) * len(state.receipts))
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure in ("IDENTITY", "STALE")


def test_substituted_receipt_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    # A genuinely foreign chain: different tokens -> different states/receipts.
    other = commit_prefill(InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot()), tokens=(5, 6, 7, 8))
    assert other.receipts[0] != state.receipts[0]
    bad = replace(state, receipts=(other.receipts[0],) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure in ("IDENTITY", "STALE")


def test_tampered_terminal_output_state_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    last = state.receipts[-1]
    bad = replace(state, receipts=state.receipts[:-1] + (replace(last, output_state="0" * 64),))
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_tampered_intermediate_output_state_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    first = state.receipts[0]
    bad = replace(state, receipts=(replace(first, output_state="f" * 64),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_wrong_input_output_link_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    # Swap input_state of receipt i with the output_state of receipt i-1 broken:
    # set receipt[1].input_state to receipt[0].output_state of a DIFFERENT chain.
    other = commit_prefill(InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot()), tokens=(5, 6, 7, 8))
    assert other.receipts[0].output_state != state.receipts[0].output_state
    bad = replace(
        state,
        receipts=state.receipts[:1]
        + (replace(state.receipts[1], input_state=other.receipts[0].output_state),)
        + state.receipts[2:],
    )
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_wrong_model_identity_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=(replace(state.receipts[0], model="0" * 64),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"


def test_wrong_tokenizer_identity_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=(replace(state.receipts[0], tokenizer="other-tokenizer"),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"


def test_wrong_context_snapshot_in_receipt_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=(replace(state.receipts[0], context_snapshot="0" * 64),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_wrong_token_in_receipt_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=(replace(state.receipts[0], token=(state.receipts[0].token + 1) % 16),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"


def test_stale_predecessor_input_state_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    # receipt[0].input_state must equal target.initial(context).digest; break it.
    bad = replace(state, receipts=(replace(state.receipts[0], input_state="1" * 64),) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_non_stepreceipt_type_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    bad = replace(state, receipts=("not-a-receipt",) + state.receipts[1:])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "IDENTITY"


def test_truncated_chain_with_extra_tokens_fails_closed(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    # Drop two receipts but keep tokens: cardinality mismatch must fail closed.
    bad = replace(state, receipts=state.receipts[:1])
    result = greedy(rt, bad)
    assert result.state == bad and result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "STALE"


def test_full_lineage_revalidates_after_growth(target):
    rt, state = InferenceEngine(target[0]), InferenceEngine(target[0]).initial(initial_snapshot())
    state = commit_prefill(rt, state)
    grown_result = greedy(rt, state, count=3)
    assert grown_result.receipt.terminal == "COMMITTED"
    grown = grown_result.state
    # The grown chain must still validate end-to-end on the next transaction.
    again = greedy(rt, grown, count=1)
    assert again.receipt.terminal == "COMMITTED"
    assert len(again.state.receipts) == len(grown.receipts) + 1
    assert len(again.state.neural.tokens) == len(grown.neural.tokens) + 1
