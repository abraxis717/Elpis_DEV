"""Surface 4: speculative exactness.

The speculative path must never commit a token the target path would reject.
Exercise zero accepted, partial, complete, early rejection, repeated rejection,
EOS boundaries, short/long suffixes, and adversarial draft proposals. Every
accepted state must equal the plain greedy execution of the same prefix.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from elpis.inference.transaction import InferenceRequest
from elpis.inference.speculative import Draft, run_speculative, verify_draft
from elpis.inference.contracts import ContractError
from elpis.inference.context import initial_snapshot

from ._helpers import build_runtime, make_drafter, prefill


def _plain_tokens(rt, state, count):
    return list(rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=count),
                           expected_state=state.digest).state.neural.tokens[len(state.neural.tokens):])


def _draft(rt, state, tokens):
    return Draft(state.digest, rt.target.model_identity, "1" * 64, tuple(tokens), (0.99,) * len(tokens))


def test_zero_accepted_returns_base(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 4)
    # Force zero acceptance: draft a token the target will not emit at position 0.
    wrong = (tokens[0] + 1) % 16
    draft = _draft(rt, state, [wrong] + tokens[1:])
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=4), draft)
    assert checked.accepted_count == 0
    assert checked.accepted_state == state
    assert checked.correction == tokens[0]


def test_partial_accepted_matches_plain_prefix(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 5)
    for accepted in range(0, 5):
        draft_tokens = list(tokens)
        if accepted < 5:
            draft_tokens[accepted] = (draft_tokens[accepted] + 1) % 16
        draft = _draft(rt, state, draft_tokens)
        checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=5), draft)
        assert checked.accepted_count == accepted
        prefix = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=accepted),
                            expected_state=state.digest)
        assert checked.accepted_state == prefix.state
        if accepted == 0:
            assert checked.accepted_state == state


def test_complete_accepted_matches_plain(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 4)
    draft = _draft(rt, state, tokens)
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=4), draft)
    assert checked.accepted_count == 4
    assert checked.correction is None
    plain = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=4),
                       expected_state=state.digest)
    assert checked.accepted_state == plain.state


def test_early_rejection_stops_at_first_mismatch(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 6)
    # Mismatch at position 2 only.
    draft_tokens = list(tokens)
    draft_tokens[2] = (draft_tokens[2] + 1) % 16
    draft = _draft(rt, state, draft_tokens)
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=6), draft)
    assert checked.accepted_count == 2
    prefix = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=2),
                        expected_state=state.digest)
    assert checked.accepted_state == prefix.state


def test_repeated_rejection_every_round(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    drafter = make_drafter(rt, state)
    # Adversarial drafter: always propose a token the target rejects.
    class BadDrafter:
        def propose(self, s, count, **_):
            plain = _plain_tokens(build_runtime(target[0]), s, count)
            return Draft(s.digest, rt.target.model_identity, "1" * 64,
                         tuple((t + 1) % 16 for t in plain), (0.99,) * count)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=4)
    out = run_speculative(rt, state, req, BadDrafter(), expected_state=state.digest, block_size=2)
    plain = rt.execute(state, req, expected_state=state.digest)
    assert out.state == plain.state
    assert out.receipt.terminal == "COMMITTED"


def test_eos_boundary_never_commits_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    # EOS token in this fixture is 15 (vocab 16, eos=15 per QwenPLE; DSV41 uses
    # dead=-1). Draft a token that is NOT the target's choice at the EOS boundary.
    tokens = _plain_tokens(rt, state, 3)
    draft_tokens = list(tokens)
    draft_tokens[-1] = (draft_tokens[-1] + 1) % 16
    draft = _draft(rt, state, draft_tokens)
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=3), draft)
    assert checked.accepted_count == 2
    prefix = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=2),
                        expected_state=state.digest)
    assert checked.accepted_state == prefix.state


def test_short_suffix(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 1)
    draft = _draft(rt, state, tokens)
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=1), draft)
    assert checked.accepted_count == 1
    assert checked.accepted_state == rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                                                expected_state=state.digest).state


def test_long_suffix(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    tokens = _plain_tokens(rt, state, 12)
    draft = _draft(rt, state, tokens)
    checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=12), draft)
    assert checked.accepted_count == 12
    assert checked.accepted_state == rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=12),
                                                expected_state=state.digest).state


def test_adversarial_draft_out_of_vocab_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    draft = _draft(rt, state, [999])
    with pytest.raises(ContractError) as exc:
        verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=1), draft)
    assert exc.value.code.value == "INVALID"


def test_adversarial_draft_too_long_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    draft = _draft(rt, state, [1] * 129)
    with pytest.raises(ContractError) as exc:
        verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=129), draft)
    assert exc.value.code.value == "LIMIT"


def test_stale_draft_base_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    draft = _draft(rt, state, [1, 2])
    stale = replace(draft, committed_base="0" * 64)
    with pytest.raises(ContractError) as exc:
        verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=2), stale)
    assert exc.value.code.value == "STALE"


def test_wrong_target_model_draft_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    draft = _draft(rt, state, [1, 2])
    wrong = replace(draft, target_model="0" * 64)
    with pytest.raises(ContractError) as exc:
        verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=2), wrong)
    assert exc.value.code.value == "STALE"


def test_speculative_equals_plain_full_run(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    drafter = make_drafter(rt, state)
    req = InferenceRequest("g", state.context.digest, "GREEDY", count=8)
    spec = run_speculative(rt, state, req, drafter, expected_state=state.digest, block_size=3)
    plain = rt.execute(state, req, expected_state=state.digest)
    assert spec.state == plain.state
    assert spec.receipt.target_steps == plain.receipt.target_steps
    assert spec.receipt.terminal == "COMMITTED"


def test_speculative_prefill_mode_rejected(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    drafter = make_drafter(rt, state)
    req = InferenceRequest("p", state.context.digest, "PREFILL", (1, 2))
    with pytest.raises(ContractError):
        run_speculative(rt, state, req, drafter, expected_state=state.digest)


def test_speculative_never_overproduces(target):
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)

    class OverDrafter:
        def propose(self, s, count, **_):
            return Draft(s.digest, rt.target.model_identity, "1" * 64, (1,) * (count + 1), (0.9,) * (count + 1))

    req = InferenceRequest("g", state.context.digest, "GREEDY", count=2)
    # Overproduction is a transaction failure: the original state is returned
    # with a FAILED receipt (the check is inside the atomic try).
    result = run_speculative(rt, state, req, OverDrafter(), expected_state=state.digest, block_size=2)
    assert result.state == state
    assert result.receipt.terminal == "FAILED"
    assert result.receipt.failure == "LIMIT"
