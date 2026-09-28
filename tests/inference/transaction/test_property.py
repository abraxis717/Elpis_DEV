"""Property-based randomized testing with recorded seeds.

Uses deterministic seeded RNGs so failures are reproducible. Exercises:
- random draft proposals (speculative exactness: never commit a rejected token);
- random receipt-chain tampering (lineage must fail closed);
- random token sequences (determinism + cardinality).

Seeds are recorded in the test names / module constant for reproducibility.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from elpis.inference.transaction import InferenceRequest
from elpis.inference.speculative import Draft, verify_draft
from elpis.inference.context import initial_snapshot

from ._helpers import build_runtime, prefill

RECORDED_SEEDS = (11, 29, 47, 103, 211)


def _plain_tokens(rt, state, count):
    return list(rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY", count=count),
                           expected_state=state.digest).state.neural.tokens[len(state.neural.tokens):])


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_random_drafts_never_commit_rejected_token(target, seed):
    """For random draft proposals, the accepted prefix must always equal the
    plain greedy prefix of the same length (target-authoritative exactness)."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state)
    rng = np.random.default_rng(seed)
    vocab = rt.target.config.vocab
    for trial in range(8):
        count = int(rng.integers(1, 9))
        plain = _plain_tokens(rt, state, count)
        # Random draft: each token is either the plain token or a random one.
        draft_tokens = []
        for t in plain:
            if rng.random() < 0.5:
                draft_tokens.append(int(rng.integers(0, vocab)))
            else:
                draft_tokens.append(t)
        draft = Draft(state.digest, rt.target.model_identity, "1" * 64,
                      tuple(draft_tokens), (0.9,) * len(draft_tokens))
        checked, _ = verify_draft(rt, state, InferenceRequest("g", state.context.digest, "GREEDY", count=count),
                                  draft)
        # The accepted prefix must equal the plain greedy prefix of that length.
        prefix = rt.execute(state, InferenceRequest("g", state.context.digest, "GREEDY",
                                                    count=checked.accepted_count),
                            expected_state=state.digest)
        assert checked.accepted_state == prefix.state
        # Never commit a token the target would reject: the committed tokens
        # beyond the base are exactly the plain prefix of the accepted length.
        committed = checked.accepted_state.neural.tokens[len(state.neural.tokens):]
        assert list(committed) == list(plain[:checked.accepted_count])


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_random_receipt_tampering_fails_closed(target, seed):
    """Randomly tamper a committed receipt chain; every tampered chain must fail
    closed (typed failure, original state returned)."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    state = prefill(rt, state, tokens=(1, 2, 3, 4))
    rng = np.random.default_rng(seed)
    for trial in range(6):
        receipts = list(state.receipts)
        idx = int(rng.integers(0, len(receipts)))
        field = rng.choice(["token", "input_state", "output_state", "context_snapshot"])
        if field == "token":
            receipts[idx] = replace(receipts[idx], token=(receipts[idx].token + 1) % 16)
        elif field == "input_state":
            receipts[idx] = replace(receipts[idx], input_state="0" * 64)
        elif field == "output_state":
            receipts[idx] = replace(receipts[idx], output_state="f" * 64)
        else:
            receipts[idx] = replace(receipts[idx], context_snapshot="0" * 64)
        bad = replace(state, receipts=tuple(receipts))
        result = rt.execute(bad, InferenceRequest("g", state.context.digest, "GREEDY", count=1),
                            expected_state=bad.digest)
        assert result.state == bad
        assert result.receipt.terminal == "FAILED"
        assert result.receipt.target_steps == ()


@pytest.mark.parametrize("seed", RECORDED_SEEDS)
def test_random_token_sequences_deterministic(target, seed):
    """Random token sequences must produce deterministic, cardinality-consistent
    committed state."""
    rt, state = build_runtime(target[0]), build_runtime(target[0]).initial(initial_snapshot())
    rng = np.random.default_rng(seed)
    vocab = rt.target.config.vocab
    for trial in range(4):
        n = int(rng.integers(1, 12))
        tokens = tuple(int(rng.integers(0, vocab)) for _ in range(n))
        req = InferenceRequest("p", state.context.digest, "PREFILL", tokens)
        a = rt.execute(state, req, expected_state=state.digest)
        b = rt.execute(state, req, expected_state=state.digest)
        assert a.state == b.state
        assert a.receipt == b.receipt
        assert len(a.state.neural.tokens) == n
        assert len(a.state.receipts) == n
