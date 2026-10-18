"""The mechanical ECS-dependence gates any future ECS<->DSV4 codec must pass (docs/qualification/ECS_CODEC_R0.md).

No ECS codec map is qualified; every canonical operation refuses ``ECS_CODEC_UNQUALIFIED`` without one. This file
proves only that the proposed gates discriminate: given TEST_ONLY TRAINING=NONE SEMANTICS=NONE fixture maps, they
accept a map whose answer is computed from the ECS readout and refuse one whose answer is not, which is the
property a hidden model (or any other cognition outside ECS) would violate:

* D1 determinism: the same text over the same ECS state yields the same tokens, in a fresh map instance too;
* D2 readout sufficiency: decoding the recorded readout alone, in a fresh map instance that never saw the tokens,
  reproduces the answer (no channel from the encode side, the text or any global state into the answer);
* D3 counterfactual sensitivity: over counterfactual ECS states (the same text; different W, including an ablated
  W = 0), the answers differ (an answer that ignores the ECS state is the same for all of them).

The fixtures' numbers mean nothing. Passing these gates says nothing about meaning or usefulness: it is necessary,
not sufficient, for an answer to be ECS computation rather than something else's.
"""
from __future__ import annotations

import numpy as np
import pytest

from elpis.runtime.cognition import QueryRequest, run_query

from ._turn_fixtures import ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import DIM, WIDTH, k1, world  # noqa: F401 (fixtures)

TEXT = "the same question"


class ReadoutBound(FixtureMap):
    """TEST_ONLY TRAINING=NONE SEMANTICS=NONE: each f_W(x) value quantized (1e-3) onto the printable bytes."""

    def decode_query(self, readout):
        self.calls.append(("decode_query", readout))
        return tuple(32 + round(v * 1000) % 94 for v in readout.values)


class ReadoutBlind(FixtureMap):
    """TEST_ONLY: answers from the text alone (as a hidden model would), whatever the ECS state computed."""

    def decode_query(self, readout):
        self.calls.append(("decode_query", readout))
        encoded = [c for c in self.calls if c[0] == "encode_query"]
        return tuple(97 + t % 26 for t in encoded[-1][1][:3]) if encoded else (63,)


class Smuggling(ReadoutBound):
    """TEST_ONLY: mostly readout-bound, but carries the encode side's tokens into the answer."""

    def decode_query(self, readout):
        bound = super().decode_query(readout)
        encoded = [c for c in self.calls if c[0] == "encode_query"]
        return bound + ((97 + encoded[-1][1][0] % 26,) if encoded else ())


def _counterfactuals(k1):   # noqa: F811
    zero = np.zeros((DIM, WIDTH))
    return [world(k1, w=np.random.default_rng(seed).normal(0.0, 0.5, size=(DIM, WIDTH)), max_rows=8)
            for seed in (1, 2, 3)] + [world(k1, w=zero, max_rows=8)]


def _ask(state, cls):
    return run_query(state, QueryRequest(TEXT, ByteTokens(), admitted(cls())))


def gates(states, cls):
    """D1, D2, D3 as booleans for one fixture class over counterfactual states."""
    results = [_ask(s, cls) for s in states]
    d1 = all(_ask(s, cls).output_tokens == r.output_tokens for s, r in zip(states, results))
    d2 = all(cls().decode_query(r.readout) == r.output_tokens for r in results)
    d3 = len({r.output_tokens for r in results}) == len(states)
    return d1, d2, d3


def test_the_gates_accept_an_answer_computed_from_the_ecs_readout(k1):   # noqa: F811
    states = _counterfactuals(k1)
    try:
        assert gates(states, ReadoutBound) == (True, True, True)
    finally:
        for s in states:
            s.close()


@pytest.mark.parametrize("cls,expected", [(ReadoutBlind, (True, False, False)), (Smuggling, (True, False, True)),
                                          (FixtureMap, (True, True, False))])
def test_the_gates_refuse_answers_that_do_not_depend_only_on_ecs_computation(k1, cls, expected):   # noqa: F811
    """ReadoutBlind (a hidden answerer) fails D2 and D3; Smuggling passes D3 but fails D2 (a channel around the ECS);
    the repository's own interface fixture is readout-blind by design (its reply ignores the answer), so it fails
    D3: it exercises the boundary's mechanics and must never be read as an ECS-dependent answer."""
    states = _counterfactuals(k1)
    try:
        assert gates(states, cls) == expected
    finally:
        for s in states:
            s.close()
