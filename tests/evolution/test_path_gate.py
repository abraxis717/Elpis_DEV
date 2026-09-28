"""The evolution path gate admits one bounded attempt per exact, history-bound assertion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import hashlib

import pytest

from elpis.evolution import (
    EvolutionAttempt,
    EvolutionGateError,
    EvolutionPathAssertion,
    EvolutionPathGate,
    GateExecuted,
    GateRejected,
)
from elpis.evolution.digests import domain_digest

from ._history import ecs_projection


def d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class FakeState:
    episode_id: str
    structural_attempt_index: int
    previous_structural_attempt_digest: str
    state_digest: str

    def digest(self) -> str:
        return self.state_digest


@pytest.fixture(scope="module")
def projection(tmp_path_factory):
    return ecs_projection(tmp_path_factory.mktemp("history") / "kernel")


def state():
    return FakeState("episode-0", 2, d("attempt-head"), d("state-before"))


def assertion(projection, **overrides):
    s = state()
    base = EvolutionPathAssertion(
        episode_id=s.episode_id,
        episode_state_digest=s.digest(),
        structural_attempt_index=s.structural_attempt_index,
        previous_structural_attempt_digest=s.previous_structural_attempt_digest,
        previous_path_receipt_digest=d("path-head"),
        candidate_manifest_digest=d("candidate"),
        hypothesis_digest=d("hypothesis"),
        component_scope=("evolution/population",),
        edit_count=1,
        edit_budget=2,
        resource_budget_digest=d("resource-budget"),
        evaluation_contract_digest=d("evaluation-contract"),
        history_projection_digest=projection.projection_digest,
        history_head_event_digest=projection.source.head_event_digest,
        history_final_state_root=projection.source.final_state_root,
    )
    return replace(base, **overrides)


def gate():
    return EvolutionPathGate(
        allowed_component_scopes=("evolution/population",),
        resource_budget_digest=d("resource-budget"),
        evaluation_contract_digest=d("evaluation-contract"),
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"episode_id": "wrong"}, "EPISODE_ID_MISMATCH"),
        ({"episode_state_digest": d("stale")}, "STALE_EPISODE_STATE"),
        ({"structural_attempt_index": 3}, "STRUCTURAL_ATTEMPT_INDEX_MISMATCH"),
        ({"previous_structural_attempt_digest": d("stale-head")}, "STRUCTURAL_ATTEMPT_HEAD_MISMATCH"),
        ({"edit_count": 3}, "EDIT_BUDGET_EXCEEDED"),
        ({"component_scope": ()}, "EMPTY_COMPONENT_SCOPE"),
        ({"component_scope": ("evolution/other",)}, "COMPONENT_SCOPE_NOT_ALLOWED"),
        ({"resource_budget_digest": d("other-resource")}, "RESOURCE_BUDGET_MISMATCH"),
        ({"evaluation_contract_digest": d("other-eval")}, "EVALUATION_CONTRACT_MISMATCH"),
        ({"history_projection_digest": d("other-projection")}, "HISTORY_PROJECTION_MISMATCH"),
        ({"history_head_event_digest": d("other-head")}, "HISTORY_HEAD_MISMATCH"),
        ({"history_final_state_root": d("other-root")}, "HISTORY_ROOT_MISMATCH"),
    ],
)
def test_fail_closed_before_advance(projection, changes, expected):
    calls = []
    result = gate().execute(assertion=assertion(projection, **changes), state=state(),
                            projection=projection, advance=lambda **kw: calls.append(kw),
                            advance_kwargs={})
    assert isinstance(result, GateRejected)
    assert result.reason == expected and result.advance_calls == 0 and calls == []


def test_look_alike_projection_is_not_a_history_binding(projection):
    fake = SimpleNamespace(projection_digest=projection.projection_digest, source=projection.source)
    result = gate().execute(assertion=assertion(projection), state=state(), projection=fake,
                            advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
    assert isinstance(result, GateRejected) and result.reason == "HISTORY_PROJECTION_INVALID"


def make_advance(outcome: str, close=None):
    def advance(*, state, **kwargs):
        after = FakeState(state.episode_id, state.structural_attempt_index + 1,
                          d("attempt-new"), d("state-after-" + outcome))
        return EvolutionAttempt(after, d("attempt-record-" + outcome), d("result-" + outcome), outcome, close)
    return advance


@pytest.mark.parametrize("outcome", ["ATTEMPT_COMMITTED", "ATTEMPT_REJECTED"])
def test_admitted_attempt_is_bound_without_reinterpretation(projection, outcome):
    result = gate().execute(assertion=assertion(projection), state=state(), projection=projection,
                            advance=make_advance(outcome), advance_kwargs={"opaque": "value"})
    assert isinstance(result, GateExecuted)
    assert result.admitted is True and result.advance_calls == 1
    receipt = result.receipt
    assert receipt.attempt_outcome == outcome
    assert receipt.path_assertion_digest == assertion(projection).digest
    assert receipt.previous_path_receipt_digest == d("path-head")
    assert receipt.structural_attempt_digest == d("attempt-record-" + outcome)
    assert receipt.refinement_result_digest == d("result-" + outcome)
    assert receipt.episode_state_after_digest == d("state-after-" + outcome)
    assert receipt.receipt_digest == domain_digest(
        "elpis.evolution-path-transition-receipt.v0", receipt.payload())


def test_attempt_must_report_a_typed_result(projection):
    with pytest.raises(EvolutionGateError):
        gate().execute(assertion=assertion(projection), state=state(), projection=projection,
                       advance=lambda **kw: {"state": "anything"}, advance_kwargs={})


def test_receipts_chain_by_digest(projection):
    first = gate().execute(assertion=assertion(projection), state=state(), projection=projection,
                           advance=make_advance("ATTEMPT_COMMITTED"), advance_kwargs={}).receipt
    s2 = FakeState("episode-0", 3, d("attempt-new"), d("state-after-ATTEMPT_COMMITTED"))
    second = gate().execute(
        assertion=assertion(projection, episode_state_digest=s2.digest(), structural_attempt_index=3,
                            previous_structural_attempt_digest=s2.previous_structural_attempt_digest,
                            previous_path_receipt_digest=first.receipt_digest),
        state=s2, projection=projection, advance=make_advance("ATTEMPT_REJECTED"),
        advance_kwargs={}).receipt
    assert second.previous_path_receipt_digest == first.receipt_digest


def test_gate_configuration_is_validated():
    with pytest.raises(ValueError):
        EvolutionPathGate(allowed_component_scopes=(), resource_budget_digest=d("r"),
                          evaluation_contract_digest=d("e"))
    with pytest.raises(ValueError):
        EvolutionPathGate(allowed_component_scopes=("x",), resource_budget_digest="R" * 64,
                          evaluation_contract_digest=d("e"))
