"""The evolution path gate admits one bounded attempt per exact, authority-bound assertion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import hashlib

import pytest

from elpis.evolution import (
    EvolutionAttempt,
    EvolutionAuthorityBinding,
    EvolutionGateError,
    EvolutionPathAssertion,
    EvolutionPathAssertionV0,
    EvolutionPathGate,
    GateExecuted,
    GateRejected,
)
from elpis.evolution.digests import domain_digest


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
def projection():
    # The current evolution authority the assertion is built against (revision 4).
    return EvolutionAuthorityBinding(4, d("evolution-authority-4"), d("path-head"))


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
        evolution_authority_revision=projection.revision,
        evolution_authority_digest=projection.digest,
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
        ({"previous_path_receipt_digest": d("forged")}, "PATH_PREDECESSOR_MISMATCH"),
        ({"episode_id": "wrong"}, "EPISODE_ID_MISMATCH"),
        ({"episode_state_digest": d("stale")}, "STALE_EPISODE_STATE"),
        ({"structural_attempt_index": 3}, "STRUCTURAL_ATTEMPT_INDEX_MISMATCH"),
        ({"previous_structural_attempt_digest": d("stale-head")}, "STRUCTURAL_ATTEMPT_HEAD_MISMATCH"),
        ({"edit_count": 3}, "EDIT_BUDGET_EXCEEDED"),
        ({"component_scope": ()}, "EMPTY_COMPONENT_SCOPE"),
        ({"component_scope": ("evolution/other",)}, "COMPONENT_SCOPE_NOT_ALLOWED"),
        ({"resource_budget_digest": d("other-resource")}, "RESOURCE_BUDGET_MISMATCH"),
        ({"evaluation_contract_digest": d("other-eval")}, "EVALUATION_CONTRACT_MISMATCH"),
        ({"evolution_authority_revision": 3}, "STALE_EVOLUTION_AUTHORITY"),
        ({"evolution_authority_digest": d("other-authority")}, "EVOLUTION_AUTHORITY_MISMATCH"),
    ],
)
def test_fail_closed_before_advance(projection, changes, expected):
    calls = []
    result = gate().execute(assertion=assertion(projection, **changes), state=state(),
                            authority=projection, advance=lambda **kw: calls.append(kw),
                            advance_kwargs={})
    assert isinstance(result, GateRejected)
    assert result.reason == expected and result.advance_calls == 0 and calls == []


def test_look_alike_authority_is_not_a_binding(projection):
    fake = SimpleNamespace(revision=projection.revision, digest=projection.digest)
    result = gate().execute(assertion=assertion(projection), state=state(), authority=fake,
                            advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
    assert isinstance(result, GateRejected) and result.reason == "EVOLUTION_AUTHORITY_INVALID"


def test_an_assertion_built_against_revision_n_is_refused_at_n_plus_one(projection):
    """Stale-authority protection: no history projection is involved."""
    advanced = EvolutionAuthorityBinding(projection.revision + 1, d("evolution-authority-5"), d("next-head"))
    result = gate().execute(assertion=assertion(projection), state=state(), authority=advanced,
                            advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
    assert isinstance(result, GateRejected) and result.reason == "STALE_EVOLUTION_AUTHORITY"


def test_retired_v0_assertions_keep_their_identity_but_are_refused(projection):
    s = state()
    v0 = EvolutionPathAssertionV0(
        s.episode_id, s.digest(), s.structural_attempt_index, s.previous_structural_attempt_digest,
        d("path-head"), d("candidate"), d("hypothesis"), ("evolution/population",), 1, 2,
        d("resource-budget"), d("evaluation-contract"), d("projection"), d("head"), d("root"))
    assert v0.payload()["schema"] == "elpis.evolution-path-assertion.v0"
    assert v0.digest == domain_digest("elpis.evolution-path-assertion.v0", v0.payload())
    result = gate().execute(assertion=v0, state=s, authority=projection,
                            advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
    assert isinstance(result, GateRejected) and result.reason == "ASSERTION_SCHEMA_RETIRED"
    v1 = assertion(projection)
    assert v1.payload()["schema"] == "elpis.evolution-path-assertion.v1" and v1.digest != v0.digest


def make_advance(outcome: str, close=None):
    def advance(*, state, **kwargs):
        after = FakeState(state.episode_id, state.structural_attempt_index + 1,
                          d("attempt-new"), d("state-after-" + outcome))
        return EvolutionAttempt(after, d("attempt-record-" + outcome), d("result-" + outcome), outcome, close)
    return advance


@pytest.mark.parametrize("outcome", ["ATTEMPT_COMMITTED", "ATTEMPT_REJECTED"])
def test_admitted_attempt_is_bound_without_reinterpretation(projection, outcome):
    result = gate().execute(assertion=assertion(projection), state=state(), authority=projection,
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
        gate().execute(assertion=assertion(projection), state=state(), authority=projection,
                       advance=lambda **kw: {"state": "anything"}, advance_kwargs={})


def test_receipts_chain_by_digest(projection):
    first = gate().execute(assertion=assertion(projection), state=state(), authority=projection,
                           advance=make_advance("ATTEMPT_COMMITTED"), advance_kwargs={}).receipt
    s2 = FakeState("episode-0", 3, d("attempt-new"), d("state-after-ATTEMPT_COMMITTED"))
    next_authority = EvolutionAuthorityBinding(projection.revision + 1, d("authority-next"), first.receipt_digest)
    second = gate().execute(
        assertion=assertion(next_authority, episode_state_digest=s2.digest(), structural_attempt_index=3,
                            previous_structural_attempt_digest=s2.previous_structural_attempt_digest,
                            previous_path_receipt_digest=first.receipt_digest),
        state=s2, authority=next_authority, advance=make_advance("ATTEMPT_REJECTED"),
        advance_kwargs={}).receipt
    assert second.previous_path_receipt_digest == first.receipt_digest


def test_gate_configuration_is_validated():
    with pytest.raises(ValueError):
        EvolutionPathGate(allowed_component_scopes=(), resource_budget_digest=d("r"),
                          evaluation_contract_digest=d("e"))
    with pytest.raises(ValueError):
        EvolutionPathGate(allowed_component_scopes=("x",), resource_budget_digest="R" * 64,
                          evaluation_contract_digest=d("e"))


def test_persisted_v1_assertion_and_v0_receipt_identities_are_unchanged(projection):
    from elpis.evolution.path_gate import PathTransitionReceipt
    # Computed using the gate at corrective base 233cf82533e0ee1f0fe97c435eb768e2163ef6fa.
    a = assertion(projection)
    assert a.digest == "8881e7e19419c7bca787572b3028147855d07c42656603e651d40190ab49a3a7"
    receipt = PathTransitionReceipt(a.digest, state().digest(), d("attempt"), d("result"),
                                    d("after"), "ATTEMPT_COMMITTED", None, d("path-head"))
    assert receipt.receipt_digest == "2c9dc7d579a4bcdeb7b71e7899f7231a1e0f7b4ebfe612b43e22a82480925216"
