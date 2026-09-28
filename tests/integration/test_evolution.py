"""Evolution end to end: gated attempts bound to the runtime's own ECS history, receipts recorded."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from elpis.evolution import EvolutionAttempt, EvolutionPathAssertion, EvolutionPathGate, GateExecuted, GateRejected
from elpis.runtime import ReceiptRecord

from .conftest import POSITIVE


def d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    structural_attempt_index: int
    previous_structural_attempt_digest: str
    state_digest: str

    def digest(self) -> str:
        return self.state_digest


GATE = EvolutionPathGate(allowed_component_scopes=("evolution/population",),
                         resource_budget_digest=d("budget"), evaluation_contract_digest=d("contract"))


def assertion(episode, projection, previous_receipt):
    return EvolutionPathAssertion(
        episode_id=episode.episode_id, episode_state_digest=episode.digest(),
        structural_attempt_index=episode.structural_attempt_index,
        previous_structural_attempt_digest=episode.previous_structural_attempt_digest,
        previous_path_receipt_digest=previous_receipt, candidate_manifest_digest=d("candidate"),
        hypothesis_digest=d("hypothesis"), component_scope=("evolution/population",), edit_count=1,
        edit_budget=2, resource_budget_digest=d("budget"), evaluation_contract_digest=d("contract"),
        history_projection_digest=projection.projection_digest,
        history_head_event_digest=projection.source.head_event_digest,
        history_final_state_root=projection.source.final_state_root)


def advance(*, state, label):
    after = Episode(state.episode_id, state.structural_attempt_index + 1, d("attempt-" + label),
                    d("state-" + label))
    return EvolutionAttempt(after, d("attempt-" + label), d("result-" + label), "ATTEMPT_COMMITTED")


def test_attempts_chain_through_the_runtime_history(runtime, ingress):
    runtime.run_ingress(ingress, POSITIVE)  # the history the first attempt reasons over is not empty
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    first_assertion = assertion(episode, runtime.history_projection(), d("path-genesis"))
    first, recorded = runtime.evolve(GATE, assertion=first_assertion, state=episode, advance=advance,
                                     advance_kwargs={"label": "1"})
    assert isinstance(first, GateExecuted) and first.advance_calls == 1
    assert recorded.record == ReceiptRecord.of(
        "evolution", "evolution.path-transition", first.receipt.receipt_digest,
        assertion=first_assertion.digest, history_projection=first_assertion.history_projection_digest,
        outcome="ATTEMPT_COMMITTED")

    # The history moved (the transition itself was recorded): the old binding is stale.
    episode = first.result.state_after
    stale = assertion(episode, runtime.history_projection(), first.receipt.receipt_digest)
    stale = replace(stale, history_projection_digest=first_assertion.history_projection_digest,
                    history_head_event_digest=first_assertion.history_head_event_digest,
                    history_final_state_root=first_assertion.history_final_state_root)
    calls = []
    rejected, none = runtime.evolve(GATE, assertion=stale, state=episode,
                                    advance=lambda **kw: calls.append(kw), advance_kwargs={})
    assert isinstance(rejected, GateRejected) and none is None and calls == []
    assert rejected.reason in ("HISTORY_PROJECTION_MISMATCH", "HISTORY_HEAD_MISMATCH")

    fresh = assertion(episode, runtime.history_projection(), first.receipt.receipt_digest)
    second, recorded2 = runtime.evolve(GATE, assertion=fresh, state=episode, advance=advance,
                                       advance_kwargs={"label": "2"})
    assert isinstance(second, GateExecuted)
    assert second.receipt.previous_path_receipt_digest == first.receipt.receipt_digest
    kinds = [r.record.kind for r in runtime.history.records()]
    assert kinds == ["ingress.proposal", "evolution.path-transition", "evolution.path-transition"]
    assert recorded2.event_index > recorded.event_index


def test_projection_of_the_history_is_read_only_and_exact(runtime, ingress):
    runtime.run_ingress(ingress, POSITIVE)
    root = runtime.history.state_root
    one = runtime.history_projection()
    two = runtime.history_projection()
    assert one.projection_digest == two.projection_digest and runtime.history.state_root == root
    assert len(one.records) == len(runtime.history.records()) == 1
