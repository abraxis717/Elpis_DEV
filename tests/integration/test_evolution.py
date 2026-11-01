"""Evolution end to end: gated attempts bound to the runtime's current evolution authority.

The gate validates and executes at the boundary; RuntimeCore (native/runtime) owns the reservation, the finalization
and the fail-stop, all in its volatile session (ABI v8): the evolution authority lives and ends with the session, and a
new session starts at revision 0. Everything below is observed through RuntimeCore and the gate's own result. (ABI
v7's durable reservation, its publication faults, process death inside it and the explicit reconciliation of a
durable pending authority retired with the durable authority.)
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pytest

from elpis.evolution import EvolutionAttempt, EvolutionPathAssertion, EvolutionPathGate, GateExecuted, GateRejected
from elpis.runtime import CompositionError, Runtime

from ..conftest import require_runtime_library, runtime_config
from ..evolution._policy_fixtures import TEST_AUTHORITY, TEST_EVOLUTION_PIN
from ..evolution import _policy_fixtures as policies
from .conftest import POSITIVE

def config_at(evolution_pin=TEST_EVOLUTION_PIN):
    return runtime_config(require_runtime_library(), evolution_pin=evolution_pin)


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


# The gate comes from the pinned TEST_ONLY evolution policy (elpis.evolution.policy), never from the caller.
GATE = TEST_AUTHORITY.gate()
POLICY = TEST_AUTHORITY.policy


def assertion(episode, authority, previous_receipt):
    return EvolutionPathAssertion(
        episode_id=episode.episode_id, episode_state_digest=episode.digest(),
        structural_attempt_index=episode.structural_attempt_index,
        previous_structural_attempt_digest=episode.previous_structural_attempt_digest,
        previous_path_receipt_digest=previous_receipt, candidate_manifest_digest=d("candidate"),
        hypothesis_digest=d("hypothesis"), component_scope=("evolution/population",), edit_count=1,
        edit_budget=2, resource_budget_digest=POLICY.resource_budget_digest,
        evaluation_contract_digest=POLICY.evaluation_contract_digest,
        evolution_authority_revision=authority.revision, evolution_authority_digest=authority.digest)


def capture_receipts(monkeypatch):
    """Record the receipt digest of every attempt the gate executes (observation only)."""
    receipts = []
    execute = EvolutionPathGate.execute

    def recording(self, **kw):
        result = execute(self, **kw)
        receipts.append(result.receipt.receipt_digest)
        return result

    monkeypatch.setattr(EvolutionPathGate, "execute", recording)
    return receipts


def advance(*, state, label):
    after = Episode(state.episode_id, state.structural_attempt_index + 1, d("attempt-" + label),
                    d("state-" + label))
    return EvolutionAttempt(after, d("attempt-" + label), d("result-" + label), "ATTEMPT_COMMITTED")


def test_attempts_advance_the_sessions_evolution_authority(runtime, ingress):
    runtime.run_ingress(ingress, POSITIVE)  # unrelated operations do not move the evolution authority
    authority0 = runtime.evolution_authority()
    assert authority0.revision == 0
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    first_assertion = assertion(episode, authority0, "0" * 64)
    first = runtime.evolve(GATE, assertion=first_assertion, state=episode, advance=advance,
                           advance_kwargs={"label": "1"})
    assert isinstance(first, GateExecuted) and first.advance_calls == 1
    authority1 = runtime.evolution_authority()
    assert authority1.revision == 1 and authority1.digest != authority0.digest
    assert runtime.continuity.snapshot().evolution.head == first.receipt.receipt_digest

    # An assertion built against revision 0 is stale at revision 1, whatever else matches.
    episode = first.result.state_after
    calls = []
    stale = assertion(episode, authority0, first.receipt.receipt_digest)
    rejected = runtime.evolve(GATE, assertion=stale, state=episode,
                              advance=lambda **kw: calls.append(kw), advance_kwargs={})
    assert isinstance(rejected, GateRejected) and calls == []
    assert rejected.reason == "STALE_EVOLUTION_AUTHORITY"
    assert runtime.evolution_authority() == authority1

    # Replaying the first assertion executes nothing either.
    replay = runtime.evolve(GATE, assertion=first_assertion, state=Episode("episode", 0, d("genesis-attempt"),
                                                                            d("state-0")),
                            advance=lambda **kw: calls.append(kw), advance_kwargs={})
    assert isinstance(replay, GateRejected) and calls == []

    second = runtime.evolve(GATE, assertion=assertion(episode, authority1, first.receipt.receipt_digest),
                            state=episode, advance=advance, advance_kwargs={"label": "2"})
    assert isinstance(second, GateExecuted)
    assert second.receipt.previous_path_receipt_digest == first.receipt.receipt_digest
    assert runtime.evolution_authority().revision == 2


def test_the_evolution_authority_lives_and_ends_with_the_session():
    config = config_at()
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    with Runtime(config) as rt:
        genesis = rt.evolution_authority()
        first = rt.evolve(GATE, assertion=assertion(episode, genesis, "0" * 64),
                          state=episode, advance=advance, advance_kwargs={"label": "1"})
        authority = rt.evolution_authority()
        assert authority.revision == 1
    with Runtime(config) as rt:     # a new session: revision 0 again, nothing of the last one remains
        fresh = rt.evolution_authority()
        assert fresh.revision == 0 and fresh.head == "0" * 64 and fresh != authority
        # An assertion built against the last session's authority is stale here.
        stale = assertion(first.result.state_after, authority, first.receipt.receipt_digest)
        rejected = rt.evolve(GATE, assertion=stale, state=first.result.state_after,
                             advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
        assert isinstance(rejected, GateRejected)


@pytest.mark.parametrize("bad_result", [False, True])
def test_advance_exception_or_invalid_result_keeps_pending_for_the_rest_of_the_session(bad_result):
    config = config_at()
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    calls = 0

    def broken(**kw):
        nonlocal calls
        calls += 1
        if bad_result:
            return None
        raise RuntimeError("external effect may already have happened")

    with Runtime(config) as rt:
        a = assertion(episode, rt.evolution_authority(), "0" * 64)
        with pytest.raises(RuntimeError):
            rt.evolve(GATE, assertion=a, state=episode, advance=broken, advance_kwargs={})
        with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
            rt.evolution_authority()
        assert rt.continuity.snapshot().evolution.pending_assertion == a.digest
        with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
            rt.evolve(GATE, assertion=a, state=episode, advance=broken, advance_kwargs={})
    with Runtime(config) as rt:     # the pending reservation ended with its session: nothing is ever re-executed
        assert rt.continuity.snapshot().evolution.pending_assertion is None
        assert rt.evolution_authority().revision == 0
    assert calls == 1


def test_forged_predecessor_refused_before_reservation_or_execution():
    with Runtime(config_at()) as rt:
        episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
        # Test genesis and a nonzero authoritative head.
        for n in range(2):
            authority = rt.evolution_authority()
            durable = rt.continuity.snapshot()
            a = assertion(episode, authority, d("forged-predecessor"))
            result = rt.evolve(GATE, assertion=a, state=episode,
                               advance=lambda **kw: pytest.fail("must not execute"), advance_kwargs={})
            assert isinstance(result, GateRejected) and result.reason == "PATH_PREDECESSOR_MISMATCH"
            assert rt.continuity.snapshot() == durable  # nothing reserved: no session transition at all
            good = rt.evolve(GATE, assertion=assertion(episode, authority, authority.head),
                             state=episode, advance=advance, advance_kwargs={"label": str(n)})
            assert good.receipt.previous_path_receipt_digest == authority.head
            episode = good.result.state_after

def test_only_the_pinned_policy_gate_reaches_reservation():
    """A caller-built gate, another policy's gate or an altered issued gate is refused before RuntimeCore reserves
    anything: no caller configures the scopes, budgets or contract of an attempt (elpis.evolution.policy)."""
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    lookalike = EvolutionPathGate(allowed_component_scopes=POLICY.candidate_scopes,
                                  resource_budget_digest=POLICY.resource_budget_digest,
                                  evaluation_contract_digest=POLICY.evaluation_contract_digest,
                                  max_edit_budget=POLICY.max_edit_budget)
    altered = TEST_AUTHORITY.gate()
    altered.allowed_component_scopes = frozenset({"evolution/population", "policy"})
    other = policies.authority(budget={"max_edit_budget": 99, "max_resource_cost": 100, "max_candidates": 4})
    calls = []
    with Runtime(config_at()) as rt:
        before = rt.continuity.snapshot()
        a = assertion(episode, rt.evolution_authority(), "0" * 64)
        for gate, code in ((lookalike, "EVOLUTION_POLICY_UNAUTHORIZED"), (altered, "EVOLUTION_POLICY_UNAUTHORIZED"),
                           (other.gate(), "EVOLUTION_POLICY_UNAUTHORIZED")):
            with pytest.raises(CompositionError) as info:
                rt.evolve(gate, assertion=a, state=episode, advance=lambda **kw: calls.append(kw), advance_kwargs={})
            assert info.value.code == code
        assert rt.continuity.snapshot() == before and rt.fault is None and calls == []
        # The pinned policy's own gate is admitted.
        assert isinstance(rt.evolve(GATE, assertion=a, state=episode, advance=advance, advance_kwargs={"label": "1"}),
                          GateExecuted)
    with Runtime(config_at(evolution_pin=None)) as rt:
        with pytest.raises(CompositionError) as info:
            rt.evolve(GATE, assertion=assertion(episode, rt.evolution_authority(), "0" * 64), state=episode,
                      advance=lambda **kw: calls.append(kw), advance_kwargs={})
        assert info.value.code == "EVOLUTION_POLICY_UNPINNED" and calls == []
