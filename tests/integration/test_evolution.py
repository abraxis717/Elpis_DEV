"""Evolution end to end: gated attempts bound to the runtime's current evolution authority (continuity)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pytest

from elpis.continuity import store as continuity_store
from elpis.evolution import EvolutionAttempt, EvolutionPathAssertion, EvolutionPathGate, GateExecuted, GateRejected
from elpis.runtime import CompositionError, Runtime, RuntimeConfig

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


def assertion(episode, authority, previous_receipt):
    return EvolutionPathAssertion(
        episode_id=episode.episode_id, episode_state_digest=episode.digest(),
        structural_attempt_index=episode.structural_attempt_index,
        previous_structural_attempt_digest=episode.previous_structural_attempt_digest,
        previous_path_receipt_digest=previous_receipt, candidate_manifest_digest=d("candidate"),
        hypothesis_digest=d("hypothesis"), component_scope=("evolution/population",), edit_count=1,
        edit_budget=2, resource_budget_digest=d("budget"), evaluation_contract_digest=d("contract"),
        evolution_authority_revision=authority.revision, evolution_authority_digest=authority.digest)


def advance(*, state, label):
    after = Episode(state.episode_id, state.structural_attempt_index + 1, d("attempt-" + label),
                    d("state-" + label))
    return EvolutionAttempt(after, d("attempt-" + label), d("result-" + label), "ATTEMPT_COMMITTED")


def test_attempts_advance_the_durable_evolution_authority(runtime, ingress):
    runtime.run_ingress(ingress, POSITIVE)  # unrelated operations do not move the evolution authority
    authority0 = runtime.evolution_authority()
    assert authority0.revision == 0
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    first_assertion = assertion(episode, authority0, d("path-genesis"))
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


def test_evolution_authority_survives_restart(tmp_path):
    config = RuntimeConfig(tmp_path / "continuity")
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    with Runtime(config) as rt:
        first = rt.evolve(GATE, assertion=assertion(episode, rt.evolution_authority(), d("path-genesis")),
                          state=episode, advance=advance, advance_kwargs={"label": "1"})
        authority = rt.evolution_authority()
    with Runtime(config) as rt:
        assert rt.evolution_authority() == authority
        stale = assertion(first.result.state_after, type(authority)(0, "0" * 64), first.receipt.receipt_digest)
        rejected = rt.evolve(GATE, assertion=stale, state=first.result.state_after,
                             advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
        assert isinstance(rejected, GateRejected)


def test_failed_authority_publication_fail_stops_evolution(tmp_path, monkeypatch):
    config = RuntimeConfig(tmp_path / "continuity")
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    with Runtime(config) as rt:
        before = rt.evolution_authority()

        def fail(fd, data, offset):
            raise OSError(28, "ENOSPC")

        monkeypatch.setattr(continuity_store.os, "pwrite", fail)
        with pytest.raises(CompositionError) as info:
            rt.evolve(GATE, assertion=assertion(episode, before, d("path-genesis")), state=episode,
                      advance=advance, advance_kwargs={"label": "1"})
        assert info.value.code == "CONTINUITY_PUBLICATION_REFUSED"
        monkeypatch.undo()
        with pytest.raises(CompositionError) as info:
            rt.evolution_authority()
        assert info.value.code == "CONTINUITY_PUBLICATION_REFUSED"
    with Runtime(config) as rt:
        assert rt.evolution_authority() == before
