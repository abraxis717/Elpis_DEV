"""Evolution end to end: gated attempts bound to the runtime's current evolution authority.

The gate validates and executes at the boundary; RuntimeCore (native/runtime) owns the durable reservation,
the finalization and the fail-stop. Everything below is observed through RuntimeCore: the continuity record,
its testing faults, and the gate's own result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pytest

from elpis.continuity import ContinuityProcessDeath
from elpis.evolution import EvolutionAttempt, EvolutionPathAssertion, EvolutionPathGate, GateExecuted, GateRejected
from elpis.runtime import CompositionError, Runtime, RuntimeConfig

from ..conftest import require_runtime_library, runtime_config
from ..evolution._policy_fixtures import TEST_AUTHORITY, TEST_EVOLUTION_PIN
from ..evolution import _policy_fixtures as policies
from .conftest import POSITIVE

SLOTS = ("continuity.a", "continuity.b")
# Testing-library fault actions (native/continuity, elpis_continuity_testing_fault).
DIE, WRITE_FAIL, TORN_FAIL, SYNC_FAIL_LOST, SYNC_FAIL_DURABLE = 1, 2, 3, 5, 6
STEPS = {"publish.begin": 0, "publish.written": 1, "publish.synced": 2}


def config_at(tmp_path, testing=False, evolution_pin=TEST_EVOLUTION_PIN):
    return runtime_config(tmp_path / "continuity", require_runtime_library(testing=testing),
                          evolution_pin=evolution_pin)


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


def test_attempts_advance_the_durable_evolution_authority(runtime, ingress):
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


def test_evolution_authority_survives_restart(tmp_path):
    config = config_at(tmp_path)
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    with Runtime(config) as rt:
        first = rt.evolve(GATE, assertion=assertion(episode, rt.evolution_authority(), "0" * 64),
                          state=episode, advance=advance, advance_kwargs={"label": "1"})
        authority = rt.evolution_authority()
    with Runtime(config) as rt:
        assert rt.evolution_authority() == authority
        stale = assertion(first.result.state_after, type(authority)(0, "0" * 64, "0" * 64), first.receipt.receipt_digest)
        rejected = rt.evolve(GATE, assertion=stale, state=first.result.state_after,
                             advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
        assert isinstance(rejected, GateRejected)


def test_reservation_refusal_executes_nothing_and_fail_stops_evolution(tmp_path):
    config = config_at(tmp_path, testing=True)
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    with Runtime(config) as rt:
        before = rt.evolution_authority()
        calls = []
        rt.continuity.testing_fault(1, WRITE_FAIL)
        with pytest.raises(CompositionError) as info:
            rt.evolve(GATE, assertion=assertion(episode, before, "0" * 64), state=episode,
                      advance=lambda **kw: calls.append(kw), advance_kwargs={})
        assert calls == []
        assert info.value.code == "CONTINUITY_PUBLICATION_REFUSED"
        with pytest.raises(CompositionError) as info:
            rt.evolution_authority()
        assert info.value.code == "CONTINUITY_PUBLICATION_REFUSED"
    with Runtime(config) as rt:
        assert rt.evolution_authority() == before

@pytest.mark.parametrize("phase", ["reservation", "finalization"])
@pytest.mark.parametrize("failure", ["refused", "torn", "uncertain-lost", "uncertain-durable"])
def test_publication_failure_never_reexecutes(tmp_path, monkeypatch, phase, failure):
    """Exercise both outcomes of uncertain publication, including lost dirty bytes."""
    config = config_at(tmp_path, testing=True)
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    calls = 0
    receipts = capture_receipts(monkeypatch)
    with Runtime(config) as rt:
        initial = rt.continuity.snapshot().evolution
        a = assertion(episode, rt.evolution_authority(), initial.head)
        library = rt.continuity.library

        def counted_advance(**kw):
            nonlocal calls
            # This assertion executes only with the exact reservation durable.
            snap = rt.continuity.snapshot()
            assert snap.evolution.pending_assertion == a.digest
            assert any(library.decode_record((config.continuity_dir / name).read_bytes()) == snap for name in SLOTS)
            calls += 1
            return advance(**kw, label="1")

        action, arg = {"refused": (WRITE_FAIL, 0), "torn": (TORN_FAIL, 80), "uncertain-lost": (SYNC_FAIL_LOST, 0),
                       "uncertain-durable": (SYNC_FAIL_DURABLE, 0)}[failure]
        rt.continuity.testing_fault(1 if phase == "reservation" else 2, action, arg)
        with pytest.raises(CompositionError) as info:
            rt.evolve(GATE, assertion=a, state=episode, advance=counted_advance, advance_kwargs={})
        code = "CONTINUITY_PUBLICATION_UNCERTAIN" if failure.startswith("uncertain") else "CONTINUITY_PUBLICATION_REFUSED"
        assert info.value.code == code
        assert calls == (0 if phase == "reservation" else 1)
        with pytest.raises(CompositionError, match=code):
            rt.evolve(GATE, assertion=a, state=episode, advance=counted_advance, advance_kwargs={})

    expected_pending = ((phase == "reservation" and failure == "uncertain-durable")
                        or (phase == "finalization" and failure != "uncertain-durable"))
    for _ in range(4):
        with Runtime(config) as rt:
            authority = rt.continuity.snapshot().evolution
            if expected_pending:
                assert authority.revision == initial.revision and authority.head == initial.head
                assert authority.pending_assertion == a.digest
                with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
                    rt.evolve(GATE, assertion=a, state=episode, advance=counted_advance, advance_kwargs={})
                with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
                    rt.evolution_authority()
            elif phase == "finalization":
                assert authority.pending_assertion is None
                assert authority.revision == initial.revision + 1 and authority.head == receipts[0]
                result = rt.evolve(GATE, assertion=a, state=episode,
                                   advance=counted_advance, advance_kwargs={})
                assert isinstance(result, GateRejected) and result.reason == "STALE_EVOLUTION_AUTHORITY"
            else:
                assert authority == initial  # Reservation was lost; no execution occurred.
            assert calls == (0 if phase == "reservation" else 1)


@pytest.mark.parametrize("bad_result", [False, True])
def test_advance_exception_or_invalid_result_preserves_pending(tmp_path, bad_result):
    config = config_at(tmp_path)
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
    with Runtime(config) as rt:
        assert rt.continuity.snapshot().evolution.pending_assertion == a.digest
        with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
            rt.evolve(GATE, assertion=a, state=episode, advance=broken, advance_kwargs={})
    assert calls == 1


def test_explicit_reconciliation_finalizes_once_without_execution(tmp_path, monkeypatch):
    from elpis.continuity import ContinuityError
    config = config_at(tmp_path, testing=True)
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    # The completed receipt, as an external reconciler could establish it. RuntimeCore never persists this
    # value or infers a result after a publication failure.
    receipts = capture_receipts(monkeypatch)
    with Runtime(config) as rt:
        a = assertion(episode, rt.evolution_authority(), "0" * 64)
        rt.continuity.testing_fault(2, WRITE_FAIL)  # the finalization's publication is refused
        with pytest.raises(CompositionError, match="CONTINUITY_PUBLICATION_REFUSED"):
            rt.evolve(GATE, assertion=a, state=episode, advance=advance, advance_kwargs={"label": "1"})
    [completed] = receipts
    with Runtime(config) as rt:
        pending = rt.continuity.snapshot().evolution
        assert pending.pending_assertion == a.digest
        final = rt.continuity.commit_evolution_transition(pending, completed).evolution
        assert final.revision == pending.revision + 1 and final.head == completed
        assert final.pending_assertion is None
        with pytest.raises(ContinuityError, match="CONTINUITY_AUTHORITY_MISMATCH"):
            rt.continuity.commit_evolution_transition(pending, completed)
        assert rt.fault is None  # an operator refusal, not a fail-stop
        result = rt.evolve(GATE, assertion=a, state=episode,
                           advance=lambda **kw: pytest.fail("must not run"), advance_kwargs={})
        assert isinstance(result, GateRejected) and result.reason == "STALE_EVOLUTION_AUTHORITY"
    with Runtime(config) as rt:
        assert rt.continuity.snapshot().evolution == final


def test_forged_predecessor_refused_before_reservation_or_execution(tmp_path):
    with Runtime(config_at(tmp_path)) as rt:
        episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
        # Test genesis and a nonzero authoritative head.
        for n in range(2):
            authority = rt.evolution_authority()
            durable = rt.continuity.snapshot()
            a = assertion(episode, authority, d("forged-predecessor"))
            result = rt.evolve(GATE, assertion=a, state=episode,
                               advance=lambda **kw: pytest.fail("must not execute"), advance_kwargs={})
            assert isinstance(result, GateRejected) and result.reason == "PATH_PREDECESSOR_MISMATCH"
            assert rt.continuity.snapshot() == durable  # nothing reserved: no publication at all
            good = rt.evolve(GATE, assertion=assertion(episode, authority, authority.head),
                             state=episode, advance=advance, advance_kwargs={"label": str(n)})
            assert good.receipt.previous_path_receipt_digest == authority.head
            episode = good.result.state_after

@pytest.mark.parametrize("phase", ["reservation", "finalization"])
@pytest.mark.parametrize("step", ["publish.begin", "publish.written", "publish.synced"])
def test_process_death_during_evolution_publication(tmp_path, phase, step):
    config = config_at(tmp_path, testing=True)
    episode = Episode("episode", 0, d("genesis-attempt"), d("state-0"))
    calls = 0
    rt = Runtime(config).open()
    a = assertion(episode, rt.evolution_authority(), "0" * 64)

    def counted(**kw):
        nonlocal calls
        calls += 1
        return advance(**kw, label="1")

    rt.continuity.testing_fault(1 if phase == "reservation" else 2, DIE, STEPS[step])
    try:
        with pytest.raises(ContinuityProcessDeath):
            rt.evolve(GATE, assertion=a, state=episode, advance=counted, advance_kwargs={})
    finally:
        rt.close()
    assert calls == (0 if phase == "reservation" else 1)
    for _ in range(3):
        with Runtime(config) as rt:
            authority = rt.continuity.snapshot().evolution
            if phase == "reservation" and step == "publish.begin":
                assert authority.pending_assertion is None and authority.revision == 0
            elif phase == "finalization" and step != "publish.begin":
                assert authority.pending_assertion is None and authority.revision == 1
                result = rt.evolve(GATE, assertion=a, state=episode,
                                   advance=counted, advance_kwargs={})
                assert isinstance(result, GateRejected)
            else:
                assert authority.pending_assertion == a.digest and authority.revision == 0
                with pytest.raises(CompositionError, match="CONTINUITY_EVOLUTION_PENDING"):
                    rt.evolve(GATE, assertion=a, state=episode, advance=counted, advance_kwargs={})
            assert calls == (0 if phase == "reservation" else 1)


def test_only_the_pinned_policy_gate_reaches_reservation(tmp_path):
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
    with Runtime(config_at(tmp_path)) as rt:
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
    with Runtime(config_at(tmp_path / "unpinned", evolution_pin=None)) as rt:
        with pytest.raises(CompositionError) as info:
            rt.evolve(GATE, assertion=assertion(episode, rt.evolution_authority(), "0" * 64), state=episode,
                      advance=lambda **kw: calls.append(kw), advance_kwargs={})
        assert info.value.code == "EVOLUTION_POLICY_UNPINNED" and calls == []
