"""Evolution may propose; it may not authorize or evaluate itself (elpis.evolution.policy).

Every authority-bearing value of an attempt and of a promotion comes from an independently pinned policy and its
measured evaluator: the gate's scopes, budgets and contract; the evidence's partitions, scope verdict and metrics;
the selection and the operator-approved grant without which nothing is materialized. Each attack below is refused
or rejected. TEST_ONLY fixtures (tests/evolution/_policy_fixtures.py); no improvement of anything is claimed.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import shutil

import pytest

from elpis.evolution import EvolutionPathAssertion, EvolutionPathGate
from elpis.evolution.policy import (
    EvaluatorReport,
    EvolutionPolicy,
    EvolutionPolicyAuthority,
    EvolutionPolicyError,
    PromotionGrant,
    policy_of,
)
from elpis.evolution.promotion import (
    CandidateRecord,
    EvaluationEvidence,
    build_manifest_from_tree,
    select,
)

from . import _policy_fixtures as F
from ._policy_fixtures import PARTITIONS, d


def _code(call):
    with pytest.raises(EvolutionPolicyError) as info:
        call()
    return info.value.code


# -- the policy document -------------------------------------------------------------------------------------------

def test_a_policy_is_only_what_its_independent_pin_names():
    raw, pin = F.document(F.spec())
    assert EvolutionPolicy(raw, expected_sha256=pin).digest == pin
    assert _code(lambda: EvolutionPolicy(raw + b" ", expected_sha256=pin)) == "EVOLUTION_POLICY_UNPINNED"
    assert _code(lambda: EvolutionPolicy(raw, expected_sha256=d("other"))) == "EVOLUTION_POLICY_UNPINNED"
    policy = EvolutionPolicy(raw, expected_sha256=pin)
    with pytest.raises(AttributeError):
        policy.max_edit_budget = 10**6


@pytest.mark.parametrize("override", [
    {"candidate_scopes": ["agent", "evaluator/hooks"]},                  # a candidate scope inside a protected one
    {"candidate_scopes": ["agent", "policy"]},
    {"protected_scopes": ["agent/evaluator"]},                            # a protected scope inside a candidate one
    {"candidate_scopes": ["../outside"]},
    {"candidate_scopes": ["/abs"]},
    {"evaluation_contract": {"partitions": {**PARTITIONS, "OOD": PARTITIONS["HELD_OUT"]}, "noise_envelope": 3,
                             "ood_regression_floor": -1}},               # held-out reused as OOD
    {"evaluation_contract": {"partitions": PARTITIONS, "noise_envelope": 0, "ood_regression_floor": -1}},
    {"evaluation_contract": {"partitions": PARTITIONS, "noise_envelope": 3, "ood_regression_floor": -50}},
    {"budget": {"max_edit_budget": 4, "max_resource_cost": 10**6, "max_candidates": 4}},
    {"budget": {"max_edit_budget": 4, "max_resource_cost": 100, "max_candidates": 0}},
    {"promotion": {"requires_operator_approval": False}},
    {"side_effects": "ANY"},
    {"confinement": "SANDBOXED"},                                        # a claim this module does not make
    {"objective": {"objective_id": "x", "metric": "self_reported_score", "direction": "MAXIMIZE"}},
    {"provenance": "synthetic"},
    {"extra": 1},
])
def test_a_policy_that_loosens_the_law_or_overlaps_its_protections_is_refused(override):
    raw, pin = F.document(F.spec(**override))
    assert _code(lambda: EvolutionPolicy(raw, expected_sha256=pin)) == "EVOLUTION_POLICY"


def test_the_evaluator_is_the_measured_pinned_implementation():
    raw, pin = F.document(F.spec())
    policy = EvolutionPolicy(raw, expected_sha256=pin)
    assert _code(lambda: EvolutionPolicyAuthority(policy, F.scribble)) == "EVALUATOR_UNPINNED"
    assert _code(lambda: EvolutionPolicyAuthority(policy, lambda **kw: None)) == "EVALUATOR_IDENTITY"
    # A pin naming other bytes of the same function is refused: the candidate never supplies the digest.
    forged = F.spec()
    forged["evaluator"] = {**forged["evaluator"], "implementation_sha256": d("forged evaluator bytes")}
    raw, pin = F.document(forged)
    assert _code(lambda: EvolutionPolicyAuthority(EvolutionPolicy(raw, expected_sha256=pin), F.score)) == \
        "EVALUATOR_UNPINNED"
    authority = F.authority()
    authority._evaluator = F.scribble        # swapped after admission: every use re-measures
    assert _code(authority.gate) == "EVALUATOR_UNPINNED"


# -- attempts: the gate comes from the policy ---------------------------------------------------------------------

def _assertion(authority, **overrides):
    values = dict(episode_id="e", episode_state_digest=d("s"), structural_attempt_index=0,
                  previous_structural_attempt_digest=d("a"), previous_path_receipt_digest="0" * 64,
                  candidate_manifest_digest=d("candidate"), hypothesis_digest=d("h"), component_scope=("agent",),
                  edit_count=1, edit_budget=2, resource_budget_digest=authority.policy.resource_budget_digest,
                  evaluation_contract_digest=authority.policy.evaluation_contract_digest,
                  evolution_authority_revision=0, evolution_authority_digest=d("authority"))
    values.update(overrides)
    return EvolutionPathAssertion(**values)


class _State:
    episode_id, structural_attempt_index, previous_structural_attempt_digest = "e", 0, d("a")

    def digest(self):
        return d("s")


def test_an_assertion_cannot_widen_its_own_scope_budget_or_contract():
    from elpis.evolution import EvolutionAuthorityBinding
    authority = F.authority()
    gate = authority.gate()
    binding = EvolutionAuthorityBinding(0, d("authority"), "0" * 64)
    assert gate.reject_reason(_assertion(authority), _State(), binding) is None
    for overrides, reason in (
            ({"edit_count": 9, "edit_budget": 99}, "EDIT_BUDGET_NOT_AUTHORIZED"),
            ({"component_scope": ("evaluator",)}, "COMPONENT_SCOPE_NOT_ALLOWED"),
            ({"component_scope": ("policy",)}, "COMPONENT_SCOPE_NOT_ALLOWED"),
            ({"resource_budget_digest": d("my own budget")}, "RESOURCE_BUDGET_MISMATCH"),
            ({"evaluation_contract_digest": d("my own contract")}, "EVALUATION_CONTRACT_MISMATCH")):
        assert gate.reject_reason(_assertion(authority, **overrides), _State(), binding) == reason, overrides


def test_only_an_issued_unaltered_gate_is_the_policy_gate():
    authority = F.authority()
    gate = authority.gate()
    assert policy_of(gate) is authority
    authority.verify_gate(gate)
    lookalike = EvolutionPathGate(allowed_component_scopes=authority.policy.candidate_scopes,
                                  resource_budget_digest=authority.policy.resource_budget_digest,
                                  evaluation_contract_digest=authority.policy.evaluation_contract_digest,
                                  max_edit_budget=authority.policy.max_edit_budget)
    assert policy_of(lookalike) is None
    assert _code(lambda: authority.verify_gate(lookalike)) == "EVOLUTION_POLICY_UNAUTHORIZED"
    assert _code(lambda: F.authority().verify_gate(gate)) == "EVOLUTION_POLICY_UNAUTHORIZED"   # another issuer
    gate.allowed_component_scopes = frozenset({"agent", "evaluator"})
    assert _code(lambda: authority.verify_gate(gate)) == "EVOLUTION_POLICY_UNAUTHORIZED"
    widened = authority.gate()
    widened.max_edit_budget = None
    assert _code(lambda: authority.verify_gate(widened)) == "EVOLUTION_POLICY_UNAUTHORIZED"


# -- evaluation, selection and promotion ---------------------------------------------------------------------------

def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


PARENT_FILES = {"agent/score.txt": b"10", "agent/notes.txt": b"notes", "evaluator/README": b"protected"}


@pytest.fixture
def parent(tmp_path):
    root = _tree(tmp_path / "parent", PARENT_FILES)
    manifest = build_manifest_from_tree(root, generation=0, parent_digest=None,
                                        candidate_manifest_digest=d("founder"), path_receipt_digest=d("founder-path"),
                                        editable_surface_digest=d("surface"))
    return root, manifest


def _child(tmp_path, parent, label, edits: dict[str, bytes], deletes=()):
    root = tmp_path / ("child-" + label)
    shutil.copytree(parent[0], root)
    _tree(root, edits)
    for name in deletes:
        (root / name).unlink()
    manifest = build_manifest_from_tree(root, generation=1, parent_digest=parent[1].digest,
                                        candidate_manifest_digest=d("candidate-" + label),
                                        path_receipt_digest=d("path-" + label), editable_surface_digest=d("surface"))
    return root, manifest


def _evaluate(authority, parent, child):
    return authority.evaluate(parent=parent[1], candidate=child[1], parent_root=parent[0], candidate_root=child[0])


def test_an_independent_evaluation_selects_and_an_approved_grant_materializes(tmp_path, parent):
    authority = F.authority()
    child = _child(tmp_path, parent, "better", {"agent/score.txt": b"20"})
    record = _evaluate(authority, parent, child)
    evidence = record.evidence
    assert (evidence.held_out_delta, evidence.source_scope_pass, record.edit_count) == (10, True, 1)
    assert evidence.evaluation_contract_digest == authority.policy.evaluation_contract_digest
    assert dict(evidence.partition_manifest_digests) == PARTITIONS
    receipt = authority.select(parent[1], [record])
    assert receipt.disposition == "SELECT_CHALLENGER"
    assert receipt.selected_harness_manifest_digest == child[1].digest
    grant = authority.approve(receipt, operator_approval_digest=d("operator approves"))
    out = authority.materialize(grant, parent_dir=parent[0], dest_dir=tmp_path / "promoted",
                                edits=[{"op": "write", "path": "agent/score.txt", "data": b"20"}],
                                expected_manifest=child[1])
    assert out["status"] == "MATERIALIZED" and (tmp_path / "promoted" / "agent" / "score.txt").read_bytes() == b"20"


def test_self_reported_or_altered_evidence_is_never_weighed(tmp_path, parent):
    authority = F.authority()
    child = _child(tmp_path, parent, "claims", {"agent/score.txt": b"11"})
    honest = _evaluate(authority, parent, child)        # +1: within noise, so the incumbent is retained
    assert authority.select(parent[1], [honest]).disposition == "RETAIN_INCUMBENT"
    boasting = CandidateRecord(child[1], replace(honest.evidence, held_out_delta=100), honest.edit_count)
    smaller = replace(honest, edit_count=0)
    forged = CandidateRecord(child[1], EvaluationEvidence(
        candidate_harness_manifest_digest=child[1].digest, parent_harness_manifest_digest=parent[1].digest,
        evaluation_contract_digest=authority.policy.evaluation_contract_digest,
        path_transition_receipt_digest=child[1].path_transition_receipt_digest,
        partition_manifest_digests=tuple(sorted(PARTITIONS.items())), correctness_pass=True, leakage_pass=True,
        resource_pass=True, source_scope_pass=True, resource_cost=0, held_out_delta=50, ood_delta=0,
        noise_envelope=3), 1)
    receipt = authority.select(parent[1], [boasting, smaller, forged])
    assert receipt.disposition == "RETAIN_INCUMBENT"
    assert [reason for _, reason in receipt.rejected] == ["EVALUATION_NOT_INDEPENDENT"] * 3
    # The same forged record passes the bare promotion law: the law alone cannot tell who evaluated.
    assert select(parent[1], [forged], authority.policy.evaluation_contract_digest,
                  PARTITIONS).disposition == "SELECT_CHALLENGER"


def test_a_candidate_cannot_edit_or_carry_its_evaluator(tmp_path, parent):
    authority = F.authority()
    protected = _child(tmp_path, parent, "protected", {"agent/score.txt": b"30", "evaluator/README": b"mine"})
    record = _evaluate(authority, parent, protected)
    assert record.evidence.source_scope_pass is False
    assert authority.select(parent[1], [record]).rejected == ((d("candidate-protected"), "SOURCE_SCOPE_GATE_FAIL"),)
    outside = _child(tmp_path, parent, "outside", {"agent/score.txt": b"30", "tools/new.py": b"x"})
    assert _evaluate(authority, parent, outside).evidence.source_scope_pass is False
    evaluator_bytes = Path(inspect.getsourcefile(F.score)).read_bytes()
    carrier = _child(tmp_path, parent, "carrier", {"agent/score.txt": b"30", "agent/evaluator.py": evaluator_bytes})
    assert _code(lambda: _evaluate(authority, parent, carrier)) == "EVALUATOR_IN_CANDIDATE"


def test_the_evaluated_bytes_are_the_manifest_and_evaluation_has_no_side_effect(tmp_path, parent):
    authority = F.authority()
    child = _child(tmp_path, parent, "drift", {"agent/score.txt": b"20"})
    (child[0] / "agent" / "score.txt").write_bytes(b"90")      # the workspace is no longer its manifest
    assert _code(lambda: _evaluate(authority, parent, child)) == "WORKSPACE_MANIFEST_MISMATCH"
    child = _child(tmp_path, parent, "honest", {"agent/score.txt": b"20"})
    assert _code(lambda: _evaluate(F.authority(F.scribble), parent, child)) == "EVALUATION_SIDE_EFFECT"
    child = _child(tmp_path, parent, "verdict", {"agent/score.txt": b"20"})
    assert _code(lambda: _evaluate(F.authority(F.unreported), parent, child)) == "EVALUATION_INVALID"
    stranger = _child(tmp_path, parent, "stranger", {"agent/score.txt": b"20"})
    orphan = replace(stranger[1], parent_harness_manifest_digest=d("another parent"))
    assert _code(lambda: authority.evaluate(parent=parent[1], candidate=orphan, parent_root=parent[0],
                                            candidate_root=stranger[0])) == "STALE_PARENT_BINDING"


def test_selection_budget_and_policy_floor(tmp_path, parent):
    authority = F.authority()
    records = [_evaluate(authority, parent, _child(tmp_path, parent, str(n), {"agent/score.txt": b"%d" % (20 + n)}))
               for n in range(5)]
    assert _code(lambda: authority.select(parent[1], records)) == "EVOLUTION_BUDGET"
    assert authority.select(parent[1], records[:4]).selected_candidate_manifest_digest_or_null == d("candidate-3")
    costly = _child(tmp_path, parent, "costly", {"agent/score.txt": b"40", **{f"agent/f{i}": b"x" for i in range(9)}})
    record = _evaluate(authority, parent, costly)
    assert record.evidence.resource_cost > authority.policy.max_resource_cost and not record.evidence.resource_pass
    assert authority.select(parent[1], [record]).rejected == ((d("candidate-costly"), "RESOURCE_GATE_FAIL"),)


def test_nothing_is_promoted_without_an_issued_selection_and_an_operator_grant(tmp_path, parent):
    authority = F.authority()
    child = _child(tmp_path, parent, "better", {"agent/score.txt": b"20"})
    record = _evaluate(authority, parent, child)
    bare = select(parent[1], [record], authority.policy.evaluation_contract_digest, PARTITIONS)
    assert _code(lambda: authority.approve(bare, operator_approval_digest=d("ok"))) == "PROMOTION_UNAUTHORIZED"
    retained = authority.select(parent[1], [])
    assert _code(lambda: authority.approve(retained, operator_approval_digest=d("ok"))) == "PROMOTION_UNAUTHORIZED"
    receipt = authority.select(parent[1], [record])
    with pytest.raises(ValueError):
        authority.approve(receipt, operator_approval_digest="yes")
    edits = [{"op": "write", "path": "agent/score.txt", "data": b"20"}]
    forged = PromotionGrant(authority.policy.digest, receipt.digest, child[1].digest, d("ok"))   # never approved
    assert _code(lambda: authority.materialize(forged, parent_dir=parent[0], dest_dir=tmp_path / "out", edits=edits,
                                               expected_manifest=child[1])) == "PROMOTION_UNAUTHORIZED"
    grant = authority.approve(receipt, operator_approval_digest=d("ok"))
    assert _code(lambda: authority.materialize(grant, parent_dir=parent[0], dest_dir=tmp_path / "out", edits=edits,
                                               expected_manifest=parent[1])) == "PROMOTION_UNAUTHORIZED"
    assert _code(lambda: F.authority().materialize(grant, parent_dir=parent[0], dest_dir=tmp_path / "out",
                                                   edits=edits, expected_manifest=child[1])) == \
        "PROMOTION_UNAUTHORIZED"
    assert not (tmp_path / "out").exists()


def test_the_evaluator_report_is_typed():
    with pytest.raises(TypeError):
        EvaluatorReport(True, True, 0, 1.5, 0)
    with pytest.raises(TypeError):
        EvaluatorReport(1, True, 0, 1, 0)
    with pytest.raises(ValueError):
        EvaluatorReport(True, True, -1, 1, 0)
