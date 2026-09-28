from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from elpis.evolution.promotion import (
    CandidateRecord,
    EvaluationEvidence,
    HarnessManifest,
    NOISE_ENVELOPE,
    SelectionReceipt,
    eligibility,
    select,
)


def d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


PARTITIONS = {
    "EVOLVE": d("EVOLVE"),
    "CALIBRATION": d("CALIBRATION"),
    "HELD_OUT": d("HELD_OUT"),
    "OOD": d("OOD"),
}
EVAL = d("evaluation-contract")


def parent() -> HarnessManifest:
    return HarnessManifest(
        generation=0,
        parent_harness_manifest_digest=None,
        candidate_manifest_digest=d("founder-candidate"),
        path_transition_receipt_digest=d("founder-path"),
        editable_surface_manifest_digest=d("editable"),
        component_content_digests=(("root.txt", d("root")),),
        configuration_digest=d("config"),
        tool_contract_digest=d("tools"),
        prompt_or_policy_surface_digest=d("policy"),
        build_or_materialization_digest=d("build"),
    )


def candidate(label: str, *, held: int = 8, ood: int = 0, cost: int = 20, edit_count: int = 1):
    p = parent()
    manifest = HarnessManifest(
        generation=1,
        parent_harness_manifest_digest=p.digest,
        candidate_manifest_digest=d("candidate-" + label),
        path_transition_receipt_digest=d("path-" + label),
        editable_surface_manifest_digest=d("editable"),
        component_content_digests=(("root.txt", d("content-" + label)),),
        configuration_digest=d("config-" + label),
        tool_contract_digest=d("tools-" + label),
        prompt_or_policy_surface_digest=d("policy-" + label),
        build_or_materialization_digest=d("build-" + label),
    )
    evidence = EvaluationEvidence(
        candidate_harness_manifest_digest=manifest.digest,
        parent_harness_manifest_digest=p.digest,
        evaluation_contract_digest=EVAL,
        path_transition_receipt_digest=manifest.path_transition_receipt_digest,
        partition_manifest_digests=tuple(sorted(PARTITIONS.items())),
        correctness_pass=True,
        leakage_pass=True,
        resource_pass=True,
        source_scope_pass=True,
        resource_cost=cost,
        held_out_delta=held,
        ood_delta=ood,
        noise_envelope=NOISE_ENVELOPE,
    )
    return CandidateRecord(manifest, evidence, edit_count)


def test_clear_challenger_selected():
    p = parent()
    c = candidate("a", held=9)
    receipt = select(p, [c], EVAL, PARTITIONS)
    assert isinstance(receipt, SelectionReceipt)
    assert receipt.disposition == "SELECT_CHALLENGER"
    assert receipt.selected_harness_manifest_digest == c.manifest.digest
    assert receipt.selected_candidate_manifest_digest_or_null == c.manifest.candidate_manifest_digest


def test_no_eligible_candidate_retains_incumbent_byte_identity():
    p = parent()
    c = candidate("weak", held=NOISE_ENVELOPE)
    receipt = select(p, [c], EVAL, PARTITIONS)
    assert receipt.disposition == "RETAIN_INCUMBENT"
    assert receipt.selected_harness_manifest_digest == p.digest
    assert receipt.selected_candidate_manifest_digest_or_null is None


def test_tie_break_order_is_frozen_and_deterministic():
    p = parent()
    a = candidate("a", held=9, cost=20, edit_count=2)
    b = candidate("b", held=9, cost=20, edit_count=1)
    receipt = select(p, [a, b], EVAL, PARTITIONS)
    assert receipt.selected_harness_manifest_digest == b.manifest.digest
    assert select(p, [b, a], EVAL, PARTITIONS).digest == receipt.digest


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda c: CandidateRecord(replace(c.manifest, parent_harness_manifest_digest=d("wrong")), c.evidence, c.edit_count), "STALE_PARENT_BINDING"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, parent_harness_manifest_digest=d("wrong")), c.edit_count), "STALE_EVALUATION_PARENT_BINDING"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, candidate_harness_manifest_digest=d("wrong")), c.edit_count), "STALE_CANDIDATE_BINDING"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, path_transition_receipt_digest=d("wrong")), c.edit_count), "PATH_RECEIPT_MISMATCH"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, evaluation_contract_digest=d("wrong")), c.edit_count), "EVALUATION_CONTRACT_MISMATCH"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, partition_manifest_digests=tuple(sorted({k:v for k,v in PARTITIONS.items() if k != "OOD"}.items()))), c.edit_count), "MISSING_REQUIRED_PARTITION"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, correctness_pass=False), c.edit_count), "CORRECTNESS_GATE_FAIL"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, leakage_pass=False), c.edit_count), "LEAKAGE_GATE_FAIL"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, resource_pass=False), c.edit_count), "RESOURCE_GATE_FAIL"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, source_scope_pass=False), c.edit_count), "SOURCE_SCOPE_GATE_FAIL"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, held_out_delta=NOISE_ENVELOPE), c.edit_count), "WITHIN_NOISE_OR_NONIMPROVING"),
        (lambda c: CandidateRecord(c.manifest, replace(c.evidence, ood_delta=-3), c.edit_count), "OOD_CATASTROPHIC_REGRESSION"),
    ],
)
def test_fail_closed_selection_reasons(mutator, expected):
    p = parent()
    c = mutator(candidate("case"))
    ok, reason = eligibility(p, c, EVAL, PARTITIONS)
    assert ok is False
    assert reason == expected
