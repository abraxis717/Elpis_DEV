from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


from elpis.pipeline.consumption.canonical import canonical_digest
from elpis.pipeline.consumption.input import create_transaction_input
from elpis.pipeline.consumption.lifecycle import create_lifecycle_entry
from elpis.pipeline.consumption.policy import (
    create_compiler_contract,
    create_consumption_policy,
)
from elpis.pipeline.consumption.transaction import consume_capability
from elpis.pipeline.application.application import apply_consumption_result
from elpis.pipeline.application.durable_ledger_v2 import DurableApplicationLedgerV2
from elpis.pipeline.application.shadow_state import ShadowCapabilityState
from elpis.pipeline.promotion.canonical import (
    CanonicalPromotionPlan,
    PhaseEvidence,
    PromotionDecision,
    SourceChain,
)
from elpis.pipeline.promotion.decision import DECISION_READY
from elpis.pipeline.promotion.source_binding import bind_g53c_application_identity
from elpis.pipeline.canonical.authority import issue_promotion_capability

CANONICAL_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "grid81"

INTENTIONS = (
    "VERIFY_CANONICAL_LEDGER_HEAD",
    "VERIFY_CAPABILITY_GRANTED_UNCONSUMED",
    "VERIFY_ARTIFACT_CANONICALLY_UNAPPLIED",
    "RESERVE_TRANSACTION_IDENTIFIER",
    "PERFORM_CANONICAL_APPLICATION",
    "APPEND_CANONICAL_RECEIPT",
    "VERIFY_POST_COMMIT_STATE",
)
PRECONDITIONS = (
    "canonical_ledger_head_verified",
    "capability_granted_and_unconsumed",
    "artifact_canonically_unapplied",
    "transaction_identifier_reserved",
    "post_commit_state_verified",
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _compiled():
    proposals = [
        canonical_digest({"proposal": 0, "i5": True}),
        canonical_digest({"proposal": 1, "i5": True}),
    ]
    capability = {
        "schema_version": "structural-influence-capability.v1",
        "capability_class": "STRUCTURAL_INFLUENCE_CAPABILITY_V1",
        "capability_digest": canonical_digest({"capability": "i5"}),
        "capability_semantic_digest": canonical_digest({"semantic": "i5"}),
        "nonce_digest": canonical_digest({"nonce": "i5"}),
        "authorized_proposal_digests": proposals,
        "authorized_consumer_class": "STRUCTURAL_INFLUENCE_COMPILER_V1",
        "authorized_operation_class": "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
        "source_request_digest": canonical_digest({"request": "i5"}),
        "source_adjudication_record_digest": canonical_digest({"adjudication": "i5"}),
        "source_proposal_set_digest": canonical_digest({"proposal-set": "i5"}),
    }
    policy = create_consumption_policy()
    compiler_contract = create_compiler_contract()
    lifecycle = create_lifecycle_entry(
        capability["capability_digest"],
        capability["nonce_digest"],
    )
    request = create_transaction_input(
        capability=capability,
        lifecycle=lifecycle,
        consumer_class="STRUCTURAL_INFLUENCE_COMPILER_V1",
        consumer_contract_digest=compiler_contract["compiler_contract_digest"],
        requested_operation_class="PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
        logical_tick=0,
        consumption_ordinal=1,
        consumption_policy_digest=policy["policy_digest"],
        claims_not_made=["i5 integration qualification"],
    )
    result = consume_capability(
        capability=capability,
        lifecycle=lifecycle,
        request=request,
        policy=policy,
        compiler_contract=compiler_contract,
    )
    return result, capability, compiler_contract


def _phase(phase_id: str, **kwargs) -> PhaseEvidence:
    return PhaseEvidence(
        phase_id=phase_id,
        source_directory=f"/runtime/{phase_id}",
        manifest_path=f"/runtime/{phase_id}/manifest.json",
        manifest_digest=_h(f"{phase_id}:manifest"),
        disposition="PASS",
        evidence_files=(),
        bundle_digest=_h(f"{phase_id}:bundle"),
        **kwargs,
    )


def test_live_application_receipt_identity_reaches_promotion_authority(tmp_path):
    result, capability, compiler_contract = _compiled()
    artifact = result["structural_influence_artifact"]
    shadow = ShadowCapabilityState(
        capability_digest=capability["capability_digest"],
        application_state="UNAPPLIED",
        consumption_count=1,
        current_lifecycle_state="CONSUMED",
        applied_artifact_digest=None,
    )

    with DurableApplicationLedgerV2(tmp_path / "i5.sqlite") as ledger:
        receipt = apply_consumption_result(
            result,
            shadow,
            ledger,
            compiler_contract_digest=compiler_contract["compiler_contract_digest"],
        )

    base = _phase(
        "G5.3C",
        lifecycle_state="GRANTED_UNCONSUMED",
        receipt_chain_digest=_h("qualified-receipt-chain"),
    )
    bound = bind_g53c_application_identity(
        base,
        structural_artifact=artifact,
        application_receipt=receipt,
    )

    assert base.artifact_digest is None
    assert base.shadow_receipt_digest is None
    assert bound.artifact_digest == artifact["artifact_digest"]
    assert bound.capability_digest == artifact["source_capability_digest"]
    assert bound.shadow_receipt_digest == receipt["receipt_digest"]
    assert bound.receipt_chain_digest == base.receipt_chain_digest
    assert bound.resulting_state_digest == receipt["resulting_state_digest"]
    assert bound.resulting_ledger_head == receipt["resulting_ledger_head"]

    chain = SourceChain(
        g53b1=_phase("G5.3B.1"),
        g53c=bound,
        g53d=_phase("G5.3D"),
    )
    decision = PromotionDecision(
        decision=DECISION_READY,
        gate_vector=(_h("gate-1"), _h("gate-2")),
        source_chain_digest=chain.chain_digest,
        expected_canonical_preconditions=PRECONDITIONS,
    )
    plan = CanonicalPromotionPlan(
        intentions=INTENTIONS,
        decision_digest=decision.digest,
        source_chain_digest=chain.chain_digest,
    )

    promotion = issue_promotion_capability(
        plan=plan,
        decision=decision,
        chain=chain,
        project_root=CANONICAL_ROOT,
        operator_approval_digest=_h("i5-operator-approval"),
        expected_publication_ledger_head=_h("i5-publication-head"),
    )
    source = promotion["source_bindings"]
    assert source["artifact_digest"] == artifact["artifact_digest"]
    assert source["structural_capability_digest"] == capability["capability_digest"]
    assert source["application_receipt_digest"] == receipt["receipt_digest"]
    assert source["resulting_state_digest"] == receipt["resulting_state_digest"]
    assert source["source_application_ledger_head"] == receipt["resulting_ledger_head"]


def test_application_identity_binder_rejects_tampered_receipt(tmp_path):
    result, capability, compiler_contract = _compiled()
    artifact = result["structural_influence_artifact"]
    shadow = ShadowCapabilityState(
        capability_digest=capability["capability_digest"],
        application_state="UNAPPLIED",
        consumption_count=1,
        current_lifecycle_state="CONSUMED",
        applied_artifact_digest=None,
    )
    with DurableApplicationLedgerV2(tmp_path / "tampered.sqlite") as ledger:
        receipt = apply_consumption_result(
            result,
            shadow,
            ledger,
            compiler_contract_digest=compiler_contract["compiler_contract_digest"],
        )

    tampered = dict(receipt)
    tampered["resulting_state_digest"] = _h("forged-state")
    with pytest.raises(ValueError, match="G53C_APPLICATION_RECEIPT_DIGEST_MISMATCH"):
        bind_g53c_application_identity(
            _phase("G5.3C"),
            structural_artifact=artifact,
            application_receipt=tampered,
        )
