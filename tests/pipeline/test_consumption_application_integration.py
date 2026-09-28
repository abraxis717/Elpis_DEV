from __future__ import annotations

import copy

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
from elpis.pipeline.application.durable_ledger_v2 import (
    DurableApplicationLedgerV2,
)
from elpis.pipeline.application.lifecycle import (
    APPLICATION_ACCEPTED,
    REJECTION_ALREADY_APPLIED_ARTIFACT,
)
from elpis.pipeline.application.shadow_state import ShadowCapabilityState


def _compiled(scope_size: int = 2):
    proposals = [
        canonical_digest({"proposal": i, "scope": scope_size})
        for i in range(scope_size)
    ]
    capability = {
        "schema_version": "structural-influence-capability.v1",
        "capability_class": "STRUCTURAL_INFLUENCE_CAPABILITY_V1",
        "capability_digest": canonical_digest({"cap": scope_size}),
        "capability_semantic_digest": canonical_digest({"sem": scope_size}),
        "nonce_digest": canonical_digest({"nonce": scope_size}),
        "authorized_proposal_digests": proposals,
        "authorized_consumer_class": "STRUCTURAL_INFLUENCE_COMPILER_V1",
        "authorized_operation_class": "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
        "source_request_digest": canonical_digest({"req": scope_size}),
        "source_adjudication_record_digest": canonical_digest({"adj": scope_size}),
        "source_proposal_set_digest": canonical_digest({"set": scope_size}),
    }
    policy = create_consumption_policy()
    contract = create_compiler_contract()
    lifecycle = create_lifecycle_entry(
        capability["capability_digest"],
        capability["nonce_digest"],
    )
    request = create_transaction_input(
        capability=capability,
        lifecycle=lifecycle,
        consumer_class="STRUCTURAL_INFLUENCE_COMPILER_V1",
        consumer_contract_digest=contract["compiler_contract_digest"],
        requested_operation_class="PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
        logical_tick=0,
        consumption_ordinal=1,
        consumption_policy_digest=policy["policy_digest"],
        claims_not_made=["integration qualification"],
    )
    result = consume_capability(
        capability=capability,
        lifecycle=lifecycle,
        request=request,
        policy=policy,
        compiler_contract=contract,
    )
    return result, capability, contract


def _shadow(capability):
    return ShadowCapabilityState(
        capability_digest=capability["capability_digest"],
        application_state="UNAPPLIED",
        consumption_count=1,
        current_lifecycle_state="CONSUMED",
        applied_artifact_digest=None,
    )


def test_real_compiler_result_applies_unchanged_through_durable_v2(tmp_path):
    result, capability, contract = _compiled()
    artifact_before = copy.deepcopy(result["structural_influence_artifact"])
    result_before = copy.deepcopy(result)

    path = tmp_path / "cross-subsystem.sqlite"
    with DurableApplicationLedgerV2(path) as ledger:
        receipt = apply_consumption_result(
            result,
            _shadow(capability),
            ledger,
            compiler_contract_digest=contract["compiler_contract_digest"],
        )

        assert receipt["application_outcome"] == APPLICATION_ACCEPTED
        assert receipt["schema_version"] == "application-receipt.v2"
        assert ledger.has_artifact(artifact_before["artifact_digest"])
        assert ledger.has_receipt(receipt["receipt_digest"])
        assert ledger.verify_chain() == (True, "valid")

    assert result == result_before
    assert result["structural_influence_artifact"] == artifact_before

    with DurableApplicationLedgerV2(path) as reopened:
        duplicate = apply_consumption_result(
            result,
            _shadow(capability),
            reopened,
            compiler_contract_digest=contract["compiler_contract_digest"],
        )
        assert (
            duplicate["application_outcome"]
            == REJECTION_ALREADY_APPLIED_ARTIFACT
        )
        assert reopened.verify_chain() == (True, "valid")


def test_tampered_transaction_result_is_rejected_before_application(tmp_path):
    result, capability, contract = _compiled()
    tampered = copy.deepcopy(result)
    tampered["structural_influence_artifact"]["logical_tick"] = 99

    with DurableApplicationLedgerV2(tmp_path / "tampered.sqlite") as ledger:
        before = ledger.to_dict()
        with pytest.raises(ValueError, match="CONSUMPTION_RESULT_DIGEST_MISMATCH"):
            apply_consumption_result(
                tampered,
                _shadow(capability),
                ledger,
                compiler_contract_digest=contract["compiler_contract_digest"],
            )
        assert ledger.to_dict() == before


def test_rejected_compiler_transaction_cannot_cross_application_boundary(tmp_path):
    result, capability, contract = _compiled()
    rejected = copy.deepcopy(result)
    rejected["transaction_outcome"] = "CONSUMPTION_REJECTED_REPLAY"

    payload = {
        k: v for k, v in rejected.items()
        if k != "transaction_result_digest"
    }
    rejected["transaction_result_digest"] = canonical_digest(payload)

    with DurableApplicationLedgerV2(tmp_path / "rejected.sqlite") as ledger:
        before = ledger.to_dict()
        with pytest.raises(ValueError, match="CONSUMPTION_RESULT_NOT_ACCEPTED"):
            apply_consumption_result(
                rejected,
                _shadow(capability),
                ledger,
                compiler_contract_digest=contract["compiler_contract_digest"],
            )
        assert ledger.to_dict() == before
