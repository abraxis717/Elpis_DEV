# Integration qualification for application executor -> durable ledger v2.

from __future__ import annotations

from elpis.pipeline.application.application import apply_artifact
from elpis.pipeline.application.artifact import validate_receipt_v2
from elpis.pipeline.application.durable_ledger_v2 import (
    DurableApplicationLedgerV2,
)
from ._fixture import (
    create_shadow_artifact,
    create_shadow_fixture,
)
from elpis.pipeline.application.ledger import ApplicationLedger
from elpis.pipeline.application.lifecycle import (
    APPLICATION_ACCEPTED,
    REJECTION_ALREADY_APPLIED_ARTIFACT,
    REJECTION_ARTIFACT_SCHEMA_INVALID,
)
from elpis.pipeline.application.shadow_state import ShadowCapabilityState


def _context():
    fixture = create_shadow_fixture(0, scope_size=1)
    artifact = create_shadow_artifact(fixture, 0)
    shadow = ShadowCapabilityState(
        capability_digest=fixture["capability_digest"],
        application_state="UNAPPLIED",
        consumption_count=1,
        current_lifecycle_state="CONSUMED",
        applied_artifact_digest=None,
    )
    return fixture, artifact, shadow, artifact["compiler_contract_digest"]


def test_apply_artifact_accepts_and_persists_exact_v2_receipt_identity(tmp_path):
    _, artifact, shadow, compiler_digest = _context()
    path = tmp_path / "application-ledger-v2.sqlite"

    with DurableApplicationLedgerV2(path) as ledger:
        before_head = ledger.head
        receipt = apply_artifact(
            artifact,
            shadow,
            ledger,
            compiler_contract_digest=compiler_digest,
        )

        assert receipt["schema_version"] == "application-receipt.v2"
        assert receipt["application_outcome"] == APPLICATION_ACCEPTED
        assert receipt["resulting_ledger_head"] == ledger.head
        assert ledger.head != before_head

        valid, issues = validate_receipt_v2(receipt)
        assert valid, issues

        assert ledger.has_artifact(artifact["artifact_digest"])
        assert ledger.has_receipt(receipt["receipt_digest"])

        evidence = ledger.to_dict()
        assert evidence["count"] == 1
        assert evidence["entries"][0]["artifact_digest"] == artifact["artifact_digest"]
        assert evidence["entries"][0]["receipt_digest"] == receipt["receipt_digest"]
        assert evidence["entries"][0]["entry_digest"] == receipt["resulting_ledger_head"]
        assert receipt["new_shadow_state"]["application_state"] == "APPLIED"
        assert ledger.verify_chain() == (True, "valid")

        committed_head = ledger.head
        committed_receipt = receipt["receipt_digest"]

    with DurableApplicationLedgerV2(path) as reopened:
        assert reopened.head == committed_head
        assert reopened.has_artifact(artifact["artifact_digest"])
        assert reopened.has_receipt(committed_receipt)

        duplicate = apply_artifact(
            artifact,
            shadow,
            reopened,
            compiler_contract_digest=compiler_digest,
        )
        assert duplicate["schema_version"] == "application-receipt.v2"
        assert duplicate["application_outcome"] == REJECTION_ALREADY_APPLIED_ARTIFACT
        assert reopened.head == committed_head
        assert reopened.verify_chain() == (True, "valid")


def test_rejected_v2_application_does_not_advance_durable_ledger(tmp_path):
    _, artifact, shadow, compiler_digest = _context()
    broken = dict(artifact)
    broken["schema_version"] = "invalid"

    with DurableApplicationLedgerV2(tmp_path / "rejected.sqlite") as ledger:
        before = ledger.to_dict()
        receipt = apply_artifact(
            broken,
            shadow,
            ledger,
            compiler_contract_digest=compiler_digest,
        )
        assert receipt["schema_version"] == "application-receipt.v2"
        assert receipt["application_outcome"] == REJECTION_ARTIFACT_SCHEMA_INVALID
        assert ledger.to_dict() == before
        assert ledger.verify_chain() == (True, "empty")


def test_historical_v1_application_path_remains_available():
    _, artifact, shadow, compiler_digest = _context()
    ledger = ApplicationLedger()

    receipt = apply_artifact(
        artifact,
        shadow,
        ledger,
        compiler_contract_digest=compiler_digest,
    )
    assert receipt["schema_version"] == "application-receipt.v1"
    assert receipt["application_outcome"] == APPLICATION_ACCEPTED
    assert receipt["resulting_ledger_head"] == ledger.head
