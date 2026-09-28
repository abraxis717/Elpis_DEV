"""Application receipt construction and validation.

Produces ApplicationReceiptV1 for accepted or rejected applications.
"""
from .canonical import canonical_digest, check_hex64


def create_application_receipt(
    artifact_digest: str,
    capability_digest: str,
    application_outcome: str,
    previous_state_digest: str,
    resulting_state_digest: str,
    previous_ledger_head: str,
    resulting_ledger_head: str,
    consumer_class: str,
) -> dict:
    """Create an ApplicationReceiptV1."""
    receipt = {
        "schema_version": "application-receipt.v1",
        "artifact_digest": artifact_digest,
        "capability_digest": capability_digest,
        "application_outcome": application_outcome,
        "previous_state_digest": previous_state_digest,
        "resulting_state_digest": resulting_state_digest,
        "previous_ledger_head": previous_ledger_head,
        "resulting_ledger_head": resulting_ledger_head,
        "consumer_class": consumer_class,
        "timestamp": "deterministic",
        "receipt_digest": "",
    }

    # Compute self-digest (excluding receipt_digest field)
    digest_payload = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    receipt["receipt_digest"] = canonical_digest(digest_payload)

    return receipt


def validate_receipt(receipt: dict) -> tuple[bool, list[str]]:
    """Validate application receipt structure and digest integrity."""
    issues = []

    if receipt.get("schema_version") != "application-receipt.v1":
        issues.append("invalid_schema_version")

    if not receipt.get("receipt_digest"):
        issues.append("missing_receipt_digest")
    else:
        # Verify self-digest
        digest_fields = {k: v for k, v in receipt.items() if k != "receipt_digest"}
        expected = canonical_digest(digest_fields)
        if receipt["receipt_digest"] != expected:
            issues.append("receipt_digest_mismatch")

    required_fields = [
        "artifact_digest", "capability_digest", "application_outcome",
        "previous_state_digest", "resulting_state_digest",
        "previous_ledger_head", "resulting_ledger_head",
        "consumer_class",
    ]
    for field in required_fields:
        if not receipt.get(field):
            issues.append(f"missing_field:{field}")

    return len(issues) == 0, issues


_RECEIPT_V2_IDENTITY_FIELDS = (
    "schema_version",
    "artifact_digest",
    "capability_digest",
    "application_outcome",
    "previous_state_digest",
    "resulting_state_digest",
    "previous_ledger_head",
    "consumer_class",
    "timestamp",
)


def _receipt_v2_digest_payload(receipt: dict) -> dict:
    """Canonical fixed ApplicationReceiptV2 transition identity."""
    return {
        field: receipt[field]
        for field in _RECEIPT_V2_IDENTITY_FIELDS
    }


def create_application_receipt_v2(
    artifact_digest: str,
    capability_digest: str,
    application_outcome: str,
    previous_state_digest: str,
    resulting_state_digest: str,
    previous_ledger_head: str,
    resulting_ledger_head: str,
    consumer_class: str,
) -> dict:
    """Create ApplicationReceiptV2 with a ledger-finalization-stable digest."""
    receipt = {
        "schema_version": "application-receipt.v2",
        "artifact_digest": artifact_digest,
        "capability_digest": capability_digest,
        "application_outcome": application_outcome,
        "previous_state_digest": previous_state_digest,
        "resulting_state_digest": resulting_state_digest,
        "previous_ledger_head": previous_ledger_head,
        "resulting_ledger_head": resulting_ledger_head,
        "consumer_class": consumer_class,
        "timestamp": "deterministic",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        _receipt_v2_digest_payload(receipt)
    )
    return receipt


def validate_receipt_v2(receipt: dict) -> tuple[bool, list[str]]:
    """Validate ApplicationReceiptV2 structure and stable digest identity."""
    issues: list[str] = []
    if type(receipt) is not dict:
        return False, ["receipt_type_invalid"]

    if receipt.get("schema_version") != "application-receipt.v2":
        issues.append("invalid_schema_version")

    digest = receipt.get("receipt_digest", "")
    if not check_hex64(digest):
        issues.append("invalid_receipt_digest")
    elif digest != canonical_digest(_receipt_v2_digest_payload(receipt)):
        issues.append("receipt_digest_mismatch")

    for field in (
        "artifact_digest",
        "capability_digest",
        "previous_state_digest",
        "resulting_state_digest",
        "previous_ledger_head",
        "resulting_ledger_head",
    ):
        if not check_hex64(receipt.get(field, "")):
            issues.append(f"invalid_digest_field:{field}")

    for field in ("application_outcome", "consumer_class"):
        value = receipt.get(field)
        if type(value) is not str or not value:
            issues.append(f"missing_field:{field}")

    return len(issues) == 0, issues


def finalize_application_receipt_v2(
    receipt: dict,
    resulting_ledger_head: str,
) -> dict:
    """Bind the durable resulting head without changing receipt_digest."""
    if type(receipt) is not dict or receipt.get("schema_version") != "application-receipt.v2":
        raise ValueError("ApplicationReceiptV2 required")
    if not check_hex64(resulting_ledger_head):
        raise ValueError("resulting_ledger_head must be a lowercase SHA-256 digest")

    finalized = dict(receipt)
    finalized["resulting_ledger_head"] = resulting_ledger_head
    ok, issues = validate_receipt_v2(finalized)
    if not ok:
        raise ValueError("Invalid finalized ApplicationReceiptV2: " + ",".join(issues))
    return finalized
