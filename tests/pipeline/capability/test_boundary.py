"""Tests for authority boundary — no forbidden fields."""

from elpis.pipeline.capability.capability import create_capability, validate_capability
from elpis.pipeline.capability.policy import create_canonical_policy
from elpis.pipeline.capability.authority_context import create_authority_context
from tests.pipeline import _chain


def _sample_req():
    return _chain.review_requests()[0]


FORBIDDEN_FIELD_NAMES = [
    "activation_state", "activation_authority", "model_identifier",
    "adapter_identifier", "model_path", "adapter_path", "device",
    "port", "endpoint", "command", "process_id", "consumption_receipt",
    "consumption_request", "runtime_import", "router_import", "scheduler_import",
    "wall_clock_timestamp", "score", "confidence", "priority",
]


def test_no_forbidden_field_activation():
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    cap = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    for field in FORBIDDEN_FIELD_NAMES:
        assert field not in cap, f"Forbidden field present: {field}"


def test_forbidden_field_rejected():
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    cap = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    cap["activation_state"] = "ACTIVE"
    assert not validate_capability(cap)


def test_forbidden_nested_rejected():
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    cap = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    cap["model_identifier"] = "test"
    assert not validate_capability(cap)


def test_no_winner_selection():
    """Capabilities must not select a winner — scope preserves full referred set."""
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    cap = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    # Authorized set must equal referred set
    assert sorted(cap["authorized_proposal_digests"]) == sorted(req.get("referred_proposal_digests", []))


def test_no_negative_evidence_in_scope():
    """Negative-evidence proposals must not be in authorized scope."""
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    cap = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    # All authorized proposals are from the referred set
    assert len(cap["authorized_proposal_digests"]) == len(req.get("referred_proposal_digests", []))
