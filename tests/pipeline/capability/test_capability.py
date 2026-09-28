"""Tests for capability compilation."""

from elpis.pipeline.capability.capability import create_capability, validate_capability
from elpis.pipeline.capability.policy import create_canonical_policy
from elpis.pipeline.capability.authority_context import create_authority_context
from tests.pipeline import _chain


def _sample_req():
    return _chain.review_requests()[0]


def test_capability_creation():
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
    assert validate_capability(cap)


def test_capability_constants():
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
    assert cap["capability_class"] == "STRUCTURAL_INFLUENCE_CAPABILITY_V1"
    assert cap["authorized_operation_class"] == "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1"
    assert cap["authorized_consumer_class"] == "STRUCTURAL_INFLUENCE_COMPILER_V1"
    assert cap["max_consumptions"] == 1
    assert cap["nontransferable"] is True


def test_capability_no_forbidden_fields():
    """Adding extra fields should fail validation."""
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


def test_capability_deterministic():
    req = _sample_req()
    policy = create_canonical_policy()
    context = create_authority_context(req.get("request_digest", ""))
    c1 = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    c2 = create_capability(
        source_request_digest=req.get("request_digest", ""),
        source_adjudication_record_digest=req.get("adjudication_record_digest", ""),
        source_proposal_set_digest=req.get("proposal_set_digest", ""),
        authorized_proposal_digests=req.get("referred_proposal_digests", []),
        authority_policy_digest=policy["policy_digest"],
        authority_context=context,
    )
    assert c1["capability_digest"] == c2["capability_digest"]


def test_capability_claims_not_made():
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
    assert len(cap["claims_not_made"]) > 0
    assert cap["claims_not_made"] == sorted(cap["claims_not_made"])
