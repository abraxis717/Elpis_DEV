"""Adjudication output stays below the capability boundary.

Review requests and adjudication records produced by the real stages over the
synthetic structural chain must not carry capability, activation or runtime
fields. Import-level boundaries are enforced repository-wide by
``tests/boundary``.
"""

from tests.pipeline import _chain

FORBIDDEN_FIELDS = [
    "capability_token", "authority_token", "model_path", "adapter_path",
    "device", "port", "command", "runtime", "selected", "activation",
    "score", "confidence", "threshold", "priority", "lifecycle_eligible",
]


def test_review_requests_no_capability_fields():
    requests = _chain.review_requests()
    assert requests
    for r in requests:
        for field in FORBIDDEN_FIELDS:
            assert field not in r, f"Forbidden field '{field}' in review request"
        assert r["required_capability_class"] == "STRUCTURAL_INFLUENCE_CAPABILITY_V1"
        assert len(r["claims_not_made"]) > 0


def test_adjudication_records_claims_not_made():
    results = _chain.adjudications()
    assert results
    for result in results:
        record = result["adjudication"]
        assert len(record.get("claims_not_made", [])) > 0
        for field in FORBIDDEN_FIELDS:
            assert field not in record, f"Forbidden field '{field}' in adjudication record"
