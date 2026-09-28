"""Tests for nonce determinism and uniqueness."""

from elpis.pipeline.capability.nonce import compute_nonce_digest, validate_nonce_digest
from elpis.pipeline.capability.policy import create_canonical_policy
from elpis.pipeline.capability.authority_context import create_authority_context

from tests.pipeline import _chain


def test_nonce_format():
    nonce = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    assert validate_nonce_digest(nonce)
    assert len(nonce) == 64


def test_nonce_deterministic():
    n1 = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    n2 = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    assert n1 == n2


def test_nonce_differs_by_request():
    n1 = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    n2 = compute_nonce_digest("d" * 64, "b" * 64, "c" * 64)
    assert n1 != n2


def test_nonce_differs_by_policy():
    n1 = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    n2 = compute_nonce_digest("a" * 64, "e" * 64, "c" * 64)
    assert n1 != n2


def test_nonce_differs_by_context():
    n1 = compute_nonce_digest("a" * 64, "b" * 64, "c" * 64)
    n2 = compute_nonce_digest("a" * 64, "b" * 64, "f" * 64)
    assert n1 != n2


def test_canonical_nonce_uniqueness():
    """Every review request of the chain yields a distinct nonce."""
    requests = _chain.review_requests()
    policy = create_canonical_policy()
    nonces = set()
    for req in requests:
        context = create_authority_context(req.get("request_digest", ""))
        nonce = compute_nonce_digest(
            req.get("request_digest", ""),
            policy["policy_digest"],
            context["authority_context_digest"],
        )
        nonces.add(nonce)
    assert len(nonces) == len(requests)


def test_invalid_nonce_format():
    assert not validate_nonce_digest("")
    assert not validate_nonce_digest("not_hex")
    assert not validate_nonce_digest("a" * 63)
    assert not validate_nonce_digest("a" * 65)
    assert not validate_nonce_digest("G" * 64)
