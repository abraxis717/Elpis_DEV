"""The query-local contract documents are the preimages of pinned native digests.

The native ingress binds every proposal to the canonical-JSON SHA-256 of the
registry contract and of the representation policy. The documents are kept
byte-identical to the donor, and this test proves that the constants compiled
into the ingress are their digests.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
INGRESS = REPO / "native" / "pipeline" / "ingress"
SOURCE = (INGRESS / "src" / "regex_hacf_query_ingress.cpp").read_text(encoding="utf-8")


def _pinned(name: str) -> str:
    match = re.search(name + r'\s*=\s*"([0-9a-f]{64})"', SOURCE)
    assert match, name
    return match.group(1)


@pytest.mark.parametrize("document,constant", [
    ("QUERY_LOCAL_PROPOSAL_REGISTRY_V1.json", "REGISTRY_CONTRACT_HEX"),
    ("QUERY_LOCAL_PROPOSAL_REPRESENTATION_POLICY_V1.json", "POLICY_HEX"),
])
def test_pinned_contract_digest_matches_document(document, constant):
    value = json.loads((INGRESS / "contracts" / document).read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == _pinned(constant)


def test_contracts_grant_no_authority():
    registry = json.loads((INGRESS / "contracts" / "QUERY_LOCAL_PROPOSAL_REGISTRY_V1.json").read_text())
    policy = json.loads((INGRESS / "contracts" / "QUERY_LOCAL_PROPOSAL_REPRESENTATION_POLICY_V1.json").read_text())
    assert registry["registry_scope"] == "QUERY_LOCAL_OVERLAY_ONLY"
    assert all(t["max_authority"] == 0 for t in registry["node_types"])
    assert policy["execution_authority"] is False
    assert policy["persistent_base_graph_admission"] is False
    assert policy["requirements"]["all_authority_fields_zero"] is True
