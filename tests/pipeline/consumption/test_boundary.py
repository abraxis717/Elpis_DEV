"""Test authority boundary enforcement."""
from elpis.pipeline.consumption.canonical import canonical_digest
from elpis.pipeline.consumption.policy import create_consumption_policy, create_compiler_contract
from elpis.pipeline.consumption.input import create_transaction_input
from elpis.pipeline.consumption.transaction import consume_capability
from elpis.pipeline.consumption.lifecycle import create_lifecycle_entry
from elpis.pipeline.consumption.boundary import (
    verify_authority_boundary, create_authority_boundary_record,
)
from elpis.pipeline.consumption.validation import FORBIDDEN_FIELDS


def _make_capability(scope_size=1):
    proposals = [canonical_digest({"proposal": i, "scope": scope_size}) for i in range(scope_size)]
    cap = {
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
    return cap


def _consume(scope_size=1):
    policy = create_consumption_policy()
    contract = create_compiler_contract()
    cap = _make_capability(scope_size)
    life = create_lifecycle_entry(cap["capability_digest"], cap["nonce_digest"])
    req = create_transaction_input(cap, life, "STRUCTURAL_INFLUENCE_COMPILER_V1",
                                    contract["compiler_contract_digest"],
                                    "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
                                    0, 1, policy["policy_digest"], ["test"])
    return consume_capability(capability=cap, lifecycle=life, request=req,
                               policy=policy, compiler_contract=contract)


def test_accepted_authority_boundary():
    result = _consume()
    artifact = result["structural_influence_artifact"]
    ok, violations = verify_authority_boundary(artifact, result)
    assert ok is True
    assert len(violations) == 0


def test_rejected_authority_boundary():
    policy = create_consumption_policy()
    contract = create_compiler_contract()
    cap = _make_capability()
    life = create_lifecycle_entry(cap["capability_digest"], cap["nonce_digest"])
    life["revocation_state"] = "REVOKED"
    req = create_transaction_input(cap, life, "STRUCTURAL_INFLUENCE_COMPILER_V1",
                                    contract["compiler_contract_digest"],
                                    "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1",
                                    0, 1, policy["policy_digest"], ["test"])
    result = consume_capability(capability=cap, lifecycle=life, request=req,
                                 policy=policy, compiler_contract=contract)
    ok, violations = verify_authority_boundary(None, result)
    assert ok is True


def test_forbidden_fields_list_nonempty():
    assert len(FORBIDDEN_FIELDS) > 0
    assert "winner" in FORBIDDEN_FIELDS
    assert "gpu" in FORBIDDEN_FIELDS
    assert "model_id" in FORBIDDEN_FIELDS


def test_authority_boundary_record():
    result = _consume()
    artifact = result["structural_influence_artifact"]
    record = create_authority_boundary_record([artifact], [result])
    assert record["violation_count"] == 0
    assert record["evidence_only"] is True


def test_no_nondeterministic_imports():
    """Consumption compilation is deterministic: no randomness, ids or sockets."""
    from pathlib import Path
    import elpis.pipeline.consumption as package

    forbidden_kw = ["import torch", "import transformers", "import socket", "import random", "import uuid"]
    sources = sorted(Path(package.__file__).parent.rglob("*.py"))
    assert sources
    for path in sources:
        content = path.read_text(encoding="utf-8")
        for kw in forbidden_kw:
            assert kw not in content, f"Forbidden import '{kw}' in {path.name}"


def test_no_activation_in_result():
    result = _consume()
    # Check field names only, not claim text values
    field_names = set()
    def collect_names(obj):
        if isinstance(obj, dict):
            field_names.update(obj.keys())
            for v in obj.values():
                collect_names(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_names(item)
    collect_names(result)
    forbidden = {"winner", "gpu", "model_id", "model_name", "adapter_id",
                 "runtime_target", "device", "endpoint", "command",
                 "process_id", "server", "ecs_entity"}
    assert field_names.isdisjoint(forbidden), f"Forbidden: {field_names & forbidden}"
