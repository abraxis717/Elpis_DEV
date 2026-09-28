from __future__ import annotations

import copy
import hashlib

from elpis.pipeline.capability.capability import validate_capability
from elpis.pipeline.capability.compiler import compile_one
from elpis.pipeline.capability.lifecycle import validate_lifecycle_entry
from elpis.pipeline.capability.policy import create_canonical_policy

from elpis.pipeline.consumption.input import create_transaction_input
from elpis.pipeline.consumption.policy import (
    create_compiler_contract,
    create_consumption_policy,
)
from elpis.pipeline.consumption.transaction import consume_capability


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_capability_grant_consumes_directly():
    proposals = sorted([
        _h("i8-proposal-a"),
        _h("i8-proposal-b"),
    ])
    source_request = {
        "request_digest": _h("i8-source-request"),
        "adjudication_record_digest": _h("i8-adjudication"),
        "proposal_set_digest": _h("i8-proposal-set"),
        "referred_proposal_digests": proposals,
        "required_capability_class": (
            "STRUCTURAL_INFLUENCE_CAPABILITY_V1"
        ),
        "source_manifest_sha256": _h("i8-source-manifest"),
    }

    authority_policy = create_canonical_policy()
    compiled = compile_one(
        copy.deepcopy(source_request),
        authority_policy,
    )

    assert compiled["decision"]["decision_outcome"] == "GRANT_CAPABILITY"
    capability = compiled["capability"]
    lifecycle = compiled["lifecycle"]

    assert capability is not None
    assert lifecycle is not None
    assert validate_capability(capability) is True
    assert validate_lifecycle_entry(lifecycle) is True

    assert capability["source_request_digest"] == source_request[
        "request_digest"
    ]
    assert capability["source_adjudication_record_digest"] == source_request[
        "adjudication_record_digest"
    ]
    assert capability["source_proposal_set_digest"] == source_request[
        "proposal_set_digest"
    ]
    assert capability["authorized_proposal_digests"] == proposals
    assert capability["authorized_consumer_class"] == (
        "STRUCTURAL_INFLUENCE_COMPILER_V1"
    )
    assert capability["authorized_operation_class"] == (
        "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1"
    )
    assert lifecycle["capability_digest"] == capability["capability_digest"]
    assert lifecycle["nonce_digest"] == capability["nonce_digest"]
    assert lifecycle["initial_lifecycle_state"] == "GRANTED_UNCONSUMED"
    assert lifecycle["consumption_count"] == 0

    capability_before = copy.deepcopy(capability)
    lifecycle_before = copy.deepcopy(lifecycle)

    consumption_policy = create_consumption_policy()
    compiler_contract = create_compiler_contract()
    request = create_transaction_input(
        capability=capability,
        lifecycle=lifecycle,
        consumer_class=capability["authorized_consumer_class"],
        consumer_contract_digest=compiler_contract[
            "compiler_contract_digest"
        ],
        requested_operation_class=capability[
            "authorized_operation_class"
        ],
        logical_tick=0,
        consumption_ordinal=1,
        consumption_policy_digest=consumption_policy["policy_digest"],
        claims_not_made=["i8 cross-component qualification"],
    )
    request_before = copy.deepcopy(request)

    result = consume_capability(
        capability=capability,
        lifecycle=lifecycle,
        request=request,
        policy=consumption_policy,
        compiler_contract=compiler_contract,
    )

    assert result["transaction_outcome"] == "CONSUMPTION_ACCEPTED"
    assert result["reason_codes"] == []
    assert result["rejection_record"] is None

    artifact = result["structural_influence_artifact"]
    receipt = result["consumption_receipt"]
    transition = result["lifecycle_transition"]

    assert artifact["source_capability_digest"] == capability[
        "capability_digest"
    ]
    assert artifact["source_capability_semantic_digest"] == capability[
        "capability_semantic_digest"
    ]
    assert artifact["source_request_digest"] == source_request[
        "request_digest"
    ]
    assert artifact["source_adjudication_record_digest"] == source_request[
        "adjudication_record_digest"
    ]
    assert artifact["source_proposal_set_digest"] == source_request[
        "proposal_set_digest"
    ]
    assert artifact["authorized_proposal_digests"] == proposals
    assert artifact["consumer_class"] == capability[
        "authorized_consumer_class"
    ]

    assert receipt["capability_digest"] == capability["capability_digest"]
    assert receipt["consumption_request_digest"] == request[
        "consumption_request_digest"
    ]
    assert receipt["produced_influence_artifact_digest"] == artifact[
        "artifact_digest"
    ]

    assert transition["previous_lifecycle_state"] == (
        "GRANTED_UNCONSUMED"
    )
    assert transition["resulting_lifecycle_state"] == "CONSUMED"
    assert transition["previous_consumption_count"] == 0
    assert transition["resulting_consumption_count"] == 1

    assert capability == capability_before
    assert lifecycle == lifecycle_before
    assert request == request_before
