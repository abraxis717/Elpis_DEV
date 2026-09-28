from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


from elpis.structure.grid81.canonical import load_current_grid81

from elpis.pipeline.consumption.canonical import canonical_digest
from elpis.pipeline.consumption.input import create_transaction_input
from elpis.pipeline.consumption.lifecycle import create_lifecycle_entry
from elpis.pipeline.consumption.policy import (
    create_compiler_contract,
    create_consumption_policy,
)
from elpis.pipeline.consumption.transaction import consume_capability

from elpis.pipeline.application import DurableApplicationLedger
from elpis.pipeline.application.application import apply_consumption_result
from elpis.pipeline.application.durable_ledger_v2 import (
    DurableApplicationLedgerV2,
)
from elpis.pipeline.application.shadow_state import ShadowCapabilityState

from elpis.pipeline.promotion.canonical import (
    CanonicalPromotionPlan,
    PhaseEvidence,
    PromotionDecision,
    SourceChain,
)
from elpis.pipeline.promotion.decision import DECISION_READY
from elpis.pipeline.promotion.source_binding import (
    bind_g53c_application_identity,
)

from elpis.pipeline.canonical.authority import issue_promotion_capability
from elpis.pipeline.canonical.candidate import construct_candidate
from elpis.pipeline.canonical.publisher import publish_candidate


SOURCE_GRID81 = Path(__file__).resolve().parents[1] / "fixtures" / "grid81" / "Canonical" / "Grid81"

INTENTIONS = (
    "VERIFY_CANONICAL_LEDGER_HEAD",
    "VERIFY_CAPABILITY_GRANTED_UNCONSUMED",
    "VERIFY_ARTIFACT_CANONICALLY_UNAPPLIED",
    "RESERVE_TRANSACTION_IDENTIFIER",
    "PERFORM_CANONICAL_APPLICATION",
    "APPEND_CANONICAL_RECEIPT",
    "VERIFY_POST_COMMIT_STATE",
)
PRECONDITIONS = (
    "canonical_ledger_head_verified",
    "capability_granted_and_unconsumed",
    "artifact_canonically_unapplied",
    "transaction_identifier_reserved",
    "post_commit_state_verified",
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _phase(phase_id: str, **kwargs) -> PhaseEvidence:
    return PhaseEvidence(
        phase_id=phase_id,
        source_directory=f"/runtime/{phase_id}",
        manifest_path=f"/runtime/{phase_id}/manifest.json",
        manifest_digest=_h(f"{phase_id}:manifest"),
        disposition="PASS",
        evidence_files=(),
        **kwargs,
    )


def _compiled():
    proposals = [
        canonical_digest({"proposal": 0, "i7": True}),
        canonical_digest({"proposal": 1, "i7": True}),
    ]
    capability = {
        "schema_version": "structural-influence-capability.v1",
        "capability_class": "STRUCTURAL_INFLUENCE_CAPABILITY_V1",
        "capability_digest": canonical_digest({"capability": "i7"}),
        "capability_semantic_digest": canonical_digest({"semantic": "i7"}),
        "nonce_digest": canonical_digest({"nonce": "i7"}),
        "authorized_proposal_digests": proposals,
        "authorized_consumer_class": "STRUCTURAL_INFLUENCE_COMPILER_V1",
        "authorized_operation_class": (
            "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1"
        ),
        "source_request_digest": canonical_digest({"request": "i7"}),
        "source_adjudication_record_digest": canonical_digest(
            {"adjudication": "i7"}
        ),
        "source_proposal_set_digest": canonical_digest(
            {"proposal-set": "i7"}
        ),
    }
    policy = create_consumption_policy()
    compiler_contract = create_compiler_contract()
    lifecycle = create_lifecycle_entry(
        capability["capability_digest"],
        capability["nonce_digest"],
    )
    request = create_transaction_input(
        capability=capability,
        lifecycle=lifecycle,
        consumer_class="STRUCTURAL_INFLUENCE_COMPILER_V1",
        consumer_contract_digest=compiler_contract[
            "compiler_contract_digest"
        ],
        requested_operation_class=(
            "PRODUCE_BOUNDED_STRUCTURAL_INFLUENCE_V1"
        ),
        logical_tick=0,
        consumption_ordinal=1,
        consumption_policy_digest=policy["policy_digest"],
        claims_not_made=["i7 integration qualification"],
    )
    result = consume_capability(
        capability=capability,
        lifecycle=lifecycle,
        request=request,
        policy=policy,
        compiler_contract=compiler_contract,
    )
    return result, capability, compiler_contract


def _copy_project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "canonical-project"
    (project_root / "Canonical").mkdir(parents=True)
    shutil.copytree(
        SOURCE_GRID81,
        project_root / "Canonical" / "Grid81",
    )
    return project_root


def test_live_chain_publishes_once_and_replays_idempotently(tmp_path):
    result, structural_capability, compiler_contract = _compiled()
    artifact = result["structural_influence_artifact"]

    shadow = ShadowCapabilityState(
        capability_digest=structural_capability["capability_digest"],
        application_state="UNAPPLIED",
        consumption_count=1,
        current_lifecycle_state="CONSUMED",
        applied_artifact_digest=None,
    )

    with DurableApplicationLedgerV2(
        tmp_path / "shadow-application.sqlite"
    ) as shadow_ledger:
        application_receipt = apply_consumption_result(
            result,
            shadow,
            shadow_ledger,
            compiler_contract_digest=compiler_contract[
                "compiler_contract_digest"
            ],
        )
        assert shadow_ledger.has_receipt(
            application_receipt["receipt_digest"]
        )

    g53c = bind_g53c_application_identity(
        _phase(
            "G5.3C",
            lifecycle_state="GRANTED_UNCONSUMED",
            receipt_chain_digest=_h("i7-qualified-receipt-chain"),
        ),
        structural_artifact=artifact,
        application_receipt=application_receipt,
    )

    chain = SourceChain(
        g53b1=_phase("G5.3B.1"),
        g53c=g53c,
        g53d=_phase("G5.3D"),
    )
    decision = PromotionDecision(
        decision=DECISION_READY,
        gate_vector=(_h("i7-gate-1"), _h("i7-gate-2")),
        source_chain_digest=chain.chain_digest,
        expected_canonical_preconditions=PRECONDITIONS,
    )
    plan = CanonicalPromotionPlan(
        intentions=INTENTIONS,
        decision_digest=decision.digest,
        source_chain_digest=chain.chain_digest,
    )

    project_root = _copy_project_root(tmp_path)
    before = load_current_grid81(project_root)

    with DurableApplicationLedger(
        tmp_path / "publication-ledger.sqlite"
    ) as publication_ledger:
        promotion = issue_promotion_capability(
            plan=plan,
            decision=decision,
            chain=chain,
            project_root=project_root,
            operator_approval_digest=_h("i7-operator-approval"),
            expected_publication_ledger_head=publication_ledger.head,
        )

        candidate_root = tmp_path / "candidate"
        constructed = construct_candidate(
            project_root=project_root,
            candidate_root=candidate_root,
            promotion_capability=promotion,
            structural_artifact=artifact,
        )

        candidate_state = load_current_grid81(candidate_root)
        assert candidate_state.generation_number == before.generation_number + 1
        assert candidate_state.canonical_digest == (
            constructed.candidate_canonical_digest
        )

        published = publish_candidate(
            project_root=project_root,
            candidate_root=candidate_root,
            ledger=publication_ledger,
            promotion_capability=promotion,
            lock_path=project_root / "Canonical",
        )

        assert published.status == "COMMITTED"
        assert published.resumed is False
        assert published.artifact_digest == artifact["artifact_digest"]
        assert published.promotion_capability_digest == promotion[
            "capability_digest"
        ]
        assert published.previous_canonical_digest == before.canonical_digest
        assert published.resulting_canonical_digest == (
            constructed.candidate_canonical_digest
        )
        assert publication_ledger.has_receipt(artifact["artifact_digest"])
        assert publication_ledger.to_dict()["count"] == 1
        assert publication_ledger.verify_chain() == (True, "valid")

        committed = load_current_grid81(project_root)
        assert committed.generation_number == before.generation_number + 1
        assert committed.canonical_digest == (
            constructed.candidate_canonical_digest
        )

        generation_path = (
            project_root
            / promotion["target_bindings"]["generation_target"]
        )
        generation = json.loads(generation_path.read_text(encoding="utf-8"))

        assert generation["source_chain_digest"] == chain.chain_digest
        assert generation["source_artifact_record"]["artifact_digest"] == (
            artifact["artifact_digest"]
        )
        assert generation["source_artifact_record"][
            "source_structural_capability_digest"
        ] == structural_capability["capability_digest"]

        shadow_evidence = generation["shadow_promotion_evidence"]
        assert shadow_evidence["application_receipt_digest"] == (
            application_receipt["receipt_digest"]
        )
        assert shadow_evidence["resulting_state_digest"] == (
            application_receipt["resulting_state_digest"]
        )
        assert shadow_evidence["source_application_ledger_head"] == (
            application_receipt["resulting_ledger_head"]
        )

        first_head = publication_ledger.head
        first_count = publication_ledger.to_dict()["count"]

        replay = publish_candidate(
            project_root=project_root,
            candidate_root=candidate_root,
            ledger=publication_ledger,
            promotion_capability=promotion,
            lock_path=project_root / "Canonical",
        )

        assert replay.status == "ALREADY_COMMITTED"
        assert replay.resumed is True
        assert replay.publication_receipt_digest == (
            published.publication_receipt_digest
        )
        assert replay.resulting_ledger_head == first_head
        assert publication_ledger.head == first_head
        assert publication_ledger.to_dict()["count"] == first_count == 1
        assert publication_ledger.verify_chain() == (True, "valid")
