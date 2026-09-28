"""Replay and idempotence tests."""


from elpis.pipeline.application.application import apply_artifact
from elpis.pipeline.application.shadow_state import ShadowCapabilityState
from elpis.pipeline.application.ledger import ApplicationLedger
from ._fixture import (
    create_shadow_fixture, create_shadow_artifact, MutableShadowState,
)
from elpis.pipeline.application.lifecycle import (
    APPLICATION_ACCEPTED,
    REJECTION_ALREADY_APPLIED_ARTIFACT,
)


class TestReplay:
    def test_double_application_rejected(self):
        fixture = create_shadow_fixture(0, scope_size=1)
        artifact = create_shadow_artifact(fixture, 0)
        shadow = ShadowCapabilityState(
            capability_digest=fixture["capability_digest"],
            application_state="UNAPPLIED",
            consumption_count=1,
            current_lifecycle_state="CONSUMED",
            applied_artifact_digest=None,
        )
        ledger = ApplicationLedger()
        ccd = artifact["compiler_contract_digest"]

        receipt1 = apply_artifact(artifact, shadow, ledger, compiler_contract_digest=ccd)
        assert receipt1["application_outcome"] == APPLICATION_ACCEPTED

        receipt2 = apply_artifact(artifact, shadow, ledger, compiler_contract_digest=ccd)
        assert receipt2["application_outcome"] == REJECTION_ALREADY_APPLIED_ARTIFACT

    def test_rejected_replay_does_not_mutate(self):
        fixture = create_shadow_fixture(0, scope_size=1)
        artifact = create_shadow_artifact(fixture, 0)
        shadow = ShadowCapabilityState(
            capability_digest=fixture["capability_digest"],
            application_state="UNAPPLIED",
            consumption_count=1,
            current_lifecycle_state="CONSUMED",
            applied_artifact_digest=None,
        )
        ledger = ApplicationLedger()
        ccd = artifact["compiler_contract_digest"]
        import copy
        mut = copy.deepcopy(artifact)
        mut["schema_version"] = "invalid"

        before_state = shadow.state_digest
        before_ledger = ledger.head

        receipt = apply_artifact(mut, shadow, ledger, compiler_contract_digest=ccd)
        assert receipt["application_outcome"] != APPLICATION_ACCEPTED
        assert shadow.state_digest == before_state
        assert ledger.head == before_ledger

    def test_concurrent_artifacts_cannot_both_commit(self):
        fixture = create_shadow_fixture(0, scope_size=1)
        artifact = create_shadow_artifact(fixture, 0)
        shadow = ShadowCapabilityState(
            capability_digest=fixture["capability_digest"],
            application_state="UNAPPLIED",
            consumption_count=1,
            current_lifecycle_state="CONSUMED",
            applied_artifact_digest=None,
        )
        ledger = ApplicationLedger()
        ccd = artifact["compiler_contract_digest"]
        mutable = MutableShadowState(shadow)

        expected_head = ledger.head

        receipt1 = apply_artifact(artifact, mutable.state, ledger, compiler_contract_digest=ccd)
        assert receipt1["application_outcome"] == APPLICATION_ACCEPTED
        mutable.transition_to_applied(artifact["artifact_digest"])

        fixture2 = create_shadow_fixture(0, scope_size=1)
        artifact2 = create_shadow_artifact(fixture2, 0)
        receipt2 = apply_artifact(artifact2, mutable.state, ledger, compiler_contract_digest=ccd,
                                   expected_ledger_head=expected_head)
        assert receipt2["application_outcome"] != APPLICATION_ACCEPTED
