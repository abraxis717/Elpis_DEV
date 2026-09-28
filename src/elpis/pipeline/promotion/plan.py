"""Non-self-executing canonical promotion plan and its closed data surface."""

from .canonical import (
    CanonicalPromotionPlan,
    PlanIntention,
    PromotionDecision,
    SourceChain,
)
from .decision import DECISION_READY


INTENTIONS = [
    PlanIntention(
        intention_type="VERIFY_CANONICAL_LEDGER_HEAD",
        description="Verify the expected canonical ledger head matches the state produced by G5.3C shadow applications.",
        parameter_digest="ledger_head_verification",
    ),
    PlanIntention(
        intention_type="VERIFY_CAPABILITY_GRANTED_UNCONSUMED",
        description="Verify the capability remains granted and canonically unconsumed before any canonical application.",
        parameter_digest="capability_state_verification",
    ),
    PlanIntention(
        intention_type="VERIFY_ARTIFACT_CANONICALLY_UNAPPLIED",
        description="Verify structural-influence artifacts remain canonically unapplied.",
        parameter_digest="artifact_state_verification",
    ),
    PlanIntention(
        intention_type="RESERVE_TRANSACTION_IDENTIFIER",
        description="Reserve a unique transaction identifier for the future canonical application.",
        parameter_digest="transaction_reservation",
    ),
    PlanIntention(
        intention_type="PERFORM_CANONICAL_APPLICATION",
        description="Execute the canonical application of shadow-qualified capability artifacts under canonical authority.",
        precondition="all_preconditions_verified",
        parameter_digest="canonical_application_execution",
    ),
    PlanIntention(
        intention_type="APPEND_CANONICAL_RECEIPT",
        description="Append a canonical application receipt to the canonical ledger.",
        precondition="canonical_application_completed",
        parameter_digest="canonical_receipt_append",
    ),
    PlanIntention(
        intention_type="VERIFY_POST_COMMIT_STATE",
        description="Verify post-commit canonical state matches expected transition.",
        precondition="canonical_receipt_appended",
        parameter_digest="post_commit_verification",
    ),
]


def render_plan(decision: PromotionDecision, chain: SourceChain) -> CanonicalPromotionPlan | None:
    """Render a non-executable promotion plan. Only when decision is READY."""
    if decision.decision != DECISION_READY:
        return None

    intention_digests = tuple(i.intention_type for i in INTENTIONS)

    return CanonicalPromotionPlan(
        intentions=intention_digests,
        decision_digest=decision.digest,
        source_chain_digest=chain.chain_digest,
        executable=False,
        self_applying=False,
        authoritative=False,
        canonical_write_permitted=False,
    )


def get_intentions() -> list:
    """Return the list of typed intentions for reporting."""
    return INTENTIONS


# The plan is data only: its closed field/intention surface is checked here and
# by the canonical promotion authority before any capability is issued.
_PLAN_INTENTIONS = (
    "VERIFY_CANONICAL_LEDGER_HEAD", "VERIFY_CAPABILITY_GRANTED_UNCONSUMED",
    "VERIFY_ARTIFACT_CANONICALLY_UNAPPLIED", "RESERVE_TRANSACTION_IDENTIFIER",
    "PERFORM_CANONICAL_APPLICATION", "APPEND_CANONICAL_RECEIPT",
    "VERIFY_POST_COMMIT_STATE",
)
_PLAN_FIELDS = frozenset({
    "intentions", "decision_digest", "source_chain_digest", "planner_version",
    "executable", "self_applying", "authoritative", "canonical_write_permitted",
})


def verify_plan_nonexecutable(plan) -> dict:
    """Validate the actual plan's closed data/capability surface.

    This proves the data contract only, not runtime isolation or observed absence
    of network access, mutation, or capability consumption.
    """
    violations = []
    if type(plan) is not CanonicalPromotionPlan:
        violations.append("PLAN_TYPE")
    else:
        if set(vars(plan)) != _PLAN_FIELDS:
            violations.append("PLAN_FIELDS")
        if type(plan.intentions) is not tuple or any(type(i) is not str for i in plan.intentions) or plan.intentions != _PLAN_INTENTIONS:
            violations.append("INTENTIONS")
        for name in ("decision_digest", "source_chain_digest"):
            value = getattr(plan, name)
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                violations.append(name.upper())
        if type(plan.planner_version) is not str or plan.planner_version != "1.0.0":
            violations.append("PLANNER_VERSION")
        for name in ("executable", "self_applying", "authoritative", "canonical_write_permitted"):
            if getattr(plan, name) is not False:
                violations.append(name.upper())
    result = {
        "schema": "elpis.grid81.promotion-plan-data-check.v2",
        "plan_non_executable": not violations,
        "violations_found": len(violations),
        "violation_details": violations,
    }
    if not violations:
        result["plan_digest"] = plan.digest
    return result
