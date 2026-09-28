"""Capability authority evaluation of one review request.

review request -> authority context -> evaluation input -> decision ->
abstention -> (on grant) scope, limit, capability, GRANTED_UNCONSUMED lifecycle.
"""

from .authority_context import create_authority_context
from .evaluation_input import create_evaluation_input
from .decision import evaluate_authority
from .abstention import create_grant_abstention, create_abstention
from .scope import create_capability_scope
from .limits import create_capability_limit
from .capability import create_capability
from .lifecycle import create_lifecycle_entry


def compile_one(source_request: dict, policy: dict) -> dict:
    """Compile a single row through the full authority pipeline."""
    # Authority context
    context = create_authority_context(source_request.get("request_digest", ""))

    # Evaluation input
    manifest_sha = source_request.get("source_manifest_sha256", "")
    eval_input = create_evaluation_input(
        source_request,
        policy["policy_digest"],
        context["authority_context_digest"],
        manifest_sha,
    )

    # Decision
    decision = evaluate_authority(eval_input, context, policy)

    # Abstention
    if decision["decision_outcome"] == "GRANT_CAPABILITY":
        abstention = create_grant_abstention()
    else:
        abstention = create_abstention(
            abstained=True,
            abstention_kind="NONE",
        )

    # Update decision with abstention digest
    decision["abstention_digest"] = abstention["abstention_digest"]

    # Capability (if granted)
    capability = None
    scope = None
    limit = None
    lifecycle = None

    if decision["decision_outcome"] == "GRANT_CAPABILITY":
        scope = create_capability_scope(source_request.get("referred_proposal_digests", []))
        limit = create_capability_limit()
        capability = create_capability(
            source_request_digest=source_request.get("request_digest", ""),
            source_adjudication_record_digest=source_request.get("adjudication_record_digest", ""),
            source_proposal_set_digest=source_request.get("proposal_set_digest", ""),
            authorized_proposal_digests=source_request.get("referred_proposal_digests", []),
            authority_policy_digest=policy["policy_digest"],
            authority_context=context,
        )
        lifecycle = create_lifecycle_entry(
            capability_digest=capability["capability_digest"],
            nonce_digest=capability["nonce_digest"],
        )

        decision["capability_digest"] = capability["capability_digest"]

    # Recompute decision digests after backfilling
    decision = recompute_decision(decision, abstention, capability)

    return {
        "context": context,
        "eval_input": eval_input,
        "decision": decision,
        "abstention": abstention,
        "scope": scope,
        "limit": limit,
        "capability": capability,
        "lifecycle": lifecycle,
    }


def recompute_decision(decision: dict, abstention: dict, capability: dict) -> dict:
    """Recompute decision digests after backfilling capability_digest."""
    from .canonical import canonical_digest

    decision["abstention_digest"] = abstention["abstention_digest"]
    if capability:
        decision["capability_digest"] = capability["capability_digest"]

    decision["authority_semantic_digest"] = canonical_digest({
        k: v for k, v in decision.items()
        if k not in ("authority_semantic_digest", "authority_decision_digest")
    })
    decision["authority_decision_digest"] = canonical_digest(decision)
    return decision
