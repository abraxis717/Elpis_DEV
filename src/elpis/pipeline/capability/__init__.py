"""Deterministic capability authority evaluation.

    CapabilityReviewRequestV1 (from adjudication)
      -> CapabilityAuthorityEvaluationInputV1
      -> CapabilityAuthorityDecisionV1
      -> StructuralInfluenceCapabilityV1
      -> GRANTED_UNCONSUMED lifecycle state

Grants are decided by the canonical authority policy only; a grant is inert
until consumed and never activates anything.
"""
