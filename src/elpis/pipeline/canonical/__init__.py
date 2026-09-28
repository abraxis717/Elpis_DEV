"""Canonical writer path: promotion authority -> candidate -> atomic publication.

The only path that changes canonical Grid81 state:

    advisory promotion plan (elpis.pipeline.promotion)
    -> authority: explicit operator approval digest -> one-use capability
    -> candidate: isolated immediate-successor candidate tree
    -> publisher: durable ledger reservation, atomic exchange, post-commit
       verification through the production reader

Each stage is a separate authority boundary; no stage can perform another's
role, and the publisher requires the exact capability object, not bare digests.
"""

from .authority import (
    AUTHORITY_POLICY_DIGEST,
    AUTHORIZED_PUBLISHER_CLASS,
    PromotionAuthorityError,
    issue_promotion_capability,
    require_promotion_capability,
    validate_promotion_capability,
)
from .candidate import (
    CandidateConstructionError,
    CandidateConstructionReceipt,
    construct_candidate,
)
from .publisher import (
    CanonicalPublicationReceipt,
    PublicationError,
    publish_candidate,
    publication_lock_path,
)

__all__ = [
    "AUTHORITY_POLICY_DIGEST", "AUTHORIZED_PUBLISHER_CLASS", "PromotionAuthorityError",
    "issue_promotion_capability", "require_promotion_capability", "validate_promotion_capability",
    "CandidateConstructionError", "CandidateConstructionReceipt", "construct_candidate",
    "CanonicalPublicationReceipt", "PublicationError", "publish_candidate", "publication_lock_path",
]
