"""Structural retrieval: HACF structural memory -> validated, bounded evidence.

query derivation -> hybrid retrieval (native HACF) -> RetrievalBundle
-> bundle validation + budget -> EvidenceEnvelope (structured observation)
"""

from .budget import BudgetDecision, RetrievalBudget, check_budget
from .contracts import EvidenceEnvelope, RetrievalBundle, RetrievalItem, RetrievalQuery
from .errors import (
    BudgetOverflowError, BundleValidationError, EvidenceError, HybridRetrievalError,
    QueryDerivationError, RetrievalError, RetrievalLibraryError,
)
from .evidence import build_evidence_envelope, evidence_envelope_digest
from .query import derive_query
from .validation import validate_bundle

__all__ = (
    "BudgetDecision", "RetrievalBudget", "check_budget",
    "EvidenceEnvelope", "RetrievalBundle", "RetrievalItem", "RetrievalQuery",
    "BudgetOverflowError", "BundleValidationError", "EvidenceError", "HybridRetrievalError",
    "QueryDerivationError", "RetrievalError", "RetrievalLibraryError",
    "build_evidence_envelope", "evidence_envelope_digest", "derive_query", "validate_bundle",
)
