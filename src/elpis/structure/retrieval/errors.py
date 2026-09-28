"""Structural retrieval errors: fail-closed, typed, deterministic."""

from __future__ import annotations


class RetrievalError(Exception):
    """Base error for structural retrieval failures."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {detail}")


class QueryDerivationError(RetrievalError):
    """Raised when query derivation fails or overflows."""


class RetrievalLibraryError(RetrievalError):
    """Raised when the native retrieval bridge cannot be loaded."""


class HybridRetrievalError(RetrievalError):
    """Raised when a native HACF retrieval call fails."""


class BundleValidationError(RetrievalError):
    """Raised when RetrievalBundle validation fails."""


class BudgetOverflowError(RetrievalError):
    """Raised when the retrieval budget is exceeded."""


class EvidenceError(RetrievalError):
    """Raised when an evidence envelope cannot be constructed."""
