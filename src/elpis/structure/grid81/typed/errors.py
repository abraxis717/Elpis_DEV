"""Error hierarchy for the Typed Projection Compiler."""


class ElpisGridError(Exception):
    """Base exception for all Grid81 typed projection errors."""


class CanonicalizationError(ElpisGridError):
    """Error in canonical serialization or domain-separated digest."""


class SourceIdentityError(ElpisGridError):
    """Error in source row identity derivation or validation."""


class TransitionCompilerError(ElpisGridError):
    """Error in transition view compilation."""


class ExpansionCompilerError(ElpisGridError):
    """Error in expansion locus view compilation."""


class QuiescenceCompilerError(ElpisGridError):
    """Error in quiescence view compilation."""


class RationaleCompilerError(ElpisGridError):
    """Error in rationale view compilation."""


class D4Error(ElpisGridError):
    """Error in D4 group operations."""


class OrbitError(ElpisGridError):
    """Error in typed orbit identity computation."""
