"""Error classes for capability consumption."""


class ConsumptionError(Exception):
    """Base class for capability consumption errors."""


class ValidationFailed(ConsumptionError):
    """Transaction input failed validation."""


class SchemaMismatch(ConsumptionError):
    """Schema version or structure mismatch."""


class ForbiddenFieldError(ConsumptionError):
    """Forbidden runtime/activation field detected in artifact."""
