"""structural-group projection compiler error types."""


class G50BError(Exception):
    """Base error for structural-group projection compiler."""


class UpstreamSealError(G50BError):
    """Upstream seal verification failed."""


class SourceJoinError(G50BError):
    """Source inventory join failed."""


class DerivationError(G50BError):
    """Evidence derivation law violated."""


class OrbitError(G50BError):
    """D4 orbit computation error."""


class ProposalError(G50BError):
    """Proposal compilation error."""


class OrderingError(G50BError):
    """Proposal ordering error."""


class ConflictError(G50BError):
    """Conflict evidence error."""


class VerificationError(G50BError):
    """Verifier found inconsistency."""
