"""Error classes and failure codes for adjudication."""


class AdjudicationError(Exception):
    """Base adjudication error."""
    def __init__(self, message, failure_code=None):
        super().__init__(message)
        self.failure_code = failure_code


# --- Source join failures ---

class SourceJoinMissingRow(AdjudicationError):
    def __init__(self, message="Source row missing from join"):
        super().__init__(message, "SOURCE_JOIN_MISSING_ROW")


class ProposalSetIncomplete(AdjudicationError):
    def __init__(self, message="Proposal set incomplete"):
        super().__init__(message, "PROPOSAL_SET_INCOMPLETE")


class ProposalSetDuplicate(AdjudicationError):
    def __init__(self, message="Duplicate proposal in envelope"):
        super().__init__(message, "PROPOSAL_SET_DUPLICATE")
