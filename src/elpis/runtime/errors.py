"""The runtime's one refusal type (shared by the composition, the canonical turn and the RuntimeCore adapter)."""
from __future__ import annotations

__all__ = ("CompositionError",)


class CompositionError(RuntimeError):
    """A composed operation was refused by one of its stages (fail closed).

    ``code`` is the stable refusal code (a RuntimeCore, continuity or boundary code); ``k1_status`` is the native
    K1 status behind an ``ECS_*`` refusal from RuntimeCore (0 otherwise).
    """

    def __init__(self, code: str, detail: str = "", *, k1_status: int = 0):
        self.code = code
        self.k1_status = k1_status
        super().__init__(f"{code}: {detail}" if detail else code)
