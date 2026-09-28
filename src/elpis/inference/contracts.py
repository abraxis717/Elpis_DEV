"""Shared, explicitly non-authoritative inference contracts.

Typed failures (:class:`Code`, :class:`ContractError`) and the integer and
digest checks come from :mod:`elpis.substrate.contracts`; identities of
inference records use the persisted ``elpis.inference.<kind>.r0`` domains of
:func:`elpis.substrate.digests.identity`.
"""
from dataclasses import dataclass

from elpis.substrate.contracts import Code, ContractError, digest_value, integer, require
from elpis.substrate.digests import identity, raw_digest

__all__ = (
    "Bank", "Code", "ContractError", "ProposalOnly", "RowIdentity",
    "digest_value", "identity", "integer", "raw_digest", "require",
)


class ProposalOnly:
    """Fixed properties, never caller-controlled admission flags."""
    __slots__ = ()
    semantic_authority = property(lambda self: False)
    admission_authority = property(lambda self: False)
    execution_authority = property(lambda self: False)
    mutation_authority = property(lambda self: False)
    runtime_admission = property(lambda self: False)


@dataclass(frozen=True)
class Bank:
    name: str
    model: str
    tokenizer: str
    scheme: str
    parameters: str
    layer: int
    rows: int
    dimension: int
    content: str
    representation: str

    def __post_init__(self):
        require(bool(self.name) and bool(self.model) and bool(self.tokenizer))
        integer(self.layer)
        integer(self.rows, 1)
        integer(self.dimension, 1)
        for d in (self.parameters, self.content, self.representation):
            digest_value(d)

    @property
    def digest(self):
        return identity('bank', self)


@dataclass(frozen=True)
class RowIdentity:
    bank: str
    row: int

    def __post_init__(self):
        digest_value(self.bank)
        integer(self.row)
