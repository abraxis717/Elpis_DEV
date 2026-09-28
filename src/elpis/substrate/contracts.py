"""Typed fail-closed contract primitives shared by the substrate and inference.

Every contract violation raises :class:`ContractError` carrying one
:class:`Code`. Callers branch on the code, never on message text. There is no
silent fallback: an unmet precondition always raises.
"""
from enum import Enum


class Code(str, Enum):
    INVALID = 'INVALID'
    IDENTITY = 'IDENTITY'
    STALE = 'STALE'
    MISSING = 'MISSING'
    IO = 'IO'
    INTEGRITY = 'INTEGRITY'
    ENCODING = 'ENCODING'
    LIMIT = 'LIMIT'
    BUSY = 'BUSY'
    UNSUPPORTED = 'UNSUPPORTED'
    DEVICE = 'DEVICE'
    CLOSED = 'CLOSED'


class ContractError(ValueError):
    def __init__(self, code: Code, detail: str):
        self.code = code
        super().__init__(f'{code.value}:{detail}')


def require(condition, code=Code.INVALID, detail='contract'):
    if not condition:
        raise ContractError(code, detail)


def integer(value, low=0, high=(1 << 63) - 1):
    require(type(value) is int and low <= value <= high, detail='integer range')
    return value


def digest_value(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), detail='digest encoding')
    return value
