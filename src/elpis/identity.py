"""Canonical content identity v1, shared by every Elpis subsystem.

Identities are ``SHA256(UTF8(domain) || NUL || canonical_json_bytes(payload))``.
The domain string is part of the persisted identity: changing a domain changes
every digest derived from it, so domains are protocol identifiers and are never
renamed for cosmetic reasons.

A digest is content identity relative to trusted starting authority. It is not
a signature and does not authenticate an issuer.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any


class CanonicalIdentityError(ValueError):
    pass


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise CanonicalIdentityError("CANONICAL_MAP_KEY_NOT_STRING")
        # "__bytes__" is the v1 type sentinel used to encode raw bytes. Ordinary
        # mappings may not claim that key, otherwise raw bytes and an ordinary
        # mapping can canonicalize to identical JSON bytes.
        if "__bytes__" in value:
            raise CanonicalIdentityError("CANONICAL_RESERVED_MAP_KEY:__bytes__")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalIdentityError("CANONICAL_NONFINITE_FLOAT")
        return value
    raise CanonicalIdentityError(
        f"CANONICAL_UNSUPPORTED_TYPE:{type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except CanonicalIdentityError:
        raise
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise CanonicalIdentityError(
            f"CANONICAL_SERIALIZATION_FAILED:{exc}"
        ) from exc


def domain_framed_bytes(domain: str, value: Any) -> bytes:
    if type(domain) is not str or not domain:
        raise CanonicalIdentityError("CANONICAL_DOMAIN_INVALID")
    if "\x00" in domain:
        raise CanonicalIdentityError("CANONICAL_DOMAIN_CONTAINS_NUL")
    return domain.encode("utf-8") + b"\x00" + canonical_json_bytes(value)


def content_digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain_framed_bytes(domain, value)).hexdigest()
