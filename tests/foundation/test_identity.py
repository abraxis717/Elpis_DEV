from dataclasses import dataclass
from enum import Enum
import hashlib
import json

import pytest

from elpis.identity import (
    CanonicalIdentityError,
    canonical_json_bytes,
    content_digest,
    domain_framed_bytes,
)


def test_non_ascii_vector():
    payload = {"label": "café", "note": "naïve"}
    expected = b'{"label":"caf\xc3\xa9","note":"na\xc3\xafve"}'
    assert canonical_json_bytes(payload) == expected
    framed = b"x.v1\x00" + expected
    assert domain_framed_bytes("x.v1", payload) == framed
    assert content_digest("x.v1", payload) == hashlib.sha256(framed).hexdigest()


def test_ascii_vector():
    payload = {"a": 1}
    framed = b'x.v1\x00{"a":1}'
    assert domain_framed_bytes("x.v1", payload) == framed


def test_reserialization_invariance():
    left = json.loads('{"note":"na\\u00efve","label":"caf\\u00e9"}')
    right = json.loads(' { "label" : "café", "note" : "naïve" } ')
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_digest("x.v1", left) == content_digest("x.v1", right)


class Mode(Enum):
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class Record:
    mode: Mode
    raw: bytes
    pair: tuple[int, int]


def test_normalization_contract():
    value = Record(Mode.ACTIVE, b"\x00\xff", (2, 1))
    assert canonical_json_bytes(value) == (
        b'{"mode":"ACTIVE","pair":[2,1],"raw":{"__bytes__":"00ff"}}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(value):
    with pytest.raises(CanonicalIdentityError, match="NONFINITE"):
        canonical_json_bytes({"x": value})


def test_non_string_mapping_key_rejected():
    with pytest.raises(CanonicalIdentityError, match="MAP_KEY"):
        canonical_json_bytes({1: "ambiguous"})


@pytest.mark.parametrize("domain", ["", "a\x00b", 3, None])
def test_invalid_domain_rejected(domain):
    with pytest.raises(CanonicalIdentityError):
        content_digest(domain, {"a": 1})

def test_bytes_type_sentinel_cannot_alias_ordinary_mapping():
    assert canonical_json_bytes(b"\x00") == b'{"__bytes__":"00"}'
    with pytest.raises(CanonicalIdentityError, match="RESERVED_MAP_KEY"):
        canonical_json_bytes({"__bytes__": "00"})
