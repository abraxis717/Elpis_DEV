"""Differential properties of the streaming lexer against the bounded lexer.

The bounded whole-input lexer is the oracle. The long-match oracle transforms
only the documented streaming evidence representation for matches longer than
4096 bytes; lexical content, offsets, payloads and digests must stay equal.
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from elpis.pipeline.ingress import IngressError, IngressLibrary, lex, lex_stream

from ...conftest import require_native_library
from ...structure.native_bridge_fixture import pin_bridge

PHRASES = [
    "at least +1.25", "strictly greater than -2.5", "at most .5", "strictly less than 2",
    "not equal to 7", "exactly equal to 4", "keep value between low and high",
    "low is the lower bound", "high as upper limit", "touching endpoints do not merge",
    "strictly overlapping", "touching ranges may also merge", "maximum ending coordinate", "minimum right edge",
]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    # TEST-ONLY pin of an isolated native fixture; never deployment authority.
    path, root, authority = pin_bridge(
        require_native_library("elpis_ingress_bridge"),
        tmp_path_factory.mktemp("sealed-regex-ingress"), "elpis_ingress_bridge")
    return IngressLibrary(path, root=root, authority=authority)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def bounded(library, source: bytes):
    try:
        return True, lex(library, source, chunk_size=len(source) + 1,
                         carry_bytes=max(256, len(source))).result_json
    except IngressError:
        return False, None


def streamed(library, source: bytes, sizes):
    chunks, offset = [], 0
    for size in sizes:
        if offset >= len(source):
            break
        chunks.append(source[offset:offset + size])
        offset += size
    if offset < len(source):
        chunks.append(source[offset:])
    try:
        return True, lex_stream(library, chunks).result_json
    except IngressError:
        return False, None


def long_representation(raw: bytes) -> bytes:
    value = json.loads(raw)
    omitted = False
    for evidence in value["ingress"]["evidence"]:
        if len(evidence["matched_text"].encode()) > 4096:
            omitted = True
            del evidence["matched_text"]
            del evidence["evidence_id"]
            evidence["matched_text_omitted"] = True
            evidence["schema"] = "elpis.regex-lexical-evidence.v2"
            evidence["evidence_id"] = hashlib.sha256(_canonical(evidence)).hexdigest()
    if omitted:
        value["ingress"]["schema"] = "elpis.regex-stream-ingress-result.v2"
    return _canonical(value)


@pytest.mark.parametrize("length", [4095, 4096, 4097, 16384])
def test_inline_to_omitted_transition(library, length):
    source = b"at" + b" " * (length - 9) + b"least 1"
    ok, oracle = bounded(library, source)
    assert ok
    for chunk in (1, 2, 17, len(source)):
        assert streamed(library, source, [chunk] * (len(source) // chunk + 1)) == (
            True, long_representation(oracle))


@pytest.mark.parametrize("phrase", PHRASES)
def test_long_whitespace_families_are_hashed_in_order(library, phrase):
    source = phrase.replace(" ", " \t\r\n" * 1300).encode()
    ok, oracle = bounded(library, source)
    assert ok
    expected = long_representation(oracle)
    for chunk in (17, 4093, len(source)):
        assert streamed(library, source, [chunk] * (len(source) // chunk + 1)) == (True, expected)


def test_sparse_multi_megabyte_input_keeps_bounded_identity(library):
    source = b"; ".join(p.encode() for p in PHRASES)
    sparse = b"z" * (2 * 1024 * 1024) + b"; " + source
    expected = bounded(library, sparse)
    assert expected[0]
    assert streamed(library, sparse, [65521] * 34) == expected


def test_seeded_mutation_corpus(library):
    rng = random.Random(0xE1F152)
    alphabet = [";", " ", "\t", " ", " ", "́", "K", "ſ", "π", "😀",
                "0", ".", "+", "-", "_", "X", "\0"]
    for i in range(300):
        text = rng.choice(PHRASES) + rng.choice(alphabet) + rng.choice(PHRASES)
        for _ in range(rng.randrange(1, 5)):
            pos = rng.randrange(len(text) + 1)
            text = text[:pos] + rng.choice(alphabet) + text[pos + rng.randrange(2):]
        source = text.encode()
        if i % 7 == 0:
            pos = rng.randrange(len(source) + 1)
            source = source[:pos] + bytes([rng.randrange(128, 256)]) + source[pos:]
        expected = bounded(library, source)
        sizes = [rng.randrange(1, 24) for _ in range(len(source))]
        for chunks in ([1] * len(source), sizes, [len(source)]):
            got = streamed(library, source, chunks)
            assert got[0] == expected[0], (i, source)
            if got[0]:
                assert got == expected, (i, source)
                result = json.loads(got[1])
                assert result["ingress"]["source_sha256"] == hashlib.sha256(source).hexdigest()
                assert result["ingress"]["source_bytes"] == len(source)


@pytest.mark.parametrize("source", [b"at lea " * 4096, b"touching endpoints do not " * 1024,
                                    b"at least " + b"9" * 65536, b"touching endpoints " + b"x " * 8192])
def test_pathological_prefixes_match_bounded_lexer(library, source):
    assert streamed(library, source, [8191] * (len(source) // 8191 + 1)) == bounded(library, source)
