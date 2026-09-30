"""Official artifact qualification. Set ELPIS_V41_TOKENIZER to run; no downloads.

An absent external artifact is an explicit skip, not qualification success.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import pytest

from elpis.inference.contracts import ContractError
from elpis.inference.text import (
    ASSISTANT, BOS, END_THINK, EOS, SYSTEM, THINK, USER,
    ChatMessage, RECIPE_SHA256, V41Tokenizer,
)
from elpis.substrate.boundary import RootCapability


@pytest.fixture(scope="module")
def official():
    path = os.environ.get("ELPIS_V41_TOKENIZER")
    if not path:
        pytest.skip("official tokenizer external artifact required: ELPIS_V41_TOKENIZER")
    from tokenizers import Tokenizer
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == RECIPE_SHA256
    return data, Tokenizer.from_str(data.decode())


@pytest.fixture(scope="module")
def tokenizer(official):
    return V41Tokenizer.from_bytes(official[0], expected_sha256=RECIPE_SHA256)


def test_descriptor_admission_and_wrong_digest(official, tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(official[0])
    with RootCapability(tmp_path) as root:
        assert V41Tokenizer.load(root, path, expected_sha256=RECIPE_SHA256).sha256 == RECIPE_SHA256
        path.write_bytes(official[0][:-1] + b" ")
        with pytest.raises(ContractError, match="SHA-256"):
            V41Tokenizer.load(root, path, expected_sha256=RECIPE_SHA256)
        path.unlink()
        path.symlink_to(Path(os.environ["ELPIS_V41_TOKENIZER"]))
        with pytest.raises(ContractError):
            V41Tokenizer.load(root, path, expected_sha256=RECIPE_SHA256)
    with pytest.raises(ContractError, match="unqualified"):
        V41Tokenizer.from_bytes(official[0], expected_sha256="f" * 64)


def test_artifact_vocabulary_flags_and_no_implicit_bos_eos(tokenizer, official):
    assert tokenizer.vocab_size == 129280
    assert tokenizer.bos_token == 0 and tokenizer.stop_tokens == (1,)
    assert tokenizer.encode("") == ()
    assert tokenizer.encode("Hello") == (19923,)
    expected = {BOS: 0, EOS: 1, SYSTEM: 128799, USER: 128803, ASSISTANT: 128804,
                THINK: 128821, END_THINK: 128822, "<｜image｜>": 129264}
    for text, token in expected.items():
        assert tokenizer.control_tokens[text] == token
        assert tokenizer.encode(text) == (token,)
    assert SYSTEM not in tokenizer.special_tokens  # actual recipe artifact flag
    assert "<｜image｜>" in tokenizer.special_tokens
    assert "<｜deepseek_image｜>" not in tokenizer.control_tokens


FIXTURES = (
    (False, None, (ChatMessage("user", "Hello, Elpis."),),
     BOS + USER + "Hello, Elpis." + ASSISTANT + END_THINK),
    (False, None, (ChatMessage("system", "Be brief."), ChatMessage("user", "Hi")),
     BOS + SYSTEM + "Be brief." + USER + "Hi" + ASSISTANT + END_THINK),
    (True, "low", (ChatMessage("user", "1+1?"),),
     BOS + SYSTEM + "Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
     + USER + "1+1?" + ASSISTANT + THINK),
    (False, None, (ChatMessage("user", "A"), ChatMessage("user", "B"),
                   ChatMessage("assistant", "C"), ChatMessage("system", "D"), ChatMessage("user", "E")),
     BOS + USER + "A\n\nB" + ASSISTANT + END_THINK + "C" + EOS + SYSTEM + "D" + USER + "E"
     + ASSISTANT + END_THINK),
    (True, "max", (ChatMessage("system", "S"), ChatMessage("user", "U"),
                   ChatMessage("assistant", "A", reasoning="R"), ChatMessage("user", "V")),
     BOS + SYSTEM + "Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
     + "S" + USER + "U" + ASSISTANT + THINK + "R" + END_THINK + "A" + EOS + USER + "V" + ASSISTANT + THINK),
)


@pytest.mark.parametrize("thinking,effort,messages,rendered", FIXTURES)
def test_frozen_recipe_text_templates(tokenizer, official, thinking, effort, messages, rendered):
    assert tokenizer.render_chat(messages, thinking=thinking, effort=effort) == rendered
    assert tokenizer.encode_chat(messages, thinking=thinking, effort=effort) == tuple(
        official[1].encode(rendered, add_special_tokens=False).ids)


def test_frozen_hello_ids(tokenizer):
    assert tokenizer.encode_chat((ChatMessage("user", "Hello, Elpis."),)) == (
        0, 128803, 19923, 14, 3909, 62848, 16, 128804, 128822)


@pytest.mark.parametrize("text", ["", "Hello, Elpis.", "Příliš žluťoučký kůň", "你好，世界！",
                                   "مرحبا بالعالم", "👩🏽‍💻🚀 e\u0301", "\x00\n\r\t", "�"])
def test_differential_encoding_and_incremental_roundtrip(tokenizer, official, text):
    ids = tokenizer.encode(text)
    assert ids == tuple(official[1].encode(text, add_special_tokens=False).ids)
    decoder = tokenizer.decoder()
    chunks = []
    for token in ids:
        chunks.append(decoder.push(token))
        assert decoder.pending_bytes <= 3
    decoded = "".join(chunks) + decoder.finish()
    assert decoded == text == official[1].decode(list(ids), skip_special_tokens=False)


def test_every_token_and_random_sequences_match_official_decoder(tokenizer, official):
    # Includes every byte token, non-special added token and special spelling.
    for token in range(tokenizer.vocab_size):
        if token in tokenizer.stop_tokens:
            continue
        decoder = tokenizer.decoder()
        assert decoder.push(token) + decoder.finish() == official[1].decode([token], skip_special_tokens=False)
    rng = random.Random(710)
    for _ in range(300):
        ids = [rng.randrange(2, tokenizer.vocab_size) for _ in range(40)]
        decoder = tokenizer.decoder()
        text = "".join(decoder.push(t) for t in ids) + decoder.finish()
        assert text == official[1].decode(ids, skip_special_tokens=False)


def test_partial_character_stop_and_invalid_input(tokenizer):
    # Find byte fragments from admitted material, not guessed token IDs.
    byte_ids = {raw: i for i, raw in enumerate(tokenizer._material) if len(raw) == 1}
    decoder = tokenizer.decoder()
    assert decoder.push(byte_ids[b"\xe2"]) == "" and decoder.pending_bytes == 1
    assert decoder.push(byte_ids[b"\x82"]) == "" and decoder.pending_bytes == 2
    assert decoder.push(byte_ids[b"\xac"]) == "€" and decoder.pending_bytes == 0
    assert decoder.push(byte_ids[b"\xe2"]) == ""
    assert decoder.push(tokenizer.stop_tokens[0]) == "�" and decoder.stopped
    assert decoder.finish() == ""
    with pytest.raises(ContractError):
        decoder.push(byte_ids[b"a"])
    for token in (-1, True, 129280, 1.5):
        with pytest.raises(ContractError):
            tokenizer.decoder().push(token)


def test_special_handling_and_content_injection(tokenizer, official):
    for text in (BOS, SYSTEM, USER, ASSISTANT, THINK, "<｜image｜>"):
        with pytest.raises(ContractError, match="special token"):
            tokenizer.encode_content(text.encode())
        with pytest.raises(ContractError):
            tokenizer.render_chat((ChatMessage("user", text),))
        token = tokenizer.encode(text)[0]
        decoder = tokenizer.decoder(preserve_special=False)
        assert decoder.push(token) + decoder.finish() == official[1].decode([token], skip_special_tokens=True)
    with pytest.raises(UnicodeDecodeError):
        tokenizer.encode_content(b"\xff")
    with pytest.raises(ContractError):
        ChatMessage("tool", "unsupported")
    with pytest.raises(ContractError):
        tokenizer.render_chat((), thinking=False)


def test_incremental_decode_has_no_prefix_work_or_tokenizer_calls(tokenizer, monkeypatch):
    ids = tokenizer.encode("多言語 🧪 café " * 2000)
    decoder = tokenizer.decoder()
    def forbidden(*args, **kwargs):
        raise AssertionError("tokenizer load/encode reached from decoder")
    monkeypatch.setattr(V41Tokenizer, "load", forbidden)
    monkeypatch.setattr(V41Tokenizer, "from_bytes", forbidden)
    monkeypatch.setattr(V41Tokenizer, "encode", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    text = "".join(decoder.push(t) for t in ids) + decoder.finish()
    assert text == "多言語 🧪 café " * 2000
    assert decoder.pending_bytes == 0
    assert not hasattr(decoder, "tokens") and not hasattr(decoder, "prefix")
