"""Digest-bound DeepSeek recipe V4.1 text boundary, outside model arithmetic.

Only text chat is supported. Tool/image templates are deliberately not inferred.
The exact recipe artifact is caller supplied; no download or file access occurs
after admission. See docs/inference/DSV41_TEXT_BOUNDARY.md for provenance.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass
import json
import os

from elpis.substrate.boundary import RootCapability, read_exact, stamp
from elpis.substrate.digests import raw_sha256
from elpis.identity import content_digest
from .contracts import Code, digest_value, integer, require

DSV41_RENDERER = "elpis.inference.render.deepseek-recipe-v41-text.v1"
RECIPE_SHA256 = "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
SYSTEM, USER, ASSISTANT = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>"
THINK, END_THINK = "<think>", "</think>"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    reasoning: str | None = None

    def __post_init__(self):
        require(self.role in ("system", "user", "assistant"), Code.UNSUPPORTED, "text chat role")
        require(type(self.content) is str, detail="chat content")
        require(self.reasoning is None or (self.role == "assistant" and type(self.reasoning) is str),
                detail="assistant reasoning")


class V41Tokenizer:
    """Admitted immutable token material and a local Rust BPE encoder.

    The artifact pin is independent of the candidate file. Identity binds both
    bytes and the adapter/engine semantics. Loading is a sequence-boundary action.
    """
    @classmethod
    def load(cls, root: RootCapability, path, *, expected_sha256: str):
        digest_value(expected_sha256)
        require(expected_sha256 == RECIPE_SHA256, Code.UNSUPPORTED, "unqualified tokenizer artifact")
        fd = root.open_file(path)
        try:
            before = os.fstat(fd)
            require(0 < before.st_size <= 8 << 20, Code.LIMIT, "tokenizer bytes")
            data = read_exact(fd, before.st_size, 0)
            require(stamp(before) == stamp(os.fstat(fd)), Code.IDENTITY, "tokenizer changed during read")
        finally:
            os.close(fd)
        return cls.from_bytes(data, expected_sha256=expected_sha256)

    @classmethod
    def from_bytes(cls, data: bytes, *, expected_sha256: str):
        require(type(data) is bytes and len(data) <= 8 << 20, Code.LIMIT, "tokenizer bytes")
        digest_value(expected_sha256)
        require(expected_sha256 == RECIPE_SHA256, Code.UNSUPPORTED, "unqualified tokenizer artifact")
        require(raw_sha256(data).hexdigest() == expected_sha256, Code.IDENTITY, "tokenizer SHA-256")
        import tokenizers
        require(tokenizers.__version__ == "0.23.2", Code.UNSUPPORTED, "qualified tokenizer engine version")
        artifact = json.loads(data)
        require(artifact["decoder"]["type"] == "ByteLevel", Code.UNSUPPORTED, "byte decoder")
        encoder = tokenizers.Tokenizer.from_str(data.decode("utf-8"))
        vocab = encoder.get_vocab()
        require(set(vocab.values()) == set(range(len(vocab))), Code.ENCODING, "contiguous vocabulary")
        added = {t["content"]: t["id"] for t in artifact["added_tokens"]}
        special = {t["content"]: t["id"] for t in artifact["added_tokens"] if t["special"]}
        for text in (BOS, EOS, SYSTEM, USER, ASSISTANT, THINK, END_THINK):
            require(text in added and encoder.encode(text, add_special_tokens=False).ids == [added[text]],
                    Code.IDENTITY, "special token identity")
        # ByteLevel's reversible alphabet: visible Latin-1 bytes retain their
        # codepoints; the remaining bytes use successive codepoints from 256.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_order = visible + [b for b in range(256) if b not in visible]
        chars = visible + list(range(256, 256 + 256 - len(visible)))
        alphabet = dict(zip(map(chr, chars), byte_order))
        material = [b""] * len(vocab)
        for text, token in vocab.items():
            material[token] = (bytes(alphabet[c] for c in text) if all(c in alphabet for c in text)
                               else text.encode("utf-8"))
        obj = object.__new__(cls)
        obj._encoder = encoder
        obj._material = tuple(material)
        obj._special = frozenset(special.values())
        obj._special_items = tuple(sorted(special.items()))
        obj._control = frozenset(added.values())
        obj._control_items = tuple(sorted(added.items()))
        obj.sha256 = expected_sha256
        obj.vocab_size = len(vocab)
        obj.bos_token = special[BOS]
        obj.stop_tokens = (special[EOS],)
        obj.identity = content_digest("elpis.inference.tokenizer.deepseek-recipe-v41.v1", dict(
            sha256=expected_sha256, engine="tokenizers-0.23.2", renderer=DSV41_RENDERER,
            decoder="bytelevel-utf8-replace.v1", vocab=obj.vocab_size, stops=obj.stop_tokens))
        return obj

    @property
    def special_tokens(self):
        return dict(self._special_items)

    @property
    def control_tokens(self):
        # Recipe chat markers are added tokens with special=false. They still
        # delimit roles; treating only special=true as controls permits injection.
        return dict(self._control_items)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text or a caller-rendered prompt; never insert BOS/EOS implicitly."""
        require(type(text) is str, detail="text input")
        return tuple(self._encoder.encode(text, add_special_tokens=False).ids)

    def encode_content(self, data: bytes) -> tuple[int, ...]:
        """HACF/plain content cannot introduce chat control tokens."""
        require(type(data) is bytes, detail="content bytes")
        tokens = self.encode(data.decode("utf-8", errors="strict"))
        require(not self._control.intersection(tokens), Code.ENCODING, "special token in plain content")
        return tokens

    def render_chat(self, messages: tuple[ChatMessage, ...], *, thinking=False, effort=None) -> str:
        require(type(messages) is tuple and bool(messages) and all(type(m) is ChatMessage for m in messages),
                detail="text chat messages")
        require(type(thinking) is bool and effort in (None, "low", "high", "xhigh", "max"),
                detail="thinking/effort")
        require(thinking or effort is None, detail="effort requires thinking")
        prompt, previous = BOS, None
        for index, msg in enumerate(messages):
            self.encode_content(msg.content.encode("utf-8"))
            if msg.reasoning is not None:
                self.encode_content(msg.reasoning.encode("utf-8"))
            if index == 0 and (thinking or msg.role == "system"):
                prompt += SYSTEM
            if index == 0 and thinking:
                score = {"low": 50, "max": 100}.get(effort, 75)
                prompt += (f"Reasoning Effort: {score} (range 1-100, the higher the value, "
                           "the more thorough the reasoning)\n\n")
            if msg.role == "system":
                prompt += (SYSTEM if index else "") + msg.content
            elif msg.role == "user":
                prompt += ("\n\n" if previous == "user" else USER) + msg.content
            else:
                reasoning = (THINK + (msg.reasoning or "") + END_THINK
                             if thinking and index > 0 else END_THINK)
                prompt += ASSISTANT + reasoning + msg.content + EOS
            previous = msg.role
        return prompt + ASSISTANT + (THINK if thinking else END_THINK)

    def encode_chat(self, messages, *, thinking=False, effort=None):
        return self.encode(self.render_chat(messages, thinking=thinking, effort=effort))

    def decoder(self, *, preserve_special=True):
        return IncrementalV41Decoder(self, preserve_special=preserve_special)


class IncrementalV41Decoder:
    """O(new token bytes), no prefix decode, lock, I/O, hashing or encoder calls.

    UTF-8 carry is at most three bytes. Invalid bytes and unfinished final
    characters use U+FFFD, matching ByteLevel's lossy UTF-8 reconstruction.
    EOS is consumed as a stop, never displayed; all other specials can be
    retained for a downstream output parser. Finish is idempotent.
    """
    def __init__(self, tokenizer: V41Tokenizer, *, preserve_special=True):
        require(type(tokenizer) is V41Tokenizer and type(preserve_special) is bool)
        self._material, self._special = tokenizer._material, tokenizer._special
        self._stops, self._preserve = tokenizer.stop_tokens, preserve_special
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.stopped = self._finished = False

    @property
    def pending_bytes(self):
        return len(self._utf8.getstate()[0])

    def push(self, token: int) -> str:
        require(not self._finished, detail="decoder finished")
        integer(token, 0, len(self._material) - 1)
        if token in self._stops:
            self.stopped = True
            return self.finish()
        if not self._preserve and token in self._special:
            return ""
        return self._utf8.decode(self._material[token], final=False)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._utf8.decode(b"", final=True)
