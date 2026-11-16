"""A reversible byte-level tokenizer for the standalone prototype.

Token ids 0..255 are the bytes themselves; three control ids follow. ``decode_bytes(encode_bytes(b)) == b``
for every byte string, so nothing is lost and nothing is learned: there is no vocabulary file and no merge
table. It is not the DSV4 codec (``elpis.inference.text``) and claims no compatibility with any pretrained
model's tokenizer. A future external tokenizer only has to satisfy ``TokenizerProtocol``; the model takes
``vocab_size`` from its configuration, never from this class.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ByteTokenizer:
    BOS = 256
    EOS = 257
    PAD = 258
    vocab_size = 259

    def encode_bytes(self, data: bytes, *, bos: bool = False, eos: bool = False) -> list[int]:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("encode_bytes expects bytes")
        return ([self.BOS] if bos else []) + list(data) + ([self.EOS] if eos else [])

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """The bytes of a token sequence. Control ids carry no bytes; anything else out of range is an error."""
        out = bytearray()
        for token in ids:
            token = int(token)
            if 0 <= token < 256:
                out.append(token)
            elif token not in (self.BOS, self.EOS, self.PAD):
                raise ValueError(f"token id {token} is outside the byte vocabulary")
        return bytes(out)

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"), bos=bos, eos=eos)

    def decode(self, ids: Sequence[int]) -> str:
        """Text for display. A generated sequence may end inside a UTF-8 character; that tail is replaced."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
