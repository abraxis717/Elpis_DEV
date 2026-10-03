"""Interface fixtures for the canonical turn. TRAINING=NONE SEMANTICS=NONE.

Stdlib only, so a clean process can run a turn without numpy or test oracles.
Neither class is a codec; they exist to exercise the boundary's mechanics.
"""
from __future__ import annotations

import codecs

from elpis.runtime.cognition import Stimulus

DIM = 6
FIXTURE = "TRAINING=NONE SEMANTICS=NONE (deterministic interface fixture; not an ECS codec)"


class ByteTokens:
    """Stand-in for the codec's token boundary: one token per UTF-8 byte. Not the V4.1 codec."""
    vocab_size = 256

    def encode(self, text):
        return tuple(text.encode("utf-8"))

    def decoder(self):
        inner = codecs.getincrementaldecoder("utf-8")()

        class Decoder:
            def push(self, token):
                return inner.decode(bytes((token,)))

            def finish(self):
                return inner.decode(b"", final=True)
        return Decoder()


class FixtureMap:
    """Deterministic interface fixture. Its numbers mean nothing."""
    classification = FIXTURE

    def __init__(self, *, rows=4, steps=2, reply=b"ok", poison_step=None):
        self.rows, self.steps, self.reply, self.poison_step = rows, steps, reply, poison_step
        self.calls = []

    def encode(self, tokens):
        self.calls.append(("encode", tokens))
        n = max(1, len(tokens))
        drives = []
        for k in range(self.steps):
            x = tuple(tuple(((tokens[(r * DIM + a + k) % n] % 17) - 8) / 16.0 if tokens else 0.25
                            for a in range(DIM)) for r in range(self.rows))
            if k == self.poison_step:
                x = tuple(tuple(v * 1e200 + 1e200 for v in row) for row in x)
            y = tuple(((tokens[(r + k) % n] % 7) - 3) / 8.0 if tokens else 0.0 for r in range(self.rows))
            drives.append((x, y))
        return Stimulus(tuple(drives))

    def decode(self, readout):
        self.calls.append(("decode", readout))
        return tuple(self.reply)
