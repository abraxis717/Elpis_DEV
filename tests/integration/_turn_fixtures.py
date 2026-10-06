"""Interface fixtures for the canonical turn. TRAINING=NONE SEMANTICS=NONE.

Stdlib only, so a clean process can run a turn without numpy or test oracles.
Neither class is a codec; they exist to exercise the boundary's mechanics. The fixture map produces the native-ready
Stimulus the canonical turn consumes (contiguous x, y and a (rows, steps) schedule); its numbers mean nothing.
"""
from __future__ import annotations

from array import array
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
    """Deterministic interface fixture. Its numbers mean nothing.

    ``experiences`` ordered experiences of ``rows`` rows each, ``steps`` K1 steps each; ``poison_experience`` makes
    that experience's inputs overflow the K1 arithmetic (finite inputs, non-finite intermediates)."""
    classification = FIXTURE

    def __init__(self, *, rows=4, experiences=2, steps=3, reply=b"ok", poison_experience=None):
        self.rows, self.experiences, self.steps = rows, experiences, steps
        self.reply, self.poison_experience = reply, poison_experience
        self.calls = []

    def encode(self, tokens):
        self.calls.append(("encode", tokens))
        n = max(1, len(tokens))
        x, y, schedule = array("d"), array("d"), array("Q")
        for k in range(self.experiences):   # the fixture plays the (unqualified) codec map, outside the ECS
            for r in range(self.rows):
                row = [((tokens[(r * DIM + a + k) % n] % 17) - 8) / 16.0 if tokens else 0.25 for a in range(DIM)]
                if k == self.poison_experience:
                    row = [v * 1e150 + 1e150 for v in row]
                x.extend(row)
                y.append(((tokens[(r + k) % n] % 7) - 3) / 8.0 if tokens else 0.0)
            schedule.extend((self.rows, self.steps))
        return Stimulus(x, y, schedule, dim=DIM)

    def decode(self, readout):
        self.calls.append(("decode", readout))
        return tuple(self.reply)
