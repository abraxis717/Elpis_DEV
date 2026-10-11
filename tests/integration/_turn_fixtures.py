"""Interface fixtures for the cognitive operations. TEST_ONLY TRAINING=NONE SEMANTICS=NONE.

Stdlib only, so a clean process can run an operation without numpy or test oracles.
Neither class is a codec; they exist to exercise the boundary's mechanics. The fixture map produces the native-ready
inputs the operations consume (a LEARN experience schedule, QUERY rows); its numbers mean nothing.

A codec map never authorizes itself (elpis.runtime.codec_authority). These tests admit the fixture through
TEST_ONLY codec authorities built here: the test author is the trust root of those catalogs, which are never
deployment authority. ``TEST_CODEC_AUTHORITY`` lists the fixture variants the managed-runtime tests use; its pin is
the test runtimes' configured ``codec_authority_sha256``.
"""
from __future__ import annotations

from array import array
import codecs
import hashlib
import json

from elpis.runtime.codec_authority import (CODEC_AUTHORITY_SCHEMA, CodecAuthority, CodecCapability, admit_codec,
                                           implementation_identity, parameters_digest)
from elpis.runtime.cognition import LearnAuthority, QueryStimulus, Stimulus

DIM = 6
FIXTURE = "TEST_ONLY TRAINING=NONE SEMANTICS=NONE (deterministic interface fixture; not an ECS codec)"
RATE = 0.002
# The explicit learning authority every LEARN in these tests carries (TEST_ONLY: an interface fixture grant).
LEARN = LearnAuthority(RATE, "TEST_ONLY interface fixture grant; TRAINING=NONE SEMANTICS=NONE")
ALL_CAPABILITIES = tuple(c.value for c in CodecCapability)


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


MODES = (None, "not_a_stimulus", "wrong_dim", "not_a_query", "wrong_query_dim", "nonfinite_query")


class FixtureMap:
    """Deterministic interface fixture. Its numbers mean nothing.

    ``experiences`` ordered experiences of ``rows`` rows each, ``steps`` K1 steps each; ``poison_experience`` makes
    that experience's inputs overflow the K1 arithmetic (finite inputs, non-finite intermediates). ``mode`` selects
    a malformed output for refusal tests (part of the parameters, so it is admitted like any other parameter).
    """
    classification = FIXTURE

    def __init__(self, *, rows=4, experiences=2, steps=3, reply=b"ok", poison_experience=None, query_rows=3,
                 mode=None):
        assert mode in MODES, mode
        self.rows, self.experiences, self.steps = rows, experiences, steps
        self.reply, self.poison_experience = reply, poison_experience
        self.query_rows, self.mode = query_rows, mode
        self.calls = []

    def parameter_bytes(self):
        """Everything besides code that determines this map (the call log is not a parameter)."""
        return json.dumps({"rows": self.rows, "experiences": self.experiences, "steps": self.steps,
                           "reply": list(self.reply), "poison_experience": self.poison_experience,
                           "query_rows": self.query_rows, "mode": self.mode},
                          sort_keys=True, separators=(",", ":")).encode()

    def experiences_of(self, tokens):
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

    def encode(self, tokens):
        self.calls.append(("encode", tokens))
        if self.mode == "not_a_stimulus":
            return ((), ())
        if self.mode == "wrong_dim":
            return Stimulus(array("d", [0.1] * 5), array("d", [0.0]), array("Q", [1, 1]), dim=5)
        return self.experiences_of(tokens)

    def decode(self, readout):
        self.calls.append(("decode", readout))
        return tuple(self.reply)

    # QUERY side: query rows from the tokens (no target: a query carries none); the reply ignores the answer.
    def query_rows_of(self, tokens):
        n = max(1, len(tokens))
        x = array("d")
        for r in range(self.query_rows):
            x.extend(((tokens[(r * DIM + a) % n] % 17) - 8) / 16.0 if tokens else 0.25 for a in range(DIM))
        return x

    def encode_query(self, tokens):
        self.calls.append(("encode_query", tokens))
        if self.mode == "not_a_query":
            return self.experiences_of(tokens)            # a LEARN stimulus is not a query
        if self.mode == "wrong_query_dim":
            return QueryStimulus(array("d", [0.1] * 5), dim=5)
        if self.mode == "nonfinite_query":
            return QueryStimulus(array("d", [float("inf")] * DIM), dim=DIM)
        return QueryStimulus(self.query_rows_of(tokens), dim=DIM)

    def decode_query(self, readout):
        self.calls.append(("decode_query", readout))
        return tuple(self.reply)


def codec_id(fixture):
    return f"test-fixture:{type(fixture).__qualname__}:{parameters_digest(fixture)[:16]}"


def fixture_entry(fixture, *, capabilities=ALL_CAPABILITIES, status="ADMITTED", classification=FIXTURE,
                  implementation=None, parameters_sha256=None, identifier=None):
    name, source_sha = implementation_identity(type(fixture))
    return {"codec_id": identifier or codec_id(fixture), "implementation": implementation or name,
            "implementation_sha256": source_sha,
            "parameters_sha256": parameters_sha256 or parameters_digest(fixture),
            "capabilities": list(capabilities), "classification": classification, "qualification": "TEST_ONLY",
            "qualification_record": None, "status": status}


def catalog(entries, *, provenance="test-fixture"):
    """A TEST_ONLY codec authority over explicit entries, pinned by its author (the test): never deployment."""
    document = json.dumps({"schema": CODEC_AUTHORITY_SCHEMA, "source": "TEST_ONLY interface fixture catalog",
                           "provenance": provenance, "codecs": list(entries)},
                          sort_keys=True, separators=(",", ":")).encode()

    return CodecAuthority(document, expected_sha256=hashlib.sha256(document).hexdigest())


def fixture_authority(*fixtures, **entry):
    seen, entries = set(), []
    for f in fixtures:
        if codec_id(f) not in seen:
            seen.add(codec_id(f))
            entries.append(fixture_entry(f, **entry))
    return catalog(entries)


# Every fixture variant the managed-runtime tests run (their runtimes pin TEST_CODEC_AUTHORITY).
MANAGED_VARIANTS = (
    {}, {"experiences": 3, "steps": 4}, {"experiences": 1, "steps": 2, "rows": 4},
    {"experiences": 2, "steps": 3, "rows": 12}, {"reply": (999,)}, {"experiences": 1, "steps": 1},
    {"experiences": 8, "steps": 60}, {"reply": (300,)}, {"mode": "nonfinite_query"},
    {"experiences": 3, "poison_experience": 1},
)
TEST_CODEC_AUTHORITY = fixture_authority(*(FixtureMap(**v) for v in MANAGED_VARIANTS))
TEST_CODEC_PIN = TEST_CODEC_AUTHORITY.digest


def admitted(fixture, authority=None):
    """``fixture`` admitted with every capability: from TEST_CODEC_AUTHORITY when it lists the variant, else from
    a catalog of exactly this fixture (accepted unmanaged; a managed runtime pinned to TEST_CODEC_AUTHORITY refuses
    it)."""
    if authority is None:
        authority = TEST_CODEC_AUTHORITY if codec_id(fixture) in TEST_CODEC_AUTHORITY.codecs \
            else fixture_authority(fixture)
    return admit_codec(fixture, authority, codec_id(fixture))
