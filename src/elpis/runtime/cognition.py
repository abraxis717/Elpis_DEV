"""The canonical cognitive turn: DSV4 codec -> ECS -> DSV4 codec (docs/ELPIS_MISSION.md).

::

    text --codec encode--> tokens
         --ECS codec map encode (UNQUALIFIED)--> Stimulus: ordered ECS_G drives (X, y)
         --ECS_G: one qualified atomic step per drive--> W_N -> W_N+k
         --readout: S3(W), the kernel's qualified coarse observable--> Readout
         --ECS codec map decode (UNQUALIFIED)--> tokens
         --codec decode--> text

The middle computation is ECS. The codec only crosses the token boundary:
``tokenizer`` supplies ``encode(text)``, ``decoder()`` and ``vocab_size`` (the
digest-bound V4.1 tokenizer, ``elpis.inference.text.V41Tokenizer``, in
production). No DSV model, parameter bank or transformer recurrence is
involved, and none is imported.

No ECS<->DSV semantic codec is defined or qualified in this repository:
neither how tokens perturb ECS state nor how ECS state is read out as tokens.
Without an explicitly supplied :class:`ECSCodecMap`, :func:`run_turn` refuses
with ``ECS_CODEC_UNQUALIFIED`` before touching anything: text generation is
unavailable. There is no fallback. A supplied map must declare its
classification, and every result carries it, so a fixture map
(``TRAINING=NONE SEMANTICS=NONE``) is never mistaken for cognition.

A turn is atomic: it runs on an independent fork of the ECS_G state and
installs the identical transition on the caller's state only after the
stimulus, the readout and the decode all succeeded. A refused turn leaves
the state and epoch unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from elpis.ECS_G.native import ECSGError, WorldState

from .composition import CompositionError

__all__ = ("CODEC_UNQUALIFIED", "ECSCodecMap", "Readout", "Stimulus", "TurnResult", "run_turn")

CODEC_UNQUALIFIED = "ECS_CODEC_UNQUALIFIED"
UNQUALIFIED_DETAIL = "ECS codec mapping not yet qualified; text generation unavailable"
_MAX_DRIVES, _MAX_ROWS, _MAX_TOKENS = 64, 256, 4096


@dataclass(frozen=True)
class Stimulus:
    """An ECS-facing perturbation in the kernel's only qualified input form.

    ``drives`` is an ordered tuple of ``(X, y)``: ``X`` a tuple of rows of
    ``dim`` finite floats, ``y`` a tuple of finite floats, one per row. Each
    drive is applied as one atomic ECS_G gradient step.
    """
    drives: tuple

    def __post_init__(self):
        if type(self.drives) is not tuple or not 1 <= len(self.drives) <= _MAX_DRIVES:
            raise CompositionError("STIMULUS", "1..64 drives")
        for drive in self.drives:
            if type(drive) is not tuple or len(drive) != 2:
                raise CompositionError("STIMULUS", "drive is (X, y)")
            x, y = drive
            if (type(x) is not tuple or type(y) is not tuple or not 1 <= len(y) <= _MAX_ROWS
                    or len(x) != len(y) or any(type(row) is not tuple for row in x)):
                raise CompositionError("STIMULUS", "X rows and y must match")
            values = [v for row in x for v in row] + list(y)
            if not all(type(v) is float and math.isfinite(v) for v in values):
                raise CompositionError("STIMULUS", "drive values must be finite floats")


@dataclass(frozen=True)
class Readout:
    """What ECS exposes to the decode boundary: S3(W) and the state it was read from."""
    s3: tuple
    epoch: int
    dim: int
    width: int


class ECSCodecMap(Protocol):
    """The missing ECS<->DSV semantic codec. No qualified implementation exists."""

    classification: str

    def encode(self, tokens: tuple) -> Stimulus: ...

    def decode(self, readout: Readout) -> tuple: ...


@dataclass(frozen=True)
class TurnResult:
    input_tokens: tuple
    output_tokens: tuple
    text: str
    readout: Readout
    epoch_before: int
    epoch_after: int
    codec: str          # the ECS codec map's declared classification


def _readout(state):
    return Readout(state.s3(), state.epoch, state.dim, state.width)


def run_turn(substrate, text, *, tokenizer, codec_map=None, learning_rate=None,
             max_output_tokens=256) -> TurnResult:
    """One canonical turn. Fails closed without a qualified ECS codec map."""
    if codec_map is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)
    classification = getattr(codec_map, "classification", None)
    if type(classification) is not str or not classification:
        raise CompositionError("CODEC_MAP", "an ECS codec map must declare its classification")
    if type(substrate) is not WorldState:
        raise CompositionError("ECS_STATE", "ECS_G WorldState required")
    if type(text) is not str:
        raise CompositionError("INPUT", "text must be str")
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise CompositionError("LEARNING_RATE", "explicit positive finite learning rate")
    if type(max_output_tokens) is not int or not 0 <= max_output_tokens <= _MAX_TOKENS:
        raise CompositionError("OUTPUT_LIMIT", "0..4096 output tokens")
    vocab = tokenizer.vocab_size
    tokens = tuple(tokenizer.encode(text))
    stimulus = codec_map.encode(tokens)
    if type(stimulus) is not Stimulus:
        raise CompositionError("STIMULUS", "the ECS codec map must return a Stimulus")
    epoch_before = substrate.epoch
    trial = substrate.fork()
    try:
        try:
            for x, y in stimulus.drives:
                trial.step(x, y, learning_rate)
        except ECSGError as exc:
            raise CompositionError("ECS_REFUSED", str(exc)) from exc
        readout = _readout(trial)
        output = codec_map.decode(readout)
        if (type(output) is not tuple or len(output) > max_output_tokens
                or not all(type(t) is int and 0 <= t < vocab for t in output)):
            raise CompositionError("DECODE", "decode must return in-vocabulary token IDs within the limit")
        decoder = tokenizer.decoder()
        rendered = "".join(decoder.push(t) for t in output) + decoder.finish()
        expected = trial.snapshot()
    finally:
        trial.close()
    # Install the identical, deterministic transition on the caller's state.
    for x, y in stimulus.drives:
        substrate.step(x, y, learning_rate)
    if substrate.snapshot() != expected:
        raise CompositionError("ECS_DIVERGED", "installed transition differs from the trial")
    return TurnResult(tokens, output, rendered, readout, epoch_before, substrate.epoch, classification)
