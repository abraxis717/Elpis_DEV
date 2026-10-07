"""The canonical cognitive turn: DSV4 codec -> ECS -> DSV4 codec (docs/ELPIS_MISSION.md).

::

    text --codec encode--> tokens
         --ECS codec map encode (UNQUALIFIED)--> Stimulus: a native-ready ordered experience schedule
         --native K1 candidate: per experience, its K1 learning steps, then the consolidation of its rows-->
           (W, epoch, H, a) candidate
         --readout: S3 of the candidate W, the kernel's qualified coarse observable, computed natively--> Readout
         --ECS codec map decode (UNQUALIFIED)--> tokens
         --codec decode--> text
         --atomic native commit of the complete (W, epoch, H, a) candidate

The ECS substrate of the canonical turn is native K1 (docs/ECS_K1_RUNTIME.md), the retained-state mechanism
Retention R3 qualified and K1N-v2 qualified natively: a standalone :class:`~elpis.ECS_G.k1.K1State` or an
FMS-resident :class:`~elpis.ECS_G.k1.K1FMSState`. Each experience is the qualified K1 transition, learn then
consolidate (``H <- H + Sigma(X_t)``, ``a <- S3(W_t)``; inputs only). The Runtime R1 ``Executor`` remains a
qualified primitive and the K1-disabled reference, but it is not the substrate of this turn.

The codec only crosses the token boundary: ``tokenizer`` supplies ``encode(text)``, ``decoder()`` and
``vocab_size`` (the digest-bound V4.1 tokenizer, ``elpis.inference.text.V41Tokenizer``, in production). No DSV
model, parameter bank or transformer recurrence is involved, and none is imported.

No ECS<->DSV semantic codec is defined or qualified in this repository: neither how tokens perturb ECS state nor
how ECS state is read out as tokens. Without an explicitly supplied :class:`ECSCodecMap`, :func:`run_turn` refuses
with ``ECS_CODEC_UNQUALIFIED`` before touching anything: text generation is unavailable. There is no fallback. A
supplied map must declare its classification, and every result carries it, so a fixture map
(``TRAINING=NONE SEMANTICS=NONE``) is never mistaken for cognition.

PYTHON MAY CONTROL THE ECS; IT MAY NOT EXECUTE THE ECS HOT PATH. A turn is three native crossings whatever the
number of experiences, rows or K1 steps: begin a transaction; run the whole experience schedule on the candidate
and read S3 of the candidate W (one call); commit or abort. The schedule crosses as contiguous buffers; nothing in
this module walks an ECS scalar, a row, a feature or a step. The decode happens between the schedule and the
commit; the candidate commits only after the stimulus, the readout and the decode all succeeded, and the commit is
refused as ``ECS_STALE`` if another commit replaced the source state meanwhile (a native generation check). A
refused turn leaves ``(W, epoch, H, a)`` byte-for-byte unchanged. Admitting more rows per experience than the
state's capacity grows it explicitly before the transaction begins (cold path).

This module performs no durable writes. The runtime composition publishes the committed transition's
retained-state identity to ``elpis.continuity`` (docs/CONTINUITY.md).
"""
from __future__ import annotations
from array import array
from dataclasses import dataclass, field
import math
from typing import Protocol

from elpis.ECS_G.k1 import MAX_EXPERIENCES, K1Error, K1FMSState, K1State

from .composition import CompositionError

__all__ = ("CODEC_UNQUALIFIED", "ECSCodecMap", "Readout", "Stimulus", "TurnResult", "run_turn")

CODEC_UNQUALIFIED = "ECS_CODEC_UNQUALIFIED"
UNQUALIFIED_DETAIL = "ECS codec mapping not yet qualified; text generation unavailable"
_MAX_ROWS, _MAX_STEPS, _MAX_TOKENS, _MAX_DIM = 256, 1 << 20, 4096, 64
_SUBSTRATES = (K1State, K1FMSState)


def _owned(values, fmt, what):
    """An owned contiguous copy of a buffer of ``fmt`` (one C-level copy; never walked in Python)."""
    try:
        view = memoryview(values)
    except TypeError as exc:
        raise CompositionError("STIMULUS", what + ": a contiguous buffer is required") from exc
    if view.format.lstrip("<=@") != fmt or view.itemsize != 8 or view.ndim != 1 or not view.c_contiguous:
        raise CompositionError("STIMULUS", f"{what}: a one-dimensional contiguous '{fmt}' buffer is required")
    out = array(fmt)
    out.frombytes(view.cast("B"))
    return out


@dataclass(frozen=True, init=False)
class Stimulus:
    """An admitted, native-ready ECS stimulus: an ordered experience schedule in the kernel's only qualified input
    form.

    * ``x``: every experience's input rows in order, row-major binary64 (``rows x dim`` values);
    * ``y``: their targets, binary64 (``rows`` values);
    * ``schedule``: uint64 ``(rows, steps)`` per experience, 1..64 experiences, 1..256 rows and 1..2^20 K1 steps
      each; experience t uses the next ``rows`` rows of ``x`` and ``y``.

    Admission validates shapes and bounds over the bounded schedule metadata and takes owned contiguous copies;
    finiteness is checked natively before the candidate is touched. Each experience is learned (its K1 steps) and
    then consolidated (its rows) natively; the turn commits all of them or none.
    """
    x: array
    y: array
    schedule: array
    dim: int
    experiences: int
    rows: int
    max_experience_rows: int
    steps: int

    def __init__(self, x, y, schedule, *, dim):
        if type(dim) is not int or not 1 <= dim <= _MAX_DIM:
            raise CompositionError("STIMULUS", "dim: 1..64")
        xs, ys, sched = _owned(x, "d", "x"), _owned(y, "d", "y"), _owned(schedule, "Q", "schedule")
        n = len(sched) // 2
        if len(sched) != 2 * n or not 1 <= n <= MAX_EXPERIENCES:
            raise CompositionError("STIMULUS", "1..64 experiences as (rows, steps) pairs")
        rows, steps = sched[0::2], sched[1::2]   # bounded metadata: at most 64 entries each
        if min(rows) < 1 or max(rows) > _MAX_ROWS or min(steps) < 1 or max(steps) > _MAX_STEPS:
            raise CompositionError("STIMULUS", "every experience: 1..256 rows and 1..2^20 K1 steps")
        total = sum(rows)
        if len(ys) != total or len(xs) != total * dim:
            raise CompositionError("STIMULUS", "x and y must hold exactly the scheduled rows")
        for name, value in (("x", xs), ("y", ys), ("schedule", sched), ("dim", dim), ("experiences", n),
                            ("rows", total), ("max_experience_rows", max(rows)), ("steps", sum(steps))):
            object.__setattr__(self, name, value)

    @classmethod
    def from_experiences(cls, experiences, *, dim):
        """Cold-path convenience (tests, tooling): ``((X rows, y, steps), ...)`` packed once into the native-ready
        form. Not used by :func:`run_turn`, which consumes an admitted Stimulus only."""
        if type(experiences) is not tuple or not 1 <= len(experiences) <= MAX_EXPERIENCES:
            raise CompositionError("STIMULUS", "1..64 experiences")
        x, y, schedule = array("d"), array("d"), array("Q")
        for item in experiences:
            if type(item) is not tuple or len(item) != 3:
                raise CompositionError("STIMULUS", "experience is (X rows, y, steps)")
            rows, targets, steps = item
            if (type(rows) is not tuple or type(targets) is not tuple or len(rows) != len(targets)
                    or type(steps) is not int or steps < 1 or any(len(r) != dim for r in rows)):
                raise CompositionError("STIMULUS", "X rows of dim values, one target per row, steps >= 1")
            try:
                for row in rows:
                    x.extend(row)
                y.extend(targets)
            except TypeError as exc:
                raise CompositionError("STIMULUS", "experience values must be floats") from exc
            schedule.extend((len(rows), steps))
        return cls(x, y, schedule, dim=dim)


@dataclass(frozen=True)
class Readout:
    """What ECS exposes to the decode boundary: S3(W) of the candidate and the state it was read from."""
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
    # Exact K1 retained-state identities (state_digest) of the committed transition, from the native commit.
    state_before_digest: bytes | None = field(default=None, compare=False)
    state_after_digest: bytes | None = field(default=None, compare=False)


def _validate_turn_request(substrate, text, *, codec_map=None, learning_rate=None,
                           max_output_tokens=256):
    """Validate the canonical turn request without reading or mutating K1 state."""
    if codec_map is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)

    classification = getattr(codec_map, "classification", None)

    if type(classification) is not str or not classification:
        raise CompositionError(
            "CODEC_MAP",
            "an ECS codec map must declare its classification",
        )

    if type(substrate) not in _SUBSTRATES:
        raise CompositionError(
            "ECS_STATE",
            "a native K1 state (K1State or K1FMSState) is required",
        )

    if type(text) is not str:
        raise CompositionError(
            "INPUT",
            "text must be str",
        )

    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise CompositionError(
            "LEARNING_RATE",
            "explicit positive finite learning rate",
        )

    if (
        type(max_output_tokens) is not int
        or not 0 <= max_output_tokens <= _MAX_TOKENS
    ):
        raise CompositionError(
            "OUTPUT_LIMIT",
            "0..4096 output tokens",
        )

    return classification


def run_turn(substrate, text, *, tokenizer, codec_map=None, learning_rate=None,
             max_output_tokens=256) -> TurnResult:
    """One canonical turn over a native K1 state. Fails closed without a qualified ECS codec map."""
    classification = _validate_turn_request(
        substrate,
        text,
        codec_map=codec_map,
        learning_rate=learning_rate,
        max_output_tokens=max_output_tokens,
    )
    vocab = tokenizer.vocab_size
    tokens = tuple(tokenizer.encode(text))
    stimulus = codec_map.encode(tokens)
    if type(stimulus) is not Stimulus:
        raise CompositionError("STIMULUS", "the ECS codec map must return a Stimulus")
    dim, width = substrate.dim, substrate.width
    if stimulus.dim != dim:
        raise CompositionError("STIMULUS", "the stimulus dimension must match the ECS state")
    try:
        if stimulus.max_experience_rows > substrate.max_rows:
            substrate.reserve(stimulus.max_experience_rows)  # explicit cold-path growth, before the transaction
        txn = substrate.transaction()
    except K1Error as exc:
        raise CompositionError("ECS_REFUSED", str(exc)) from exc
    with txn:  # aborted unless committed
        try:  # one native call: every experience learned and consolidated, then S3 of the candidate W
            prepared = txn.run_schedule(stimulus.x, stimulus.y, stimulus.schedule, learning_rate)
        except K1Error as exc:
            raise CompositionError("ECS_REFUSED", str(exc)) from exc
        readout = Readout(prepared.s3, prepared.epoch_after, dim, width)
        output = codec_map.decode(readout)
        if (type(output) is not tuple or len(output) > max_output_tokens
                or not all(type(t) is int and 0 <= t < vocab for t in output)):
            raise CompositionError("DECODE", "decode must return in-vocabulary token IDs within the limit")
        decoder = tokenizer.decoder()
        rendered = "".join(decoder.push(t) for t in output) + decoder.finish()
        try:
            committed = txn.commit_identity()  # third crossing: commit plus exact retained-state identities
        except K1Error as exc:
            raise CompositionError("ECS_STALE" if exc.code == "STALE" else "ECS_REFUSED", str(exc)) from exc
    return TurnResult(tokens, output, rendered, readout, committed.commit.epoch_before,
                      committed.commit.epoch_after, classification,
                      committed.state_before_digest, committed.state_after_digest)
