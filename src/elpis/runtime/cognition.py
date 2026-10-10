"""The canonical cognitive operations: DSV4 codec -> ECS -> DSV4 codec (docs/ELPIS_MISSION.md, docs/COGNITION_R0.md).

Cognitive R0 defines two different operations, and this module keeps them apart by type, by entry point and by the
native calls each one can reach:

* **QUERY** (:class:`QueryRequest`, :func:`run_query`, :meth:`elpis.runtime.Runtime.run_query`): read-only. ::

      text --codec encode--> tokens --QUERY encode (UNQUALIFIED)--> QueryStimulus: query rows x
           --native K1 query: f_W(x) by the qualified forward map of the authoritative W, together with the
             retained-state identity of that same state (one guarded native call)--> QueryReadout
           --QUERY decode (UNQUALIFIED)--> tokens --codec decode--> text

  STATE CHANGE = NONE. A query never learns, never consolidates, never advances the epoch, never writes W, H or a,
  opens no transaction and publishes nothing to continuity. Its answer is ``f_W(x)``, never ``S3`` and never a
  lookup of training data: nothing but the authoritative K1 state and the query rows enters it.

* **LEARN** (:class:`LearnRequest`, :func:`run_learn`, :meth:`elpis.runtime.Runtime.run_learn`): requires an
  explicit :class:`LearnAuthority`. ::

      text --codec encode--> tokens --LEARN encode (UNQUALIFIED)--> Stimulus: ordered experience schedule
           --native K1 transaction candidate: per experience, its K1 learning steps, then the consolidation of
             its rows (one native call)--> (W, epoch, H, a) candidate
           --atomic native commit of the complete candidate, or nothing
           --(managed) one continuity publication of the new retained-state identity

  A LEARN returns the committed transition and its exact retained-state identities. It produces no text: a
  response is a QUERY.

* **Legacy learned turn** (:func:`run_turn`, :meth:`elpis.runtime.Runtime.run_turn`): the original synthetic
  scaffold, LEARN followed by the decode of ``S3`` of the candidate W (the kernel's coarse observable) into tokens.
  It is kept, explicitly classified ``LEGACY_LEARNED_TURN``, for the replay of its tests and identities. It is a
  LEARN: it needs the same explicit :class:`LearnAuthority`, and it is never a query.

The ECS substrate is native K1 (docs/ECS_K1_RUNTIME.md): a standalone :class:`~elpis.ECS.k1.K1State` or an
FMS-resident :class:`~elpis.ECS.k1.K1FMSState`. The Runtime R1 ``Executor`` remains a qualified primitive and the
K1-disabled reference; it is not accepted here.

The codec only crosses the token boundary: ``tokenizer`` supplies ``encode(text)``, ``decoder()`` and
``vocab_size``. No DSV model, parameter bank or transformer recurrence is involved, and none is imported.

No ECS<->DSV semantic codec is defined or qualified in this repository. Without an explicitly supplied codec every
operation refuses with ``ECS_CODEC_UNQUALIFIED`` before touching anything: text generation is unavailable. There is
no fallback. A codec map never authorizes itself: an operation runs it only as an
:class:`~elpis.runtime.codec_authority.AdmittedCodec`, bound to an independently pinned codec authority that grants
it the operation's capabilities (QUERY encode/decode, LEARN encode, legacy LEARN decode) and fixes the
classification every result carries; the map's own ``classification`` is reporting metadata that must match it. The
only admitted maps in the repository are ``TEST_ONLY TRAINING=NONE SEMANTICS=NONE`` interface fixtures.

PYTHON MAY CONTROL THE ECS; IT MAY NOT EXECUTE THE ECS HOT PATH. A query is one native crossing; a LEARN is three
(begin, the whole schedule on the candidate, commit) whatever the number of experiences, rows or K1 steps. Inputs
cross as contiguous buffers; nothing in this module walks an ECS scalar, a row, a feature or a step. A refused
operation leaves ``(W, epoch, H, a)`` byte-for-byte unchanged.

This module performs no durable writes and holds no state. Its functions are the unmanaged operations over a native
K1 state (no continuity); the managed operations of :class:`elpis.runtime.Runtime` share this module's boundary
(request validation, encode, decode) and hand the native work and its continuity publication to RuntimeCore
(native/runtime, docs/RUNTIME_CORE.md).
"""
from __future__ import annotations
from array import array
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Protocol

from elpis.ECS.k1 import MAX_EXPERIENCES, K1Error, K1FMSState, K1State

from . import codec_authority as _ca
from .codec_authority import AdmittedCodec
from .errors import CompositionError
from .fuel import CEILING, CognitiveBudget, admit_learn, admit_query

__all__ = ("CODEC_UNQUALIFIED", "LEGACY_LEARNED_TURN", "CognitiveOperation", "ECSCodecMap", "ECSQueryCodecMap",
           "LearnAuthority", "LearnRequest", "LearnResult", "QueryReadout", "QueryRequest", "QueryResult",
           "QueryStimulus", "Readout", "Stimulus", "TurnResult", "run_learn", "run_query", "run_turn")

CODEC_UNQUALIFIED = "ECS_CODEC_UNQUALIFIED"
UNQUALIFIED_DETAIL = "ECS codec mapping not yet qualified; text generation unavailable"
LEGACY_LEARNED_TURN = "LEGACY_LEARNED_TURN"
_MAX_ROWS, _MAX_STEPS, _MAX_TOKENS, _MAX_DIM = 256, 1 << 20, 4096, 64
MAX_QUERY_ROWS = 4096
_SUBSTRATES = (K1State, K1FMSState)


class CognitiveOperation(str, Enum):
    """The two cognitive operations of Cognitive R0. They are different operations, not modes of one."""

    QUERY = "QUERY"     # f_W(x) from the authoritative state; state change none
    LEARN = "LEARN"     # an admitted experience schedule committed atomically under explicit authority


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
    """An admitted, native-ready LEARN stimulus: an ordered experience schedule in the kernel's only qualified input
    form.

    * ``x``: every experience's input rows in order, row-major binary64 (``rows x dim`` values);
    * ``y``: their targets, binary64 (``rows`` values);
    * ``schedule``: uint64 ``(rows, steps)`` per experience, 1..64 experiences, 1..256 rows and 1..2^20 K1 steps
      each; experience t uses the next ``rows`` rows of ``x`` and ``y``.

    Admission validates shapes and bounds over the bounded schedule metadata and takes owned contiguous copies;
    finiteness is checked natively before the candidate is touched. Each experience is learned (its K1 steps) and
    then consolidated (its rows) natively; the operation commits all of them or none.
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
        form. Not used by the operations, which consume an admitted Stimulus only."""
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


@dataclass(frozen=True, init=False)
class QueryStimulus:
    """An admitted, native-ready QUERY input: ``rows`` query rows of ``dim`` binary64 values, row-major.

    A query carries no target, no schedule and no learning parameter: there is nothing in it that could make it a
    LEARN. Admission takes an owned contiguous copy; finiteness is checked natively.
    """
    x: array
    dim: int
    rows: int

    def __init__(self, x, *, dim):
        if type(dim) is not int or not 1 <= dim <= _MAX_DIM:
            raise CompositionError("STIMULUS", "dim: 1..64")
        xs = _owned(x, "d", "x")
        rows = len(xs) // dim
        if len(xs) != rows * dim or not 1 <= rows <= MAX_QUERY_ROWS:
            raise CompositionError("STIMULUS", f"query: 1..{MAX_QUERY_ROWS} rows of dim values")
        for name, value in (("x", xs), ("dim", dim), ("rows", rows)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Readout:
    """LEGACY learned turn: what ECS exposed to its decode, ``S3(W)`` of the candidate and the state it was read
    from. ``S3`` is a diagnostic coarse observable; it is not a query response."""
    s3: tuple
    epoch: int
    dim: int
    width: int


@dataclass(frozen=True)
class QueryReadout:
    """What a QUERY exposes to its decode: ``f_W(x)`` per query row, computed natively from the authoritative W,
    and the retained-state identity (``state_digest``) of the state that computed it."""
    values: tuple
    dim: int
    width: int
    state_digest: bytes


class ECSCodecMap(Protocol):
    """The missing ECS<->DSV semantic codec, LEARN side: ``encode`` (tokens -> experience schedule) and, for the
    legacy learned turn only, ``decode`` (S3 readout -> tokens). No qualified implementation exists."""

    classification: str

    def encode(self, tokens: tuple) -> Stimulus: ...

    def decode(self, readout: Readout) -> tuple: ...


class ECSQueryCodecMap(Protocol):
    """The missing ECS<->DSV semantic codec, QUERY side: tokens -> query rows, ``f_W(x)`` -> tokens. No qualified
    implementation exists."""

    classification: str

    def encode_query(self, tokens: tuple) -> QueryStimulus: ...

    def decode_query(self, readout: QueryReadout) -> tuple: ...


@dataclass(frozen=True)
class LearnAuthority:
    """Explicit learning authority: without it nothing learns.

    QUERY has no parameter that can carry it, and no LEARN entry point runs without it. It binds the program
    parameters of the learning law that are not learned state (docs/COGNITION_R0.md): the explicit positive finite
    learning rate, and the total fuel one LEARN may use (``budget``: experiences, rows, the reserve ceiling,
    learning steps and ECS work units, admitted before any reserve or transaction; at most the canonical ceiling).
    ``grant`` names who granted it and why (a non-empty statement; it is reported, never interpreted). It is not a
    credential against an in-process caller, which can construct one: it makes learning an explicit act at every
    call site instead of a side effect of answering.
    """

    learning_rate: float
    grant: str
    budget: CognitiveBudget = CEILING

    def __post_init__(self):
        if type(self.budget) is not CognitiveBudget:
            raise CompositionError("LEARN_AUTHORITY", "budget must be a CognitiveBudget")
        rate = self.learning_rate
        if type(rate) is not float or not math.isfinite(rate) or rate <= 0:
            raise CompositionError("LEARNING_RATE", "explicit positive finite learning rate")
        if type(self.grant) is not str or not self.grant.strip():
            raise CompositionError("LEARN_AUTHORITY", "a learning grant must state who granted it and why")


@dataclass(frozen=True)
class QueryRequest:
    """A QUERY: ``text`` through ``tokenizer`` and the QUERY side of an admitted ``codec``. Nothing in it can learn."""

    text: str
    tokenizer: object
    codec: AdmittedCodec | None = None
    max_output_tokens: int = 256
    budget: CognitiveBudget = CEILING   # query rows, output tokens, ECS work units; admitted before the native query

    operation = CognitiveOperation.QUERY

    def __post_init__(self):
        if type(self.budget) is not CognitiveBudget:
            raise CompositionError("OPERATION", "a QUERY carries a CognitiveBudget, never learning authority")


@dataclass(frozen=True)
class LearnRequest:
    """A LEARN: ``text`` through ``tokenizer`` and the LEARN encode of an admitted ``codec``, under explicit
    ``authority``."""

    text: str
    tokenizer: object
    codec: AdmittedCodec | None = None
    authority: LearnAuthority | None = None

    operation = CognitiveOperation.LEARN


@dataclass(frozen=True)
class QueryResult:
    input_tokens: tuple
    output_tokens: tuple
    text: str
    readout: QueryReadout
    codec: str          # the classification the codec authority admitted
    operation: CognitiveOperation = CognitiveOperation.QUERY

    @property
    def state_digest(self) -> bytes:
        """The retained-state identity the answer was computed from (unchanged by the query)."""
        return self.readout.state_digest


@dataclass(frozen=True)
class LearnResult:
    input_tokens: tuple
    experiences: int
    epoch_before: int
    epoch_after: int
    codec: str          # the classification the codec authority admitted
    grant: str          # the learning authority's grant
    state_before_digest: bytes
    state_after_digest: bytes
    operation: CognitiveOperation = CognitiveOperation.LEARN


@dataclass(frozen=True)
class TurnResult:
    """LEGACY learned turn result."""
    input_tokens: tuple
    output_tokens: tuple
    text: str
    readout: Readout
    epoch_before: int
    epoch_after: int
    codec: str          # the classification the codec authority admitted
    # Exact K1 retained-state identities (state_digest) of the committed transition, from the native commit.
    state_before_digest: bytes | None = field(default=None, compare=False)
    state_after_digest: bytes | None = field(default=None, compare=False)
    operation: CognitiveOperation = field(default=CognitiveOperation.LEARN, compare=False)
    surface: str = field(default=LEGACY_LEARNED_TURN, compare=False)


# --- the boundary: validation, encode, decode (shared by the managed operations) --------------------------------

def _admitted(codec, needed, deployment_pin):
    """The admitted codec map and its admitted classification; refuses (fail closed) before anything else without a
    codec, and before any ECS call without a valid, sufficient admission."""
    if codec is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)
    if type(codec) is not AdmittedCodec:
        raise CompositionError("CODEC_UNADMITTED", "an ECS codec map runs only as an independently AdmittedCodec")
    return codec.require(needed, deployment_pin), codec.classification


def _validate_common(substrate, text):
    if type(substrate) not in _SUBSTRATES:
        raise CompositionError("ECS_STATE", "a native K1 state (K1State or K1FMSState) is required")
    if type(text) is not str:
        raise CompositionError("INPUT", "text must be str")


def _validate_output_limit(max_output_tokens):
    if type(max_output_tokens) is not int or not 0 <= max_output_tokens <= _MAX_TOKENS:
        raise CompositionError("OUTPUT_LIMIT", "0..4096 output tokens")


def _validate_authority(authority):
    if authority is None:
        raise CompositionError("LEARN_UNAUTHORIZED", "LEARN requires an explicit LearnAuthority")
    if type(authority) is not LearnAuthority:
        raise CompositionError("LEARN_AUTHORITY", "authority must be a LearnAuthority")
    return authority


def _validate_query_request(substrate, request, deployment_pin=None):
    """Validate a QUERY without reading or mutating K1 state; returns (codec map, admitted classification)."""
    if type(request) is not QueryRequest:
        raise CompositionError("OPERATION", "a QUERY takes a QueryRequest")
    if request.codec is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)
    _validate_common(substrate, request.text)
    _validate_output_limit(request.max_output_tokens)
    return _admitted(request.codec, _ca.QUERY, deployment_pin)


def _validate_learn_request(substrate, request, deployment_pin=None):
    """Validate a LEARN without reading or mutating K1 state; returns (codec map, classification, authority)."""
    if type(request) is not LearnRequest:
        raise CompositionError("OPERATION", "a LEARN takes a LearnRequest")
    if request.codec is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)
    authority = _validate_authority(request.authority)
    _validate_common(substrate, request.text)
    codec_map, classification = _admitted(request.codec, _ca.LEARN, deployment_pin)
    return codec_map, classification, authority


def _validate_turn_request(substrate, text, *, codec=None, authority=None, max_output_tokens=256,
                           deployment_pin=None):
    """Validate the legacy learned turn without reading or mutating K1 state; returns (codec map, classification,
    authority)."""
    if codec is None:
        raise CompositionError(CODEC_UNQUALIFIED, UNQUALIFIED_DETAIL)
    authority = _validate_authority(authority)
    _validate_common(substrate, text)
    _validate_output_limit(max_output_tokens)
    codec_map, classification = _admitted(codec, _ca.LEGACY_LEARNED_TURN_CAPABILITIES, deployment_pin)
    return codec_map, classification, authority


def _admit_turn(substrate, stimulus, authority, max_output_tokens):
    """The legacy learned turn's fuel: its LEARN schedule and its decoded output, under the authority's budget."""
    admit_learn(stimulus, substrate.dim, substrate.width, authority.budget)
    if max_output_tokens > authority.budget.max_output_tokens:
        raise CompositionError("COGNITION_FUEL_EXCEEDED", "output tokens")


def _encode(substrate, text, tokenizer, codec_map):
    """LEARN encode: text -> tokens -> an admitted experience Stimulus of the state's dimension."""
    tokens = tuple(tokenizer.encode(text))
    stimulus = codec_map.encode(tokens)
    if type(stimulus) is not Stimulus:
        raise CompositionError("STIMULUS", "the ECS codec map must return a Stimulus")
    if stimulus.dim != substrate.dim:
        raise CompositionError("STIMULUS", "the stimulus dimension must match the ECS state")
    return tokens, stimulus


def _encode_query(substrate, text, tokenizer, codec_map):
    """QUERY encode: text -> tokens -> admitted query rows of the state's dimension."""
    tokens = tuple(tokenizer.encode(text))
    stimulus = codec_map.encode_query(tokens)
    if type(stimulus) is not QueryStimulus:
        raise CompositionError("STIMULUS", "the QUERY encode must return a QueryStimulus")
    if stimulus.dim != substrate.dim:
        raise CompositionError("STIMULUS", "the query dimension must match the ECS state")
    return tokens, stimulus


def _render(output, tokenizer, max_output_tokens):
    vocab = tokenizer.vocab_size
    if (type(output) is not tuple or len(output) > max_output_tokens
            or not all(type(t) is int and 0 <= t < vocab for t in output)):
        raise CompositionError("DECODE", "decode must return in-vocabulary token IDs within the limit")
    decoder = tokenizer.decoder()
    return output, "".join(decoder.push(t) for t in output) + decoder.finish()


def _decode(codec_map, readout, tokenizer, max_output_tokens):
    """LEGACY decode: S3 readout -> in-vocabulary token IDs within the limit -> text."""
    return _render(codec_map.decode(readout), tokenizer, max_output_tokens)


def _decode_query(codec_map, readout, tokenizer, max_output_tokens):
    """QUERY decode: f_W(x) -> in-vocabulary token IDs within the limit -> text."""
    return _render(codec_map.decode_query(readout), tokenizer, max_output_tokens)


# --- the unmanaged operations over a native K1 state (no continuity) ----------------------------------------------

def run_query(substrate, request: QueryRequest) -> QueryResult:
    """One unmanaged QUERY: read-only, one native call (``f_W(x)`` and the identity of the state that answered).

    Fails closed without a qualified ECS codec map. Nothing is written: W, epoch, H, a and the generation are
    unchanged whether the query succeeds or is refused.
    """
    codec_map, classification = _validate_query_request(substrate, request)
    tokens, stimulus = _encode_query(substrate, request.text, request.tokenizer, codec_map)
    admit_query(stimulus, substrate.dim, substrate.width, request.budget, request.max_output_tokens)
    try:
        values, digest = substrate.query_identity(stimulus.x)
    except K1Error as exc:
        raise CompositionError("ECS_REFUSED", str(exc)) from exc
    readout = QueryReadout(values, substrate.dim, substrate.width, digest)
    output, rendered = _decode_query(codec_map, readout, request.tokenizer, request.max_output_tokens)
    return QueryResult(tokens, output, rendered, readout, classification)


def run_learn(substrate, request: LearnRequest) -> LearnResult:
    """One unmanaged LEARN under explicit authority: the experience schedule on a native candidate, committed
    atomically (``W``, epoch, ``H``, ``a`` together) or not at all. Three native crossings."""
    codec_map, classification, authority = _validate_learn_request(substrate, request)
    tokens, stimulus = _encode(substrate, request.text, request.tokenizer, codec_map)
    admit_learn(stimulus, substrate.dim, substrate.width, authority.budget)   # before any reserve or transaction
    committed = _learn_native(substrate, stimulus, authority.learning_rate, None)[1]
    return LearnResult(tokens, stimulus.experiences, committed.commit.epoch_before, committed.commit.epoch_after,
                       classification, authority.grant, committed.state_before_digest, committed.state_after_digest)


def _learn_native(substrate, stimulus, learning_rate, between):
    """begin -> one native schedule on the candidate -> ``between(prepared)`` -> commit, or abort on any refusal."""
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
        decoded = between(prepared) if between is not None else None
        try:
            committed = txn.commit_identity()  # third crossing: commit plus exact retained-state identities
        except K1Error as exc:
            raise CompositionError("ECS_STALE" if exc.code == "STALE" else "ECS_REFUSED", str(exc)) from exc
    return decoded, committed


def run_turn(substrate, text, *, tokenizer, codec=None, authority=None, max_output_tokens=256) -> TurnResult:
    """LEGACY learned turn (unmanaged): a LEARN under explicit authority whose S3 readout is decoded to text.

    Kept for the replay of the original synthetic scaffold; it is never a query. Fails closed without a qualified
    ECS codec, refuses ``LEARN_UNAUTHORIZED`` without a :class:`LearnAuthority`, and needs a codec admitted for
    LEARN encode and LEARN decode.
    """
    codec_map, classification, authority = _validate_turn_request(substrate, text, codec=codec, authority=authority,
                                                                  max_output_tokens=max_output_tokens)
    tokens, stimulus = _encode(substrate, text, tokenizer, codec_map)
    _admit_turn(substrate, stimulus, authority, max_output_tokens)   # before any reserve or transaction

    def decode(prepared):  # between the schedule and the commit: the candidate commits only after the decode
        readout = Readout(prepared.s3, prepared.epoch_after, substrate.dim, substrate.width)
        return readout, _decode(codec_map, readout, tokenizer, max_output_tokens)

    (readout, (output, rendered)), committed = _learn_native(substrate, stimulus, authority.learning_rate, decode)
    return TurnResult(tokens, output, rendered, readout, committed.commit.epoch_before,
                      committed.commit.epoch_after, classification,
                      committed.state_before_digest, committed.state_after_digest)
