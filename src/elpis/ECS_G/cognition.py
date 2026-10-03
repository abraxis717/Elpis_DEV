"""Cognitive R0: ECS-native stateful query and learning on one ECS_G state.

See docs/COGNITION_R0.md. Two distinct operations on one authoritative state:

* QUERY ``x -> f_W(x)``: the native forward map of the current ``W``
  (:meth:`Executor.forward`), one native call. Read-only.
* LEARN ``(X, y)``: ``K`` qualified G1 gradient steps in one native call
  (:meth:`Executor.learn`; the ``K`` loop is native), committed as one
  transition: ``W -> W'``, epoch ``+K``. A refused step leaves ``W`` and the
  epoch unchanged.

The only learned state is ``W`` with its epoch, owned by the native ECS_G
executor. The core holds that executor and a fixed learning rate, nothing
else: no examples, labels, caches or receipts. :meth:`CognitiveCore.s3` is a
diagnostic projection; nothing runs from ``S3`` alone, and snapshots persist
``W`` itself.

Python controls the ECS here and never executes its hot path
(docs/ECS_RUNTIME_R1.md). A :class:`Transition` receipt is audit packaging
produced outside the numerical operation: two snapshot identities around the
one native learn call and the experience digest. Its cost does not grow with
``K``, and ``learn(..., receipt=False)`` skips it entirely.

R0 claims no more than that (docs/COGNITION_R0.md): no language, and no
promotion of the cubic kernel, ``S3`` or gradient descent beyond R0. Standard
library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .native import DEFAULT_MAX_ROWS, ECSGError, Executor

__all__ = ("CognitiveCore", "Transition", "experience_digest")

TRANSITION_SCHEMA = "elpis.ecsg.cognition-r0.transition.v1"
EXPERIENCE_SCHEMA = "elpis.ecsg.cognition-r0.experience.v1"
_MAX_STEPS = 100_000


def _digest(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def experience_digest(x_rows, y, *, dim=None):
    """Identity of one experience ``(X, y)`` in the domain ``elpis.ecsg.cognition-r0.experience.v1``.

    SHA-256 over the domain, a NUL byte and the canonical JSON (sorted keys,
    no whitespace, NaN/Infinity refused) of ``{"x": rows, "y": targets}``,
    where every value is a binary64 written as Python's shortest round-trip
    ``repr``. It is a digest of those JSON representations, in order, not of
    raw binary64 bit patterns. A flat binary64 buffer of rows needs ``dim``.
    """
    try:
        if dim is not None and _is_buffer(x_rows):
            flat = _buffer_values(x_rows)
            x_rows = [flat[i:i + dim] for i in range(0, len(flat), dim)]
        if _is_buffer(y):
            y = _buffer_values(y)
        rows = [[float(v) for v in row] for row in x_rows]
        targets = [float(v) for v in y]
        return _digest(EXPERIENCE_SCHEMA, {"x": rows, "y": targets})
    except (TypeError, ValueError) as exc:
        raise ECSGError("INVALID", "experience values") from exc


def _buffer_values(buffer):
    return memoryview(buffer).cast("B").cast("d").tolist()


def _is_buffer(value):
    try:
        memoryview(value)
    except TypeError:
        return False
    return True


@dataclass(frozen=True)
class Transition:
    """Receipt of one committed LEARN: which state became which, through which experience."""
    before: str          # SHA-256 of the snapshot before
    after: str           # SHA-256 of the snapshot after
    epoch_before: int
    epoch_after: int
    experience: str      # experience_digest(X, y)
    steps: int
    learning_rate: float
    schema: str = TRANSITION_SCHEMA

    @property
    def digest(self):
        return _digest(TRANSITION_SCHEMA, asdict(self))


def _identity(snapshot):
    return hashlib.sha256(snapshot).hexdigest()


class CognitiveCore:
    """One ECS_G executor that answers queries from W and learns into W."""

    __slots__ = ("_state", "learning_rate")

    def __init__(self, state, *, learning_rate):
        if type(state) is not Executor:
            raise ECSGError("INVALID", "an ECS_G Executor is required")
        if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ECSGError("INVALID", "explicit positive finite learning rate")
        state._live()
        self._state, self.learning_rate = state, learning_rate

    @classmethod
    def create(cls, api, dim, width, initial_w, *, learning_rate, max_rows=DEFAULT_MAX_ROWS):
        return cls(Executor.create(api, dim, width, initial_w, max_rows=max_rows), learning_rate=learning_rate)

    @classmethod
    def restore(cls, api, snapshot, *, learning_rate, max_rows=DEFAULT_MAX_ROWS):
        return cls(Executor.restore(api, snapshot, max_rows=max_rows), learning_rate=learning_rate)

    @property
    def dim(self):
        return self._state.dim

    @property
    def width(self):
        return self._state.width

    @property
    def epoch(self):
        return self._state.epoch

    @property
    def max_rows(self):
        """Experience rows one learn may admit; grown only by :meth:`reserve`."""
        return self._state.max_rows

    @property
    def identity(self):
        """SHA-256 of the current snapshot: the identity of the learned state."""
        return _identity(self._state.snapshot())

    def query(self, x_rows):
        """QUERY: ``f_W(x)`` for each row, from the current authoritative W. Read-only."""
        return self._state.forward(x_rows)

    def query_into(self, x, out):
        """QUERY into a caller-owned binary64 buffer (no allocation, no conversion). Returns the row count."""
        return self._state.forward_into(x, out)

    def learn(self, x_rows, y, *, steps=1, receipt=True):
        """LEARN: ``steps`` qualified G1 steps on ``(X, y)`` in one native call, committed atomically.

        Returns a :class:`Transition` receipt, or with ``receipt=False`` the
        native :class:`~elpis.ECS_G.native.Commit` alone.
        """
        if type(steps) is not int or not 1 <= steps <= _MAX_STEPS:
            raise ECSGError("INVALID", "steps must be 1..100000")
        state = self._state
        if not receipt:
            return state.learn(x_rows, y, self.learning_rate, steps)
        rows, targets = _materialized(x_rows), _materialized(y)
        experience = experience_digest(rows, targets, dim=state.dim)
        before = _identity(state.snapshot())
        commit = state.learn(rows, targets, self.learning_rate, steps)
        return Transition(before, _identity(state.snapshot()), commit.epoch_before, commit.epoch_after,
                          experience, steps, self.learning_rate)

    def reserve(self, max_rows):
        """Cold path: grow the admitted-experience capacity explicitly."""
        self._state.reserve(max_rows)

    def snapshot(self):
        """Portable bytes of W and its epoch (the complete learned state)."""
        return self._state.snapshot()

    def s3(self):
        """Diagnostic coarse observable S3(W). Not sufficient for future dynamics."""
        return self._state.s3()

    def close(self):
        self._state.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _materialized(values):
    """One pass over the caller's input, shared by the receipt and the native call."""
    return values if isinstance(values, (list, tuple)) or _is_buffer(values) else tuple(values)
