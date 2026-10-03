"""Cognitive R0: ECS-native stateful query and learning on one ECS_G state.

See docs/COGNITION_R0.md. Two distinct operations on one authoritative state:

* QUERY ``x -> f_W(x)``: the native forward map of the current ``W``
  (:meth:`WorldState.forward`). Read-only.
* LEARN ``(X, y)``: ``K`` qualified G1 gradient steps on a fork of ``W``,
  then one atomic adoption (:meth:`WorldState.adopt`): ``W -> W'``, epoch
  ``+K``. A refused step or a stale adoption leaves ``W`` and the epoch
  unchanged.

The only learned state is ``W`` with its epoch, owned by the native ECS_G
state. The core holds that state and a fixed learning rate, nothing else: no
examples, labels, caches or receipts. :meth:`CognitiveCore.s3` is a
diagnostic projection; nothing runs from ``S3`` alone, and snapshots persist
``W`` itself.

R0 claims no more than that (docs/COGNITION_R0.md): no language, and no
promotion of the cubic kernel, ``S3`` or gradient descent beyond R0. Standard
library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .native import ECSGError, WorldState

__all__ = ("CognitiveCore", "Transition", "experience_digest")

TRANSITION_SCHEMA = "elpis.ecsg.cognition-r0.transition.v1"
EXPERIENCE_SCHEMA = "elpis.ecsg.cognition-r0.experience.v1"
_MAX_STEPS = 100_000


def _digest(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def experience_digest(x_rows, y):
    """Identity of one experience ``(X, y)``: canonical binary64 values, in order."""
    try:
        rows = [[float(v) for v in row] for row in x_rows]
        targets = [float(v) for v in y]
        return _digest(EXPERIENCE_SCHEMA, {"x": rows, "y": targets})
    except (TypeError, ValueError) as exc:
        raise ECSGError("INVALID", "experience values") from exc


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
    """One ECS_G state that answers queries from W and learns into W."""

    __slots__ = ("_state", "learning_rate")

    def __init__(self, state, *, learning_rate):
        if type(state) is not WorldState:
            raise ECSGError("INVALID", "an ECS_G WorldState is required")
        if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ECSGError("INVALID", "explicit positive finite learning rate")
        state._live()
        self._state, self.learning_rate = state, learning_rate

    @classmethod
    def create(cls, api, dim, width, initial_w, *, learning_rate):
        return cls(WorldState.create(api, dim, width, initial_w), learning_rate=learning_rate)

    @classmethod
    def restore(cls, api, snapshot, *, learning_rate):
        return cls(WorldState.restore(api, snapshot), learning_rate=learning_rate)

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
    def identity(self):
        """SHA-256 of the current snapshot: the identity of the learned state."""
        return _identity(self._state.snapshot())

    def query(self, x_rows):
        """QUERY: ``f_W(x)`` for each row, from the current authoritative W. Read-only."""
        return self._state.forward(x_rows)

    def learn(self, x_rows, y, *, steps=1):
        """LEARN: ``steps`` qualified G1 steps on ``(X, y)``, committed as one atomic transition."""
        if type(steps) is not int or not 1 <= steps <= _MAX_STEPS:
            raise ECSGError("INVALID", "steps must be 1..100000")
        rows, targets = tuple(x_rows), tuple(y)
        experience = experience_digest(rows, targets)
        before, epoch_before = _identity(self._state.snapshot()), self._state.epoch
        candidate = self._state.fork()
        try:
            for _ in range(steps):
                candidate.step(rows, targets, self.learning_rate)
            self._state.adopt(candidate)
        finally:
            candidate.close()  # no-op once adopted
        return Transition(before, _identity(self._state.snapshot()), epoch_before, self._state.epoch,
                          experience, steps, self.learning_rate)

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
