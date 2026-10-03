"""Turn-boundary conditioning in, turn observation out (driver-neutral contracts).

Inference knows only that it received *admitted* conditioning: a bounded,
finite vector frozen for one complete principal sequence, together with an
opaque provenance digest chosen by whoever admitted it. It does not know which
subsystem produced it, and it never calls back into that subsystem.

* :class:`TurnConditioning` enters at ``PrincipalEngine.begin`` and is frozen
  for the whole sequence. Nothing about it is consulted per token beyond the
  target's own preprojected, immutable copy. Its digest is bound into the
  commit, so replay requires the same conditioning.
* :class:`TurnObservation` leaves at ``PrincipalEngine.finalize`` and only for
  a committed sequence: a fixed, token-identity-free summary of the final
  next-token distribution (sorted top probabilities, tail mass, normalized
  entropy), bound to the commit digest. It grants nothing.

This module imports no numpy so contract holders (the runtime) stay light.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import Code, digest_value, identity, require

__all__ = ("CONDITIONING_SCHEMA", "OBSERVATION_FEATURES", "OBSERVATION_SCHEMA", "TurnConditioning",
           "TurnObservation")

CONDITIONING_SCHEMA = "elpis.inference.turn-conditioning.v1"
OBSERVATION_SCHEMA = "elpis.inference.turn-observation.v1"
# top1..top6 sorted probabilities, tail mass (1 - top6 sum), entropy / ln(vocab)
OBSERVATION_FEATURES = 8
_MAX_VALUES = 4096


def _finite_floats(values, what):
    require(type(values) is tuple and 1 <= len(values) <= _MAX_VALUES, Code.LIMIT, what)
    require(all(type(v) is float and math.isfinite(v) for v in values), Code.ENCODING, what)


@dataclass(frozen=True)
class TurnConditioning:
    """Admitted, immutable conditioning for exactly one principal sequence."""

    values: tuple[float, ...]
    provenance: str            # opaque digest supplied by the admitting layer
    schema: str = CONDITIONING_SCHEMA

    def __post_init__(self):
        require(self.schema == CONDITIONING_SCHEMA, Code.UNSUPPORTED, "turn conditioning schema")
        _finite_floats(self.values, "turn conditioning values")
        digest_value(self.provenance)

    @property
    def digest(self):
        return identity("turn-conditioning", self)


@dataclass(frozen=True)
class TurnObservation:
    """Bounded summary of one committed sequence's final next-token distribution."""

    commit: str                # PrincipalCommit digest
    features: tuple[float, ...]
    outputs: int
    stop_reason: str
    schema: str = OBSERVATION_SCHEMA

    def __post_init__(self):
        require(self.schema == OBSERVATION_SCHEMA, Code.UNSUPPORTED, "turn observation schema")
        digest_value(self.commit)
        _finite_floats(self.features, "turn observation features")
        require(len(self.features) == OBSERVATION_FEATURES, Code.ENCODING, "turn observation width")
        require(type(self.outputs) is int and self.outputs >= 0, detail="turn observation outputs")
        require(type(self.stop_reason) is str and bool(self.stop_reason), detail="turn observation stop")

    @property
    def digest(self):
        return identity("turn-observation", self)
