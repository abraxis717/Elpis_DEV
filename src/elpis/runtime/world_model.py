"""Turn-boundary ECS_G world-model loop, owned by the runtime composition.

Information crosses between the native geometric world model (ECS_G) and
principal inference only at sequence boundaries, and only through the
runtime:

* Before turn N begins, :meth:`WorldModelLoop.turn_conditioning` projects the
  current microscopic state to ``S3(W_N)`` and freezes it as admitted
  :class:`~elpis.inference.conditioning.TurnConditioning`. Its provenance
  digest binds the ECS_G snapshot it was read from. Inference sees only that
  admitted vector; it never learns that ECS_G produced it.
* While turn N streams, nothing here runs.
* After turn N commits, :meth:`WorldModelLoop.advance` maps the committed
  :class:`~elpis.inference.conditioning.TurnObservation` through an explicit,
  digest-bound :class:`DriveMap` to one cubic drive ``(X, y)`` and applies
  exactly one atomic ECS_G gradient step: ``W_N -> W_N+1``. Only a commit
  that was conditioned by this loop's current state, and has not been
  applied before, is accepted. A failed or uncommitted turn never reaches
  this method, so it changes nothing.

The drive map is the explicit observation adapter. It reads only the
observation's distribution-summary features (never token identities and
never Engram address rows), in a fixed summation order with binary64
arithmetic, so the same commits always produce the same world. No learned
production drive map ships; qualification uses deterministic fixtures.

ECS_G's ``W`` stays the authority for world state. The receipt history
(ECS_C) records only what it already records for a principal commit, plus
the conditioning digest whose provenance names the ECS_G snapshot; that
digest chain is not world state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from elpis.ECS_G.native import ECSGError, ECSGLibrary, WorldState
from elpis.identity import content_digest
from elpis.inference.conditioning import OBSERVATION_FEATURES, TurnConditioning, TurnObservation

__all__ = ("DRIVE_MAP_SCHEMA", "DriveMap", "WorldModelError", "WorldModelLoop", "load_world_library")

DRIVE_MAP_SCHEMA = "elpis.runtime.world-drive-map.v1"
_PROVENANCE = "elpis.runtime.world-conditioning-provenance.v1"
_MAX_ROWS = 256


class WorldModelError(RuntimeError):
    """The world-model loop refused an operation; ECS_G state is unchanged."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _floats(values, count, what):
    if type(values) is not tuple or len(values) != count or not all(
            type(v) is float and math.isfinite(v) for v in values):
        raise WorldModelError("DRIVE_MAP", what)


@dataclass(frozen=True)
class DriveMap:
    """Explicit affine map from one TurnObservation to one ECS_G drive ``(X, y)``.

    With ``f`` the observation's features (``OBSERVATION_FEATURES`` values)::

        X[r][a] = x_bias[r*dim + a] + sum_k x_weight[(r*dim + a)*F + k] * f[k]
        y[r]    = y_bias[r]         + sum_k y_weight[r*F + k] * f[k]

    summed in ascending ``k`` in binary64. ``rows`` is the drive batch size R.
    """

    dim: int
    rows: int
    x_weight: tuple[float, ...]
    x_bias: tuple[float, ...]
    y_weight: tuple[float, ...]
    y_bias: tuple[float, ...]
    schema: str = DRIVE_MAP_SCHEMA

    def __post_init__(self):
        if self.schema != DRIVE_MAP_SCHEMA:
            raise WorldModelError("DRIVE_MAP", "schema")
        if type(self.dim) is not int or type(self.rows) is not int or not (
                1 <= self.dim <= 64 and 1 <= self.rows <= _MAX_ROWS):
            raise WorldModelError("DRIVE_MAP", "dim/rows")
        f, cells = OBSERVATION_FEATURES, self.rows * self.dim
        _floats(self.x_weight, cells * f, "x_weight")
        _floats(self.x_bias, cells, "x_bias")
        _floats(self.y_weight, self.rows * f, "y_weight")
        _floats(self.y_bias, self.rows, "y_bias")

    @property
    def digest(self):
        return content_digest(DRIVE_MAP_SCHEMA, self)

    def drive(self, observation: TurnObservation):
        if type(observation) is not TurnObservation:
            raise WorldModelError("OBSERVATION", "TurnObservation required")
        f, F, d = observation.features, OBSERVATION_FEATURES, self.dim

        def affine(weights, bias, row):
            total = bias
            for k in range(F):
                total += weights[row * F + k] * f[k]
            return total

        x = tuple(tuple(affine(self.x_weight, self.x_bias[r * d + a], r * d + a) for a in range(d))
                  for r in range(self.rows))
        y = tuple(affine(self.y_weight, self.y_bias[r], r) for r in range(self.rows))
        return x, y


def load_world_library(root, library, *, authority, library_id):
    """Open ``libelpis_ecsg_math`` through the substrate's sealed, pinned loader."""
    from elpis.substrate.authority import PinnedAuthority
    from elpis.substrate.boundary import RootCapability, load_native

    if type(authority) is not PinnedAuthority or library_id not in authority.libraries:
        raise WorldModelError("IDENTITY", "pinned ECS_G library")
    with RootCapability(root) as boundary:
        return ECSGLibrary(load_native(boundary, library, authority.libraries[library_id]))


class WorldModelLoop:
    """One ECS_G world advanced once per committed, conditioned principal turn."""

    __slots__ = ("_state", "drive_map", "learning_rate", "_conditioning", "_applied", "transitions")

    def __init__(self, state: WorldState, drive_map: DriveMap, *, learning_rate: float, applied: str | None = None):
        if type(state) is not WorldState:
            raise WorldModelError("STATE", "ECS_G WorldState required")
        if type(drive_map) is not DriveMap or drive_map.dim != state.dim:
            raise WorldModelError("DRIVE_MAP", "drive map dimension must equal the world dimension")
        if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise WorldModelError("LEARNING_RATE", "explicit positive finite learning rate")
        if applied is not None and (type(applied) is not str or len(applied) != 64):
            raise WorldModelError("STATE", "applied commit digest")
        self._state, self.drive_map, self.learning_rate = state, drive_map, learning_rate
        self._applied = applied        # last commit digest applied to W (idempotence and lineage)
        self.transitions = 0           # steps applied by this loop instance
        self._conditioning = self._project()

    @classmethod
    def restore(cls, api: ECSGLibrary, snapshot: bytes, drive_map: DriveMap, *, learning_rate: float,
                applied: str | None):
        """Continue exactly from :meth:`checkpoint` output."""
        return cls(WorldState.restore(api, snapshot), drive_map, learning_rate=learning_rate, applied=applied)

    def _project(self):
        state = self._state
        snapshot = state.snapshot()
        provenance = content_digest(_PROVENANCE, dict(
            snapshot=hashlib.sha256(snapshot).hexdigest(), epoch=state.epoch, dim=state.dim,
            width=state.width, applied=self._applied, drive=self.drive_map.digest, rate=self.learning_rate))
        return TurnConditioning(tuple(float(v) for v in state.s3()), provenance)

    @property
    def epoch(self):
        return self._state.epoch

    @property
    def applied(self):
        return self._applied

    def turn_conditioning(self) -> TurnConditioning:
        """The frozen S3(W) conditioning for the next turn (no ECS_G call; precomputed)."""
        return self._conditioning

    def advance(self, result) -> TurnConditioning:
        """Apply one committed turn: exactly one atomic ECS_G step, then re-project.

        ``result`` must be a committed PrincipalResult conditioned by this
        loop's current conditioning, carrying its observation. Anything else
        is refused with W unchanged.
        """
        commit = getattr(result, "commit", None)
        observation = getattr(result, "observation", None)
        if commit is None or getattr(result, "failure", None) is not None:
            raise WorldModelError("UNCOMMITTED", "only a committed turn advances the world")
        digest = commit.digest
        if digest == self._applied:
            raise WorldModelError("REPLAYED", "commit already applied")
        if commit.conditioning != self._conditioning.digest:
            raise WorldModelError("STALE", "commit was not conditioned by the current world state")
        if type(observation) is not TurnObservation or observation.commit != digest:
            raise WorldModelError("OBSERVATION", "observation must be bound to the commit")
        x, y = self.drive_map.drive(observation)
        try:
            self._state.step(x, y, self.learning_rate)
        except ECSGError as exc:
            raise WorldModelError("REFUSED", str(exc)) from exc
        self._applied = digest
        self.transitions += 1
        self._conditioning = self._project()
        return self._conditioning

    def checkpoint(self):
        """(ECS_G snapshot bytes, last applied commit digest): everything needed to continue."""
        return self._state.snapshot(), self._applied

    def s3(self):
        return self._state.s3()
