"""A tiny deterministic corridor environment with replay identity. RESEARCH_ONLY. NO_CLAIM.

* **State**: ``(position, step, done)`` in a corridor of ``length`` cells; the goal cell is fixed per world.
* **Action**: ``LEFT``, ``STAY`` or ``RIGHT``.
* **Transition**: the position moves by -1, 0 or +1, clamped to the corridor; the episode ends at the goal or after
  ``2 * length`` steps.
* **Feedback**: +10 on reaching the goal, -1 for every other step.
* **Terminal evaluation**: an episode's score is the sum of its feedback, recomputed by the environment from the
  actions alone (:func:`replay`); a recorded trajectory is verified step by step against that replay (:func:`verify`).
* **Replay identity**: a hash chain over the spec, the world and every ``(state, action, feedback, next state)``.

A world is a string identifier; its start and goal derive from the SHA-256 of ``spec digest || world id``. The agent
observes only the direction to the goal and a distance bucket (:func:`observation`): it never sees the world id.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json

ENVIRONMENT_SCHEMA = "elpis.research.evolution-fitness.corridor.v1"
GOAL_FEEDBACK, STEP_FEEDBACK = 10, -1


class Action(str, Enum):
    LEFT = "LEFT"
    STAY = "STAY"
    RIGHT = "RIGHT"


_MOVE = {Action.LEFT: -1, Action.STAY: 0, Action.RIGHT: 1}


def _digest(domain: str, payload) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(domain.encode() + b"\0" + body).hexdigest()


@dataclass(frozen=True)
class EnvironmentSpec:
    length: int

    def __post_init__(self):
        if type(self.length) is not int or not 3 <= self.length <= 64:
            raise ValueError("corridor length in 3..64")

    @property
    def max_steps(self) -> int:
        return 2 * self.length

    @property
    def digest(self) -> str:
        return _digest(ENVIRONMENT_SCHEMA, {"length": self.length, "goal_feedback": GOAL_FEEDBACK,
                                            "step_feedback": STEP_FEEDBACK, "max_steps": self.max_steps})


@dataclass(frozen=True)
class State:
    position: int
    step: int
    done: bool


@dataclass(frozen=True)
class World:
    spec: EnvironmentSpec
    world_id: str

    @property
    def layout(self) -> tuple[int, int]:
        """``(start, goal)``, distinct cells, from the SHA-256 of the spec and the world id."""
        raw = hashlib.sha256(bytes.fromhex(self.spec.digest) + self.world_id.encode()).digest()
        start = raw[0] % self.spec.length
        goal = (start + 1 + raw[1] % (self.spec.length - 1)) % self.spec.length
        return start, goal

    def initial(self) -> State:
        return State(self.layout[0], 0, False)


def observation(world: World, state: State) -> str:
    """What the agent sees: the direction to the goal and the distance bucket 0..3 (``"-1:2"``, say)."""
    delta = world.layout[1] - state.position
    return f"{(delta > 0) - (delta < 0)}:{min(abs(delta), 3)}"


OBSERVATIONS = tuple(f"{s}:{b}" for s in (-1, 0, 1) for b in range(4) if (s == 0) == (b == 0))


def transition(world: World, state: State, action: Action) -> tuple[State, int]:
    if state.done:
        raise ValueError("the episode is over")
    position = min(max(state.position + _MOVE[Action(action)], 0), world.spec.length - 1)
    step = state.step + 1
    reached = position == world.layout[1]
    done = reached or step >= world.spec.max_steps
    return State(position, step, done), GOAL_FEEDBACK if reached else STEP_FEEDBACK


@dataclass(frozen=True)
class Step:
    state: State
    action: Action
    feedback: int
    next_state: State
    chain: str


@dataclass(frozen=True)
class Trajectory:
    world: World
    steps: tuple[Step, ...]

    @property
    def score(self) -> int:
        return sum(s.feedback for s in self.steps)

    @property
    def identity(self) -> str:
        return self.steps[-1].chain if self.steps else _origin(self.world)


def _origin(world: World) -> str:
    return _digest(ENVIRONMENT_SCHEMA + ".origin", {"spec": world.spec.digest, "world": world.world_id})


def _link(previous: str, state: State, action: Action, feedback: int, after: State) -> str:
    return _digest(ENVIRONMENT_SCHEMA + ".step", [previous, [state.position, state.step, state.done], action.value,
                                                   feedback, [after.position, after.step, after.done]])


def _advance(world: World, state: State, chain: str, action) -> tuple[State, str, Step]:
    action = Action(action)
    after, feedback = transition(world, state, action)
    chain = _link(chain, state, action, feedback, after)
    return after, chain, Step(state, action, feedback, after, chain)


def run(world: World, policy) -> Trajectory:
    """One episode of ``policy`` (observation -> Action) from the world's start to termination."""
    state, chain, steps = world.initial(), _origin(world), []
    while not state.done:
        state, chain, step = _advance(world, state, chain, policy(observation(world, state)))
        steps.append(step)
    return Trajectory(world, tuple(steps))


def replay(world: World, actions) -> Trajectory:
    """The trajectory the environment produces for exactly these actions, start to termination."""
    state, chain, steps = world.initial(), _origin(world), []
    for action in actions:
        if state.done:
            raise ValueError("actions beyond the end of the episode")
        state, chain, step = _advance(world, state, chain, action)
        steps.append(step)
    if not state.done:
        raise ValueError("the actions end before the episode does")
    return Trajectory(world, tuple(steps))


def verify(trajectory: Trajectory) -> bool:
    """Whether a recorded trajectory is exactly the environment's replay of its own actions."""
    try:
        return replay(trajectory.world, [s.action for s in trajectory.steps]) == trajectory
    except ValueError:
        return False
