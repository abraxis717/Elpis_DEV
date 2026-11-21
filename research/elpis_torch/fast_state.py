"""R12: a private, volatile, episode-local fast state for ONE expert-local specialist core. RESEARCH_ONLY.

    Expert(h) = SwiGLU(h) + g * CGC(h; W_active)            W_active = W_base + dW_session

What changes, and what never does
    Only ``dW_session`` changes. It is a tensor this object allocates (zeros, in the core's precision) for the
    specialist CORE parameters of one expert:
    - the micro CGC's ``cgc.core.weight`` ``[5, 3, 8]`` (120 values, 480 bytes in float32);
    - or the four core tensors of R11's matched GELU control (120 values).
    The model's parameters, its ``state_dict`` and every file stay untouched. Every computation under the fast state
    substitutes ``W_base + dW`` through ``torch.func.functional_call`` (the R0 session's convention), so the block's
    own routed channel runs: its ``SparseChannel`` routes, and its ``ChannelExpert`` computes FFN + gate * specialist
    with the substituted core. Nothing bypasses the expert.

The update
    ``session.bounded_update``, unchanged. It takes the gradient of the support loss with respect to ``dW``, applies a
    global gradient-norm clip and an SGD step, projects onto a delta-norm ball, checks finiteness, and then commits
    atomically or refuses. R12's defaults are R11 Stage B's protocol: lr 0.08, clip 2.0, 8 steps, no momentum. The
    delta bound equals the largest displacement those 8 steps can reach (8 x 0.08 x 2.0 = 1.28). It is therefore a
    hard safety limit that cannot bind inside the protocol, and the update is exactly R11's.

Where it may act
    Only in the LAST block. That block's ERSA state, attention window and evidence read are all computed before its
    channel, which feeds only the readout, so no recurrent or cached state depends on its core. This is the R1
    ``lastblock`` argument, tested here for ``ErsaLM`` and for the R11 stack.
    - Any other block is REFUSED. Its channel feeds every later block's state, so a changed core would need a replay
      of the history, which this contract does not allow.
    - The block's channel must hold exactly one routed expert and no shared expert: a fixed assignment, labelled a
      laboratory condition. Learned routing among several adapted experts is out of scope.

Teaching signal
    ``ObservedSupport``: explicit (input, target) pairs whose targets are observations, named by a source. The caller
    asserts the provenance. Nothing else teaches: no model output, no generated token, no confidence, no ERSA state.

Lifetime
    - **Identity.** Each state carries an explicit session identity, the model's parameter digest, its layer and
      expert identity, a version and a step counter.
    - **Reset.** ``reset`` restores the base exactly (``dW = 0``).
    - **Close.** ``close`` drops the state and re-verifies the model digest.
    - **No persistence.** The state is never pickled, copied or written anywhere.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .cgc_specialist import specialist_core_names
from .lastblock import capture_pre
from .model import parameter_digest
from .session import AdaptResult, SessionConfig, Status, bounded_update


class FastStateError(RuntimeError):
    """A fast-state contract was violated (closed state, unfrozen model, a block other than the last, ...)."""


@dataclass(frozen=True)
class FastStateConfig:
    learning_enabled: bool = True
    lr: float = 0.08                  # R11 Stage B / R97W inner learning rate
    grad_clip: float = 2.0            # global L2 gradient-norm ceiling over the core's tensors
    max_steps: int = 8                # committed updates per session; reset restores the budget
    delta_bound: float | None = None  # Frobenius bound per core tensor; None: max_steps * lr * grad_clip (unreachable)
    reject_loss_increase: bool = False

    def __post_init__(self):
        for name in ("lr", "grad_clip"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite")
        if type(self.max_steps) is not int or self.max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        if self.delta_bound is not None and not (math.isfinite(self.delta_bound) and self.delta_bound > 0):
            raise ValueError("delta_bound must be None or positive and finite")

    def bound(self) -> float:
        return self.max_steps * self.lr * self.grad_clip if self.delta_bound is None else float(self.delta_bound)

    def session_config(self) -> SessionConfig:
        """The R0 session's transaction parameters. ``bounded_update`` bounds each delta by ``max_delta_ratio *
        base_norm[name]``; the fast state passes ``base_norm[name] = bound()`` with ratio 1, i.e. an absolute bound."""
        return SessionConfig(learning_enabled=self.learning_enabled, lr=self.lr, momentum=0.0,
                             grad_clip=self.grad_clip, max_delta_ratio=1.0, max_steps=self.max_steps,
                             reject_loss_increase=self.reject_loss_increase, fast_scope="last")

    def replace(self, **changes) -> "FastStateConfig":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class ObservedSupport:
    """Observed teaching pairs for one episode: model inputs that carry NO target, and the observed targets.

    Build it with ``ObservedSupport.observed``. The caller asserts the provenance (``source``). The constructor can
    only check shapes and finiteness; ``fast_episodes`` additionally checks that no target value is present in the
    inputs (the leakage contract)."""
    inputs: torch.Tensor
    targets: torch.Tensor
    source: str

    @classmethod
    def observed(cls, inputs: torch.Tensor, targets: torch.Tensor, *, source: str) -> "ObservedSupport":
        if not isinstance(source, str) or not source.strip():
            raise FastStateError("observed support must name its source")
        if inputs.shape[:-1] != targets.shape[:-1] or inputs.ndim != 3:
            raise FastStateError("support inputs [1, n, features] and targets [1, n, outputs] must align")
        if not bool(torch.isfinite(targets).all()):
            raise FastStateError("observed targets must be finite")
        return cls(inputs, targets, source)


def last_channel(model):
    """The last block's routed channel and its single expert, or a refusal (the contract's laboratory condition)."""
    block = model.blocks[-1]
    channel = getattr(block, "sparse", None)
    if channel is None:
        raise FastStateError("the last block has no routed channel (a dense channel is the R0/R1 session's domain)")
    if len(channel.experts) != 1 or channel.shared is not None:
        raise FastStateError("the fast state needs a fixed assignment: exactly one routed expert and no shared "
                             "expert in the last block (learned routing among adapted experts is out of scope)")
    return block, channel, channel.experts[0]


def head(model, x: torch.Tensor) -> torch.Tensor:
    """The model's output from the last block's stream: the R11 stack's readout or an ``ErsaLM``'s logits."""
    out = model.final_norm(x)
    if hasattr(model, "readout"):
        return model.readout(out)
    return F.linear(out, model.embed.weight) if model.head is None else model.head(out)


class SpecialistFastState:
    """One session's private fast state for the last block's specialist core (see the module docstring)."""

    def __init__(self, model, *, session: str, config: FastStateConfig = FastStateConfig(), block: int | None = None):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise FastStateError("open a fast state only over a frozen model (eval mode, no parameter requires grad)")
        last = len(model.blocks) - 1
        if block is not None and block != last:
            raise FastStateError(f"block {block} is not the last block ({last}): its channel feeds later blocks' "
                                 "recurrent state, and a changed core would need an unbounded replay (refused)")
        if not isinstance(session, str) or not session.strip():
            raise FastStateError("a fast state needs an explicit session identity")
        self._model = model
        self._block, self._channel, self._expert = last_channel(model)
        names = specialist_core_names(self._expert)
        if not names:
            raise FastStateError("the expert has no specialist core (an FFN-only expert cannot be adapted)")
        params = dict(self._expert.named_parameters())
        self._base = {n: params[n] for n in names}
        self.layer, self.expert_index, self.session, self.config = last, 0, session, config
        self._transaction = config.session_config()
        self._bound = {n: config.bound() for n in names}
        self._digest = parameter_digest(model)
        self._version = 0
        self._closed = False
        self._start()

    def _start(self) -> None:
        self._delta = {n: torch.zeros_like(p) for n, p in self._base.items()}
        self._velocity = {n: torch.zeros_like(p) for n, p in self._base.items()}
        self._steps = 0

    # -- contracts ---------------------------------------------------------------------------------------------

    def __getstate__(self):
        raise FastStateError("a fast state is volatile and is never serialized")

    def __reduce_ex__(self, protocol):
        raise FastStateError("a fast state is volatile and is never serialized")

    def __deepcopy__(self, memo):
        raise FastStateError("a fast state is never copied")

    def _live(self) -> None:
        if self._closed:
            raise FastStateError("the fast state is closed")

    @property
    def names(self) -> tuple:
        return tuple(self._base)

    @property
    def steps_used(self) -> int:
        return self._steps

    @property
    def version(self) -> int:
        return self._version

    def identity(self) -> dict:
        return {"session": self.session, "model_digest": self._digest, "layer": self.layer,
                "expert": self.expert_index, "version": self._version, "steps": self._steps,
                "core": list(self._base)}

    def delta(self) -> dict:
        """A detached copy of the current ``dW`` (the state itself is never handed out)."""
        return {n: d.detach().clone() for n, d in self._delta.items()}

    def delta_norm(self) -> float:
        return math.sqrt(sum(float(d.pow(2).sum()) for d in self._delta.values()))

    def nbytes(self) -> dict:
        size = lambda ts: sum(t.numel() * t.element_size() for t in ts)   # noqa: E731
        out = {"delta": size(self._delta.values()), "velocity": size(self._velocity.values())}
        out["total"] = out["delta"] + out["velocity"]
        return out

    def verify_base_unchanged(self) -> None:
        if parameter_digest(self._model) != self._digest:
            raise FastStateError("the model's parameters changed while a fast state was open")

    # -- execution under W_base + dW -----------------------------------------------------------------------------

    def active(self, delta: dict | None = None) -> dict:
        delta = self._delta if delta is None else delta
        return {n: self._base[n] + delta[n] for n in self._base}

    def overrides(self, delta: dict | None = None) -> dict:
        """``W_active`` under the model's own parameter names (for a full ``functional_call`` of the model)."""
        prefix = f"blocks.{self.layer}.sparse.experts.0."
        return {prefix + n: w for n, w in self.active(delta).items()}

    def channel(self, h: torch.Tensor, delta: dict | None = None) -> torch.Tensor:
        """The last block's routed channel on ``h`` (its channel-norm output) under ``W_base + dW``."""
        self._live()
        return functional_call(self._channel, {f"experts.0.{n}": w for n, w in self.active(delta).items()}, (h,))

    def output(self, pre: torch.Tensor, delta: dict | None = None) -> torch.Tensor:
        """The model's output from the last block's pre-channel stream ``pre`` (cached path)."""
        return head(self._model, pre + self.channel(self._block.channel_norm(pre), delta))

    # -- the bounded update ----------------------------------------------------------------------------------------

    def adapt(self, loss_of, steps: int = 1) -> list[AdaptResult]:
        """Up to ``steps`` bounded transactions. ``loss_of(delta)`` is the OBSERVED support's loss under
        ``W_base + delta``. Stops at the first refusal; nothing changes unless a step is APPLIED."""
        self._live()
        results = []
        for _ in range(steps):
            status, fields, candidate, velocity = bounded_update(
                self._transaction, self._bound, self._delta, self._velocity, self._steps, loss_of)
            if status is not Status.APPLIED:
                results.append(AdaptResult(status, delta_norm=self.delta_norm(), steps_used=self._steps, **fields))
                break
            self._delta, self._velocity = candidate, velocity
            self._steps += 1
            self._version += 1
            results.append(AdaptResult(Status.APPLIED, delta_norm=self.delta_norm(), steps_used=self._steps, **fields))
        return results

    def support_loss(self, support: ObservedSupport, pre: torch.Tensor):
        """``loss_of`` for an observed support whose last-block stream ``pre`` was computed (once, without a graph)
        from ``support.inputs``: the mean squared error of the cached-path output against the observed targets."""
        if not isinstance(support, ObservedSupport):
            raise FastStateError("adaptation needs ObservedSupport.observed(...)")
        targets = support.targets

        def loss_of(delta):
            return (self.output(pre, delta) - targets).pow(2).mean()
        return loss_of

    # -- lifetime ----------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        """Discard ``dW``, the optimizer state and the step count: exactly the base behaviour again."""
        self._live()
        self._version += 1
        self._start()

    def close(self) -> None:
        """End the session: the volatile state is dropped, never exported, and the model digest is re-verified."""
        if self._closed:
            return
        self._closed = True
        self._delta, self._velocity = {}, {}
        self._version += 1
        self.verify_base_unchanged()


@torch.no_grad()
def frozen_pass(model, tokens: torch.Tensor):
    """``(output, pre)``: the frozen model's output and its last block's pre-channel stream for ``tokens``."""
    with capture_pre(model) as captured:
        out = model(tokens, model.new_state(tokens.shape[0]))
    return out, captured.pre
