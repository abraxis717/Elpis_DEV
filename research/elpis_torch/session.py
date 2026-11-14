"""Volatile, session-local CGC fast weights over an immutable checkpoint.

    W_active = W_base + dW_session                 for every CGC core weight, and for nothing else

``W_base`` is the frozen model's own tensor: the session reads it and never writes it. ``dW_session`` is a
separate tensor the session allocates at open (all zeros) and discards at reset or close. Every forward pass of
a session goes through ``torch.func.functional_call`` with ``W_base + dW`` substituted for the CGC core weights,
so the module's parameters are never touched, and two sessions over one model share nothing mutable.

What may teach the fast weights
    Only genuinely observed next tokens: tokens the caller ``observe``s into the session's stream, or an explicit
    supervised ``SupportSet``. Tokens the session generated itself are recorded as GENERATED and are never a
    target; they remain context. No self-supervised or self-evaluated signal exists here.

What an update may do
    One update is a bounded, atomic transaction: a candidate ``dW`` is computed out of place, clipped, projected
    onto the delta-norm ball, validated finite, tried on the support, and only then committed. Any failure
    leaves the session exactly as it was and is reported as a refusal (``AdaptResult.status``).

Attention caches
    The KV cache of the session's stream holds hidden states computed under the weights of the moment. A
    committed update changes those weights, so the cache is dropped at commit and rebuilt from the stream's
    tokens under the new weights before its next use. The weights tag (session identity, version) makes any
    older cache unusable: ``ElpisTorchLM.forward`` refuses it with ``StaleCacheError``.

Nothing here is written to storage: there is no save, export or state-dict method, and pickling or copying a
session is refused. See docs/research/elpis_torch/DESIGN.md, "Zero-write properties", for what is verified in
code and what only the host can enforce.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch
from torch.func import functional_call

from .model import ContextLengthError, ElpisTorchLM, KVCache, next_token_loss, parameter_digest, sample_next

FAST_WEIGHT_SUFFIX = "cgc.core.weight"


class SessionError(RuntimeError):
    """A session contract was violated (closed session, unfrozen model, serialization attempt, ...)."""


class BaseMutatedError(SessionError):
    """The immutable checkpoint's parameters no longer match the digest taken when the session opened."""


class Provenance(enum.Enum):
    OBSERVED = "observed"      # supplied by the caller as evidence: may be a learning target
    GENERATED = "generated"    # sampled by this session: context only, never a target


class Status(enum.Enum):
    APPLIED = "applied"
    REFUSED_LEARNING_DISABLED = "refused: learning is disabled for this session"
    REFUSED_NO_FAST_WEIGHTS = "refused: the model has no CGC core weights"
    REFUSED_STEP_BUDGET = "refused: the session's update budget is exhausted"
    REFUSED_NO_OBSERVED_TARGET = "refused: no observed next-token target is available"
    REFUSED_NONFINITE_LOSS = "refused: the support loss is not finite"
    REFUSED_NONFINITE_GRADIENT = "refused: the gradient is not finite"
    REFUSED_NONFINITE_CANDIDATE = "refused: the candidate fast weights are not finite"
    ROLLED_BACK_NONFINITE_AFTER = "rolled back: the candidate's support loss is not finite"
    ROLLED_BACK_LOSS_INCREASED = "rolled back: the candidate increased the support loss"


@dataclass(frozen=True)
class SessionConfig:
    learning_enabled: bool = True
    lr: float = 0.05
    momentum: float = 0.0
    grad_clip: float = 1.0            # global gradient-norm ceiling over all fast weights
    max_delta_ratio: float = 0.5      # per tensor: ||dW|| <= max_delta_ratio * ||W_base|| (Frobenius)
    max_steps: int = 64               # committed updates per session; reset restores the budget
    reject_loss_increase: bool = False
    verify_base_on_close: bool = True

    def __post_init__(self):
        for name in ("lr", "grad_clip", "max_delta_ratio"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if type(self.max_steps) is not int or self.max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")


@dataclass(frozen=True)
class SupportSet:
    """An explicit supervised support set: token sequences whose next tokens are genuine observations.

    Build it with ``SupportSet.observed``. The caller asserts the provenance; the only thing the constructor
    can check is that the tokens are not the direct output of a session's ``generate``.
    """
    tokens: torch.Tensor
    mask: torch.Tensor | None
    source: str

    @classmethod
    def observed(cls, tokens, *, mask: torch.Tensor | None = None, source: str) -> "SupportSet":
        if isinstance(tokens, GeneratedTokens):
            raise SessionError("generated tokens are not observations and cannot form a support set")
        if not isinstance(source, str) or not source.strip():
            raise SessionError("a support set must name its source")
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise SessionError("support tokens must be [n, T] with T >= 2")
        if mask is not None and (mask.dtype != torch.bool or mask.shape != (tokens.shape[0], tokens.shape[1] - 1)):
            raise SessionError("support mask must be a bool tensor of shape [n, T-1]")
        return cls(tokens, mask, source)


class GeneratedTokens(tuple):
    """Token ids a session sampled. A distinct type so they cannot be passed off as a support set directly."""


@dataclass(frozen=True)
class AdaptResult:
    status: Status
    loss_before: float = math.nan
    loss_after: float = math.nan
    grad_norm: float = math.nan
    clipped: bool = False
    projected: bool = False
    delta_norm: float = 0.0            # global Frobenius norm of all fast-weight deltas after this call
    steps_used: int = 0

    @property
    def applied(self) -> bool:
        return self.status is Status.APPLIED


class VolatileSession:
    """One inference session: a token stream, its KV cache, and the session-owned CGC fast-weight deltas."""

    def __init__(self, model: ElpisTorchLM, config: SessionConfig = SessionConfig()):
        if not isinstance(model, ElpisTorchLM):
            raise SessionError("a session needs an ElpisTorchLM")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise SessionError("open a session only over a frozen checkpoint (training.freeze / load_checkpoint)")
        self._model, self.config = model, config
        self._base = {name: p for name, p in model.named_parameters() if name.endswith(FAST_WEIGHT_SUFFIX)}
        self._base_norm = {name: float(p.norm()) for name, p in self._base.items()}
        self._base_digest = parameter_digest(model)
        self._identity = object()       # unique per session object: no other session's cache can carry it
        self._version = 0
        self._closed = False
        self.cache_rebuilds = 0
        self._start()

    def _start(self) -> None:
        learn = self.config.learning_enabled
        self._delta = {name: torch.zeros_like(p) for name, p in self._base.items()} if learn else {}
        self._velocity = {name: torch.zeros_like(p) for name, p in self._delta.items()}
        self._steps = 0
        self._tokens: list[int] = []
        self._provenance: list[Provenance] = []
        self._cache: KVCache | None = None
        self._next_logits: torch.Tensor | None = None

    # -- contracts ----------------------------------------------------------------------------------------

    def __getstate__(self):
        raise SessionError("a session's volatile state is never serialized")

    def __reduce_ex__(self, protocol):
        raise SessionError("a session's volatile state is never serialized")

    def __deepcopy__(self, memo):
        raise SessionError("a session's volatile state is never copied")

    def __enter__(self) -> "VolatileSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _live(self) -> None:
        if self._closed:
            raise SessionError("the session is closed")

    @property
    def _tag(self) -> tuple:
        return (self._identity, self._version)

    @property
    def fast_weight_names(self) -> tuple:
        return tuple(self._base)

    @property
    def steps_used(self) -> int:
        return self._steps

    @property
    def tokens(self) -> tuple:
        return tuple(self._tokens)

    @property
    def provenance(self) -> tuple:
        return tuple(self._provenance)

    def delta_norm(self) -> float:
        return math.sqrt(sum(float(d.pow(2).sum()) for d in self._delta.values()))

    def volatile_bytes(self) -> dict:
        """The session's volatile memory: fixed by the model, bounded by ``max_seq_len``."""
        delta = sum(d.numel() * d.element_size() for d in (*self._delta.values(), *self._velocity.values()))
        return {"fast_weights": delta, "kv_cache": self._cache.nbytes() if self._cache else 0,
                "stream_tokens": len(self._tokens)}

    def verify_base_unchanged(self) -> None:
        if parameter_digest(self._model) != self._base_digest:
            raise BaseMutatedError("the checkpoint's parameters changed while a session was open")

    # -- execution under the active weights ---------------------------------------------------------------

    def _call(self, tokens: torch.Tensor, cache: KVCache | None = None, delta: dict | None = None):
        delta = self._delta if delta is None else delta
        active = {name: self._base[name] + d for name, d in delta.items()}
        return functional_call(self._model, active, (tokens,), {"cache": cache, "weights_tag": self._tag})

    def new_cache(self, batch_size: int) -> KVCache:
        """A cache bound to this session's current weights; it stops being accepted after the next update."""
        self._live()
        return self._model.new_cache(batch_size, weights_tag=self._tag)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Logits under ``W_base + dW``. Does not touch the session's own stream."""
        self._live()
        return self._call(tokens, cache)

    @torch.no_grad()
    def loss(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """Teacher-forced loss under the active weights (evaluation only: nothing is learned from it)."""
        self._live()
        return float(next_token_loss(self._call(tokens[:, :-1]), tokens, mask))

    # -- the session's stream -----------------------------------------------------------------------------

    @torch.no_grad()
    def _ensure_cache(self) -> None:
        """Rebuild the stream's cache from its tokens if the weights changed since it was written."""
        if self._cache is not None and self._cache.weights_tag == self._tag:
            return
        self._cache = self._model.new_cache(1, weights_tag=self._tag)
        self._next_logits = None
        if self._tokens:
            self._next_logits = self._call(torch.tensor([self._tokens]), self._cache)[0, -1]
            self.cache_rebuilds += 1

    @torch.no_grad()
    def _append(self, tokens: list[int], provenance: Provenance) -> torch.Tensor:
        if len(self._tokens) + len(tokens) > self._model.config.max_seq_len:
            raise ContextLengthError("the session stream would exceed max_seq_len; nothing was appended")
        self._ensure_cache()
        logits = self._call(torch.tensor([tokens]), self._cache)[0]
        self._tokens += tokens
        self._provenance += [provenance] * len(tokens)
        self._next_logits = logits[-1]
        return logits

    def observe(self, tokens) -> list[float]:
        """Append observed evidence to the stream.

        Returns the prequential loss of each observed token: its negative log-probability under the prediction
        the session made BEFORE seeing it (``nan`` for the very first token of the stream, which had no
        prediction). These tokens become eligible next-token targets for ``adapt_on_stream``.
        """
        self._live()
        if isinstance(tokens, GeneratedTokens):
            raise SessionError("generated tokens are not observations")
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise SessionError("observe needs at least one token")
        self._ensure_cache()
        before = self._next_logits
        logits = self._append(tokens, Provenance.OBSERVED)
        predictions = logits[:-1] if before is None else torch.cat((before[None], logits[:-1]))
        targets = torch.tensor(tokens if before is not None else tokens[1:], dtype=torch.long)
        nll = -torch.log_softmax(predictions.double(), dim=-1).gather(-1, targets[:, None])[:, 0]
        return ([] if before is not None else [math.nan]) + nll.tolist()

    def next_logits(self) -> torch.Tensor:
        """The next-token logits of the stream under the active weights (cache rebuilt first if stale)."""
        self._live()
        self._ensure_cache()
        if self._next_logits is None:
            raise SessionError("the stream is empty")
        return self._next_logits.clone()

    def generate(self, max_new_tokens: int, *, temperature: float = 0.0, top_k: int | None = None,
                 generator: torch.Generator | None = None) -> GeneratedTokens:
        """Continue the stream. The new tokens are recorded as GENERATED: context, never learning targets."""
        self._live()
        if max_new_tokens < 1:
            raise SessionError("max_new_tokens must be positive")
        if len(self._tokens) + max_new_tokens > self._model.config.max_seq_len:
            raise ContextLengthError("the continuation would exceed max_seq_len; nothing was generated")
        out = []
        for _ in range(max_new_tokens):
            token = int(sample_next(self.next_logits()[None], temperature=temperature, top_k=top_k,
                                    generator=generator)[0])
            self._append([token], Provenance.GENERATED)
            out.append(token)
        return GeneratedTokens(out)

    # -- bounded fast-weight adaptation -------------------------------------------------------------------

    def adapt(self, support: SupportSet, *, steps: int = 1) -> list[AdaptResult]:
        """Up to ``steps`` bounded updates on an explicit observed support set. Stops at the first refusal."""
        self._live()
        if not isinstance(support, SupportSet):
            raise SessionError("adapt needs a SupportSet (SupportSet.observed)")
        return self._adapt(support.tokens, support.mask, steps)

    def adapt_on_stream(self, *, steps: int = 1) -> list[AdaptResult]:
        """Up to ``steps`` bounded updates on the stream's own OBSERVED next tokens.

        A position is a target only if the token there was observed. Generated tokens stay in the input as
        context but contribute no loss.
        """
        self._live()
        observed = [p is Provenance.OBSERVED for p in self._provenance[1:]]
        if not any(observed):
            return [self._refusal(Status.REFUSED_NO_OBSERVED_TARGET)]
        return self._adapt(torch.tensor([self._tokens]), torch.tensor([observed]), steps)

    def _refusal(self, status: Status, **fields) -> AdaptResult:
        return AdaptResult(status, delta_norm=self.delta_norm(), steps_used=self._steps, **fields)

    def _adapt(self, tokens: torch.Tensor, mask: torch.Tensor | None, steps: int) -> list[AdaptResult]:
        results = []
        for _ in range(steps):
            results.append(self._update(tokens, mask))
            if not results[-1].applied:
                break
        return results

    def _update(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> AdaptResult:
        """One atomic update: nothing of the session changes unless every check passes."""
        c = self.config
        if not c.learning_enabled:
            return self._refusal(Status.REFUSED_LEARNING_DISABLED)
        if not self._base:
            return self._refusal(Status.REFUSED_NO_FAST_WEIGHTS)
        if self._steps >= c.max_steps:
            return self._refusal(Status.REFUSED_STEP_BUDGET)
        names = list(self._delta)
        trial = {name: self._delta[name].clone().requires_grad_(True) for name in names}
        with torch.enable_grad():
            loss = next_token_loss(self._call(tokens[:, :-1], delta=trial), tokens, mask)
            before = float(loss.detach())
            if not math.isfinite(before):
                return self._refusal(Status.REFUSED_NONFINITE_LOSS, loss_before=before)
            grads = torch.autograd.grad(loss, [trial[name] for name in names])
        grad_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
        if not math.isfinite(grad_norm):
            return self._refusal(Status.REFUSED_NONFINITE_GRADIENT, loss_before=before, grad_norm=grad_norm)
        scale = min(1.0, c.grad_clip / grad_norm) if grad_norm > 0 else 1.0
        candidate, velocity, projected = {}, {}, False
        with torch.no_grad():
            for name, grad in zip(names, grads):
                velocity[name] = c.momentum * self._velocity[name] + scale * grad
                value = self._delta[name] - c.lr * velocity[name]
                limit = c.max_delta_ratio * self._base_norm[name]
                norm = float(value.norm())
                if not math.isfinite(norm):      # an overflowed norm must refuse, never rescale to zero
                    return self._refusal(Status.REFUSED_NONFINITE_CANDIDATE, loss_before=before,
                                         grad_norm=grad_norm)
                if norm > limit:
                    value, projected = value * (limit / norm), True
                candidate[name] = value
            if not all(bool(torch.isfinite(v).all()) for v in (*candidate.values(), *velocity.values())):
                return self._refusal(Status.REFUSED_NONFINITE_CANDIDATE, loss_before=before, grad_norm=grad_norm)
            after = float(next_token_loss(self._call(tokens[:, :-1], delta=candidate), tokens, mask))
        fields = dict(loss_before=before, loss_after=after, grad_norm=grad_norm, clipped=scale < 1.0,
                      projected=projected)
        if not math.isfinite(after):
            return self._refusal(Status.ROLLED_BACK_NONFINITE_AFTER, **fields)
        if c.reject_loss_increase and after > before:
            return self._refusal(Status.ROLLED_BACK_LOSS_INCREASED, **fields)
        # Commit. The weights changed, so every cache written before this point is stale: drop the stream's
        # cache (rebuilt on next use) and advance the version so no older cache is ever accepted again.
        self._delta, self._velocity = candidate, velocity
        self._steps += 1
        self._version += 1
        self._cache, self._next_logits = None, None
        return AdaptResult(Status.APPLIED, delta_norm=self.delta_norm(), steps_used=self._steps, **fields)

    # -- lifetime -----------------------------------------------------------------------------------------

    def reset(self) -> None:
        """Discard every fast weight, the optimizer state, the stream and its cache: a fresh session."""
        self._live()
        self._version += 1
        self._start()

    def close(self) -> None:
        """End the session. Its volatile state is dropped, never exported; the base digest is re-verified."""
        if self._closed:
            return
        self._closed = True
        self._delta, self._velocity, self._tokens, self._provenance = {}, {}, [], []
        self._cache, self._next_logits = None, None
        self._version += 1
        if self.config.verify_base_on_close:
            self.verify_base_unchanged()
