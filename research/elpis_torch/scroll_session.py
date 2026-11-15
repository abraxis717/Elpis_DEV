"""A volatile session over a scrolling-context model: last-block CGC fast weights, no cache invalidation.

    W_active = W_base + dW_session          for the FINAL block's CGC core weight only

Why only the final block. Every key, value, sink and compressed block the stream has written was computed
from the stream entering some block's attention. The final block's CGC acts after the final attention, so
none of that state depends on its weight (``lastblock.py``). Changing it therefore invalidates nothing: a
million tokens of rolling memory stay exactly as they are, bit for bit. An R0-style update of every block
would change the stream below later attentions and would require recomputing the whole history, which a
scrolling session cannot do (the tokens beyond the horizon are gone). ``fast_scope="all"`` is refused here.

What is learned from. The session keeps a bounded ring of the last ``learn_window`` positions: the final
block's pre-channel stream at that position and the token that followed it, with its provenance. An update
is the R0 bounded transaction on those pairs whose following token was OBSERVED. It never re-runs the model
and never touches the stream state. Generated tokens are context and never targets.

Nothing is written to storage, serialized or exported; the checkpoint digest is re-verified at close.
"""
from __future__ import annotations

import math

import torch
from torch.func import functional_call
from torch.nn import functional as F

from .lastblock import LAST_BLOCK_FAST_WEIGHT, last_block_logits
from .model import parameter_digest, sample_next
from .scroll import ScrollLM
from .session import (AdaptResult, BaseMutatedError, GeneratedTokens, Provenance, SessionConfig, SessionError, Status,
                      bounded_update)


class ScrollSession:
    def __init__(self, model: ScrollLM, config: SessionConfig = SessionConfig(fast_scope="last"), *,
                 learn_window: int = 256, origin: int = 0):
        if not isinstance(model, ScrollLM):
            raise SessionError("a scrolling session needs a ScrollLM")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise SessionError("open a session only over a frozen checkpoint (training.freeze)")
        if config.fast_scope != "last":
            raise SessionError("a scrolling session adapts the last block only: anything else would invalidate "
                               "rolling memory that can no longer be recomputed")
        if type(learn_window) is not int or learn_window < 1:
            raise SessionError("learn_window must be a positive integer")
        self._model, self.config, self._window, self._origin = model, config, learn_window, origin
        self._name = f"blocks.{len(model.blocks) - 1}.{LAST_BLOCK_FAST_WEIGHT}"
        named = dict(model.named_parameters())
        self._base = {self._name: named[self._name]} if self._name in named else {}
        self._base_norm = {name: float(p.norm()) for name, p in self._base.items()}
        self._base_digest = parameter_digest(model)
        self._closed = False
        self._start()

    def _start(self) -> None:
        hidden, dtype = self._model.config.d_model, getattr(torch, self._model.config.dtype)
        learn = self.config.learning_enabled
        self._delta = {name: torch.zeros_like(p) for name, p in self._base.items()} if learn else {}
        self._velocity = {name: torch.zeros_like(p) for name, p in self._delta.items()}
        self._steps = 0
        self.state = self._model.new_state(1, origin=self._origin)
        self._pre = torch.zeros(self._window, hidden, dtype=dtype)       # ring: pre-channel stream at position p
        self._next = torch.zeros(self._window, dtype=torch.long)         # the token that followed position p
        self._teach = torch.zeros(self._window, dtype=torch.bool)        # ... and whether it was OBSERVED
        self._last_pre: torch.Tensor | None = None
        self._next_logits: torch.Tensor | None = None

    # -- contracts ----------------------------------------------------------------------------------------

    def __getstate__(self):
        raise SessionError("a session's volatile state is never serialized")

    def __reduce_ex__(self, protocol):
        raise SessionError("a session's volatile state is never serialized")

    def __deepcopy__(self, memo):
        raise SessionError("a session's volatile state is never copied")

    def __enter__(self) -> "ScrollSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _live(self) -> None:
        if self._closed:
            raise SessionError("the session is closed")

    @property
    def fast_weight_names(self) -> tuple:
        return tuple(self._base)

    @property
    def steps_used(self) -> int:
        return self._steps

    @property
    def position(self) -> int:
        return self.state.position

    def delta_norm(self) -> float:
        return math.sqrt(sum(float(d.pow(2).sum()) for d in self._delta.values()))

    def volatile_bytes(self) -> dict:
        """Everything the session holds. Fixed at open: it does not grow with the stream."""
        size = lambda t: t.numel() * t.element_size()   # noqa: E731
        return {"stream_state": self.state.nbytes()["total"],
                "fast_weights": sum(size(t) for t in (*self._delta.values(), *self._velocity.values())),
                "learn_window": size(self._pre) + size(self._next) + size(self._teach)}

    def verify_base_unchanged(self) -> None:
        if parameter_digest(self._model) != self._base_digest:
            raise BaseMutatedError("the checkpoint's parameters changed while a session was open")

    def _weight(self, delta: dict | None = None):
        delta = self._delta if delta is None else delta
        return self._base[self._name] + delta[self._name] if delta else None

    # -- the stream ---------------------------------------------------------------------------------------

    @torch.no_grad()
    def _feed(self, tokens: list[int], provenance: Provenance) -> torch.Tensor:
        """Advance the stream. Returns the logits after each fed token ``[n, vocab]``."""
        chunk, out = self._model.scroll.chunk, []
        active = {self._name: self._weight()} if self._delta else {}
        teach = provenance is Provenance.OBSERVED
        for at in range(0, len(tokens), chunk):
            piece = tokens[at:at + chunk]
            first = self.state.position - self._origin              # stream index of piece[0]
            logits, pre = functional_call(self._model, active, (torch.tensor([piece]), self.state),
                                          {"return_pre": True})
            index = torch.arange(first, first + len(piece))
            # Position p's follower is the token at p + 1: piece[j] follows position first + j - 1.
            self._pre[index % self._window] = pre[0]
            self._teach[index % self._window] = False                # overwritten slots: no follower known yet
            followers = index - 1
            keep = followers >= 0
            self._next[followers[keep] % self._window] = torch.tensor(piece)[keep]
            self._teach[followers[keep] % self._window] = teach
            self._last_pre = pre[0, -1]
            out.append(logits[0])
        self._next_logits = out[-1][-1]
        return torch.cat(out)

    def observe(self, tokens) -> list[float]:
        """Append observed evidence. Returns each token's prequential loss (``nan`` for the stream's first)."""
        self._live()
        if isinstance(tokens, GeneratedTokens):
            raise SessionError("generated tokens are not observations")
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise SessionError("observe needs at least one token")
        before = self._next_logits
        logits = self._feed(tokens, Provenance.OBSERVED)
        predictions = logits[:-1] if before is None else torch.cat((before[None], logits[:-1]))
        targets = torch.tensor(tokens if before is not None else tokens[1:], dtype=torch.long)
        nll = -torch.log_softmax(predictions.double(), dim=-1).gather(-1, targets[:, None])[:, 0]
        return ([] if before is not None else [math.nan]) + nll.tolist()

    def next_logits(self) -> torch.Tensor:
        self._live()
        if self._next_logits is None:
            raise SessionError("the stream is empty")
        return self._next_logits.clone()

    def generate(self, max_new_tokens: int, *, temperature: float = 0.0, top_k: int | None = None,
                 generator: torch.Generator | None = None) -> GeneratedTokens:
        """Continue the stream. The new tokens are GENERATED: context, never learning targets."""
        self._live()
        if max_new_tokens < 1:
            raise SessionError("max_new_tokens must be positive")
        out = []
        for _ in range(max_new_tokens):
            token = int(sample_next(self.next_logits()[None], temperature=temperature, top_k=top_k,
                                    generator=generator)[0])
            self._feed([token], Provenance.GENERATED)
            out.append(token)
        return GeneratedTokens(out)

    # -- bounded last-block adaptation --------------------------------------------------------------------

    def support_size(self) -> int:
        """Observed (position, next token) pairs currently in the learning window."""
        return int(self._teach.sum())

    def adapt_on_stream(self, *, steps: int = 1) -> list[AdaptResult]:
        """Up to ``steps`` bounded updates on the learning window's OBSERVED next tokens.

        The stream state is not read, not written and not invalidated.
        """
        self._live()
        if not int(self._teach.sum()):
            return [self._result(Status.REFUSED_NO_OBSERVED_TARGET, {})]
        pre, targets = self._pre[self._teach], self._next[self._teach]

        def loss_of(delta):
            return F.cross_entropy(last_block_logits(self._model, pre, self._weight(delta)), targets)

        results = []
        for _ in range(steps):
            status, fields, candidate, velocity = bounded_update(
                self.config, self._base_norm, self._delta, self._velocity, self._steps, loss_of)
            if status is Status.APPLIED:
                self._delta, self._velocity = candidate, velocity
                self._steps += 1
                if self._last_pre is not None:
                    with torch.no_grad():
                        self._next_logits = last_block_logits(self._model, self._last_pre, self._weight())
            results.append(self._result(status, fields))
            if status is not Status.APPLIED:
                break
        return results

    def _result(self, status: Status, fields: dict) -> AdaptResult:
        return AdaptResult(status, delta_norm=self.delta_norm(), steps_used=self._steps, **fields)

    # -- lifetime -----------------------------------------------------------------------------------------

    def reset(self) -> None:
        """A fresh session: zero fast weights, an empty stream, an empty learning window."""
        self._live()
        self._start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._delta, self._velocity, self.state = {}, {}, None
        self._pre = self._next = self._teach = self._last_pre = self._next_logits = None
        if self.config.verify_base_on_close:
            self.verify_base_unchanged()
