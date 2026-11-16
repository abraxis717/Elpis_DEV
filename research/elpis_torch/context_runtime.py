"""R4: the context runtime of ``ErsaHistoryLM``. Engineering only: no learning rule, objective or selection rule
is changed here.

    exact_topk      top-K with a DETERMINISTIC tie rule (score descending, then the newer block first). Used by
                    the R3 reference path and by the tiled path, so both select the same blocks even on exact ties.
    tiled_select    the R3 scoring rule, evaluated over the index in tiles with a running exact top-K merge. It never
                    materializes the ``[B, T, horizon / block]`` score, distance and eligibility tensors.
    BlockCache      an optional, session-local, volatile cache of ENCODED historical blocks with a hard byte bound,
                    deterministic LRU eviction, purging of blocks that left the horizon, and invalidation whenever an
                    encoder or embedding parameter changes. Bypassed whenever gradients are enabled.
    RuntimeOptions  what a ``HistoryState`` carries to select the execution path (``None``: the R3 reference).
    ContextSession  bounded streaming inference over a frozen model: chunked prefill that keeps only the logits the
                    caller asks for, incremental decoding, continuation over calls, reset in place, close, budget
                    refusal, no serialization and no storage writes.

Exactness. With unchanged weights, the tiled path and the cache change only the order of floating-point
reductions (batched versus per-tile matrix products, a block encoded alone or in a batch). Parity is tested to
1e-10 in float64 (logits) with identical selected block ids; float32 differences are reported, not asserted
bitwise.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

NEG = float("-inf")


class BudgetError(MemoryError):
    """A session would exceed its declared context-memory budget. Refused at creation; nothing is allocated."""


class SessionClosedError(RuntimeError):
    """The session was closed (or never usable)."""


# -- deterministic exact top-K --------------------------------------------------------------------------------

def exact_topk(scores: torch.Tensor, ids: torch.Tensor, k: int):
    """The ``k`` best entries of ``scores [..., N]`` (``-inf``: ineligible) under the rule: higher score first,
    and among equal scores the larger block id (the newer block) first. ``ids`` ``[N]`` or ``[..., N]``.

    Returns ``(values [..., k], positions [..., k])``; positions index the last dimension. Fewer than ``k``
    eligible entries: the rest are ``-inf`` with position 0. The common case (no tie at the k-th score) costs one
    ``topk`` and a sort of ``k`` elements; rows with a tie at the boundary are resolved by a stable full sort.
    """
    n = scores.shape[-1]
    ids = ids.expand_as(scores) if ids.shape != scores.shape else ids
    if n == 0:
        shape = (*scores.shape[:-1], k)
        return scores.new_full(shape, NEG), torch.zeros(shape, dtype=torch.long, device=scores.device)
    width = min(k, n)
    values, positions = scores.topk(width, dim=-1)
    kth = values[..., -1:]
    # A row needs the full stable sort only if an unselected entry ties the k-th (finite) score.
    boundary = ((scores == kth) & torch.isfinite(kth)).sum(-1) > (values == kth).sum(-1)
    if bool(boundary.any()):
        rows = boundary.nonzero(as_tuple=True)
        sub_scores, sub_ids = scores[rows], ids[rows]
        order = torch.argsort(sub_ids, dim=-1, descending=True, stable=True)
        order = order.gather(-1, torch.argsort(sub_scores.gather(-1, order), dim=-1, descending=True, stable=True))
        positions = positions.clone()
        positions[rows] = order[..., :width]
        values = scores.gather(-1, positions)
    # Canonical order of the selection itself: score descending, then id descending (stable two-key sort).
    chosen_ids = ids.gather(-1, positions)
    order = torch.argsort(chosen_ids, dim=-1, descending=True, stable=True)
    order = order.gather(-1, torch.argsort(values.gather(-1, order), dim=-1, descending=True, stable=True))
    values, positions = values.gather(-1, order), positions.gather(-1, order)
    positions = positions.masked_fill(~torch.isfinite(values), 0)
    if width < k:
        pad = k - width
        values = torch.cat((values, values.new_full((*values.shape[:-1], pad), NEG)), dim=-1)
        positions = torch.cat((positions, positions.new_zeros((*positions.shape[:-1], pad))), dim=-1)
    return values, positions


def tiled_select(reader, u: torch.Tensor, segments, positions: torch.Tensor, *, block: int, slots: int, k: int,
                 tile: int, tile_elements: int = 0, counters: dict | None = None):
    """R3's learned selection, exact, over index ``segments``.

    Each segment is ``(keys [B, n, per, dim], ids [n], first_block)``: contiguous records holding blocks
    ``first_block, first_block + 1, ...`` in order (views of the index ring, never copies). For each tile: the R3
    score (``reader.scores``), the R3 eligibility (complete at the query and inside its horizon), then an exact
    top-K merge with the running best. Tiles that no query of the chunk can see are skipped by scalar bounds.
    The tile holds ``max(tile, tile_elements // (B T))`` records, so decoding (``T = 1``) scores the index in few
    tiles while a prefill chunk stays within ``O(max(B T tile, tile_elements))`` scratch, independent of the
    horizon. Keys may live on another device (a CPU index): each tile is copied to the query's device.
    Returns absolute block ids ``[B, T, k]`` (-1: none) and scores ``[B, T, k]`` (-inf: none).
    """
    batch, length = u.shape[0], u.shape[1]
    width = max(tile, tile_elements // max(1, batch * length))
    best = u.new_full((batch, length, k), NEG)
    best_ids = torch.full((batch, length, k), -1, dtype=torch.long, device=u.device)
    complete = (positions + 1) // block
    first_position, last_position = int(positions[0]), int(positions[-1])
    lowest = (first_position + 1) // block - slots              # the oldest block any query of the chunk may see
    highest = (last_position + 1) // block - 1                  # the newest block complete by the chunk's end
    for keys, ids, first_block in segments:
        count = ids.shape[0]
        for at in range(0, count, width):
            span_lo, span_hi = first_block + at, first_block + min(at + width, count) - 1
            if span_hi < lowest or span_lo > highest:
                continue
            tile_ids = ids[at:at + width].to(u.device)
            tile_keys = keys[:, at:at + width].to(u.device)
            distance = positions[:, None] - ((tile_ids + 1) * block - 1)[None, :]                     # [T, n]
            eligible = (tile_ids[None, :] >= 0) & (distance >= 0) & (tile_ids[None, :] >= complete[:, None] - slots)
            scores = reader.scores(u, tile_keys, distance).masked_fill(~eligible[None], NEG)
            if counters is not None:
                counters["records_scored"] = counters.get("records_scored", 0) + int(tile_ids.shape[0]) * length
            pool = torch.cat((best, scores), dim=-1)
            pool_ids = torch.cat((best_ids, tile_ids[None, None, :].expand(batch, length, -1)), dim=-1)
            best, where = exact_topk(pool, pool_ids, k)
            best_ids = torch.where(torch.isfinite(best), pool_ids.gather(-1, where), torch.full_like(where, -1))
    return best_ids, best


def ring_segments(index):
    """The occupied records of a ``BlockIndex`` as at most two contiguous views, in increasing block order:
    ``[(keys, ids, first_block), ...]``. Nothing is copied."""
    if index.next_block is None:
        return []
    held = min(index.next_block, index.slots)
    first = index.next_block - held
    start = first % index.slots
    head = min(held, index.slots - start)
    out = [(index.keys[:, start:start + head], index.ids[start:start + head], first)]
    if held > head:
        out.append((index.keys[:, :held - head], index.ids[:held - head], first + head))
    return out


# -- the encoded-block cache ----------------------------------------------------------------------------------

class BlockCache:
    """Encoded historical blocks ``[B, s + 1, width]`` keyed by absolute block id. Volatile and session-local.

    * Hard bound: ``capacity_bytes``; an entry larger than the capacity is never stored.
    * Deterministic eviction: least recently used first (access order is deterministic for a given stream).
    * ``purge_below(block)``: entries older than the horizon are dropped (no stale reference survives eviction
      from the token history or rollover).
    * Invalidation: a fingerprint of the embedding and encoder parameters (storage pointer, version counter,
      dtype, device) is checked on every use; any change clears the cache.
    * Never used while gradients are enabled: cached tensors carry no graph (``bypassed`` counts those calls).
    """

    def __init__(self, capacity_bytes: int):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a non-negative integer")
        self.capacity_bytes = capacity_bytes
        self.entries: OrderedDict = OrderedDict()
        self.bytes = 0
        self.fingerprint = None
        self.stats = {"hits": 0, "misses": 0, "inserts": 0, "evictions": 0, "purged": 0, "invalidations": 0,
                      "bypassed": 0, "rejected_too_large": 0}

    def __getstate__(self):
        raise TypeError("a block cache is volatile session state and is never serialized")

    @staticmethod
    def model_fingerprint(model) -> tuple:
        tensors = [model.embed.weight, *model.encoder.parameters()]
        return tuple((t.data_ptr(), t._version, t.dtype, str(t.device), tuple(t.shape)) for t in tensors)

    def check(self, model) -> None:
        fingerprint = self.model_fingerprint(model)
        if fingerprint != self.fingerprint:
            if self.entries:
                self.stats["invalidations"] += 1
            self.clear()
            self.fingerprint = fingerprint

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0

    def get(self, block: int):
        entry = self.entries.get(block)
        if entry is None:
            self.stats["misses"] += 1
            return None
        self.entries.move_to_end(block)
        self.stats["hits"] += 1
        return entry

    def put(self, block: int, states: torch.Tensor) -> None:
        size = states.numel() * states.element_size()
        if size > self.capacity_bytes:
            self.stats["rejected_too_large"] += 1
            return
        if block in self.entries:
            old = self.entries.pop(block)
            self.bytes -= old.numel() * old.element_size()
        while self.entries and self.bytes + size > self.capacity_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes -= evicted.numel() * evicted.element_size()
            self.stats["evictions"] += 1
        self.entries[block] = states.detach()
        self.bytes += size
        self.stats["inserts"] += 1

    def purge_below(self, block: int) -> None:
        for old in [b for b in self.entries if b < block]:
            entry = self.entries.pop(old)
            self.bytes -= entry.numel() * entry.element_size()
            self.stats["purged"] += 1


@dataclass
class RuntimeOptions:
    """Execution options a ``HistoryState`` carries. ``scorer``: "reference" (R3, full table) or "tiled".
    ``key_grad_blocks``: in training, recompute (with gradient) the index keys of resident blocks among the most
    recent ``key_grad_blocks`` blocks instead of using the detached stored keys (``-1``: all resident blocks)."""
    scorer: str = "reference"
    tile: int = 4096
    tile_elements: int = 1 << 21                 # scores per tile at least this many (B T tile), see tiled_select
    cache: BlockCache | None = None
    key_grad_blocks: int = 0
    counters: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.scorer not in ("reference", "tiled"):
            raise ValueError("scorer must be 'reference' or 'tiled'")
        if type(self.tile) is not int or self.tile < 1:
            raise ValueError("tile must be a positive integer")
        if type(self.tile_elements) is not int or self.tile_elements < 0:
            raise ValueError("tile_elements must be a non-negative integer")
        if type(self.key_grad_blocks) is not int or self.key_grad_blocks < -1:
            raise ValueError("key_grad_blocks must be -1 (all) or a non-negative integer")

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount


# -- the streaming session ------------------------------------------------------------------------------------

@dataclass
class PrefillResult:
    last_logits: torch.Tensor | None          # [B, vocab]: the prediction after the final input token
    probes: dict                               # absolute position -> [B, vocab], for requested positions only
    tokens: int


class ContextSession:
    """Bounded streaming inference over a FROZEN ``ErsaHistoryLM`` (or ``ErsaLM``).

    Memory held: the model's fixed-size state (ERSA cells, window rings, token history, block index) plus the
    optional block cache (bounded) plus per-chunk scratch (bounded by ``chunk`` and ``tile``). Logits are computed
    only at the positions the caller asks for. Nothing is written to storage; the session cannot be pickled or
    copied; ``close`` drops every tensor it holds.
    """

    def __init__(self, model, *, batch: int = 1, chunk: int | None = None, scorer: str = "tiled", tile: int = 4096,
                 cache_bytes: int = 0, history_device=None, index_device=None, budget_bytes: int | None = None,
                 origin: int = 0):
        from .history_model import ErsaHistoryLM
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("a context session needs a frozen model (eval mode, no parameter requires grad)")
        self.model = model
        self.history = isinstance(model, ErsaHistoryLM)
        self.chunk = chunk or model.ersa.chunk
        if not 1 <= self.chunk <= model.ersa.chunk:
            raise ValueError(f"chunk must be in [1, {model.ersa.chunk}] (the model's chunk limit)")
        self.batch, self.origin = batch, origin
        self._options = dict(scorer=scorer, tile=tile, cache_bytes=cache_bytes)
        self._placement = dict(history_device=history_device, index_device=index_device)
        if budget_bytes is not None:
            need = self.projected_bytes()
            if need["total"] > budget_bytes:
                raise BudgetError(f"projected context memory {need['total']} bytes exceeds the budget {budget_bytes}")
        self.budget_bytes = budget_bytes
        self.state = None
        self._closed = False
        self._last = None
        self._allocate()

    # -- lifetime ---------------------------------------------------------------------------------------------

    def _allocate(self) -> None:
        if self.history:
            self.state = self.model.new_state(self.batch, origin=self.origin, **self._placement)
            cache = BlockCache(self._options["cache_bytes"]) if self._options["cache_bytes"] else None
            self.state.runtime = RuntimeOptions(scorer=self._options["scorer"], tile=self._options["tile"],
                                                cache=cache)
        else:
            self.state = self.model.new_state(self.batch)
        self._last = None

    def reset(self) -> None:
        """A fresh stream in the same buffers: every state tensor zeroed in place, the cache emptied."""
        self._live()
        self.state.reset()
        if self.history and self.state.runtime.cache is not None:
            self.state.runtime.cache.clear()
        self._last = None

    def close(self) -> None:
        self.state, self._last, self._closed = None, None, True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        raise TypeError("a context session is volatile and is never serialized")

    def __reduce_ex__(self, protocol):
        raise TypeError("a context session is volatile and is never serialized")

    def __deepcopy__(self, memo):
        raise TypeError("a context session is volatile and is never copied")

    def _live(self) -> None:
        if self._closed or self.state is None:
            raise SessionClosedError("the session is closed")

    # -- accounting -------------------------------------------------------------------------------------------

    @property
    def position(self) -> int:
        self._live()
        return self.state.position

    def counters(self) -> dict:
        self._live()
        if not self.history:
            return {}
        out = dict(self.state.runtime.counters)
        if self.state.runtime.cache is not None:
            out.update({f"cache_{k}": v for k, v in self.state.runtime.cache.stats.items()})
            out["cache_bytes"] = self.state.runtime.cache.bytes
        return out

    def nbytes(self) -> dict:
        """Bytes held now: fixed state by tier, and the cache's current (not configured) size."""
        self._live()
        out = self.state.nbytes()
        if self.history and self.state.runtime.cache is not None:
            out = dict(out)
            out["block_cache"] = self.state.runtime.cache.bytes
            out["total"] = out["total"] + out["block_cache"]
        return out

    def projected_bytes(self) -> dict:
        """Upper bound of the session's context memory, from the configuration: fixed state + configured cache
        capacity + per-chunk scratch of the selected scorer. Used for budget refusal."""
        from .scaling import fixed_state_bytes, session_scratch_bytes
        fixed = fixed_state_bytes(self.model, self.batch)["total"]
        scratch = session_scratch_bytes(self.model, batch=self.batch, chunk=self.chunk,
                                        scorer=self._options["scorer"], tile=self._options["tile"])
        out = {"state": fixed, "block_cache": self._options["cache_bytes"], "scratch": scratch["total"]}
        out["total"] = sum(out.values())
        return out

    # -- streaming --------------------------------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, *, probes=None, keep_last: bool = True) -> PrefillResult:
        """Feed ``tokens [B, n]`` (any n) chunk by chunk. Returns the logits after the last token (``keep_last``)
        and at the requested absolute ``probes`` positions only; nothing else is kept."""
        self._live()
        if tokens.ndim != 2 or tokens.shape[0] != self.batch:
            raise ValueError("tokens must be [batch, n]")
        start = self.state.position
        wanted = sorted({int(p) for p in (probes or ())})
        if wanted and not all(start <= p < start + tokens.shape[1] for p in wanted):
            raise ValueError("probe positions must lie in this call's span")
        out = {}
        for at in range(0, tokens.shape[1], self.chunk):
            piece = tokens[:, at:at + self.chunk]
            first = start + at
            local = [p - first for p in wanted if first <= p < first + piece.shape[1]]
            last = at + piece.shape[1] == tokens.shape[1]
            if keep_last and last:
                local = sorted(set(local) | {piece.shape[1] - 1})
            index = torch.tensor(local, dtype=torch.long, device=piece.device) if local else \
                torch.zeros(0, dtype=torch.long, device=piece.device)
            logits = self.model(piece, self.state, logit_positions=index)
            for column, position in enumerate(local):
                if first + position in wanted:
                    out[first + position] = logits[:, column]
                if last and keep_last and position == piece.shape[1] - 1:
                    self._last = logits[:, column]
        return PrefillResult(self._last if keep_last else None, out, tokens.shape[1])

    @torch.no_grad()
    def step(self, token: torch.Tensor) -> torch.Tensor:
        """One token per stream ``[B]`` -> the next-token logits ``[B, vocab]``."""
        self._live()
        token = token.reshape(self.batch, 1)
        index = torch.zeros(1, dtype=torch.long, device=token.device)
        self._last = self.model(token, self.state, logit_positions=index)[:, 0]
        return self._last

    @torch.no_grad()
    def generate(self, count: int, *, temperature: float = 0.0, top_k: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Continue every stream by ``count`` tokens from the last logits. Unbounded by the horizon."""
        from .model import sample_next
        self._live()
        if self._last is None:
            raise ValueError("generate needs a prefill (or a step) first")
        out = []
        for _ in range(count):
            token = sample_next(self._last, temperature=temperature, top_k=top_k, generator=generator)
            out.append(token)
            self.step(token)
        return torch.stack(out, dim=1)
