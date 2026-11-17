"""R3: the historical substrate. Exact original tokens in a rolling ring, and a bounded ring of block keys.

Nothing here learns, ranks or decides relevance. ``TokenHistory`` keeps token ids; ``BlockIndex`` keeps whatever
key vectors the model computed for completed blocks, with their absolute block numbers. Which blocks are read is
decided by the model (``history_model.HistoryReader``); the host side only stores, evicts and gathers.

Coordinates (all absolute integers, never wrapped by the caller):

    position   p      token p is the p-th token of the stream (the stream starts at ``origin``)
    block      b      block b covers positions [b s, (b + 1) s); the stream origin is a multiple of s
    complete          block b is complete once position (b + 1) s - 1 has been appended
    slot              token p lives in token slot p mod capacity; block b in index slot b mod index_slots

Eviction is overwriting: appending token p overwrites token p - capacity, writing block b overwrites block
b - index_slots. Nothing else removes anything, nothing is concatenated, and every buffer is allocated once.
Batches are independent streams advancing in lock step: they share coordinates, never contents.
"""
from __future__ import annotations

import torch


class EvictedError(LookupError):
    """A position or block that the ring no longer holds (or never held) was requested."""


class TokenHistory:
    """The last ``capacity`` token ids of ``batch`` streams, int32, with absolute positions."""

    def __init__(self, batch: int, capacity: int, *, origin: int = 0, device=None):
        for name, value in (("batch", batch), ("capacity", capacity)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(origin) is not int or origin < 0:
            raise ValueError("origin must be a non-negative integer")
        self.batch, self.capacity, self.origin, self.position = batch, capacity, origin, origin
        self.tokens = torch.zeros(batch, capacity, dtype=torch.int32, device=device)

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def oldest(self) -> int:
        """The oldest absolute position still held."""
        return max(self.origin, self.position - self.capacity)

    def append(self, tokens: torch.Tensor) -> None:
        """Write ``[batch, n]`` token ids at the next ``n`` positions. Beyond ``capacity`` only the newest are held."""
        if tokens.ndim != 2 or tokens.shape[0] != self.batch:
            raise ValueError("append expects [batch, n] token ids for this history's batch")
        if tokens.shape[1] < 1:
            raise ValueError("append needs at least one token")
        if tokens.dtype not in (torch.int32, torch.int64) or int(tokens.min()) < 0:
            raise ValueError("token ids must be non-negative integers")
        if tokens.shape[1] > self.capacity:                 # the older ones would be evicted at once
            self.position += tokens.shape[1] - self.capacity
            tokens = tokens[:, -self.capacity:]
        length = tokens.shape[1]
        slots = (self.position + torch.arange(length, device=self.device)) % self.capacity
        self.tokens.index_copy_(1, slots, tokens.to(device=self.device, dtype=torch.int32))
        self.position += length

    def reset(self) -> None:
        """An empty history in the same buffer."""
        self.tokens.zero_()
        self.position = self.origin

    def read(self, start: int, length: int) -> torch.Tensor:
        """Token ids at absolute positions ``start .. start + length - 1``: ``[batch, length]`` int64."""
        if length < 0 or start < self.oldest or start + length > self.position:
            raise EvictedError(f"positions {start} .. {start + length - 1} are not all held "
                               f"(held: {self.oldest} .. {self.position - 1})")
        slots = (start + torch.arange(length, device=self.device)) % self.capacity
        return self.tokens[:, slots].long()

    def gather(self, positions: torch.Tensor) -> torch.Tensor:
        """Token ids at arbitrary absolute positions ``positions`` (any shape, shared by every stream):
        ``[batch, *positions.shape]`` int64. Refuses any position that is not held."""
        if positions.numel() and (int(positions.min()) < self.oldest or int(positions.max()) >= self.position):
            raise EvictedError("a requested position is outside the held history")
        return self.tokens[:, positions.to(self.device) % self.capacity].long()

    def nbytes(self) -> int:
        return self.tokens.numel() * self.tokens.element_size()


class BlockIndex:
    """A ring of ``slots`` index records per stream: ``keys [batch, slots, per_block, dim]`` and the absolute
    block number held in each slot (``-1``: empty). Records are written in increasing block order.

    Occupancy invariant (R4.1): ``count`` is the number of records written and still held, so the held blocks
    are exactly ``next_block - count .. next_block - 1`` in slots ``(block mod slots)``, whatever the stream's
    absolute origin. It equals ``(ids >= 0).sum()`` at all times (tested), without a scan."""

    def __init__(self, batch: int, slots: int, per_block: int, dim: int, *, dtype=torch.float32, device=None):
        for name, value in (("batch", batch), ("slots", slots), ("per_block", per_block), ("dim", dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.batch, self.slots = batch, slots
        self.keys = torch.zeros(batch, slots, per_block, dim, dtype=dtype, device=device)
        self.ids = torch.full((slots,), -1, dtype=torch.int64, device=device)
        self.next_block = None                       # the block number the next write must carry
        self.count = 0                               # R4.1: records written and still held (0 .. slots)

    def write(self, first_block: int, keys: torch.Tensor) -> None:
        """Store records for blocks ``first_block .. first_block + n - 1`` (``keys [batch, n, per_block, dim]``),
        overwriting the oldest. Blocks must arrive in order, without gaps or repeats."""
        count = keys.shape[1]
        if self.next_block is not None and first_block != self.next_block:
            raise ValueError(f"index records must be consecutive: expected block {self.next_block}, got {first_block}")
        if count > self.slots:                       # only the newest ``slots`` can be held; the rest are evicted
            keys, first_block, count = keys[:, -self.slots:], first_block + count - self.slots, self.slots
        blocks = first_block + torch.arange(count, device=self.ids.device)
        slots = blocks % self.slots
        self.keys.index_copy_(1, slots, keys.detach().to(self.keys.dtype))
        self.ids.index_copy_(0, slots, blocks)
        self.next_block = first_block + count
        self.count = min(self.count + count, self.slots)

    def reset(self) -> None:
        """An empty index in the same buffers."""
        self.keys.zero_()
        self.ids.fill_(-1)
        self.next_block = None
        self.count = 0

    def resident(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ids [n], keys [batch, n, per_block, dim])`` of the held records, oldest first."""
        held = self.ids >= 0
        ids = self.ids[held]
        order = torch.argsort(ids)
        return ids[order], self.keys[:, held][:, order]

    def nbytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.ids.numel() * self.ids.element_size()
