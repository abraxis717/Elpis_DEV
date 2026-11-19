"""R5: the exact history with identity. A FIFO ring of exactly ``capacity`` token ids and a bounded ring of block
descriptors (the coarse index), where every live token and block has an address that is validated before any
read. Nothing here learns or ranks; the model computes descriptors and chooses blocks.

Identity of a block address (all integers, never wrapped by the caller):

    session      unique per ring object (a process-wide counter): an address never crosses sessions
    epoch        incremented by ``reset``: an address never survives a reset
    block        absolute block number b, covering positions [b s, (b + 1) s); the origin is a multiple of s
    slot         b mod slots (index) and p mod capacity (tokens): where the record lives
    generation   b // slots: how many times that slot has been reused

A block is LIVE for a query at absolute position ``at`` iff it is complete at ``at`` ((b + 1) s <= at + 1), all
of its tokens AND its ``overlap`` tokens of left context are still held (max(origin, b s - overlap) >= at + 1 -
capacity), and the index slot still carries b. Requiring the left context keeps every reread identical to the
encoding the descriptor was computed from, whatever the chunking; it costs one block at the edge of the ring. A reread that fails
any check raises ``StaleAddressError`` and is counted; nothing is ever read from a reused slot. FIFO only:
appending token p overwrites token p - capacity, writing block b overwrites block b - slots. Every buffer is
allocated once; nothing grows with the stream.

R8: an optional second index tier, the FINE KEYS of every written block (``[batch, slots, block, fine_dim]``), lives
in the same slots as the descriptors: written by the same ``write``, overwritten with them, zeroed by ``reset``, and
read (``read_fine``) only through the same validation as a reread. Like the descriptors it is derived from the
block's exact tokens; the tokens stay the authority.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from .history import EvictedError


class StaleAddressError(EvictedError):
    """An address whose session, epoch, slot generation or liveness does not match the ring: refused, never read."""


_SESSIONS = itertools.count(1)


@dataclass(frozen=True)
class BlockAddress:
    session: int
    epoch: int
    block: int
    slot: int
    generation: int
    start: int
    end: int


class EvidenceRing:
    """``batch`` lock-step streams: tokens ``[batch, capacity]`` int32 and descriptors ``[batch, slots, per_block,
    dim]`` with the absolute block id held in each slot (``-1``: empty). ``slots = capacity // block``. With
    ``fine_dim`` (R8), also the fine keys ``[batch, slots, block, fine_dim]`` in ``fine_dtype``."""

    def __init__(self, batch: int, *, capacity: int = 1 << 20, block: int = 32, per_block: int = 4, dim: int = 64,
                 origin: int = 0, overlap: int = 0, key_dtype=torch.float32, device=None,
                 fine_dim: int | None = None, fine_dtype=torch.float32):
        for name, value in (("batch", batch), ("capacity", capacity), ("block", block), ("per_block", per_block),
                            ("dim", dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if capacity % block:
            raise ValueError("capacity must be a multiple of block")
        if type(origin) is not int or origin < 0 or origin % block:
            raise ValueError("origin must be a non-negative multiple of block")
        if type(overlap) is not int or not 0 <= overlap < capacity - block:
            raise ValueError("overlap must be a non-negative integer well below the capacity")
        if fine_dim is not None and (type(fine_dim) is not int or fine_dim < 1):
            raise ValueError("fine_dim must be a positive integer (or None: no fine-key tier)")
        self.batch, self.capacity, self.block, self.overlap = batch, capacity, block, overlap
        self.slots = capacity // block
        self.tokens = torch.zeros(batch, capacity, dtype=torch.int32, device=device)
        self.keys = torch.zeros(batch, self.slots, per_block, dim, dtype=key_dtype, device=device)
        self.ids = torch.full((self.slots,), -1, dtype=torch.int64, device=device)
        self.fine = None if fine_dim is None else \
            torch.zeros(batch, self.slots, block, fine_dim, dtype=fine_dtype, device=device)
        self.session, self.epoch = next(_SESSIONS), 0
        self.origin = self.position = origin
        self.next_block, self.count = origin // block, 0
        self.stats = {"stale_rejections": 0, "rereads": 0, "written_blocks": 0}
        if self.fine is not None:
            self.stats["fine_reads"] = 0

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def oldest(self) -> int:
        """The oldest absolute position still held."""
        return max(self.origin, self.position - self.capacity)

    # -- identity ----------------------------------------------------------------------------------------------

    def address(self, block: int) -> BlockAddress:
        """The address of a block the ring holds now (``StaleAddressError`` otherwise)."""
        self._check(torch.tensor([block], device=self.device), None)
        return BlockAddress(self.session, self.epoch, block, block % self.slots, block // self.slots,
                            block * self.block, (block + 1) * self.block)

    def check(self, address: BlockAddress, at: int | None = None) -> None:
        """Refuse an address from another session or epoch, of a reused slot, evicted, or not live at ``at``."""
        if address.session != self.session or address.epoch != self.epoch:
            self.stats["stale_rejections"] += 1
            raise StaleAddressError(f"address of session {address.session} epoch {address.epoch}; this ring is "
                                    f"session {self.session} epoch {self.epoch}")
        if address.slot != address.block % self.slots or address.generation != address.block // self.slots:
            self.stats["stale_rejections"] += 1
            raise StaleAddressError("address slot or generation does not match its block")
        self._check(torch.tensor([address.block], device=self.device),
                    None if at is None else torch.tensor([at], device=self.device))

    def _check(self, blocks: torch.Tensor, at: torch.Tensor | None) -> None:
        """Vectorized liveness and slot identity of absolute ``blocks [n]`` (for queries at ``at [n]``)."""
        if not blocks.numel():
            return
        s = self.block
        start = (blocks * s - self.overlap).clamp(min=self.origin)            # the span a reread needs
        ok = (blocks >= 0) & (self.ids[blocks.clamp(min=0) % self.slots] == blocks)
        ok &= (start >= self.oldest) & ((blocks + 1) * s <= self.position)
        if at is not None:
            ok &= ((blocks + 1) * s <= at + 1) & (start >= at + 1 - self.capacity)
        if not bool(ok.all()):
            bad = int((~ok).sum())
            self.stats["stale_rejections"] += bad
            raise StaleAddressError(f"{bad} block address(es) are not live (slot reused, evicted, or not complete "
                                    f"at the query): first {int(blocks[~ok][0])}")

    def live(self) -> tuple[int, int]:
        """``(lo, hi)``: the blocks ``lo .. hi - 1`` are written and held now with their left context."""
        s = self.block
        lo = self.next_block - self.count
        if self.oldest > self.origin:
            lo = max(lo, -(-(self.oldest + self.overlap) // s))
        return lo, self.next_block

    # -- tokens ------------------------------------------------------------------------------------------------

    def append(self, tokens: torch.Tensor) -> None:
        """Write ``[batch, n]`` token ids at the next ``n`` positions (FIFO; beyond capacity only the newest)."""
        if tokens.ndim != 2 or tokens.shape[0] != self.batch or tokens.shape[1] < 1:
            raise ValueError("append expects [batch, n >= 1] token ids for this ring's batch")
        if tokens.shape[1] > self.capacity:
            self.position += tokens.shape[1] - self.capacity
            tokens = tokens[:, -self.capacity:]
        length = tokens.shape[1]
        slots = (self.position + torch.arange(length, device=self.device)) % self.capacity
        self.tokens.index_copy_(1, slots, tokens.to(device=self.device, dtype=torch.int32))
        self.position += length

    def gather(self, positions: torch.Tensor) -> torch.Tensor:
        """Token ids at absolute ``positions`` (any shape): ``[batch, *shape]`` int64. Refuses positions not held."""
        if positions.numel() and (int(positions.min()) < self.oldest or int(positions.max()) >= self.position):
            raise EvictedError("a requested position is outside the held history")
        return self.tokens[:, positions.to(self.device) % self.capacity].long()

    def reread(self, blocks: torch.Tensor, overlap: int, at: torch.Tensor | None = None):
        """MANDATORY EXACT REREAD. The original token ids of ``blocks [n]`` with ``overlap`` tokens of left
        context: ``(tokens [batch, n, overlap + block] int64, valid [n, overlap + block] bool)``. Every block is
        validated first (``_check``); context positions that are no longer held (or precede the origin) are
        masked, never read from a reused slot."""
        self._check(blocks, at)
        s = self.block
        positions = (blocks * s - overlap)[:, None] + torch.arange(overlap + s, device=blocks.device)
        valid = positions >= self.oldest
        safe = torch.where(valid, positions, blocks[:, None] * s)
        self.stats["rereads"] += int(blocks.numel())
        return self.tokens[:, safe.to(self.device) % self.capacity].long(), valid

    # -- descriptors -------------------------------------------------------------------------------------------

    def write(self, first_block: int, keys: torch.Tensor, fine: torch.Tensor | None = None) -> None:
        """Descriptors of blocks ``first_block .. first_block + n - 1`` (consecutive, in order, every token already
        appended): ``keys [batch, n, per_block, dim]``. Overwrites the oldest records (whose tokens are gone). A
        ring with a fine-key tier takes the same blocks' fine keys ``fine [batch, n, block, fine_dim]`` in the same
        call (and refuses one without them); a ring without refuses them."""
        count = keys.shape[1]
        if (fine is None) != (self.fine is None):
            raise ValueError("fine keys are written exactly when the ring holds a fine-key tier")
        if fine is not None and tuple(fine.shape) != (self.batch, count, *self.fine.shape[2:]):
            raise ValueError(f"fine keys must be [batch, n, block, fine_dim]; got {tuple(fine.shape)}")
        if count == 0:
            return
        if first_block != self.next_block:
            raise ValueError(f"descriptors must be consecutive: expected block {self.next_block}, got {first_block}")
        if (first_block + count) * self.block > self.position:
            raise ValueError("a descriptor was written before all of its block's tokens were appended")
        if count > self.slots:
            keys, first_block, count = keys[:, -self.slots:], first_block + count - self.slots, self.slots
            fine = None if fine is None else fine[:, -self.slots:]
        blocks = first_block + torch.arange(count, device=self.ids.device)
        slots = blocks % self.slots
        self.keys.index_copy_(1, slots.to(self.keys.device), keys.detach().to(self.keys.dtype))
        if fine is not None:
            self.fine.index_copy_(1, slots.to(self.fine.device), fine.detach().to(self.fine.device, self.fine.dtype))
        self.ids.index_copy_(0, slots, blocks)
        self.next_block = first_block + count
        self.count = min(self.count + count, self.slots)
        self.stats["written_blocks"] += count

    def read_fine(self, blocks: torch.Tensor) -> torch.Tensor:
        """R8: the stored fine keys of ``blocks [n]``: ``[batch, n, block, fine_dim]``. Validated exactly like a
        reread (``_check``): an evicted, overwritten, incomplete or never-written block raises
        ``StaleAddressError``; nothing is ever read from a reused slot."""
        if self.fine is None:
            raise ValueError("this ring holds no fine-key tier")
        self._check(blocks, None)
        self.stats["fine_reads"] += int(blocks.numel())
        return self.fine[:, (blocks % self.slots).to(self.fine.device)]

    def reset(self) -> None:
        """An empty ring in the same buffers, under a new epoch: every earlier address becomes stale."""
        self.tokens.zero_()
        self.keys.zero_()
        if self.fine is not None:
            self.fine.zero_()
        self.ids.fill_(-1)
        self.position, self.next_block, self.count = self.origin, self.origin // self.block, 0
        self.epoch += 1

    def nbytes(self) -> dict:
        size = lambda t: t.numel() * t.element_size()    # noqa: E731
        out = {"history_tokens": size(self.tokens), "index_keys": size(self.keys), "index_ids": size(self.ids)}
        if self.fine is not None:
            out["index_fine_keys"] = size(self.fine)
        return out
