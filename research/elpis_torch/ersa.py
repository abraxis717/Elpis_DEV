"""ERSA R2: a recurrent associative token mixer on one bounded state, and the same state used as exact cells.

Elpis-native implementation. No code is taken from the historical ERSA archive (``research/ersa``), from FLA,
or from the Qwenputer-family projects; see ``PROVENANCE.md`` for what was studied and what was not copied.

State. Per head, one matrix ``S`` of shape ``[value_dim, key_dim]``: column ``i`` is the cell addressed by key
direction ``i``. Its size is fixed at creation and does not depend on how many tokens have been processed.

One update kernel (``delta_step``), evaluated in this order:

    S' = retain * S  -  (retain * erase) * (S k) k^T  +  write * v k^T          read: o = S' q

* ``retain`` in [0, 1] scales everything (1 keeps, 0 wipes, 1/2 halves).
* ``erase`` in [0, 1] removes what is currently stored along the key direction ``k`` (the correction term:
  ``S k`` is what the state predicts for this key, and it is subtracted before the new value is written).
* ``write`` in [0, 1] stores the value ``v`` along ``k``.

Two policies use that kernel. They are not the same numerical regime and are not interchangeable:

    LEARNING  (``ERSAMixer`` with trained weights)
        keys      dense, L2-normalized: ``I - erase k k^T`` is then a contraction, so the state cannot blow up
                  through the erase path;
        gates     graded, input dependent (hard sigmoid: exactly 0 and exactly 1 are reachable, so "do not
                  write" and "do not forget" can be exact rather than asymptotic);
        erase     tied to write (the delta rule) unless ``tie_erase=False``;
        read      RMS-normalized per head, then projected; differentiable end to end;
        exact?    no. Stored values are superpositions; recall is approximate and capacity is finite.

    EXACT  (``ERSACells``, or an ``ERSAMixer`` whose weights were compiled by ``compile_cell_mixer``)
        keys      one-hot: a key names one cell;
        gates     saturated: retain in {0, 1/2, 1}, erase and write in {0, 1};
        erase     independent of write: erase=1, write=1 overwrites; erase=0, write=1 accumulates;
        read      no normalization: a one-hot query returns the cell, a multi-hot query the sum of cells;
        exact?    yes, bit for bit, for finite values: with a one-hot key the kernel computes
                  ``x - x*1 = 0`` and ``0 + v = v`` in the written column and adds ``+-0`` to every other.

A bounded state is not a million-token memory. In the learning policy it holds on the order of ``key_dim``
separable associations per head (measured in the R2 experiment); in the exact policy it holds exactly
``key_dim`` cells per head, for as long as nothing overwrites them.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def hard_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """``clamp(z / 4 + 1/2, 0, 1)``: a gate that can be exactly closed, exactly half and exactly open."""
    return (z * 0.25 + 0.5).clamp(0.0, 1.0)


def delta_step(state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, retain: torch.Tensor,
               erase: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    """One state transition. ``state`` ``[..., V, K]``, ``key`` ``[..., K]``, ``value`` ``[..., V]``, gates ``[...]``.

    Never in place: the previous state is still valid after the call.
    """
    held = retain[..., None, None] * state
    stored = (state @ key[..., None])[..., 0]                               # what the state holds along the key
    removed = (retain * erase)[..., None, None] * (stored[..., :, None] * key[..., None, :])
    added = write[..., None, None] * (value[..., :, None] * key[..., None, :])
    return (held - removed) + added


def read_cells(state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """``o = S q``: ``[..., V, K]`` and ``[..., K]`` -> ``[..., V]``."""
    return (state @ query[..., None])[..., 0]


def serial_scan(state, query, key, value, retain, erase, write):
    """The reference recurrence: ``delta_step`` then ``read_cells``, one position at a time.

    ``state`` ``[B, h, V, K]``; ``query``/``key`` ``[B, T, h, K]``; ``value`` ``[B, T, h, V]``; gates ``[B, T, h]``.
    Returns the reads ``[B, T, h, V]`` and the final state.
    """
    reads = []
    for t in range(key.shape[1]):
        state = delta_step(state, key[:, t], value[:, t], retain=retain[:, t], erase=erase[:, t], write=write[:, t])
        reads.append(read_cells(state, query[:, t]))
    return torch.stack(reads, dim=1), state


def chunked_scan(state, query, key, value, retain, erase, write, *, block: int = 64):
    """The same recurrence as ``serial_scan``, evaluated ``block`` positions at a time (R3).

    Algebraically equal, not an approximation. Write ``S_t = r_t S_{t-1} + u_t k_t^T`` with
    ``u_t = w_t v_t - r_t e_t S_{t-1} k_t``. Inside a block that starts from ``S_0``, with
    ``G[t, i] = r_{i+1} ... r_t`` (``G[t, t] = 1``) and ``g_t = r_1 ... r_t``:

        S_{t-1} k_t r_t = g_t S_0 k_t + sum_{i<t} G[t, i] (k_i . k_t) u_i
        (I + L) U = diag(w) V - diag(e g) (K S_0^T),     L[t, i] = e_t G[t, i] (k_t . k_i) for i < t
        o_t = g_t S_0 q_t + sum_{i<=t} G[t, i] (k_i . q_t) u_i
        S_end = g_end S_0 + sum_i G[end, i] u_i k_i^T

    ``G`` is a cumulative PRODUCT of masked gates (never a ratio of cumulative products), so a retain of
    exactly 0 (a wipe) is exact and nothing is divided. ``I + L`` is unit lower triangular: the solve is
    forward substitution of the same recurrence on ``u``. Rounding differs from the serial loop (different
    summation order); parity is tested to 1e-10 in float64 and to the float32 tolerance in the tests.
    """
    batch, length, heads, _ = key.shape
    reads = []
    for at in range(0, length, block):
        q, k, v = (t[:, at:at + block].transpose(1, 2) for t in (query, key, value))      # [B, h, C, .]
        r, e, w = (t[:, at:at + block].transpose(1, 2) for t in (retain, erase, write))   # [B, h, C]
        size = k.shape[2]
        lower = torch.ones(size, size, dtype=torch.bool, device=k.device).tril(-1)       # i < t
        # G[t, i] = prod_{j = i+1 .. t} r_j: a cumulative product along t of r_t masked to 1 where t <= i.
        masked = torch.where(lower, r[..., :, None], torch.ones_like(r)[..., :, None])    # [B, h, t, i]
        decay = torch.cumprod(masked, dim=2)                                               # G, valid for t >= i
        decay = decay.masked_fill(~(lower | torch.eye(size, dtype=torch.bool, device=k.device)), 0.0)
        g = torch.cumprod(r, dim=2)                                                        # [B, h, C]
        s0k = k @ state.transpose(-1, -2)                                                  # rows (S_0 k_t)^T
        rhs = w[..., None] * v - (e * g)[..., None] * s0k
        system = (e[..., None] * decay * (k @ k.transpose(-1, -2))).masked_fill(~lower, 0.0)
        eye = torch.eye(size, dtype=k.dtype, device=k.device)
        u = torch.linalg.solve_triangular(eye + system, rhs, upper=False, unitriangular=True)
        attend = decay * (q @ k.transpose(-1, -2))                                         # [B, h, t, i], i <= t
        reads.append(g[..., None] * (q @ state.transpose(-1, -2)) + attend @ u)
        state = g[..., -1, None, None] * state + (decay[..., -1, :, None] * u).transpose(-1, -2) @ k
    return torch.cat(reads, dim=2).transpose(1, 2), state


class ERSAMixer(nn.Module):
    """Causal, content-dependent token mixing through a recurrent associative state.

    ``forward(x [B, T, D], state [B, heads, V, K], tail) -> (y [B, T, D], new state, new tail)``.
    Token ``t`` writes into the state and then reads it, so ``y_t`` depends on tokens ``<= t`` only. ``tail`` is the
    last ``conv_kernel - 1`` inputs of the previous chunk (``None`` when there is no convolution).
    """

    def __init__(self, dim: int, *, heads: int, key_dim: int, value_dim: int, conv_kernel: int = 0,
                 tie_erase: bool = True, key_norm: bool = True, read_norm: bool = True, out_gate: bool = True,
                 dtype: torch.dtype = torch.float32, kernel: str = "serial", scan_block: int = 64):
        super().__init__()
        if kernel not in ("serial", "chunked"):
            raise ValueError("kernel must be 'serial' (the reference) or 'chunked'")
        if type(scan_block) is not int or scan_block < 1:
            raise ValueError("scan_block must be a positive integer")
        self.kernel, self.scan_block = kernel, scan_block
        for name, number in (("dim", dim), ("heads", heads), ("key_dim", key_dim), ("value_dim", value_dim)):
            if type(number) is not int or number < 1:
                raise ValueError(f"ERSAMixer {name} must be a positive integer")
        if type(conv_kernel) is not int or conv_kernel < 0 or conv_kernel == 1:
            raise ValueError("conv_kernel must be 0 (off) or at least 2")
        self.dim, self.heads, self.key_dim, self.value_dim = dim, heads, key_dim, value_dim
        self.conv_kernel, self.tie_erase, self.key_norm, self.read_norm = conv_kernel, tie_erase, key_norm, read_norm
        # A short causal depthwise convolution: local order information the associative state does not have.
        self.conv = nn.Conv1d(dim, dim, conv_kernel, groups=dim, bias=False, dtype=dtype) if conv_kernel else None
        self.q = nn.Linear(dim, heads * key_dim, bias=False, dtype=dtype)
        self.k = nn.Linear(dim, heads * key_dim, bias=False, dtype=dtype)
        self.v = nn.Linear(dim, heads * value_dim, bias=False, dtype=dtype)
        # Per head: forget, write and (if untied) erase.
        self.gates = nn.Linear(dim, heads * (2 if tie_erase else 3), dtype=dtype)
        self.og = nn.Linear(dim, heads, dtype=dtype) if out_gate else None
        self.out = nn.Linear(heads * value_dim, dim, bias=False, dtype=dtype)

    def new_state(self, batch: int) -> torch.Tensor:
        """Zero state on the module's device, in its parameter dtype (the accumulation precision)."""
        weight = self.q.weight
        return torch.zeros(batch, self.heads, self.value_dim, self.key_dim, dtype=weight.dtype, device=weight.device)

    def new_tail(self, batch: int) -> torch.Tensor | None:
        if self.conv is None:
            return None
        weight = self.q.weight
        return torch.zeros(batch, self.conv_kernel - 1, self.dim, dtype=weight.dtype, device=weight.device)

    def signals(self, x: torch.Tensor, tail: torch.Tensor | None):
        """Everything a chunk needs except the recurrence itself, computed for all its tokens at once."""
        batch, length, _ = x.shape
        new_tail = None
        if self.conv is not None:
            padded = torch.cat((tail, x), dim=1)
            new_tail = padded[:, -(self.conv_kernel - 1):]
            x = self.conv(padded.transpose(1, 2)).transpose(1, 2)
        split = lambda t, d: t.view(batch, length, self.heads, d)     # noqa: E731
        query, key, value = split(self.q(x), self.key_dim), split(self.k(x), self.key_dim), split(
            self.v(x), self.value_dim)
        if self.key_norm:
            key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        gates = hard_sigmoid(self.gates(x)).view(batch, length, self.heads, -1)
        retain, write = 1.0 - gates[..., 0], gates[..., 1]
        erase = write if self.tie_erase else gates[..., 2]
        opened = None if self.og is None else torch.sigmoid(self.og(x))
        return query, key, value, retain, erase, write, opened, new_tail

    def forward(self, x: torch.Tensor, state: torch.Tensor, tail: torch.Tensor | None = None):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"ERSAMixer expects [batch, T, {self.dim}]")
        if state.shape != (x.shape[0], self.heads, self.value_dim, self.key_dim):
            raise ValueError("the state does not belong to this mixer and batch")
        query, key, value, retain, erase, write, opened, new_tail = self.signals(x, tail)
        # The recurrence runs in the state's dtype with autocast off: under mixed precision the projections may
        # be 16-bit, but the state is accumulated in its own (float32 or float64) precision.
        precision = state.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            signals = [t.to(precision) for t in (query, key, value, retain, erase, write)]
            if self.kernel == "chunked":
                read, state = chunked_scan(state, *signals, block=self.scan_block)
            else:
                read, state = serial_scan(state, *signals)      # [B, T, heads, V]
            if new_tail is not None:
                new_tail = new_tail.to(precision)
        if self.read_norm:
            read = read * torch.rsqrt(read.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        if opened is not None:
            read = read * opened[..., None]
        return self.out(read.flatten(-2)), state, new_tail

    def state_bytes(self, batch: int = 1) -> int:
        width = self.q.weight.element_size()
        tail = (self.conv_kernel - 1) * self.dim if self.conv is not None else 0
        return batch * (self.heads * self.value_dim * self.key_dim + tail) * width


class ERSACells:
    """The exact policy, as an addressable bank: ``cells`` cells of ``width`` numbers, on ``delta_step``.

    Every operation is one call of the kernel (or of ``read_cells``) with a one-hot or multi-hot key and
    saturated gates. Nothing is computed outside the state tensor: a sum of cells is a multi-hot read.
    Volatile, bounded, independent per instance; ``reset`` restores the all-zero state.
    """

    def __init__(self, cells: int, width: int = 1, *, dtype: torch.dtype = torch.float64):
        if type(cells) is not int or cells < 1 or type(width) is not int or width < 1:
            raise ValueError("cells and width must be positive integers")
        self.cells, self.width, self.dtype = cells, width, dtype
        self.state = torch.zeros(width, cells, dtype=dtype)
        self.transitions = 0

    def _key(self, *addresses: int) -> torch.Tensor:
        key = torch.zeros(self.cells, dtype=self.dtype)
        for address in addresses:
            if type(address) is not int or not 0 <= address < self.cells:
                raise IndexError(f"cell address {address!r} is outside [0, {self.cells})")
            key[address] = 1.0
        return key

    def _value(self, value) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=self.dtype).reshape(-1)
        if value.numel() != self.width or not bool(torch.isfinite(value).all()):
            raise ValueError(f"a cell holds {self.width} finite numbers")
        return value

    def _step(self, key, value, retain: float, erase: float, write: float) -> None:
        gate = lambda g: torch.tensor(g, dtype=self.dtype)     # noqa: E731
        new = delta_step(self.state, key, value, retain=gate(retain), erase=gate(erase), write=gate(write))
        if not bool(torch.isfinite(new).all()):
            raise FloatingPointError("the transition would leave the state non-finite; nothing was changed")
        self.state, self.transitions = new, self.transitions + 1

    def write(self, address: int, value) -> None:
        """Overwrite one cell exactly. No other cell changes."""
        self._step(self._key(address), self._value(value), 1.0, 1.0, 1.0)

    def add(self, address: int, value) -> None:
        """Accumulate into one cell (erase off): a register that sums what it is given."""
        self._step(self._key(address), self._value(value), 1.0, 0.0, 1.0)

    def clear(self) -> None:
        """Wipe every cell (retain 0)."""
        self._step(torch.zeros(self.cells, dtype=self.dtype), torch.zeros(self.width, dtype=self.dtype), 0.0, 0.0, 0.0)

    def halve(self) -> None:
        """Halve every cell (retain 1/2): k calls scale by exactly 2^-k."""
        self._step(torch.zeros(self.cells, dtype=self.dtype), torch.zeros(self.width, dtype=self.dtype), 0.5, 0.0, 0.0)

    def read(self, address: int) -> torch.Tensor:
        return read_cells(self.state, self._key(address))

    def reduce(self, *addresses: int) -> torch.Tensor:
        """The sum of several cells, computed by one multi-hot read of the state."""
        return read_cells(self.state, self._key(*addresses))

    def reset(self) -> None:
        self.state, self.transitions = torch.zeros(self.width, self.cells, dtype=self.dtype), 0

    def nbytes(self) -> int:
        return self.state.numel() * self.state.element_size()


# -- the exact policy through the mixer's own forward pass ------------------------------------------------------

STORE, ADD, HALVE, CLEAR = 0, 1, 2, 3          # flag columns of a compiled cell mixer's input


def cell_code(cells: int, width: int, *, address: int | None = None, value=None, store: bool = False,
              add: bool = False, halve: bool = False, clear: bool = False, dtype=torch.float64) -> torch.Tensor:
    """One instruction as an input vector ``[one-hot address | value | flags]`` for a compiled cell mixer.

    No flag: a read of ``address`` (or a no-op when ``address`` is ``None``).
    """
    code = torch.zeros(cells + width + 4, dtype=dtype)
    if address is not None:
        code[address] = 1.0
    if value is not None:
        code[cells:cells + width] = torch.as_tensor(value, dtype=dtype).reshape(-1)
    for column, flag in ((STORE, store), (ADD, add), (HALVE, halve), (CLEAR, clear)):
        code[cells + width + column] = float(flag)
    return code


@torch.no_grad()
def compile_cell_mixer(cells: int, width: int, *, dtype: torch.dtype = torch.float64) -> ERSAMixer:
    """An ``ERSAMixer`` whose weights are COMPUTED, not learned, so that its ordinary forward pass is exact RAM.

    Input ``[address one-hot | value | STORE, ADD, HALVE, CLEAR]``. The key and query maps copy the address, the
    value map copies the value, and the gate biases are large enough to saturate the hard sigmoids:
    STORE -> erase 1, write 1 (overwrite); ADD -> erase 0, write 1 (accumulate); HALVE -> retain 1/2;
    CLEAR -> retain 0. The output is the addressed cell after the instruction.
    """
    dim, big = cells + width + 4, 64.0
    mixer = ERSAMixer(dim, heads=1, key_dim=cells, value_dim=width, tie_erase=False, key_norm=True,
                      read_norm=False, out_gate=False, dtype=dtype)
    flags = cells + width
    for parameter in mixer.parameters():
        parameter.zero_()
    mixer.q.weight[:, :cells] = torch.eye(cells, dtype=dtype)
    mixer.k.weight[:, :cells] = torch.eye(cells, dtype=dtype)
    mixer.v.weight[:, cells:flags] = torch.eye(width, dtype=dtype)
    mixer.out.weight[cells:flags, :] = torch.eye(width, dtype=dtype)
    forget, write, erase = 0, 1, 2
    mixer.gates.bias[forget] = -big                                       # retain 1 unless told otherwise
    mixer.gates.weight[forget, flags + HALVE] = big                       # -big + big = 0   -> forget 1/2
    mixer.gates.weight[forget, flags + CLEAR] = 2 * big                   # -big + 2 big     -> forget 1
    mixer.gates.bias[write] = -big
    mixer.gates.weight[write, flags + STORE] = 2 * big
    mixer.gates.weight[write, flags + ADD] = 2 * big
    mixer.gates.bias[erase] = -big
    mixer.gates.weight[erase, flags + STORE] = 2 * big
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)
    return mixer.eval()


def recall_capacity(key_dim: int, value_dim: int, pairs: int, *, seed: int = 0, dtype=torch.float64) -> dict:
    """Store ``pairs`` random associations with the learning policy's kernel, then recall each key.

    Keys are random unit vectors (not orthogonal), written once each with erase = write = 1, retain = 1.
    Returns the fraction recalled (nearest stored value by cosine) and the mean cosine to the true value:
    the state's capacity for exact recall, as a property of the update rule and not of any training.
    """
    generator = torch.Generator().manual_seed(seed)
    keys = F.normalize(torch.randn(pairs, key_dim, generator=generator, dtype=dtype), dim=-1)
    values = torch.randn(pairs, value_dim, generator=generator, dtype=dtype)
    state, one = torch.zeros(value_dim, key_dim, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    for key, value in zip(keys, values):
        state = delta_step(state, key, value, retain=one, erase=one, write=one)
    recalled = (state @ keys.T).T                                          # [pairs, V]
    cosine = F.cosine_similarity(recalled, values, dim=-1)
    nearest = F.normalize(recalled, dim=-1) @ F.normalize(values, dim=-1).T
    return {"key_dim": key_dim, "pairs": pairs, "load": pairs / key_dim,
            "recalled_fraction": float((nearest.argmax(dim=-1) == torch.arange(pairs)).double().mean()),
            "mean_cosine": float(cosine.mean()), "state_norm": float(state.norm() / math.sqrt(pairs))}
