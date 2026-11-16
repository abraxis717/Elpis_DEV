"""The last block's channel transformation and the head, as a function of that block's CGC core weight.

Everything a decoder computes before the last block's channel transformation (every key, every value, every
compressed block, the stream entering the last block and its attention output) is independent of the last
block's CGC weight. So if ``pre`` is the last block's stream after attention, the logits under ANY last-block
core weight ``W`` are

    logits(W) = head(final_norm(pre + channel_W(channel_norm(pre))))

and nothing upstream has to be recomputed, and no attention cache is touched, when ``W`` changes. This module
is that function, for ``ElpisTorchLM`` and ``ScrollLM`` alike (they share the block's attribute names).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .cgc import phi, soft_bound

LAST_BLOCK_FAST_WEIGHT = "cgc.core.weight"


def capture_pre(model):
    """A context manager that records the last block's pre-channel stream of every forward pass in it."""
    class _Capture:
        def __enter__(self):
            self.pre = None
            self._handle = model.blocks[-1].channel_norm.register_forward_pre_hook(
                lambda module, args: setattr(self, "pre", args[0]))
            return self

        def __exit__(self, *exc):
            self._handle.remove()

    return _Capture()


def last_block_logits(model, pre: torch.Tensor, core_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Logits from the last block's pre-channel stream ``pre`` ``[..., hidden]`` under ``core_weight``."""
    block = model.blocks[-1]
    x = block.channel_norm(pre)
    if block.cgc is None:
        channel = block.ffn(x)
    else:
        core = block.cgc.core
        weight = core.weight if core_weight is None else core_weight
        code = x if block.cgc.proj_in is None else block.cgc.proj_in(x)
        units = soft_bound(code @ weight.reshape(core.in_dim, core.channels * core.units), core.preact_bound)
        cells = block.cgc.proj_out(phi(units).unflatten(-1, (core.channels, core.units)).sum(dim=-1))
        channel = cells if block.ffn is None else block.ffn(x) + block.cgc_gate * cells
    out = model.final_norm(pre + channel)
    return F.linear(out, model.embed.weight) if model.head is None else model.head(out)
