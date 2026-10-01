"""Single-caller-thread F32 equations; no BLAS pool, GPU or donor imports.

Low precision here is the donor's cache quantize/dequantize operation, not a
packed parameter serialization. F32 reductions define a distinct CPU profile.
"""
import math
import numpy as np
from ...contracts import Code, require

F32 = np.float32


def linear(x, weight, out=None):
    return np.einsum("...i,oi->...o", x, weight, dtype=F32, optimize=False, out=out)


def rms(x, weight, eps):
    return (x * (F32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F32) + F32(eps)))) * weight


def sigmoid(x):
    return np.exp(-np.logaddexp(F32(0), -x)).astype(F32)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True, dtype=F32)


# Positive finite format encodings in code order; code parity is the tie bit.
_E4M3 = np.array([math.ldexp(m, -9) if e == 0 else math.ldexp(8 + m, e - 10)
                 for e in range(16) for m in range(8)][:-1], dtype=F32)
_E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=F32)


def quantize_values(x, levels):
    """Saturating round-nearest-even finite floating point lattice."""
    value = np.minimum(np.abs(x), levels[-1])
    upper = np.minimum(np.searchsorted(levels, value), len(levels) - 1)
    lower = np.maximum(upper - 1, 0)
    dl, du = value - levels[lower], levels[upper] - value
    choose_upper = (du < dl) | ((du == dl) & ((upper & 1) == 0))
    return np.copysign(levels[np.where(choose_upper, upper, lower)], x).astype(F32)


def bf16_round(x):
    bits = np.asarray(x, dtype="<f4").view("<u4")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view("<f4")


def quant_dequant(x, mode):
    """Donor group/scale/minimum rules, with F32 output in the CPU profile."""
    block = 16 if mode == "compressed" else 32
    shape = x.shape
    require(shape[-1] % block == 0, detail="cache quantization group")
    groups = np.asarray(x, dtype=F32).reshape(*shape[:-1], -1, block)
    amax = np.max(np.abs(groups), axis=-1, keepdims=True)
    if mode == "local":
        scale = np.exp2(np.ceil(np.log2(np.maximum(amax, F32(1e-4)) * F32(1 / 448))))
        levels = _E4M3
    elif mode == "index":
        scale = np.exp2(np.ceil(np.log2(np.maximum(amax, F32(6 * 2.0 ** -126)) * F32(1 / 6))))
        levels = _E2M1
    else:
        require(mode == "compressed", detail="cache quantization mode")
        scale = quantize_values(np.maximum(amax, F32(6 * 2.0 ** -9)) / F32(6), _E4M3)
        levels = _E2M1
    out = (quantize_values(groups / scale, levels) * scale).reshape(shape)
    require(np.all(np.isfinite(out)), Code.ENCODING, "cache quantization overflow")
    return out


def rope_frequencies(c, compressed):
    base = c.compress_rope_theta if compressed else c.rope_theta
    freq = F32(1) / np.power(F32(base), np.arange(0, c.rope_dim, 2, dtype=F32) / F32(c.rope_dim))
    if compressed and c.original_seq_len > 0:
        def corrected(rotations):
            return c.rope_dim * math.log(c.original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected(c.beta_fast)), 0)
        high = min(math.ceil(corrected(c.beta_slow)), c.rope_dim - 1)
        smooth = 1 - np.clip((np.arange(c.rope_dim // 2, dtype=F32) - low) / max(high - low, 1e-3), 0, 1)
        freq = freq / F32(c.rope_factor) * (1 - smooth) + freq * smooth
    angles = np.arange(c.max_tokens, dtype=F32)[:, None] * freq
    result = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(F32)
    result.flags.writeable = False
    return result


def rotate(x, freq, inverse=False):
    out = x.copy()
    rd = freq.shape[0] * 2
    tail = x[..., -rd:].reshape(*x.shape[:-1], rd // 2, 2)
    cosine, sine = freq[:, 0], freq[:, 1] * (-1 if inverse else 1)
    pairs = out[..., -rd:].reshape(tail.shape)
    pairs[..., 0] = tail[..., 0] * cosine - tail[..., 1] * sine
    pairs[..., 1] = tail[..., 1] * cosine + tail[..., 0] * sine
    return out


def hc_mixes(x, fn, scale, base, c):
    flat = x.reshape(-1)
    mix = linear(flat, fn) / np.sqrt(np.mean(flat * flat, dtype=F32) + F32(c.norm_eps))
    h = c.hc_mult
    pre = sigmoid(mix[:h] * scale[0] + base[:h]) + F32(c.hc_eps)
    post = 2 * sigmoid(mix[h:2*h] * scale[1] + base[h:2*h])
    comb = softmax((mix[2*h:] * scale[2] + base[2*h:]).reshape(h, h)) + F32(c.hc_eps)
    comb /= np.sum(comb, axis=0, keepdims=True, dtype=F32) + F32(c.hc_eps)
    for _ in range(c.hc_sinkhorn_iters - 1):
        comb /= np.sum(comb, axis=1, keepdims=True, dtype=F32) + F32(c.hc_eps)
        comb /= np.sum(comb, axis=0, keepdims=True, dtype=F32) + F32(c.hc_eps)
    return pre, post, comb


def hc_pre(x, pre):
    return np.sum(pre[:, None] * x, axis=0, dtype=F32)


def hc_post(x, residual, post, comb):
    # Source sums the INPUT copy axis, i.e. comb.T @ residual.
    return post[:, None] * x + np.einsum("ij,id->jd", comb, residual, optimize=False, dtype=F32)
