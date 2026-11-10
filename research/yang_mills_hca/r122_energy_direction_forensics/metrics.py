"""R122-B exact error decomposition and explicit sample classification. numpy + stdlib only.

For a prediction ``p`` and a teacher target ``t`` (vectors over the last axis)::

    E_abs = |p - t|^2            E_rel = E_abs / |t|^2
    r   = |p| / |t|              c = p.t / (|p| |t|)            rho = p.t / |t|^2 = r c

    E_rel = 1 + r^2 - 2 r c                                         cosine identity
          = (1 - rho)^2 + |p - rho t|^2 / |t|^2 = E_gain + E_perp   projection identity
          = (1 - r)^2   + 2 r (1 - c)           = E_mag  + E_ang    norm / angle identity

    E_direction = 2 (1 - c)      E_energy_ratio = r^2

``E_gain`` is the error of the signed amplitude along the teacher direction (a sign
reversal lives there: rho < 0); ``E_perp`` is the energy of the component orthogonal to
the teacher (a genuine rotation). No denominator floor is ever added: a zero or
numerically degenerate norm is *classified*, never regularized.

Numerics: every sample is prescaled by an exact power of two so that its largest
|component| lies in [0.5, 1); sums of squares then cannot overflow or underflow for any
finite input, and all ratio metrics are computed from the scaled sums. Absolute energies
are rescaled with ``ldexp``. ``exact_sums`` recomputes the sums in rational arithmetic
(binary64 values are rationals) for audits.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

# ---- registered constants (PROTOCOL_R122.json repeats them; the runner checks equality) ----
R_FLOOR = 0.01              # minimum meaningful norm ratio: direction is classified only for r in [R_FLOOR, 1/R_FLOOR]
COS_ALIGNED = 0.9           # c >= 0.9        -> ALIGNED    (E_direction <= 0.2)
COS_DEVIATED = 0.5          # 0.5 <= c < 0.9  -> DEVIATED
COS_REVERSED = -0.5         # c <= -0.5       -> REVERSED   (E_direction >= 3);  -0.5 < c < 0.5 -> ORTHOGONAL
COS_ANTIPARALLEL = -0.9     # c <= -0.9       -> REVERSED and flagged antiparallel (E_direction >= 3.8)
NEAR_ZERO_EXPONENT = -511   # largest |component| < 2^-511: a square would leave the normal binary64 range
PAIR_CONDITION_LIMIT = float(2 ** 22)  # (|a| + |b|) / |a - b| above this: a paired difference keeps < ~9 digits

NORM_CLASSES = ("ORDINARY", "BOTH_ZERO", "ZERO_TARGET", "ZERO_PREDICTION", "NEAR_ZERO", "ILL_CONDITIONED", "NONFINITE")
DIR_CLASSES = ("ALIGNED", "DEVIATED", "ORTHOGONAL", "REVERSED", "EXTINGUISHED", "TARGET_NEGLIGIBLE", "UNDEFINED")
N_ORD, N_BOTH0, N_T0, N_P0, N_NEAR0, N_ILL, N_NONFIN = range(7)
D_ALIGNED, D_DEVIATED, D_ORTHO, D_REVERSED, D_EXTINGUISHED, D_TNEG, D_UNDEF = range(7)

BASE_FIELDS = ("tt", "pp", "pt", "ee", "pe")  # |t|^2, |p|^2, p.t, |p-t|^2, |p - rho t|^2


def base_sums(p, t):
    """Scale-safe sums over the last axis.

    Returns ``(absolute, scaled, flags)``: both sum arrays are ``[..., 5]`` in BASE_FIELDS
    order; ``scaled`` uses the per-sample power-of-two normalization (ratios are scale
    free); ``flags`` holds boolean arrays ``nonfinite``, ``t_zero``, ``p_zero``, ``near_zero``.
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("SHAPE")
    finite = np.isfinite(p).all(-1) & np.isfinite(t).all(-1)
    ps = np.where(finite[..., None], p, 0.0)
    ts = np.where(finite[..., None], t, 0.0)
    pm = np.abs(ps).max(-1)
    tm = np.abs(ts).max(-1)
    _, e = np.frexp(np.maximum(pm, tm))      # max = mantissa * 2**e, mantissa in [0.5, 1); 0 -> e == 0
    e = e.astype(np.int64)
    sp = np.ldexp(ps, -e[..., None])         # exact: multiplication by a power of two
    st = np.ldexp(ts, -e[..., None])
    tt = (st * st).sum(-1)
    pp = (sp * sp).sum(-1)
    pt = (sp * st).sum(-1)
    d = sp - st
    ee = (d * d).sum(-1)
    rho = np.divide(pt, tt, out=np.zeros_like(pt), where=tt > 0)
    q = sp - rho[..., None] * st
    pe = (q * q).sum(-1)
    scaled = np.stack([tt, pp, pt, ee, pe], axis=-1)
    with np.errstate(over="ignore", under="ignore"):
        # absolute energies can leave the binary64 range for extreme inputs; the ratio metrics
        # are computed from the scaled sums and are unaffected (such samples are flagged)
        absolute = np.ldexp(scaled, 2 * e[..., None])
    floor = 2.0 ** NEAR_ZERO_EXPONENT
    flags = {
        "nonfinite": ~finite,
        "t_zero": finite & (tm == 0.0),
        "p_zero": finite & (pm == 0.0),
        "near_zero": finite & (((tm > 0) & (tm < floor)) | ((pm > 0) & (pm < floor))),
    }
    return absolute, scaled, flags


def _norm_class(flags, shape, pair_condition):
    norm_class = np.full(shape, N_ORD, dtype=np.int8)
    norm_class[flags["near_zero"]] = N_NEAR0
    if pair_condition is not None:
        ill = np.broadcast_to(np.asarray(pair_condition, dtype=np.float64) > PAIR_CONDITION_LIMIT, shape)
        norm_class[ill & (norm_class == N_ORD)] = N_ILL
    norm_class[flags["p_zero"] & ~flags["t_zero"]] = N_P0
    norm_class[flags["t_zero"] & ~flags["p_zero"]] = N_T0
    norm_class[flags["t_zero"] & flags["p_zero"]] = N_BOTH0
    norm_class[flags["nonfinite"]] = N_NONFIN
    return norm_class


def _derive(sums, norm_class):
    """Ratio metrics, identity residuals and direction classes from (possibly scaled) sums."""
    tt, pp, pt, ee, pe = (sums[..., i] for i in range(5))
    shape = tt.shape
    has_t = (norm_class != N_NONFIN) & (tt > 0)
    has_both = has_t & (pp > 0)
    nan = np.full(shape, np.nan)
    tt1 = np.where(has_t, tt, 1.0)
    pp1 = np.where(has_both, pp, 1.0)
    r = np.where(has_t, np.sqrt(pp / tt1), nan)
    rho = np.where(has_t, pt / tt1, nan)
    erel = np.where(has_t, ee / tt1, nan)
    eperp = np.where(has_t, pe / tt1, nan)
    c = np.where(has_both, pt / (np.sqrt(pp1) * np.sqrt(tt1)), nan)
    c = np.clip(c, -1.0, 1.0)                # |c| may exceed 1 by an ulp
    egain = (1.0 - rho) ** 2
    emag = (1.0 - r) ** 2
    eang = np.where(has_both, 2.0 * r * (1.0 - c), np.where(has_t, 0.0, nan))
    scale = 1.0 + r * r
    with np.errstate(invalid="ignore"):
        res_cos = np.where(has_both, np.abs(erel - (1.0 + r * r - 2.0 * r * c)) / scale,
                           np.where(has_t, np.abs(erel - 1.0), nan))
        res_proj = np.where(has_t, np.abs(erel - (egain + eperp)) / scale, nan)
        res_norm = np.where(has_t, np.abs(erel - (emag + eang)) / scale, nan)
        ordv = norm_class == N_ORD
        dir_class = np.full(shape, D_UNDEF, dtype=np.int8)
        dir_class[ordv & (r < R_FLOOR)] = D_EXTINGUISHED
        dir_class[ordv & (r > 1.0 / R_FLOOR)] = D_TNEG
        mid = ordv & (r >= R_FLOOR) & (r <= 1.0 / R_FLOOR)
        dir_class[mid & (c >= COS_ALIGNED)] = D_ALIGNED
        dir_class[mid & (c < COS_ALIGNED) & (c >= COS_DEVIATED)] = D_DEVIATED
        dir_class[mid & (c < COS_DEVIATED) & (c > COS_REVERSED)] = D_ORTHO
        dir_class[mid & (c <= COS_REVERSED)] = D_REVERSED
        antiparallel = mid & (c <= COS_ANTIPARALLEL)
    return {"r": r, "c": c, "rho": rho, "erel": erel, "egain": egain, "eperp": eperp, "emag": emag, "eang": eang,
            "edir": 2.0 * (1.0 - c), "energy_ratio": r * r, "res_cos": res_cos, "res_proj": res_proj,
            "res_norm": res_norm, "norm_class": norm_class, "dir_class": dir_class, "antiparallel": antiparallel,
            "eligible": ordv}


def decompose(p, t, pair_condition=None):
    """Per-sample decomposition and classification of vectors (last axis). See module docstring.

    ``pair_condition`` (optional, broadcastable to the sample shape) is the conditioning
    number ``(|a| + |b|) / |a - b|`` of a view that was built as a paired difference.
    """
    absolute, scaled, flags = base_sums(p, t)
    out = _derive(scaled, _norm_class(flags, scaled.shape[:-1], pair_condition))
    for i, name in enumerate(BASE_FIELDS):
        out[name] = absolute[..., i]
    return out


def from_base(base, pair_condition=None):
    """The same decomposition from stored absolute base sums ``[..., 5]`` (BASE_FIELDS order).

    Used by the analysis pass on the raw measurement tensor. Zero classes are recognised
    from exact zeros of the stored energies; NEAR_ZERO from energies below 2^(2*NEAR_ZERO_EXPONENT).
    """
    base = np.asarray(base, dtype=np.float64)
    tt, pp = base[..., 0], base[..., 1]
    finite = np.isfinite(base).all(-1)
    tiny = 2.0 ** (2 * NEAR_ZERO_EXPONENT)
    with np.errstate(invalid="ignore"):
        flags = {"nonfinite": ~finite, "t_zero": finite & (tt == 0), "p_zero": finite & (pp == 0),
                 "near_zero": finite & (((tt > 0) & (tt < tiny)) | ((pp > 0) & (pp < tiny)))}
    out = _derive(np.where(finite[..., None], base, 0.0), _norm_class(flags, tt.shape, pair_condition))
    for i, name in enumerate(BASE_FIELDS):
        out[name] = base[..., i]
    return out


def pair_condition(a, b):
    """Conditioning of the paired difference ``a - b`` over the last axis: (|a| + |b|) / |a - b|."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.sqrt((a * a).sum(-1))
    nb = np.sqrt((b * b).sum(-1))
    d = a - b
    nd = np.sqrt((d * d).sum(-1))
    out = np.divide(na + nb, nd, out=np.ones_like(nd), where=nd > 0)
    return np.where((nd == 0) & (na + nb > 0), np.inf, out)


# ---- independent exact reference -------------------------------------------------------------
def exact_sums(p, t):
    """(tt, pp, pt, ee) of one sample in exact rational arithmetic."""
    pf = [Fraction(float(x)) for x in np.asarray(p, dtype=np.float64).reshape(-1)]
    tf = [Fraction(float(x)) for x in np.asarray(t, dtype=np.float64).reshape(-1)]
    if len(pf) != len(tf):
        raise ValueError("SHAPE")
    tt = sum((x * x for x in tf), Fraction(0))
    pp = sum((x * x for x in pf), Fraction(0))
    pt = sum((x * y for x, y in zip(pf, tf)), Fraction(0))
    ee = sum(((x - y) * (x - y) for x, y in zip(pf, tf)), Fraction(0))
    return tt, pp, pt, ee


def exact_audit(p, t):
    """Deviation of the float sums from exact rational values for one sample.

    Returns ``(worst, identity)``: ``worst`` is the largest error of tt, pp, pt, ee relative
    to ``pp + tt`` and of E_rel relative to ``1 + pp/tt``; ``identity`` is the exact value of
    ``ee - (pp + tt - 2 pt)``, which is identically zero.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    absolute, _, _ = base_sums(p, t)
    tt, pp, pt, ee = exact_sums(p, t)
    identity = ee - (pp + tt - 2 * pt)
    den = pp + tt
    if den == 0:
        return (0.0 if not np.any(absolute[:4]) else float("inf")), identity
    worst = max(float(abs(Fraction(float(absolute[i])) - ref) / den) for i, ref in enumerate((tt, pp, pt, ee)))
    if tt != 0:
        got = Fraction(float(absolute[3])) / Fraction(float(absolute[0]))
        worst = max(worst, float(abs(got - ee / tt) / (1 + pp / tt)))
    return worst, identity


def summary(values):
    """Summary of a finite 1-D sample (callers filter eligibility first)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"n": 0}
    if not np.all(np.isfinite(v)):
        raise ValueError("NONFINITE_SUMMARY_INPUT")
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(q[1]), "q10": float(q[0]),
            "q90": float(q[2]), "min": float(v.min()), "max": float(v.max())}
