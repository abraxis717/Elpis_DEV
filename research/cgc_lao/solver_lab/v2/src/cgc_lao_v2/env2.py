"""SLB2 simulator: the SLB1 world under a new RNG root, fresh partitions and added corruption mechanisms.

Episodes are returned as the v1 ``EpisodeBatch`` container, so the v1 rollout, selectors and agents run on
them unchanged. Streams are keyed by ``SHA-256("elpis/cgc_lao/solver_lab/v2/" + labels)``; elements are the
v1 splitmix64 counter streams, so an episode's data never depends on batching.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from cgc_lao_lab import rng as rng1
from cgc_lao_lab.env import (CONTRACTS as V1_CONTRACTS, D, K, Q, R_MAX, STOP, STOP_PAYOFF, T_MAX, Contract,
                             EpisodeBatch, logits, sigmoid)

ROOT = "elpis/cgc_lao/solver_lab/v2/"
ROUTER_SCALE = 0.65
BIAS_SCALE = 1.0
PARTITIONS = {"TRAIN": 64, "DEV": 256, "VALIDATION": 32, "HOLDOUT": 64, "OOD": 64}
__all__ = ["K", "D", "Q", "T_MAX", "R_MAX", "STOP", "STOP_PAYOFF", "logits", "sigmoid", "Contract"]


def derive_key(*labels: object) -> int:
    text = ROOT + "/".join(str(label) for label in labels)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class Regime:
    name: str
    mechanism: str     # symmetric | qunif | burst | event | correlated | shared_source | second_event | asym10 | context
    q: float = 0.0
    qmax: float = 0.5
    rho: float = 0.0
    q_shared: float = 0.0
    q_noise: float = 0.0
    bias_scale: float = BIAS_SCALE
    router_scale: float = ROUTER_SCALE
    enter: float = 0.25
    exit: float = 0.25
    flip_in_burst: float = 0.5
    cls: str = "DESIGN"


def _r(name, mechanism, cls, **kw):
    return Regime(name, mechanism, cls=cls, **kw)


REGIMES = {r.name: r for r in (
    _r("CLEAN", "symmetric", "DESIGN", q=0.0),
    _r("FLIP20", "symmetric", "DESIGN", q=0.2),
    _r("FLIP40", "symmetric", "DESIGN", q=0.4),
    _r("FLIP45", "symmetric", "ROBUSTNESS", q=0.45),
    _r("FLIP48", "symmetric", "ROBUSTNESS", q=0.48),
    _r("QUNIF", "qunif", "ROBUSTNESS", qmax=0.5),
    _r("BURST", "burst", "ROBUSTNESS", enter=0.25, exit=0.25, flip_in_burst=0.5),
    _r("EVENT40", "event", "ROBUSTNESS", q=0.4),
    _r("CORR10", "correlated", "CORRELATION", q=0.4, rho=0.1),
    _r("CORR25", "correlated", "CORRELATION", q=0.4, rho=0.25),
    _r("CORR50", "correlated", "CORRELATION", q=0.4, rho=0.5),
    _r("CORR75", "correlated", "CORRELATION", q=0.4, rho=0.75),
    _r("CORR90", "correlated", "CORRELATION", q=0.4, rho=0.9),
    _r("FLIP30", "symmetric", "OOD", q=0.3),
    _r("ASYM40", "asym10", "OOD", q=0.4),
    _r("CTXQ40", "context", "OOD", q=0.4),
    _r("BIASWIDE", "symmetric", "OOD", q=0.0, bias_scale=1.6),
    _r("BIASWIDE40", "symmetric", "OOD", q=0.4, bias_scale=1.6),
    _r("ROUTERWIDE40", "symmetric", "OOD", q=0.4, router_scale=1.0),
    _r("SHARED_SRC", "shared_source", "OOD", q_shared=0.3, q_noise=0.15),
    _r("SECOND_EVENT40", "second_event", "OOD", q=0.4),
)}
DESIGN = tuple(n for n, r in REGIMES.items() if r.cls == "DESIGN")
ROBUSTNESS = tuple(n for n, r in REGIMES.items() if r.cls == "ROBUSTNESS")
CORRELATION = tuple(n for n, r in REGIMES.items() if r.cls == "CORRELATION")
OOD = tuple(n for n, r in REGIMES.items() if r.cls == "OOD")
ALL_MIXTURE = ("CLEAN", "FLIP20", "FLIP40")
REGIME_USE = {"TRAIN": DESIGN, "DEV": DESIGN + ROBUSTNESS + CORRELATION,
              "VALIDATION": DESIGN + ROBUSTNESS, "HOLDOUT": DESIGN + ROBUSTNESS,
              "OOD": DESIGN + ROBUSTNESS + CORRELATION + OOD}
CONTRACTS = {name: V1_CONTRACTS[name] for name in ("ORIGINAL", "DUP2_4", "DUP2")}
MODES = ("DUPLICATE", "FIRST_ONLY", "SECOND_EVENT")


def _f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32).astype(np.float64)


def _check(partition: str, seed: int, regime: Regime) -> None:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    if not (isinstance(seed, int) and 0 <= seed < PARTITIONS[partition]):
        raise ValueError(f"seed {seed!r} outside partition {partition}")
    if regime.name not in REGIME_USE[partition]:
        raise ValueError(f"regime {regime.name} is not admitted on partition {partition}")


def router(partition: str, seed: int, scale: float = ROUTER_SCALE) -> np.ndarray:
    if partition not in PARTITIONS or not (0 <= seed < PARTITIONS[partition]):
        raise ValueError("ROUTER_SEED")
    unit = rng1.normal(derive_key(partition, "router", seed), K * D).reshape(K, D)
    w = _f32(scale * unit)
    w.setflags(write=False)
    return w


def episodes(partition: str, seed: int, regime: Regime | str, first: int, count: int) -> EpisodeBatch:
    """Episodes ``first .. first+count-1`` of (partition, router seed, regime); identical whatever the batching."""
    if isinstance(regime, str):
        regime = REGIMES[regime]
    _check(partition, seed, regime)
    if first < 0 or count < 1:
        raise ValueError("EPISODE_RANGE")
    w = router(partition, seed, regime.router_scale)

    def key(stream: str) -> int:
        return derive_key(partition, "router", seed, "regime", regime.name, stream)

    E = count
    b = regime.bias_scale * rng1.normal(key("bias"), E * K, first * K).reshape(E, K)
    x = _f32(rng1.normal(key("ctx"), E * T_MAX * D, first * T_MAX * D).reshape(E, T_MAX, D))
    xq = _f32(rng1.normal(key("query"), E * Q * D, first * Q * D).reshape(E, Q, D))
    p = sigmoid(logits(w, x) + b[:, None, :])
    pq = sigmoid(logits(w, xq) + b[:, None, :])
    n_tka = E * T_MAX * K
    ur = rng1.uniform(key("reward"), n_tka, first * T_MAX * K).reshape(E, T_MAX, K)
    clean = (ur < p).astype(np.uint8)
    uf = rng1.uniform(key("flip"), n_tka * R_MAX, first * T_MAX * K * R_MAX).reshape(E, T_MAX, K, R_MAX)
    q_episode = np.full(E, regime.q)
    m = regime.mechanism
    if m == "symmetric":
        flips = uf < regime.q
    elif m == "qunif":
        q_episode = regime.qmax * rng1.uniform(key("qdraw"), E, first)
        flips = uf < q_episode[:, None, None, None]
    elif m == "burst":
        ub = rng1.uniform(key("burst"), E * T_MAX, first * T_MAX).reshape(E, T_MAX)
        state = np.zeros((E, T_MAX), dtype=bool)
        state[:, 0] = ub[:, 0] < regime.enter / (regime.enter + regime.exit)
        for t in range(1, T_MAX):
            state[:, t] = np.where(state[:, t - 1], ub[:, t] >= regime.exit, ub[:, t] < regime.enter)
        flips = state[:, :, None, None] & (uf < regime.flip_in_burst)
        q_episode = regime.flip_in_burst * state[:, :8].mean(1)
    elif m == "event":
        flips = np.repeat(uf[..., :1] < regime.q, R_MAX, axis=-1)
    elif m == "correlated":
        uc = rng1.uniform(key("corr"), n_tka, first * T_MAX * K).reshape(E, T_MAX, K)
        f0 = uf[..., 0] < regime.q
        f1 = np.where(uc < regime.rho, f0, uf[..., 1] < regime.q)
        flips = np.stack((f0, f1), -1)
    elif m == "shared_source":
        us = rng1.uniform(key("shared"), n_tka, first * T_MAX * K).reshape(E, T_MAX, K)
        flips = (us < regime.q_shared)[..., None] ^ (uf < regime.q_noise)
        q_episode = np.full(E, regime.q_shared * (1 - regime.q_noise) + (1 - regime.q_shared) * regime.q_noise)
    elif m == "second_event":
        ur2 = rng1.uniform(key("reward2"), n_tka, first * T_MAX * K).reshape(E, T_MAX, K)
        clean2 = (ur2 < p).astype(np.uint8)
        f = uf < regime.q
        y1 = clean2 ^ f[..., 1]
        flips = np.stack((f[..., 0], (y1 ^ clean).astype(bool)), -1)   # relative to the first event's clean outcome
    elif m == "asym10":
        flips = (clean[..., None] == 1) & (uf < regime.q)
    elif m == "context":
        q_t = np.where(x[..., 0] > 0, regime.q, 0.0)
        flips = uf < q_t[:, :, None, None]
    else:
        raise ValueError(f"unknown mechanism {m!r}")
    flips = flips.astype(np.uint8)
    y = clean[..., None] ^ flips
    for arr in (x, xq, b, p, pq, clean, flips, y):
        arr.setflags(write=False)
    return EpisodeBatch(partition, seed, regime, first, count, w, x, xq, b, p, pq, clean, flips, y, q_episode)
