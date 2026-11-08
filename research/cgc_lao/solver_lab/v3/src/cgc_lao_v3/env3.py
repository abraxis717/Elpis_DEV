"""SLB2 episodes under a configurable RNG root. ``episodes`` uses the v3 root and v3 partitions; with the v2 root
the generator reproduces ``cgc_lao_v2.env2.episodes`` bitwise (tested), so the benchmark is unchanged and
only the seeds are fresh."""
from __future__ import annotations

import hashlib

import numpy as np

from cgc_lao_lab import rng as rng1
from cgc_lao_lab.env import D, K, Q, R_MAX, T_MAX, EpisodeBatch, logits, sigmoid
from cgc_lao_v2 import env2

ROOT = "elpis/cgc_lao/solver_lab/v3/"
PARTITIONS = {"DEV": 256, "VALIDATION": 32, "HOLDOUT": 64, "OOD": 64}
REGIME_USE = {"DEV": env2.DESIGN + env2.ROBUSTNESS + env2.CORRELATION,
              "VALIDATION": env2.DESIGN + env2.ROBUSTNESS, "HOLDOUT": env2.DESIGN + env2.ROBUSTNESS,
              "OOD": env2.DESIGN + env2.ROBUSTNESS + env2.CORRELATION + env2.OOD}


def derive_key(root: str, *labels: object) -> int:
    text = root + "/".join(str(label) for label in labels)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _f32(a):
    return a.astype(np.float32).astype(np.float64)


def router(partition: str, seed: int, scale: float = env2.ROUTER_SCALE, root: str = ROOT) -> np.ndarray:
    unit = rng1.normal(derive_key(root, partition, "router", seed), K * D).reshape(K, D)
    w = _f32(scale * unit)
    w.setflags(write=False)
    return w


def generate(root: str, partition: str, seed: int, regime, first: int, count: int) -> EpisodeBatch:
    """The SLB2 generator (statement for statement as env2.episodes), parameterised by the RNG root."""
    w = router(partition, seed, regime.router_scale, root)

    def key(stream: str) -> int:
        return derive_key(root, partition, "router", seed, "regime", regime.name, stream)

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
        flips = np.stack((f[..., 0], (y1 ^ clean).astype(bool)), -1)
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


def episodes(partition: str, seed: int, regime, first: int, count: int) -> EpisodeBatch:
    if isinstance(regime, str):
        regime = env2.REGIMES[regime]
    if partition not in PARTITIONS or not (isinstance(seed, int) and 0 <= seed < PARTITIONS[partition]):
        raise ValueError(f"seed {seed!r} outside v3 partition {partition}")
    if regime.name not in REGIME_USE[partition]:
        raise ValueError(f"regime {regime.name} is not admitted on v3 partition {partition}")
    if first < 0 or count < 1:
        raise ValueError("EPISODE_RANGE")
    return generate(ROOT, partition, seed, regime, first, count)
