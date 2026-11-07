"""SLB1 simulator: routers, episodes, exactly indexed potential outcomes and corruption mechanisms.

The simulator is the only owner of hidden quantities (biases, clean probabilities, clean events,
corruption labels, regime parameters). A learner receives a :class:`PublicEpisodes` view and the
readings of the experts it selects, through :meth:`EpisodeBatch.readings`; nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rng

K = 3            # experts
D = 4            # context dimension
T_PRIMARY = 8    # the original horizon
T_MAX = 16       # potential outcomes are generated for this many steps
R_MAX = 2        # and this many independent readings per event
Q = 16           # terminal queries per episode
STOP_PAYOFF = 0.6
ROUTER_SCALE = 0.65
BIAS_SCALE = 1.0
STOP = K         # decision code for STOP

PARTITIONS = {"TRAIN": 64, "DEV": 256, "VALIDATION": 32, "HOLDOUT": 64}


@dataclass(frozen=True)
class Regime:
    name: str
    mechanism: str          # symmetric | qunif | asym10 | burst | context | event
    q: float = 0.0
    qmax: float = 0.5
    bias_scale: float = BIAS_SCALE
    router_scale: float = ROUTER_SCALE
    enter: float = 0.25
    exit: float = 0.25
    flip_in_burst: float = 0.5


DESIGN_REGIMES = {
    "CLEAN": Regime("CLEAN", "symmetric", 0.0),
    "FLIP20": Regime("FLIP20", "symmetric", 0.2),
    "FLIP40": Regime("FLIP40", "symmetric", 0.4),
}
SHIFT_REGIMES = {
    "QUNIF": Regime("QUNIF", "qunif", qmax=0.5),
    "FLIP30": Regime("FLIP30", "symmetric", 0.3),
    "FLIP45": Regime("FLIP45", "symmetric", 0.45),
    "BURST": Regime("BURST", "burst", enter=0.25, exit=0.25, flip_in_burst=0.5),
    "ASYM40": Regime("ASYM40", "asym10", 0.4),
    "CTXQ40": Regime("CTXQ40", "context", 0.4),
    "BIASWIDE": Regime("BIASWIDE", "symmetric", 0.0, bias_scale=1.6),
    "BIASWIDE40": Regime("BIASWIDE40", "symmetric", 0.4, bias_scale=1.6),
    "ROUTERWIDE40": Regime("ROUTERWIDE40", "symmetric", 0.4, router_scale=1.0),
    "EVENT40": Regime("EVENT40", "event", 0.4),
}
REGIMES = {**DESIGN_REGIMES, **SHIFT_REGIMES}
ALL_MIXTURE = ("CLEAN", "FLIP20", "FLIP40")
PRIMARY_CORRUPTED = "FLIP40"
PRIMARY_SHIFTS = ("QUNIF", "BURST")


@dataclass(frozen=True)
class Contract:
    name: str
    reads: tuple            # readings per acquisition step; len(reads) is the horizon

    @property
    def horizon(self) -> int:
        return len(self.reads)

    @property
    def changed(self) -> bool:
        return self.name != "ORIGINAL"

    @property
    def max_reads(self) -> int:
        return max(self.reads)


CONTRACTS = {
    "ORIGINAL": Contract("ORIGINAL", (1,) * 8),
    "DUP2": Contract("DUP2", (2,) * 8),
    "DUP2_4": Contract("DUP2_4", (2,) * 4 + (1,) * 4),
    "T12": Contract("T12", (1,) * 12),
    "T16": Contract("T16", (1,) * 16),
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function (float64 or float32, dtype preserved)."""
    z = np.asarray(z)
    e = np.exp(-np.abs(z))
    one = z.dtype.type(1)
    return np.where(z >= 0, one / (one + e), e / (one + e))


def logits(router_w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Public base logits ``W x`` for contexts ``x[..., 4]`` -> ``[..., 3]``, in a fixed summation order."""
    out = x[..., 0:1] * router_w[:, 0]
    for d in range(1, D):
        out = out + x[..., d:d + 1] * router_w[:, d]
    return out


def _f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32).astype(np.float64)


def router(partition: str, seed: int, scale: float = ROUTER_SCALE) -> np.ndarray:
    """The public frozen router of a router seed (float32-representable float64 [3, 4])."""
    _check_seed(partition, seed)
    unit = rng.normal(rng.derive_key(partition, "router", seed), K * D).reshape(K, D)
    w = _f32(scale * unit)
    w.setflags(write=False)
    return w


def _check_seed(partition: str, seed: int) -> None:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    if not (isinstance(seed, int) and 0 <= seed < PARTITIONS[partition]):
        raise ValueError(f"seed {seed!r} outside partition {partition}")


@dataclass(frozen=True)
class PublicEpisodes:
    """Everything a deployed learner may see before feedback: router, contexts, terminal queries."""
    router_w: np.ndarray    # [3, 4]
    x: np.ndarray           # [E, T_MAX, 4]
    xq: np.ndarray          # [E, Q, 4]


@dataclass
class EpisodeBatch:
    """Episodes ``[first, first + count)`` of one (partition, router seed, regime). Holds hidden truth."""
    partition: str
    seed: int
    regime: Regime
    first: int
    count: int
    router_w: np.ndarray
    x: np.ndarray
    xq: np.ndarray
    b: np.ndarray           # hidden biases [E, 3]
    p: np.ndarray           # clean probabilities [E, T_MAX, 3]
    pq: np.ndarray          # terminal clean probabilities [E, Q, 3]
    clean: np.ndarray       # clean events C [E, T_MAX, 3] uint8
    flips: np.ndarray       # corruption labels F [E, T_MAX, 3, R_MAX] uint8
    y: np.ndarray           # readings Y [E, T_MAX, 3, R_MAX] uint8
    q_episode: np.ndarray   # per-episode nominal corruption rate (diagnostic only) [E]
    access_log: list = field(default_factory=list)

    def public(self) -> PublicEpisodes:
        return PublicEpisodes(self.router_w, self.x, self.xq)

    def readings(self, t: int, actions: np.ndarray, reads: int) -> np.ndarray:
        """Readings ``Y[e, t, a_e, 0..reads-1]`` of the selected experts only -> uint8 ``[E, reads]``."""
        actions = np.asarray(actions)
        if actions.shape != (self.count,) or actions.dtype.kind not in "iu":
            raise ValueError("READINGS_ACTION_CONTRACT")
        if not (0 <= t < T_MAX and 1 <= reads <= R_MAX) or actions.min() < 0 or actions.max() >= K:
            raise ValueError("READINGS_INDEX_CONTRACT")
        self.access_log.append((t, actions.copy(), reads))
        return self.y[np.arange(self.count), t, actions, :reads].copy()


def episodes(partition: str, seed: int, regime: Regime | str, first: int, count: int) -> EpisodeBatch:
    """Generate episodes ``first .. first+count-1``; identical whatever the batching."""
    _check_seed(partition, seed)
    if isinstance(regime, str):
        regime = REGIMES[regime]
    if first < 0 or count < 1:
        raise ValueError("EPISODE_RANGE")
    w = router(partition, seed, regime.router_scale)

    def key(stream: str) -> int:
        return rng.derive_key(partition, "router", seed, "regime", regime.name, stream)

    E = count
    b = regime.bias_scale * rng.normal(key("bias"), E * K, first * K).reshape(E, K)
    x = _f32(rng.normal(key("ctx"), E * T_MAX * D, first * T_MAX * D).reshape(E, T_MAX, D))
    xq = _f32(rng.normal(key("query"), E * Q * D, first * Q * D).reshape(E, Q, D))
    p = sigmoid(logits(w, x) + b[:, None, :])
    pq = sigmoid(logits(w, xq) + b[:, None, :])
    ur = rng.uniform(key("reward"), E * T_MAX * K, first * T_MAX * K).reshape(E, T_MAX, K)
    clean = (ur < p).astype(np.uint8)
    uf = rng.uniform(key("flip"), E * T_MAX * K * R_MAX, first * T_MAX * K * R_MAX).reshape(E, T_MAX, K, R_MAX)
    q_episode = np.full(E, regime.q)
    m = regime.mechanism
    if m == "symmetric":
        flips = uf < regime.q
    elif m == "qunif":
        q_episode = regime.qmax * rng.uniform(key("qdraw"), E, first)
        flips = uf < q_episode[:, None, None, None]
    elif m == "asym10":
        flips = (clean[..., None] == 1) & (uf < regime.q)
    elif m == "context":
        q_t = np.where(x[..., 0] > 0, regime.q, 0.0)
        flips = uf < q_t[:, :, None, None]
    elif m == "event":
        flips = np.repeat(uf[..., :1] < regime.q, R_MAX, axis=-1)
    elif m == "burst":
        ub = rng.uniform(key("burst"), E * T_MAX, first * T_MAX).reshape(E, T_MAX)
        stationary = regime.enter / (regime.enter + regime.exit)
        state = np.zeros((E, T_MAX), dtype=bool)
        state[:, 0] = ub[:, 0] < stationary
        for t in range(1, T_MAX):
            prev = state[:, t - 1]
            state[:, t] = np.where(prev, ub[:, t] >= regime.exit, ub[:, t] < regime.enter)
        flips = state[:, :, None, None] & (uf < regime.flip_in_burst)
        q_episode = regime.flip_in_burst * state[:, :T_PRIMARY].mean(1)
    else:
        raise ValueError(f"unknown mechanism {m!r}")
    flips = flips.astype(np.uint8)
    y = clean[..., None] ^ flips
    for arr in (x, xq, b, p, pq, clean, flips, y):
        arr.setflags(write=False)
    return EpisodeBatch(partition, seed, regime, first, count, w, x, xq, b, p, pq, clean, flips, y, q_episode)


def terminal_regret(pq: np.ndarray, decisions: np.ndarray, stop_payoff: float = STOP_PAYOFF) -> np.ndarray:
    """Per-episode terminal regret for decisions ``[E, Q]`` in ``{0, 1, 2, STOP}``."""
    best = np.maximum(pq.max(-1), stop_payoff)
    chosen = np.take_along_axis(pq, np.minimum(decisions, K - 1)[..., None], -1)[..., 0]
    util = np.where(decisions == STOP, stop_payoff, chosen)
    return (best - util).mean(-1)


def online_regret_step(p_t: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per-episode online regret of one acquisition step."""
    return p_t.max(-1) - p_t[np.arange(len(actions)), actions]
