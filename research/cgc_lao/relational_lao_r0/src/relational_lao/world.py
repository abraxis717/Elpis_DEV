"""RLB1: the relational LAO benchmark (SPEC.md section 3). Stochastic model, separate from the query utility.

Every random quantity is a counter-RNG potential outcome indexed [episode, step, relation(, reading)] under the
root ``elpis/cgc_lao/relational_lao_r0/``, so every policy and every learner faces the same world. Corruption
mechanisms are SLB2's (``cgc_lao_v2.env2.REGIMES``), applied per reading of the selected relation's event.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from cgc_lao_lab import rng as rng1
from cgc_lao_lab.env import sigmoid
from cgc_lao_v2 import env2

ROOT = "elpis/cgc_lao/relational_lao_r0/"
PARTITIONS = {"TRAIN": 32, "DEV": 256, "VALIDATION": 32, "HOLDOUT": 64, "OOD": 64}
T, D, R_MAX, QN, NSRC, NPLAN = 8, 4, 8, 16, 3, 4
READS = {"I": (2, 2, 2, 2, 1, 1, 1, 1), "II": (1, 1, 1, 1, 1, 1, 1, 1)}
STOP_BASE, SECONDARY_P = 0.6, 0.3
COMPOSITIONAL = ("PATH", "CONJ", "REACH")
AUTHORITY_KINDS = ("STATIC", "REV_DRIFT", "REV_SAME", "DORMANT", "RECLAIM_DRIFT")


def derive_key(*labels) -> int:
    text = ROOT + "/".join(str(label) for label in labels)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _f32(a):
    return np.asarray(a).astype(np.float32).astype(np.float64)


# -- query structures and templates ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Query:
    """An alternative is an OR of edge-disjoint paths; a path is an AND of distinct relations."""
    name: str
    kind: str                      # DIRECT | PATH | CONJ | REACH
    alternatives: tuple            # ((path, ...), ...); path = (relation index, ...)

    @property
    def stop(self) -> float:
        if self.kind == "REACH":
            return STOP_BASE
        return STOP_BASE ** len(self.alternatives[0][0])


@dataclass(frozen=True)
class Template:
    name: str
    entities: tuple
    relations: tuple               # ((from, to), ...) entity indices; at most R_MAX
    queries: tuple

    def __post_init__(self):
        if not (1 <= len(self.relations) <= R_MAX) or len(set(self.relations)) != len(self.relations):
            raise ValueError("TEMPLATE_RELATIONS")
        for q in self.queries:
            lengths = {len(p) for alt in q.alternatives for p in alt}
            if not (1 <= len(q.alternatives) <= 4) or max(len(a) for a in q.alternatives) > 3 or max(lengths) > 3:
                raise ValueError("TEMPLATE_QUERY_BOUNDS")
            if q.kind != "REACH" and len(lengths) != 1:
                raise ValueError("TEMPLATE_QUERY_LENGTHS")
            for alt in q.alternatives:
                used = [r for p in alt for r in p]
                if len(used) != len(set(used)) or not all(0 <= r < len(self.relations) for r in used):
                    raise ValueError("TEMPLATE_QUERY_DISJOINT")


def _q(name, kind, *alts):
    return Query(name, kind, tuple(tuple(tuple(p) for p in a) for a in alts))


D_DIAMOND = Template("D_DIAMOND", ("S", "M1", "M2", "M3", "T", "U"),
                     ((0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (1, 5), (3, 5)),
                     (_q("DIRECT_S", "DIRECT", [(0,)], [(1,)], [(2,)]),
                      _q("PATH_ST", "PATH", [(0, 3)], [(1, 4)], [(2, 5)]),
                      _q("PATH_SU", "PATH", [(0, 6)], [(2, 7)]),
                      _q("CONJ_TU", "CONJ", [(3, 6)], [(5, 7)]),
                      _q("REACH_ST", "REACH", [(0, 3), (1, 4), (2, 5)])))
D_FAN = Template("D_FAN", ("S", "A", "B", "C", "D", "T"),
                 ((0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)),
                 (_q("DIRECT_S", "DIRECT", [(0,)], [(1,)]),
                  _q("PATH_SC", "PATH", [(0, 2)], [(1, 4)]),
                  _q("PATH_SD", "PATH", [(0, 3)], [(1, 5)]),
                  _q("PATH3_ST", "PATH", [(0, 2, 6)], [(0, 3, 7)], [(1, 4, 6)], [(1, 5, 7)]),
                  _q("REACH_SC", "REACH", [(0, 2), (1, 4)]),
                  _q("CONJ_CD", "CONJ", [(2, 3)], [(4, 5)])))
V_LADDER = Template("V_LADDER", ("S", "A1", "A2", "B1", "B2", "T"),
                    ((0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5), (1, 4), (3, 2)),
                    (_q("PATH3_ST", "PATH", [(0, 1, 2)], [(3, 4, 5)], [(0, 6, 5)], [(3, 7, 2)]),
                     _q("PATH_SA2", "PATH", [(0, 1)], [(3, 7)]),
                     _q("PATH_SB2", "PATH", [(3, 4)], [(0, 6)]),
                     _q("DIRECT_S", "DIRECT", [(0,)], [(3,)]),
                     _q("REACH_ST", "REACH", [(0, 1, 2), (3, 4, 5)]),
                     _q("CONJ_X", "CONJ", [(1, 6)], [(4, 7)])))
FIXED_TEMPLATES = {t.name: t for t in (D_DIAMOND, D_FAN, V_LADDER)}


def random_template(key: int, attempt_limit: int = 512) -> Template:
    """H_RANDOM: an 8-edge DAG over 6 or 7 nodes in topological order, with automatically derived queries (at least
    one DIRECT, at least three compositional, at least six relations used). Deterministic in ``key``."""
    for attempt in range(attempt_limit):
        u = rng1.uniform(key, 64, 64 * attempt)
        n = 6 + int(u[0] * 2)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i <= 3]
        order = np.argsort(u[1:1 + len(pairs)], kind="stable")
        edges = tuple(sorted(pairs[k] for k in order[:8]))
        idx = {e: r for r, e in enumerate(edges)}
        out = {i: [idx[e] for e in edges if e[0] == i] for i in range(n)}

        def paths(a, b, k):
            if k == 1:
                return [(idx[(a, b)],)] if (a, b) in idx else []
            res = []
            for (s, m), r in idx.items():
                if s == a and m < b:
                    res += [(r,) + p for p in paths(m, b, k - 1)]
            return res
        queries = []
        hubs = [i for i in range(n) if len(out[i]) >= 2]
        if hubs:
            h = hubs[0]
            queries.append(_q(f"DIRECT_{h}", "DIRECT", *[[(r,)] for r in out[h][:4]]))
        for k in (2, 3):
            for a in range(n):
                for b in range(a + k, n):
                    ps = paths(a, b, k)
                    if len(ps) >= 2:
                        queries.append(_q(f"PATH{k}_{a}_{b}", "PATH", *[[p] for p in ps[:4]]))
        for a in range(n):
            for b in range(a + 2, n):
                ps = sorted(paths(a, b, 2) + paths(a, b, 3), key=len)
                chosen = []
                for p in ps:
                    if all(not set(p) & set(c) for c in chosen):
                        chosen.append(p)
                if len(chosen) >= 2:
                    queries.append(_q(f"REACH_{a}_{b}", "REACH", chosen[:3]))
        if len(hubs) >= 2:
            queries.append(_q("CONJ_HUBS", "CONJ", *[[tuple(out[h][:2])] for h in hubs[:3]]))
        direct = [q for q in queries if q.kind == "DIRECT"][:1]
        comp = [q for q in queries if q.kind != "DIRECT"]
        picked = direct + [q for q in comp if q.kind == "PATH"][:3] + [q for q in comp if q.kind == "CONJ"][:1] + \
            [q for q in comp if q.kind == "REACH"][:1]
        used = {r for q in picked for a in q.alternatives for p in a for r in p}
        if direct and len(picked) - 1 >= 3 and len(used) >= 6:
            return Template(f"H_RANDOM_{key & 0xFFFF:04x}", tuple(f"N{i}" for i in range(n)), edges, tuple(picked))
    raise RuntimeError("H_RANDOM_NO_TEMPLATE")


def template_for(partition: str, seed: int) -> Template:
    """Templates enter partitions as fixed in SPEC.md: D_* everywhere, V_LADDER from VALIDATION, H_RANDOM only on
    HOLDOUT and OOD (a fresh topology per world seed)."""
    if partition in ("TRAIN", "DEV"):
        return (D_DIAMOND, D_FAN)[seed % 2]
    if partition == "VALIDATION":
        return V_LADDER if seed % 2 == 0 else (D_DIAMOND, D_FAN)[(seed // 2) % 2]
    if partition == "HOLDOUT":
        return random_template(derive_key(partition, "world", seed, "topology")) if seed % 2 == 0 \
            else (D_DIAMOND, D_FAN)[(seed // 2) % 2]
    if partition == "OOD":
        return (D_DIAMOND, D_FAN, V_LADDER, None)[seed % 4] or \
            random_template(derive_key(partition, "world", seed, "topology"))
    raise ValueError(f"unknown partition {partition}")


# -- regimes -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RelRegime:
    name: str
    corruption: env2.Regime
    authority: str = "STATIC"


def regime(name: str) -> RelRegime:
    """'FLIP40' (STATIC) or 'CLEAN+REV_DRIFT' (corruption + authority event)."""
    corr, _, auth = name.partition("+")
    auth = auth or "STATIC"
    if corr not in env2.REGIMES or auth not in AUTHORITY_KINDS:
        raise ValueError(f"unknown regime {name}")
    return RelRegime(name, env2.REGIMES[corr], auth)


DESIGN = ("CLEAN", "FLIP20", "FLIP40")
ROBUSTNESS = ("FLIP45", "FLIP48", "QUNIF", "BURST", "EVENT40")
AUTHORITY = tuple(f"{c}+{a}" for c in ("CLEAN", "FLIP20") for a in AUTHORITY_KINDS[1:])
CORRELATION = env2.CORRELATION
OOD_CORRUPTION = ("SHARED_SRC", "SECOND_EVENT40", "ASYM40", "CTXQ40", "FLIP30")


# -- episodes ----------------------------------------------------------------------------------------------------
@dataclass
class Batch:
    partition: str
    seed: int
    regime: RelRegime
    template: Template
    first: int
    count: int
    W: np.ndarray                  # [R, D] public relational prior weights
    x: np.ndarray                  # [E, T, D] step contexts
    xq: np.ndarray                 # [E, QN, D] terminal query contexts
    xp: np.ndarray                 # [E, NPLAN, D] planning contexts (never the terminal ones)
    b0: np.ndarray                 # [E, R] hidden biases before any authority event
    b1: np.ndarray                 # [E, R] after it (drifted relations redrawn)
    p: np.ndarray                  # [E, T, R] true event probabilities at each step
    pq: np.ndarray                 # [E, QN, R] true probabilities at the terminal queries (final world)
    clean: np.ndarray              # [E, T, R] clean outcomes
    flips: np.ndarray              # [E, T, R, 2]
    y: np.ndarray                  # [E, T, R, 2] readings
    q_episode: np.ndarray          # [E] privileged corruption rate
    rho_episode: float
    own1: np.ndarray               # [E, R] primary source
    own2: np.ndarray               # [E, R] secondary source or -1
    t_ev: np.ndarray               # [E] authority event step (T: none)
    s_ev: np.ndarray               # [E] the source the event concerns
    perm: np.ndarray               # [E, R] derangement of slots (identity ablation)
    query_struct: list = field(default_factory=list)   # [QN] index into template.queries

    @property
    def R(self) -> int:
        return len(self.template.relations)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Public logits W x for contexts x[..., D] -> [..., R] (fixed summation order)."""
        out = x[..., 0:1] * self.W[:, 0]
        for d in range(1, D):
            out = out + x[..., d:d + 1] * self.W[:, d]
        return out

    def reporter(self, t: int) -> np.ndarray:
        """[E, R]: the source reporting an event of each relation at step t (primary on even steps)."""
        return np.where((t % 2 == 1) & (self.own2 >= 0), self.own2, self.own1)

    def epoch(self, src: np.ndarray, t: int) -> np.ndarray:
        """Binding epoch of source(s) at time t (a revision or reclamation starts epoch 1)."""
        auth = self.regime.authority
        ev = (src == self.s_ev[:, None]) & (t >= self.t_ev[:, None]) if src.ndim == 2 else \
            (src == self.s_ev) & (t >= self.t_ev)
        return (ev & (auth in ("REV_DRIFT", "REV_SAME", "RECLAIM_DRIFT"))).astype(np.int8)

    def active(self, src: np.ndarray, t: int) -> np.ndarray:
        if self.regime.authority != "DORMANT":
            return np.ones(src.shape, dtype=bool)
        s, t0 = (self.s_ev[:, None], self.t_ev[:, None]) if src.ndim == 2 else (self.s_ev, self.t_ev)
        return ~((src == s) & (t >= t0) & (t < t0 + 2))

    def live(self, src: np.ndarray, epoch: np.ndarray, t: int) -> np.ndarray:
        return (self.epoch(src, t) == epoch) & self.active(src, t)

    def admissible(self, t: int) -> np.ndarray:
        """[E, R]: relations whose reporting source is ACTIVE at step t."""
        return self.active(self.reporter(t), t)


def _flips(regime: RelRegime, key, E, R, first, x, clean, p):
    """SLB2 mechanisms over [E, T, R, 2] (statement for statement as env3.generate, relations for experts)."""
    reg = regime.corruption
    n = E * T * R
    uf = rng1.uniform(key("flip"), n * 2, first * T * R * 2).reshape(E, T, R, 2)
    q_episode, rho = np.full(E, reg.q), 0.0
    m = reg.mechanism
    if m == "symmetric":
        flips = uf < reg.q
    elif m == "qunif":
        q_episode = reg.qmax * rng1.uniform(key("qdraw"), E, first)
        flips = uf < q_episode[:, None, None, None]
    elif m == "burst":
        ub = rng1.uniform(key("burst"), E * T, first * T).reshape(E, T)
        state = np.zeros((E, T), dtype=bool)
        state[:, 0] = ub[:, 0] < reg.enter / (reg.enter + reg.exit)
        for t in range(1, T):
            state[:, t] = np.where(state[:, t - 1], ub[:, t] >= reg.exit, ub[:, t] < reg.enter)
        flips = state[:, :, None, None] & (uf < reg.flip_in_burst)
        q_episode = reg.flip_in_burst * state.mean(1)
    elif m == "event":
        flips = np.repeat(uf[..., :1] < reg.q, 2, axis=-1)
        rho = 1.0
    elif m == "correlated":
        uc = rng1.uniform(key("corr"), n, first * T * R).reshape(E, T, R)
        f0 = uf[..., 0] < reg.q
        f1 = np.where(uc < reg.rho, f0, uf[..., 1] < reg.q)
        flips, rho = np.stack((f0, f1), -1), reg.rho
    elif m == "shared_source":
        us = rng1.uniform(key("shared"), n, first * T * R).reshape(E, T, R)
        flips = (us < reg.q_shared)[..., None] ^ (uf < reg.q_noise)
        q_episode = np.full(E, reg.q_shared * (1 - reg.q_noise) + (1 - reg.q_shared) * reg.q_noise)
    elif m == "second_event":
        ur2 = rng1.uniform(key("reward2"), n, first * T * R).reshape(E, T, R)
        clean2 = (ur2 < p).astype(np.uint8)
        f = uf < reg.q
        flips = np.stack((f[..., 0], ((clean2 ^ f[..., 1]) ^ clean).astype(bool)), -1)
    elif m == "asym10":
        flips = (clean[..., None] == 1) & (uf < reg.q)
    elif m == "context":
        q_t = np.where(x[..., 0] > 0, reg.q, 0.0)
        flips = uf < q_t[:, :, None, None]
    else:
        raise ValueError(f"unknown mechanism {m!r}")
    return flips.astype(np.uint8), q_episode, rho


def episodes(partition: str, seed: int, regime_name: str, first: int, count: int) -> Batch:
    if partition not in PARTITIONS or not (isinstance(seed, int) and 0 <= seed < PARTITIONS[partition]):
        raise ValueError(f"seed {seed!r} outside RLB1 partition {partition}")
    if first < 0 or count < 1:
        raise ValueError("EPISODE_RANGE")
    reg = regime(regime_name)
    tpl = template_for(partition, seed)
    R, E = len(tpl.relations), count
    W = _f32(reg.corruption.router_scale * rng1.normal(derive_key(partition, "world", seed, "W"), R_MAX * D)
             .reshape(R_MAX, D)[:R])

    def key(stream: str) -> int:
        return derive_key(partition, "world", seed, "regime", reg.name, stream)

    b0 = reg.corruption.bias_scale * rng1.normal(key("bias"), E * R, first * R).reshape(E, R)
    bnew = reg.corruption.bias_scale * rng1.normal(key("drift"), E * R, first * R).reshape(E, R)
    x = _f32(rng1.normal(key("ctx"), E * T * D, first * T * D).reshape(E, T, D))
    xq = _f32(rng1.normal(key("query"), E * QN * D, first * QN * D).reshape(E, QN, D))
    xp = _f32(rng1.normal(key("plan"), E * NPLAN * D, first * NPLAN * D).reshape(E, NPLAN, D))
    own1 = np.minimum((rng1.uniform(key("own"), E * R, first * R) * NSRC).astype(np.int64), NSRC - 1).reshape(E, R)
    sec = rng1.uniform(key("secondary"), E * R * 2, first * R * 2).reshape(E, R, 2)
    own2 = np.where(sec[..., 0] < SECONDARY_P, (own1 + 1 + (sec[..., 1] * 2).astype(np.int64)) % NSRC, -1)
    ue = rng1.uniform(key("event"), E * 2, first * 2).reshape(E, 2)
    auth = reg.authority
    if auth == "STATIC":
        t_ev = np.full(E, T, dtype=np.int64)
    elif auth == "DORMANT":
        t_ev = 2 + (ue[:, 0] * 3).astype(np.int64)
    else:
        t_ev = 3 + (ue[:, 0] * 3).astype(np.int64)
    s_ev = own1[np.arange(E), np.minimum((ue[:, 1] * R).astype(np.int64), R - 1)]
    drift = (own1 == s_ev[:, None]) & (auth in ("REV_DRIFT", "RECLAIM_DRIFT"))
    b1 = np.where(drift, bnew, b0)
    shift = 1 + (rng1.uniform(key("perm"), E, first) * (R - 1)).astype(np.int64)
    perm = (np.arange(R)[None, :] + shift[:, None]) % R
    probe = Batch(partition, seed, reg, tpl, first, count, W, x, xq, xp, b0, b1, None, None, None, None, None,
                  None, 0.0, own1, own2, t_ev, s_ev, perm)
    z = probe.logits(x)                                        # [E, T, R]
    after = (np.arange(T)[None, :, None] >= t_ev[:, None, None])
    p = sigmoid(z + np.where(after, b1[:, None, :], b0[:, None, :]))
    pq = sigmoid(probe.logits(xq) + b1[:, None, :])
    ur = rng1.uniform(key("reward"), E * T * R, first * T * R).reshape(E, T, R)
    clean = (ur < p).astype(np.uint8)
    flips, q_episode, rho = _flips(reg, key, E, R, first, x, clean, p)
    y = clean[..., None] ^ flips
    probe.p, probe.pq, probe.clean, probe.flips, probe.y = p, pq, clean, flips, y
    probe.q_episode, probe.rho_episode = q_episode, rho
    probe.query_struct = [j % len(tpl.queries) for j in range(QN)]
    for arr in (W, x, xq, xp, b0, b1, p, pq, clean, flips, y, own1, own2, t_ev, s_ev, perm):
        arr.setflags(write=False)
    return probe


def query_tensors(template: Template):
    """Padded index tensors of the template's queries: idx [S, A, P, L] (-1 padding), alt_mask [S, A], stop [S]."""
    S = len(template.queries)
    idx = -np.ones((S, 4, 3, 3), dtype=np.int64)
    alt_mask = np.zeros((S, 4), dtype=bool)
    stop = np.zeros(S)
    for s, q in enumerate(template.queries):
        stop[s] = q.stop
        for a, alt in enumerate(q.alternatives):
            alt_mask[s, a] = True
            for pi, path in enumerate(alt):
                idx[s, a, pi, :len(path)] = path
    return idx, alt_mask, stop


def utilities(m: np.ndarray, template: Template, struct: list) -> np.ndarray:
    """Utility of every alternative [..., J, A] from relation probabilities m [..., J, R] (exact for independent,
    edge-disjoint terms): 1 - prod_paths (1 - prod_relations m); padded alternatives are -inf."""
    idx, alt_mask, _ = query_tensors(template)
    sidx = idx[list(struct)]                                   # [J, A, P, L]
    J = len(struct)
    lead = m.shape[:-2]
    flat = m.reshape((-1, J, m.shape[-1]))                     # [N, J, R]
    gidx = np.broadcast_to(np.maximum(sidx, 0).reshape(1, J, 36), (flat.shape[0], J, 36))
    g = np.take_along_axis(flat, gidx, axis=-1).reshape(-1, J, 4, 3, 3)
    g = np.where(sidx[None] >= 0, g, 1.0)
    path = np.where(sidx[None, ..., 0] >= 0, g.prod(-1), 0.0)  # [N, J, A, P]
    alt = 1.0 - (1.0 - path).prod(-1)                          # [N, J, A]
    alt = np.where(alt_mask[list(struct)][None], alt, -np.inf)
    return alt.reshape(lead + (J, 4))
