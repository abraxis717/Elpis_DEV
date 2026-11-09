"""Layer B: bounded relational belief states (SPEC.md section 4), vectorised over episodes.

RelationalFilter is the RMF3 update law (cgc_lao_v2.scalar2 / filter2) with the expert index generalised to a
fixed-capacity relation slot and the reliability log-weights decomposed per slot: Lambda[h, r] is the summed log
evidence of slot r's readings under hypothesis h. Under every h the slots are independent, so
log w_h = log pi_h + sum_r Lambda[h, r] exactly; removing a slot's term removes its evidence exactly, and replaying
the retained LIVE events reproduces the state a learner would hold had it never seen the retired ones.

Learners differ only in what they read out and how they invalidate:
  coupled     E[U] = sum_h w_h U(m_h)           (RRMF: shared reliability, hypothesis-coupled composition)
  marginal    E[U] = U(sum_h w_h m_h)           (RRMF_MARGINAL: shared reliability, marginal product)
  independent E[U] = U(sum_h w_{h,r} m_{h,r})   (INDEPENDENT_RMF: a reliability posterior per slot)
"""
from __future__ import annotations

import numpy as np

from cgc_lao_lab.env import sigmoid
from cgc_lao_lab.filters import LW_FLOOR, NODES, V_FLOOR, WEIGHTS

from .world import T

H3 = ((0.0, 0.0), (0.4, 0.0), (0.48, 0.0))
H4 = ((0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.48, 0.0))
STATE_CAP = 1.5
KAPPA = 8.0          # STATIC_COUNTS prior strength: the v2 TRAIN value of BETA_ALL (coefficients/baselines.v2.json)


def gh_mean(z: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    """E[sigmoid(z + b)], b ~ N(mu, v), by the 16-node Gauss-Hermite rule (broadcasting z against mu, v)."""
    sd = np.sqrt(v)
    acc = np.zeros(np.broadcast(z, mu).shape)
    for g in range(16):
        acc = acc + WEIGHTS[g] * sigmoid(z + (mu + sd * NODES[g]))
    return acc


def sorted_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A sum independent of the order of its terms (relation slots): sort, then add left to right."""
    s = np.sort(x, axis=axis)
    out = np.take(s, 0, axis=axis).copy()
    for i in range(1, s.shape[axis]):
        out = out + np.take(s, i, axis=axis)
    return out


class EventLog:
    """The bounded, provenance-indexed buffer of one batch: at most one event per step (T = 8 events, at most
    12 readings). Fields per event: relation, logit, readings, binding (source index, epoch)."""

    def __init__(self, E: int):
        self.rel = -np.ones((E, T), dtype=np.int64)
        self.z = np.zeros((E, T))
        self.reads = np.zeros((E, T, 2), dtype=np.uint8)
        self.n = np.zeros(T, dtype=np.int64)
        self.src = np.zeros((E, T), dtype=np.int64)
        self.epoch = np.zeros((E, T), dtype=np.int8)

    def add(self, t, rel, z, reads, src, epoch, mask):
        self.rel[:, t] = np.where(mask, rel, -1)
        self.z[:, t], self.n[t] = z, reads.shape[1]
        self.reads[:, t, :reads.shape[1]] = reads
        self.src[:, t], self.epoch[:, t] = src, epoch

    def live(self, batch, t_now: int) -> np.ndarray:
        """[E, T]: events that are LIVE under the authority at time t_now (the view's law)."""
        return (self.rel >= 0) & batch.live(self.src, self.epoch, t_now)

    def retired(self, batch, t_now: int) -> np.ndarray:
        """[E, T]: events whose binding is RETIRED (a newer epoch: revision or reclamation)."""
        return (self.rel >= 0) & (batch.epoch(self.src, t_now) != self.epoch)


class RelationalFilter:
    """mu, v, lam [E, M, R] and live counts [E, R]: the RMF3 law per selected slot."""

    def __init__(self, hypotheses=H3, prior=(1.0, 1.0, 1.0), bias_var: float = 1.0):
        self.q = np.array([h[0] for h in hypotheses], dtype=np.float64)
        self.rho = np.array([h[1] for h in hypotheses], dtype=np.float64)
        prior = np.asarray(prior, dtype=np.float64)
        self.log_prior = np.log(prior / prior.sum())
        self.M, self.bias_var = len(self.q), float(bias_var)

    def reset(self, E: int, R: int, q_override=None, rho_override=None) -> None:
        self.E, self.R, self.override = E, R, (q_override, rho_override)
        self.mu = np.zeros((E, self.M, R))
        self.v = np.full((E, self.M, R), self.bias_var)
        self.lam = np.zeros((E, self.M, R))
        self.counts = np.zeros((E, R), dtype=np.int64)
        qq = self.q if q_override is None else np.asarray(q_override, float).reshape(E, 1)
        rr = self.rho if rho_override is None else np.full((E, 1), float(rho_override))
        self.qe = np.broadcast_to(qq, (E, self.M)).copy()
        self.rhoe = np.broadcast_to(rr, (E, self.M)).copy()

    def state_scalars(self) -> int:
        return 3 * self.M * self.R + self.R

    def evidence(self, reads: np.ndarray):
        """A1, A0 [E, M]: probability of the readings given clean outcome 1 / 0 (scalar2._evidence, vectorised)."""
        q, rho = self.qe, self.rhoe
        out = []
        for c in (1, 0):
            f = (reads ^ c).astype(np.float64)
            if reads.shape[1] == 1:
                f0 = f[:, 0:1]
                p = q ** f0 * (1.0 - q) ** (1.0 - f0)
            else:
                s = (f[:, 0] + f[:, 1])[:, None]
                indep = q ** s * (1.0 - q) ** (2.0 - s)
                f0 = f[:, 0:1]
                shared = np.where(s == 1.0, 0.0, q ** f0 * (1.0 - q) ** (1.0 - f0))
                p = (1.0 - rho) * indep + rho * shared
            out.append(p)
        return out[0], out[1]

    def likelihood(self, slot: np.ndarray, z: np.ndarray, reads: np.ndarray):
        """L [E, M] and the matched moments (m1, m2) of slot `slot` after the readings; no state change."""
        idx = np.arange(self.E)
        a1, a0 = self.evidence(reads)
        mu, v = self.mu[idx, :, slot], self.v[idx, :, slot]
        b = mu[..., None] + np.sqrt(v)[..., None] * NODES
        p = sigmoid(z[:, None, None] + b)
        lik = (p * a1[..., None] + (1.0 - p) * a0[..., None]) * WEIGHTS
        L = lik.sum(-1)
        Ls = np.where(L > 1e-300, L, 1.0)
        m1 = (lik * b).sum(-1) / Ls
        m2 = (lik * b * b).sum(-1) / Ls
        return L, m1, m2, mu, v

    def update(self, slot: np.ndarray, z: np.ndarray, reads: np.ndarray, mask: np.ndarray) -> None:
        idx = np.arange(self.E)
        slot = np.where(mask, slot, 0)
        L, m1, m2, mu, v = self.likelihood(slot, z, reads)
        ok = L > 1e-300
        upd = mask[:, None]
        self.mu[idx, :, slot] = np.where(upd & ok, m1, mu)
        self.v[idx, :, slot] = np.where(upd & ok, np.maximum(m2 - m1 * m1, V_FLOOR), v)
        lam = self.lam[idx, :, slot]
        self.lam[idx, :, slot] = np.where(upd, lam + np.where(ok, np.log(np.where(ok, L, 1.0)), LW_FLOOR), lam)
        self.counts[idx, slot] += mask.astype(np.int64)

    def log_weights(self) -> np.ndarray:
        return self.log_prior[None, :] + sorted_sum(self.lam, axis=-1)

    def weights(self) -> np.ndarray:
        lw = self.log_weights()
        w = np.exp(lw - lw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def slot_weights(self) -> np.ndarray:
        """[E, M, R]: the per-slot reliability posteriors (INDEPENDENT_RMF)."""
        lw = self.log_prior[None, :, None] + self.lam
        w = np.exp(lw - lw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def means(self, z: np.ndarray) -> np.ndarray:
        """[E, M, J, R]: E_h[p_r] at logits z [E, J, R]."""
        return gh_mean(z[:, None, :, :], self.mu[:, :, None, :], self.v[:, :, None, :])

    def replay(self, log, keep: np.ndarray, upto: int) -> None:
        """Rebuild the state from the kept events of steps < upto, in order (exact: the same arithmetic)."""
        self.reset(self.E, self.R, *self.override)
        for s in range(upto):
            n = int(log.n[s])
            if n == 0:
                continue
            m = keep[:, s] & (log.rel[:, s] >= 0)
            self.update(log.rel[:, s], log.z[:, s], log.reads[:, s, :n], m)


class Learner:
    """Common interface: reset(batch), sync(batch, t), observe(...), components(z) -> [(w [E], m [E, J, R])],
    predictive(z) -> [E, R], live_counts() -> [E, R], state_scalars()."""
    name = "LEARNER"
    first_only = False
    privileged = False

    def reset(self, batch) -> None:
        self.batch, self.E, self.R = batch, batch.count, batch.R
        self.log = EventLog(self.E)
        self.t = 0

    def sync(self, batch, t: int) -> None:
        self.t = t

    def observe(self, t, rel, z, reads, src, epoch, mask) -> None:
        reads = reads[:, :1] if self.first_only else reads
        self.log.add(t, rel, z, reads, src, epoch, mask)
        self._update(t, rel, z, reads, mask)
        self.t = t + 1

    def _update(self, t, rel, z, reads, mask) -> None:
        pass

    def predictive(self, z: np.ndarray) -> np.ndarray:
        comps = self.components(z[:, None, :])
        return sum(w[:, None, None] * m for w, m in comps)[:, 0, :]

    def live_counts(self) -> np.ndarray:
        live = self.log.live(self.batch, self.t)
        out = np.zeros((self.E, self.R), dtype=np.int64)
        for s in range(T):
            m = live[:, s]
            np.add.at(out, (np.nonzero(m)[0], self.log.rel[m, s]), 1)
        return out


def prior_mean(z: np.ndarray) -> np.ndarray:
    return gh_mean(z, np.zeros(1), np.ones(1))


class Noop(Learner):
    name = "NOOP"

    def components(self, z):
        return [(np.ones(self.E), prior_mean(z))]

    def state_scalars(self) -> int:
        return 0


class Bayes(Learner):
    """RRMF (readout 'coupled'), RRMF_MARGINAL ('marginal'), INDEPENDENT_RMF ('independent'); invalidation
    'replay' (strategy B), 'slot' (strategy A) or 'none' (stale ablation); permuted: the identity ablation."""

    def __init__(self, name, hypotheses=H3, prior=(1.0, 1.0, 1.0), readout="coupled", invalidation="replay",
                 first_only=False, permuted=False, oracle=False):
        self.name, self.readout, self.invalidation = name, readout, invalidation
        self.first_only, self.permuted, self.privileged = first_only, permuted, oracle
        self.oracle = oracle
        self.filter = RelationalFilter(hypotheses, prior)

    def reset(self, batch) -> None:
        super().reset(batch)
        if self.oracle:
            self.filter = RelationalFilter(((0.25, 0.0),), (1.0,))
            self.filter.reset(self.E, self.R, batch.q_episode, batch.rho_episode)
        else:
            self.filter.reset(self.E, self.R)
        self.dropped_before = np.zeros((self.E, self.R), dtype=np.int64)
        self.kept = np.zeros((self.E, T), dtype=bool)

    def _slot(self, rel):
        return self.batch.perm[np.arange(self.E), rel] if self.permuted else rel

    def _update(self, t, rel, z, reads, mask) -> None:
        self.filter.update(self._slot(np.where(mask, rel, 0)), z, reads, mask)
        self.kept[:, t] = mask                                   # the event is applied: part of the kept state

    def _keep(self, batch, t):
        if self.invalidation == "none":
            return self.log.rel >= 0
        live = self.log.live(batch, t)
        if self.invalidation == "replay":                     # strategy B: exactly the LIVE events
            return live
        # strategy A (binding-local reset): a slot with retired support is reset (its events before t are dropped;
        # fresh evidence re-activates it); a slot with suspended support is excluded while it lasts
        valid = self.log.rel >= 0
        rel = np.where(valid, self.log.rel, 0)
        step = np.broadcast_to(np.arange(T)[None, :], rel.shape)
        current = valid & (step >= np.take_along_axis(self.dropped_before, rel, axis=1))
        rows = np.broadcast_to(np.arange(self.E)[:, None], rel.shape)
        retired = current & self.log.retired(batch, t)
        hit = np.zeros((self.E, self.R), dtype=bool)
        np.logical_or.at(hit, (rows[retired], rel[retired]), True)
        self.dropped_before = np.where(hit, t, self.dropped_before)
        current = valid & (step >= np.take_along_axis(self.dropped_before, rel, axis=1))
        suspended = current & ~live
        excluded = np.zeros((self.E, self.R), dtype=bool)
        np.logical_or.at(excluded, (rows[suspended], rel[suspended]), True)
        return current & ~np.take_along_axis(excluded, rel, axis=1)

    def sync(self, batch, t: int) -> None:
        super().sync(batch, t)
        keep = self._keep(batch, t)
        changed = (keep != self.kept).any(1) & (self.log.rel >= 0).any(1)
        if changed.any():
            saved = (self.filter.mu.copy(), self.filter.v.copy(), self.filter.lam.copy(), self.filter.counts.copy())
            self.filter.replay(_PermLog(self.log, self) if self.permuted else self.log, keep, t)
            for arr, old in zip((self.filter.mu, self.filter.v, self.filter.lam, self.filter.counts), saved):
                arr[~changed] = old[~changed]
        self.kept = keep

    def live_counts(self) -> np.ndarray:
        return self.filter.counts.copy()

    def components(self, z):
        m = self.filter.means(z)                                 # [E, M, J, R]
        if self.readout == "coupled":
            w = self.filter.weights()
            return [(w[:, h], m[:, h]) for h in range(self.filter.M)]
        if self.readout == "marginal":
            w = self.filter.weights()
            return [(np.ones(self.E), (w[:, :, None, None] * m).sum(1))]
        ws = self.filter.slot_weights()                          # [E, M, R]
        return [(np.ones(self.E), (ws[:, :, None, :] * m).sum(1))]

    def state_scalars(self) -> int:
        return self.filter.state_scalars()


class _PermLog:
    """The event log as the permuted learner saw it (relations mapped through the derangement)."""

    def __init__(self, log, learner):
        self.n, self.z, self.reads = log.n, log.z, log.reads
        valid = log.rel >= 0
        self.rel = np.where(valid, np.take_along_axis(learner.batch.perm, np.maximum(log.rel, 0), axis=1), -1)


class Exact(Learner):
    """EXACT_RELATIONAL: the live view's net polarity per relation, read as truth (+: 1, -: 0, none: prior)."""
    name = "EXACT_RELATIONAL"

    def components(self, z):
        net = self.net()
        pm = prior_mean(z)
        m = np.where(net[:, None, :] > 0, 1.0, np.where(net[:, None, :] < 0, 0.0, pm))
        return [(np.ones(self.E), m)]

    def net(self) -> np.ndarray:
        live = self.log.live(self.batch, self.t)
        net = np.zeros((self.E, self.R))
        for s in range(T):
            n = int(self.log.n[s])
            if n == 0:
                continue
            m = live[:, s]
            vote = (2.0 * self.log.reads[:, s, :n].astype(np.float64) - 1.0).sum(1)
            np.add.at(net, (np.nonzero(m)[0], self.log.rel[m, s]), vote[m])
        return net

    def state_scalars(self) -> int:
        return 2 * self.R


class Counts(Learner):
    """STATIC_COUNTS: Beta-Bernoulli per relation centred on the public prior (kappa = 8), live events only; each
    event contributes the mean of its readings (v2 BETA_ALL); plug-in composition."""
    name = "STATIC_COUNTS"

    def components(self, z):
        live = self.log.live(self.batch, self.t)
        succ, pred, n = (np.zeros((self.E, self.R)) for _ in range(3))
        for s in range(T):
            k = int(self.log.n[s])
            if k == 0:
                continue
            m = live[:, s]
            e, r = np.nonzero(m)[0], self.log.rel[m, s]
            np.add.at(succ, (e, r), self.log.reads[m, s, :k].mean(1))
            np.add.at(pred, (e, r), sigmoid(self.log.z[m, s]))
            np.add.at(n, (e, r), 1.0)
        seen = n > 0
        mbar = np.clip(np.where(seen, pred / np.maximum(n, 1), 0.5), 1e-6, 1 - 1e-6)
        rate = np.clip((KAPPA * mbar + succ) / (KAPPA + n), 1e-6, 1 - 1e-6)
        est = np.where(seen, np.clip(np.log(rate / (1 - rate)) - np.log(mbar / (1 - mbar)), -STATE_CAP, STATE_CAP), 0.0)
        return [(np.ones(self.E), sigmoid(z + est[:, None, :]))]

    def state_scalars(self) -> int:
        return 3 * self.R


# -- the decision layer (host-side evaluation of learner proposals) ----------------------------------------------
def expected_utilities(learner: Learner, z: np.ndarray, template, struct) -> np.ndarray:
    """[E, J, A]: the learner's expected utility of every alternative (sum over its readout components)."""
    from .world import utilities
    out = None
    for w, m in learner.components(z):
        u = utilities(m, template, struct)
        u = w[:, None, None] * np.where(np.isfinite(u), u, 0.0)
        out = u if out is None else out + u
    from .world import query_tensors
    _, alt_mask, _ = query_tensors(template)
    return np.where(alt_mask[list(struct)][None], out, -np.inf)


def decide(eu: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Argmax alternative (first index on ties) or STOP (-1) when the best expected utility is below the payoff."""
    best = eu.argmax(-1)
    top = np.take_along_axis(eu, best[..., None], -1)[..., 0]
    return np.where(top < stops[None, :], -1, best)
