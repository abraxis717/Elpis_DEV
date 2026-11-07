"""v2 agents: the generalised reliability-mixture learner, equal-information controls and the privileged oracle.

Every agent keeps the v1 deployment contract (public router, contexts, its own selected readings, terminal
queries; frozen coefficients; fixed-size volatile state reset by ``begin``). ``observe_extra`` incorporates a
reading of a *second, independent event* of the selected expert at the same context without advancing the
acquisition step (provider mode SECOND_EVENT); only Bayes agents and NOOP support it.
"""
from __future__ import annotations

import numpy as np

from cgc_lao_lab import agents as v1
from cgc_lao_lab import cgc as cgcmod
from cgc_lao_lab.env import K, logits, sigmoid
from cgc_lao_lab.filters import NODES, V_FLOOR, WEIGHTS
from cgc_lao_lab.selector import STATE_CAP, greedy_with_deadline, probe_early_from_pred

from .filter2 import ReliabilityMixtureFilter2

# Corruption-rate grid of the disagreement-only reliability estimate used by two controls.
DISAGREE_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.48])
_PD = 2.0 * DISAGREE_GRID * (1.0 - DISAGREE_GRID)


class BayesAgent2(v1.BayesAgent):
    """v1 BayesAgent on the generalised filter: hypotheses [(q or [q nodes], rho), ...]."""
    name = "RMF2"

    def __init__(self, hypotheses, prior, bias_var=1.0, acquisition="probe", decision="bayes", lam=1.0, tau=0.0,
                 single_read=False, name=None, privileged=False):
        self.filter = ReliabilityMixtureFilter2(hypotheses, prior, bias_var)
        if acquisition not in ("probe", "greedy", "infoq") or decision not in ("bayes", "conservative", "minimax"):
            raise ValueError("BAYES_AGENT_RULES")
        self.acquisition, self.decision = acquisition, decision
        self.lam, self.tau = float(lam), float(tau)
        self.single_read = bool(single_read)
        self.privileged = bool(privileged)
        if name:
            self.name = name

    def observe_extra(self, t, x_t, actions, reads) -> None:
        if t != self.t - 1:
            raise ValueError("EXTRA_EVENT_STEP_ORDER")
        z = logits(self.router_w, x_t)[np.arange(self.E), actions]
        self.filter.update(actions, z, self._readings(reads))


class NoopAgent2(v1.NoopAgent):
    def observe_extra(self, t, x_t, actions, reads) -> None:
        if t != self.t - 1:
            raise ValueError("EXTRA_EVENT_STEP_ORDER")


class BetaAllAgent(v1.BetaAgent):
    """Equal-information Beta-Bernoulli: each event contributes the mean of all its readings."""
    name = "BETA_ALL"
    single_read = False


class SGDAllAgent(v1.SGDAgent):
    """Equal-information bounded SGD on the mean of all readings of the event."""
    name = "SGD_ALL"
    single_read = False


def disagreement_posterior(pairs: np.ndarray, disagree: np.ndarray) -> np.ndarray:
    """Posterior over DISAGREE_GRID (uniform prior) from duplicate pairs and their disagreements; [E, G]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logl = disagree[:, None] * np.log(_PD) + (pairs - disagree)[:, None] * np.log1p(-_PD)
    logl = np.where(np.isnan(logl), 0.0, logl)          # 0 * log(0) when no disagreement was seen
    w = np.exp(logl - logl.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


class DisagreeSGDAgent(v1.PlugInAgent):
    """Disagreement-aware SGD: the step is scaled by the posterior mean information ratio E[(1-2q)^2], with
    q inferred from duplicate disagreements alone (bias-free)."""
    name = "DISAGREE_SGD"
    single_read = False

    def __init__(self, eta: float):
        self.eta = float(eta)

    def _reset(self, privileged) -> None:
        super()._reset(privileged)
        self.pairs = np.zeros(self.E)
        self.disagree = np.zeros(self.E)

    def _update(self, actions, z, reads) -> None:
        if reads.shape[1] == 2:
            self.pairs += 1.0
            self.disagree += (reads[:, 0] != reads[:, 1])
        ratio = (disagreement_posterior(self.pairs, self.disagree) * (1.0 - 2.0 * DISAGREE_GRID) ** 2).sum(1)
        idx = np.arange(self.E)
        s = self.state[idx, actions]
        self.state[idx, actions] = np.clip(s + self.eta * ratio * (reads.mean(-1) - sigmoid(z + s)), -STATE_CAP, STATE_CAP)

    def state_scalars(self) -> int:
        return 2 * K + 2

    def volatile_state(self) -> dict:
        return {**super().volatile_state(), "pairs": self.pairs, "disagree": self.disagree}


class PluginQBayesAgent(v1.Agent):
    """Simpler Bayesian inference with the same readings: one Gaussian bias belief per expert, updated by the
    same ADF moment matching, under a plug-in corruption rate q_hat = posterior mean of q given duplicate
    disagreements. No hypothesis mixture; PROBE_EARLY and the Bayes decision as RMF."""
    name = "PLUGIN_Q_BAYES"
    single_read = False

    def _reset(self, privileged) -> None:
        self.mu = np.zeros((self.E, K))
        self.v = np.ones((self.E, K))
        self.pairs = np.zeros(self.E)
        self.disagree = np.zeros(self.E)

    def _predictive(self, z: np.ndarray) -> np.ndarray:
        shape = z.shape
        zz = z.reshape(shape[0], -1, K)
        b = self.mu[:, None, :, None] + np.sqrt(self.v)[:, None, :, None] * NODES
        return (sigmoid(zz[..., None] + b) * WEIGHTS).sum(-1).reshape(shape)

    def act(self, t, x_t) -> np.ndarray:
        pred = self._predictive(logits(self.router_w, x_t)[:, None, :])[:, 0, :]
        return probe_early_from_pred(t, pred, self.counts, self.horizon)

    def observe(self, t, x_t, actions, reads) -> None:
        reads = self._readings(reads)
        if reads.shape[1] == 2:
            self.pairs += 1.0
            self.disagree += (reads[:, 0] != reads[:, 1])
        q = (disagreement_posterior(self.pairs, self.disagree) * DISAGREE_GRID).sum(1)[:, None]
        n1 = reads.sum(-1).astype(np.float64)[:, None]
        n0 = reads.shape[1] - n1
        a1 = (1.0 - q) ** n1 * q ** n0
        a0 = q ** n1 * (1.0 - q) ** n0
        idx = np.arange(self.E)
        z = logits(self.router_w, x_t)[idx, actions]
        mu, v = self.mu[idx, actions], self.v[idx, actions]
        b = mu[:, None] + np.sqrt(v)[:, None] * NODES
        p = sigmoid(z[:, None] + b)
        lik = (p * a1 + (1.0 - p) * a0) * WEIGHTS
        L = lik.sum(-1)
        ok = L > 1e-300
        Ls = np.where(ok, L, 1.0)
        m1 = (lik * b).sum(-1) / Ls
        m2 = (lik * b * b).sum(-1) / Ls
        self.mu[idx, actions] = np.where(ok, m1, mu)
        self.v[idx, actions] = np.where(ok, np.maximum(m2 - m1 * m1, V_FLOOR), v)
        self._advance(t, actions)

    def decide(self, xq) -> np.ndarray:
        return v1.decide_from_pred(self._predictive(logits(self.router_w, xq)))

    def state_scalars(self) -> int:
        return 2 * K + K + 2

    def volatile_state(self) -> dict:
        return {"mu": self.mu, "v": self.v, "counts": self.counts, "pairs": self.pairs, "disagree": self.disagree}


class CGCMeanReadingAgent(v1.CGCAgent):
    """Static attenuation with the same readings: the frozen CGC proposal (x attenuation) on the residual
    against the mean reading of the event."""
    name = "STATIC_ATT_ALL"

    def observe(self, t, x_t, actions, reads) -> None:
        base = logits(self.w32, x_t.astype(np.float32))
        reward = reads.mean(-1).astype(np.float32)
        self.state, _ = cgcmod.cgc_step(t, base, self.state, actions, reward, self.w, self.attenuation)
        self._advance(t, actions)


def known_q_oracle2(regime):
    """PRIVILEGED: told the regime's corruption rate, reading correlation and bias scale (symmetric, event,
    correlated, qunif mechanisms only)."""
    m = regime.mechanism
    var = regime.bias_scale ** 2
    if m in ("symmetric", "event", "correlated"):
        rho = 1.0 if m == "event" else (regime.rho if m == "correlated" else 0.0)
        return BayesAgent2([(regime.q, rho)], [1.0], var, name="KNOWN_Q_ORACLE", privileged=True), None
    if m == "qunif":
        return BayesAgent2([(0.0, 0.0)], [1.0], var, name="KNOWN_Q_ORACLE", privileged=True), {"q_override_from": "q_episode"}
    return None, None
