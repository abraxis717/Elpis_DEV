"""Deployed learners. Each agent sees only the public router, contexts, its own selected readings and the
terminal queries (``begin``/``act``/``observe``/``decide``). Coefficients are frozen at construction;
volatile state is created by ``begin`` (a reset) and has a fixed size per episode.

Baselines: NOOP, SGD, BETA, CGC_REFERENCE, STATIC_ATTENUATION, CLEAN_BAYES.
Privileged reference (never a candidate): KNOWN_Q_ORACLE.
Candidates: RMF agents (reliability-mixture filter + acquisition + decision rule) and the gated-CGC hybrid.
"""
from __future__ import annotations

import numpy as np

from . import cgc as cgcmod
from .env import K, STOP, STOP_PAYOFF, Contract, logits, sigmoid
from .filters import ReliabilityMixtureFilter
from .selector import STATE_CAP, choose_probe_early, greedy_with_deadline, probe_early_from_pred


def decide_from_pred(pred: np.ndarray, stop_payoff: float = STOP_PAYOFF) -> np.ndarray:
    """Expected-utility decision: the best expert, or STOP when no expert beats the STOP payoff."""
    return np.where(pred.max(-1) < stop_payoff, STOP, pred.argmax(-1))


class Agent:
    """Interface and shared bookkeeping (public router copy, coverage counts)."""
    name = "AGENT"
    privileged = False
    single_read = True   # True: the agent's model uses the first reading of an event only

    def begin(self, router_w: np.ndarray, E: int, contract: Contract, privileged=None) -> None:
        if privileged is not None and not self.privileged:
            raise PermissionError("PRIVILEGED_INPUT_TO_ADMISSIBLE_AGENT")
        if router_w.shape != (K, 4) or not np.isfinite(router_w).all():
            raise ValueError("ROUTER_CONTRACT")
        self.router_w = np.array(router_w, dtype=np.float64)
        self.router_w.setflags(write=False)
        self.E = E
        self.horizon = contract.horizon
        self.counts = np.zeros((E, K))
        self.t = 0
        self._reset(privileged)

    def _reset(self, privileged) -> None:
        raise NotImplementedError

    def _readings(self, reads: np.ndarray) -> np.ndarray:
        return reads[:, :1] if self.single_read else reads

    def _advance(self, t: int, actions: np.ndarray) -> None:
        if t != self.t:
            raise ValueError("STEP_ORDER")
        self.counts[np.arange(self.E), actions] += 1
        self.t += 1

    def state_scalars(self) -> int:
        raise NotImplementedError

    def volatile_state(self) -> dict:
        raise NotImplementedError


class PlugInAgent(Agent):
    """A bias-correction state s[E, 3]; PROBE_EARLY on sigmoid(Wx + s); plug-in terminal decision."""

    def _reset(self, privileged) -> None:
        self.state = np.zeros((self.E, K))

    def act(self, t: int, x_t: np.ndarray) -> np.ndarray:
        return choose_probe_early(t, logits(self.router_w, x_t), self.state, self.counts, self.horizon)

    def observe(self, t, x_t, actions, reads) -> None:
        z = logits(self.router_w, x_t)[np.arange(self.E), actions]
        self._update(actions, z, self._readings(reads))
        self._advance(t, actions)

    def _update(self, actions, z, reads) -> None:
        pass

    def decide(self, xq: np.ndarray) -> np.ndarray:
        return decide_from_pred(sigmoid(logits(self.router_w, xq) + self.state[:, None, :]))

    def state_scalars(self) -> int:
        return 2 * K   # state + counts

    def volatile_state(self) -> dict:
        return {"state": self.state, "counts": self.counts}


class NoopAgent(PlugInAgent):
    name = "NOOP"


class SGDAgent(PlugInAgent):
    """Bounded online logistic correction: s_a <- clip(s_a + eta * (y - sigmoid(z + s_a)), +-1.5)."""
    name = "SGD"

    def __init__(self, eta: float):
        self.eta = float(eta)

    def _update(self, actions, z, reads) -> None:
        idx = np.arange(self.E)
        s = self.state[idx, actions]
        y = reads.mean(-1)
        self.state[idx, actions] = np.clip(s + self.eta * (y - sigmoid(z + s)), -STATE_CAP, STATE_CAP)


class BetaAgent(PlugInAgent):
    """Selected-feedback Beta-Bernoulli calibration per expert. With n observations, S successes and
    mean router prediction m, the rate posterior mean is r = (kappa*m + S)/(kappa + n) under a
    Beta(kappa*m, kappa*(1-m)) prior centred on the router; the bias estimate is logit(r) - logit(m)."""
    name = "BETA"

    def __init__(self, kappa: float):
        self.kappa = float(kappa)

    def _reset(self, privileged) -> None:
        super()._reset(privileged)
        self.succ = np.zeros((self.E, K))
        self.pred_sum = np.zeros((self.E, K))

    def _update(self, actions, z, reads) -> None:
        idx = np.arange(self.E)
        self.succ[idx, actions] += reads.mean(-1)
        self.pred_sum[idx, actions] += sigmoid(z)
        n = self.counts.copy()
        n[idx, actions] += 1   # counts are advanced after the update
        seen = n > 0
        m = np.clip(np.where(seen, self.pred_sum / np.maximum(n, 1), 0.5), 1e-6, 1 - 1e-6)
        r = np.clip((self.kappa * m + self.succ) / (self.kappa + n), 1e-6, 1 - 1e-6)
        est = np.log(r / (1 - r)) - np.log(m / (1 - m))
        self.state = np.where(seen, np.clip(est, -STATE_CAP, STATE_CAP), 0.0)

    def state_scalars(self) -> int:
        return 4 * K   # state, counts, successes, prediction sums

    def volatile_state(self) -> dict:
        return {**super().volatile_state(), "succ": self.succ, "pred_sum": self.pred_sum}


class CGCAgent(Agent):
    """The checked-in Q4E5 CGC episode (float32), with frozen externally trained weights."""
    name = "CGC_REFERENCE"

    def __init__(self, weights: np.ndarray, attenuation: float = 1.0, name: str | None = None):
        self.w = cgcmod.frozen_weights(weights)
        self.attenuation = float(attenuation)
        if name:
            self.name = name

    def _reset(self, privileged) -> None:
        if self.horizon != cgcmod.HORIZON:
            raise ValueError("CGC_REFERENCE_IS_DEFINED_FOR_T8_ONLY")
        self.w32 = self.router_w.astype(np.float32)
        self.state = np.zeros((self.E, K), dtype=np.float32)
        self.counts = np.zeros((self.E, K), dtype=np.float32)

    def act(self, t, x_t) -> np.ndarray:
        base = logits(self.w32, x_t.astype(np.float32))
        return choose_probe_early(t, base, self.state, self.counts, self.horizon)

    def observe(self, t, x_t, actions, reads) -> None:
        base = logits(self.w32, x_t.astype(np.float32))
        reward = reads[:, 0].astype(np.float32)
        self.state, _ = cgcmod.cgc_step(t, base, self.state, actions, reward, self.w, self.attenuation)
        self._advance(t, actions)

    def decide(self, xq) -> np.ndarray:
        prob = sigmoid(logits(self.w32, xq.astype(np.float32)) + self.state[:, None, :])
        return decide_from_pred(prob.astype(np.float64))

    def state_scalars(self) -> int:
        return 2 * K

    def volatile_state(self) -> dict:
        return {"state": self.state, "counts": self.counts}


class BayesAgent(Agent):
    """Reliability-mixture filter + acquisition rule + decision rule.

    acquisition: 'probe'  - PROBE_EARLY on posterior predictive success probabilities;
                 'greedy' - argmax predictive probability, with PROBE_EARLY's deadline coverage;
                 'infoq'  - argmax [reliability information + lam * predictive probability], same coverage.
    decision:    'bayes'  - argmax posterior expected utility (experts and STOP);
                 'conservative' - keep the no-learning decision unless the posterior expected advantage
                 of the Bayes decision exceeds tau posterior standard deviations.
    """
    name = "RMF"

    def __init__(self, qs, prior, bias_var=1.0, kinds=None, acquisition="probe", decision="bayes",
                 lam=1.0, tau=0.0, single_read=False, name=None, privileged=False):
        self.filter = ReliabilityMixtureFilter(qs, prior, bias_var, kinds)
        if acquisition not in ("probe", "greedy", "infoq") or decision not in ("bayes", "conservative", "minimax"):
            raise ValueError("BAYES_AGENT_RULES")
        self.acquisition, self.decision = acquisition, decision
        self.lam, self.tau = float(lam), float(tau)
        self.single_read = bool(single_read)
        self.privileged = bool(privileged)
        if name:
            self.name = name

    def _reset(self, privileged) -> None:
        q_override = None
        if privileged is not None:
            q_override = privileged.get("q_override")
        self.filter.reset(self.E, q_override)

    def act(self, t, x_t) -> np.ndarray:
        base = logits(self.router_w, x_t)
        pred = self.filter.predictive(base[:, None, :])[:, 0, :]
        if self.acquisition == "probe":
            return probe_early_from_pred(t, pred, self.counts, self.horizon)
        if self.acquisition == "greedy":
            return greedy_with_deadline(t, pred, self.counts, self.horizon)
        score = self.filter.reliability_information(base) + self.lam * pred
        return greedy_with_deadline(t, score, self.counts, self.horizon)

    def observe(self, t, x_t, actions, reads) -> None:
        z = logits(self.router_w, x_t)[np.arange(self.E), actions]
        self.filter.update(actions, z, self._readings(reads))
        self._advance(t, actions)

    def decide(self, xq) -> np.ndarray:
        base = logits(self.router_w, xq)
        if self.decision == "bayes":
            return decide_from_pred(self.filter.predictive(base))
        if self.decision == "minimax":
            return minimax_regret_decisions(self.filter, base, self.tau)
        return conservative_decisions(self.filter, base, self.tau)

    def state_scalars(self) -> int:
        return self.filter.state_scalars() + K

    def volatile_state(self) -> dict:
        return {**self.filter.volatile_state(), "counts": self.counts}


def _utility_moments(mean, second, d):
    """Per-hypothesis mean and second moment of U(d) for decisions d[E, Q] (STOP = constant payoff)."""
    dd = np.minimum(d, K - 1)[:, :, None, None]
    m = np.take_along_axis(mean, np.broadcast_to(dd, mean.shape[:3] + (1,)), -1)[..., 0]
    s = np.take_along_axis(second, np.broadcast_to(dd, second.shape[:3] + (1,)), -1)[..., 0]
    stop = (d == STOP)[:, :, None]
    return np.where(stop, STOP_PAYOFF, m), np.where(stop, STOP_PAYOFF ** 2, s)


def conservative_decisions(filt: ReliabilityMixtureFilter, base: np.ndarray, tau: float) -> np.ndarray:
    """Adopt the Bayes decision over the no-learning decision only when its posterior expected advantage
    exceeds ``tau`` posterior standard deviations of the advantage (mixture over hypotheses; experts are
    a-posteriori independent within a hypothesis)."""
    mean, second = filt.component_moments(base)
    w = filt.weights()[:, None, :]
    d_star = decide_from_pred((mean * w[..., None]).sum(2))
    d0 = decide_from_pred(sigmoid(base))
    m1, s1 = _utility_moments(mean, second, d_star)
    m0, s0 = _utility_moments(mean, second, d0)
    both_experts = (d_star != STOP) & (d0 != STOP)
    cross = np.where(both_experts[:, :, None], m1 * m0, np.where((d_star == STOP)[:, :, None], STOP_PAYOFF * m0,
                                                                  STOP_PAYOFF * m1))
    ed = ((m1 - m0) * w).sum(-1)
    ed2 = ((s1 + s0 - 2 * cross) * w).sum(-1)
    sd = np.sqrt(np.maximum(ed2 - ed * ed, 0.0))
    adopt = (d_star != d0) & (ed > tau * sd)
    return np.where(adopt, d_star, d0)


def minimax_regret_decisions(filt: ReliabilityMixtureFilter, base: np.ndarray, floor: float) -> np.ndarray:
    """Minimax expected regret across the reliability hypotheses still plausible (posterior weight above
    ``floor``): for each hypothesis, the expected utility of every action (experts and STOP); choose the
    action whose worst-case shortfall from that hypothesis's best action is smallest."""
    mean, _ = filt.component_moments(base)                           # [E, Q, M, K]
    util = np.concatenate((mean, np.full(mean.shape[:3] + (1,), STOP_PAYOFF)), -1)   # [E, Q, M, K+1]
    shortfall = util.max(-1, keepdims=True) - util
    plausible = (filt.weights() > floor)[:, None, :, None]
    worst = np.where(plausible, shortfall, -np.inf).max(2)           # [E, Q, K+1]
    return worst.argmin(-1)


class GatedCGCAgent(Agent):
    """Family B hybrid: the frozen CGC proposes the state correction; a reliability-mixture filter run in
    parallel scales each proposal by the posterior mean information ratio E[(1-2q)^2 | history]."""
    name = "GATED_CGC"

    def __init__(self, weights, qs, prior, bias_var=1.0, name=None):
        self.w = cgcmod.frozen_weights(weights)
        self.filter = ReliabilityMixtureFilter(qs, prior, bias_var)
        if name:
            self.name = name

    def _reset(self, privileged) -> None:
        if self.horizon != cgcmod.HORIZON:
            raise ValueError("CGC_REFERENCE_IS_DEFINED_FOR_T8_ONLY")
        self.w32 = self.router_w.astype(np.float32)
        self.state = np.zeros((self.E, K), dtype=np.float32)
        self.counts = np.zeros((self.E, K), dtype=np.float32)
        self.filter.reset(self.E)

    def act(self, t, x_t):
        return choose_probe_early(t, logits(self.w32, x_t.astype(np.float32)), self.state, self.counts,
                                  self.horizon)

    def observe(self, t, x_t, actions, reads) -> None:
        base = logits(self.w32, x_t.astype(np.float32))
        reads = reads[:, :1]
        self.filter.update(actions, logits(self.router_w, x_t)[np.arange(self.E), actions], reads)
        gate = self.filter.gain_ratio().astype(np.float32)
        nxt, delta = cgcmod.cgc_step(t, base, self.state, actions, reads[:, 0].astype(np.float32), self.w)
        applied = gate[:, None] * np.clip(delta, -cgcmod.STEP_CAP, cgcmod.STEP_CAP)
        self.state = np.clip(self.state - applied, -cgcmod.STATE_CAP, cgcmod.STATE_CAP).astype(np.float32)
        self._advance(t, actions)

    def decide(self, xq):
        prob = sigmoid(logits(self.w32, xq.astype(np.float32)) + self.state[:, None, :])
        return decide_from_pred(prob.astype(np.float64))

    def state_scalars(self) -> int:
        return 2 * K + self.filter.state_scalars()

    def volatile_state(self) -> dict:
        return {"state": self.state, "counts": self.counts, **self.filter.volatile_state()}


def clean_bayes(acquisition="probe") -> BayesAgent:
    """Admissible clean-specialised Bayes learner (assumes uncorrupted feedback; first reading only)."""
    return BayesAgent([0.0], [1.0], acquisition=acquisition, single_read=True, name="CLEAN_BAYES")


def known_q_oracle(regime, contract: Contract) -> tuple[BayesAgent, dict | None]:
    """PRIVILEGED diagnostic reference: the Bayes learner told the true corruption rate (and, for
    ``event`` corruption, that readings share their corruption). Defined for symmetric/qunif/event."""
    kind = "event" if regime.mechanism == "event" else "channel"
    if regime.mechanism in ("symmetric", "event"):
        agent = BayesAgent([regime.q], [1.0], bias_var=regime.bias_scale ** 2, kinds=[kind],
                           single_read=False, name="KNOWN_Q_ORACLE", privileged=True)
        return agent, None
    if regime.mechanism == "qunif":
        agent = BayesAgent([0.0], [1.0], bias_var=regime.bias_scale ** 2, single_read=False,
                           name="KNOWN_Q_ORACLE", privileged=True)
        return agent, {"q_override_from": "q_episode"}
    return None, None
