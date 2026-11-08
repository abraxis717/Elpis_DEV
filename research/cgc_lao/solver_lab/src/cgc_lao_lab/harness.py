"""Causal rollouts and per-seed evaluation.

``rollout`` is the only place where an agent meets the simulator. The agent receives the public view
(router, contexts, queries) and, after each selection, ``EpisodeBatch.readings`` of the selected expert:
the correctly indexed potential outcome for *its own* action (on-policy) or for a supplied action log
(fixed-log diagnostic). Regret is computed afterwards from the hidden truth, outside the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import env
from .agents import Agent, decide_from_pred
from .env import CONTRACTS, K, REGIMES, STOP, Contract, EpisodeBatch


@dataclass
class Rollout:
    actions: np.ndarray      # [E, T]
    decisions: np.ndarray    # [E, Q]
    terminal: np.ndarray     # [E]
    online: np.ndarray       # [E]


def privileged_inputs(spec: dict | None, batch: EpisodeBatch) -> dict | None:
    """Map a privileged agent's declared needs to hidden simulator values (oracles only)."""
    if spec is None:
        return None
    if spec.get("q_override_from") == "q_episode":
        return {"q_override": batch.q_episode[:, None]}
    raise ValueError("UNKNOWN_PRIVILEGED_SPEC")


def rollout(agent: Agent, batch: EpisodeBatch, contract: Contract, actions: np.ndarray | None = None,
            privileged: dict | None = None) -> Rollout:
    pub = batch.public()
    E, T = batch.count, contract.horizon
    if T > env.T_MAX or contract.max_reads > env.R_MAX:
        raise ValueError("CONTRACT_EXCEEDS_GENERATED_OUTCOMES")
    agent.begin(pub.router_w, E, contract, privileged)
    taken = np.zeros((E, T), dtype=np.int64)
    online = np.zeros(E)
    for t in range(T):
        x_t = pub.x[:, t]
        a = agent.act(t, x_t) if actions is None else actions[:, t]
        a = np.asarray(a, dtype=np.int64)
        reads = batch.readings(t, a, contract.reads[t])
        agent.observe(t, x_t, a, reads)
        taken[:, t] = a
        online += env.online_regret_step(batch.p[:, t], a)
    decisions = np.asarray(agent.decide(pub.xq), dtype=np.int64)
    if decisions.shape != (E, env.Q) or decisions.min() < 0 or decisions.max() > STOP:
        raise ValueError("DECISION_CONTRACT")
    return Rollout(taken, decisions, env.terminal_regret(batch.pq, decisions), online / T)


def noop_decisions(batch: EpisodeBatch) -> np.ndarray:
    return decide_from_pred(env.sigmoid(env.logits(batch.router_w, batch.xq)))


def regret_split(batch: EpisodeBatch, decisions: np.ndarray):
    """Terminal regret split into STOP-boundary errors and expert-ranking errors (per episode)."""
    pq, s = batch.pq, env.STOP_PAYOFF
    best_expert = pq.max(-1)
    opt_stop = best_expert < s
    chosen = np.take_along_axis(pq, np.minimum(decisions, K - 1)[..., None], -1)[..., 0]
    util = np.where(decisions == STOP, s, chosen)
    regret = np.maximum(best_expert, s) - util
    boundary = (decisions == STOP) != opt_stop
    return (regret * boundary).mean(-1), (regret * ~boundary).mean(-1)


def evaluate_batch(make_agent, batch: EpisodeBatch, contract: Contract, noop_dec: np.ndarray) -> dict:
    """Run one arm on one batch; return per-episode sums for aggregation."""
    agent, priv_spec = make_agent(batch.regime, contract)
    if agent is None:
        return {}
    out = rollout(agent, batch, contract, privileged=privileged_inputs(priv_spec, batch))
    boundary, ranking = regret_split(batch, out.decisions)
    res = {
        "terminal": out.terminal.sum(),
        "online": out.online.sum(),
        "stop": (out.decisions == STOP).mean(-1).sum(),
        "diff_noop": (out.decisions != noop_dec).mean(-1).sum(),
        "boundary": boundary.sum(),
        "ranking": ranking.sum(),
    }
    filt = getattr(agent, "filter", None)
    if filt is not None and 0.0 in list(filt.qs) and not agent.privileged:
        res["p_clean"] = filt.weights()[:, list(filt.qs).index(0.0)].sum()
    return res


def evaluate_seed(partition: str, seed: int, regime_names, arm_factories: dict, contract_name: str,
                  episodes: int, chunk: int = 2048) -> dict:
    """Per-arm, per-regime episode means for one router seed."""
    contract = CONTRACTS[contract_name]
    out: dict = {}
    for rname in regime_names:
        regime = REGIMES[rname]
        sums: dict = {}
        for first in range(0, episodes, chunk):
            batch = env.episodes(partition, seed, regime, first, min(chunk, episodes - first))
            nd = noop_decisions(batch)
            for arm, make in arm_factories.items():
                res = evaluate_batch(make, batch, contract, nd)
                acc = sums.setdefault(arm, {})
                for key, value in res.items():
                    acc[key] = acc.get(key, 0.0) + float(value)
        out[rname] = {arm: {k: v / episodes for k, v in acc.items()} for arm, acc in sums.items()}
    return out
