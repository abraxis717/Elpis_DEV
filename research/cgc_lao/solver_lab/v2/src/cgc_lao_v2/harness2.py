"""v2 rollouts: the v1 causal loop plus provider modes (DUPLICATE, FIRST_ONLY, SECOND_EVENT).

The agent meets the simulator only here, through the public view and ``EpisodeBatch.readings`` of its own
selected expert. In FIRST_ONLY mode only one reading is ever requested; in SECOND_EVENT mode the second
reading is delivered as an independent event (a non-advancing update). Regret is computed afterwards.
"""
from __future__ import annotations

import numpy as np

from cgc_lao_lab import env as env1
from cgc_lao_lab.harness import Rollout, noop_decisions, privileged_inputs, regret_split
from cgc_lao_lab.env import STOP

from . import env2

MODES = env2.MODES


def rollout2(agent, batch, contract, mode: str = "DUPLICATE", privileged=None) -> Rollout:
    if mode not in MODES:
        raise ValueError("UNKNOWN_PROVIDER_MODE")
    pub = batch.public()
    E, T = batch.count, contract.horizon
    agent.begin(pub.router_w, E, contract, privileged)
    taken = np.zeros((E, T), dtype=np.int64)
    online = np.zeros(E)
    for t in range(T):
        x_t = pub.x[:, t]
        a = np.asarray(agent.act(t, x_t), dtype=np.int64)
        R = contract.reads[t]
        reads = batch.readings(t, a, 1 if mode == "FIRST_ONLY" else R)
        if mode == "SECOND_EVENT" and R == 2:
            agent.observe(t, x_t, a, reads[:, :1])
            agent.observe_extra(t, x_t, a, reads[:, 1:2])
        else:
            agent.observe(t, x_t, a, reads)
        taken[:, t] = a
        online += env1.online_regret_step(batch.p[:, t], a)
    decisions = np.asarray(agent.decide(pub.xq), dtype=np.int64)
    if decisions.shape != (E, env1.Q) or decisions.min() < 0 or decisions.max() > STOP:
        raise ValueError("DECISION_CONTRACT")
    return Rollout(taken, decisions, env1.terminal_regret(batch.pq, decisions), online / T)


def _p_clean(agent):
    f = getattr(agent, "filter", None)
    if f is None or agent.privileged:
        return None
    qbar = list(getattr(f, "qbar", getattr(f, "qs", [])))
    rho = list(getattr(f, "rho", [0.0] * len(qbar)))
    for m, (q, r) in enumerate(zip(qbar, rho)):
        if q == 0.0 and r == 0.0:
            return f.weights()[:, m]
    return None


def evaluate_batch2(spec: dict, batch, noop_dec) -> dict:
    agent, priv = spec["factory"](batch.regime, env2.CONTRACTS[spec["contract"]])
    if agent is None:
        return {}
    out = rollout2(agent, batch, env2.CONTRACTS[spec["contract"]], spec["mode"], privileged_inputs(priv, batch))
    boundary, ranking = regret_split(batch, out.decisions)
    res = {"terminal": out.terminal.sum(), "online": out.online.sum(),
           "stop": (out.decisions == STOP).mean(-1).sum(), "diff_noop": (out.decisions != noop_dec).mean(-1).sum(),
           "boundary": boundary.sum(), "ranking": ranking.sum()}
    pc = _p_clean(agent)
    if pc is not None:
        res["p_clean"] = pc.sum()
    return res


def evaluate_seed2(partition: str, seed: int, regimes, specs: dict, episodes: int, chunk: int = 2048) -> dict:
    out: dict = {}
    for rname in regimes:
        sums: dict = {}
        for first in range(0, episodes, chunk):
            batch = env2.episodes(partition, seed, rname, first, min(chunk, episodes - first))
            nd = noop_decisions(batch)
            for label, spec in specs.items():
                res = evaluate_batch2(spec, batch, nd)
                acc = sums.setdefault(label, {})
                for key, value in res.items():
                    acc[key] = acc.get(key, 0.0) + float(value)
        out[rname] = {label: {k: v / episodes for k, v in acc.items()} for label, acc in sums.items() if acc}
    return out
