"""Causal rollouts over correctly indexed potential outcomes (SPEC.md sections 3-4).

One DRIVER learner chooses the relation at every step with its acquisition rule; every other learner is a shadow
fed exactly the same admitted readings (matched information). Authority events are applied at the start of each
step and at the terminal time through sync() (each learner's own invalidation). Terminal decisions of every learner
are scored on the same true utilities. Online regret is the driver's: max over admissible relations of p - p of the
selected one.
"""
from __future__ import annotations

import numpy as np

from .acquisition import probe_r, voi_r
from .learners import decide, expected_utilities
from .world import COMPOSITIONAL, QN, READS, T, query_tensors, utilities


def rollout(batch, learners: dict, driver: str, policy: str = "PROBE", contract: str = "I") -> dict:
    E = batch.count
    idx = np.arange(E)
    sched = READS[contract]
    for L in learners.values():
        L.reset(batch)
    online = np.zeros(E)
    steps = np.zeros(E)
    for t in range(T):
        for L in learners.values():
            L.sync(batch, t)
        adm = batch.admissible(t)
        if policy == "PROBE":
            act = probe_r(learners[driver], batch, t, adm)
        elif policy == "VOI":
            act = voi_r(learners[driver], batch, t, adm, sched[t])
        else:
            raise ValueError(policy)
        mask = act >= 0
        a = np.where(mask, act, 0)
        p_t = batch.p[:, t]
        best = np.where(adm, p_t, -np.inf).max(1)
        online += np.where(mask, best - p_t[idx, a], 0.0)
        steps += mask
        reads = batch.y[idx, t, a, :sched[t]]
        z = batch.logits(batch.x[:, t])[idx, a]
        src = batch.reporter(t)[idx, a]
        epoch = batch.epoch(src, t)
        for L in learners.values():
            L.observe(t, a, z, reads, src, epoch, mask)
    for L in learners.values():
        L.sync(batch, T)
    out = terminal(batch, learners)
    out[driver]["online"] = online / np.maximum(steps, 1)
    return out


def terminal(batch, learners: dict) -> dict:
    tpl, struct = batch.template, batch.query_struct
    _, _, stop_s = query_tensors(tpl)
    stops = stop_s[struct]
    U = utilities(batch.pq, tpl, struct)                          # [E, QN, A] true
    best_true = np.maximum(np.where(np.isfinite(U), U, -np.inf).max(-1), stops[None])
    stop_best = U.max(-1) < stops[None]
    kinds = np.array([tpl.queries[s].kind for s in struct])
    comp = np.isin(kinds, COMPOSITIONAL)
    direct = kinds == "DIRECT"
    zq = batch.logits(batch.xq)
    out = {}
    for name, L in learners.items():
        eu = expected_utilities(L, zq, tpl, struct)
        choice = decide(eu, stops)
        got = np.where(choice >= 0, np.take_along_axis(U, np.maximum(choice, 0)[..., None], -1)[..., 0], stops[None])
        regret = best_true - got                                  # [E, QN]
        boundary = (choice < 0) != stop_best
        pred = sum(w[:, None, None] * m for w, m in L.components(zq))
        res = {"terminal": regret.mean(1), "comp": regret[:, comp].mean(1),
               "direct": regret[:, direct].mean(1) if direct.any() else np.zeros(len(regret)),
               "stop": (choice < 0).mean(1), "boundary": (regret * boundary).mean(1),
               "edge_mse": ((pred - batch.pq) ** 2).mean((1, 2))}
        for k in COMPOSITIONAL:
            sel = kinds == k
            if sel.any():
                res[k.lower()] = regret[:, sel].mean(1)
        out[name] = res
    return out
