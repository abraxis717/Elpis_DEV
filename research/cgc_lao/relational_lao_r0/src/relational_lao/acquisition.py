"""Acquisition rules over admissible relations (SPEC.md section 4).

PROBE_R  the historical PROBE_EARLY schedule (cgc_lao_lab.selector) over relation slots: argmax predictive success,
         except that the best unseen admissible relation is probed at steps 1 and 2 and forced when the unseen
         relations equal the remaining steps.
VOI_R    the myopic expected value of sample information of the next event for the relational decision: for every
         admissible relation and every reading pattern of this step, the posterior after the event is formed exactly
         (the update law, no state change), and the decision value of the template's query structures is read at
         planning contexts (an independent stream, never the terminal contexts). The relation with the largest
         expected gain is selected (first index on ties). Bounded: R x 2^reads posterior updates per step.
"""
from __future__ import annotations

import numpy as np

from cgc_lao_lab.filters import V_FLOOR

from .learners import gh_mean
from .world import NPLAN, T, query_tensors, utilities


def probe_r(learner, batch, t: int, admissible: np.ndarray) -> np.ndarray:
    z = batch.logits(batch.x[:, t])
    pred = learner.predictive(z)
    unseen = (learner.live_counts() == 0) & admissible
    need = unseen.sum(1)
    best = np.where(admissible, pred, -np.inf).argmax(1)
    unseen_best = np.where(unseen, pred, -np.inf).argmax(1)
    force = (need > 0) & ((need >= T - t) | (t in (1, 2)))
    act = np.where(force, unseen_best, best)
    return np.where(admissible.any(1), act, -1)


def _value(w, m, template, struct, alt_mask, stops):
    """Decision value sum_j max(max_a E[U_ja], tau_j) of a coupled posterior (w [E, M], m [E, M, J, R])."""
    U = utilities(m, template, struct)                          # [E, M, J, A]
    EU = (w[:, :, None, None] * np.where(np.isfinite(U), U, 0.0)).sum(1)
    EU = np.where(alt_mask[None], EU, -np.inf)
    return np.maximum(EU.max(-1), stops[None]).sum(-1)


def planning(batch):
    S = len(batch.template.queries)
    struct = [s for _ in range(NPLAN) for s in range(S)]
    zp = np.repeat(batch.logits(batch.xp), S, axis=1)           # [E, NPLAN * S, R]
    _, alt_mask, stop = query_tensors(batch.template)
    return struct, zp, alt_mask[struct], stop[struct]


def voi_r(learner, batch, t: int, admissible: np.ndarray, n_reads: int) -> np.ndarray:
    """VOI_R for a Bayes learner or a readout view of one (the decision value is the filter's coupled readout)."""
    f = learner.filter if hasattr(learner, "filter") else learner.base.filter
    E, R = batch.count, batch.R
    struct, zp, alt_mask, stops = planning(batch)
    z_t = batch.logits(batch.x[:, t])
    w = f.weights()
    m = f.means(zp)
    v0 = _value(w, m, batch.template, struct, alt_mask, stops)
    patterns = [(0,), (1,)] if n_reads == 1 else [(0, 0), (0, 1), (1, 0), (1, 1)]
    gain = np.full((E, R), -np.inf)
    for r in range(R):
        if not admissible[:, r].any():
            continue
        slot = np.full(E, r)
        total = np.zeros(E)
        for pat in patterns:
            reads = np.tile(np.array(pat, dtype=np.uint8), (E, 1))
            L, m1, m2, mu, v = f.likelihood(slot, z_t[:, r], reads)
            ok = L > 1e-300
            py = (w * L).sum(1)
            w2 = w * L
            w2 = w2 / np.maximum(w2.sum(1, keepdims=True), 1e-300)
            mu2 = np.where(ok, m1, mu)
            v2 = np.where(ok, np.maximum(m2 - m1 * m1, V_FLOOR), v)
            mm = m.copy()
            mm[:, :, :, r] = gh_mean(zp[:, None, :, r], mu2[:, :, None], v2[:, :, None])
            total = total + py * _value(w2, mm, batch.template, struct, alt_mask, stops)
        gain[:, r] = np.where(admissible[:, r], total - v0, -np.inf)
    act = gain.argmax(1)
    return np.where(admissible.any(1), act, -1)
