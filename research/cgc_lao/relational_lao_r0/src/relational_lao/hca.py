"""Layer C: the research-only geometric projection and its HCA readout (docs/ECS_C_HCA.md).

A relation's salience s_r in [0, 1000] enters HCA as the integer weight of a unary factor from a seed cell into
the relation's cell (score = weight, by the message law weight * min(inputs) / 1000 with a seed at 1000). A path is
a factor (all its relation cells -> the path cell) of weight 1000: its score is min over its relations. An
alternative of several edge-disjoint paths receives max over its paths (a cell keeps the maximum message). B0, the
exact closure, gives exactly these values on these acyclic graphs, so the numpy readout here equals native B0 (the
differential test builds the native graphs). Every score is SALIENCE_ONLY.

The host decides from the saliences: the best alternative (first on ties) is answered when its salience reaches
round(1000 * tau^(1/k)) for paths of k relations (the per-relation level whose k-fold product is the STOP payoff),
otherwise STOP. Nothing is written back to the learner or to ECS_C.
"""
from __future__ import annotations

import numpy as np

from .learners import Exact, Learner, prior_mean
from .world import query_tensors


def project(p: np.ndarray) -> np.ndarray:
    """The non-authoritative projection: u16 saliences round(1000 p), clamped to [0, 1000]."""
    return np.clip(np.rint(1000.0 * p), 0, 1000).astype(np.int64)


def readout(sal: np.ndarray, template, struct) -> np.ndarray:
    """Alternative saliences [..., J, A] (-1 for padding) from relation saliences sal [..., J, R]."""
    idx, alt_mask, _ = query_tensors(template)
    sidx = idx[list(struct)]                                    # [J, A, P, L]
    J = len(struct)
    flat = sal.reshape((-1, J, sal.shape[-1]))
    g = np.take_along_axis(flat, np.broadcast_to(np.maximum(sidx, 0).reshape(1, J, 36), (flat.shape[0], J, 36)),
                           axis=-1).reshape(-1, J, 4, 3, 3)
    g = np.where(sidx[None] >= 0, g, 1000)
    path = np.where(sidx[None, ..., 0] >= 0, g.min(-1), -1)     # min over a path's relations
    alt = path.max(-1)                                          # max over an alternative's paths
    alt = np.where(alt_mask[list(struct)][None], alt, -1)
    return alt.reshape(sal.shape[:-2] + (J, 4))


def thresholds(template, struct) -> np.ndarray:
    out = []
    for s in struct:
        q = template.queries[s]
        k = len(q.alternatives[0][0]) if q.kind != "REACH" else max(len(p) for p in q.alternatives[0])
        out.append(int(round(1000.0 * q.stop ** (1.0 / k))))
    return np.array(out)


def decide(alt_sal: np.ndarray, theta: np.ndarray) -> np.ndarray:
    best = alt_sal.argmax(-1)
    top = np.take_along_axis(alt_sal, best[..., None], -1)[..., 0]
    return np.where(top >= theta[None, :], best, -1)


class HcaOnly(Exact):
    """HCA_ONLY: HCA B0 over the admitted graph: a relation with LIVE positive net evidence is a seed-backed cell
    (salience 1000); every other relation is absent (0). No learning."""
    name = "HCA_ONLY"

    def decide_queries(self, zq, template, struct):
        sal = np.where(self.net() > 0, 1000, 0)[:, None, :].repeat(len(struct), 1)
        return decide(readout(sal, template, struct), thresholds(template, struct))


class LaoHca(Learner):
    """LAO_PLUS_HCA: HCA B0 over the learner's projection (salience = round(1000 E[p_r(x')]) per query context)."""

    def __init__(self, learner):
        self.inner = learner
        self.name = "LAO_PLUS_HCA"

    def reset(self, batch):
        super().reset(batch)
        self.inner.reset(batch)

    def sync(self, batch, t):
        self.inner.sync(batch, t)

    def observe(self, t, rel, z, reads, src, epoch, mask):
        self.inner.observe(t, rel, z, reads, src, epoch, mask)

    def components(self, z):
        return self.inner.components(z)

    def live_counts(self):
        return self.inner.live_counts()

    def decide_queries(self, zq, template, struct):
        p = sum(w[:, None, None] * m for w, m in self.inner.components(zq))
        return decide(readout(project(p), template, struct), thresholds(template, struct))

    def state_scalars(self) -> int:
        return self.inner.state_scalars()


__all__ = ["HcaOnly", "LaoHca", "project", "readout", "thresholds", "decide", "prior_mean"]
