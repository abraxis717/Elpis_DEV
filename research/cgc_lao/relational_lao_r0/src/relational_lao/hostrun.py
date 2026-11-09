"""Drive the research host through one RLB1 episode with the real ECS_C view (instrumented provider).

The episode's sources become real elpis_ecs_c_entity SOURCE rows; its authority events become real successor
tables (version bump; DORMANT then ACTIVE; TERMINATED, reclaimed, and a new source reusing the 32-bit
source_authority); every reading is admitted into the view before the learner sees it. Identities are fresh per
episode. Used by the host/vectorised differential and by the provider and invalidation tests.
"""
from __future__ import annotations

from cgc_lao_v2.providers import Reading

from . import ecsc
from .world import NSRC, READS, T


class EpisodeWorld:
    def __init__(self, batch, e: int):
        self.batch, self.e = batch, e
        tag = (batch.partition, batch.seed, batch.regime.name, batch.first + e)
        self.world = ecsc.AuthorityWorld(ecsc.digest("rlb1/lineage", *tag))
        scope = ecsc.digest("rlb1/scope", *tag)
        self.row_of = [self.world.found(ecsc.SOURCE, ecsc.ACTIVE, authority=1000 + s, version=1, scope=scope)
                       for s in range(NSRC)]
        self.scope = scope
        ents = [ecsc.digest("rlb1/entity", *tag, name) for name in batch.template.entities]
        self.relations = [(ents[a], ents[b]) for a, b in batch.template.relations]

    def reporters(self, t: int, rel: int) -> int:
        return self.row_of[int(self.batch.reporter(t)[self.e, rel])]

    def transitions(self, t: int) -> list:
        """The authority transitions at the start of step t (each a table the host must sync)."""
        b, e = self.batch, self.e
        kind, t_ev, s = b.regime.authority, int(b.t_ev[e]), int(b.s_ev[e])
        w, out = self.world, []
        if kind in ("REV_DRIFT", "REV_SAME") and t == t_ev:
            w.set_version(self.row_of[s], 2)
            out.append(w)
        elif kind == "DORMANT" and t in (t_ev, t_ev + 2):
            w.set_lifecycle(self.row_of[s], ecsc.DORMANT if t == t_ev else ecsc.ACTIVE)
            out.append(w)
        elif kind == "RECLAIM_DRIFT" and t == t_ev:
            row = self.row_of[s]
            w.set_lifecycle(row, ecsc.TERMINATED)
            out.append(w)
            w.reclaim(row)
            self.row_of = [r - 1 if r > row else r for r in self.row_of]
            self.row_of[s] = w.found(ecsc.SOURCE, ecsc.ACTIVE, authority=1000 + s, version=1, scope=self.scope)
            out.append(w)
        return out


def run_episode(host, batch, e: int, tamper=None):
    """One episode through the host: transitions + syncs, proposals, admitted readings, terminal decisions.
    Returns (decisions per terminal query, relation selected per step)."""
    ew = EpisodeWorld(batch, e)
    host.reset_episode(ew.world, ew.relations, batch.W, ew.reporters)
    chosen = []
    for t in range(T):
        for table in ew.transitions(t):
            host.sync(table)
        slot = host.propose_relation(batch.x[e, t])
        if slot < 0:                                        # nothing admissible: the step passes without evidence
            chosen.append(-1)
            continue
        rel = host.rel_of[slot]
        n = READS["I"][t]
        readings = [Reading(f"{e}/{t}", f"{e}/{t}/{k}", f"chan-{'AB'[k]}", slot, int(batch.y[e, t, rel, k]))
                    for k in range(n)]
        if tamper is not None:
            readings = tamper(t, readings)
        host.observe_selected(slot, readings)
        chosen.append(rel)
    for table in ew.transitions(T):
        host.sync(table)
    tpl = batch.template
    queries = [(tpl.queries[s].alternatives, tpl.queries[s].stop) for s in batch.query_struct]
    return host.propose_terminal(queries, batch.xq[e]), chosen
