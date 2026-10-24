"""Stateful learned comparator: the trained max-plus typed recurrence run as an
online engine with genuinely retained per-query state.

Retained state is the per-query, per-round log-score vector A[q][r].  An event
that changes a factor's provenance class updates only the witnesses containing
that factor, then propagates forward through the remaining rounds.  Regime-B
damage corrupts `units` entries of that retained state; the engine keeps running
and its answers become wrong.  `repair()` re-derives every round from the
undamaged round-0 seed, which is the model's only repair mechanism.

The edge set is fixed at construction (all factors whose scope/kind match the
query); only the provenance class of each edge changes over time, exactly as in
the batch model used for training.
"""
import json
import numpy as np
from collections import Counter

from learned import LAYERS, NEG, CLASS_INDEX


def logsumexp(vals):
    if not vals:
        return NEG
    m = max(vals)
    return m + np.log(sum(np.exp(v - m) for v in vals))


class LearnedOnline:
    def __init__(self, net, case):
        self.net = net
        self.case = case
        toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                      | {q["from"] for q in case["queries"]}
                      | {q["to"] for q in case["queries"]})
        self.node_index = {t: i for i, t in enumerate(toks)}
        self.nc = len(toks)
        self.factors = {f["id"]: f for f in case["factors"]}
        self.status, self.present, self.src = {}, {}, {}
        self.src = {f["id"]: f["source"] for f in case["factors"]}
        self.n = 0
        self.work = Counter()

        self.q = {}
        for query in case["queries"]:
            edges = []
            for f in case["factors"]:
                if f["rel"] not in LAYERS:
                    continue
                if f["roles"]["scope"] != query["scope"] or f["roles"]["kind"] != query["kind"]:
                    continue
                a = self.node_index[f["roles"]["from"]]
                b = self.node_index[f["roles"]["to"]]
                if a == b:
                    continue
                edges.append((LAYERS.index(f["rel"]), a, b, f["id"]))
            inc = [[[] for _ in range(self.nc)] for _ in range(net.rounds)]
            out = [[[] for _ in range(self.nc)] for _ in range(net.rounds)]
            for i, (r, a, b, fid) in enumerate(edges):
                inc[r][b].append((i, a))
                out[r][a].append(b)
            A = [np.full(self.nc, NEG) for _ in range(net.rounds + 1)]
            A[0][self.node_index[query["from"]]] = 0.0
            cls = [CLASS_INDEX["DROPPED"]] * len(edges)
            qd = {"query": query, "edges": edges, "inc": inc, "out": out, "cls": cls, "A": A}
            self.q[query["id"]] = qd
            self._recompute_all(qd)

    def _class_of(self, fid):
        """Provenance class from the engine's own maintained state (no rescan)."""
        f = self.factors[fid]
        if not self.present.get(fid, False):
            return "DROPPED"
        st = self.status.get(self.src.get(fid, f["source"]))
        if st is None:
            return "ABSENT"
        if st[0] != "VALID":
            return "STALE" if st[0] == "STALE" else "UNKNOWN"
        return "VALID_MATCH" if st[1] == f["version"] else "VALID_MISMATCH"

    def _recompute_all(self, qd):
        """Full re-derivation of every round from the round-0 seed."""
        t = self.net.t
        A = qd["A"]
        for r in range(self.net.rounds):
            self.work["witness_recombine"] += len(qd["edges"])
            A[r + 1] = np.array([logsumexp([A[r][a] + t[r][qd["cls"][i]]
                                           for i, a in qd["inc"][r][b]])
                                 for b in range(self.nc)])
            self.work["state_write"] += self.nc

    def _propagate(self, qd, start_round, nodes):
        t = self.net.t
        A = qd["A"]
        nodes = set(nodes)
        for r in range(start_round, self.net.rounds):
            nxt = set()
            for b in nodes:
                vals = [A[r][a] + t[r][qd["cls"][i]] for i, a in qd["inc"][r][b]]
                new = logsumexp(vals)
                self.work["witness_recombine"] += len(vals)
                if new != A[r + 1][b]:
                    A[r + 1][b] = new
                    self.work["state_write"] += 1
                    if r + 1 < self.net.rounds:
                        nxt.update(qd["out"][r + 1][b])
            nodes = nxt
            if not nodes:
                break

    def apply(self, event):
        t = event["type"]
        changed = set()
        if t == "SOURCE":
            self.status[event["source"]] = (event["status"], event["epoch"])
            changed = {fid for fid in self.src if self.src[fid] == event["source"]}
        elif t == "EMIT":
            self.present[event["factor"]] = True
            changed = {event["factor"]}
        elif t == "DROP":
            self.present[event["factor"]] = False
            changed = {event["factor"]}
        else:
            self.src[event["factor"]] = event["source"]
            changed = {event["factor"]}
        self.n += 1
        self.work["event_apply"] += 1
        for fid in changed:
            ci = CLASS_INDEX[self._class_of(fid)]
            for qid, qd in self.q.items():
                for i, (r, a, b, f2) in enumerate(qd["edges"]):
                    if f2 != fid:
                        continue
                    if qd["cls"][i] == ci:
                        continue
                    qd["cls"][i] = ci
                    self.work["witness_update"] += 1
                    self._propagate(qd, r, {b})
        return changed

    def snapshot(self):
        out = {}
        for qid, qd in self.q.items():
            logit = qd["A"][self.net.rounds][self.node_index[qd["query"]["to"]]] - self.net.theta
            out[qid] = int(logit > 0.0)
            self.work["query_snapshot"] += 1
        return out

    def footprint(self):
        return len(json.dumps([[list(map(float, row)) for row in qd["A"]]
                              for qd in self.q.values()]))

    def damage(self, rng, units):
        """Corrupt `units` entries of the retained activation that are actually
        in use (non-sentinel), so the damage is not trivially invisible."""
        n = 0
        attempts = 0
        keys = list(self.q)
        while n < units and attempts < units * 40 and keys:
            k = rng.choice(keys)
            r = rng.randrange(1, self.net.rounds + 1)
            active = [i for i in range(self.nc) if self.q[k]["A"][r][i] > NEG / 2]
            if not active:
                attempts += 1
                continue
            i = rng.choice(active)
            self.q[k]["A"][r][i] = rng.uniform(-3.0, 3.0)
            n += 1
            attempts += 1
        return n

    def repair(self):
        """Re-derive every round from the undamaged round-0 seed."""
        for qid, qd in self.q.items():
            self._recompute_all(qd)
        self.work["repair"] += 1
