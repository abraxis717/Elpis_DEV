"""H1D0 learned local comparator (DEV only, explicitly NOT B3).

A trained typed factor-node recurrence in the max-plus (tropical) semiring,
softened with log-sum-exp so that every witness contributes gradient.

State is one bounded log-score per entity per query.  A query seeds its
`from` node with 0 and every other node with -infinity.  Round r combines the
live factors whose relation is layer r and whose scope/kind match the query:

    v_{r+1}[b] = logsumexp_{(a -> b) live, scope, kind} ( v_r[a] + t[r][class(a->b)] )

where `class` is the factor's provenance class and t[r][class] is a learned
admission score.  The prediction is `v_rounds[target] > theta`.

This is the same typed-conjunction reachability shape as the native HCA, with
the hand-written admission table replaced by learned parameters.  Capacity is
tiny and prospectively declared: 3 layers x 6 provenance classes + 1 threshold
= 19 parameters.
"""
import json
import numpy as np

CLASSES = ("VALID_MATCH", "VALID_MISMATCH", "STALE", "UNKNOWN", "ABSENT", "DROPPED")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
LAYERS = ("R0", "R1", "R2")
NEG = -60.0


def provenance_class(case, n, factor):
    status = {}
    present = {}
    src = {}
    for e in case["events"][:n]:
        t = e["type"]
        if t == "SOURCE":
            status[e["source"]] = (e["status"], e["epoch"])
        elif t == "EMIT":
            present[e["factor"]] = True
        elif t == "DROP":
            present[e["factor"]] = False
        else:
            src[e["factor"]] = e["source"]
    s = src.get(factor["id"], factor["source"])
    if not present.get(factor["id"], False):
        return "DROPPED"
    st = status.get(s)
    if st is None:
        return "ABSENT"
    if st[0] != "VALID":
        return "STALE" if st[0] == "STALE" else "UNKNOWN"
    return "VALID_MATCH" if st[1] == factor["version"] else "VALID_MISMATCH"


def edge_arrays(case, n, node_index, query):
    rel, ea, eb, ec = [], [], [], []
    for f in case["factors"]:
        if f["rel"] not in LAYERS:
            continue
        if f["roles"]["scope"] != query["scope"] or f["roles"]["kind"] != query["kind"]:
            continue
        cls = provenance_class(case, n, f)
        a = node_index[f["roles"]["from"]]
        b = node_index[f["roles"]["to"]]
        if a == b:
            continue
        rel.append(LAYERS.index(f["rel"]))
        ea.append(a)
        eb.append(b)
        ec.append(CLASS_INDEX[cls])
    return (np.array(rel, dtype=np.int64), np.array(ea, dtype=np.int64),
            np.array(eb, dtype=np.int64), np.array(ec, dtype=np.int64))


def _round(v, rel, ea, eb, ec, r, t):
    sel = rel == r
    nc = len(v)
    if not sel.any():
        return np.full(nc, NEG), None
    a, b, c = ea[sel], eb[sel], ec[sel]
    vals = v[a] + t[r][c]
    m = np.full(nc, NEG)
    np.maximum.at(m, b, vals)
    s = np.zeros(nc)
    np.add.at(s, b, np.exp(vals - m[b]))
    nv = np.where(s > 0, m + np.log(np.maximum(s, 1e-300)), NEG)
    weight = np.where(s[b] > 0, np.exp(vals - m[b]) / np.maximum(s[b], 1e-300), 0.0)
    return nv, (a, b, c, weight)


class FactorNet:
    def __init__(self, rounds=3, seed=20261023):
        self.rounds = rounds
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.t = rng.normal(0.0, 0.2, (rounds, len(CLASSES)))
        self.theta = 0.0

    def forward(self, nc, seed_idx, target_idx, edges, want_grad=False):
        rel, ea, eb, ec = edges
        v = np.full(nc, NEG)
        v[seed_idx] = 0.0
        vs = [v]
        grads = []
        for r in range(self.rounds):
            v, g = _round(v, rel, ea, eb, ec, r, self.t)
            vs.append(v)
            grads.append(g)
        logit = v[target_idx] - self.theta
        if not want_grad:
            return logit
        return logit, vs, grads

    def loss_and_grad(self, samples):
        """Exact BCE loss and its exact gradient w.r.t. (t, theta)."""
        loss = 0.0
        dt = np.zeros_like(self.t)
        dtheta = 0.0
        for nc, s, tg, edges, y in samples:
            logit, _vs, grads = self.forward(nc, s, tg, edges, want_grad=True)
            z = np.clip(logit, -60, 60)
            loss += np.log1p(np.exp(-abs(z))) + max(z, 0.0) - y * z
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            gv = np.zeros(nc)
            gv[tg] = err
            dtheta -= err
            for r in range(self.rounds - 1, -1, -1):
                if grads[r] is None:
                    continue
                a, b, c, weight = grads[r]
                contrib = gv[b] * weight
                np.add.at(dt[r], c, contrib)
                gv_prev = np.zeros(nc)
                np.add.at(gv_prev, a, contrib)
                gv = gv_prev
        return loss, dt, dtheta

    def fit(self, samples, epochs=60, lr=0.05):
        rng = np.random.default_rng(self.seed + 7)
        idx = rng.permutation(len(samples))
        mw = np.zeros_like(self.t); mv = np.zeros_like(self.t)
        mb = 0.0; mbv = 0.0
        eps = 1e-8
        for ep in range(epochs):
            _loss, dt, dtheta = self.loss_and_grad([samples[k] for k in idx])
            self.t -= lr * mw / (np.sqrt(mv) + eps)
            self.theta -= lr * mb / (np.sqrt(mbv) + eps)
            mw = 0.9 * mw + 0.1 * dt
            mv = 0.999 * mv + 0.001 * dt * dt
            mb = 0.9 * mb + 0.1 * dtheta
            mbv = 0.999 * mbv + 0.001 * dtheta * dtheta
        return self

    def predict(self, nc, seed_idx, target_idx, edges):
        logit = self.forward(nc, seed_idx, target_idx, edges)
        return int(logit > 0.0), logit

    def param_bytes(self):
        return len(json.dumps({"t": self.t.tolist(), "theta": self.theta}))
