"""Symbolic and control baselines. All are TRAIN-only: they are fit on the
TRAIN split and evaluated on TUNE.  They never see the truth log or sealed data.
"""
from collections import Counter, defaultdict
from hashlib import sha256


def prefix_state(case, n):
    status, present, src = {}, {}, {}
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
    live = [f for f in case["factors"]
            if present.get(f["id"], False)
            and status.get(src.get(f["id"], f["source"])) == ("VALID", f["version"])]
    return status, present, src, live


def features(case, n, q):
    """Bounded local feature vocabulary over the visible prefix."""
    _s, _p, _src, live = prefix_state(case, n)
    by_rel = defaultdict(list)
    for f in live:
        by_rel[f["rel"]].append(f["roles"])
    r0 = [r for r in by_rel["R0"] if r["from"] == q["from"] and r["scope"] == q["scope"] and r["kind"] == q["kind"]]
    r1 = [r for r in by_rel["R1"] if r["scope"] == q["scope"] and r["kind"] == q["kind"]]
    r2 = [r for r in by_rel["R2"] if r["to"] == q["to"] and r["scope"] == q["scope"] and r["kind"] == q["kind"]]
    mid1 = {r["to"] for r in r0}
    mid2 = {r["from"] for r in r2}
    two_hop = any(r["from"] in mid1 for r in r1)
    back_hop = any(r["to"] in mid2 for r in r1)
    full = any(r["from"] in mid1 and r["to"] in mid2 for r in r1)
    return {
        "n_live": len(live),
        "n_r0": len(r0),
        "n_r1": len(r1),
        "n_r2": len(r2),
        "two_hop": int(two_hop),
        "back_hop": int(back_hop),
        "full_chain": int(full),
        "n_events": n,
        "last_status": case["events"][n - 1]["status"] if n and case["events"][n - 1]["type"] == "SOURCE" else "NONE",
    }


class SymbolicInducer:
    """Greedy decision-list learner over the declared feature vocabulary.

    depth='full' allows the full-chain atom; depth='local' forbids it, which
    models a genuinely bounded local rule.
    """

    ATOMS = ("n_live", "n_r0", "n_r1", "n_r2", "two_hop", "back_hop", "full_chain")

    def __init__(self, depth="full"):
        self.depth = depth
        self.rules = []
        self.default = 0

    def allowed(self, atom):
        if atom == "full_chain":
            return self.depth == "full"
        return True

    def fit(self, samples):
        pos = [s[1] for s in samples if s[2] == 1]
        neg = [s[1] for s in samples if s[2] == 0]
        self.default = 1 if len(pos) > len(neg) else 0
        remaining_pos, remaining_neg = list(pos), list(neg)
        for _ in range(12):
            if not remaining_pos or not remaining_neg:
                break
            best = None
            for atom in self.ATOMS:
                if not self.allowed(atom):
                    continue
                for thr in range(1, max(1, max(f[atom] for f in remaining_pos) + 1)):
                    tp = sum(1 for f in remaining_pos if f[atom] >= thr)
                    fp = sum(1 for f in remaining_neg if f[atom] >= thr)
                    if tp == 0:
                        continue
                    score = tp - fp
                    if best is None or score > best[0]:
                        best = (score, atom, thr)
            if best is None or best[0] <= 0:
                break
            _score, atom, thr = best
            self.rules.append((atom, thr))
            remaining_pos = [f for f in remaining_pos if f[atom] < thr]
            remaining_neg = [f for f in remaining_neg if f[atom] < thr]
        return self

    def predict(self, feats):
        for atom, thr in self.rules:
            if feats[atom] >= thr:
                return 1
        return self.default


class Controls:
    """Memorization, majority, graph-statistic, identity-only and order-only."""

    def __init__(self):
        self.majority = 0
        self.case_memory = {}
        self.query_memory = {}
        self.stat_threshold = 0
        self.order_rule = "NONE"

    def fit(self, samples):
        labels = [s[2] for s in samples]
        self.majority = 1 if sum(labels) > len(labels) / 2 else 0
        by_case = defaultdict(list)
        by_query = defaultdict(list)
        for key, feats, y in samples:
            by_case[key].append(y)
            by_query[key[1]].append(y)
        for k, ys in by_case.items():
            self.case_memory[k] = 1 if sum(ys) > len(ys) / 2 else 0
        for k, ys in by_query.items():
            self.query_memory[k] = 1 if sum(ys) > len(ys) / 2 else 0
        pos = [feats["n_live"] for _k, feats, y in samples if y == 1]
        neg = [feats["n_live"] for _k, feats, y in samples if y == 0]
        self.stat_threshold = (max(pos) + min(neg or [0])) / 2 if pos else 0
        self.order_rule = "VALID" if sum(labels) > len(labels) / 2 else "STALE"
        return self

    def predict(self, key, feats):
        case_id = key[0]
        return self.case_memory.get(case_id, self.majority)

    def predict_majority(self, feats):
        return self.majority

    def predict_stat(self, feats):
        return int(feats["n_live"] > self.stat_threshold)

    def predict_query_identity(self, key):
        return self.query_memory.get(key[1], self.majority)

    def predict_order(self, feats):
        return int(feats["last_status"] == self.order_rule)
