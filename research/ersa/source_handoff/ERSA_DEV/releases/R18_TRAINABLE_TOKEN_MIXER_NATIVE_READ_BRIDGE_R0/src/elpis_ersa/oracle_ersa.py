"""ERSA ORACLE R0 — independent dense ground truth (plain dicts/lists).

Does NOT import bank_ersa.  Expectations come only from here; a backend never
supplies its own answers.  The oracle scans the FULL live relation set, so it is
also the dense-attention ground truth: an indexed backend must match it
exactly whenever the candidate ceiling covers the live set, and must be scored
for candidate recall when it does not.
"""
from __future__ import annotations
import math
from . import contract_ersa as C

MASK64 = (1 << 64) - 1
MIX_A = 0x9E3779B97F4A7C15


def _sm(x):
    x = (x + MIX_A) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def fp(key):
    h = 0xCBF29CE484222325
    for v in key:
        h = _sm(h ^ (int(v) & MASK64))
    return h & MASK64


def subk(key, k, bits):
    base = fp(key)
    return [(_sm(base ^ ((j + 1) * 0x100000001B3)) & ((1 << bits) - 1))
            for j in range(k)]


def cos(a, b):
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na < 1e-14 or nb < 1e-14:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b)) / (na * nb)


W = {"qcompat": 1.00, "utility": 0.50, "freshness": 0.35, "credibility": 0.40,
     "contradiction": -0.60, "goalrel": 0.45, "protected": 0.30,
     "geomcompat": 0.55}


class NaiveOracle:
    def __init__(self, spec, registry, transport, authority):
        self.spec = spec
        self.registry = {int(r.relation_id): r for r in registry}
        self.transport = transport
        a = dict(authority or {})
        self.denied = set(a.get("denied_provenances", ()))
        self.expired = set(a.get("expired_tickets", ()))
        self.protected = set(a.get("protected_relations", ()))
        self.gauge_relation = tuple(a.get("gauge_relation", (0.0, 0.0)))
        self.gauge_participant = {int(k): tuple(v)
                                  for k, v in (a.get("gauge_participant")
                                               or {}).items()}
        self.participant_angles = {int(k): tuple(v)
                                   for k, v in (a.get("participant_angles")
                                                or {}).items()}
        self.state = {}
        self.tick = 0

    def _admitted(self, rid):
        rec = self.registry.get(int(rid))
        if rec is None:
            return False
        if rec.provenance in self.denied or rec.ticket in self.expired:
            return False
        return self.tick <= rec.valid_until_tick

    def insert(self, relation_id, key, value4, provenance=0, participant=0,
               role=0, ordinal=0, flags=0, ticket=0, utility=1.0,
               credibility=1.0, goal=None, contradicts=-1, angle=(0.0, 0.0)):
        rid = int(relation_id)
        self.state[rid] = {
            "key": tuple(int(x) for x in key), "value": tuple(map(float, value4)),
            "prov": int(provenance), "part": int(participant), "flags": int(flags),
            "ticket": int(ticket), "issued": self.tick, "util": float(utility),
            "cred": float(credibility),
            "goal": tuple(map(float, goal)) if goal else (0.0,) * len(key),
            "contra": int(contradicts), "angle": tuple(map(float, angle)),
        }

    def evict(self, rid):
        self.state.pop(int(rid), None)

    def reinforce(self, rid, co=1.0, fb=1.0):
        if rid in self.state:
            self.state[rid]["util"] = min(4.0, self.state[rid]["util"]
                                          + 0.25 * float(co) * float(fb))

    def decay(self, ticks=1):
        for rid in self.state:
            self.state[rid]["util"] *= math.exp(-float(ticks) / self.spec.halflife_ticks)

    def step(self, n=1):
        self.tick += int(n)

    def reset(self):
        self.state = {}
        self.tick = 0

    def _frame_bound(self, rid):
        if self.transport is None or not self.participant_angles:
            return True
        return self.state[rid]["part"] in self.participant_angles

    def transported(self, rid):
        rec = self.state[rid]
        if self.transport is None:
            return rec["value"]
        beta = self.participant_angles.get(rec["part"], (0.0, 0.0))
        gam = self.gauge_relation
        gp = self.gauge_participant.get(rec["part"], (0.0, 0.0))
        u = (rec["angle"][0] + gam[0] - beta[0] - gp[0],
             rec["angle"][1] + gam[1] - beta[1] - gp[1])
        return _rho(u, rec["value"])

    def features(self, rid, qkey, qgoal, qdir):
        rec = self.state[rid]
        age = max(0, self.tick - rec["issued"])
        tv = self.transported(rid)
        contra = rec["contra"]
        contradiction = (-1.0 if (contra in self.state
                                  and self._admitted(contra)) else 0.0)
        return {"qcompat": cos(rec["key"], qkey),
                "utility": rec["util"],
                "freshness": math.exp(-age / self.spec.halflife_ticks),
                "credibility": rec["cred"],
                "contradiction": contradiction,
                "goalrel": cos(rec["goal"], qgoal) if qgoal else 0.0,
                "protected": (1.0 if (rec["flags"] & C.FLAG_PROTECTED
                                      or rid in self.protected) else 0.0),
                "geomcompat": cos(tv, qdir)}

    def score(self, feats):
        return math.fsum(W[k] * v for k, v in feats.items())

    def expected_selection(self, qkey, qgoal=None, k=None, qdir=None):
        spec = self.spec
        k = int(k if k is not None else spec.max_candidates)
        out = []
        for rid in sorted(self.state):
            if (not self._admitted(rid) or self.state[rid]["prov"] in self.denied
                    or not self._frame_bound(rid)):
                continue
            f = self.features(rid, qkey, qgoal, qdir if qdir is not None else qkey)
            out.append((-round(self.score(f), 12), rid, f))
        out.sort()
        return [(rid, f) for _, rid, f in out[:k]]

    def expected_aggregate(self, qkey, qgoal=None, k=None, qdir=None):
        sel = self.expected_selection(qkey, qgoal, k, qdir)
        tv, w = [], []
        for rid, f in sel:
            tv.append(self.transported(rid))
            w.append(self.score(f))
        if not tv:
            return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
        z = math.fsum(abs(x) for x in w)
        if z < 1e-12:
            sal = tuple(math.fsum(v[j] for v in tv) / len(tv) for j in range(4))
        else:
            sal = tuple(math.fsum((w[i] / z) * tv[i][j]
                                  for i in range(len(tv))) for j in range(4))
        col = tuple(math.fsum(v[j] for v in tv) / len(tv) for j in range(4))
        return sal, col

    def expected_candidates(self, qkey, k_bits, k_sub, degree, participant=None,
                            limit=None):
        """Dense ground-truth candidate universe = every live relation."""
        return sorted(self.state)


def _rho(angles, v):
    a1, a2 = angles
    x1, x2, x3, x4 = v
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    return (c1 * x1 - s1 * x2, s1 * x1 + c1 * x2,
            c2 * x3 - s2 * x4, s2 * x3 + c2 * x4)
