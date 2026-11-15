"""COMPETING FAMILY 3 — recurrent retention / compressed-state reference.

Modelled after RetNet-style decayed state plus Titans/MIRAS-style bounded
segment memory (published mathematics as scientific context; no code copied).
It keeps NO per-relation records: provenance, authority, utility, credibility,
contradiction and protected status are structurally unavailable to it.  That is
the point of the comparison, not an oversight.

REFERENCE_NOT_NATIVE_GDN.  CPU reference only.
"""
from __future__ import annotations
import math
import numpy as np
from . import contract_ersa as C


class RetentionBank:
    def __init__(self):
        self._open = False
        self._spec = None
        self._state = None
        self._mem = None
        self._tick = 0
        self._counters = None
        self._last = []
        self._last_selection = []

    def begin_session(self, spec=None, registry=None, transport=None,
                      authority=None):
        spec = spec or C.SessionSpec()
        self._spec = spec
        cap = spec.max_active_relations
        self._state = np.zeros(4, dtype=np.float64)
        self._mem = {
            "live": np.zeros(cap, dtype=np.uint8),
            "key": np.zeros((cap, spec.key_feature_dim), dtype=np.int64),
            "value": np.zeros((cap, 4), dtype=np.float64),
            "issued": np.zeros(cap, dtype=np.int64),
            "n": 0,
        }
        self._tick = 0
        self._counters = {"inserts": 0, "evictions": 0, "queries": 0,
                          "live": 0}
        self._open = True
        return C.receipt(C.STATUS_OK, "", "RETENTION_OPEN")

    def insert_relation(self, relation_id, key, value4, **_ignored):
        """Authority/identity/utility fields are IGNORED by this family."""
        if not self._open:
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "NO_SESSION")
        spec = self._spec
        v = np.asarray([float(x) for x in value4], dtype=np.float64)
        # recurrent decayed state (RetNet-style scalar decay)
        lam = 0.99
        self._state = lam * self._state + v
        m = self._mem
        slot = int(relation_id) % spec.max_active_relations
        if m["live"][slot]:
            self._counters["evictions"] += 1
        m["live"][slot] = 1
        m["key"][slot] = [int(x) for x in key]
        m["value"][slot] = v
        m["issued"][slot] = self._tick
        self._counters["inserts"] += 1
        self._counters["live"] = int(np.count_nonzero(m["live"]))
        return C.receipt(C.STATUS_OK, "", "RETAINED")

    def query_candidates(self, query_key, **_ignored):
        if self._mem is None:
            return C.receipt(C.STATUS_REJECTED, "OP_BUDGET", "NO_SESSION")
        self._counters["queries"] += 1
        spec = self._spec
        qk = np.asarray([float(x) for x in query_key], dtype=np.float64)
        scored = []
        for s in range(spec.max_active_relations):
            if not self._mem["live"][s]:
                continue
            k = self._mem["key"][s].astype(float)
            nk = float(np.sqrt(np.sum(k * k)))
            nq = float(np.sqrt(np.sum(qk * qk)))
            sim = (float(np.sum(k * qk)) / (nk * nq)) if nk > 1e-12 and nq > 1e-12 else 0.0
            age = self._tick - int(self._mem["issued"][s])
            scored.append((-round(sim + 0.25 * math.exp(-age / 64.0), 12), s))
        scored.sort()
        self._last = [s for _, s in scored[:spec.max_candidates]]
        return C.receipt(C.STATUS_OK, "", "RET_CANDIDATES:" +
                         ",".join(str(s) for s in self._last))

    def select(self, query_key, **_ignored):
        self.query_candidates(query_key)
        self._last_selection = list(self._last)
        return C.receipt(C.STATUS_OK, "", "RET_SELECTED:" +
                         ",".join(str(s) for s in self._last))

    def aggregate(self, query_key, **_ignored):
        self.select(query_key)
        spec = self._spec
        vals = [tuple(self._mem["value"][s]) for s in self._last]
        if not vals:
            return C.receipt(C.STATUS_OK, "", "AGGREGATE:0,0,0,0")
        agg = tuple(float(np.mean([v[j] for v in vals])) for j in range(4))
        beta = 0.25
        agg = tuple((1 - beta) * agg[j] + beta * float(self._state[j])
                    for j in range(4))
        self._last_aggregate = agg
        return C.receipt(C.STATUS_OK, "", "AGGREGATE:" +
                         ",".join(f"{x:.12g}" for x in agg))

    def propose_update(self, target, delta4, gain=0.25):
        """This family has no proposal boundary: it mutates retained state."""
        v = np.asarray([float(x) for x in delta4], dtype=np.float64)
        self._state = self._state + float(gain) * v
        return C.receipt(C.STATUS_PENDING, "", "APPLIED_IMMEDIATELY")

    def commit_validated_update(self, stage):
        return C.receipt(C.STATUS_REJECTED, "STALE_PROPOSAL",
                         "NO_PROPOSAL_BOUNDARY_IN_FAMILY")

    def evict(self, relation_id):
        return C.receipt(C.STATUS_REJECTED, "UNKNOWN_RELATION",
                         "NO_RELATION_IDENTITIES")

    def step(self, n=1):
        self._tick += int(n)
        return C.receipt(C.STATUS_OK, "", "STEPPED")

    def reset(self):
        self._state = None
        self._mem = None
        self._counters = None
        self._open = False
        return C.receipt(C.STATUS_OK, "", "RESET_EMPTY")

    def occupancy(self):
        if self._mem is None:
            return {"live": 0, "capacity": 0, "tick": 0}
        return {"live": self._mem["n"] if False else self._counters["live"],
                "capacity": int(self._spec.max_active_relations),
                "tick": int(self._tick), "inserts": self._counters["inserts"],
                "evictions": self._counters["evictions"],
                "queries": self._counters["queries"]}

    def policy(self):
        return {"family": "retention", "state_dim": 4,
                "segment_memory": int(self._spec.max_active_relations)}

    def snapshot(self):
        if self._mem is None:
            return {"relations": {}, "state": None, "tick": 0}
        rel = {}
        for s in range(self._spec.max_active_relations):
            if self._mem["live"][s]:
                rel[s] = {"key": tuple(int(x) for x in self._mem["key"][s]),
                          "value": tuple(float(x) for x in self._mem["value"][s])}
        return {"relations": rel, "state": tuple(float(x) for x in self._state),
                "tick": int(self._tick)}
