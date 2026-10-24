"""H1D0 engines. Each engine sees ONLY the public case (events, factors,
queries) and never the truth log, the oracle, or the labels.

Every engine exposes:
    apply(event)       -> one online event
    snapshot()         -> {query_id: 0/1}
    footprint()        -> serialized retained-state bytes
    damage(rng, units) -> corrupt `units` entries of its OWN retained state
    work               -> Counter of logical operations
"""
import json
from collections import defaultdict, Counter
from hashlib import sha256


def enc(obj):
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


# --------------------------------------------------------------------------
# 1. Exact stateless replay
# --------------------------------------------------------------------------
class Replay:
    """Rebuild the whole state and rescan every factor at every prefix.

    No derived cache exists, so regime-B damage is a declared no-op.
    """

    def __init__(self, case):
        self.factors = case["factors"]
        self.queries = case["queries"]
        self.events = []
        self.work = Counter()
        self.peak = 0

    def apply(self, event):
        self.events.append(event)
        self.work["log_append"] += 1

    def snapshot(self):
        status, present, src = {}, {}, {}
        for e in self.events:
            t = e["type"]
            if t == "SOURCE":
                status[e["source"]] = (e["status"], e["epoch"])
            elif t == "EMIT":
                present[e["factor"]] = True
            elif t == "DROP":
                present[e["factor"]] = False
            else:
                src[e["factor"]] = e["source"]
            self.work["replayed_event"] += 1
        live = []
        for f in self.factors:
            self.work["factor_scan"] += 1
            if present.get(f["id"], False) and status.get(src.get(f["id"], f["source"])) == ("VALID", f["version"]):
                live.append(f)
        row = {}
        for q in self.queries:
            frontier = {q["from"]}
            for rel in ("R0", "R1", "R2"):
                nxt = set()
                for f in live:
                    self.work["factor_visit"] += 1
                    r = f["roles"]
                    if f["rel"] == rel and r["scope"] == q["scope"] and r["kind"] == q["kind"] and r["from"] in frontier:
                        nxt.add(r["to"])
                frontier = nxt
            row[q["id"]] = int(q["to"] in frontier)
            self.work["query_snapshot"] += 1
        self.peak = max(self.peak, enc({"status": status, "live": [f["id"] for f in live]}))
        return row

    def footprint(self):
        return enc({"log": self.events, "factors": self.factors})

    def damage(self, rng, units):
        return 0


# --------------------------------------------------------------------------
# 2. Exact incremental dependency maintenance
# --------------------------------------------------------------------------
class Incremental:
    """Source-aware witness-index maintenance with invalidation counting."""

    def __init__(self, case, integrity=False):
        self.factors = {f["id"]: f for f in case["factors"]}
        if len(self.factors) != len(case["factors"]):
            raise ValueError("factor id collision")
        self.queries = {q["id"]: q for q in case["queries"]}
        self.status = {}
        self.present = {}
        self.src = {f["id"]: f["source"] for f in case["factors"]}
        self.live = set()
        self.work = Counter()
        self.paths = []
        self.dependents = defaultdict(list)
        self.missing = []
        self.supports = {k: 0 for k in self.queries}
        rels = defaultdict(list)
        for f in case["factors"]:
            rels[f["rel"]].append(f)
        for q in case["queries"]:
            for a in rels["R0"]:
                self.work["setup_candidate"] += 1
                ra = a["roles"]
                if (ra["from"], ra["scope"], ra["kind"]) != (q["from"], q["scope"], q["kind"]):
                    continue
                for b in rels["R1"]:
                    self.work["setup_candidate"] += 1
                    rb = b["roles"]
                    if (rb["from"], rb["scope"], rb["kind"]) != (ra["to"], q["scope"], q["kind"]):
                        continue
                    for c in rels["R2"]:
                        self.work["setup_candidate"] += 1
                        rc = c["roles"]
                        if (rc["from"], rc["to"], rc["scope"], rc["kind"]) != (rb["to"], q["to"], q["scope"], q["kind"]):
                            continue
                        pid = len(self.paths)
                        self.paths.append((q["id"], (a["id"], b["id"], c["id"])))
                        self.missing.append(3)
                        for fid in (a["id"], b["id"], c["id"]):
                            self.dependents[fid].append(pid)
                        self.work["setup_path"] += 1
        self.work["setup_factor"] += len(case["factors"])
        self.work["setup_query"] += len(case["queries"])
        self.integrity = integrity
        self.check = self._checksum()
        self.rebuilds = 0
        self.invariant_violations = 0

    def _checksum(self):
        return sha256(("|".join(str(m) for m in self.missing) + "|" +
                       "|".join(str(v) for v in self.supports.values())).encode()).hexdigest()

    def _is_live(self, fid):
        f = self.factors[fid]
        return self.present.get(fid, False) and self.status.get(self.src[fid]) == ("VALID", f["version"])

    def _flip(self, fid, now):
        old = fid in self.live
        if old == now:
            return
        self.work["factor_live_flip"] += 1
        if now:
            self.live.add(fid)
        else:
            self.live.remove(fid)
        for pid in self.dependents.get(fid, []):
            self.work["path_dependency_update"] += 1
            before = self.missing[pid]
            after = before - 1 if now else before + 1
            # An undamaged engine can never leave [0, 3].  A damaged engine can,
            # and must keep running so that the wrongness is observable rather
            # than a crash.  The integrity-checked variant detects the drift and
            # rebuilds; the plain variant simply carries the corrupted counter.
            if not 0 <= after <= 3:
                self.work["invariant_violation"] += 1
                self.invariant_violations += 1
                after = min(3, max(0, after))
            self.missing[pid] = after
            qid = self.paths[pid][0]
            if before == 0:
                self.supports[qid] -= 1
            if after == 0:
                self.supports[qid] += 1
            if self.supports[qid] < 0:
                self.work["invariant_violation"] += 1
                self.invariant_violations += 1
                self.supports[qid] = 0

    def apply(self, event):
        self.work["event_apply"] += 1
        t = event["type"]
        if t == "SOURCE":
            self.status[event["source"]] = (event["status"], event["epoch"])
            for fid in self.factors:
                if self.src[fid] == event["source"]:
                    self.work["affected_factor_visit"] += 1
                    self._flip(fid, self._is_live(fid))
        elif t == "EMIT":
            self.present[event["factor"]] = True
            self.work["affected_factor_visit"] += 1
            self._flip(event["factor"], self._is_live(event["factor"]))
        elif t == "DROP":
            self.present[event["factor"]] = False
            self.work["affected_factor_visit"] += 1
            self._flip(event["factor"], self._is_live(event["factor"]))
        elif t == "REBIND":
            fid = event["factor"]
            self.src[fid] = event["source"]
            self.work["affected_factor_visit"] += 1
            self._flip(fid, self._is_live(fid))
        if self.integrity:
            self.work["integrity_check"] += 1
            if self._checksum() != self.check:
                self.work["integrity_rebuild"] += 1
                self.rebuilds += 1
                self.rebuild()

    def rebuild(self):
        self.work["rebuild_path"] += len(self.paths)
        self.missing = [3] * len(self.paths)
        self.supports = {k: 0 for k in self.queries}
        for pid, (qid, members) in enumerate(self.paths):
            for fid in members:
                if fid in self.live:
                    self.missing[pid] -= 1
            if self.missing[pid] == 0:
                self.supports[qid] += 1
        self.check = self._checksum()

    def snapshot(self):
        self.work["query_snapshot"] += len(self.queries)
        out = {k: 0 for k in self.queries}
        for pid, (qid, _members) in enumerate(self.paths):
            self.work["path_read"] += 1
            if self.missing[pid] == 0:
                out[qid] = 1
        return out

    def footprint(self):
        return enc({"paths": self.paths, "missing": self.missing,
                    "dependents": dict(self.dependents), "live": sorted(self.live),
                    "status": self.status, "supports": self.supports,
                    "present": self.present, "src": self.src})

    def damage(self, rng, units):
        n = 0
        attempts = 0
        while n < units and attempts < units * 20 and self.missing:
            i = rng.randrange(len(self.missing))
            cur = self.missing[i]
            if cur == 0:
                new = 1
            elif cur == 3:
                new = 2
            else:
                new = rng.choice((cur - 1, cur + 1))
            if new == cur:
                attempts += 1
                continue
            self.missing[i] = new
            n += 1
            attempts += 1
        return n


# --------------------------------------------------------------------------
# 3. Checkpoint + replay recovery
# --------------------------------------------------------------------------
class CheckpointReplay:
    """Exact engine that keeps periodic checkpoints and recovers by replay."""

    def __init__(self, case, every=8):
        self.factors = case["factors"]
        self.queries = case["queries"]
        self.every = every
        self.status, self.present, self.src = {}, {}, {f["id"]: f["source"] for f in case["factors"]}
        self.live = set()
        self.work = Counter()
        self.checkpoints = []
        self.n = 0
        self._checkpoint()

    def _live(self):
        return {f["id"] for f in self.factors
                if self.present.get(f["id"], False)
                and self.status.get(self.src[f["id"]]) == ("VALID", f["version"])}

    def _checkpoint(self):
        self.checkpoints.append((self.n, dict(self.status), dict(self.present),
                                 dict(self.src), set(self.live)))
        self.work["checkpoint_write"] += 1

    def apply(self, event):
        t = event["type"]
        if t == "SOURCE":
            self.status[event["source"]] = (event["status"], event["epoch"])
        elif t == "EMIT":
            self.present[event["factor"]] = True
        elif t == "DROP":
            self.present[event["factor"]] = False
        else:
            self.src[event["factor"]] = event["source"]
        self.live = self._live()
        self.n += 1
        self.work["event_apply"] += 1
        if self.n % self.every == 0:
            self._checkpoint()

    def snapshot(self):
        row = {}
        for q in self.queries:
            frontier = {q["from"]}
            for rel in ("R0", "R1", "R2"):
                nxt = set()
                for f in self.factors:
                    if f["id"] not in self.live:
                        continue
                    self.work["factor_visit"] += 1
                    r = f["roles"]
                    if f["rel"] == rel and r["scope"] == q["scope"] and r["kind"] == q["kind"] and r["from"] in frontier:
                        nxt.add(r["to"])
                frontier = nxt
            row[q["id"]] = int(q["to"] in frontier)
            self.work["query_snapshot"] += 1
        return row

    def footprint(self):
        return enc({"checkpoints": [(n, s, p, sr, sorted(l)) for n, s, p, sr, l in self.checkpoints],
                    "status": self.status, "present": self.present, "src": self.src,
                    "live": sorted(self.live)})

    def damage(self, rng, units):
        n = 0
        if self.checkpoints:
            idx = rng.randrange(len(self.checkpoints))
            _n, st, pr, sr, lv = self.checkpoints[idx]
            for _ in range(units):
                if st:
                    k = rng.choice(list(st))
                    st[k] = (rng.choice(("VALID", "STALE", "UNKNOWN")), rng.randrange(2))
                    n += 1
            self.checkpoints[idx] = (idx, st, pr, sr, lv)
        return n

    def recover(self):
        n, st, pr, sr, lv = self.checkpoints[-1]
        self.status, self.present, self.src, self.live = dict(st), dict(pr), dict(sr), set(lv)
        self.work["recover_replay"] += 1
        return self.snapshot()


# --------------------------------------------------------------------------
# 4. Native B0/B1/B2 reference semantics (typed weighted conjunction closure)
# --------------------------------------------------------------------------
class HcaNativeSemantics:
    """Faithful port of the reference B0/B1/B2 typed-conjunction comparators.

    Encoding (declared, and cross-checked against the reference implementation):
      * a cell is a (token, layer) pair.  Layers are L0..L3 plus the shared
        S (scope) and K (kind) cells.
      * a factor of relation type R0 is the 4-ary conjunction
        (from,L0) & (scope,S) & (kind,K) => (to,L1);  R1: L1 => L2;  R2: L2 => L3.
      * RX factors are not admitted by the predicate mask, exactly as the
        reference refuses a factor whose schema is not named by a predicate.
      * message = weight * min(input scores) // 1000, and 0 unless the factor's
        source admits (VALID with an epoch equal to the factor version).
      * seeds for a query are (from,L0), (scope,S), (kind,K) at 1000.
      * readout is whether (to,L3) has positive salience.

    B0 = exact worklist closure.  B1 = synchronous rounds (truncated by the
    round budget).  B2 = deterministic asynchronous recurrence with a bounded
    update budget per tick and an explicit retained activation.
    """

    SCHEDULE = b"elpis.hca.h1d0.b2.schedule\0"
    LAYER_IN = {"R0": "L0", "R1": "L1", "R2": "L2"}
    LAYER_OUT = {"R0": "L1", "R1": "L2", "R2": "L3"}

    def __init__(self, case, method="B2", rounds=4, updates_per_tick=256, weight=1000):
        if method not in ("B0", "B1", "B2"):
            raise ValueError("method must be B0, B1 or B2")
        self.case = case
        self.method = method
        self.rounds = rounds
        self.updates = updates_per_tick
        self.weight = weight
        self.work = Counter()
        cells = set()
        for f in case["factors"]:
            if f["rel"] in ("R0", "R1", "R2"):
                cells.add((f["roles"]["from"], self.LAYER_IN[f["rel"]]))
                cells.add((f["roles"]["to"], self.LAYER_OUT[f["rel"]]))
                cells.add((f["roles"]["scope"], "S"))
                cells.add((f["roles"]["kind"], "K"))
        for q in case["queries"]:
            cells.add((q["from"], "L0"))
            cells.add((q["to"], "L3"))
            cells.add((q["scope"], "S"))
            cells.add((q["kind"], "K"))
        self.cells = sorted(cells)
        self.index = {k: i for i, k in enumerate(self.cells)}
        self.incoming = defaultdict(list)
        self.outgoing = defaultdict(list)
        for f in case["factors"]:
            if f["rel"] in ("R0", "R1", "R2"):
                out = (f["roles"]["to"], self.LAYER_OUT[f["rel"]])
                self.incoming[out].append(f)
                for role in ("from", "scope", "kind"):
                    layer = self.LAYER_IN[f["rel"]] if role == "from" else ("S" if role == "scope" else "K")
                    self.outgoing[(f["roles"][role], layer)].append(f)
        self.status = {}
        self.present = {}
        self.src = {f["id"]: f["source"] for f in case["factors"]}
        self.values = {q["id"]: [0] * len(self.cells) for q in case["queries"]}
        self.step = 0

    def _live(self, f):
        return (self.present.get(f["id"], False)
                and self.status.get(self.src[f["id"]]) == ("VALID", f["version"]))

    def apply(self, event):
        t = event["type"]
        if t == "SOURCE":
            self.status[event["source"]] = (event["status"], event["epoch"])
        elif t == "EMIT":
            self.present[event["factor"]] = True
        elif t == "DROP":
            self.present[event["factor"]] = False
        else:
            self.src[event["factor"]] = event["source"]
        self.work["event_apply"] += 1
        if self.method == "B2":
            self._tick()

    def _message(self, f, scores):
        self.work["factor_update"] += 1
        if not self._live(f):
            return 0
        value = 1000
        for role, layer in (("from", self.LAYER_IN[f["rel"]]), ("scope", "S"), ("kind", "K")):
            self.work["input_read"] += 1
            value = min(value, scores[self.index[(f["roles"][role], layer)]])
        return self.weight * value // 1000

    def _seeds(self, q):
        return {self.index[(q["from"], "L0")],
                self.index[(q["scope"], "S")],
                self.index[(q["kind"], "K")]}

    def _closure(self, q):
        scores = [0] * len(self.cells)
        for i in self._seeds(q):
            scores[i] = 1000
        if self.method == "B0":
            queue = sorted(self._seeds(q))
            queued = set(queue)
            self.work["queue_init"] += len(queue)
            while queue:
                self.work["queue_pop"] += 1
                key = queue.pop(0)
                queued.discard(key)
                for f in self.outgoing[self.cells[key]]:
                    if f["roles"]["scope"] != q["scope"] or f["roles"]["kind"] != q["kind"]:
                        continue
                    value = self._message(f, scores)
                    self.work["cell_update"] += 1
                    out = self.index[(f["roles"]["to"], self.LAYER_OUT[f["rel"]])]
                    if value > scores[out]:
                        scores[out] = value
                        if out not in queued:
                            self.work["queue_push"] += 1
                            queue.append(out)
                            queued.add(out)
        else:
            for _ in range(self.rounds):
                self.work["cell_copy"] += len(scores)
                updated = list(scores)
                for f in self.case["factors"]:
                    if f["rel"] not in ("R0", "R1", "R2"):
                        continue
                    if f["roles"]["scope"] != q["scope"] or f["roles"]["kind"] != q["kind"]:
                        continue
                    value = self._message(f, scores)
                    self.work["cell_update"] += 1
                    out = self.index[(f["roles"]["to"], self.LAYER_OUT[f["rel"]])]
                    if value > updated[out]:
                        updated[out] = value
                scores = updated
        return scores

    def _tick(self):
        order = sorted(range(len(self.cells)),
                      key=lambda i: (sha256(self.SCHEDULE + self.step.to_bytes(8, "big")
                                          + i.to_bytes(8, "big")).digest(), i))
        self.step += 1
        for q in self.case["queries"]:
            scores = self.values[q["id"]]
            self.work["schedule_order"] += len(order)
            for i in order[:self.updates]:
                self.work["cell_fire"] += 1
                value = 1000 if i in self._seeds(q) else 0
                for f in self.incoming[self.cells[i]]:
                    if f["roles"]["scope"] != q["scope"] or f["roles"]["kind"] != q["kind"]:
                        continue
                    value = max(value, self._message(f, scores))
                scores[i] = value
            self.work["state_write"] += len(scores)

    def snapshot(self):
        if self.method == "B2":
            return {q["id"]: int(self.values[q["id"]][self.index[(q["to"], "L3")]] > 0)
                    for q in self.case["queries"]}
        return {q["id"]: int(self._closure(q)[self.index[(q["to"], "L3")]] > 0)
                for q in self.case["queries"]}

    def footprint(self):
        if self.method == "B2":
            return enc({"values": self.values, "step": self.step})
        return enc({"status": self.status, "present": self.present, "src": self.src})

    def damage(self, rng, units):
        if self.method != "B2":
            return 0
        n = 0
        sessions = list(self.values)
        for _ in range(units * 4):
            if not sessions:
                break
            sid = rng.choice(sessions)
            i = rng.randrange(len(self.cells))
            self.values[sid][i] = rng.randrange(1001)
            n += 1
            if n >= units:
                break
        return n
