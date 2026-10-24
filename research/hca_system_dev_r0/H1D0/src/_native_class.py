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
        for f in case["factors"]:
            if f["rel"] in ("R0", "R1", "R2"):
                self.incoming[(f["roles"]["to"], self.LAYER_OUT[f["rel"]])].append(f)
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
                for f in self.incoming[self.cells[key]]:
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
