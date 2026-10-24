"""Find the first HcaNativeSemantics B0 disagreement with the oracle and dump the state."""
import sys
sys.path.insert(0, "src")
from collections import Counter, defaultdict
from generator import build_split
from oracle import gold_for_case
from engines import HcaNativeSemantics

TR, TRT = build_split("TRAIN", 12345, 4)
cases = [c for w in TR["worlds"] for c in w["cases"]]
truth = {c["case_id"]: c["events"] for c in cases}

first = None
counts = Counter()
for c in cases:
    gold = gold_for_case(c, truth[c["case_id"]])
    eng = HcaNativeSemantics(c, method="B0")
    for n in range(len(c["events"]) + 1):
        if n:
            eng.apply(c["events"][n - 1])
        snap = eng.snapshot()
        for q in c["queries"]:
            if snap[q["id"]] != gold[n][q["id"]]:
                counts[c["kind"]] += 1
                if first is None:
                    first = (c, n, q, snap[q["id"]], gold[n][q["id"]])
print("total b0 mismatches", sum(counts.values()), "by kind", dict(counts))
if first:
    c, n, q, got, want = first
    print("case", c["case_id"], "kind", c["kind"], "prefix", n, "query", q)
    print("got", got, "want", want)
    live = [f for f in c["factors"]
            if eng.present.get(f["id"], False)
            and eng.status.get(eng.src[f["id"]]) == ("VALID", f["version"])]
    print("live rel counts", Counter(f["rel"] for f in live))
    for f in live:
        r = f["roles"]
        if r["scope"] == q["scope"] and r["kind"] == q["kind"]:
            print("  ", f["rel"], r["from"][:8], "->", r["to"][:8], "src", f["source"][:8], "ver", f["version"])
    scores = eng._closure(q)
    for cell in eng.cells:
        if cell[0] in (q["from"], q["to"]) or cell[1] in ("S", "K"):
            print("  cell", cell[0][:8], cell[1], scores[eng.index[cell]])
