"""Explain every HcaNativeSemantics B2 disagreement with the independent oracle."""
import sys
sys.path.insert(0, "src")
from collections import Counter
from generator import build_split
from oracle import gold_for_case
from engines import HcaNativeSemantics

TR, TRT = build_split("TRAIN", 12345, 4)
cases = [c for w in TR["worlds"] for c in w["cases"]]
truth = {c["case_id"]: c["events"] for c in cases}

examples = []
for c in cases:
    gold = gold_for_case(c, truth[c["case_id"]])
    eng = HcaNativeSemantics(c, method="B2", rounds=4, updates_per_tick=256)
    for n in range(len(c["events"]) + 1):
        if n:
            eng.apply(c["events"][n - 1])
        snap = eng.snapshot()
        for q in c["queries"]:
            if snap[q["id"]] != gold[n][q["id"]]:
                examples.append((c, n, q, snap[q["id"]], gold[n][q["id"]]))
print("b2 vs gold mismatches:", len(examples), "all false negatives:",
      all(g == 1 and p == 0 for _c, _n, _q, p, g in examples))
print("prefix histogram:", Counter(n for _c, n, _q, _p, _g in examples))

c, n, q, got, want = examples[0]
print("\ncase", c["case_id"], "kind", c["kind"], "prefix", n, "query", q)
eng = HcaNativeSemantics(c, method="B2", rounds=4, updates_per_tick=256)
for ev in c["events"][:n]:
    eng.apply(ev)
scores = eng.values[q["id"]]
print("cells", len(eng.cells), "step", eng.step)
for cell in eng.cells:
    print("  ", cell[0][:10], cell[1], scores[eng.index[cell]])
live = [f for f in c["factors"] if eng._live(f)
        and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]]
print("live matching factors:")
for f in live:
    print("  ", f["rel"], f["roles"]["from"][:10], "->", f["roles"]["to"][:10])
print("extra-tick convergence:")
for extra in range(0, 10):
    print("  extra ticks", extra, "answer", eng.snapshot()[q["id"]], "gold", want)
    eng._tick()
