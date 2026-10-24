"""Quantify the B2 asynchronous recurrence's propagation lag after each event.

Question: is the B2 disagreement with the exact closure a permanent semantic
difference, or a transient convergence lag that disappears with extra ticks?
"""
import sys
sys.path.insert(0, "src")
from collections import Counter
from generator import build_split
from oracle import gold_for_case
from engines import HcaNativeSemantics

TR, TRT = build_split("TRAIN", 12345, 4)
TU, TUT = build_split("TUNE", 67890, 2)
cases = [c for w in TR["worlds"] for c in w["cases"]] + [c for w in TU["worlds"] for c in w["cases"]]
truth = {c["case_id"]: c["events"] for c in cases}

lag = Counter()
decisions = 0
never = 0
for c in cases:
    gold = gold_for_case(c, truth[c["case_id"]])
    for n in range(1, len(c["events"]) + 1):
        b2 = HcaNativeSemantics(c, method="B2", rounds=4, updates_per_tick=256)
        b0 = HcaNativeSemantics(c, method="B0")
        for ev in c["events"][:n]:
            b2.apply(ev)
            b0.apply(ev)
        exact = b0.snapshot()
        snap = b2.snapshot()
        for q in c["queries"]:
            decisions += 1
            if snap[q["id"]] == exact[q["id"]]:
                lag[0] += 1
                continue
            k = 0
            while k < 30:
                b2._tick()
                k += 1
                if b2.snapshot()[q["id"]] == exact[q["id"]]:
                    break
            if k >= 30:
                never += 1
            lag[k] += 1
print("decisions", decisions)
print("extra ticks needed to reach the exact closure:", dict(sorted(lag.items())))
print("never converged within 30 ticks:", never)
print("fraction already converged at the event boundary:", round(lag[0] / decisions, 6))
