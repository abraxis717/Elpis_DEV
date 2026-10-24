"""Full reference-substrate differential: H1D0 port vs the real Elpis DEV substrate.

Every TRAIN case, every query, at three prefixes, for B0, B1 and B2.
Compares:
  reference B0/B1/B2  vs  independent oracle gold
  H1D0 port B0/B1/B2  vs  reference B0/B1/B2
"""
import sys, time
from collections import Counter
sys.path.insert(0, "src")
from generator import build_split
from oracle import gold_for_case
from engines import HcaNativeSemantics
from native_ref import reference_predict

TR, TRT = build_split("TRAIN", 12345, 4)
cases = [c for w in TR["worlds"] for c in w["cases"]]
diff = Counter()
comparisons = 0
t0 = time.perf_counter()
for c in cases:
    gold = gold_for_case(c, TRT[c["case_id"]])
    n = len(c["events"])
    half = n // 2
    for pref in (half, n):
        port = {m: HcaNativeSemantics(c, method=m) for m in ("B0", "B1", "B2")}
        for ev in c["events"][:pref]:
            for eng in port.values():
                eng.apply(ev)
        snaps = {m: eng.snapshot() for m, eng in port.items()}
        for q in c["queries"]:
            for m, rounds in (("B0", 4), ("B1", 4), ("B2", 4)):
                try:
                    ref, _w = reference_predict(c, pref, q, m, rounds=rounds)
                except Exception as e:
                    diff["reference_error_" + m] += 1
                    continue
                comparisons += 1
                if m in ("B0", "B1") and ref != gold[pref][q["id"]]:
                    diff["ref" + m + "_vs_gold"] += 1
                if ref != snaps[m][q["id"]]:
                    diff["port" + m + "_vs_ref" + m] += 1
print(time.perf_counter() - t0)
print("comparisons", comparisons)
print("differences", dict(diff))
