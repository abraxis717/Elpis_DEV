import numpy as np
from generator import build_split
from oracle import gold_for_case
from learned import FactorNet, edge_arrays

def nodes_of(c):
    toks = sorted({v for f in c["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in c["queries"]} | {q["to"] for q in c["queries"]})
    return {t: i for i, t in enumerate(toks)}

def samples(ds, truth):
    out = []
    for w in ds["worlds"]:
        for c in w["cases"]:
            gold = gold_for_case(c, truth[c["case_id"]])
            ni = nodes_of(c)
            cache = {}
            for n in range(len(c["events"]) + 1):
                for q in c["queries"]:
                    k = (n, q["scope"], q["kind"])
                    if k not in cache:
                        cache[k] = edge_arrays(c, n, ni, q)
                    out.append((len(ni), ni[q["from"]], ni[q["to"]], cache[k], gold[n][q["id"]]))
    return out

tr, tt = build_split("TRAIN", 12345, 2)
S = samples(tr, tt)
tu, t2 = build_split("TUNE", 987654, 2)
T = samples(tu, t2)
print("train samples", len(S), "tune samples", len(T))
for seed in (20261023, 20261029, 20261031, 20261037):
    for lr in (0.1, 0.2):
        net = FactorNet(rounds=3, seed=seed)
        net.fit(S, epochs=100, lr=lr)
        a = sum(1 for nc, x, y, e, yy in S if net.predict(nc, x, y, e)[0] == yy) / len(S)
        b = sum(1 for nc, x, y, e, yy in T if net.predict(nc, x, y, e)[0] == yy) / len(T)
        print(f"seed={seed} lr={lr} train={a:.4f} tune={b:.4f} s={net.s.round(3).tolist()} g={net.g:.3f} d={net.d:.3f}")
