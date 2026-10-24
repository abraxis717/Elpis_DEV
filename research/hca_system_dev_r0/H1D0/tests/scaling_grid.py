"""H1D0 scaling probe: does exact incremental dependency maintenance become the
dominant cost as the query/factor ratio and event-log length grow?

Synthetic worlds are built directly (not from the H1D0 counterfactual generator)
so that the size parameters can be swept independently.  Every engine sees only
the public case.  Correctness is checked against the independent oracle.
"""
import json, random, sys, time
sys.path.insert(0, "src")
from oracle import gold_for_case
from engines import Replay, Incremental, CheckpointReplay, HcaNativeSemantics
from learned import FactorNet, edge_arrays
from learned_online import LearnedOnline

REL = ("R0", "R1", "R2", "RX")


def token(rng):
    return "z" + format(rng.getrandbits(112), "028x")


def make_world(rng, n_ent, n_src, n_fac, n_queries, n_events):
    ents = [token(rng) for _ in range(n_ent)]
    srcs = [token(rng) for _ in range(n_src)]
    scopes = [token(rng) for _ in range(2)]
    kinds = [token(rng) for _ in range(2)]
    ids = [token(rng) for _ in range(n_fac)]
    factors = []
    for i, fid in enumerate(ids):
        rel = REL[i % 4]
        a, b = ents[(2 * i) % n_ent], ents[(2 * i + 1) % n_ent]
        factors.append({"id": fid, "rel": rel,
                        "roles": {"from": a, "to": b,
                                 "scope": scopes[i % 2], "kind": kinds[i % 2]},
                        "source": srcs[i % n_src], "version": i % 3})
    queries = [{"id": token(rng), "from": ents[i % n_ent], "to": ents[(i * 7 + 3) % n_ent],
                "scope": scopes[i % 2], "kind": kinds[i % 2]} for i in range(n_queries)]
    events = []
    for s in srcs:
        events.append({"type": "SOURCE", "source": s, "status": "VALID", "epoch": 0})
    for f in factors:
        events.append({"type": "EMIT", "factor": f["id"]})
    i = 0
    while len(events) < n_events:
        r = rng.random()
        if r < 0.4:
            events.append({"type": "SOURCE", "source": srcs[rng.randrange(n_src)],
                           "status": rng.choice(("VALID", "STALE")),
                           "epoch": rng.randrange(3)})
        elif r < 0.7:
            events.append({"type": "REBIND", "factor": ids[rng.randrange(n_fac)],
                           "source": srcs[rng.randrange(n_src)]})
        elif r < 0.85:
            events.append({"type": "DROP", "factor": ids[rng.randrange(n_fac)]})
        else:
            events.append({"type": "EMIT", "factor": ids[rng.randrange(n_fac)]})
        i += 1
    return {"case_id": token(rng), "pair_id": token(rng), "kind": "scaling",
            "target_query_id": queries[0]["id"], "factors": factors,
            "queries": queries, "events": events, "pending": [],
            "damage_prefix": len(events) // 2}


def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}


def ctor(case, name, net=None):
    if name == "replay":
        return Replay(case)
    if name == "incremental":
        return Incremental(case)
    if name == "checkpoint":
        return CheckpointReplay(case, every=8)
    if name == "hca_b0":
        return HcaNativeSemantics(case, method="B0")
    if name == "hca_b1":
        return HcaNativeSemantics(case, method="B1")
    if name == "hca_b2":
        return HcaNativeSemantics(case, method="B2")
    if name == "learned_online":
        return LearnedOnline(net, case)
    raise KeyError(name)


def nodes_of_case(case):
    return nodes_of(case)


GRID = [(64, 10, 48, 8, 37), (64, 10, 200, 8, 60), (128, 20, 500, 32, 120),
        (256, 40, 1000, 64, 200), (512, 60, 2000, 128, 300)]

names = ["replay", "incremental", "checkpoint", "hca_b0", "hca_b1", "hca_b2",
         "learned_online"]
out = []
for (ne, ns, nf, nq, nev) in GRID:
    rng = random.Random(4242)
    case = make_world(rng, ne, ns, nf, nq, nev)
    truth = case["events"]
    gold = gold_for_case(case, truth)
    ni = nodes_of(case)
    S = []
    for n in range(0, nev + 1, 5):
        for q in case["queries"]:
            S.append((len(ni), ni[q["from"]], ni[q["to"]],
                      edge_arrays(case, n, ni, q), gold[n][q["id"]]))
    net = FactorNet(rounds=3, seed=1)
    net.fit(S, epochs=20, lr=0.05)
    row = {"entities": ne, "factors": nf, "queries": nq, "events": nev}
    for name in names:
        eng = ctor(case, name, net)
        t0 = time.perf_counter()
        for e in case["events"]:
            eng.apply(e)
        snap = eng.snapshot()
        t = time.perf_counter() - t0
        correct = sum(snap[q["id"]] == gold[nev][q["id"]] for q in case["queries"])
        row[name] = {"s": round(t, 5), "bytes": eng.footprint(),
                    "work": sum(eng.work.values()),
                    "accuracy": round(correct / nq, 6)}
    out.append(row)
print(json.dumps(out, indent=1))
json.dump(out, open("runs/scaling_grid.json", "w"), indent=1)
