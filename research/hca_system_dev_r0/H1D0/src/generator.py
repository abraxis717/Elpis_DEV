"""H1D0 generator. Generator-side only; never handed to predictors as features.

Deterministic: split seed -> per-world seed -> case construction.
build_split returns (public_dataset, truth) where truth holds the complete
event log for every case.  Only `public_dataset` is ever given to a method.
"""
import random
from copy import deepcopy

from contract import (SCHEMA, KINDS, RELS, MAX_PENDING, MAX_EVENTS, MAX_FACTORS)


def token(rng):
    return "z" + format(rng.getrandbits(112), "028x")


def ev(step, etype, **kw):
    e = {"step": step, "type": etype}
    if etype == "SOURCE":
        e.update({"source": kw["source"], "status": kw["status"], "epoch": kw["epoch"]})
    elif etype in ("EMIT", "DROP"):
        e["factor"] = kw["factor"]
    else:
        e.update({"factor": kw["factor"], "source": kw["source"]})
    return e


def factor(fid, rel, a, b, scope, kind, source, version=0):
    return {"id": fid, "rel": rel,
            "roles": {"from": a, "to": b, "scope": scope, "kind": kind},
            "source": source, "version": version}


def make_pair(rng, kind):
    ents = [token(rng) for _ in range(24)]
    srcs = [token(rng) for _ in range(10)]
    scopes = [token(rng) for _ in range(2)]
    kinds = [token(rng) for _ in range(2)]
    ids = [token(rng) for _ in range(48)]

    a, b, c, d = ents[0], ents[1], ents[2], ents[3]
    b2, c2 = ents[16], ents[17]
    S0, K0 = scopes[0], kinds[0]
    S1, K1 = scopes[1], kinds[1]

    # Primary witness chain a->b->c->d.  A second R1 edge b2->c2 exists in the
    # same scope/kind but has no R0 into b2 and no R2 out of c2, so exchanging
    # the role bindings of the two R1 edges changes the witness set while
    # leaving the multiset of (relation, role, value) triples identical.
    r0 = factor(ids[0], "R0", a, b, S0, K0, srcs[0])
    r1a = factor(ids[1], "R1", b, c, S0, K0, srcs[1], version=1)
    r1b = factor(ids[2], "R1", b2, c2, S0, K0, srcs[3], version=1)
    r2 = factor(ids[3], "R2", c, d, S0, K0, srcs[2])

    # Independent chains sharing sources with the primary chain, so a plain
    # "how many sources are VALID" statistic is not sufficient.
    other = []
    for k in range(3):
        u, v, w, y = ents[6 + 2 * k], ents[7 + 2 * k], ents[8 + 2 * k], ents[9 + 2 * k]
        base = 4 + 3 * k
        other.append(factor(ids[base], "R0", u, v, S1, K1, srcs[4 + (k % 3)]))
        other.append(factor(ids[base + 1], "R1", v, w, S1, K1, srcs[5 + (k % 3)]))
        other.append(factor(ids[base + 2], "R2", w, y, S1, K1, srcs[4 + (k % 3)]))

    fillers = []
    for i in range(13, 17):
        rel = RELS[rng.randrange(4)]
        u = ents[10 + rng.randrange(14)]
        v = ents[10 + rng.randrange(14)]
        fillers.append(factor(ids[i], rel, u, v,
                              scopes[rng.randrange(2)], kinds[rng.randrange(2)],
                              srcs[3 + rng.randrange(6)]))

    fillers.append(factor(ids[21], "RX", ents[14], ents[15], S1, K1, srcs[8]))
    fillers.append(factor(ids[22], "RX", ents[15], ents[14], S1, K1, srcs[9]))
    fillers.append(factor(ids[23], "RX", ents[18], ents[19], S1, K1, srcs[7]))
    fillers.append(factor(ids[24], "RX", ents[20], ents[21], S1, K1, srcs[1]))
    # Adversarial decoys: RX factors whose role bindings exactly mirror the
    # primary witness chain.  An engine that ignores the relation type (or
    # collapses the typed hypergraph into a plain adjacency) would find a
    # witness through these; the contract forbids it.
    fillers.append(factor(ids[25], "RX", b, c, S0, K0, srcs[5]))
    fillers.append(factor(ids[26], "RX", c, d, S0, K0, srcs[6]))

    queries = [{"id": token(rng), "from": a, "to": d, "scope": S0, "kind": K0},
               {"id": token(rng), "from": a, "to": d, "scope": S0, "kind": K1},
               {"id": token(rng), "from": ents[6], "to": ents[9], "scope": S1, "kind": K1},
               {"id": token(rng), "from": ents[8], "to": ents[11], "scope": S1, "kind": K1},
               {"id": token(rng), "from": ents[10], "to": ents[13], "scope": S1, "kind": K1}]
    for _ in range(3):
        queries.append({"id": token(rng), "from": ents[rng.randrange(24)],
                        "to": ents[rng.randrange(24)],
                        "scope": scopes[rng.randrange(2)],
                        "kind": kinds[rng.randrange(2)]})
    target = queries[0]["id"]
    rng.shuffle(queries)

    pair_id = token(rng)
    cases = []
    for positive in (False, True):
        factors = [r0, r1a, r1b, r2] + other + fillers
        events = []
        step = 0

        def add(etype, **kw):
            nonlocal step
            events.append(ev(step, etype, **kw))
            step += 1

        for s in srcs[:7]:
            add("SOURCE", source=s, status="VALID", epoch=0)
        for s in srcs[7:9]:
            add("SOURCE", source=s, status="UNKNOWN", epoch=0)
        for f in factors:
            add("EMIT", factor=f["id"])

        if kind == "source_swap":
            add("SOURCE", source=srcs[1], status="VALID" if positive else "STALE", epoch=1)
            add("SOURCE", source=srcs[3], status="STALE" if positive else "VALID", epoch=1)
        elif kind == "event_order":
            if positive:
                add("SOURCE", source=srcs[1], status="STALE", epoch=1)
                add("SOURCE", source=srcs[1], status="VALID", epoch=1)
            else:
                add("SOURCE", source=srcs[1], status="VALID", epoch=1)
                add("SOURCE", source=srcs[1], status="STALE", epoch=1)
            add("SOURCE", source=srcs[3], status="VALID", epoch=1)
        elif kind == "epoch_rebind":
            add("SOURCE", source=srcs[1], status="VALID", epoch=1 if positive else 0)
            add("SOURCE", source=srcs[3], status="VALID", epoch=0 if positive else 1)
        elif kind == "structural":
            if positive:
                factors[1] = factor(r1a["id"], "R1", b, c, S0, K0, srcs[1], 1)
                factors[2] = factor(r1b["id"], "R1", b2, c2, S0, K0, srcs[3], 1)
            else:
                factors[1] = factor(r1a["id"], "R1", b, c2, S0, K0, srcs[1], 1)
                factors[2] = factor(r1b["id"], "R1", b2, c, S0, K0, srcs[3], 1)
            add("SOURCE", source=srcs[1], status="VALID", epoch=1)
            add("SOURCE", source=srcs[3], status="VALID", epoch=1)
        elif kind == "cancellation":
            add("SOURCE", source=srcs[1], status="VALID", epoch=1)
            add("SOURCE", source=srcs[3], status="VALID", epoch=1)
            if positive:
                add("DROP", factor=r2["id"])
                add("EMIT", factor=r2["id"])
            else:
                add("EMIT", factor=r2["id"])
                add("DROP", factor=r2["id"])
        elif kind == "provenance":
            add("SOURCE", source=srcs[1], status="STALE", epoch=1)
            add("SOURCE", source=srcs[3], status="VALID", epoch=1)
            add("SOURCE", source=srcs[9], status="VALID", epoch=1)
            factors[1] = factor(r1a["id"], "R1", b, c, S0, K0,
                                srcs[9] if positive else srcs[1], 1)
        else:
            raise ValueError("unknown kind " + kind)

        # trailing nuisance evolution that must not change any answer
        add("SOURCE", source=srcs[8], status="STALE", epoch=0)
        add("SOURCE", source=srcs[8], status="VALID", epoch=0)

        if len(events) > MAX_EVENTS:
            raise ValueError("event budget exceeded")
        if len(factors) > MAX_FACTORS:
            raise ValueError("factor budget exceeded")
        cases.append({"case_id": token(rng), "pair_id": pair_id, "kind": kind,
                      "target_query_id": target,
                      "factors": deepcopy(factors),
                      "queries": deepcopy(queries),
                      "events": deepcopy(events),
                      "pending": [],
                      "damage_prefix": len(events) // 2})
    rng.shuffle(cases)
    return cases


def withhold(rng, case, prob):
    """Regime C1: remove a bounded set of events and publish them as a pool."""
    events = case["events"]
    chosen = [i for i in range(len(events)) if rng.random() < prob][:MAX_PENDING]
    if not chosen:
        return case
    pool = [deepcopy(events[i]) for i in chosen]
    case["events"] = [e for i, e in enumerate(events) if i not in set(chosen)]
    case["pending"] = pool
    case["damage_prefix"] = len(case["events"]) // 2
    return case


def build_split(split, seed, worlds, pending_prob=0.0):
    if split not in ("TRAIN", "TUNE"):
        raise ValueError("no SEALED split generator exists in H1D0")
    rng = random.Random(seed)
    out = {"schema": SCHEMA, "split": split, "worlds": []}
    truth = {}
    for _ in range(worlds):
        wr = random.Random(rng.getrandbits(128))
        cases = []
        for kind in KINDS:
            for _ in range(2):
                cases.extend(make_pair(wr, kind))
        for c in cases:
            truth[c["case_id"]] = deepcopy(c["events"])
            if pending_prob > 0.0:
                withhold(wr, c, pending_prob)
        wr.shuffle(cases)
        out["worlds"].append({"world_id": token(wr), "cases": cases})
    return out, truth
