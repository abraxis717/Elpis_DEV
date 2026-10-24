"""Independent scratch oracle for H1D0.

Written directly from the contract in contract.py.  It imports nothing from
generator.py, checker.py or any engine.  It is deliberately naive: it rebuilds
the whole state from scratch at every prefix and enumerates witnesses by brute
force.  Its purpose is to be a second, unshared source of truth.
"""
from itertools import combinations
from math import comb


def _state(events, factors, n):
    status = {}
    present = {}
    src = {}
    for e in events[:n]:
        t = e["type"]
        if t == "SOURCE":
            status[e["source"]] = (e["status"], e["epoch"])
        elif t == "EMIT":
            present[e["factor"]] = True
        elif t == "DROP":
            present[e["factor"]] = False
        elif t == "REBIND":
            src[e["factor"]] = e["source"]
    live = []
    for f in factors:
        if not present.get(f["id"], False):
            continue
        s = src.get(f["id"], f["source"])
        st = status.get(s)
        if st is not None and st[0] == "VALID" and st[1] == f["version"]:
            live.append(f)
    return live


def answer(events, factors, query, n):
    live = _state(events, factors, n)
    for f0 in live:
        if f0["rel"] != "R0":
            continue
        r = f0["roles"]
        if r["from"] != query["from"] or r["scope"] != query["scope"] or r["kind"] != query["kind"]:
            continue
        for f1 in live:
            if f1["rel"] != "R1":
                continue
            r1 = f1["roles"]
            if r1["from"] != r["to"] or r1["scope"] != query["scope"] or r1["kind"] != query["kind"]:
                continue
            for f2 in live:
                if f2["rel"] != "R2":
                    continue
                r2 = f2["roles"]
                if (r2["from"] == r1["to"] and r2["to"] == query["to"]
                        and r2["scope"] == query["scope"] and r2["kind"] == query["kind"]):
                    return 1
    return 0


def prefix_labels(events, factors, queries):
    return [{q["id"]: answer(events, factors, q, n) for q in queries}
            for n in range(len(events) + 1)]


def gold_for_case(case, full_events):
    return prefix_labels(full_events, case["factors"], case["queries"])


def _key(e):
    if e["type"] == "SOURCE":
        return ("SOURCE", e["source"])
    if e["type"] in ("EMIT", "DROP"):
        return ("FACTOR", e["factor"])
    return ("REBIND", e["factor"])


def _apply(state, e):
    status, present, src = state
    if e["type"] == "SOURCE":
        status[e["source"]] = (e["status"], e["epoch"])
    elif e["type"] == "EMIT":
        present[e["factor"]] = True
    elif e["type"] == "DROP":
        present[e["factor"]] = False
    else:
        src[e["factor"]] = e["source"]


def possible_final_states(visible, pool):
    """Every final state the withheld pool could have produced.

    The fold is last-wins per key (source, factor, rebind-target), so the set of
    reachable final states is exactly the product over keys of
    {last visible value} union {each pending value}.  This is an exact
    enumeration, not a sample.
    """
    status, present, src = {}, {}, {}
    for e in visible:
        _apply((status, present, src), e)
    base = (dict(status), dict(present), dict(src))
    by_key = {}
    for e in pool:
        by_key.setdefault(_key(e), []).append(e)
    choices = []
    for key, evs in by_key.items():
        opts = [None] + evs          # None = the visible event remains last
        choices.append((key, opts))
    states = [base]
    for key, opts in choices:
        nxt = []
        for st in states:
            for opt in opts:
                s2 = (dict(st[0]), dict(st[1]), dict(st[2]))
                if opt is not None:
                    _apply(s2, opt)
                nxt.append(s2)
        states = nxt
    return states


def answer_from_state(state, factors, query):
    status, present, src = state
    live = [f for f in factors
            if present.get(f["id"], False)
            and status.get(src.get(f["id"], f["source"])) == ("VALID", f["version"])]
    s1 = {f["roles"]["to"] for f in live if f["rel"] == "R0"
          and f["roles"]["from"] == query["from"]
          and f["roles"]["scope"] == query["scope"] and f["roles"]["kind"] == query["kind"]}
    s2 = {f["roles"]["to"] for f in live if f["rel"] == "R1" and f["roles"]["from"] in s1
          and f["roles"]["scope"] == query["scope"] and f["roles"]["kind"] == query["kind"]}
    return int(any(f["rel"] == "R2" and f["roles"]["from"] in s2 and f["roles"]["to"] == query["to"]
                   and f["roles"]["scope"] == query["scope"] and f["roles"]["kind"] == query["kind"]
                   for f in live))


def completions(visible, pool):
    """Every way the pending pool could have been delivered into the visible log.

    Each pending event is either delivered or not; delivered events may appear
    in any relative order at any positions.  Returns distinct resulting logs.
    """
    m = len(pool)
    seen = set()
    out = []
    for k in range(m + 1):
        for subset in combinations(range(m), k):
            chosen = [pool[i] for i in subset]
            for order in _orders(chosen):
                for positions in _insert_positions(len(visible), k):
                    log = list(visible)
                    for pos, e in zip(positions, order):
                        log.insert(pos, e)
                    key = tuple((e["type"], e.get("source"), e.get("factor"),
                                 e["status"] if e["type"] == "SOURCE" else None,
                                 e.get("epoch")) for e in log)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(log)
    return out


def _orders(items):
    if not items:
        return [()]
    res = []
    for i, it in enumerate(items):
        for rest in _orders(items[:i] + items[i + 1:]):
            res.append((it,) + rest)
    return res


def _insert_positions(n, k):
    """Choose k insertion positions (non-decreasing) among n+1 slots."""
    for combo in combinations(range(n + k + 1), k):
        yield tuple(c - i for i, c in enumerate(combo))


def identifiability(case):
    """Per-query identifiability for regime C1 at the decision point.

    'IDENTIFIABLE'      every reachable completion of the pending pool yields
                        the same answer
    'UNDERDETERMINED'   reachable completions disagree
    """
    visible = case["events"]
    pool = case["pending"]
    if not pool:
        return {"status": "NO_PENDING", "queries": {}, "states": 1}
    states = possible_final_states(visible, pool)
    res = {}
    for q in case["queries"]:
        vals = {answer_from_state(st, case["factors"], q) for st in states}
        res[q["id"]] = "IDENTIFIABLE" if len(vals) == 1 else "UNDERDETERMINED"
    return {"status": "OK", "queries": res, "states": len(states)}
