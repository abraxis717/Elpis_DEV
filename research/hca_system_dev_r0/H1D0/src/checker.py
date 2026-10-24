"""Independent contract checker for H1D0.

Separate from the generator and from every engine.  It re-derives the gold
labels with its own naive semantics (checker_local_labels) and refuses any
dataset that violates the declared invariants.  Every refusal is a raised
ContractViolation, never a swallowed exception.
"""
from collections import Counter, defaultdict

from contract import (SCHEMA, STATUSES, EPOCHS, RELS, ROLES, EVENT_TYPES, KINDS)


class ContractViolation(ValueError):
    pass


def need(cond, why):
    if not cond:
        raise ContractViolation(why)


# ---------------------------------------------------------------- semantics --
def checker_local_labels(events, factors, queries):
    """Second, independent implementation of the declared fold."""
    rows = []
    status, present, src = {}, {}, {}
    rows.append({q["id"]: witness(events, factors, src, status, present, q) for q in queries})
    for e in events:
        if e["type"] == "SOURCE":
            status[e["source"]] = (e["status"], e["epoch"])
        elif e["type"] == "EMIT":
            present[e["factor"]] = True
        elif e["type"] == "DROP":
            present[e["factor"]] = False
        else:
            src[e["factor"]] = e["source"]
        rows.append({q["id"]: witness(events, factors, src, status, present, q) for q in queries})
    return rows


def witness(events, factors, src, status, present, q):
    live = [f for f in factors
            if present.get(f["id"], False)
            and status.get(src.get(f["id"], f["source"])) == ("VALID", f["version"])]
    step1 = {f["roles"]["to"] for f in live
             if f["rel"] == "R0" and f["roles"]["from"] == q["from"]
             and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]}
    step2 = {f["roles"]["to"] for f in live
             if f["rel"] == "R1" and f["roles"]["from"] in step1
             and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]}
    return int(any(f["rel"] == "R2" and f["roles"]["from"] in step2 and f["roles"]["to"] == q["to"]
                    and f["roles"]["scope"] == q["scope"] and f["roles"]["kind"] == q["kind"]
                    for f in live))


# ------------------------------------------------------------- structural ------
def factor_bag(case):
    return Counter((f["id"], f["rel"], f["source"], f["version"],
                    tuple(sorted(f["roles"].items()))) for f in case["factors"])


def role_bag(case):
    return Counter((f["rel"], key, value) for f in case["factors"] for key, value in f["roles"].items())


def event_bag(case):
    return Counter((e["type"], e.get("source"), e.get("factor"),
                    e.get("status"), e.get("epoch")) for e in case["events"])


def status_bag(case):
    return Counter((e["status"], e["epoch"]) for e in case["events"] if e["type"] == "SOURCE")


def source_bag(case):
    return Counter(f["source"] for f in case["factors"])


def validate_dataset(ds, truth, which):
    need(set(ds) == {"schema", "split", "worlds"}, "dataset schema")
    need(ds["schema"] == SCHEMA, "schema mismatch")
    need(ds["split"] == which, "split mismatch")
    need(which in ("TRAIN", "TUNE"), "sealed split is not permitted in H1D0")

    case_ids, world_ids, pairs = set(), set(), defaultdict(list)
    stats = Counter()
    for w in ds["worlds"]:
        need(set(w) == {"world_id", "cases"}, "world schema")
        need(w["world_id"] not in world_ids, "duplicate world id")
        world_ids.add(w["world_id"])
        for case in w["cases"]:
            cid = case["case_id"]
            need(cid not in case_ids, "duplicate case id")
            case_ids.add(cid)
            need(set(case) == {"case_id", "pair_id", "kind", "target_query_id",
                               "factors", "queries", "events", "pending",
                               "damage_prefix"}, "case schema / unexpected field")
            need(case["kind"] in KINDS, "unknown intervention kind")
            need(len(case["factors"]) <= 64, "factor budget")
            need(len(case["queries"]) == 8, "query denominator")
            need(len(case["events"]) <= 40, "event budget")
            need(0 <= case["damage_prefix"] <= len(case["events"]), "damage prefix bounds")

            fids, tokens = set(), set()
            for f in case["factors"]:
                need(set(f) == {"id", "rel", "roles", "source", "version"}, "factor schema")
                need(f["id"] not in fids, "duplicate factor id")
                fids.add(f["id"])
                need(f["rel"] in RELS, "relation enum")
                need(tuple(sorted(f["roles"])) == tuple(sorted(ROLES)), "role schema")
                need(all(isinstance(v, str) and v for v in f["roles"].values()), "empty role token")
                need(type(f["version"]) is int and f["version"] in EPOCHS, "invalid factor version")
                tokens.update(f["roles"].values())
                tokens.add(f["source"])
            qids = [q["id"] for q in case["queries"]]
            need(len(qids) == len(set(qids)), "duplicate query id")
            need(case["target_query_id"] in qids, "target query absent")
            for q in case["queries"]:
                need(set(q) == {"id", "from", "to", "scope", "kind"}, "query schema")
                tokens.update([q["from"], q["to"], q["scope"], q["kind"]])
            steps = []
            for e in case["events"]:
                need(e["type"] in EVENT_TYPES, "event type enum")
                need(type(e["step"]) is int, "step type")
                steps.append(e["step"])
                if e["type"] == "SOURCE":
                    need(e["status"] in STATUSES, "status enum")
                    need(type(e["epoch"]) is int and e["epoch"] in EPOCHS, "epoch enum")
                    need(e["source"] in tokens, "unknown source token")
                else:
                    need(e["factor"] in fids, "unknown factor in event")
                if e["type"] == "REBIND":
                    need(e["source"] in tokens, "unknown rebind source")
            need(all(x < y for x, y in zip(steps, steps[1:])), "steps not strictly increasing")
            need(len(case["pending"]) <= 3, "pending pool budget")
            for e in case["pending"]:
                need(e["type"] in EVENT_TYPES, "pending event type enum")

            gold = checker_local_labels(case["events"], case["factors"], case["queries"])
            full = truth[cid]
            need(len(full) >= len(case["events"]), "truth log shorter than visible log")
            stats["cases"] += 1
            stats["events"] += len(case["events"])
            stats["pending"] += len(case["pending"])
            stats["rows"] += len(gold) * len(qids)
            stats["positives"] += sum(sum(r.values()) for r in gold)
            pairs[(w["world_id"], case["pair_id"])].append((case, gold, full))
    need(len(world_ids) == len(ds["worlds"]), "world uniqueness")
    return case_ids, pairs, stats


def check_pairs(pairs, expected_per_kind):
    counts = Counter()
    for (wid, pid), items in pairs.items():
        need(len(items) == 2, "pair multiplicity")
        (a, ga, fa), (b, gb, fb) = items
        need(a["kind"] == b["kind"], "pair kind drift")
        need(a["queries"] == b["queries"], "pair query drift")
        need(a["target_query_id"] == b["target_query_id"], "pair target drift")
        need(sorted([ga[-1][a["target_query_id"]], gb[-1][b["target_query_id"]]]) == [0, 1],
            "counterfactual pair lacks opposite gold")
        kind = a["kind"]
        if kind in ("source_swap", "event_order", "epoch_rebind"):
            need(factor_bag(a) == factor_bag(b), kind + ": factor nuisance drift")
            need(status_bag(a) == status_bag(b), kind + ": status nuisance drift")
            need(source_bag(a) == source_bag(b), kind + ": source nuisance drift")
        elif kind == "structural":
            need(role_bag(a) == role_bag(b), "structural: role bag drift")
            need(event_bag(a) == event_bag(b), "structural: event bag drift")
        elif kind == "cancellation":
            need(factor_bag(a) == factor_bag(b), "cancellation: factor nuisance drift")
            need(event_bag(a) == event_bag(b), "cancellation: event multiset drift")
        elif kind == "provenance":
            need(role_bag(a) == role_bag(b), "provenance: role bag drift")
            need(status_bag(a) == status_bag(b), "provenance: status bag drift")
        counts["total_" + kind] += 1
        counts["both_" + kind] += 1
    for kind in KINDS:
        need(counts["total_" + kind] == expected_per_kind[kind],
            "kind coverage " + kind + " = " + str(counts["total_" + kind]))
    return dict(counts)


def nuisance_ablation(case):
    """Rename every token consistently; the gold must be unchanged.

    Proves the semantics does not depend on token spelling.
    """
    tokens = set()
    for f in case["factors"]:
        tokens.update(f["roles"].values())
        tokens.add(f["source"])
    for q in case["queries"]:
        tokens.update([q["from"], q["to"], q["scope"], q["kind"]])
    mapping = {t: "r" + str(i) for i, t in enumerate(sorted(tokens))}
    remapped = {
        "factors": [{"id": f["id"], "rel": f["rel"], "source": mapping[f["source"]],
                     "version": f["version"],
                     "roles": {k: mapping[v] for k, v in f["roles"].items()}}
                    for f in case["factors"]],
        "queries": [{"id": q["id"], "from": mapping[q["from"]], "to": mapping[q["to"]],
                    "scope": mapping[q["scope"]], "kind": mapping[q["kind"]]}
                    for q in case["queries"]],
        "events": [{"step": e["step"], "type": e["type"],
                    **({"source": mapping[e["source"]]} if "source" in e else {}),
                    **({"factor": e["factor"]} if "factor" in e else {}),
                    **({"status": e["status"]} if "status" in e else {}),
                    **({"epoch": e["epoch"]} if "epoch" in e else {})}
                    for e in case["events"]],
    }
    original = checker_local_labels(case["events"], case["factors"], case["queries"])
    renamed = checker_local_labels(remapped["events"], remapped["factors"], remapped["queries"])
    need(original == renamed, "gold depends on token spelling")
    return True
