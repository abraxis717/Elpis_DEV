"""Scoring of the ECS Memory MF-R0 laboratory, exactly as PREREGISTRATION.md ("Hypotheses") fixes it.

Pure function of the laboratory rows: accuracies are pooled over seeds (correct / total); verdicts are PASS or
NONPASS; a hypothesis whose rows are missing is NOT_RUN, never PASS. Standard library only.
"""
from __future__ import annotations

CANDIDATES = 64
CHANCE = 1.0 / CANDIDATES


def pooled(rows):
    table = {}
    for r in rows:
        key = (r["scenario"], r["condition"], r["arm"])
        c, t = table.get(key, (0, 0))
        table[key] = (c + r["correct"], t + r["total"])
    return table


def accuracy(table, scenario, condition, arm):
    c, t = table.get((scenario, condition, arm), (0, 0))
    if not t:
        raise KeyError((scenario, condition, arm))
    return c / t


def _verdict(checks):
    return "PASS" if all(checks.values()) else "NONPASS"


def score(rows):
    table = pooled(rows)
    acc = {"|".join(k): (v[0] / v[1] if v[1] else None) for k, v in sorted(table.items())}
    verdicts = {}

    def run(name, fn):
        try:
            checks, values = fn()
            verdicts[name] = {"verdict": _verdict(checks), "checks": checks, "values": values}
        except KeyError as missing:
            verdicts[name] = {"verdict": "NOT_RUN", "missing": list(missing.args[0])}

    def h1():
        a = {arm: accuracy(table, "S1", "base", arm) for arm in ("FAST", "NONE", "FIXED", "PERM")}
        controls = max(a["NONE"], a["FIXED"], a["PERM"])
        checks = {
            "FAST >= 0.95": a["FAST"] >= 0.95,
            "NONE <= chance + 0.05": a["NONE"] <= CHANCE + 0.05,
            "FIXED <= chance + 0.05": a["FIXED"] <= CHANCE + 0.05,
            "PERM <= chance + 0.05": a["PERM"] <= CHANCE + 0.05,
            "FAST - max(controls) >= 0.80": a["FAST"] - controls >= 0.80,
        }
        return checks, a

    def h2():
        a = {f"FAST/{c}": accuracy(table, "S2", c, "FAST") for c in ("AB", "ABA", "ABAB")}
        a["HEBB/ABAB"] = accuracy(table, "S2", "ABAB", "HEBB")
        checks = {f"{k} >= 0.90": v >= 0.90 for k, v in a.items() if k.startswith("FAST/")}
        checks["HEBB/ABAB <= FAST/ABAB - 0.20"] = a["HEBB/ABAB"] <= a["FAST/ABAB"] - 0.20
        return checks, a

    def h3():
        a = {"FAST/UNIQUE/agent": accuracy(table, "S5", "UNIQUE/agent", "FAST")}
        return {"FAST/UNIQUE/agent >= 0.80": a["FAST/UNIQUE/agent"] >= 0.80}, a

    def h4():
        a, checks = {}, {}
        for k in ("K=0", "K=256", "K=1024", "K=4096"):
            a[f"FAST/{k}"] = accuracy(table, "S6", k, "FAST")
            a[f"KEYED/{k}"] = accuracy(table, "S6", k, "KEYED")
            checks[f"FAST/{k} >= KEYED/{k}"] = a[f"FAST/{k}"] >= a[f"KEYED/{k}"]
        return checks, a

    def h5():
        a = {
            "SLOW/RECUR": accuracy(table, "S7", "RECUR", "SLOW"),
            "FAST/RECUR": accuracy(table, "S7", "RECUR", "FAST"),
            "SLOW/NORECUR": accuracy(table, "S7", "NORECUR", "SLOW"),
        }
        checks = {
            "SLOW/RECUR >= FAST/RECUR + 0.25": a["SLOW/RECUR"] >= a["FAST/RECUR"] + 0.25,
            "SLOW/RECUR >= SLOW/NORECUR + 0.25": a["SLOW/RECUR"] >= a["SLOW/NORECUR"] + 0.25,
        }
        return checks, a

    for name, fn in (("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4), ("H5", h5)):
        run(name, fn)
    return {"chance": CHANCE, "accuracy": acc, "verdicts": verdicts}
