"""Derived v4 reports, computed only from the sealed records with the frozen, preregistered rules.

    python3 research/cgc_lao/solver_lab/v4/scripts/report4.py disposition   # evidence/DISPOSITION_V4.json
    python3 research/cgc_lao/solver_lab/v4/scripts/report4.py ood           # evidence/ood/characterization.json
    python3 research/cgc_lao/solver_lab/v4/scripts/report4.py tables        # report tables (stdout, markdown)

Gates and family claims: ``cgc_lao_v2.report2`` (95%; G7/F6 97.5%; families Bonferroni 99.375%).
rho_admit (PREREGISTRATION_V2.md, Milestone B): the largest grid value such that every rho' <= rho_admit has
Delta vs NOOP 95% upper < 0 for the frozen primary in DUPLICATE mode. Nothing here runs an agent.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4 / "src"))

from cgc_lao_v4 import study4  # noqa: E402
from cgc_lao_lab import stats  # noqa: E402
from cgc_lao_v2 import report2  # noqa: E402

EV = V4 / "evidence"
RHO_GRID = ((0.0, "FLIP40"), (0.1, "CORR10"), (0.25, "CORR25"), (0.5, "CORR50"), (0.75, "CORR75"),
            (0.9, "CORR90"), (1.0, "EVENT40"))
SCOPE = ("Synthetic SLB2 qualification only. It is not a production admission: a real host must establish "
         "its provider's measurement independence separately (registered, audited provenance).")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen() -> tuple[str, str]:
    f = load(V4 / "FROZEN_V4.json")
    return f["candidates"]["PRIMARY"]["name"], f["candidates"]["FALLBACK"]["name"]


def d(rec, regime, a, b, level=0.95) -> dict:
    return stats.paired(report2.series(rec, regime, a), report2.series(rec, regime, b), level)


def triple(x: dict) -> list:
    return [x["mean"], x["lo"], x["hi"]]


def phase(name: str, primary: str, fallback: str) -> dict | None:
    qpath, ppath = EV / name.lower() / "qualification.json", EV / name.lower() / "native_parity.json"
    if not qpath.exists():
        return None
    rec, par = load(qpath), load(ppath)
    g, f = report2.primary_gates(rec, primary), report2.fallback_gates(rec, fallback)
    return {"primary": {k: bool(g[k]["pass"]) for k in study4.PRIMARY_GATES},
            "fallback": {k: bool(f[k]["pass"]) for k in study4.FALLBACK_GATES},
            "families": {fam: v["claim"] for fam, v in g["families"].items()},
            "clean_retention": {"primary": g["clean_retention"], "fallback": f["clean_retention"]},
            "FLIP40_vs_NOOP": triple(g["G1"]), "FLIP45_vs_NOOP": triple(g["G9"]["vs_NOOP"]),
            "FLIP48_vs_NOOP": triple(d(rec, "FLIP48", primary, "NOOP")),
            "intervals": "[mean, lo, hi], paired Student-t over router seeds, 95% two-sided",
            "native_parity": {role: bool(arm["pass"]) for role, arm in par["arms"].items()},
            "record_sha256": sha(qpath), "native_parity_sha256": sha(ppath)}


def disposition() -> dict:
    primary, fallback = frozen()
    phases = {p: s for p in ("VALIDATION", "HOLDOUT") if (s := phase(p, primary, fallback)) is not None}
    failing = [f"{k}@{p}" for p, s in phases.items() for k, ok in s["primary"].items() if not ok]
    ffailing = [f"{k}@{p}" for p, s in phases.items() for k, ok in s["fallback"].items() if not ok]
    complete = set(phases) == {"VALIDATION", "HOLDOUT"}
    return {"schema": "cgc-lao-solver-lab.disposition.v4", "phases": phases,
            "primary": {"name": primary, "failing": failing,
                        "status": ("QUALIFIED_SYNTHETIC_CONTRACT_I" if complete and not failing else
                                   "NOT_QUALIFIED" if failing else "PENDING")},
            "fallback": {"name": fallback, "failing": ffailing,
                         "status": ("QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE" if complete and not ffailing else
                                    "NOT_QUALIFIED" if ffailing else "PENDING")},
            "holdout": "OPENED_ONCE" if "HOLDOUT" in phases else "NOT_OPENED",
            "ood": "opened once after this disposition was committed: evidence/ood/characterization.json",
            "scope": SCOPE}


def ood() -> dict:
    primary, fallback = frozen()
    path = EV / "ood" / "qualification.json"
    rec = load(path)
    sweep, rho_admit, admitted = [], None, True
    for rho, regime in RHO_GRID:
        x, fb = d(rec, regime, primary, "NOOP"), d(rec, regime, fallback, "NOOP")
        admitted = admitted and x["hi"] < 0
        rho_admit = rho if admitted else rho_admit
        sweep.append({"rho": rho, "regime": regime, "primary_vs_NOOP": triple(x), "primary_wins": x.get("wins"),
                      "fallback_vs_NOOP": triple(fb), "primary_upper_below_zero": x["hi"] < 0})
    regimes = {}
    for regime in rec["regimes"]:
        row = {}
        for label in (primary, f"{primary}@SECOND_EVENT", fallback, "RMF_V1", "CLEAN_BAYES", "KNOWN_Q_DUP"):
            if report2.has(rec, regime, label):
                x = d(rec, regime, label, "NOOP")
                row[label] = {"vs_NOOP": triple(x),
                              "claim": "USEFUL" if x["hi"] < 0 else ("SAFE" if x["hi"] <= report2.TAU_SAFE else "HARM")}
        row["NOOP_regret"] = float(report2.series(rec, regime, "NOOP").mean())
        regimes[regime] = row
    return {"schema": "cgc-lao-solver-lab.ood-characterization.v4", "record_sha256": sha(path),
            "primary": primary, "fallback": fallback,
            "rule": "rho_admit = largest grid rho such that every rho' <= rho has primary (DUPLICATE) Delta vs NOOP "
                    "95% upper < 0 (PREREGISTRATION_V2.md); claims per regime at 95%: USEFUL upper < 0, SAFE upper "
                    "<= +0.0005, else HARM",
            "rho_sweep": sweep, "rho_admit": rho_admit, "regimes": regimes, "scope": SCOPE}


def tables() -> None:
    primary, fallback = frozen()
    rec = load(EV / "holdout" / "qualification.json")
    cols = ("CLEAN", "FLIP20", "FLIP40", "FLIP45", "FLIP48", "QUNIF", "BURST", "EVENT40", "ALL")

    def row(label, a, b):
        cells = []
        for r in cols:
            x = d(rec, r, a, b)
            cells.append(f"{x['mean']:+.5f}{'' if x['hi'] < 0 or x['lo'] > 0 else '~'}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("HOLDOUT, Delta = row arm - column... (~: 95% interval contains 0)\n")
    print("| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    row(f"{primary} - NOOP", primary, "NOOP")
    row(f"{fallback} - NOOP", fallback, "NOOP")
    for c in study4.CONTROLS4:
        if c != primary and report2.has(rec, "CLEAN", c):
            row(f"{primary} - {c}", primary, c)
    for b in ("SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES", "KNOWN_Q_ORACLE", "KNOWN_Q_DUP"):
        row(f"{primary} - {b}", primary, b)
    print("\n| arm | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for a in ("NOOP", primary, fallback, "RMF_V1", "CLEAN_BAYES", "KNOWN_Q_DUP"):
        print(f"| {a} | " + " | ".join(f"{float(report2.series(rec, r, a).mean()):.5f}" for r in cols) + " |")


def main(argv) -> None:
    cmd = argv[0] if argv else ""
    if cmd == "disposition":
        out = EV / "DISPOSITION_V4.json"
        out.write_text(json.dumps(disposition(), indent=1) + "\n", encoding="utf-8")
        print(out, sha(out))
    elif cmd == "ood":
        out = EV / "ood" / "characterization.json"
        out.write_text(json.dumps(ood(), indent=1) + "\n", encoding="utf-8")
        print(out, sha(out))
    elif cmd == "tables":
        tables()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
