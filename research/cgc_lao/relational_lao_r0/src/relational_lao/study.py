"""Relational LAO R0 study layer: arms, parallel runs over world seeds, compact sealed records, gates, selection
(DEV_PROTOCOL.md; PREREGISTRATION_R6.md for VALIDATION and HOLDOUT).

Arms are built from a few filters and read-only views of them: a candidate, its marginal readout, its
independent-reliability counterpart and its HCA projection share one filter state, because they differ only in the
readout. Ablation arms (identity-permuted, stale) have filters of their own.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from cgc_lao_lab import evaluate as ev1
from cgc_lao_lab import stats

from . import R0_ROOT
from .hca import HcaOnly, LaoHca
from .harness import rollout
from .learners import H3, H4, Bayes, Counts, Exact, Learner, Noop
from .world import AUTHORITY, DESIGN, ROBUSTNESS

SRC = R0_ROOT / "src" / "relational_lao"
LAB = R0_ROOT.parent / "solver_lab"
METRICS = ("comp", "direct", "terminal", "stop", "edge_mse", "online")
CANDIDATES = {
    "RRMF3": (H3, (1.0, 1.0, 1.0), "coupled", "replay"),
    "RRMF3_M": (H3, (1.0, 1.0, 1.0), "marginal", "replay"),
    "RRMF3_A": (H3, (1.0, 1.0, 1.0), "coupled", "slot"),
    "RRMF4": (H4, (1.0, 1.0, 1.0, 1.0), "coupled", "replay"),
    "RRMF4_M": (H4, (1.0, 1.0, 1.0, 1.0), "marginal", "replay"),
    "RRMF4_W": (H4, (1.0, 0.5, 1.0, 2.0), "coupled", "replay"),
    "RRMF4_W_M": (H4, (1.0, 0.5, 1.0, 2.0), "marginal", "replay"),
}
FALLBACKS = {"SAFE4_P8": (H4, (1.0, 1.0, 8.0, 8.0)), "SAFE3_P4": (H3, (1.0, 4.0, 4.0))}
CONTROLS = ("NOOP", "EXACT_RELATIONAL", "STATIC_COUNTS", "HCA_ONLY", "KNOWN_RELIABILITY")
REVISION = ("CLEAN+REV_DRIFT", "FLIP20+REV_DRIFT", "CLEAN+RECLAIM_DRIFT", "FLIP20+RECLAIM_DRIFT")


class View(Learner):
    """A read-only readout of another arm's filter (no update of its own)."""

    def __init__(self, name: str, base: Bayes, readout: str):
        self.name, self.base, self.readout = name, base, readout

    def reset(self, batch):
        self.batch, self.E, self.R = batch, batch.count, batch.R

    def sync(self, batch, t):
        pass

    def observe(self, *args):
        pass

    def components(self, z):
        saved = self.base.readout
        self.base.readout = self.readout
        try:
            return self.base.components(z)
        finally:
            self.base.readout = saved

    def live_counts(self):
        return self.base.live_counts()

    def state_scalars(self):
        return self.base.state_scalars()


def _family(prefix: str, hyp, prior, invalidation: str, authority: bool) -> dict:
    """A filter family: the coupled filter with its views, and its permuted and stale ablation filters."""
    base = Bayes(prefix, hyp, prior, "coupled", invalidation)
    arms = {prefix: base, f"{prefix}_M": View(f"{prefix}_M", base, "marginal"),
            f"{prefix}:IND": View(f"{prefix}:IND", base, "independent"),
            f"{prefix}:HCA": LaoHca(View(f"{prefix}:HCA", base, "marginal"))}
    perm = Bayes(f"{prefix}:PERM", hyp, prior, "coupled", invalidation, permuted=True)
    arms[f"{prefix}:PERM"] = perm
    arms[f"{prefix}_M:PERM"] = View(f"{prefix}_M:PERM", perm, "marginal")
    if authority:
        stale = Bayes(f"{prefix}:STALE", hyp, prior, "coupled", "none")
        arms[f"{prefix}:STALE"] = stale
        arms[f"{prefix}_M:STALE"] = View(f"{prefix}_M:STALE", stale, "marginal")
    return arms


def arms_for(regime: str, driver: str = "NOOP", subset=None) -> dict:
    """Every arm of a round, the driver first (its acquisition drives the shared log)."""
    authority = "+" in regime
    arms: dict = {"NOOP": Noop()}
    arms.update(_family("RRMF3", H3, (1.0, 1.0, 1.0), "replay", authority))
    a = Bayes("RRMF3_A", H3, (1.0, 1.0, 1.0), "coupled", "slot")
    arms.update({"RRMF3_A": a, "RRMF3_A:IND": View("RRMF3_A:IND", a, "independent"),
                 "RRMF3_A:HCA": LaoHca(View("RRMF3_A:HCA", a, "marginal"))})
    arms.update(_family("RRMF4", H4, (1.0, 1.0, 1.0, 1.0), "replay", authority))
    arms.update(_family("RRMF4_W", H4, (1.0, 0.5, 1.0, 2.0), "replay", authority))
    for name, (hyp, prior) in FALLBACKS.items():
        arms[name] = Bayes(name, hyp, prior, "coupled", "replay", first_only=True)
    arms.update({"EXACT_RELATIONAL": Exact(), "STATIC_COUNTS": Counts(), "HCA_ONLY": HcaOnly(),
                 "KNOWN_RELIABILITY": Bayes("KNOWN_RELIABILITY", oracle=True)})
    if driver != "NOOP":
        arms = {driver: arms[driver], **{k: v for k, v in arms.items() if k != driver}}
    if subset is not None:
        keep = [driver] + [k for k in subset if k != driver]
        arms = {k: arms[k] for k in keep}
        for k, v in list(arms.items()):                       # a view needs its base arm's updates
            base = getattr(getattr(v, "inner", v), "base", None)
            if base is not None and base not in arms.values():
                arms = {**arms, f"__base_{k}": base}
    return arms


def counterpart(cand: str, ablation: str) -> str:
    """The ablation arm of a candidate (RRMF4_M -> RRMF4:IND, RRMF4_M:PERM, RRMF4_M:STALE, RRMF4:HCA)."""
    base = cand[:-2] if cand.endswith("_M") else cand
    if ablation in ("IND", "HCA"):
        return f"{base}:{ablation}"
    return f"{cand}:{ablation}" if cand != "RRMF3_A" else f"RRMF3:{ablation}"


def complexity(cand: str) -> tuple:
    """(logical state scalars, likelihood evaluations per update, readout products per alternative)."""
    hyp = CANDIDATES[cand][0]
    M = len(hyp)
    return 3 * M * 8 + 8, 16 * M, 1 if CANDIDATES[cand][2] == "marginal" else M


# -- runs ---------------------------------------------------------------------------------------------------------
def evaluate_seed(partition, seed, regimes, n_episodes, driver="NOOP", policy="PROBE", subset=None,
                  chunk=2048) -> dict:
    from .world import episodes
    out = {}
    for regime in regimes:
        sums: dict = {}
        for first in range(0, n_episodes, chunk):
            batch = episodes(partition, seed, regime, first, min(chunk, n_episodes - first))
            res = rollout(batch, arms_for(regime, driver, subset), driver, policy)
            for arm, metrics in res.items():
                if arm.startswith("__base_"):
                    continue
                acc = sums.setdefault(arm, {})
                for m, v in metrics.items():
                    if m in METRICS:
                        acc[m] = acc.get(m, 0.0) + float(v.sum())
        out[regime] = {arm: {m: v / n_episodes for m, v in acc.items()} for arm, acc in sums.items()}
    return out


def _worker(task):
    partition, seed, regimes, n, driver, policy, subset = task
    return seed, evaluate_seed(partition, seed, regimes, n, driver, policy, subset)


def run(partition, seeds, regimes, n_episodes, driver="NOOP", policy="PROBE", subset=None, workers=4) -> dict:
    seeds, t0 = list(seeds), time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
        by_seed = dict(ex.map(_worker, [(partition, s, list(regimes), n_episodes, driver, policy, subset)
                                        for s in seeds]))
    results = {}
    for r in regimes:
        for arm in by_seed[seeds[0]][r]:
            results.setdefault(r, {})[arm] = {m: [float(f"{by_seed[s][r][arm][m]:.10g}") for s in seeds]
                                              for m in by_seed[seeds[0]][r][arm]}
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": n_episodes, "regimes": list(regimes),
            "driver": driver, "policy": policy, "subset": subset, "results": results,
            "seconds": round(time.time() - t0, 1)}


def source_file_digests() -> dict:
    out = {f"relational_lao/{p.name}": ev1.file_digest(p) for p in sorted(SRC.glob("*.py"))}
    out["SPEC.md"] = ev1.file_digest(R0_ROOT / "SPEC.md")
    for rel in ("src/cgc_lao_lab/rng.py", "src/cgc_lao_lab/stats.py", "src/cgc_lao_lab/filters.py",
                "src/cgc_lao_lab/env.py", "src/cgc_lao_lab/selector.py", "v2/src/cgc_lao_v2/env2.py",
                "v2/src/cgc_lao_v2/providers.py", "v2/src/cgc_lao_v2/host.py", "host_closure/strict_host.py"):
        out[f"solver_lab/{rel}"] = ev1.file_digest(LAB / rel)
    return out


def source_digest() -> str:
    h = hashlib.sha256()
    for name, digest in sorted(source_file_digests().items()):
        h.update(name.encode() + b"\0" + digest.encode() + b"\0")
    return h.hexdigest()


def seal(record: dict, phase: str, note: str = "") -> dict:
    return {"schema": "relational-lao-r0.record.v1", "phase": phase, "note": note,
            "rng_root": "elpis/cgc_lao/relational_lao_r0/", "source_sha256": source_digest(),
            "source_files_sha256": source_file_digests(), "environment": ev1.environment(), **record}


def write(record: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, separators=(",", ":"), sort_keys=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# -- gates --------------------------------------------------------------------------------------------------------
TAU_SAFE = 0.0005


def series(rec, regime, arm, metric="comp") -> np.ndarray:
    if regime == "ALL":
        return np.mean([rec["results"][r][arm][metric] for r in DESIGN], axis=0)
    if regime == "REVISION":
        return np.mean([rec["results"][r][arm][metric] for r in REVISION], axis=0)
    return np.asarray(rec["results"][regime][arm][metric])


def d(rec, regime, a, b, level, metric="comp"):
    return stats.paired(series(rec, regime, a, metric), series(rec, regime, b, metric), level)


def gates(rec, cand: str, scales: dict, level: float, online: dict | None = None) -> dict:
    """The candidate's gates at `level` (RG5 at max(level, 0.975)); online: an on-policy record of the candidate."""
    controls = ["EXACT_RELATIONAL", "STATIC_COUNTS", "HCA_ONLY", counterpart(cand, "IND"), counterpart(cand, "HCA")]
    g = {"RG1": d(rec, "ALL", cand, "NOOP", level)["hi"] < 0,
         "RG2": d(rec, "CLEAN", cand, "NOOP", level)["hi"] < 0
         and d(rec, "CLEAN", cand, "KNOWN_RELIABILITY", level)["hi"] <= scales["delta_clean"],
         "RG3": all(d(rec, "ALL", cand, c, level)["hi"] < 0 for c in controls),
         "RG4": all(d(rec, r, cand, "NOOP", level)["hi"] <= TAU_SAFE for r in ("FLIP40", "FLIP45")),
         "RG5": any(d(rec, r, cand, "NOOP", max(level, 0.975))["hi"] < 0 for r in ("QUNIF", "BURST")),
         "RG6": d(rec, "ALL", cand, counterpart(cand, "PERM"), level)["hi"] < 0}
    if all(r in rec["results"] for r in REVISION):
        g["RG7"] = d(rec, "REVISION", cand, counterpart(cand, "STALE"), level)["hi"] < 0 \
            and d(rec, "REVISION", cand, "NOOP", level)["hi"] < 0
    if online is not None:
        g["RG9"] = stats.mean_ci(series(online, "ALL", cand, "online"), level)["hi"] <= scales["online_budget"]
    return g


def fallback_gates(rec, cand: str, level: float) -> dict:
    return {"F1": d(rec, "FLIP40", cand, "NOOP", level)["hi"] <= TAU_SAFE,
            "F2": d(rec, "CLEAN", cand, "NOOP", level)["hi"] < 0,
            "F3": all(d(rec, r, cand, "NOOP", level)["hi"] <= TAU_SAFE for r in ("FLIP45", "FLIP48")),
            "F4": d(rec, "EVENT40", cand, "NOOP", level)["hi"] <= TAU_SAFE}


def select(rec, scales: dict, online_records: dict, level: float = 0.999) -> dict:
    rows = {}
    for c in CANDIDATES:
        g = gates(rec, c, scales, level, online_records.get(c))
        rows[c] = {"gates": g, "eligible": all(g.values()), "complexity": list(complexity(c)),
                   "ALL": float(series(rec, "ALL", c).mean()),
                   "REVISION": float(series(rec, "REVISION", c).mean()) if "RG7" in g else None}
    eligible = [c for c in rows if rows[c]["eligible"]]
    chosen = min(eligible, key=lambda c: (tuple(rows[c]["complexity"]), rows[c]["ALL"], rows[c]["REVISION"] or 0.0,
                                          c)) if eligible else None
    frows = {}
    for c in FALLBACKS:
        g = fallback_gates(rec, c, level)
        frows[c] = {"gates": g, "eligible": all(g.values()), "CLEAN": float(series(rec, "CLEAN", c).mean())}
    feligible = [c for c in frows if frows[c]["eligible"]]
    fchosen = min(feligible, key=lambda c: (frows[c]["CLEAN"], c)) if feligible else None
    return {"primary": chosen, "fallback": fchosen, "rows": rows, "fallback_rows": frows, "level": level}


ROUND_REGIMES = list(DESIGN + ROBUSTNESS + AUTHORITY)
__all__ = ["run", "seal", "write", "gates", "select", "arms_for", "ROUND_REGIMES", "CANDIDATES", "FALLBACKS"]


# -- CLI ----------------------------------------------------------------------------------------------------------
EVIDENCE = R0_ROOT / "evidence"


def scales_from(train: dict) -> dict:
    gain = series(train, "CLEAN", "NOOP") - series(train, "CLEAN", "KNOWN_RELIABILITY")
    return {"delta_clean": float(f"{0.25 * float(gain.mean()):.8f}"),
            "online_budget": float(f"{1.10 * float(series(train, 'ALL', 'NOOP', 'online').mean()):.8f}"),
            "tau_safe": TAU_SAFE}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("train", "dev", "onpolicy", "select"))
    ap.add_argument("--candidate", default="RRMF3")
    ap.add_argument("--policy", default="PROBE", choices=("PROBE", "VOI"))
    a = ap.parse_args(argv)
    if a.command == "train":
        rec = run("TRAIN", range(32), list(DESIGN), 2048)
        print(write(seal(rec, "R0-TRAIN", "gate scales only"), EVIDENCE / "train" / "round.json"))
        sc = scales_from(rec)
        (EVIDENCE / "train" / "scales.json").write_text(json.dumps(sc, indent=1) + "\n", encoding="utf-8")
        print(sc)
    elif a.command == "dev":
        rec = run("DEV", range(32), ROUND_REGIMES, 2048)
        print(write(seal(rec, "R0-DEV", "the single preregistered DEV round (DEV_PROTOCOL.md)"),
                    EVIDENCE / "dev" / "round.json"))
    elif a.command == "onpolicy":
        regimes = list(DESIGN) + (["FLIP45"] if a.policy == "VOI" else [])
        rec = run("DEV", range(32, 64), regimes, 2048, driver=a.candidate, policy=a.policy,
                  subset=[a.candidate, "NOOP"])
        print(write(seal(rec, "R0-DEV-ONPOLICY", f"{a.candidate} drives with {a.policy}"),
                    EVIDENCE / "dev" / f"onpolicy_{a.candidate}_{a.policy}.json"))
    elif a.command == "select":
        dev = json.loads((EVIDENCE / "dev" / "round.json").read_text(encoding="utf-8"))
        sc = json.loads((EVIDENCE / "train" / "scales.json").read_text(encoding="utf-8"))
        online = {}
        for c in CANDIDATES:
            p = EVIDENCE / "dev" / f"onpolicy_{c}_PROBE.json"
            if p.exists():
                online[c] = json.loads(p.read_text(encoding="utf-8"))
        out = select(dev, sc, online)
        out = {"schema": "relational-lao-r0.selection.v1", **out, "scales": sc,
               "record_sha256": hashlib.sha256((EVIDENCE / "dev" / "round.json").read_bytes()).hexdigest()}
        (R0_ROOT / "SELECTION_R6.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(out["primary"], out["fallback"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
