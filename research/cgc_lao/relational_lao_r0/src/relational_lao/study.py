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
METRICS = ("comp", "direct", "terminal", "stop", "boundary", "edge_mse", "online", "path", "conj", "reach")
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
    if subset is not None:                                    # a :STALE arm exists only in authority regimes
        keep = [driver] + [k for k in subset if k != driver and k in arms]
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
            names = dict.fromkeys(m for s in seeds for m in by_seed[s][r][arm])   # a topology may lack a kind
            results.setdefault(r, {})[arm] = {m: [float(f"{by_seed[s][r][arm][m]:.10g}")
                                                  if m in by_seed[s][r][arm] else None for s in seeds]
                                              for m in names}
    from .world import template_for
    return {"partition": partition, "seeds": seeds, "episodes_per_seed": n_episodes, "regimes": list(regimes),
            "templates": [template_for(partition, s).name for s in seeds], "driver": driver, "policy": policy,
            "subset": subset, "results": results, "seconds": round(time.time() - t0, 1)}


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


def series(rec, regime, arm, metric="comp", seeds=None) -> np.ndarray:
    if regime == "ALL":
        out = np.mean([rec["results"][r][arm][metric] for r in DESIGN], axis=0)
    elif regime == "REVISION":
        out = np.mean([rec["results"][r][arm][metric] for r in REVISION], axis=0)
    else:
        out = np.asarray(rec["results"][regime][arm][metric])
    return out if seeds is None else out[seeds]


def heldout_seeds(rec) -> np.ndarray | None:
    """Seed positions of the partition's held-out topology (V_LADDER on VALIDATION, H_RANDOM on HOLDOUT)."""
    names = rec.get("templates", [])
    if rec["partition"] == "VALIDATION":
        mask = np.array([n == "V_LADDER" for n in names])
    elif rec["partition"] == "HOLDOUT":
        mask = np.array([n.startswith("H_RANDOM") for n in names])
    else:
        return None
    return np.nonzero(mask)[0] if mask.any() else None


def d(rec, regime, a, b, level, metric="comp", seeds=None):
    return stats.paired(series(rec, regime, a, metric, seeds), series(rec, regime, b, metric, seeds), level)


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
    held = heldout_seeds(rec)
    if held is not None:
        g["RG8"] = d(rec, "ALL", cand, "NOOP", level, seeds=held)["hi"] < 0 \
            and d(rec, "ALL", cand, counterpart(cand, "IND"), level, seeds=held)["hi"] < 0
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


# -- qualification analysis (PREREGISTRATION_R6.md) -------------------------------------------------------------
QUAL_LEVEL = 0.95
FAMILY_REGIMES = ("CLEAN", "FLIP20", "FLIP40", "FLIP45", "FLIP48", "QUNIF", "BURST", "EVENT40")
FAMILY_LEVEL = 1.0 - 0.05 / len(FAMILY_REGIMES)                 # Bonferroni over the 8 families: 99.375%
RHO_GRID = (("FLIP40", 0.0), ("CORR10", 0.1), ("CORR25", 0.25), ("CORR50", 0.5), ("CORR75", 0.75),
            ("CORR90", 0.9), ("EVENT40", 1.0))
KINDS = ("path", "conj", "reach")
COUPLING_PARTS = ("RG3_IND", "RG8_IND")


def _iv(x: dict) -> list:
    return [x["mean"], x["lo"], x["hi"]]


def gate_parts(rec, cand: str, scales: dict, level: float, online: dict | None = None) -> dict:
    """The gates of `gates` split into their named comparisons (the disposition names each failing part)."""
    hi = lambda regime, b, lv=level, seeds=None: d(rec, regime, cand, b, lv, seeds=seeds)["hi"]  # noqa: E731
    p = {"RG1": hi("ALL", "NOOP") < 0, "RG2_NOOP": hi("CLEAN", "NOOP") < 0,
         "RG2_KNOWN": hi("CLEAN", "KNOWN_RELIABILITY") <= scales["delta_clean"]}
    for c in ("EXACT_RELATIONAL", "STATIC_COUNTS", "HCA_ONLY"):
        p[f"RG3_{c}"] = hi("ALL", c) < 0
    p["RG3_IND"] = hi("ALL", counterpart(cand, "IND")) < 0
    p["RG3_HCA"] = hi("ALL", counterpart(cand, "HCA")) < 0
    p["RG4_FLIP40"], p["RG4_FLIP45"] = hi("FLIP40", "NOOP") <= TAU_SAFE, hi("FLIP45", "NOOP") <= TAU_SAFE
    p["RG5"] = any(hi(r, "NOOP", max(level, 0.975)) < 0 for r in ("QUNIF", "BURST"))
    p["RG6"] = hi("ALL", counterpart(cand, "PERM")) < 0
    if all(r in rec["results"] for r in REVISION):
        p["RG7_STALE"] = hi("REVISION", counterpart(cand, "STALE")) < 0
        p["RG7_NOOP"] = hi("REVISION", "NOOP") < 0
    held = heldout_seeds(rec)
    if held is not None:
        p["RG8_NOOP"] = hi("ALL", "NOOP", seeds=held) < 0
        p["RG8_IND"] = hi("ALL", counterpart(cand, "IND"), seeds=held) < 0
    if online is not None:
        p["RG9"] = stats.mean_ci(series(online, "ALL", cand, "online"), level)["hi"] <= scales["online_budget"]
    whole = gates(rec, cand, scales, level, online)
    for g, ok in whole.items():                                 # the parts are exactly the gates
        assert ok == all(v for k, v in p.items() if k == g or k.startswith(g + "_")), g
    return {k: bool(v) for k, v in p.items()}


def families(rec, cand: str, level: float = FAMILY_LEVEL) -> dict:
    out = {}
    for r in FAMILY_REGIMES:
        hi = d(rec, r, cand, "NOOP", level)["hi"]
        out[r] = "USEFUL" if hi < 0 else "SAFE" if hi <= TAU_SAFE else "NOT_QUALIFIED"
    return out


def _kind(rec, regime: str, arm: str, kind: str) -> np.ndarray:
    regs = DESIGN if regime == "ALL" else (regime,)
    n = len(rec["seeds"])
    rows = [[np.nan if v is None else v for v in rec["results"][r][arm].get(kind, [None] * n)] for r in regs]
    return np.mean(np.asarray(rows, dtype=np.float64), axis=0)


def claims(rec, cand: str, level: float = QUAL_LEVEL) -> dict:
    """The secondary estimates reported with the disposition (95% paired intervals, [mean, lo, hi]; not gates)."""
    controls = ["EXACT_RELATIONAL", "STATIC_COUNTS", "HCA_ONLY", counterpart(cand, "IND"), counterpart(cand, "HCA")]
    best = min(controls, key=lambda c: (float(series(rec, "ALL", c).mean()), c))
    gain = series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", cand)
    oracle = series(rec, "CLEAN", "NOOP") - series(rec, "CLEAN", "KNOWN_RELIABILITY")
    kinds = {}
    for k in KINDS:
        a, b = _kind(rec, "ALL", cand, k), _kind(rec, "ALL", "NOOP", k)
        ok = np.isfinite(a) & np.isfinite(b)
        kinds[k] = _iv(stats.paired(a[ok], b[ok], level)) + [int(ok.sum())] if ok.sum() >= 2 else None
    out = {"level": level,
           "DIRECT_RELATION_GAIN": _iv(d(rec, "ALL", cand, "NOOP", level, "direct")),
           "COMPOSITIONAL_GAIN_VS_NOOP": _iv(d(rec, "ALL", cand, "NOOP", level)),
           "COMPOSITIONAL_GAIN_VS_BEST_CONTROL": {"control": best, "delta": _iv(d(rec, "ALL", cand, best, level))},
           "RELATIONAL_COUPLING_VS_IND": _iv(d(rec, "ALL", cand, counterpart(cand, "IND"), level)),
           "RELATION_IDENTITY_ABLATION": _iv(d(rec, "ALL", cand, counterpart(cand, "PERM"), level)),
           "GEOMETRIC_PROJECTION_GAIN": _iv(d(rec, "ALL", counterpart(cand, "HCA"), cand, level)),
           "CONTRADICTION_VS_NOOP": {r: _iv(d(rec, r, cand, "NOOP", level)) for r in ("FLIP45", "FLIP48", "EVENT40")},
           "STOP": {"stop_rate": float(series(rec, "ALL", cand, "stop").mean()),
                    "noop_stop_rate": float(series(rec, "ALL", "NOOP", "stop").mean()),
                    "known_stop_rate": float(series(rec, "ALL", "KNOWN_RELIABILITY", "stop").mean()),
                    "boundary_vs_NOOP": _iv(d(rec, "ALL", cand, "NOOP", level, "boundary"))},
           "ONLINE_REGRET": _iv(stats.mean_ci(series(rec, "ALL", cand, "online"), level)),
           "CLEAN_VS_KNOWN": _iv(d(rec, "CLEAN", cand, "KNOWN_RELIABILITY", level)),
           "CLEAN_RETENTION": float(gain.mean() / oracle.mean()),
           "EDGE_MSE_VS_NOOP": _iv(d(rec, "ALL", cand, "NOOP", level, "edge_mse")),
           "KINDS_VS_NOOP": kinds}
    if all(r in rec["results"] for r in REVISION):
        out["SOURCE_REVISION_VS_STALE"] = _iv(d(rec, "REVISION", cand, counterpart(cand, "STALE"), level))
        out["SOURCE_REVISION_VS_NOOP"] = _iv(d(rec, "REVISION", cand, "NOOP", level))
    held = heldout_seeds(rec)
    if held is not None:
        out["HELDOUT_TOPOLOGY"] = {"seeds": int(len(held)),
                                   "vs_NOOP": _iv(d(rec, "ALL", cand, "NOOP", level, seeds=held)),
                                   "vs_IND": _iv(d(rec, "ALL", cand, counterpart(cand, "IND"), level, seeds=held))}
    return out


def primary_status(parts: dict) -> tuple:
    """parts: {phase: gate parts or None (not opened)}. Returns (status, failing parts per opened phase)."""
    failing = {ph: [k for k, ok in p.items() if not ok] for ph, p in parts.items() if p is not None}
    if parts.get("HOLDOUT") is None:
        return "NOT_QUALIFIED", failing
    if not failing["VALIDATION"] and not failing["HOLDOUT"]:
        return "QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_I", failing
    if not failing["VALIDATION"] and set(failing["HOLDOUT"]) <= set(COUPLING_PARTS):
        return "INDEPENDENT_RELATIONAL_FILTER_ADAPTER", failing
    return "NOT_QUALIFIED", failing


def fallback_status(parts: dict) -> tuple:
    failing = {ph: [k for k, ok in p.items() if not ok] for ph, p in parts.items() if p is not None}
    ok = parts.get("HOLDOUT") is not None and not any(failing.values())
    return ("QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_II_CONSERVATIVE" if ok else "NOT_QUALIFIED"), failing


def rho_admission(ood, cand: str, level: float = QUAL_LEVEL) -> dict:
    """The largest grid rho whose every rho' <= rho has compositional C - NOOP upper <= tau_safe (None if rho = 0
    already fails); the host can use at most the strict cap MAX_QUALIFIED_RHO."""
    from host_closure.strict_host import MAX_QUALIFIED_RHO
    rows, admit, ok = [], None, True
    for regime, rho in RHO_GRID:
        x = d(ood, regime, cand, "NOOP", level)
        safe = x["hi"] <= TAU_SAFE
        ok = ok and safe
        if ok:
            admit = rho
        rows.append({"regime": regime, "rho": rho, "vs_NOOP": _iv(x), "safe": bool(safe)})
    return {"rho_admit": admit, "host_rho_admit": None if admit is None else min(admit, MAX_QUALIFIED_RHO),
            "strict_cap": MAX_QUALIFIED_RHO, "grid": rows}


def ood_characterisation(ood, cand: str, level: float = QUAL_LEVEL) -> dict:
    regimes = {}
    for r in ood["regimes"]:
        x = d(ood, r, cand, "NOOP", level)
        regimes[r] = {"vs_NOOP": _iv(x), "label": "USEFUL" if x["hi"] < 0 else "SAFE" if x["hi"] <= TAU_SAFE
                      else "NOT_QUALIFIED"}
    groups: dict = {}
    for i, name in enumerate(ood["templates"]):
        groups.setdefault("H_RANDOM" if name.startswith("H_RANDOM") else name, []).append(i)
    topologies = {g: {"seeds": len(ix),
                      "vs_NOOP": _iv(d(ood, "ALL", cand, "NOOP", level, seeds=np.array(ix))),
                      "vs_IND": _iv(d(ood, "ALL", cand, counterpart(cand, "IND"), level, seeds=np.array(ix)))}
                  if len(ix) >= 2 else {"seeds": len(ix)} for g, ix in groups.items()}
    return {"level": level, "regimes": regimes, "topologies": topologies}


ROUND_REGIMES = list(DESIGN + ROBUSTNESS + AUTHORITY)
__all__ = ["run", "seal", "write", "gates", "select", "arms_for", "ROUND_REGIMES", "CANDIDATES", "FALLBACKS"]


# -- CLI ----------------------------------------------------------------------------------------------------------
EVIDENCE = R0_ROOT / "evidence"


def scales_from(train: dict) -> dict:
    gain = series(train, "CLEAN", "NOOP") - series(train, "CLEAN", "KNOWN_RELIABILITY")
    return {"delta_clean": float(f"{0.25 * float(gain.mean()):.8f}"),
            "online_budget": float(f"{1.10 * float(series(train, 'ALL', 'NOOP', 'online').mean()):.8f}"),
            "tau_safe": TAU_SAFE}


PLANS = {
    "VALIDATION": {"partition": "VALIDATION", "seeds": [0, 32], "episodes": 4096, "regimes": ROUND_REGIMES},
    "HOLDOUT": {"partition": "HOLDOUT", "seeds": [0, 64], "episodes": 4096, "regimes": ROUND_REGIMES},
    "OOD": {"partition": "OOD", "seeds": [0, 64], "episodes": 2048,
            "regimes": ["CLEAN", "FLIP20", "FLIP40", "CORR10", "CORR25", "CORR50", "CORR75", "CORR90", "EVENT40",
                        "SHARED_SRC", "SECOND_EVENT40", "ASYM40", "CTXQ40", "FLIP30"]},
}


FROZEN_DOCUMENTS = ("DEV_PROTOCOL.md", "PREREGISTRATION_R6.md", "SELECTION_R6.json", "evidence/train/scales.json",
                    "evidence/train/round.json", "evidence/dev/round.json")


def qualification_arms(primary: str, fallback: str) -> list:
    """The frozen primary (the driver), every arm its gates read, the fallback and the controls."""
    arms = [primary, "NOOP", counterpart(primary, "IND"), counterpart(primary, "HCA"), counterpart(primary, "PERM"),
            counterpart(primary, "STALE"), fallback, "EXACT_RELATIONAL", "STATIC_COUNTS", "HCA_ONLY",
            "KNOWN_RELIABILITY"]
    return list(dict.fromkeys(arms))


def freeze() -> dict:
    sel = json.loads((R0_ROOT / "SELECTION_R6.json").read_text(encoding="utf-8"))
    primary, fallback = sel["primary"], sel["fallback"]
    return {"schema": "relational-lao-r0.freeze.v1",
            "note": "Frozen before any VALIDATION, HOLDOUT or OOD run; selection by SELECTION_R6.json.",
            "primary": {"name": primary, "config": [list(map(list, CANDIDATES[primary][0])),
                                                     list(CANDIDATES[primary][1])] + list(CANDIDATES[primary][2:]),
                        "contract": "I", "acquisition": sel.get("acquisition", "PROBE")},
            "fallback": {"name": fallback, "config": [list(map(list, FALLBACKS[fallback][0])),
                                                       list(FALLBACKS[fallback][1])], "contract": "II"},
            "arms": qualification_arms(primary, fallback), "scales": sel["scales"],
            "source_sha256": source_digest(), "source_files_sha256": source_file_digests(),
            "documents_sha256": {name: ev1.file_digest(R0_ROOT / name) for name in FROZEN_DOCUMENTS}, "plans": PLANS,
            "holdout_rule": "open HOLDOUT only if the primary passes every gate RG1-RG9 on VALIDATION at 95%"}


def acquisition_decision(cand: str, sc: dict, level: float = 0.999) -> dict:
    """DEV_PROTOCOL: VOI_R replaces PROBE_R only if on-policy ALL compositional regret is lower at 99.9% AND VOI_R's
    online regret meets the budget (RG9)."""
    if cand is None:
        return {"adopt_voi": False, "reason": "NO_PRIMARY"}
    paths = {k: EVIDENCE / "dev" / f"onpolicy_{cand}_{k}.json" for k in ("PROBE", "VOI")}
    if not all(p.exists() for p in paths.values()):
        return {"adopt_voi": False, "reason": "NO_VOI_RECORD"}
    rec = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    diff = stats.paired(series(rec["VOI"], "ALL", cand), series(rec["PROBE"], "ALL", cand), level)
    online = stats.mean_ci(series(rec["VOI"], "ALL", cand, "online"), level)
    better, within = diff["hi"] < 0, online["hi"] <= sc["online_budget"]
    return {"adopt_voi": bool(better and within), "voi_minus_probe_ALL": [diff["mean"], diff["lo"], diff["hi"]],
            "voi_online_ALL": [online["mean"], online["lo"], online["hi"]], "online_budget": sc["online_budget"],
            "better_at_999": bool(better), "online_within_budget": bool(within)}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("train", "dev", "onpolicy", "select", "freeze", "qualify", "disposition",
                                        "ood"))
    ap.add_argument("--phase", default="")
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
        out["acquisition"], out["acquisition_rule"] = "PROBE", acquisition_decision(out["primary"], sc)
        if out["acquisition_rule"].get("adopt_voi"):
            out["acquisition"] = "VOI"
        out = {"schema": "relational-lao-r0.selection.v1", **out, "scales": sc,
               "record_sha256": hashlib.sha256((EVIDENCE / "dev" / "round.json").read_bytes()).hexdigest()}
        (R0_ROOT / "SELECTION_R6.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(out["primary"], out["fallback"])
    elif a.command == "freeze":
        (R0_ROOT / "FROZEN_R6.json").write_text(json.dumps(freeze(), indent=1) + "\n", encoding="utf-8")
        print(source_digest())
    elif a.command == "qualify":
        frozen = json.loads((R0_ROOT / "FROZEN_R6.json").read_text(encoding="utf-8"))
        if frozen["source_sha256"] != source_digest():
            raise SystemExit("SOURCES_DIFFER_FROM_FROZEN_R6")
        out = EVIDENCE / a.phase.lower() / "qualification.json"
        if out.exists():
            raise SystemExit(f"{a.phase}_ALREADY_RUN")                # every partition is opened once
        if a.phase == "HOLDOUT":
            if not (EVIDENCE / "validation" / "qualification.json").exists():
                raise SystemExit("HOLDOUT_STAYS_SEALED no VALIDATION record")
            val = _load(EVIDENCE / "validation" / "qualification.json")
            parts = gate_parts(val, frozen["primary"]["name"], frozen["scales"], QUAL_LEVEL, val)
            if not all(parts.values()):
                raise SystemExit(f"HOLDOUT_STAYS_SEALED {[k for k, v in parts.items() if not v]}")
        if a.phase == "OOD" and not (EVIDENCE / "holdout" / "qualification.json").exists():
            raise SystemExit("OOD_AFTER_HOLDOUT_ONLY")
        plan = frozen["plans"][a.phase]
        rec = run(plan["partition"], range(*plan["seeds"]), plan["regimes"], plan["episodes"],
                  driver=frozen["primary"]["name"], policy=frozen["primary"]["acquisition"], subset=frozen["arms"])
        print(write(seal(rec, f"R6-{a.phase}"), out), rec["seconds"])
    elif a.command == "disposition":
        frozen = json.loads((R0_ROOT / "FROZEN_R6.json").read_text(encoding="utf-8"))
        prim, fb, sc = frozen["primary"]["name"], frozen["fallback"]["name"], frozen["scales"]
        phases, pparts, fparts = {}, {}, {}
        for ph in ("VALIDATION", "HOLDOUT"):
            path = EVIDENCE / ph.lower() / "qualification.json"
            if not path.exists():
                pparts[ph] = fparts[ph] = None
                continue
            rec = _load(path)
            pparts[ph] = gate_parts(rec, prim, sc, QUAL_LEVEL, rec)
            fparts[ph] = fallback_gates(rec, fb, QUAL_LEVEL)
            phases[ph] = {"primary": pparts[ph], "fallback": fparts[ph], "families": families(rec, prim),
                          "claims": claims(rec, prim), "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "source_sha256": rec["source_sha256"]}
        status, failing = primary_status(pparts)
        fstatus, ffailing = fallback_status(fparts)
        out = {"schema": "relational-lao-r0.disposition.v1", "level": QUAL_LEVEL, "family_level": FAMILY_LEVEL,
               "intervals": "[mean, lo, hi], paired Student-t over world seeds, two-sided",
               "primary": {"name": prim, "status": status, "failing": failing},
               "fallback": {"name": fb, "status": fstatus, "failing": ffailing},
               "holdout": "OPENED_ONCE" if pparts["HOLDOUT"] is not None else "SEALED",
               "frozen_source_sha256": frozen["source_sha256"], "phases": phases}
        (EVIDENCE / "DISPOSITION_R6.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(status, fstatus, failing)
    elif a.command == "ood":
        frozen = json.loads((R0_ROOT / "FROZEN_R6.json").read_text(encoding="utf-8"))
        path = EVIDENCE / "ood" / "qualification.json"
        rec, prim = _load(path), frozen["primary"]["name"]
        out = {"schema": "relational-lao-r0.ood.v1", "primary": prim, "rho": rho_admission(rec, prim),
               "characterisation": ood_characterisation(rec, prim),
               "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_sha256": rec["source_sha256"]}
        (EVIDENCE / "ood" / "characterization.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(out["rho"]["rho_admit"], out["rho"]["host_rho_admit"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
