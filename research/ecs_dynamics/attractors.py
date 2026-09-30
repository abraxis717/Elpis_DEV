"""Conservative identification of sampled attractors. RESEARCH_ONLY.

Only two kinds of object are identified: fixed points and periodic orbits
that repeat within tolerance across several cycles (see
:func:`diagnostics.classify_tail`). Trajectories that are neither are counted
as unresolved and are never merged into an attractor. The result is a
*sampled* attractor inventory for the initial conditions tried. It is not a
census of all attractors of the system.
"""
from __future__ import annotations

import numpy as np

from .diagnostics import classify_tail
from .systems import RecurrentSystem, perturb, rollout, run


def _cycle(tail: np.ndarray, period: int) -> np.ndarray:
    return tail[-period:].copy()


def _same_cycle(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if a.shape != b.shape:
        return False
    p = a.shape[0]
    return any(float(np.max(np.abs(a - np.roll(b, s, axis=0)))) <= tol for s in range(p))


def _canonical(cycle: np.ndarray) -> np.ndarray:
    """Phase-independent representative: the cycle state with the lexicographically smallest rounded tuple."""
    keys = [tuple(np.round(c, 6)) for c in cycle]
    return cycle[int(min(range(len(keys)), key=lambda i: keys[i]))].copy()


class AttractorInventory:
    """Sampled attractors of one system: types, periods, representatives, basin counts, recovery."""

    def __init__(self, *, warmup: int, tail: int, fixed_tol: float, recurrence_tol: float, max_period: int,
                 min_cycles: int = 3, match_tol: float = 1e-6):
        self.warmup, self.tail = warmup, tail
        self.fixed_tol, self.recurrence_tol = fixed_tol, recurrence_tol
        self.max_period, self.min_cycles, self.match_tol = max_period, min_cycles, match_tol
        self.attractors: list[dict] = []
        self.unresolved = 0
        self.unbounded = 0

    def classify(self, system: RecurrentSystem, x0: np.ndarray) -> tuple[dict, np.ndarray]:
        x = run(system, x0, self.warmup)
        tail = rollout(system, x, self.tail - 1)
        return classify_tail(tail, fixed_tol=self.fixed_tol, recurrence_tol=self.recurrence_tol,
                             max_period=self.max_period, min_cycles=self.min_cycles), tail

    def match(self, info: dict, tail: np.ndarray) -> int | None:
        if info["class"] == "FIXED_POINT":
            rep = tail[-1]
            for k, a in enumerate(self.attractors):
                if a["type"] == "FIXED_POINT" and float(np.max(np.abs(a["_cycle"][0] - rep))) <= self.match_tol:
                    return k
            return None
        if info["class"] == "PERIODIC":
            cyc = _cycle(tail, info["period"])
            for k, a in enumerate(self.attractors):
                if a["type"] == "PERIODIC" and a["period"] == info["period"] and _same_cycle(a["_cycle"], cyc,
                                                                                               self.match_tol):
                    return k
        return None

    def label(self, system: RecurrentSystem, x0: np.ndarray, register: bool = True) -> int:
        """Attractor index, or -1 for unresolved and -2 for unbounded trajectories."""
        info, tail = self.classify(system, x0)
        if info["class"] == "UNRESOLVED_NONPERIODIC":
            if register:
                self.unresolved += 1
            return -1
        if info["class"] == "UNBOUNDED":
            if register:
                self.unbounded += 1
            return -2
        k = self.match(info, tail)
        if k is None:
            if not register:
                return -3  # a resolved attractor not in the inventory
            cyc = tail[-1:].copy() if info["class"] == "FIXED_POINT" else _cycle(tail, info["period"])
            self.attractors.append({"type": info["class"], "period": info["period"], "_cycle": cyc,
                                    "representative": _canonical(cyc), "basin_count": 0})
            k = len(self.attractors) - 1
        if register:
            self.attractors[k]["basin_count"] += 1
        return k

    def add_recovery(self, system: RecurrentSystem, rng: np.random.Generator, magnitude: float) -> None:
        """Perturb each representative and record whether the trajectory returns to the same attractor."""
        for k, a in enumerate(self.attractors):
            lab = self.label(system, perturb(a["representative"], rng, magnitude), register=False)
            a["recovery"] = "RETURNS" if lab == k else ("UNRESOLVED" if lab == -1 else "LEAVES")

    def summary(self, samples: int) -> dict:
        resolved = sum(a["basin_count"] for a in self.attractors)
        basins = sorted((a["basin_count"] for a in self.attractors), reverse=True)
        return {
            "samples": samples,
            "attractor_count_proxy": len(self.attractors),
            "fixed_points": sum(1 for a in self.attractors if a["type"] == "FIXED_POINT"),
            "periodic_orbits": sum(1 for a in self.attractors if a["type"] == "PERIODIC"),
            "periods": sorted({a["period"] for a in self.attractors if a["type"] == "PERIODIC"}),
            "resolved_fraction": resolved / samples if samples else 0.0,
            "unresolved_fraction": self.unresolved / samples if samples else 0.0,
            "unbounded_fraction": self.unbounded / samples if samples else 0.0,
            "largest_basin_fraction": basins[0] / samples if basins and samples else 0.0,
            "basin_occupancy": [b / samples for b in basins],
            "attractors": [{"type": a["type"], "period": a["period"], "basin_count": a["basin_count"],
                            "recovery": a.get("recovery"),
                            "representative_norm": float(np.linalg.norm(a["representative"])),
                            "representative_head": [float(v) for v in a["representative"][:4]]}
                           for a in self.attractors],
        }
