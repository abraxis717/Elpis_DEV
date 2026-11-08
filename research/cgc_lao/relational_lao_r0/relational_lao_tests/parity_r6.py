"""R6 native and host parity (PREREGISTRATION_R6.md, run only after a qualifying disposition).

Through the research host over the real ECS_C relational view:
- the native C11 adapter receives every learner operation the scalar reference receives (update, rebuild, drop, slot
  reset, predictive, every alternative's expected utility) and must agree bitwise after each one;
- the host must reproduce the vectorised frozen arm's relation selections and terminal decisions, except documented
  near-ties (a gap under 1e-9 at the decision), for the primary (Contract I) and the fallback (Contract II).

    PYTHONPATH=src:. ELPIS_NATIVE_BUILD=<build> python3 -m relational_lao_tests.parity_r6 --phase VALIDATION
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

from relational_lao_tests import _native
import relational_lao  # noqa: F401  (paths)
from cgc_lao_v2.providers import MechanismRegistry, ProviderDeclaration, slb2_digest
from relational_lao import R0_ROOT, harness, host as H, hostrun, learners as Lr, study, world as W
from relational_lao.scalar import ScalarRelational

NEAR_TIE = 1e-9
SAMPLE = {"VALIDATION": ("VALIDATION", range(8), 256), "HOLDOUT": ("HOLDOUT", range(8), 256)}
REGISTRY = MechanismRegistry.slb2()
DUPLICATE = ProviderDeclaration("p", "INDEPENDENT_REREAD", 2, slb2_digest("slb2/independent-reread"), None)
FIRST_ONLY = ProviderDeclaration("p", "INDEPENDENT_REREAD", 1, slb2_digest("slb2/independent-reread"), None)


class ParityError(AssertionError):
    pass


class DualRelational:
    """The scalar reference with the native adapter mirrored call for call. Any difference raises ParityError."""
    operations = 0

    def __init__(self, hypotheses, prior, slots=8, bias_var=1.0, readout="coupled"):
        self.s = ScalarRelational(hypotheses, prior, slots=slots, bias_var=bias_var, readout=readout)
        self.n = _native.NativeRelational(hypotheses, prior, slots=slots, bias_var=bias_var, readout=readout)
        self.ids: dict = {}
        self._same("reset")

    @property
    def buffer(self):
        return self.s.buffer

    @property
    def counts(self):
        return self.s.counts

    @property
    def M(self):
        return self.s.M

    def _same(self, op: str) -> None:
        DualRelational.operations += 1
        if self.s.state() != self.n.state() or len(self.s.buffer) != self.n.buffered():
            raise ParityError(f"state after {op}")
        if self.s.weights() != self.n.weights():
            raise ParityError(f"weights after {op}")

    def update(self, r, z, reads, step=None, binding=None, record=True):
        if not record:
            raise ParityError("unrecorded update")
        self.s.update(r, z, reads, step, binding)
        self.n.update(r, z, reads, step, self.ids.setdefault(binding, len(self.ids)))
        self._same("update")

    def rebuild(self, apply):
        keep = sum(1 << i for i, e in enumerate(self.s.buffer) if apply(e))
        self.s.rebuild(apply)
        self.n.rebuild(keep)
        self._same("rebuild")

    def drop(self, predicate):
        mask = sum(1 << i for i, e in enumerate(self.s.buffer) if predicate(e))
        self.s.drop(predicate)
        self.n.drop(mask)
        self._same("drop")

    def reset_slot(self, r):
        self.s.reset_slot(r)
        self.n.reset_slot(r)
        self._same("reset_slot")

    def predictive(self, z_row):
        a = self.s.predictive(z_row)
        if a != self.n.predictive(z_row):
            raise ParityError("predictive")
        DualRelational.operations += 1
        return a

    def expected_utility(self, alternative, z_row):
        a = self.s.expected_utility(alternative, z_row)
        if a != self.n.expected_utility(alternative, z_row):
            raise ParityError("expected_utility")
        DualRelational.operations += 1
        return a

    def decide(self, alternatives, stop, z_row):
        for alt in alternatives:
            self.expected_utility(alt, z_row)
        return self.s.decide(alternatives, stop, z_row)

    def zeroize(self):
        self.s.zeroize()
        self.n.zeroize()


def _margins(eu: np.ndarray, stop: float) -> float:
    """Distance of a vectorised decision from its nearest tie (best vs second alternative, best vs STOP)."""
    v = np.sort(eu[np.isfinite(eu)])[::-1]
    gaps = [abs(v[0] - stop)] + ([v[0] - v[1]] if len(v) > 1 else [])
    return float(min(gaps))


def _compare(kind, b, e, got, chosen, dec, eu, stops, rel, out):
    out["episodes"] += 1
    if list(chosen) != list(rel[e]):
        out["selection_mismatches"].append([b.regime.name, b.seed, e])
    for j, (g, v) in enumerate(zip(got, dec[e])):
        if g != int(v):
            m = _margins(eu[e, j], float(stops[j]))
            out["near_ties" if m < NEAR_TIE else "decision_mismatches"].append([b.regime.name, b.seed, e, j, m])


def run(frozen: dict, phase: str) -> dict:
    partition, seeds, n = SAMPLE[phase]
    cfg = H.config_from_freeze(frozen)
    prim, fb = frozen["primary"]["name"], frozen["fallback"]["name"]
    if frozen["primary"]["acquisition"] != "PROBE":
        raise SystemExit("the host implements PROBE_R only")
    saved, H.ScalarRelational = H.ScalarRelational, DualRelational
    res = {k: {"episodes": 0, "selection_mismatches": [], "decision_mismatches": [], "near_ties": []}
           for k in ("PRIMARY", "FALLBACK")}
    t0 = time.time()
    try:
        for regime in study.ROUND_REGIMES:
            for seed in seeds:
                b = W.episodes(partition, seed, regime, 0, n)
                tpl, struct = b.template, b.query_struct
                _, _, st = W.query_tensors(tpl)
                stops, zq = st[struct], b.logits(b.xq)
                arms = study.arms_for(regime, prim, [prim])
                harness.rollout(b, arms, prim, "PROBE")
                eu = Lr.expected_utilities(arms[prim], zq, tpl, struct)
                dec, rel = Lr.decide(eu, stops), arms[prim].base.log.rel
                hyp, prior = study.FALLBACKS[fb]
                fa = Lr.Bayes(fb, hyp, prior, "coupled", "replay", first_only=True)
                harness.rollout(b, {fb: fa}, fb, "PROBE")
                feu = Lr.expected_utilities(fa, zq, tpl, struct)
                fdec, frel = Lr.decide(feu, stops), fa.log.rel
                for e in range(b.count):
                    for kind, decl, d_, eu_, rel_ in (("PRIMARY", DUPLICATE, dec, eu, rel),
                                                      ("FALLBACK", FIRST_ONLY, fdec, feu, frel)):
                        h = H.RelationalHost(REGISTRY, 0.0, cfg)
                        adm = h.initialize(decl)
                        if adm.configuration != kind:
                            raise ParityError(f"admission {adm}")
                        got, chosen = hostrun.run_episode(h, b, e)
                        _compare(kind, b, e, got, chosen, d_, eu_, stops, rel_, res[kind])
                        h.close_episode()
            print(regime, f"{time.time() - t0:.0f}s", {k: (v["episodes"], len(v["selection_mismatches"]),
                                                        len(v["decision_mismatches"]), len(v["near_ties"]))
                                                    for k, v in res.items()}, flush=True)
    finally:
        H.ScalarRelational = saved
    native = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((R0_ROOT / "native").rglob("*")) if p.is_file()}
    ok = all(not v["selection_mismatches"] and not v["decision_mismatches"] for v in res.values())
    return {"schema": "relational-lao-r0.parity.v1", "phase": phase, "partition": partition, "seeds": list(seeds),
            "episodes_per_seed": n, "regimes": study.ROUND_REGIMES, "near_tie": NEAR_TIE,
            "native_bitwise": True, "native_operations_compared": DualRelational.operations,
            "host_matches_vectorised": ok, "results": res, "frozen_source_sha256": frozen["source_sha256"],
            "sources_match_frozen": study.source_digest() == frozen["source_sha256"],
            "native_sources_sha256": native, "seconds": round(time.time() - t0, 1)}


def main(argv) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(SAMPLE))
    a = ap.parse_args(argv)
    frozen = json.loads((R0_ROOT / "FROZEN_R6.json").read_text(encoding="utf-8"))
    out = run(frozen, a.phase)
    path = R0_ROOT / "evidence" / a.phase.lower() / "parity.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(hashlib.sha256(path.read_bytes()).hexdigest(), out["host_matches_vectorised"], out["seconds"])


if __name__ == "__main__":
    main(sys.argv[1:])
