"""ERSA HARNESS R0 — replays frozen fixtures against any backend and scores
independent metrics.  Expectations come only from oracle_ersa.

Metric contract (restated in every report):
  exact_match      backend selection == dense oracle selection
  candidate_recall |backend candidates ∩ required| / |required| over the live set
  required_recall  required evidence present in the committed selection
  forbidden_leak   forbidden/authority-denied evidence reached a committed result
  directional_fidelity  1 - cos(backend aggregate, oracle aggregate)
  gauge_covariance max |agg' - rho(G) agg| over the fixture's gauges
"""
from __future__ import annotations
import math
import numpy as np
from . import contract_ersa as C
from . import geometry_ersa as G
from . import oracle_ersa as O
from . import fixtures_ersa as F

TOL = 1e-9


def replay(backend, scenario, gauge=None):
    """Replay the frozen op-trace into a backend. Returns backend."""
    auth = dict(scenario.get("authority") or {})
    if gauge:
        auth = {**auth, "gauge_relation": gauge[0],
                "gauge_participant": gauge[1]}
    b = backend()
    b.begin_session(scenario["spec"], scenario["records"],
                    _transport(scenario, gauge), auth)
    for ins in scenario["inserts"]:
        b.insert_relation(**ins)
    for act in scenario.get("extra", ()):
        if act[0] == "propose_commit":
            stage = b.propose_update(act[1], act[2], act[3])
            b.commit_validated_update(stage)
    if scenario.get("advance"):
        b.step(scenario["advance"])
    return b


def _transport(scenario, gauge=None):
    auth = dict(scenario.get("authority") or {})
    gr = gp = None
    if gauge:
        gr, gp = gauge
    pa = auth.get("participant_angles") or {}
    if not pa:
        return None
    return G.ReferenceTransport(participant_angles=pa,
                                gauge_relation=gr or (0.0, 0.0),
                                gauge_participant=gp)


def _oracle(scenario, gauge=None):
    auth = dict(scenario.get("authority") or {})
    if gauge:
        auth = {**auth, "gauge_relation": gauge[0],
                "gauge_participant": gauge[1]}
    return O.NaiveOracle(scenario["spec"], scenario["records"],
                         _transport(scenario, gauge), auth)


def _replay_oracle(scenario, gauge=None):
    orc = _oracle(scenario, gauge)
    for ins in scenario["inserts"]:
        orc.insert(**ins)
    for act in scenario.get("extra", ()):
        if act[0] == "propose_commit":
            rid, delta, gain = act[1], act[2], act[3]
            if rid in orc.state:
                old = orc.state[rid]["value"]
                orc.state[rid]["value"] = tuple(
                    old[i] + float(gain) * delta[i] for i in range(4))
    if scenario.get("advance"):
        orc.step(scenario["advance"])
    return orc


def evaluate_backend(backend_cls, scenario, name):
    """Return per-family metric dict for one backend on one frozen family."""
    b = replay(backend_cls, scenario)
    orc = _replay_oracle(scenario)
    if scenario.get("teardown"):
        b.reset()
        orc.reset()
    rows = []
    cap_test = bool(scenario.get("capacity_test"))
    must_retain = set(scenario.get("must_retain") or [])
    for q in scenario["queries"]:
        rec = b.query_candidates(q["qkey"], query_goal=q.get("qgoal"))
        srec = b.select(q["qkey"], query_goal=q.get("qgoal"), k=q["k"],
                        qdir=q.get("qdir"))
        arec = b.aggregate(q["qkey"], query_goal=q.get("qgoal"), k=q["k"],
                           qdir=q.get("qdir"))
        got = _ids(getattr(b, "_last_selection", []))
        exp_sel = orc.expected_selection(q["qkey"], q.get("qgoal"), q["k"],
                                         q.get("qdir"))
        exp_ids = [rid for rid, _ in exp_sel]
        exp_agg, _ = orc.expected_aggregate(q["qkey"], q.get("qgoal"),
                                            q["k"], q.get("qdir"))
        got_agg = tuple(getattr(b, "_last_aggregate", (0.0,) * 4))
        live = sorted(orc.state)
        cands = set(getattr(b, "_last_candidates", []))
        required = q.get("required") or []
        forbidden = set(q.get("forbidden") or [])
        ev_ids = {e["relation_id"] for e in _evidence(b)}
        live_rids = set(int(x) for x in _snapshot_rids(b))
        rows.append({
            "query": q.get("note", ""),
            "exact_match": (None if cap_test else got == exp_ids),
            "top1_correct": (bool(got and required and got[0] == required[0])
                             if required else (got == exp_ids[:1])),
            "must_retain_ok": (must_retain <= live_rids if must_retain else True),
            "required_recall": (sum(1 for r in required if r in got) / len(required)
                                if required else 1.0),
            "forbidden_leak": bool(forbidden & (set(got) | ev_ids)),
            "directional_fidelity": (0.0 if _norm(exp_agg) < 1e-12 else
                                     G.directional_fidelity(got_agg, exp_agg)),
            "candidate_recall": (sum(1 for r in required if r in cands) / len(required)
                                 if required else 1.0),
            "candidate_count": len(cands),
            "live": len(live),
            "authority_refused": _refused(b),
        })
    gauge_cov = _gauge_covariance(backend_cls, scenario)
    det = _determinism(backend_cls, scenario)
    out = {
        "queries": rows,
        "exact_match": all(r["exact_match"] for r in rows
                           if r["exact_match"] is not None),
        "top1_correct": all(r["top1_correct"] for r in rows),
        "must_retain_ok": all(r["must_retain_ok"] for r in rows),
        "required_recall": min(r["required_recall"] for r in rows),
        "forbidden_leak": any(r["forbidden_leak"] for r in rows),
        "directional_fidelity": max(r["directional_fidelity"] for r in rows),
        "candidate_recall": min(r["candidate_recall"] for r in rows),
        "gauge_covariance": gauge_cov,
        "determinism": det,
    }
    return out


def _ids(sel):
    out = []
    for item in sel:
        out.append(int(item[0]) if isinstance(item, tuple) else int(item))
    return out


def _evidence(bank):
    if hasattr(bank, "_last_evidence"):
        return bank._last_evidence
    return []


def _norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def _refused(bank):
    occ = bank.occupancy()
    return int(occ.get("rejections", 0))


def _snapshot_rids(bank):
    try:
        return list(bank.snapshot().get("relations", {}).keys())
    except Exception:
        return []


def _gauge_covariance(backend_cls, scenario):
    """Full-operator covariance: operator(g·input) = g·operator(input).

    The query's canonical-frame direction lives IN the canonical frame, so it
    transforms with G_r as well; key-space features (qcompat/goalrel) are
    gauge-invariant and are left untouched.
    """
    g = scenario.get("gauge")
    if not g:
        return 0.0
    gam, gp = g
    base = replay(backend_cls, scenario)
    q = scenario["queries"][0]
    agg0 = _run_agg(base, q)
    g2 = replay(backend_cls, scenario, gauge=(gam, gp))
    q2 = dict(q)
    if q.get("qdir"):
        q2["qdir"] = G.rho_apply(gam, tuple(q["qdir"]))
    agg1 = _run_agg(g2, q2)
    expected = G.rho_apply(gam, agg0)
    return G.max_abs_diff(agg1, expected)


def _run_agg(bank, q):
    bank.aggregate(q["qkey"], query_goal=q.get("qgoal"), k=q["k"],
                   qdir=q.get("qdir"))
    return tuple(getattr(bank, "_last_aggregate", (0.0,) * 4))


def _determinism(backend_cls, scenario):
    a = _run_agg(replay(backend_cls, scenario), scenario["queries"][0])
    b = _run_agg(replay(backend_cls, scenario), scenario["queries"][0])
    return max(abs(x - y) for x, y in zip(a, b)) == 0.0


def _evidence_hook(backend_cls):
    """ErsBank stores evidence in _last_evidence; ensure aggregate() sets it."""
    return True


def summary(results):
    fams = sorted(results)
    out = {}
    keys = ("exact_match", "top1_correct", "must_retain_ok",
            "required_recall", "forbidden_leak",
            "directional_fidelity", "candidate_recall",
            "gauge_covariance", "determinism")
    for fam in fams:
        r = results[fam]
        if "error" in r:
            out[fam] = {"error": r["error"]}
        else:
            out[fam] = {k: r.get(k) for k in keys}
    return out
