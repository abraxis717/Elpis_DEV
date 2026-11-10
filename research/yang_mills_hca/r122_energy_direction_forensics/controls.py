"""R122 controls: detector validation that must pass before any prospective fixture is consumed.

* ``synthetic_classifier_controls``  constructed vectors with known geometry: aligned,
  antiparallel, orthogonal, exact zeros, tiny-but-aligned, tiny-but-inverted, subnormal,
  and gauge-inconsistent predictions (state rotated, target not).
* ``regression_fixture``  the consumed R121 ``R121_HOLDOUT_2`` fixture (published seeds and
  structure): the published cell means must be reproduced, the published unsigned R114
  inversion must be detected as REVERSED at k = 0 while NATIVE stays ALIGNED, and the
  nearly extinguished k = 6 predictions must *not* be reported as reversals.
* ``gauge_controls``  on real trajectories: joint relation-gauge transformation of frames and
  states leaves every metric invariant; a state-only transformation (frames untouched)
  does not; a prediction-only rotation is caught by the direction classes.

None of these read or produce prospective data.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import gauge as G118
from research.yang_mills_hca.r118_scalable_awr import provenance as P

from . import harness as H
from . import measure as MS
from . import metrics as M

YM = P.YM_ROOT
R121_PROTOCOL = YM / "r121_temporal_topology" / "PROTOCOL_R121.json"
R121_RESULT = YM / "r121_temporal_topology" / "evidence" / "R121_RESULT.json"
REGRESSION_GRAPH = "R121_HOLDOUT_2"
INVERSION_CELLS = ("54007/STATE8", "54007/STATE8_TF1")


def _block_rotation(theta1, theta2):
    c1, s1, c2, s2 = math.cos(theta1), math.sin(theta1), math.cos(theta2), math.sin(theta2)
    return np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, c2, -s2], [0, 0, s2, c2]])


def synthetic_classifier_controls(seed: int = 122600) -> dict:
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(64, 4))
    perp = rng.normal(size=(64, 4))
    perp -= (perp * t).sum(-1, keepdims=True) / (t * t).sum(-1, keepdims=True) * t
    perp *= np.sqrt((t * t).sum(-1, keepdims=True) / (perp * perp).sum(-1, keepdims=True))
    half_turn = _block_rotation(math.pi, math.pi)
    quarter = _block_rotation(math.pi / 2, math.pi / 2)
    g = _block_rotation(0.7, -1.9)
    zero = np.zeros_like(t)
    tiny = t * 2.0 ** -600
    cases = {
        # name: (prediction, target, expected norm class, expected direction class, expected E_direction or None)
        "identical": (t, t, "ORDINARY", "ALIGNED", 0.0),
        "antiparallel": (-t, t, "ORDINARY", "REVERSED", 4.0),
        "orthogonal": (perp, t, "ORDINARY", "ORTHOGONAL", 2.0),
        "aligned_half_amplitude": (0.5 * t, t, "ORDINARY", "ALIGNED", 0.0),
        "small_inverted_5_percent": (-0.05 * t, t, "ORDINARY", "REVERSED", 4.0),
        "aligned_but_extinguished": (1e-3 * t, t, "ORDINARY", "EXTINGUISHED", 0.0),
        "inverted_but_extinguished": (-1e-3 * t, t, "ORDINARY", "EXTINGUISHED", 4.0),
        "target_negligible": (1e3 * t, t, "ORDINARY", "TARGET_NEGLIGIBLE", 0.0),
        "zero_prediction": (zero, t, "ZERO_PREDICTION", "UNDEFINED", None),
        "zero_target": (t, zero, "ZERO_TARGET", "UNDEFINED", None),
        "both_zero": (zero, zero, "BOTH_ZERO", "UNDEFINED", None),
        "subnormal_range_pair": (tiny, tiny, "NEAR_ZERO", "UNDEFINED", 0.0),
        "joint_gauge_transform": (t @ g.T, t @ g.T, "ORDINARY", "ALIGNED", 0.0),
        "prediction_only_half_turn": (t @ half_turn.T, t, "ORDINARY", "REVERSED", 4.0),
        "prediction_only_quarter_turn": (t @ quarter.T, t, "ORDINARY", "ORTHOGONAL", 2.0),
    }
    out, ok = {}, True
    worst_res = 0.0
    for name, (p, tt, nclass, dclass, edir) in cases.items():
        d = M.decompose(p, tt)
        good = bool(np.all(d["norm_class"] == M.NORM_CLASSES.index(nclass)) and np.all(d["dir_class"] == M.DIR_CLASSES.index(dclass)))
        if edir is not None:
            good = good and bool(np.nanmax(np.abs(d["edir"] - edir)) < 1e-9)
        for key in ("res_cos", "res_proj", "res_norm"):
            if np.isfinite(d[key]).any():
                worst_res = max(worst_res, float(np.nanmax(d[key])))
        out[name] = {"expected_norm_class": nclass, "expected_direction_class": dclass, "pass": good}
        ok = ok and good
    # zero-prediction: E_rel is exactly 1 and never called a reversal; zero-target: no E_rel is produced
    d = M.decompose(zero, t)
    ok = ok and bool(np.all(d["erel"] == 1.0)) and not bool(d["antiparallel"].any())
    ok = ok and bool(np.all(np.isnan(M.decompose(t, zero)["erel"])))
    # conditioning of a paired difference is flagged, not silently used
    a = rng.normal(size=(8, 4))
    b = a * (1.0 + 2.0 ** -40)
    dcond = M.decompose(a - b, t[:8], pair_condition=M.pair_condition(a, b))
    ok = ok and bool(np.all(dcond["norm_class"] == M.N_ILL))
    # exact rational audit of the float sums
    audit = max(M.exact_audit(p_, t_)[0] for p_, t_ in zip((t[:8] * 1.3 + perp[:8] * 0.2), t[:8]))
    identity_zero = all(M.exact_audit(p_, t_)[1] == 0 for p_, t_ in zip(perp[:8], t[:8]))
    ok = ok and audit <= 1e-12 and identity_zero and worst_res <= 1e-12
    return {"pass": bool(ok), "cases": out, "max_identity_residual": worst_res, "exact_audit_max_relative": audit,
            "exact_identity_is_zero": bool(identity_zero), "seed": seed}


def regression_config():
    proto = json.loads(R121_PROTOCOL.read_bytes())
    names = sorted(proto["graphs"])
    gi = names.index(REGRESSION_GRAPH)
    return {"graphs": {REGRESSION_GRAPH: proto["graphs"][REGRESSION_GRAPH]}, "structure_seed": proto["structure_seed"] + gi,
            "evaluation_seeds": proto["evaluation_seeds"], "cases": proto["cases"], "k_values": proto["k_values"],
            "scales": {"1": 1.0}, "ratio": 160, "batch": 4, "deep_check_stride": 4, "reference_teacher_steps": [1, 2, 8, 20]}


def regression_fixture(oracle, tolerances, progress=print) -> dict:
    """Known-result diagnostic on consumed evidence only (R121 seeds, structure and graph)."""
    cfg = regression_config()
    published = json.loads(R121_RESULT.read_bytes())["by_cell"]
    raw, info = MS.run(cfg, oracle, progress=progress)
    R = H.VIEWS.index("READOUT")
    worst, compared, exact = 0.0, 0, 0
    fails = []
    for ki, k in enumerate(cfg["k_values"]):
        for ci, ck in enumerate(P.CHECKPOINTS):
            for ti, pol in enumerate(("NATIVE", "R114")):
                for h in (8, 20):
                    b = raw["AUTO"][0, ki, 0, :, ci, ti, h - 1, R]          # [F, N, 5]
                    tt, pp, pt, ee = (b[..., i].reshape(-1) for i in range(4))
                    mine = {"nse": float(np.mean(ee / tt)),
                            "matched_nse": float(np.mean(2.0 * (1.0 - pt / (np.sqrt(pp * tt) + 1e-30)))),
                            "energy_ratio": float(np.mean(pp / tt))}
                    ref = published[f"{REGRESSION_GRAPH}/k{k}/{pol}/{ck}/H{h}"]
                    for key, val in mine.items():
                        dev = abs(val - ref[key])
                        compared += 1
                        exact += int(val == ref[key])
                        worst = max(worst, dev / max(abs(ref[key]), 1e-300))
                        if dev > tolerances["published_cell_relative"] * abs(ref[key]) + tolerances["published_cell_absolute"]:
                            fails.append(f"k{k}/{pol}/{ck}/H{h}/{key}")
    # detector: the published inversion at k = 0 and the non-conflation of extinguished predictions at k = 6
    det = {}
    ok = not fails
    for ck in INVERSION_CELLS:
        ci = P.CHECKPOINTS.index(ck)
        for ki, k in enumerate(cfg["k_values"]):
            for ti, pol in enumerate(("NATIVE", "R114")):
                d = M.from_base(raw["AUTO"][0, ki, 0, :, ci, ti, H.HMAX - 1, R], raw["PCOND"][0, ki, 0, :, ci, ti, H.HMAX - 1, 0])
                det[f"k{k}/{pol}/{ck}/H20"] = {
                    "classes": {name: int((d["dir_class"] == i).sum()) for i, name in enumerate(M.DIR_CLASSES) if (d["dir_class"] == i).any()},
                    "c_min": float(d["c"].min()), "c_max": float(d["c"].max()), "r_median": float(np.median(d["r"])),
                    "edir_mean": float(d["edir"].mean()), "antiparallel": int(d["antiparallel"].sum()), "n": int(d["c"].size)}
        k0 = det[f"k0/R114/{ck}/H20"]
        n0 = det[f"k0/NATIVE/{ck}/H20"]
        k6 = det[f"k6/R114/{ck}/H20"]
        ok = ok and k0["classes"] == {"REVERSED": k0["n"]} and k0["antiparallel"] == k0["n"] and k0["edir_mean"] >= 3.99
        ok = ok and n0["classes"] == {"ALIGNED": n0["n"]} and n0["c_min"] >= 0.999
        ok = ok and k6["classes"].get("REVERSED", 0) == 0 and k6["classes"].get("EXTINGUISHED", 0) == k6["n"]
    c = info["counters"]
    integrity = (c["nonfinite"] == 0 and c["autonomous_recheck_mismatch"] == 0 and c["delayed_spliced_vs_full_mismatch"] == 0
                 and c["delayed_h8_vs_native_mismatch"] == 0 and c["fresh_state_mismatch"] == 0 and c["exact_r118_mismatch"] == 0
                 and info["maxima"]["teacher_k0_vs_historical_max_abs"] <= tolerances["teacher_k0_parity_abs"]
                 and info["maxima"]["reference_teacher_max_rel"] <= tolerances["reference_teacher_relative"])
    return {"pass": bool(ok and integrity), "source": "CONSUMED_R121_EVIDENCE_ONLY", "graph": REGRESSION_GRAPH,
            "r121_result_sha256": hashlib.sha256(R121_RESULT.read_bytes()).hexdigest(),
            "published_cells_compared": compared, "published_cells_bit_exact": exact,
            "published_cells_worst_relative_deviation": worst, "published_cells_out_of_tolerance": fails,
            "detector": det, "measurement": info}


def gauge_controls(oracle, S, parts, X0, T, checkpoints, rng, tolerances) -> dict:
    """Joint-gauge invariance, state-only negative control and inconsistent-gauge detection on real trajectories."""
    Sg, g_rows = H.gauged_system(S, parts, S.k, rng)
    Xg0 = H.gauge_state(X0, g_rows)
    Tg = Sg.teacher(oracle, Xg0)
    cov = float(np.abs(Tg - H.gauge_state(T, g_rows)).max())
    inv = 0.0
    neg = np.inf
    detect_total = detect_rev = detect_orth = 0
    half = G118.rotation_blocks(np.array([math.pi]), np.array([math.pi]))[0]
    quarter = G118.rotation_blocks(np.array([math.pi / 2]), np.array([math.pi / 2]))[0]
    for ck in checkpoints:
        for pol in ("NATIVE", "R114"):
            X = S.rollout(X0, ck, pol)
            Xg = Sg.rollout(Xg0, ck, pol)
            cov = max(cov, float(np.abs(Xg - H.gauge_state(X, g_rows)).max()))
            Xbad = S.rollout(Xg0, ck, pol)                 # states transformed, frames not
            neg = min(neg, float(np.abs(Xbad[1:] - H.gauge_state(X[1:], g_rows)).max()))
            for name in H.VIEWS:
                a = M.decompose(S.view(X[1:], name), S.view(T[1:], name))
                b = M.decompose(Sg.view(Xg[1:], name), Sg.view(Tg[1:], name))
                for key in ("erel", "r", "c"):
                    both = np.isfinite(a[key]) & np.isfinite(b[key])
                    if both.any():
                        inv = max(inv, float(np.abs(a[key][both] - b[key][both]).max() / (1.0 + np.abs(a[key][both]).max())))
            # inconsistent gauge: rotate the predicted readout only
            p, t = S.view(X[1:], "READOUT"), S.view(T[1:], "READOUT")
            base = M.decompose(p, t)
            sel = base["dir_class"] == M.D_ALIGNED
            rev = M.decompose(G118.bapply(half, p), t)
            orth = M.decompose(G118.bapply(quarter, p), t)
            detect_total += int(sel.sum())
            detect_rev += int((rev["dir_class"][sel] == M.D_REVERSED).sum())
            detect_orth += int((orth["dir_class"][sel] == M.D_ORTHO).sum())
    ok = (cov <= tolerances["gauge_covariance_abs"] and inv <= tolerances["gauge_metric_invariance"]
          and neg >= tolerances["gauge_negative_control_min"] and detect_total > 0
          and detect_rev == detect_total and detect_orth == detect_total)
    return {"pass": bool(ok), "joint_covariance_max_abs": cov, "metric_invariance_max": inv,
            "state_only_negative_control_min": float(neg), "inconsistent_gauge_samples": detect_total,
            "half_turn_detected_as_reversed": detect_rev, "quarter_turn_detected_as_orthogonal": detect_orth}
