"""R122 registered analysis: raw measurement tensors -> decompositions, classes, decisions.

Pure numpy; no model execution. Every reported quantity is derived from the base sums
written by ``measure.py``. Thresholds come from the protocol (``thr``); nothing is fitted.

Exact identities used (each is checked and its residual reported):

  I1  E_rel = 1 + r^2 - 2 r c = E_gain + E_perp = E_mag + E_ang             (per sample)
  I2  e_h = P_h + L_h  =>  |e|^2 = |P|^2 + |L|^2 + 2 <P, L>                 (one-step telescoping)
  I3  e_20 = sum_b D_b  =>  sum_b <D_b, e_20> = |e_20|^2                    (restart telescoping)
  I4  log E_rel(h') - log E_rel(h) = [log E_abs(h') - log E_abs(h)] + [log T(h) - log T(h')]
  I5  E_rel(READOUT) = E_rel(PAIRED_STATE) * A_den                          (definition of A_den)

Stratum = (graph, overlap k, amplitude scale, checkpoint cohort); its samples are the
case x seed x checkpoint x pair samples. Strata of one graph are correlated; replication
claims are made over graph classes only.

Mechanism label of a stratum (registered decision tree, READOUT view, one horizon):

  INVALID                   fewer than ``min_eligible_fraction`` ordinary samples
  DENOMINATOR_COLLAPSE      median A_den >= ``a_den_collapse``: the readout NSE exceeds the
                            NSE of the whole paired fixture state by that factor, i.e. the
                            readout target has lost energy relative to the state it is read from
  LOCAL_TRANSITION          median lambda >= ``lambda_local``,  lambda = |L|^2 / |e|^2: one
                            teacher-forced step already produces that share of the error energy
  ACCUMULATED_DRIFT         median lambda <= ``lambda_accumulated``: the autonomous error
                            energy is at least 1/lambda_accumulated times a single local defect
  MIXED_LOCAL_ACCUMULATED   otherwise
"""
from __future__ import annotations

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import provenance as P

from . import harness as H
from . import measure as MS
from . import metrics as M

COHORTS = ("SIGNED", "UNSIGNED")
MECHANISMS = ("DENOMINATOR_COLLAPSE", "LOCAL_TRANSITION", "ACCUMULATED_DRIFT", "MIXED_LOCAL_ACCUMULATED", "INVALID")
LOCALIZING = ("DENOMINATOR_COLLAPSE", "LOCAL_TRANSITION", "ACCUMULATED_DRIFT")
GROWTH = ("NO_GROWTH", "NUMERATOR_DRIVEN", "DENOMINATOR_DRIVEN")
VI = {name: i for i, name in enumerate(H.VIEWS)}
STRATUM_FIELDS = (
    "n", "eligible_fraction", "erel_mean", "erel_median", "erel_ratio_of_sums", "eabs_sum", "teacher_energy_sum",
    "r_median", "c_median", "c_min", "rho_negative_fraction", "egain_mean", "eperp_mean", "perp_share_median",
    "state_erel_median", "state_c_median", "state_r_median", "state_perp_share_median", "a_den_median",
    "local_erel_median", "local_erel_mean", "persistence_erel_median", "local_skill_median", "lambda_local_median",
    "local_projection_share_median", "cancellation_index_median", "reversed", "extinguished")


def cohort_indices():
    return {"SIGNED": [i for i, ck in enumerate(P.CHECKPOINTS) if H.is_signed(ck)],
            "UNSIGNED": [i for i, ck in enumerate(P.CHECKPOINTS) if not H.is_signed(ck)]}


def traj_index(policy):
    return [H.TRAJECTORIES.index(H.trajectory_of(ck, policy)) for ck in P.CHECKPOINTS]


def by_policy(arr, policy):
    """[G,K,S,F,C,TR,...] -> [G,K,S,F,C,...]: each checkpoint's trajectory for ``policy``."""
    return np.stack([arr[:, :, :, :, ci, ti] for ci, ti in enumerate(traj_index(policy))], axis=4)


def _f(x, digits: int = 10):
    """JSON-safe float (None for non-finite)."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{digits}g}") if np.isfinite(x) else None


def _med(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanmean(x, axis=None):
    """nanmean that returns NaN for an all-NaN slice without a warning."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    n = ok.sum(axis=axis)
    tot = np.where(ok, x, 0.0).sum(axis=axis)
    return np.where(n > 0, tot / np.maximum(n, 1), np.nan)


def _safe_div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=(b != 0) & np.isfinite(a) & np.isfinite(b))
    return out


# ---------------------------------------------------------------------------------------------
def decomposition_audit(raw):
    """I1 residuals and explicit norm-class counts for every stored comparison family."""
    out = {}
    worst = 0.0
    for fam in ("AUTO", "TF"):
        base = raw[fam]
        cond = None
        if fam == "AUTO":
            cond = np.ones(base.shape[:-1])
            cond[..., VI["READOUT"], :] = raw["PCOND"][..., 0, :]
            cond[..., VI["PAIRED_STATE"], :] = raw["PCOND"][..., 1, :]
        d = M.from_base(base, pair_condition=cond)
        res = {k: (float(np.nanmax(d[k])) if np.isfinite(d[k]).any() else 0.0) for k in ("res_cos", "res_proj", "res_norm")}
        worst = max(worst, *res.values())
        counts = {name: {cls: int((d["norm_class"][..., vi, :] == i).sum()) for i, cls in enumerate(M.NORM_CLASSES)}
                  for vi, name in enumerate(H.VIEWS)}
        out[fam] = {"max_identity_residual": res, "norm_class_counts": counts, "samples": int(d["norm_class"].size)}
    d = M.from_base(raw["AUTO"][..., VI["READOUT"], :, :], pair_condition=raw["PCOND"][..., 0, :])
    nc = d["norm_class"]                              # [G,K,S,F,C,TR,H,N]
    per_h = {}
    for h in range(H.HMAX):
        row = {cls: int((nc[..., h, :] == i).sum()) for i, cls in enumerate(M.NORM_CLASSES)}
        if row["ORDINARY"] != nc[..., h, :].size:
            per_h[str(h + 1)] = row
    out["readout_non_ordinary_by_horizon"] = per_h
    out["max_identity_residual"] = worst
    out["max_pair_condition"] = {"teacher": float(raw["TCH"][..., 2].max()), "prediction": float(raw["PCOND"].max())}
    out["nonfinite_samples"] = int(sum(v["norm_class_counts"][name]["NONFINITE"] for v in (out["AUTO"], out["TF"]) for name in H.VIEWS))
    return out


def telescoping_audit(raw):
    """I2 and I3 residuals over all samples."""
    ee = raw["AUTO"][..., 3]                           # [G,K,S,F,C,TR,H,V,N]
    ll, le, pn = (raw["D3"][..., i] for i in range(3))
    r2 = _safe_div(np.abs(ee - (pn + 2.0 * le - ll)), ee + pn + ll)
    e20 = ee[..., H.HMAX - 1, :, :]                    # [G,K,S,F,C,TR,V,N]
    dd, de = raw["RST"][..., 0], raw["RST"][..., 1]   # [G,K,S,F,C,TR,NB,V,N]
    amp = np.sqrt(dd).sum(axis=6)                      # the blocks may be much larger than their sum
    r3 = _safe_div(np.abs(de.sum(axis=6) - e20), np.sqrt(e20) * amp)
    return {"one_step_max_residual": float(np.nanmax(r2)), "restart_max_residual": float(np.nanmax(r3)),
            "restart_max_block_to_total_amplitude": float(np.nanmax(_safe_div(amp, np.sqrt(e20))))}


# ---------------------------------------------------------------------------------------------
def _ss(arr, gi, ki, si, cidx):
    """arr [G,K,S,F,C,...] -> samples of one stratum [F * |cohort|, ...]."""
    x = arr[gi, ki, si][:, cidx]
    return x.reshape((-1,) + x.shape[2:])


def _label(item, thr):
    if item["eligible_fraction"] < thr["min_eligible_fraction"]:
        return "INVALID"
    if item["a_den_median"] is not None and item["a_den_median"] >= thr["a_den_collapse"]:
        return "DENOMINATOR_COLLAPSE"
    if item["lambda_local_median"] is None:
        return "INVALID"
    if item["lambda_local_median"] >= thr["lambda_local"]:
        return "LOCAL_TRANSITION"
    if item["lambda_local_median"] <= thr["lambda_accumulated"]:
        return "ACCUMULATED_DRIFT"
    return "MIXED_LOCAL_ACCUMULATED"


def stratum_table(raw, cfg, thr, policy="NATIVE"):
    """Per-stratum registered quantities at every registered horizon (READOUT primary)."""
    hs = cfg["horizons"]
    auto = by_policy(raw["AUTO"], policy)              # [G,K,S,F,C,H,V,N,5]
    d3 = by_policy(raw["D3"], policy)
    rst = by_policy(raw["RST"], policy)                # [G,K,S,F,C,NB,V,N,3]
    pcond = by_policy(raw["PCOND"], policy)            # [G,K,S,F,C,H,2,N]
    ops = [[MS.OPS.index(H.controller_at(ck, policy, h)) for h in range(1, H.HMAX + 1)] for ck in P.CHECKPOINTS]
    tf = np.stack([np.stack([raw["TF"][:, :, :, :, ci, ops[ci][h], h] for h in range(H.HMAX)], axis=4)
                   for ci in range(len(P.CHECKPOINTS))], axis=4)      # [G,K,S,F,C,H,V,N,5]
    tch = raw["TCH"]                                   # [G,K,S,F,H+1,V,N,3]
    names, ks, scales = list(cfg["graphs"]), list(cfg["k_values"]), list(cfg["scales"])
    R, PS = VI["READOUT"], VI["PAIRED_STATE"]
    rows = {}
    for gi, g in enumerate(names):
        for ki, k in enumerate(ks):
            for si, s in enumerate(scales):
                for cname, cidx in cohort_indices().items():
                    rec = {}
                    for h in hs:
                        hi = h - 1
                        a_r = M.from_base(_ss(auto[..., hi, R, :, :], gi, ki, si, cidx), _ss(pcond[..., hi, 0, :], gi, ki, si, cidx))
                        a_s = M.from_base(_ss(auto[..., hi, PS, :, :], gi, ki, si, cidx), _ss(pcond[..., hi, 1, :], gi, ki, si, cidx))
                        t_r = M.from_base(_ss(tf[..., hi, R, :, :], gi, ki, si, cidx))
                        dd = _ss(d3[..., hi, R, :, :], gi, ki, si, cidx)
                        ok = a_r["eligible"] & a_s["eligible"] & (a_r["ee"] > 0) & (a_s["ee"] > 0) & (t_r["tt"] > 0)
                        tshape = (tch.shape[3], len(cidx), tch.shape[6])
                        tt_h = np.broadcast_to(tch[gi, ki, si][:, None, h, R, :, 0], tshape).reshape(-1, tshape[2])
                        per_h = np.broadcast_to(tch[gi, ki, si][:, None, h, R, :, 1], tshape).reshape(-1, tshape[2])
                        eps_per = _safe_div(per_h, tt_h)
                        anyok = bool(ok.any())
                        item = {
                            "n": int(ok.size), "eligible_fraction": _f(ok.mean()),
                            "erel_mean": _f(_mean(a_r["erel"][ok])), "erel_median": _f(_med(a_r["erel"][ok])),
                            "erel_ratio_of_sums": _f(a_r["ee"][ok].sum() / a_r["tt"][ok].sum()) if anyok else None,
                            "eabs_sum": _f(a_r["ee"][ok].sum()), "teacher_energy_sum": _f(a_r["tt"][ok].sum()),
                            "r_median": _f(_med(a_r["r"][ok])), "c_median": _f(_med(a_r["c"][ok])),
                            "c_min": _f(a_r["c"][ok].min()) if anyok else None,
                            "rho_negative_fraction": _f((a_r["rho"][ok] < 0).mean()) if anyok else None,
                            "egain_mean": _f(_mean(a_r["egain"][ok])), "eperp_mean": _f(_mean(a_r["eperp"][ok])),
                            "perp_share_median": _f(_med(_safe_div(a_r["eperp"], a_r["erel"])[ok])),
                            "state_erel_median": _f(_med(a_s["erel"][ok])), "state_c_median": _f(_med(a_s["c"][ok])),
                            "state_r_median": _f(_med(a_s["r"][ok])),
                            "state_perp_share_median": _f(_med(_safe_div(a_s["eperp"], a_s["erel"])[ok])),
                            "a_den_median": _f(_med(_safe_div(a_r["erel"], a_s["erel"])[ok])),
                            "local_erel_median": _f(_med(t_r["erel"][ok])), "local_erel_mean": _f(_mean(t_r["erel"][ok])),
                            "persistence_erel_median": _f(_med(eps_per[ok])),
                            "local_skill_median": _f(_med((1.0 - _safe_div(t_r["erel"], eps_per))[ok])),
                            "lambda_local_median": _f(_med(_safe_div(dd[..., 0], a_r["ee"])[ok])),
                            "local_projection_share_median": _f(_med(_safe_div(dd[..., 1], a_r["ee"])[ok])),
                            "cancellation_index_median": _f(_med(_safe_div(np.sqrt(dd[..., 2]) + np.sqrt(dd[..., 0]), np.sqrt(a_r["ee"]))[ok])),
                            "reversed": int((a_r["dir_class"] == M.D_REVERSED).sum()),
                            "extinguished": int((a_r["dir_class"] == M.D_EXTINGUISHED).sum()),
                        }
                        item["mechanism"] = _label(item, thr)
                        rec["H" + str(h)] = item
                    # growth split between two registered horizons (I4, I5): sample-wise, then medians
                    h0, h1 = thr["growth_from"], thr["growth_to"]
                    b0, b1 = _ss(auto[..., h0 - 1, R, :, :], gi, ki, si, cidx), _ss(auto[..., h1 - 1, R, :, :], gi, ki, si, cidx)
                    s0, s1 = _ss(auto[..., h0 - 1, PS, :, :], gi, ki, si, cidx), _ss(auto[..., h1 - 1, PS, :, :], gi, ki, si, cidx)
                    ok = ((b0[..., 0] > 0) & (b1[..., 0] > 0) & (b0[..., 3] > 0) & (b1[..., 3] > 0)
                          & (s0[..., 0] > 0) & (s1[..., 0] > 0) & (s0[..., 3] > 0) & (s1[..., 3] > 0))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        g_num = np.log(b1[..., 3] / b0[..., 3])[ok]
                        g_den = np.log(b0[..., 0] / b1[..., 0])[ok]
                        g_state = (np.log(s1[..., 3] / s1[..., 0]) - np.log(s0[..., 3] / s0[..., 0]))[ok]
                    g_tot = g_num + g_den
                    mt, mn = _med(g_tot), _med(g_num)
                    gclass = "NO_GROWTH" if not mt > 0 else ("NUMERATOR_DRIVEN" if mn > 0 else "DENOMINATOR_DRIVEN")
                    rec["growth"] = {"from": h0, "to": h1, "n": int(ok.sum()), "dlog_erel_median": _f(mt),
                                     "dlog_eabs_median": _f(mn), "dlog_inverse_teacher_energy_median": _f(_med(g_den)),
                                     "dlog_state_erel_median": _f(_med(g_state)),
                                     "dlog_a_den_median": _f(_med(g_tot - g_state)), "class": gclass,
                                     "r93_pattern_sample_fraction": _f(((g_num <= 0) & (g_tot > 0)).mean()) if ok.any() else None}
                    # origin blocks of the H20 error (I3)
                    for vname in ("READOUT", "PAIRED_STATE"):
                        v = VI[vname]
                        blk = np.moveaxis(_ss(rst[..., v, :, :], gi, ki, si, cidx), 1, -2).reshape(-1, MS.NB, 3)
                        last = _ss(auto[..., H.HMAX - 1, v, :, :], gi, ki, si, cidx)
                        e20, t20 = last[..., 3].reshape(-1), last[..., 0].reshape(-1)
                        ok = (e20 > 0) & (t20 > 0)
                        shares = _safe_div(blk[..., 1], e20[:, None])[ok]
                        rec["origin_" + vname] = {
                            "share_median": [_f(_med(shares[:, b])) for b in range(MS.NB)],
                            "block_energy_over_teacher_median": [_f(_med(_safe_div(blk[..., 0], t20[:, None])[ok][:, b])) for b in range(MS.NB)],
                            "restart_erel_median": [_f(_med(_safe_div(blk[..., 2], t20[:, None])[ok][:, b])) for b in range(MS.NB)],
                            "absolute_share_sum_median": _f(_med(np.abs(shares).sum(axis=1))),
                        }
                    rows[f"{g}/k{k}/{s}/{cname}"] = rec
    return rows


def compact_strata(strata, digits=7, keep=None):
    """Compact (fields + rows) form of a stratum table for the evidence file.

    ``keep`` optionally restricts the entries per stratum (e.g. two horizons for controlled policies).
    """
    out = {"fields": list(STRATUM_FIELDS) + ["mechanism"], "strata": {}}
    for key, rec in strata.items():
        row = {}
        for name, item in rec.items():
            if keep is not None and name not in keep:
                continue
            if name.startswith("H"):
                row[name] = [(_f(item[f], digits) if isinstance(item[f], float) else item[f]) for f in STRATUM_FIELDS] + [item["mechanism"]]
            else:
                row[name] = item
        out["strata"][key] = row
    return out


# ---------------------------------------------------------------------------------------------
def dominance(labels_by_graph, classes, thr):
    """Registered dominance rule over strata grouped by graph class.

    A class is dominant iff it holds in >= ``dominant_fraction`` of all strata *and* in a
    strict majority of the strata of every graph class.
    """
    flat = [x for v in labels_by_graph.values() for x in v]
    counts = {c: flat.count(c) for c in classes}
    total = len(flat)
    dominant = None
    for c in classes:
        if total and counts[c] / total >= thr["dominant_fraction"] and all(
                v.count(c) * 2 > len(v) for v in labels_by_graph.values()):
            dominant = c
    return {"counts": counts, "total": total, "dominant": dominant,
            "by_graph": {g: {c: v.count(c) for c in classes if v.count(c)} for g, v in labels_by_graph.items()}}


def _scoped(strata, cfg, thr, scale_label, getter, classes):
    names, ks = list(cfg["graphs"]), list(cfg["k_values"])
    out = {}
    for scope, cohorts in (("ALL", COHORTS), ("SIGNED", ("SIGNED",)), ("UNSIGNED", ("UNSIGNED",))):
        lab = {g: [getter(strata[f"{g}/k{k}/{scale_label}/{c}"]) for k in ks for c in cohorts] for g in names}
        out[scope] = dominance(lab, classes, thr)
    return out


def mechanism_decision(strata, cfg, thr, scale_label, horizon):
    out = _scoped(strata, cfg, thr, scale_label, lambda r: r["H" + str(horizon)]["mechanism"], MECHANISMS)
    return out, out["ALL"]["counts"]["INVALID"] / max(out["ALL"]["total"], 1)


def growth_decision(strata, cfg, thr, scale_label):
    return _scoped(strata, cfg, thr, scale_label, lambda r: r["growth"]["class"], GROWTH)


def disposition(decision, invalid_fraction, thr):
    """Registered scientific disposition (before the integrity gates, which can only demote to NONQUAL)."""
    if invalid_fraction > thr["max_invalid_strata_fraction"]:
        return "MECHANISM_INDETERMINATE"
    if decision["ALL"]["dominant"] in LOCALIZING:
        return "MECHANISM_LOCALIZED_BOUNDED"
    return "MECHANISM_MIXED_BOUNDED"


# ---------------------------------------------------------------------------------------------
def directional_events(raw, cfg, thr):
    """Sign-reversal census on the READOUT view for every policy, all twenty steps.

    ``induced``: the policy's sample is REVERSED (c <= COS_REVERSED with a meaningful norm)
    while the NATIVE sample of the same fixture, checkpoint, pair and step is ALIGNED.
    Verdicts are per amplitude regime (in-range scales; above-clamp stress scale) and count
    every one of the twenty recorded steps.
    """
    R = VI["READOUT"]
    names, ks, scales = list(cfg["graphs"]), list(cfg["k_values"]), list(cfg["scales"])
    in_range = [si for si, s in enumerate(scales) if s not in thr["above_clamp_scales"]]
    above = [si for si, s in enumerate(scales) if s in thr["above_clamp_scales"]]
    nat = M.from_base(by_policy(raw["AUTO"], "NATIVE")[..., R, :, :], by_policy(raw["PCOND"], "NATIVE")[..., 0, :])
    reg_h = [h - 1 for h in cfg["horizons"]]
    coh = cohort_indices()
    out = {}
    for pol in H.POLICIES:
        d = M.from_base(by_policy(raw["AUTO"], pol)[..., R, :, :], by_policy(raw["PCOND"], pol)[..., 0, :])
        rev = d["dir_class"] == M.D_REVERSED            # [G,K,S,F,C,H,N]
        ext = d["dir_class"] == M.D_EXTINGUISHED
        induced = rev & (nat["dir_class"] == M.D_ALIGNED)
        ext_ind = ext & (nat["dir_class"] == M.D_ALIGNED)
        cells = []
        for gi, g in enumerate(names):
            for ki, k in enumerate(ks):
                for si, s in enumerate(scales):
                    for ci, ck in enumerate(P.CHECKPOINTS):
                        r = rev[gi, ki, si, :, ci]       # [F, H, N]
                        if not r.any():
                            continue
                        per_h = r.sum(axis=(0, 2))
                        ind = induced[gi, ki, si, :, ci].sum(axis=(0, 2))
                        hs_any = np.nonzero(per_h)[0]
                        cells.append([f"{g}/k{k}/{s}/{ck}", int(hs_any[0]) + 1, int(hs_any[-1]) + 1, int(len(hs_any)),
                                      int(per_h.max()), int(ind.sum()), int(np.nonzero(ind)[0][0]) + 1 if ind.any() else None,
                                      _f(np.nanmax(np.where(r, d["edir"][gi, ki, si, :, ci], np.nan)), 6),
                                      _f(_med(np.where(r, d["r"][gi, ki, si, :, ci], np.nan)), 6)])
        first = {}
        for c in cells:
            first[str(c[1])] = first.get(str(c[1]), 0) + 1

        def census(mask):
            return {"all_steps": int(mask.sum()), "registered_horizons": int(mask[..., reg_h, :].sum()),
                    "by_horizon": [int(x) for x in mask.sum(axis=(0, 1, 2, 3, 4, 6))],
                    "by_scale": {s: int(mask[:, :, si].sum()) for si, s in enumerate(scales)},
                    "by_scale_registered_horizons": {s: int(mask[:, :, si][..., reg_h, :].sum()) for si, s in enumerate(scales)},
                    "by_cohort": {c: int(mask[:, :, :, :, idx].sum()) for c, idx in coh.items()},
                    "by_graph": {g: int(mask[gi].sum()) for gi, g in enumerate(names)},
                    "by_k": {str(k): int(mask[:, ki].sum()) for ki, k in enumerate(ks)}}

        # the safety verdict uses every recorded step, not only the registered reporting horizons
        n_in = int(induced[:, :, in_range].sum())
        n_above = int(induced[:, :, above].sum()) if above else 0

        def verdict(n):
            return "HAZARD_CONTROLLER_INDUCED_REVERSAL" if n > thr["max_induced_reversals"] else "NO_INDUCED_REVERSAL_OBSERVED"

        out[pol] = {
            "samples_all_steps": int(rev.size), "samples_registered_horizons": int(rev[..., reg_h, :].size),
            "reversed": census(rev), "extinguished": census(ext),
            "induced_reversal": census(induced), "induced_extinction": census(ext_ind),
            "max_edir": _f(np.nanmax(np.where(rev, d["edir"], np.nan))) if rev.any() else None,
            "cells_with_reversal": len(cells), "first_reversal_horizon_histogram": first,
            "cell_fields": ["cell", "first_horizon", "last_horizon", "horizons_with_reversal", "max_reversed_samples_at_a_horizon",
                            "induced_samples", "first_induced_horizon", "max_edir", "median_r_when_reversed"],
            "cells": cells,
            "verdict_in_range_scales": "REFERENCE" if pol == "NATIVE" else verdict(n_in),
            "verdict_above_clamp_scales": "REFERENCE" if pol == "NATIVE" else (verdict(n_above) if above else "NOT_MEASURED"),
            "induced_in_range_all_steps": n_in, "induced_above_clamp_all_steps": n_above,
            "induced_in_range_registered_horizons": int(induced[:, :, in_range][..., reg_h, :].sum()),
        }
    return out


def policy_comparison(raw, cfg, thr, scale_label):
    """R121-comparable pooled READOUT NSE, paired harms vs NATIVE and graph-cluster bootstrap intervals."""
    R = VI["READOUT"]
    names, ks, scales = list(cfg["graphs"]), list(cfg["k_values"]), list(cfg["scales"])
    si = scales.index(scale_label)
    rng = np.random.default_rng(thr["bootstrap_seed"])
    boot_idx = rng.integers(0, len(names), size=(thr["bootstrap_resamples"], len(names)))
    cellmean = {}
    for pol in H.POLICIES:
        b = by_policy(raw["AUTO"], pol)[:, :, si][..., R, :, :]       # [G,K,F,C,H,N,5]
        # [G, K, C, H]: mean over fixtures x pairs (the R118-E / R121 cell statistic); cells whose
        # target is exactly zero (readout not yet reached) have no E_rel and stay NaN -> None
        cellmean[pol] = _nanmean(_safe_div(b[..., 3], b[..., 0]), axis=(2, 5))
    out = {"scale": scale_label, "pooled": {}, "paired_vs_native": {}, "worst_graph_harm_h20": {}}
    worst_all = {}
    for ki, k in enumerate(ks):
        for h in cfg["horizons"]:
            for pol in H.POLICIES:
                out["pooled"][f"k{k}/{pol}/H{h}"] = _f(_nanmean(cellmean[pol][:, ki, :, h - 1]))
        for pol in H.POLICIES[1:]:
            diff = cellmean[pol][:, ki, :, H.HMAX - 1] - cellmean["NATIVE"][:, ki, :, H.HMAX - 1]     # [G, C]
            gmean = np.nanmean(diff, axis=1)
            boots = np.nanmean(gmean[boot_idx], axis=1)
            out["paired_vs_native"][f"k{k}/{pol}/H20"] = {
                "mean_delta_erel": _f(np.nanmean(gmean)), "graph_blocks_improved": int((gmean < 0).sum()),
                "graph_blocks": len(names), "graph_cluster_bootstrap_ci95": [_f(np.quantile(boots, 0.025)), _f(np.quantile(boots, 0.975))],
                "by_graph": {g: _f(gmean[gi]) for gi, g in enumerate(names)}}
            out["worst_graph_harm_h20"][f"k{k}/{pol}"] = {"max_graph_block_delta_erel": _f(np.nanmax(gmean)),
                                                         "graph": names[int(np.nanargmax(gmean))]}
            worst_all[pol] = max(worst_all.get(pol, -np.inf), float(np.nanmax(gmean)))
    out["worst_graph_harm_h20_over_k"] = {pol: _f(v) for pol, v in worst_all.items()}
    return out


def controller_effects(raw, cfg):
    """Controller-specific contributions (READOUT): local one-step harm and per-step displacement effect."""
    R = VI["READOUT"]
    scales = list(cfg["scales"])
    coh = cohort_indices()
    tf = raw["TF"][..., R, :, :]                       # [G,K,S,F,C,OP,H,N,5]
    erel = _safe_div(tf[..., 3], tf[..., 0])
    out = {"teacher_forced_one_step": {}, "autonomous_step_effect": {}}
    for cname, cidx in coh.items():
        e = erel[:, :, :, :, cidx]
        valid = np.isfinite(e[:, :, :, :, :, 0]) & np.isfinite(e[:, :, :, :, :, 1])
        worse = e[:, :, :, :, :, 1] > e[:, :, :, :, :, 0]
        out["teacher_forced_one_step"][cname] = {
            "fraction_samples_local_nse_worse_under_r114": _f(worse[valid].mean()),
            "fraction_by_scale": {s: _f(worse[:, :, si][valid[:, :, si]].mean()) for si, s in enumerate(scales)},
            "fraction_by_horizon": [_f(worse[..., h, :][valid[..., h, :]].mean()) if valid[..., h, :].any() else None
                                    for h in range(H.HMAX)],
            "median_local_erel_native_by_horizon": [_f(_med(e[:, :, :, :, :, 0, h]), 7) for h in range(H.HMAX)],
            "median_local_erel_r114_by_horizon": [_f(_med(e[:, :, :, :, :, 1, h]), 7) for h in range(H.HMAX)]}
    # along the controlled autonomous trajectories: did the controller's displacement at step h raise the error?
    ctrl = raw["CTRL"][..., R, :, :]                    # [G,K,S,F,C,2,H,N,4]
    for qi, tr in enumerate(MS.CTRL_TR):
        ee = raw["AUTO"][:, :, :, :, :, H.TRAJECTORIES.index(tr)][..., R, :, 3]       # [G,K,S,F,C,H,N]
        ne = ctrl[:, :, :, :, :, qi, :, :, 3]
        kk = ctrl[:, :, :, :, :, qi, :, :, 0]
        active = kk > 0
        for cname, cidx in coh.items():
            a = active[:, :, :, :, cidx]
            if not a.any():
                continue
            harm = (ee[:, :, :, :, cidx] > ne[:, :, :, :, cidx])
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.log(ee[:, :, :, :, cidx] / ne[:, :, :, :, cidx])
            out["autonomous_step_effect"][f"{tr}/{cname}"] = {
                "fraction_steps_error_raised_by_controller": _f(harm[a].mean()),
                "fraction_by_scale": {s: _f(harm[:, :, si][a[:, :, si]].mean()) for si, s in enumerate(scales)},
                "fraction_by_horizon": [_f(harm[..., h, :][a[..., h, :]].mean()) if a[..., h, :].any() else None
                                        for h in range(H.HMAX)],
                "median_log_error_ratio_by_horizon": [_f(_med(lg[..., h, :][a[..., h, :]]), 7) if a[..., h, :].any() else None
                                                      for h in range(H.HMAX)]}
    return out


def curves(raw, cfg, digits=6):
    """Per-step pooled curves (all twenty steps) by policy, cohort, k, scale and view."""
    ks, scales = list(cfg["k_values"]), list(cfg["scales"])
    out = {"fields": ["erel_mean", "erel_median", "r_median", "c_median", "eligible_fraction"], "data": {}}
    for pol in H.POLICIES:
        b = by_policy(raw["AUTO"], pol)
        pc = by_policy(raw["PCOND"], pol)
        for cname, cidx in cohort_indices().items():
            for ki, k in enumerate(ks):
                for si, s in enumerate(scales):
                    for vname, pci in (("READOUT", 0), ("PAIRED_STATE", 1), ("WEAK_STATE", None), ("STRONG_STATE", None)):
                        x = b[:, ki, si][:, :, cidx][..., VI[vname], :, :]           # [G,F,c,H,N,5]
                        cond = None if pci is None else pc[:, ki, si][:, :, cidx][..., pci, :]
                        d = M.from_base(x, cond)
                        rows = []
                        for h in range(H.HMAX):
                            m = d["eligible"][..., h, :]
                            rows.append([_f(_mean(d["erel"][..., h, :][m]), digits), _f(_med(d["erel"][..., h, :][m]), digits),
                                         _f(_med(d["r"][..., h, :][m]), digits), _f(_med(d["c"][..., h, :][m]), digits),
                                         _f(m.mean(), digits)])
                        out["data"][f"{pol}/{cname}/k{k}/{s}/{vname}"] = rows
    tch = raw["TCH"]
    out["teacher_energy_median"] = {f"k{k}/{s}/{v}": [_f(_med(tch[:, ki, si, :, h, VI[v], :, 0]), digits) for h in range(H.HMAX + 1)]
                                    for ki, k in enumerate(ks) for si, s in enumerate(scales) for v in H.VIEWS}
    return out


def cell_table(raw, cfg, digits=6):
    """Compact per-cell table (graph, k, scale, policy, checkpoint) x registered horizon, READOUT view."""
    R = VI["READOUT"]
    names, ks, scales = list(cfg["graphs"]), list(cfg["k_values"]), list(cfg["scales"])
    fields = ["n_ordinary", "erel_mean", "erel_median", "r_median", "c_median", "c_min", "eabs_sum", "teacher_energy_sum",
              "n_reversed", "n_extinguished"]
    data = {}
    for pol in H.POLICIES:
        d = M.from_base(by_policy(raw["AUTO"], pol)[..., R, :, :], by_policy(raw["PCOND"], pol)[..., 0, :])
        for gi, g in enumerate(names):
            for ki, k in enumerate(ks):
                for si, s in enumerate(scales):
                    for ci, ck in enumerate(P.CHECKPOINTS):
                        rows = []
                        for h in cfg["horizons"]:
                            sl = (gi, ki, si, slice(None), ci, h - 1)
                            ok = d["eligible"][sl]
                            tail = [int((d["dir_class"][sl] == M.D_REVERSED).sum()), int((d["dir_class"][sl] == M.D_EXTINGUISHED).sum())]
                            if not ok.any():
                                rows.append([0] + [None] * 7 + tail)
                                continue
                            rows.append([int(ok.sum()), _f(d["erel"][sl][ok].mean(), digits), _f(np.median(d["erel"][sl][ok]), digits),
                                         _f(np.median(d["r"][sl][ok]), digits), _f(np.median(d["c"][sl][ok]), digits),
                                         _f(d["c"][sl][ok].min(), digits), _f(d["ee"][sl][ok].sum(), digits),
                                         _f(d["tt"][sl][ok].sum(), digits)] + tail)
                        data[f"{g}/k{k}/{s}/{pol}/{ck}"] = rows
    return {"view": "READOUT", "horizons": list(cfg["horizons"]), "fields": fields, "cells": data}


def secondary_findings(strata, cfg, thr, scale_label, horizon):
    """Registered secondary findings over the primary strata (descriptive; fixed criteria)."""
    names, ks = list(cfg["graphs"]), list(cfg["k_values"])
    items = [strata[f"{g}/k{k}/{scale_label}/{c}"]["H" + str(horizon)] for g in names for k in ks for c in COHORTS]
    n = len(items)

    def frac(pred):
        vals = [pred(i) for i in items]
        return sum(bool(v) for v in vals) / n

    f_col = frac(lambda i: i["perp_share_median"] is not None and i["perp_share_median"] <= thr["collinear_perp_share"])
    f_dir = frac(lambda i: i["state_c_median"] is not None and i["state_c_median"] < M.COS_ALIGNED)
    f_skill = frac(lambda i: i["local_skill_median"] is not None and i["local_skill_median"] > 0)
    dom = thr["dominant_fraction"]
    return {
        "S1_readout_error_collinear_with_teacher": {"strata_fraction": _f(f_col), "supported": bool(f_col >= dom),
                                                    "criterion": f"median E_perp/E_rel <= {thr['collinear_perp_share']} in >= {dom} of primary strata"},
        "S2_paired_state_direction_not_preserved": {"strata_fraction": _f(f_dir), "supported": bool(f_dir >= dom),
                                                    "criterion": f"median PAIRED_STATE cosine < {M.COS_ALIGNED} in >= {dom} of primary strata"},
        "S3_local_step_beats_persistence": {"strata_fraction": _f(f_skill),
                                            "class": "LOCALLY_SKILLFUL" if f_skill >= dom else ("LOCALLY_UNSKILLED" if f_skill <= 1 - dom else "MIXED_LOCAL_SKILL"),
                                            "criterion": "median teacher-forced one-step skill vs teacher persistence > 0"},
    }


# ---------------------------------------------------------------------------------------------
def analyze(raw, cfg, thr):
    """Complete registered analysis. Returns (result, tables) with tables = strata / cells / curves."""
    ps, ph = thr["primary_scale"], thr["primary_horizon"]
    out = {"decomposition_audit": decomposition_audit(raw), "telescoping_audit": telescoping_audit(raw)}
    strata = {pol: stratum_table(raw, cfg, thr, pol) for pol in H.POLICIES}
    mech, invalid = mechanism_decision(strata["NATIVE"], cfg, thr, ps, ph)
    out["mechanism"] = {"policy": "NATIVE", "view": "READOUT", "scale": ps, "horizon": ph, "decision": mech,
                        "invalid_fraction": _f(invalid)}
    out["scientific_disposition_before_integrity"] = disposition(mech, invalid, thr)
    out["cohort_dispositions"] = {c: ("LOCALIZED:" + mech[c]["dominant"] if mech[c]["dominant"] in LOCALIZING else "NOT_LOCALIZED")
                                  for c in COHORTS}
    out["mechanism_by_scale"] = {s: mechanism_decision(strata["NATIVE"], cfg, thr, s, ph)[0] for s in cfg["scales"]}
    out["mechanism_by_horizon"] = {str(h): mechanism_decision(strata["NATIVE"], cfg, thr, ps, h)[0] for h in cfg["horizons"]}
    out["mechanism_controlled_policies"] = {pol: mechanism_decision(strata[pol], cfg, thr, ps, ph)[0] for pol in H.POLICIES[1:]}
    out["growth"] = {pol: {s: growth_decision(strata[pol], cfg, thr, s) for s in cfg["scales"]} for pol in H.POLICIES}
    names, ks = list(cfg["graphs"]), list(cfg["k_values"])
    rob = {}
    for s in cfg["scales"]:
        if s == ps:
            continue
        agree = [strata["NATIVE"][f"{g}/k{k}/{s}/{c}"]["H" + str(ph)]["mechanism"]
                 == strata["NATIVE"][f"{g}/k{k}/{ps}/{c}"]["H" + str(ph)]["mechanism"]
                 for g in names for k in ks for c in COHORTS]
        rob[s] = {"label_agreement_with_primary_scale": _f(np.mean(agree)), "robust": bool(np.mean(agree) >= thr["dominant_fraction"])}
    out["amplitude_robustness"] = rob
    out["secondary_findings"] = secondary_findings(strata["NATIVE"], cfg, thr, ps, ph)
    out["directional"] = directional_events(raw, cfg, thr)
    out["policy_comparison"] = policy_comparison(raw, cfg, thr, ps)
    out["controller_effects"] = controller_effects(raw, cfg)
    keep = ("H" + str(thr["growth_from"]), "H" + str(ph), "growth", "origin_READOUT")
    tables = {"strata": {pol: compact_strata(strata[pol], keep=None if pol == "NATIVE" else keep) for pol in H.POLICIES},
              "cells": cell_table(raw, cfg), "curves": curves(raw, cfg)}
    return out, tables
