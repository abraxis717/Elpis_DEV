"""R122 measurement pass: raw base-sum tensors for every registered comparison.

``run(config, oracle)`` executes, for every fixture (graph, overlap k, amplitude scale,
case, seed) and every frozen checkpoint:

AUTO   autonomous trajectories X^tr_h (tr in NATIVE, R114, DELAYED8), h = 1..20, vs teacher T_h
TF     offline teacher-forced one-step outputs Phi^op(T_{h-1}) (op in NATIVE, R114) vs T_h
D3     one-step telescoping  e_h = P_h + L_h  with  L_h = Phi^{op(tr,h)}(T_{h-1}) - T_h  (local
       transition defect) and  P_h = X^tr_h - Phi^{op(tr,h)}(T_{h-1})  (propagated incoming mismatch)
RST    teacher-prefix restarts Y^(j)_20 = policy run autonomously from T_j to step 20,
       j in RESTARTS; with Y^(0) = X_20 and Y^(20) = T_20 the differences
       D_b = Y^(j_b) - Y^(j_{b+1}) telescope exactly to e_20: block b is the effect at step 20
       of the local defects committed in steps j_b + 1 .. j_{b+1}
CTRL   controller displacement K_h = X^tr_h - N(X^tr_{h-1}) along the controlled trajectory
       (N = native step from the same incoming state), for tr in R114, DELAYED8
TCH    teacher energies, one-step persistence baseline and pair conditioning

Each comparison is stored as absolute sums (see ``metrics.BASE_FIELDS``) per sample and per
view; nothing is averaged here. The analysis pass (``analyze.py``) derives every reported
quantity from these tensors. Integrity counters are returned alongside.

Autonomy contract: an autonomous trajectory is produced by engine steps from X0 only.
Teacher states enter engine calls exclusively as inputs of TF / RST diagnostics, and the
autonomous trajectories of one rotating checkpoint are recomputed after all diagnostics
of a fixture and compared bit for bit.
"""
from __future__ import annotations

import time

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import provenance as P

from . import harness as H
from . import metrics as M

RESTARTS = (4, 8, 12, 16)
BLOCK_EDGES = (0,) + RESTARTS + (H.HMAX,)            # blocks: steps 1-4, 5-8, 9-12, 13-16, 17-20
NB = len(BLOCK_EDGES) - 1
OPS = ("NATIVE", "R114")
CTRL_TR = ("R114", "DELAYED8")
NV, NS = len(H.VIEWS), H.PAIRS


def shapes(cfg):
    G, K, S = len(cfg["graphs"]), len(cfg["k_values"]), len(cfg["scales"])
    F = len(cfg["cases"]) * len(cfg["evaluation_seeds"])
    C = len(P.CHECKPOINTS)
    lead = (G, K, S, F)
    return {
        "AUTO": lead + (C, len(H.TRAJECTORIES), H.HMAX, NV, NS, 5),
        "TF": lead + (C, len(OPS), H.HMAX, NV, NS, 5),
        "D3": lead + (C, len(H.TRAJECTORIES), H.HMAX, NV, NS, 3),
        "RST": lead + (C, len(H.TRAJECTORIES), NB, NV, NS, 3),
        "CTRL": lead + (C, len(CTRL_TR), H.HMAX, NV, NS, 4),
        "TCH": lead + (H.HMAX + 1, NV, NS, 3),
        "PCOND": lead + (C, len(H.TRAJECTORIES), H.HMAX, 2, NS),
    }


def raw_bytes(cfg) -> int:
    return int(sum(8 * int(np.prod(s)) for s in shapes(cfg).values()))


def engine_step_bound(cfg) -> int:
    """Upper bound on HCA passes of the main loop (registered work budget)."""
    C_signed = sum(H.is_signed(ck) for ck in P.CHECKPOINTS)
    C_unsigned = len(P.CHECKPOINTS) - C_signed
    rst = sum(H.HMAX - j for j in RESTARTS)
    per_unsigned = 2 * H.HMAX + 2 * H.HMAX + H.HMAX + 2 * rst
    per_signed = per_unsigned + 2 * (H.HMAX - 8) + (H.HMAX - RESTARTS[0])
    recheck = 2 * H.HMAX
    deep = C_signed * H.HMAX                      # full DELAYED8 rollouts on deep-check fixtures (upper bound: all)
    fixtures = (len(cfg["graphs"]) * len(cfg["k_values"]) * len(cfg["scales"]) * len(cfg["cases"])
                * len(cfg["evaluation_seeds"]))
    return fixtures * (C_unsigned * per_unsigned + C_signed * per_signed + recheck + deep)


def _dot(a, b):
    return (a * b).sum(-1)


def _fixture(S, oracle, X0, T, f_index, counters, maxima, deep_checks):
    """All comparisons for one fixture; returns dict of arrays without the (G,K,S,F) lead."""
    C = len(P.CHECKPOINTS)
    R = X0.shape[0]
    XS = np.empty((C, 3, H.HMAX + 1, R, 8))
    FS = np.empty((C, 2, H.HMAX, R, 8))            # Phi^op(T_{h-1}), h = 1..20
    NSt = np.empty((C, 2, H.HMAX, R, 8))           # N(X^tr_{h-1}) for tr in CTRL_TR
    YS = np.empty((C, 3, NB + 1, R, 8))            # Y^(j) at step 20 for j in BLOCK_EDGES
    for ci, ck in enumerate(P.CHECKPOINTS):
        signed = H.is_signed(ck)
        Xn = S.rollout(X0, ck, "NATIVE")
        Xc = S.rollout(X0, ck, "R114")
        if signed:
            tail = S.rollout(Xn[8], ck, "DELAYED8", start=8)
            Xd = np.concatenate([Xn[:9], tail[1:]], axis=0)
        else:
            Xd = Xn
        XS[ci, 0], XS[ci, 1], XS[ci, 2] = Xn, Xc, Xd
        for h in range(1, H.HMAX + 1):
            FS[ci, 0, h - 1] = S.step(T[h - 1], ck, "NATIVE")
            FS[ci, 1, h - 1] = S.step(T[h - 1], ck, "R114")
            NSt[ci, 0, h - 1] = S.step(Xc[h - 1], ck, "NATIVE")
            if signed and h > 8:
                NSt[ci, 1, h - 1] = S.step(Xd[h - 1], ck, "NATIVE")
            else:
                NSt[ci, 1, h - 1] = Xd[h]          # controller inactive: the step taken is the native step
        YS[ci, :, 0] = XS[ci, :, H.HMAX]
        YS[ci, :, NB] = T[H.HMAX]
        for bi, j in enumerate(RESTARTS, start=1):
            YS[ci, 0, bi] = S.rollout(T[j], ck, "NATIVE", start=j)[-1]
            YS[ci, 1, bi] = S.rollout(T[j], ck, "R114", start=j)[-1]
            if not signed:
                YS[ci, 2, bi] = YS[ci, 0, bi]
            elif j >= 8:
                YS[ci, 2, bi] = YS[ci, 1, bi]      # every remaining DELAYED8 step is R114
            else:
                YS[ci, 2, bi] = S.rollout(T[j], ck, "DELAYED8", start=j)[-1]
        if deep_checks and signed:
            full = S.rollout(X0, ck, "DELAYED8")
            counters["delayed_spliced_vs_full_mismatch"] += int(not np.array_equal(full.view(np.uint64), Xd.view(np.uint64)))
            counters["delayed_h8_vs_native_mismatch"] += int(not np.array_equal(full[8].view(np.uint64), Xn[8].view(np.uint64)))
            counters["delayed_checks"] += 1
    # autonomy isolation: recompute one checkpoint's autonomous trajectories after all diagnostics
    ci = f_index % C
    ck = P.CHECKPOINTS[ci]
    for ti, pol in ((0, "NATIVE"), (1, "R114")):
        again = S.rollout(X0, ck, pol)
        counters["autonomous_recheck_mismatch"] += int(not np.array_equal(again.view(np.uint64), XS[ci, ti].view(np.uint64)))
        counters["autonomous_rechecks"] += 1
    if not (np.isfinite(XS).all() and np.isfinite(FS).all() and np.isfinite(YS).all() and np.isfinite(NSt).all()):
        counters["nonfinite"] += 1
    if deep_checks:
        # exact rational audit of the float sums on real vectors (independent reference arithmetic)
        for name in ("READOUT", "PAIRED_STATE"):
            for h in (1, 8, H.HMAX):
                pv, tv = S.view(XS[ci, f_index % 2, h], name), S.view(T[h], name)
                for n in range(NS):
                    worst, identity = M.exact_audit(pv[n], tv[n])
                    maxima["exact_audit_max_relative"] = max(maxima["exact_audit_max_relative"], worst)
                    counters["exact_identity_nonzero"] += int(identity != 0)
                    counters["exact_audit_samples"] += 1
    out = {k: None for k in ("AUTO", "TF", "D3", "RST", "CTRL", "TCH", "PCOND")}
    auto = np.empty((C, 3, H.HMAX, NV, NS, 5))
    tf = np.empty((C, 2, H.HMAX, NV, NS, 5))
    d3 = np.empty((C, 3, H.HMAX, NV, NS, 3))
    rst = np.empty((C, 3, NB, NV, NS, 3))
    ctrl = np.empty((C, 2, H.HMAX, NV, NS, 4))
    tch = np.empty((H.HMAX + 1, NV, NS, 3))
    pcond = np.ones((C, 3, H.HMAX, 2, NS))
    op_index = np.array([[[OPS.index(H.controller_at(ck, tr, h)) for h in range(1, H.HMAX + 1)]
                          for tr in H.TRAJECTORIES] for ck in P.CHECKPOINTS])       # [C, 3, 20]
    for vi, name in enumerate(H.VIEWS):
        tv = S.view(T, name)                       # [21, 4, d]
        xv = S.view(XS, name)                      # [C, 3, 21, 4, d]
        fv = S.view(FS, name)                      # [C, 2, 20, 4, d]
        nv = S.view(NSt, name)                     # [C, 2, 20, 4, d]
        yv = S.view(YS, name)                      # [C, 3, NB+1, 4, d]
        t1 = tv[1:]
        auto[:, :, :, vi] = M.base_sums(xv[:, :, 1:], np.broadcast_to(t1, xv[:, :, 1:].shape))[0]
        tf[:, :, :, vi] = M.base_sums(fv, np.broadcast_to(t1, fv.shape))[0]
        # D3: per trajectory, the operator actually used at step h
        fsel = np.take_along_axis(fv[:, None].repeat(3, axis=1), op_index[:, :, None, :, None, None], axis=2)[:, :, 0]
        L = fsel - t1                              # [C, 3, 20, 4, d]
        e = xv[:, :, 1:] - t1
        Pp = xv[:, :, 1:] - fsel
        d3[:, :, :, vi, :, 0] = _dot(L, L)
        d3[:, :, :, vi, :, 1] = _dot(L, e)
        d3[:, :, :, vi, :, 2] = _dot(Pp, Pp)
        # restarts
        e20 = yv[:, :, 0] - yv[:, :, NB]           # [C, 3, 4, d]
        for b in range(NB):
            D = yv[:, :, b] - yv[:, :, b + 1]
            ye = yv[:, :, b] - yv[:, :, NB]
            rst[:, :, b, vi, :, 0] = _dot(D, D)
            rst[:, :, b, vi, :, 1] = _dot(D, e20)
            rst[:, :, b, vi, :, 2] = _dot(ye, ye)
        # controller displacement along the controlled trajectories
        for qi, ti in enumerate((1, 2)):
            Kd = xv[:, ti, 1:] - nv[:, qi]         # [C, 20, 4, d]
            eh = xv[:, ti, 1:] - t1
            ne = nv[:, qi] - t1
            ctrl[:, qi, :, vi, :, 0] = _dot(Kd, Kd)
            ctrl[:, qi, :, vi, :, 1] = _dot(Kd, eh)
            ctrl[:, qi, :, vi, :, 2] = _dot(Kd, np.broadcast_to(t1, Kd.shape))
            ctrl[:, qi, :, vi, :, 3] = _dot(ne, ne)
        # teacher: energy, persistence baseline, pair conditioning
        tch[:, vi, :, 0] = _dot(tv, tv)
        tch[0, vi, :, 1] = 0.0
        dT = tv[1:] - tv[:-1]
        tch[1:, vi, :, 1] = _dot(dT, dT)
        parts = S.view_parts(T, name)
        tch[:, vi, :, 2] = 1.0 if parts is None else M.pair_condition(*parts)
        if parts is not None:
            a, b = S.view_parts(XS[:, :, 1:], name)
            pcond[:, :, :, 0 if name == "READOUT" else 1] = M.pair_condition(a, b)
    out.update(AUTO=auto, TF=tf, D3=d3, RST=rst, CTRL=ctrl, TCH=tch, PCOND=pcond)
    return out, XS


def run(cfg, oracle, progress=print):
    """Execute the measurement pass. ``cfg`` keys: graphs (ordered {name: graph}), structure_seed,
    evaluation_seeds, cases, k_values, scales (ordered {label: value}), ratio, batch,
    deep_check_stride, reference_teacher_steps."""
    t_start = time.monotonic()
    shp = shapes(cfg)
    raw = {name: np.full(s, np.nan) for name, s in shp.items()}
    counters = {k: 0 for k in (
        "fixtures", "nonfinite", "autonomous_rechecks", "autonomous_recheck_mismatch", "delayed_checks",
        "delayed_spliced_vs_full_mismatch", "delayed_h8_vs_native_mismatch", "fresh_state_checks",
        "fresh_state_mismatch", "exact_r118_checks", "exact_r118_mismatch", "teacher_k0_checks",
        "teacher_k0_bit_mismatch", "reference_teacher_checks", "engine_steps", "exact_audit_samples",
        "exact_identity_nonzero")}
    maxima = {"teacher_k0_vs_historical_max_abs": 0.0, "reference_teacher_max_abs": 0.0,
              "reference_teacher_max_rel": 0.0, "exact_audit_max_relative": 0.0}
    names = list(cfg["graphs"])
    scales = list(cfg["scales"].items())
    seeds = list(cfg["evaluation_seeds"])
    cases = list(cfg["cases"])
    torch = oracle.torch
    for gi, name in enumerate(names):
        graph = cfg["graphs"][name]
        systems, parts = H.build_systems(oracle, graph, cfg["structure_seed"] + gi, ks=tuple(cfg["k_values"]))
        _, adj = oracle.graph_adjs(graph)
        for ki, k in enumerate(cfg["k_values"]):
            S = systems[k]
            for si, (slabel, scale) in enumerate(scales):
                t_cell = time.monotonic()
                for ci_case, case in enumerate(cases):
                    for qi, seed in enumerate(seeds):
                        f = ci_case * len(seeds) + qi
                        st0 = oracle.r64.pair_state(seed, case, cfg["ratio"], batch=cfg["batch"])
                        A0, B0 = st0[0].numpy(), st0[1].numpy()
                        X0 = S.x0(A0, B0, scale)
                        T = S.teacher(oracle, X0)
                        deep = (f % cfg["deep_check_stride"] == 0)
                        res, XS = _fixture(S, oracle, X0, T, f, counters, maxima, deep)
                        for key, arr in res.items():
                            raw[key][gi, ki, si, f] = arr
                        counters["fixtures"] += 1
                        if k == 0:
                            # composed teacher at k = 0 must equal the historical isolated teacher rollout
                            with torch.no_grad():
                                hist = oracle.r64.rollout((st0[0] * scale, st0[1] * scale), adj,
                                                          oracle.r30.NonlinearTeacher(torch), H.HMAX,
                                                          keep=tuple(range(1, H.HMAX + 1)))
                            for h in range(1, H.HMAX + 1):
                                w = T[h][S.fidx]                       # [8, 4, 8]
                                ref = np.concatenate([hist[h][0].numpy(), hist[h][1].numpy()], axis=-1)
                                maxima["teacher_k0_vs_historical_max_abs"] = max(
                                    maxima["teacher_k0_vs_historical_max_abs"], float(np.abs(w - ref).max()))
                                counters["teacher_k0_bit_mismatch"] += int(
                                    not np.array_equal(w.view(np.uint64), ref.view(np.uint64)))
                                counters["teacher_k0_checks"] += 1
                        if f == 0:
                            # independent (torch-free) teacher reference along the oracle trajectory
                            for h in cfg["reference_teacher_steps"]:
                                got = H.reference_teacher_step(S, T[h - 1])
                                dev = float(np.abs(got - T[h]).max())
                                maxima["reference_teacher_max_abs"] = max(maxima["reference_teacher_max_abs"], dev)
                                maxima["reference_teacher_max_rel"] = max(
                                    maxima["reference_teacher_max_rel"], dev / float(np.abs(T[h]).max()))
                                counters["reference_teacher_checks"] += 1
                            # fresh volatile state and frozen R118 kernel must reproduce the reused-state rollouts
                            fresh = H.System(*parts, k)
                            for ci, ck in ((0, P.CHECKPOINTS[0]), (4, P.CHECKPOINTS[4])):
                                for ti, pol in enumerate(H.TRAJECTORIES):
                                    a = fresh.rollout(X0, ck, pol)
                                    counters["fresh_state_mismatch"] += int(
                                        not np.array_equal(a.view(np.uint64), XS[ci, ti].view(np.uint64)))
                                    counters["fresh_state_checks"] += 1
                                    b = fresh.rollout(X0, ck, pol, mode="EXACT_R118")
                                    counters["exact_r118_mismatch"] += int(
                                        not np.array_equal(b.view(np.uint64), XS[ci, ti].view(np.uint64)))
                                    counters["exact_r118_checks"] += 1
                progress(f"R122_CELL graph={name} ({gi + 1}/{len(names)}) k={k} scale={slabel} "
                         f"fixtures={counters['fixtures']} cell_s={time.monotonic() - t_cell:.1f} "
                         f"elapsed_s={time.monotonic() - t_start:.0f}")
            counters["engine_steps"] += S.engine_steps
    raw_nan = {k: int(np.isnan(v).sum()) for k, v in raw.items()}
    return raw, {"counters": counters, "maxima": maxima, "raw_nan_entries": raw_nan,
                 "seconds": round(time.monotonic() - t_start, 2)}
