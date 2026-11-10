"""R119 qualification: executes PROTOCOL_R119.json (registered before measurement).

    python -B -m research.yang_mills_hca.r119_numerical_hotpath.qualify_r119 --all --out PATH

Parts (each also runnable alone): ``--controlled`` and ``--organic`` (latency + bit identity
+ deviations; torch never imported), ``--numerical`` (kernel differential campaign,
historical replay through the substrate against torch, covariance per mode and scale).
``--all`` runs each part in its own subprocess (clean peak RSS) and applies the
registered classification and decision rules.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import bench as B
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.awr_state import AWRState, StateCapacity
from research.yang_mills_hca.r118_scalable_awr.fixtures import bulk_synthetic_registry, extract_window

from .engine_r119 import R119Engine

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.join(HERE, "PROTOCOL_R119.json")
MODES = ("EXACT_R118", "EXACT_GUARDED", "FTZ_DAZ")
FLUSH = 2.0 ** -500
TOL = 2.0 ** -1000


def protocol():
    raw = open(PROTOCOL, "rb").read()
    return json.loads(raw), P.sha256_bytes(raw)


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pct(v, p):
    s = np.sort(np.asarray(v))
    return float(s[max(0, int(np.ceil(p / 100.0 * s.size)) - 1)])


def stats(v):
    v = np.asarray(v)
    return {"n": int(v.size), "p50_ms": pct(v, 50), "p95_ms": pct(v, 95), "p99_ms": pct(v, 99), "max_ms": float(v.max())}


def build_states(cfg, copies):
    """R118-D ``bench.build`` with the registry shared by ``copies`` identical states."""
    reg = bulk_synthetic_registry(cfg["participants"], cfg["relations"], cfg["seed"], arity=tuple(cfg["arity"]),
                                  zipf=cfg["zipf"])
    n = cfg["instances"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    seeds = rng.choice(reg.num_relations, size=n, replace=False)
    rel = np.zeros((n, 4), dtype=np.uint64)
    mask = np.zeros((n, 4), dtype=bool)
    for i, s in enumerate(seeds.tolist()):
        w = extract_window(reg, s, max_incidences=65536)
        perm = rng.permutation(4)[:len(w)]
        rel[i, perm] = reg.relation_ids[w]
        mask[i, perm] = True
    states = []
    for _ in range(copies):
        st = AWRState(reg, StateCapacity(max_instances=n, max_materialized_relations=4 * n,
                                         max_bindings_per_relation=1024))
        st.create_instances(np.arange(n, dtype=np.uint64) * 2654435761 + 17, rel, mask)
        rows = np.nonzero(st.row_ref > 0)[0]
        st.pool[rows] = np.random.default_rng(118500).normal(0.0, 1.0, size=(rows.size, 8))
        states.append(st)
    for st in states[1:]:
        assert np.array_equal(st.pool.view(np.uint64), states[0].pool.view(np.uint64))
        assert np.array_equal(st.slot_row, states[0].slot_row)
    return reg, states


def bitdiff(a, b):
    return int(np.sum(a.view(np.uint64) != b.view(np.uint64)))


# ---------------------------------------------------------------------------------- controlled
def run_controlled() -> dict:
    proto, psha = protocol()
    r118, _ = B.protocol()
    cfg = r118["registries"]["STRESS1M"]
    c = proto["controlled_regimes"]
    t0 = time.time()
    _, (st,) = build_states(cfg, 1)
    base = st.pool.copy()
    engines = {m: R119Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=B.HOT, mode=m) for m in MODES}
    live = np.nonzero(st.inst_live)[0].astype(np.int32)
    out = {"protocol_sha256": psha, "registry": "STRESS1M", "build_s": round(time.time() - t0, 1), "cells": []}
    for ri, (name, scale) in enumerate(c["scales"].items()):
        if name == "MIXED":
            e = np.random.default_rng(119500).uniform(-320.0, 0.0, size=(base.shape[0], 1))
            snap = base * 10.0 ** e
        else:
            snap = base * float(scale)
        st.pool[:] = snap
        for si, A in enumerate(c["active_sizes"]):
            rng = np.random.default_rng(119000 + 10 * ri + si)
            seq = [np.sort(rng.choice(live, size=A, replace=False)) for _ in range(c["warmup_steps"] + c["measured_steps"])]
            lat = {m: [] for m in MODES}
            mism = fails = 0
            ftz_dev = flush_dev = 0.0
            ftz_bits = 0
            sub_out = 0
            for i, act in enumerate(seq):
                rows = st.slot_row[act]
                rows = np.unique(rows[rows >= 0])
                outs = {}
                for m in MODES:
                    st.pool[rows] = snap[rows]
                    t1 = time.perf_counter_ns()
                    r = engines[m].step(act)
                    dt = (time.perf_counter_ns() - t1) / 1e6
                    fails += r.disposition != "EXECUTED"
                    outs[m] = st.pool[rows].copy()
                    if i >= c["warmup_steps"]:
                        lat[m].append(dt)
                st.pool[rows] = snap[rows]
                ref = outs["EXACT_R118"]
                mism += bitdiff(ref, outs["EXACT_GUARDED"])
                ftz_bits += bitdiff(ref, outs["FTZ_DAZ"])
                ftz_dev = max(ftz_dev, float(np.abs(outs["FTZ_DAZ"] - ref).max()))
                small = np.abs(ref) < FLUSH
                flush_dev = max(flush_dev, float(np.abs(ref[small]).max()) if small.any() else 0.0)
                sub_out += int(np.sum((ref != 0) & (np.abs(ref) < 2.2250738585072014e-308)))
            cell = {"regime": name, "scale": scale, "A": A, "failures": fails,
                    "guarded_vs_r118_bit_mismatches": mism, "ftz_vs_r118_bit_mismatches": ftz_bits,
                    "ftz_max_abs_deviation": ftz_dev, "state_flush_max_abs_deviation": flush_dev,
                    "subnormal_output_coords_exact": sub_out,
                    "latency": {m: stats(lat[m]) for m in MODES}}
            out["cells"].append(cell)
            print(f"[R119] controlled {name:10s} A={A:5d} " + " ".join(
                f"{m}:p95={cell['latency'][m]['p95_ms']:.3f}" for m in MODES) + f" mism={mism}", flush=True)
    out["guard_report"] = engines["EXACT_GUARDED"].guard_report()
    out["theta"] = engines["EXACT_GUARDED"].theta
    out["libm_selfcheck"] = engines["EXACT_GUARDED"].libm_selfcheck
    out["kernels"] = {m: engines[m].kernel_info for m in MODES}
    out["scratch_bytes"] = engines["EXACT_GUARDED"].scratch_bytes
    out["peak_rss_mib"] = rss_mib()
    out["torch_imported"] = "torch" in sys.modules
    return out


# ------------------------------------------------------------------------------------ organic
def _organic(cfg, act_sizes, fixed_all, steps, block):
    reg, states = build_states(cfg, len(MODES) + 1)
    sts = dict(zip(MODES + ("STATE_FLUSH",), states))
    engines = {m: R119Engine(sts[m], "54007/STATE8", controller="R104", budgets=B.HOT, mode=m) for m in MODES}
    engines["STATE_FLUSH"] = R119Engine(sts["STATE_FLUSH"], "54007/STATE8", controller="R104", budgets=B.HOT,
                                        mode="EXACT_R118")
    live = np.nonzero(states[0].inst_live)[0].astype(np.int32)
    init = states[0].pool.copy()  # registered N(0,1) initial state (seed 118500)
    res = []
    for A in act_sizes:
        for st in states:
            st.pool[:] = init  # every trajectory starts from the registered initial state
        act = live if fixed_all else np.sort(np.random.default_rng(119100 + A).choice(live, size=A, replace=False))
        rows = states[0].slot_row[act]
        rows = np.unique(rows[rows >= 0])
        lat = {m: [] for m in engines}
        mism = fails = ftz_bits = 0
        ftz_dev = flush_dev = 0.0
        trace = []
        for step in range(1, steps + 1):
            for m, eng in engines.items():
                t1 = time.perf_counter_ns()
                r = eng.step(act)
                lat[m].append((time.perf_counter_ns() - t1) / 1e6)
                fails += r.disposition != "EXECUTED"
                if m == "STATE_FLUSH":
                    p = sts[m].pool
                    sel = p[rows]
                    sel[np.abs(sel) < FLUSH] = 0.0
                    p[rows] = sel
            ref = sts["EXACT_R118"].pool[rows]
            mism += bitdiff(ref, sts["EXACT_GUARDED"].pool[rows])
            ftz_bits += bitdiff(ref, sts["FTZ_DAZ"].pool[rows])
            ftz_dev = max(ftz_dev, float(np.abs(sts["FTZ_DAZ"].pool[rows] - ref).max()))
            flush_dev = max(flush_dev, float(np.abs(sts["STATE_FLUSH"].pool[rows] - ref).max()))
            if step % block == 0:
                v = np.abs(ref)
                trace.append({"step": step, "state_abs_max": float(v.max()),
                              "state_abs_median_nonzero": float(np.median(v[v > 0])) if (v > 0).any() else 0.0})
        blocks = {m: [pct(lat[m][i:i + block], 95) for i in range(0, steps, block)] for m in engines}
        res.append({"A": int(act.size), "steps": steps, "failures": fails, "guarded_vs_r118_bit_mismatches": mism,
                    "ftz_vs_r118_bit_mismatches": ftz_bits,
                    "ftz_max_abs_deviation": ftz_dev, "state_flush_max_abs_deviation": flush_dev,
                    "latency": {m: dict(stats(lat[m]), max_block_p95_ms=max(blocks[m]), block_p95_ms=blocks[m])
                                for m in engines},
                    "trace": trace})
        print(f"[R119] organic A={act.size:5d} " + " ".join(
            f"{m}:maxblockp95={max(blocks[m]):.2f}" for m in MODES) + f" mism={mism}", flush=True)
    return res


def run_organic() -> dict:
    proto, psha = protocol()
    r118, _ = B.protocol()
    o = proto["organic_trajectories"]
    t0 = time.time()
    out = {"protocol_sha256": psha}
    out["STRESS1M"] = _organic(r118["registries"]["STRESS1M"], (64, 256, 1024), False, 4000, o["block"])
    out["SMALL"] = _organic(r118["registries"]["SMALL"], (1024,), True, 4000, o["block"])
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_rss_mib"] = rss_mib()
    out["torch_imported"] = "torch" in sys.modules
    return out


# ---------------------------------------------------------------------------------- numerical
def run_numerical() -> dict:
    from research.yang_mills_hca.r118_scalable_awr import gauge as G
    from research.yang_mills_hca.r118_scalable_awr.engine import Budgets
    from research.yang_mills_hca.r118_scalable_awr.fixtures import historical_registry, synthetic_registry
    from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
    from research.yang_mills_hca.r118_scalable_awr.registry import RegistryLimits
    from research.yang_mills_hca.r118_scalable_awr.tests._util import random_windows

    from . import differential as D

    proto, psha = protocol()
    out = {"protocol_sha256": psha}
    # 1. kernel differential campaign
    rng = np.random.default_rng(119200)
    fams = {"connected": synthetic_registry(300, 1200, 17, arity=(2, 5)),
            "zipf": synthetic_registry(200, 700, 18, arity=(1, 8), zipf=1.3),
            "hub16": synthetic_registry(80, 300, 19, arity=(8, 16), hub_fraction=0.9, limits=RegistryLimits(max_arity=16))}
    comps = mism_g = mism_u = 0
    hits = np.zeros(8, dtype=np.int64)
    for fname, reg in fams.items():
        for ck in P.CHECKPOINTS:
            for ctrl in ("NATIVE", "R104", "R114"):
                dif = D.Differential(reg, ck, ctrl)
                for mode in ("uniform_scale", "mixed_rows", "mixed_coords", "zeros_and_signs", "subnormal", "threshold"):
                    for _ in range(4):
                        slot_rel, slot_cmp, nr = D.windows_for(reg, 32, rng)
                        cin = D.threshold_states(rng, nr, dif.theta) if mode == "threshold" else D.random_states(rng, nr, mode)
                        rc, mb, mu = dif.compare(slot_rel, slot_cmp, cin)
                        comps += 1
                        mism_g += mb if rc == 0 else 10 ** 9
                        mism_u += mu if rc == 0 else 10 ** 9
                hits += dif.gc
    out["differential"] = {"comparisons": comps, "instances_per_comparison": 32, "guarded_bit_mismatches": mism_g,
                           "unguarded_bit_mismatches": mism_u,
                           "guard_hits": hits[:4].tolist(), "guard_evaluations": hits[4:].tolist()}
    # 2. historical replay through the substrate, all modes, against torch
    o = get_oracle()
    T = o.torch
    graphs = dict(o.r61["r55_structural"]["graphs"])
    graphs.update(o.r77.CFG["graphs"])
    st0 = o.r64.pair_state(53039, "anti", 160, batch=4)
    Bn = st0[0].shape[0]
    hr = {"cases": 0, "guarded_vs_r118_bit_mismatches": 0, "ftz_vs_r118_bit_mismatches": 0,
          "guarded_vs_torch_h1_max_abs": 0.0, "guarded_vs_torch_h20_max_abs": 0.0}
    bud = Budgets(max_passes=20)
    for gname, graph in sorted(graphs.items()):
        for mode_name, idx in (("BASE_SEPARABLE", 0), ("PARTICIPANT_NONFLAT", 1)):
            reg, windows = historical_registry(o, graph, mode=mode_name, copies=Bn)
            adj = o.graph_adjs(graph)[idx]
            for key in P.CHECKPOINTS:
                with T.no_grad():
                    hist = o.r64.rollout(st0, adj, o.models()[key], 20, keep=(1, 20))
                for ctrl in ("NATIVE", "R114"):
                    pools = {}
                    for m in MODES:
                        st = AWRState(reg, StateCapacity(max_instances=Bn, max_materialized_relations=4 * Bn))
                        hs = np.array([st.create_instance(1000 + c, w)[0] for c, w in enumerate(windows)])
                        for c, w in enumerate(windows):
                            for d in range(4):
                                st.set_relation_state(w[d], np.concatenate([st0[0][c, d].numpy(), st0[1][c, d].numpy()]))
                        eng = R119Engine(st, key, controller=ctrl, budgets=bud, mode=m)
                        assert eng.step(hs).exact
                        h1 = st.pool.copy()
                        assert eng.step(hs, passes=19).exact
                        pools[m] = (h1, st.pool.copy(), st)
                    for hi in (0, 1):
                        hr["guarded_vs_r118_bit_mismatches"] += bitdiff(pools["EXACT_R118"][hi], pools["EXACT_GUARDED"][hi])
                        hr["ftz_vs_r118_bit_mismatches"] += bitdiff(pools["EXACT_R118"][hi], pools["FTZ_DAZ"][hi])
                    if ctrl == "NATIVE":
                        st = pools["EXACT_GUARDED"][2]
                        for hi, H in ((0, 1), (1, 20)):
                            pool = pools["EXACT_GUARDED"][hi]
                            a = np.array([[pool[st.rel_row[st.registry.relation_index([w[d]])[0]], :4] for d in range(4)]
                                          for w in windows])
                            b = np.array([[pool[st.rel_row[st.registry.relation_index([w[d]])[0]], 4:] for d in range(4)]
                                          for w in windows])
                            dv = max(np.abs(a - hist[H][0].numpy()).max(), np.abs(b - hist[H][1].numpy()).max())
                            k = "guarded_vs_torch_h1_max_abs" if H == 1 else "guarded_vs_torch_h20_max_abs"
                            hr[k] = max(hr[k], float(dv))
                hr["cases"] += 1
    out["historical_replay"] = hr
    # 3. covariance per mode and scale, overlapping windows
    reg = synthetic_registry(120, 400, 44, arity=(2, 7), zipf=1.1)
    rng = np.random.default_rng(118)
    g = G.random_so2x2(rng, reg.num_relations)
    h = G.random_so2x2(rng, reg.num_participants)
    rel, mask = random_windows(reg, 64, 9)
    reg_g, reg_h = G.gauge_registry(reg, g_rel=g), G.gauge_registry(reg, h_part=h)
    cov = {}
    for scale in (1.0, 1e-104, 1e-157, 1e-300, 1e-310):
        for m in MODES:
            def make(registry):
                st = AWRState(registry, StateCapacity(max_instances=64, max_materialized_relations=256,
                                                      max_bindings_per_relation=256))
                hh, _ = st.create_instances(np.arange(64, dtype=np.uint64) + 1, rel, mask)
                rows = np.nonzero(st.row_ref > 0)[0]
                st.pool[rows] = np.random.default_rng(218).normal(0.0, 1.0, size=(rows.size, 8)) * scale
                return st, hh
            base, hb = make(reg)
            init = base.pool.copy()
            gst, hg = make(reg_g)
            gst.pool[:] = G.gauge_rows(base, g)
            pst, hp = make(reg_h)
            neg, hn = make(reg)
            neg.pool[:] = G.gauge_rows(base, g)
            for st_, hh in ((base, hb), (gst, hg), (pst, hp), (neg, hn)):
                assert R119Engine(st_, "54007/STATE8_SIGNED8", controller="R114", budgets=bud, mode=m).step(hh, passes=6).exact
            expect = G.gauge_rows(base, g)
            sc = float(np.abs(expect).max()) or 1.0
            cov[f"{m}/{scale:g}"] = {
                "relation_gauge_abs": float(np.abs(gst.pool - expect).max()),
                "relation_gauge_rel": float(np.abs(gst.pool - expect).max()) / sc,
                "participant_gauge_abs": float(np.abs(pst.pool - base.pool).max()),
                "participant_gauge_rel": float(np.abs(pst.pool - base.pool).max()) / sc,
                "negative_state_only_rel": float(np.abs(neg.pool - expect).max()) / sc,
                "state_scale": sc, "initial_scale": scale}
            del init
    out["covariance"] = cov
    out["weights_unchanged"] = o.parameter_pins() == P.CHECKPOINT_SHA256 and all(
        P.checkpoint_hash(r["weights"], r["bias"]) == P.CHECKPOINT_SHA256[k] for k, r in P.checkpoint_records().items())
    out["torch"] = T.__version__
    return out


# --------------------------------------------------------------------------------- decision
def decide(proto, controlled, organic, numerical) -> dict:
    targets = {int(k): v for k, v in proto["inherited_targets"]["p95_ms"].items()}
    lat = {m: {} for m in MODES}
    for cell in controlled["cells"]:
        for m in MODES:
            ok = cell["latency"][m]["p95_ms"] <= targets[cell["A"]] and cell["failures"] == 0
            lat[m][f"controlled/{cell['regime']}/A{cell['A']}"] = ok
    for name in ("STRESS1M", "SMALL"):
        for traj in organic[name]:
            A = traj["A"]
            for m in MODES:
                ok = traj["latency"][m]["max_block_p95_ms"] <= targets[A] and traj["failures"] == 0
                lat[m][f"organic/{name}/A{A}"] = ok
    latency_qualified = {m: all(v.values()) for m, v in lat.items()}
    exact_guarded = (all(c["guarded_vs_r118_bit_mismatches"] == 0 for c in controlled["cells"])
                     and all(t["guarded_vs_r118_bit_mismatches"] == 0 for n in ("STRESS1M", "SMALL") for t in organic[n])
                     and numerical["differential"]["guarded_bit_mismatches"] == 0
                     and numerical["historical_replay"]["guarded_vs_r118_bit_mismatches"] == 0
                     and numerical["historical_replay"]["guarded_vs_torch_h1_max_abs"] <= 1e-12
                     and numerical["historical_replay"]["guarded_vs_torch_h20_max_abs"] <= 1e-10)
    cov = numerical["covariance"]

    def tol_class(dev_key, bits_key):
        normal_bits = all(c[bits_key] == 0 for c in controlled["cells"] if c["regime"] in ("NORMAL", "NEARZERO")) \
            if bits_key else all(c["state_flush_max_abs_deviation"] == 0 for c in controlled["cells"]
                                 if c["regime"] in ("NORMAL", "NEARZERO"))
        dev = max([c[dev_key] for c in controlled["cells"]] +
                  [t[dev_key] for n in ("STRESS1M", "SMALL") for t in organic[n]])
        return normal_bits, dev

    ftz_normal_bits, ftz_dev = tol_class("ftz_max_abs_deviation", "ftz_vs_r118_bit_mismatches")
    flush_normal_bits, flush_dev = tol_class("state_flush_max_abs_deviation", None)
    ftz_cov = all(v["relation_gauge_abs"] <= 1e-9 and v["participant_gauge_abs"] <= 1e-9
                  for k, v in cov.items() if k.startswith("FTZ_DAZ/"))
    weights = numerical["weights_unchanged"]
    classes = {
        "EXACT_R118": "EXACT (reference)",
        "EXACT_GUARDED": "EXACT" if exact_guarded and weights else "NONQUALIFIED",
        "FTZ_DAZ": ("EXACT" if (all(c["ftz_vs_r118_bit_mismatches"] == 0 for c in controlled["cells"])
                                and all(t["ftz_vs_r118_bit_mismatches"] == 0 for n in ("STRESS1M", "SMALL")
                                        for t in organic[n])
                                and numerical["historical_replay"]["ftz_vs_r118_bit_mismatches"] == 0)
                    else "TOLERANCE_QUALIFIED" if (ftz_normal_bits and ftz_dev <= TOL and ftz_cov and weights)
                    else "NONQUALIFIED"),
        "STATE_FLUSH_2P-500": ("TOLERANCE_QUALIFIED" if (flush_normal_bits and flush_dev <= TOL and weights)
                               else "NONQUALIFIED"),
    }
    closes = classes["EXACT_GUARDED"] == "EXACT" and latency_qualified["EXACT_GUARDED"]
    return {
        "latency_checks": lat, "latency_qualified": latency_qualified, "numerical_class": classes,
        "ftz_max_abs_deviation": ftz_dev, "state_flush_max_abs_deviation": flush_dev, "tolerance": TOL,
        "ftz_bit_identical_in_normal_and_nearzero": ftz_normal_bits, "ftz_covariance_abs_ok": ftz_cov,
        "r118d_numerical_exception": "CLOSED_FOR_DEFAULT_EXACT_PATH" if closes else "OPEN",
        "default_mode": "EXACT_GUARDED" if closes else "EXACT_R118",
        "mode_dispositions": {
            "EXACT_R118": "EXACT; " + ("LATENCY_QUALIFIED" if latency_qualified["EXACT_R118"] else "LATENCY_NONQUAL"),
            "EXACT_GUARDED": classes["EXACT_GUARDED"] + "; " + (
                "LATENCY_QUALIFIED" if latency_qualified["EXACT_GUARDED"] else "LATENCY_NONQUAL"),
            "FTZ_DAZ": "OPT_IN; " + classes["FTZ_DAZ"] + "; " + (
                "LATENCY_QUALIFIED" if latency_qualified["FTZ_DAZ"] else "LATENCY_NONQUAL"),
            "STATE_FLUSH_2P-500": classes["STATE_FLUSH_2P-500"] + " (negative control)",
        },
        "r93": "ENERGY_NORMALIZED_DENOMINATOR_HOLD", "r118e": "DESCRIPTIVE_ONLY",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--controlled", "--organic", "--numerical", "--all"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.controlled:
        res = run_controlled()
    elif args.organic:
        res = run_organic()
    elif args.numerical:
        res = run_numerical()
    else:
        proto, psha = protocol()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        parts = {}
        with P.volatile_scratch("ym_r119_parts_") as tmp:
            for part in ("controlled", "organic", "numerical"):
                path = str(tmp / (part + ".json"))
                p = subprocess.run([sys.executable, "-B", "-m", __spec__.name, "--" + part, "--out", path], env=env,
                                   capture_output=True, text=True)
                if p.returncode != 0:
                    raise SystemExit(f"PART_FAILED:{part}:{p.stderr[-3000:]}")
                with open(path) as f:
                    parts[part] = json.load(f)
        res = {"phase": "R119", "protocol_sha256": psha, **parts,
               "decision": decide(proto, parts["controlled"], parts["organic"], parts["numerical"]),
               "environment": P.environment_fingerprint(),
               "r118_component_digest": P.component_digest(), "r119_sources": sources()}
    blob = json.dumps(res, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0


def sources() -> dict:
    out = {}
    for root, _, files in os.walk(HERE):
        for name in sorted(files):
            if name.endswith((".py", ".c", ".json")) and "evidence" not in root:
                path = os.path.join(root, name)
                out[os.path.relpath(path, HERE)] = P.sha256_file(path)
    return dict(sorted(out.items()))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
