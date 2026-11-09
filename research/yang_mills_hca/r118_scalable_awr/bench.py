"""R118-D performance measurement, executing PROTOCOL_R118D.json exactly.

    python -B -m research.yang_mills_hca.r118_scalable_awr.bench --all --out PATH

``--all`` runs one subprocess per registry (clean peak-RSS accounting), then the
torch oracle baseline in its own process (skipped, and reported as
NOT_MEASURED, when torch is unavailable), and applies the frozen admission rule.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from . import engine as E
from . import provenance as P
from .awr_state import AWRState, StateCapacity
from .fixtures import bulk_synthetic_registry, extract_window

PROTOCOL_PATH = P.COMPONENT / "PROTOCOL_R118D.json"
HOT = E.Budgets(max_active_instances=1024, max_active_relations=4096, max_active_memberships=1 << 16,
                max_links_per_pass=1 << 17, max_passes=8, max_work_units=1 << 22)


def protocol():
    raw = PROTOCOL_PATH.read_bytes()
    return json.loads(raw), P.sha256_bytes(raw)


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pct(sorted_vals, p):
    n = len(sorted_vals)
    return float(sorted_vals[max(0, math.ceil(p / 100.0 * n) - 1)])


def summarize(lat):
    s = np.sort(lat)
    return {"n": int(s.size), "p50_ms": pct(s, 50), "p95_ms": pct(s, 95), "p99_ms": pct(s, 99),
            "max_ms": float(s[-1]), "mean_ms": float(s.mean())}


def dist(x):
    x = np.asarray(x)
    return {"min": float(x.min()), "p50": float(np.percentile(x, 50)), "mean": float(x.mean()),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def build(cfg):
    t = {}
    t0 = time.perf_counter()
    reg = bulk_synthetic_registry(cfg["participants"], cfg["relations"], cfg["seed"], arity=tuple(cfg["arity"]),
                                  zipf=cfg["zipf"])
    t["registry_build_s"] = time.perf_counter() - t0
    n = cfg["instances"]
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"] + 1)
    seeds = rng.choice(reg.num_relations, size=n, replace=False)
    rel = np.zeros((n, 4), dtype=np.uint64)
    mask = np.zeros((n, 4), dtype=bool)
    for i, s in enumerate(seeds.tolist()):
        w = extract_window(reg, s, max_incidences=65536)
        perm = rng.permutation(4)[:len(w)]
        rel[i, perm] = reg.relation_ids[w]
        mask[i, perm] = True
    t["window_extraction_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    st = AWRState(reg, StateCapacity(max_instances=n, max_materialized_relations=4 * n,
                                     max_bindings_per_relation=1024))  # registered windows bind up to 411x
    st.create_instances(np.arange(n, dtype=np.uint64) * 2654435761 + 17, rel, mask)
    t["instance_creation_s"] = time.perf_counter() - t0
    rows = np.nonzero(st.row_ref > 0)[0]
    st.pool[rows] = np.random.default_rng(118500).normal(0.0, 1.0, size=(rows.size, 8))
    live = np.nonzero(st.inst_live)[0]
    fixture = {
        "participants": reg.num_participants, "relations": reg.num_relations, "memberships": reg.num_incidences,
        "instances": int(live.size), "materialized_relations": int(rows.size),
        "participant_degree": dist(reg.participant_degree()),
        "relation_arity": {str(a): int(c) for a, c in zip(*np.unique(np.diff(reg.rel_offsets), return_counts=True))},
        "window_bound_slots": {str(a): int(c) for a, c in zip(*np.unique(st.inst_bound[live], return_counts=True))},
        "window_links": dist(st.inst_links[live]),
        "relation_binding_overlap": dist(st.row_ref[rows]),
        "registry_bytes": reg.nbytes(), "state_bytes": st.nbytes()["total"],
    }
    return reg, st, t, fixture


def measure(eng, st, A, warmup, steps, seed, passes=1):
    live = np.nonzero(st.inst_live)[0].astype(np.int32)
    rng = np.random.default_rng(seed)
    seq = [np.sort(rng.choice(live, size=A, replace=False)) for _ in range(warmup + steps)]
    for i in range(warmup):
        eng.step(seq[i], passes=passes)
    lat = np.empty(steps)
    fails = 0
    links = 0
    for i in range(steps):
        a = seq[warmup + i]
        t0 = time.perf_counter_ns()
        r = eng.step(a, passes=passes)
        lat[i] = (time.perf_counter_ns() - t0) / 1e6
        fails += r.disposition != "EXECUTED"
        links += r.links
    out = summarize(lat)
    out.update({"failures": fails, "mean_links_per_step": links / steps})
    return out, seq


def allocations(eng, seq):
    """Untimed: transient peak bytes above the pre-step baseline, net retained bytes and blocks."""
    gc.collect()
    tracemalloc.start()
    peaks, retained, blocks = [], [], []
    for a in seq[:200]:
        base = tracemalloc.get_traced_memory()[0]
        b0 = sys.getallocatedblocks()
        tracemalloc.reset_peak()
        eng.step(a)
        cur, peak = tracemalloc.get_traced_memory()
        peaks.append(peak - base)
        retained.append(cur - base)
        blocks.append(sys.getallocatedblocks() - b0)
    tracemalloc.stop()
    return {"steps": len(peaks), "transient_peak_bytes_mean": float(np.mean(peaks)),
            "transient_peak_bytes_max": int(max(peaks)), "net_retained_bytes_mean": float(np.mean(retained)),
            "net_retained_bytes_sum": int(sum(retained)), "net_live_blocks_mean": float(np.mean(blocks))}


def parity(st, eng, seq, n=3):
    from . import simple_reference as SR
    from .tests._util import simple_model

    worst = 0.0
    for a in seq[:n]:
        relations, instances, state = simple_model(st, a)
        ref = SR.step(eng.W.tolist(), eng.b.tolist(), relations, instances, state,
                      [int(st.inst_id[h]) for h in a.tolist()], eng.checkpoint, eng.controller)
        assert eng.step(a).exact
        for rid, vec in ref.items():
            r = int(st.registry.relation_index([rid])[0])
            worst = max(worst, float(np.abs(np.array(vec) - st.pool[st.rel_row[r]]).max()))
    return worst


def run_registry(name) -> dict:
    proto, psha = protocol()
    cfg = proto["registries"][name]
    out = {"registry": name, "protocol_sha256": psha, "config": cfg}
    out["cold"] = {}
    reg, st, t, fixture = build(cfg)
    out["cold"].update(t)
    out["fixture"] = fixture
    t0 = time.perf_counter()
    from . import native

    native.build()
    out["cold"]["kernel_build_s"] = time.perf_counter() - t0
    out["rss_after_build_mib"] = rss_mib()
    configs = [dict(proto["primary_configuration"], role="primary")] + [dict(c, role="secondary")
                                                                         for c in proto["secondary_configurations"]]
    out["latency"] = []
    primary_seq = {}
    for c in configs:
        t0 = time.perf_counter()
        eng = E.Engine(st, c["checkpoint"], controller=c["controller"], budgets=HOT)
        build_s = time.perf_counter() - t0
        for A in proto["active_sizes"]:
            if A > st.live_count:
                continue
            res, seq = measure(eng, st, A, proto["warmup_steps"], proto["measured_steps"], 118400 + A, c["passes"])
            res.update({"A": A, "controller": c["controller"], "checkpoint": c["checkpoint"], "passes": c["passes"],
                        "role": c["role"], "engine_construction_s": build_s, "scratch_bytes": eng.scratch_bytes})
            out["latency"].append(res)
            if c["role"] == "primary":
                primary_seq[A] = seq
                out.setdefault("allocations", {})[str(A)] = allocations(eng, seq)
    out["rss_after_measurement_mib"] = rss_mib()
    if name != "SMALL":
        sweep_bud = E.Budgets(max_active_instances=st.live_count, max_active_relations=4 * st.live_count,
                              max_active_memberships=1 << 22, max_links_per_pass=1 << 24, max_passes=1,
                              max_work_units=1 << 25, max_scratch_bytes=1 << 31)
        eng = E.Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=sweep_bud)
        allh = np.nonzero(st.inst_live)[0]
        times, links, disp = [], 0, set()
        for i in range(12):
            t0 = time.perf_counter_ns()
            r = eng.step(allh)
            if i >= 2:
                times.append((time.perf_counter_ns() - t0) / 1e6)
            links, disp = r.links, disp | {r.disposition}
        out["full_sweep"] = {"instances": int(allh.size), "links": int(links), "dispositions": sorted(disp),
                             **summarize(np.array(times)), "scratch_bytes": eng.scratch_bytes}
    if name == "STRESS1M":
        eng = E.Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=HOT)
        out["parity_native_vs_simple_max_abs"] = parity(st, eng, primary_seq[64])
    out["environment"] = P.environment_fingerprint()
    out["torch_imported"] = "torch" in sys.modules
    return out


def oracle_baseline() -> dict:
    proto, _ = protocol()
    cfg = proto["registries"]["SMALL"]
    reg, st, _, _ = build(cfg)
    eng = E.Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=HOT, backend="oracle")
    live = np.nonzero(st.inst_live)[0].astype(np.int32)
    rng = np.random.default_rng(118464)
    seq = [np.sort(rng.choice(live, size=64, replace=False)) for _ in range(23)]
    for a in seq[:3]:
        eng.step(a)
    lat = []
    for a in seq[3:]:
        t0 = time.perf_counter_ns()
        assert eng.step(a).exact
        lat.append((time.perf_counter_ns() - t0) / 1e6)
    return {"A": 64, "controller": "R114", **summarize(np.array(lat))}


def admission(results, proto) -> dict:
    targets = {int(k): v for k, v in proto["admission_targets_p95_ms"].items()}
    prim = {(r["registry"], x["A"]): x for r in results for x in r["latency"] if x["role"] == "primary"}
    checks, worst = {}, {}
    for (reg, A), x in sorted(prim.items()):
        checks[f"{reg}/A{A}/p95<={targets[A]}ms"] = x["p95_ms"] <= targets[A] and x["failures"] == 0
    for A in targets:
        small = prim[("SMALL", A)]["p95_ms"]
        for big in ("INST50K", "STRESS1M"):
            ratio = prim[(big, A)]["p95_ms"] / small
            worst[f"{big}/A{A}"] = ratio
            checks[f"{big}/A{A}/capacity_ratio<=2.0"] = ratio <= 2.0
    par = [r.get("parity_native_vs_simple_max_abs") for r in results if r["registry"] == "STRESS1M"]
    checks["parity_on_fixture<=1e-12"] = bool(par) and par[0] <= 1e-12
    ok = all(checks.values())
    return {"checks": checks, "capacity_p95_ratios": worst,
            "disposition": "HOT_PATH_QUALIFIED" if ok else "PERFORMANCE_NONQUAL"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry")
    ap.add_argument("--oracle-baseline", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.registry:
        res = run_registry(args.registry)
    elif args.oracle_baseline:
        res = oracle_baseline()
    else:
        proto, psha = protocol()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        results = []
        for name in proto["registries"]:
            p = subprocess.run([sys.executable, "-B", "-m", __spec__.name, "--registry", name], env=env,
                               capture_output=True, text=True, check=True)
            results.append(json.loads(p.stdout))
        p = subprocess.run([sys.executable, "-B", "-m", __spec__.name, "--oracle-baseline"], env=env,
                           capture_output=True, text=True)
        oracle = json.loads(p.stdout) if p.returncode == 0 else {"status": "NOT_MEASURED", "stderr": p.stderr[-500:]}
        res = {"phase": "R118-D", "protocol_sha256": psha, "registries": results, "oracle_baseline": oracle,
               "admission": admission(results, proto)}
    blob = json.dumps(res, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
