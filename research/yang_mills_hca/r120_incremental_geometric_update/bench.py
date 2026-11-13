"""R120E performance measurement, executing PROTOCOL_R120U.json.

    python -B -m research.yang_mills_hca.r120_incremental_geometric_update.bench --all --out /dev/shm/r120_bench.json

``--all`` runs one subprocess per fixture. The measured quantity is the complete
update-and-propagate call (``IncrementalEngine.apply``): numpy event arrays in, committed
state out. The R119 EXACT_GUARDED step on the same affected window sets is measured
interleaved in the same process as the reference. No FTZ/DAZ mode is used.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import bench as B118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine

from . import engine_r120 as E
from . import native_r120 as N

HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = HERE / "PROTOCOL_R120U.json"
NS_FIELDS = ("ns_validate_update", "ns_select", "ns_hca", "ns_reconcile", "ns_commit", "ns_total")


def protocol():
    raw = PROTOCOL.read_bytes()
    return json.loads(raw), P.sha256_bytes(raw)


def build(cfg):
    """The R118 fixture builder, unmodified, producing an IncrementalState (same RNG stream, same fixture)."""
    orig = B118.AWRState
    B118.AWRState = E.IncrementalState
    try:
        return B118.build(cfg)
    finally:
        B118.AWRState = orig


def binding_csr(st):
    """row -> bound instance handles, built from slot_row (independent of the binding lists the kernel walks)."""
    live = np.nonzero(st.inst_live)[0]
    rows = st.slot_row[live].reshape(-1)
    inst = np.repeat(live, 4)
    keep = rows >= 0
    rows, inst = rows[keep], inst[keep]
    order = np.argsort(rows, kind="stable")
    rows, inst = rows[order], inst[order].astype(np.int32)
    off = np.zeros(st.capacity.max_materialized_relations + 1, dtype=np.int64)
    np.add.at(off, rows + 1, 1)
    np.cumsum(off, out=off)
    return off, inst


def make_batches(st, csr, A, count, seed):
    """Event batches whose radius-1 affected window set has exactly ``A`` windows."""
    reg = st.registry
    off, inst = csr
    rng = np.random.default_rng(seed)
    mat = np.nonzero(st.row_ref > 0)[0]
    mark = np.zeros(st.capacity.max_instances, dtype=bool)
    out = []
    for _ in range(count):
        chosen, handles, n_aff = [], [], 0
        tries = 0
        while n_aff < A:
            tries += 1
            if tries > 200:
                raise RuntimeError("WORKLOAD_GENERATION")
            for row in rng.choice(mat, size=min(mat.size, 4 * A), replace=False).tolist():
                b = inst[off[row]:off[row + 1]]
                new = b[~mark[b]]
                if new.size == 0 or n_aff + new.size > A:
                    continue
                mark[new] = True
                handles.append(new)
                chosen.append(row)
                n_aff += new.size
                if n_aff == A:
                    break
        handles = np.sort(np.concatenate(handles)).astype(np.int32)
        mark[handles] = False
        rows = np.array(chosen, dtype=np.int64)
        rel = st.row_rel[rows]
        lo = reg.rel_offsets[rel]
        ar = reg.rel_offsets[rel + 1] - lo
        parts = reg.members[lo + (rng.random(rows.size) * ar).astype(np.int64)].astype(np.int32)
        all_rows = st.slot_row[handles].reshape(-1)
        out.append({"ids": reg.relation_ids[rel].copy(), "parts": parts, "obs": rng.normal(size=(rows.size, 4)),
                    "weight": rng.uniform(-1.0, 1.0, rows.size), "aux": rng.uniform(0.0, 1.0, rows.size),
                    "rows": rows, "handles": handles, "all_rows": np.unique(all_rows[all_rows >= 0])})
    return out


def bits_equal(a, b):
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def measure_cell(st, eng, r119, batches, warmup, parity_n=10):
    """Interleaved: R120 apply | R119 step | R120 propagation-only step, all from the same pre-call state."""
    pool = st.pool
    steps = len(batches) - warmup
    lat = {k: np.empty(steps) for k in ("apply", "r119_step", "r120_step")}
    ns = {k: np.empty(steps) for k in NS_FIELDS}
    fails = step_mism = comp_mism = comp_n = 0
    events = links = rows_active = cached = 0
    for i, b in enumerate(batches):
        rows = b["all_rows"]
        seq = np.full(b["ids"].size, i + 1, dtype=np.uint64)
        save = pool[rows].copy()
        t0 = time.perf_counter_ns()
        rep = eng.apply(b["ids"], seq, b["parts"], b["obs"], b["weight"], b["aux"])
        t1 = time.perf_counter_ns()
        out = pool[rows].copy()
        pool[rows] = save
        t2 = time.perf_counter_ns()
        r = r119.step(b["handles"])
        t3 = time.perf_counter_ns()
        ref = pool[rows].copy()
        pool[rows] = save
        t4 = time.perf_counter_ns()
        rs = eng.step(b["handles"])
        t5 = time.perf_counter_ns()
        step_mism += int(not bits_equal(pool[rows], ref))
        if i < parity_n:
            # composition parity on the benchmark fixture: native law per event, then an R119 step
            pool[rows] = save
            for e, row in enumerate(b["rows"].tolist()):
                rel = int(st.row_rel[row])
                lo = int(st.registry.rel_offsets[rel])
                inc = lo + int(np.nonzero(st.registry.members_of(rel) == b["parts"][e])[0][0])
                new, rc = eng.update_probe(pool[row], st.registry.frames[inc], b["obs"][e], b["weight"][e], b["aux"][e])
                assert rc == 0
                pool[row] = new
            assert r119.step(b["handles"]).exact
            comp_n += 1
            comp_mism += int(not bits_equal(pool[rows], out))
        pool[rows] = out
        ok = rep.disposition == "OK" and rep.instances == b["handles"].size and r.exact and rs.disposition == "OK"
        if i >= warmup:
            j = i - warmup
            lat["apply"][j] = (t1 - t0) / 1e6
            lat["r119_step"][j] = (t3 - t2) / 1e6
            lat["r120_step"][j] = (t5 - t4) / 1e6
            for k in NS_FIELDS:
                ns[k][j] = getattr(rep, k)
            fails += int(not ok)
            events += b["ids"].size
            links += rep.links
            rows_active += rep.rows
            cached += rep.cached_instances
    res = {k: B118.summarize(v) for k, v in lat.items()}
    res["kernel_total"] = B118.summarize(ns["ns_total"] / 1e6)
    total = float(ns["ns_total"].mean())
    res["stage_mean_us"] = {k[3:]: float(ns[k].mean()) / 1e3 for k in NS_FIELDS}
    res["stage_share"] = {k[3:]: float(ns[k].mean()) / total for k in NS_FIELDS if k != "ns_total"}
    res["wrapper_overhead_mean_us"] = float(lat["apply"].mean() * 1e3 - total / 1e3)
    res.update({"failures": int(fails), "mean_events_per_batch": events / steps, "mean_links": links / steps,
                "mean_active_rows": rows_active / steps, "mean_cached_instances": cached / steps,
                "step_bit_mismatches_vs_r119": int(step_mism), "step_comparisons": len(batches),
                "composition_bit_mismatches": int(comp_mism), "composition_comparisons": int(comp_n)})
    return res


def measure_extra(st, eng, batches, n=200):
    """Untimed-state passes on the first ``n`` batches: zero-copy commit and the profiled CGC share."""
    pool = st.pool
    n = min(n, len(batches))
    lat = np.empty(n)
    cgc = np.empty(n)
    upd = np.empty(n)
    ev = eng.ev
    for i, b in enumerate(batches[:n]):
        rows = b["all_rows"]
        m = b["ids"].size
        save = pool[rows].copy()
        ev["rel_id"][:m], ev["seq"][:m], ev["part"][:m] = b["ids"], 1_000_000 + i, b["parts"]
        ev["obs"][:m], ev["weight"][:m], ev["aux"][:m], ev["exp_gen"][:m] = b["obs"], b["weight"], b["aux"], 0
        t0 = time.perf_counter_ns()
        rep = eng.commit(m)
        lat[i] = (time.perf_counter_ns() - t0) / 1e6
        assert rep.disposition == "OK"
        pool[rows] = save
        rep = eng.apply(b["ids"], np.full(m, 2_000_000 + i, np.uint64), b["parts"], b["obs"], b["weight"], b["aux"],
                        profile=True)
        cgc[i], upd[i] = rep.ns_cgc / m, rep.ns_validate_update / m
        pool[rows] = save
    return {"zero_copy_commit": B118.summarize(lat), "profiled_update_law_ns_per_event_mean": float(cgc.mean()),
            "profiled_validate_update_ns_per_event_mean": float(upd.mean()),
            "note": "the profiled pass adds two clock reads per event; its per-event figure is an upper bound"}


def measure_update_only(st, eng, sizes, warmup, steps, seed):
    """propagate=False: validation + CGC + in-place update + commit + pending bookkeeping."""
    reg = st.registry
    mat = np.nonzero(st.row_ref > 0)[0]
    out = {}
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        lat = np.empty(steps)
        fails = 0
        for i in range(warmup + steps):
            rows = rng.choice(mat, size=n, replace=False)
            rel = st.row_rel[rows]
            lo = reg.rel_offsets[rel]
            ar = reg.rel_offsets[rel + 1] - lo
            parts = reg.members[lo + (rng.random(n) * ar).astype(np.int64)].astype(np.int32)
            ids, obs = reg.relation_ids[rel], rng.normal(size=(n, 4))
            wt, aux = rng.uniform(-1.0, 1.0, n), rng.uniform(0.0, 1.0, n)
            seq = st.row_seq[rows] + np.uint64(1)
            save = st.pool[rows].copy()
            t0 = time.perf_counter_ns()
            rep = eng.apply(ids, seq, parts, obs, wt, aux, propagate=False)
            t1 = time.perf_counter_ns()
            fails += int(not (rep.deferred and rep.applied == n))
            st.pool[rows] = save                       # untimed: keep the state stationary, clear the pending marks
            eng._drop_pending(rows)
            if i >= warmup:
                lat[i - warmup] = (t1 - t0) / 1e6
        res = B118.summarize(lat)
        res.update({"events": n, "failures": int(fails), "p50_ns_per_event": res["p50_ms"] * 1e6 / n,
                    "p95_ns_per_event": res["p95_ms"] * 1e6 / n})
        out[str(n)] = res
    return out


def allocations(st, eng, batches):
    gc.collect()
    tracemalloc.start()
    peaks, retained, blocks = [], [], []
    for i, b in enumerate(batches[:200]):
        seq = np.full(b["ids"].size, 3_000_000 + i, dtype=np.uint64)
        base = tracemalloc.get_traced_memory()[0]
        b0 = sys.getallocatedblocks()
        tracemalloc.reset_peak()
        rep = eng.apply(b["ids"], seq, b["parts"], b["obs"], b["weight"], b["aux"])
        cur, peak = tracemalloc.get_traced_memory()
        del rep
        peaks.append(peak - base)
        retained.append(tracemalloc.get_traced_memory()[0] - base)
        blocks.append(sys.getallocatedblocks() - b0)
    tracemalloc.stop()
    return {"calls": len(peaks), "transient_peak_bytes_mean": float(np.mean(peaks)), "transient_peak_bytes_max": int(max(peaks)),
            "net_retained_bytes_sum": int(sum(retained)), "net_live_blocks_sum": int(sum(blocks)),
            "note": "transient bytes are the returned report snapshot and call temporaries; the kernel allocates nothing"}


def run_fixture(name, warmup=None, steps=None) -> dict:
    proto, psha = protocol()
    bp = proto["benchmark"]
    registered = warmup is None and steps is None
    cfg = {k: v for k, v in bp["fixtures"][name].items() if k != "note"}
    out = {"fixture_name": name, "protocol_sha256": psha, "config": cfg, "cold": {}, "registered_run": registered}
    reg, st, t, fixture = build(cfg)
    out["cold"].update(t)
    out["fixture"] = fixture
    t0 = time.perf_counter()
    lib, kinfo = N.kernel()
    out["cold"]["kernel_build_s"] = time.perf_counter() - t0
    out["kernel"] = kinfo
    out["openmp_max_threads"] = int(lib.awr120_openmp())
    init_pool = st.pool.copy()
    csr = binding_csr(st)
    warm = bp["warmup_steps"] if warmup is None else warmup
    steps = bp["measured_steps"] if steps is None else steps
    t0 = time.perf_counter()
    batches = {A: make_batches(st, csr, A, warm + steps, 120400 + A) for A in bp["affected_sizes"] if A <= st.live_count}
    out["cold"]["workload_generation_s"] = time.perf_counter() - t0
    ck = bp["checkpoint"]
    out["latency"] = []
    prim = bp["primary_configuration"]
    for controller in ("R114", "NATIVE"):
        r119 = R119Engine(st, ck, controller=controller, budgets=B118.HOT, mode="EXACT_GUARDED")
        for threads in (1, 4):
            st._engines.clear()
            t0 = time.perf_counter()
            eng = E.IncrementalEngine(st, ck, controller=controller, limits=E.Limits(threads=threads))
            build_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            cache = eng.build_cache()
            cache_s = time.perf_counter() - t0
            for A, bl in batches.items():
                st.pool[:] = init_pool
                st.row_seq[:] = 0
                res = measure_cell(st, eng, r119, bl, warm)
                res.update({"A": A, "controller": controller, "threads_requested": threads, "threads": eng.threads,
                            "role": "primary" if (name == prim["fixture"] and controller == prim["controller"]
                                                  and threads == prim["threads"]) else "secondary",
                            "engine_construction_s": build_s, "cache_build_s": cache_s, "cache": cache})
                tgt = bp["preferred_targets_p95_ms"][str(A)]
                res["preferred_target_p95_ms"] = tgt
                res["target_met"] = bool(res["failures"] == 0 and res["apply"]["p95_ms"] <= tgt)
                if threads == 1:
                    res["extra"] = measure_extra(st, eng, bl)
                out["latency"].append(res)
                print(f"R120E {name} {controller} threads={eng.threads} A={A}: apply p50/p95/p99 "
                      f"{res['apply']['p50_ms']:.3f}/{res['apply']['p95_ms']:.3f}/{res['apply']['p99_ms']:.3f} ms | "
                      f"R119 step {res['r119_step']['p50_ms']:.3f}/{res['r119_step']['p95_ms']:.3f} | "
                      f"mismatches {res['step_bit_mismatches_vs_r119']}+{res['composition_bit_mismatches']}", flush=True)
            if controller == "R114" and threads == 1:
                st.pool[:] = init_pool
                st.row_seq[:] = 0
                out["update_only"] = measure_update_only(st, eng, [n for n in (1, 16, 64, 256, 1024) if n <= eng.limits.max_events],
                                                         warm, steps, 120500)
                st.pool[:] = init_pool
                st.row_seq[:] = 0
                out["allocations"] = {str(A): allocations(st, eng, bl) for A, bl in batches.items()}
                side = {k: int(getattr(st, k).nbytes) for k in ("row_gen", "row_seq", "row_pending", "pending_list", "inst_mark")}
                out["memory"] = {"registry_bytes": reg.nbytes(), "state_bytes_r118_layout": st.nbytes()["total"],
                                 "r120_side_table_bytes": side, "r120_side_table_total": sum(side.values()),
                                 "engine_scratch_bytes": int(eng.scratch_bytes), "window_cache_used_bytes": int(eng.wc_used),
                                 "window_cache_reserved_bytes": int(eng.wc_data.nbytes), "window_cache_offsets_bytes": int(eng.wc_off.nbytes),
                                 "per_update_growth_bytes": 0,
                                 "note": "all R120 storage is allocated at construction; an update writes only existing rows"}
    out["rss_peak_mib"] = B118.rss_mib()
    out["environment"] = P.environment_fingerprint()
    out["torch_imported"] = "torch" in sys.modules
    return out


def run_all(progress=print) -> dict:
    proto, psha = protocol()
    res = {"phase": "ELPIS_YM_AWR_R120", "schema": "elpis.ym_hca.r120u.benchmark.v1", "protocol_sha256": psha, "fixtures": {}}
    for name in proto["benchmark"]["fixtures"]:
        tmp = pathlib.Path("/dev/shm") / f"r120_bench_{name}.json"
        cmd = [sys.executable, "-B", "-m", "research.yang_mills_hca.r120_incremental_geometric_update.bench",
               "--fixture", name, "--out", str(tmp)]
        proc = subprocess.run(cmd, cwd=str(P.YM_ROOT.parent.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        progress(proc.stdout.rstrip())
        if proc.returncode != 0:
            raise RuntimeError("BENCH_SUBPROCESS:" + name)
        res["fixtures"][name] = json.loads(tmp.read_text())
        tmp.unlink()
    cells = [c for f in res["fixtures"].values() for c in f["latency"]]
    prim = [c for f, v in res["fixtures"].items() for c in v["latency"] if c["role"] == "primary"]
    res["primary"] = {str(c["A"]): {"apply_p50_ms": c["apply"]["p50_ms"], "apply_p95_ms": c["apply"]["p95_ms"],
                                    "apply_p99_ms": c["apply"]["p99_ms"], "apply_max_ms": c["apply"]["max_ms"],
                                    "r119_step_p95_ms": c["r119_step"]["p95_ms"], "target_met": c["target_met"]} for c in prim}
    res["failures"] = int(sum(c["failures"] for c in cells))
    res["bit_mismatches"] = int(sum(c["step_bit_mismatches_vs_r119"] + c["composition_bit_mismatches"] for c in cells))
    res["comparisons"] = int(sum(c["step_comparisons"] + c["composition_comparisons"] for c in cells))
    res["ftz_daz_used"] = False
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fixture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-steps", type=int, help="development only: shortened run, marked unregistered")
    args = ap.parse_args(argv)
    path = pathlib.Path(args.out).resolve()
    if not path.is_relative_to("/dev/shm"):
        raise SystemExit("VOLATILE_OUTPUT_ONLY")
    if args.all:
        res = run_all(lambda s: print(s, flush=True))
    elif args.dev_steps:
        res = run_fixture(args.fixture, warmup=10, steps=args.dev_steps)
    else:
        res = run_fixture(args.fixture)
    path.write_text(json.dumps(res, indent=1, sort_keys=True, default=P.json_default) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
