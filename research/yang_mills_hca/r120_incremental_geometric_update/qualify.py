"""R120D adversarial qualification of the incremental update path.

    python -B -m research.yang_mills_hca.r120_incremental_geometric_update.qualify --out /dev/shm/r120_qualification.json

Every check is executed against criteria fixed in PROTOCOL_R120U.json before this run. Checks
that need torch (the frozen CGC reference forward) are reported NOT_EXECUTED when torch is
absent. Nothing is written under the repository.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import sys
import time
import tracemalloc

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import contract as C118
from research.yang_mills_hca.r118_scalable_awr import gauge as G118
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.fixtures import synthetic_registry
from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine
from research.yang_mills_hca.r118_scalable_awr import engine as E118

from . import cgc_bridge as CGC
from . import engine_r120 as E
from . import fixtures_r120 as F
from . import native_r120 as N
from . import reference as REF

HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = HERE / "PROTOCOL_R120U.json"
CGC_REFERENCE = P.YM_ROOT.parent / "cgc_lao" / "reference" / "src" / "cgc" / "lao_q4e5.py"
SOURCE_FILES = ("__init__.py", "native_r120.py", "cgc_bridge.py", "engine_r120.py", "reference.py", "fixtures_r120.py",
                "qualify.py", "bench.py", "science.py", "csrc/awr_kernel_r120.c")


def sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def component_digest() -> str:
    return hashlib.sha256("".join(f"{sha(HERE / n)}  {n}\n" for n in sorted(SOURCE_FILES)).encode()).hexdigest()


def bits_equal(a, b) -> bool:
    return bool(np.array_equal(np.ascontiguousarray(a, dtype=np.float64).view(np.uint64),
                               np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)))


def state_vs_reference(st, ref_state):
    """(max abs deviation, rows that are not bit-identical) of the pool against a reference dict."""
    reg = st.registry
    worst, bad = 0.0, 0
    for rid, vec in ref_state.items():
        got = st.pool[st.rel_row[int(reg.relation_index([rid])[0])]]
        worst = max(worst, float(np.abs(got - np.array(vec)).max()))
        bad += int(not bits_equal(got, vec))
    return worst, bad


# ---------------------------------------------------------------------------------------------
def check_pins(tol):
    out = {"frozen_manifest_files": P.verify_manifest()["files_verified"],
           "checkpoint_records_authenticated": sorted(P.checkpoint_records()) == list(P.CHECKPOINTS),
           "r118_component_digest": P.component_digest(),
           "r119_kernel_sha256": sha(P.YM_ROOT / "r119_numerical_hotpath/csrc/awr_kernel_r119.c"),
           "cgc_reference_sha256": sha(CGC_REFERENCE)}
    out["pass"] = (out["frozen_manifest_files"] == 39 and out["checkpoint_records_authenticated"]
                   and out["r118_component_digest"] == tol["r118_component_digest"]
                   and out["r119_kernel_sha256"] == N.R119_KERNEL_SHA256 == tol["r119_kernel_sha256"]
                   and out["cgc_reference_sha256"] == tol["cgc_reference_sha256"])
    return out


def check_cgc_and_law(tol):
    """Cubic operator and update law: native vs independent reference (bitwise), torch reference (tolerance)."""
    rng = np.random.default_rng(1201)
    reg, st, *_ = F.small_state()
    eng = E.IncrementalEngine(st, F.CHECKPOINT)
    lib = eng._lib
    w = eng.cgc_w
    mism = 0
    xs = rng.uniform(-1, 1, size=(2000, 5)).astype(np.float32)
    out = np.zeros(3, dtype=np.float32)
    for x in xs:
        lib.awr120_cgc_probe(w.ctypes.data, x.ctypes.data, out.ctypes.data)
        ref = np.array(CGC.forward_reference(w, x), dtype=np.float32)
        mism += int(not np.array_equal(out.view(np.uint32), ref.view(np.uint32)))
    torch_dev = "NOT_EXECUTED"
    try:
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("lao_q4e5_frozen", CGC_REFERENCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ctl = mod.FrozenCGC3(torch.from_numpy(np.array(w))).eval()
        with torch.no_grad():
            t = ctl(torch.from_numpy(xs)).numpy()
        mine = np.array([CGC.forward_reference(w, x) for x in xs], dtype=np.float32)
        torch_dev = float(np.abs(t - mine).max() / (1.0 + np.abs(t).max()))
    except ImportError:
        pass
    # update law on detached states across regimes, including exact zeros, signed zeros and subnormals
    law_mism = law_n = 0
    frames = G118.random_so2x2(rng, 64)
    for scale in (1.0, 3.5, 1e-3, 1e-20, 1e-160, 1e-300, 1e-310, 0.0):
        for _ in range(250):
            s = rng.normal(size=8) * scale
            if rng.random() < 0.2:
                s[rng.integers(0, 8)] = -0.0
            o = rng.normal(size=4) * (scale if rng.random() < 0.5 else 1.0)
            fr = frames[rng.integers(0, 64)] if rng.random() < 0.9 else None
            if fr is None:
                o = np.zeros(4)
            wt, aux = float(rng.uniform(-1, 1)), float(rng.uniform(0, 1))
            got, rc = eng.update_probe(s, fr, o, wt, aux)
            ref = REF.update_state(s, fr, o, wt, aux, w, eng.eta, eng.gamma, eng.rho_b, eng.kappa)
            law_n += 1
            law_mism += int(rc != 0 or ref is None or not bits_equal(got, ref))
    zero, _ = eng.update_probe(np.zeros(8), None, np.zeros(4), 0.7, 0.3)
    fr, o = frames[0], rng.normal(size=4)
    inj, _ = eng.update_probe(np.zeros(8), fr, o, 1.0, 0.0)
    v = G118.bapply(fr, o)
    exp_inj = np.concatenate([eng.gamma * v, (eng.gamma * eng.rho_b) * v])
    res = {"cgc_native_vs_reference_bit_mismatches": mism, "cgc_samples": int(xs.shape[0]),
           "cgc_reference_vs_torch_frozen_cgc3_max_rel": torch_dev,
           "law_native_vs_reference_bit_mismatches": law_mism, "law_samples": law_n,
           "zero_state_zero_observation_stays_exactly_zero": bool(not zero.any()),
           "zero_state_injection_equals_gamma_v": bits_equal(inj, exp_inj),
           "cgc_weights": {"label": CGC.SYNTHETIC_LABEL, "sha256": eng.cgc_sha256}}
    res["pass"] = (mism == 0 and law_mism == 0 and res["zero_state_zero_observation_stays_exactly_zero"]
                   and res["zero_state_injection_equals_gamma_v"]
                   and (torch_dev == "NOT_EXECUTED" or torch_dev <= tol["cgc_torch_relative"]))
    return res


def check_reference_parity(tol):
    """Fused native update-and-propagate vs the independent pure-Python reference, bitwise."""
    total = bad_rows = bad_disp = bad_active = 0
    worst = 0.0
    cells = []
    for name, kw in (("sparse", dict(seed=5)), ("zipf_heavy", dict(seed=7, zipf=1.2)),
                     ("high_overlap", dict(seed=9, participants=12, relations=40, windows=120)),
                     ("hub_arity16", dict(seed=13, participants=40, relations=120, windows=60, arity=(2, 12), hub=0.6))):
        for controller in ("NATIVE", "R104", "R114"):
            for scale in (1.0, 1e-150, 1e-310, "mixed"):
                reg, st, *_ = F.small_state(**kw)
                rng = np.random.default_rng(hash((name, controller, str(scale))) % (1 << 31))
                rows = np.nonzero(st.row_ref > 0)[0]
                if scale == "mixed":
                    st.pool[rows] *= 10.0 ** rng.uniform(-320, 0, size=(rows.size, 1))
                else:
                    st.pool[rows] *= scale
                eng = E.IncrementalEngine(st, F.CHECKPOINT, controller=controller,
                                          limits=E.Limits(threads=1 if rng.random() < 0.5 else 4, parallel_min_instances=2))
                if rng.random() < 0.5:
                    eng.build_cache()
                n_bad = 0
                for _ in range(12):
                    n = int(rng.integers(1, 14))
                    oscale = 1.0 if scale == "mixed" else float(scale)
                    batch = F.random_batch(st, rng, n, scale=oscale if rng.random() < 0.7 else 1.0)
                    radius, passes = int(rng.integers(1, 3)), int(rng.integers(1, 4))
                    model, params = F.model_of(st, eng)
                    disp, ref_state, ref_seq, info = REF.apply_batch(model, F.events_of(st, *batch), params,
                                                                      radius=radius, passes=passes)
                    rep = eng.apply(*batch, radius=radius, passes=passes)
                    dev, bad = state_vs_reference(st, ref_state)
                    total += 1
                    worst = max(worst, dev)
                    n_bad += bad
                    bad_disp += int(rep.disposition != disp)
                    bad_active += int(rep.instances != len(info["active"]))
                bad_rows += n_bad
                cells.append(f"{name}/{controller}/{scale}:{n_bad}")
    res = {"batches": total, "rows_not_bit_identical": bad_rows, "disposition_mismatches": bad_disp,
           "affected_set_size_mismatches": bad_active, "max_abs_deviation": worst, "cells": len(cells)}
    res["pass"] = bad_rows == 0 and bad_disp == 0 and bad_active == 0
    return res


def check_r119_regression(tol):
    """(a) fused step == R119 EXACT_GUARDED step; (b) fused update+propagate == reference update + R119 step."""
    step_mism = step_n = comp_mism = comp_n = 0
    for seed, kw in ((21, {}), (22, dict(participants=12, relations=40, windows=120))):
        for controller in ("NATIVE", "R104", "R107", "R114"):
            for ck in (P.CHECKPOINTS[0], P.CHECKPOINTS[1], P.CHECKPOINTS[5]):
                reg, st, handles, _, wins = F.small_state(seed=seed, **kw)
                rng = np.random.default_rng(seed * 7 + len(controller))
                eng = E.IncrementalEngine(st, ck, controller=controller, limits=E.Limits(threads=4, parallel_min_instances=2))
                eng.build_cache()
                r119 = R119Engine(st, ck, controller=controller, mode="EXACT_GUARDED", budgets=E118.Budgets(max_passes=8))
                for _ in range(6):
                    a = np.sort(rng.choice(handles, size=int(rng.integers(1, len(handles))), replace=False))
                    passes = int(rng.integers(1, 4))
                    base = st.pool.copy()
                    assert r119.step(a, passes=passes).exact
                    ref = st.pool.copy()
                    st.pool[:] = base
                    assert eng.step(a, passes=passes).committed
                    step_n += 1
                    step_mism += int(not bits_equal(st.pool, ref))
                    # composition: update rows with the independent reference law, then an R119 step
                    batch = F.random_batch(st, rng, int(rng.integers(1, 10)))
                    base, seq0 = st.pool.copy(), st.row_seq.copy()
                    rep = eng.apply(*batch, passes=passes)
                    fused = st.pool.copy()
                    st.pool[:] = base
                    st.row_seq[:] = seq0
                    ids, sq, parts, obs, wt, aux = batch
                    touched = []
                    for i in range(len(ids)):
                        r = int(reg.relation_index([ids[i]])[0])
                        row = int(st.rel_row[r])
                        lo = int(reg.rel_offsets[r])
                        inc = lo + int(np.nonzero(reg.members_of(r) == parts[i])[0][0])
                        new = REF.update_state(st.pool[row], reg.frames[inc], obs[i], wt[i], aux[i], eng.cgc_w, eng.eta,
                                               eng.gamma, eng.rho_b, eng.kappa)
                        st.set_relation_state(int(ids[i]), new)
                        touched.append(row)
                    assert r119.step(F.affected_handles(st, touched), passes=passes).exact
                    comp_n += 1
                    comp_mism += int(not bits_equal(st.pool, fused) or not rep.committed)
    res = {"step_comparisons": step_n, "step_bit_mismatches": step_mism, "composition_comparisons": comp_n,
           "composition_bit_mismatches": comp_mism,
           "weights_sha256_match_pins": all(E118.load_weights(ck)[2] == P.CHECKPOINT_SHA256[ck] for ck in P.CHECKPOINTS)}
    res["pass"] = step_mism == 0 and comp_mism == 0 and res["weights_sha256_match_pins"]
    return res


def check_identity_and_memory(tol):
    reg, st, handles, gens, wins = F.small_state(seed=31)
    eng = E.IncrementalEngine(st, F.CHECKPOINT)
    eng.build_cache()
    rng = np.random.default_rng(31)
    before = F.snapshot(st, eng)
    weights = (eng.W.tobytes(), eng.b.tobytes(), eng.cgc_w.tobytes())
    ids_of = {int(k): (id(v), v.ctypes.data) for k, v in enumerate((st.pool, st.row_seq, eng.wc_data, eng._buf["cand"]))}
    for _ in range(50):                      # warm every code path before measuring allocations
        eng.apply(*F.random_batch(st, rng, 8))
    batches = [F.random_batch(st, rng, 8) for _ in range(tol["memory_updates"])]
    # sequence numbers were drawn against the state at generation time; make them increase globally
    for k, b in enumerate(batches):
        b[1][:] = 1000 + 10 * k + np.arange(8, dtype=np.uint64)
    gc.collect()
    tracemalloc.start()
    base_cur = tracemalloc.get_traced_memory()[0]
    blocks0 = sys.getallocatedblocks()
    applied = 0
    for b in batches:
        applied += eng.apply(*b).applied
    cur, peak = tracemalloc.get_traced_memory()
    blocks1 = sys.getallocatedblocks()
    tracemalloc.stop()
    same_identity = all(np.array_equal(getattr(st, k), before[k]) for k in ("row_rel", "rel_row", "slot_rel", "slot_row",
                                                                           "inst_gen", "row_gen"))
    res = {
        "updates_applied": int(applied), "relation_rows_and_generations_unchanged": bool(same_identity),
        "live_instances_unchanged": st.live_count == before["live"],
        "materialized_rows_unchanged": int((st.row_ref > 0).sum()) == int((before["row_rel"] >= 0).sum()),
        "buffers_not_reallocated": all(ids_of[k] == (id(v), v.ctypes.data) for k, v in
                                       enumerate((st.pool, st.row_seq, eng.wc_data, eng._buf["cand"]))),
        "hca88_and_cgc_weights_unchanged": weights == (eng.W.tobytes(), eng.b.tobytes(), eng.cgc_w.tobytes()),
        "weights_read_only": not (eng.W.flags.writeable or eng.b.flags.writeable or eng.cgc_w.flags.writeable),
        "net_retained_python_bytes": int(cur - base_cur), "transient_peak_python_bytes": int(peak - base_cur),
        "net_allocated_blocks": int(blocks1 - blocks0), "engine_scratch_bytes": int(eng.scratch_bytes),
        "window_cache_bytes": int(eng.wc_used),
    }
    res["pass"] = (same_identity and res["live_instances_unchanged"] and res["materialized_rows_unchanged"]
                   and res["buffers_not_reallocated"] and res["hca88_and_cgc_weights_unchanged"] and res["weights_read_only"]
                   and res["net_retained_python_bytes"] <= tol["max_net_retained_bytes"]
                   and abs(res["net_allocated_blocks"]) <= tol["max_net_allocated_blocks"])
    return res


def check_atomicity(tol):
    """Every refusal leaves pool, sequences, pending list, identities and scratch marks untouched."""
    reg, st, handles, gens, wins = F.small_state(seed=41)
    rng = np.random.default_rng(41)
    tiny = E.Limits(max_events=16, max_active_instances=3, max_active_relations=10, max_links=1 << 17)
    cases, ok = {}, True

    def run(name, expect, limits=E.Limits(), mutate=None, n=6, **kw):
        nonlocal ok
        st._engines.clear()
        eng = E.IncrementalEngine(st, F.CHECKPOINT, limits=limits)
        batch = list(F.random_batch(st, rng, n, repeat=False))
        gen = np.zeros(n, dtype=np.uint32)
        if mutate:
            mutate(batch, gen)
        snap = F.snapshot(st, eng)
        rep = eng.apply(*batch, exp_gen=gen, **kw)
        good = rep.disposition == expect and F.unchanged(st, snap, eng)
        cases[name] = {"disposition": rep.disposition, "expected": expect, "state_unchanged": F.unchanged(st, snap, eng),
                       "fail_index": rep.fail_index}
        ok = ok and good

    def set_last(field, value):
        def m(batch, gen):
            batch[field][-1] = value
        return m

    run("nan_observation_last", "REFUSED_INVALID_VALUE", mutate=lambda b, g: b[3].__setitem__((5, 2), np.nan))
    run("inf_observation_first", "REFUSED_INVALID_VALUE", mutate=lambda b, g: b[3].__setitem__((0, 0), np.inf))
    run("weight_out_of_range", "REFUSED_INVALID_VALUE", mutate=set_last(4, 1.5))
    run("aux_out_of_range", "REFUSED_INVALID_VALUE", mutate=set_last(5, -0.1))
    run("unknown_relation", "REFUSED_UNKNOWN_RELATION", mutate=set_last(0, np.uint64(12345)))
    unbound = int(np.nonzero(st.rel_row < 0)[0][0])
    run("relation_not_materialized", "REFUSED_NOT_MATERIALIZED", mutate=set_last(0, reg.relation_ids[unbound]))
    run("stale_generation", "REFUSED_STALE_RELATION", mutate=lambda b, g: g.__setitem__(-1, 77))

    def not_incident(batch, gen):
        r = int(reg.relation_index([batch[0][-1]])[0])
        outside = np.setdiff1d(np.arange(reg.num_participants), reg.members_of(r))
        batch[2][-1] = int(outside[0])
    run("participant_not_incident", "REFUSED_NOT_INCIDENT", mutate=not_incident)
    run("observation_without_frame", "REFUSED_NOT_INCIDENT", mutate=set_last(2, -1))

    def conflict(batch, gen):
        batch[0][-1], batch[1][-1], batch[2][-1] = batch[0][0], batch[1][0], batch[2][0]
        batch[3][-1] = batch[3][0] + 1.0
    run("conflicting_same_sequence", "REFUSED_CONFLICT", mutate=conflict)
    run("event_budget", "REFUSED_EVENT_BUDGET", limits=tiny, n=17)
    run("instance_budget", "REFUSED_INSTANCE_BUDGET", limits=E.Limits(max_active_instances=3), n=12)
    run("row_budget", "REFUSED_ROW_BUDGET", limits=E.Limits(max_active_instances=64, max_active_relations=6), n=5)
    run("link_budget", "REFUSED_LINK_BUDGET", limits=E.Limits(max_links=4), n=3)
    run("bad_radius", "REFUSED_ARGUMENT", radius=9)
    # stale relation reference after release and re-creation of the only binding window
    st._engines.clear()
    eng = E.IncrementalEngine(st, F.CHECKPOINT)
    single = np.nonzero(st.row_ref == 1)[0]
    row = int(single[0])
    rid = int(reg.relation_ids[st.row_rel[row]])
    _, gen0 = st.relation_ref(rid)
    h = int(st.instances_binding(int(st.row_rel[row]))[0])
    slots = [None if r < 0 else int(reg.relation_ids[r]) for r in st.slot_rel[h]]
    st.release_instance(h, int(st.inst_gen[h]))
    st.create_instance(999999, slots)
    part = int(reg.members_of(int(reg.relation_index([rid])[0]))[0])
    snap = F.snapshot(st, eng)
    rep = eng.apply(np.array([rid], np.uint64), np.array([1], np.uint64), np.array([part], np.int32), np.ones((1, 4)),
                    np.array([0.5]), np.array([0.5]), exp_gen=np.array([gen0], np.uint32))
    cases["stale_reference_after_rematerialization"] = {"disposition": rep.disposition, "expected": "REFUSED_STALE_RELATION",
                                                        "state_unchanged": F.unchanged(st, snap, eng)}
    ok = ok and rep.disposition == "REFUSED_STALE_RELATION" and F.unchanged(st, snap, eng)
    _, gen1 = st.relation_ref(rid)
    rep = eng.apply(np.array([rid], np.uint64), np.array([1], np.uint64), np.array([part], np.int32), np.ones((1, 4)),
                    np.array([0.5]), np.array([0.5]), exp_gen=np.array([gen1], np.uint32))
    cases["fresh_reference_accepted"] = {"disposition": rep.disposition, "generation_changed": gen1 != gen0}
    ok = ok and rep.disposition == "OK" and gen1 != gen0
    return {"cases": cases, "pass": bool(ok)}


def check_event_semantics(tol):
    reg, st, handles, gens, wins = F.small_state(seed=51)
    rng = np.random.default_rng(51)
    eng = E.IncrementalEngine(st, F.CHECKPOINT)
    res = {}
    # idempotent redelivery
    batch = F.random_batch(st, rng, 9)
    r1 = eng.apply(*batch)
    after = st.pool.copy()
    r2 = eng.apply(*batch)
    res["redelivery_skipped_and_state_unchanged"] = bool(r1.applied == 9 and r2.applied == 0 and r2.skipped_duplicate == 9
                                                         and bits_equal(st.pool, after))
    # identical duplicate inside one batch is applied once
    b = [np.concatenate([x[:1], x[:1]]) for x in F.random_batch(st, rng, 1)]
    r3 = eng.apply(*b)
    res["in_batch_exact_duplicate_applied_once"] = bool(r3.applied == 1 and r3.skipped_duplicate == 1)
    # order independence across distinct relations (each relation once)
    base, seq0 = st.pool.copy(), st.row_seq.copy()
    batch = F.random_batch(st, rng, 10, repeat=False)
    eng.apply(*batch)
    a = st.pool.copy()
    st.pool[:], st.row_seq[:] = base, seq0
    perm = rng.permutation(10)
    eng.apply(*[x[perm] for x in batch])
    res["batch_order_independent_for_distinct_relations"] = bits_equal(st.pool, a)
    # sequential semantics for repeated events on one relation: one batch == consecutive batches + one propagation
    st.pool[:], st.row_seq[:] = base, seq0
    row = int(np.nonzero(st.row_ref > 0)[0][3])
    rid = reg.relation_ids[st.row_rel[row]]
    part = int(reg.members_of(int(st.row_rel[row]))[0])
    k = 5
    ids = np.full(k, rid, np.uint64)
    seq = int(st.row_seq[row]) + 1 + np.arange(k, dtype=np.uint64)
    parts = np.full(k, part, np.int32)
    obs, wt, aux = rng.normal(size=(k, 4)), rng.uniform(-1, 1, k), rng.uniform(0, 1, k)
    rc = eng.apply(ids, seq, parts, obs, wt, aux)
    coalesced = st.pool.copy()
    st.pool[:], st.row_seq[:] = base, seq0
    for i in range(k):
        eng.apply(ids[i:i + 1], seq[i:i + 1], parts[i:i + 1], obs[i:i + 1], wt[i:i + 1], aux[i:i + 1], propagate=False)
    rp = eng.propagate_pending()
    res["repeated_events_coalesce_to_one_propagation"] = bool(bits_equal(st.pool, coalesced) and rc.touched_rows == 1
                                                              and rc.applied == k and rp.disposition == "OK")
    # deferred propagation is reported and equals immediate propagation once executed
    st.pool[:], st.row_seq[:] = base, seq0
    batch = F.random_batch(st, rng, 6, repeat=False)
    eng.apply(*batch)
    immediate = st.pool.copy()
    st.pool[:], st.row_seq[:] = base, seq0
    rd = eng.apply(*batch, propagate=False)
    untouched_neighbors = True
    touched_rows = {int(st.rel_row[int(reg.relation_index([i])[0])]) for i in batch[0]}
    for rw in np.nonzero(st.row_ref > 0)[0].tolist():
        if rw not in touched_rows:
            untouched_neighbors = untouched_neighbors and bits_equal(st.pool[rw], base[rw])
    pend = eng.pending_rows
    rp = eng.propagate_pending()
    res["deferred_reported_then_equal_to_immediate"] = bool(rd.deferred and pend == 6 and untouched_neighbors
                                                            and rp.disposition == "OK" and eng.pending_rows == 0
                                                            and bits_equal(st.pool, immediate))
    # budget overflow with deferral: updates committed, propagation pending, nothing dropped
    st._engines.clear()
    small = E.IncrementalEngine(st, F.CHECKPOINT, limits=E.Limits(max_active_instances=2))
    st.pool[:], st.row_seq[:] = base, seq0
    rb = small.apply(*batch, defer_on_budget=True)
    res["budget_overflow_deferred_not_dropped"] = bool(rb.deferred and small.pending_rows == 6 and rb.applied == 6)
    # unaffected relations are untouched by an immediate update
    st._engines.clear()
    eng = E.IncrementalEngine(st, F.CHECKPOINT)
    st.pool[:], st.row_seq[:] = base, seq0
    one = F.random_batch(st, rng, 1)
    rep = eng.apply(*one)
    row = int(st.rel_row[int(reg.relation_index([one[0][0]])[0])])
    aff = F.affected_handles(st, [row])
    aff_rows = set(st.slot_row[aff][st.slot_row[aff] >= 0].tolist())
    same = all(bits_equal(st.pool[rw], base[rw]) for rw in np.nonzero(st.row_ref > 0)[0].tolist() if rw not in aff_rows)
    res["rows_outside_affected_windows_bitwise_unchanged"] = bool(same and rep.instances == aff.size and rep.rows == len(aff_rows))
    res["pass"] = all(v for v in res.values())
    return res


def _gauged(reg, st, wins, g_rel=None, h_part=None):
    reg2 = G118.gauge_registry(reg, g_rel=g_rel, h_part=h_part)
    st2 = F.clone_state(reg2, st, wins)
    if g_rel is not None:
        rows = np.nonzero(st2.row_ref > 0)[0]
        g = np.asarray(g_rel)[st2.row_rel[rows]]
        st2.pool[rows, :4] = G118.bapply(g, st2.pool[rows, :4])
        st2.pool[rows, 4:] = G118.bapply(g, st2.pool[rows, 4:])
    return reg2, st2


def check_geometry(tol):
    worst_cov = worst_part = worst_perm = 0.0
    neg_state_only = neg_obs_only = neg_no_transport = np.inf
    feature_dev = 0.0
    for seed, controller in ((61, "NATIVE"), (62, "R114"), (63, "NATIVE")):
        reg, st, handles, gens, wins = F.small_state(seed=seed, participants=14 if seed == 63 else 60,
                                                     relations=50 if seed == 63 else 200, windows=100 if seed == 63 else 80)
        rng = np.random.default_rng(seed)
        g_rel = G118.random_so2x2(rng, reg.num_relations)
        h_part = G118.random_so2x2(rng, reg.num_participants)
        batches = [F.random_batch(st, rng, 10) for _ in range(4)]
        for k, b in enumerate(batches):
            b[1][:] = 100 * (k + 1) + np.arange(10, dtype=np.uint64)

        def run(reg_x, st_x, obs_map=None):
            eng = E.IncrementalEngine(st_x, F.CHECKPOINT, controller=controller)
            for b in batches:
                obs = b[3] if obs_map is None else obs_map(b)
                assert eng.apply(b[0], b[1], b[2], obs, b[4], b[5], radius=2, passes=2).committed
            return {int(reg_x.relation_ids[st_x.row_rel[r]]): st_x.pool[r].copy() for r in np.nonzero(st_x.row_ref > 0)[0]}

        base = run(reg, F.clone_state(reg, st, wins))
        # (1) relation gauge: frames and states transformed jointly -> results transform covariantly
        reg_g, st_g = _gauged(reg, st, wins, g_rel=g_rel)
        out_g = run(reg_g, st_g)
        # (2) participant gauge: frames U h^T and observations h o -> identical results
        reg_p, st_p = _gauged(reg, st, wins, h_part=h_part)
        out_p = run(reg_p, st_p, obs_map=lambda b: G118.bapply(h_part[b[2]], b[3]))
        # negative controls
        st_bad = F.clone_state(reg, st, wins)                    # states transformed, frames not
        rows = np.nonzero(st_bad.row_ref > 0)[0]
        gg = g_rel[st_bad.row_rel[rows]]
        st_bad.pool[rows, :4] = G118.bapply(gg, st_bad.pool[rows, :4])
        st_bad.pool[rows, 4:] = G118.bapply(gg, st_bad.pool[rows, 4:])
        out_bad = run(reg, st_bad)
        out_obs = run(reg, F.clone_state(reg, st, wins), obs_map=lambda b: G118.bapply(h_part[b[2]], b[3]))   # obs rotated only
        for rid, vec in base.items():
            r = int(reg.relation_index([rid])[0])
            exp = np.concatenate([G118.bapply(g_rel[r], vec[:4]), G118.bapply(g_rel[r], vec[4:])])
            worst_cov = max(worst_cov, float(np.abs(out_g[rid] - exp).max()))
            worst_part = max(worst_part, float(np.abs(out_p[rid] - vec).max()))
        neg_state_only = min(neg_state_only, max(float(np.abs(out_bad[rid] - np.concatenate(
            [G118.bapply(g_rel[int(reg.relation_index([rid])[0])], v[:4]),
             G118.bapply(g_rel[int(reg.relation_index([rid])[0])], v[4:])])).max()) for rid, v in base.items()))
        neg_obs_only = min(neg_obs_only, max(float(np.abs(out_obs[rid] - v).max()) for rid, v in base.items()))
        # (3) a non-geometric update (observation used without frame transport) is not covariant
        eng0 = E.IncrementalEngine(F.clone_state(reg, st, wins), F.CHECKPOINT)
        ident = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        dev = 0.0
        for _ in range(50):
            r = int(rng.integers(0, reg.num_relations))
            s, o = rng.normal(size=8), rng.normal(size=4)
            fr = reg.frames[int(reg.rel_offsets[r])]
            g = g_rel[r]
            plain, _ = eng0.update_probe(s, ident, o, 0.8, 0.4)                    # no transport
            sg = np.concatenate([G118.bapply(g, s[:4]), G118.bapply(g, s[4:])])
            plain_g, _ = eng0.update_probe(sg, ident, o, 0.8, 0.4)                # gauged state, same raw observation
            dev = max(dev, float(np.abs(plain_g - np.concatenate([G118.bapply(g, plain[:4]), G118.bapply(g, plain[4:])])).max()))
            # the geometric law on the same inputs is covariant, and its CGC features are invariant
            geo, _ = eng0.update_probe(s, fr, o, 0.8, 0.4)
            geo_g, _ = eng0.update_probe(sg, G118.bmul(g, fr), o, 0.8, 0.4)
            feature_dev = max(feature_dev, float(np.abs(geo_g - np.concatenate([G118.bapply(g, geo[:4]), G118.bapply(g, geo[4:])])).max()))
        neg_no_transport = min(neg_no_transport, dev)
        # (4) slot permutation of every window leaves each relation's result unchanged
        rel, mask = wins
        perm = np.array([rng.permutation(4) for _ in range(rel.shape[0])])
        rel_p = np.take_along_axis(rel, perm, axis=1)
        mask_p = np.take_along_axis(mask, perm, axis=1)
        st_perm = F.clone_state(reg, st, (rel_p, mask_p))
        out_perm = run(reg, st_perm)
        worst_perm = max(worst_perm, max(float(np.abs(out_perm[rid] - v).max()) for rid, v in base.items()))
    res = {"relation_gauge_covariance_max_abs": worst_cov, "participant_gauge_invariance_max_abs": worst_part,
           "single_update_covariance_max_abs": feature_dev, "slot_permutation_max_abs": worst_perm,
           "negative_state_only_gauge_min": float(neg_state_only), "negative_observation_only_rotation_min": float(neg_obs_only),
           "negative_untransported_observation_min": float(neg_no_transport)}
    res["pass"] = (worst_cov <= tol["covariance_abs"] and worst_part <= tol["covariance_abs"]
                   and feature_dev <= tol["covariance_abs"] and worst_perm <= tol["covariance_abs"]
                   and neg_state_only >= tol["negative_control_min"] and neg_obs_only >= tol["negative_control_min"]
                   and neg_no_transport >= tol["negative_control_min"])
    return res


def check_stability_and_replay(tol):
    reg, st, handles, gens, wins = F.small_state(seed=71)
    rows = np.nonzero(st.row_ref > 0)[0]
    out = {}
    finals = {}
    for threads in (1, 4):
        st_x = F.clone_state(reg, st, wins)
        eng = E.IncrementalEngine(st_x, F.CHECKPOINT, limits=E.Limits(threads=threads, parallel_min_instances=2))
        eng.build_cache()
        rng = np.random.default_rng(71)
        max_norm, sat, n = 0.0, 0, 0
        for step in range(tol["long_horizon_batches"]):
            b = F.random_batch(st_x, rng, 8, scale=3.0)
            rep = eng.apply(*b)
            assert rep.committed
            if step % 500 == 0 or step == tol["long_horizon_batches"] - 1:
                live = st_x.pool[np.nonzero(st_x.row_ref > 0)[0]]
                assert np.all(np.isfinite(live))
                na, nb = np.linalg.norm(live[:, :4], axis=1), np.linalg.norm(live[:, 4:], axis=1)
                max_norm = max(max_norm, float(max(na.max(), nb.max())))
                sat += int(((na > 3.999) | (nb > 3.999)).sum())
                n += live.shape[0]
        finals[threads] = st_x.pool.copy()
        out[f"threads_{threads}"] = {"max_channel_norm": max_norm, "saturated_fraction_sampled": sat / n,
                                     "batches": tol["long_horizon_batches"]}
    # same stream replayed in a fresh process-independent state: bit identical; and across thread counts
    out["serial_vs_parallel_bit_identical"] = bits_equal(finals[1], finals[4])
    st_r = F.clone_state(reg, st, wins)
    eng = E.IncrementalEngine(st_r, F.CHECKPOINT)
    rng = np.random.default_rng(71)
    for _ in range(tol["long_horizon_batches"]):
        eng.apply(*F.random_batch(st_r, rng, 8, scale=3.0))
    out["deterministic_replay_bit_identical"] = bits_equal(st_r.pool, finals[1])
    # high-frequency repeated updates of one relation, propagated every time, hostile evidence amplitude
    st_h = F.clone_state(reg, st, wins)
    eng = E.IncrementalEngine(st_h, F.CHECKPOINT, controller="R114")
    row = int(rows[0])
    rid = np.array([reg.relation_ids[st_h.row_rel[row]]], np.uint64)
    part = np.array([int(reg.members_of(int(st_h.row_rel[row]))[0])], np.int32)
    rng = np.random.default_rng(72)
    peak = 0.0
    for k in range(tol["high_frequency_updates"]):
        rep = eng.apply(rid, np.array([k + 1], np.uint64), part, rng.normal(size=(1, 4)) * 1e3, np.array([1.0]),
                        np.array([1.0]))
        assert rep.committed
        if k % 1000 == 0:
            live = st_h.pool[np.nonzero(st_h.row_ref > 0)[0]]
            peak = max(peak, float(np.abs(live).max()))
    live = st_h.pool[np.nonzero(st_h.row_ref > 0)[0]]
    out["high_frequency"] = {"updates": tol["high_frequency_updates"], "all_finite": bool(np.all(np.isfinite(live))),
                             "max_abs_coordinate": max(peak, float(np.abs(live).max())),
                             "updated_row_channel_norms": [float(np.linalg.norm(st_h.pool[row, :4])),
                                                           float(np.linalg.norm(st_h.pool[row, 4:]))]}
    bound = tol["state_coordinate_bound"]
    out["high_frequency"]["updated_row_after_update_only_norm_bound_holds"] = True
    for k in range(200):                     # the update itself (no propagation) always leaves the row within the clamp
        rep = eng.apply(rid, np.array([tol["high_frequency_updates"] + k + 1], np.uint64), part,
                        rng.normal(size=(1, 4)) * 1e3, np.array([1.0]), np.array([1.0]), propagate=False)
        ok = rep.deferred and max(np.linalg.norm(st_h.pool[row, :4]), np.linalg.norm(st_h.pool[row, 4:])) <= bound
        out["high_frequency"]["updated_row_after_update_only_norm_bound_holds"] &= bool(ok)
    out["pass"] = (out["serial_vs_parallel_bit_identical"] and out["deterministic_replay_bit_identical"]
                   and out["high_frequency"]["all_finite"]
                   and out["high_frequency"]["max_abs_coordinate"] <= tol["controller_coordinate_bound"]
                   and out["high_frequency"]["updated_row_after_update_only_norm_bound_holds"]
                   and all(out[f"threads_{t}"]["max_channel_norm"] <= bound for t in (1, 4)))
    return out


CHECKS = (("pins", check_pins), ("cgc_and_update_law", check_cgc_and_law), ("reference_parity", check_reference_parity),
          ("r119_regression", check_r119_regression), ("identity_and_memory", check_identity_and_memory),
          ("atomicity", check_atomicity), ("event_semantics", check_event_semantics), ("geometry", check_geometry),
          ("stability_and_replay", check_stability_and_replay))


def run(progress=print) -> dict:
    proto = json.loads(PROTOCOL.read_bytes())
    tol = proto["qualification"]
    t0 = time.time()
    out = {"phase": "ELPIS_YM_AWR_R120", "schema": "elpis.ym_hca.r120u.qualification.v1",
           "protocol_sha256": sha(PROTOCOL), "component_digest": component_digest(), "checks": {}}
    for name, fn in CHECKS:
        t1 = time.time()
        res = fn(tol)
        res["seconds"] = round(time.time() - t1, 1)
        out["checks"][name] = res
        progress(f"R120D {name}: {'PASS' if res['pass'] else 'FAIL'} ({res['seconds']} s)")
    lib, info = N.kernel()
    out["kernel"] = info
    out["openmp_max_threads"] = int(lib.awr120_openmp())
    out["checkpoints_unchanged"] = all(E118.load_weights(ck)[2] == P.CHECKPOINT_SHA256[ck] for ck in P.CHECKPOINTS)
    out["frozen_tree_unchanged"] = P.verify_manifest()["files_verified"] == 39
    out["environment"] = P.environment_fingerprint()
    out["pass"] = bool(all(c["pass"] for c in out["checks"].values()) and out["checkpoints_unchanged"]
                       and out["frozen_tree_unchanged"])
    out["r93"] = "ENERGY_NORMALIZED_DENOMINATOR_HOLD"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    path = pathlib.Path(args.out).resolve()
    if not path.is_relative_to("/dev/shm"):
        raise SystemExit("VOLATILE_OUTPUT_ONLY")
    res = run(progress=lambda s: print(s, flush=True))
    path.write_text(json.dumps(res, indent=1, sort_keys=True, default=P.json_default) + "\n")
    print("R120D_QUALIFICATION=" + ("PASS" if res["pass"] else "FAIL"), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
