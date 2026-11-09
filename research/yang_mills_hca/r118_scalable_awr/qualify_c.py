"""R118-C qualification: the authentic frozen HCA inside the sparse substrate.

    python -B -m research.yang_mills_hca.r118_scalable_awr.qualify_c [--out PATH]

Thresholds are fixed here before execution (PROTOCOL_R118.json repeats them):

* oracle-through-substrate vs historical rollout: bitwise (max abs == 0);
* native vs oracle: <= 1e-12 after one pass, <= 1e-10 after <= 20 passes;
* native vs simple reference: <= 1e-12 (observed bitwise);
* gauge covariance / invariance: <= 1e-9; negative controls must exceed 1e-6.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import engine as E
from . import gauge as G
from . import provenance as P
from .awr_state import AWRState, StateCapacity
from .fixtures import historical_registry, synthetic_registry
from .registry import RegistryLimits, build_registry, rotation_blocks
from .tests._util import compare_with_simple, random_windows, seed_states

ONE_PASS_TOL = 1e-12
MULTI_PASS_TOL = 1e-10
COV_TOL = 1e-9
NEG_MIN = 1e-6
BUD = E.Budgets(max_passes=20)


def _hist_state(o, reg, windows, st0, cap):
    st = AWRState(reg, StateCapacity(max_instances=cap, max_materialized_relations=4 * cap))
    hs = []
    for c, w in enumerate(windows):
        h, _ = st.create_instance(1000 + c, w)
        hs.append(h)
        for d in range(4):
            st.set_relation_state(w[d], np.concatenate([st0[0][c, d].numpy(), st0[1][c, d].numpy()]))
    return st, np.array(hs)


def _readback(st, windows):
    a = np.array([[st.relation_state(w[d])[:4] for d in range(4)] for w in windows])
    b = np.array([[st.relation_state(w[d])[4:] for d in range(4)] for w in windows])
    return a, b


def historical_replay(o) -> dict:
    """Every historical graph, both transports, all six checkpoints: H1 and H20."""
    T = o.torch
    graphs = dict(o.r61["r55_structural"]["graphs"])
    graphs.update(o.r77.CFG["graphs"])
    out = {"cases": 0, "oracle_bitwise_cases": 0, "oracle_max_abs": 0.0, "native_h1_max_abs": 0.0,
           "native_h20_max_abs": 0.0, "native_controlled_vs_oracle_h20_max_abs": 0.0}
    st0 = o.r64.pair_state(53039, "anti", 160, batch=4)
    B = st0[0].shape[0]
    for gname, graph in sorted(graphs.items()):
        for mode, idx in (("BASE_SEPARABLE", 0), ("PARTICIPANT_NONFLAT", 1)):
            reg, windows = historical_registry(o, graph, mode=mode, copies=B)
            adj = o.graph_adjs(graph)[idx]
            for key in P.CHECKPOINTS:
                with T.no_grad():
                    hist = o.r64.rollout(st0, adj, o.models()[key], 20, keep=(1, 20))
                for backend in ("oracle", "native"):
                    st, hs = _hist_state(o, reg, windows, st0, B)
                    eng = E.Engine(st, key, controller="NATIVE", budgets=BUD, backend=backend)
                    assert eng.step(hs).exact
                    a1, b1 = _readback(st, windows)
                    assert eng.step(hs, passes=19).exact
                    a20, b20 = _readback(st, windows)
                    d1 = max(np.abs(a1 - hist[1][0].numpy()).max(), np.abs(b1 - hist[1][1].numpy()).max())
                    d20 = max(np.abs(a20 - hist[20][0].numpy()).max(), np.abs(b20 - hist[20][1].numpy()).max())
                    if backend == "oracle":
                        out["oracle_max_abs"] = max(out["oracle_max_abs"], d1, d20)
                        out["oracle_bitwise_cases"] += int(d1 == 0.0 and d20 == 0.0)
                    else:
                        out["native_h1_max_abs"] = max(out["native_h1_max_abs"], d1)
                        out["native_h20_max_abs"] = max(out["native_h20_max_abs"], d20)
                # controlled (R114) through both backends: the published controller has no historical rollout
                res = {}
                for backend in ("oracle", "native"):
                    st, hs = _hist_state(o, reg, windows, st0, B)
                    assert E.Engine(st, key, controller="R114", budgets=BUD, backend=backend).step(hs, passes=20).exact
                    res[backend] = _readback(st, windows)
                out["native_controlled_vs_oracle_h20_max_abs"] = max(
                    out["native_controlled_vs_oracle_h20_max_abs"],
                    max(np.abs(x - y).max() for x, y in zip(res["oracle"], res["native"])))
                out["cases"] += 1
    return out


def _pair(reg, n, seed):
    """Twin states (oracle / native) with identical windows and initial states."""
    sts = []
    for _ in range(2):
        st = AWRState(reg, StateCapacity(max_instances=n, max_materialized_relations=4 * n,
                                         max_bindings_per_relation=256))
        rel, mask = random_windows(reg, n, seed)
        h, _ = st.create_instances(np.arange(n, dtype=np.uint64) + 1, rel, mask)
        seed_states(st, seed + 100)
        sts.append((st, h))
    return sts


def sparse_parity(o) -> dict:
    """Native vs oracle on overlapping synthetic windows (connected, Zipf, hub, high arity)."""
    fams = {
        "connected_sparse": synthetic_registry(200, 700, 41, arity=(2, 5)),
        "zipf_overlap": synthetic_registry(150, 500, 42, arity=(1, 8), zipf=1.3),
        "hub_high_arity": synthetic_registry(80, 300, 43, arity=(8, 16), hub_fraction=0.9,
                                             limits=RegistryLimits(max_arity=16)),
    }
    out = {}
    for name, reg in fams.items():
        rec = {"one_pass_max_abs": 0.0, "multi_pass_max_abs": 0.0, "max_slot_in_degree": 0,
               "max_relation_bindings": 0, "native_vs_simple_max_abs": 0.0}
        for controller, key in (("NATIVE", "54031/STATE8_TF1"), ("R104", "54007/STATE8"),
                                ("R114", "54007/STATE8_SIGNED8"), ("R114", "54031/STATE8")):
            (so, ho), (sn, hn) = _pair(reg, 48, 5)
            rec["max_relation_bindings"] = max(rec["max_relation_bindings"], int(so.row_ref.max()))
            for h in ho.tolist():
                links = reg.window_links(so.slot_rel[h])
                for d in range(4):
                    rec["max_slot_in_degree"] = max(rec["max_slot_in_degree"], sum(1 for lk in links if lk[0] == d))
            eo = E.Engine(so, key, controller=controller, budgets=BUD, backend="oracle")
            en = E.Engine(sn, key, controller=controller, budgets=BUD, backend="native")
            assert eo.step(ho).exact and en.step(hn).exact
            rec["one_pass_max_abs"] = max(rec["one_pass_max_abs"], float(np.abs(so.pool - sn.pool).max()))
            assert eo.step(ho, passes=9).exact and en.step(hn, passes=9).exact
            rec["multi_pass_max_abs"] = max(rec["multi_pass_max_abs"], float(np.abs(so.pool - sn.pool).max()))
            res, diff, _, _ = compare_with_simple(en, hn, passes=2)
            rec["native_vs_simple_max_abs"] = max(rec["native_vs_simple_max_abs"], diff)
        out[name] = rec
    return out


def covariance(o) -> dict:
    """End-to-end gauge covariance with overlapping windows, MEAN reconciliation, controller."""
    reg = synthetic_registry(120, 400, 44, arity=(2, 7), zipf=1.1)
    rng = np.random.default_rng(118)
    g = G.random_so2x2(rng, reg.num_relations)
    h = G.random_so2x2(rng, reg.num_participants)
    out = {"max_relation_bindings": 0}
    for backend in ("native", "oracle"):
        for controller, key in (("NATIVE", "54007/STATE8_TF1"), ("R114", "54007/STATE8_SIGNED8"),
                                ("R114", "54031/STATE8")):
            n, passes = (64, 6) if backend == "native" else (24, 3)
            (s0, h0), _ = _pair(reg, n, 9)
            out["max_relation_bindings"] = max(out["max_relation_bindings"], int(s0.row_ref.max()))
            rel, mask = random_windows(reg, n, 9)

            def twin(registry, pool):
                st = AWRState(registry, StateCapacity(max_instances=n, max_materialized_relations=4 * n,
                                                      max_bindings_per_relation=256))
                hh, _ = st.create_instances(np.arange(n, dtype=np.uint64) + 1, rel, mask)
                st.pool[:] = pool
                return st, hh

            init = s0.pool.copy()
            reg_g = G.gauge_registry(reg, g_rel=g)
            reg_h = G.gauge_registry(reg, h_part=h)
            base, hb = twin(reg, init)
            gst, hg = twin(reg_g, G.gauge_rows(base, g))
            pst, hp = twin(reg_h, init)
            negs = {"state_only": twin(reg, G.gauge_rows(base, g)), "frames_only": twin(reg_g, init),
                    "inverse_frames": twin(G.gauge_registry(reg, g_rel=G.btrans(g)), G.gauge_rows(base, g))}
            for st_, hh in [(base, hb), (gst, hg), (pst, hp)] + list(negs.values()):
                assert E.Engine(st_, key, controller=controller, budgets=BUD, backend=backend).step(hh, passes=passes).exact
            expect = G.gauge_rows(base, g)
            tag = f"{backend}/{controller}/{key}"
            out[tag] = {
                "relation_gauge_covariance": float(np.abs(gst.pool - expect).max()),
                "participant_gauge_invariance": float(np.abs(pst.pool - base.pool).max()),
                "negative_state_only": float(np.abs(negs["state_only"][0].pool - expect).max()),
                "negative_frames_only": float(np.abs(negs["frames_only"][0].pool - expect).max()),
                "negative_inverse_frames": float(np.abs(negs["inverse_frames"][0].pool - expect).max()),
            }
    return out


def hidden_channels_and_passes(o) -> dict:
    """Outputs depend only on the window's relation states and *shared-participant* frames."""
    reg = synthetic_registry(200, 600, 45, arity=(3, 6), zipf=0.7)
    (st, hs), _ = _pair(reg, 40, 13)
    active = hs[:20]
    init = st.pool.copy()
    eng = E.Engine(st, "54007/STATE8_SIGNED8", controller="R114", budgets=BUD)
    assert eng.step(active, passes=5).exact
    ref = st.pool.copy()
    act_rows = np.unique(st.slot_row[active][st.slot_row[active] >= 0])
    act_rels = st.row_rel[act_rows]
    used = set()
    for h in active.tolist():
        for lk in reg.window_links(st.slot_rel[h]):
            used.update((lk[3], lk[4]))
    unused_inc = np.array([i for r in act_rels.tolist() for i in range(reg.rel_offsets[r], reg.rel_offsets[r + 1])
                           if i not in used])
    other_rel = np.setdiff1d(np.arange(reg.num_relations), act_rels)
    rng = np.random.default_rng(7)
    fr = reg.frames.copy()
    fr[unused_inc] = G.random_so2x2(rng, unused_inc.size)       # unshared incidences of active relations
    fr[np.isin(reg.inc_relation, other_rel)] = G.random_so2x2(rng, int(np.isin(reg.inc_relation, other_rel).sum()))
    reg2 = build_registry(reg.participant_ids, reg.relation_ids, reg.rel_offsets, reg.participant_ids[reg.members],
                          fr, reg.limits)
    st2 = AWRState(reg2, st.capacity)
    rel, mask = random_windows(reg, 40, 13)
    h2, _ = st2.create_instances(np.arange(40, dtype=np.uint64) + 1, rel, mask)
    pool2 = init.copy()
    inactive_rows = np.setdiff1d(np.nonzero(st2.row_ref > 0)[0], act_rows)
    pool2[inactive_rows] += rng.normal(0, 3.0, size=(inactive_rows.size, 8))  # perturb inactive relation states
    st2.pool[:] = pool2
    assert E.Engine(st2, "54007/STATE8_SIGNED8", controller="R114", budgets=BUD).step(h2[:20], passes=5).exact
    k_calls = AWRState(reg, st.capacity)
    k_calls.create_instances(np.arange(40, dtype=np.uint64) + 1, rel, mask)
    k_calls.pool[:] = init
    e3 = E.Engine(k_calls, "54007/STATE8_SIGNED8", controller="R114", budgets=BUD)
    for _ in range(5):
        assert e3.step(active).exact
    return {
        "unshared_and_foreign_frames_perturbed": int(unused_inc.size),
        "foreign_relations_perturbed": int(other_rel.size),
        "inactive_states_perturbed": int(inactive_rows.size),
        "active_rows_bitwise_unchanged": bool(np.array_equal(ref[act_rows], st2.pool[act_rows])),
        "five_passes_one_call_equals_five_calls_bitwise": bool(np.array_equal(ref, k_calls.pool)),
    }


def run() -> dict:
    t0 = time.time()
    from .frozen_oracle import get_oracle

    o = get_oracle()
    pins = o.parameter_pins()
    ev = {"phase": "R118-C", "schema": "elpis.ym_hca.r118c.v1",
          "thresholds": {"one_pass": ONE_PASS_TOL, "multi_pass": MULTI_PASS_TOL, "covariance": COV_TOL,
                         "negative_min": NEG_MIN}}
    ev["historical_replay"] = historical_replay(o)
    ev["sparse_parity"] = sparse_parity(o)
    ev["covariance"] = covariance(o)
    ev["hidden_channels"] = hidden_channels_and_passes(o)
    ev["weights_unchanged"] = o.parameter_pins() == pins == P.CHECKPOINT_SHA256
    hr, sp, cv, hc = ev["historical_replay"], ev["sparse_parity"], ev["covariance"], ev["hidden_channels"]
    cov_cases = [v for k, v in cv.items() if isinstance(v, dict)]
    gates = {
        "historical_replay_oracle_bitwise": hr["oracle_max_abs"] == 0.0 and hr["oracle_bitwise_cases"] == hr["cases"],
        "historical_replay_native": hr["native_h1_max_abs"] <= ONE_PASS_TOL and hr["native_h20_max_abs"] <= MULTI_PASS_TOL,
        "controlled_native_vs_oracle": hr["native_controlled_vs_oracle_h20_max_abs"] <= MULTI_PASS_TOL,
        "sparse_native_vs_oracle": all(r["one_pass_max_abs"] <= ONE_PASS_TOL and r["multi_pass_max_abs"] <= MULTI_PASS_TOL
                                       for r in sp.values()),
        "native_vs_simple": all(r["native_vs_simple_max_abs"] <= ONE_PASS_TOL for r in sp.values()),
        "covariance": all(c["relation_gauge_covariance"] <= COV_TOL and c["participant_gauge_invariance"] <= COV_TOL
                          for c in cov_cases),
        "negative_controls": all(min(c["negative_state_only"], c["negative_frames_only"], c["negative_inverse_frames"])
                                 >= NEG_MIN for c in cov_cases),
        "no_hidden_channels": hc["active_rows_bitwise_unchanged"],
        "pass_composition": hc["five_passes_one_call_equals_five_calls_bitwise"],
        "weights_unchanged": ev["weights_unchanged"],
    }
    gates = {k: bool(v) for k, v in gates.items()}
    ev["gates"] = gates
    ev["gate"] = "R118C_PASS" if all(gates.values()) else "R118C_FAIL"
    ev["environment"] = P.environment_fingerprint()
    ev["component_digest"] = P.component_digest()
    ev["component_sources"] = P.component_sources()
    from . import native

    ev["native_kernel"] = native.build().info
    ev["seconds"] = round(time.time() - t0, 1)
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    ev = run()
    blob = json.dumps(ev, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0 if ev["gate"] == "R118C_PASS" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
