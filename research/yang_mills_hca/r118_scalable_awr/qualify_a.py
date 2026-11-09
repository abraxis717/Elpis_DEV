"""R118-A qualification: authentic operator reconstruction, parity and covariance.

Run from the repository root with an interpreter that has torch:

    python -B -m research.yang_mills_hca.r118_scalable_awr.qualify_a [--no-replay] [--out PATH]

Every check below executes the frozen operator itself (``r62.graph_step_split``
with the reconstructed ``r64.HCA`` checkpoints). Thresholds are fixed here,
before execution, and are not tuned to observed values:

* covariance / invariance / parity checks pass at max abs error <= 1e-9
  (the historical R56/R61 ``covariance_tolerance``);
* a negative control is *detected* only if its error is >= 1e-6 (1000x the
  pass tolerance); an undetected negative control fails the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time

from . import contract as C
from . import provenance as P

PASS_TOL = 1e-9
NEGATIVE_MIN = 1e-6
HORIZON = 20
GAUGE_SEED = 118_001  # draws random gauge transformations only; no fixture or training data


def _graphs(o):
    graphs = dict(o.r61["r55_structural"]["graphs"])
    graphs.update(o.r77.CFG["graphs"])
    return graphs


def _fixture(o, seed=53003, case="weak", ratio=160, batch=4):
    """Historical R77/R78 paired fixture (source at slot 0, both channels)."""
    return o.r64.pair_state(seed, case, ratio, batch=batch)


def _rollout(o, state, adj, model, steps, gains=None):
    """Historical recurrence; with ``gains`` the per-instance controller is applied each step."""
    T = o.torch
    s = state
    for _ in range(steps):
        p = o.step(s, adj, model)
        if gains is not None:
            p = _correct_torch(T, s, p, gains)
        s = p
    return s


def _correct_torch(T, s, p, gains):
    """Vectorized controller over each instance's 32 coordinates (both channels, 4 slots)."""
    alpha, beta = gains
    s32 = T.cat((s[0].reshape(s[0].shape[0], 16), s[1].reshape(s[1].shape[0], 16)), dim=1)
    p32 = T.cat((p[0].reshape(p[0].shape[0], 16), p[1].reshape(p[1].shape[0], 16)), dim=1)
    d = p32 - s32
    norm = (s32 * s32).sum(-1, keepdim=True)
    dot = (s32 * d).sum(-1, keepdim=True)
    radial = s32 * dot / T.clamp(norm, min=C.RADIAL_EPSILON)
    out = p32 + alpha * radial + beta * (d - radial)
    return out[:, :16].reshape(-1, 4, 4), out[:, 16:].reshape(-1, 4, 4)


def _maxdiff(x, y):
    return max(float((u - v).abs().max()) for u, v in zip(x, y))


def _rand_so2x2(rng, o):
    return o.pm.rotation(rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi))


def _rand_o4(rng, T):
    """Random orthogonal 4x4 (QR of a Gaussian; random sign so reflections occur)."""
    g = T.Generator().manual_seed(rng.randrange(1 << 30))
    q, r = T.linalg.qr(T.randn(4, 4, generator=g, dtype=T.float64))
    q = q * T.sign(T.diagonal(r))
    if rng.random() < 0.5:
        q[:, 0] = -q[:, 0]
    return q


def covariance_suite(o) -> dict:
    T = o.torch
    pm = o.pm
    models = o.models()
    graphs = _graphs(o)
    rng = random.Random(GAUGE_SEED)
    state = _fixture(o)
    res = {
        "so2x2_covariance_native_h1": 0.0,
        "so2x2_covariance_native_h20": 0.0,
        "so2x2_covariance_controlled_h20": 0.0,
        "o4_covariance_native_h20": 0.0,
        "participant_gauge_invariance_h20": 0.0,
        "slot_permutation_equivariance_h20": 0.0,
        "negative_state_only_h1": math.inf,
        "negative_links_only_h1": math.inf,
        "negative_transposed_link_convention_h1": math.inf,
        "negative_nonorthogonal_scaling_h1": math.inf,
        "negative_slot_mismatched_gauge_h1": math.inf,
        "cases": 0,
    }
    with T.no_grad():
        for gname, graph in sorted(graphs.items()):
            flat, nonflat = o.graph_adjs(graph)
            for adj_name, adj in (("BASE", flat), ("NONFLAT", nonflat)):
                for key in sorted(models):
                    m = models[key]
                    gains = C.controller_gains("R114", key)
                    res["cases"] += 1
                    g = {d: _rand_so2x2(rng, o) for d in range(4)}
                    gs, gadj = pm.gauge_state(state, g), pm.gauged_links(adj, g)
                    # Positive: jointly transformed state and links.
                    y1 = o.step(state, adj, m)
                    res["so2x2_covariance_native_h1"] = max(
                        res["so2x2_covariance_native_h1"], _maxdiff(o.step(gs, gadj, m), pm.gauge_state(y1, g)))
                    y20 = _rollout(o, state, adj, m, HORIZON)
                    res["so2x2_covariance_native_h20"] = max(
                        res["so2x2_covariance_native_h20"],
                        _maxdiff(_rollout(o, gs, gadj, m, HORIZON), pm.gauge_state(y20, g)))
                    c20 = _rollout(o, state, adj, m, HORIZON, gains)
                    res["so2x2_covariance_controlled_h20"] = max(
                        res["so2x2_covariance_controlled_h20"],
                        _maxdiff(_rollout(o, gs, gadj, m, HORIZON, gains), pm.gauge_state(c20, g)))
                    # O(4) per relation node (finding: the per-node law uses only R^4 dot products).
                    q = {d: _rand_o4(rng, T) for d in range(4)}
                    res["o4_covariance_native_h20"] = max(
                        res["o4_covariance_native_h20"],
                        _maxdiff(_rollout(o, pm.gauge_state(state, q), pm.gauged_links(adj, q), m, HORIZON),
                                 pm.gauge_state(y20, q)))
                    # Negative controls: each must change the output by >= NEGATIVE_MIN.
                    expect = pm.gauge_state(y1, g)
                    res["negative_state_only_h1"] = min(
                        res["negative_state_only_h1"], _maxdiff(o.step(gs, adj, m), expect))
                    res["negative_links_only_h1"] = min(
                        res["negative_links_only_h1"], _maxdiff(o.step(state, gadj, m), expect))
                    wrong = {d: (src.clone(), T.stack([g[int(s)] @ op @ g[d].T for s, op in zip(src, ops)]))
                             for d, (src, ops) in adj.items()}
                    res["negative_transposed_link_convention_h1"] = min(
                        res["negative_transposed_link_convention_h1"], _maxdiff(o.step(gs, wrong, m), expect))
                    sc = {d: 1.5 * T.eye(4, dtype=T.float64) for d in range(4)}
                    sstate = pm.gauge_state(state, sc)
                    sadj = {d: (src.clone(), T.stack([sc[d] @ op @ T.linalg.inv(sc[int(s)]) for s, op in zip(src, ops)]))
                            for d, (src, ops) in adj.items()}
                    res["negative_nonorthogonal_scaling_h1"] = min(
                        res["negative_nonorthogonal_scaling_h1"],
                        _maxdiff(o.step(sstate, sadj, m), pm.gauge_state(y1, sc)))
                    gperm = {d: g[(d + 1) % 4] for d in range(4)}
                    res["negative_slot_mismatched_gauge_h1"] = min(
                        res["negative_slot_mismatched_gauge_h1"],
                        _maxdiff(o.step(pm.gauge_state(state, gperm), gadj, m), expect))
            # Participant gauge: U[r,p] -> U[r,p] h_p^T leaves every link unchanged.
            base = {(rid, p): pm.rotation(.19 + .17 * rid - .067 * p, -.41 + .103 * rid + .049 * p)
                    for rid, mem in enumerate(graph) for p in mem}
            mats = o.r62.transport_matrices(None, base, "PARTICIPANT_NONFLAT", o.r61, pm.R3Adapter(), T)
            h = {p: _rand_so2x2(rng, o) for p in range(10)}
            adj_h = pm.link_builder(graph, {(r, p): u @ h[p].T for (r, p), u in mats.items()})
            adj_0 = pm.link_builder(graph, mats)
            # Slot permutation: relabel relation rows and their frames consistently.
            perm = [2, 0, 3, 1]  # new slot i holds old relation perm[i]
            inv = {old: new for new, old in enumerate(perm)}
            pgraph = [graph[perm[i]] for i in range(4)]
            pmats = {(inv[r], p): u for (r, p), u in mats.items()}
            adj_p = pm.link_builder(pgraph, pmats)
            pstate = tuple(ch[:, perm, :] for ch in state)
            for key in sorted(models):
                m = models[key]
                y0 = _rollout(o, state, adj_0, m, HORIZON)
                res["participant_gauge_invariance_h20"] = max(
                    res["participant_gauge_invariance_h20"], _maxdiff(_rollout(o, state, adj_h, m, HORIZON), y0))
                yp = _rollout(o, pstate, adj_p, m, HORIZON)
                res["slot_permutation_equivariance_h20"] = max(
                    res["slot_permutation_equivariance_h20"], _maxdiff(yp, tuple(ch[:, perm, :] for ch in y0)))
    return res


def isolation_suite(o) -> dict:
    """Batch rows are independent instances; empty / isolated slots are inert."""
    T = o.torch
    models = o.models()
    graph = o.r77.CFG["graphs"]["R77_OOD_D4"]
    _, adj = o.graph_adjs(graph)
    m = models["54031/STATE8_SIGNED8"]
    out = {}
    with T.no_grad():
        batch = _fixture(o, seed=53011, case="anti", batch=4)  # 8 instances
        full = _rollout(o, batch, adj, m, HORIZON)
        worst = 0.0
        bitwise = True
        for i in range(batch[0].shape[0]):
            one = _rollout(o, tuple(ch[i:i + 1] for ch in batch), adj, m, HORIZON)
            worst = max(worst, _maxdiff(one, tuple(ch[i:i + 1] for ch in full)))
            bitwise &= all(bool(T.equal(u, v[i:i + 1])) for u, v in zip(one, full))
        out["batch_vs_single_instance_h20_max_abs"] = worst
        out["batch_vs_single_instance_bitwise"] = bitwise
        # Perturbing instance 0 must not move any other instance (no cross-instance channel).
        pert = (batch[0].clone(), batch[1].clone())
        pert[0][0, 1, :] += 0.5
        moved = _rollout(o, pert, adj, m, HORIZON)
        out["cross_instance_leak_h20"] = max(float((u[1:] - v[1:]).abs().max()) for u, v in zip(moved, full))
        out["perturbed_instance_changed_h20"] = max(float((u[:1] - v[:1]).abs().max()) for u, v in zip(moved, full))
        # Slot without in-links and zero state stays exactly zero (historical R63 control).
        iso = {d: v for d, v in adj.items() if d != 3}
        iso = {d: (src[src != 3], ops[src != 3]) for d, (src, ops) in iso.items()}
        iso = {d: v for d, v in iso.items() if len(v[0])}
        z = (batch[0].clone(), batch[1].clone())
        z[0][:, 3, :] = 0
        z[1][:, 3, :] = 0
        y = _rollout(o, z, iso, m, HORIZON)
        out["isolated_zero_slot_max_abs_h20"] = max(float(ch[:, 3, :].abs().max()) for ch in y)
    return out


def controller_parity(o) -> dict:
    """Vectorized controller vs the published scalar algebra (reference.correct_node)."""
    T = o.torch
    models = o.models()
    graph = o.r77.CFG["graphs"]["R77_OOD_A3"]
    _, adj = o.graph_adjs(graph)
    worst = 0.0
    n = 0
    with T.no_grad():
        s = _fixture(o, seed=53017, case="reverse", batch=4)
        for key in sorted(models):
            for controller in ("R104", "R107", "R114"):
                gains = C.controller_gains(controller, key)
                p = o.step(s, adj, models[key])
                vec = _correct_torch(T, s, p, gains)
                for i in range(s[0].shape[0]):
                    s32 = C.to_state32(s[0][i].tolist(), s[1][i].tolist())
                    p32 = C.to_state32(p[0][i].tolist(), p[1][i].tolist())
                    ex = C.correct_instance_exact(s32, p32, controller, key)
                    v32 = C.to_state32(vec[0][i].tolist(), vec[1][i].tolist())
                    worst = max(worst, max(abs(a - b) for a, b in zip(ex, v32)))
                    n += 1
    return {"vectorized_vs_reference_max_abs": worst, "instances_compared": n}


def timing(o) -> dict:
    """Historical-size transition latency of the oracle (R77 evaluation batch, 6 stacked models)."""
    T = o.torch
    models = o.models()
    keys = sorted(models)
    state = o.r77.join_states()
    per = state[0].shape[0]
    w = T.stack([models[k].gates.weight.detach() for k in keys])
    b = T.stack([models[k].gates.bias.detach() for k in keys])
    law = o.r62._StackedGates(w, b, o.r30, T, per)
    init = tuple(x.repeat(len(keys), 1, 1) for x in state)
    _, adj = o.graph_adjs(o.r77.CFG["graphs"]["R77_OOD_D4"])
    samples = []
    with T.no_grad():
        for _ in range(5):
            o.r62.graph_step_split(init, adj, adj, law, T)
        for _ in range(50):
            t0 = time.perf_counter_ns()
            o.r62.graph_step_split(init, adj, adj, law, T)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    return {
        "instances_per_step": init[0].shape[0],
        "step_ms_p50": samples[len(samples) // 2],
        "step_ms_max": samples[-1],
        "torch_threads": T.get_num_threads(),
    }


def run(replay=True) -> dict:
    t0 = time.time()
    from . import frozen_oracle

    ev = {"phase": "R118-A", "schema": "elpis.ym_hca.r118a.v1"}
    ev["provenance"] = P.verify_manifest()
    ev["snapshot_digest"] = P.snapshot_digest()
    ev["checkpoints"] = {k: v["sha256"] for k, v in P.checkpoint_records().items()}
    if replay:
        from . import historical_replay

        rep = historical_replay.replay(("verify", "spot", "run"))
        ev["historical_replay"] = {
            "disposition": rep["disposition"],
            "r78_result_byte_identical": rep["modes"]["run"]["byte_identical"],
            "r78_replayed_sha256": rep["modes"]["run"]["replayed_sha256"],
            "verify_tail": rep["modes"]["verify"]["stdout_tail"][-1],
            "spot_tail": rep["modes"]["spot"]["stdout_tail"][-1],
            "exits": {k: v["exit"] for k, v in rep["modes"].items()},
        }
    o = frozen_oracle.get_oracle()
    models = o.models()
    ev["oracle"] = {
        "executed_modules": sorted(o.executed_modules),
        "source_sha256": o.source_sha256,
        "torch_threads_before_import": o.torch_threads_before,
        "torch_threads_after_import": o.torch_threads_after,
        "parameter_pins": o.parameter_pins(),
        "parameter_count": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
    }
    pins_before = o.parameter_pins()
    rows = o.r77.evaluate(models)
    sealed = o.r77_report()["rows"]
    canon = lambda obj: hashlib.sha256(P.canonical_json(obj)).hexdigest()  # noqa: E731
    ev["r77_full_evaluation_replay"] = {
        "rows": len(rows),
        "rows_canonical_sha256": canon(rows),
        "sealed_rows_canonical_sha256": canon(sealed),
        "bitwise_identical": canon(rows) == canon(sealed),
    }
    ev["covariance"] = covariance_suite(o)
    ev["isolation"] = isolation_suite(o)
    ev["controller_parity"] = controller_parity(o)
    ev["oracle_timing"] = timing(o)
    ev["weights_unchanged_after_suite"] = o.parameter_pins() == pins_before == ev["checkpoints"]
    cov = ev["covariance"]
    positives = [k for k in cov if not k.startswith("negative") and k != "cases"]
    negatives = [k for k in cov if k.startswith("negative")]
    gates = {
        "provenance": ev["provenance"]["files_verified"] == P.MANIFEST_FILE_COUNT,
        "historical_replay": (not replay) or ev["historical_replay"]["disposition"] == "HISTORICAL_R78_REPLAY_PASS",
        "checkpoint_hashes": all(ev["oracle"]["parameter_pins"][k] == P.CHECKPOINT_SHA256[k] for k in P.CHECKPOINTS),
        "r77_parity": ev["r77_full_evaluation_replay"]["bitwise_identical"],
        "covariance": all(cov[k] <= PASS_TOL for k in positives),
        "negative_controls_detected": all(cov[k] >= NEGATIVE_MIN for k in negatives),
        "isolation": ev["isolation"]["batch_vs_single_instance_h20_max_abs"] <= PASS_TOL
        and ev["isolation"]["cross_instance_leak_h20"] == 0.0
        and ev["isolation"]["perturbed_instance_changed_h20"] >= NEGATIVE_MIN
        and ev["isolation"]["isolated_zero_slot_max_abs_h20"] == 0.0,
        "controller_parity": ev["controller_parity"]["vectorized_vs_reference_max_abs"] <= PASS_TOL,
        "no_weight_mutation": ev["weights_unchanged_after_suite"],
    }
    ev["gates"] = gates
    ev["gate"] = "R118A_PASS" if all(gates.values()) else "R118A_FAIL"
    ev["thresholds"] = {"pass_tol": PASS_TOL, "negative_min": NEGATIVE_MIN, "horizon": HORIZON}
    ev["environment"] = P.environment_fingerprint()
    ev["component_digest"] = P.component_digest()
    ev["component_sources"] = P.component_sources()
    ev["seconds"] = round(time.time() - t0, 2)
    ev["provenance_after"] = P.verify_manifest()
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    ev = run(replay=not args.no_replay)
    blob = json.dumps(ev, indent=1, sort_keys=True, default=P.json_default)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob + "\n")
    print(blob)
    return 0 if ev["gate"] == "R118A_PASS" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
