"""R120F exploratory utility study: does the geometric update make states *useful*?

    python -B -m research.yang_mills_hca.r120_incremental_geometric_update.science --out /dev/shm/r120_science.json

Synthetic latent tracking, executing the ``utility_study`` section of PROTOCOL_R120U.json.
State fidelity and the downstream gauge-invariant link readout are reported separately.
The CGC weights are synthetic or randomly searched; none is a trained relational CGC.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import gauge as G118
from research.yang_mills_hca.r118_scalable_awr import provenance as P

from . import cgc_bridge as CGC
from . import engine_r120 as E
from . import fixtures_r120 as F

HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = HERE / "PROTOCOL_R120U.json"
REGIMES = ("CLEAN", "NOISY", "REVERSED_FLAGGED", "CONTRADICTORY_UNFLAGGED", "MIXED_RELIABILITY", "CHANGE_POINT")
ENGINE_ARMS = ("GEOMETRIC_ADDITIVE", "CGC_SYNTHETIC", "CGC_SEARCHED")
ARMS = ("NO_UPDATE", "HCA_ONLY", "DIRECT_NONGEOMETRIC", "GEOMETRIC_ADDITIVE", "GEOMETRIC_EMA", "CGC_SYNTHETIC", "CGC_SEARCHED")
LIMITS = E.Limits(max_events=64, max_active_instances=256, max_active_relations=1024, window_cache_bytes=0)
EVAL_LAST = 50


def clamp4(v):
    n = np.sqrt((v * v).sum(-1, keepdims=True))
    return v * np.minimum(1.0, 4.0 / (n + 1e-30))


class System:
    """One synthetic relational system with its link list (independent of the engine's binding lists)."""

    def __init__(self, seed, cfg):
        self.reg, self.st, self.handles, _, self.wins = F.small_state(
            seed=seed, participants=cfg["participants"], relations=cfg["relations"], windows=cfg["windows"],
            arity=tuple(cfg["arity"]), zipf=cfg["zipf"], state_seed=seed + 1, scale=0.3)
        reg, st = self.reg, self.st
        self.init = st.pool.copy()
        self.rows = np.nonzero(st.row_ref > 0)[0]
        links = {}
        for h in np.nonzero(st.inst_live)[0].tolist():
            rels = [int(r) for r in st.slot_rel[h] if r >= 0]
            for i, d in enumerate(rels):
                for s in rels[i + 1:]:
                    md = {int(p): int(reg.rel_offsets[d]) + k for k, p in enumerate(reg.members_of(d))}
                    for k, p in enumerate(reg.members_of(s).tolist()):
                        if p in md:
                            links[(min(d, s), max(d, s), p)] = (md[p], int(reg.rel_offsets[s]) + k, d, s)
        vals = sorted(links.values())
        self.l_rowd = np.array([st.rel_row[v[2]] for v in vals], dtype=np.int64)
        self.l_rows = np.array([st.rel_row[v[3]] for v in vals], dtype=np.int64)
        self.l_L = G118.bmul(reg.frames[[v[0] for v in vals]], G118.btrans(reg.frames[[v[1] for v in vals]]))

        self.eng = E.IncrementalEngine(st, F.CHECKPOINT, controller="NATIVE", limits=LIMITS, cgc_weights=CGC.zero_weights())

    def engine(self, cgc_weights):
        """The system's single engine, reset to the initial state, with the given frozen CGC weights."""
        st = self.st
        self.eng._drop_pending(self.rows)            # update-only arms leave deferred records; the state is reset
        st.pool[:] = self.init
        st.row_seq[:] = 0
        self.eng.replace_cgc_weights(cgc_weights)
        return self.eng


def make_stream(sys_, seed, regime, cfg):
    """Pre-generated event stream and latent trajectory for one (system, regime)."""
    reg, st = sys_.reg, sys_.st
    rng = np.random.default_rng([seed, REGIMES.index(regime)])
    T, n = cfg["batches"], cfg["events_per_batch"]
    z = np.zeros((st.pool.shape[0], 4))
    z[sys_.rows] = rng.normal(0.0, 0.5, size=(sys_.rows.size, 4))
    z2 = z.copy()
    z2[sys_.rows] = rng.normal(0.0, 0.5, size=(sys_.rows.size, 4))
    batches, seen = [], np.zeros(st.pool.shape[0], dtype=bool)
    for t in range(T):
        zt = z2 if (regime == "CHANGE_POINT" and t >= T // 2) else z
        rows = rng.choice(sys_.rows, size=n, replace=False)
        rel = st.row_rel[rows]
        lo = reg.rel_offsets[rel]
        inc = lo + (rng.random(n) * (reg.rel_offsets[rel + 1] - lo)).astype(np.int64)
        sd = np.full(n, 1.0 if regime == "NOISY" else 0.1)
        aux = np.full(n, 0.5)
        sign = np.ones(n)
        weight = np.ones(n)
        if regime == "MIXED_RELIABILITY":
            bad = rng.random(n) < 0.5
            sd, aux = np.where(bad, 1.5, 0.1), np.where(bad, 0.0, 1.0)
        if regime in ("REVERSED_FLAGGED", "CONTRADICTORY_UNFLAGGED"):
            sign = np.where(rng.random(n) < 0.3, -1.0, 1.0)
            if regime == "REVERSED_FLAGGED":
                weight = sign.copy()
        frames = reg.frames[inc]
        obs = G118.bapply(G118.btrans(frames), sign[:, None] * zt[rows]) + sd[:, None] * rng.normal(size=(n, 4))
        seen[rows] = True
        batches.append({"rows": rows, "ids": reg.relation_ids[rel].copy(), "parts": reg.members[inc].astype(np.int32),
                        "frames": frames, "obs": obs, "weight": weight, "aux": aux,
                        "seq": np.full(n, t + 1, dtype=np.uint64), "handles": F.affected_handles(st, rows)})
    return {"batches": batches, "z": z, "z2": z2, "eval_rows": np.nonzero(seen)[0], "T": T, "regime": regime}


def metrics(sys_, stream, t):
    st = sys_.st
    z = stream["z2"] if (stream["regime"] == "CHANGE_POINT" and t >= stream["T"] // 2) else stream["z"]
    rows = stream["eval_rows"]
    A, zz = st.pool[rows, :4], z[rows]
    na, nz = np.linalg.norm(A, axis=1), np.linalg.norm(zz, axis=1)
    cos = np.where(na > 0, (A * zz).sum(1) / (np.maximum(na, 1e-300) * nz), 0.0)
    rel = np.linalg.norm(A - zz, axis=1) / nz
    Ad, As, zd, zs = st.pool[sys_.l_rowd, :4], st.pool[sys_.l_rows, :4], z[sys_.l_rowd], z[sys_.l_rows]
    pred = (Ad * G118.bapply(sys_.l_L, As)).sum(1)
    truth = (zd * G118.bapply(sys_.l_L, zs)).sum(1)
    corr = 0.0 if pred.std() == 0 else float(np.corrcoef(pred, truth)[0, 1])
    live = st.pool[sys_.rows]
    norm = float(max(np.linalg.norm(live[:, :4], axis=1).max(), np.linalg.norm(live[:, 4:], axis=1).max()))
    return np.array([cos.mean(), rel.mean(), float((np.sign(pred) == np.sign(truth)).mean()), corr, norm])


METRIC_NAMES = ("state_cosine", "state_relative_error", "link_sign_accuracy", "link_correlation", "max_channel_norm")


def run_arm(sys_, stream, arm, propagate, weights=None, gamma=0.25, rho=CGC.RHO_B, cos_only=False):
    """One arm on one stream. Returns the metric vector averaged over the last EVAL_LAST batches."""
    eng = sys_.engine(weights if arm in ENGINE_ARMS and weights is not None else CGC.zero_weights())
    st = sys_.st
    acc, count = np.zeros(len(METRIC_NAMES)), 0
    T = stream["T"]
    for t, b in enumerate(stream["batches"]):
        rows = b["rows"]
        if arm == "NO_UPDATE":
            pass
        elif arm == "HCA_ONLY":
            assert eng.step(b["handles"]).committed
        elif arm in ENGINE_ARMS:
            rep = eng.apply(b["ids"], b["seq"], b["parts"], b["obs"], b["weight"], b["aux"], propagate=propagate)
            assert rep.committed and rep.applied == rows.size
        else:
            if arm == "DIRECT_NONGEOMETRIC":
                v = b["weight"][:, None] * b["obs"]                       # participant frame, no transport
                st.pool[rows, :4] = clamp4(st.pool[rows, :4] + gamma * v)
                st.pool[rows, 4:] = clamp4(st.pool[rows, 4:] + gamma * rho * v)
            elif arm == "GEOMETRIC_EMA":
                v = b["weight"][:, None] * G118.bapply(b["frames"], b["obs"])
                st.pool[rows, :4] = clamp4(st.pool[rows, :4] + gamma * (v - st.pool[rows, :4]))
                st.pool[rows, 4:] = clamp4(st.pool[rows, 4:] + gamma * rho * (v - st.pool[rows, 4:]))
            else:
                raise ValueError(arm)
            if propagate:
                assert eng.step(b["handles"]).committed
        if t >= T - EVAL_LAST:
            acc += metrics(sys_, stream, t)
            count += 1
    return acc / count


def search_weights(proto, progress):
    """Random search (not training): best of N draws by mean final state cosine on the training seeds."""
    us = proto["utility_study"]
    systems = [System(s, us["system"]) for s in us["seeds"]["search_training"]]
    streams = [(sy, make_stream(sy, s, rg, us["stream"])) for sy, s in zip(systems, us["seeds"]["search_training"]) for rg in REGIMES]
    rng = np.random.default_rng(130200)
    best, best_score, scores = None, -np.inf, []
    for k in range(256):
        w = rng.normal(0.0, (0.25, 0.5, 1.0)[k % 3], size=(3, 5, 8)).astype(np.float32)
        score = float(np.mean([run_arm(sy, sm, "CGC_SEARCHED", True, weights=w)[0] for sy, sm in streams]))
        scores.append(score)
        if score > best_score:
            best, best_score = w, score
            progress(f"R120F search candidate {k}: training cosine {score:.4f}")
    zero = float(np.mean([run_arm(sy, sm, "GEOMETRIC_ADDITIVE", True, weights=CGC.zero_weights())[0] for sy, sm in streams]))
    return best, {"candidates": 256, "best_training_cosine": best_score, "median_training_cosine": float(np.median(scores)),
                  "zero_weight_training_cosine": zero, "weights_sha256": CGC.weights_sha256(best),
                  "kind": "RANDOM_SEARCH_NOT_TRAINED"}


def run(progress=print) -> dict:
    proto = json.loads(PROTOCOL.read_bytes())
    us = proto["utility_study"]
    t0 = time.time()
    searched, search_info = search_weights(proto, progress)
    weights = {"GEOMETRIC_ADDITIVE": CGC.zero_weights(), "CGC_SYNTHETIC": CGC.synthetic_weights(), "CGC_SEARCHED": searched}
    raw = {}
    for seed in us["seeds"]["evaluation"]:
        sy = System(seed, us["system"])
        for rg in REGIMES:
            sm = make_stream(sy, seed, rg, us["stream"])
            for arm in ARMS:
                for prop in (False, True):
                    if arm in ("NO_UPDATE", "HCA_ONLY") and prop != (arm == "HCA_ONLY"):
                        continue
                    key = (rg, arm, "PROPAGATE" if prop else "UPDATE_ONLY")
                    raw.setdefault(key, []).append(run_arm(sy, sm, arm, prop, weights=weights.get(arm)))
        progress(f"R120F evaluation seed {seed} done")
    table = {}
    for (rg, arm, var), vals in raw.items():
        v = np.array(vals)
        table.setdefault(rg, {}).setdefault(arm, {})[var] = {
            name: {"mean": float(v[:, i].mean()), "sd": float(v[:, i].std(ddof=1)), "min": float(v[:, i].min()),
                   "max": float(v[:, i].max())} for i, name in enumerate(METRIC_NAMES)}

    def m(rg, arm, var, name):
        return table[rg][arm][var][name]["mean"]

    def per_seed(rg, arm, var, i):
        return np.array(raw[(rg, arm, var)])[:, i]

    verdict = {}
    for var in ("UPDATE_ONLY", "PROPAGATE"):
        v = {}
        for cgc in ("CGC_SYNTHETIC", "CGC_SEARCHED"):
            v[cgc + "_beats_both_lightweight_downstream_every_regime"] = bool(all(
                m(rg, cgc, var, "link_correlation") > max(m(rg, "GEOMETRIC_ADDITIVE", var, "link_correlation"),
                                                          m(rg, "GEOMETRIC_EMA", var, "link_correlation")) for rg in REGIMES))
            v[cgc + "_regimes_better_than_additive_downstream"] = [rg for rg in REGIMES if m(rg, cgc, var, "link_correlation")
                                                                   > m(rg, "GEOMETRIC_ADDITIVE", var, "link_correlation")]
            v[cgc + "_regimes_better_than_ema_downstream"] = [rg for rg in REGIMES if m(rg, cgc, var, "link_correlation")
                                                              > m(rg, "GEOMETRIC_EMA", var, "link_correlation")]
        v["geometric_beats_nongeometric_downstream_every_regime_every_seed"] = bool(all(
            np.all(per_seed(rg, "GEOMETRIC_ADDITIVE", var, 3) > per_seed(rg, "DIRECT_NONGEOMETRIC", var, 3)) for rg in REGIMES))
        v["geometric_beats_nongeometric_state_cosine_every_regime_every_seed"] = bool(all(
            np.all(per_seed(rg, "GEOMETRIC_ADDITIVE", var, 0) > per_seed(rg, "DIRECT_NONGEOMETRIC", var, 0)) for rg in REGIMES))
        v["best_arm_by_downstream_correlation"] = {rg: max((a for a in ARMS if var in table[rg][a]),
                                                           key=lambda a: m(rg, a, var, "link_correlation")) for rg in REGIMES}
        v["best_arm_by_state_cosine"] = {rg: max((a for a in ARMS if var in table[rg][a]),
                                                 key=lambda a: m(rg, a, var, "state_cosine")) for rg in REGIMES}
        verdict[var] = v
    verdict["propagation_effect_on_downstream_correlation"] = {
        arm: {rg: m(rg, arm, "PROPAGATE", "link_correlation") - m(rg, arm, "UPDATE_ONLY", "link_correlation") for rg in REGIMES}
        for arm in ARMS if arm not in ("NO_UPDATE", "HCA_ONLY")}
    cgc_claim = any(verdict[var][c + "_beats_both_lightweight_downstream_every_regime"]
                    for var in ("UPDATE_ONLY", "PROPAGATE") for c in ("CGC_SYNTHETIC", "CGC_SEARCHED"))
    return {"phase": "ELPIS_YM_AWR_R120", "schema": "elpis.ym_hca.r120u.utility.v1", "status": us["status"],
            "protocol_sha256": P.sha256_file(PROTOCOL), "metrics": METRIC_NAMES, "regimes": REGIMES, "arms": ARMS,
            "evaluation_seeds": us["seeds"]["evaluation"], "table": table, "verdict": verdict, "search": search_info,
            "synthetic_weights_sha256": CGC.weights_sha256(CGC.synthetic_weights()),
            "cgc_utility_established": bool(cgc_claim),
            "limitations": ["synthetic task and synthetic or randomly searched CGC weights; no trained relational CGC exists in the repository",
                            "the frozen HCA88 was trained for Yang-Mills relaxation, not for this readout",
                            "five evaluation seeds; exploratory, not a prospective scientific claim"],
            "environment": P.environment_fingerprint(), "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    path = pathlib.Path(args.out).resolve()
    if not path.is_relative_to("/dev/shm"):
        raise SystemExit("VOLATILE_OUTPUT_ONLY")
    res = run(progress=lambda s: print(s, flush=True))
    path.write_text(json.dumps(res, indent=1, sort_keys=True, default=P.json_default) + "\n")
    print("R120F_CGC_UTILITY_ESTABLISHED=" + str(res["cgc_utility_established"]).upper(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
