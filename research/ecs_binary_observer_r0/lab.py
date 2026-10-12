"""ECS Binary Observation R0 laboratory (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).

Worlds are real native K1 states learning under the Cognitive R0 regime. At every state the system answers two
observed QUERYs (the standing X_REF batch and a workload batch of 1..256 rows); the observer encodes each accepted
answer. The laboratory separately holds ground truth the observer never sees: f on X_HELD, W itself (for the
neuron-permuted copy), snapshots and counters. Nothing here is imported by src/.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import struct
from array import array
from pathlib import Path

import numpy as np

from elpis.ECS.k1 import K1Library, K1State

from . import law as L
from . import protocol as P
from .observer import Observer

S = P.SPEC
D, N = S["dim"], S["width"]
T = S["episodes"] + 1
M64 = (1 << 64) - 1


def load_k1(path: Path) -> K1Library:
    return K1Library(ctypes.CDLL(str(path)))


def cubic(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Z = np.atleast_2d(X) @ W
    return (0.5 * Z + 0.5 * Z * Z + 0.5 * Z * Z * Z).sum(axis=1)


def _inputs(g: np.random.Generator, rows: int, signs) -> np.ndarray:
    centre = np.zeros((rows, D))
    centre[:, 0] = np.asarray(signs, dtype=float) * S["region_offset"]
    return np.ascontiguousarray(g.normal(0.0, S["input_scale"], size=(rows, D)) + centre)


def _nonidentity_permutation(g: np.random.Generator, n: int) -> np.ndarray:
    p = g.permutation(n)
    while (p == np.arange(n)).all():
        p = g.permutation(n)
    return p


_ALT = np.where(np.arange(S["ref_rows"]) % 2 == 0, 1.0, -1.0)
X_REF = _inputs(P.rng("global", "x-ref"), S["ref_rows"], _ALT)
X_HELD = _inputs(P.rng("global", "x-held"), S["held_rows"], np.where(np.arange(S["held_rows"]) % 2 == 0, -1.0, 1.0))
ROWPERM = _nonidentity_permutation(P.rng("global", "rowperm"), S["ref_rows"])
X_ROWPERM = np.ascontiguousarray(X_REF[ROWPERM])


class World:
    def __init__(self, wid: str):
        self.id = wid
        self.w0 = P.rng(wid, "init").normal(0.0, S["init_scale"], size=(D, N))
        teachers = [P.rng(wid, f"teacher-{k}").normal(0.0, S["teacher_scale"], size=(D, S["teacher_width"]))
                    for k in range(S["teachers_per_world"])]
        ge = P.rng(wid, "episodes")
        self.episodes = []
        for e in range(S["episodes"]):
            k = int(ge.integers(S["teachers_per_world"]))
            sign = 1.0 if int(ge.integers(2)) else -1.0
            steps = int(ge.choice(S["step_choices"]))
            x = _inputs(P.rng(wid, f"experience-{e}"), S["experience_rows"], np.full(S["experience_rows"], sign))
            self.episodes.append((x, np.ascontiguousarray(cubic(teachers[k], x)), steps))
        gw = P.rng(wid, "workload")
        self.workload = []
        for t in range(T):
            rows = int(gw.choice(S["workload_rows"]))
            signs = np.where(gw.random(rows) < 0.5, 1.0, -1.0)
            self.workload.append(_inputs(P.rng(wid, f"workload-{t}"), rows, signs))
        self.pi = _nonidentity_permutation(P.rng(wid, "neuron-perm"), N)
        self.random_bits = P.rng(wid, "random-bits").integers(0, 2, size=(T, L.PAYLOAD_BITS))


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(bytes(p) if isinstance(p, (bytes, bytearray)) else np.asarray(p, dtype="<f8").tobytes())
    return h.hexdigest()


def _open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def run_world(k1: K1Library, world: World, observers: dict | None = None, parity: bool = False) -> dict:
    """One world: LEARN episodes between states; at each state the observed QUERYs, the lab probes and, when
    observers are given, the live records. Returns raw observations, records, and the interference witnesses."""
    states, failures = [], []
    fds = _open_fds()
    with K1State.create(k1, D, N, np.ascontiguousarray(world.w0).ravel(), max_rows=S["max_rows"]) as st:
        for t in range(T):
            if t > 0:
                x, y, steps = world.episodes[t - 1]
                st.learn(x, y, S["learning_rate"], steps)
            wl = world.workload[t]
            ref_v, ref_d = st.query_identity(X_REF)          # observed
            wl_v, wl_d = st.query_identity(wl)                # observed
            rp_v, rp_d = st.query_identity(X_ROWPERM)        # ROWPERM control probe
            held = np.array(st.query(X_HELD))                 # laboratory ground truth: never observed
            w = np.array(st.w()).reshape(D, N)                # laboratory ground truth: never observed
            snap = st.snapshot()
            if not (ref_d == wl_d == rp_d == st.state_digest()):
                failures.append(f"{world.id} t={t}: identities of one state disagree")
            with K1State.create(k1, D, N, np.ascontiguousarray(w[:, world.pi]).ravel(), max_rows=64) as twin:
                pv, pd = twin.query_identity(X_REF)           # neuron-permuted twin (lab-only state)
            s = {"t": t, "digest": ref_d, "ref": np.array(ref_v), "rowperm": np.array(rp_v), "held": held,
                 "wl_rows": int(wl.shape[0]), "perm_digest": pd, "perm_ref": np.array(pv),
                 "answers_sha": _sha(ref_v, wl_v, rp_v, pv), "snapshot_sha": hashlib.sha256(snap).hexdigest()}
            if observers:
                s["records"] = {}
                queries = {"ref": (X_REF, ref_v, ref_d), "wl": (wl, wl_v, wl_d), "rowperm": (X_ROWPERM, rp_v, rp_d),
                           "perm": (X_REF, pv, pd)}
                for name, obs in observers.items():
                    recs = {}
                    for q, (xq, vq, dq) in queries.items():
                        rows = int(xq.shape[0])
                        values = array("d", vq)
                        rec = obs.encode(D, rows, xq, values, dq)
                        if parity:
                            again = obs.encode(D, rows, xq, values, dq)
                            ref = L.encode(obs.law, D, rows, xq.ravel().tolist(), list(vq), dq)
                            if rec != again or rec != ref or obs.check(rec) != 0 or len(rec) != L.RECORD_BYTES:
                                failures.append(f"{world.id} t={t} {name}/{q}: repeatability or parity")
                            if (rows > L.WINDOW_ROWS) != bool(rec[7] & L.FLAG_TRUNCATED):
                                failures.append(f"{world.id} t={t} {name}/{q}: window flag")
                        recs[q] = rec
                    s["records"][name] = recs
            states.append(s)
        counters = st.stats()
    if _open_fds() != fds:
        failures.append(f"{world.id}: open file descriptors changed")
    return {"world": world.id, "states": states, "counters": counters, "failures": failures}


def interference_witness(result: dict) -> dict:
    keys = ("forward_calls", "learn_calls", "commits", "refusals", "heap_allocations", "steps_executed")
    return {"answers": [s["answers_sha"] for s in result["states"]],
            "digests": [s["digest"].hex() for s in result["states"]],
            "snapshots": [s["snapshot_sha"] for s in result["states"]],
            "counters": {k: result["counters"][k] for k in keys}}


def records_digest(results: list) -> str:
    h = hashlib.sha256()
    for r in results:
        for s in r["states"]:
            for name in sorted(s.get("records", {})):
                for q in sorted(s["records"][name]):
                    h.update(s["records"][name][q])
    return h.hexdigest()


def observations_digest(results: list) -> str:
    h = hashlib.sha256()
    for r in results:
        for s in r["states"]:
            for key in ("ref", "rowperm", "held", "perm_ref"):
                h.update(np.asarray(s[key], dtype="<f8").tobytes())
            h.update(s["digest"] + s["perm_digest"] + s["snapshot_sha"].encode())
    return h.hexdigest()


def encode_post_hoc(results: list, observers: dict) -> None:
    """DESIGN only: records computed after calibration from the stored answers (same law, same native observer)."""
    for r in results:
        for s in r["states"]:
            s["records"] = {}
            for name, obs in observers.items():
                s["records"][name] = {
                    "ref": obs.encode(D, S["ref_rows"], X_REF, array("d", s["ref"]), s["digest"]),
                    "rowperm": obs.encode(D, S["ref_rows"], X_ROWPERM, array("d", s["rowperm"]), s["digest"]),
                    "perm": obs.encode(D, S["ref_rows"], X_REF, array("d", s["perm_ref"]), s["perm_digest"])}


# --- calibration (DESIGN only) ---------------------------------------------------------------------------------

def _splitmix(seed: int):
    state = seed
    while True:
        state = (state + 0x9E3779B97F4A7C15) & M64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        yield z ^ (z >> 31)


def sproj_tables():
    stream = _splitmix(int.from_bytes(hashlib.sha256(b"ebo-r0/sproj/v1").digest()[:8], "little"))
    index, sign = [], []
    for _ in range(L.PAYLOAD_BITS):
        rows, signs = [], []
        while len(rows) < L.SPROJ_TERMS:
            r = next(stream) % L.WINDOW_ROWS
            if r not in rows:
                rows.append(r)
        for _ in range(L.SPROJ_TERMS):
            signs.append(1 if next(stream) >> 63 else -1)
        index.append(tuple(rows))
        sign.append(tuple(signs))
    return index, sign


def calibrate(design: list) -> dict:
    pooled = np.concatenate([s["ref"] for r in design for s in r["states"]])
    thresholds = np.quantile(pooled, [k / 16 for k in range(1, 16)], method="linear")
    gray = L.gray4_blob(thresholds)
    L.parse_law(gray)                                  # refuses unless strictly increasing
    mu, sd = float(pooled.mean()), float(pooled.std())
    index, sign = sproj_tables()
    probe = L.parse_law(L.sproj_blob(mu, 1.0 / sd, index, sign, [0.0] * L.PAYLOAD_BITS))
    sums = np.array([L.sproj_sums(probe, list(s["ref"]), S["ref_rows"]) for r in design for s in r["states"]])
    tau = np.median(sums, axis=0)
    sproj = L.sproj_blob(mu, 1.0 / sd, index, sign, tau.tolist())
    return {"GRAY4": gray, "SPROJ": sproj, "pooled_values": int(pooled.size)}


# --- features ----------------------------------------------------------------------------------------------------

def _pm(bits) -> np.ndarray:
    return 2.0 * np.asarray(bits, dtype=float) - 1.0


def _bytes_bits(data: bytes) -> list:
    return [(data[k >> 3] >> (k & 7)) & 1 for k in range(8 * len(data))]


def payload_features(name: str, law: L.Law, record: bytes) -> np.ndarray:
    payload = record[32:64]
    if name == "GRAY4":
        mids = L.gray4_midpoints(law)
        return np.array([mids[v] for v in L.gray4_levels(payload, record[11])])
    return _pm(L.payload_bits(payload))


def family_features(name: str, s: dict, laws: dict, world: World, query: str = "ref") -> np.ndarray:
    if name in P.ENCODERS:
        return payload_features(name, laws[name], s["records"][name][query])
    values = s["ref"] if query == "ref" else s["rowperm"]
    if name == "RAW16":
        raw = L.raw16(list(values))
        return np.array(struct.unpack("<16e", raw))
    if name == "RAW4":
        return np.array(struct.unpack("<4d", L.raw4(list(values))))
    if name == "DIGEST":
        return _pm(_bytes_bits(s["digest"][:8]))
    if name == "RANDOM":
        return _pm(world.random_bits[s["t"]])
    raise KeyError(name)


FAMILIES = P.ENCODERS + P.COMPARATORS + P.CONTROLS


def state_matrix(results: list, worlds: dict, laws: dict, name: str, query: str = "ref"):
    X = np.array([family_features(name, s, laws, worlds[r["world"]], query) for r in results for s in r["states"]])
    Y = np.array([s["held"] for r in results for s in r["states"]])
    groups = np.array([i for i, r in enumerate(results) for _ in r["states"]])
    return X, Y, groups


def transition_matrix(results: list, worlds: dict, laws: dict, name: str, reset: bool = False):
    X, Y, groups = [], [], []
    for i, r in enumerate(results):
        f = [family_features(name, s, laws, worlds[r["world"]]) for s in r["states"]]
        for t in range(len(r["states"]) - 1):
            a, b = (f[0], f[0]) if reset else (f[t], f[t + 1])
            X.append(np.concatenate([a, b]))
            Y.append(r["states"][t + 1]["held"] - r["states"][t]["held"])
            groups.append(i)
    return np.array(X), np.array(Y), np.array(groups)


# --- decoders ----------------------------------------------------------------------------------------------------

class Ridge:
    def __init__(self, X: np.ndarray, Y: np.ndarray, lam: float):
        self.mu, sd = X.mean(axis=0), X.std(axis=0)
        self.keep = sd > 0
        self.sd = np.where(self.keep, sd, 1.0)
        Z = ((X - self.mu) / self.sd)[:, self.keep]
        self.ym = Y.mean(axis=0)
        self.B = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - self.ym)) if Z.shape[1] else None
        self.lam = lam

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.B is None:
            return np.tile(self.ym, (X.shape[0], 1))
        return ((X - self.mu) / self.sd)[:, self.keep] @ self.B + self.ym


def choose_lambda(X: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> float:
    best, best_mse = None, math.inf
    for lam in P.RIDGE_GRID:
        err = 0.0
        for g in np.unique(groups):
            train, test = groups != g, groups == g
            err += float(((Ridge(X[train], Y[train], lam).predict(X[test]) - Y[test]) ** 2).sum())
        mse = err / Y.size
        if mse <= best_mse:   # ties go to the larger lambda (the grid ascends)
            best, best_mse = lam, mse
    return best


def r2(pred: np.ndarray, Y: np.ndarray) -> float:
    return float(1.0 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum())


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = rankdata(np.asarray(a, dtype=float)), rankdata(np.asarray(b, dtype=float))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def train_decoders(design: list, worlds: dict, laws: dict) -> dict:
    out = {}
    for name in FAMILIES:
        Xs, Ys, gs = state_matrix(design, worlds, laws, name)
        Xt, Yt, gt = transition_matrix(design, worlds, laws, name)
        out[name] = {"state_lambda": choose_lambda(Xs, Ys, gs), "transition_lambda": choose_lambda(Xt, Yt, gt)}
    return out


def fit_decoders(design: list, worlds: dict, laws: dict, lambdas: dict) -> dict:
    fitted = {}
    for name in FAMILIES:
        Xs, Ys, _ = state_matrix(design, worlds, laws, name)
        Xt, Yt, _ = transition_matrix(design, worlds, laws, name)
        fitted[name] = (Ridge(Xs, Ys, lambdas[name]["state_lambda"]),
                        Ridge(Xt, Yt, lambdas[name]["transition_lambda"]))
    return fitted


# --- probes ------------------------------------------------------------------------------------------------------

def _world_shift(n_states: int, per_world: int) -> np.ndarray:
    """A derangement pairing each state (or transition) with the same index of the next world."""
    return (np.arange(n_states) + per_world) % n_states


def probes(evaluation: list, worlds: dict, laws: dict, fitted: dict) -> dict:
    out = {"S1": {}, "S2a": {}, "S2b": {}, "S3": {}, "S4": {}, "information": {}}
    states = [(r, s) for r in evaluation for s in r["states"]]
    for name in P.ENCODERS:
        recs = [s["records"][name]["ref"] for _, s in states]
        perms = [s["records"][name]["perm"] for _, s in states]
        out["S1"][name] = {
            "states": len(states),
            "digest_prefix_differs": sum(s["digest"][:8] != s["perm_digest"][:8] for _, s in states) / len(states),
            "payload_identical": sum(a[32:] == b[32:] for a, b in zip(recs, perms)) / len(states),
            "mean_bit_disagreement": float(np.mean([hamming(a[32:], b[32:]) / 256 for a, b in zip(recs, perms)])),
            "digest_bytes_identical_to_twin": sum(a[16:24] == b[16:24] for a, b in zip(recs, perms))}
        changed, total = 0, 0
        for r in evaluation:
            for t in range(len(r["states"]) - 1):
                a, b = r["states"][t]["records"][name]["ref"], r["states"][t + 1]["records"][name]["ref"]
                changed += a[32:] != b[32:]
                total += 1
        out["S2a"][name] = {"transitions": total, "payload_differs": changed / total}

    # S2b: graded similarity on within-world state pairs
    pairs = [(r, i, j) for r in evaluation for i in range(len(r["states"])) for j in range(i + 1, len(r["states"]))]
    dist = [float(np.linalg.norm(r["states"][i]["held"] - r["states"][j]["held"])) for r, i, j in pairs]
    for name in P.ENCODERS:
        ham = [hamming(r["states"][i]["records"][name]["ref"][32:], r["states"][j]["records"][name]["ref"][32:])
               for r, i, j in pairs]
        out["S2b"][name] = {"pairs": len(pairs), "rho": spearman(ham, dist)}
    out["S2b"]["DIGEST"] = {"pairs": len(pairs), "rho": spearman(
        [hamming(r["states"][i]["digest"][:8], r["states"][j]["digest"][:8]) for r, i, j in pairs], dist)}
    for name in P.COMPARATORS:
        feats = {id(s): family_features(name, s, laws, worlds[r["world"]]) for r in evaluation for s in r["states"]}
        out["S2b"][name] = {"pairs": len(pairs), "rho": spearman(
            [float(np.linalg.norm(feats[id(r["states"][i])] - feats[id(r["states"][j])])) for r, i, j in pairs], dist),
            "metric": "L2 of decoded values"}

    # S3: held-out state prediction; S4: transition prediction
    per_world = len(evaluation[0]["states"])
    for name in FAMILIES:
        Xs, Ys, _ = state_matrix(evaluation, worlds, laws, name)
        Xt, Yt, _ = transition_matrix(evaluation, worlds, laws, name)
        sdec, tdec = fitted[name]
        entry_s = {"r2": r2(sdec.predict(Xs), Ys), "lambda": sdec.lam, "features": int(Xs.shape[1])}
        entry_t = {"r2": r2(tdec.predict(Xt), Yt), "lambda": tdec.lam, "features": int(Xt.shape[1])}
        if name in P.ENCODERS:
            shift_s = _world_shift(len(Ys), per_world)
            reset_s = np.array([family_features(name, r["states"][0], laws, worlds[r["world"]])
                                for r in evaluation for _ in r["states"]])
            rowperm_s, _, _ = state_matrix(evaluation, worlds, laws, name, query="rowperm")
            entry_s["controls"] = {"SHUFFLED": r2(sdec.predict(Xs), Ys[shift_s]),
                                   "RESET": r2(sdec.predict(reset_s), Ys),
                                   "ROWPERM": r2(sdec.predict(rowperm_s), Ys)}
            shift_t = _world_shift(len(Yt), per_world - 1)
            Xr, _, _ = transition_matrix(evaluation, worlds, laws, name, reset=True)
            entry_t["controls"] = {"SHUFFLED": r2(tdec.predict(Xt), Yt[shift_t]), "RESET": r2(tdec.predict(Xr), Yt)}
        out["S3"][name] = entry_s
        out["S4"][name] = entry_t
    for name in P.ENCODERS:
        for key, extra in (("S3", ("DIGEST", "RANDOM")), ("S4", ("DIGEST", "RANDOM"))):
            controls = dict(out[key][name]["controls"])
            controls.update({c: out[key][c]["r2"] for c in extra})
            out[key][name]["controls"] = controls
            out[key][name]["margin"] = out[key][name]["r2"] - max(controls.values())
            out[key][name]["vs_RAW16"] = out[key][name]["r2"] - out[key]["RAW16"]["r2"]
    out["information"] = information(evaluation, laws)
    return out


def information(evaluation: list, laws: dict) -> dict:
    states = [s for r in evaluation for s in r["states"]]
    info = {}
    for name in P.ENCODERS:
        payloads = [s["records"][name]["ref"][32:] for s in states]
        bits = np.array([L.payload_bits(p) for p in payloads], dtype=float)
        p = bits.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0.0))
        collisions = 0
        seen = {}
        for s, pay in zip(states, payloads):
            seen.setdefault(pay, set()).add(s["digest"])
        for digests in seen.values():
            n = len(digests)
            collisions += n * (n - 1) // 2
        n_states = len(states)
        entry = {"states": n_states, "distinct_payloads": len(seen), "colliding_state_pairs": collisions,
                 "collision_rate": collisions / (n_states * (n_states - 1) // 2),
                 "bit_entropy_mean": float(h.mean()), "bit_entropy_min": float(h.min()),
                 "constant_bits": int(((p == 0) | (p == 1)).sum())}
        law = laws[name]
        values = np.concatenate([s["ref"] for s in states])
        if name == "GRAY4":
            mids = np.array(L.gray4_midpoints(law))
            levels = np.array([L.gray4_level(law, float(v)) for v in values])
            t = np.array(law.thresholds)
            entry.update({"midpoint_rmse_over_std": float(np.sqrt(((mids[levels] - values) ** 2).mean()) / values.std()),
                          "outer_level_share": float(((levels == 0) | (levels == 15)).mean()),
                          "boundary_proximal_values": int((np.abs(values[:, None] - t[None, :]) <= 1e-9 * np.abs(t)
                                                           [None, :]).any(axis=1).sum())})
        else:
            sums = np.array([L.sproj_sums(law, list(s["ref"]), S["ref_rows"]) for s in states])
            entry["boundary_proximal_projections"] = int((np.abs(sums - np.array(law.tau)[None, :]) <= 1e-9).sum())
        info[name] = entry
    wl = [s["wl_rows"] for s in states]
    info["window_truncation_share"] = sum(r > L.WINDOW_ROWS for r in wl) / len(wl)
    info["observed_values_per_state"] = {"ref": S["ref_rows"], "window": L.WINDOW_ROWS}
    return info


def gates(probe: dict) -> dict:
    g = {}
    for name in P.ENCODERS:
        s1, s2a, s2b, s3, s4 = (probe[k][name] for k in ("S1", "S2a", "S2b", "S3", "S4"))
        digest_rho = probe["S2b"]["DIGEST"]["rho"]
        g[name] = {
            "S1": s1["digest_prefix_differs"] == 1.0 and s1["payload_identical"] >= 0.99,
            "S2a": s2a["payload_differs"] >= 0.95,
            "S2b": s2b["rho"] >= 0.6 and -0.2 <= digest_rho <= 0.2,
            "S3": s3["r2"] >= 0.5 and s3["margin"] >= 0.25,
            "S4": s4["r2"] >= 0.3 and s4["margin"] >= 0.2,
        }
        g[name]["USEFUL"] = all(g[name][k] for k in ("S1", "S2a", "S2b", "S3"))
        g[name]["matched_budget"] = "BINARY_AT_OR_ABOVE_RAW" if s3["vs_RAW16"] >= -0.05 else "RAW_SUPERIOR"
    return g
