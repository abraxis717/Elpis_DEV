"""Resource measurements of deployed learners: volatile state, peak temporary allocation, latency.

State is counted in scalars and bytes per episode. Peak temporary allocation is the tracemalloc peak
(numpy registers its buffers with tracemalloc) for one episode's full rollout at batch size 1, which
bounds the per-update and per-decision working set. Latency is wall clock per update and per terminal
decision (16 queries), at batch size 1 (numpy and the scalar reference) and amortised at batch 4096.
Wall-clock numbers are printed measurements of this machine, never asserted.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import arms, env, evaluate, harness
from .env import CONTRACTS, logits
from .scalar_rmf import ScalarRMF

LAB = Path(__file__).resolve().parents[2]


def measure(arm: str, contract: str, repeats: int = 200) -> dict:
    c = CONTRACTS[contract]
    fac = arms.factory(arm)
    agent, _ = fac(None, c)
    one = env.episodes("DEV", 200, "FLIP40", 0, 1)
    agent.begin(one.router_w, 1, c)
    state_scalars = agent.state_scalars()
    state_bytes = int(sum(v.nbytes for v in agent.volatile_state().values()))
    tracemalloc.start()
    harness.rollout(fac(None, c)[0], one, c)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # latency at batch size 1
    upd, dec = [], []
    for r in range(repeats):
        batch = env.episodes("DEV", 200, "FLIP40", r, 1)
        ag, _ = fac(None, c)
        ag.begin(batch.router_w, 1, c)
        for t in range(c.horizon):
            a = ag.act(t, batch.x[:, t])
            reads = batch.readings(t, a, c.reads[t])
            t0 = time.perf_counter()
            ag.observe(t, batch.x[:, t], a, reads)
            upd.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ag.decide(batch.xq)
        dec.append(time.perf_counter() - t0)
    big = env.episodes("DEV", 201, "FLIP40", 0, 4096)
    t0 = time.perf_counter()
    harness.rollout(fac(None, c)[0], big, c)
    amortised = (time.perf_counter() - t0) / 4096
    return {"arm": arm, "contract": contract, "state_scalars": state_scalars, "state_bytes_float64": state_bytes,
            "state_bytes_float32_equivalent": 4 * state_scalars, "peak_temporary_bytes_episode": int(peak),
            "updates_per_episode": c.horizon, "terminal_decisions_per_episode": env.Q,
            "update_latency_us_median_b1": round(1e6 * float(np.median(upd)), 1),
            "decision_latency_us_median_b1": round(1e6 * float(np.median(dec)), 1),
            "episode_latency_us_amortised_b4096": round(1e6 * amortised, 2)}


def scalar_latency(cfg: dict, contract: str, repeats: int = 100) -> dict:
    c = CONTRACTS[contract]
    upd, dec = [], []
    for r in range(repeats):
        batch = env.episodes("DEV", 202, "FLIP40", r, 1)
        f = ScalarRMF(cfg["qs"], cfg["prior"], kinds=cfg.get("kinds"), horizon=c.horizon,
                      acquisition=cfg.get("acquisition", "probe"))
        z = logits(batch.router_w, batch.x[0])
        for t in range(c.horizon):
            a = f.select(list(z[t]))
            reads = list(batch.readings(t, np.array([a]), c.reads[t])[0])
            t0 = time.perf_counter()
            f.update(a, float(z[t, a]), reads)
            upd.append(time.perf_counter() - t0)
        zq = logits(batch.router_w, batch.xq[0])
        t0 = time.perf_counter()
        for j in range(env.Q):
            f.decide(list(zq[j]))
        dec.append(time.perf_counter() - t0)
    m = len(cfg["qs"])
    return {"scalar_update_latency_us_median": round(1e6 * float(np.median(upd)), 1),
            "scalar_decision_latency_us_median_16_queries": round(1e6 * float(np.median(dec)), 1),
            "flops_per_update_approx": m * 16 * 12, "flops_per_query_decision_approx": m * 3 * 16 * 8}


def main(argv) -> None:
    from .candidates import CANDIDATES
    pairs = [(a, "ORIGINAL") for a in ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES")]
    pairs += [(a, c) for a, c in (x.split("@") for x in argv[1:])]
    rows = []
    for arm, contract in pairs:
        row = measure(arm, contract)
        if arm in CANDIDATES and CANDIDATES[arm].get("family") != "B_GATED_CGC":
            row.update(scalar_latency(CANDIDATES[arm], contract))
        rows.append(row)
        print(json.dumps(row))
    rec = evaluate.seal({"rows": rows}, "RESOURCES", "measured on this machine; not asserted by tests")
    print("sha256", evaluate.write(rec, LAB / "evidence" / "resources.json"))


if __name__ == "__main__":
    main(sys.argv)
