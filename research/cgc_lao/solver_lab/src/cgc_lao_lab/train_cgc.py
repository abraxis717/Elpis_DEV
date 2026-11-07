"""Offline training of the CGC_REFERENCE coefficients (TRAIN partition only; never at inference).

The checked-in reference ships no trained checkpoint, so the lab trains its own 120 float32 weights,
reproducibly: antithetic evolution strategies (centred-rank fitness shaping, Adam) minimising the mean
terminal binary cross-entropy between the episode's terminal predictions sigmoid(Wx' + s_T) and the
clean success probabilities, on on-policy PROBE_EARLY episodes of the ALL mixture (CLEAN, FLIP20,
FLIP40). The historical ledger records a BCE-trained pinned CGC; this is a local analogue, not a
reconstruction. All noise comes from the lab's counter streams, so the run is bit-reproducible on a
given platform.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import env, rng
from .agents import CGCAgent
from .cgc import frozen_weights, weights_digest
from .env import CONTRACTS, logits, sigmoid
from .harness import rollout

CONFIG = {
    "partition": "TRAIN",
    "train_seeds": list(range(0, 48)),
    "check_seeds": list(range(48, 64)),
    "regimes": ["CLEAN", "FLIP20", "FLIP40"],
    "episodes_per_batch": 512,
    "generations": 400,
    "pairs": 16,
    "sigma": 0.02,
    "lr": 0.004,
    "init_scale": 0.05,
    "adam": [0.9, 0.999, 1e-8],
}


def terminal_bce(agent: CGCAgent, batch) -> float:
    prob = sigmoid(logits(agent.w32, batch.xq.astype(np.float32)) + agent.state[:, None, :]).astype(np.float64)
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    pq = batch.pq
    return float(-(pq * np.log(prob) + (1 - pq) * np.log(1 - prob)).mean())


def objective(theta: np.ndarray, batches) -> float:
    w = frozen_weights(theta.astype(np.float32))
    total = 0.0
    for batch in batches:
        agent = CGCAgent(w)
        rollout(agent, batch, CONTRACTS["ORIGINAL"])
        total += terminal_bce(agent, batch)
    return total / len(batches)


def generation_batches(g: int, cfg=CONFIG):
    seeds = cfg["train_seeds"]
    n = cfg["episodes_per_batch"]
    out = []
    for j, regime in enumerate(cfg["regimes"]):
        seed = seeds[(3 * g + j) % len(seeds)]
        first = n * (g // len(seeds))
        out.append(env.episodes(cfg["partition"], seed, regime, first, n))
    return out


def centred_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x))
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def train(cfg=CONFIG, log=print) -> dict:
    theta = cfg["init_scale"] * rng.normal(rng.derive_key("TRAIN", "cgc-es", "init"), 120)
    m = np.zeros(120)
    v = np.zeros(120)
    b1, b2, eps = cfg["adam"]
    history = []
    t0 = time.time()
    for g in range(cfg["generations"]):
        batches = generation_batches(g, cfg)
        noise = rng.normal(rng.derive_key("TRAIN", "cgc-es", "noise", g), cfg["pairs"] * 120).reshape(cfg["pairs"], 120)
        fit = []
        for e in noise:
            fit.append(objective(theta + cfg["sigma"] * e, batches))
            fit.append(objective(theta - cfg["sigma"] * e, batches))
        fit = np.array(fit)
        shaped = centred_ranks(fit)
        grad = ((shaped[0::2] - shaped[1::2])[:, None] * noise).sum(0) / (cfg["pairs"] * cfg["sigma"])
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** (g + 1))
        vhat = v / (1 - b2 ** (g + 1))
        theta = theta - cfg["lr"] * mhat / (np.sqrt(vhat) + eps)
        if g % 25 == 0 or g == cfg["generations"] - 1:
            current = objective(theta, batches)
            history.append([g, round(float(fit.mean()), 7), round(current, 7)])
            log(f"gen {g:4d} mean-pop-bce {fit.mean():.6f} centre-bce {current:.6f} ({time.time() - t0:.0f}s)")
    w = frozen_weights(theta.astype(np.float32))
    return {"weights": w, "history": history, "seconds": round(time.time() - t0, 1)}


def check(w: np.ndarray, cfg=CONFIG, episodes: int = 1024) -> dict:
    """BCE of the trained and zero (NOOP-equivalent) controllers on the TRAIN check seeds."""
    out = {}
    for regime in cfg["regimes"]:
        vals_w, vals_0 = [], []
        for seed in cfg["check_seeds"]:
            batch = env.episodes(cfg["partition"], seed, regime, 0, episodes)
            vals_w.append(objective(w.astype(np.float64).ravel(), [batch]))
            vals_0.append(objective(np.zeros(120), [batch]))
        out[regime] = {"trained_bce": round(float(np.mean(vals_w)), 7), "zero_bce": round(float(np.mean(vals_0)), 7)}
    return out


def record(result: dict, checks: dict) -> dict:
    w = result["weights"]
    return {
        "schema": "cgc-lao-solver-lab.cgc-coefficients.v1",
        "arm": "CGC_REFERENCE",
        "shape": [3, 5, 8],
        "dtype": "float32",
        "weights_float32_hex": [float(x).hex() for x in w.ravel()],
        "weights_sha256": weights_digest(w),
        "training": {**CONFIG, "objective": "mean terminal BCE vs clean probabilities, on-policy PROBE_EARLY, ALL mixture",
                     "history": result["history"], "seconds": result["seconds"]},
        "check": checks,
        "provenance": "Locally reproducible experimental coefficients for the checked-in Q4E5 interface; "
                      "not the historical Q4E5 checkpoint (which is not published).",
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1])
    res = train()
    chk = check(res["weights"])
    print(json.dumps(chk, indent=1))
    out.write_text(json.dumps(record(res, chk), indent=1) + "\n", encoding="utf-8")
    print("wrote", out, weights_digest(res["weights"]))
