"""Native parity on qualification samples (PREREGISTRATION_V2.md, "Native component").

For router seeds 0-7 of a partition and 256 episodes per regime, the native C11 learner runs every episode
on-policy, and its proposals, state and terminal decisions are compared with the vectorised frozen learner on
the same episodes. Also, on the first 32 episodes per seed, it is compared bitwise with the explicit-loop
scalar reference. Tolerances are fixed in the preregistration.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from cgc_lao_lab.env import logits

from . import V2_ROOT, agents2, candidates2, env2, evaluate2, harness2, native
from .scalar2 import ScalarRMF2

SEEDS, EPISODES, BITWISE_EPISODES = range(0, 8), 256, 32
STATE_TOL, TIE_TOL = 1e-9, 1e-9


def _feed(learner, mode, a, z, y, reads_t):
    y = list(y[:reads_t])
    if mode == "FIRST_ONLY" or len(y) == 1:
        learner.update(a, z, y[:1])
    elif mode == "DUPLICATE":
        learner.update(a, z, y)
    else:
        learner.update(a, z, y[:1])
        learner.update(a, z, y[1:], advance=False)


def check_arm(partition: str, regimes, cfg: dict, mode: str, seeds=SEEDS, episodes=EPISODES) -> dict:
    contract = env2.CONTRACTS["DUP2_4"]
    out = {"episodes": 0, "trajectory_divergences": 0, "decision_mismatches": 0, "near_tie_mismatches": 0,
           "max_state_abs_diff": 0.0, "bitwise_episodes": 0, "bitwise_failures": 0}
    for seed in seeds:
        for regime in regimes:
            batch = env2.episodes(partition, seed, regime, 0, episodes)
            agent = agents2.BayesAgent2(cfg["hypotheses"], cfg["prior"])
            ref = harness2.rollout2(agent, batch, contract, mode)
            for e in range(batch.count):
                n = native.NativeRMF(cfg["hypotheses"], cfg["prior"])
                s = ScalarRMF2(cfg["hypotheses"], cfg["prior"]) if e < BITWISE_EPISODES else None
                z = logits(batch.router_w, batch.x[e])
                diverged, bitwise_ok = False, True
                for t in range(contract.horizon):
                    a = n.select(list(z[t]))
                    if s is not None:
                        bitwise_ok &= s.select(list(z[t])) == a
                    if a != ref.actions[e, t]:
                        diverged = True
                    y = [int(v) for v in batch.y[e, t, a, :]]
                    _feed(n, mode, a, float(z[t, a]), y, contract.reads[t])
                    if s is not None:
                        _feed(s, mode, a, float(z[t, a]), y, contract.reads[t])
                        bitwise_ok &= (s.mu == n.mu and s.v == n.v and s.lw == n.lw)
                out["episodes"] += 1
                zq = logits(batch.router_w, batch.xq[e])
                if s is not None:
                    bitwise_ok &= all(s.decide(list(zq[j])) == n.decide(list(zq[j])) for j in range(16))
                    out["bitwise_episodes"] += 1
                    out["bitwise_failures"] += int(not bitwise_ok)
                if diverged:
                    out["trajectory_divergences"] += 1
                    continue
                diff = max(float(np.max(np.abs(np.array(n.mu) - agent.filter.mu[e]))),
                           float(np.max(np.abs(np.array(n.v) - agent.filter.v[e]))),
                           float(np.max(np.abs(np.array(n.lw) - agent.filter.lw[e]))))
                out["max_state_abs_diff"] = max(out["max_state_abs_diff"], diff)
                for j in range(16):
                    d = n.decide(list(zq[j]))
                    if d != ref.decisions[e, j]:
                        p = n.predictive(list(zq[j]))
                        gaps = [abs(p[i] - p[k]) for i in range(3) for k in range(i + 1, 3)] + [abs(max(p) - 0.6)]
                        if min(gaps) < TIE_TOL:
                            out["near_tie_mismatches"] += 1
                        else:
                            out["decision_mismatches"] += 1
    out["pass"] = (out["bitwise_failures"] == 0 and out["decision_mismatches"] == 0
                   and out["max_state_abs_diff"] <= STATE_TOL)
    return out


def main(argv) -> None:
    partition, path = argv[1], argv[2]
    frozen = json.loads((V2_ROOT / "FROZEN_V2.json").read_text(encoding="utf-8"))
    regimes = env2.DESIGN + env2.ROBUSTNESS
    t0 = time.time()
    rec = {"partition": partition, "seeds": list(SEEDS), "episodes_per_seed_and_regime": EPISODES,
           "bitwise_episodes_per_seed_and_regime": BITWISE_EPISODES, "regimes": list(regimes),
           "tolerances": {"state_abs": STATE_TOL, "near_tie": TIE_TOL, "scalar": "bitwise"},
           "arms": {}}
    for role in ("PRIMARY", "FALLBACK"):
        arm = frozen["candidates"][role]
        rec["arms"][role] = {"name": arm["name"], "mode": arm["mode"],
                             **check_arm(partition, regimes, candidates2.BAYES_CONFIGS[arm["name"]], arm["mode"])}
    rec["seconds"] = round(time.time() - t0, 1)
    print(rec["arms"])
    print(path, evaluate2.write(evaluate2.seal(rec, f"V2-{partition}-native-parity"), path))


if __name__ == "__main__":
    main(sys.argv)
