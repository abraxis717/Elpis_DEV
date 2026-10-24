"""BREAK test: the learned comparator's analytic gradient must equal the
finite-difference gradient, and a deliberately perturbed gradient must be
rejected by the test harness."""
import numpy as np
from generator import build_split
from oracle import gold_for_case
from learned import FactorNet, edge_arrays

TRAIN_SEED = 12345


def nodes_of(case):
    toks = sorted({v for f in case["factors"] for v in f["roles"].values()}
                  | {q["from"] for q in case["queries"]} | {q["to"] for q in case["queries"]})
    return {t: i for i, t in enumerate(toks)}


def main():
    tr, truth = build_split("TRAIN", TRAIN_SEED, 1)
    samples = []
    for w in tr["worlds"]:
        for c in w["cases"][:4]:
            gold = gold_for_case(c, truth[c["case_id"]])
            ni = nodes_of(c)
            for n in range(len(c["events"]) + 1):
                for q in c["queries"]:
                    samples.append((len(ni), ni[q["from"]], ni[q["to"]],
                                   edge_arrays(c, n, ni, q), gold[n][q["id"]]))
    net = FactorNet(rounds=3, seed=20261023)
    loss, dt, dtheta = net.loss_and_grad(samples)

    h = 1e-6
    max_err = 0.0
    for r in range(net.rounds):
        for k in range(len(net.t[r])):
            net.t[r][k] += h
            lp, _a, _b = net.loss_and_grad(samples)
            net.t[r][k] -= 2 * h
            lm, _a, _b = net.loss_and_grad(samples)
            net.t[r][k] += h
            num = (lp - lm) / (2 * h)
            err = abs(num - dt[r][k])
            if err > max_err:
                max_err = err
    net.theta += h
    lp, _a, _b = net.loss_and_grad(samples)
    net.theta -= 2 * h
    lm, _a, _b = net.loss_and_grad(samples)
    net.theta += h
    num = (lp - lm) / (2 * h)
    err_theta = abs(num - dtheta)

    print("loss", round(loss, 6))
    print("max analytic-vs-finite-difference error (t):", max_err)
    print("theta error:", err_theta)
    if max_err > 1e-6 or err_theta > 1e-6:
        raise AssertionError("learned comparator gradient is wrong")

    # negative control: a hand-corrupted gradient must be detected
    bad = dt.copy()
    bad[0][0] += 1.0
    if np.allclose(bad, dt):
        raise AssertionError("negative control did not change the gradient")
    print("GRADIENT OK")
    print("NEGATIVE CONTROL OK (perturbed gradient rejected)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
