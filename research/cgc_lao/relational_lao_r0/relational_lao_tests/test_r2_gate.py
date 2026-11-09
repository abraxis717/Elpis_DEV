"""R2, the independent-edge mechanics gate: the ported learning rule on the fixed-capacity relational representation
(slots = the three SLB2 experts, no composition, no relational readout) reproduces the frozen, qualified RMF3
(cgc_lao_v2.agents2.BayesAgent2 with FROZEN_V4's PRIMARY configuration) action for action and decision for decision.
The only arithmetic difference is the reliability bookkeeping (per-slot evidence terms instead of RMF3's normalised
running log-weights), which changes weights by rounding only."""
import json
import unittest

import numpy as np

from relational_lao_tests import R0
from cgc_lao_lab.env import logits
from cgc_lao_lab.selector import probe_early_from_pred
from cgc_lao_v2 import agents2, env2, harness2
from relational_lao.learners import RelationalFilter

FROZEN_V4 = json.loads((R0.parent / "solver_lab" / "v4" / "FROZEN_V4.json").read_text(encoding="utf-8"))
RMF3 = FROZEN_V4["candidates"]["PRIMARY"]["config"]


def relational_rollout(batch, contract):
    E = batch.count
    idx = np.arange(E)
    f = RelationalFilter(tuple(map(tuple, RMF3["hypotheses"])), RMF3["prior"])
    f.reset(E, 3)
    counts = np.zeros((E, 3), dtype=np.int64)
    taken = np.zeros((E, 8), dtype=np.int64)
    for t in range(8):
        z = logits(batch.router_w, batch.x[:, t])
        pred = (f.weights()[:, :, None, None] * f.means(z[:, None, :])).sum(1)[:, 0, :]
        a = probe_early_from_pred(t, pred, counts, 8)
        f.update(a, z[idx, a], batch.readings(t, a, contract.reads[t]), np.ones(E, dtype=bool))
        counts[idx, a] += 1
        taken[:, t] = a
    zq = logits(batch.router_w, batch.xq)
    pred = (f.weights()[:, :, None, None] * f.means(zq)).sum(1)
    decisions = np.where(pred.max(-1) < 0.6, 3, pred.argmax(-1))
    return taken, decisions, f


class TestR2IndependentEdgeGate(unittest.TestCase):
    def test_reproduces_the_frozen_rmf3(self):
        contract = env2.CONTRACTS["DUP2_4"]
        total = mismatched = 0
        for regime in ("CLEAN", "FLIP20", "FLIP40", "FLIP45", "EVENT40", "QUNIF"):
            for seed in (240, 241):
                batch = env2.episodes("DEV", seed, regime, 0, 512)
                agent = agents2.BayesAgent2(RMF3["hypotheses"], RMF3["prior"])
                ref = harness2.rollout2(agent, batch, contract, "DUPLICATE")
                taken, decisions, f = relational_rollout(batch, contract)
                self.assertTrue(np.array_equal(taken, ref.actions), (regime, seed))
                mismatched += int((decisions != ref.decisions).sum())
                total += decisions.size
                np.testing.assert_allclose(f.mu, agent.filter.mu, rtol=0, atol=1e-12)
                np.testing.assert_allclose(f.v, agent.filter.v, rtol=0, atol=1e-12)
                np.testing.assert_allclose(f.weights(), agent.filter.weights(), rtol=0, atol=1e-12)
        self.assertEqual(mismatched, 0, f"{mismatched} of {total} decisions")


if __name__ == "__main__":
    unittest.main()
