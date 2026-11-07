"""The dependency-free scalar filter reproduces the vectorised learner episode by episode."""
import unittest

import numpy as np

from cgc_lao_lab_tests import LAB  # noqa: F401
from cgc_lao_lab import agents as ag
from cgc_lao_lab import env, harness
from cgc_lao_lab.env import logits
from cgc_lao_lab.scalar_rmf import ScalarRMF

CONFIGS = [
    ({"qs": [0.0, 0.2, 0.4], "prior": [1, 1, 1]}, "ORIGINAL", "probe"),
    ({"qs": [0.0, 0.2, 0.4, 0.4], "prior": [1, 1, 1, 0.5], "kinds": ["channel"] * 3 + ["event"]}, "DUP2_4", "greedy"),
    ({"qs": [0.0, 0.4], "prior": [1, 2]}, "DUP2", "probe"),
]


class TestScalarReference(unittest.TestCase):
    def test_scalar_filter_matches_vectorised_agent(self):
        for cfg, cname, acq in CONFIGS:
            contract = env.CONTRACTS[cname]
            batch = env.episodes("DEV", 199, "FLIP20", 0, 24)
            agent = ag.BayesAgent(cfg["qs"], cfg["prior"], kinds=cfg.get("kinds"), acquisition=acq)
            out = harness.rollout(agent, batch, contract)
            for e in range(batch.count):
                f = ScalarRMF(cfg["qs"], cfg["prior"], kinds=cfg.get("kinds"), horizon=contract.horizon,
                              acquisition=acq)
                z = logits(batch.router_w, batch.x[e])
                for t in range(contract.horizon):
                    a = f.select(list(z[t]))
                    self.assertEqual(a, out.actions[e, t], (cname, e, t))
                    f.update(a, float(z[t, a]), [int(r) for r in batch.y[e, t, a, :contract.reads[t]]])
                self.assertTrue(np.allclose(f.mu, agent.filter.mu[e], atol=1e-10), (cname, e))
                self.assertTrue(np.allclose(f.v, agent.filter.v[e], atol=1e-10), (cname, e))
                self.assertTrue(np.allclose(f.weights(), agent.filter.weights()[e], atol=1e-10), (cname, e))
                zq = logits(batch.router_w, batch.xq[e])
                self.assertEqual([f.decide(list(zq[j])) for j in range(env.Q)], list(out.decisions[e]), (cname, e))
            self.assertEqual(f.state_scalars(), agent.state_scalars())


if __name__ == "__main__":
    unittest.main()
