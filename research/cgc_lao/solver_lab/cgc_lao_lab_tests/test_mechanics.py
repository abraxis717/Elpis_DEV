"""Selector parity with the checked-in reference, CGC equations, reliability-mixture filter correctness."""
import importlib.util
import unittest

import numpy as np

from cgc_lao_lab_tests import LAB
from cgc_lao_lab import agents as ag
from cgc_lao_lab import cgc, env, harness, rng, selector
from cgc_lao_lab.filters import NODES, WEIGHTS, ReliabilityMixtureFilter

REFERENCE_SRC = LAB.parent / "reference" / "src"


def reference_choose_action(t, state, count, base):
    """The reference test's verbatim PROBE_EARLY semantics, transcribed to numpy."""
    pred = 1.0 / (1.0 + np.exp(-(base + state)))
    best = pred.argmax(-1)
    unseen = count == 0
    need = unseen.sum(-1)
    unseen_best = np.where(unseen, pred, -1e9).argmax(-1)
    deadline_force = need >= (8 - t)
    probe = (t in (1, 2)) & (need > 0)
    return np.where(deadline_force | probe, unseen_best, best)


def small_weights(seed=88711, scale=0.05):
    return cgc.frozen_weights(scale * rng.normal(rng.derive_key("unit", "cgcw", seed), 120))


class TestSelectorParity(unittest.TestCase):
    def test_probe_early_matches_reference_semantics_at_every_step(self):
        for t in range(8):
            for size in (1, 3, 32):
                key = rng.derive_key("unit", "sel", t, size)
                base = rng.normal(key, size * 3).reshape(size, 3)
                state = np.clip(rng.normal(key + 1, size * 3).reshape(size, 3), -1.5, 1.5)
                obs = (rng.uniform(key + 2, size * t) * 3).astype(int).reshape(size, t)
                count = np.zeros((size, 3))
                for i in range(t):
                    count[np.arange(size), obs[:, i]] += 1
                got = selector.choose_probe_early(t, base, state, count)
                self.assertTrue(np.array_equal(got, reference_choose_action(t, state, count, base)), (t, size))

    def test_tie_and_saturated_logit_parity(self):
        for t in range(8):
            base = np.array([[90., 91., 92.], [0., 0., 0.]], dtype=np.float32)
            state = np.zeros_like(base)
            count = np.zeros_like(base)
            if t > 0:
                count[:, 0] = float(t)
            self.assertTrue(np.array_equal(selector.choose_probe_early(t, base, state, count),
                                           reference_choose_action(t, state, count, base)))

    def test_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "COUNTS_DO_NOT_MATCH_STEP"):
            selector.choose_probe_early(3, np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "STATE_CAP"):
            selector.choose_probe_early(0, np.zeros((2, 3)), np.full((2, 3), 2.0), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "STEP_OUT_OF_RANGE"):
            selector.choose_probe_early(8, np.zeros((2, 3)), np.zeros((2, 3)), np.full((2, 3), 8 / 3))

    def test_coverage_by_deadline(self):
        for horizon in (8, 12, 16):
            batch = env.episodes("DEV", 1, "FLIP40", 0, 256)
            out = harness.rollout(ag.NoopAgent(), batch, env.Contract("X", (1,) * horizon))
            for e in range(256):
                self.assertEqual(set(out.actions[e]), {0, 1, 2})
            agent = ag.BayesAgent([0.0], [1.0], acquisition="greedy")
            out = harness.rollout(agent, batch, env.Contract("X", (1,) * horizon))
            self.assertTrue(all(set(row) == {0, 1, 2} for row in out.actions))


class TestCGCPort(unittest.TestCase):
    def test_weight_contract(self):
        w = small_weights()
        self.assertEqual((w.shape, w.dtype, w.flags.writeable), ((3, 5, 8), np.float32, False))
        with self.assertRaises(ValueError):
            cgc.frozen_weights(np.full(120, np.nan))
        with self.assertRaises(ValueError):
            w[0, 0, 0] = 1.0

    def test_episode_update_matches_reference_test_equations(self):
        """Transcription of test_full_episode_reference_parity_and_coverage's per-step equations."""
        w = small_weights()
        batch = env.episodes("TRAIN", 9, "FLIP20", 0, 27)
        agent = ag.CGCAgent(w)
        agent.begin(batch.router_w, 27, env.CONTRACTS["ORIGINAL"])
        w32 = batch.router_w.astype(np.float32)
        state = np.zeros((27, 3), np.float32)
        count = np.zeros((27, 3), np.float32)
        for t in range(8):
            x = batch.x[:, t].astype(np.float32)
            base = env.logits(w32, x)
            act_ref = reference_choose_action(t, state, count, base)
            act = agent.act(t, batch.x[:, t])
            self.assertTrue(np.array_equal(act, act_ref))
            selected = batch.y[np.arange(27), t, act_ref, 0].astype(np.float32)
            pred = env.sigmoid(base + state)
            onehot = np.eye(3, dtype=np.float32)[act_ref]
            residual = pred[np.arange(27), act_ref] - selected
            feat = np.concatenate((np.tanh(3 * onehot * residual[:, None]), np.tanh(residual[:, None] ** 2),
                                   np.full((27, 1), (t + 1) / 8, np.float32)), -1).astype(np.float32)
            z = np.einsum("bd,odn->bon", feat, w)
            delta = np.float32(0.25) * np.tanh(np.float32(0.5) * (z + z * z + z * z * z).sum(-1))
            state = np.clip(state - np.clip(delta, -0.25, 0.25), -1.5, 1.5).astype(np.float32)
            count = count + onehot
            agent.observe(t, batch.x[:, t], act, batch.readings(t, act, 1))
            self.assertTrue(np.allclose(agent.state, state, atol=1e-7), t)
            self.assertTrue(np.array_equal(agent.counts, count))
        self.assertTrue((count > 0).all())

    def test_static_attenuation_scales_the_applied_proposal(self):
        w = small_weights(scale=0.2)
        base = np.zeros((4, 3), np.float32)
        state = np.zeros((4, 3), np.float32)
        acts = np.array([0, 1, 2, 0])
        full, delta = cgc.cgc_step(0, base, state, acts, np.array([1, 0, 1, 0], np.float32), w, 1.0)
        half, _ = cgc.cgc_step(0, base, state, acts, np.array([1, 0, 1, 0], np.float32), w, 0.5)
        self.assertTrue(np.allclose(half, 0.5 * full, atol=1e-7))
        self.assertTrue(np.allclose(full, -np.clip(delta, -0.25, 0.25)))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional: torch not installed")
    def test_parity_with_the_checked_in_torch_reference(self):
        import sys
        import torch
        sys.path.insert(0, str(REFERENCE_SRC))
        try:
            from cgc.lao_q4e5 import FrozenCGC3, ProbeEarlyEpisode
        finally:
            sys.path.remove(str(REFERENCE_SRC))
        w = small_weights(scale=0.2)
        for regime in ("CLEAN", "FLIP40"):
            batch = env.episodes("TRAIN", 11, regime, 0, 64)
            ref = ProbeEarlyEpisode(torch.tensor(batch.router_w, dtype=torch.float32),
                                    FrozenCGC3(torch.tensor(w)).eval(), 64)
            mine = ag.CGCAgent(w)
            mine.begin(batch.router_w, 64, env.CONTRACTS["ORIGINAL"])
            for t in range(8):
                a_ref = ref.select(torch.tensor(batch.x[:, t], dtype=torch.float32)).numpy()
                a = mine.act(t, batch.x[:, t])
                self.assertTrue(np.array_equal(a, a_ref), (regime, t))
                r = batch.readings(t, a, 1)
                ref.observe(torch.tensor(r[:, 0], dtype=torch.float32))
                mine.observe(t, batch.x[:, t], a, r)
                self.assertTrue(np.allclose(ref.state.numpy(), mine.state, atol=2e-6), (regime, t))
            q = torch.tensor(np.transpose(batch.xq, (1, 0, 2)), dtype=torch.float32)
            p_ref = ref.terminal_expert_probabilities(q).numpy().transpose(1, 0, 2)
            p_mine = env.sigmoid(env.logits(mine.w32, batch.xq.astype(np.float32)) + mine.state[:, None, :])
            self.assertTrue(np.allclose(p_ref, p_mine, atol=2e-6))


def grid_posterior(observations, q, prior_var=1.0):
    """Brute-force posterior of one expert's bias on a fine grid; observations: [(z, reads tuple)]."""
    b = np.linspace(-12, 12, 48001)
    logp = -0.5 * b * b / prior_var
    for z, reads in observations:
        p = 1 / (1 + np.exp(-(z + b)))
        n1 = sum(reads)
        n0 = len(reads) - n1
        logp = logp + np.log(p * (1 - q) ** n1 * q ** n0 + (1 - p) * q ** n1 * (1 - q) ** n0)
    w = np.exp(logp - logp.max())
    norm = w.sum()
    mean = (w * b).sum() / norm
    var = (w * b * b).sum() / norm - mean * mean
    log_evidence = np.log(norm * (b[1] - b[0])) + logp.max() - 0.5 * np.log(2 * np.pi * prior_var)
    return mean, var, log_evidence


class TestReliabilityMixtureFilter(unittest.TestCase):
    def test_quadrature_rule(self):
        self.assertAlmostEqual(float(WEIGHTS.sum()), 1.0, places=14)
        self.assertAlmostEqual(float((WEIGHTS * NODES ** 2).sum()), 1.0, places=12)
        self.assertAlmostEqual(float((WEIGHTS * NODES ** 4).sum()), 3.0, places=11)
        self.assertAlmostEqual(float((WEIGHTS * NODES ** 30).sum()), float(np.prod(np.arange(29, 0, -2))),
                               delta=1e-6 * float(np.prod(np.arange(29, 0, -2))))
        x, wq = np.polynomial.hermite_e.hermegauss(16)
        self.assertTrue(np.allclose(NODES, x, atol=1e-12) and np.allclose(WEIGHTS, wq / wq.sum(), atol=1e-13))

    def test_first_update_is_exact_bayes(self):
        for q in (0.0, 0.2, 0.4):
            for reads in ((1,), (0,), (1, 1), (1, 0)):
                if q == 0.0 and reads == (1, 0):
                    continue
                f = ReliabilityMixtureFilter([q], [1.0])
                f.reset(1)
                f.update(np.array([1]), np.array([0.7]), np.array([reads]))
                mean, var, _ = grid_posterior([(0.7, reads)], q)
                self.assertAlmostEqual(float(f.mu[0, 0, 1]), mean, delta=1e-7, msg=(q, reads))
                self.assertAlmostEqual(float(f.v[0, 0, 1]), var, delta=1e-7, msg=(q, reads))
                self.assertEqual(float(f.mu[0, 0, 0]), 0.0)

    def test_mixture_weights_follow_bayes_rule(self):
        obs = [(0, 1.3, (1,)), (2, -0.4, (0,)), (1, 2.1, (0,))]   # distinct experts: ADF moments are exact
        qs, prior = [0.0, 0.2, 0.4], [0.5, 0.3, 0.2]
        f = ReliabilityMixtureFilter(qs, prior)
        f.reset(1)
        for a, z, reads in obs:
            f.update(np.array([a]), np.array([z]), np.array([reads]))
        logpost = np.log(prior) + np.array([sum(grid_posterior([(z, r)], q)[2] for _, z, r in obs) for q in qs])
        post = np.exp(logpost - logpost.max())
        self.assertTrue(np.allclose(f.weights()[0], post / post.sum(), atol=1e-6))

    def test_adf_tracks_exact_posterior_over_a_full_expert_history(self):
        hist = [(0.3, (1,)), (-1.1, (1,)), (0.8, (0,)), (1.9, (1,)), (-0.2, (0,))]
        for q in (0.0, 0.4):
            f = ReliabilityMixtureFilter([q], [1.0])
            f.reset(1)
            for z, reads in hist:
                f.update(np.array([2]), np.array([z]), np.array([reads]))
            mean, var, _ = grid_posterior(hist, q)
            self.assertAlmostEqual(float(f.mu[0, 0, 2]), mean, delta=0.01)
            self.assertAlmostEqual(float(f.v[0, 0, 2]), var, delta=0.01)

    def test_disagreeing_readings_eliminate_clean_and_event_hypotheses(self):
        f = ReliabilityMixtureFilter([0.0, 0.4, 0.4], [1, 1, 1], kinds=["channel", "channel", "event"])
        f.reset(2)
        f.update(np.array([0, 0]), np.array([0.5, 0.5]), np.array([[1, 0], [1, 1]]))
        w = f.weights()
        self.assertEqual(float(w[0, 0]), 0.0)
        self.assertEqual(float(w[0, 2]), 0.0)
        self.assertAlmostEqual(float(w[0, 1]), 1.0)
        self.assertTrue((w[1] > 0).all())

    def test_state_size_and_fail_closed(self):
        f = ReliabilityMixtureFilter([0.0, 0.2, 0.4], [1, 1, 1])
        self.assertEqual(f.state_scalars(), 21)
        with self.assertRaises(ValueError):
            ReliabilityMixtureFilter([0.5], [1.0])
        with self.assertRaises(ValueError):
            ReliabilityMixtureFilter([0.1], [0.0])
        f.reset(1)
        with self.assertRaises(ValueError):
            f.update(np.array([0]), np.array([0.0]), np.array([[2]]))

    def test_predictive_is_a_posterior_mean(self):
        f = ReliabilityMixtureFilter([0.0], [1.0])
        f.reset(1)
        z = np.array([[[0.4, -1.0, 2.0]]])
        b = np.linspace(-12, 12, 48001)
        dens = np.exp(-0.5 * b * b)
        exact = [(dens / (1 + np.exp(-(zz + b)))).sum() / dens.sum() for zz in z[0, 0]]
        self.assertTrue(np.allclose(f.predictive(z)[0, 0], exact, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
