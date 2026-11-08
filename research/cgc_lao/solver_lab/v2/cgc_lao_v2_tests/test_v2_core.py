"""SLB2 spec conformance, namespace and partition separation, mechanisms, generalised filter, scalar reference,
equal-information controls and deployment boundaries of every v2 arm."""
import dataclasses
import json
import math
import unittest

import numpy as np

from cgc_lao_v2_tests import V2
from cgc_lao_lab import agents as v1
from cgc_lao_lab import env as env1
from cgc_lao_lab import harness as h1
from cgc_lao_lab import rng as rng1
from cgc_lao_v2 import agents2, arms2, candidates2, env2, harness2
from cgc_lao_v2.filter2 import ReliabilityMixtureFilter2
from cgc_lao_v2.scalar2 import ScalarRMF2

SPEC = json.loads((V2 / "specs" / "slb2.spec.json").read_text(encoding="utf-8"))
DUP = env2.CONTRACTS["DUP2_4"]
ORIG = env2.CONTRACTS["ORIGINAL"]


class TestSLB2Spec(unittest.TestCase):
    def test_constants_regimes_contracts_partitions(self):
        self.assertEqual(env2.ROOT, SPEC["rng_root"])
        self.assertEqual(env2.PARTITIONS, {k: v["router_seeds"] for k, v in SPEC["partitions"].items()})
        spec_regimes = {n: (cls, f) for cls, rs in SPEC["regimes"].items() for n, f in rs.items()}
        self.assertEqual(set(spec_regimes), set(env2.REGIMES))
        for name, (cls, fields) in spec_regimes.items():
            r = env2.REGIMES[name]
            self.assertEqual((r.cls, r.mechanism), (cls, fields["mechanism"]), name)
            for key, value in fields.items():
                if key != "mechanism":
                    self.assertEqual(getattr(r, key), value, (name, key))
        for name, fields in SPEC["contracts"].items():
            self.assertEqual(list(env2.CONTRACTS[name].reads), fields["reads"])
        self.assertEqual(list(env2.ALL_MIXTURE), SPEC["all_mixture"])
        self.assertEqual(tuple(SPEC["provider_modes"]), env2.MODES)
        self.assertEqual((env1.STOP_PAYOFF, env1.Q, env1.T_MAX, env1.R_MAX), (SPEC["stop_payoff"], SPEC["terminal_queries"],
                         SPEC["horizon_max_generated"], SPEC["reads_max_generated"]))

    def test_namespace_and_partitions_are_fresh(self):
        v1_keys = {rng1.derive_key(p, "router", s) for p, n in env1.PARTITIONS.items() for s in range(n)}
        v2_keys = {env2.derive_key(p, "router", s) for p, n in env2.PARTITIONS.items() for s in range(n)}
        self.assertEqual(len(v2_keys), sum(env2.PARTITIONS.values()))
        self.assertFalse(v1_keys & v2_keys)
        routers = {p: {env2.router(p, s).tobytes() for s in range(n)} for p, n in env2.PARTITIONS.items()}
        for p in routers:
            for p2 in routers:
                if p != p2:
                    self.assertFalse(routers[p] & routers[p2], (p, p2))
        with self.assertRaises(ValueError):
            env2.episodes("HOLDOUT", 0, "CORR50", 0, 4)     # correlation regimes never on HOLDOUT
        with self.assertRaises(ValueError):
            env2.episodes("DEV", 0, "ASYM40", 0, 4)         # OOD regimes only on the OOD partition
        with self.assertRaises(ValueError):
            env2.episodes("TRAIN", 0, "FLIP45", 0, 4)
        with self.assertRaises(ValueError):
            env2.episodes("OOD", 64, "CLEAN", 0, 4)

    def test_batching_invariance(self):
        whole = env2.episodes("DEV", 9, "CORR50", 0, 40)
        parts = [env2.episodes("DEV", 9, "CORR50", 0, 15), env2.episodes("DEV", 9, "CORR50", 15, 25)]
        for field in ("x", "xq", "b", "clean", "flips", "y"):
            self.assertTrue(np.array_equal(getattr(whole, field), np.concatenate([getattr(p, field) for p in parts])))


class TestMechanisms(unittest.TestCase):
    def test_correlated_readings_law(self):
        for name, rho in (("FLIP40", 0.0), ("CORR25", 0.25), ("CORR75", 0.75), ("EVENT40", 1.0)):
            b = env2.episodes("DEV", 11, name, 0, 6000)
            f = b.flips.astype(float)
            self.assertAlmostEqual(float(f[..., 0].mean()), 0.4, delta=0.006, msg=name)
            self.assertAlmostEqual(float(f[..., 1].mean()), 0.4, delta=0.006, msg=name)
            corr = np.corrcoef(f[..., 0].ravel(), f[..., 1].ravel())[0, 1]
            self.assertAlmostEqual(float(corr), rho, delta=0.01, msg=name)
            disagree = float((b.y[..., 0] != b.y[..., 1]).mean())
            self.assertAlmostEqual(disagree, (1 - rho) * 2 * 0.4 * 0.6, delta=0.006, msg=name)

    def test_other_v2_mechanisms(self):
        s = env2.episodes("OOD", 2, "SHARED_SRC", 0, 6000)
        self.assertAlmostEqual(float(s.flips.mean()), 0.3 * 0.85 + 0.7 * 0.15, delta=0.006)
        self.assertAlmostEqual(float((s.y[..., 0] != s.y[..., 1]).mean()), 2 * 0.15 * 0.85, delta=0.006)
        e = env2.episodes("OOD", 2, "SECOND_EVENT40", 0, 6000)
        p = e.p
        # two independent events of p, each read through a q = 0.4 channel
        pr1 = 0.4 + 0.2 * p
        self.assertAlmostEqual(float((e.y[..., 0] != e.y[..., 1]).mean()), float((2 * pr1 * (1 - pr1)).mean()), delta=0.006)
        f48 = env2.episodes("DEV", 2, "FLIP48", 0, 4000)
        self.assertAlmostEqual(float(f48.flips.mean()), 0.48, delta=0.006)


class TestGeneralisedFilter(unittest.TestCase):
    def test_v1_hypotheses_reproduce_v1_bitwise(self):
        b = env2.episodes("DEV", 3, "QUNIF", 0, 300)
        for cname in ("ORIGINAL", "DUP2_4", "DUP2"):
            a1 = v1.BayesAgent([0.0, 0.2, 0.4, 0.4], [1, 1, 1, 0.25], kinds=["channel"] * 3 + ["event"])
            a2 = agents2.BayesAgent2(candidates2.H_V1, [1, 1, 1, 0.25])
            r1 = h1.rollout(a1, b, env2.CONTRACTS[cname])
            r2 = harness2.rollout2(a2, b, env2.CONTRACTS[cname], "DUPLICATE")
            self.assertTrue(np.array_equal(r1.actions, r2.actions) and np.array_equal(r1.decisions, r2.decisions))
            for k in ("mu", "v", "lw"):
                self.assertTrue(np.array_equal(getattr(a1.filter, k), getattr(a2.filter, k)), (cname, k))

    def test_correlation_parameter_interpolates_channel_and_event(self):
        reads = np.array([[1, 1], [1, 0], [0, 0]])
        z = np.array([0.3, 0.3, 0.3])
        out = {}
        for rho in (0.0, 0.5, 1.0):
            f = ReliabilityMixtureFilter2([(0.4, rho)], [1.0])
            f.reset(3)
            a1, a0 = f._evidence(reads.astype(np.uint8))
            out[rho] = (a1[:, 0, 0], a0[:, 0, 0])
        q = 0.4
        self.assertTrue(np.allclose(out[0.0][0], [(1 - q) ** 2, q * (1 - q), q ** 2]))
        self.assertTrue(np.allclose(out[1.0][0], [1 - q, 0.0, q]))
        self.assertTrue(np.allclose(out[0.5][0], 0.5 * out[0.0][0] + 0.5 * out[1.0][0]))
        f.update(np.zeros(3, dtype=np.int64), z, reads)

    def test_band_hypothesis_is_a_node_average(self):
        f = ReliabilityMixtureFilter2([([0.42, 0.46, 0.49], 0.0)], [1.0])
        f.reset(1)
        a1, _ = f._evidence(np.array([[1, 0]], dtype=np.uint8))
        self.assertAlmostEqual(float(a1[0, 0, 0]), float(np.mean([q * (1 - q) for q in (0.42, 0.46, 0.49)])), places=15)
        self.assertEqual(f.state_scalars(), 7)

    def test_stability_as_q_approaches_one_half(self):
        for q in (0.49, 0.499, 0.4999, 0.49999):
            f = ReliabilityMixtureFilter2([(0.0, 0.0), (q, 0.0), ([0.45, q], 0.0)], [1, 1, 1])
            f.reset(64)
            g = np.random.default_rng(5)
            for t in range(16):
                acts = g.integers(0, 3, 64)
                reads = g.integers(0, 2, (64, 2 if t % 2 else 1))
                f.update(acts, g.normal(0, 3, 64), reads)
            for arr in (f.mu, f.v, f.lw, f.weights()):
                self.assertTrue(np.isfinite(arr).all(), q)
            self.assertTrue((f.v >= 1e-6).all())
            # a channel at q ~ 0.5 carries almost no information: its bias belief barely moves
            self.assertLess(float(np.abs(f.mu[:, 1]).max()), 0.05 if q >= 0.499 else 0.3)
        with self.assertRaises(ValueError):
            ReliabilityMixtureFilter2([(0.5, 0.0)], [1.0])


class TestScalarReference(unittest.TestCase):
    def test_scalar_reference_matches_vectorised_learner(self):
        cfgs = [(candidates2.H5, [1, 1, 1, 1, 0.25], "probe"), (candidates2.H5_BAND, [1, 1, 1, 1, 0.25], "greedy")]
        for hyps, prior, acq in cfgs:
            for mode in ("DUPLICATE", "FIRST_ONLY", "SECOND_EVENT"):
                b = env2.episodes("DEV", 200, "FLIP40", 0, 24)
                agent = agents2.BayesAgent2(hyps, prior, acquisition=acq)
                out = harness2.rollout2(agent, b, DUP, mode)
                for e in range(b.count):
                    s = ScalarRMF2(hyps, prior, acquisition=0 if acq == "probe" else 1)
                    z = env1.logits(b.router_w, b.x[e])
                    for t in range(8):
                        a = s.select(list(z[t]))
                        self.assertEqual(a, out.actions[e, t], (mode, e, t))
                        y = [int(v) for v in b.y[e, t, a, :DUP.reads[t]]]
                        if mode == "FIRST_ONLY" or DUP.reads[t] == 1:
                            s.update(a, float(z[t, a]), y[:1])
                        elif mode == "DUPLICATE":
                            s.update(a, float(z[t, a]), y)
                        else:
                            s.update(a, float(z[t, a]), y[:1])
                            s.update(a, float(z[t, a]), y[1:], advance=False)
                    self.assertTrue(np.allclose(s.mu, agent.filter.mu[e], atol=1e-10, rtol=0))
                    self.assertTrue(np.allclose(s.v, agent.filter.v[e], atol=1e-10, rtol=0))
                    self.assertTrue(np.allclose(s.weights(), agent.filter.weights()[e], atol=1e-10, rtol=0))
                    zq = env1.logits(b.router_w, b.xq[e])
                    self.assertEqual([s.decide(list(zq[j])) for j in range(16)], list(out.decisions[e]))
                self.assertEqual(s.state_scalars(), agent.state_scalars())


def v2_arm_names():
    return (list(candidates2.PRIMARY) + list(candidates2.FALLBACK) + list(candidates2.BAYES_CONTROLS)
            + ["NOOP", "BETA_ALL", "SGD_ALL", "DISAGREE_SGD", "PLUGIN_Q_BAYES"])


class TestV2DeploymentBoundaries(unittest.TestCase):
    def test_budget_reset_replay_privilege_and_poisoning(self):
        batch = env2.episodes("DEV", 199, "FLIP40", 0, 40)
        other = env2.episodes("DEV", 198, "CLEAN", 0, 40)
        for name in v2_arm_names():
            fac = arms2.factory(name)
            agent, priv = fac(env2.REGIMES["CLEAN"], DUP)
            self.assertIsNone(priv)
            self.assertFalse(agent.privileged)
            self.assertLessEqual(agent.state_scalars(), 64, name)
            with self.assertRaises(PermissionError):
                agent.begin(batch.router_w, 4, DUP, {"q_override": np.zeros((4, 1))})
            for mode in ("DUPLICATE", "FIRST_ONLY"):
                ref = harness2.rollout2(fac(None, DUP)[0], batch, DUP, mode)
                again = harness2.rollout2(fac(env2.REGIMES["FLIP45"], DUP)[0], batch, DUP, mode)
                self.assertTrue(np.array_equal(ref.decisions, again.decisions), (name, mode))
                reused = fac(None, DUP)[0]
                harness2.rollout2(reused, other, DUP, mode)
                self.assertTrue(np.array_equal(harness2.rollout2(reused, batch, DUP, mode).decisions, ref.decisions))
                # unselected potential outcomes inverted, hidden truth destroyed: identical behaviour
                taken = np.zeros_like(batch.y, dtype=bool)
                for t in range(8):
                    taken[np.arange(batch.count), t, ref.actions[:, t], :] = True
                bad = dataclasses.replace(batch, y=np.where(taken, batch.y, 1 - batch.y).astype(np.uint8),
                                          b=np.full_like(batch.b, np.nan), access_log=[])
                out = harness2.rollout2(fac(None, DUP)[0], bad, DUP, mode)
                self.assertTrue(np.array_equal(out.actions, ref.actions) and np.array_equal(out.decisions, ref.decisions),
                                (name, mode))
            agent.begin(batch.router_w, batch.count, DUP)
            per_episode = sum(v.size for v in agent.volatile_state().values()) // batch.count
            self.assertEqual(per_episode, agent.state_scalars(), name)

    def test_first_only_mode_never_requests_a_second_reading(self):
        batch = env2.episodes("DEV", 197, "EVENT40", 0, 16)
        harness2.rollout2(arms2.factory("SAFE4_P4")(None, DUP)[0], batch, DUP, "FIRST_ONLY")
        self.assertTrue(all(reads == 1 for _, _, reads in batch.access_log))
        self.assertEqual(len(batch.access_log), 8)

    def test_disagreement_posterior(self):
        w = agents2.disagreement_posterior(np.array([0.0, 4.0, 4.0]), np.array([0.0, 0.0, 2.0]))
        self.assertTrue(np.allclose(w[0], 1 / 6))
        self.assertGreater(w[1, 0], w[1, -1])
        self.assertEqual(w[2, 0], 0.0)
        self.assertTrue(np.allclose(w.sum(1), 1.0))
        self.assertTrue(math.isclose(float((w[1] * agents2.DISAGREE_GRID).sum()) < 0.2, True))


if __name__ == "__main__":
    unittest.main()
