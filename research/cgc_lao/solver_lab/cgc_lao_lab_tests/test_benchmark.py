"""Spec conformance, counter RNG, simulator, corruption mechanisms, potential-outcome indexing, partitions."""
import hashlib
import json
import unittest

import numpy as np

from cgc_lao_lab_tests import LAB
from cgc_lao_lab import env, rng

SPEC = json.loads((LAB / "specs" / "slb1.spec.json").read_text(encoding="utf-8"))


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


class TestBenchmarkSpecConformance(unittest.TestCase):
    def test_constants_equal_the_frozen_spec(self):
        self.assertEqual((env.K, env.D, env.T_PRIMARY, env.T_MAX, env.R_MAX, env.Q),
                         (SPEC["experts"], SPEC["context_dim"], SPEC["horizon_primary"],
                          SPEC["horizon_max_generated"], SPEC["reads_max_generated"], SPEC["terminal_queries"]))
        self.assertEqual(env.STOP_PAYOFF, SPEC["stop_payoff"])
        self.assertEqual(env.ROUTER_SCALE, SPEC["router"]["scale"])
        self.assertEqual(env.BIAS_SCALE, SPEC["bias"]["scale"])
        self.assertEqual({k: v["router_seeds"] for k, v in SPEC["partitions"].items()}, env.PARTITIONS)
        self.assertEqual(list(env.ALL_MIXTURE), SPEC["all_mixture"])
        self.assertEqual(env.PRIMARY_CORRUPTED, SPEC["primary_corrupted_regime"])
        self.assertEqual(list(env.PRIMARY_SHIFTS), SPEC["primary_shift_regimes"])

    def test_regimes_equal_the_frozen_spec(self):
        spec = {**SPEC["design_regimes"], **SPEC["shift_regimes"]}
        self.assertEqual(set(spec), set(env.REGIMES))
        self.assertEqual(set(SPEC["design_regimes"]), set(env.DESIGN_REGIMES))
        for name, fields in spec.items():
            regime = env.REGIMES[name]
            self.assertEqual(regime.mechanism, fields["mechanism"], name)
            for key, attr in (("q", "q"), ("qmax", "qmax"), ("bias_scale", "bias_scale"),
                              ("router_scale", "router_scale"), ("enter", "enter"), ("exit", "exit"),
                              ("flip_in_burst", "flip_in_burst")):
                if key in fields:
                    self.assertEqual(getattr(regime, attr), fields[key], (name, key))

    def test_contracts_equal_the_frozen_spec(self):
        self.assertEqual(set(SPEC["contracts"]), set(env.CONTRACTS))
        for name, fields in SPEC["contracts"].items():
            c = env.CONTRACTS[name]
            self.assertEqual(list(c.reads), fields["reads"])
            self.assertEqual(c.horizon, fields["horizon"])
            self.assertEqual(c.changed, fields["changed"])


class TestCounterRNG(unittest.TestCase):
    def test_pinned_stream_values_are_bitwise_stable(self):
        key = rng.derive_key("TRAIN", "router", 0)
        self.assertEqual(key, 7263180687321709323)
        self.assertEqual([int(x) for x in rng.bits(key, 3)],
                         [18239051996137043571, 5582917842893469956, 11577464626272957147])
        self.assertEqual([float(x).hex() for x in rng.uniform(key, 3)],
                         ["0x1.fa3c424fcb416p-1", "0x1.35ea08cb4f6bep-2", "0x1.4156d579a14cep-1"])

    def test_offsets_address_elements_not_draw_order(self):
        key = rng.derive_key("unit", "offsets")
        whole = rng.uniform(key, 100)
        self.assertTrue(np.array_equal(whole[37:61], rng.uniform(key, 24, 37)))
        self.assertTrue(np.array_equal(rng.normal(key, 50)[10:20], rng.normal(key, 10, 10)))

    def test_distributions(self):
        u = rng.uniform(rng.derive_key("unit", "u"), 200000)
        self.assertTrue((u >= 0).all() and (u < 1).all())
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.003)
        n = rng.normal(rng.derive_key("unit", "n"), 200000)
        self.assertAlmostEqual(n.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(n.var(), 1.0, delta=0.015)

    def test_distinct_labels_give_distinct_keys(self):
        keys = {rng.derive_key(p, "router", s) for p, n in env.PARTITIONS.items() for s in range(n)}
        self.assertEqual(len(keys), sum(env.PARTITIONS.values()))


class TestSimulator(unittest.TestCase):
    PINS = {  # (partition, seed, regime) -> digests of clean events, corruption labels, readings
        ("TRAIN", 5, "FLIP40"): ("e483db351a2816f0", "f0cf148b7b88629f", "f06f9746cd75066c"),
        ("DEV", 0, "CLEAN"): ("ea7ce84e7a0dcc42", "fd9243e1ba57263e", "d8766a23b1d82a6a"),
        ("VALIDATION", 1, "BURST"): ("7c1a7f39aee0dbd4", "e8f2098494205565", "b7bd2bddeed8351e"),
        ("HOLDOUT", 2, "EVENT40"): ("6dda93f533ed407e", "53f14d26ccad9b1c", "222ad5687c7e31f3"),
    }

    def test_reproducible_generation_matches_pins(self):
        for (part, seed, regime), pins in self.PINS.items():
            b = env.episodes(part, seed, regime, 3, 64)
            self.assertEqual((digest(b.clean), digest(b.flips), digest(b.y)), pins, (part, seed, regime))
        w = env.router("TRAIN", 0)
        self.assertEqual([float(x).hex() for x in w.ravel()[:4]],
                         ["-0x1.43cf1c0000000p-1", "-0x1.9fb0120000000p-1", "-0x1.c8bf140000000p-7",
                          "0x1.6e71720000000p-4"])

    def test_episode_data_does_not_depend_on_batching(self):
        whole = env.episodes("DEV", 7, "QUNIF", 0, 50)
        parts = [env.episodes("DEV", 7, "QUNIF", 0, 20), env.episodes("DEV", 7, "QUNIF", 20, 30)]
        for field in ("x", "xq", "b", "clean", "flips", "y", "q_episode"):
            self.assertTrue(np.array_equal(getattr(whole, field), np.concatenate([getattr(p, field) for p in parts])),
                            field)

    def test_public_quantities_are_float32_representable_and_read_only(self):
        b = env.episodes("TRAIN", 1, "CLEAN", 0, 8)
        for arr in (b.router_w, b.x, b.xq):
            self.assertTrue(np.array_equal(arr, arr.astype(np.float32).astype(np.float64)))
            self.assertFalse(arr.flags.writeable)
        with self.assertRaises(ValueError):
            b.y[0, 0, 0, 0] = 1

    def test_clean_probabilities_follow_the_model(self):
        b = env.episodes("TRAIN", 2, "CLEAN", 0, 16)
        z = env.logits(b.router_w, b.x) + b.b[:, None, :]
        self.assertTrue(np.allclose(b.p, 1 / (1 + np.exp(-z)), atol=1e-15))
        big = env.episodes("TRAIN", 2, "CLEAN", 0, 4000)
        self.assertAlmostEqual(float(big.clean.mean()), float(big.p.mean()), delta=0.004)

    def test_corruption_mechanisms(self):
        n = 3000
        clean = env.episodes("TRAIN", 3, "CLEAN", 0, n)
        self.assertEqual(int(clean.flips.sum()), 0)
        self.assertTrue(np.array_equal(clean.y[..., 0], clean.clean))
        for name, q in (("FLIP20", 0.2), ("FLIP40", 0.4), ("FLIP45", 0.45)):
            b = env.episodes("TRAIN", 3, name, 0, n)
            self.assertAlmostEqual(float(b.flips.mean()), q, delta=0.006, msg=name)
            r0, r1 = b.flips[..., 0].astype(bool), b.flips[..., 1].astype(bool)
            self.assertAlmostEqual(float((r0 & r1).mean()), q * q, delta=0.006, msg=name)  # independent readings
        ev = env.episodes("TRAIN", 3, "EVENT40", 0, n)
        self.assertTrue(np.array_equal(ev.y[..., 0], ev.y[..., 1]))
        self.assertAlmostEqual(float(ev.flips[..., 0].mean()), 0.4, delta=0.006)
        asym = env.episodes("TRAIN", 3, "ASYM40", 0, n)
        self.assertEqual(int((asym.flips.astype(bool) & (asym.clean[..., None] == 0)).sum()), 0)
        ctx = env.episodes("TRAIN", 3, "CTXQ40", 0, n)
        self.assertEqual(int(ctx.flips[ctx.x[..., 0] <= 0].sum()), 0)
        burst = env.episodes("TRAIN", 3, "BURST", 0, n)
        self.assertAlmostEqual(float(burst.flips.mean()), 0.25, delta=0.01)
        qu = env.episodes("TRAIN", 3, "QUNIF", 0, n)
        self.assertTrue((qu.q_episode >= 0).all() and (qu.q_episode < 0.5).all())
        self.assertAlmostEqual(float(qu.flips.mean()), float(qu.q_episode.mean()), delta=0.006)
        wide = env.episodes("TRAIN", 3, "BIASWIDE", 0, n)
        self.assertAlmostEqual(float(wide.b.std()), 1.6, delta=0.05)
        rw = env.episodes("TRAIN", 3, "ROUTERWIDE40", 0, 4)
        self.assertTrue(np.allclose(rw.router_w, env.router("TRAIN", 3, 1.0)))

    def test_readings_return_exactly_indexed_potential_outcomes(self):
        b = env.episodes("DEV", 4, "FLIP40", 0, 32)
        acts = (np.arange(32) % 3).astype(np.int64)
        for t in (0, 7, 15):
            for reads in (1, 2):
                got = b.readings(t, acts, reads)
                self.assertEqual(got.shape, (32, reads))
                for e in range(32):
                    self.assertTrue(np.array_equal(got[e], b.y[e, t, acts[e], :reads]))
        with self.assertRaises(ValueError):
            b.readings(16, acts, 1)
        with self.assertRaises(ValueError):
            b.readings(0, acts, 3)
        with self.assertRaises(ValueError):
            b.readings(0, np.full(32, 3), 1)
        with self.assertRaises(ValueError):
            b.readings(0, acts.astype(float), 1)

    def test_seed_partitions_are_closed(self):
        with self.assertRaises(ValueError):
            env.router("HOLDOUT", 64)
        with self.assertRaises(ValueError):
            env.episodes("VALIDATION", 32, "CLEAN", 0, 1)
        with self.assertRaises(ValueError):
            env.router("TEST", 0)
        routers = {p: {env.router(p, s).tobytes() for s in range(n)} for p, n in env.PARTITIONS.items()}
        for p, rs in routers.items():
            self.assertEqual(len(rs), env.PARTITIONS[p])
            for p2, rs2 in routers.items():
                if p2 != p:
                    self.assertFalse(rs & rs2, (p, p2))

    def test_utilities(self):
        pq = np.array([[[0.9, 0.2, 0.1], [0.3, 0.4, 0.5]]])
        oracle = np.array([[0, env.STOP]])
        self.assertTrue(np.allclose(env.terminal_regret(pq, oracle), 0.0))
        self.assertTrue(np.allclose(env.terminal_regret(pq, np.array([[env.STOP, 2]])), ((0.9 - 0.6) + (0.6 - 0.5)) / 2))
        self.assertTrue(np.allclose(env.online_regret_step(np.array([[0.2, 0.7, 0.4]]), np.array([2])), 0.3))
        s = env.sigmoid(np.array([-800.0, 0.0, 800.0]))
        self.assertTrue(np.array_equal(s, [0.0, 0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
