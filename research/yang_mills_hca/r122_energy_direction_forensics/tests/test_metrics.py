"""Exact decomposition identities and explicit classification (numpy only)."""
import unittest

import numpy as np

from research.yang_mills_hca.r122_energy_direction_forensics import controls as C
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M


class TestDecomposition(unittest.TestCase):
    def test_identities_across_dynamic_range(self):
        rng = np.random.default_rng(5)
        for exponent in (0, -3, -20, -150, -300, 150, 300):
            t = rng.normal(size=(200, 32)) * 10.0 ** exponent
            p = t * rng.uniform(-2, 2, size=(200, 1)) + rng.normal(size=(200, 32)) * 10.0 ** exponent * rng.uniform(0, 1)
            d = M.decompose(p, t)
            # below 2^-511 the sample is flagged NEAR_ZERO; the identities still hold exactly
            self.assertTrue(np.all(d["norm_class"] == (M.N_NEAR0 if exponent == -300 else M.N_ORD)), exponent)
            for key in ("res_cos", "res_proj", "res_norm"):
                self.assertLess(float(d[key].max()), 1e-12, (exponent, key))
            self.assertTrue(np.allclose(d["rho"], d["r"] * d["c"], rtol=1e-12, atol=1e-15))
            self.assertTrue(np.all(d["eperp"] >= 0) and np.all(d["egain"] >= 0))

    def test_scale_invariance_is_exact_for_powers_of_two(self):
        rng = np.random.default_rng(6)
        t, p = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
        a = M.decompose(p, t)
        b = M.decompose(np.ldexp(p, -700), np.ldexp(t, -700))
        for key in ("erel", "r", "c", "rho", "egain", "eperp"):
            self.assertTrue(np.array_equal(a[key], b[key]), key)
        self.assertTrue(np.all(b["norm_class"] == M.N_NEAR0))     # exact ratios, but flagged

    def test_no_denominator_floor(self):
        t = np.array([[1e-200, 0.0, 0.0, 0.0]])
        p = np.array([[3e-200, 0.0, 0.0, 0.0]])
        d = M.decompose(p, t)
        self.assertEqual(float(d["erel"][0]), 4.0)
        self.assertEqual(float(d["r"][0]), 3.0)

    def test_from_base_matches_decompose(self):
        rng = np.random.default_rng(7)
        t, p = rng.normal(size=(40, 32)), rng.normal(size=(40, 32)) * 0.3
        a = M.decompose(p, t)
        base = np.stack([a[k] for k in M.BASE_FIELDS], axis=-1)
        b = M.from_base(base)
        for key in ("erel", "r", "c", "rho", "eperp"):
            self.assertTrue(np.allclose(a[key], b[key], rtol=1e-13, atol=0), key)
        self.assertTrue(np.array_equal(a["dir_class"], b["dir_class"]))

    def test_exact_rational_audit(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            t, p = rng.normal(size=32) * 1e-9, rng.normal(size=32) * 1e-7
            worst, identity = M.exact_audit(p, t)
            self.assertLess(worst, 1e-13)
            self.assertEqual(identity, 0)

    def test_classifier_controls(self):
        res = C.synthetic_classifier_controls()
        self.assertTrue(res["pass"], [k for k, v in res["cases"].items() if not v["pass"]])

    def test_zero_is_not_antiparallel(self):
        t = np.ones((3, 4))
        d = M.decompose(np.zeros((3, 4)), t)
        self.assertTrue(np.all(d["norm_class"] == M.N_P0) and np.all(d["dir_class"] == M.D_UNDEF))
        d = M.decompose(-1e-4 * t, t)
        self.assertTrue(np.all(d["dir_class"] == M.D_EXTINGUISHED))
        d = M.decompose(-0.02 * t, t)
        self.assertTrue(np.all(d["dir_class"] == M.D_REVERSED) and bool(d["antiparallel"].all()))


if __name__ == "__main__":
    unittest.main()
