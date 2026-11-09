"""The native C11 relational adapter: bitwise parity with the scalar reference (state, weights, predictive, every
readout, rebuild, slot reset) over RLB1 event streams, its bounded layout, and its refusals."""
import unittest

import numpy as np

from relational_lao_tests import REQUIRE_NATIVE
from relational_lao import scalar, world as W
from relational_lao.learners import H3, H4

CONFIGS = {"H3": (H3, (1.0, 1.0, 1.0)), "H4_W": (H4, (1.0, 0.5, 1.0, 2.0)), "SAFE4_P8": (H4, (1.0, 1.0, 8.0, 8.0)),
           "SAFE3_P4": (H3, (1.0, 4.0, 4.0))}                      # the frozen primary uses H3; SAFE3_P4 is its fallback


def native_or_skip(test):
    try:
        from relational_lao_tests import _native
        _native.library()
        return _native
    except Exception as exc:  # noqa: BLE001 - compiler or loader missing
        if REQUIRE_NATIVE:
            raise
        test.skipTest(f"native adapter unavailable: {exc}")


class TestNativeAdapter(unittest.TestCase):
    def test_bitwise_parity_on_rlb1_streams(self):
        N = native_or_skip(self)
        for regime in ("CLEAN", "FLIP40", "EVENT40"):
            b = W.episodes("DEV", 47, regime, 0, 24)
            for name, (hyp, prior) in CONFIGS.items():
                for readout in ("coupled", "marginal", "independent"):
                    for e in range(b.count):
                        s = scalar.ScalarRelational(hyp, prior, slots=b.R, readout=readout)
                        n = N.NativeRelational(hyp, prior, slots=b.R, readout=readout)
                        for t in range(W.T):
                            r = (e + 3 * t) % b.R
                            reads = tuple(int(v) for v in b.y[e, t, r, :W.READS["I"][t]])
                            z = float(b.logits(b.x[e, t])[r])
                            s.update(r, z, reads, t, None)
                            n.update(r, z, reads, t, 0)
                            self.assertEqual(s.state(), n.state(), (regime, name, readout, e, t))
                        zq = [float(v) for v in b.logits(b.xq[e, 0])]
                        self.assertEqual(s.weights(), n.weights())
                        self.assertEqual(s.predictive(zq), n.predictive(zq))
                        for q in b.template.queries:
                            for alt in q.alternatives:
                                self.assertEqual(s.expected_utility(alt, zq), n.expected_utility(alt, zq),
                                                 (regime, name, readout, q.name))

    def test_invalidation_parity(self):
        N = native_or_skip(self)
        rng = np.random.default_rng(13)
        for trial in range(60):
            s = scalar.ScalarRelational()
            n = N.NativeRelational(H3, (1.0, 1.0, 1.0))
            for t in range(8):
                r, z = int(rng.integers(8)), float(rng.normal())
                reads = tuple(int(v) for v in rng.integers(0, 2, size=2 if t < 4 else 1))
                s.update(r, z, reads, t, t)
                n.update(r, z, reads, t, t)
            keep = int(rng.integers(256))
            s.rebuild(lambda e, k=keep: bool((k >> e[0]) & 1))         # buffered steps 0..7 are the bit indices
            n.rebuild(keep)
            self.assertEqual(s.state(), n.state(), trial)
            victim = int(rng.integers(8))
            s.reset_slot(victim)
            n.reset_slot(victim)
            self.assertEqual(s.state(), n.state(), trial)
            self.assertEqual(len(s.buffer), n.buffered())

    def test_layout_and_refusals(self):
        N = native_or_skip(self)
        n = N.NativeRelational(H3, (1.0, 1.0, 1.0))
        self.assertEqual(n.logical_scalars(), 80)
        self.assertEqual(N.library().rlao_state_size(), 1016)
        for t in range(8):
            n.update(t, 0.1, (1,), t, 0)
        before = n.state()
        with self.assertRaises(N.NativeError):
            n.update(0, 0.1, (1,), 8, 0)                              # capacity: refused, never grown
        self.assertEqual(n.state(), before)
        with self.assertRaises(N.NativeError):
            n.expected_utility(((0, 0),), [0.0] * 8)                 # a repeated relation is refused
        n.zeroize()
        with self.assertRaises(N.NativeError):
            n.weights()


if __name__ == "__main__":
    unittest.main()
