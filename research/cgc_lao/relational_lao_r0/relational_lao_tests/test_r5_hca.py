"""R5 mechanics: the research-only projection and its HCA readout. The numpy readout used at scale equals native
HCA B0 (native/ECS_C/hca) on every query of fixed and held-out topologies; the LAO_PLUS_HCA arm's decisions equal
the decisions read from native scores; the projection is one-way (the learner's state is untouched by it)."""
import unittest

import numpy as np

from relational_lao_tests import REQUIRE_NATIVE
from relational_lao import hca, harness, world as W
from relational_lao.learners import Bayes


def hca_or_skip(test):
    try:
        from relational_lao import hca_native
        hca_native.library()
        return hca_native
    except (RuntimeError, OSError) as exc:
        if REQUIRE_NATIVE:
            raise
        test.skipTest(f"native HCA unavailable: {exc}")


class TestHcaProjection(unittest.TestCase):
    def test_numpy_readout_equals_native_b0(self):
        native = hca_or_skip(self)
        rng = np.random.default_rng(5)
        templates = [W.D_DIAMOND, W.D_FAN, W.V_LADDER] + [
            W.random_template(W.derive_key("OOD", "world", s, "topology")) for s in (3, 7, 11)]
        for tpl in templates:
            struct = list(range(len(tpl.queries)))
            for trial in range(12):
                sal = rng.integers(0, 1001, size=len(tpl.relations))
                if trial % 3 == 0:
                    sal[rng.integers(len(tpl.relations))] = 0
                ref = hca.readout(np.tile(sal, (len(struct), 1))[None], tpl, struct)[0]
                for j, s in enumerate(struct):
                    got = native.query_scores(tpl.queries[s], sal)
                    self.assertEqual(got, [int(v) for v in ref[j][:len(got)]], (tpl.name, tpl.queries[s].name))

    def test_lao_plus_hca_decisions_equal_native_and_the_projection_is_one_way(self):
        native = hca_or_skip(self)
        b = W.episodes("DEV", 41, "FLIP20", 0, 24)
        base = Bayes("R")
        arm = hca.LaoHca(base)
        harness.rollout(b, {"R": base, "H": arm}, "R")
        state = (base.filter.mu.copy(), base.filter.v.copy(), base.filter.lam.copy())
        zq = b.logits(b.xq)
        dec = arm.decide_queries(zq, b.template, b.query_struct)
        p = sum(w[:, None, None] * m for w, m in base.components(zq))
        sal = hca.project(p)
        theta = hca.thresholds(b.template, b.query_struct)
        for e in range(b.count):
            for j, s in enumerate(b.query_struct):
                scores = native.query_scores(b.template.queries[s], sal[e, j])
                best = int(np.argmax(scores))
                self.assertEqual(best if scores[best] >= theta[j] else -1, int(dec[e, j]), (e, j))
        for before, after in zip(state, (base.filter.mu, base.filter.v, base.filter.lam)):
            self.assertTrue(np.array_equal(before, after))
        self.assertTrue(((sal >= 0) & (sal <= 1000)).all() and sal.dtype.kind == "i")


if __name__ == "__main__":
    unittest.main()
