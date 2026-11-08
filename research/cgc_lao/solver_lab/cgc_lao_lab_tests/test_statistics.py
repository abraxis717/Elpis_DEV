"""Student-t intervals, paired summaries, per-seed evaluation and record sealing."""
import json
import math
import unittest

import numpy as np

from cgc_lao_lab_tests import LAB
from cgc_lao_lab import arms, evaluate, harness, stats


class TestStatistics(unittest.TestCase):
    def test_t_quantiles_match_reference_tables(self):
        cases = {1: 12.706204736174698, 2: 4.302652729749464, 15: 2.131449545559323,
                 31: 2.0395134463964077, 63: 1.9983405417721956, 1000: 1.9623390808264078}
        for df, ref in cases.items():
            self.assertAlmostEqual(stats.t_quantile(0.975, df), ref, delta=2e-8, msg=df)
        self.assertAlmostEqual(stats.t_quantile(0.025, 15), -stats.t_quantile(0.975, 15), places=12)
        for df, p in ((63, 0.9875), (31, 0.9875), (5, 0.9)):
            t = stats.t_quantile(p, df)
            # independent check: trapezoidal integral of the t density from 0 to t
            x = np.linspace(0.0, t, 200001)
            logc = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
            dens = np.exp(logc - (df + 1) / 2 * np.log1p(x * x / df))
            area = float(((dens[1:] + dens[:-1]) * 0.5 * np.diff(x)).sum())
            self.assertAlmostEqual(0.5 + area, p, delta=1e-9, msg=(df, p))
        self.assertAlmostEqual(stats.t_cdf(0.0, 7), 0.5, places=14)

    def test_paired_interval(self):
        a = np.array([0.10, 0.12, 0.11, 0.13, 0.09])
        b = np.array([0.11, 0.12, 0.13, 0.12, 0.10])
        out = stats.paired(a, b)
        d = a - b
        half = stats.t_quantile(0.975, 4) * d.std(ddof=1) / math.sqrt(5)
        self.assertAlmostEqual(out["mean"], d.mean(), places=15)
        self.assertAlmostEqual(out["hi"] - out["mean"], half, places=12)
        self.assertEqual(out["wins"], 3)
        with self.assertRaises(ValueError):
            stats.mean_ci([1.0])


class TestEvaluation(unittest.TestCase):
    def test_evaluate_seed_is_chunk_invariant(self):
        facs = {"NOOP": arms.factory("NOOP"), "RMF_U3": arms.factory("RMF_U3")}
        a = harness.evaluate_seed("DEV", 255, ["FLIP40"], facs, "ORIGINAL", 96, chunk=96)
        b = harness.evaluate_seed("DEV", 255, ["FLIP40"], facs, "ORIGINAL", 96, chunk=40)
        for arm in facs:
            for metric, value in a["FLIP40"][arm].items():
                self.assertAlmostEqual(value, b["FLIP40"][arm][metric], places=12, msg=(arm, metric))
        noop = a["FLIP40"]["NOOP"]
        self.assertAlmostEqual(noop["terminal"], noop["boundary"] + noop["ranking"], places=12)
        self.assertEqual(noop["diff_noop"], 0.0)

    def test_run_and_seal(self):
        rec = evaluate.run("DEV", [254, 253], ["CLEAN"], evaluate.specs(["NOOP"]), episodes=32, workers=1)
        self.assertEqual(rec["seeds"], [254, 253])
        self.assertEqual(len(rec["results"]["CLEAN"]["NOOP"]["terminal"]), 2)
        sealed = evaluate.seal(rec, "unit")
        self.assertEqual(sealed["spec_sha256"], evaluate.file_digest(LAB / "specs" / "slb1.spec.json"))
        self.assertEqual(len(sealed["source_sha256"]), 64)
        json.dumps(sealed)


if __name__ == "__main__":
    unittest.main()
