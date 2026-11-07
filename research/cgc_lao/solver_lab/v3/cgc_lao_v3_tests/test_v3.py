"""v3: faithful benchmark under fresh seeds, selection recompute, freeze, frozen plans, disposition, chronology."""
import json
import subprocess
import unittest

import numpy as np

from cgc_lao_v3_tests import V3
from cgc_lao_lab import env as env1
from cgc_lao_lab import rng as rng1
from cgc_lao_v2 import env2, report2
from cgc_lao_v3 import env3, study3

EVIDENCE = V3 / "evidence"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=V3, capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


class TestV3Benchmark(unittest.TestCase):
    def test_generator_is_slb2_bitwise(self):
        for regime in ("FLIP40", "QUNIF", "BURST", "CORR50", "EVENT40"):
            a = env2.episodes("DEV", 5, regime, 3, 40)
            b = env3.generate(env2.ROOT, "DEV", 5, env2.REGIMES[regime], 3, 40)
            for field in ("router_w", "x", "xq", "b", "p", "pq", "clean", "flips", "y", "q_episode"):
                self.assertTrue(np.array_equal(getattr(a, field), getattr(b, field)), (regime, field))
        b = env3.generate(env2.ROOT, "OOD", 1, env2.REGIMES["SECOND_EVENT40"], 0, 20)
        self.assertTrue(np.array_equal(env2.episodes("OOD", 1, "SECOND_EVENT40", 0, 20).y, b.y))

    def test_partitions_are_fresh(self):
        keys = {"v1": {rng1.derive_key(p, "router", s) for p, n in env1.PARTITIONS.items() for s in range(n)},
                "v2": {env2.derive_key(p, "router", s) for p, n in env2.PARTITIONS.items() for s in range(n)},
                "v3": {env3.derive_key(env3.ROOT, p, "router", s) for p, n in env3.PARTITIONS.items() for s in range(n)}}
        self.assertEqual(len(keys["v3"]), sum(env3.PARTITIONS.values()))
        self.assertFalse(keys["v3"] & keys["v1"])
        self.assertFalse(keys["v3"] & keys["v2"])
        with self.assertRaises(ValueError):
            env3.episodes("TRAIN", 0, "CLEAN", 0, 2)
        with self.assertRaises(ValueError):
            env3.episodes("HOLDOUT", 0, "CORR50", 0, 2)

    def test_v3_candidates_are_bounded_and_admissible(self):
        for name in list(study3.PRIMARY3) + list(study3.FALLBACK3):
            agent, priv = study3.factory(name)(env2.REGIMES["CLEAN"], env2.CONTRACTS["DUP2_4"])
            self.assertIsNone(priv)
            self.assertFalse(agent.privileged)
            self.assertLessEqual(agent.state_scalars(), 64)
        self.assertEqual(study3.complexity("RMF3")[0], 24)


class TestV3Evidence(unittest.TestCase):
    def test_selection_is_recomputed(self):
        path = EVIDENCE / "dev" / "selection_round.json"
        if not (V3 / "SELECTION_V3.json").exists():
            self.skipTest("no v3 selection yet")
        sel = load(V3 / "SELECTION_V3.json")
        got = study3.select(load(path))
        self.assertEqual((got["primary"], got["fallback"]), (sel["primary"], sel["fallback"]))

    def test_freeze_binds_current_sources_and_plans(self):
        if not (V3 / "FROZEN_V3.json").exists():
            self.skipTest("no v3 freeze yet")
        frozen = load(V3 / "FROZEN_V3.json")
        self.assertEqual(frozen["source_files_sha256"], study3.source_file_digests())
        self.assertEqual(frozen["native_files_sha256"], study3.evaluate2.native_file_digests())
        for phase, plan in frozen["plans"].items():
            path = EVIDENCE / phase.lower() / "qualification.json"
            if not path.exists():
                continue
            rec = load(path)
            self.assertEqual(rec["source_files_sha256"], frozen["source_files_sha256"], phase)
            self.assertEqual((rec["partition"], rec["seeds"], rec["episodes_per_seed"], rec["regimes"]),
                             (plan["partition"], list(range(*plan["seeds"])), plan["episodes"], plan["regimes"]))
            self.assertEqual(rec["arms"], {k: {kk: v[kk] for kk in ("arm", "contract", "mode", "overrides")}
                                           for k, v in frozen["specs"][phase].items()})
            parity = EVIDENCE / phase.lower() / "native_parity.json"
            if phase != "OOD":
                self.assertTrue(parity.exists(), phase)
                for role, arm in load(parity)["arms"].items():
                    self.assertTrue(arm["pass"], (phase, role))

    def test_disposition_is_recomputed(self):
        path = EVIDENCE / "DISPOSITION_V3.json"
        if not path.exists():
            self.skipTest("no v3 disposition yet")
        disp, frozen = load(path), load(V3 / "FROZEN_V3.json")
        primary, fallback = frozen["candidates"]["PRIMARY"]["name"], frozen["candidates"]["FALLBACK"]["name"]
        for phase in ("VALIDATION", "HOLDOUT"):
            qpath = EVIDENCE / phase.lower() / "qualification.json"
            if not qpath.exists():
                continue
            rec = load(qpath)
            g, f = report2.primary_gates(rec, primary), report2.fallback_gates(rec, fallback)
            self.assertEqual({k: g[k]["pass"] for k in study3.PRIMARY_GATES}, disp["phases"][phase]["primary"])
            self.assertEqual({k: f[k]["pass"] for k in study3.FALLBACK_GATES}, disp["phases"][phase]["fallback"])
            self.assertEqual({fam: v["claim"] for fam, v in g["families"].items()}, disp["phases"][phase]["families"])

    def test_chronology(self):
        if git("rev-parse", "--is-shallow-repository") != "false":
            self.skipTest("needs a full git history")

        def added(rel):
            out = git("log", "--format=%H", "--diff-filter=A", "--", rel)
            return out.splitlines()[-1] if out else None
        order = [added("PREREGISTRATION_V3.md")]
        self.assertIsNotNone(order[0])
        for rel in ("evidence/dev/selection_round.json", "SELECTION_V3.json", "FROZEN_V3.json",
                    "evidence/validation/qualification.json", "evidence/holdout/qualification.json",
                    "evidence/ood/qualification.json"):
            commit = added(rel)
            if commit is None:
                break
            if commit != order[-1]:
                self.assertIsNotNone(git("merge-base", "--is-ancestor", order[-1], commit), rel)
            order.append(commit)
        self.assertEqual(len(set(order[:2])), min(2, len(order)))   # the preregistration precedes the DEV round


if __name__ == "__main__":
    unittest.main()
